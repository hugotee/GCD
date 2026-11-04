#!/usr/bin/env python3
"""Native baseline evaluation on a sustained low-utilization authorized GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime, validate_gpus


def initialize_eval_runner(runner, checkpoint):
    """MMEngine test() does not initialize this project's custom checkpoint."""
    import torch
    runner._init_model_weights()
    model = runner.model.module if hasattr(runner.model, 'module') else runner.model
    expected = torch.load(checkpoint, map_location='cpu')['state_dict']
    actual = model.state_dict()
    if set(actual) != set(expected) or any(not torch.equal(actual[n].detach().cpu(), expected[n]) for n in actual):
        raise RuntimeError('Loaded evaluation state differs from the requested checkpoint')
    return dict(passed=True,checkpoint_state_exact=True,state_tensors=len(actual),checkpoint_sha256=sha256(checkpoint))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--gpu', default='auto')
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    allowed = runtime['authorized_gpus'] if a.gpu == 'auto' else validate_gpus(a.gpu, runtime['authorized_gpus'])
    samples = []
    for i in range(3):
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
        samples.append({r[0]: r for r in [list(map(int, x.split(','))) for x in rows.strip().splitlines()]})
        if i < 2:
            time.sleep(2)
    available = [g for g in allowed if all(s[g][2] - s[g][1] >= 6144 and s[g][3] <= 5 for s in samples)]
    if not available:
        raise RuntimeError('No authorized GPU has sustained low utilization and at least 6 GiB free; no evaluation started')
    gpu = max(available, key=lambda g: min(s[g][2] - s[g][1] for s in samples))
    os.environ['CUDA_VISIBLE_DEVICES'], os.environ['TOKENIZERS_PARALLELISM'] = str(gpu), 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), 'eval', '--dry-run',
        '--runtime', str(a.runtime), '--gpus', str(gpu), '--phase', 'task2',
        '--init', str(a.init), '--work-dir', str(a.work_dir)], check=True)
    sys.path.insert(0, runtime['mmdet_root'])
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    import gcd.model
    import gcd.hooks
    torch.set_num_threads(2)
    register_all_modules()
    cfg = Config.fromfile(str(a.work_dir / 'resolved_config.py'))
    provenance = json.loads((a.work_dir / 'provenance.json').read_text())
    (a.work_dir / 'provenance.json').rename(a.work_dir / 'template_provenance.json')
    usage = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    used, total, utilization = map(int, usage.strip().split(','))
    if total - used < 6144 or utilization > 5:
        raise RuntimeError('GPU became busy before model construction; no evaluation started')
    provenance.update(config_only=False, evaluation_completed=False,
        launcher_code_sha256=sha256(__file__), cpu_threads=2,
        resource_policy=dict(authorized_gpus=allowed, selected_physical_gpu=gpu,
            samples=samples, used_MiB_before_start=used, free_MiB_before_start=total-used,
            minimum_free_MiB=6144, maximum_utilization_percent=5,
            note='Existing resident processes untouched; low-utilization inference with VRAM headroom.'))
    write_json(a.work_dir / 'provenance.json', provenance)
    runner = Runner.from_cfg(cfg)
    initialization = initialize_eval_runner(runner, a.init)
    write_json(a.work_dir / 'initialization_audit.json', initialization)
    provenance.update(model_initialization_verified=True,loaded_checkpoint_state_exact=True)
    write_json(a.work_dir / 'provenance.json', provenance)
    # Four fixed training-side images exercise the complete prediction path.
    import copy
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS
    manifest = json.loads(Path(runtime['manifest']).read_text())
    smoke_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
    smoke_cfg.ann_file = manifest['phases']['task2']['smoke_eval_ann']
    smoke_cfg.data_prefix.img = manifest['phases']['task2']['image_root'] + '/'
    smoke = DATASETS.build(smoke_cfg)
    runner.model.eval()
    checks = []
    with torch.inference_mode():
        for i in range(len(smoke)):
            pred = runner.model.test_step(pseudo_collate([smoke[i]]))[0].pred_instances
            assert len(pred.scores) and torch.isfinite(pred.scores).all() and torch.isfinite(pred.bboxes).all()
            checks.append(dict(image_id=int(smoke[i]['data_samples'].img_id),detections=len(pred.scores),max_score=float(pred.scores.max())))
    write_json(a.work_dir / 'prediction_preflight.json',dict(passed=True,engineering_only=True,
        full_resolution=True,num_queries=900,images=checks,note='Train-side engineering preflight, not AP.'))
    runner.test()
    gt = json.loads(Path(manifest['test']['ann']).read_text())
    provenance.update(evaluation_completed=True, evaluated_image_ids=[i['id'] for i in gt['images']])
    write_json(a.work_dir / 'provenance.json', provenance)
    write_json(a.work_dir / 'run_complete.json', dict(passed=True, mode='eval', engineering_only=False,
        physical_gpu=gpu, evaluated_images=len(gt['images'])))


if __name__ == '__main__':
    main()
