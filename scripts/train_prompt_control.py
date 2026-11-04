#!/usr/bin/env python3
"""Train one registered prompt arm using the unchanged baseline config builder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.prompt_control import arm_spec
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime, validate_gpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--arm', choices=['new5', 'all20'], required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--preflight', action='store_true', help='8 microsteps only; engineering check')
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    assert validate_gpus('2', runtime['authorized_gpus']) == [2]
    spec = json.loads((ROOT / 'configs/experiments/task2_prompt_control_v1.json').read_text())
    plan = json.loads(a.plan.read_text())
    assert plan['control_spec_sha256'] == sha256(ROOT / 'configs/experiments/task2_prompt_control_v1.json')
    assert plan['initial_checkpoint_sha256'] == sha256(a.init) == spec['initial_checkpoint_sha256']
    assert plan['manifest_sha256'] == sha256(runtime['manifest'])
    for rel, digest in plan['source_sha256'].items():
        assert sha256(ROOT / rel) == digest, rel
    os.environ['CUDA_VISIBLE_DEVICES'], os.environ['TOKENIZERS_PARALLELISM'] = '2', 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    # The baseline dry-run produces all loss/optimizer/pipeline settings unchanged.
    # Only the registered epoch budget/schedule and three prompt/label fields vary.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    exp = json.loads((ROOT / spec['base_experiment']).read_text())
    exp.update(epochs=spec['epochs'], seed=spec['seed'], lr_milestones=[])
    exp_path = a.out.parent / f'{a.out.name}_experiment.json'
    if exp_path.exists():
        raise FileExistsError(exp_path)
    write_json(exp_path, exp)
    subprocess.run([sys.executable, str(ROOT / 'scripts/run.py'), 'train', '--dry-run',
        '--runtime', str(a.runtime), '--experiment', str(exp_path), '--phase', 'task2',
        '--gpus', '2', '--init', str(a.init), '--work-dir', str(a.out)], check=True)
    sys.path.insert(0, runtime['mmdet_root'])
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    import gcd.model
    import gcd.hooks
    import gcd.control_hooks
    torch.set_num_threads(2)
    register_all_modules()
    cfg = Config.fromfile(str(a.out / 'resolved_config.py'))
    template = json.loads((a.out / 'provenance.json').read_text())
    (a.out / 'resolved_config.py').rename(a.out / 'template_config.py')
    (a.out / 'provenance.json').rename(a.out / 'template_provenance.json')
    arm = arm_spec(a.arm)
    cfg.model.expected_classes = arm['classes']
    cfg.model.train_label_ids = arm['local_label_ids']
    cfg.train_dataloader.dataset.metainfo.classes = tuple(arm['classes'])
    cfg.custom_hooks.append(dict(type='PromptControlAuditHook', arm=a.arm))
    cfg.gcd_experiment_hash = hashlib.sha256((sha256(a.plan) + a.arm + str(a.preflight)).encode()).hexdigest()
    cfg.default_hooks.checkpoint.max_keep_ckpts = 1
    if a.preflight:
        cfg.engineering_only = True
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=8)
        cfg.param_scheduler = []
        cfg.default_hooks.checkpoint.update(by_epoch=False, interval=8)
        cfg.log_processor.by_epoch = False
        cfg.custom_hooks[0].update(profile=True, warmup=0)
    cfg.dump(str(a.out / 'resolved_config.py'))
    provenance = dict(template, config_only=False, engineering_only=a.preflight,
        diagnostic_short_training=True, arm=a.arm, training_classes=arm['classes'],
        classes=arm['classes'], expected_local_labels=arm['local_label_ids'],
        plan_sha256=sha256(a.plan), run_identity_sha256=cfg.gcd_experiment_hash,
        effective_experiment_sha256=sha256(exp_path), resolved_config_sha256=sha256(a.out / 'resolved_config.py'))
    write_json(a.out / 'provenance.json', provenance)
    usage = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    memory, utilization = map(int, usage.strip().split(','))
    if memory > 512 or utilization > 5:
        raise RuntimeError('GPU2 is busy; no training started')
    runner = Runner.from_cfg(cfg)
    runner.train()
    checkpoint = a.out / ('iter_8.pth' if a.preflight else 'epoch_1.pth')
    write_json(a.out / 'run_complete.json', dict(passed=True, arm=a.arm,
        diagnostic_short_training=True, engineering_only=a.preflight,
        checkpoint_sha256=sha256(checkpoint), checkpoint=str(checkpoint),
        plan_sha256=sha256(a.plan), physical_gpu=2))


if __name__ == '__main__':
    main()
