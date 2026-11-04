#!/usr/bin/env python3
"""Read-only audit of real assets and historical result provenance; no CUDA use."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime


def semantic_annotations(data):
    # Match by image filename rather than relying on a converter's integer IDs.
    images = {i['id']: i['file_name'] for i in data['images']}
    return sorted((images[a['image_id']], a['category_id'], tuple(a['bbox']), int(a['iscrowd']))
                  for a in data['annotations'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--b0-predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    ref = a.reference_root
    manifest = json.loads(Path(runtime['manifest']).read_text())
    previous = json.loads((ref / 'data/manifest.json').read_text())
    source_hash = sha256(runtime['pretrained_checkpoint'])
    assert source_hash == runtime['pretrained_sha256']
    stages = {}
    for key in ('task1', 'task2'):
        new = json.loads(Path(manifest['phases'][key]['train_ann']).read_text())
        old = json.loads(Path(previous['phases'][key]['train_ann']).read_text())
        assert sorted(i['file_name'] for i in new['images']) == sorted(i['file_name'] for i in old['images'])
        assert semantic_annotations(new) == semantic_annotations(old)
        stages[key] = dict(images=len(new['images']), annotations=len(new['annotations']), semantic_match=True)
    newtest = json.loads(Path(manifest['test']['ann']).read_text())
    oldtest = json.loads(Path(previous['test']['ann']).read_text())
    assert semantic_annotations(newtest) == semantic_annotations(oldtest)
    def dimensions(data):
        return sorted((i['file_name'], i['width'], i['height']) for i in data['images'])
    assert dimensions(newtest) == dimensions(oldtest)
    results = {}
    for model in ('B0', 'B1', 'B2'):
        summary = json.loads((ref / f'results/{model}/summary.json').read_text())
        prediction = a.b0_predictions if model == 'B0' else ref / f'runs/{model}_eval/predictions.bbox.json'
        assert sha256(prediction) == summary['prediction_sha256']
        assert not summary['engineering_only']
        rows = list(csv.DictReader((ref / f'results/{model}/per_class_ap.csv').open()))
        assert len(rows) == 20 and {int(r['category_id']) for r in rows} == set(range(1, 21))
        for group in ('old15', 'new5', 'all20'):
            selected = [r for r in rows if group == 'all20' or r['group'] == group]
            for metric in ('VOC07_AP50', 'COCO_AP50', 'COCO_AP'):
                mean = sum(float(r[metric]) for r in selected) / len(selected)
                assert abs(mean - summary['groups'][group][metric]) < 1e-8
        checkpoint = Path(summary['provenance'].get('initial_checkpoint', runtime['pretrained_checkpoint']))
        actual_checkpoint_hash = sha256(checkpoint)
        assert actual_checkpoint_hash == summary['provenance']['initial_checkpoint_sha256']
        results[model] = dict(prediction_sha256=summary['prediction_sha256'],
            prediction_bytes=prediction.stat().st_size, checkpoint_sha256=actual_checkpoint_hash,
            groups=summary['groups'], per_class_groups_recomputed=True)
    b1_eval = json.loads((ref / 'runs/B1_eval/provenance.json').read_text())
    b2_train = json.loads((ref / 'runs/B2_formal/provenance.json').read_text())
    assert b1_eval['initial_checkpoint_sha256'] == b2_train['initial_checkpoint_sha256']
    packages = {}
    for name in ('torch', 'torchvision', 'mmcv', 'mmengine', 'transformers', 'numpy', 'pycocotools'):
        packages[name] = importlib.metadata.version(name)
    gpu_inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                                           '--format=csv,noheader'], text=True)
    write_json(a.out, dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        new_gpu_jobs_started=False, authorized_gpus=runtime['authorized_gpus'],
        pretrained_sha256=source_hash, packages=packages, gpu_inventory=gpu_inventory,
        phases=stages, test_semantic_match=True, test_images=len(newtest['images']),
        note='Converters use different numeric image IDs; compare/remap by filename before reusing predictions.',
        B2_init_equals_B1_evaluated_checkpoint=True, reference_results=results))
    print(a.out.read_text())


if __name__ == '__main__':
    main()
