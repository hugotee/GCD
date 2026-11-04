#!/usr/bin/env python3
"""Re-evaluate genuine frozen predictions with the new XML-derived GT and evaluator."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.metrics import evaluate
from gcd.protocol import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter()
    summary = json.loads((a.reference_root / 'results/B0/summary.json').read_text())
    assert sha256(a.predictions) == summary['prediction_sha256']
    new_manifest = json.loads(a.manifest.read_text())
    old_manifest = json.loads((a.reference_root / 'data/manifest.json').read_text())
    gt = json.loads(Path(new_manifest['test']['ann']).read_text())
    old_gt = json.loads(Path(old_manifest['test']['ann']).read_text())
    by_name = {i['file_name']: i['id'] for i in gt['images']}
    remap = {i['id']: by_name[i['file_name']] for i in old_gt['images']}
    predictions = json.loads(a.predictions.read_text())
    for row in predictions:
        row['image_id'] = remap[row['image_id']]
    result = evaluate(gt, predictions)
    for group, scores in result['groups'].items():
        for metric, value in scores.items():
            assert abs(value - summary['groups'][group][metric]) < 1e-6, (group, metric, value)
    result.update(passed=True, new_gpu_inference=False, source='historical full-test B0 predictions',
        prediction_sha256=summary['prediction_sha256'], test_annotation_sha256=sha256(new_manifest['test']['ann']),
        numeric_image_ids_remapped_by_filename=True, images=len(gt['images']),
        elapsed_seconds=time.perf_counter() - start)
    write_json(a.out, result)
    print(result['groups'])


if __name__ == '__main__':
    main()
