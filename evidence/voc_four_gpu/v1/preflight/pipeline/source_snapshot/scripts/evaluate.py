#!/usr/bin/env python3
"""Evaluate completed full-test predictions; reject smoke and incomplete runs."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gcd.metrics import evaluate
from gcd.protocol import sha256, write_json, CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['B0', 'B1', 'B2'], required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    provenance = json.loads(a.provenance.read_text())
    gt = json.loads(Path(manifest['test']['ann']).read_text())
    assert not manifest.get('engineering_only') and not provenance['engineering_only']
    assert not provenance.get('config_only') and provenance['evaluation_completed']
    assert provenance['evaluation_ann_sha256'] == sha256(manifest['test']['ann']) == manifest['test']['sha256']
    assert provenance['classes'] == list(CLASSES)
    assert sorted(provenance['evaluated_image_ids']) == sorted(i['id'] for i in gt['images'])
    assert len(gt['images']) == 4952, 'Formal protocol requires the complete VOC07 test'
    if a.model == 'B0':
        assert provenance['mode'] == 'frozen'
        assert provenance['initial_checkpoint_sha256'] == provenance['pretrained_sha256']
    else:
        assert provenance['mode'] == 'eval'
        assert provenance['phase'] == ('task1' if a.model == 'B1' else 'task2')
    result = evaluate(gt, json.loads(a.predictions.read_text()))
    a.out.mkdir(parents=True, exist_ok=False)
    rows = result.pop('per_class')
    with (a.out / 'per_class_ap.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result.update(model=a.model, prediction_sha256=sha256(a.predictions),
                  provenance=provenance, engineering_only=False)
    write_json(a.out / 'summary.json', result)
    print(json.dumps(result['groups'], indent=2))


if __name__ == '__main__':
    main()
