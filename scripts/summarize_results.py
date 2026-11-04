#!/usr/bin/env python3
"""Combine genuine B0/B1/B2 summaries and report absolute AP-point changes."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    models = {}
    for name in ('B0', 'B1', 'B2'):
        summary = json.loads((a.results_root / name / 'summary.json').read_text())
        assert not summary['engineering_only']
        models[name] = summary['groups']
    rows = []
    for name in models:
        row = dict(model=name)
        for metric in ('VOC07_AP50', 'COCO_AP50', 'COCO_AP'):
            for group in ('old15', 'new5', 'all20'):
                row[f'{group}_{metric}'] = models[name][group][metric]
            row[f'F_old_{metric}'] = (models['B1']['old15'][metric] - models['B2']['old15'][metric]) if name == 'B2' else ''
            row[f'G_new_{metric}'] = (models['B2']['new5'][metric] - models['B1']['new5'][metric]) if name == 'B2' else ''
        rows.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(a.out)


if __name__ == '__main__':
    main()
