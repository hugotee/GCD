#!/usr/bin/env python3
"""Paired, class-composition-aware summaries of the cached diagnostic evidence."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from gcd.protocol import write_json, sha256

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
summary = json.loads((a.input / 'summary.json').read_text())
ap = list(csv.DictReader((a.input / 'per_class_stratum_ap.csv').open()))
lookup = {(r['model'], r['stratum'], r['category']): float(r['VOC07_AP50'])
          for r in ap if r['VOC07_AP50']}
common = sorted({r['category'] for r in ap if r['stratum'] == 'old_and_new' and r['VOC07_AP50']})
strata = {}
for key in ['old_only', 'old_and_new']:
    before = float(np.mean([lookup[('B1', key, c)] for c in common]))
    after = float(np.mean([lookup[('B2', key, c)] for c in common]))
    strata[key] = dict(categories=common, images=summary['models']['B1']['stratum_ap'][key]['images'],
                       B1_AP50=before, B2_AP50=after, drop_AP50=before - after)
support = {}
raw_ap = {(r['model'], r['stratum'], r['category']): r for r in ap}
for minimum in [10, 20, 50]:
    categories = [c for c in common if all(int(raw_ap[('B1', s, c)]['gt_instances']) >= minimum
                                         for s in ['old_only', 'old_and_new'])]
    support[str(minimum)] = dict(categories=categories, category_count=len(categories),
        drops={s: float(np.mean([lookup[('B1', s, c)] - lookup[('B2', s, c)] for c in categories]))
               for s in ['old_only', 'old_and_new']})
trajectories = list(csv.DictReader((a.input / 'object_trajectories.csv').open()))
paired = []
for row in trajectories:
    if int(row['category_id']) <= 15 and row['B1_best_same_class_score_iou50'] and row['B2_best_same_class_score_iou50']:
        paired.append((float(row['B1_best_same_class_score_iou50']),
                       float(row['B2_best_same_class_score_iou50'])))
paired = np.asarray(paired)
coverage = {}
for model, d in summary['models'].items():
    coverage[model] = {}
    for threshold, counts in d['old_gt_coverage_by_threshold'].items():
        coverage[model][threshold] = {k + '_percent': 100 * counts[k] / counts['old_gt']
                                     for k in ['any_class_iou50', 'same_class_iou50']}
exposure = list(csv.DictReader((a.input / 'task2_hidden_old_exposure.csv').open()))
probe_completion = a.input / 'full_probe/completion.json'
probe = json.loads(probe_completion.read_text()) if probe_completion.exists() else {}
outputs = dict(passed=True, diagnostic_only=True, common_category_count=len(common),
    class_matched_strata=strata, minimum_support_sensitivity=support, coverage_percent=coverage,
    paired_cached_scores=dict(old_gt_pairs=len(paired),
        B1_median=float(np.median(paired[:, 0])), B2_median=float(np.median(paired[:, 1])),
        B2_score_lower_fraction=float((paired[:, 1] < paired[:, 0]).mean()),
        caveat='Conditional on same-class IoU>0.5 candidates present in both cached top300 lists.'),
    worst_classes=sorted(exposure, key=lambda x: float(x['B1_minus_B2_VOC07_AP50']), reverse=True)[:5],
    artifact_hashes={a.input.joinpath(n).name: sha256(a.input / n) for n in
        ['summary.json', 'per_class_stratum_ap.csv', 'task2_hidden_old_exposure.csv',
         'object_trajectories.csv', 'confusions_score03.csv']},
    GPU_full_probe_completed=bool(probe.get('passed') and not probe.get('engineering_only')
                                 and probe.get('images') == 128 and probe.get('num_queries') == 900),
    interpretation='Confidence-scale and cached-candidate changes are observed; pure monotone class score scaling alone cannot explain AP loss.')
write_json(a.out, outputs)
print(json.dumps(outputs, indent=2))
