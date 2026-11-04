#!/usr/bin/env python3
"""Analyze completed full-size raw-query probes; refuse CPU-smoke mechanism claims."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from gcd.protocol import CLASSES, sha256, write_json
from gcd.diagnostics import representation_changes

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
completion = json.loads((a.input / 'completion.json').read_text())
assert completion['passed'] and not completion['engineering_only']
assert completion['num_queries'] == 900 and completion['images'] == 128
assert completion['training_performed'] is False
assert sha256(a.input / 'object_scores.json') == completion['object_scores_sha256']
assert sha256(a.input / 'text_features.npz') == completion['text_feature_sha256']
rows = json.loads((a.input / 'object_scores.json').read_text())
features = np.load(a.input / 'text_features.npz')
conditions, perclass = {}, []
for model in ['B0', 'B1', 'B2']:
    for prompt in ['old15', 'all20']:
        selected = [r for r in rows if r['model'] == model and r['prompt'] == prompt]
        counts = Counter(r['category_id'] for r in selected)
        conditions[f'{model}_{prompt}'] = dict(old_gt_instances=len(selected), classes_with_positives=len(counts),
            any_raw_query_iou50_percent=100 * float(np.mean([r['any_raw_query_iou50'] for r in selected])),
            gt_score_coverage_percent={str(t):100 * float(np.mean([r[f'true_score_ge_{t}'] for r in selected])) for t in [.05, .3, .5]})
        for cat in sorted(counts):
            group = [r for r in selected if r['category_id'] == cat]
            perclass.append(dict(model=model, prompt=prompt, category=CLASSES[cat - 1], gt_instances=len(group),
                any_raw_query_iou50_percent=100 * float(np.mean([r['any_raw_query_iou50'] for r in group])),
                same_class_score03_percent=100 * float(np.mean([r['true_score_ge_0.3'] for r in group]))))
        conditions[f'{model}_{prompt}']['macro_class_score03_percent'] = float(np.mean(
            [r['same_class_score03_percent'] for r in perclass if r['model'] == model and r['prompt'] == prompt]))
geometry = {}
for prompt in ['old15', 'all20']:
    geometry[prompt] = {stage: representation_changes(features[f'B1_{prompt}_{stage}'], features[f'B2_{prompt}_{stage}'])
                        for stage in ['pre', 'fused']}
prompt_geometry = {model: representation_changes(features[f'{model}_old15_fused'], features[f'{model}_all20_fused'])
                   for model in ['B0', 'B1', 'B2']}
index = {(r['model'], r['prompt'], r['annotation_id']): r for r in rows}
keys = sorted({r['annotation_id'] for r in rows})
paired = {}
for prompt in ['old15', 'all20']:
    valid = [k for k in keys if all(index[(m, prompt, k)]['any_raw_query_iou50'] for m in ['B1', 'B2'])]
    before = [index[('B1', prompt, k)] for k in valid]
    after = [index[('B2', prompt, k)] for k in valid]
    paired[prompt] = dict(geometry_matched_gt_pairs=len(valid),
        B1_median_best_true_score=float(np.median([r['best_true_class_score_iou50'] for r in before])),
        B2_median_best_true_score=float(np.median([r['best_true_class_score_iou50'] for r in after])),
        B1_mean_query_norm=float(np.mean([r['query_norm'] for r in before])),
        B2_mean_query_norm=float(np.mean([r['query_norm'] for r in after])),
        B1_mean_true_text_norm=float(np.mean([r['text_norm'] for r in before])),
        B2_mean_true_text_norm=float(np.mean([r['text_norm'] for r in after])),
        B1_mean_best_iou_query_text_cosine=float(np.mean([r['best_iou_query_text_cosine'] for r in before])),
        B2_mean_best_iou_query_text_cosine=float(np.mean([r['best_iou_query_text_cosine'] for r in after])))
a.out.mkdir(parents=True, exist_ok=False)
with (a.out / 'per_class_query_coverage.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(perclass[0]));writer.writeheader();writer.writerows(perclass)
write_json(a.out / 'summary.json', dict(passed=True, diagnostic_only=True, formal_AP=False,
    completion=completion, conditions=conditions, B1_to_B2_text_geometry=geometry,
    same_model_prompt_text_geometry=prompt_geometry, geometry_matched_gt_pairs=paired,
    probe_source_sha256=sha256(a.input / 'completion.json'),
    caveats=['Training-side fixed 128 images, not generalization AP.',
             'GT query coverage can share queries; not official detection recall.',
             'Geometry/score changes are measurements, not proof of a unique causal mechanism.']))
print(json.dumps(dict(conditions=conditions, geometry=geometry, paired=paired), indent=2))
