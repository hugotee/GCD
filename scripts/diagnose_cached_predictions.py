#!/usr/bin/env python3
"""Read-only full-test error/candidate analysis and train-side missing-label exposure."""
import argparse
from collections import Counter, defaultdict
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from gcd.diagnostics import candidate_rows, prediction_bins, image_stratum
from gcd.metrics import voc07_class_ap
from gcd.protocol import CLASSES, convert_voc, sha256, write_json


def write_csv(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--b0-predictions', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = json.loads(a.manifest.read_text())
    prior = json.loads((a.reference_root / 'data/manifest.json').read_text())
    gt = json.loads(Path(manifest['test']['ann']).read_text())
    old_gt = json.loads(Path(prior['test']['ann']).read_text())
    by_name = {i['file_name']: i['id'] for i in gt['images']}
    names = {i['id']: i['file_name'] for i in gt['images']}
    remap = {i['id']: by_name[i['file_name']] for i in old_gt['images']}
    anns = defaultdict(list)
    for ann in gt['annotations']:
        anns[ann['image_id']].append(ann)
    strata = {i['id']: image_stratum(anns[i['id']]) for i in gt['images']}
    stratum_counts = dict(Counter(strata.values()))
    summaries, perclass, confusion_rows, trajectory = {}, [], [], {}
    for model in ('B0', 'B1', 'B2'):
        source = a.b0_predictions if model == 'B0' else a.reference_root / f'runs/{model}_eval/predictions.bbox.json'
        reference = json.loads((a.reference_root / f'results/{model}/summary.json').read_text())
        assert sha256(source) == reference['prediction_sha256']
        raw = json.loads(source.read_text())
        indexed = defaultdict(list)
        for pred in raw:
            pred['image_id'] = remap[pred['image_id']]
            indexed[pred['image_id']].append(pred)
        counts, conf, candidates = Counter(), Counter(), Counter()
        class_candidates = defaultdict(Counter)
        thresholds = {str(x): Counter() for x in [0., .05, .3, .5]}
        for n, image in enumerate(gt['images']):
            iid = image['id']
            objects = candidate_rows(anns[iid], indexed[iid])
            for row in objects:
                if row['category_id'] <= 15:
                    candidates[row['candidate_bin']] += 1
                    class_candidates[row['category_id']][row['candidate_bin']] += 1
                key = row['annotation_id']
                if key not in trajectory:
                    trajectory[key] = dict(annotation_id=key, image_id=iid, file_name=names[iid],
                        category=CLASSES[row['category_id'] - 1], category_id=row['category_id'],
                        stratum=strata[iid])
                for field in ('candidate_bin', 'best_same_class_score_iou50',
                              'max_cached_iou', 'strongest_localized_label'):
                    trajectory[key][model + '_' + field] = row[field]
            for t, total in thresholds.items():
                for row in objects if float(t) == .3 else candidate_rows(anns[iid], indexed[iid], float(t)):
                    if row['category_id'] <= 15:
                        total['old_gt'] += 1
                        total['any_class_iou50'] += row['any_class_iou50_at_threshold']
                        total['same_class_iou50'] += row['same_class_iou50_at_threshold']
            errors, wrong = prediction_bins(anns[iid], indexed[iid])
            counts.update(errors)
            conf.update(wrong)
            if (n + 1) % 500 == 0:
                write_json(a.out / 'progress.json', dict(model=model, images=n + 1,
                                                        elapsed_seconds=time.perf_counter() - started))
        ap_strata = {}
        for stratum in ['all', 'old_only', 'old_and_new']:
            ids = {i for i, s in strata.items() if stratum == 'all' or s == stratum}
            subset = dict(gt, images=[i for i in gt['images'] if i['id'] in ids],
                          annotations=[ann for ann in gt['annotations'] if ann['image_id'] in ids])
            predictions = [d for d in raw if d['image_id'] in ids]
            values = []
            for cat in range(1, 16):
                ap = voc07_class_ap(subset, predictions, cat)
                values.append(ap)
                perclass.append(dict(model=model, stratum=stratum, category=CLASSES[cat - 1],
                    category_id=cat, VOC07_AP50=ap,
                    gt_instances=sum(x['category_id'] == cat and not x['iscrowd'] for x in subset['annotations'])))
            ap_strata[stratum] = dict(images=len(ids), categories_with_positives=sum(v is not None for v in values),
                old15_VOC07_AP50=float(np.mean([v for v in values if v is not None])))
        assert abs(ap_strata['all']['old15_VOC07_AP50'] - reference['groups']['old15']['VOC07_AP50']) < 1e-6
        for (true, predicted), count in conf.most_common():
            confusion_rows.append(dict(model=model, true_category=CLASSES[true - 1],
                predicted_category=CLASSES[predicted - 1], true_category_id=true,
                predicted_category_id=predicted, count=count))
        summaries[model] = dict(source_prediction_sha256=reference['prediction_sha256'],
            prediction_count=len(raw), prediction_bins_at_score03=dict(counts),
            old_gt_candidate_bins_at_score03=dict(candidates),
            old_gt_candidate_bins_by_class={CLASSES[k - 1]: dict(v) for k, v in class_candidates.items()},
            old_gt_coverage_by_threshold={k: dict(v) for k, v in thresholds.items()},
            stratum_ap=ap_strata)
        del raw, indexed, predictions
        gc.collect()
        print(model, ap_strata, flush=True)
    train = convert_voc(manifest['voc_root'], 'trainval')
    t2 = json.loads(Path(manifest['phases']['task2']['train_ann']).read_text())
    t2ids = {i['id'] for i in t2['images']}
    exposure = []
    for cat in range(1, 16):
        total = [x for x in train['annotations'] if x['category_id'] == cat and not x['iscrowd']]
        hidden = [x for x in total if x['image_id'] in t2ids]
        before = next(r['VOC07_AP50'] for r in perclass if r['model'] == 'B1' and r['stratum'] == 'all' and r['category_id'] == cat)
        after = next(r['VOC07_AP50'] for r in perclass if r['model'] == 'B2' and r['stratum'] == 'all' and r['category_id'] == cat)
        exposure.append(dict(category=CLASSES[cat - 1], category_id=cat,
            trainval_instances=len(total), hidden_old_instances_in_task2=len(hidden),
            task2_images_containing_old_class=len({x['image_id'] for x in hidden}),
            hidden_fraction_of_trainval_instances=len(hidden) / len(total),
            B1_minus_B2_VOC07_AP50=before - after))
    corr = float(np.corrcoef([x['hidden_fraction_of_trainval_instances'] for x in exposure],
                            [x['B1_minus_B2_VOC07_AP50'] for x in exposure])[0, 1])
    write_csv(a.out / 'per_class_stratum_ap.csv', perclass)
    write_csv(a.out / 'confusions_score03.csv', confusion_rows)
    write_csv(a.out / 'object_trajectories.csv', list(trajectory.values()))
    write_csv(a.out / 'task2_hidden_old_exposure.csv', exposure)
    write_json(a.out / 'summary.json', dict(passed=True, diagnostic_only=True, training_performed=False,
        source_test_sha256=sha256(manifest['test']['ann']), test_images=len(gt['images']),
        score_thresholds=[0., .05, .3, .5], iou_strict_greater_than=.5,
        strata=stratum_counts, models=summaries, task2_images=len(t2ids),
        hidden_old_instances=sum(x['hidden_old_instances_in_task2'] for x in exposure),
        exposure_forgetting_pearson_over_15_classes=corr,
        elapsed_seconds=time.perf_counter() - started,
        caveats=['Cached top300 box candidates are not all900 raw queries.',
                 'Missing same-class cached scores are null, not zero network probability.',
                 'GT candidate bins can share predictions; they are not official one-to-one recall or TIDE dAP.',
                 'Exposure correlation and strata are descriptive, not causal identification.',
                 'All historical B1/B2 evaluations already used the same 20-class prompt.']))


if __name__ == '__main__':
    main()
