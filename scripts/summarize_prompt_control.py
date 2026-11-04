#!/usr/bin/env python3
"""Accept the two-arm result only after paired inputs and fixed budgets pass."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.prompt_control import compare_traces
from gcd.protocol import CLASSES, sha256, write_json

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--reference-summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
spec = plan['control_spec']
reference = json.loads(a.reference_summary.read_text())
assert reference['provenance']['initial_checkpoint_sha256'] == plan['initial_checkpoint_sha256']
outputs, audits, traces, provenance = {}, {}, {}, {}
for arm in spec['arms']:
    train, result = a.root / arm, a.root / f'{arm}_results'
    complete = json.loads((train / 'run_complete.json').read_text())
    audit = json.loads((train / 'audit_rank0.json').read_text())
    control = json.loads((train / 'control_audit.json').read_text())
    source = json.loads((train / 'provenance.json').read_text())
    evaluation = json.loads((result / 'summary.json').read_text())
    assert complete['passed'] and not complete['engineering_only']
    assert complete['plan_sha256'] == source['plan_sha256'] == sha256(a.plan)
    assert source['initial_checkpoint_sha256'] == plan['initial_checkpoint_sha256']
    assert audit['passed'] and audit['frozen_parameters_unchanged'] and control['passed']
    assert audit['iterations'] == spec['expected_microsteps']
    assert audit['optimizer_steps'] == audit['expected_optimizer_steps'] == spec['expected_optimizer_updates']
    assert audit['amp_skipped_steps'] == 0 and audit['changed_lora_tensors'] == 48
    ep = evaluation['provenance']
    assert ep['evaluation_completed'] and not ep['engineering_only'] and not ep['config_only']
    assert ep['classes'] == list(CLASSES) and ep['num_queries'] == 900
    assert len(set(ep['evaluated_image_ids'])) == spec['evaluation_images']
    assert ep['initial_checkpoint_sha256'] == complete['checkpoint_sha256']
    if 'launcher_code_sha256' in ep:
        assert ep.get('model_initialization_verified') and ep.get('loaded_checkpoint_state_exact')
    outputs[arm] = dict(groups=evaluation['groups'],
        delta_vs_B1={g: {m: evaluation['groups'][g][m] - reference['groups'][g][m]
                          for m in reference['groups'][g]} for g in reference['groups']})
    audits[arm] = dict(training=audit, control=control)
    traces[arm] = json.loads((train / 'input_trace.json').read_text())
    provenance[arm] = dict(checkpoint_sha256=complete['checkpoint_sha256'],
        prediction_sha256=evaluation['prediction_sha256'],
        train_provenance_sha256=sha256(train / 'provenance.json'),
        evaluation_summary_sha256=sha256(result / 'summary.json'),
        input_trace_sha256=sha256(train / 'input_trace.json'))
assert audits['new5']['control']['initial_adapter_sha256'] == audits['all20']['control']['initial_adapter_sha256']
pair = compare_traces(traces['new5'], traces['all20'], spec['expected_train_images'])
assert sorted(i for r in traces['new5'] for i in r['image_ids']) == plan['sorted_training_image_ids']
difference = {g: {m: outputs['new5']['groups'][g][m] - outputs['all20']['groups'][g][m]
                  for m in reference['groups'][g]} for g in reference['groups']}
a.out.mkdir(parents=True, exist_ok=False)
rows = []
for name, groups in [('B1_initial', reference['groups']), *[(k, v['groups']) for k, v in outputs.items()]]:
    rows.extend(dict(model=name, group=g, **v) for g, v in groups.items())
with (a.out / 'group_metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
class_sources = {'B1': a.reference_summary.with_name('per_class_ap.csv'),
                 **{arm: a.root / f'{arm}_results/per_class_ap.csv' for arm in spec['arms']}}
class_values = {name: {r['category']: r for r in csv.DictReader(path.open())}
                for name, path in class_sources.items()}
assert all(set(values) == set(CLASSES) for values in class_values.values())
perclass = []
for index, category in enumerate(CLASSES):
    row = dict(category_id=index + 1, category=category, group='old15' if index < 15 else 'new5')
    for metric in ['VOC07_AP50', 'COCO_AP50', 'COCO_AP']:
        for name, values in class_values.items():
            row[f'{name}_{metric}'] = float(values[category][metric])
        row[f'new5_minus_all20_{metric}'] = row[f'new5_{metric}'] - row[f'all20_{metric}']
    perclass.append(row)
for name, values in class_values.items():
    groups = reference['groups'] if name == 'B1' else outputs[name]['groups']
    for group in groups:
        selected = [r for r in perclass if group == 'all20' or r['group'] == group]
        for metric, reported in groups[group].items():
            assert abs(sum(r[f'{name}_{metric}'] for r in selected) / len(selected) - reported) < 1e-6
with (a.out / 'per_class_comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(perclass[0])); writer.writeheader(); writer.writerows(perclass)
write_json(a.out / 'summary.json', dict(passed=True, diagnostic_short_training=True,
    formal_12_epoch_baseline=False, plan_sha256=sha256(a.plan), paired_inputs=pair,
    reference_summary_sha256=sha256(a.reference_summary),
    B1_initial_groups=reference['groups'], arms=outputs, new5_minus_all20=difference,
    audits=audits, provenance=provenance,
    caveats=['Single seed, one complete Task2 epoch, last checkpoint only.',
             'Uniform all20 full-test evaluation for both arms.',
             'Prompt change also changes text conditioning and matching; not pure negative-loss removal.',
             'A retention difference must be assessed together with new-class adaptation.']))
print(json.dumps(dict(paired_inputs=pair, arms=outputs, new5_minus_all20=difference), indent=2))
