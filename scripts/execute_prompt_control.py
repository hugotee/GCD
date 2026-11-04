#!/usr/bin/env python3
"""Finite two-arm pipeline; reuse verified completed stages, reject partial ones."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.prompt_control import compare_traces
from gcd.protocol import CLASSES, sha256, write_json
from gcd.runtime import read_runtime, validate_gpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--init', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--reference-summary', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--eval-gpu', default='auto', help='Authorized GPU(s), or auto by sustained utilization/free VRAM')
    p.add_argument('--validate-only', action='store_true', help='Check completed stages without starting a job')
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    assert validate_gpus('2', runtime['authorized_gpus']) == [2]
    plan = json.loads(a.plan.read_text())
    assert sha256(a.init) == plan['initial_checkpoint_sha256']
    assert sha256(runtime['manifest']) == plan['manifest_sha256']
    for path, digest in plan['source_sha256'].items():
        assert sha256(ROOT / path) == digest, path
    a.out.mkdir(parents=True, exist_ok=True)

    def run(name, target, command):
        if a.validate_only:
            raise RuntimeError(f'Incomplete stage {name}; validation did not launch a job')
        if target.exists():
            raise FileExistsError(f'Partial stage preserved at {target}; use a new attempt or inspect it')
        logs = a.out / 'pipeline_logs'; logs.mkdir(exist_ok=True)
        write_json(a.out / 'pipeline_progress.json', dict(stage=name, status='running', command=command))
        with (logs / f'{name}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        write_json(a.out / 'pipeline_progress.json', dict(stage=name, status='completed'))

    completed, traces = {}, {}
    for arm in ['new5', 'all20']:
        work = a.out / arm
        marker = work / 'run_complete.json'
        if not marker.is_file():
            run(f'train_{arm}', work, [sys.executable, str(ROOT / 'scripts/train_prompt_control.py'),
                '--runtime', str(a.runtime), '--init', str(a.init), '--plan', str(a.plan),
                '--arm', arm, '--out', str(work)])
        completed[arm] = json.loads(marker.read_text())
        assert completed[arm]['passed'] and not completed[arm]['engineering_only']
        assert completed[arm]['plan_sha256'] == sha256(a.plan)
        assert sha256(work / 'epoch_1.pth') == completed[arm]['checkpoint_sha256']
        traces[arm] = json.loads((work / 'input_trace.json').read_text())
    pair = compare_traces(traces['new5'], traces['all20'], plan['control_spec']['expected_train_images'])
    assert sorted(i for row in traces['new5'] for i in row['image_ids']) == plan['sorted_training_image_ids']

    evaluation_gpus = {}
    for arm in ['new5', 'all20']:
        work, metrics = a.out / f'{arm}_eval', a.out / f'{arm}_results'
        if not (work / 'run_complete.json').is_file():
            run(f'eval_{arm}', work, [sys.executable, str(ROOT / 'scripts/evaluate_available_gpu.py'),
                '--runtime', str(a.runtime), '--gpu', a.eval_gpu,
                '--init', str(a.out / arm / 'epoch_1.pth'), '--work-dir', str(work)])
        provenance = json.loads((work / 'provenance.json').read_text())
        assert provenance['evaluation_completed'] and not provenance['engineering_only']
        assert not provenance['config_only'] and provenance['classes'] == list(CLASSES)
        assert provenance['initial_checkpoint_sha256'] == completed[arm]['checkpoint_sha256']
        if 'launcher_code_sha256' in provenance:
            assert provenance.get('model_initialization_verified') and provenance.get('loaded_checkpoint_state_exact')
        evaluation_gpus[arm] = provenance['physical_gpus']
        if not (metrics / 'summary.json').is_file():
            run(f'metrics_{arm}', metrics, [sys.executable, str(ROOT / 'scripts/evaluate.py'),
                '--manifest', runtime['manifest'], '--predictions', str(work / 'predictions.bbox.json'),
                '--provenance', str(work / 'provenance.json'), '--model', 'B2', '--out', str(metrics)])
        assert sha256(work / 'predictions.bbox.json') == json.loads((metrics / 'summary.json').read_text())['prediction_sha256']
    analysis = a.out / 'analysis'
    if not (analysis / 'summary.json').is_file():
        run('analysis', analysis, [sys.executable, str(ROOT / 'scripts/summarize_prompt_control.py'),
            '--root', str(a.out), '--plan', str(a.plan), '--reference-summary', str(a.reference_summary),
            '--out', str(analysis)])
    summary = json.loads((analysis / 'summary.json').read_text())
    assert summary['passed'] and summary['plan_sha256'] == sha256(a.plan)
    assert summary['paired_inputs'] == pair
    write_json(a.out / 'pipeline_complete.json', dict(passed=True, validation_only=a.validate_only,
        training_physical_gpu=2, evaluation_physical_gpus=evaluation_gpus,
        all_training_and_evaluation_completed=True,
        plan_sha256=sha256(a.plan), summary_sha256=sha256(analysis / 'summary.json'),
        source_script_sha256=sha256(__file__)))
    print(json.dumps(dict(passed=True, validation_only=a.validate_only, paired_inputs=pair), indent=2))


if __name__ == '__main__':
    main()
