#!/usr/bin/env python3
"""Summarize actual single-GPU measurements and a clearly separated time estimate."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import write_json

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=ROOT)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
stages = {}
for phase in ('task1', 'task2'):
    directory = a.root / 'outputs' / f'profile_{phase}'
    row = json.loads((directory / 'audit_rank0.json').read_text())
    assert row['passed'] and row['engineering_only']
    assert row['iterations'] == 200 and row['optimizer_steps'] == 100
    seconds_per_step = row['steady_seconds'] / (row['iterations'] - row['warmup'])
    stages[phase] = dict(microsteps=row['iterations'], warmup=row['warmup'],
        optimizer_updates=row['optimizer_steps'], amp_skips=row['amp_skipped_steps'],
        images_exposed=row['iterations'] * 2, steady_images=row['steady_images'],
        steady_seconds=row['steady_seconds'], seconds_per_microstep=seconds_per_step,
        images_per_second=row['steady_images'] / row['steady_seconds'],
        peak_allocated_gib=row['peak_allocated_bytes'] / 2**30,
        peak_reserved_gib=row['peak_reserved_bytes'] / 2**30,
        actual_profile_seconds_including_hash_and_checkpoint=row['elapsed_seconds'],
        loader_microsteps_per_epoch=row['loader_length'],
        estimated_12epoch_training_hours=seconds_per_step * row['loader_length'] * 12 / 3600,
        caveat='Training estimate excludes loading, checkpointing, evaluation and server load changes.')
for phase in ('task1', 'task2'):
    directory = a.root / 'outputs' / f'smoke_{phase}'
    assert json.loads((directory / 'audit_rank0.json').read_text())['passed']
    assert json.loads((directory / 'reload_check.json').read_text())['passed']
util_path = a.root / 'evidence/current/gpu_utilization.json'
util = json.loads(util_path.read_text()) if util_path.exists() else dict(samples=[])
samples = util['samples']
write_json(a.out, dict(passed=True, engineering_only=True, physical_gpu=2,
    batch_per_gpu=2, accumulation=2, effective_batch=4,
    formal_num_queries=900, train_short_edges=[640, 672, 704, 736, 768, 800],
    max_long_edge=1333, stages=stages,
    estimated_total_12epoch_training_hours=sum(s['estimated_12epoch_training_hours'] for s in stages.values()),
    sampled_gpu_utilization=dict(samples=len(samples),
        mean_percent=sum(s['utilization_percent'] for s in samples) / len(samples) if samples else None,
        max_percent=max((s['utilization_percent'] for s in samples), default=None),
        note='Read-only nvidia-smi samples include load/save/transition idle time; not a steady-state-only mean.'),
    new_formal_training_started=False))
print(a.out.read_text())
