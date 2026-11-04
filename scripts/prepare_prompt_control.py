#!/usr/bin/env python3
"""Register data, source and budget hashes before the prompt control is run."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime, validate_manifest

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--init', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert not a.out.exists(), 'Registered plans are immutable'
r = read_runtime(a.runtime)
m = json.loads(Path(r['manifest']).read_text())
validate_manifest(m)
s = ROOT / 'configs/experiments/task2_prompt_control_v1.json'
spec = json.loads(s.read_text())
assert sha256(a.init) == spec['initial_checkpoint_sha256']
train = json.loads(Path(m['phases']['task2']['train_ann']).read_text())
assert len(train['images']) == spec['expected_train_images']
assert len(train['annotations']) == spec['expected_train_annotations']
assert {x['category_id'] for x in train['annotations']} == set(range(16, 21))
paths = ['scripts/train_prompt_control.py', 'src/gcd/control_hooks.py', 'src/gcd/prompt_control.py',
         'src/gcd/cli.py', 'src/gcd/model.py', 'src/gcd/hooks.py', 'src/gcd/peft.py',
         spec['base_experiment'], 'configs/model/gdino_t.py']
write_json(a.out, dict(registered_before_training=True, diagnostic_short_training=True,
    control_spec=spec, control_spec_sha256=sha256(s), initial_checkpoint_sha256=sha256(a.init),
    manifest_sha256=sha256(r['manifest']), training_annotation_sha256=sha256(m['phases']['task2']['train_ann']),
    sorted_training_image_ids=sorted(i['id'] for i in train['images']),
    source_sha256={p: sha256(ROOT / p) for p in paths},
    decision_rule='Compare fixed final checkpoints only; no test-based threshold, seed or checkpoint selection.',
    hypothesis='Including old-class tokens without old GT may increase old-class suppression.',
    falsification='If removing old tokens does not improve old retention at comparable new adaptation, prompt-negative supervision alone is insufficient.',
    caveats=['A single-seed one-epoch screen, not the 12-epoch baseline.',
             'Prompt tokens also change encoder/query conditioning and matching; this is not a pure loss-channel intervention.',
             'No hidden old annotations used for optimization.']))
print(a.out)
