#!/usr/bin/env python3
"""Freeze a training-side probe selection before any model-output inspection."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gcd.diagnostics import choose_probe, image_stratum
from gcd.protocol import convert_voc, sha256, write_json

p = argparse.ArgumentParser()
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise FileExistsError('Refuse to replace the registered selection')
m = json.loads(a.manifest.read_text())
data = convert_voc(m['voc_root'], 'trainval')
ids = choose_probe(data)
annotations = defaultdict(list)
for row in data['annotations']:
    annotations[row['image_id']].append(row)
names = {i['id']: i['file_name'] for i in data['images']}
write_json(a.out, dict(diagnostic_only=True, split='VOC2007 trainval', seed=20261104,
    manifest_sha256=sha256(a.manifest), selected_image_ids=ids,
    selected_filenames=[names[i] for i in ids],
    strata=dict(Counter(image_stratum(annotations[i]) for i in ids)),
    old_instances_per_class=dict(Counter(x['category_id'] for i in ids for x in annotations[i]
                                        if x['category_id'] <= 15 and not x['iscrowd'])),
    planned_models=['B0', 'B1', 'B2'], planned_prompts=['old15', 'all20'],
    planned_num_queries=900, planned_short_edge=800, planned_max_long_edge=1333,
    physical_gpu=2, status='not_run_GPU2_busy',
    note='Selected using training annotations and fixed seed only; not a formal evaluation subset.'))
print(a.out)
