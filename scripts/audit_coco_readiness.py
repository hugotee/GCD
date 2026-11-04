#!/usr/bin/env python3
"""CPU-only taxonomy, prompt-length and optional full COCO2017 file audit."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--taxonomy', type=Path, default=ROOT / 'configs/datasets/coco80_reference.json')
p.add_argument('--coco-root', type=Path)
p.add_argument('--user-confirmed-not-prepared', action='store_true')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert not a.out.exists(), 'Refuse to overwrite a readiness snapshot'
assert bool(a.coco_root) != a.user_confirmed_not_prepared
os.environ['CUDA_VISIBLE_DEVICES'], os.environ['TOKENIZERS_PARALLELISM'] = '', 'false'
from transformers import AutoTokenizer
import torch
torch.set_num_threads(2)
runtime = read_runtime(a.runtime)
taxonomy = json.loads(a.taxonomy.read_text())
categories = taxonomy['categories']
assert len(categories) == 80 and [c['index'] for c in categories] == list(range(80))
assert [c['id'] for c in categories] == sorted({c['id'] for c in categories})
overlap = {x['coco_category_id'] for x in taxonomy['voc20_semantic_mapping'].values()}
assert len(overlap) == 20
tokenizer = AutoTokenizer.from_pretrained(runtime['bert_path'], local_files_only=True)
prompts = {}
for name, selected in [('all80', categories), ('first70', categories[:70]),
                       ('first40', categories[:40]), ('VOC_unadapted60', [c for c in categories if c['id'] not in overlap])]:
    caption = ''.join(c['name'] + '. ' for c in selected)
    tokens = tokenizer(caption, add_special_tokens=True, truncation=False)
    count = len(tokens['input_ids'])
    prompts[name] = dict(classes=len(selected), tokens_including_special=count, fits_256=count <= 256)
data = dict(prepared=False, user_confirmed_not_prepared=a.user_confirmed_not_prepared,
            supplied_root=str(a.coco_root) if a.coco_root else None)
if a.coco_root:
    splits = {}
    for split, expected_images in [('train2017', 118287), ('val2017', 5000)]:
        ann = a.coco_root / 'annotations' / f'instances_{split}.json'
        if not ann.is_file():
            splits[split] = dict(ready=False, missing_annotation=str(ann)); continue
        source = json.loads(ann.read_text())
        actual = {c['id']: c['name'] for c in source['categories']}
        assert actual == {c['id']: c['name'] for c in categories}, 'COCO category identity mismatch'
        missing = [i['file_name'] for i in source['images'] if not (a.coco_root / split / i['file_name']).is_file()]
        splits[split] = dict(ready=not missing and len(source['images']) == expected_images,
            images=len(source['images']), annotations=len(source['annotations']),
            missing_images=len(missing), missing_examples=missing[:10], annotation_sha256=sha256(ann))
    data.update(splits=splits, prepared=all(x['ready'] for x in splits.values()))
assert not torch.cuda.is_initialized()
write_json(a.out, dict(audit_completed=True, dataset=data, protocol_frozen=False,
    taxonomy_sha256=sha256(a.taxonomy), source_script_sha256=sha256(__file__),
    tokenizer_class=type(tokenizer).__name__, prompt_checks=prompts, cuda_initialized=False,
    incremental_training_started=False,
    note='VOC_unadapted60 means not supervised during VOC adaptation, not unseen in foundation pretraining.'))
print(a.out.read_text())
