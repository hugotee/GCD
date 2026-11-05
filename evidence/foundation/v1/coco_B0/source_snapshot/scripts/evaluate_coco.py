#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gcd.protocol import sha256,write_json
from gcd.runtime import validate_manifest
from gcd.coco_metrics import evaluate_coco
p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--predictions',type=Path,required=True)
p.add_argument('--provenance',type=Path,required=True)
p.add_argument('--model',choices=['B0','B1','B2'],required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--split-at',type=int,choices=[40,70])
a=p.parse_args()
m=json.loads(a.manifest.read_text());validate_manifest(m,training=False)
assert m['dataset']=='coco2017'
pr=json.loads(a.provenance.read_text());gt=json.loads(Path(m['test']['ann']).read_text())
assert not m['engineering_only'] and not pr['engineering_only'] and not pr['config_only']
assert pr['evaluation_completed'] and pr['model_initialization_verified']
assert pr['classes']==m['classes'] and pr['evaluation_ann_sha256']==sha256(m['test']['ann'])==m['test']['sha256']
assert len(gt['images'])==5000 and sorted(pr['evaluated_image_ids'])==sorted(i['id'] for i in gt['images'])
if a.model=='B0':assert pr['mode']=='frozen' and pr['initial_checkpoint_sha256']==pr['pretrained_sha256']
else:assert pr['mode']=='eval' and pr['phase']==('task1' if a.model=='B1' else 'task2')
r=evaluate_coco(gt,json.loads(a.predictions.read_text()),m['category_ids'],a.split_at or m.get('num_old_classes'))
a.out.mkdir(parents=True,exist_ok=False)
rows=r.pop('per_class')
with (a.out/'per_class_ap.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
r.update(dataset='coco2017',model=a.model,engineering_only=False,prediction_sha256=sha256(a.predictions),provenance=pr)
write_json(a.out/'summary.json',r);print(json.dumps(r['groups'],indent=2))
