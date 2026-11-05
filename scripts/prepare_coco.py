#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from gcd.coco import prepare_coco
p=argparse.ArgumentParser()
p.add_argument('--coco-root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--old-classes',type=int,choices=[40,70],default=70)
p.add_argument('--evaluation-only',action='store_true')
a=p.parse_args()
m=prepare_coco(a.coco_root,a.out,a.old_classes,a.evaluation_only)
print(dict(protocol=m['protocol'],phases={k:v['images'] for k,v in m['phases'].items()},validation_images=m['test']['images']))
