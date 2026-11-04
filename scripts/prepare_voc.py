#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gcd.protocol import prepare

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voc-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    m = prepare(args.voc_root, args.out)
    print({k: (v['images'], v['annotations']) for k, v in m['phases'].items()})
