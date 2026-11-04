#!/usr/bin/env python3
"""Bounded, read-only nvidia-smi sampling for one authorized physical GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--until', type=Path, required=True)
p.add_argument('--seconds', type=int, default=300)
a = p.parse_args()
assert a.gpu in json.loads(a.runtime.read_text())['authorized_gpus']
rows, start = [], time.monotonic()
while time.monotonic() - start < a.seconds:
    value = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
    util, memory = map(int, value.strip().split(','))
    rows.append(dict(at_utc=datetime.now(timezone.utc).isoformat(), physical_gpu=a.gpu,
                     utilization_percent=util, memory_mib=memory))
    a.out.write_text(json.dumps(dict(samples=rows, completed=a.until.is_file()), indent=2) + '\n')
    if a.until.is_file():
        break
    time.sleep(2)
