#!/usr/bin/env python3
"""Persist a native frozen-model evaluation job and its source identity."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime, validate_gpus

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--experiment', type=Path, default=ROOT/'configs/experiments/voc15_5.json')
p.add_argument('--code-commit', required=True)
a = p.parse_args()
runtime = read_runtime(a.runtime)
assert len(validate_gpus(a.gpu, runtime['authorized_gpus'])) == 1
manifest = json.loads(Path(runtime['manifest']).read_text())
dataset = manifest.get('dataset', 'voc2007')
a.out.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
sources = ['src/gcd/cli.py','src/gcd/runtime.py','src/gcd/model.py','src/gcd/peft.py',
           'scripts/run.py','scripts/run_frozen_baseline.py','configs/model/gdino_t.py']
for name in sources:
    target = a.out/'source_snapshot'/name
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name,target)
write_json(a.out/'job_identity.json',dict(dataset=dataset,code_commit=a.code_commit,
    source_sha256={n:sha256(ROOT/n) for n in sources},manifest_sha256=sha256(runtime['manifest']),
    pretrained_sha256=runtime['pretrained_sha256'],physical_gpu=int(a.gpu)))
commands = [
    ('inference',[sys.executable,str(ROOT/'scripts/run.py'),'frozen','--runtime',str(a.runtime),
                  '--experiment',str(a.experiment),'--gpus',a.gpu,'--work-dir',str(a.out/'inference')]),
    ('metrics',[sys.executable,str(ROOT/('scripts/evaluate_coco.py' if dataset=='coco2017' else 'scripts/evaluate.py')),
        '--manifest',runtime['manifest'],'--predictions',str(a.out/'inference/predictions.bbox.json'),
        '--provenance',str(a.out/'inference/provenance.json'),'--model','B0','--out',str(a.out/'results')])]
for stage, command in commands:
    write_json(a.out/'job_status.json',dict(stage=stage,status='running'))
    with (a.out/f'{stage}.log').open('x') as log:
        result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(a.out/'job_status.json',dict(stage=stage,status='failed',returncode=result.returncode))
        raise SystemExit(result.returncode)
write_json(a.out/'job_complete.json',dict(passed=True,dataset=dataset,physical_gpu=int(a.gpu),
    evaluated_images=manifest['test']['images'],elapsed_seconds=time.perf_counter()-started,
    result_sha256=sha256(a.out/'results/summary.json')))
