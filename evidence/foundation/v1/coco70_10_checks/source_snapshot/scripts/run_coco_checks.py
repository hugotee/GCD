#!/usr/bin/env python3
"""Run short native COCO training checks, then estimate full training cost."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gcd.protocol import sha256,write_json
from gcd.runtime import read_runtime,validate_manifest,validate_gpus

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--experiment',type=Path,required=True)
p.add_argument('--gpu',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--code-commit',required=True)
a=p.parse_args()
runtime=read_runtime(a.runtime)
assert len(validate_gpus(a.gpu,runtime['authorized_gpus']))==1
manifest=json.loads(Path(runtime['manifest']).read_text());validate_manifest(manifest)
assert manifest['dataset']=='coco2017' and not manifest['engineering_only']
exp=json.loads(a.experiment.read_text())
assert exp['protocol']==manifest['protocol']
a.out.mkdir(parents=True,exist_ok=False)
sources=list((ROOT/'src/gcd').glob('*.py'))+[ROOT/'scripts/run.py',Path(__file__).resolve(),
    ROOT/'configs/model/gdino_t.py',a.experiment.resolve(),ROOT/'configs/datasets/coco80_reference.json']
for source in sources:
    destination=a.out/'source_snapshot'/source.relative_to(ROOT)
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
write_json(a.out/'job_identity.json',dict(code_commit=a.code_commit,protocol=manifest['protocol'],
    physical_gpu=int(a.gpu),manifest_sha256=sha256(runtime['manifest']),experiment_sha256=sha256(a.experiment),
    source_sha256={str(s.relative_to(ROOT)):sha256(s) for s in sources},engineering_only=True))
started=time.perf_counter()
for mode in ['smoke','profile']:
    for phase in ['task1','task2']:
        name=f'{mode}_{phase}'
        command=[sys.executable,str(ROOT/'scripts/run.py'),mode,'--runtime',str(a.runtime),
            '--experiment',str(a.experiment),'--gpus',a.gpu,'--phase',phase,
            '--work-dir',str(a.out/name),'--steps','8' if mode=='smoke' else '200',
            '--warmup','0' if mode=='smoke' else '20']
        if mode=='smoke':command+=['--verify']
        if phase=='task2':command+=['--init',str(a.out/f'{mode}_task1'/('iter_8.pth' if mode=='smoke' else 'iter_200.pth'))]
        write_json(a.out/'job_status.json',dict(stage=name,status='running',command=command))
        with (a.out/f'{name}.log').open('x') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(a.out/'job_status.json',dict(stage=name,status='failed',returncode=result.returncode))
            raise SystemExit(result.returncode)
        for filename in ['audit_rank0.json','initialization_rank0.json','run_complete.json']+(['reload_check.json'] if mode=='smoke' else []):
            assert json.loads((a.out/name/filename).read_text())['passed'],filename

stages={}
for phase in ['task1','task2']:
    row=json.loads((a.out/f'profile_{phase}/audit_rank0.json').read_text())
    assert row['engineering_only'] and row['iterations']==200
    seconds=row['steady_seconds']/(row['iterations']-row['warmup'])
    stages[phase]=dict(training_images=manifest['phases'][phase]['images'],
        microsteps=row['iterations'],warmup=row['warmup'],optimizer_updates=row['optimizer_steps'],amp_skips=row['amp_skipped_steps'],
        seconds_per_microstep=seconds,images_per_second=row['steady_images']/row['steady_seconds'],
        peak_allocated_gib=row['peak_allocated_bytes']/2**30,peak_reserved_gib=row['peak_reserved_bytes']/2**30,
        loader_microsteps_per_epoch=row['loader_length'],
        estimated_training_hours=seconds*row['loader_length']*exp['epochs']/3600)
summary=dict(passed=True,engineering_only=True,dataset='coco2017',protocol=manifest['protocol'],
    physical_gpu=int(a.gpu),batch_per_gpu=exp['batch_per_gpu'],accumulation=exp['accumulation'],
    num_queries=exp['num_queries'],estimated_epochs_per_phase=exp['epochs'],stages=stages,
    estimated_total_training_hours=sum(s['estimated_training_hours'] for s in stages.values()),
    caveat='Estimate from 200-microstep synchronized profiles; excludes loading, saving, evaluation and load changes.',
    formal_training_started=False,elapsed_seconds=time.perf_counter()-started)
write_json(a.out/'profile_summary.json',summary)
write_json(a.out/'job_complete.json',dict(passed=True,engineering_only=True,physical_gpu=int(a.gpu),elapsed_seconds=time.perf_counter()-started))
write_json(a.out/'job_status.json',dict(stage='complete',status='passed'))
print(json.dumps(summary,indent=2))
