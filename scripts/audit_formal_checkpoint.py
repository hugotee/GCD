#!/usr/bin/env python3
"""Check saved formal weights, optimizer, scheduler and per-rank audit identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gcd.initialization import IGNORED_PRETRAINED_KEYS
from gcd.protocol import sha256,write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work-dir',type=Path,required=True)
p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--phase',choices=['task1','task2'],required=True)
p.add_argument('--world-size',type=int,default=4)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
checkpoint=a.work_dir/'epoch_12.pth'
state=torch.load(checkpoint,map_location='cpu')
pretrained=torch.load(a.pretrained,map_location='cpu')
base=pretrained.get('state_dict',pretrained)
provenance=json.loads((a.work_dir/'provenance.json').read_text())
assert {'state_dict','optimizer','param_schedulers','meta'}<=state.keys()
assert state['meta']['epoch']==12 and state['meta']['gcd_phase']==a.phase
assert not state['meta']['gcd_engineering_only']
assert state['meta']['gcd_experiment_hash']==provenance['run_identity_sha256']
audits=[json.loads((a.work_dir/f'audit_rank{rank}.json').read_text()) for rank in range(a.world_size)]
assert all(x['passed'] and not x['engineering_only'] for x in audits)
assert len({x['lora_sha256'] for x in audits})==len({x['optimizer_steps'] for x in audits})==1
assert all(x['iterations']==state['meta']['iter']==x['loader_length']*12 for x in audits)
saved=state['state_dict'];shared=set(base)&set(saved)
assert set(base)-set(saved)<=IGNORED_PRETRAINED_KEYS
assert len(shared)==900 and len(saved)==948
assert all(torch.equal(base[n],saved[n]) for n in shared)
assert all(bool(torch.isfinite(t).all()) for t in saved.values())
h=hashlib.sha256()
for name,value in saved.items():
    if name.endswith(('lora_A','lora_B')):
        h.update(name.encode());h.update(value.contiguous().numpy().tobytes())
assert h.hexdigest()==audits[0]['lora_sha256']
optimizer=state['optimizer']
assert len(optimizer['state'])==48 and optimizer['loss_scaler'] and state['param_schedulers']
assert all(abs(g['lr']-1e-5)<1e-12 for g in optimizer['param_groups'])
for values in optimizer['state'].values():
    assert all(bool(torch.isfinite(values[n]).all()) for n in ['exp_avg','exp_avg_sq'])
write_json(a.out,dict(passed=True,engineering_only=False,phase=a.phase,checkpoint=str(checkpoint),
    checkpoint_sha256=sha256(checkpoint),pretrained_sha256=sha256(a.pretrained),
    base_tensors_exact=900,saved_model_tensors=948,all_model_tensors_finite=True,
    checkpoint_adapter_matches_all_ranks=True,lora_sha256=h.hexdigest(),
    epoch=12,microsteps_per_rank=state['meta']['iter'],optimizer_updates=audits[0]['optimizer_steps'],
    amp_skips=audits[0]['amp_skipped_steps'],optimizer_parameter_states=48,
    optimizer_moments_finite=True,amp_scaler_state_present=True,scheduler_state_present=True,
    final_learning_rates=[g['lr'] for g in optimizer['param_groups']],audit_code_sha256=sha256(__file__)))
print(a.out)
