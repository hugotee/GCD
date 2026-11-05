#!/usr/bin/env python3
"""Check all selected ranks communicate correctly before launching training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=90))
rank,world=dist.get_rank(),dist.get_world_size()
data=torch.full((1024,),float(rank+1),device='cuda')
dist.all_reduce(data)
assert torch.equal(data,torch.full_like(data,world*(world+1)/2))
a.out.mkdir(parents=True,exist_ok=True)
(a.out/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,rank=rank,world_size=world,
    device=torch.cuda.get_device_name(),all_reduce_exact=True),indent=2)+'\n')
dist.barrier();dist.destroy_process_group()
