#!/usr/bin/env python3
"""Run the explicitly authorized four-GPU VOC baseline with gated checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gcd.protocol import sha256,write_json
from gcd.runtime import read_runtime,validate_gpus,validate_manifest,validate_experiment


def read(path):return json.loads(Path(path).read_text())


def check_training(path,world):
    audits=[read(path/f'audit_rank{rank}.json') for rank in range(world)]
    for rank,audit in enumerate(audits):
        assert audit['passed'] and audit['frozen_parameters_unchanged']
        assert read(path/f'initialization_rank{rank}.json')['passed']
    assert len({a['lora_sha256'] for a in audits})==1
    assert len({a['optimizer_steps'] for a in audits})==1
    assert read(path/'run_complete.json')['passed']
    return audits


def check_profile(audits):
    """AMP may skip synchronized updates; never count them as successful ones."""
    for audit in audits:
        assert audit['passed'] and audit['iterations']==200
        assert audit['expected_optimizer_steps']==200
        assert audit['optimizer_steps']+audit['amp_skipped_steps']==200
        assert 0<=audit['amp_skipped_steps']<=10
    assert len({a['optimizer_steps'] for a in audits})==1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--experiment',type=Path,default=ROOT/'configs/experiments/voc15_5_4gpu.json')
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--code-commit',required=True)
    p.add_argument('--continue-checked',action='store_true',help='Reuse only completed, fully audited stages; never restart an incomplete native run')
    a=p.parse_args()
    runtime=read_runtime(a.runtime);gpus=validate_gpus(a.gpus,runtime['authorized_gpus'])
    assert len(gpus)==4
    manifest=read(runtime['manifest']);validate_manifest(manifest)
    assert manifest.get('dataset','voc2007')=='voc2007'
    exp=read(a.experiment);validate_experiment(exp)
    assert exp['expected_world_size']==4 and exp['batch_per_gpu']==exp['accumulation']==1
    assert exp['effective_batch']==4 and exp['epochs']==12 and exp['num_queries']==900
    assert exp['protocol']==manifest['protocol']
    b0_source=ROOT/'outputs/foundation_v1/voc_B0/results/summary.json'
    b0=read(b0_source)
    assert b0['model']=='B0' and not b0['engineering_only']
    assert b0['provenance']['evaluation_ann_sha256']==manifest['test']['sha256']
    a.out=a.out.resolve()
    sources=list((ROOT/'src/gcd').glob('*.py'))+[ROOT/'scripts/run.py',Path(__file__).resolve(),
        ROOT/'scripts/evaluate.py',ROOT/'configs/model/gdino_t.py',a.experiment.resolve()]
    if a.continue_checked:
        identity=read(a.out/'job_identity.json')
        assert identity['manifest_sha256']==sha256(runtime['manifest']) and identity['experiment_sha256']==sha256(a.experiment)
        for source in sources:
            if source != Path(__file__).resolve():
                assert sha256(source)==identity['source_sha256'][str(source.relative_to(ROOT))]
        recovery=a.out/'driver_recovery';recovery.mkdir(exist_ok=False)
        shutil.copy2(a.out/'job_status.json',recovery/'previous_job_status.json')
        shutil.copy2(__file__,recovery/'run_voc_four_gpu.py')
        write_json(recovery/'identity.json',dict(code_commit=a.code_commit,driver_sha256=sha256(__file__),
            note='Record AMP skips and reuse completed audited native runs. Original snapshots and logs preserved.'))
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'results/B0').mkdir(parents=True)
        shutil.copy2(b0_source,a.out/'results/B0/summary.json')
        write_json(a.out/'B0_reuse.json',dict(source=str(b0_source),sha256=sha256(b0_source),
            note='Reuse the completed fresh 4952-image frozen run; no new B0 inference.'))
        for source in sources:
            target=a.out/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        write_json(a.out/'job_identity.json',dict(code_commit=a.code_commit,physical_gpus=gpus,
            world_size=4,effective_batch=4,manifest_sha256=sha256(runtime['manifest']),
            experiment_sha256=sha256(a.experiment),pretrained_sha256=runtime['pretrained_sha256'],
            source_sha256={str(s.relative_to(ROOT)):sha256(s) for s in sources},
            sampler_note='Native distributed sampler pads Task1 by 3 and Task2 by 2 image exposures per epoch.'))
    stages=[]
    for mode in ['smoke','profile']:
        for phase in ['task1','task2']:
            stages.append((f'{mode}_{phase}',mode,phase))
    stages += [('B1','train','task1'),('B2','train','task2'),
               ('B1_eval','eval','task1'),('B1_metrics','metrics','task1'),
               ('B2_eval','eval','task2'),('B2_metrics','metrics','task2')]
    records=[];started=time.perf_counter()
    for name,mode,phase in stages:
        if a.continue_checked and (a.out/name/'run_complete.json').exists():
            if mode in ['train','smoke','profile']:
                audits=check_training(a.out/name,4)
                if mode=='smoke':assert read(a.out/name/'reload_check.json')['predictions_exact']
                if mode=='profile':check_profile(audits)
            elif mode=='eval':assert read(a.out/name/'initialization_audit.json')['checkpoint_tensors_exact']
            records.append(dict(stage=name,passed=True,elapsed_seconds=None,reused_completed_stage=True))
            write_json(a.out/'milestones.json',records)
            continue
        if mode=='metrics':
            model='B1' if phase=='task1' else 'B2'
            command=[sys.executable,str(ROOT/'scripts/evaluate.py'),'--manifest',runtime['manifest'],
                '--predictions',str(a.out/f'{model}_eval/predictions.bbox.json'),
                '--provenance',str(a.out/f'{model}_eval/provenance.json'),'--model',model,
                '--out',str(a.out/'results'/model)]
        else:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            write_json(a.out/f'{name}_resources.json',dict(rows=rows,physical_gpus=gpus))
            for row in rows.strip().splitlines():
                gpu,memory,utilization=map(int,row.split(','))
                if gpu in gpus and (memory>512 or utilization>5):
                    raise RuntimeError(f'GPU{gpu} became busy before {name}; no other job stopped')
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                str(ROOT/'scripts/run.py'),mode,'--runtime',str(a.runtime),'--experiment',str(a.experiment),
                '--gpus',a.gpus,'--phase',phase,'--work-dir',str(a.out/name)]
            if mode in ['smoke','profile']:
                command+=['--steps','8' if mode=='smoke' else '200','--warmup','0' if mode=='smoke' else '20']
                if mode=='smoke':command+=['--verify']
                if phase=='task2':command+=['--init',str(a.out/f'{mode}_task1'/('iter_8.pth' if mode=='smoke' else 'iter_200.pth'))]
            elif name=='B2':command+=['--init',str(a.out/'B1/epoch_12.pth')]
            elif mode=='eval':command+=['--init',str(a.out/('B1' if phase=='task1' else 'B2')/'epoch_12.pth')]
        write_json(a.out/'job_status.json',dict(stage=name,status='running',command=command,completed=records))
        stage_started=time.perf_counter()
        env=dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=a.gpus)
        with (a.out/f'{name}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(a.out/'active_process.json',dict(stage=name,pid=process.pid,command=command))
            try:
                result=process.wait(timeout=8*3600)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        if result:
            write_json(a.out/'job_status.json',dict(stage=name,status='failed',returncode=result,completed=records))
            raise SystemExit(result)
        if mode in ['train','smoke','profile']:
            audits=check_training(a.out/name,4)
            if mode=='smoke':assert read(a.out/name/'reload_check.json')['predictions_exact']
            if mode=='profile':
                check_profile(audits)
        elif mode=='eval':
            assert read(a.out/name/'run_complete.json')['passed']
            assert read(a.out/name/'initialization_audit.json')['checkpoint_tensors_exact']
        elif mode=='metrics':assert not read(a.out/'results'/('B1' if phase=='task1' else 'B2')/'summary.json')['engineering_only']
        records.append(dict(stage=name,passed=True,elapsed_seconds=time.perf_counter()-stage_started))
        write_json(a.out/'milestones.json',records)
    b1=read(a.out/'results/B1/summary.json');b2=read(a.out/'results/B2/summary.json')
    summary=dict(passed=True,engineering_only=False,physical_gpus=gpus,world_size=4,effective_batch=4,
        epochs_per_stage=12,groups_B0=b0['groups'],groups_B1=b1['groups'],groups_B2=b2['groups'],
        old_forgetting={m:b1['groups']['old15'][m]-b2['groups']['old15'][m] for m in b1['groups']['old15']},
        new_adaptation={m:b2['groups']['new5'][m]-b1['groups']['new5'][m] for m in b1['groups']['new5']},
        elapsed_seconds=time.perf_counter()-started,stages=records,
        elapsed_note='Current controller invocation; recovered short checks are excluded.' if a.continue_checked else 'Whole controller invocation')
    write_json(a.out/'summary.json',summary)
    write_json(a.out/'job_complete.json',dict(passed=True,engineering_only=False,elapsed_seconds=time.perf_counter()-started))
    write_json(a.out/'job_status.json',dict(stage='complete',status='passed',completed=records))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
