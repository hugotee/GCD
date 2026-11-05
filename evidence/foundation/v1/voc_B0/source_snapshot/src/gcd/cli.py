"""Native MMEngine execution for frozen/eval, Single LoRA, smoke and profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from gcd.protocol import sha256, write_json
from gcd.runtime import read_runtime, validate_gpus, validate_manifest, validate_resume, validate_experiment

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['frozen', 'train', 'eval', 'smoke', 'profile'])
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--experiment', type=Path, default=ROOT / 'configs/experiments/voc15_5.json')
    p.add_argument('--phase', choices=['task1', 'task2'], default='task1')
    p.add_argument('--gpus', help='Physical IDs; defaults to the sole authorized GPU')
    p.add_argument('--work-dir', type=Path, required=True)
    exclusive = p.add_mutually_exclusive_group()
    exclusive.add_argument('--init', type=Path)
    exclusive.add_argument('--resume', type=Path)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--verify', action='store_true', help='Four training-side images; smoke only')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    runtime = read_runtime(a.runtime)
    if a.gpus is None:
        if len(runtime['authorized_gpus']) != 1:
            p.error('Select physical GPU IDs explicitly when several are authorized')
        a.gpus = str(runtime['authorized_gpus'][0])
    gpus = validate_gpus(a.gpus, runtime['authorized_gpus'])
    if a.mode == 'frozen' and (a.init or a.resume):
        p.error('B0 always starts from the configured W0; no --init or --resume')
    if a.mode == 'eval' and not a.init:
        p.error('Evaluation of B1/B2 requires --init checkpoint')
    if a.mode == 'train' and a.phase == 'task2' and not (a.init or a.resume):
        p.error('B2 must start from B1 via --init or resume its own checkpoint')
    if a.resume and a.mode != 'train':
        p.error('--resume is restricted to formal training')
    if a.verify and a.mode != 'smoke':
        p.error('--verify is restricted to smoke mode')
    if a.mode == 'profile' and not (100 <= a.steps <= 300 and 0 <= a.warmup < a.steps):
        p.error('Profile requires 100-300 microsteps and a smaller warmup')
    if a.mode == 'smoke' and not (4 <= a.steps <= 20):
        p.error('Smoke requires 4-20 steps; pass --steps 8 --warmup 0')
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # Explicit official source takes precedence over any old editable DGS package.
    upstream = Path(runtime['mmdet_root']).resolve()
    critical_hashes = {
        'models/detectors/grounding_dino.py': 'a79d59ca49fe66734273a6df4a338aeb1505a2ea54fc1acd164e6ef638ae3692',
        'models/dense_heads/grounding_dino_head.py': 'd90d306bf51b4065ab042aefed56e7e3337c0d9e7cdac126b249d4b38dd05650',
        'models/layers/transformer/grounding_dino_layers.py': 'cb212298c7598ab5482eec3b96ea2215b5145138118590f99d8510da1d4e7ece'}
    for name, digest in critical_hashes.items():
        if sha256(upstream / 'mmdet' / name) != digest:
            raise RuntimeError(f'Official upstream source mismatch: {name}')
    sys.path.insert(0, str(upstream))
    import torch
    from mmengine.config import Config
    from mmengine.dist import init_dist, get_rank, get_world_size, barrier
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    import mmdet
    if not Path(mmdet.__file__).resolve().is_relative_to(upstream):
        raise RuntimeError('A non-project MMDetection package was imported')
    register_all_modules()
    import gcd.model
    import gcd.hooks
    exp = json.loads(a.experiment.read_text())
    validate_experiment(exp)
    manifest = json.loads(Path(runtime['manifest']).read_text())
    engineering = a.mode in ['smoke', 'profile']
    validate_manifest(manifest, engineering)
    if sha256(runtime['pretrained_checkpoint']) != runtime['pretrained_sha256']:
        raise RuntimeError('W0 hash mismatch')
    initial = a.init or Path(runtime['pretrained_checkpoint'])
    if not initial.is_file():
        raise FileNotFoundError(initial)
    if a.init and a.mode in ['train', 'smoke', 'profile']:
        initial_state = torch.load(a.init, map_location='cpu')
        if a.phase == 'task2' and not any('lora_' in n for n in initial_state.get('state_dict', initial_state)):
            raise ValueError('Task2 requires the preceding Single LoRA checkpoint')
        if a.mode == 'train' and initial_state.get('meta', {}).get('gcd_engineering_only'):
            raise ValueError('Engineering checkpoints cannot initialize a formal baseline')
        del initial_state
    identity = dict(experiment_sha256=sha256(a.experiment),
        model_config_sha256=sha256(ROOT / 'configs/model/gdino_t.py'),
        manifest_sha256=sha256(runtime['manifest']), pretrained_sha256=runtime['pretrained_sha256'],
        model_code_sha256=sha256(ROOT / 'src/gcd/model.py'),
        peft_code_sha256=sha256(ROOT / 'src/gcd/peft.py'),
        hooks_code_sha256=sha256(ROOT / 'src/gcd/hooks.py'))
    run_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if a.resume:
        validate_resume(torch.load(a.resume, map_location='cpu'), a.phase, run_hash)
    if not a.dry_run:
        if not torch.cuda.is_available():
            raise RuntimeError('This execution path requires CUDA; CPU tests are separate')
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
            for line in rows.strip().splitlines():
                idx, memory, utilization = map(int, line.split(','))
                if idx in gpus and (memory > 512 or utilization > 5):
                    raise RuntimeError(f'GPU{idx} busy ({memory} MiB, {utilization}%); no job started')
    if len(gpus) > 1 and not a.dry_run:
        if int(os.environ.get('WORLD_SIZE', '0')) != len(gpus):
            raise ValueError('Use torchrun with one process per selected GPU')
        init_dist('pytorch', backend='nccl')
    rank, world = (0, len(gpus)) if a.dry_run else (get_rank(), get_world_size())
    work = a.work_dir.resolve()
    if rank == 0:
        if not a.resume:
            work.mkdir(parents=True, exist_ok=False)
        elif not work.is_dir():
            raise FileNotFoundError('Resume requires the existing work directory')
    barrier()
    phase = manifest['phases'][a.phase]
    model = Config.fromfile(str(ROOT / 'configs/model/gdino_t.py')).model
    for key in ('as_two_stage', 'share_pred_layer', 'num_pred_layer', 'train_cfg', 'test_cfg'):
        model.bbox_head.pop(key, None)
    model.update(type='SingleLoRAGroundingDINO', initial_checkpoint=str(initial),
        adapter=a.mode != 'frozen', lora=exp['lora'],
        train_label_ids=phase['expected_coco_dataset_label_ids'], expected_classes=phase['classes'])
    model.language_model.name = runtime['bert_path']
    model.encoder.num_cp, model.backbone.with_cp = 0, False
    model.num_queries = exp['num_queries']
    model.test_cfg.max_per_img = exp['max_per_image']
    meta = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities')
    trainpipe = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=.5), dict(type='RandomChoiceResize',
             scales=[(s, exp['max_long_edge']) for s in exp['train_short_edges']], keep_ratio=True),
        dict(type='PackDetInputs', meta_keys=meta + ('flip', 'flip_direction'))]
    testpipe = [dict(type='LoadImageFromFile'), dict(type='FixScaleResize',
        scale=(exp['eval_short_edge'], exp['max_long_edge']), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs', meta_keys=meta)]
    def loader(train=False, smoke=False):
        return dict(batch_size=exp['batch_per_gpu'] if train else 1,
            num_workers=exp['workers'], persistent_workers=exp['workers'] > 0,
            sampler=dict(type='DefaultSampler', shuffle=train), drop_last=False,
            dataset=dict(type='CocoDataset', ann_file=phase['train_ann'] if train else
                phase['smoke_eval_ann'] if smoke else manifest['test']['ann'],
                data_prefix=dict(img=(phase['image_root'] if train or smoke else manifest['test']['image_root']) + '/'),
                metainfo=dict(classes=tuple(phase['classes'] if train or smoke else manifest['classes'])),
                pipeline=trainpipe if train else testpipe, return_classes=True, test_mode=not train, filter_cfg=None))
    cfg = Config(dict(model=model, work_dir=str(work), default_scope='mmdet',
        gcd_phase=a.phase, gcd_experiment_hash=run_hash, engineering_only=engineering,
        randomness=dict(seed=exp['seed'], diff_rank_seed=False, deterministic=False),
        env_cfg=dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                     dist_cfg=dict(backend='nccl')), log_level='INFO', visualizer=None,
        log_processor=dict(type='LogProcessor', by_epoch=a.mode == 'train'),
        default_hooks=dict(logger=dict(type='LoggerHook', interval=20), timer=dict(type='IterTimerHook'),
                           sampler_seed=dict(type='DistSamplerSeedHook')),
        launcher='pytorch' if world > 1 else 'none'))
    if world > 1:
        cfg.model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True, broadcast_buffers=False)
    training = a.mode in ['train', 'smoke', 'profile']
    if training:
        cfg.train_dataloader = loader(True)
        cfg.optim_wrapper = dict(type='AmpOptimWrapper',
            loss_scale=dict(init_scale=exp['amp_initial_scale'], growth_interval=2000),
            accumulative_counts=exp['accumulation'], optimizer=dict(type='AdamW',
                lr=exp['learning_rate'], weight_decay=exp['weight_decay']),
            clip_grad=dict(max_norm=exp['clip_grad'], norm_type=2))
        cfg.train_cfg = (dict(type='EpochBasedTrainLoop', max_epochs=exp['epochs'], val_interval=exp['epochs'] + 1)
                         if a.mode == 'train' else dict(type='IterBasedTrainLoop', max_iters=a.steps))
        cfg.param_scheduler = ([dict(type='MultiStepLR', by_epoch=True,
            milestones=exp['lr_milestones'], gamma=exp['lr_gamma'])] if a.mode == 'train' else [])
        cfg.default_hooks.checkpoint = dict(type='CheckpointHook', by_epoch=a.mode == 'train',
            interval=1 if a.mode == 'train' else a.steps, save_last=True, max_keep_ckpts=2)
        cfg.custom_hooks = [dict(type='BaselineAuditHook', profile=engineering,
            warmup=a.warmup if a.mode == 'profile' else 0)]
    if a.resume:
        cfg.load_from, cfg.resume = str(a.resume.resolve()), True
    if a.mode in ['frozen', 'eval'] or a.verify:
        cfg.test_dataloader, cfg.test_cfg = loader(smoke=a.verify), dict(type='TestLoop')
        cfg.test_evaluator = (dict(type='DumpDetResults', out_file_path=str(work / 'engineering_predictions.pkl'))
            if a.verify else dict(type='CocoMetric', ann_file=manifest['test']['ann'], metric='bbox',
                classwise=True, outfile_prefix=str(work / 'predictions')))
    provenance = dict(mode=a.mode, engineering_only=engineering, config_only=a.dry_run,
        phase=a.phase, physical_gpus=gpus, classes=manifest['classes'] if a.mode in ['eval', 'frozen'] else phase['classes'],
        initial_checkpoint=str(initial), initial_checkpoint_sha256=sha256(initial),
        pretrained_sha256=runtime['pretrained_sha256'], experiment_sha256=sha256(a.experiment),
        evaluation_ann_sha256=manifest['test']['sha256'], num_queries=model.num_queries,
        mmdet_version=mmdet.__version__, mmdet_source=str(Path(mmdet.__file__).resolve()),
        seed=exp['seed'], resumed_from=str(a.resume) if a.resume else None)
    provenance.update(identity, run_identity_sha256=run_hash)
    if rank == 0:
        cfg.dump(str(work / 'resolved_config.py'))
        write_json(work / ('resume_provenance.json' if a.resume else 'provenance.json'), provenance)
    if not a.dry_run:
        runner = Runner.from_cfg(cfg)
        if training:
            runner.train()
        else:
            runner._init_model_weights()
        if a.mode in ['eval', 'frozen'] or a.verify:
            runner.test()
            if a.verify:
                import pickle
                first = pickle.loads((work / 'engineering_predictions.pkl').read_bytes()) if rank == 0 else None
                unwrap = gcd.hooks.unwrap
                checkpoint = torch.load(work / f'iter_{a.steps}.pth', map_location='cpu')
                unwrap(runner.model).load_state_dict(checkpoint['state_dict'], strict=True)
                runner.test()
                if rank == 0:
                    second = pickle.loads((work / 'engineering_predictions.pkl').read_bytes())
                    assert len(first) == len(second) == 4
                    for x, y in zip(first, second):
                        assert x['img_id'] == y['img_id']
                        for key in ('bboxes', 'scores', 'labels'):
                            assert torch.equal(x['pred_instances'][key], y['pred_instances'][key])
                    write_json(work / 'reload_check.json', dict(passed=True, predictions_exact=True, engineering_only=True))
            elif rank == 0:
                test = json.loads(Path(manifest['test']['ann']).read_text())
                provenance['evaluated_image_ids'] = [i['id'] for i in test['images']]
                provenance['evaluation_completed'] = True
                write_json(work / 'provenance.json', provenance)
        if rank == 0:
            write_json(work / 'run_complete.json', dict(passed=True, mode=a.mode, engineering_only=engineering))
    barrier()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
