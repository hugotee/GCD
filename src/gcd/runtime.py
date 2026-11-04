"""Runtime guards and configuration checks without importing CUDA/PyTorch."""
import json
from pathlib import Path

UPSTREAM_COMMIT = '44ebd17b145c2372c4b700bfb9cb20dbd28ab64a'


def read_runtime(path):
    data = json.loads(Path(path).read_text())
    required = {'pretrained_checkpoint', 'pretrained_sha256', 'bert_path',
                'mmdet_root', 'manifest', 'authorized_gpus'}
    if required - data.keys():
        raise ValueError(f'Missing runtime keys: {required - data.keys()}')
    return data


def validate_gpus(value, authorized):
    gpus = [int(s) for s in value.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= set(authorized):
        raise ValueError(f'GPU selection {gpus} outside authorized physical IDs {authorized}')
    return gpus


def validate_experiment(data):
    if data['lora_targets'] != 'vision_and_text_encoder_ffn_only':
        raise ValueError('This baseline implements encoder FFN LoRA only')
    if any(data[key] for key in ('denoising', 'replay', 'distillation')):
        raise ValueError('DN, replay and KD are not implemented in this baseline')
    if any(data[key] <= 0 for key in ('batch_per_gpu', 'accumulation', 'epochs', 'num_queries')):
        raise ValueError('Batch, accumulation, epochs and queries must be positive')


def validate_manifest(manifest, engineering=False):
    from gcd.protocol import CLASSES, sha256
    if not manifest.get('passed') or not manifest.get('protocol_frozen'):
        raise ValueError('Protocol has not been verified/frozen')
    if manifest.get('engineering_only') and not engineering:
        raise ValueError('Synthetic/engineering data cannot enter a formal run')
    if manifest['classes'] != list(CLASSES):
        raise ValueError('Unexpected class order')
    for p in manifest['phases'].values():
        if sha256(p['train_ann']) != p['train_sha256']:
            raise ValueError('Training annotation hash changed')
    if sha256(manifest['test']['ann']) != manifest['test']['sha256']:
        raise ValueError('Test annotation hash changed')


def validate_resume(checkpoint, phase, config_hash):
    """Called after trusted local torch.load; don't accept weights-only resumes."""
    if not {'state_dict', 'optimizer', 'param_schedulers', 'meta'} <= checkpoint.keys():
        raise ValueError('Resume requires model, optimizer, scheduler and loop metadata')
    if checkpoint['meta'].get('gcd_phase') != phase:
        raise ValueError('Cannot resume a different phase; use --init for B1 -> B2')
    if checkpoint['meta'].get('gcd_experiment_hash') != config_hash:
        raise ValueError('Experiment configuration differs from the resumed run')
    if checkpoint['meta'].get('gcd_engineering_only'):
        raise ValueError('An engineering checkpoint cannot resume formal training')
