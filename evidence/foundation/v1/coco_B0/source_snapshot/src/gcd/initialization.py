"""Check the actual initialized model, not only the checkpoint filename."""
import torch
from gcd.protocol import sha256

IGNORED_PRETRAINED_KEYS = {
    'language_model.language_backbone.body.model.pooler.dense.weight',
    'language_model.language_backbone.body.model.pooler.dense.bias',
    'language_model.language_backbone.body.model.embeddings.position_ids'}


def verify_loaded_checkpoint(model, path):
    model=model.module if hasattr(model,'module') else model
    checkpoint=torch.load(path,map_location='cpu')
    expected=checkpoint.get('state_dict',checkpoint)
    actual=model.state_dict()
    unused=set(expected)-set(actual)
    missing=set(actual)-set(expected)
    adapter_checkpoint=any('lora_' in n for n in expected)
    allowed_missing=set() if adapter_checkpoint else {n for n in actual if n.endswith(('lora_A','lora_B'))}
    if not unused <= IGNORED_PRETRAINED_KEYS or missing != allowed_missing:
        raise RuntimeError('Initialized model/checkpoint tensor keys differ')
    if any(not torch.equal(actual[n].detach().cpu(),expected[n]) for n in set(actual)&set(expected)):
        raise RuntimeError('Initialized model tensors differ from the requested checkpoint')
    if any(bool(actual[n].count_nonzero()) for n in missing if n.endswith('lora_B')):
        raise RuntimeError('New LoRA B must be zero when initializing from W0')
    return dict(passed=True,checkpoint_tensors_exact=True,checkpoint_sha256=sha256(path),
        compared_tensors=len(set(actual)&set(expected)),new_adapter_tensors=len(missing),
        ignored_checkpoint_keys=sorted(unused))
