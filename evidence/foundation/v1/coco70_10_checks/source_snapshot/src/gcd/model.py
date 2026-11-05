"""Native official MMDetection GDINO plus a project-local single adapter."""
import hashlib

import torch
from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.registry import MODELS

from gcd.peft import inject_encoder_ffn, audit_trainables


@MODELS.register_module()
class NoDenoisingGroundingDINOHead(GroundingDINOHead):
    """Fix v3.3.0's None metadata access; all native matching/losses unchanged."""
    @staticmethod
    def split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta):
        if dn_meta is None:
            return all_layers_cls_scores, all_layers_bbox_preds, None, None
        return GroundingDINOHead.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, dn_meta)


class EmptyDenoising(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, samples):
        ref = samples[0].gt_instances.bboxes
        return (ref.new_empty((len(samples), 0, self.dim)),
                ref.new_empty((len(samples), 0, 4)), None, None)


@MODELS.register_module()
class SingleLoRAGroundingDINO(GroundingDINO):
    def __init__(self, initial_checkpoint, adapter=True, lora=None,
                 train_label_ids=None, expected_classes=None, **kwargs):
        self.initial_checkpoint = initial_checkpoint
        self.train_label_ids = train_label_ids
        self.expected_classes = expected_classes
        self.adapter_enabled = adapter
        kwargs['bbox_head']['type'] = 'NoDenoisingGroundingDINOHead'
        # Official DINO constructs this interface unconditionally. It is replaced
        # immediately below; no denoising queries/loss are used in this baseline.
        if kwargs.get('dn_cfg') is None:
            kwargs['dn_cfg'] = dict(label_noise_scale=.5, box_noise_scale=1.,
                                    group_cfg=dict(dynamic=True, num_dn_queries=0))
        super().__init__(**kwargs)
        self.dn_query_generator = EmptyDenoising(self.embed_dims)
        self.requires_grad_(False)
        self.lora_paths = inject_encoder_ffn(self, **(lora or {})) if adapter else []
        if adapter and len(self.lora_paths) != 24:
            raise RuntimeError(f'Expected 24 FFN LoRA layers, got {len(self.lora_paths)}')

    def init_weights(self):
        if self._is_init:
            return
        super().init_weights()
        checkpoint = torch.load(self.initial_checkpoint, map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        is_adapter = any('lora_' in k for k in state)
        if is_adapter:
            self.load_state_dict(state, strict=True)
        else:
            missing, extra = self.load_state_dict(state, strict=False)
            lora_names = {n for n, _ in self.named_parameters() if n.endswith(('lora_A', 'lora_B'))}
            allowed = {'language_model.language_backbone.body.model.pooler.dense.weight',
                       'language_model.language_backbone.body.model.pooler.dense.bias',
                       'language_model.language_backbone.body.model.embeddings.position_ids'}
            if set(missing) != lora_names or not set(extra).issubset(allowed):
                raise RuntimeError(f'Incomplete pretrained detector: missing={missing}, extra={extra}')
            for name, param in self.named_parameters():
                if name.endswith('lora_B'):
                    torch.nn.init.zeros_(param)
        if self.adapter_enabled:
            audit_trainables(self)

    def loss(self, batch_inputs, batch_data_samples):
        for sample in batch_data_samples:
            if list(sample.text) != self.expected_classes:
                raise ValueError('Training prompt differs from the frozen phase protocol')
            if not set(sample.gt_instances.labels.tolist()).issubset(self.train_label_ids):
                raise ValueError('Old/future annotations leaked into current-phase supervision')
        return super().loss(batch_inputs, batch_data_samples)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.language_model.eval()
        return self


def frozen_digest(model):
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            h.update(name.encode())
            h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
