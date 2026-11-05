"""An unmerged, single LoRA adapter with explicit trainable-parameter auditing."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Keep weight/bias state names compatible with ordinary Linear checkpoints."""

    def __init__(self, in_features, out_features, rank=16, alpha=32.0,
                 dropout=0.0, bias=True, device=None, dtype=None):
        if rank <= 0 or not 0 <= dropout < 1 or alpha <= 0:
            raise ValueError("Invalid LoRA rank, alpha or dropout")
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(self.weight.new_empty(rank, in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        # Two matmuls deliberately preserve the historical adapter arithmetic.
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + delta * self.scaling


def inject_encoder_ffn(model, rank=16, alpha=32.0, dropout=0.0):
    """Only vision/text encoder FFN Linear layers; no fusion, backbone or head."""
    model.requires_grad_(False)
    paths = []
    for name, layer in list(model.named_modules()):
        if (isinstance(layer, nn.Linear) and not isinstance(layer, LoRALinear) and
                name.startswith(('encoder.layers.', 'encoder.text_layers.')) and
                '.ffn.' in name):
            replacement = LoRALinear(layer.in_features, layer.out_features, rank,
                                     alpha, dropout, layer.bias is not None,
                                     layer.weight.device, layer.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(layer.weight)
                if layer.bias is not None:
                    replacement.bias.copy_(layer.bias)
            parent, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), attr, replacement)
            paths.append(name)
    if not paths:
        raise RuntimeError("No encoder FFN Linear matched; inspect the installed model")
    audit_trainables(model)
    return paths


def audit_trainables(model):
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not n.endswith(('lora_A', 'lora_B')) for n in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    return dict(trainable=sum(trainable.values()),
                frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                names=trainable)
