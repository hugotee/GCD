import copy
import io
import unittest

import torch
from torch import nn

from gcd.peft import LoRALinear, inject_encoder_ffn, audit_trainables


class LoRATests(unittest.TestCase):
    def test_zero_initialized_adapter_and_mode_switch_do_not_merge(self):
        torch.manual_seed(10)
        layer = LoRALinear(8, 4, rank=2, alpha=4)
        x = torch.randn(3, 8)
        base = nn.functional.linear(x, layer.weight, layer.bias)
        self.assertTrue(torch.equal(base, layer(x)))
        original = layer.weight.detach().clone()
        layer.eval()
        layer.train()
        self.assertTrue(torch.equal(original, layer.weight))
        layer(x).square().sum().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertIsNone(layer.bias.grad)
        self.assertGreater(float(layer.lora_B.grad.abs().sum()), 0)

    def test_target_scope_excludes_heads_fusion_and_backbone(self):
        model = nn.Module()
        model.encoder = nn.Module()
        class FrameworkLinear(nn.Linear):
            pass  # MMCV's Linear is also a subclass, not exact nn.Linear.
        for group in ['layers', 'text_layers']:
            layer = nn.Module()
            layer.ffn = nn.Sequential(FrameworkLinear(4, 8), nn.ReLU(), FrameworkLinear(8, 4))
            setattr(model.encoder, group, nn.ModuleList([layer]))
        model.encoder.fusion_layers = nn.ModuleList([nn.Linear(4, 4)])
        model.backbone = nn.Linear(4, 4)
        model.bbox_head = nn.Linear(4, 4)
        paths = inject_encoder_ffn(model, rank=2)
        self.assertEqual(len(paths), 4)
        self.assertEqual(audit_trainables(model)['trainable'], 96)
        self.assertFalse(model.bbox_head.weight.requires_grad)
        self.assertFalse(model.encoder.fusion_layers[0].weight.requires_grad)

    def test_strict_checkpoint_and_optimizer_resume_preserve_next_update(self):
        torch.manual_seed(7)
        layer = LoRALinear(5, 3, rank=2)
        optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.01)
        x, y = torch.randn(4, 5), torch.randn(4, 3)
        def step(model, opt):
            opt.zero_grad()
            (model(x) - y).square().mean().backward()
            opt.step()
        step(layer, optimizer)
        buf = io.BytesIO()
        torch.save(dict(state_dict=layer.state_dict(), optimizer=optimizer.state_dict()), buf)
        buf.seek(0)
        state = torch.load(buf, map_location='cpu')
        restored = LoRALinear(5, 3, rank=2)
        restored.load_state_dict(state['state_dict'], strict=True)
        opt2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.01)
        opt2.load_state_dict(copy.deepcopy(state['optimizer']))
        step(layer, optimizer)
        step(restored, opt2)
        for a, b in zip(layer.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))
        with self.assertRaises(RuntimeError):
            LoRALinear(5, 3, rank=3).load_state_dict(state['state_dict'], strict=True)
