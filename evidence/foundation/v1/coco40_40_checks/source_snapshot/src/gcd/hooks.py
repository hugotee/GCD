"""Training invariants and actual CUDA throughput; no automatic job launcher."""
import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from gcd.model import frozen_digest
from gcd.peft import audit_trainables
from gcd.protocol import write_json


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class BaselineAuditHook(Hook):
    priority = 'ABOVE_NORMAL'

    def __init__(self, warmup=20, profile=False):
        self.warmup, self.profile = warmup, profile

    def before_train(self, runner):
        model = unwrap(runner.model)
        from gcd.initialization import verify_loaded_checkpoint
        checkpoint=runner.cfg.get('load_from') or model.initial_checkpoint
        initialization=verify_loaded_checkpoint(model,checkpoint)
        write_json(Path(runner.work_dir)/f'initialization_rank{get_rank()}.json',initialization)
        self.rank, self.world = get_rank(), get_world_size()
        self.base = frozen_digest(model)
        self.trainables = audit_trainables(model)
        self.initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.steps, self.seen, self.rows, self.grads = 0, 0, [], []
        self.peak_allocated = self.peak_reserved = 0
        self.started = time.perf_counter()
        self.step_handle = runner.optim_wrapper.optimizer.register_step_post_hook(
            lambda *args: setattr(self, 'steps', self.steps + 1))
        def observe(g):
            if self.seen < 3:
                self.grads.append(dict(iteration=runner.iter,
                    finite=bool(torch.isfinite(g).all()), nonzero=bool((g != 0).any())))
        selected = {'encoder.layers.0.ffn.layers.0.0.lora_B',
                    'encoder.text_layers.0.ffn.layers.0.0.lora_B'}
        self.handles = [p.register_hook(observe) for n, p in model.named_parameters() if n in selected]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.previous_end = time.perf_counter()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.start = time.perf_counter()
        self.count = len(data_batch['inputs'])

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.profile:
            torch.cuda.synchronize()
        end = time.perf_counter()
        loss = float(outputs['loss'].detach())
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite detection loss')
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())
        self.rows.append(dict(iteration=runner.iter + 1, seconds=end - self.previous_end,
                              data_wait_seconds=self.start - self.previous_end,
                              images=self.count, loss=loss, optimizer_steps=self.steps))
        self.seen += 1
        if self.seen % 20 == 0:
            write_json(Path(runner.work_dir) / f'progress_rank{self.rank}.json', self.rows[-1])
        self.previous_end = time.perf_counter()

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint['meta']['gcd_phase'] = runner.cfg.gcd_phase
        checkpoint['meta']['gcd_experiment_hash'] = runner.cfg.gcd_experiment_hash
        checkpoint['meta']['gcd_engineering_only'] = runner.cfg.engineering_only

    def after_train(self, runner):
        model = unwrap(runner.model)
        changed = sum(bool((p.detach().cpu() - self.initial[n]).abs().max() > 0)
                      for n, p in model.named_parameters() if p.requires_grad)
        h = hashlib.sha256()
        for n, p in model.named_parameters():
            if p.requires_grad:
                h.update(n.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
        if self.world > 1:
            import torch.distributed as dist
            digests = [None] * self.world
            dist.all_gather_object(digests, h.hexdigest())
            if len(set(digests)) != 1:
                raise RuntimeError('DDP adapter states diverged')
        rows = self.rows[self.warmup:] if self.profile else self.rows
        expected = math.ceil(self.seen / runner.optim_wrapper._accumulative_counts)
        passed = (self.base == frozen_digest(model) and changed > 0 and
                  len(self.grads) == 6 and all(g['finite'] and g['nonzero'] for g in self.grads)
                  and self.steps >= expected * .95)
        summary = dict(passed=passed, engineering_only=runner.cfg.engineering_only,
            measurement='CUDA synchronized per microstep' if self.profile else 'training wall time',
            rank=self.rank, world_size=self.world, iterations=self.seen,
            warmup=self.warmup if self.profile else 0,
            steady_seconds=sum(r['seconds'] for r in rows),
            steady_images=sum(r['images'] for r in rows),
            p50_seconds=float(np.percentile([r['seconds'] for r in rows], 50)),
            p95_seconds=float(np.percentile([r['seconds'] for r in rows], 95)),
            optimizer_steps=self.steps, expected_optimizer_steps=expected,
            amp_skipped_steps=expected - self.steps,
            frozen_parameters_unchanged=self.base == frozen_digest(model),
            changed_lora_tensors=changed, gradient_checks=self.grads,
            lora_sha256=h.hexdigest(), peak_allocated_bytes=self.peak_allocated,
            peak_reserved_bytes=self.peak_reserved, trainable_parameters=self.trainables['trainable'],
            total_parameters=sum(p.numel() for p in model.parameters()),
            loader_length=len(runner.train_dataloader), elapsed_seconds=time.perf_counter() - self.started)
        path = Path(runner.work_dir)
        write_json(path / f'audit_rank{self.rank}.json', summary)
        write_json(path / f'timings_rank{self.rank}.json', self.rows)
        self.step_handle.remove()
        for handle in self.handles:
            handle.remove()
        if not passed:
            raise RuntimeError('Training invariant failed; inspect saved audit evidence')
