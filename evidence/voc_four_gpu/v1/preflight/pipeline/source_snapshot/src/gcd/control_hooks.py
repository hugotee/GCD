"""Audit actual inputs, initial adapters and native positive-token targets."""
import hashlib
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from gcd.hooks import unwrap
from gcd.prompt_control import arm_spec, sample_fingerprint
from gcd.protocol import CLASSES, write_json


def adapter_digest(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@HOOKS.register_module()
class PromptControlAuditHook(Hook):
    def __init__(self, arm):
        self.arm, self.spec = arm, arm_spec(arm)

    def before_train(self, runner):
        model = unwrap(runner.model)
        self.initial = adapter_digest(model)
        self.rows, self.target_check = [], None

        def check_targets(module, args, kwargs, output):
            if self.target_check is not None or kwargs.get('mode') != 'loss':
                return
            checks = []
            for sample in kwargs['data_samples']:
                tokenized, caption, spans, _ = model.get_tokens_and_prompts(sample.text, True)
                labels = sample.gt_instances.labels.tolist()
                _, expected = model.get_positive_map(tokenized, [spans[k] for k in labels])
                actual = sample.gt_instances.positive_maps.detach().cpu()
                assert torch.equal(actual, expected.bool().float())
                names = [CLASSES[k + self.spec['global_label_offset']] for k in labels]
                extracted = [' '.join(caption[start:end] for start, end in spans[k]) for k in labels]
                assert names == extracted
                checks.append(dict(image_id=int(sample.img_id), local_labels=labels,
                    global_labels=[k + self.spec['global_label_offset'] for k in labels],
                    ground_truth_names=names, native_positive_map_verified=True,
                    text_token_count=int(sample.gt_instances.text_token_mask[0].sum())))
            self.target_check = checks

        self.handle = model.register_forward_hook(check_targets, with_kwargs=True)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        fingerprints, ids = [], []
        for image, sample in zip(data_batch['inputs'], data_batch['data_samples']):
            assert list(sample.text) == self.spec['classes']
            gt = sample.gt_instances
            ids.append(int(sample.img_id))
            fingerprints.append(sample_fingerprint(sample.img_id, image.cpu().numpy(),
                gt.bboxes.cpu().numpy(), gt.labels.cpu().numpy(), self.arm))
        self.rows.append(dict(image_ids=ids, sample_sha256=fingerprints))

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint['meta'].update(gcd_prompt_arm=self.arm, gcd_diagnostic_only=True,
                                  gcd_budget_epochs=1)

    def after_train(self, runner):
        self.handle.remove()
        assert self.target_check is not None
        work = Path(runner.work_dir)
        write_json(work / 'input_trace.json', self.rows)
        write_json(work / 'control_audit.json', dict(passed=True, arm=self.arm,
            diagnostic_short_training=True, initial_adapter_sha256=self.initial,
            final_adapter_sha256=adapter_digest(unwrap(runner.model)),
            native_target_checks=self.target_check, microsteps=len(self.rows)))
