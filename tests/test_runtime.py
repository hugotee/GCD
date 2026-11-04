import unittest
from gcd.runtime import validate_gpus, validate_resume, validate_experiment


class RuntimeTests(unittest.TestCase):
    def test_configuration_cannot_silently_enable_unimplemented_methods(self):
        from pathlib import Path
        import json
        exp = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/voc15_5.json').read_text())
        validate_experiment(exp)
        for key, value in [('denoising', True), ('replay', True), ('distillation', True),
                           ('lora_targets', 'fusion'), ('num_queries', 0)]:
            with self.assertRaises(ValueError):
                validate_experiment(dict(exp, **{key: value}))

    def test_gpu_authorization_is_physical_and_explicit(self):
        self.assertEqual(validate_gpus('0', [0]), [0])
        for value in ['1', '0,2', '0,0']:
            with self.assertRaises(ValueError):
                validate_gpus(value, [0])

    def test_resume_requires_full_state_and_matching_phase_config(self):
        ckpt = dict(state_dict={}, optimizer={}, param_schedulers=[],
                    meta=dict(gcd_phase='task1', gcd_experiment_hash='abc'))
        validate_resume(ckpt, 'task1', 'abc')
        for phase, digest in [('task2', 'abc'), ('task1', 'def')]:
            with self.assertRaises(ValueError):
                validate_resume(ckpt, phase, digest)
        with self.assertRaises(ValueError):
            validate_resume(dict(state_dict={}), 'task1', 'abc')
        ckpt['meta']['gcd_engineering_only'] = True
        with self.assertRaises(ValueError):
            validate_resume(ckpt, 'task1', 'abc')
