"""Exercise admission with mocked telemetry; no CUDA process is launched."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('available_gpu_launcher', ROOT / 'scripts/evaluate_available_gpu.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class NativeConfigReached(Exception):
    pass


class GPULauncherTests(unittest.TestCase):
    def admission(self, samples, authorized, expected_gpu=None):
        argv = ['evaluate_available_gpu.py', '--runtime', 'unused.json', '--init', 'unused.pth',
                '--work-dir', 'unused_work_dir']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(launcher, 'read_runtime', return_value=dict(authorized_gpus=authorized)), \
             patch.object(launcher.time, 'sleep'), \
             patch.object(launcher.subprocess, 'check_output', side_effect=samples), \
             patch.object(launcher.subprocess, 'run', side_effect=NativeConfigReached) as native:
            if expected_gpu is None:
                with self.assertRaisesRegex(RuntimeError, 'No authorized GPU'):
                    launcher.main()
                native.assert_not_called()
            else:
                with self.assertRaises(NativeConfigReached):
                    launcher.main()
                command = native.call_args.args[0]
                self.assertEqual(command[command.index('--gpus') + 1], str(expected_gpu))
                self.assertIn('--dry-run', command)

    def test_never_uses_an_unauthorized_free_gpu(self):
        self.admission(['1, 12, 24576, 0\n2, 14000, 24576, 100\n'] * 3, [2])

    def test_rejects_transient_idle_and_low_free_memory(self):
        early = '0, 12, 24576, 0\n1, 22000, 24576, 0\n2, 12, 24576, 70\n3, 12, 24576, 0\n'
        late = early.replace('0, 12, 24576, 0', '0, 12, 24576, 30')
        self.admission([early, early, late], [0, 1, 2, 3], expected_gpu=3)

    def test_allows_low_compute_utilization_with_sufficient_headroom(self):
        sample = '0, 18000, 24576, 80\n1, 16711, 24576, 0\n2, 19000, 24576, 100\n3, 18000, 24576, 50\n'
        self.admission([sample] * 3, [0, 1, 2, 3], expected_gpu=1)


if __name__ == '__main__':
    unittest.main()
