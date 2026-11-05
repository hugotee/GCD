from pathlib import Path
import tempfile
import unittest
import torch
from gcd.initialization import verify_loaded_checkpoint

class InitializationTests(unittest.TestCase):
    def test_zero_adapter_can_initialize_from_exact_pretrained_base(self):
        model=torch.nn.Linear(2,1)
        expected={k:v.clone() for k,v in model.state_dict().items()}
        model.register_parameter('lora_A',torch.nn.Parameter(torch.ones(1,2)))
        model.register_parameter('lora_B',torch.nn.Parameter(torch.zeros(1,1)))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'w0.pth';torch.save(dict(state_dict=expected),p)
            self.assertEqual(verify_loaded_checkpoint(model,p)['new_adapter_tensors'],2)
            with torch.no_grad():model.lora_B.fill_(1)
            with self.assertRaises(RuntimeError):verify_loaded_checkpoint(model,p)

    def test_adapter_checkpoint_and_corrupted_base_are_checked_exactly(self):
        model=torch.nn.Linear(2,1)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.pth';torch.save(dict(state_dict=model.state_dict()),p)
            self.assertTrue(verify_loaded_checkpoint(model,p)['passed'])
            with torch.no_grad():model.weight.add_(1)
            with self.assertRaises(RuntimeError):verify_loaded_checkpoint(model,p)
