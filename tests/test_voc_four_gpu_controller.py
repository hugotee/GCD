import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('voc_four_gpu',Path(__file__).resolve().parents[1]/'scripts/run_voc_four_gpu.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class FourGpuControllerTests(unittest.TestCase):
    def audit(self,updates=200,**kw):
        return dict(passed=True,iterations=200,expected_optimizer_steps=200,
                    optimizer_steps=updates,amp_skipped_steps=200-updates,**kw)

    def test_synchronized_amp_skip_is_recorded_without_counting_it_as_update(self):
        module.check_profile([self.audit(199) for _ in range(4)])

    def test_rank_disagreement_and_excessive_skips_are_rejected(self):
        for rows in [[self.audit(199),self.audit()], [self.audit(189)]*4]:
            with self.assertRaises(AssertionError):module.check_profile(rows)

    def test_incomplete_or_failed_measurement_is_rejected(self):
        for key,value in [('iterations',199),('expected_optimizer_steps',199),('passed',False),('amp_skipped_steps',1)]:
            row=self.audit();row[key]=value
            with self.assertRaises(AssertionError):module.check_profile([row]*4)
