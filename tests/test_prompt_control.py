import unittest
import numpy as np
from gcd.prompt_control import arm_spec, sample_fingerprint, compare_traces


class PromptControlTests(unittest.TestCase):
    def test_local_labels_remap_to_identical_global_supervision(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        boxes = [[0, 0, 1, 1], [0, 0, 2, 2]]
        a = sample_fingerprint(7, image, boxes, [0, 4], 'new5')
        b = sample_fingerprint(7, image, boxes, [15, 19], 'all20')
        self.assertEqual(a, b)
        self.assertEqual(arm_spec('new5')['classes'][0], 'pottedplant')
        with self.assertRaises(ValueError):
            sample_fingerprint(7, image, boxes, [14], 'all20')
        self.assertNotEqual(a, sample_fingerprint(7, image + 1, boxes, [0, 4], 'new5'))
        self.assertNotEqual(a, sample_fingerprint(7, image, [[0, 0, 2, 1], boxes[1]], [0, 4], 'new5'))

    def test_pair_rejects_order_change_duplicate_and_partial_epoch(self):
        a = [dict(image_ids=[1, 2], sample_sha256=['a', 'b']),
             dict(image_ids=[3], sample_sha256=['c'])]
        self.assertTrue(compare_traces(a, a, 3)['passed'])
        for b, count in [(list(reversed(a)), 3), (a, 4), (a + a, 6)]:
            with self.assertRaises(ValueError):
                compare_traces(a, b, count)


if __name__ == '__main__':
    unittest.main()
