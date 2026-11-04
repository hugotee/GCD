import unittest
import contextlib
import io
import json
from gcd.metrics import voc07_class_ap, evaluate


class MetricTests(unittest.TestCase):
    def test_difficult_detection_is_ignored_and_duplicate_is_false_positive(self):
        gt = dict(images=[dict(id=1), dict(id=2)], annotations=[
            dict(image_id=1, category_id=1, bbox=[0, 0, 10, 10], iscrowd=0),
            dict(image_id=2, category_id=1, bbox=[0, 0, 10, 10], iscrowd=1)])
        pred = [dict(image_id=i, category_id=1, bbox=[0, 0, 10, 10], score=s)
                for i, s in [(2, .99), (1, .9), (1, .8)]]
        self.assertAlmostEqual(voc07_class_ap(gt, pred, 1), 100)
        self.assertEqual(voc07_class_ap(gt, [], 1), 0)

    def test_exact_half_iou_is_not_positive_in_voc07(self):
        gt = dict(images=[dict(id=1)], annotations=[
            dict(image_id=1, category_id=1, bbox=[0, 0, 10, 10], iscrowd=0)])
        pred = [dict(image_id=1, category_id=1, bbox=[0, 0, 5, 10], score=.9)]
        self.assertEqual(voc07_class_ap(gt, pred, 1), 0)

    def test_nonpositive_category_is_not_silently_zero_ap(self):
        self.assertIsNone(voc07_class_ap(dict(images=[], annotations=[]), [], 3))

    def test_perfect_20_class_predictions_and_json_serialization(self):
        gt = dict(images=[dict(id=1, width=100, height=100)],
            categories=[dict(id=i, name=f'class{i}') for i in range(1, 21)],
            annotations=[dict(id=i, image_id=1, category_id=i, bbox=[0, 0, 10, 10],
                              area=100, iscrowd=0) for i in range(1, 21)])
        pred = [dict(image_id=1, category_id=i, bbox=[0, 0, 10, 10], score=.9)
                for i in range(1, 21)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate(gt, pred)
        json.dumps(result)  # COCO params.catIds can otherwise leak numpy.int64.
        for group in result['groups'].values():
            for value in group.values():
                self.assertAlmostEqual(value, 100)
