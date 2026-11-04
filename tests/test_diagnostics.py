import unittest
from gcd.diagnostics import candidate_rows, prediction_bins, image_stratum, iou_xywh, choose_probe, representation_changes


class DiagnosticTests(unittest.TestCase):
    def test_norm_scaling_is_separate_from_direction_and_topology(self):
        import numpy as np
        a = np.asarray([[[1., 0.], [0., 1.]]])
        result = representation_changes(a, a * 2)
        self.assertAlmostEqual(result['mean_norm_ratio'], 2)
        self.assertAlmostEqual(result['mean_corresponding_cosine'], 1)
        self.assertAlmostEqual(result['mean_abs_pairwise_cosine_change'], 0)

    def test_rigid_rotation_can_change_directions_but_preserve_topology(self):
        import numpy as np
        a = np.asarray([[[1., 0.], [0., 1.]]])
        b = a @ np.asarray([[0., -1.], [1., 0.]])
        result = representation_changes(a, b)
        self.assertAlmostEqual(result['mean_corresponding_cosine'], 0)
        self.assertAlmostEqual(result['mean_abs_pairwise_cosine_change'], 0)

    def ann(self, cat=1, difficult=0, identity=1):
        return dict(id=identity, image_id=1, category_id=cat, bbox=[0, 0, 10, 10], iscrowd=difficult)

    def pred(self, cat=1, score=.9):
        return dict(image_id=1, category_id=cat, bbox=[0, 0, 10, 10], score=score)

    def test_score_suppression_is_separate_from_missing_cached_box(self):
        row = candidate_rows([self.ann()], [self.pred(1, .02), self.pred(16, .9)])[0]
        self.assertEqual(row['candidate_bin'], 'other_class_localized')
        self.assertEqual(row['best_same_class_score_iou50'], .02)
        self.assertTrue(row['any_class_iou50_at_threshold'])
        row = candidate_rows([self.ann()], [self.pred(16)])[0]
        self.assertIsNone(row['best_same_class_score_iou50'])

    def test_difficult_and_duplicate_are_not_counted_as_new_true_positives(self):
        counts, _ = prediction_bins([self.ann()], [self.pred(), self.pred(score=.8)])
        self.assertEqual(counts['old:true_positive'], 1)
        self.assertEqual(counts['old:duplicate'], 1)
        counts, _ = prediction_bins([self.ann(difficult=1)], [self.pred()])
        self.assertEqual(counts['old:ignored_difficult'], 1)

    def test_confusion_orientation_is_ground_truth_to_predicted_class(self):
        counts, matrix = prediction_bins([self.ann(1)], [self.pred(16)])
        self.assertEqual(matrix[(1, 16)], 1)
        self.assertEqual(counts['new:wrong_class'], 1)

    def test_exact_half_iou_and_empty_predictions(self):
        self.assertEqual(float(iou_xywh([[0, 0, 5, 10]], [[0, 0, 10, 10]])[0, 0]), .5)
        row = candidate_rows([self.ann()], [])[0]
        self.assertIsNone(row['best_same_class_score_iou50'])
        self.assertEqual(row['candidate_bin'], 'no_candidate_overlap')
        counts, _ = prediction_bins([], [self.pred()])
        self.assertEqual(counts['old:background'], 1)

    def test_monotone_score_change_preserves_ap_but_changes_fixed_threshold_coverage(self):
        from gcd.metrics import voc07_class_ap
        gt = dict(images=[dict(id=1)], annotations=[self.ann()])
        before, after = [self.pred(score=.9)], [self.pred(score=.009)]
        self.assertEqual(voc07_class_ap(gt, before, 1), voc07_class_ap(gt, after, 1))
        self.assertTrue(candidate_rows(gt['annotations'], before)[0]['same_class_iou50_at_threshold'])
        self.assertFalse(candidate_rows(gt['annotations'], after)[0]['same_class_iou50_at_threshold'])

    def test_probe_uses_annotations_only_and_is_deterministic(self):
        data = dict(images=[dict(id=i) for i in range(8)], annotations=[])
        for i in range(8):
            data['annotations'].append(dict(self.ann(), image_id=i))
            if i >= 4:
                data['annotations'].append(dict(self.ann(16), image_id=i))
        self.assertEqual(choose_probe(data, 4), choose_probe(data, 4))
        ids = choose_probe(data, 4)
        self.assertEqual(sum(i < 4 for i in ids), 2)
        self.assertEqual(image_stratum([self.ann(), self.ann(16)]), 'old_and_new')
