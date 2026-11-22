import unittest

from wildfire_detection.evaluate import Box, iou, match_detections, summarize_counts


class EvaluationTests(unittest.TestCase):
    def test_duplicate_and_wrong_class_cannot_take_a_ground_truth(self):
        coords = (0.0, 0.0, 1.0, 1.0)
        truth = [Box(0, coords), Box(1, (0.0, 0.0, 0.1, 0.1))]
        best = Box(0, coords, 0.90)
        duplicate = Box(0, coords, 0.50)
        wrong_class = Box(1, coords, 0.95)
        tp, fp, fn = match_detections([duplicate, best, wrong_class], truth, 0.5)
        self.assertEqual(tp, [best])
        self.assertEqual(fp, [wrong_class, duplicate])
        self.assertEqual(fn, [truth[1]])

    def test_iou_boundary_is_inclusive(self):
        prediction = Box(0, (0.0, 0.0, 0.5, 1.0), 0.65)
        truth = Box(0, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(iou(prediction.xyxy, truth.xyxy), 0.5)
        self.assertEqual(match_detections([prediction], [truth], 0.5), ([prediction], [], []))

    def test_precision_and_recall_with_no_predictions(self):
        report = summarize_counts({"smoke": {"tp": 0, "fp": 0, "fn": 3}})
        self.assertIsNone(report["smoke"]["precision"])
        self.assertEqual(report["smoke"]["recall"], 0.0)
