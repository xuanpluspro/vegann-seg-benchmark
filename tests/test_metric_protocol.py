"""Hand-calculated protocol checks; CPU NumPy/SciPy only, no model downloads."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from segmentation_metrics import (DatasetSegmentationMetrics, RESULT_FIELDS,
    surface_scores, write_run_results, append_history)


class MetricProtocolTests(unittest.TestCase):
    def test_hand_calculated_regions(self):
        meter = DatasetSegmentationMetrics(3)
        meter.update(np.array([0,1,2,0,2]), np.array([0,0,1,1,2]))
        result = meter.compute()
        self.assertEqual(result['confusion_matrix'], [[1,1,0],[1,0,1],[0,0,1]])
        self.assertAlmostEqual(result['miou'], (1/3+0+1/2)/3)
        self.assertAlmostEqual(result['foreground_miou'], .25)
        self.assertAlmostEqual(result['dice'], 1/3)
        self.assertAlmostEqual(result['dice_all'], (1/2+0+2/3)/3)
        self.assertAlmostEqual(result['union_foreground_iou'], .5)
        self.assertAlmostEqual(result['macc'], .5)
        self.assertAlmostEqual(result['pixacc'], .4)

    def test_class_swaps_are_not_correct_foreground(self):
        truth = np.array([[[0,1,2],[0,1,2],[0,1,2]]])
        prediction = np.array([[[0,2,1],[0,2,1],[0,2,1]]])
        meter = DatasetSegmentationMetrics(3, boundary_tolerance=0)
        meter.update(prediction, truth)
        result = meter.compute()
        self.assertEqual(result['union_foreground_iou'], 1)
        self.assertEqual(result['foreground_miou'], 0)
        self.assertEqual(result['dice'], 0)
        self.assertEqual(result['boundary_f1'], 0)

    def test_absent_false_positive_and_ignored(self):
        meter = DatasetSegmentationMetrics(4)
        meter.update(np.array([0,1,99]), np.array([0,0,255]))
        result = meter.compute()
        self.assertEqual(result['per_class'][1]['iou'], 0)
        self.assertIsNone(result['per_class'][1]['recall'])
        self.assertIsNone(result['per_class'][2]['iou'])
        self.assertEqual(result['evaluated_iou_classes'], 2)
        self.assertEqual(result['valid_pixels'], 2)
        self.assertAlmostEqual(result['miou'], .25)
        empty = DatasetSegmentationMetrics(2)
        empty.update(np.array([99]), np.array([255]))
        self.assertIsNone(empty.compute()['miou'])
        json.dumps(result, allow_nan=False)

    def test_boundaries_empty_and_distance(self):
        empty = np.zeros((4,4), bool)
        point = empty.copy(); point[1,1] = True
        other = empty.copy(); other[1,3] = True
        self.assertIsNone(surface_scores(empty, empty))
        self.assertEqual(surface_scores(point, other), (2.,2.,1.))
        self.assertEqual(surface_scores(point, other, 1), (2.,2.,0.))
        hd, assd, f1 = surface_scores(empty, point)
        self.assertEqual(hd, np.hypot(4,4)); self.assertEqual(assd, hd)
        self.assertEqual(f1, 0)
        meter = DatasetSegmentationMetrics(2)
        meter.update(np.zeros((2,4,4), int), np.zeros((2,4,4), int))
        self.assertIsNone(meter.compute()['boundary_f1'])
        self.assertEqual(meter.compute()['boundary_image_count'], 0)

    def test_boundary_averages_classes_then_images(self):
        truth = np.zeros((2,4,4), int); prediction = truth.copy()
        truth[0,1,1] = prediction[0,1,1] = 1
        truth[0,2,2] = 2  # one-empty class: F1=0, distance=diagonal
        truth[1,1,1] = prediction[1,1,1] = 1
        meter = DatasetSegmentationMetrics(3)
        meter.update(prediction, truth)
        result = meter.compute()
        self.assertEqual(result['boundary_f1'], .75)
        self.assertAlmostEqual(result['hd95'], np.hypot(4,4)/4)
        self.assertEqual(result['per_class'][1]['boundary_image_count'], 2)
        self.assertEqual(result['per_class'][2]['boundary_image_count'], 1)

    def test_batch_invariance_and_background_index(self):
        random = np.random.default_rng(15)
        for classes in (2,4,6):
            pred = random.integers(classes, size=(3,5,6))
            truth = random.integers(classes, size=(3,5,6)); truth[0,0,0] = 255
            first = DatasetSegmentationMetrics(classes)
            second = DatasetSegmentationMetrics(classes)
            first.update(pred, truth)
            for p,t in zip(pred,truth): second.update(p,t)
            a,b = first.compute(), second.compute()
            self.assertEqual(a,b)
        meter = DatasetSegmentationMetrics(3, background_index=2)
        meter.update(np.array([2,0,1]), np.array([2,1,0]))
        result = meter.compute()
        self.assertEqual(result['background_iou'], 1)
        self.assertEqual(result['foreground_miou'], 0)
        self.assertEqual(result['union_foreground_iou'], 1)

    def test_exports_share_fixed_fields_and_distinguish_precision(self):
        meter = DatasetSegmentationMetrics(2)
        meter.update(np.array([0,1]), np.array([0,1]))
        result = dict(best_epoch=3, parameters=8, test=meter.compute(),
                      benchmark={'precision': 'FP32', 'params_m': 8e-6})
        with tempfile.TemporaryDirectory() as directory:
            write_run_results(result,directory)
            path = Path(directory)
            self.assertEqual({p.name for p in path.iterdir()},
                {'metrics.json','benchmark.json','results.csv','class_metrics.csv','confusion_matrix.csv'})
            with (path/'results.csv').open(encoding='utf-8-sig') as stream:
                reader=csv.DictReader(stream); row=next(reader)
                self.assertEqual(reader.fieldnames,list(RESULT_FIELDS))
                self.assertEqual(row['precision'],'1.0')
                self.assertEqual(row['precision_mode'],'FP32')
            with (path/'confusion_matrix.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),4)
            history=path/'history.csv'
            append_history({'epoch':1,'loss':.5},history)
            append_history({'epoch':2,'loss':.4,'miou':.6},history)
            with history.open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['loss'],'0.5'); self.assertEqual(rows[1]['miou'],'0.6')

    def test_invalid_labels_rejected(self):
        for value in (np.array([3]), np.array([1.0])):
            with self.assertRaises(ValueError):
                DatasetSegmentationMetrics(2).update(value,np.array([0]))
        with self.assertRaises(ValueError): DatasetSegmentationMetrics(3,ignore_index=1)
        with self.assertRaises(ValueError): DatasetSegmentationMetrics(2,boundary_tolerance=-1)


if __name__ == '__main__':
    unittest.main()
