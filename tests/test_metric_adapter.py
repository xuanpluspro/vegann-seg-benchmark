"""Exercise each repository's actual adapter against the canonical metric core."""
from pathlib import Path
import unittest
import numpy as np
from segmentation_metrics import DatasetSegmentationMetrics, METRIC_KEYS


class AdapterTests(unittest.TestCase):
    def test_repository_adapter_matches_core(self):
        root = Path(__file__).resolve().parents[1]
        if (root/'utils/metrics.py').exists():
            from utils.metrics import SegmentationMetrics
            factory = lambda k: SegmentationMetrics(k)
            def update(meter, prediction, target, k):
                logits = np.eye(k)[prediction].transpose(0,3,1,2)
                meter.update(logits, target)
            classes = (2,4,6)
        else:
            import metrics
            if hasattr(metrics, 'MulticlassSegmentationMetrics'):
                factory = metrics.MulticlassSegmentationMetrics
                classes = (2,4,6)
            else:
                factory = lambda k: metrics.BinarySegmentationMetrics()
                classes = (2,)
            def update(meter, prediction, target, k):
                meter.update(prediction, target)
        rng = np.random.default_rng(32)
        for k in classes:
            prediction = rng.integers(k, size=(3,8,9))
            target = rng.integers(k, size=(3,8,9))
            adapter = factory(k); core = DatasetSegmentationMetrics(k)
            update(adapter,prediction,target,k); core.update(prediction,target)
            expected,actual = core.compute(),adapter.compute()
            for key in (*METRIC_KEYS,'confusion_matrix','valid_pixels','boundary_image_count'):
                self.assertEqual(actual[key],expected[key],key)


if __name__ == '__main__':
    unittest.main()
