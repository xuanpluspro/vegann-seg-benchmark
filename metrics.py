"""Binary-label adapter to the shared dataset metric protocol."""
import numpy as np
from segmentation_metrics import DatasetSegmentationMetrics, METRIC_KEYS, _array

BINARY_METRIC_KEYS = METRIC_KEYS


class BinarySegmentationMetrics:
    def __init__(self, foreground_class=1, background_class=0, ignore_index=255,
                 boundary_tolerance=2.0):
        if {foreground_class, background_class} != {0, 1}:
            raise ValueError("binary metric classes must be 0 and 1")
        self.foreground_class = foreground_class
        self.background_class = background_class
        self.ignore_index = ignore_index
        self.meter = DatasetSegmentationMetrics(2, ["background", "foreground"],
            ignore_index=-1, boundary_tolerance=boundary_tolerance)

    def update(self, prediction, target):
        pred, truth = _array(prediction), _array(target)
        if pred.shape != truth.shape or pred.ndim != 3:
            raise ValueError("prediction/target must have matching [B,H,W] shapes")
        valid = truth != self.ignore_index
        if not np.isin(pred, [0, 1]).all() or not np.isin(truth[valid], [0, 1]).all():
            raise ValueError("Expected binary label IDs")
        labels = (truth == self.foreground_class).astype(np.int64)
        labels[~valid] = -1
        self.meter.update((pred == self.foreground_class).astype(np.int64), labels)

    def reduce_from_all_processes(self):
        # Preserve the optional distributed interface. Current trainers use one GPU.
        import torch
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
        for name in ("confusion", "surface_sums", "class_surface_sums", "class_surface_counts"):
            tensor = torch.as_tensor(getattr(self.meter, name), device=device)
            dist.all_reduce(tensor)
            setattr(self.meter, name, tensor.cpu().numpy())
        count = torch.tensor(self.meter.boundary_image_count, dtype=torch.int64, device=device)
        dist.all_reduce(count)
        self.meter.boundary_image_count = int(count.item())

    def compute(self):
        result = self.meter.compute()
        result.update(foreground_iou=result["foreground_miou"], accuracy=result["pixacc"])
        return result
