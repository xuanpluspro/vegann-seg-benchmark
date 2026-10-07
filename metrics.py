"""Region and boundary metrics owned by the standalone Code_MPSA package."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from scipy import ndimage


BINARY_METRIC_KEYS = (
    "foreground_iou",
    "background_iou",
    "miou",
    "dice",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "hd95",
    "assd",
    "boundary_f1",
)


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class BinarySegmentationMetrics:
    """Accumulate dataset region scores and mean per-image surface scores.

    The project label convention is fixed to class ``0`` for plant foreground,
    class ``1`` for background, and ``255`` for ignored pixels by default.
    HD95 and ASSD are measured in pixels at evaluation resolution.
    """

    def __init__(
        self,
        foreground_class: int = 0,
        background_class: int = 1,
        ignore_index: int = 255,
        boundary_tolerance: float = 2.0,
    ) -> None:
        if {foreground_class, background_class} != {0, 1}:
            raise ValueError("binary metric classes must be 0 and 1")
        if foreground_class == background_class:
            raise ValueError("foreground and background classes must differ")
        if boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be non-negative")
        self.foreground_class = foreground_class
        self.background_class = background_class
        self.ignore_index = ignore_index
        self.boundary_tolerance = float(boundary_tolerance)
        self.confusion: Optional[torch.Tensor] = None
        self.surface_sums: Optional[torch.Tensor] = None
        self.image_count: Optional[torch.Tensor] = None

    @staticmethod
    def _safe_ratio(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        empty_value: float = 1.0,
    ) -> float:
        if denominator.item() == 0:
            return float(empty_value)
        return float((numerator / denominator).item())

    @staticmethod
    def _surface(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        eroded = ndimage.binary_erosion(
            mask,
            structure=ndimage.generate_binary_structure(2, 1),
            border_value=0,
        )
        return mask & ~eroded

    def _surface_metrics(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, float, float]:
        prediction_surface = self._surface(prediction)
        target_surface = self._surface(target)
        prediction_empty = not prediction_surface.any()
        target_empty = not target_surface.any()

        if prediction_empty and target_empty:
            return 0.0, 0.0, 1.0
        if prediction_empty or target_empty:
            diagonal = float(np.hypot(*prediction.shape))
            return diagonal, diagonal, 0.0

        distance_to_target = ndimage.distance_transform_edt(~target_surface)
        distance_to_prediction = ndimage.distance_transform_edt(
            ~prediction_surface
        )
        prediction_distances = distance_to_target[prediction_surface]
        target_distances = distance_to_prediction[target_surface]
        symmetric_distances = np.concatenate(
            (prediction_distances, target_distances)
        )
        hd95 = float(
            max(
                np.percentile(prediction_distances, 95),
                np.percentile(target_distances, 95),
            )
        )
        assd = float(symmetric_distances.mean())

        boundary_precision = float(
            np.mean(prediction_distances <= self.boundary_tolerance)
        )
        boundary_recall = float(
            np.mean(target_distances <= self.boundary_tolerance)
        )
        denominator = boundary_precision + boundary_recall
        boundary_f1 = (
            2.0 * boundary_precision * boundary_recall / denominator
            if denominator > 0.0
            else 0.0
        )
        return hd95, assd, boundary_f1

    def _initialize(self, device: torch.device) -> None:
        self.confusion = torch.zeros((2, 2), dtype=torch.int64, device=device)
        self.surface_sums = torch.zeros(3, dtype=torch.float64, device=device)
        self.image_count = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim != 3 or target.ndim != 3:
            raise ValueError("prediction and target must have shape [N, H, W]")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if prediction.device != target.device:
            raise ValueError("prediction and target must use the same device")
        if self.confusion is None:
            self._initialize(target.device)
        assert self.confusion is not None
        assert self.surface_sums is not None
        assert self.image_count is not None

        valid = target != self.ignore_index
        invalid_target = valid & (target != self.foreground_class) & (
            target != self.background_class
        )
        if invalid_target.any():
            values = torch.unique(target[invalid_target]).detach().cpu().tolist()
            raise ValueError(f"target contains unsupported class values: {values}")
        invalid_prediction = (prediction != self.foreground_class) & (
            prediction != self.background_class
        )
        if invalid_prediction.any():
            values = (
                torch.unique(prediction[invalid_prediction])
                .detach()
                .cpu()
                .tolist()
            )
            raise ValueError(f"prediction contains unsupported class values: {values}")

        target_binary = (target == self.background_class).to(torch.int64)
        prediction_binary = (prediction == self.background_class).to(torch.int64)
        indices = 2 * target_binary[valid] + prediction_binary[valid]
        self.confusion += torch.bincount(indices, minlength=4).reshape(2, 2)

        prediction_cpu = prediction.detach().cpu().numpy()
        target_cpu = target.detach().cpu().numpy()
        valid_cpu = valid.detach().cpu().numpy()
        surface_totals = np.zeros(3, dtype=np.float64)
        valid_image_count = 0
        for predicted_image, target_image, valid_image in zip(
            prediction_cpu, target_cpu, valid_cpu
        ):
            if not valid_image.any():
                continue
            predicted_foreground = (
                (predicted_image == self.foreground_class) & valid_image
            )
            target_foreground = (
                (target_image == self.foreground_class) & valid_image
            )
            surface_totals += self._surface_metrics(
                predicted_foreground,
                target_foreground,
            )
            valid_image_count += 1

        self.surface_sums += torch.as_tensor(
            surface_totals,
            dtype=torch.float64,
            device=target.device,
        )
        self.image_count += valid_image_count

    def reduce_from_all_processes(self) -> None:
        if not _distributed_ready():
            return
        if self.confusion is None:
            raise RuntimeError("cannot reduce metrics before update")
        assert self.surface_sums is not None
        assert self.image_count is not None
        dist.barrier()
        dist.all_reduce(self.confusion)
        dist.all_reduce(self.surface_sums)
        dist.all_reduce(self.image_count)

    def compute(self) -> dict[str, float]:
        if self.confusion is None:
            raise RuntimeError("cannot compute metrics before update")
        assert self.surface_sums is not None
        assert self.image_count is not None

        histogram = self.confusion.double()
        true_positive = histogram[0, 0]
        false_negative = histogram[0, 1]
        false_positive = histogram[1, 0]
        true_negative = histogram[1, 1]

        foreground_union = true_positive + false_positive + false_negative
        background_union = true_negative + false_positive + false_negative
        foreground_iou = self._safe_ratio(true_positive, foreground_union)
        background_iou = self._safe_ratio(true_negative, background_union)
        dice = self._safe_ratio(
            2.0 * true_positive,
            2.0 * true_positive + false_positive + false_negative,
        )
        precision = self._safe_ratio(
            true_positive,
            true_positive + false_positive,
            empty_value=float(false_negative.item() == 0),
        )
        recall = self._safe_ratio(
            true_positive,
            true_positive + false_negative,
        )
        specificity = self._safe_ratio(
            true_negative,
            true_negative + false_positive,
        )
        accuracy = self._safe_ratio(
            true_positive + true_negative,
            histogram.sum(),
        )
        surface_means = self.surface_sums / self.image_count.clamp_min(1)

        return {
            "foreground_iou": foreground_iou,
            "background_iou": background_iou,
            "miou": 0.5 * (foreground_iou + background_iou),
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "hd95": float(surface_means[0].item()),
            "assd": float(surface_means[1].item()),
            "boundary_f1": float(surface_means[2].item()),
        }
