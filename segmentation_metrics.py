"""Shared segmentation metric protocol v2 (identical in all three repositories).

Region scores use a dataset confusion matrix. Macro region means skip only
undefined denominators. Boundary scores first average active foreground classes
within each image, then average eligible images. Both-empty classes are skipped;
one-empty classes score F1=0 and distance=hypot(H,W). No eligible pairs -> None.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

PROTOCOL_VERSION = "bimseg-segmentation-v2"
REGION_KEYS = ("miou", "foreground_miou", "union_foreground_iou", "background_iou",
               "dice", "dice_all", "precision", "recall", "specificity", "pixacc",
               "macc", "macc_fg", "fwiou")
BOUNDARY_KEYS = ("boundary_f1", "hd95", "assd")
METRIC_KEYS = REGION_KEYS + BOUNDARY_KEYS
RESULT_FIELDS = ("metric_protocol", "name", "arch", "encoder", "seed", "status",
                 "split", "best_epoch", "num_classes", "background_index",
                 *METRIC_KEYS, "valid_pixels", "evaluated_iou_classes",
                 "evaluated_acc_classes", "boundary_image_count", "parameters",
                 "params_m", "gflops", "counted_gflops", "flops_complete",
                 "latency_ms", "latency_median_ms", "latency_p95_ms", "fps",
                 "precision_mode", "gpu_name")
CLASS_FIELDS = ("split", "class_id", "class_name", "is_background", "iou", "dice",
                "precision", "recall", "specificity", "support", "predicted_pixels",
                "boundary_f1", "hd95", "assd", "boundary_image_count")


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _ratio(a, b):
    return float(a / b) if b else None


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def surface_scores(prediction, target, tolerance=2.0):
    """Return HD95, ASSD, BF1; None when both masks are empty."""
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if not prediction.any() and not target.any():
        return None
    if not prediction.any() or not target.any():
        diagonal = float(np.hypot(*prediction.shape))
        return diagonal, diagonal, 0.0
    structure = ndimage.generate_binary_structure(2, 1)
    p = prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    t = target & ~ndimage.binary_erosion(target, structure=structure, border_value=0)
    dp = ndimage.distance_transform_edt(~t)[p]
    dt = ndimage.distance_transform_edt(~p)[t]
    hd95 = float(max(np.percentile(dp, 95), np.percentile(dt, 95)))
    assd = float(np.concatenate([dp, dt]).mean())
    precision, recall = float(np.mean(dp <= tolerance)), float(np.mean(dt <= tolerance))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return hd95, assd, f1


class DatasetSegmentationMetrics:
    def __init__(self, num_classes, class_names=None, ignore_index=255,
                 background_index=0, boundary_tolerance=2.0):
        if num_classes < 2 or not 0 <= background_index < num_classes:
            raise ValueError("Require >=2 classes and a valid background index")
        if ignore_index is not None and ignore_index in range(num_classes):
            raise ValueError("ignore_index cannot be a valid class ID")
        if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0:
            raise ValueError("boundary tolerance must be finite and nonnegative")
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(num_classes)]
        if len(self.class_names) != num_classes or len(set(self.class_names)) != num_classes:
            raise ValueError("Unique class names must match num_classes")
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.boundary_tolerance = float(boundary_tolerance)
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.surface_sums = np.zeros(3, dtype=np.float64)
        self.boundary_image_count = 0
        self.class_surface_sums = np.zeros((num_classes, 3), dtype=np.float64)
        self.class_surface_counts = np.zeros(num_classes, dtype=np.int64)

    def update(self, prediction, target):
        pred, truth = _array(prediction), _array(target)
        if pred.shape != truth.shape:
            raise ValueError("Prediction/target shape mismatch")
        if not np.issubdtype(pred.dtype, np.integer) or not np.issubdtype(truth.dtype, np.integer):
            raise ValueError("Metrics require integer class IDs")
        if truth.ndim not in (1, 2, 3):
            raise ValueError("Expected flat labels, [H,W] or [B,H,W]")
        valid = np.ones(truth.shape, bool) if self.ignore_index is None else truth != self.ignore_index
        p, t = pred[valid], truth[valid]
        k = self.num_classes
        if ((p < 0) | (p >= k) | (t < 0) | (t >= k)).any():
            raise ValueError("Out-of-range class ID")
        self.confusion += np.bincount(k * t.astype(np.int64) + p.astype(np.int64), minlength=k*k).reshape(k, k)
        # One-dimensional arrays support region-only callers/tests. Images are
        # [H,W] or [B,H,W]; no boundary geometry is invented for flattened data.
        if truth.ndim == 1:
            return
        if truth.ndim == 2:
            pred, truth, valid = pred[None], truth[None], valid[None]
        elif truth.ndim != 3:
            raise ValueError("Use [H,W] or [B,H,W] labels (or flat region-only arrays)")
        for pi, ti, vi in zip(pred, truth, valid):
            if not vi.any():
                continue
            active = []
            for c in range(k):
                if c == self.background_index:
                    continue
                scores = surface_scores((pi == c) & vi, (ti == c) & vi, self.boundary_tolerance)
                if scores is not None:
                    self.class_surface_sums[c] += scores
                    self.class_surface_counts[c] += 1
                    active.append(scores)
            if active:
                self.surface_sums += np.mean(active, axis=0)
                self.boundary_image_count += 1

    def compute(self):
        cm = self.confusion
        support, predicted = cm.sum(1), cm.sum(0)
        total = int(cm.sum())
        rows = []
        for c in range(self.num_classes):
            tp = int(cm[c, c]); fp = int(predicted[c]) - tp; fn = int(support[c]) - tp
            tn = total - tp - fp - fn
            n = int(self.class_surface_counts[c])
            surface = self.class_surface_sums[c] / n if n else (None, None, None)
            rows.append(dict(class_id=c, class_name=self.class_names[c],
                is_background=c == self.background_index,
                iou=_ratio(tp, tp+fp+fn), dice=_ratio(2*tp, 2*tp+fp+fn),
                precision=_ratio(tp, tp+fp), recall=_ratio(tp, tp+fn),
                specificity=_ratio(tn, tn+fp), support=int(support[c]),
                predicted_pixels=int(predicted[c]), hd95=None if surface[0] is None else float(surface[0]),
                assd=None if surface[1] is None else float(surface[1]),
                boundary_f1=None if surface[2] is None else float(surface[2]),
                boundary_image_count=n))
        fg = [r for r in rows if not r["is_background"]]
        bg = self.background_index
        tn = int(cm[bg, bg]); fp = int(cm[bg].sum()) - tn
        fn = int(cm[:, bg].sum()) - tn; tp = total - tn - fp - fn
        n = self.boundary_image_count
        surface = self.surface_sums / n if n else (None, None, None)
        result = dict(metric_protocol=PROTOCOL_VERSION, num_classes=self.num_classes,
            background_index=bg, miou=_mean([r["iou"] for r in rows]),
            foreground_miou=_mean([r["iou"] for r in fg]),
            union_foreground_iou=_ratio(tp, tp+fp+fn), background_iou=rows[bg]["iou"],
            dice=_mean([r["dice"] for r in fg]), dice_all=_mean([r["dice"] for r in rows]),
            precision=_mean([r["precision"] for r in fg]), recall=_mean([r["recall"] for r in fg]),
            specificity=_ratio(tn, tn+fp), pixacc=_ratio(int(np.trace(cm)), total),
            macc=_mean([r["recall"] for r in rows]), macc_fg=_mean([r["recall"] for r in fg]),
            fwiou=sum(r["support"] * (r["iou"] or 0.0) for r in rows) / total if total else None,
            hd95=None if surface[0] is None else float(surface[0]),
            assd=None if surface[1] is None else float(surface[1]),
            boundary_f1=None if surface[2] is None else float(surface[2]),
            boundary_image_count=n, boundary_tolerance=self.boundary_tolerance,
            boundary_scope="per-image macro foreground classes, then mean over eligible images; both-empty class pairs excluded",
            dice_scope="dataset confusion matrix; macro foreground classes with defined denominator",
            metric_units={"region_scores": "0-1", "boundary_f1": "0-1", "hd95": "pixels", "assd": "pixels"},
            empty_policy="undefined region denominator -> null; both-empty boundary pair skipped; one-empty -> image diagonal and BF1=0",
            valid_pixels=total, evaluated_iou_classes=sum(r["iou"] is not None for r in rows),
            evaluated_acc_classes=sum(r["recall"] is not None for r in rows),
            per_class=rows, confusion_matrix=cm.tolist())
        return result


def result_rows(result, name=None, seed=None, status=0):
    """One standardized row per split; reused by per-run and batch exporters."""
    report = result.get("benchmark") or {}
    rows = []
    for split in ("val", "test"):
        if split not in result:
            continue
        m = result[split]
        row = {key: m.get(key) for key in METRIC_KEYS}
        row.update({key: m.get(key) for key in ("num_classes", "background_index", "valid_pixels",
                    "evaluated_iou_classes", "evaluated_acc_classes", "boundary_image_count")})
        row.update(metric_protocol=m.get("metric_protocol", "legacy-unversioned"),
            name=name if name is not None else result.get("name", result.get("arch", "")),
            arch=result.get("arch", ""), encoder=result.get("encoder", ""),
            seed=seed if seed is not None else result.get("seed"), status=status,
            split=split, best_epoch=result.get("best_epoch"), parameters=result.get("parameters", report.get("parameters")))
        for key in ("params_m", "gflops", "counted_gflops", "flops_complete", "latency_ms",
                    "latency_median_ms", "latency_p95_ms", "fps", "gpu_name"):
            row[key] = report.get(key, result.get(key))
        if row["params_m"] is None and row["parameters"] is not None:
            row["params_m"] = row["parameters"] / 1e6
        row["precision_mode"] = report.get("precision")
        rows.append(row)
    return rows


def write_summary(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_run_results(result, output):
    """Write the same four result files for binary and multiclass experiments."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result["metric_protocol"] = PROTOCOL_VERSION
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if result.get("benchmark") is not None:
        (output / "benchmark.json").write_text(json.dumps(result["benchmark"], indent=2, allow_nan=False), encoding="utf-8")
    write_summary(result_rows(result), output / "results.csv")
    with (output / "class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CLASS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for split in ("val", "test"):
            for row in result.get(split, {}).get("per_class", []):
                writer.writerow(dict(split=split, **row))
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "true_class", "predicted_class", "pixels"))
        writer.writeheader()
        for split in ("val", "test"):
            for i, row in enumerate(result.get(split, {}).get("confusion_matrix", [])):
                for j, value in enumerate(row):
                    writer.writerow(dict(split=split, true_class=i, predicted_class=j, pixels=value))


def append_history(row, path):
    """Preserve old history rows when protocol v2 introduces new scalar columns."""
    path = Path(path)
    previous = []
    fields = []
    if path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or [])
            previous = list(reader)
    fields = list(dict.fromkeys([*fields, *row]))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(previous); writer.writerow(row)
