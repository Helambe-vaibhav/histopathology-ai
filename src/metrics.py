"""
metrics.py
----------
Functions:
  dice()             - pixel-level Dice coefficient
  iou()              - pixel-level Intersection over Union
  precision()        - pixel-level precision
  recall()           - pixel-level recall
  panoptic_quality() - instance-level PQ = SQ × DQ
  bootstrap_ci()     - image-level bootstrap confidence interval
"""

import numpy as np
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Pixel-level metrics  (operate on binary numpy arrays)
# ---------------------------------------------------------------------------

def _confusion(pred: np.ndarray, target: np.ndarray):
    """Return (TP, FP, FN) for flattened binary arrays."""
    pred = pred.astype(bool).ravel()
    target = target.astype(bool).ravel()
    TP = np.logical_and(pred, target).sum()
    FP = np.logical_and(pred, ~target).sum()
    FN = np.logical_and(~pred, target).sum()
    return TP, FP, FN


def dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice coefficient: 2·TP / (2·TP + FP + FN)

    Args:
        pred:   binary (H, W) numpy array
        target: binary (H, W) numpy array

    Returns:
        float in [0, 1]
    """
    TP, FP, FN = _confusion(pred, target)
    return float((2 * TP + smooth) / (2 * TP + FP + FN + smooth))


def iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Intersection over Union (Jaccard): TP / (TP + FP + FN)
    """
    TP, FP, FN = _confusion(pred, target)
    return float((TP + smooth) / (TP + FP + FN + smooth))


def precision(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Precision: TP / (TP + FP)"""
    TP, FP, _ = _confusion(pred, target)
    return float((TP + smooth) / (TP + FP + smooth))


def recall(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Recall: TP / (TP + FN)"""
    TP, _, FN = _confusion(pred, target)
    return float((TP + smooth) / (TP + FN + smooth))


def pixel_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute all four pixel-level metrics at once."""
    return {
        "dice": dice(pred, target),
        "iou": iou(pred, target),
        "precision": precision(pred, target),
        "recall": recall(pred, target),
    }


compute_pixel_metrics = pixel_metrics


# ---------------------------------------------------------------------------
# Instance-level: Panoptic Quality
# ---------------------------------------------------------------------------

def _instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray,
                   pred_id: int, gt_id: int) -> float:
    """IoU between a single predicted instance and a single GT instance."""
    p = pred_mask == pred_id
    g = gt_mask == gt_id
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / (union + 1e-8)


def panoptic_quality(
    pred_instance: np.ndarray,
    gt_instance: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Panoptic Quality: PQ = SQ × DQ

    SQ = mean IoU of matched pairs
    DQ = 2·TP / (2·TP + FP + FN)  (at instance level)

    Args:
        pred_instance: (H, W) uint32 instance label map (0 = background)
        gt_instance:   (H, W) uint32 instance label map (0 = background)
        iou_threshold: minimum IoU to count as a match (default 0.5)

    Returns:
        dict with keys: pq, sq, dq, tp, fp, fn, matched_ious
    """
    pred_ids = np.unique(pred_instance)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = np.unique(gt_instance)
    gt_ids = gt_ids[gt_ids != 0]

    matched_ious = []
    matched_gt = set()
    matched_pred = set()

    # Greedy matching: for each GT find the best overlapping prediction
    for gid in gt_ids:
        gt_region = gt_instance == gid
        # Find overlapping predictions
        overlapping_preds = np.unique(pred_instance[gt_region])
        overlapping_preds = overlapping_preds[overlapping_preds != 0]

        best_iou = 0.0
        best_pid = None
        for pid in overlapping_preds:
            if pid in matched_pred:
                continue
            iou_val = _instance_iou(pred_instance, gt_instance, pid, gid)
            if iou_val > best_iou:
                best_iou = iou_val
                best_pid = pid

        if best_pid is not None and best_iou >= iou_threshold:
            matched_ious.append(best_iou)
            matched_gt.add(gid)
            matched_pred.add(best_pid)

    TP = len(matched_ious)
    FP = len(pred_ids) - TP
    FN = len(gt_ids) - TP

    SQ = float(np.mean(matched_ious)) if matched_ious else 0.0
    DQ = float((2 * TP) / (2 * TP + FP + FN + 1e-8))
    PQ = SQ * DQ

    return {
        "pq": PQ,
        "sq": SQ,
        "dq": DQ,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "matched_ious": matched_ious,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence interval  (image-level)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    metric_fn: Callable,
    predictions: list,
    targets: list,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Image-level bootstrap confidence interval.

    Args:
        metric_fn:    function(pred, target) → float  (e.g., dice)
        predictions:  list of (H, W) binary arrays, one per image
        targets:      list of (H, W) binary arrays, one per image
        n_bootstrap:  number of bootstrap samples
        alpha:        significance level (0.05 → 95% CI)
        seed:         random seed

    Returns:
        (mean, lower, upper)
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)
    scores_orig = np.array([metric_fn(p, t) for p, t in zip(predictions, targets)])
    mean_orig = float(scores_orig.mean())

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = scores_orig[idx]
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_orig, lower, upper


def report_metrics(predictions: list, targets: list, prefix: str = "") -> dict:
    """
    Compute all pixel-level metrics with 95% bootstrap CI over a list of images.

    Returns dict with keys like 'dice_mean', 'dice_lower', 'dice_upper', etc.
    """
    results = {}
    for name, fn in [("dice", dice), ("iou", iou), ("precision", precision), ("recall", recall)]:
        mean, lo, hi = bootstrap_ci(fn, predictions, targets)
        results[f"{prefix}{name}_mean"] = mean
        results[f"{prefix}{name}_lower"] = lo
        results[f"{prefix}{name}_upper"] = hi
        print(f"{prefix}{name}: {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    return results
