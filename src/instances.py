"""
instances.py
------------
Functions:
  binary_to_instances()  - distance transform + watershed → instance labels
  match_instances()      - IoU-based GT–prediction matching
  classify_errors()      - tag each instance as correct/merged/split/missed/FP
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label


# ---------------------------------------------------------------------------
# Binary mask → instance labels
# ---------------------------------------------------------------------------

def binary_to_instances(
    binary_mask: np.ndarray,
    min_distance: int = 5,
    intensity_image: np.ndarray = None,
    min_area: int = 20,
    max_eccentricity: float = 0.98,
) -> np.ndarray:
    """
    Convert a binary segmentation mask to instance labels using
    distance transform + watershed, optionally guided by optical stain intensity
    and filtered by biological morphological constraints.

    Args:
        binary_mask:       (H, W) binary array (0/1 or bool)
        min_distance:      minimum distance between watershed seed peaks
        intensity_image:   (H, W) grayscale or RGB array for optical ridge guidance
        min_area:          minimum pixel area to retain (filters dust/debris)
        max_eccentricity:  maximum allowed ellipse eccentricity (filters thin collagen fibers)

    Returns:
        instance_mask:     (H, W) uint32 array, 0 = background, i = nucleus i
    """
    binary = binary_mask.astype(bool)
    if not binary.any():
        return np.zeros_like(binary_mask, dtype=np.uint32)

    # 1. Euclidean distance transform
    dist = ndi.distance_transform_edt(binary)

    # 2. Optical Guidance (if intensity image provided, deepen saddles at cell membranes)
    if intensity_image is not None:
        if intensity_image.ndim == 3:
            gray = 0.299 * intensity_image[:, :, 0] + 0.587 * intensity_image[:, :, 1] + 0.114 * intensity_image[:, :, 2]
        else:
            gray = intensity_image.astype(float)
        grad = ndi.gaussian_gradient_magnitude(gray, sigma=1.0)
        norm_grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        surface = dist * (1.0 - 0.4 * norm_grad)
    else:
        surface = dist

    # 3. Find local maxima as watershed seeds
    coords = peak_local_max(
        surface,
        min_distance=min_distance,
        labels=binary,
        threshold_abs=1.0,
    )
    if len(coords) == 0:
        labels_ws, _ = ndi.label(binary)
        labels_ws = labels_ws.astype(np.uint32)
    else:
        markers = np.zeros_like(surface, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        labels_ws = watershed(-surface, markers, mask=binary).astype(np.uint32)

    # 4. Biological Morphological Filtering (prune non-cellular fibers and debris)
    if min_area > 0 or max_eccentricity < 1.0:
        from skimage.measure import regionprops
        props = regionprops(labels_ws)
        valid_labels = [p.label for p in props if p.area >= min_area and p.eccentricity <= max_eccentricity]
        cleaned = np.zeros_like(labels_ws)
        new_id = 1
        for orig_lbl in valid_labels:
            cleaned[labels_ws == orig_lbl] = new_id
            new_id += 1
        return cleaned.astype(np.uint32)

    return labels_ws


# ---------------------------------------------------------------------------
# Instance matching
# ---------------------------------------------------------------------------

def match_instances(
    pred_instance: np.ndarray,
    gt_instance: np.ndarray,
    iou_threshold: float = 0.5,
) -> list:
    """
    Match predicted instances to GT instances by IoU.

    Returns list of matched (pred_id, gt_id, iou) tuples.
    """
    pred_ids = np.unique(pred_instance)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = np.unique(gt_instance)
    gt_ids = gt_ids[gt_ids != 0]

    matches = []
    matched_gt = set()
    matched_pred = set()

    for gid in gt_ids:
        gt_region = gt_instance == gid
        overlapping = np.unique(pred_instance[gt_region])
        overlapping = overlapping[overlapping != 0]

        best_iou = 0.0
        best_pid = None
        for pid in overlapping:
            if pid in matched_pred:
                continue
            p_region = pred_instance == pid
            inter = np.logical_and(p_region, gt_region).sum()
            union = np.logical_or(p_region, gt_region).sum()
            iou_val = inter / (union + 1e-8)
            if iou_val > best_iou:
                best_iou = iou_val
                best_pid = pid

        if best_pid is not None and best_iou >= iou_threshold:
            matches.append((int(best_pid), int(gid), float(best_iou)))
            matched_gt.add(gid)
            matched_pred.add(best_pid)

    return matches


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_errors(
    pred_instance: np.ndarray,
    gt_instance: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Classify each predicted / GT nucleus into an error category.

    Categories:
      correct       - 1 pred matched to 1 GT (IoU >= threshold)
      merged        - 1 pred covers 2+ GT nuclei
      split         - 1 GT covered by 2+ pred nuclei
      missed        - GT nucleus with no matching prediction
      false_positive- predicted nucleus with no matching GT

    Returns:
        dict with keys: correct, merged, split, missed, false_positive
        Each value is a list of (pred_id, gt_id) or single ids.
    """
    pred_ids = set(np.unique(pred_instance).tolist()) - {0}
    gt_ids   = set(np.unique(gt_instance).tolist()) - {0}

    errors = {
        "correct": [],
        "merged": [],
        "split": [],
        "missed": [],
        "false_positive": [],
    }

    matched_gt = {}    # gt_id  → list of pred_ids that overlap
    matched_pred = {}  # pred_id → list of gt_ids that overlap

    # Build overlap map
    for pid in pred_ids:
        p_region = pred_instance == pid
        overlapping_gt = np.unique(gt_instance[p_region])
        overlapping_gt = [g for g in overlapping_gt if g != 0]

        for gid in overlapping_gt:
            g_region = gt_instance == gid
            inter = float(np.logical_and(p_region, g_region).sum())
            union = float(np.logical_or(p_region, g_region).sum())
            iou_val = inter / (union + 1e-8)

            if iou_val >= float(iou_threshold):
                matched_gt.setdefault(gid, []).append(pid)
                matched_pred.setdefault(pid, []).append(gid)

    classified_gt = set()
    classified_pred = set()

    # Merged: 1 pred → 2+ GT
    for pid, gids in matched_pred.items():
        if len(gids) > 1:
            errors["merged"].append((pid, gids))
            classified_pred.add(pid)
            classified_gt.update(gids)

    # Split: 2+ preds → 1 GT
    for gid, pids in matched_gt.items():
        if len(pids) > 1 and gid not in classified_gt:
            errors["split"].append((pids, gid))
            classified_gt.add(gid)
            classified_pred.update(pids)

    # Correct: 1 pred ↔ 1 GT
    for pid, gids in matched_pred.items():
        if pid not in classified_pred and len(gids) == 1:
            gid = gids[0]
            if matched_gt.get(gid) == [pid]:
                errors["correct"].append((pid, gid))
                classified_pred.add(pid)
                classified_gt.add(gid)

    # Missed GT
    for gid in gt_ids:
        if gid not in classified_gt:
            errors["missed"].append(gid)

    # False positives
    for pid in pred_ids:
        if pid not in classified_pred:
            errors["false_positive"].append(pid)

    return errors


def error_summary(errors: dict) -> dict:
    """Convert error dict to count/percentage summary."""
    counts = {k: len(v) for k, v in errors.items()}
    total = sum(counts.values())
    pcts = {k: round(100 * v / total, 1) if total > 0 else 0.0
            for k, v in counts.items()}
    # Return both direct keys (for backward compatibility) and nested dicts
    return {**counts, "counts": counts, "percentages": pcts, "total": total}
