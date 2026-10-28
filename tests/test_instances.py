"""
test_instances.py
-----------------
Unit tests for instance separation and error classification.

Run with: pytest tests/test_instances.py -v
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from instances import binary_to_instances, match_instances, classify_errors, error_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_circle_mask(H=100, W=200):
    """Binary mask with two non-touching circles."""
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx1, cx2, r = H // 2, W // 4, 3 * W // 4, H // 4
    Y, X = np.ogrid[:H, :W]
    mask[(Y - cy)**2 + (X - cx1)**2 <= r**2] = 1
    mask[(Y - cy)**2 + (X - cx2)**2 <= r**2] = 1
    return mask


# ---------------------------------------------------------------------------
# Test 1 — Separated nuclei produce correct instance count
# ---------------------------------------------------------------------------

def test_two_nuclei_detected():
    mask = _two_circle_mask()
    instances = binary_to_instances(mask, min_distance=5)
    n = len(np.unique(instances)) - 1  # exclude background

    assert n == 2, f"Expected 2 instances, got {n}"


# ---------------------------------------------------------------------------
# Test 2 — Background stays zero
# ---------------------------------------------------------------------------

def test_background_is_zero():
    mask = _two_circle_mask()
    instances = binary_to_instances(mask, min_distance=5)
    # All background pixels in the binary mask should be 0 in instance mask
    assert (instances[mask == 0] == 0).all(), "Background pixels got non-zero labels"


# ---------------------------------------------------------------------------
# Test 3 — match_instances returns correct number of matches
# ---------------------------------------------------------------------------

def test_match_instances_perfect():
    gt = np.zeros((50, 100), dtype=np.uint32)
    gt[:, :45] = 1
    gt[:, 55:] = 2

    pred = gt.copy()   # identical

    matches = match_instances(pred, gt, iou_threshold=0.5)
    assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
    for pid, gid, iou_val in matches:
        assert iou_val == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 4 — classify_errors: correct category
# ---------------------------------------------------------------------------

def test_classify_correct():
    gt   = np.array([[0, 1, 0, 2]], dtype=np.uint32)
    pred = np.array([[0, 1, 0, 2]], dtype=np.uint32)

    errors = classify_errors(pred, gt)
    assert len(errors["correct"]) == 2
    assert len(errors["false_positive"]) == 0
    assert len(errors["missed"]) == 0


# ---------------------------------------------------------------------------
# Test 5 — classify_errors: false positive
# ---------------------------------------------------------------------------

def test_classify_false_positive():
    gt   = np.array([[0, 0, 0, 0]], dtype=np.uint32)
    pred = np.array([[0, 1, 0, 0]], dtype=np.uint32)

    errors = classify_errors(pred, gt)
    assert len(errors["false_positive"]) == 1
    assert len(errors["missed"]) == 0


# ---------------------------------------------------------------------------
# Test 6 — classify_errors: missed nucleus
# ---------------------------------------------------------------------------

def test_classify_missed():
    gt   = np.array([[0, 1, 0, 2]], dtype=np.uint32)
    pred = np.array([[0, 0, 0, 0]], dtype=np.uint32)

    errors = classify_errors(pred, gt)
    assert len(errors["missed"]) == 2
    assert len(errors["false_positive"]) == 0


# ---------------------------------------------------------------------------
# Test 7 — error_summary percentages sum to 100
# ---------------------------------------------------------------------------

def test_error_summary_percentages():
    gt   = np.array([[0, 1, 0, 2]], dtype=np.uint32)
    pred = np.array([[0, 1, 0, 0]], dtype=np.uint32)

    errors = classify_errors(pred, gt)
    summary = error_summary(errors)
    total_pct = sum(summary["percentages"].values())
    assert abs(total_pct - 100.0) < 0.1 or summary["total"] == 0


if __name__ == "__main__":
    test_two_nuclei_detected()
    test_background_is_zero()
    test_match_instances_perfect()
    test_classify_correct()
    test_classify_false_positive()
    test_classify_missed()
    test_error_summary_percentages()
    print("All instance tests passed.")
