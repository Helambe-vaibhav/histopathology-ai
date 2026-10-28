"""
test_metrics.py
---------------
Unit tests for pixel metrics and the PQ unit test demonstrating
that Dice can remain high while PQ correctly penalizes merged instances.

Run with: pytest tests/test_metrics.py -v
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from metrics import dice, iou, precision, recall, panoptic_quality, bootstrap_ci, report_metrics


# ---------------------------------------------------------------------------
# Test 1 — Perfect prediction
# ---------------------------------------------------------------------------

def test_perfect_prediction():
    gt = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    assert dice(gt, gt)      == pytest.approx(1.0, abs=1e-4)
    assert iou(gt, gt)       == pytest.approx(1.0, abs=1e-4)
    assert precision(gt, gt) == pytest.approx(1.0, abs=1e-4)
    assert recall(gt, gt)    == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 2 — Empty prediction / empty GT
# ---------------------------------------------------------------------------

def test_all_zeros():
    gt   = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4), dtype=np.uint8)
    # Both empty → metrics should be ~1.0 (smooth avoids 0/0)
    assert dice(pred, gt) > 0.99


def test_all_background_pred():
    gt   = np.array([[1, 1], [1, 0]])
    pred = np.zeros_like(gt)
    assert precision(pred, gt) == pytest.approx(1.0, abs=1e-4)  # no FP
    assert recall(pred, gt)    < 0.1                             # all FN


# ---------------------------------------------------------------------------
# Test 3 — PQ unit test (MUST demonstrate merged instance failure)
# ---------------------------------------------------------------------------

def test_pq_penalizes_merged_instances():
    """
    Ground truth: 3 separate nuclei (labels 1, 2, 3).
    Prediction:   1 large blob covering all three (label 1).

    Expected outcome:
      - Dice (pixel-level): reasonably high (most foreground pixels are covered)
      - PQ (instance-level): low, because the single blob matches at most 1 GT nucleus
        at IoU >= 0.5
    """
    H, W = 50, 150

    # Ground truth: three 40×40 nuclei spaced along the width
    gt_binary   = np.zeros((H, W), dtype=np.uint8)
    gt_instance = np.zeros((H, W), dtype=np.uint32)

    nuc_cols = [(5, 45), (55, 95), (105, 145)]
    for idx, (c_start, c_end) in enumerate(nuc_cols, start=1):
        gt_binary[:, c_start:c_end]   = 1
        gt_instance[:, c_start:c_end] = idx

    # Prediction: single blob covering the whole width
    pred_binary   = np.ones((H, W), dtype=np.uint8)
    pred_instance = np.ones((H, W), dtype=np.uint32)  # single label

    # Pixel-level Dice
    d = dice(pred_binary, gt_binary)

    # Instance-level PQ
    pq_result = panoptic_quality(pred_instance, gt_instance, iou_threshold=0.5)

    print(f"\n[PQ Unit Test]")
    print(f"  Dice (pixel-level): {d:.3f}")
    print(f"  PQ  (instance):     {pq_result['pq']:.3f}")
    print(f"  TP={pq_result['tp']}  FP={pq_result['fp']}  FN={pq_result['fn']}")

    # Dice should be moderate to high (lots of overlapping pixels)
    assert d > 0.5, f"Dice unexpectedly low: {d:.3f}"

    # PQ should be low — the blob can match at most 1 of 3 GT nuclei at IoU >= 0.5
    assert pq_result["pq"] < 0.5, \
        f"PQ should be penalized for merging, got {pq_result['pq']:.3f}"

    # At most 1 TP; FN should be >= 2
    assert pq_result["tp"] <= 1, f"Expected ≤1 TP, got {pq_result['tp']}"
    assert pq_result["fn"] >= 2, f"Expected ≥2 FN, got {pq_result['fn']}"


# ---------------------------------------------------------------------------
# Test 4 — PQ perfect instance match
# ---------------------------------------------------------------------------

def test_pq_perfect():
    """PQ should be 1.0 when pred == GT (instance-level)."""
    gt = np.array([
        [0, 1, 0, 2, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint32)

    result = panoptic_quality(gt, gt, iou_threshold=0.5)
    assert result["pq"] == pytest.approx(1.0, abs=1e-4)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


# ---------------------------------------------------------------------------
# Test 5 — Bootstrap CI sanity
# ---------------------------------------------------------------------------

def test_bootstrap_ci_bounds():
    """CI lower <= mean <= upper, and CI is non-degenerate."""
    preds   = [np.random.randint(0, 2, (100, 100)).astype(np.uint8) for _ in range(20)]
    targets = [np.random.randint(0, 2, (100, 100)).astype(np.uint8) for _ in range(20)]

    mean, lower, upper = bootstrap_ci(dice, preds, targets, n_bootstrap=200, seed=0)

    assert lower <= mean <= upper, f"CI not ordered: [{lower}, {mean}, {upper}]"
    assert upper > lower, "Degenerate CI (zero width)"


# ---------------------------------------------------------------------------
# Test 6 — report_metrics dictionary keys and values
# ---------------------------------------------------------------------------

def test_report_metrics():
    preds = [np.ones((20, 20), dtype=np.uint8), np.zeros((20, 20), dtype=np.uint8)]
    targets = [np.ones((20, 20), dtype=np.uint8), np.zeros((20, 20), dtype=np.uint8)]
    rep = report_metrics(preds, targets)
    for m in ["dice", "iou", "precision", "recall"]:
        assert f"{m}_mean" in rep
        assert f"{m}_lower" in rep
        assert f"{m}_upper" in rep
        assert rep[f"{m}_lower"] <= rep[f"{m}_mean"] <= rep[f"{m}_upper"]


if __name__ == "__main__":
    test_perfect_prediction()
    test_all_zeros()
    test_all_background_pred()
    test_pq_penalizes_merged_instances()
    test_pq_perfect()
    test_bootstrap_ci_bounds()
    test_report_metrics()
    print("All metric tests passed.")
