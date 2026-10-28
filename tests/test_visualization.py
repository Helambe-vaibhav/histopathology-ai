"""
test_visualization.py
---------------------
Unit tests for visualization utilities:
- plot_feature_distribution
- plot_organ_comparison
- plot_ablation_table

Run with: pytest tests/test_visualization.py -v
"""

import sys
sys.path.insert(0, "src")

import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization import (
    plot_feature_distribution,
    plot_organ_comparison,
    plot_ablation_table,
)


def test_plot_feature_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "dist.png"
        values = np.random.normal(100, 15, size=200)
        fig = plot_feature_distribution(values, "Nuclear Area", save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_plot_organ_comparison():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "organ.png"
        data = {
            "Breast": [0.81, 0.83, 0.85],
            "Lung": [0.78, 0.80],
            "Kidney": [0.84],
            "Prostate": [0.79],
        }
        fig = plot_organ_comparison(data, "Dice", save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0


def test_plot_ablation_table():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "table.png"
        records = [
            {"config": "M1: Baseline", "dice": "0.78", "iou": "0.64", "pq": "0.45", "count_error": "25%", "area_error": "18%"},
            {"config": "M4: Combined", "dice": "0.83", "iou": "0.71", "pq": "0.55", "count_error": "12%", "area_error": "8%"},
        ]
        fig = plot_ablation_table(records, save_path=str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0
