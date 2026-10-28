"""
test_features.py
----------------
Unit tests for morphological, spatial, and agreement features.

Run with: pytest tests/test_features.py -v
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest
from features import (
    extract_morphological_features,
    extract_spatial_features,
    nuclei_density,
    nearest_neighbor_distances,
    measurement_agreement,
    aggregate_agreements,
)


def test_morphological_features_extraction():
    # Create two artificial nuclei
    mask = np.zeros((100, 100), dtype=np.uint32)
    mask[20:30, 20:30] = 1  # 100 px square
    mask[60:80, 60:80] = 2  # 400 px square

    df = extract_morphological_features(mask)
    assert len(df) == 2
    assert "area" in df.columns
    assert "circularity" in df.columns
    assert "eccentricity" in df.columns
    assert df.loc[df["nucleus_id"] == 1, "area"].values[0] == 100
    assert df.loc[df["nucleus_id"] == 2, "area"].values[0] == 400


def test_spatial_features_and_density():
    mask = np.zeros((100, 100), dtype=np.uint32)
    mask[20:30, 20:30] = 1
    mask[60:70, 60:70] = 2
    tissue = np.ones((100, 100), dtype=np.uint8)

    density = nuclei_density(mask, tissue)
    assert density == pytest.approx(2.0 / 10000, abs=1e-6)

    spatial = extract_spatial_features(mask, tissue)
    assert spatial["nuclei_count"] == 2
    assert spatial["nuclei_density"] == pytest.approx(2.0 / 10000, abs=1e-6)
    assert spatial["mean_nn_distance"] > 0


def test_measurement_agreement_and_aggregation():
    # 3 slides
    records = []
    for count in [50, 100, 150]:
        records.append({
            "pred_count": count,
            "gt_count": count + 5,
            "pred_mean_area": count * 2.0,
            "gt_mean_area": count * 2.1,
            "count_error": 5.0 / (count + 5),
            "area_error": 0.05,
        })

    summary = aggregate_agreements(records)
    assert "pearson_count" in summary
    assert "pearson_area" in summary
    assert summary["pearson_count"] > 0.99  # Perfectly linear relation
    assert summary["pearson_area"] > 0.99
