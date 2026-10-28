"""
features.py
-----------
Functions:
  extract_morphological_features()  - area, perimeter, circularity, eccentricity, mean_color
  nuclei_density()                  - nuclei count / tissue area
  nearest_neighbor_distances()      - KD-tree based NN distances between centroids
  measurement_agreement()           - count error and area error vs GT
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# Morphological features
# ---------------------------------------------------------------------------

def extract_morphological_features(
    instance_mask: np.ndarray,
    image: np.ndarray = None,
) -> pd.DataFrame:
    """
    Extract per-nucleus morphological features from an instance mask.

    Args:
        instance_mask: (H, W) uint32 instance label map (0 = background)
        image:         (H, W, 3) uint8 RGB image (optional, for mean color)

    Returns:
        DataFrame with columns:
          nucleus_id, area, perimeter, circularity, eccentricity,
          centroid_row, centroid_col,
          [mean_r, mean_g, mean_b  — if image provided]
    """
    records = []
    props = regionprops(instance_mask, intensity_image=None)

    for p in props:
        area = p.area
        perimeter = p.perimeter if p.perimeter > 0 else 1e-6
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        eccentricity = p.eccentricity  # 0 = circle, 1 = line
        cy, cx = p.centroid

        row = {
            "nucleus_id": int(p.label),
            "area": int(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "eccentricity": float(eccentricity),
            "centroid_row": float(cy),
            "centroid_col": float(cx),
        }

        if image is not None:
            # Mean RGB inside nucleus
            nucleus_pixels = image[instance_mask == p.label]
            row["mean_r"] = float(nucleus_pixels[:, 0].mean()) if len(nucleus_pixels) else 0.0
            row["mean_g"] = float(nucleus_pixels[:, 1].mean()) if len(nucleus_pixels) else 0.0
            row["mean_b"] = float(nucleus_pixels[:, 2].mean()) if len(nucleus_pixels) else 0.0

        records.append(row)

    return pd.DataFrame(records)


compute_morphological_features = extract_morphological_features


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------

def nuclei_density(instance_mask: np.ndarray, tissue_mask: np.ndarray) -> float:
    """
    Nuclei density = N / TissueArea (nuclei per tissue pixel).

    Args:
        instance_mask: (H, W) instance label map
        tissue_mask:   (H, W) binary tissue mask (1 = tissue)

    Returns:
        float density
    """
    n = int((np.unique(instance_mask) != 0).sum())
    tissue_area = int(tissue_mask.sum())
    return n / (tissue_area + 1e-8)


def nearest_neighbor_distances(centroids: np.ndarray) -> np.ndarray:
    """
    Compute nearest-neighbor distance for each nucleus centroid using KD-tree.

    Args:
        centroids: (N, 2) array of (row, col) centroid coordinates

    Returns:
        (N,) array of distances to nearest other nucleus
        Returns array of zeros if fewer than 2 nuclei.
    """
    if len(centroids) < 2:
        return np.zeros(len(centroids))

    tree = KDTree(centroids)
    # k=2: closest is self (distance 0), second is nearest neighbor
    distances, _ = tree.query(centroids, k=2)
    return distances[:, 1]


def extract_spatial_features(
    instance_mask: np.ndarray,
    tissue_mask: np.ndarray,
) -> dict:
    """
    Compute spatial features for an image.

    Returns:
        dict with nuclei_count, density, mean_nn_distance, median_nn_distance
    """
    props = regionprops(instance_mask)
    centroids = np.array([p.centroid for p in props]) if props else np.empty((0, 2))

    nn_dists = nearest_neighbor_distances(centroids)
    density = nuclei_density(instance_mask, tissue_mask)

    return {
        "nuclei_count": len(props),
        "nuclei_density": float(density),
        "mean_nn_distance": float(nn_dists.mean()) if len(nn_dists) else 0.0,
        "median_nn_distance": float(np.median(nn_dists)) if len(nn_dists) else 0.0,
    }


# ---------------------------------------------------------------------------
# Measurement agreement (second-level experiment)
# ---------------------------------------------------------------------------

def measurement_agreement(
    pred_instance: np.ndarray,
    gt_instance: np.ndarray,
) -> dict:
    """
    Compare predicted vs. GT quantitative measurements.

    Returns:
        dict with:
          pred_count, gt_count, count_error (relative)
          pred_mean_area, gt_mean_area, area_error (relative)
    """
    pred_props = regionprops(pred_instance)
    gt_props   = regionprops(gt_instance)

    pred_count = len(pred_props)
    gt_count   = len(gt_props)

    pred_areas = [p.area for p in pred_props]
    gt_areas   = [p.area for p in gt_props]

    pred_mean_area = float(np.mean(pred_areas)) if pred_areas else 0.0
    gt_mean_area   = float(np.mean(gt_areas))   if gt_areas   else 0.0

    count_error = abs(pred_count - gt_count) / (gt_count + 1e-8)
    area_error  = abs(pred_mean_area - gt_mean_area) / (gt_mean_area + 1e-8)

    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
        "count_error": float(count_error),
        "pred_mean_area": pred_mean_area,
        "gt_mean_area": gt_mean_area,
        "area_error": float(area_error),
    }


def aggregate_agreements(records: list) -> dict:
    """
    Aggregate per-image measurement_agreement dicts.
    Computes mean errors and Pearson correlation between pred/GT counts and areas.
    """
    from scipy.stats import pearsonr

    pred_counts = [r["pred_count"]     for r in records]
    gt_counts   = [r["gt_count"]       for r in records]
    pred_areas  = [r["pred_mean_area"] for r in records]
    gt_areas    = [r["gt_mean_area"]   for r in records]

    corr_count, _ = pearsonr(pred_counts, gt_counts) if len(records) > 2 else (0.0, 1.0)
    corr_area,  _ = pearsonr(pred_areas,  gt_areas)  if len(records) > 2 else (0.0, 1.0)

    return {
        "mean_count_error": float(np.mean([r["count_error"] for r in records])),
        "mean_area_error":  float(np.mean([r["area_error"]  for r in records])),
        "pearson_count":    float(corr_count),
        "pearson_area":     float(corr_area),
    }
