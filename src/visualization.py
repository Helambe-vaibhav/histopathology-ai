"""
visualization.py
----------------
Functions:
  plot_overlay()             - H&E + segmentation overlay
  plot_instance_overlay()    - H&E + colored instance overlay
  plot_error_map()           - green/red/blue TP/FP/FN overlay
  plot_feature_distribution()- histogram / violin of morphological features
  plot_ablation_table()      - render ablation results as a formatted table figure
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import label2rgb


# ---------------------------------------------------------------------------
# Segmentation overlays
# ---------------------------------------------------------------------------

def plot_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (1.0, 0.2, 0.2),
    ax=None,
    title: str = "Segmentation Overlay",
):
    """
    Draw binary prediction mask as a semi-transparent color overlay on the H&E image.

    Args:
        image:     (H, W, 3) uint8 RGB
        pred_mask: (H, W) binary array
        alpha:     overlay transparency
        color:     RGB tuple for the mask color
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    overlay = image.copy().astype(float) / 255.0
    for i, c in enumerate(color):
        overlay[:, :, i] = np.where(pred_mask > 0,
                                    (1 - alpha) * overlay[:, :, i] + alpha * c,
                                    overlay[:, :, i])
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_instance_overlay(
    image: np.ndarray,
    instance_mask: np.ndarray,
    ax=None,
    title: str = "Instance Overlay",
    alpha: float = 0.5,
):
    """
    Draw each nucleus in a distinct random color.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    colored = label2rgb(instance_mask, image=image, alpha=alpha, bg_label=0, kind="overlay")
    ax.imshow(colored)
    ax.set_title(title)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Error map (TP=green, FP=red, FN=blue)
# ---------------------------------------------------------------------------

def plot_error_map(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    ax=None,
    title: str = "Error Map",
    alpha: float = 0.5,
):
    """
    Overlay a pixel-level error map on the H&E image.

    Green  = True Positive
    Red    = False Positive
    Blue   = False Negative

    Args:
        image:     (H, W, 3) uint8 RGB
        pred_mask: (H, W) binary prediction
        gt_mask:   (H, W) binary ground truth
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)

    overlay = image.copy().astype(float) / 255.0
    # Green for TP
    overlay[:, :, 0] = np.where(tp, (1-alpha)*overlay[:,:,0], overlay[:,:,0])
    overlay[:, :, 1] = np.where(tp, (1-alpha)*overlay[:,:,1] + alpha, overlay[:,:,1])
    overlay[:, :, 2] = np.where(tp, (1-alpha)*overlay[:,:,2], overlay[:,:,2])
    # Red for FP
    overlay[:, :, 0] = np.where(fp, (1-alpha)*overlay[:,:,0] + alpha, overlay[:,:,0])
    overlay[:, :, 1] = np.where(fp, (1-alpha)*overlay[:,:,1], overlay[:,:,1])
    overlay[:, :, 2] = np.where(fp, (1-alpha)*overlay[:,:,2], overlay[:,:,2])
    # Blue for FN
    overlay[:, :, 0] = np.where(fn, (1-alpha)*overlay[:,:,0], overlay[:,:,0])
    overlay[:, :, 1] = np.where(fn, (1-alpha)*overlay[:,:,1], overlay[:,:,1])
    overlay[:, :, 2] = np.where(fn, (1-alpha)*overlay[:,:,2] + alpha, overlay[:,:,2])

    overlay = np.clip(overlay, 0, 1)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")

    patches = [
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red",   label="FP"),
        mpatches.Patch(color="blue",  label="FN"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Feature distributions
# ---------------------------------------------------------------------------

def plot_feature_distribution(
    values: np.ndarray,
    feature_name: str,
    bins: int = 30,
    save_path: str = None,
):
    """Plot histogram of a morphological feature across all nuclei."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {feature_name}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_organ_comparison(
    data: dict,
    feature: str,
    save_path: str = None,
):
    """
    Box/violin plot comparing a feature across organs.

    Args:
        data: dict mapping organ_name → list of feature values
        feature: feature name string
    """
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 4))
    labels = list(data.keys())
    vals   = [data[k] for k in labels]
    try:
        ax.violinplot(vals, positions=range(len(labels)), showmedians=True)
    except Exception:
        ax.boxplot(vals, positions=range(len(labels)))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by Organ (dataset variability, not biological ground truth)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Ablation result table figure
# ---------------------------------------------------------------------------

def plot_ablation_table(results: list, save_path: str = None):
    """
    Render ablation ladder as a matplotlib table figure.

    Args:
        results: list of dicts with keys:
                 config, dice, iou, pq, count_error, area_error
    """
    import pandas as pd
    df = pd.DataFrame(results)
    cols = ["config", "dice", "iou", "pq", "count_error", "area_error"]
    df = df.reindex(columns=cols)

    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    ax.set_title("Ablation Ladder", pad=12, fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
