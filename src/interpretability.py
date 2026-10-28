"""
interpretability.py
-------------------
End-to-end Visual Interpretability & Step-by-Step Pipeline Reconstruction Engine.

Transforms the histopathology AI pipeline from a "black box" into a transparent "glass box"
by reconstructing high-resolution diagnostic images at every single algorithmic stage:

  Stage 0:  Raw Input Tissue Crop
  Stage 1:  Tissue Foreground Mask & Glass QC (Tissue kept vs Glass rejected)
  Stage 2A: Macenko Stain Normalization (Multi-hospital color standardization)
  Stage 2B: Pure Hematoxylin (H-Stain) Optical Density Channel (Nuclei only)
  Stage 2C: Pure Eosin (E-Stain) Optical Density Channel (Cytoplasm/Stroma only)
  Stage 3:  Sliding Tile Grid & Overlap Coverage (256x256 tiles, 50% stride)
  Stage 4:  U-Net Neural Network Confidence Heatmap (Continuous 0.0 - 1.0 probability)
  Stage 5:  Calibrated Binary Mask & Morphological Cleanup (Hole-filling + Dust removal)
  Stage 6:  Distance Transform Topography Map ("The Mountain Range")
  Stage 7:  Peak Local Maxima Markers (Seed points placed at cell summits)
  Stage 8:  Marker-Controlled Watershed Instance Segmentation (Touching cells cleaved)
  Stage 9:  Clinical Cellularity & KD-Tree Spatial Crowding Graph
  Stage 10: Diagnostic Error Map (Optional: Green=TP, Red=FP, Blue=FN vs Ground Truth)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import KDTree, Delaunay
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.color import label2rgb

import torch
from torchvision import transforms


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array safely to [0, 1] range."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-7:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def reconstruct_tissue_qc_overlay(image: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    """
    Stage 1: Reconstruct an overlay showing tissue kept vs empty glass rejected.
    Tissue is shown in full color; glass background is dimmed/tinted.
    """
    h, w = image.shape[:2]
    overlay = image.copy().astype(np.float32) / 255.0
    # Dim non-tissue glass
    glass_mask = (tissue_mask == 0)
    overlay[glass_mask] = overlay[glass_mask] * 0.3 + np.array([0.2, 0.2, 0.25]) * 0.7
    
    # Draw green border around tissue contours
    boundaries = find_boundaries(tissue_mask, mode="thick")
    overlay[boundaries] = [0.0, 1.0, 0.0]
    
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


def reconstruct_stain_channels(image: np.ndarray, beta: float = 0.15) -> tuple:
    """
    Stage 2: Deconstruct H&E into isolated pure Hematoxylin and pure Eosin optical channels.
    Returns:
      (h_reconstruction_rgb, e_reconstruction_rgb, h_density_gray, e_density_gray)
    """
    img = np.clip(image.astype(np.float64), 1, 255)
    od = -np.log(img / 255.0)
    od_flat = od.reshape(-1, 3)
    mask = np.all(od_flat > beta, axis=1)
    
    od_tissue = od_flat[mask]
    if od_tissue.shape[0] < 10:
        # Fallback standard stain matrix
        HE = np.array([[0.644, 0.717, 0.267],
                       [0.093, 0.954, 0.283]])
    else:
        _, _, Vt = np.linalg.svd(od_tissue, full_matrices=False)
        plane = Vt[:2].T
        proj = od_tissue @ plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        phi1 = np.percentile(angles, 1.0)
        phi2 = np.percentile(angles, 99.0)
        v1 = plane @ np.array([np.cos(phi1), np.sin(phi1)])
        v2 = plane @ np.array([np.cos(phi2), np.sin(phi2)])
        if v1[0] < 0: v1 = -v1
        if v2[0] < 0: v2 = -v2
        if v1[2] > v2[2]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        HE = HE / np.linalg.norm(HE, axis=1, keepdims=True)

    # Solve for concentrations: OD = C @ HE  -> C = OD @ pinv(HE)
    H, W = image.shape[:2]
    HE_pinv = np.linalg.pinv(HE)  # (3, 2)
    C = od_flat @ HE_pinv         # (H*W, 2)
    C = np.maximum(C, 0)
    
    C_H = C[:, 0:1] # (H*W, 1)
    C_E = C[:, 1:2] # (H*W, 1)
    
    # Reconstruct pure H RGB image: I_H = 255 * exp(-C_H @ HE[0:1, :])
    OD_H = C_H @ HE[0:1, :]
    img_H = np.clip(np.exp(-OD_H) * 255.0, 0, 255).astype(np.uint8).reshape(H, W, 3)
    
    # Reconstruct pure E RGB image: I_E = 255 * exp(-C_E @ HE[1:2, :])
    OD_E = C_E @ HE[1:2, :]
    img_E = np.clip(np.exp(-OD_E) * 255.0, 0, 255).astype(np.uint8).reshape(H, W, 3)
    
    # Grayscale density maps (0 to 255)
    h_gray = (np.clip(_normalize_01(C_H.reshape(H, W)) * 255.0, 0, 255)).astype(np.uint8)
    e_gray = (np.clip(_normalize_01(C_E.reshape(H, W)) * 255.0, 0, 255)).astype(np.uint8)
    
    return img_H, img_E, h_gray, e_gray


def reconstruct_tiling_grid_overlay(image: np.ndarray, tile_size: int = 256, stride: int = 128) -> np.ndarray:
    """
    Stage 3: Reconstruct the slide with the sliding tile grid and 50% overlap lines.
    """
    h, w = image.shape[:2]
    canvas = image.copy().astype(np.float32) / 255.0
    
    # Draw grid boxes
    for y in range(0, max(1, h - tile_size + 1), stride):
        for x in range(0, max(1, w - tile_size + 1), stride):
            # Box borders
            y1, y2 = y, min(h - 1, y + tile_size)
            x1, x2 = x, min(w - 1, x + tile_size)
            canvas[y1:y1+2, x1:x2] = [0.0, 0.8, 1.0] # Top cyan
            canvas[y2-2:y2, x1:x2] = [0.0, 0.8, 1.0] # Bottom cyan
            canvas[y1:y2, x1:x1+2] = [0.0, 0.8, 1.0] # Left cyan
            canvas[y1:y2, x2-2:x2] = [0.0, 0.8, 1.0] # Right cyan
            
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def reconstruct_probability_heatmap(image: np.ndarray, prob_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Stage 4: Reconstruct model confidence heatmap overlaid on the original tissue.
    Uses 'plasma' colormap to show network attention.
    """
    cmap = plt.get_cmap("plasma")
    heatmap_rgba = cmap(prob_map)
    heatmap_rgb = heatmap_rgba[:, :, :3]
    
    img_norm = image.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * img_norm + alpha * heatmap_rgb
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def reconstruct_distance_transform_heatmap(dist_map: np.ndarray) -> np.ndarray:
    """
    Stage 6: Reconstruct the Euclidean distance transform topography ("Mountain Range").
    Uses 'turbo' or 'viridis' colormap.
    """
    norm_dist = _normalize_01(dist_map)
    cmap = plt.get_cmap("viridis")
    dist_rgb = cmap(norm_dist)[:, :, :3]
    return np.clip(dist_rgb * 255.0, 0, 255).astype(np.uint8)


def reconstruct_markers_overlay(image: np.ndarray, markers: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Stage 7: Reconstruct the seed markers detected at the local maxima peaks.
    Draws high-visibility red target dots at each cell peak.
    """
    import cv2
    canvas = image.copy()
    coords = np.argwhere(markers > 0)
    for y, x in coords:
        cv2.circle(canvas, (x, y), radius, (255, 0, 0), -1)       # Red fill
        cv2.circle(canvas, (x, y), radius + 1, (255, 255, 255), 1) # White ring
    return canvas


def reconstruct_watershed_instances(image: np.ndarray, instance_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Stage 8: Reconstruct colored instance map with boundaries separating touching nuclei.
    """
    colored = label2rgb(instance_mask, image=image, alpha=alpha, bg_label=0, kind="overlay")
    boundaries = find_boundaries(instance_mask, mode="thick")
    # Draw yellow boundary lines between touching cells
    colored[boundaries] = [1.0, 1.0, 0.0]
    return np.clip(colored * 255.0, 0, 255).astype(np.uint8)


def reconstruct_spatial_density_graph(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    """
    Stage 9: Reconstruct KD-Tree / Delaunay spatial connectivity graph.
    Visualizes tumor cellularity and cell crowding directly on the tissue.
    """
    from skimage.measure import regionprops
    props = regionprops(instance_mask)
    if len(props) < 3:
        return image.copy()
        
    centroids = np.array([p.centroid for p in props]) # (N, 2) in (y, x)
    points_xy = centroids[:, [1, 0]]                  # (N, 2) in (x, y)
    
    # KD-Tree for local density (mean distance to 5 nearest neighbors)
    kdtree = KDTree(points_xy)
    k_nn = min(6, len(points_xy))
    dists, _ = kdtree.query(points_xy, k=k_nn)
    mean_nn_dist = dists[:, 1:].mean(axis=1) if dists.ndim > 1 else np.ones(len(points_xy)) * 50.0
    
    # Closer distance = higher crowding (inverse)
    crowding_score = _normalize_01(1.0 / np.maximum(mean_nn_dist, 1.0))
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image)
    
    # Delaunay triangulation edges
    try:
        tri = Delaunay(points_xy)
        ax.triplot(points_xy[:, 0], points_xy[:, 1], tri.simplices, color="#00E5FF", lw=0.6, alpha=0.55)
    except Exception:
        pass
        
    # Scatter centroids colored by crowding
    scatter = ax.scatter(
        points_xy[:, 0], points_xy[:, 1],
        c=crowding_score, cmap="coolwarm", s=18, edgecolors="white", linewidths=0.4, zorder=4
    )
    ax.set_title(f"Clinical Spatial Graph (N={len(props)} nuclei)\nBlue=Spaced out, Red=Crowded Tumor", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    
    # Convert figure to RGB numpy array
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    img_pil = Image.open(buf).convert("RGB")
    return np.array(img_pil)


def reconstruct_error_map_overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Stage 10: Reconstruct diagnostic error map.
    Green = True Positive, Red = False Positive, Blue = False Negative.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    
    overlay = image.copy().astype(float) / 255.0
    # TP = Green
    overlay[:, :, 0] = np.where(tp, (1 - alpha) * overlay[:, :, 0], overlay[:, :, 0])
    overlay[:, :, 1] = np.where(tp, (1 - alpha) * overlay[:, :, 1] + alpha, overlay[:, :, 1])
    overlay[:, :, 2] = np.where(tp, (1 - alpha) * overlay[:, :, 2], overlay[:, :, 2])
    # FP = Red
    overlay[:, :, 0] = np.where(fp, (1 - alpha) * overlay[:, :, 0] + alpha, overlay[:, :, 0])
    overlay[:, :, 1] = np.where(fp, (1 - alpha) * overlay[:, :, 1], overlay[:, :, 1])
    overlay[:, :, 2] = np.where(fp, (1 - alpha) * overlay[:, :, 2], overlay[:, :, 2])
    # FN = Blue
    overlay[:, :, 0] = np.where(fn, (1 - alpha) * overlay[:, :, 0], overlay[:, :, 0])
    overlay[:, :, 1] = np.where(fn, (1 - alpha) * overlay[:, :, 1], overlay[:, :, 1])
    overlay[:, :, 2] = np.where(fn, (1 - alpha) * overlay[:, :, 2] + alpha, overlay[:, :, 2])
    
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


def run_full_pipeline_interpretability(
    image: np.ndarray,
    model,
    device,
    gt_binary: np.ndarray = None,
    tile_size: int = 256,
    stride: int = 128,
    threshold: float = 0.15,
    min_dist: int = 5,
    use_tta: bool = True,
) -> dict:
    """
    Executes the full pipeline while capturing and reconstructing intermediate images
    at every single stage. Returns a rich dictionary of reconstructed diagnostic artifacts.
    """
    from preprocessing import create_tissue_mask, macenko_normalize, fit_macenko_reference, tile_image
    from evaluate import stitch_hann_predictions, predict_tile_tta
    from instances import binary_to_instances
    
    H, W = image.shape[:2]
    stages = {}
    
    # 0. Raw Input Image
    stages["0_input_image"] = image.copy()
    
    # 1. Tissue Mask & Glass QC
    tissue_mask = create_tissue_mask(image)
    stages["1_tissue_qc_overlay"] = reconstruct_tissue_qc_overlay(image, tissue_mask)
    
    # 2. Stain Separation & Normalization
    try:
        ref_maxC, ref_HE = fit_macenko_reference(image)
        norm_img = macenko_normalize(image, target_concentrations=(ref_maxC, ref_HE))
    except Exception:
        norm_img = image.copy()
    stages["2a_macenko_normalized"] = norm_img
    
    img_H, img_E, h_gray, e_gray = reconstruct_stain_channels(image)
    stages["2b_hematoxylin_channel"] = img_H
    stages["2c_eosin_channel"] = img_E
    
    # 3. Tiling & Overlap Mosaic
    stages["3_tiling_grid_overlay"] = reconstruct_tiling_grid_overlay(image, tile_size=tile_size, stride=stride)
    
    # 4. Neural Network Inference
    # Use stain-normalized image to match model training distribution
    infer_img = norm_img if norm_img is not None else image
    tiles_info = tile_image(infer_img, tile_size=tile_size, stride=stride)
    to_tensor = transforms.ToTensor()
    pred_tiles_info = []
    
    model.eval()
    with torch.no_grad():
        for t_info in tiles_info:
            tile = t_info["tile"]
            if use_tta:
                prob = predict_tile_tta(model, tile, device, to_tensor)
            else:
                t_tensor = to_tensor(Image.fromarray(tile)).unsqueeze(0).to(device)
                logits = model(t_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_tiles_info.append({
                "tile": prob,
                "h_start": t_info["h_start"],
                "w_start": t_info["w_start"],
            })
            
    stitched_prob = stitch_hann_predictions(pred_tiles_info, output_shape=(H, W), tile_size=tile_size)
    stages["4_model_probability_heatmap"] = reconstruct_probability_heatmap(image, stitched_prob)
    
    # 5. Calibrated Binary Mask, Tissue QC Gating & Cleanup
    raw_binary = (stitched_prob > threshold).astype(np.uint8) * tissue_mask
    clean_binary = ndi.binary_fill_holes(raw_binary).astype(np.uint8)
    stages["5_binary_mask_cleaned"] = (clean_binary * 255).astype(np.uint8)
    
    # 6. Optical Intensity Guidance Surface ("The Mountain Range")
    # Deconstruct saddles using Pure Hematoxylin optical gradient
    dist_map = ndi.distance_transform_edt(clean_binary.astype(bool))
    grad = ndi.gaussian_gradient_magnitude(h_gray.astype(float), sigma=1.0)
    norm_grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    surface = dist_map * (1.0 - 0.4 * norm_grad)
    stages["6_distance_transform"] = reconstruct_distance_transform_heatmap(surface)
    
    # 7. Seed Markers (Peak Local Maxima on Optical Mountain Range)
    coords = peak_local_max(surface, min_distance=min_dist, labels=clean_binary.astype(bool), threshold_abs=1.0)
    markers = np.zeros_like(surface, dtype=np.int32)
    if len(coords) > 0:
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        labels_ws = watershed(-surface, markers, mask=clean_binary.astype(bool)).astype(np.uint32)
    else:
        labels_ws, _ = ndi.label(clean_binary)
        labels_ws = labels_ws.astype(np.uint32)
        
    # Morphological Biological Filtering (prune non-cellular collagen streaks & debris)
    from skimage.measure import regionprops
    props = regionprops(labels_ws)
    valid_labels = [p.label for p in props if p.area >= 20 and p.eccentricity <= 0.98]
    cleaned_ws = np.zeros_like(labels_ws)
    for new_idx, orig_lbl in enumerate(valid_labels, 1):
        cleaned_ws[labels_ws == orig_lbl] = new_idx
    labels_ws = cleaned_ws.astype(np.uint32)
    
    stages["7_markers_overlay"] = reconstruct_markers_overlay(image, markers)
    stages["8_watershed_instances"] = reconstruct_watershed_instances(image, labels_ws)
    
    # 9. Clinical Cellularity & Spatial Graph
    stages["9_spatial_crowding_graph"] = reconstruct_spatial_density_graph(image, labels_ws)
    
    # 10. Optional Error Map
    if gt_binary is not None:
        stages["10_diagnostic_error_map"] = reconstruct_error_map_overlay(image, clean_binary, gt_binary)
        
    stages["labels_instances"] = labels_ws
    stages["meta"] = {
        "nuclei_count": int(labels_ws.max()),
        "mean_confidence": float(stitched_prob[clean_binary > 0].mean()) if clean_binary.sum() > 0 else 0.0,
        "tissue_coverage_pct": float(tissue_mask.mean() * 100.0),
    }
    
    return stages


def save_master_interpretability_figure(stages: dict, save_path: str = "interpretability_dashboard.png"):
    """
    Assembles all reconstructed pipeline stages into a single high-resolution
    3x4 clinical interpretability dashboard.
    """
    panels = [
        ("1. Raw Slide Input", stages.get("0_input_image")),
        ("2. Glass QC & Tissue Mask", stages.get("1_tissue_qc_overlay")),
        ("3. Macenko Normalization", stages.get("2a_macenko_normalized")),
        ("4. Pure Hematoxylin (H-Stain)", stages.get("2b_hematoxylin_channel")),
        ("5. Pure Eosin (E-Stain)", stages.get("2c_eosin_channel")),
        ("6. Sliding Tile Grid (256x256)", stages.get("3_tiling_grid_overlay")),
        ("7. U-Net Attention Heatmap", stages.get("4_model_probability_heatmap")),
        ("8. Cleaned Binary Mask", stages.get("5_binary_mask_cleaned")),
        ("9. Distance Mountain Range", stages.get("6_distance_transform")),
        ("10. Watershed Peak Seeds", stages.get("7_markers_overlay")),
        ("11. Watershed Instances Cleaved", stages.get("8_watershed_instances")),
        ("12. Spatial Crowding Graph", stages.get("9_spatial_crowding_graph")),
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=150)
    axes = axes.flatten()
    
    for idx, (title, img_arr) in enumerate(panels):
        ax = axes[idx]
        if img_arr is not None:
            if img_arr.ndim == 2:
                ax.imshow(img_arr, cmap="gray")
            else:
                ax.imshow(img_arr)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")
        
    meta = stages.get("meta", {})
    plt.suptitle(
        f"Computational Pathology End-to-End Pipeline Interpretability Dashboard\n"
        f"Detected Nuclei: {meta.get('nuclei_count', 'N/A')} | "
        f"Mean Confidence: {meta.get('mean_confidence', 0.0):.2f} | "
        f"Tissue Area: {meta.get('tissue_coverage_pct', 0.0):.1f}%",
        fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Master Interpretability Dashboard saved to: {save_path}")
