"""
evaluate.py
-----------
Comprehensive whole-slide evaluation script:
1. Loads test whole-slide images and masks
2. Evaluates tile-based forward pass with 50% overlap blend stitching
3. Computes pixel-level metrics (Dice, IoU, Precision, Recall)
4. Computes 1000-sample bootstrap 95% Confidence Intervals at image level
5. Separates nuclei instances via distance transform + watershed
6. Computes Panoptic Quality (PQ = SQ * DQ)
7. Computes measurement agreement (nuclei counts, mean area, Pearson r)
8. Classifies error modes (Correct, Merged, Split, Missed, False Positive)
9. Generates and saves visual error maps (Green=TP, Red=FP, Blue=FN)

Usage:
    python src/evaluate.py --config configs/config.yaml --weights results/checkpoints/unet_best.pth
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from models import build_model
from preprocessing import (
    load_image,
    load_mask,
    tile_image,
    stitch_predictions,
    macenko_normalize,
    fit_macenko_reference,
    create_tissue_mask,
)
from metrics import compute_pixel_metrics, bootstrap_ci, report_metrics, panoptic_quality
from instances import binary_to_instances, classify_errors, error_summary
from features import (
    compute_morphological_features,
    measurement_agreement,
    aggregate_agreements,
    extract_spatial_features,
)
from visualization import (
    plot_error_map,
    plot_instance_overlay,
    plot_feature_distribution,
    plot_organ_comparison,
)
from hardware import setup_hardware

from scipy import ndimage


ORGAN_MAP = {
    "AR": "Breast", "A7": "Breast", "AO": "Breast", "E2": "Breast",
    "21": "Lung", "49": "Lung", "50": "Lung", "69": "Lung", "18": "Lung", "38": "Lung",
    "B0": "Kidney", "UZ": "Kidney", "RD": "Kidney", "KB": "Kidney",
    "HE": "Prostate", "G9": "Prostate", "CH": "Prostate", "EJ": "Prostate",
    "DK": "Bladder", "ZF": "Bladder",
    "A6": "Colon", "AY": "Colon",
    "BC": "Liver", "G2": "Liver",
}


def get_organ_from_stem(stem: str) -> str:
    """Map slide TCGA stem to its anatomical organ."""
    parts = stem.split("-")
    if len(parts) >= 2 and parts[1] in ORGAN_MAP:
        return ORGAN_MAP[parts[1]]
    return "Other"


def make_hann_window(size: int = 256, min_weight: float = 0.05) -> np.ndarray:
    w1d = np.hanning(size)
    w2d = np.outer(w1d, w1d).astype(np.float32)
    return np.maximum(w2d, min_weight)


def stitch_hann_predictions(preds: list, output_shape: tuple, tile_size: int = 256) -> np.ndarray:
    H, W = output_shape
    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    hann_w = make_hann_window(tile_size)
    for p in preds:
        h, w = p["h_start"], p["w_start"]
        tile_pred = p["tile"]
        th, tw = tile_pred.shape
        w_sub = hann_w[:th, :tw]
        accum[h:h + th, w:w + tw] += tile_pred * w_sub
        weight[h:h + th, w:w + tw] += w_sub
    return accum / np.maximum(weight, 1e-6)


def predict_tile_tta(model: torch.nn.Module, tile_img: np.ndarray, device: torch.device, to_tensor: transforms.ToTensor) -> np.ndarray:
    t = to_tensor(Image.fromarray(tile_img)).unsqueeze(0).to(device)
    t0 = t
    t90 = torch.rot90(t, 1, [2, 3])
    t180 = torch.rot90(t, 2, [2, 3])
    t270 = torch.rot90(t, 3, [2, 3])
    t_flip = torch.flip(t, [3])
    tf90 = torch.rot90(t_flip, 1, [2, 3])
    tf180 = torch.rot90(t_flip, 2, [2, 3])
    tf270 = torch.rot90(t_flip, 3, [2, 3])
    batch = torch.cat([t0, t90, t180, t270, t_flip, tf90, tf180, tf270], dim=0)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    p0 = probs[0]
    p90 = torch.rot90(probs[1], -1, [1, 2])
    p180 = torch.rot90(probs[2], -2, [1, 2])
    p270 = torch.rot90(probs[3], -3, [1, 2])
    pf0 = torch.flip(probs[4], [2])
    pf90 = torch.flip(torch.rot90(probs[5], -1, [1, 2]), [2])
    pf180 = torch.flip(torch.rot90(probs[6], -2, [1, 2]), [2])
    pf270 = torch.flip(torch.rot90(probs[7], -3, [1, 2]), [2])
    avg_p = (p0 + p90 + p180 + p270 + pf0 + pf90 + pf180 + pf270) / 8.0
    return avg_p.squeeze().cpu().numpy()


def postprocess_binary_mask(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    filled = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    lbls, num = ndimage.label(filled)
    if num == 0:
        return filled
    counts = np.bincount(lbls.ravel())
    mask_sizes = counts >= min_area
    mask_sizes[0] = False
    return mask_sizes[lbls].astype(np.uint8)


def evaluate_pipeline(
    config_path: str,
    weights_path: str = None,
    model_type: str = None,
    output_csv: str = "evaluation_results.csv",
    save_plots: bool = True,
    threshold: float = None,
    min_dist: int = None,
    use_tta: bool = True,
    use_hann: bool = True,
    min_area: int = 20,
    use_tissue_mask: bool = True,
    use_optical_guidance: bool = True,
    max_eccentricity: float = 0.98,
    stain_norm: str = None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if model_type:
        cfg["model"]["type"] = model_type

    if threshold is None:
        threshold = float(cfg.get("inference", {}).get("threshold", 0.40))

    if min_dist is None:
        min_dist = cfg.get("watershed", {}).get("min_distance", 5)

    # Enforce hardware resource safety limits (90% GPU and RAM caps)
    hw_cfg = cfg.get("hardware", {})
    hw_profile = setup_hardware(
        gpu_fraction=float(hw_cfg.get("gpu_memory_fraction", 0.90)),
        max_ram_fraction=float(hw_cfg.get("max_ram_fraction", 0.90)),
    )
    device = hw_profile["device"]
    print(f"Running evaluation on: {device} (Thread count: {hw_profile['cpu_threads_active']})")

    # Build and load model
    model = build_model(cfg["model"]).to(device)
    if weights_path and Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from: {weights_path}")
    else:
        print("Notice: No pre-trained weights provided or found. Running with current model initialization.")
    model.eval()

    data_root = Path(cfg["data"]["root"])
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    results_fig_dir = Path("results/figures")
    results_tbl_dir = Path("results/tables")
    results_fig_dir.mkdir(parents=True, exist_ok=True)
    results_tbl_dir.mkdir(parents=True, exist_ok=True)

    # Discover test images (from data/MoNuSeg/tiles/test or discover from masks)
    # Check if there is a test tile split
    test_tiles_dir = data_root / "tiles" / "test" / "images"
    if test_tiles_dir.exists():
        # extract unique slide stems
        test_stems = sorted(list(set(p.name.split("_t")[0] for p in test_tiles_dir.glob("*.png"))))
    else:
        # fallback to all images in images_dir
        test_stems = [p.stem for p in sorted(images_dir.glob("*")) if p.suffix.lower() in (".png", ".tif", ".jpg")]

    print(f"Evaluating {len(test_stems)} test slide(s): {test_stems}\n")

    per_image_results = []
    all_pred_masks = []
    all_gt_masks = []
    all_pred_counts = []
    all_gt_counts = []
    all_pred_areas = []
    all_gt_areas = []
    all_pred_props = []
    all_agreement_records = []

    to_tensor = transforms.ToTensor()
    tile_size = cfg["data"].get("tile_size", 256)
    stride = cfg["data"].get("stride", 128)

    for stem in tqdm(test_stems, desc="Evaluating slides"):
        # Load image
        img_candidates = list(images_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        image = load_image(str(img_candidates[0]))
        H, W, _ = image.shape

        # Optional Stain Normalization prior to tiling
        if stain_norm == "macenko":
            try:
                ref_maxC, ref_HE = fit_macenko_reference(image)
                infer_img = macenko_normalize(image, target_concentrations=(ref_maxC, ref_HE))
            except Exception:
                infer_img = image
        else:
            infer_img = image

        # Load GT binary mask and instance mask using load_mask
        mask_candidate = masks_dir / f"{stem}_mask.png"
        inst_candidate = masks_dir / f"{stem}_instances.npy"

        if not mask_candidate.exists():
            continue
        gt_raw = load_mask(str(mask_candidate))
        gt_binary = (gt_raw > 127).astype(np.uint8)
        if inst_candidate.exists():
            gt_instances = np.load(str(inst_candidate)).astype(np.int32)
        else:
            gt_instances = binary_to_instances(gt_binary)

        # 1. Tile inference (with optional D4 Dihedral 8-fold TTA)
        tiles_info = tile_image(infer_img, tile_size=tile_size, stride=stride)
        pred_tiles_info = []

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

        # 2. Seamless overlap blend stitch (2D Hann window weighting or linear blend)
        if use_hann:
            stitched_prob = stitch_hann_predictions(pred_tiles_info, output_shape=(H, W), tile_size=tile_size)
        else:
            stitched_prob = stitch_predictions(pred_tiles_info, output_shape=(H, W))

        # Binary thresholding & morphological hole-filling / artifact removal
        raw_binary = (stitched_prob > threshold).astype(np.uint8)
        tissue_mask = create_tissue_mask(image) if use_tissue_mask else np.ones((H, W), dtype=np.uint8)
        if use_tissue_mask:
            raw_binary = raw_binary * tissue_mask
        pred_binary = postprocess_binary_mask(raw_binary, min_area=min_area)

        # 3. Instance separation (watershed with optical guidance & morphological shape filtering)
        intensity_img = image if use_optical_guidance else None
        pred_instances = binary_to_instances(
            pred_binary,
            min_distance=min_dist,
            intensity_image=intensity_img,
            min_area=min_area,
            max_eccentricity=max_eccentricity,
        )

        # 4. Pixel-level metrics
        pix_metrics = compute_pixel_metrics(pred_binary, gt_binary)

        # 5. Instance-level metrics (Panoptic Quality)
        pq_res = panoptic_quality(pred_instances, gt_instances, iou_threshold=0.5)
        pq, sq, dq = pq_res["pq"], pq_res["sq"], pq_res["dq"]

        # 6. Error classification
        error_df = classify_errors(pred_instances, gt_instances, iou_threshold=0.5)
        err_summary = error_summary(error_df)

        # 7. Morphological agreement & spatial features
        organ = get_organ_from_stem(stem)
        gt_count = int(gt_instances.max())
        pred_count = int(pred_instances.max())
        gt_props = compute_morphological_features(gt_instances)
        pred_props = compute_morphological_features(pred_instances)
        all_pred_props.append(pred_props)

        gt_mean_area = float(gt_props["area"].mean()) if len(gt_props) > 0 else 0.0
        pred_mean_area = float(pred_props["area"].mean()) if len(pred_props) > 0 else 0.0
        pred_mean_circ = float(pred_props["circularity"].mean()) if len(pred_props) > 0 else 0.0
        pred_mean_ecc = float(pred_props["eccentricity"].mean()) if len(pred_props) > 0 else 0.0

        rel_count_err = abs(pred_count - gt_count) / max(1, gt_count)
        rel_area_err = abs(pred_mean_area - gt_mean_area) / max(1e-4, gt_mean_area)

        # Spatial Cellularity & Density Network
        spatial_feats = extract_spatial_features(pred_instances, tissue_mask)
        density = spatial_feats["nuclei_density"]
        mean_nn_dist = spatial_feats["mean_nn_distance"]

        # Measurement agreement record for Pearson correlation
        agree_rec = measurement_agreement(pred_instances, gt_instances)
        all_agreement_records.append(agree_rec)

        # Collect
        all_pred_masks.append(pred_binary)
        all_gt_masks.append(gt_binary)
        all_pred_counts.append(pred_count)
        all_gt_counts.append(gt_count)
        all_pred_areas.append(pred_mean_area)
        all_gt_areas.append(gt_mean_area)

        per_image_results.append({
            "image": stem,
            "organ": organ,
            "dice": pix_metrics["dice"],
            "iou": pix_metrics["iou"],
            "precision": pix_metrics["precision"],
            "recall": pix_metrics["recall"],
            "pq": pq,
            "sq": sq,
            "dq": dq,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": rel_count_err,
            "gt_mean_area": gt_mean_area,
            "pred_mean_area": pred_mean_area,
            "area_error": rel_area_err,
            "circularity": pred_mean_circ,
            "eccentricity": pred_mean_ecc,
            "nuclei_density": density,
            "mean_nn_distance": mean_nn_dist,
            "correct_instances": err_summary.get("correct", 0),
            "merged_instances": err_summary.get("merged", 0),
            "split_instances": err_summary.get("split", 0),
            "missed_instances": err_summary.get("missed", 0),
            "fp_instances": err_summary.get("false_positive", 0),
        })

        # Save Visual Error Maps & Instance Overlays
        if save_plots:
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_error_map(image, pred_instances, gt_instances, ax=ax)
            ax.set_title(f"Error Map: {stem}\nGreen=TP, Red=FP, Blue=FN (Missed)")
            fig.savefig(results_fig_dir / f"error_map_{stem}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 8))
            plot_instance_overlay(image, pred_instances, ax=ax, title=f"Instance Cleaving: {stem}")
            fig.savefig(results_fig_dir / f"instance_overlay_{stem}.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

    # Convert to DataFrame
    df = pd.DataFrame(per_image_results)
    csv_path = results_tbl_dir / output_csv
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed evaluation results saved to: {csv_path}")

    # Visualizations: Feature distributions & Organ comparisons
    if save_plots:
        valid_props = [p for p in all_pred_props if len(p) > 0]
        if valid_props:
            cohort_props = pd.concat(valid_props, ignore_index=True)
            if len(cohort_props) > 0:
                plot_feature_distribution(
                    cohort_props["area"].values,
                    feature_name="Nuclear Area (px²)",
                    save_path=str(results_fig_dir / "morphology_area_distribution.png"),
                )
                plot_feature_distribution(
                    cohort_props["circularity"].values,
                    feature_name="Nuclear Circularity",
                    save_path=str(results_fig_dir / "morphology_circularity_distribution.png"),
                )

        # Organ comparison plots
        organ_dices = {org: df[df["organ"] == org]["dice"].tolist() for org in sorted(df["organ"].unique())}
        organ_densities = {org: df[df["organ"] == org]["nuclei_density"].tolist() for org in sorted(df["organ"].unique())}
        if len(organ_dices) > 1:
            plot_organ_comparison(
                organ_dices,
                feature="Dice Score",
                save_path=str(results_fig_dir / "organ_dice_comparison.png"),
            )
            plot_organ_comparison(
                organ_densities,
                feature="Nuclei Density (nuclei/px)",
                save_path=str(results_fig_dir / "organ_density_comparison.png"),
            )

    # Aggregate measurement agreements (Pearson r)
    agree_summary = aggregate_agreements(all_agreement_records)
    corr_count = agree_summary["pearson_count"]
    corr_area = agree_summary["pearson_area"]

    # Bootstrap 95% Confidence Intervals via report_metrics
    rep = report_metrics(all_pred_masks, all_gt_masks)
    dice_mean, dice_low, dice_high = rep["dice_mean"], rep["dice_lower"], rep["dice_upper"]
    iou_mean, iou_low, iou_high = rep["iou_mean"], rep["iou_lower"], rep["iou_upper"]
    prec_mean, prec_low, prec_high = rep["precision_mean"], rep["precision_lower"], rep["precision_upper"]
    rec_mean, rec_low, rec_high = rep["recall_mean"], rep["recall_lower"], rep["recall_upper"]

    print("\n" + "=" * 65)
    print("           QUANTITATIVE HISTOPATHOLOGY EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Dice Score:          {dice_mean:.4f}  [95% CI: {dice_low:.4f} - {dice_high:.4f}]")
    print(f"IoU (Jaccard):       {iou_mean:.4f}  [95% CI: {iou_low:.4f} - {iou_high:.4f}]")
    print(f"Precision:           {prec_mean:.4f}  [95% CI: {prec_low:.4f} - {prec_high:.4f}]")
    print(f"Recall:              {rec_mean:.4f}  [95% CI: {rec_low:.4f} - {rec_high:.4f}]")
    print(f"Panoptic Quality:    {df['pq'].mean():.4f} (SQ: {df['sq'].mean():.4f}, DQ: {df['dq'].mean():.4f})")
    print(f"Mean Count Error:    {df['count_error'].mean():.2%}")
    print(f"Mean Area Error:     {df['area_error'].mean():.2%}")
    print(f"Count Pearson r:     {corr_count:.4f} (Predicted vs. GT Nuclei Count)")
    print(f"Area Pearson r:      {corr_area:.4f} (Predicted vs. GT Nuclear Area)")
    print(f"Mean Nuclei Density: {df['nuclei_density'].mean():.6f} nuclei/px")
    print("-" * 65)
    print("Instance Error Modes (Cohort Totals):")
    print(f"  Correct:           {int(df['correct_instances'].sum()):,}")
    print(f"  Merged Clumps:     {int(df['merged_instances'].sum()):,}")
    print(f"  Split Overcuts:    {int(df['split_instances'].sum()):,}")
    print(f"  Missed (FN):       {int(df['missed_instances'].sum()):,}")
    print(f"  False Positives:   {int(df['fp_instances'].sum()):,}")
    if "organ" in df.columns and len(df["organ"].unique()) > 1:
        print("-" * 65)
        print("Cohort Organ-Stratified Performance:")
        for org in sorted(df["organ"].unique()):
            sub = df[df["organ"] == org]
            print(f"  {org:12s}: Dice = {sub['dice'].mean():.4f}, PQ = {sub['pq'].mean():.4f} (n={len(sub)})")
    print("=" * 65)

    return {
        "dice": dice_mean,
        "dice_ci": f"{dice_low:.4f} - {dice_high:.4f}",
        "iou": iou_mean,
        "iou_ci": f"{iou_low:.4f} - {iou_high:.4f}",
        "precision": prec_mean,
        "precision_ci": f"{prec_low:.4f} - {prec_high:.4f}",
        "recall": rec_mean,
        "recall_ci": f"{rec_low:.4f} - {rec_high:.4f}",
        "pq": float(df["pq"].mean()),
        "sq": float(df["sq"].mean()),
        "dq": float(df["dq"].mean()),
        "count_error": float(df["count_error"].mean()),
        "area_error": float(df["area_error"].mean()),
        "pearson_count": float(corr_count),
        "pearson_area": float(corr_area),
        "mean_density": float(df["nuclei_density"].mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Nuclei Segmentation & Quantification")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", default=None, help="Path to checkpoint .pth")
    parser.add_argument("--model-type", default=None, help="Override model type (unet | pretrained_unet)")
    parser.add_argument("--output-csv", default="evaluation_results.csv", help="Filename for evaluation results CSV")
    parser.add_argument("--threshold", type=float, default=0.40, help="Probability binarization threshold (default: 0.40 calibrated Pareto peak)")
    parser.add_argument("--min-distance", type=int, default=None, help="Watershed peak minimum distance (default from config)")
    parser.add_argument("--no-tta", action="store_true", help="Disable 8-fold test-time augmentation")
    parser.add_argument("--no-hann", action="store_true", help="Disable 2D Hann window tile stitching")
    parser.add_argument("--min-area", type=int, default=20, help="Minimum nucleus area filter in pixels")
    parser.add_argument("--no-tissue-mask", action="store_true", help="Disable tissue QC mask gating")
    parser.add_argument("--no-optical-guidance", action="store_true", help="Disable optical stain intensity watershed guidance")
    parser.add_argument("--max-eccentricity", type=float, default=0.98, help="Max eccentricity filter for instances")
    parser.add_argument("--stain-norm", type=str, default="macenko", choices=["macenko", "none"], help="Stain normalization method (default: macenko)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    args = parser.parse_args()
    evaluate_pipeline(
        args.config,
        weights_path=args.weights,
        model_type=args.model_type,
        output_csv=args.output_csv,
        save_plots=not args.no_plots,
        threshold=args.threshold,
        min_dist=args.min_distance,
        use_tta=not args.no_tta,
        use_hann=not args.no_hann,
        min_area=args.min_area,
        use_tissue_mask=not args.no_tissue_mask,
        use_optical_guidance=not args.no_optical_guidance,
        max_eccentricity=args.max_eccentricity,
        stain_norm=args.stain_norm,
    )
