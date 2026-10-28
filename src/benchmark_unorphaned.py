"""
benchmark_unorphaned.py
-----------------------
Runs a side-by-side evaluation across all 9 test slides comparing:
1. Baseline (Orphaned):
   - No tissue QC mask gating
   - Pure Euclidean distance transform (no optical stain guidance)
   - No morphological filtering (debris/fibers kept)
2. Un-orphaned Pipeline:
   - Tissue QC mask gating (glass debris eliminated)
   - Pure Hematoxylin optical gradient watershed guidance (cleaves touching cells)
   - Morphological biological filtering (min_area=20, max_eccentricity=0.98)
   - Outlier stain harmonization for pale slides (TCGA-HE-7130)

Outputs comparative metrics table:
- Pixel Metrics: Dice, IoU, Precision, Recall
- Instance Metrics: Panoptic Quality (PQ), Segmentation Quality (SQ), Detection Quality (DQ)
- Quantification: Nuclei Counts, Count Error, Merged Clumps, False Positives
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

# Set path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from models import build_model
from preprocessing import load_image, tile_image, create_tissue_mask
from evaluate import stitch_hann_predictions, predict_tile_tta, postprocess_binary_mask
from instances import binary_to_instances, classify_errors, error_summary
from metrics import compute_pixel_metrics, panoptic_quality


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")

    weights_path = BASE_DIR.parent / "results/checkpoints/m3_aug_best.pth"
    if not weights_path.exists():
        weights_path = BASE_DIR.parent / "results/checkpoints/unet_best.pth"
    print(f"Using model weights: {weights_path.name}")

    cfg_model = {"type": "unet", "encoder": "resnet34", "encoder_weights": None, "in_channels": 3, "classes": 1}
    model = build_model(cfg_model).to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()

    data_root = BASE_DIR.parent / "data/MoNuSeg"
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    test_tiles_dir = data_root / "tiles/test/images"

    if test_tiles_dir.exists():
        test_stems = sorted(list(set(p.name.split("_t")[0] for p in test_tiles_dir.glob("*.png"))))
    else:
        test_stems = [p.stem for p in sorted(images_dir.glob("*.tif"))]

    print(f"Found {len(test_stems)} test slides: {test_stems}\n")

    to_tensor = transforms.ToTensor()
    results_base = []
    results_unorph = []

    for stem in tqdm(test_stems, desc="Evaluating test cohort"):
        img_candidates = list(images_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        image = load_image(str(img_candidates[0]))
        H, W, _ = image.shape

        mask_candidate = masks_dir / f"{stem}_mask.png"
        inst_candidate = masks_dir / f"{stem}_instances.npy"
        if not mask_candidate.exists():
            continue

        gt_binary = (np.array(Image.open(mask_candidate)) > 127).astype(np.uint8)
        if inst_candidate.exists():
            gt_instances = np.load(str(inst_candidate)).astype(np.int32)
        else:
            gt_instances = binary_to_instances(gt_binary)

        gt_count = int(gt_instances.max())

        # Forward pass (TTA 8-fold + Hann window stitching)
        tiles_info = tile_image(image, tile_size=256, stride=128)
        pred_tiles_info = []
        with torch.no_grad():
            for t_info in tiles_info:
                p = predict_tile_tta(model, t_info["tile"], device, to_tensor)
                pred_tiles_info.append({"tile": p, "h_start": t_info["h_start"], "w_start": t_info["w_start"]})

        prob = stitch_hann_predictions(pred_tiles_info, (H, W))

        # --- 1. BASELINE (Orphaned) ---
        raw_base = (prob > 0.15).astype(np.uint8)
        bin_base = postprocess_binary_mask(raw_base, min_area=20)
        inst_base = binary_to_instances(bin_base, min_distance=5, intensity_image=None, min_area=0, max_eccentricity=1.0)

        pix_b = compute_pixel_metrics(bin_base, gt_binary)
        pq_b = panoptic_quality(inst_base, gt_instances)
        err_b = error_summary(classify_errors(inst_base, gt_instances))
        cnt_b = int(inst_base.max())

        results_base.append({
            "slide": stem,
            "dice": pix_b["dice"],
            "iou": pix_b["iou"],
            "precision": pix_b["precision"],
            "recall": pix_b["recall"],
            "pq": pq_b["pq"],
            "sq": pq_b["sq"],
            "dq": pq_b["dq"],
            "gt_count": gt_count,
            "pred_count": cnt_b,
            "count_error": abs(cnt_b - gt_count) / max(1, gt_count),
            "merged": err_b.get("merged", 0),
            "fp": err_b.get("false_positive", 0),
            "fn": err_b.get("missed", 0),
        })

        # --- 2. UN-ORPHANED ---
        tissue_mask = create_tissue_mask(image)
        raw_unorph = (prob > 0.15).astype(np.uint8) * tissue_mask
        bin_unorph = postprocess_binary_mask(raw_unorph, min_area=20)
        inst_unorph = binary_to_instances(
            bin_unorph,
            min_distance=5,
            intensity_image=image,
            min_area=20,
            max_eccentricity=0.98,
        )

        pix_u = compute_pixel_metrics(bin_unorph, gt_binary)
        pq_u = panoptic_quality(inst_unorph, gt_instances)
        err_u = error_summary(classify_errors(inst_unorph, gt_instances))
        cnt_u = int(inst_unorph.max())

        results_unorph.append({
            "slide": stem,
            "dice": pix_u["dice"],
            "iou": pix_u["iou"],
            "precision": pix_u["precision"],
            "recall": pix_u["recall"],
            "pq": pq_u["pq"],
            "sq": pq_u["sq"],
            "dq": pq_u["dq"],
            "gt_count": gt_count,
            "pred_count": cnt_u,
            "count_error": abs(cnt_u - gt_count) / max(1, gt_count),
            "merged": err_u.get("merged", 0),
            "fp": err_u.get("false_positive", 0),
            "fn": err_u.get("missed", 0),
        })

    df_b = pd.DataFrame(results_base)
    df_u = pd.DataFrame(results_unorph)

    out_tbl_dir = BASE_DIR.parent / "results/tables"
    out_tbl_dir.mkdir(parents=True, exist_ok=True)
    df_b.to_csv(out_tbl_dir / "eval_cohort_baseline.csv", index=False)
    df_u.to_csv(out_tbl_dir / "eval_cohort_unorphaned.csv", index=False)

    print("\n" + "=" * 78)
    print("      COHORT BENCHMARK COMPARISON: BASELINE (ORPHANED) vs UN-ORPHANED")
    print("=" * 78)
    print(f"{'Metric':<24} | {'Baseline (Orphaned)':<24} | {'Un-orphaned':<24}")
    print("-" * 78)
    print(f"{'Mean Dice Score':<24} | {df_b['dice'].mean():<24.4f} | {df_u['dice'].mean():<24.4f}")
    print(f"{'Mean IoU (Jaccard)':<24} | {df_b['iou'].mean():<24.4f} | {df_u['iou'].mean():<24.4f}")
    print(f"{'Mean Precision':<24} | {df_b['precision'].mean():<24.4f} | {df_u['precision'].mean():<24.4f}")
    print(f"{'Mean Recall':<24} | {df_b['recall'].mean():<24.4f} | {df_u['recall'].mean():<24.4f}")
    print(f"{'Panoptic Quality (PQ)':<24} | {df_b['pq'].mean():<24.4f} | {df_u['pq'].mean():<24.4f}")
    print(f"{'Segmentation Quality (SQ)':<24} | {df_b['sq'].mean():<24.4f} | {df_u['sq'].mean():<24.4f}")
    print(f"{'Detection Quality (DQ)':<24} | {df_b['dq'].mean():<24.4f} | {df_u['dq'].mean():<24.4f}")
    print(f"{'Mean Count Error':<24} | {df_b['count_error'].mean():<24.2%} | {df_u['count_error'].mean():<24.2%}")
    print(f"{'Total Merged Clumps':<24} | {df_b['merged'].sum():<24,d} | {df_u['merged'].sum():<24,d}")
    print(f"{'Total False Positives':<24} | {df_b['fp'].sum():<24,d} | {df_u['fp'].sum():<24,d}")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
