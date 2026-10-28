"""
evaluate_all_fixes.py
---------------------
Systematically implements and benchmarks all fixes one by one on RAW images (as M3/M4 expect):
Step 0: Baseline M3 (Raw, T=0.50, min_dist=10)
Step 1: Fix 1 - Calibrated Threshold (T=0.15, min_dist=10)
Step 2: Fix 2 - Adaptive Watershed Peak Separation (T=0.15, min_dist=5)
Step 3: Fix 3 - Fine-grained Watershed for Small Nuclei (T=0.15, min_dist=3)
Step 4: Fix 4 - M4 Combined Loss Model with Calibrated Threshold & Watershed
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from models import build_model
from preprocessing import load_image, tile_image, stitch_predictions
from metrics import compute_pixel_metrics, panoptic_quality
from instances import binary_to_instances

def run_eval_config(
    model,
    device,
    test_stems,
    images_dir,
    masks_dir,
    tile_size=256,
    stride=128,
    threshold=0.5,
    min_dist=10,
):
    to_tensor = transforms.ToTensor()
    results = []

    for stem in test_stems:
        img_candidates = list(images_dir.glob(f"{stem}.*"))
        mask_candidate = masks_dir / f"{stem}_mask.png"
        inst_candidate = masks_dir / f"{stem}_instances.npy"

        if not img_candidates or not mask_candidate.exists():
            continue

        raw_image = load_image(str(img_candidates[0]))
        H, W, _ = raw_image.shape
        gt_binary = (np.array(Image.open(mask_candidate)) > 127).astype(np.uint8)

        if inst_candidate.exists():
            gt_inst = np.load(str(inst_candidate)).astype(np.int32)
        else:
            gt_inst = binary_to_instances(gt_binary)

        tiles_info = tile_image(raw_image, tile_size=tile_size, stride=stride)
        pred_tiles_info = []

        with torch.no_grad():
            for t_info in tiles_info:
                tile = t_info["tile"]
                t_tensor = to_tensor(Image.fromarray(tile)).unsqueeze(0).to(device)
                logits = model(t_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_tiles_info.append({
                    "tile": prob,
                    "h_start": t_info["h_start"],
                    "w_start": t_info["w_start"],
                })

        stitched_prob = stitch_predictions(pred_tiles_info, output_shape=(H, W))
        pred_bin = (stitched_prob > threshold).astype(np.uint8)

        pix = compute_pixel_metrics(pred_bin, gt_binary)
        pred_inst = binary_to_instances(pred_bin, min_distance=min_dist)
        pq_res = panoptic_quality(pred_inst, gt_inst, iou_threshold=0.5)

        gt_count = int(gt_inst.max())
        pred_count = int(pred_inst.max())
        count_err = abs(pred_count - gt_count) / max(1, gt_count)

        results.append({
            "stem": stem,
            "dice": pix["dice"],
            "iou": pix["iou"],
            "precision": pix["precision"],
            "recall": pix["recall"],
            "pq": pq_res["pq"],
            "sq": pq_res["sq"],
            "dq": pq_res["dq"],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "count_error": count_err,
        })

    df = pd.DataFrame(results)
    return {
        "mean_dice": float(df["dice"].mean()),
        "mean_iou": float(df["iou"].mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_pq": float(df["pq"].mean()),
        "mean_count_error": float(df["count_error"].mean()),
        "details_df": df,
    }

def main():
    config_path = "configs/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = Path(cfg["data"]["root"])
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    test_tiles_dir = data_root / "tiles" / "test" / "images"

    if test_tiles_dir.exists():
        test_stems = sorted(list(set(p.name.split("_t")[0] for p in test_tiles_dir.glob("*.png"))))
    else:
        test_stems = [p.stem for p in sorted(images_dir.glob("*")) if p.suffix.lower() in (".png", ".tif", ".jpg")]

    # Build M3 model
    m3_model = build_model(cfg["model"]).to(device)
    m3_ckpt = "results/checkpoints/m3_aug_best.pth"
    m3_model.load_state_dict(torch.load(m3_ckpt, map_location=device))
    m3_model.eval()

    # Build M4 model
    m4_model = build_model(cfg["model"]).to(device)
    m4_ckpt = "results/checkpoints/m4_combined_best.pth"
    m4_model.load_state_dict(torch.load(m4_ckpt, map_location=device))
    m4_model.eval()

    benchmarks = []

    print("\n" + "=" * 80)
    print("      BENCHMARKING FIXES STEP-BY-STEP ON MONUSEG TEST COHORT")
    print("=" * 80)

    # 0. Baseline M3
    print("\n[Step 0] Baseline M3: Raw Input, Threshold = 0.50, min_distance = 10")
    b0 = run_eval_config(m3_model, device, test_stems, images_dir, masks_dir, threshold=0.50, min_dist=10)
    benchmarks.append({
        "Configuration": "Step 0: Baseline M3 (T=0.50, dist=10)",
        "Dice": b0["mean_dice"],
        "IoU": b0["mean_iou"],
        "Precision": b0["mean_precision"],
        "Recall": b0["mean_recall"],
        "PQ": b0["mean_pq"],
        "Count Error": b0["mean_count_error"],
    })
    print(f"-> Dice: {b0['mean_dice']:.4f} | IoU: {b0['mean_iou']:.4f} | Prec: {b0['mean_precision']:.4f} | Rec: {b0['mean_recall']:.4f} | PQ: {b0['mean_pq']:.4f} | CountErr: {b0['mean_count_error']:.1%}")

    # 1. Fix 1: Calibrated Threshold
    print("\n[Step 1] Fix 1: Calibrated Threshold (T = 0.15, min_distance = 10)")
    b1 = run_eval_config(m3_model, device, test_stems, images_dir, masks_dir, threshold=0.15, min_dist=10)
    benchmarks.append({
        "Configuration": "Step 1: Fix 1 (Calibrated T=0.15, dist=10)",
        "Dice": b1["mean_dice"],
        "IoU": b1["mean_iou"],
        "Precision": b1["mean_precision"],
        "Recall": b1["mean_recall"],
        "PQ": b1["mean_pq"],
        "Count Error": b1["mean_count_error"],
    })
    print(f"-> Dice: {b1['mean_dice']:.4f} | IoU: {b1['mean_iou']:.4f} | Prec: {b1['mean_precision']:.4f} | Rec: {b1['mean_recall']:.4f} | PQ: {b1['mean_pq']:.4f} | CountErr: {b1['mean_count_error']:.1%}")

    # 2. Fix 2: Adaptive Watershed min_dist = 5
    print("\n[Step 2] Fix 2: Adaptive Watershed (T = 0.15, min_distance = 5)")
    b2 = run_eval_config(m3_model, device, test_stems, images_dir, masks_dir, threshold=0.15, min_dist=5)
    benchmarks.append({
        "Configuration": "Step 2: Fix 1+2 (+ Adaptive Watershed dist=5)",
        "Dice": b2["mean_dice"],
        "IoU": b2["mean_iou"],
        "Precision": b2["mean_precision"],
        "Recall": b2["mean_recall"],
        "PQ": b2["mean_pq"],
        "Count Error": b2["mean_count_error"],
    })
    print(f"-> Dice: {b2['mean_dice']:.4f} | IoU: {b2['mean_iou']:.4f} | Prec: {b2['mean_precision']:.4f} | Rec: {b2['mean_recall']:.4f} | PQ: {b2['mean_pq']:.4f} | CountErr: {b2['mean_count_error']:.1%}")

    # 3. Fix 3: Compact Watershed min_dist = 3
    print("\n[Step 3] Fix 3: Compact Watershed for Dense Slides (T = 0.15, min_distance = 3)")
    b3 = run_eval_config(m3_model, device, test_stems, images_dir, masks_dir, threshold=0.15, min_dist=3)
    benchmarks.append({
        "Configuration": "Step 3: Fix 1+3 (+ Compact Watershed dist=3)",
        "Dice": b3["mean_dice"],
        "IoU": b3["mean_iou"],
        "Precision": b3["mean_precision"],
        "Recall": b3["mean_recall"],
        "PQ": b3["mean_pq"],
        "Count Error": b3["mean_count_error"],
    })
    print(f"-> Dice: {b3['mean_dice']:.4f} | IoU: {b3['mean_iou']:.4f} | Prec: {b3['mean_precision']:.4f} | Rec: {b3['mean_recall']:.4f} | PQ: {b3['mean_pq']:.4f} | CountErr: {b3['mean_count_error']:.1%}")

    # 4. Fix 4: M4 Model with Calibrated Threshold
    print("\n[Step 4] Fix 4: Evaluate M4 Model (Combined Loss, T = 0.15, min_dist = 5)")
    b4 = run_eval_config(m4_model, device, test_stems, images_dir, masks_dir, threshold=0.15, min_dist=5)
    benchmarks.append({
        "Configuration": "Step 4: M4 Combined Loss (T=0.15, dist=5)",
        "Dice": b4["mean_dice"],
        "IoU": b4["mean_iou"],
        "Precision": b4["mean_precision"],
        "Recall": b4["mean_recall"],
        "PQ": b4["mean_pq"],
        "Count Error": b4["mean_count_error"],
    })
    print(f"-> Dice: {b4['mean_dice']:.4f} | IoU: {b4['mean_iou']:.4f} | Prec: {b4['mean_precision']:.4f} | Rec: {b4['mean_recall']:.4f} | PQ: {b4['mean_pq']:.4f} | CountErr: {b4['mean_count_error']:.1%}")

    # Save summary table
    summary_df = pd.DataFrame(benchmarks)
    out_csv = Path("results/tables/fixes_progression_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("                    FINAL FIXES BENCHMARK PROGRESSION")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {out_csv}")

    # Save best detailed slide-level CSV
    best_df = b2["details_df"]
    best_df_path = Path("results/tables/eval_m3_fixed_details.csv")
    best_df.to_csv(best_df_path, index=False)
    print(f"Saved optimized slide-level details to: {best_df_path}")

if __name__ == "__main__":
    main()
