"""
evaluate_improved.py
--------------------
High-performance whole-slide evaluation pipeline for histopathology nuclei segmentation.

Key Architectural Improvements:
1. 8-Fold Dihedral Test-Time Augmentation (D4 TTA: 4 rotations × 2 flips)
2. 2D Hann (Cosine-Bell) Window Stitching for edge-artifact-free tile blending
3. Calibrated thresholding (T = 0.15 - 0.25) to correct for training distribution shifts
4. High-resolution Distance Transform + Tuned Watershed (min_distance = 5, area filter = 20)
5. Comprehensive metrics: Dice, IoU, Precision, Recall, PQ, SQ, DQ, Count Error, Area Error
"""

import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

import torch
from torchvision import transforms

# Ensure local src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import build_model
from preprocessing import load_image, tile_image, create_tissue_mask
from metrics import compute_pixel_metrics, panoptic_quality
from instances import binary_to_instances


def make_hann_window(size: int = 256, min_weight: float = 0.05) -> np.ndarray:
    """Generates a 2D Hann window for smooth tile blending."""
    w1d = np.hanning(size)
    w2d = np.outer(w1d, w1d).astype(np.float32)
    return np.maximum(w2d, min_weight)


def stitch_hann_predictions(preds: list, output_shape: tuple, tile_size: int = 256) -> np.ndarray:
    """Stitches overlapping tile predictions using 2D Hann window weighting."""
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
    """
    Computes 8-fold D4 dihedral Test-Time Augmentation for a single tile.
    (4 rotations × 2 horizontal flips).
    """
    t = to_tensor(Image.fromarray(tile_img)).unsqueeze(0).to(device)

    # 4 rotations of original
    t0 = t
    t90 = torch.rot90(t, 1, [2, 3])
    t180 = torch.rot90(t, 2, [2, 3])
    t270 = torch.rot90(t, 3, [2, 3])

    # 4 rotations of horizontally flipped image
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


def postprocess_mask(binary_mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Fills interior holes and filters tiny noise specks."""
    # Fill holes in binary mask
    filled = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)

    # Remove small objects below min_area
    lbls, num = ndimage.label(filled)
    if num == 0:
        return filled

    counts = np.bincount(lbls.ravel())
    mask_sizes = counts >= min_area
    mask_sizes[0] = False  # background
    cleaned = mask_sizes[lbls].astype(np.uint8)
    return cleaned


def run_improved_evaluation(
    config_path: str,
    weights_path: str,
    output_csv: str = "results/tables/eval_improved_bench.csv",
    threshold: float = 0.15,
    min_dist: int = 5,
    use_tta: bool = True,
    use_hann: bool = True,
    min_area: int = 20,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    base_dir = Path(config_path).resolve().parent.parent
    data_root = Path(cfg["data"]["root"])
    if not data_root.is_absolute():
        data_root = (base_dir / data_root).resolve()
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"

    # Find test stems
    test_tiles_dir = data_root / "tiles" / "test" / "images"
    if test_tiles_dir.exists():
        test_stems = sorted(list(set(p.name.split("_t")[0] for p in test_tiles_dir.glob("*.png"))))
    else:
        test_stems = [p.stem for p in sorted(images_dir.glob("*.tif"))]

    print(f"Evaluating {len(test_stems)} test slides with TTA={use_tta}, Hann={use_hann}, Threshold={threshold}, min_dist={min_dist}...")

    to_tensor = transforms.ToTensor()
    results = []

    for stem in tqdm(test_stems, desc="Evaluating slides"):
        img_path = images_dir / f"{stem}.tif"
        if not img_path.exists():
            img_path = images_dir / f"{stem}.png"
        raw_img = load_image(str(img_path))
        H, W, _ = raw_img.shape

        gt_mask_path = masks_dir / f"{stem}_mask.png"
        gt_binary = (np.array(Image.open(gt_mask_path)) > 127).astype(np.uint8)

        inst_npy_path = masks_dir / f"{stem}_instances.npy"
        if inst_npy_path.exists():
            gt_inst = np.load(str(inst_npy_path)).astype(np.int32)
        else:
            gt_inst = binary_to_instances(gt_binary)

        tiles = tile_image(raw_img, 256, 128)
        preds = []

        with torch.no_grad():
            for t in tiles:
                tile_arr = t["tile"]
                if use_tta:
                    p = predict_tile_tta(model, tile_arr, device, to_tensor)
                else:
                    logits = model(to_tensor(Image.fromarray(tile_arr)).unsqueeze(0).to(device))
                    p = torch.sigmoid(logits).squeeze().cpu().numpy()
                preds.append({"tile": p, "h_start": t["h_start"], "w_start": t["w_start"]})

        if use_hann:
            prob_map = stitch_hann_predictions(preds, (H, W))
        else:
            from preprocessing import stitch_predictions
            prob_map = stitch_predictions(preds, (H, W))

        # Binary thresholding, Tissue QC Gating & post-processing
        tissue_mask = create_tissue_mask(raw_img)
        raw_binary = (prob_map > threshold).astype(np.uint8) * tissue_mask
        pred_binary = postprocess_mask(raw_binary, min_area=min_area)

        # Pixel metrics
        pix_m = compute_pixel_metrics(pred_binary, gt_binary)

        # Instance segmentation & PQ (guided by optical stain intensity & filtered by shape)
        pred_inst = binary_to_instances(
            pred_binary,
            min_distance=min_dist,
            intensity_image=raw_img,
            min_area=min_area,
            max_eccentricity=0.98,
        )
        pq_res = panoptic_quality(pred_inst, gt_inst)

        # Counting & Area
        gt_cnt = int(gt_inst.max())
        pred_cnt = int(pred_inst.max())
        cnt_err = abs(pred_cnt - gt_cnt) / max(1, gt_cnt)

        gt_areas = [int((gt_inst == i).sum()) for i in range(1, gt_cnt + 1)] if gt_cnt > 0 else [0]
        pred_areas = [int((pred_inst == i).sum()) for i in range(1, pred_cnt + 1)] if pred_cnt > 0 else [0]
        mean_gt_area = float(np.mean(gt_areas))
        mean_pred_area = float(np.mean(pred_areas))
        area_err = abs(mean_pred_area - mean_gt_area) / max(1.0, mean_gt_area)

        row = {
            "image": stem,
            "dice": pix_m["dice"],
            "iou": pix_m["iou"],
            "precision": pix_m["precision"],
            "recall": pix_m["recall"],
            "pq": pq_res["pq"],
            "sq": pq_res["sq"],
            "dq": pq_res["dq"],
            "gt_count": gt_cnt,
            "pred_count": pred_cnt,
            "count_error": cnt_err,
            "gt_mean_area": mean_gt_area,
            "pred_mean_area": mean_pred_area,
            "area_error": area_err,
        }
        results.append(row)

    df = pd.DataFrame(results)
    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)

    print("\n" + "=" * 65)
    print("IMPROVED COHORT EVALUATION RESULTS (N=9 Whole Slides):")
    print("=" * 65)
    print(f"Mean Dice:         {df['dice'].mean():.4f}")
    print(f"Mean IoU:          {df['iou'].mean():.4f}")
    print(f"Mean Precision:    {df['precision'].mean():.4f}")
    print(f"Mean Recall:       {df['recall'].mean():.4f}")
    print(f"Mean PQ:           {df['pq'].mean():.4f}")
    print(f"Mean SQ:           {df['sq'].mean():.4f}")
    print(f"Mean DQ:           {df['dq'].mean():.4f}")
    print(f"Mean Count Error:  {df['count_error'].mean():.1%}")
    print(f"Mean Area Error:   {df['area_error'].mean():.1%}")
    print("=" * 65)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", default="results/checkpoints/m3_aug_best.pth")
    parser.add_argument("--output_csv", default="results/tables/eval_improved_bench.csv")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--min_dist", type=int, default=5)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--no_hann", action="store_true")
    parser.add_argument("--min_area", type=int, default=20)
    args = parser.parse_args()

    run_improved_evaluation(
        config_path=args.config,
        weights_path=args.weights,
        output_csv=args.output_csv,
        threshold=args.threshold,
        min_dist=args.min_dist,
        use_tta=not args.no_tta,
        use_hann=not args.no_hann,
        min_area=args.min_area,
    )
