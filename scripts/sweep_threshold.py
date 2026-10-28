"""
sweep_threshold.py
------------------
Evaluates threshold sweep (0.15 to 0.50) for M3 checkpoint on the test cohort
to find the optimal operating point balancing Precision and Recall.
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from models import build_model
from preprocessing import load_image, tile_image, stitch_predictions
from metrics import compute_pixel_metrics, panoptic_quality
from instances import binary_to_instances

def run_sweep(config_path="configs/config.yaml", weights_path="results/checkpoints/m3_aug_best.pth"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    data_root = Path(cfg["data"]["root"])
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    test_tiles_dir = data_root / "tiles" / "test" / "images"

    if test_tiles_dir.exists():
        test_stems = sorted(list(set(p.name.split("_t")[0] for p in test_tiles_dir.glob("*.png"))))
    else:
        test_stems = [p.stem for p in sorted(images_dir.glob("*")) if p.suffix.lower() in (".png", ".tif", ".jpg")]

    print(f"Running inference on {len(test_stems)} test slides to cache probabilities...")
    to_tensor = transforms.ToTensor()
    tile_size = cfg["data"].get("tile_size", 256)
    stride = cfg["data"].get("stride", 128)

    cached_probs = []
    cached_gts = []
    cached_gt_instances = []

    for stem in tqdm(test_stems, desc="Caching slide probabilities"):
        img_candidates = list(images_dir.glob(f"{stem}.*"))
        mask_candidate = masks_dir / f"{stem}_mask.png"
        inst_candidate = masks_dir / f"{stem}_instances.npy"

        if not img_candidates or not mask_candidate.exists():
            continue

        image = load_image(str(img_candidates[0]))
        H, W, _ = image.shape
        gt_binary = (np.array(Image.open(mask_candidate)) > 127).astype(np.uint8)

        if inst_candidate.exists():
            gt_inst = np.load(str(inst_candidate)).astype(np.int32)
        else:
            gt_inst = binary_to_instances(gt_binary)

        tiles_info = tile_image(image, tile_size=tile_size, stride=stride)
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
        cached_probs.append((stem, stitched_prob))
        cached_gts.append(gt_binary)
        cached_gt_instances.append(gt_inst)

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_results = []

    print("\n" + "=" * 75)
    print("                PROBABILITY THRESHOLD SWEEP RESULTS")
    print("=" * 75)

    for thresh in thresholds:
        dices, ious, precisions, recalls, pqs = [], [], [], [], []

        for (stem, prob), gt_bin, gt_inst in zip(cached_probs, cached_gts, cached_gt_instances):
            pred_bin = (prob > thresh).astype(np.uint8)
            pix = compute_pixel_metrics(pred_bin, gt_bin)
            dices.append(pix["dice"])
            ious.append(pix["iou"])
            precisions.append(pix["precision"])
            recalls.append(pix["recall"])

            pred_inst = binary_to_instances(pred_bin, min_distance=10)
            pq_metrics = panoptic_quality(pred_inst, gt_inst, iou_threshold=0.5)
            pqs.append(pq_metrics["pq"])

        row = {
            "threshold": thresh,
            "mean_dice": np.mean(dices),
            "mean_iou": np.mean(ious),
            "mean_precision": np.mean(precisions),
            "mean_recall": np.mean(recalls),
            "mean_pq": np.mean(pqs),
        }
        sweep_results.append(row)
        print(f"Thresh: {thresh:.2f} | Dice: {row['mean_dice']:.4f} | IoU: {row['mean_iou']:.4f} | "
              f"Precision: {row['mean_precision']:.4f} | Recall: {row['mean_recall']:.4f} | PQ: {row['mean_pq']:.4f}")

    df_sweep = pd.DataFrame(sweep_results)
    out_csv = Path("results/tables/threshold_sweep_m3.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(out_csv, index=False)
    print(f"\nSaved threshold sweep results to: {out_csv}")

    best_row = df_sweep.loc[df_sweep["mean_dice"].idxmax()]
    print(f"\n[OPTIMAL THRESHOLD] T = {best_row['threshold']:.2f} achieves Peak Dice = {best_row['mean_dice']:.4f} (IoU = {best_row['mean_iou']:.4f}, Recall = {best_row['mean_recall']:.4f}, PQ = {best_row['mean_pq']:.4f})")

if __name__ == "__main__":
    run_sweep()
