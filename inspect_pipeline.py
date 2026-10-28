"""
inspect_pipeline.py
-------------------
Command-line runner to inspect and reconstruct all intermediate pipeline images.

Usage:
    python inspect_pipeline.py --image data/MoNuSeg/images/TCGA-21-5784-01Z-00-DX1.tif
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import torch

# Ensure src is in python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from models import build_model
from preprocessing import load_image
from interpretability import run_full_pipeline_interpretability, save_master_interpretability_figure


def main():
    parser = argparse.ArgumentParser(description="End-to-End Computational Pathology Pipeline Visual Inspector")
    parser.add_argument("--image", default=None, help="Path to input H&E image (.tif / .png / .jpg)")
    parser.add_argument("--mask", default=None, help="Optional ground truth mask for error map")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--weights", default="results/checkpoints/unet_best.pth", help="Model weights path")
    parser.add_argument("--out-dir", default="results/figures/interpretability", help="Output directory for stage images")
    parser.add_argument("--threshold", type=float, default=0.15, help="Binarization threshold")
    parser.add_argument("--min-distance", type=int, default=5, help="Watershed min peak distance")
    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Select image
    if args.image:
        img_path = Path(args.image)
    else:
        # Default to first available image in MoNuSeg
        default_dir = ROOT / "data/MoNuSeg/images"
        candidates = list(default_dir.glob("*.tif")) + list(default_dir.glob("*.png"))
        if not candidates:
            print("ERROR: No images found. Please specify --image <path>")
            sys.exit(1)
        img_path = candidates[0]

    print(f"\n[1/4] Loading input biopsy slide: {img_path.name}")
    image = load_image(str(img_path))
    H, W = image.shape[:2]
    print(f"      Resolution: {W} x {H} px, Channels: {image.shape[2]}")

    # Load ground truth if exists
    gt_binary = None
    if args.mask:
        gt_binary = (np.array(Image.open(args.mask)) > 127).astype(np.uint8)
    else:
        stem = img_path.stem
        candidate_mask = ROOT / f"data/MoNuSeg/masks/{stem}_mask.png"
        if candidate_mask.exists():
            gt_binary = (np.array(Image.open(candidate_mask)) > 127).astype(np.uint8)
            print(f"      Matched ground truth mask: {candidate_mask.name}")

    # Build model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/4] Initializing U-Net model on device: {device}")
    model = build_model(cfg["model"]).to(device)
    
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path
    if weights_path.exists():
        model.load_state_dict(torch.load(str(weights_path), map_location=device))
        print(f"      Loaded model checkpoint: {weights_path.name}")
    else:
        print(f"      WARNING: Checkpoint {weights_path} not found. Running with initialized weights.")

    # Run full interpretability pipeline
    print(f"[3/4] Reconstructing intermediate images across all pipeline stages...")
    stages = run_full_pipeline_interpretability(
        image=image,
        model=model,
        device=device,
        gt_binary=gt_binary,
        tile_size=cfg["data"].get("tile_size", 256),
        stride=cfg["data"].get("stride", 128),
        threshold=args.threshold,
        min_dist=args.min_distance,
    )

    # Save output artifacts
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[4/4] Saving reconstructed diagnostic images to: {out_dir}")
    # 1. Save individual stage images
    for stage_key, stage_img in stages.items():
        if stage_key in ("meta", "labels_instances"):
            continue
        stage_filename = out_dir / f"{stage_key}_{img_path.stem}.png"
        Image.fromarray(stage_img).save(stage_filename)
        print(f"      Saved: {stage_filename.name}")

    # 2. Save master summary dashboard
    master_path = out_dir / f"master_interpretability_dashboard_{img_path.stem}.png"
    save_master_interpretability_figure(stages, save_path=str(master_path))

    meta = stages.get("meta", {})
    print("\n" + "=" * 65)
    print("      PIPELINE RECONSTRUCTION & INTERPRETABILITY SUMMARY")
    print("=" * 65)
    print(f"Slide Name:           {img_path.name}")
    print(f"Tissue Coverage:      {meta.get('tissue_coverage_pct', 0.0):.1f}%")
    print(f"Nuclei Identified:    {meta.get('nuclei_count', 0):,}")
    print(f"Mean Confidence:      {meta.get('mean_confidence', 0.0):.2%}")
    print(f"Master Dashboard:     {master_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
