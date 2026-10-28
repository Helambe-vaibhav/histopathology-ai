"""
prepare_dataset.py
------------------
Prepares MoNuSeg dataset for training:
1. Converts XML annotations to binary & instance masks
2. Splits dataset at the IMAGE level (prevents data leakage across tiles)
3. Fits Macenko reference slide
4. Extracts 256x256 tiles with 50% overlap (stride=128)
5. Saves train/val/test tile splits

Usage:
    python src/prepare_dataset.py --config configs/config.yaml
"""

import os
import argparse
import yaml
import random
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from preprocessing import (
    load_image,
    xml_to_masks,
    create_tissue_mask,
    macenko_normalize,
    fit_macenko_reference,
    tile_image
)


def prepare_dataset(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["root"])
    images_dir = data_root / "images"
    annotations_dir = data_root / "annotations"
    masks_dir = data_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all image files (.png, .tif, .tiff)
    image_files = sorted(
        [p for p in images_dir.glob("*") if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg")]
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"Found {len(image_files)} slides in {images_dir}")

    # 2. Image-level train / val / test split (CRITICAL: Never tile before splitting)
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    indices = list(range(len(image_files)))
    random.shuffle(indices)

    train_ratio = cfg["data"].get("train_ratio", 0.70)
    val_ratio = cfg["data"].get("val_ratio", 0.15)
    
    n_train = max(1, int(len(image_files) * train_ratio))
    n_val = max(1, int(len(image_files) * val_ratio)) if len(image_files) > 2 else 1
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if not test_idx:
        test_idx = val_idx  # fallback for tiny smoke tests

    splits = {
        "train": [image_files[i] for i in train_idx],
        "val":   [image_files[i] for i in val_idx],
        "test":  [image_files[i] for i in test_idx],
    }

    print("\nImage-Level Split Summary (Zero Leakage):")
    for split_name, files in splits.items():
        print(f"  {split_name.capitalize():5s}: {len(files)} slides -> {[f.stem for f in files]}")

    # 3. Setup output tile directories
    tile_size = cfg["data"].get("tile_size", 256)
    stride = cfg["data"].get("stride", 128)

    tiles_root = data_root / "tiles"
    for split_name in ("train", "val", "test"):
        (tiles_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (tiles_root / split_name / "masks").mkdir(parents=True, exist_ok=True)

    # 4. Optional Macenko reference fit on a train image
    ref_image_path = splits["train"][0]
    ref_img = load_image(str(ref_image_path))
    ref_tissue = create_tissue_mask(ref_img)
    macenko_ref = fit_macenko_reference(ref_img)
    print(f"\nMacenko reference fitted on training slide: {ref_image_path.name}")

    # 5. Process and tile each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} slides...")
        img_out_dir = tiles_root / split_name / "images"
        mask_out_dir = tiles_root / split_name / "masks"

        tile_counter = 0
        for img_path in tqdm(files, desc=f"  {split_name}"):
            stem = img_path.stem
            img = load_image(str(img_path))
            H, W, _ = img.shape

            # Find matching annotation XML or pre-existing mask
            xml_path = annotations_dir / f"{stem}.xml"
            pre_mask_png = masks_dir / f"{stem}_mask.png"

            if xml_path.exists():
                binary_mask, instance_mask = xml_to_masks(str(xml_path), (H, W))
                # Save whole image GT masks for later whole-slide evaluation
                Image.fromarray((binary_mask * 255).astype(np.uint8)).save(masks_dir / f"{stem}_mask.png")
                np.save(str(masks_dir / f"{stem}_instances.npy"), instance_mask)
            elif pre_mask_png.exists():
                binary_mask = (np.array(Image.open(pre_mask_png)) > 127).astype(np.uint8)
            else:
                print(f"Warning: No annotation or mask found for {stem}, skipping.")
                continue

            # Macenko normalize using reference statistics
            norm_img = macenko_normalize(img, target_concentrations=macenko_ref)

            # Tile image and mask
            img_tiles_info = tile_image(norm_img, tile_size=tile_size, stride=stride)
            mask_tiles_info = tile_image(binary_mask, tile_size=tile_size, stride=stride)

            for idx, (it_info, mt_info) in enumerate(zip(img_tiles_info, mask_tiles_info)):
                itile = it_info["tile"]
                mtile = mt_info["tile"]
                r = it_info["h_start"]
                c = it_info["w_start"]
                tile_filename = f"{stem}_t{idx:04d}_r{r}_c{c}.png"
                Image.fromarray(itile).save(img_out_dir / tile_filename)
                Image.fromarray((mtile * 255).astype(np.uint8)).save(mask_out_dir / tile_filename)
                tile_counter += 1

        print(f"  Generated {tile_counter} tiles for {split_name}.")

    print("\nDataset preparation complete!")
    print(f"Tiles stored in: {tiles_root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MoNuSeg dataset")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    prepare_dataset(args.config)
