"""
setup_monuseg_data.py
---------------------
Helper script to unpack and structure the full MoNuSeg dataset.

Expected inputs:
- Downloaded zip from Kaggle: `kaggle datasets download -d nebulae/monuseg`
  or Kumar et al. repository / Grand Challenge.
- Or an existing folder containing MoNuSeg .tif images and .xml annotations.

Usage:
    python scripts/setup_monuseg_data.py --archive /path/to/monuseg.zip
    or
    python scripts/setup_monuseg_data.py --source /path/to/extracted/folder
"""

import os
import sys
import shutil
import zipfile
import tarfile
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from prepare_dataset import prepare_dataset


def extract_archive(archive_path: Path, target_dir: Path):
    print(f"Extracting {archive_path.name}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif archive_path.suffix.lower() in (".tar", ".gz", ".tgz"):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")


def organize_monuseg(source_dir: Path, dest_root: Path):
    images_dest = dest_root / "images"
    annotations_dest = dest_root / "annotations"
    images_dest.mkdir(parents=True, exist_ok=True)
    annotations_dest.mkdir(parents=True, exist_ok=True)

    # Search for all image files (.tif, .tiff, .png, .jpg)
    img_extensions = {".tif", ".tiff", ".png", ".jpg"}
    found_images = [
        p for p in source_dir.rglob("*")
        if p.suffix.lower() in img_extensions and not p.name.startswith(".") and "mask" not in p.name.lower()
    ]

    # Search for all xml files
    found_xmls = {
        p.stem: p for p in source_dir.rglob("*.xml")
        if not p.name.startswith(".")
    }

    print(f"Discovered {len(found_images)} images and {len(found_xmls)} XML annotations in source.")

    paired_count = 0
    for img_path in found_images:
        stem = img_path.stem
        # Copy image
        target_img = images_dest / img_path.name
        if not target_img.exists() or target_img.resolve() != img_path.resolve():
            shutil.copy2(img_path, target_img)

        # Check for matching XML
        if stem in found_xmls:
            xml_path = found_xmls[stem]
            target_xml = annotations_dest / f"{stem}.xml"
            if not target_xml.exists() or target_xml.resolve() != xml_path.resolve():
                shutil.copy2(xml_path, target_xml)
            paired_count += 1
        else:
            print(f"Warning: No XML match for image {img_path.name}")

    print(f"Successfully organized {paired_count} paired slides into {dest_root.resolve()}")
    return paired_count


def main():
    parser = argparse.ArgumentParser(description="Organize and structure MoNuSeg dataset")
    parser.add_argument("--archive", type=str, default=None, help="Path to downloaded zip/tar archive")
    parser.add_argument("--source", type=str, default=None, help="Path to directory containing MoNuSeg files")
    parser.add_argument("--dest", type=str, default="data/MoNuSeg", help="Target dataset root")
    parser.add_argument("--prepare", action="store_true", default=True, help="Automatically run zero-leakage tiling")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive_path = Path(args.archive)
        temp_extract = dest_root / "_temp_raw"
        extract_archive(archive_path, temp_extract)
        organize_monuseg(temp_extract, dest_root)
        shutil.rmtree(temp_extract, ignore_errors=True)
    elif args.source:
        organize_monuseg(Path(args.source), dest_root)
    else:
        print("No --archive or --source provided. Checking if data/MoNuSeg already has images...")
        n_imgs = len(list((dest_root / "images").glob("*"))) if (dest_root / "images").exists() else 0
        print(f"Found {n_imgs} images in {dest_root / 'images'}.")

    if args.prepare:
        print("\nTriggering zero-leakage image-level splitting and tiling...")
        prepare_dataset(args.config)


if __name__ == "__main__":
    main()
