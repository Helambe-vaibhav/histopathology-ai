"""
unpack_official_dataset.py
--------------------------
Extracts all clean images and XML annotations from:
- D:/Career/Compitations/ComputerVision/MoNuSeg 2018 Training Data.zip
- D:/Career/Compitations/ComputerVision/MoNuSegTestData.zip
into data/MoNuSeg/images and data/MoNuSeg/annotations.
"""

import os
import shutil
import zipfile
from pathlib import Path


def main():
    dest = Path("data/MoNuSeg")
    images_dir = dest / "images"
    ann_dir = dest / "annotations"
    masks_dir = dest / "masks"
    tiles_dir = dest / "tiles"

    # Clean existing directories
    for d in [images_dir, ann_dir, masks_dir, tiles_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    zips = [
        Path("D:/Career/Compitations/ComputerVision/MoNuSeg 2018 Training Data.zip"),
        Path("D:/Career/Compitations/ComputerVision/MoNuSegTestData.zip"),
    ]

    print("Unpacking official MoNuSeg datasets...")
    for zpath in zips:
        if not zpath.exists():
            print(f"Error: {zpath} not found!")
            continue
        print(f"Reading {zpath.name}...")
        with zipfile.ZipFile(zpath, "r") as z:
            for member in z.namelist():
                base = os.path.basename(member)
                if not base or base.startswith(".") or "__MACOSX" in member:
                    continue
                if base.lower().endswith((".tif", ".tiff")):
                    with z.open(member) as src_f, open(images_dir / base, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                elif base.lower().endswith(".xml"):
                    with z.open(member) as src_f, open(ann_dir / base, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)

    # Verify pairing
    img_files = sorted(list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff")))
    xml_files = sorted(list(ann_dir.glob("*.xml")))
    paired = [img for img in img_files if (ann_dir / f"{img.stem}.xml").exists()]

    print(f"\nUnpacking Complete:")
    print(f"  Total Images:      {len(img_files)}")
    print(f"  Total XML Files:   {len(xml_files)}")
    print(f"  Paired Slide Sets: {len(paired)}")


if __name__ == "__main__":
    main()
