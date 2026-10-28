"""
generate_synthetic_monuseg.py
------------------------------
Generates a small synthetic MoNuSeg-like dataset for quick smoke-testing,
continuous integration, and training validation without downloading gigabytes.

Usage:
    python scripts/generate_synthetic_monuseg.py --output data/MoNuSeg --num-samples 6
"""

import os
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path


def create_synthetic_slide(image_size=(512, 512), num_nuclei=35, seed=42):
    np.random.seed(seed)
    H, W = image_size
    
    # 1. Background tissue (eosin-rich pinkish-purple)
    img = np.full((H, W, 3), 245, dtype=np.uint8)  # glass slide
    
    # Tissue area (irregular ellipse)
    cv2.ellipse(img, (W // 2, H // 2), (int(W * 0.42), int(H * 0.38)), 15, 0, 360, (220, 150, 205), -1)
    
    # Add slight texture noise to tissue
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    polygons = []
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    instance_mask = np.zeros((H, W), dtype=np.int32)
    
    # 2. Add nuclei (hematoxylin-rich deep violet-blue)
    for inst_id in range(1, num_nuclei + 1):
        # Sample inside tissue region
        r = np.random.uniform(0, 0.32)
        theta = np.random.uniform(0, 2 * np.pi)
        cx = int(W // 2 + r * W * np.cos(theta))
        cy = int(H // 2 + r * H * np.sin(theta))
        
        rx = np.random.randint(9, 18)
        ry = np.random.randint(7, 15)
        angle = np.random.randint(0, 180)
        
        # Generate polygon vertices along ellipse
        n_vertices = np.random.randint(12, 20)
        t_vals = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
        rad_angle = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)
        
        pts = []
        for t in t_vals:
            # Ellipse coords before rotation
            jitter = np.random.uniform(0.85, 1.15)
            x_el = rx * np.cos(t) * jitter
            y_el = ry * np.sin(t) * jitter
            px = int(cx + x_el * cos_a - y_el * sin_a)
            py = int(cy + x_el * sin_a + y_el * cos_a)
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
            pts.append((px, py))
            
        poly_arr = np.array(pts, dtype=np.int32)
        
        # Draw on image
        nucleus_color = (
            int(np.random.randint(55, 80)),
            int(np.random.randint(20, 45)),
            int(np.random.randint(95, 130))
        )
        cv2.fillPoly(img, [poly_arr], nucleus_color)
        cv2.fillPoly(binary_mask, [poly_arr], 1)
        cv2.fillPoly(instance_mask, [poly_arr], inst_id)
        
        polygons.append(pts)
        
    return img, binary_mask, instance_mask, polygons


def export_monuseg_xml(polygons, output_path):
    """Saves polygon annotations in MoNuSeg XML format."""
    root = ET.Element("Annotations")
    for idx, poly in enumerate(polygons):
        ann = ET.SubElement(root, "Annotation", Id=str(idx + 1))
        regions = ET.SubElement(ann, "Regions")
        region = ET.SubElement(regions, "Region", Id=str(idx + 1))
        vertices = ET.SubElement(region, "Vertices")
        for x, y in poly:
            ET.SubElement(vertices, "Vertex", X=str(float(x)), Y=str(float(y)))
            
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MoNuSeg dataset")
    parser.add_argument("--output", type=str, default="data/MoNuSeg", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of images to generate")
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    images_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    masks_dir = out_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_samples} synthetic slides into: {out_dir}")
    
    for i in range(1, args.num_samples + 1):
        img_id = f"sample_{i:02d}"
        img, binary_mask, instance_mask, polygons = create_synthetic_slide(seed=100 + i)
        
        # Save image (tif & png)
        cv2.imwrite(str(images_dir / f"{img_id}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Save XML
        export_monuseg_xml(polygons, ann_dir / f"{img_id}.xml")
        # Save GT mask
        cv2.imwrite(str(masks_dir / f"{img_id}_mask.png"), binary_mask * 255)
        np.save(str(masks_dir / f"{img_id}_instances.npy"), instance_mask)
        
    print(f"Successfully generated {args.num_samples} samples.")
    print("Dataset layout:")
    print(f"  Images:      {images_dir}")
    print(f"  Annotations: {ann_dir}")
    print(f"  Masks:       {masks_dir}")


if __name__ == "__main__":
    main()
