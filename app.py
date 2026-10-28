"""
app.py — Gradio Demo
--------------------
Upload an H&E image → preprocessing → U-Net → watershed → feature extraction → display.

Usage:
    python app.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import yaml
import torch
import gradio as gr
from PIL import Image
from pathlib import Path

from preprocessing import (
    load_image,
    create_tissue_mask,
    macenko_normalize,
    fit_macenko_reference,
    tile_image,
    stitch_predictions,
)
from evaluate import stitch_hann_predictions, postprocess_binary_mask
from models import build_model
from instances import binary_to_instances
from features import extract_morphological_features, extract_spatial_features
from visualization import plot_overlay, plot_instance_overlay, plot_feature_distribution

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


# ---------------------------------------------------------------------------
# Load config and model
# ---------------------------------------------------------------------------

CONFIG_PATH = "configs/config.yaml"
TILE_SIZE   = 256
STRIDE      = 128
MIN_DIST    = 5
THRESHOLD   = 0.15

# Automatically select the highest tier available checkpoint
CANDIDATE_CKPTS = [
    "results/checkpoints/m4_combined_best.pth",
    "results/checkpoints/m3_aug_best.pth",
    "results/checkpoints/unet_best.pth",
]
CHECKPOINT = CANDIDATE_CKPTS[-1]
for cand in CANDIDATE_CKPTS:
    if Path(cand).exists():
        CHECKPOINT = cand
        break

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_fraction = float(cfg.get("hardware", {}).get("gpu_memory_fraction", 0.90))
    torch.cuda.set_per_process_memory_fraction(gpu_fraction, 0)
    torch.cuda.empty_cache()

model  = build_model(cfg["model"]).to(device)

ckpt_path = Path(CHECKPOINT)
if ckpt_path.exists():
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(f"WARNING: No checkpoint at {ckpt_path}. Using random weights.")

model.eval()


# ---------------------------------------------------------------------------
# Helper: run full inference pipeline on a numpy RGB image
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_pipeline(image_np: np.ndarray):
    """
    Full pipeline:
      image → tissue mask → Macenko norm → tile → U-Net → stitch → watershed → features

    Returns:
        pred_binary   : (H, W) binary prediction
        instance_mask : (H, W) instance labels
        features_df   : DataFrame of morphological features
        spatial_feats : dict of spatial features
    """
    H, W = image_np.shape[:2]

    # 1. Tissue mask
    tissue_mask = create_tissue_mask(image_np)

    # 2. Macenko normalization (fit on whole image)
    try:
        ref_maxC, ref_HE = fit_macenko_reference(image_np)
        image_norm = macenko_normalize(image_np, target_concentrations=(ref_maxC, ref_HE))
    except Exception:
        image_norm = image_np   # fallback: no normalization

    # 3. Tile
    tiles = tile_image(image_norm, tile_size=TILE_SIZE, stride=STRIDE)

    # 4. Inference on each tile
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    for t in tiles:
        tile_rgb = t["tile"]
        tensor = to_tensor(Image.fromarray(tile_rgb)).unsqueeze(0).to(device)
        logit  = model(tensor)
        prob   = torch.sigmoid(logit).squeeze().cpu().numpy()
        t["pred"] = prob

    # 5. Seamless Hann-window prediction stitching
    pred_tiles_info = [
        {"tile": t["pred"], "h_start": t["h_start"], "w_start": t["w_start"]}
        for t in tiles
    ]
    prob_map = stitch_hann_predictions(pred_tiles_info, output_shape=(H, W), tile_size=TILE_SIZE)
    raw_binary = (prob_map > THRESHOLD).astype(np.uint8) * tissue_mask
    pred_binary = postprocess_binary_mask(raw_binary, min_area=20)

    # 6. Optically-guided instance separation & morphological cleaving
    instance_mask = binary_to_instances(
        pred_binary,
        min_distance=MIN_DIST,
        intensity_image=image_np,
        min_area=20,
        max_eccentricity=0.98,
    )

    # 7. Features
    features_df  = extract_morphological_features(instance_mask, image_np)
    spatial_feats = extract_spatial_features(instance_mask, tissue_mask)

    return pred_binary, instance_mask, features_df, spatial_feats


# ---------------------------------------------------------------------------
# Gradio helper: render figure to PIL Image
# ---------------------------------------------------------------------------

def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Main inference function called by Gradio
# ---------------------------------------------------------------------------

def analyze(input_image):
    if input_image is None:
        return None, None, None, "Please upload an image."

    # Convert PIL → numpy
    image_np = np.array(input_image.convert("RGB"))

    pred_binary, instance_mask, feat_df, spatial = run_pipeline(image_np)

    # Panel 1 — segmentation overlay
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plot_overlay(image_np, pred_binary, ax=ax1, title="Segmentation Overlay")
    img_seg = _fig_to_pil(fig1)

    # Panel 2 — instance overlay
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    plot_instance_overlay(image_np, instance_mask, ax=ax2, title="Instance Cleaving Overlay")
    img_inst = _fig_to_pil(fig2)

    # Panel 3 — morphological distribution
    if not feat_df.empty and len(feat_df) > 0:
        fig3 = plot_feature_distribution(feat_df["area"].values, feature_name="Nuclear Area (px²)")
        img_dist = _fig_to_pil(fig3)
    else:
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.text(0.5, 0.5, "No nuclei detected", ha="center")
        img_dist = _fig_to_pil(fig3)

    # Summary text
    n          = spatial["nuclei_count"]
    density    = spatial["nuclei_density"]
    mean_area  = feat_df["area"].mean()    if not feat_df.empty else 0.0
    mean_ecc   = feat_df["eccentricity"].mean() if not feat_df.empty else 0.0
    mean_circ  = feat_df["circularity"].mean() if not feat_df.empty else 0.0

    summary = (
        f"Nuclei count:       {n}\n"
        f"Nuclei density:     {density:.6f}  (nuclei/tissue px)\n"
        f"Mean nuclear area:  {mean_area:.1f} px²\n"
        f"Mean circularity:   {mean_circ:.3f}\n"
        f"Mean eccentricity:  {mean_ecc:.3f}"
    )

    return img_seg, img_inst, img_dist, summary


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Histopathology Nuclei Segmentation") as demo:
    gr.Markdown(
        """
        # Quantitative Histopathology Analysis
        **Robust Nuclei Segmentation & Morphological Quantification**

        Upload an H&E-stained histopathology image to segment nuclei and extract measurements.
        """
    )

    with gr.Row():
        input_img = gr.Image(type="pil", label="H&E Image (upload .tif / .png / .jpg)")

    with gr.Row():
        btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        out_seg  = gr.Image(label="Segmentation Overlay")
        out_inst = gr.Image(label="Instance Cleaving Overlay")
        out_dist = gr.Image(label="Nuclear Area Distribution")

    summary_box = gr.Textbox(label="Quantitative Summary", lines=5)

    btn.click(
        fn=analyze,
        inputs=[input_img],
        outputs=[out_seg, out_inst, out_dist, summary_box],
    )

    gr.Markdown(
        """
        ---
        **Pipeline:** Tissue QC Gating → Macenko Normalization → Hann Overlap Stitching → Optical Gradient Watershed → Morphological Filtering
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
