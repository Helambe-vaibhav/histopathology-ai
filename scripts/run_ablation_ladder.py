"""
run_ablation_ladder.py
----------------------
Orchestrates the 5-stage ablation ladder:
  M1: Baseline (Scratch U-Net + BCE, No Aug)
  M2: + Macenko Normalization (Scratch U-Net + BCE, No Aug)
  M3: + Spatial & Color Augmentations (Scratch U-Net + BCE + Aug)
  M4: + Combined BCE & Dice Loss (Scratch U-Net + Combined + Aug) [Uses completed 100-epoch ckpt]
  M5: + ImageNet Pretrained Encoder (PretrainedUNet ResNet-34 + Combined + Aug)

Usage:
  python scripts/run_ablation_ladder.py --config configs/config.yaml [--stages M5 M3 M2 M1] [--epochs 35]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import pandas as pd
import yaml

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import train
from evaluate import evaluate_pipeline
from visualization import plot_ablation_table


STAGES = {
    "M1": {
        "name": "M1: Baseline (Scratch U-Net + BCE, No Aug)",
        "model_type": "unet",
        "loss": "bce",
        "augment": False,
        "ckpt_name": "m1_baseline_best.pth",
        "output_csv": "eval_m1_baseline.csv",
    },
    "M2": {
        "name": "M2: + Macenko Normalization (Scratch U-Net + BCE, No Aug)",
        "model_type": "unet",
        "loss": "bce",
        "augment": False,
        "ckpt_name": "m2_macenko_best.pth",
        "output_csv": "eval_m2_macenko.csv",
    },
    "M3": {
        "name": "M3: + Spatial & Color Augmentations (Scratch U-Net + BCE + Aug)",
        "model_type": "unet",
        "loss": "bce",
        "augment": True,
        "ckpt_name": "m3_aug_best.pth",
        "output_csv": "eval_m3_aug.csv",
    },
    "M4": {
        "name": "M4: + Combined BCE & Dice Loss (Scratch U-Net + Combined + Aug)",
        "model_type": "unet",
        "loss": "combined",
        "augment": True,
        "ckpt_name": "m4_combined_best.pth",
        "output_csv": "eval_m4_combined.csv",
    },
    "M5": {
        "name": "M5: + ImageNet Pretrained Encoder (ResNet-34)",
        "model_type": "pretrained_unet",
        "loss": "combined",
        "augment": True,
        "ckpt_name": "m5_pretrained_best.pth",
        "output_csv": "eval_m5_pretrained.csv",
    },
}


def run_ablation(config_path: str, stages_to_run: list, default_epochs: int = 35):
    # Ensure offline mode for Hugging Face Hub so cached weights load instantly
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = Path("results/tables")
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Check if M4 can use the existing converged 100-epoch checkpoint
    unet_best = ckpt_dir / "unet_best.pth"
    m4_ckpt = ckpt_dir / "m4_combined_best.pth"
    if unet_best.exists() and not m4_ckpt.exists():
        shutil.copy(unet_best, m4_ckpt)
        print(f"Copied converged 100-epoch checkpoint to {m4_ckpt}")

    ablation_summary = []

    for stage_key in stages_to_run:
        if stage_key not in STAGES:
            print(f"Unknown stage: {stage_key}, skipping.")
            continue

        stage_cfg = STAGES[stage_key]
        ckpt_path = ckpt_dir / stage_cfg["ckpt_name"]

        print("\n" + "=" * 70)
        print(f"  EXECUTING ABLATION STAGE: {stage_cfg['name']}")
        print("=" * 70)

        # 1. Train if checkpoint doesn't exist
        if not ckpt_path.exists():
            print(f"Training {stage_key} for {default_epochs} epochs...")
            train(
                cfg_path=config_path,
                epochs_override=default_epochs,
                model_type_override=stage_cfg["model_type"],
                loss_override=stage_cfg["loss"],
                augment_override=stage_cfg["augment"],
                ckpt_name_override=stage_cfg["ckpt_name"],
            )
        else:
            print(f"Checkpoint {ckpt_path} already exists. Proceeding directly to evaluation.")

        # 2. Evaluate
        print(f"\nEvaluating {stage_key} on test cohort...")
        metrics = evaluate_pipeline(
            config_path=config_path,
            weights_path=str(ckpt_path),
            model_type=stage_cfg["model_type"],
            output_csv=stage_cfg["output_csv"],
            save_plots=False,
        )

        ablation_summary.append({
            "stage": stage_key,
            "configuration": stage_cfg["name"],
            "dice": round(metrics["dice"], 4),
            "dice_ci": metrics["dice_ci"],
            "iou": round(metrics["iou"], 4),
            "iou_ci": metrics["iou_ci"],
            "pq": round(metrics["pq"], 4),
            "sq": round(metrics["sq"], 4),
            "dq": round(metrics["dq"], 4),
            "count_error": f"{metrics['count_error']:.2%}",
            "area_error": f"{metrics['area_error']:.2%}",
        })

    # Save summary table
    summary_df = pd.DataFrame(ablation_summary)
    summary_csv = tbl_dir / "ablation_ladder_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved Ablation Ladder Scorecard to: {summary_csv}")

    # Render formatted ablation table figure
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_fig_path = fig_dir / "ablation_ladder_table.png"
    table_records = [
        {
            "config": r["stage"],
            "dice": r["dice"],
            "iou": r["iou"],
            "pq": r["pq"],
            "count_error": r["count_error"],
            "area_error": r["area_error"],
        }
        for r in ablation_summary
    ]
    plot_ablation_table(table_records, save_path=str(table_fig_path))
    print(f"Rendered Ablation Ladder table figure to: {table_fig_path}")

    # Display Markdown Table
    print("\n" + "=" * 70)
    print("                    ABLATION LADDER FINAL RESULTS")
    print("=" * 70)
    print(summary_df.to_markdown(index=False))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stages", nargs="+", default=["M5", "M4", "M3", "M2", "M1"])
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    run_ablation(args.config, args.stages, default_epochs=args.epochs)
