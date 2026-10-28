#!/usr/bin/env bash
set -e

cd /home/vhelambe/tmp/histopathology-ai
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

LOG_FILE="/home/vhelambe/tmp/histopathology-ai/retrain_and_evaluate.log"
CKPT="unet_full_pipeline_best.pth"
CSV_OUT="eval_retrained_full_pipeline.csv"

echo "==========================================================" | tee "$LOG_FILE"
echo " HISTOPATHOLOGY AI: FULL PIPELINE RETRAINING & EVALUATION" | tee -a "$LOG_FILE"
echo " Model: U-Net with Combined Loss (70% Dice, 30% BCE)       " | tee -a "$LOG_FILE"
echo " Augmentation: Spatial D4 + Histology Stain/Hue/Sat Jitter" | tee -a "$LOG_FILE"
echo " Hardware: RTX 2080 (8GB VRAM) Capped at 90% Allocation   " | tee -a "$LOG_FILE"
echo " Target Checkpoint: results/checkpoints/$CKPT             " | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Step 1: Retrain Model
echo "[1/2] Training model for 50 epochs..." | tee -a "$LOG_FILE"
/home/vhelambe/tmp/gpu_env/bin/python src/train.py \
    --config configs/config.yaml \
    --epochs 50 \
    --loss combined \
    --ckpt-name "$CKPT" >> "$LOG_FILE" 2>&1

echo "[+] Training complete! Best weights saved to results/checkpoints/$CKPT" | tee -a "$LOG_FILE"

# Step 2: Comprehensive Evaluation with Un-orphaned Pipeline
echo "[2/2] Running full un-orphaned clinical evaluation on test cohort..." | tee -a "$LOG_FILE"
/home/vhelambe/tmp/gpu_env/bin/python src/evaluate.py \
    --config configs/config.yaml \
    --weights "results/checkpoints/$CKPT" \
    --stain-norm macenko \
    --output-csv "$CSV_OUT" >> "$LOG_FILE" 2>&1

# Step 3: Multi-Stage Interpretability Dashboard Generation
echo "[3/3] Generating multi-stage interpretability dashboards across test organs..." | tee -a "$LOG_FILE"
mkdir -p results/figures/interpretability
for SLIDE in TCGA-21-5784-01Z-00-DX1 TCGA-AR-A1AK-01Z-00-DX1 TCGA-B0-5698-01Z-00-DX1 TCGA-HE-7130-01Z-00-DX1; do
    echo "  Generating interpretability dashboard for $SLIDE..." | tee -a "$LOG_FILE"
    /home/vhelambe/tmp/gpu_env/bin/python inspect_pipeline.py \
        --image "data/MoNuSeg/images/$SLIDE.tif" \
        --weights "results/checkpoints/$CKPT" \
        --threshold 0.15 \
        --min-distance 5 \
        --out-dir "results/figures/interpretability" >> "$LOG_FILE" 2>&1
done

echo "==========================================================" | tee -a "$LOG_FILE"
echo " RETRAINING, EVALUATION & INTERPRETABILITY COMPLETE!       " | tee -a "$LOG_FILE"
echo " Results saved to: results/tables/$CSV_OUT                " | tee -a "$LOG_FILE"
echo " Diagnostic Figures: results/figures/                      " | tee -a "$LOG_FILE"
echo " Interpretability Dashboards: results/figures/interpretability/ " | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"
