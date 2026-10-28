#!/usr/bin/env bash
# ==========================================================================
#  finetune_continue.sh
#  Fine-tune continuation from the best checkpoint produced by the full
#  retrain. Strategy:
#    - Resume weights from unet_full_pipeline_best.pth
#    - Lower LR to 1e-5 (10x reduction = fine-tune regime)
#    - Cosine Annealing Warm Restarts (T_0=20 epochs) to escape local minima
#    - Time-limited: max 3 hours (10,800 seconds)
#    - Early stopping: patience = 20 epochs, min_delta = 0.0005
#    - Save best to: unet_finetune_best.pth
#    - Compare with previous best (0.7513) and report improvement
# ==========================================================================
set -e

cd /home/vhelambe/tmp/histopathology-ai
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

LOG_FILE="/home/vhelambe/tmp/histopathology-ai/finetune_continue.log"
CKPT_IN="results/checkpoints/unet_full_pipeline_best.pth"
CKPT_OUT="unet_finetune_best.pth"
CSV_OUT="eval_finetune_best.csv"
PYTHON="/home/vhelambe/tmp/gpu_env/bin/python"

echo "==========================================================" | tee "$LOG_FILE"
echo "  HISTOPATHOLOGY AI: FINE-TUNE CONTINUATION (3-HOUR RUN)" | tee -a "$LOG_FILE"
echo "  Resume:      $CKPT_IN"                                    | tee -a "$LOG_FILE"
echo "  Output:      results/checkpoints/$CKPT_OUT"              | tee -a "$LOG_FILE"
echo "  LR:          1e-5 (fine-tune regime, 10x lower)"         | tee -a "$LOG_FILE"
echo "  Scheduler:   Cosine Annealing Warm Restarts T_0=20"      | tee -a "$LOG_FILE"
echo "  Max epochs:  150 (~3 hours), Early stop patience=20"     | tee -a "$LOG_FILE"
echo "  Baseline:    Val Dice = 0.7513 (unet_full_pipeline_best)"| tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# Step 1: Write inline Python fine-tune script
cat > /tmp/finetune_run.py << 'PYEOF'
"""
finetune_run.py  — inline fine-tune script
Resumes from a pre-trained checkpoint, applies lower LR and
CosineAnnealingWarmRestarts, with time-limit + early stopping.
"""
import sys, os, time, yaml, random, argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, "/home/vhelambe/tmp/histopathology-ai/src")
from train import NucleiDataset, train_one_epoch, validate
from models import build_model
from losses import get_loss_fn
from hardware import setup_hardware, check_system_resources

# ── Config ──────────────────────────────────────────────────────────────────
CFG_PATH      = "/home/vhelambe/tmp/histopathology-ai/configs/config.yaml"
CKPT_IN       = "/home/vhelambe/tmp/histopathology-ai/results/checkpoints/unet_full_pipeline_best.pth"
CKPT_OUT_NAME = "unet_finetune_best.pth"
CKPT_DIR      = Path("/home/vhelambe/tmp/histopathology-ai/results/checkpoints")

FINETUNE_LR   = 1e-5           # 10x lower than original 1e-4
MAX_EPOCHS    = 150             # upper bound — time limit will stop earlier
T_0           = 20              # cosine warm restart period (epochs)
PATIENCE      = 20              # early stopping patience
MIN_DELTA     = 5e-4            # min improvement to reset patience counter
PREV_BEST     = 0.7513          # val dice from the previous 50-epoch run
TIME_LIMIT_S  = 10_500          # 2h 55m — leaves 5 min for eval; 3h = 10800s

# ── Setup ────────────────────────────────────────────────────────────────────
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

random.seed(cfg["training"]["seed"])
np.random.seed(cfg["training"]["seed"])
torch.manual_seed(cfg["training"]["seed"])

hw_cfg = cfg.get("hardware", {})
hw     = setup_hardware(
    gpu_fraction   = float(hw_cfg.get("gpu_memory_fraction", 0.90)),
    max_ram_fraction = float(hw_cfg.get("max_ram_fraction", 0.90))
)
device = hw["device"]
use_amp = hw_cfg.get("amp", False) and device.type == "cuda"
scaler  = torch.amp.GradScaler("cuda") if use_amp else None

print("=" * 60)
print("       FINE-TUNE CONTINUATION — HARDWARE PROFILE")
print("=" * 60)
print(f"  Device:   {device}")
if hw["gpu"]:
    g = hw["gpu"]
    print(f"  GPU:      {g.get('name','?')}  VRAM cap={g['vram_cap_mb']}MB")
print(f"  LR:       {FINETUNE_LR}   (down from 1e-4)")
print(f"  Prev best Dice: {PREV_BEST}")
print(f"  Max epochs: {MAX_EPOCHS}  |  Time limit: {TIME_LIMIT_S}s ({TIME_LIMIT_S/3600:.2f}h)")
print("=" * 60)

# ── Data ─────────────────────────────────────────────────────────────────────
data_root  = Path(cfg["data"]["root"])
train_ds   = NucleiDataset(data_root/"tiles"/"train"/"images",
                           data_root/"tiles"/"train"/"masks", augment=True)
val_ds     = NucleiDataset(data_root/"tiles"/"val"/"images",
                           data_root/"tiles"/"val"/"masks",   augment=False)

num_workers = hw_cfg.get("num_workers", 4)
pin_mem     = device.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                          shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                          shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

print(f"\nDataset — train: {len(train_ds)} tiles | val: {len(val_ds)} tiles")

# ── Model — load checkpoint ──────────────────────────────────────────────────
model = build_model(cfg["model"]).to(device)
state = torch.load(CKPT_IN, map_location=device)
model.load_state_dict(state)
print(f"  [OK] Loaded weights from {CKPT_IN}")

# ── Optimizer & Scheduler ────────────────────────────────────────────────────
loss_fn   = get_loss_fn(cfg["training"]["loss"], cfg["training"]["lambda_bce"])
optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_0, T_mult=1, eta_min=1e-7
)

# ── Training loop ─────────────────────────────────────────────────────────────
best_dice    = PREV_BEST          # we beat the old record, not just 0
no_improve   = 0
start_time   = time.time()
grad_accum   = int(cfg["training"].get("gradient_accumulation_steps", 1))

print(f"\n  Training starts — beating {PREV_BEST:.4f} is the goal\n")

for epoch in range(1, MAX_EPOCHS + 1):
    elapsed = time.time() - start_time
    if elapsed >= TIME_LIMIT_S:
        print(f"\n  [TIME] {elapsed/3600:.2f}h elapsed — stopping before epoch {epoch}.")
        break

    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,
                                 device, scaler=scaler, grad_accum_steps=grad_accum)
    val_loss, val_dice = validate(model, val_loader, loss_fn, device)
    scheduler.step(epoch)          # CosineAnnealingWarmRestarts uses epoch float

    res      = check_system_resources()
    vram_str = (f" | VRAM: {res.get('vram_allocated_mb',0):.0f}MB"
                f" (Peak: {res.get('vram_peak_mb',0):.0f}MB)"
                if "vram_allocated_mb" in res else "")
    ram_str  = f" | RAM: {res['ram_percent']:.1f}%"
    cur_lr   = optimizer.param_groups[0]["lr"]
    elapsed_h = (time.time() - start_time) / 3600

    print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
          f"Dice: {val_dice:.4f} | LR: {cur_lr:.2e} | "
          f"Elapsed: {elapsed_h:.2f}h{vram_str}{ram_str}")

    # Save best
    if val_dice > best_dice + MIN_DELTA:
        best_dice  = val_dice
        no_improve = 0
        out_path   = CKPT_DIR / CKPT_OUT_NAME
        torch.save(model.state_dict(), out_path)
        print(f"  [BEST ★] Dice {best_dice:.4f} → saved to {out_path}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n  [EARLY STOP] No improvement for {PATIENCE} epochs. Best={best_dice:.4f}")
            break

total_h = (time.time() - start_time) / 3600
print(f"\n{'='*60}")
print(f"  Fine-tune complete in {total_h:.2f}h")
print(f"  Previous best Dice:  {PREV_BEST:.4f}")
print(f"  Final best Dice:     {best_dice:.4f}")
if best_dice > PREV_BEST:
    delta = best_dice - PREV_BEST
    print(f"  IMPROVEMENT:        +{delta:.4f} (+{delta/PREV_BEST*100:.2f}%)")
else:
    print(f"  No improvement over previous best — keep unet_full_pipeline_best.pth")
print(f"{'='*60}")
PYEOF

echo "[1/3] Starting fine-tune..." | tee -a "$LOG_FILE"
$PYTHON /tmp/finetune_run.py 2>&1 | tee -a "$LOG_FILE"

# Step 2: Only run evaluation if we have a better checkpoint
FINETUNE_CKPT="results/checkpoints/$CKPT_OUT"
if [ -f "$FINETUNE_CKPT" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "[2/3] Fine-tuned checkpoint found — running full test cohort evaluation..." | tee -a "$LOG_FILE"
    $PYTHON src/evaluate.py \
        --config configs/config.yaml \
        --weights "$FINETUNE_CKPT" \
        --stain-norm macenko \
        --output-csv "$CSV_OUT" 2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "[3/3] Generating interpretability dashboard for comparison slide..." | tee -a "$LOG_FILE"
    $PYTHON inspect_pipeline.py \
        --image "data/MoNuSeg/images/TCGA-21-5784-01Z-00-DX1.tif" \
        --weights "$FINETUNE_CKPT" \
        --threshold 0.15 \
        --min-distance 5 \
        --out-dir "results/figures/interpretability" 2>&1 | tee -a "$LOG_FILE"
else
    echo "[SKIP] No improvement checkpoint saved — previous best remains unet_full_pipeline_best.pth" | tee -a "$LOG_FILE"
fi

echo "==========================================================" | tee -a "$LOG_FILE"
echo "  FINE-TUNE RUN COMPLETE — check finetune_continue.log   " | tee -a "$LOG_FILE"
echo "  Eval results: results/tables/$CSV_OUT                  " | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"
