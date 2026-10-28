"""
train.py
--------
Dataset, DataLoader, training loop, validation, checkpointing.

Usage:
    python src/train.py --config configs/config.yaml
"""

import os
import random
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from models import build_model
from losses import get_loss_fn
from metrics import dice as dice_metric
from hardware import setup_hardware, check_system_resources


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NucleiDataset(Dataset):
    """
    Loads image tiles and corresponding binary mask tiles.

    Args:
        image_dir: directory containing image tiles (PNG)
        mask_dir:  directory containing binary mask tiles (PNG)
        augment:   whether to apply data augmentation
    """

    def __init__(self, image_dir: str, mask_dir: str, augment: bool = False):
        self.image_paths = sorted(Path(image_dir).glob("*.png"))
        self.mask_dir = Path(mask_dir)
        self.augment = augment

        self.to_tensor = transforms.ToTensor()  # converts [0,255] uint8 → [0,1] float

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")   # grayscale 0/255

        if self.augment:
            image, mask = self._augment(image, mask)

        image = self.to_tensor(image)                 # (3, H, W) float32 [0,1]
        mask  = self.to_tensor(mask)                  # (1, H, W) float32 [0,1]
        mask  = (mask > 0.5).float()                  # binarize

        return image, mask

    def _augment(self, image, mask):
        """Apply random augmentations consistently to image and mask."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # Random 90° rotation
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle)

        # Strong histological stain & color augmentation on image only
        if random.random() > 0.3:
            image = TF.adjust_brightness(image, 1 + random.uniform(-0.25, 0.25))
            image = TF.adjust_contrast(image, 1 + random.uniform(-0.25, 0.25))
            image = TF.adjust_saturation(image, 1 + random.uniform(-0.35, 0.35))
            image = TF.adjust_hue(image, random.uniform(-0.06, 0.06))

        return image, mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, grad_accum_steps: int = 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images = images.to(device)
        masks  = masks.to(device)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, masks) / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks) / grad_accum_steps
            loss.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        loss   = loss_fn(logits, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        gts   = masks.cpu().numpy().astype(np.uint8)

        for p, g in zip(preds, gts):
            dice_scores.append(dice_metric(p[0], g[0]))

    return total_loss / len(loader), float(np.mean(dice_scores))


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def train(
    cfg_path: str,
    epochs_override: int = None,
    model_type_override: str = None,
    loss_override: str = None,
    augment_override: bool = None,
    ckpt_name_override: str = None,
):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    # Configure hardware limits (90% GPU VRAM cap, 90% RAM cap, multi-core CPU threads)
    hw_cfg = cfg.get("hardware", {})
    gpu_fraction = float(hw_cfg.get("gpu_memory_fraction", 0.90))
    max_ram_frac = float(hw_cfg.get("max_ram_fraction", 0.90))
    hw_profile = setup_hardware(gpu_fraction=gpu_fraction, max_ram_fraction=max_ram_frac)

    device = hw_profile["device"]
    print("=" * 60)
    print("             SYSTEM HARDWARE & RESOURCE PROFILE")
    print("=" * 60)
    print(f"  Compute Device:     {device} ({hw_profile['gpu'].get('name', 'CPU') if hw_profile['gpu'] else 'CPU'})")
    if hw_profile["gpu"]:
        print(f"  GPU VRAM Cap (90%): {hw_profile['gpu']['vram_cap_mb']} MB / {hw_profile['gpu']['total_vram_mb']} MB")
    print(f"  CPU Cores:          {hw_profile['cpu_cores_total']} total ({hw_profile['cpu_threads_active']} active PyTorch threads)")
    print(f"  System RAM:         {hw_profile['available_ram_gb']} GB free / {hw_profile['total_ram_gb']} GB total (Current: {hw_profile['ram_usage_percent']:.1f}%)")
    print(f"  RAM Safety Ceiling: {max_ram_frac * 100:.0f}%")
    print("=" * 60)

    use_amp = hw_cfg.get("amp", False) and device.type == "cuda"
    if use_amp:
        print("  Automatic Mixed Precision (AMP): Enabled (FP16)")

    # Paths
    data_root = Path(cfg["data"]["root"])
    train_img  = data_root / "tiles" / "train" / "images"
    train_mask = data_root / "tiles" / "train" / "masks"
    val_img    = data_root / "tiles" / "val"   / "images"
    val_mask   = data_root / "tiles" / "val"   / "masks"
    ckpt_dir   = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if loss_override:
        cfg["training"]["loss"] = loss_override
    if augment_override is not None:
        cfg["augmentation"]["enabled"] = augment_override
    if model_type_override:
        cfg["model"]["type"] = model_type_override

    aug = cfg["augmentation"]["enabled"]
    train_ds = NucleiDataset(train_img, train_mask, augment=aug)
    val_ds   = NucleiDataset(val_img,   val_mask,   augment=False)

    num_workers = cfg.get("hardware", {}).get("num_workers", 0 if os.name == "nt" else 4)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    model    = build_model(cfg["model"]).to(device)
    loss_fn  = get_loss_fn(cfg["training"]["loss"], cfg["training"]["lambda_bce"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    
    total_epochs = epochs_override if epochs_override is not None else cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    best_dice  = -1.0
    model_name = cfg["model"]["type"]

    ckpt_save_name = ckpt_name_override if ckpt_name_override else f"{model_name}_best.pth"

    print(f"\nTraining {model_name} for {total_epochs} epochs (Batch size: {cfg['training']['batch_size']})")
    print(f"Loss: {cfg['training']['loss']} | Augmentation: {aug} | Target ckpt: {ckpt_save_name}")
    print(f"Train: {len(train_ds)} tiles | Val: {len(val_ds)} tiles\n")

    grad_accum_steps = int(cfg["training"].get("gradient_accumulation_steps", 1))
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler, grad_accum_steps=grad_accum_steps)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        # Log current allocated memory
        # Real-time resource check (RAM & VRAM)
        res = check_system_resources()
        vram_str = f" | VRAM: {res.get('vram_allocated_mb', 0):.0f}MB (Peak: {res.get('vram_peak_mb', 0):.0f}MB)" if "vram_allocated_mb" in res else ""
        ram_str = f" | RAM: {res['ram_percent']:.1f}% ({res['ram_used_gb']}GB)"

        print(f"Epoch {epoch:3d}/{total_epochs} | "
              f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}{vram_str}{ram_str}")

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = ckpt_dir / ckpt_save_name
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [BEST] New best Dice {best_dice:.4f} -> saved to {ckpt_path}")

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")
    return best_dice, str(ckpt_dir / ckpt_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--model-type", type=str, default=None, help="Override model type (unet | pretrained_unet)")
    parser.add_argument("--loss", type=str, default=None, help="Override loss (bce | dice | combined)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Override saved checkpoint filename")
    args = parser.parse_args()
    
    aug_override = False if args.no_augment else None
    train(
        args.config,
        epochs_override=args.epochs,
        model_type_override=args.model_type,
        loss_override=args.loss,
        augment_override=aug_override,
        ckpt_name_override=args.ckpt_name,
    )
