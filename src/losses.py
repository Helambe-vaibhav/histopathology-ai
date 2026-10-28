"""
losses.py
---------
Functions:
  dice_loss()      - soft Dice loss
  bce_loss()       - binary cross-entropy with logits
  combined_loss()  - λ·BCE + (1-λ)·Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss (numerically stable for AMP FP16).
    """
    probs = torch.sigmoid(logits.float())
    probs = probs.contiguous().view(-1)
    targets = targets.contiguous().view(-1).float()

    intersection = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + smooth
    dice = (2.0 * intersection + smooth) / denom
    return 1.0 - dice


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss with logits (numerically stable for AMP FP16).
    """
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_bce: float = 0.5,
) -> torch.Tensor:
    """
    Combined loss: λ·BCE + (1-λ)·Dice

    Args:
        logits:     (B, 1, H, W)
        targets:    (B, 1, H, W) float binary ground truth
        lambda_bce: weight for BCE term (0..1)

    Returns:
        scalar combined loss
    """
    bce = bce_loss(logits, targets)
    dice = dice_loss(logits, targets)
    return lambda_bce * bce + (1.0 - lambda_bce) * dice


def get_loss_fn(loss_name: str, lambda_bce: float = 0.5):
    """
    Return loss function by name.

    Args:
        loss_name: 'bce' | 'dice' | 'combined'
        lambda_bce: weight for BCE in combined loss
    """
    if loss_name == "bce":
        return bce_loss
    elif loss_name == "dice":
        return dice_loss
    elif loss_name == "combined":
        return lambda logits, targets: combined_loss(logits, targets, lambda_bce=lambda_bce)
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose bce | dice | combined")
