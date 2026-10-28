"""
models.py
---------
Models:
  UNet              - Minimal U-Net built from scratch
  PretrainedUNet    - U-Net with ResNet-34 encoder (segmentation_models_pytorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3×3 conv layers, each followed by BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample then concatenate skip, then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if shapes don't match exactly
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                       diffH // 2, diffH - diffH // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net from scratch  (minimal: 4 encoder stages)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Minimal U-Net from scratch.

    Channels: 64 → 128 → 256 → 512 → 1024 (bottleneck)

    Args:
        in_channels: number of input channels (3 for RGB)
        out_channels: number of output channels (1 for binary segmentation)
        base_ch: base channel width (default 64)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_ch: int = 64):
        super().__init__()
        b = base_ch

        # Encoder
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)

        # Bottleneck
        self.bottleneck = Down(b * 8, b * 16)

        # Decoder
        self.dec4 = Up(b * 16 + b * 8, b * 8)
        self.dec3 = Up(b * 8 + b * 4, b * 4)
        self.dec2 = Up(b * 4 + b * 2, b * 2)
        self.dec1 = Up(b * 2 + b, b)

        # Output
        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)

        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)   # raw logits, (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Pretrained encoder U-Net  (segmentation_models_pytorch)
# ---------------------------------------------------------------------------

def PretrainedUNet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    out_channels: int = 1,
):
    """
    U-Net with a pretrained encoder.

    Returns a segmentation_models_pytorch.Unet instance.
    Call as a drop-in replacement for UNet().
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError("Install segmentation-models-pytorch: pip install segmentation-models-pytorch")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,  # raw logits — loss functions apply sigmoid/softmax
    )
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Build model from config dict.

    cfg keys: type ('unet' | 'pretrained_unet'), encoder, pretrained
    """
    model_type = cfg.get("type", "unet")
    if model_type == "unet":
        return UNet()
    elif model_type == "pretrained_unet":
        weights = "imagenet" if cfg.get("pretrained", True) else None
        return PretrainedUNet(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=weights,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
