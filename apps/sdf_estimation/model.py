"""
3D U-Net for multi-field SDF estimation.

Component hierarchy
-------------------
UNet3DSDF
├── encoder    : UNet3DEncoder    -- input -> (e1, e2, e3, e4)
├── bottleneck : UNet3DBottleneck -- e4 -> b
├── decoder    : UNet3DDecoder    -- (b, e4..e1) -> (d1, d2, d3, d4)
└── sdf_head   : SDFHead          -- d1 -> [B, num_sdf_fields, D, H, W]  float32

Key differences from the segmentation UNet
-------------------------------------------
- Output is raw float32 (no softmax / sigmoid); values represent signed
  distances in the same physical units as the ground-truth SDF files.
- ``num_sdf_fields`` replaces ``num_classes``; the head is a single linear
  1×1×1 convolution (no activation).
- No EvalHead / InferHead are needed -- the raw output IS the prediction.
- Deep supervision is not included; the SDF Eikonal loss must be applied at
  full resolution to be physically meaningful.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def _GroupNorm(channels: int) -> nn.GroupNorm:
    """Largest power-of-2 group count that evenly divides *channels*, max 32."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


# ── Primitive blocks (shared with labelmap_segmentation) ──────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  mid_ch, kernel_size=3, padding=1, bias=False),
            _GroupNorm(mid_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _GroupNorm(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Decoder block: upsample then concatenate skip connection."""

    def __init__(self, in_ch: int, out_ch: int, trilinear: bool = True) -> None:
        super().__init__()
        if trilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2,
                        dy // 2, dy - dy // 2,
                        dz // 2, dz - dz // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ── Structural components ──────────────────────────────────────────────────────

class UNet3DEncoder(nn.Module):
    """Four-level encoder. Returns (e1, e2, e3, e4) from finest to coarsest."""

    def __init__(self, in_channels: int, base_features: int) -> None:
        super().__init__()
        f = base_features
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,     f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4


class UNet3DBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Down(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DDecoder(nn.Module):
    """Four-level decoder with skip connections. Returns (d1, d2, d3, d4) finest -> coarsest."""

    def __init__(self, base_features: int, trilinear: bool = True) -> None:
        super().__init__()
        f      = base_features
        factor = 2 if trilinear else 1
        self.dec4 = Up(f * 16, f * 8  // factor, trilinear)
        self.dec3 = Up(f * 8,  f * 4  // factor, trilinear)
        self.dec2 = Up(f * 4,  f * 2  // factor, trilinear)
        self.dec1 = Up(f * 2,  f,                trilinear)

    def forward(
        self,
        b:  torch.Tensor,
        e4: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return d1, d2, d3, d4


# ── SDF output head ────────────────────────────────────────────────────────────

class SDFHead(nn.Module):
    """
    Maps finest-scale decoder features to multi-channel SDF predictions.

    Output is raw float32 with **no activation** -- values represent signed
    distances and can be positive, negative, or zero.

    Args:
        base_features:  Feature channels at the finest decoder level (d1).
        num_sdf_fields: Number of SDF outputs to predict simultaneously.
    """

    def __init__(self, base_features: int, num_sdf_fields: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(base_features, num_sdf_fields, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Composite model ────────────────────────────────────────────────────────────

class UNet3DSDF(nn.Module):
    """
    3D U-Net for volumetric multi-field SDF estimation.

    Args:
        in_channels:    Number of input modalities / channels.
        num_sdf_fields: Number of SDF fields to predict simultaneously.
        base_features:  Feature channels at the first encoder level.
        trilinear:      True = trilinear upsampling, False = transposed conv.

    Forward
    -------
    ``forward(x) -> Tensor [B, num_sdf_fields, D, H, W]``  raw float32 SDF values.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_sdf_fields: int = 1,
        base_features: int = 32,
        trilinear: bool = True,
    ) -> None:
        super().__init__()
        f      = base_features
        factor = 2 if trilinear else 1

        self.encoder    = UNet3DEncoder(in_channels, f)
        self.bottleneck = UNet3DBottleneck(f * 8, f * 16 // factor)
        self.decoder    = UNet3DDecoder(f, trilinear)
        self.sdf_head   = SDFHead(f, num_sdf_fields)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw SDF predictions ``[B, num_sdf_fields, D, H, W]``.
        No activation is applied -- values encode signed distances.
        """
        e1, e2, e3, e4 = self.encoder(x)
        b  = self.bottleneck(e4)
        d1, *_ = self.decoder(b, e4, e3, e2, e1)
        return self.sdf_head(d1)
