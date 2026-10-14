"""
UNet3D encoders and bottlenecks.

Every encoder maps ``input -> (e1, e2, e3, e4)`` (finest -> coarsest) with channels
``(f, 2f, 4f, 8f)`` at strides ``(1, 2, 4, 8)``, and every bottleneck maps
``e4 -> b`` (stride 16, ``16f // factor`` channels) so that any encoder can be paired
with any decoder (see ``decoder.py``).

encoder_type  block                                         reference
------------  --------------------------------------------  ------------------------------
conv3d        DoubleConv + MaxPool (plain U-Net)            Cicek 2016 / nnU-Net 2021
res           residual blocks, strided-conv downsampling    nnU-Net ResEnc 2024
convnext      depthwise large-kernel ConvNeXt blocks        MedNeXt 2023 / 3D UX-Net 2022
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.blocks import (
    ConvNeXtStage,
    DoubleConv,
    Down,
    ResBlock,
    StridedDown,
    _GroupNorm,
)


class UNet3DEncoder(nn.Module):
    """Four-level encoder. Returns (e1, e2, e3, e4) from finest to coarsest."""

    def __init__(self, in_channels: int, base_features: int) -> None:
        super().__init__()
        f = base_features
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4


class UNet3DBottleneck(nn.Module):
    """Bridge between encoder and decoder (single strided-conv block)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = Down(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Residual encoder ───────────────────────────────────────────────────────────


class _ResStage(nn.Sequential):
    """``n`` residual blocks; the first one changes channels / strides."""

    def __init__(self, in_ch: int, out_ch: int, n: int, stride: int) -> None:
        blocks = [ResBlock(in_ch, out_ch, stride)]
        blocks += [ResBlock(out_ch, out_ch) for _ in range(n - 1)]
        super().__init__(*blocks)


class ResEncoder(nn.Module):
    """Residual encoder with ``depths`` blocks per stage and stride-2 convs for downsampling."""

    def __init__(
        self,
        in_channels: int,
        base_features: int,
        depths: Sequence[int] = (1, 2, 2, 2),
    ) -> None:
        super().__init__()
        f = base_features
        self.enc1 = _ResStage(in_channels, f, depths[0], 1)
        self.enc2 = _ResStage(f, f * 2, depths[1], 2)
        self.enc3 = _ResStage(f * 2, f * 4, depths[2], 2)
        self.enc4 = _ResStage(f * 4, f * 8, depths[3], 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        return e1, e2, e3, e4


class ResBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = _ResStage(in_channels, out_channels, 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── ConvNeXt (MedNeXt-style) encoder ───────────────────────────────────────────


class ConvNeXtEncoder(nn.Module):
    """3x3x3 conv stem, then ConvNeXt stages joined by stride-2 convs."""

    def __init__(
        self,
        in_channels: int,
        base_features: int,
        depths: Sequence[int] = (1, 2, 2, 2),
        kernel_size: int = 5,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        f = base_features
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, f, 3, padding=1, bias=False), _GroupNorm(f)
        )
        self.enc1 = ConvNeXtStage(f, f, depths[0], kernel_size, expansion)
        self.down2 = StridedDown(f, f * 2)
        self.enc2 = ConvNeXtStage(f * 2, f * 2, depths[1], kernel_size, expansion)
        self.down3 = StridedDown(f * 2, f * 4)
        self.enc3 = ConvNeXtStage(f * 4, f * 4, depths[2], kernel_size, expansion)
        self.down4 = StridedDown(f * 4, f * 8)
        self.enc4 = ConvNeXtStage(f * 8, f * 8, depths[3], kernel_size, expansion)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        e4 = self.enc4(self.down4(e3))
        return e1, e2, e3, e4


class ConvNeXtBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = StridedDown(in_channels, out_channels)
        self.stage = ConvNeXtStage(out_channels, out_channels, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(self.down(x))


# ── Factory ────────────────────────────────────────────────────────────────────

_ENCODERS: Dict[str, Tuple[Callable[..., nn.Module], Callable[..., nn.Module]]] = {
    "conv3d": (UNet3DEncoder, UNet3DBottleneck),
    "res": (ResEncoder, ResBottleneck),
    "convnext": (ConvNeXtEncoder, ConvNeXtBottleneck),
}


def available_encoders() -> Tuple[str, ...]:
    return tuple(_ENCODERS)


def build_encoder(
    encoder_type: str, in_channels: int, base_features: int, trilinear: bool = True
) -> Tuple[nn.Module, nn.Module]:
    """Return ``(encoder, bottleneck)`` for *encoder_type*."""
    if encoder_type not in _ENCODERS:
        raise KeyError(f"Unknown encoder_type {encoder_type!r}. Available: {available_encoders()}")
    enc_cls, bott_cls = _ENCODERS[encoder_type]
    factor = 2 if trilinear else 1
    f = base_features
    return enc_cls(in_channels, f), bott_cls(f * 8, f * 16 // factor)
