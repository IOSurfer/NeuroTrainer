"""
UNet3D decoders.

Both decoders map ``(b, e4, e3, e2, e1) -> (d1, d2, d3, d4)`` (finest -> coarsest) and
expose

    out_channels  : channels of (d1, d2, d3, d4), used to size the segmentation heads
    output_stride : stride of d1 w.r.t. the input (1 = full resolution; 2 = half
                    resolution, ``UNet3D`` upsamples the logits)

decoder_type  description
------------  ------------------------------------------------------------------
conv3d        U-Net decoder: upsample, concat skip, DoubleConv
res           upsample, concat skip, 1x1 fuse, residual block
effidec       EffiDec3D: channel-reduced residual skips summed with transposed-conv
              upsampling, no full-resolution decoder stage (about half the memory)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.blocks import (
    ChannelReductionResidualBlock,
    ResidualUpBlock,
    ResUp,
    Up,
)


class UNet3DDecoder(nn.Module):
    """Four-level decoder with skip connections. Returns (d1, d2, d3, d4) finest -> coarsest."""

    output_stride = 1

    def __init__(self, base_features: int, trilinear: bool = True) -> None:
        super().__init__()
        f = base_features
        factor = 2 if trilinear else 1
        self.dec4 = Up(f * 16, f * 8 // factor, trilinear)
        self.dec3 = Up(f * 8, f * 4 // factor, trilinear)
        self.dec2 = Up(f * 4, f * 2 // factor, trilinear)
        self.dec1 = Up(f * 2, f, trilinear)
        self.out_channels = (f, f * 2 // factor, f * 4 // factor, f * 8 // factor)

    def forward(
        self,
        b: torch.Tensor,
        e4: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return d1, d2, d3, d4  # finest -> coarsest


class ResDecoder(nn.Module):
    """U-Net decoder built from residual blocks (same channel plan as UNet3DDecoder)."""

    output_stride = 1

    def __init__(self, base_features: int, trilinear: bool = True) -> None:
        super().__init__()
        f = base_features
        factor = 2 if trilinear else 1
        b_ch = f * 16 // factor
        o4, o3, o2, o1 = f * 8 // factor, f * 4 // factor, f * 2 // factor, f
        self.dec4 = ResUp(b_ch, f * 8, o4)
        self.dec3 = ResUp(o4, f * 4, o3)
        self.dec2 = ResUp(o3, f * 2, o2)
        self.dec1 = ResUp(o2, f, o1)
        self.out_channels = (o1, o2, o3, o4)

    def forward(
        self,
        b: torch.Tensor,
        e4: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return d1, d2, d3, d4


class EffiDec3D(nn.Module):
    """
    EffiDec3D: every stage works with ``base_features`` channels, skips are reduced by a
    residual 1x1/3x3 block and added to the transposed-conv upsampled path. The highest
    resolution stage is dropped, so d1 is at stride 2 and logits must be upsampled.
    """

    output_stride = 2

    def __init__(self, base_features: int, trilinear: bool = True) -> None:
        super().__init__()
        f = base_features
        b_ch = f * 16 // (2 if trilinear else 1)
        self.dec4 = ChannelReductionResidualBlock(b_ch, f)
        self.dec3 = ResidualUpBlock(f * 8, f)
        self.dec2 = ResidualUpBlock(f * 4, f)
        self.dec1 = ResidualUpBlock(f * 2, f)
        self.out_channels = (f, f, f, f)

    def forward(
        self,
        b: torch.Tensor,
        e4: torch.Tensor,
        e3: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        d4 = self.dec4(b)
        d3 = self.dec3(d4, e4)
        d2 = self.dec2(d3, e3)
        d1 = self.dec1(d2, e2)  # e1 (full resolution) is intentionally unused
        return d1, d2, d3, d4


_DECODERS: Dict[str, Callable[..., nn.Module]] = {
    "conv3d": UNet3DDecoder,
    "res": ResDecoder,
    "effidec": EffiDec3D,
}


def available_decoders() -> Tuple[str, ...]:
    return tuple(_DECODERS)


def build_decoder(decoder_type: str, base_features: int, trilinear: bool = True) -> nn.Module:
    if decoder_type not in _DECODERS:
        raise KeyError(f"Unknown decoder_type {decoder_type!r}. Available: {available_decoders()}")
    return _DECODERS[decoder_type](base_features, trilinear)
