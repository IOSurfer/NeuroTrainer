"""
UNet3D decoders.

UNet3DDecoder : (b, e4..e1) -> (d1, d2, d3, d4)  finest -> coarsest, used by UNet3D.
EffiDec3D     : alternative residual/upsampling decoder (not currently wired into UNet3D).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.blocks import (
    ChannelReductionResidualBlock,
    ResidualUpBlock,
    Up,
)


class UNet3DDecoder(nn.Module):
    """Four-level decoder with skip connections. Returns (d1, d2, d3, d4) finest -> coarsest."""

    def __init__(self, base_features: int, trilinear: bool = True) -> None:
        super().__init__()
        f = base_features
        factor = 2 if trilinear else 1
        self.dec4 = Up(f * 16, f * 8 // factor, trilinear)
        self.dec3 = Up(f * 8, f * 4 // factor, trilinear)
        self.dec2 = Up(f * 4, f * 2 // factor, trilinear)
        self.dec1 = Up(f * 2, f, trilinear)

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


class EffiDec3D(nn.Module):

    def __init__(self, base_features: int) -> None:
        super().__init__()
        f = base_features
        self.dec4 = ChannelReductionResidualBlock(f * 16, f)
        self.dec3 = ResidualUpBlock(f * 8, f)
        self.dec2 = ResidualUpBlock(f * 4, f)
        self.dec1 = ResidualUpBlock(f * 2, f)

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
        d1 = self.dec1(d2, e2)
        return d1, d2, d3, d4
