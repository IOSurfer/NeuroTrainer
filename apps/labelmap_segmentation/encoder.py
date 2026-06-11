"""
UNet3D encoder and bottleneck.

UNet3DEncoder    : input -> (e1, e2, e3, e4)  finest -> coarsest
UNet3DBottleneck : e4 -> b                     bridge to the decoder
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.blocks import DoubleConv, Down


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
