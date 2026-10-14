"""
Primitive 3D building blocks shared by the UNet3D encoder, decoder and heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Helpers ────────────────────────────────────────────────────────────────────


def _GroupNorm(channels: int) -> nn.GroupNorm:
    """Largest power-of-2 group count that evenly divides *channels*, max 32."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)  # fallback (always divides)


# ── Primitive blocks ───────────────────────────────────────────────────────────


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None) -> None:
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
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
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2]
        )
        return self.conv(torch.cat([x2, x1], dim=1))


# ── Residual / ConvNeXt blocks (alternative encoders & decoders) ───────────────


class ResBlock(nn.Module):
    """Basic residual block (nnU-Net ResEnc style): conv-norm-act, conv-norm, + skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _GroupNorm(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _GroupNorm(out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                _GroupNorm(out_ch),
            )
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt / MedNeXt style block: depthwise large-kernel conv -> norm ->
    1x1x1 expand -> GELU -> 1x1x1 compress, with an identity residual.
    """

    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv3d(
            ch, ch, kernel_size, padding=kernel_size // 2, groups=ch, bias=False
        )
        self.norm = _GroupNorm(ch)
        self.expand = nn.Conv3d(ch, ch * expansion, 1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(ch * expansion, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.compress(self.act(self.expand(self.norm(self.dw(x)))))


class ConvNeXtStage(nn.Module):
    """Optional channel projection followed by ``depth`` ConvNeXt blocks."""

    def __init__(
        self, in_ch: int, out_ch: int, depth: int, kernel_size: int = 5, expansion: int = 2
    ) -> None:
        super().__init__()
        self.proj = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(nn.Conv3d(in_ch, out_ch, 1, bias=False), _GroupNorm(out_ch))
        )
        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(out_ch, kernel_size, expansion) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class StridedDown(nn.Module):
    """Learned 2x downsampling (2x2x2 stride-2 conv + norm), used by ResNet/ConvNeXt stages."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            _GroupNorm(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResUp(nn.Module):
    """Decoder block: trilinear upsample, concat skip, 1x1 fuse, then a residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 1, bias=False), _GroupNorm(out_ch)
        )
        self.res = ResBlock(out_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2]
        )
        return self.res(self.fuse(torch.cat([x2, x1], dim=1)))


# ── EffiDec3D blocks ───────────────────────────────────────────────────────────


class ChannelReductionResidualBlock(nn.Module):

    def __init__(self, in_ch: int, num_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, num_features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=num_features),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=num_features),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv3d(in_ch, num_features, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(num_features=num_features),
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        x_res = self.res(x)
        return self.relu(y + x_res)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_ch: int, num_features: int) -> None:
        super().__init__()
        self.reduction_residual = ChannelReductionResidualBlock(in_ch, num_features)
        self.up = nn.ConvTranspose3d(
            num_features, num_features, kernel_size=2, stride=2
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        u = self.up(x1)
        f = self.reduction_residual(x2)
        dz = f.size(2) - u.size(2)
        dy = f.size(3) - u.size(3)
        dx = f.size(4) - u.size(4)
        u = F.pad(
            u, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2]
        )
        return u + f
