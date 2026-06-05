"""
3D U-Net building blocks and composite model for label-map segmentation.

Component hierarchy
-------------------
UNet3D
├── encoder    : UNet3DEncoder    -- input -> (e1, e2, e3, e4)
├── bottleneck : UNet3DBottleneck -- e4 -> b
├── decoder    : UNet3DDecoder    -- (b, e4..e1) -> (d1, d2, d3, d4)
├── train_head : TrainHead        -- decoder features -> List[logits per level]
├── eval_head  : EvalHead         -- logits -> (labelmap, uncertainty)
└── infer_head : InferHead        -- logits -> labelmap (argmax only)

Forward entry points on UNet3D
-------------------------------
forward(x)         -> Tensor            main logits [B, C, D, H, W]
forward_train(x)   -> List[Tensor]      logits per supervision level (finest first)
forward_eval(x)    -> (Tensor, Tensor)  (labelmap [B,D,H,W], uncertainty [B,D,H,W])
forward_infer(x)   -> Tensor            argmax labelmap [B, D, H, W]
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

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
    """Bridge between encoder and decoder (single strided-conv block)."""

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
        return d1, d2, d3, d4  # finest -> coarsest


# ── Task heads ─────────────────────────────────────────────────────────────────

class TrainHead(nn.Module):
    """
    Projects decoder feature maps to raw logits at each active supervision level.

    Index 0 is the main output (finest resolution = full spatial size).
    Subsequent entries are auxiliary outputs at 1/2, 1/4, 1/8 spatial sizes,
    used only when num_supervision_levels > 1 (deep supervision).

    Deep-supervision loss example::

        logits_list = model.forward_train(x)
        loss = sum(
            w * criterion(lg, downsample_target(y, i))
            for i, (w, lg) in enumerate(
                zip(TrainHead.SUPERVISION_WEIGHTS, logits_list))
        )

    Returns List[Tensor] of shape [B, num_classes, *spatial] per level.
    """

    SUPERVISION_WEIGHTS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)

    def __init__(
        self,
        base_features: int,
        num_classes: int,
        trilinear: bool = True,
        num_supervision_levels: int = 1,
    ) -> None:
        super().__init__()
        factor = 2 if trilinear else 1
        # Output-feature channels at each decoder level (finest -> coarsest)
        ch_per_level: Tuple[int, ...] = (
            base_features,               # d1
            base_features * 2 // factor, # d2
            base_features * 4 // factor, # d3
            base_features * 8 // factor, # d4
        )
        n = min(max(num_supervision_levels, 1), len(ch_per_level))
        self.convs = nn.ModuleList(
            nn.Conv3d(ch_per_level[i], num_classes, kernel_size=1)
            for i in range(n)
        )

    def forward(self, decoder_features: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        # decoder_features: (d1, d2, d3, d4) finest -> coarsest
        return [conv(feat) for conv, feat in zip(self.convs, decoder_features)]


class EvalHead(nn.Module):
    """
    Converts main logits to a predicted label-map and a normalized-entropy
    uncertainty map.

    Input : logits [B, C, D, H, W]  (raw, from TrainHead.convs[0])
    Output:
        labelmap    [B, D, H, W]  int64  predicted class per voxel
        uncertainty [B, D, H, W]  float  normalized entropy in [0, 1]
                                          0 = certain  /  1 = maximally uncertain

    No trainable parameters — pure post-processing.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self._max_entropy = math.log(max(num_classes, 2))

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_classes == 1:
            probs   = torch.sigmoid(logits).squeeze(1)        # [B, D, H, W]
            labelmap = (probs > 0.5).long()
            p        = probs.clamp(1e-6, 1 - 1e-6)
            entropy  = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            uncertainty = (entropy / math.log(2)).clamp(0.0, 1.0)
        else:
            probs   = F.softmax(logits, dim=1)                 # [B, C, D, H, W]
            labelmap = probs.argmax(dim=1)
            entropy  = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)
            uncertainty = (entropy / self._max_entropy).clamp(0.0, 1.0)
        return labelmap, uncertainty


class InferHead(nn.Module):
    """
    Fastest inference head: argmax over raw logits (softmax skipped — monotone).

    Input : logits [B, C, D, H, W]
    Output: labelmap [B, D, H, W]  int64 class indices

    No trainable parameters.
    """

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(1) == 1:
            return (logits.squeeze(1) > 0.0).long()
        return logits.argmax(dim=1)


# ── Composite model ────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    3D U-Net for volumetric label-map segmentation.

    Args:
        in_channels:            Number of input modalities / channels.
        num_classes:            Number of output classes (1 = binary).
        base_features:          Feature channels at the first encoder level.
        trilinear:              True = trilinear upsampling, False = transposed conv.
        num_supervision_levels: 1 = standard single output;
                                >1 = deep supervision with auxiliary heads at
                                coarser resolutions (max 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 32,
        trilinear: bool = True,
        num_supervision_levels: int = 1,
    ) -> None:
        super().__init__()
        f      = base_features
        factor = 2 if trilinear else 1

        self.encoder    = UNet3DEncoder(in_channels, f)
        self.bottleneck = UNet3DBottleneck(f * 8, f * 16 // factor)
        self.decoder    = UNet3DDecoder(f, trilinear)
        self.train_head = TrainHead(f, num_classes, trilinear, num_supervision_levels)
        self.eval_head  = EvalHead(num_classes)
        self.infer_head = InferHead()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    # ── Backbone ───────────────────────────────────────────────────────────────

    def _backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encoder -> bottleneck -> decoder. Returns (d1, d2, d3, d4)."""
        e1, e2, e3, e4 = self.encoder(x)
        b = self.bottleneck(e4)
        return self.decoder(b, e4, e3, e2, e1)

    # ── Forward entry points ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main logits at finest scale [B, C, D, H, W]. Backward-compatible with loss functions."""
        return self.train_head(self._backbone(x))[0]

    def forward_train(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        One logit tensor per active supervision level, finest first.

        Single level -> identical to forward().
        Multiple levels -> use with TrainHead.SUPERVISION_WEIGHTS for a
        weighted deep-supervision loss.
        """
        return self.train_head(self._backbone(x))

    def forward_eval(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (labelmap [B, D, H, W], uncertainty [B, D, H, W]).

        For quantitative metrics that need raw logits, use forward() instead.
        """
        logits = self.train_head(self._backbone(x))[0]
        return self.eval_head(logits)

    def forward_infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns argmax label-map [B, D, H, W].

        Preferred for deployment: skips softmax computation entirely.
        """
        return self.infer_head(self.train_head(self._backbone(x))[0])
