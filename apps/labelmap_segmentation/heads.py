"""
UNet3D output heads.

TrainHead : decoder features -> List[logits per supervision level]
EvalHead  : main logits -> (labelmap, normalized-entropy uncertainty)
InferHead : main logits -> labelmap (argmax only, deployment fast path)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            base_features,  # d1
            base_features * 2 // factor,  # d2
            base_features * 4 // factor,  # d3
            base_features * 8 // factor,  # d4
        )
        n = min(max(num_supervision_levels, 1), len(ch_per_level))
        self.convs = nn.ModuleList(
            nn.Conv3d(ch_per_level[i], num_classes, kernel_size=1) for i in range(n)
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
        self.num_classes = num_classes
        self._max_entropy = math.log(max(num_classes, 2))

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_classes == 1:
            probs = torch.sigmoid(logits).squeeze(1)  # [B, D, H, W]
            labelmap = (probs > 0.5).to(torch.uint8)
            p = probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            uncertainty = (entropy / math.log(2)).clamp(0.0, 1.0)
        else:
            probs = F.softmax(logits, dim=1)  # [B, C, D, H, W]
            labelmap = probs.argmax(dim=1).to(torch.uint8)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)
            uncertainty = (entropy / self._max_entropy).clamp(0.0, 1.0)
        return labelmap, uncertainty


class InferHead(nn.Module):
    """
    Fastest inference head: argmax over raw logits (softmax skipped — monotone).

    Input : logits [B, C, D, H, W]
    Output: labelmap [B, D, H, W]  uint8 class indices

    No trainable parameters.
    """

    @torch.no_grad()
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.size(1) == 1:
            return (logits.squeeze(1) > 0.0).to(torch.uint8)
        return logits.argmax(dim=1).to(torch.uint8)
