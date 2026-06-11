"""
Composite 3D U-Net model for label-map segmentation.

Component hierarchy
-------------------
UNet3D
├── encoder    : UNet3DEncoder    -- input -> (e1, e2, e3, e4)        (apps.labelmap_segmentation.encoder)
├── bottleneck : UNet3DBottleneck -- e4 -> b                           (apps.labelmap_segmentation.encoder)
├── decoder    : UNet3DDecoder    -- (b, e4..e1) -> (d1, d2, d3, d4)   (apps.labelmap_segmentation.decoder)
├── train_head : TrainHead        -- decoder features -> List[logits per level]  (apps.labelmap_segmentation.heads)
├── eval_head  : EvalHead         -- logits -> (labelmap, uncertainty) (apps.labelmap_segmentation.heads)
└── infer_head : InferHead        -- logits -> labelmap (argmax only)  (apps.labelmap_segmentation.heads)

Forward entry points on UNet3D
-------------------------------
forward(x)         -> Tensor            main logits [B, C, D, H, W]  float32
forward_train(x)   -> List[Tensor]      logits per supervision level (finest first)
forward_eval(x)    -> (Tensor, Tensor)  labelmap [B,D,H,W] uint8, uncertainty [B,D,H,W] float32
forward_infer(x)   -> Tensor            argmax labelmap [B, D, H, W]  uint8
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.decoder import UNet3DDecoder
from apps.labelmap_segmentation.encoder import UNet3DBottleneck, UNet3DEncoder
from apps.labelmap_segmentation.heads import EvalHead, InferHead, TrainHead


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
        f = base_features
        factor = 2 if trilinear else 1

        self.encoder = UNet3DEncoder(in_channels, f)
        self.bottleneck = UNet3DBottleneck(f * 8, f * 16 // factor)
        self.decoder = UNet3DDecoder(f, trilinear)
        self.train_head = TrainHead(f, num_classes, trilinear, num_supervision_levels)
        self.eval_head = EvalHead(num_classes)
        self.infer_head = InferHead()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )

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
