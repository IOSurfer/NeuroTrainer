"""
Composite 3D U-Net with multiple independent segmentation task heads sharing
one encoder / bottleneck / decoder backbone.

Component hierarchy
-------------------
MultiHeadUNet3D
├── encoder     : UNet3DEncoder    -- input -> (e1, e2, e3, e4)        (apps.labelmap_segmentation.encoder)
├── bottleneck  : UNet3DBottleneck -- e4 -> b                           (apps.labelmap_segmentation.encoder)
├── decoder     : UNet3DDecoder    -- (b, e4..e1) -> (d1, d2, d3, d4)   (apps.labelmap_segmentation.decoder)
├── train_heads : {task: TrainHead}  -- decoder features -> List[logits per level], per task
├── eval_heads  : {task: EvalHead}   -- logits -> (labelmap, uncertainty), per task
└── infer_heads : {task: InferHead}  -- logits -> labelmap (argmax only), per task

The backbone and head classes are reused unmodified from
apps.labelmap_segmentation -- TrainHead/EvalHead/InferHead are already
parameterised per-task by num_classes, so a multi-head model only needs one
head instance per task name, sharing a single backbone. ``encoder`` /
``bottleneck`` / ``decoder`` keep the same submodule names as ``UNet3D`` so
that checkpoint partial-loading (encoder / encoder_decoder) transfers
unchanged, including from a single-task UNet3D checkpoint.

Forward entry points on MultiHeadUNet3D
----------------------------------------
forward(x)         -> Dict[str, Tensor]                  main logits per task [B, C_t, D, H, W]
forward_train(x)   -> Dict[str, List[Tensor]]             logits per supervision level, per task
forward_eval(x)    -> Dict[str, Tuple[Tensor, Tensor]]    (labelmap, uncertainty) per task
forward_infer(x)   -> Dict[str, Tensor]                   argmax labelmap per task
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.decoder import UNet3DDecoder
from apps.labelmap_segmentation.encoder import UNet3DBottleneck, UNet3DEncoder
from apps.labelmap_segmentation.heads import EvalHead, InferHead, TrainHead


class MultiHeadUNet3D(nn.Module):
    """
    3D U-Net with one independent segmentation head per task.

    Args:
        in_channels:            Number of input modalities / channels.
        task_num_classes:       Mapping task name -> num_classes (1 = binary).
                                 One TrainHead/EvalHead/InferHead is created
                                 per entry.
        base_features:          Feature channels at the first encoder level.
        trilinear:              True = trilinear upsampling, False = transposed conv.
        num_supervision_levels: 1 = standard single output per task;
                                 >1 = deep supervision with auxiliary heads at
                                 coarser resolutions (max 4), applied identically
                                 to every task.
    """

    def __init__(
        self,
        in_channels: int = 1,
        task_num_classes: Dict[str, int] = None,
        base_features: int = 32,
        trilinear: bool = True,
        num_supervision_levels: int = 1,
    ) -> None:
        super().__init__()
        if not task_num_classes:
            raise ValueError(
                "task_num_classes must be a non-empty {task_name: num_classes} mapping"
            )

        f = base_features
        factor = 2 if trilinear else 1
        self.task_names: List[str] = list(task_num_classes.keys())

        self.encoder = UNet3DEncoder(in_channels, f)
        self.bottleneck = UNet3DBottleneck(f * 8, f * 16 // factor)
        self.decoder = UNet3DDecoder(f, trilinear)

        self.train_heads = nn.ModuleDict(
            {
                name: TrainHead(f, nc, trilinear, num_supervision_levels)
                for name, nc in task_num_classes.items()
            }
        )
        self.eval_heads = nn.ModuleDict(
            {name: EvalHead(nc) for name, nc in task_num_classes.items()}
        )
        self.infer_heads = nn.ModuleDict(
            {name: InferHead() for name in task_num_classes}
        )

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Main logits per task at finest scale, each [B, C_t, D, H, W]."""
        feats = self._backbone(x)
        return {name: head(feats)[0] for name, head in self.train_heads.items()}

    def forward_train(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """One logit list per task (finest first), for deep-supervision losses."""
        feats = self._backbone(x)
        return {name: head(feats) for name, head in self.train_heads.items()}

    def forward_eval(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Per task: (labelmap [B,D,H,W], uncertainty [B,D,H,W])."""
        feats = self._backbone(x)
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, head in self.train_heads.items():
            logits = head(feats)[0]
            out[name] = self.eval_heads[name](logits)
        return out

    def forward_infer(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Per task: argmax label-map [B, D, H, W]."""
        feats = self._backbone(x)
        out: Dict[str, torch.Tensor] = {}
        for name, head in self.train_heads.items():
            logits = head(feats)[0]
            out[name] = self.infer_heads[name](logits)
        return out
