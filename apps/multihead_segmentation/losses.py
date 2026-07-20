"""
Multi-task loss for the Multi-Head Segmentation application.

Wraps one independent per-task criterion -- DiceLoss / DiceCELoss reused
unchanged from apps.labelmap_segmentation.losses -- and combines them into a
single scalar via a weighted sum (LossConfig.task_weights, default: equal
weight 1.0 per task).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from apps.labelmap_segmentation.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss


class MultiHeadLoss(nn.Module):
    """
    Args:
        task_num_classes: Mapping task name -> num_classes.
        loss_type:        dice | dice_ce | ce | focal | dice_focal -- applied identically to every task.
        dice_weight:      Dice term weight, used when loss_type == 'dice_ce' / 'dice_focal'.
        ce_weight:        Cross-entropy term weight, used when loss_type == 'dice_ce'.
        focal_gamma:      Focal loss γ, used when loss_type == 'focal' / 'dice_focal'.
        focal_weight:     Focal term weight, used when loss_type == 'dice_focal'.
        alpha:            Per-class weights [w0, w1, ...] for ce / focal / dice_focal (None = uniform).
        task_weights:     Mapping task name -> weight in the cross-task sum
                          (None = equal weight 1.0 for every task).
    """

    def __init__(
        self,
        task_num_classes: Dict[str, int],
        loss_type: str = "dice_ce",
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        alpha: Optional[list] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.task_weights = task_weights or {name: 1.0 for name in task_num_classes}

        def _make(nc: int) -> nn.Module:
            if loss_type == "dice":
                return DiceLoss(nc)
            if loss_type == "focal":
                return FocalLoss(nc, gamma=focal_gamma, alpha=alpha)
            if loss_type == "dice_focal":
                return DiceFocalLoss(nc, dice_weight, focal_weight, focal_gamma, alpha)
            if loss_type == "ce":
                if nc == 1:
                    return nn.BCEWithLogitsLoss()
                if alpha:
                    return nn.CrossEntropyLoss(weight=torch.tensor(alpha, dtype=torch.float32))
                return nn.CrossEntropyLoss()
            return DiceCELoss(nc, dice_weight, ce_weight)

        self.criteria = nn.ModuleDict(
            {name: _make(nc) for name, nc in task_num_classes.items()}
        )

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits:  {task_name: [B, C_t, D, H, W] raw model output}
            targets: {task_name: [B, 1, D, H, W] integer class indices}

        Returns:
            (total_loss, {'<task>_dice': ..., '<task>_ce': ...})
            Component tensors are detached scalars suitable for logging.
        """
        total: torch.Tensor = 0.0
        components: Dict[str, torch.Tensor] = {}

        for name, crit in self.criteria.items():
            lg, tgt = logits[name], targets[name]

            if self.loss_type == "ce":
                loss = (
                    crit(lg, tgt.float())
                    if lg.size(1) == 1
                    else crit(lg, tgt.squeeze(1).long())
                )
                comp = {"ce": loss.detach()}
            elif self.loss_type == "dice":
                loss = crit(lg, tgt)
                comp = {"dice": loss.detach()}
            elif self.loss_type == "focal":
                loss = crit(lg, tgt)
                comp = {"focal": loss.detach()}
            else:  # dice_ce / dice_focal
                loss, comp = crit(lg, tgt)

            total = total + self.task_weights.get(name, 1.0) * loss
            for k, v in comp.items():
                components[f"{name}_{k}"] = v

        return total, components
