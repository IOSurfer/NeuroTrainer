from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss, optionally ignoring the background class."""

    def __init__(
        self, num_classes: int, smooth: float = 1e-5, ignore_background: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, D, H, W] raw model output
        targets: [B, 1, D, H, W] integer class indices
        """
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            t = targets.float()
            inter = (probs * t).sum()
            return 1.0 - (2.0 * inter + self.smooth) / (
                probs.sum() + t.sum() + self.smooth
            )

        probs = F.softmax(logits, dim=1)
        t_long = targets.squeeze(1).long()
        t_max = int(t_long.max().item())
        if t_max >= self.num_classes:
            raise ValueError(
                f"DiceLoss: label value {t_max} is out of range for "
                f"num_classes={self.num_classes} (valid: 0–{self.num_classes - 1}). "
                f"Unique values in this batch: {t_long.unique().cpu().tolist()}"
            )
        t_oh = F.one_hot(t_long, self.num_classes).permute(0, 4, 1, 2, 3).float()

        # Reduce over batch + spatial dims at once, keeping the class dim --
        # avoids a Python loop (and per-class kernel launches) over classes.
        dims = (0,) + tuple(range(2, probs.dim()))
        inter = (probs * t_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + t_oh.sum(dim=dims)
        dice_per_class = (2.0 * inter + self.smooth) / (union + self.smooth)

        start = 1 if self.ignore_background else 0
        return (1.0 - dice_per_class[start:]).mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017).

    For ``num_classes == 1`` uses binary sigmoid focal loss; otherwise
    softmax focal loss with optional per-class alpha weighting.
    When ``ignore_background=True`` (default) background voxels are excluded
    from the mean for the multi-class case.
    """

    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_background = ignore_background
        weight = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.register_buffer("alpha", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            t = targets.float()
            p_t = probs * t + (1.0 - probs) * (1.0 - t)
            focal_w = (1.0 - p_t) ** self.gamma
            return (focal_w * F.binary_cross_entropy_with_logits(logits, t, reduction="none")).mean()

        t_long = targets.squeeze(1).long()
        t_max = int(t_long.max().item())
        if t_max >= self.num_classes:
            raise ValueError(
                f"FocalLoss: 标签值 {t_max} 超出范围，num_classes={self.num_classes}（合法范围 0–{self.num_classes - 1}）。"
                f"本批次唯一标签值：{t_long.unique().cpu().tolist()}"
            )
        log_probs = F.log_softmax(logits, dim=1)
        # p_t: probability of the true class at each voxel [B, D, H, W]
        p_t = log_probs.exp().gather(dim=1, index=t_long.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - p_t) ** self.gamma
        # alpha-weighted CE per voxel [B, D, H, W]
        ce = F.nll_loss(log_probs, t_long, weight=self.alpha, reduction="none")  # type: ignore[arg-type]
        if self.ignore_background:
            fg = (t_long != 0).float()
            return (focal_w * ce * fg).sum() / (fg.sum() + 1e-5)
        return (focal_w * ce).mean()


class DiceFocalLoss(nn.Module):
    """Weighted combination of Dice loss and Focal loss."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        smooth: float = 1e-5,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(num_classes, smooth, ignore_background)
        self.focal = FocalLoss(num_classes, gamma, alpha, ignore_background)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)
        total = self.dice_weight * loss_dice + self.focal_weight * loss_focal
        return total, {"dice": loss_dice.detach(), "focal": loss_focal.detach()}


class DiceCELoss(nn.Module):
    """Weighted combination of Dice loss and Cross-Entropy loss."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        ignore_background: bool = True,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes, smooth, ignore_background)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self.ce = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns (total_loss, {'dice': dice_term, 'ce': ce_term}).
        Both component tensors are detached scalars suitable for logging.
        """
        loss_dice = self.dice(logits, targets)
        loss_ce = (
            self.ce(logits, targets.float())
            if self.num_classes == 1
            else self.ce(logits, targets.squeeze(1).long())
        )
        total = self.dice_weight * loss_dice + self.ce_weight * loss_ce
        return total, {"dice": loss_dice.detach(), "ce": loss_ce.detach()}
