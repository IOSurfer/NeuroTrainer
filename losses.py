import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss, optionally ignoring the background class."""

    def __init__(self, num_classes: int, smooth: float = 1e-5, ignore_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  [B, C, D, H, W] — raw (un-activated) model output
        targets: [B, 1, D, H, W] — integer class indices
        """
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
            t = targets.float()
            inter = (probs * t).sum()
            return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + t.sum() + self.smooth)

        probs = F.softmax(logits, dim=1)  # [B, C, D, H, W]
        t_long = targets.squeeze(1).long()  # [B, D, H, W]
        t_oh = F.one_hot(t_long, self.num_classes).permute(0, 4, 1, 2, 3).float()

        start = 1 if self.ignore_background else 0
        dice_losses = []
        for c in range(start, self.num_classes):
            p_c = probs[:, c]
            t_c = t_oh[:, c]
            inter = (p_c * t_c).sum()
            dice_c = (2.0 * inter + self.smooth) / (p_c.sum() + t_c.sum() + self.smooth)
            dice_losses.append(1.0 - dice_c)

        return torch.stack(dice_losses).mean()


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

        if num_classes == 1:
            self.ce = nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_dice = self.dice(logits, targets)
        if self.num_classes == 1:
            loss_ce = self.ce(logits, targets.float())
        else:
            loss_ce = self.ce(logits, targets.squeeze(1).long())
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce
