from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute per-class Dice scores (background excluded from mean).

    Args:
        logits:      [B, C, D, H, W] raw model output
        targets:     [B, 1, D, H, W] integer class indices
        num_classes: total number of classes (including background)

    Returns:
        dict with keys 'dice_class_<c>' for c >= 1, and 'mean_dice'
    """
    if num_classes == 1:
        preds = (torch.sigmoid(logits) > threshold).float()
        t = targets.float()
        inter = (preds * t).sum()
        dice = (2.0 * inter + smooth) / (preds.sum() + t.sum() + smooth)
        score = dice.item()
        return {'dice_class_1': score, 'mean_dice': score}

    preds = logits.argmax(dim=1)          # [B, D, H, W]
    t = targets.squeeze(1).long()         # [B, D, H, W]

    scores: Dict[str, float] = {}
    total = 0.0

    for c in range(1, num_classes):       # skip background
        pred_c = (preds == c).float()
        t_c = (t == c).float()
        inter = (pred_c * t_c).sum()
        dice = (2.0 * inter + smooth) / (pred_c.sum() + t_c.sum() + smooth)
        scores[f'dice_class_{c}'] = dice.item()
        total += dice.item()

    scores['mean_dice'] = total / (num_classes - 1)
    return scores


@torch.no_grad()
def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute per-class Intersection-over-Union (background excluded)."""
    if num_classes == 1:
        preds = (torch.sigmoid(logits) > threshold).float()
        t = targets.float()
        inter = (preds * t).sum()
        union = preds.sum() + t.sum() - inter
        iou = (inter + smooth) / (union + smooth)
        score = iou.item()
        return {'iou_class_1': score, 'mean_iou': score}

    preds = logits.argmax(dim=1)
    t = targets.squeeze(1).long()

    scores: Dict[str, float] = {}
    total = 0.0

    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        t_c = (t == c).float()
        inter = (pred_c * t_c).sum()
        union = pred_c.sum() + t_c.sum() - inter
        iou = (inter + smooth) / (union + smooth)
        scores[f'iou_class_{c}'] = iou.item()
        total += iou.item()

    scores['mean_iou'] = total / (num_classes - 1)
    return scores
