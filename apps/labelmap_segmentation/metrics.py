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
    Per-class Dice scores (background excluded from mean).

    logits:  [B, C, D, H, W] raw model output
    targets: [B, 1, D, H, W] integer class indices
    Returns: {'dice_class_<c>': ..., 'mean_dice': ...}
    """
    if num_classes == 1:
        preds = (torch.sigmoid(logits) > threshold).float()
        t = targets.float()
        inter = (preds * t).sum()
        score = ((2.0 * inter + smooth) / (preds.sum() + t.sum() + smooth)).item()
        return {"dice_class_1": score, "mean_dice": score}

    preds = logits.argmax(dim=1)
    t = targets.squeeze(1).long()
    scores: Dict[str, float] = {}
    total = 0.0

    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        t_c = (t == c).float()
        inter = (pred_c * t_c).sum()
        dice = ((2.0 * inter + smooth) / (pred_c.sum() + t_c.sum() + smooth)).item()
        scores[f"dice_class_{c}"] = dice
        total += dice

    scores["mean_dice"] = total / (num_classes - 1)
    return scores


@torch.no_grad()
def compute_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-class IoU (background excluded from mean)."""
    if num_classes == 1:
        preds = (torch.sigmoid(logits) > threshold).float()
        t = targets.float()
        inter = (preds * t).sum()
        union = preds.sum() + t.sum() - inter
        score = ((inter + smooth) / (union + smooth)).item()
        return {"iou_class_1": score, "mean_iou": score}

    preds = logits.argmax(dim=1)
    t = targets.squeeze(1).long()
    scores: Dict[str, float] = {}
    total = 0.0

    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        t_c = (t == c).float()
        inter = (pred_c * t_c).sum()
        union = pred_c.sum() + t_c.sum() - inter
        iou = ((inter + smooth) / (union + smooth)).item()
        scores[f"iou_class_{c}"] = iou
        total += iou

    scores["mean_iou"] = total / (num_classes - 1)
    return scores
