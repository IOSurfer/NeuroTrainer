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

    # One-hot + reduce over batch + spatial dims at once, then a single
    # CPU transfer -- avoids a Python loop with one .item() sync per class.
    pred_oh = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3).float()
    t_oh = F.one_hot(t, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0,) + tuple(range(2, pred_oh.dim()))
    inter = (pred_oh * t_oh).sum(dim=dims)
    union = pred_oh.sum(dim=dims) + t_oh.sum(dim=dims)
    dice_per_class = ((2.0 * inter + smooth) / (union + smooth)).cpu().numpy()

    scores: Dict[str, float] = {
        f"dice_class_{c}": float(dice_per_class[c]) for c in range(1, num_classes)
    }
    scores["mean_dice"] = float(dice_per_class[1:].mean())
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

    pred_oh = F.one_hot(preds, num_classes).permute(0, 4, 1, 2, 3).float()
    t_oh = F.one_hot(t, num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0,) + tuple(range(2, pred_oh.dim()))
    inter = (pred_oh * t_oh).sum(dim=dims)
    union = pred_oh.sum(dim=dims) + t_oh.sum(dim=dims) - inter
    iou_per_class = ((inter + smooth) / (union + smooth)).cpu().numpy()

    scores: Dict[str, float] = {
        f"iou_class_{c}": float(iou_per_class[c]) for c in range(1, num_classes)
    }
    scores["mean_iou"] = float(iou_per_class[1:].mean())
    return scores
