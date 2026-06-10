"""
Loss functions for the SDF Estimation application.

SDFLoss
-------
Combined loss with multiple terms, each independently weighted via
``LossConfig``:

1. **MSE / Smooth-L1** -- pixel-wise reconstruction error between the
   predicted SDF and the ground-truth SDF.

2. **Eikonal** -- penalises deviations of the predicted gradient magnitude
   from unity ( |Nabla SDF| = 1 ), the fundamental property of a signed
   distance field.

3. **Normal** -- cosine-similarity term encouraging the predicted gradient
   direction to match the ground-truth gradient direction.

4. **Level-set overlap** -- soft Dice overlap between the zero level-sets of
   the predicted and ground-truth SDFs (Heaviside-relaxed).

5. **Boundary** -- reconstruction error weighted by a Gaussian centred on the
   zero level-set, focusing supervision near the surface.

Gradients are approximated with central finite differences along each of the
three spatial axes (replicate padding) and computed once, shared between the
Eikonal and Normal terms.

All terms are averaged over all SDF channels so they scale naturally when
num_sdf_fields > 1.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _eikonal_loss(g_pred: torch.Tensor) -> torch.Tensor:
    grad_norm = torch.sqrt((g_pred ** 2).sum(dim=0) + 1e-6)
    return ((grad_norm - 1.0) ** 2).mean()


def _normal_loss(g_pred: torch.Tensor, g_gt: torch.Tensor) -> torch.Tensor:
    dot = (g_pred * g_gt).sum(dim=0)
    pred_norm = torch.sqrt((g_pred * g_pred).sum(dim=0) + 1e-6)
    gt_norm = torch.sqrt((g_gt * g_gt).sum(dim=0) + 1e-6)
    cos = dot / (pred_norm * gt_norm + 1e-6)
    return (1.0 - cos).mean()


def _heaviside(phi: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(alpha * phi))


def _levelset_overlap_loss(
    pred_sdf: torch.Tensor, gt_sdf: torch.Tensor, alpha: float = 10.0, eps: float = 1e-6
) -> torch.Tensor:
    p = _heaviside(pred_sdf, alpha)
    g = _heaviside(gt_sdf, alpha)

    intersection = (p * g).sum()
    union = p.sum() + g.sum()

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice


def _boundary_weight(phi: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(-(phi ** 2) / (2 * sigma ** 2))


def _soft_boundary_loss(pred_sdf: torch.Tensor, gt_sdf: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    w = _boundary_weight(gt_sdf, sigma)
    return (w * (pred_sdf - gt_sdf) ** 2).mean()


def _mse_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, gt, beta=1.0)


class SDFLoss(nn.Module):
    """
    Combined multi-term loss for multi-field SDF estimation.

    Args:
        mse_weight:      Weight of the MSE / Smooth-L1 reconstruction term.
        eikonal_weight:  Weight of the Eikonal constraint.
        normal_weight:   Weight of the gradient-direction (normal) consistency term.
        overlap_weight:  Weight of the soft level-set Dice overlap term.
        boundary_weight: Weight of the boundary-focused reconstruction term.
        boundary_sigma:  Gaussian width (in voxels) for the boundary weighting.
        levelset_alpha:  Heaviside steepness for the level-set overlap term.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        eikonal_weight: float = 0.1,
        normal_weight: float = 0.0,
        overlap_weight: float = 0.0,
        boundary_weight: float = 0.0,
        boundary_sigma: float = 1.0,
        levelset_alpha: float = 10.0,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.eikonal_weight = eikonal_weight
        self.normal_weight = normal_weight
        self.overlap_weight = overlap_weight
        self.boundary_weight = boundary_weight
        self.boundary_sigma = boundary_sigma
        self.levelset_alpha = levelset_alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            pred:   ``[B, num_sdf_fields, D, H, W]`` raw model output.
            target: ``[B, num_sdf_fields, D, H, W]`` ground-truth SDF (float32).

        Returns:
            ``(total_loss, {'mse', 'eikonal', 'normal', 'overlap', 'boundary'})``
            Component tensors are detached scalars suitable for logging.
        """
        device = pred.device
        dtype = pred.dtype
        C = pred.shape[1]

        # Central-difference kernels for d/dD, d/dH, d/dW
        kernel_d = torch.zeros((C, 1, 3, 1, 1), device=device, dtype=dtype)
        kernel_d[:, 0, :, 0, 0] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        kernel_h = torch.zeros((C, 1, 1, 3, 1), device=device, dtype=dtype)
        kernel_h[:, 0, 0, :, 0] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        kernel_w = torch.zeros((C, 1, 1, 1, 3), device=device, dtype=dtype)
        kernel_w[:, 0, 0, 0, :] = torch.tensor([-0.5, 0.0, 0.5], device=device, dtype=dtype)

        # Gradients of prediction (replicate padding)
        pred_pad_d = F.pad(pred, (0, 0, 0, 0, 1, 1), mode="replicate")
        pred_pad_h = F.pad(pred, (0, 0, 1, 1, 0, 0), mode="replicate")
        pred_pad_w = F.pad(pred, (1, 1, 0, 0, 0, 0), mode="replicate")

        gd_pred = F.conv3d(pred_pad_d, kernel_d, groups=C)
        gh_pred = F.conv3d(pred_pad_h, kernel_h, groups=C)
        gw_pred = F.conv3d(pred_pad_w, kernel_w, groups=C)

        g_pred = torch.stack([gd_pred, gh_pred, gw_pred], dim=0)

        # Gradients of ground truth (replicate padding)
        gt_pad_d = F.pad(target, (0, 0, 0, 0, 1, 1), mode="replicate")
        gt_pad_h = F.pad(target, (0, 0, 1, 1, 0, 0), mode="replicate")
        gt_pad_w = F.pad(target, (1, 1, 0, 0, 0, 0), mode="replicate")

        gd_gt = F.conv3d(gt_pad_d, kernel_d, groups=C)
        gh_gt = F.conv3d(gt_pad_h, kernel_h, groups=C)
        gw_gt = F.conv3d(gt_pad_w, kernel_w, groups=C)

        g_gt = torch.stack([gd_gt, gh_gt, gw_gt], dim=0)

        loss_mse = _mse_loss(pred, target)
        loss_eik = _eikonal_loss(g_pred)
        loss_normal = _normal_loss(g_pred, g_gt)
        loss_overlap = _levelset_overlap_loss(pred, target, alpha=self.levelset_alpha)
        loss_boundary = _soft_boundary_loss(pred, target, sigma=self.boundary_sigma)

        total = (
            self.mse_weight * loss_mse
            + self.eikonal_weight * loss_eik
            + self.normal_weight * loss_normal
            + self.overlap_weight * loss_overlap
            + self.boundary_weight * loss_boundary
        )
        return total, {
            'mse':      loss_mse.detach(),
            'eikonal':  loss_eik.detach(),
            'normal':   loss_normal.detach(),
            'overlap':  loss_overlap.detach(),
            'boundary': loss_boundary.detach(),
        }
