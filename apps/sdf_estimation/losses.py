"""
Loss functions for the SDF Estimation application.

SDFLoss
-------
Combined loss with multiple terms, each independently weighted via
``LossConfig``:

1. **Reconstruction** -- Smooth-L1 error between the predicted SDF and the
   ground-truth SDF, boosted near the zero level-set by a Gaussian boundary
   weight (a single term covering both the global and boundary-focused
   reconstruction signal).

2. **Eikonal** -- penalises deviations of the predicted gradient magnitude
   from unity ( |Nabla SDF| = 1 ), the fundamental property of a signed
   distance field.

3. **Normal** -- cosine-similarity term encouraging the predicted gradient
   direction to match the ground-truth gradient direction, weighted by the
   same Gaussian boundary weight as the reconstruction term (gradient
   direction is most meaningful near the zero level-set).

Gradients are approximated with central finite differences along each of the
three spatial axes (replicate padding) and computed once, shared between the
Eikonal and Normal terms.

The boundary weight is computed once from the ground-truth SDF and shared
between the reconstruction and normal terms.

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


def _boundary_weight(phi: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(-(phi ** 2) / (2 * sigma ** 2))


def _weighted_recon_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    per_voxel = F.smooth_l1_loss(pred, target, beta=1.0, reduction='none')
    return ((1.0 + weight) * per_voxel).mean()


def _weighted_normal_loss(
    g_pred: torch.Tensor, g_gt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    dot = (g_pred * g_gt).sum(dim=0)
    pred_norm = torch.sqrt((g_pred * g_pred).sum(dim=0) + 1e-6)
    gt_norm = torch.sqrt((g_gt * g_gt).sum(dim=0) + 1e-6)
    cos = dot / (pred_norm * gt_norm + 1e-6)
    return (weight * (1.0 - cos)).mean()


class SDFLoss(nn.Module):
    """
    Combined multi-term loss for multi-field SDF estimation.

    Args:
        recon_weight:    Weight of the combined Smooth-L1 reconstruction term
                          (boosted near the zero level-set).
        eikonal_weight:  Weight of the Eikonal constraint.
        normal_weight:   Weight of the gradient-direction (normal) consistency
                          term (weighted near the zero level-set).
        boundary_sigma:  Gaussian width (in voxels) for the boundary weighting,
                          shared by the reconstruction and normal terms.
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        eikonal_weight: float = 0.1,
        normal_weight: float = 0.0,
        boundary_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.eikonal_weight = eikonal_weight
        self.normal_weight = normal_weight
        self.boundary_sigma = boundary_sigma

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
            ``(total_loss, {'recon', 'eikonal', 'normal'})``
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

        # Boundary weight, precomputed once and shared between the
        # reconstruction and normal terms
        boundary_w = _boundary_weight(target, self.boundary_sigma)

        loss_recon = _weighted_recon_loss(pred, target, boundary_w)
        loss_eik = _eikonal_loss(g_pred)
        loss_normal = _weighted_normal_loss(g_pred, g_gt, boundary_w)

        total = (
            self.recon_weight * loss_recon
            + self.eikonal_weight * loss_eik
            + self.normal_weight * loss_normal
        )
        return total, {
            'recon':   loss_recon.detach(),
            'eikonal': loss_eik.detach(),
            'normal':  loss_normal.detach(),
        }
