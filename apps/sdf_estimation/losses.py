"""
Loss functions for the SDF Estimation application.

SDFLoss
-------
Combined loss with two terms:

1. **MSE** -- pixel-wise mean-squared error between the predicted SDF and the
   ground-truth SDF.  Drives the network to recover accurate distance values
   everywhere.

2. **Eikonal** -- penalises deviations of the predicted gradient magnitude from
   unity ( ``‖∇SDF‖ = 1`` ).  This is the fundamental property of a signed
   distance field and helps the network produce geometrically valid outputs.

   Gradients are approximated with forward finite differences along each of the
   three spatial axes and the magnitude is computed over the inner voxel region
   where all three partial derivatives are defined.

Both terms are averaged over all SDF channels so they scale naturally when
``num_sdf_fields > 1``.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn.functional as F


def _eikonal_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the Eikonal loss for a batch of SDF predictions.

    Args:
        pred: ``[B, C, D, H, W]`` predicted SDF values (raw float32).

    Returns:
        Scalar loss  ``mean( |L2(\Delta pred) - 1| )``.
    """

    device = pred.device
    dtype = pred.dtype
    C = pred.shape[1]

    # d/dD
    kernel_d = torch.zeros((C, 1, 3, 1, 1), device=device, dtype=dtype)
    kernel_d[:, 0, :, 0, 0] = torch.tensor(
        [-0.5, 0.0, 0.5],
        device=device,
        dtype=dtype
    )

    # d/dH
    kernel_h = torch.zeros((C, 1, 1, 3, 1), device=device, dtype=dtype)
    kernel_h[:, 0, 0, :, 0] = torch.tensor(
        [-0.5, 0.0, 0.5],
        device=device,
        dtype=dtype
    )

    # d/dW
    kernel_w = torch.zeros((C, 1, 1, 1, 3), device=device, dtype=dtype)
    kernel_w[:, 0, 0, 0, :] = torch.tensor(
        [-0.5, 0.0, 0.5],
        device=device,
        dtype=dtype
    )

    # replicate padding
    pred_pad_d = F.pad(pred, (0, 0, 0, 0, 1, 1), mode="replicate")
    pred_pad_h = F.pad(pred, (0, 0, 1, 1, 0, 0), mode="replicate")
    pred_pad_w = F.pad(pred, (1, 1, 0, 0, 0, 0), mode="replicate")

    gd = F.conv3d(pred_pad_d, kernel_d, groups=C)
    gh = F.conv3d(pred_pad_h, kernel_h, groups=C)
    gw = F.conv3d(pred_pad_w, kernel_w, groups=C)

    grad_mag = torch.sqrt(
        gd * gd +
        gh * gh +
        gw * gw +
        1e-8
    )

    return (grad_mag - 1.0).abs().mean()

class SDFLoss(nn.Module):
    """
    Combined MSE + Eikonal loss for multi-field SDF estimation.

    Args:
        mse_weight:     Weight of the MSE reconstruction term.
        eikonal_weight: Weight of the Eikonal constraint.
    """

    def __init__(self, mse_weight: float = 1.0, eikonal_weight: float = 0.1) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.eikonal_weight = eikonal_weight

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
            ``(total_loss, {'mse': ..., 'eikonal': ...})``
            Component tensors are detached scalars suitable for logging.
        """
        loss_mse = F.mse_loss(pred, target)
        loss_eik = _eikonal_loss(pred)

        total = self.mse_weight * loss_mse + self.eikonal_weight * loss_eik
        return total, {
            'mse':     loss_mse.detach(),
            'eikonal': loss_eik.detach(),
        }
