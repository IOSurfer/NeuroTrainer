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


def _eikonal_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the Eikonal loss for a batch of SDF predictions.

    Args:
        pred: ``[B, C, D, H, W]`` predicted SDF values (raw float32).

    Returns:
        Scalar loss  ``mean( |‖∇pred‖ - 1| )``.
    """
    # Forward finite differences along each spatial axis
    gd = pred[:, :, 1:, :,  :] - pred[:, :, :-1, :,  :]   # [B, C, D-1, H,   W  ]
    gh = pred[:, :, :,  1:, :] - pred[:, :, :,  :-1, :]   # [B, C, D,   H-1, W  ]
    gw = pred[:, :, :,  :,  1:] - pred[:, :, :,  :, :-1]  # [B, C, D,   H,   W-1]

    # Trim to the common inner region [B, C, D-1, H-1, W-1]
    D_inner = gd.shape[2]   # D - 1
    H_inner = gh.shape[3]   # H - 1
    W_inner = gw.shape[4]   # W - 1

    gd = gd[:, :,        :, :H_inner, :W_inner]
    gh = gh[:, :, :D_inner, :,        :W_inner]
    gw = gw[:, :, :D_inner, :H_inner, :]

    grad_mag = torch.sqrt(gd ** 2 + gh ** 2 + gw ** 2 + 1e-8)
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
