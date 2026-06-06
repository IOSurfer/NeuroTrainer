"""
Evaluation metrics for the SDF Estimation application.

Both functions are ``@torch.no_grad()`` and return plain Python dicts so they
can be logged directly to TensorBoard or written to JSON.

Metrics per field
-----------------
- ``mae_field_<i>``:  mean absolute error for the i-th SDF channel.
- ``mse_field_<i>``:  mean squared error for the i-th SDF channel.
- ``mean_mae``:       average MAE across all SDF channels.
- ``mean_mse``:       average MSE across all SDF channels.
"""

from typing import Dict

import torch


@torch.no_grad()
def compute_sdf_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Per-field MAE and MSE plus their cross-field means.

    Args:
        pred:   ``[B, num_sdf_fields, D, H, W]`` model output (float32).
        target: ``[B, num_sdf_fields, D, H, W]`` ground-truth SDF (float32).

    Returns:
        Dict with keys ``mae_field_<i>``, ``mse_field_<i>``,
        ``mean_mae``, ``mean_mse``.
    """
    num_fields = pred.shape[1]
    scores: Dict[str, float] = {}
    total_mae = 0.0
    total_mse = 0.0

    for i in range(num_fields):
        diff = pred[:, i] - target[:, i]                    # [B, D, H, W]
        mae = diff.abs().mean().item()
        mse = (diff ** 2).mean().item()
        scores[f'mae_field_{i}'] = mae
        scores[f'mse_field_{i}'] = mse
        total_mae += mae
        total_mse += mse

    scores['mean_mae'] = total_mae / num_fields
    scores['mean_mse'] = total_mse / num_fields
    return scores
