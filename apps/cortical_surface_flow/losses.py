"""
Training losses.

closed_loop   beta-NLL of the end-point Gaussian, error split at each vertex into the
              part along the target normal (1 dim, variance var_n) and the part in its
              tangent plane (2 dims, variance var_t):

                  nll = 0.5 (e_n^2 / var_n + log var_n) + 0.5 (|e_t|^2 / var_t + 2 log var_t)

              each term weighted by stop_grad(var^beta) (Seitzer et al. 2022), plus
              normal_weight * (1 - n_hat . n) for the predicted end-point normal.
              The tangential variance is learned per vertex: where along the surface a
              vertex lands is FreeSurfer's bookkeeping and partly invisible in the T1.

flow          flow matching on straight paths, velocity error split the same way:
                  L = mean(e_n^2 + tangential_weight * |e_t|^2) / 3
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def closed_loop_loss(mu, lv_n, lv_t, n_hat, p1, nrm, half_fov: float, beta: float,
                     normal_weight: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """All positions normalised; errors are evaluated in mm (half_fov) so the variances
    are in mm^2."""
    e = (mu - p1) * half_fov
    en = (e * nrm).sum(-1)
    et2 = ((e * e).sum(-1) - en ** 2).clamp(min=0)
    nll_n = 0.5 * (en ** 2 * torch.exp(-lv_n) + lv_n)
    nll_t = 0.5 * (et2 * torch.exp(-lv_t) + 2.0 * lv_t)
    if beta > 0:
        nll_n = nll_n * torch.exp(beta * lv_n).detach()
        nll_t = nll_t * torch.exp(beta * lv_t).detach()
    cos = (n_hat * nrm).sum(-1)
    loss = (nll_n + nll_t).mean() + normal_weight * (1.0 - cos).mean()
    with torch.no_grad():
        stats = {"endpoint_rmse_mm": (e * e).sum(-1).mean().sqrt(),
                 "endpoint_normal_rmse_mm": (en ** 2).mean().sqrt(),
                 "endpoint_tangential_rmse_mm": et2.mean().sqrt(),
                 "sigma_normal_mm": (0.5 * lv_n).exp().median(),
                 "sigma_tangential_mm": (0.5 * lv_t).exp().median(),
                 "normal_cos": cos.mean()}
    return loss, stats


def flow_loss(v, v_target, nrm, half_fov: float,
              tangential_weight: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    e = v - v_target
    e2 = (e * e).sum(-1)
    en2 = ((e * nrm).sum(-1)) ** 2
    loss = ((en2 + tangential_weight * (e2 - en2)) / 3.0).mean()
    with torch.no_grad():
        stats = {"velocity_rmse_mm": e2.mean().sqrt() * half_fov,
                 "velocity_normal_rmse_mm": en2.mean().sqrt() * half_fov}
    return loss, stats
