"""
Run inference and evaluation on the test split for an SDF Estimation experiment.

Usage
-----
    python -m apps.sdf_estimation.test ^
        --config_dir output/exp_001 ^
        --checkpoint output/exp_001/checkpoints/best.pth ^
        --output_dir output/exp_001/test_eval

Outputs (under --output_dir)
-----------------------------
    predictions/<subject_id>_<sdf_name>.nii.gz   -- predicted SDF field (float32)
    metrics.json                                  -- per-subject + aggregate MAE / MSE
    model_profile.json                            -- total/per-layer parameter counts and GFLOPs

Notes
-----
* EMA weights are applied automatically when present in the checkpoint and
  TrainingConfig.ema is True.
* SDF estimation is always full-volume -- there is no patch-based inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchio as tio

from configuration.manager import ConfigManager
from apps.sdf_estimation.dataset import create_sdf_datasets
from apps.sdf_estimation.metrics import compute_sdf_metrics
from apps.sdf_estimation.model import UNet3DSDF
from apps.sdf_estimation.sdf_config import (
    AugmentConfig,
    DataConfig,
    InfraConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    UNet3DConfig,
)

# ── Config loader ──────────────────────────────────────────────────────────────


def _load_configs(config_dir: str) -> ConfigManager:
    """Populate a fresh ConfigManager from the JSON files in *config_dir*."""
    m = ConfigManager.get()
    base = Path(config_dir)
    type_map = {
        ConfigManager.DATA: DataConfig,
        ConfigManager.AUGMENT: AugmentConfig,
        ConfigManager.MODEL: UNet3DConfig,
        ConfigManager.LOSS: LossConfig,
        ConfigManager.OPTIMIZER: OptimizerConfig,
        ConfigManager.SCHEDULER: SchedulerConfig,
        ConfigManager.TRAINING: TrainingConfig,
        ConfigManager.INFRA: InfraConfig,
    }
    for key, cls in type_map.items():
        cfg = cls()
        json_path = base / f"{key.lower()}.json"
        if json_path.exists():
            cfg.load(str(json_path))
        else:
            print(f"[test] {json_path.name} not found -- using defaults")
        m.register(key, cfg)
    return m


# ── EMA ────────────────────────────────────────────────────────────────────────


def _apply_ema(model: UNet3DSDF, ckpt: dict, device: torch.device) -> bool:
    """Overwrite model parameters with EMA shadow weights from *ckpt*."""
    shadow = ckpt.get("ema", {}).get("shadow", {})
    if not shadow:
        return False
    updated = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in shadow:
            param.data.copy_(shadow[name].to(device))
            updated += 1
    return updated > 0


# ── Model profiling: parameters + FLOPs ──────────────────────────────────────────


def _count_parameters(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    """Returns (total_params, {leaf_module_name: params})."""
    total = sum(p.numel() for p in model.parameters())
    per_layer: Dict[str, int] = {}
    for name, module in model.named_modules():
        if list(module.children()):
            continue  # not a leaf
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            per_layer[name] = n
    return total, per_layer


def _count_flops(
    model: nn.Module, input_shape: Tuple[int, ...], device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_gflops, {leaf_module_name: gflops}) for a single forward pass.

    Only Conv3d / ConvTranspose3d layers are counted (they dominate a 3D U-Net's
    compute budget); norm / activation / upsample layers contribute negligibly.
    """
    flops: Dict[str, float] = {}
    hooks = []

    def _hook(name: str, module: nn.Module):
        def fn(_module, inputs, output):
            k = module.kernel_size
            kernel_vol = k[0] * k[1] * k[2]
            if isinstance(module, nn.ConvTranspose3d):
                elems = inputs[0].numel()
                channels = module.out_channels // module.groups
            else:
                elems = output.numel()
                channels = module.in_channels // module.groups
            flops[name] = flops.get(name, 0.0) + 2.0 * elems * kernel_vol * channels

        return fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            hooks.append(module.register_forward_hook(_hook(name, module)))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))

    for h in hooks:
        h.remove()

    total = sum(flops.values())
    return total / 1e9, {k: v / 1e9 for k, v in flops.items()}


# ── Shape helper ──────────────────────────────────────────────────────────────


def _input_shape(
    data_cfg: DataConfig,
    model_cfg: UNet3DConfig,
    override: Optional[Tuple[int, int, int]],
) -> Tuple[int, ...]:
    c = model_cfg.encoder.in_channels
    if override:
        return (1, c, *override)
    if data_cfg.target_shape:
        return (1, c, *data_cfg.target_shape)
    print("[test] No spatial shape found in config -- defaulting to 128x128x128")
    return (1, c, 128, 128, 128)


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict_subject(
    model: UNet3DSDF,
    subject: tio.Subject,
    data_cfg: DataConfig,
    device: torch.device,
) -> torch.Tensor:
    """Returns raw SDF predictions ``[1, num_sdf_fields, D, H, W]``."""
    modalities = data_cfg.modalities
    imgs = (
        torch.cat([subject[m][tio.DATA] for m in modalities], dim=0)
        .unsqueeze(0)
        .to(device)
    )
    return model(imgs)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ── Main evaluation pipeline ────────────────────────────────────────────────────


def run_test(
    config_dir: str,
    checkpoint: str,
    output_dir: str,
    device: str = "auto",
    save_predictions: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    data_root: Optional[str] = None,
) -> None:
    print(f"[test] Config dir : {config_dir}")
    print(f"[test] Checkpoint : {checkpoint}")

    m = _load_configs(config_dir)
    data_cfg: DataConfig = m.get_config(ConfigManager.DATA)
    model_cfg: UNet3DConfig = m.get_config(ConfigManager.MODEL)
    train_cfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)

    if data_root:
        data_cfg.data_root = data_root

    dev = _resolve_device(device)
    print(f"[test] Device     : {dev}")

    out_dir = Path(output_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("[test] Building test dataset...")
    _, _, test_ds = create_sdf_datasets()
    print(f"[test] Test subjects: {len(test_ds)}")
    if len(test_ds) == 0:
        sys.exit("[test] ERROR: test dataset is empty")

    sdf_names: List[str] = list(data_cfg.sdf_names)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = UNet3DSDF(
        in_channels=model_cfg.encoder.in_channels,
        num_sdf_fields=data_cfg.num_sdf_fields,
        base_features=model_cfg.encoder.base_features,
        trilinear=model_cfg.decoder.trilinear,
    ).to(dev)

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"[test] ERROR: checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f'[test] Checkpoint epoch: {ckpt.get("epoch", "?")}')

    if train_cfg.ema and "ema" in ckpt:
        applied = _apply_ema(model, ckpt, dev)
        print(
            f'[test] EMA: {"applied" if applied else "shadow dict empty -- raw weights kept"}'
        )
    elif "ema" in ckpt and not train_cfg.ema:
        print(
            "[test] EMA state found in checkpoint but TrainingConfig.ema=False -- skipped"
        )

    model.eval()

    # ── Model profile: parameters + FLOPs ─────────────────────────────────────
    total_params, per_layer_params = _count_parameters(model)
    shape = _input_shape(data_cfg, model_cfg, input_shape)
    total_gflops, per_layer_gflops = _count_flops(model, shape, dev)

    layer_names = sorted(set(per_layer_params) | set(per_layer_gflops))
    profile = {
        "total_params": total_params,
        "total_gflops": total_gflops,
        "flops_input_shape": list(shape),
        "layers": {
            name: {
                "params": per_layer_params.get(name, 0),
                "gflops": per_layer_gflops.get(name, 0.0),
            }
            for name in layer_names
        },
    }
    (out_dir / "model_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    print(f"[test] Total params: {total_params:,}")
    print(f"[test] Total GFLOPs (input {list(shape)}): {total_gflops:.2f}")

    # ── Inference + per-subject metrics ───────────────────────────────────────
    per_subject = []
    for subject in test_ds:
        subject_id = getattr(subject, "subject_id", f"sub_{len(per_subject)}")

        pred = _predict_subject(model, subject, data_cfg, dev)
        target = (
            torch.cat([subject[s][tio.DATA] for s in sdf_names], dim=0)
            .unsqueeze(0)
            .float()
            .to(dev)
        )

        result = {"subject": subject_id, **compute_sdf_metrics(pred, target)}
        per_subject.append(result)

        print(
            f'  {subject_id}: mean_mae {result["mean_mae"]:.4f}  '
            f'mean_mse {result["mean_mse"]:.4f}'
        )

        if save_predictions:
            ref = subject[data_cfg.modalities[0]]
            for i, sdf_name in enumerate(sdf_names):
                pred_img = tio.ScalarImage(
                    tensor=pred[0, i : i + 1].cpu().float(), affine=ref.affine
                )
                pred_img.save(str(pred_dir / f"{subject_id}_{sdf_name}.nii.gz"))

    # ── Aggregate ───────────────────────────────────────────────────────────────
    keys = [k for k in per_subject[0].keys() if k != "subject"]
    aggregate = {}
    for k in keys:
        vals = np.array([s[k] for s in per_subject], dtype=np.float64)
        aggregate[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    metrics = {"per_subject": per_subject, "aggregate": aggregate}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(
        f'[test] mean MAE: {aggregate["mean_mae"]["mean"]:.4f} ± {aggregate["mean_mae"]["std"]:.4f}  '
        f'mean MSE: {aggregate["mean_mse"]["mean"]:.4f} ± {aggregate["mean_mse"]["std"]:.4f}'
    )
    print(f"[test] Results written to {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference + evaluation on the test split for an SDF Estimation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config_dir",
        required=True,
        help="Experiment directory containing the *.json config files "
        "(e.g. output/exp_001/)",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth checkpoint (e.g. output/exp_001/checkpoints/best.pth)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <config_dir>/test_eval)",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Skip writing predicted SDF fields to disk (metrics are still computed)",
    )
    p.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Spatial shape used for the FLOPs dummy input. "
        "Defaults to DataConfig.target_shape, then 128x128x128.",
    )
    p.add_argument(
        "--data_root",
        default=None,
        help="Override DataConfig.data_root (use if the experiment was trained on a different machine)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or str(Path(args.config_dir) / "test_eval")
    run_test(
        config_dir=args.config_dir,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        device=args.device,
        save_predictions=not args.no_save_predictions,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
