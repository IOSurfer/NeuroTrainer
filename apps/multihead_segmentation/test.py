"""
Run inference and evaluation on the test split for a Multi-Head Segmentation experiment.

Usage
-----
    python -m apps.multihead_segmentation.test ^
        --config_dir output/exp_001 ^
        --checkpoint output/exp_001/checkpoints/best.pth ^
        --output_dir output/exp_001/test_eval

Outputs (under --output_dir)
-----------------------------
    predictions/<subject_id>_<task>.nii.gz   -- predicted label map per task (uint8)
    metrics.json                              -- per-subject + aggregate Dice / IoU / HD95, per task
    model_profile.json                        -- total/per-layer parameter counts and GFLOPs

Notes
-----
* EMA weights are applied automatically when present in the checkpoint and
  TrainingConfig.ema is True.
* Patch-based inference (GridSampler + one GridAggregator per task) is used
  automatically when PatchConfig.enabled is True.
* Generic helpers (_apply_ema, _count_parameters, _count_flops, _input_shape,
  _hausdorff_distance_95, _resolve_device) are reused unchanged from
  apps.labelmap_segmentation.test -- only the multi-task-aware pieces
  (_load_configs, _predict_subject, run_test) are new.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader

from configuration.manager import ConfigManager
from apps.labelmap_segmentation.metrics import compute_dice, compute_iou
from apps.labelmap_segmentation.segmentation_config import (
    AugmentConfig,
    InfraConfig,
    OptimizerConfig,
    PatchConfig,
    SchedulerConfig,
    TrainingConfig,
    UNet3DConfig,
)
from apps.labelmap_segmentation.test import (
    _apply_ema,
    _count_flops,
    _count_parameters,
    _hausdorff_distance_95,
    _input_shape,
    _resolve_device,
)
from apps.multihead_segmentation.dataset import create_multihead_datasets
from apps.multihead_segmentation.model import MultiHeadUNet3D
from apps.multihead_segmentation.multihead_config import DataConfig, LossConfig

# ── Config loader ──────────────────────────────────────────────────────────────


def _load_configs(config_dir: str) -> ConfigManager:
    """Populate a fresh ConfigManager from the JSON files in *config_dir*."""
    m = ConfigManager.get()
    base = Path(config_dir)
    type_map = {
        ConfigManager.DATA: DataConfig,
        ConfigManager.PATCH: PatchConfig,
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


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict_subject(
    model: MultiHeadUNet3D,
    subject: tio.Subject,
    data_cfg: DataConfig,
    patch_cfg: PatchConfig,
    train_cfg: TrainingConfig,
    task_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Returns {task_name: logits [1, C_t, D, H, W]}."""
    modalities = data_cfg.modalities

    if patch_cfg.enabled:
        grid = tio.GridSampler(subject, patch_cfg.size, patch_cfg.overlap)
        loader = DataLoader(grid, batch_size=train_cfg.batch_size, num_workers=0)
        aggrs = {
            task: tio.data.GridAggregator(grid, overlap_mode="average")
            for task in task_names
        }
        for pb in loader:
            imgs = torch.cat([pb[m][tio.DATA] for m in modalities], dim=1).to(device)
            logits_dict = model(imgs)
            for task in task_names:
                aggrs[task].add_batch(logits_dict[task], pb[tio.LOCATION])
        return {
            task: aggrs[task].get_output_tensor().unsqueeze(0).to(device)
            for task in task_names
        }

    imgs = (
        torch.cat([subject[m][tio.DATA] for m in modalities], dim=0)
        .unsqueeze(0)
        .to(device)
    )
    return model(imgs)


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
    patch_cfg: PatchConfig = m.get_config(ConfigManager.PATCH)
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
    _, _, test_ds = create_multihead_datasets()
    print(f"[test] Test subjects: {len(test_ds)}")
    if len(test_ds) == 0:
        sys.exit("[test] ERROR: test dataset is empty")

    task_names: List[str] = list(data_cfg.tasks.keys())

    # ── Model ──────────────────────────────────────────────────────────────────
    model = MultiHeadUNet3D(
        in_channels=model_cfg.encoder.in_channels,
        task_num_classes=data_cfg.tasks,
        base_features=model_cfg.encoder.base_features,
        trilinear=model_cfg.decoder.trilinear,
        num_supervision_levels=model_cfg.num_supervision_levels,
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
    shape = _input_shape(data_cfg, patch_cfg, model_cfg, input_shape)
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

        logits = _predict_subject(
            model, subject, data_cfg, patch_cfg, train_cfg, task_names, dev
        )

        result: Dict[str, float] = {"subject": subject_id}
        per_task_mean_dice = []

        for task in task_names:
            nc = data_cfg.tasks[task]
            label = subject[task][tio.DATA].unsqueeze(0).to(dev)

            dice = compute_dice(logits[task], label, nc)
            iou = compute_iou(logits[task], label, nc)

            pred_np = logits[task].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_np = label[0, 0].cpu().numpy().astype(np.uint8)
            spacing = subject[task].spacing

            hd_vals = []
            for c in range(1, nc):
                d = _hausdorff_distance_95(pred_np == c, gt_np == c, spacing)
                result[f"{task}_hd95_class_{c}"] = d
                hd_vals.append(d)
            result[f"{task}_mean_hd95"] = (
                float(np.nanmean(hd_vals)) if hd_vals else float("nan")
            )

            for k, v in dice.items():
                result[f"{task}_{k}"] = v
            for k, v in iou.items():
                result[f"{task}_{k}"] = v
            per_task_mean_dice.append(dice["mean_dice"])

            if save_predictions:
                ref = subject[data_cfg.modalities[0]]
                pred_img = tio.LabelMap(
                    tensor=torch.from_numpy(pred_np).unsqueeze(0), affine=ref.affine
                )
                pred_img.save(str(pred_dir / f"{subject_id}_{task}.nii.gz"))

        mean_dice = float(np.mean(per_task_mean_dice))
        result["mean_dice"] = mean_dice
        per_subject.append(result)

        per_task_log = "  ".join(
            f'{t} dice {result[f"{t}_mean_dice"]:.4f}' for t in task_names
        )
        print(f"  {subject_id}: {per_task_log}  | overall {mean_dice:.4f}")

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
        f'[test] mean Dice (avg across tasks): {aggregate["mean_dice"]["mean"]:.4f} '
        f'± {aggregate["mean_dice"]["std"]:.4f}'
    )
    print(f"[test] Results written to {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference + evaluation on the test split for a "
        "Multi-Head Segmentation experiment",
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
        help="Skip writing predicted label maps to disk (metrics are still computed)",
    )
    p.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Spatial shape used for the FLOPs dummy input. "
        "Defaults to PatchConfig.size, then DataConfig.target_shape, then 128x128x128.",
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
