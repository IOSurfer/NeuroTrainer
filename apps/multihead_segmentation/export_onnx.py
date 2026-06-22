"""
Export a trained MultiHeadUNet3D checkpoint to ONNX.

Usage
-----
# Standard deployment export (argmax heads, dynamic spatial dims):
    python -m apps.multihead_segmentation.export_onnx ^
        --config_dir  output/exp_001 ^
        --checkpoint  output/exp_001/checkpoints/best.pth ^
        --output      model.onnx

# Evaluation heads (labelmap + normalized-entropy uncertainty, per task):
    python -m apps.multihead_segmentation.export_onnx ^
        --config_dir  output/exp_001 ^
        --checkpoint  output/exp_001/checkpoints/best.pth ^
        --head        eval ^
        --output      model_eval.onnx

# Fixed spatial size (static graph -- faster on TensorRT / CoreML):
    python -m apps.multihead_segmentation.export_onnx ^
        --config_dir  output/exp_001 ^
        --checkpoint  output/exp_001/checkpoints/best.pth ^
        --static_shape ^
        --input_shape 128 128 128

Notes
-----
* EMA weights are applied automatically when present in the checkpoint and
  TrainingConfig.ema is True.
* Install onnxsim (pip install onnxsim) for an optional graph-simplification
  pass after export.
* Every task produces its own labelmap output (and uncertainty output when
  --head eval), named "<task>_labelmap" / "<task>_uncertainty", in the order
  tasks were declared in DataConfig.tasks.
* Generic helpers (_apply_ema, _input_shape, _check_onnx, _simplify) are
  reused unchanged from apps.labelmap_segmentation.export_onnx -- only the
  multi-task-aware pieces (_load_configs, the ONNX wrappers, export) are new.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from configuration.manager import ConfigManager
from apps.labelmap_segmentation.export_onnx import (
    _apply_ema,
    _check_onnx,
    _input_shape,
    _simplify,
)
from apps.labelmap_segmentation.segmentation_config import (
    AugmentConfig,
    InfraConfig,
    OptimizerConfig,
    PatchConfig,
    SchedulerConfig,
    TrainingConfig,
    UNet3DConfig,
)
from apps.multihead_segmentation.model import MultiHeadUNet3D
from apps.multihead_segmentation.multihead_config import DataConfig, LossConfig

# ── Config loader ──────────────────────────────────────────────────────────────


def _load_configs(config_dir: str) -> ConfigManager:
    """
    Populate a fresh ConfigManager from the JSON files in *config_dir*.
    Missing JSON files are silently ignored (defaults are used instead).
    """
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
            print(f"[export] {json_path.name} not found -- using defaults")
        m.register(key, cfg)
    return m


# ── ONNX wrapper modules ───────────────────────────────────────────────────────


class _InferWrapper(nn.Module):
    """Traces MultiHeadUNet3D.forward_infer: one labelmap output per task, in task order."""

    def __init__(self, model: MultiHeadUNet3D, task_names: List[str]) -> None:
        super().__init__()
        self.model = model
        self.task_names = task_names

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.model.forward_infer(x)
        return tuple(out[t] for t in self.task_names)


class _EvalWrapper(nn.Module):
    """Traces MultiHeadUNet3D.forward_eval: (labelmap, uncertainty) per task, flattened in task order."""

    def __init__(self, model: MultiHeadUNet3D, task_names: List[str]) -> None:
        super().__init__()
        self.model = model
        self.task_names = task_names

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.model.forward_eval(x)
        flat: List[torch.Tensor] = []
        for t in self.task_names:
            labelmap, uncertainty = out[t]
            flat.append(labelmap)
            flat.append(uncertainty)
        return tuple(flat)


# ── Main export logic ─────────────────────────────────────────────────────────


def export(
    config_dir: str,
    checkpoint: str,
    output: str,
    head: str = "infer",
    opset: int = 17,
    static_shape: bool = False,
    input_shape: Optional[Tuple[int, int, int]] = None,
    device: str = "cpu",
) -> None:
    """
    Full export pipeline:
        1. Load configs from *config_dir*.
        2. Reconstruct and populate MultiHeadUNet3D from *checkpoint*.
        3. Apply EMA shadow weights when available.
        4. Trace with the requested head and export to *output*.
        5. Verify ONNX graph and optionally run onnx-simplifier.
    """
    print(f"[export] Config dir : {config_dir}")
    print(f"[export] Checkpoint : {checkpoint}")
    print(f"[export] Head       : {head}")
    print(f"[export] Opset      : {opset}")
    print(f"[export] Static     : {static_shape}")

    m = _load_configs(config_dir)
    data_cfg: DataConfig = m.get_config(ConfigManager.DATA)
    patch_cfg: PatchConfig = m.get_config(ConfigManager.PATCH)
    model_cfg: UNet3DConfig = m.get_config(ConfigManager.MODEL)
    train_cfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)

    dev = torch.device(device)
    task_names: List[str] = list(data_cfg.tasks.keys())

    # ── Build model ────────────────────────────────────────────────────────────
    model = MultiHeadUNet3D(
        in_channels=model_cfg.encoder.in_channels,
        task_num_classes=data_cfg.tasks,
        base_features=model_cfg.encoder.base_features,
        trilinear=model_cfg.decoder.trilinear,
        num_supervision_levels=model_cfg.num_supervision_levels,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[export] Trainable params: {n_params:,}")
    print(f"[export] Tasks: {task_names}")

    # ── Load weights ───────────────────────────────────────────────────────────
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"[export] ERROR: checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    print(f"[export] Checkpoint epoch: {epoch}")

    # ── EMA ────────────────────────────────────────────────────────────────────
    if train_cfg.ema and "ema" in ckpt:
        applied = _apply_ema(model, ckpt, dev)
        status = "applied" if applied else "shadow dict empty -- raw weights kept"
        print(f"[export] EMA: {status}")
    elif "ema" in ckpt and not train_cfg.ema:
        print(
            "[export] EMA state found in checkpoint but TrainingConfig.ema=False -- skipped"
        )

    model.eval()

    # ── Wrapper & output names ─────────────────────────────────────────────────
    if head == "eval":
        wrapper = _EvalWrapper(model, task_names).to(dev)
        output_names = []
        for t in task_names:
            output_names += [f"{t}_labelmap", f"{t}_uncertainty"]
        print("[export] EvalHead: outputs labelmap (uint8) + uncertainty (float32) per task")
    else:
        wrapper = _InferWrapper(model, task_names).to(dev)
        output_names = [f"{t}_labelmap" for t in task_names]
        print("[export] InferHead: output labelmap (uint8) per task")

    # ── Dummy input ────────────────────────────────────────────────────────────
    shape = _input_shape(data_cfg, patch_cfg, model_cfg, input_shape)
    dummy = torch.zeros(shape, device=dev)
    print(f"[export] Dummy input: {list(shape)} {dummy.dtype}")

    # ── Dynamic axes ───────────────────────────────────────────────────────────
    dynamic_axes: Optional[dict] = None
    if not static_shape:
        dynamic_axes = {"input": {0: "batch", 2: "depth", 3: "height", 4: "width"}}
        for oname in output_names:
            # outputs are [B, D, H, W] -- axes 0-3
            dynamic_axes[oname] = {0: "batch", 1: "depth", 2: "height", 3: "width"}

    # ── Export ─────────────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export] Exporting to {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"[export] Export done")

    # ── Post-export passes ─────────────────────────────────────────────────────
    _check_onnx(str(out_path))
    _simplify(str(out_path), shape)

    print(f"[export] Finished: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export MultiHeadUNet3D checkpoint to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config_dir",
        required=True,
        help="Experiment directory that contains the *.json config files "
        "(e.g. output/exp_001/)",
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth checkpoint (e.g. output/exp_001/checkpoints/best.pth)",
    )
    p.add_argument("--output", default="model.onnx", help="Output ONNX file path")
    p.add_argument(
        "--head",
        default="infer",
        choices=["infer", "eval"],
        help='"infer": argmax labelmap only per task (fastest, deployment default). '
        '"eval": labelmap + normalized-entropy uncertainty map per task.',
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument(
        "--static_shape",
        action="store_true",
        help="Fix all spatial dimensions in the graph (no dynamic axes). "
        "Required for TensorRT or CoreML compilation. "
        "Use with --input_shape to control the fixed size.",
    )
    p.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Override the dummy-input spatial shape used for tracing. "
        "Defaults to PatchConfig.size, then DataConfig.target_shape, "
        "then 128x128x128.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for tracing. cpu is recommended for portability.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    export(
        config_dir=args.config_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        head=args.head,
        opset=args.opset,
        static_shape=args.static_shape,
        input_shape=tuple(args.input_shape) if args.input_shape else None,
        device=args.device,
    )


if __name__ == "__main__":
    main()
