"""
Export a trained UNet3D checkpoint to ONNX.

Usage
-----
# Standard deployment export (argmax head, dynamic spatial dims):
    python -m apps.labelmap_segmentation.export_onnx ^
        --config_dir  output/exp_001 ^
        --checkpoint  output/exp_001/checkpoints/best.pth ^
        --output      model.onnx

# Evaluation head (labelmap + normalized-entropy uncertainty map):
    python -m apps.labelmap_segmentation.export_onnx ^
        --config_dir  output/exp_001 ^
        --checkpoint  output/exp_001/checkpoints/best.pth ^
        --head        eval ^
        --output      model_eval.onnx

# Fixed spatial size (static graph -- faster on TensorRT / CoreML):
    python -m apps.labelmap_segmentation.export_onnx ^
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
* Both heads output uint8 labelmap (max 255 classes), compatible with most
  runtimes including TensorRT and ONNX Runtime without additional casting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from configuration.manager import ConfigManager
from apps.labelmap_segmentation.model import UNet3D
from apps.labelmap_segmentation.segmentation_config import (
    AugmentConfig,
    DataConfig,
    InfraConfig,
    LossConfig,
    OptimizerConfig,
    PatchConfig,
    SchedulerConfig,
    TrainingConfig,
    UNet3DConfig,
)

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
    """Traces UNet3D.forward_infer: single int64 labelmap output."""

    def __init__(self, model: UNet3D) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_infer(x)


class _EvalWrapper(nn.Module):
    """Traces UNet3D.forward_eval: (labelmap, uncertainty) outputs."""

    def __init__(self, model: UNet3D) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_eval(x)


# ── Weight helpers ────────────────────────────────────────────────────────────


def _apply_ema(model: UNet3D, ckpt: dict, device: torch.device) -> bool:
    """
    Overwrite model parameters with EMA shadow weights from *ckpt*.
    Returns True iff at least one parameter was updated.
    """
    shadow = ckpt.get("ema", {}).get("shadow", {})
    if not shadow:
        return False
    updated = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in shadow:
            param.data.copy_(shadow[name].to(device))
            updated += 1
    return updated > 0


# ── Shape helpers ─────────────────────────────────────────────────────────────


def _input_shape(
    data_cfg: DataConfig,
    patch_cfg: PatchConfig,
    model_cfg: UNet3DConfig,
    override: Optional[Tuple[int, int, int]],
) -> Tuple[int, ...]:
    """Return a concrete (1, C, D, H, W) dummy-input shape."""
    c = model_cfg.encoder.in_channels
    if override:
        return (1, c, *override)
    if patch_cfg.enabled:
        return (1, c, *patch_cfg.size)
    if data_cfg.target_shape:
        return (1, c, *data_cfg.target_shape)
    print("[export] No spatial shape found in config -- defaulting to 128x128x128")
    return (1, c, 128, 128, 128)


# ── Post-export passes ─────────────────────────────────────────────────────────


def _check_onnx(path: str) -> None:
    try:
        import onnx

        proto = onnx.load(path)
        onnx.checker.check_model(proto)
        proto = onnx.shape_inference.infer_shapes(proto)
        onnx.save(proto, path)
        mb = Path(path).stat().st_size / 1e6
        print(f"[export] ONNX check OK -- {mb:.1f} MB  (shape inference applied)")
    except ImportError:
        print("[export] onnx not installed -- skipping graph check  (pip install onnx)")
    except Exception as exc:
        print(f"[export] ONNX check warning: {exc}")


def _simplify(path: str, shape: Tuple[int, ...]) -> None:
    try:
        import onnx
        import onnxsim

        print("[export] Running onnx-simplifier ...")
        proto = onnx.load(path)
        simplified, ok = onnxsim.simplify(
            proto,
            input_shapes={"input": list(shape)},
        )
        if ok:
            onnx.save(simplified, path)
            mb = Path(path).stat().st_size / 1e6
            print(f"[export] Simplified -- {mb:.1f} MB")
        else:
            print(
                "[export] onnx-simplifier: simplification not possible -- original kept"
            )
    except ImportError:
        print(
            "[export] onnx-simplifier not installed -- skipping  (pip install onnxsim)"
        )
    except Exception as exc:
        print(f"[export] Simplification warning: {exc}")


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
        2. Reconstruct and populate UNet3D from *checkpoint*.
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

    # ── Build model ────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels=model_cfg.encoder.in_channels,
        num_classes=data_cfg.num_classes,
        base_features=model_cfg.encoder.base_features,
        trilinear=model_cfg.decoder.trilinear,
        num_supervision_levels=model_cfg.num_supervision_levels,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[export] Trainable params: {n_params:,}")

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
        wrapper = _EvalWrapper(model).to(dev)
        output_names = ["labelmap", "uncertainty"]
        print("[export] EvalHead: outputs labelmap (uint8) + uncertainty (float32)")
    else:
        wrapper = _InferWrapper(model).to(dev)
        output_names = ["labelmap"]
        print("[export] InferHead: output labelmap (uint8)")

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
        description="Export UNet3D checkpoint to ONNX",
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
        help="Path to .pth checkpoint " "(e.g. output/exp_001/checkpoints/best.pth)",
    )
    p.add_argument("--output", default="model.onnx", help="Output ONNX file path")
    p.add_argument(
        "--head",
        default="infer",
        choices=["infer", "eval"],
        help='"infer": argmax labelmap only (fastest, deployment default). '
        '"eval": labelmap + normalized-entropy uncertainty map.',
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
