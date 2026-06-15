"""
SDF Estimation -- training entry point.

Usage
-----
Train from scratch::

    python -m apps.sdf_estimation.train \\
        --data_root /data/sdf_dataset \\
        --sdf_names sdf_bone sdf_muscle \\
        --experiment_name exp_sdf_001

Reproduce a past experiment exactly::

    python -m apps.sdf_estimation.train \\
        --reproduce output/exp_sdf_001/

See --help for all options.

Notes
-----
- Patch-based training is **not supported**.  SDF estimation requires the full
  spatial context of the volume; sub-patches break the global distance property.
- Best-model selection is based on validation **mean MAE** (lower is better).
- Intensity augmentations are restricted to input modalities; SDF fields are
  only subject to spatial augmentations.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configuration.manager import ConfigManager
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
from apps.sdf_estimation.dataset import (
    create_data_loaders,
    create_sdf_datasets,
)
from apps.sdf_estimation.losses import SDFLoss
from apps.sdf_estimation.metrics import compute_sdf_metrics
from apps.sdf_estimation.model import UNet3DSDF

# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDF Estimation -- 3D U-Net trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reproduce",
        default=None,
        metavar="EXP_DIR",
        help="Reproduce an experiment by loading its saved configs. "
        "All other flags are ignored when this is set.",
    )

    g = p.add_argument_group("Data")
    g.add_argument("--data_root")
    g.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Input modality folder names (None = auto-detect)",
    )
    g.add_argument(
        "--sdf_names",
        nargs="+",
        required=False,
        default=None,
        help="SDF field folder names, e.g. --sdf_names sdf_bone sdf_muscle",
    )
    g.add_argument(
        "--target_spacing", type=float, nargs=3, default=None, metavar=("X", "Y", "Z")
    )
    g.add_argument(
        "--target_shape", type=int, nargs=3, default=None, metavar=("D", "H", "W")
    )
    g.add_argument(
        "--normalization", default="znorm", choices=["znorm", "rescale", "none"]
    )

    g = p.add_argument_group("Augmentation")
    g.add_argument("--no_augment", action="store_true", help="Disable all augmentation")
    g.add_argument("--no_flip", action="store_true")
    g.add_argument("--no_affine", action="store_true")
    g.add_argument(
        "--no_elastic",
        action="store_true",
        help="Disable elastic deformation (enabled by default)",
    )
    g.add_argument("--no_noise", action="store_true")
    g.add_argument("--no_blur", action="store_true")
    g.add_argument("--no_gamma", action="store_true")

    g = p.add_argument_group("Model")
    g.add_argument("--base_features", type=int, default=32)
    g.add_argument("--trilinear", action="store_true", default=True)
    g.add_argument("--no_trilinear", action="store_false", dest="trilinear")

    g = p.add_argument_group("Loss")
    g.add_argument("--recon_weight", type=float, default=1.0)
    g.add_argument("--eikonal_weight", type=float, default=0.1)
    g.add_argument("--normal_weight", type=float, default=0.0)
    g.add_argument("--boundary_sigma", type=float, default=1.0)

    g = p.add_argument_group("Optimiser / Scheduler")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch_size", type=int, default=1)
    g.add_argument("--gradient_accumulation", type=bool, default=True)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "sgd"])
    g.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "plateau", "step", "none"]
    )
    g.add_argument("--warmup_epochs", type=int, default=30)
    g.add_argument("--scheduler_patience", type=int, default=10)
    g.add_argument("--scheduler_factor", type=float, default=0.5)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--amp", action="store_true")
    g.add_argument("--ema", action="store_true")
    g.add_argument("--ema_decay", type=float, default=0.99)

    g = p.add_argument_group("Early stopping")
    g.add_argument("--early_stopping", action="store_true")
    g.add_argument("--early_stopping_patience", type=int, default=30)

    g = p.add_argument_group("Infrastructure")
    g.add_argument("--output_dir", default="./output")
    g.add_argument("--experiment_name", default="sdf_estimation")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--resume", default=None)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--val_interval", type=int, default=1)
    g.add_argument("--save_interval", type=int, default=10)
    g.add_argument("--log_interval", type=int, default=10)
    g.add_argument("--log_images", action="store_true")
    g.add_argument("--log_images_interval", type=int, default=10)
    g.add_argument("--torch_compile", action="store_true")

    return p.parse_args()


# ── ConfigManager setup ───────────────────────────────────────────────────────


def setup_manager(args: argparse.Namespace) -> ConfigManager:
    if not args.data_root:
        print("Error: --data_root is required")

    m = ConfigManager.get()

    dc = DataConfig()
    dc.data_root = args.data_root
    dc.modalities = args.modalities
    dc.sdf_names = args.sdf_names
    dc.target_spacing = tuple(args.target_spacing) if args.target_spacing else None
    dc.target_shape = tuple(args.target_shape) if args.target_shape else None
    dc.normalization = args.normalization

    ac = AugmentConfig()
    ac.enabled = not args.no_augment
    ac.flip = not args.no_flip
    ac.affine = not args.no_affine
    ac.elastic = not args.no_elastic
    ac.noise = not args.no_noise
    ac.blur = not args.no_blur
    ac.gamma = not args.no_gamma

    mc = UNet3DConfig()
    mc.encoder.base_features = args.base_features
    mc.decoder.trilinear = args.trilinear

    lc = LossConfig()
    lc.recon_weight = args.recon_weight
    lc.eikonal_weight = args.eikonal_weight
    lc.normal_weight = args.normal_weight
    lc.boundary_sigma = args.boundary_sigma

    oc = OptimizerConfig()
    oc.type = args.optimizer
    oc.lr = args.lr
    oc.weight_decay = args.weight_decay
    oc.grad_clip = args.grad_clip

    sc = SchedulerConfig()
    sc.type = args.scheduler
    sc.warmup_epochs = args.warmup_epochs
    sc.patience = args.scheduler_patience
    sc.factor = args.scheduler_factor

    tc = TrainingConfig()
    tc.epochs = args.epochs
    tc.batch_size = args.batch_size
    tc.gradient_accumulation = args.gradient_accumulation
    tc.amp = args.amp
    tc.early_stopping = args.early_stopping
    tc.early_stopping_patience = args.early_stopping_patience
    tc.ema = args.ema
    tc.ema_decay = args.ema_decay

    ic = InfraConfig()
    ic.output_dir = args.output_dir
    ic.experiment_name = args.experiment_name
    ic.num_workers = args.num_workers
    ic.device = args.device
    ic.seed = args.seed
    ic.resume = args.resume
    ic.val_interval = args.val_interval
    ic.save_interval = args.save_interval
    ic.log_interval = args.log_interval
    ic.log_images = args.log_images
    ic.log_images_interval = args.log_images_interval
    ic.torch_compile = args.torch_compile

    m.register(ConfigManager.DATA, dc)
    m.register(ConfigManager.AUGMENT, ac)
    m.register(ConfigManager.MODEL, mc)
    m.register(ConfigManager.LOSS, lc)
    m.register(ConfigManager.OPTIMIZER, oc)
    m.register(ConfigManager.SCHEDULER, sc)
    m.register(ConfigManager.TRAINING, tc)
    m.register(ConfigManager.INFRA, ic)

    return m


def load_manager_from_directory(directory: str) -> ConfigManager:
    """Restore the ConfigManager from a previously saved experiment directory."""
    m = ConfigManager.get()
    dir_path = Path(directory)

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
    for type_name, cfg_cls in type_map.items():
        cfg = cfg_cls()
        path = dir_path / f"{type_name.lower()}.json"
        if path.exists():
            cfg.load(str(path))
        m.register(type_name, cfg)

    return m


# ── EMA ────────────────────────────────────────────────────────────────────────


class _EMA:
    """Shadow-parameter EMA for any nn.Module."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.n_updates: int = 0
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def reset(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].copy_(p.data)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        self.n_updates += 1

    def apply_shadow(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "n_updates": self.n_updates,
            "shadow": {n: v.cpu() for n, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay = state["decay"]
        self.n_updates = state.get("n_updates", 0)
        self.shadow = {n: v.to(device) for n, v in state["shadow"].items()}


# ── Trainer ────────────────────────────────────────────────────────────────────


class Trainer:
    """
    SDF Estimation training loop.

    All behaviour is driven by configs registered in the ConfigManager.
    Patch-based training is explicitly disallowed; only full-volume mode
    is supported.
    """

    def __init__(self) -> None:
        m = ConfigManager.get()
        self.data_cfg: DataConfig = m.get_config(ConfigManager.DATA)
        self.aug_cfg: AugmentConfig = m.get_config(ConfigManager.AUGMENT)
        self.model_cfg: UNet3DConfig = m.get_config(ConfigManager.MODEL)
        self.loss_cfg: LossConfig = m.get_config(ConfigManager.LOSS)
        self.opt_cfg: OptimizerConfig = m.get_config(ConfigManager.OPTIMIZER)
        self.sched_cfg: SchedulerConfig = m.get_config(ConfigManager.SCHEDULER)
        self.train_cfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
        self.infra_cfg: InfraConfig = m.get_config(ConfigManager.INFRA)

        self.device = self._resolve_device()
        self._seed_everything()

        self.exp_dir = Path(self.infra_cfg.output_dir) / self.infra_cfg.experiment_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.tb_dir = self.exp_dir / "tensorboard"
        for d in (self.ckpt_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        ConfigManager.get().save_all(str(self.exp_dir))
        self.log.info(f"Configs saved to {self.exp_dir}")

        self.log.info("Building datasets...")
        self.train_ds, self.val_ds, self.test_ds = create_sdf_datasets()
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_ds, self.val_ds
        )
        self.log.info(
            f"Subjects -- train: {len(self.train_ds)}  "
            f"val: {len(self.val_ds)}  test: {len(self.test_ds)}"
        )

        if not self.data_cfg.modalities:
            raise RuntimeError("data_cfg.modalities is empty after dataset creation.")
        if not self.data_cfg.sdf_names:
            raise RuntimeError("data_cfg.sdf_names is empty after dataset creation.")

        self.sdf_names: List[str] = list(self.data_cfg.sdf_names)
        num_sdf_fields: int = self.data_cfg.num_sdf_fields

        # Resolve in_channels now that modalities are known
        self.model_cfg.encoder.in_channels = len(self.data_cfg.modalities)

        self.model = UNet3DSDF(
            in_channels=self.model_cfg.encoder.in_channels,
            num_sdf_fields=num_sdf_fields,
            base_features=self.model_cfg.encoder.base_features,
            trilinear=self.model_cfg.decoder.trilinear,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info(
            f"UNet3DSDF -- in_channels={self.model_cfg.encoder.in_channels}  "
            f"num_sdf_fields={num_sdf_fields}  trainable_params={n_params:,}"
        )

        self.criterion = SDFLoss(
            recon_weight=self.loss_cfg.recon_weight,
            eikonal_weight=self.loss_cfg.eikonal_weight,
            normal_weight=self.loss_cfg.normal_weight,
            boundary_sigma=self.loss_cfg.boundary_sigma,
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda")
            if self.train_cfg.amp and self.device.type == "cuda"
            else None
        )

        self.ema: Optional[_EMA] = (
            _EMA(self.model, self.train_cfg.ema_decay) if self.train_cfg.ema else None
        )
        if self.ema:
            self.log.info(f"EMA enabled -- decay {self.train_cfg.ema_decay}")

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self._log_model_graph()

        # Keep an uncompiled reference for EMA and grad-norm inspection.
        self._raw_model: nn.Module = self.model

        if self.infra_cfg.torch_compile:
            if hasattr(torch, "compile"):
                self.log.info(
                    "torch.compile(dynamic=False) -- first batch triggers compilation"
                )
                self.model = torch.compile(self.model, dynamic=False)
            else:
                self.log.warning(
                    "torch_compile=True but torch.compile is unavailable "
                    "(requires PyTorch >= 2.0) -- skipped"
                )

        self.start_epoch = 0
        self.best_val_mae = float("inf")
        self.stale_epochs = 0

        if self.infra_cfg.resume:
            self._load_checkpoint(self.infra_cfg.resume)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        spec = self.infra_cfg.device
        return (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if spec == "auto"
            else torch.device(spec)
        )

    def _seed_everything(self) -> None:
        import random

        s = self.infra_cfg.seed
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.exp_dir / "training.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.log = logging.getLogger("trainer")

    def _build_optimizer(self) -> optim.Optimizer:
        c, p = self.opt_cfg, self.model.parameters()
        if c.type == "adam":
            return optim.Adam(p, lr=c.lr, weight_decay=c.weight_decay)
        if c.type == "adamw":
            return optim.AdamW(p, lr=c.lr, weight_decay=c.weight_decay)
        return optim.SGD(
            p, lr=c.lr, momentum=0.9, weight_decay=c.weight_decay, nesterov=True
        )

    def _build_scheduler(self):
        c, o, e = self.sched_cfg, self.optimizer, self.train_cfg.epochs
        if c.type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                o, T_max=max(e - c.warmup_epochs, 1)
            )
        if c.type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                o, mode="min", factor=c.factor, patience=c.patience
            )
        if c.type == "step":
            return optim.lr_scheduler.StepLR(o, step_size=c.patience, gamma=c.factor)
        return None

    @contextlib.contextmanager
    def _ema_scope(self):
        """Temporarily apply EMA shadow weights, then restore."""
        if self.ema is None or self.ema.n_updates == 0:
            yield
            return
        backup = {
            n: p.data.clone()
            for n, p in self._raw_model.named_parameters()
            if p.requires_grad
        }
        self.ema.apply_shadow(self._raw_model)
        try:
            yield
        finally:
            for n, p in self._raw_model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])

    def _log_model_graph(self) -> None:
        try:
            shape = self.data_cfg.target_shape or (64, 64, 64)
            dummy = torch.zeros(
                1, len(self.data_cfg.modalities), *shape, device=self.device
            )
            self.writer.add_graph(self.model, dummy)
        except Exception as exc:
            self.log.warning(f"Model graph logging skipped: {exc}")

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _batch_to_tensors(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: ``[B, C_mod, D, H, W]`` input modalities.
            y: ``[B, num_sdf_fields, D, H, W]`` SDF targets (float32).
        """
        x = torch.cat([batch[m][tio.DATA] for m in self.data_cfg.modalities], dim=1).to(
            self.device
        )
        y = (
            torch.cat([batch[s][tio.DATA] for s in self.sdf_names], dim=1)
            .float()
            .to(self.device)
        )
        return x, y

    def _forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = self.model(x)
                raw = self.criterion(pred, y)
        else:
            pred = self.model(x)
            raw = self.criterion(pred, y)
        loss, components = raw
        return pred, loss, components

    def _block_grad_norms(self) -> Dict[str, float]:
        norms: Dict[str, float] = {}
        for name, module in self._raw_model.named_children():
            params = [
                p for p in module.parameters() if p.requires_grad and p.grad is not None
            ]
            if params:
                norms[name] = (
                    torch.stack([p.grad.detach().norm() for p in params]).norm().item()
                )
        return norms

    # ── LR warmup / scheduler step ────────────────────────────────────────────

    def _apply_warmup(self, epoch: int) -> None:
        c = self.sched_cfg
        if epoch < c.warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.opt_cfg.lr * (epoch + 1) / c.warmup_epochs

    def _step_scheduler(self, val_mae: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_mae)
        else:
            self.scheduler.step()

    # ── Epoch loops ───────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self._apply_warmup(epoch)

        if self.ema is not None and epoch == self.sched_cfg.warmup_epochs:
            self.ema.reset(self._raw_model)
            self.log.info(
                f"Epoch {epoch}: warmup complete -- EMA shadow synced to current model weights"
            )

        accum: Dict[str, float] = {}
        accum_gnorms: Dict[str, float] = {}
        n = 0
        step_count = 0
        accum_steps = (
            self.train_cfg.batch_size if self.train_cfg.gradient_accumulation else 1
        )
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            x, y = self._batch_to_tensors(batch)
            self.optimizer.zero_grad()
            pred, loss, components = self._forward(x, y)

            loss_scaled = loss / accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            with torch.no_grad():
                metrics = compute_sdf_metrics(pred, y)

            batch_m: Dict[str, float] = {"loss": loss.item(), **metrics}
            for k, v in components.items():
                batch_m[f"loss_{k}"] = v.item()
            for k, v in batch_m.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1

            do_step = ((i + 1) % accum_steps == 0) or (i == len(self.train_loader) - 1)

            if do_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                block_norms = self._block_grad_norms()
                total_norm = nn.utils.clip_grad_norm_(
                    self._raw_model.parameters(),
                    (
                        self.opt_cfg.grad_clip
                        if self.opt_cfg.grad_clip > 0
                        else float("inf")
                    ),
                ).item()
                accum_gnorms["total"] = accum_gnorms.get("total", 0.0) + total_norm
                for blk, nrm in block_norms.items():
                    accum_gnorms[blk] = accum_gnorms.get(blk, 0.0) + nrm
                step_count += 1

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None and epoch >= self.sched_cfg.warmup_epochs:
                    self.ema.update(self._raw_model)

            if i % self.infra_cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar("train/batch_loss", loss.item(), step)
                self.writer.add_scalar(
                    "train/batch_mean_mae", metrics["mean_mae"], step
                )
                self.writer.add_scalar(
                    "train/batch_mean_mse", metrics["mean_mse"], step
                )
                if "recon" in components:
                    self.writer.add_scalar(
                        "train/batch_loss_recon", components["recon"].item(), step
                    )
                if "eikonal" in components:
                    self.writer.add_scalar(
                        "train/batch_loss_eikonal", components["eikonal"].item(), step
                    )
                if "normal" in components:
                    self.writer.add_scalar(
                        "train/batch_loss_normal", components["normal"].item(), step
                    )
                self.writer.add_scalar("train/lr", lr, step)
                self.log.info(
                    f"Epoch {epoch:04d} | Batch {i}/{len(self.train_loader)} | "
                    f'loss {loss.item():.4f} | mae {metrics["mean_mae"]:.4f} | lr {lr:.2e}'
                )

        result = {k: v / n for k, v in accum.items()}
        if step_count:
            result["grad_norm"] = accum_gnorms.get("total", 0.0) / step_count
            for blk, nrm_sum in accum_gnorms.items():
                if blk != "total":
                    result[f"grad_norm_{blk}"] = nrm_sum / step_count
        return result

    @torch.no_grad()
    def val_epoch(self) -> Dict:
        self.model.eval()
        accum: Dict[str, float] = {}
        n = 0

        for batch in self.val_loader:
            x, y = self._batch_to_tensors(batch)
            pred, loss, _ = self._forward(x, y)
            metrics = compute_sdf_metrics(pred, y)

            if n == 0:
                accum = {"loss": 0.0, **{k: 0.0 for k in metrics}}
            accum["loss"] += loss.item()
            for k, v in metrics.items():
                accum[k] += v
            n += 1

        return {k: v / n for k, v in accum.items()} if n else accum

    # ── TensorBoard logging ───────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict) -> None:
        num_fields = self.data_cfg.num_sdf_fields

        self.writer.add_scalars(
            "epoch/loss", {"train": train_m["loss"], "val": val_m.get("loss", 0)}, epoch
        )

        for key in ("loss_recon", "loss_eikonal", "loss_normal"):
            if key in train_m:
                self.writer.add_scalar(f"epoch/{key}", train_m[key], epoch)

        self.writer.add_scalars(
            "epoch/mean_mae",
            {"train": train_m["mean_mae"], "val": val_m.get("mean_mae", 0)},
            epoch,
        )
        self.writer.add_scalars(
            "epoch/mean_mse",
            {"train": train_m["mean_mse"], "val": val_m.get("mean_mse", 0)},
            epoch,
        )

        for i in range(num_fields):
            km, ks = f"mae_field_{i}", f"mse_field_{i}"
            scalars: Dict[str, float] = {}
            if km in train_m:
                scalars["train"] = train_m[km]
            if km in val_m:
                scalars["val"] = val_m[km]
            if scalars:
                self.writer.add_scalars(f"epoch/mae_field_{i}", scalars, epoch)
            if ks in val_m:
                self.writer.add_scalar(f"epoch/val_mse_field_{i}", val_m[ks], epoch)

        if "grad_norm" in train_m:
            self.writer.add_scalar("train/grad_norm", train_m["grad_norm"], epoch)
        for block_name in self._raw_model._modules:
            key = f"grad_norm_{block_name}"
            if key in train_m:
                self.writer.add_scalar(
                    f"train/grad_norm_{block_name}", train_m[key], epoch
                )

        self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)

        if epoch % 10 == 0:
            for name, param in self._raw_model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f"weights/{name}", param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f"grads/{name}", param.grad, epoch)

    def _log_images(self, epoch: int) -> None:
        """Log mid-slice visualisations for ALL SDF channels."""
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            x, y = self._batch_to_tensors(batch)

            with torch.no_grad():
                pred = self.model(x)

            # First modality image (normalize to [0, 1])
            img = x[0, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            D, H, W = img.shape

            def _row(*slices):
                return torch.stack([s.unsqueeze(0) for s in slices])

            C = pred.shape[1]  # number of SDF channels

            for c in range(C):
                sdf_pred = pred[0, c]
                sdf_gt = y[0, c]

                # normalize per-channel jointly
                lo = min(sdf_pred.min(), sdf_gt.min())
                hi = max(sdf_pred.max(), sdf_gt.max())
                rng = hi - lo + 1e-8

                sdf_pred_n = (sdf_pred - lo) / rng
                sdf_gt_n = (sdf_gt - lo) / rng

                tag_prefix = f"slices/sdf_channel_{c}"

                # axial
                self.writer.add_images(
                    f"{tag_prefix}/axial",
                    _row(img[D // 2], sdf_gt_n[D // 2], sdf_pred_n[D // 2]),
                    epoch,
                )

                # coronal
                self.writer.add_images(
                    f"{tag_prefix}/coronal",
                    _row(
                        img[:, H // 2, :],
                        sdf_gt_n[:, H // 2, :],
                        sdf_pred_n[:, H // 2, :],
                    ),
                    epoch,
                )

                # sagittal
                self.writer.add_images(
                    f"{tag_prefix}/sagittal",
                    _row(
                        img[:, :, W // 2],
                        sdf_gt_n[:, :, W // 2],
                        sdf_pred_n[:, :, W // 2],
                    ),
                    epoch,
                )

        except Exception as exc:
            self.log.warning(f"Image logging failed: {exc}")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_m: Dict, is_best: bool) -> None:
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_metrics": val_m,
            "best_val_mae": self.best_val_mae,
        }
        if self.scheduler:
            ckpt["scheduler"] = self.scheduler.state_dict()
        if self.scaler:
            ckpt["scaler"] = self.scaler.state_dict()
        if self.ema:
            ckpt["ema"] = self.ema.state_dict()

        torch.save(ckpt, self.ckpt_dir / "latest.pth")
        if epoch % self.infra_cfg.save_interval == 0:
            torch.save(ckpt, self.ckpt_dir / f"epoch_{epoch:04d}.pth")
        if is_best:
            torch.save(ckpt, self.ckpt_dir / "best.pth")
            self.log.info(f'  -> New best -- val_mae {val_m["mean_mae"]:.4f}')

    def _load_checkpoint(self, path: str) -> None:
        self.log.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_mae = ckpt.get("best_val_mae", float("inf"))
        if "scheduler" in ckpt and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler"])
        if "ema" in ckpt and self.ema:
            self.ema.load_state_dict(ckpt["ema"], self.device)

    # ── Test evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self) -> float:
        best = self.ckpt_dir / "best.pth"
        if best.exists():
            self.log.info("Loading best checkpoint for test evaluation...")
            self._load_checkpoint(str(best))

        with self._ema_scope():
            return self._run_test()

    def _run_test(self) -> float:
        self.model.eval()
        all_mae, all_mse = [], []

        for subject in self.test_ds:
            subject_id = getattr(subject, "subject_id", "unknown")

            imgs = (
                torch.cat(
                    [subject[m][tio.DATA] for m in self.data_cfg.modalities], dim=0
                )
                .unsqueeze(0)
                .to(self.device)
            )
            sdf_gt = (
                torch.cat([subject[s][tio.DATA] for s in self.sdf_names], dim=0)
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

            pred = self.model(imgs)
            metrics = compute_sdf_metrics(pred, sdf_gt)

            all_mae.append(metrics["mean_mae"])
            all_mse.append(metrics["mean_mse"])
            self.log.info(
                f'  {subject_id}: mae {metrics["mean_mae"]:.4f}  mse {metrics["mean_mse"]:.4f}'
            )

        mean_mae = float(np.mean(all_mae))
        std_mae = float(np.std(all_mae))
        mean_mse = float(np.mean(all_mse))
        self.log.info(
            f"Test -- mean MAE: {mean_mae:.4f} ± {std_mae:.4f}  mean MSE: {mean_mse:.4f}"
        )

        self.writer.add_scalar("test/mean_mae", mean_mae)
        self.writer.add_scalar("test/std_mae", std_mae)
        self.writer.add_scalar("test/mean_mse", mean_mse)

        (self.exp_dir / "test_results.json").write_text(
            json.dumps(
                {
                    "mean_mae": mean_mae,
                    "std_mae": std_mae,
                    "mean_mse": mean_mse,
                    "per_subject": [
                        {
                            "subject": getattr(s, "subject_id", f"sub_{i}"),
                            "mae": float(d),
                            "mse": float(u),
                        }
                        for i, (s, d, u) in enumerate(
                            zip(self.test_ds, all_mae, all_mse)
                        )
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return mean_mae

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.log.info(f"Device: {self.device}")
        self.log.info(f"Experiment: {self.exp_dir}")

        self.writer.add_text(
            "config/summary", f"```\n{ConfigManager.get().print_all}\n```"
        )

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            t0 = time.perf_counter()
            train_m = self.train_epoch(epoch)
            val_m: Dict = {}

            if (epoch + 1) % self.infra_cfg.val_interval == 0:
                with self._ema_scope():
                    val_m = self.val_epoch()
                    if (
                        self.infra_cfg.log_images
                        and (epoch + 1) % self.infra_cfg.log_images_interval == 0
                    ):
                        self._log_images(epoch)
                self._log_epoch(epoch, train_m, val_m)

                if epoch >= self.sched_cfg.warmup_epochs:
                    self._step_scheduler(val_m.get("mean_mae", float("inf")))

                current_mae = val_m.get("mean_mae", float("inf"))
                is_best = current_mae < self.best_val_mae
                if is_best:
                    self.best_val_mae = current_mae
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1

                self._save_checkpoint(epoch, val_m, is_best)

                self.log.info(
                    f"Epoch {epoch:04d}/{self.train_cfg.epochs} | "
                    f'train_loss {train_m["loss"]:.4f}  '
                    f'train_mae {train_m["mean_mae"]:.4f} | '
                    f'val_loss {val_m.get("loss", 0):.4f}  '
                    f"val_mae {current_mae:.4f} | "
                    f"best {self.best_val_mae:.4f} | "
                    f"stale {self.stale_epochs} | "
                    f"{time.perf_counter() - t0:.0f}s"
                )

                if (
                    self.train_cfg.early_stopping
                    and self.stale_epochs >= self.train_cfg.early_stopping_patience
                ):
                    self.log.info(
                        f"Early stopping after {self.stale_epochs} epochs without improvement."
                    )
                    break

        self.writer.add_hparams(
            {
                "lr": self.opt_cfg.lr,
                "batch_size": self.train_cfg.batch_size,
                "base_features": self.model_cfg.encoder.base_features,
                "optimizer": self.opt_cfg.type,
                "recon_weight": self.loss_cfg.recon_weight,
                "eikonal_weight": self.loss_cfg.eikonal_weight,
                "normal_weight": self.loss_cfg.normal_weight,
                "boundary_sigma": self.loss_cfg.boundary_sigma,
                "num_sdf_fields": self.data_cfg.num_sdf_fields,
                "ema": self.train_cfg.ema,
                "ema_decay": self.train_cfg.ema_decay if self.train_cfg.ema else 0.0,
            },
            {"hparam/best_val_mae": self.best_val_mae},
        )
        self.writer.close()
        self.log.info("Training complete. Running test evaluation...")
        self.test()


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.reproduce:
        load_manager_from_directory(args.reproduce)
        print(f"Reproducing experiment from: {args.reproduce}")
    else:
        setup_manager(args)

    ConfigManager.get().print_all()
    Trainer().run()


if __name__ == "__main__":
    main()
