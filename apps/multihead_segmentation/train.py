"""
Multi-Head Segmentation -- training entry point.

Usage
-----
Train from scratch:
    python -m apps.multihead_segmentation.train \\
        --data_root /data/brain_mri --tasks organs:14 tumor:2 \\
        --patch_based --patch_size 128 128 128 \\
        --experiment_name exp_001 --log_images

Reproduce a past experiment exactly:
    python -m apps.multihead_segmentation.train \\
        --reproduce output/exp_001/

See --help for all options.

Design
------
``Trainer`` subclasses ``apps.labelmap_segmentation.train.Trainer`` to reuse
its generic machinery unchanged: device/seed setup, optimizer/scheduler
construction, the EMA shadow scope, gradient-norm bookkeeping, checkpoint
save/load (including partial-load by encoder/encoder_decoder/full -- the
prefixes match because ``MultiHeadUNet3D`` keeps the same ``encoder`` /
``bottleneck`` / ``decoder`` submodule names as ``UNet3D``), and the entire
``run()`` orchestration loop. Only methods that index a single task / single
``num_classes`` are overridden, and they keep populating the same
``"loss"`` / ``"mean_dice"`` keys the inherited methods read -- with
``mean_dice`` now defined as the average of every task's own mean_dice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
from apps.labelmap_segmentation.train import Trainer as LabelMapTrainer
from apps.labelmap_segmentation.train import _EMA
from apps.multihead_segmentation.dataset import (
    create_data_loaders,
    create_multihead_datasets,
)
from apps.multihead_segmentation.losses import MultiHeadLoss
from apps.multihead_segmentation.model import MultiHeadUNet3D
from apps.multihead_segmentation.multihead_config import DataConfig, LossConfig

# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_kv_pairs(items: Optional[List[str]], value_type):
    """Parse ``["name:value", ...]`` into ``{name: value_type(value)}``, or None."""
    if not items:
        return None
    result = {}
    for item in items:
        name, sep, raw_value = item.partition(":")
        if not sep:
            raise ValueError(f"Expected NAME:VALUE, got {item!r}")
        result[name] = value_type(raw_value)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-Head Segmentation -- 3D U-Net trainer with multiple "
        "independent task heads sharing one backbone",
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
    g.add_argument("--modalities", nargs="+", default=None)
    g.add_argument(
        "--tasks",
        nargs="+",
        metavar="NAME:NUM_CLASSES",
        help="Task definitions, one per segmentation head, e.g. "
        "--tasks organs:14 tumor:2. Each NAME must match a label "
        "subfolder under every subject.",
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
    g.add_argument(
        "--znorm_mask_name",
        default=None,
        help="Labelmap folder whose non-zero voxels define the ZNorm mask "
        "(default: None = normalize over the whole volume; "
        "may equal one of --tasks' names to reuse that task's mask)",
    )
    g.add_argument(
        "--foreground_mask_name",
        default=None,
        help="Labelmap folder whose non-zero voxels are kept after "
        "normalization, with all other voxels set to 0 "
        "(default: None = disabled; may equal one of --tasks' names or "
        "--znorm_mask_name to reuse that mask)",
    )

    g = p.add_argument_group("Patch sampling")
    g.add_argument("--patch_based", action="store_true")
    g.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    g.add_argument("--patch_overlap", type=int, nargs=3, default=[64, 64, 64])
    g.add_argument("--samples_per_volume", type=int, default=4)
    g.add_argument("--queue_max_length", type=int, default=256)
    g.add_argument("--weighted_sampling", action="store_true")

    g = p.add_argument_group("Augmentation")
    g.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable all augmentation (overrides individual toggles)",
    )
    g.add_argument("--no_flip", action="store_true", help="Disable random flip")
    g.add_argument("--no_affine", action="store_true", help="Disable random affine")
    g.add_argument("--no_noise", action="store_true", help="Disable random noise")
    g.add_argument("--no_blur", action="store_true", help="Disable random blur")
    g.add_argument("--no_gamma", action="store_true", help="Disable random gamma")
    g.add_argument(
        "--elastic",
        action="store_true",
        help="Enable random elastic deformation (disabled by default)",
    )

    g = p.add_argument_group("Model")
    g.add_argument("--base_features", type=int, default=32)
    g.add_argument("--trilinear", action="store_true", default=True)
    g.add_argument("--no_trilinear", action="store_false", dest="trilinear")
    g.add_argument(
        "--num_supervision_levels",
        type=int,
        default=1,
        help="1 = standard; >1 = deep supervision with auxiliary heads "
        "(applied identically to every task)",
    )

    g = p.add_argument_group("Loss")
    g.add_argument("--loss", default="dice_ce", choices=["dice", "dice_ce", "ce"])
    g.add_argument("--dice_weight", type=float, default=0.5)
    g.add_argument("--ce_weight", type=float, default=0.5)
    g.add_argument(
        "--task_weights",
        nargs="+",
        default=None,
        metavar="NAME:WEIGHT",
        help="Per-task weight in the cross-task loss sum, e.g. "
        "--task_weights organs:1.0 tumor:2.0 (default: equal weight 1.0/task)",
    )

    g = p.add_argument_group("Optimiser / Scheduler")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch_size", type=int, default=2)
    g.add_argument("--gradient_accumulation", type=bool, default=True)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument(
        "--optimizer", default="adamw", choices=["adam", "adamw", "muon", "sgd"]
    )
    g.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "plateau", "step", "none"]
    )
    g.add_argument("--warmup_epochs", type=int, default=30)
    g.add_argument("--scheduler_patience", type=int, default=10)
    g.add_argument("--scheduler_factor", type=float, default=0.5)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--amp", action="store_true")
    g.add_argument("--ema", action="store_true", help="Enable EMA of model weights")
    g.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")

    g = p.add_argument_group("Early stopping")
    g.add_argument("--early_stopping", action="store_true")
    g.add_argument("--early_stopping_patience", type=int, default=30)

    g = p.add_argument_group("Infrastructure")
    g.add_argument("--output_dir", default="./output")
    g.add_argument("--experiment_name", default="multihead_seg")
    g.add_argument("--num_workers", type=int, default=4)
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--resume", default=None)
    g.add_argument(
        "--no_resume_state",
        dest="resume_state",
        action="store_false",
        default=True,
        help="Load only model weights from the checkpoint; skip optimizer / scheduler / "
        "EMA state. Use for fine-tuning or transfer-learning from a pre-trained checkpoint.",
    )
    g.add_argument(
        "--partial_load",
        default="full",
        choices=["full", "encoder", "encoder_decoder"],
        help="Which model components to initialise from the checkpoint: "
        "full (default) | encoder (encoder + bottleneck) | "
        "encoder_decoder (encoder + bottleneck + decoder, heads re-initialised).",
    )
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--val_interval", type=int, default=1)
    g.add_argument("--save_interval", type=int, default=10)
    g.add_argument("--log_interval", type=int, default=10)
    g.add_argument("--log_images", action="store_true")
    g.add_argument("--log_images_interval", type=int, default=10)
    g.add_argument(
        "--torch_compile",
        action="store_true",
        help="torch.compile(model, dynamic=False) -- requires PyTorch >= 2.0",
    )

    return p.parse_args()


# ── ConfigManager setup ───────────────────────────────────────────────────────


def setup_manager(args: argparse.Namespace) -> ConfigManager:
    """Build and register all configs from parsed CLI arguments."""

    if args.data_root is None or not args.tasks:
        print("Error: the following arguments are required: --data_root, --tasks")

    m = ConfigManager.get()

    dc = DataConfig()
    dc.data_root = args.data_root
    dc.modalities = args.modalities
    dc.tasks = _parse_kv_pairs(args.tasks, int)
    dc.target_spacing = tuple(args.target_spacing) if args.target_spacing else None
    dc.target_shape = tuple(args.target_shape) if args.target_shape else None
    dc.normalization = args.normalization
    dc.znorm_mask_name = args.znorm_mask_name
    dc.foreground_mask_name = args.foreground_mask_name

    pc = PatchConfig()
    pc.enabled = args.patch_based
    pc.size = tuple(args.patch_size)
    pc.overlap = tuple(args.patch_overlap)
    pc.samples_per_volume = args.samples_per_volume
    pc.queue_max_length = args.queue_max_length
    pc.weighted_sampling = args.weighted_sampling

    ac = AugmentConfig()
    ac.enabled = not args.no_augment
    ac.flip = not args.no_flip
    ac.affine = not args.no_affine
    ac.noise = not args.no_noise
    ac.blur = not args.no_blur
    ac.gamma = not args.no_gamma
    ac.elastic = args.elastic

    mc = UNet3DConfig()
    mc.encoder.base_features = args.base_features
    mc.decoder.trilinear = args.trilinear
    mc.num_supervision_levels = args.num_supervision_levels
    # encoder.in_channels is resolved in Trainer after modality auto-discovery

    lc = LossConfig()
    lc.type = args.loss
    lc.dice_weight = args.dice_weight
    lc.ce_weight = args.ce_weight
    lc.task_weights = _parse_kv_pairs(args.task_weights, float)

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
    ic.resume_state = args.resume_state
    ic.partial_load = args.partial_load
    ic.val_interval = args.val_interval
    ic.save_interval = args.save_interval
    ic.log_interval = args.log_interval
    ic.log_images = args.log_images
    ic.log_images_interval = args.log_images_interval
    ic.torch_compile = args.torch_compile

    m.register(ConfigManager.DATA, dc)
    m.register(ConfigManager.PATCH, pc)
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
        ConfigManager.PATCH: PatchConfig,
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


# ── Graph tracing helper ───────────────────────────────────────────────────────


class _GraphTraceWrapper(nn.Module):
    """
    Flattens MultiHeadUNet3D's ``{task: Tensor}`` output into a plain tuple.

    torch.jit's tracer (used internally by SummaryWriter.add_graph) cannot
    reliably handle a dict output, so the inherited ``_log_model_graph``
    needs this wrapper instead of tracing ``self.model`` directly.
    """

    def __init__(self, model: nn.Module, task_names: List[str]) -> None:
        super().__init__()
        self.model = model
        self.task_names = task_names

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[t] for t in self.task_names)


# ── Trainer ────────────────────────────────────────────────────────────────────


class Trainer(LabelMapTrainer):
    """
    Multi-Head Segmentation training loop.

    Subclasses ``apps.labelmap_segmentation.train.Trainer`` -- see module
    docstring for which methods are inherited unchanged versus overridden.
    """

    def __init__(self) -> None:
        # Intentionally does not call super().__init__(): the parent builds a
        # single-task UNet3D + DiceCELoss from DataConfig.label_name/num_classes,
        # neither of which exists on this app's DataConfig. Every individual
        # piece below (configs, dataset factory, _EMA, SummaryWriter, device /
        # seeding helpers) is still reused.
        m = ConfigManager.get()
        self.data_cfg: DataConfig = m.get_config(ConfigManager.DATA)
        self.patch_cfg: PatchConfig = m.get_config(ConfigManager.PATCH)
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
        self.train_ds, self.val_ds, self.test_ds = create_multihead_datasets()
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_ds, self.val_ds
        )
        self.log.info(
            f"Subjects -- train: {len(self.train_ds)}  "
            f"val: {len(self.val_ds)}  test: {len(self.test_ds)}"
        )

        if not self.data_cfg.modalities:
            raise RuntimeError("data_cfg.modalities is empty after dataset creation.")

        self.task_names: List[str] = list(self.data_cfg.tasks.keys())

        # Resolve in_channels now that modalities are known, then persist
        self.model_cfg.encoder.in_channels = len(self.data_cfg.modalities)

        self.model = MultiHeadUNet3D(
            in_channels=self.model_cfg.encoder.in_channels,
            task_num_classes=self.data_cfg.tasks,
            base_features=self.model_cfg.encoder.base_features,
            trilinear=self.model_cfg.decoder.trilinear,
            num_supervision_levels=self.model_cfg.num_supervision_levels,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info(
            f"MultiHeadUNet3D -- tasks={self.task_names}  trainable_parameters={n_params:,}"
        )

        self.criterion = self._build_loss()
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

        # _raw_model always refers to the original MultiHeadUNet3D (never the
        # compiled wrapper) -- see parent class docstring for why.
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
        self.best_val_dice = -float("inf")
        self.stale_epochs = 0

        if self.infra_cfg.resume:
            self._load_checkpoint(self.infra_cfg.resume)

    # ── Model graph logging ──────────────────────────────────────────────────

    def _log_model_graph(self) -> None:
        """Same as the parent's, but traces a tuple-output wrapper instead of
        the model directly -- see _GraphTraceWrapper docstring."""
        try:
            shape = (
                self.patch_cfg.size
                if self.patch_cfg.enabled
                else self.data_cfg.target_shape or (64, 64, 64)
            )
            dummy = torch.zeros(
                1, len(self.data_cfg.modalities), *shape, device=self.device
            )
            wrapper = _GraphTraceWrapper(self.model, self.task_names)
            self.writer.add_graph(wrapper, dummy)
        except Exception as exc:
            self.log.warning(f"Model graph logging skipped: {exc}")

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _build_loss(self) -> nn.Module:
        c = self.loss_cfg
        return MultiHeadLoss(
            task_num_classes=self.data_cfg.tasks,
            loss_type=c.type,
            dice_weight=c.dice_weight,
            ce_weight=c.ce_weight,
            task_weights=c.task_weights,
        )

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _batch_to_tensors(
        self, batch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.cat([batch[m][tio.DATA] for m in self.data_cfg.modalities], dim=1).to(
            self.device
        )
        y = {task: batch[task][tio.DATA].to(self.device) for task in self.task_names}
        return x, y

    def _forward(self, x: torch.Tensor, y: Dict[str, torch.Tensor]):
        """Returns (logits, loss, components) -- logits is {task: Tensor}."""
        if self.scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = self.model(x)
                loss, components = self.criterion(logits, y)
        else:
            logits = self.model(x)
            loss, components = self.criterion(logits, y)
        return logits, loss, components

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
            logits, loss, components = self._forward(x, y)

            loss_scaled = loss / accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            with torch.no_grad():
                task_dice: Dict[str, float] = {}
                for task in self.task_names:
                    d = compute_dice(logits[task], y[task], self.data_cfg.tasks[task])
                    for k, v in d.items():
                        task_dice[f"{task}_{k}"] = v
                mean_dice = float(
                    np.mean([task_dice[f"{t}_mean_dice"] for t in self.task_names])
                )

            batch_m: Dict[str, float] = {
                "loss": loss.item(),
                "mean_dice": mean_dice,
                **task_dice,
            }
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
                self.writer.add_scalar("train/batch_mean_dice", mean_dice, step)
                for k, v in components.items():
                    self.writer.add_scalar(f"train/batch_loss_{k}", v.item(), step)
                self.writer.add_scalar("train/lr", lr, step)
                self.log.info(
                    f"Epoch {epoch:04d} | Batch {i}/{len(self.train_loader)} | "
                    f"loss {loss.item():.4f} | mean_dice {mean_dice:.4f} | lr {lr:.2e}"
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
            logits, loss, _ = self._forward(x, y)

            task_metrics: Dict[str, float] = {"loss": loss.item()}
            per_task_mean_dice = []
            for task in self.task_names:
                nc = self.data_cfg.tasks[task]
                dice = compute_dice(logits[task], y[task], nc)
                iou = compute_iou(logits[task], y[task], nc)
                for k, v in dice.items():
                    task_metrics[f"{task}_{k}"] = v
                for k, v in iou.items():
                    task_metrics[f"{task}_{k}"] = v
                per_task_mean_dice.append(dice["mean_dice"])
            task_metrics["mean_dice"] = float(np.mean(per_task_mean_dice))

            if n == 0:
                accum = {k: 0.0 for k in task_metrics}
            for k, v in task_metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            n += 1

        return {k: v / n for k, v in accum.items()} if n else accum

    # ── TensorBoard logging ───────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict) -> None:
        self.writer.add_scalars(
            "epoch/loss", {"train": train_m["loss"], "val": val_m.get("loss", 0)}, epoch
        )
        for key, value in train_m.items():
            if key.startswith("loss_"):
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

        self.writer.add_scalars(
            "epoch/mean_dice",
            {"train": train_m["mean_dice"], "val": val_m.get("mean_dice", 0)},
            epoch,
        )

        for task in self.task_names:
            nc = self.data_cfg.tasks[task]

            tk = f"{task}_mean_dice"
            scalars: Dict[str, float] = {}
            if tk in train_m:
                scalars["train"] = train_m[tk]
            if tk in val_m:
                scalars["val"] = val_m[tk]
            if scalars:
                self.writer.add_scalars(f"epoch/{task}/mean_dice", scalars, epoch)

            for c in range(1, nc):
                k = f"{task}_dice_class_{c}"
                scalars = {}
                if k in train_m:
                    scalars["train"] = train_m[k]
                if k in val_m:
                    scalars["val"] = val_m[k]
                if scalars:
                    self.writer.add_scalars(f"epoch/{task}/dice_class_{c}", scalars, epoch)
                ki = f"{task}_iou_class_{c}"
                if ki in val_m:
                    self.writer.add_scalar(
                        f"epoch/{task}/val_iou_class_{c}", val_m[ki], epoch
                    )

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
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            x, y = self._batch_to_tensors(batch)
            with torch.no_grad():
                logits = self.model(x)

            img = x[0, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            D, H, W = img.shape

            def _row(*slices):
                return torch.stack([s.unsqueeze(0) for s in slices])

            for task in self.task_names:
                nc = self.data_cfg.tasks[task]
                lg = logits[task]
                pred = (
                    (torch.sigmoid(lg) > 0.5).float()
                    if nc == 1
                    else lg.argmax(dim=1, keepdim=True).float() / max(nc - 1, 1)
                )
                lbl = y[task][0, 0].float() / max(nc - 1, 1)
                prd = pred[0, 0]

                tag_prefix = f"slices/{task}"
                self.writer.add_images(
                    f"{tag_prefix}/axial",
                    _row(img[D // 2], lbl[D // 2], prd[D // 2]),
                    epoch,
                )
                self.writer.add_images(
                    f"{tag_prefix}/coronal",
                    _row(img[:, H // 2, :], lbl[:, H // 2, :], prd[:, H // 2, :]),
                    epoch,
                )
                self.writer.add_images(
                    f"{tag_prefix}/sagittal",
                    _row(img[:, :, W // 2], lbl[:, :, W // 2], prd[:, :, W // 2]),
                    epoch,
                )
        except Exception as exc:
            self.log.warning(f"Image logging failed: {exc}")

    # ── Test evaluation ───────────────────────────────────────────────────────

    def _run_test(self) -> float:
        self.model.eval()
        per_subject = []
        all_mean_dice = []

        for subject in self.test_ds:
            subject_id = getattr(subject, "subject_id", "unknown")

            if self.patch_cfg.enabled:
                grid = tio.GridSampler(
                    subject, self.patch_cfg.size, self.patch_cfg.overlap
                )
                loader = DataLoader(
                    grid, batch_size=self.train_cfg.batch_size, num_workers=0
                )
                aggrs = {
                    task: tio.data.GridAggregator(grid, overlap_mode="average")
                    for task in self.task_names
                }
                for pb in loader:
                    imgs = torch.cat(
                        [pb[m][tio.DATA] for m in self.data_cfg.modalities], dim=1
                    ).to(self.device)
                    logits_dict = self.model(imgs)
                    for task in self.task_names:
                        aggrs[task].add_batch(logits_dict[task], pb[tio.LOCATION])
                logits = {
                    task: aggrs[task].get_output_tensor().unsqueeze(0).to(self.device)
                    for task in self.task_names
                }
            else:
                imgs = (
                    torch.cat(
                        [subject[m][tio.DATA] for m in self.data_cfg.modalities], dim=0
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.model(imgs)

            targets = {
                task: subject[task][tio.DATA].unsqueeze(0).to(self.device)
                for task in self.task_names
            }

            result: Dict[str, float] = {"subject": subject_id}
            per_task_mean_dice = []
            for task in self.task_names:
                nc = self.data_cfg.tasks[task]
                dice = compute_dice(logits[task], targets[task], nc)
                iou = compute_iou(logits[task], targets[task], nc)
                for k, v in dice.items():
                    result[f"{task}_{k}"] = v
                for k, v in iou.items():
                    result[f"{task}_{k}"] = v
                per_task_mean_dice.append(dice["mean_dice"])

            mean_dice = float(np.mean(per_task_mean_dice))
            result["mean_dice"] = mean_dice
            per_subject.append(result)
            all_mean_dice.append(mean_dice)

            per_task_log = "  ".join(
                f'{t} dice {result[f"{t}_mean_dice"]:.4f}' for t in self.task_names
            )
            self.log.info(f"  {subject_id}: {per_task_log}  | overall {mean_dice:.4f}")

        overall_mean = float(np.mean(all_mean_dice))
        overall_std = float(np.std(all_mean_dice))
        self.log.info(
            f"Test -- mean Dice (avg across tasks): {overall_mean:.4f} ± {overall_std:.4f}"
        )

        self.writer.add_scalar("test/mean_dice", overall_mean)
        self.writer.add_scalar("test/std_dice", overall_std)
        for task in self.task_names:
            task_vals = [s[f"{task}_mean_dice"] for s in per_subject]
            self.writer.add_scalar(f"test/{task}_mean_dice", float(np.mean(task_vals)))

        (self.exp_dir / "test_results.json").write_text(
            json.dumps(
                {
                    "mean_dice": overall_mean,
                    "std_dice": overall_std,
                    "per_subject": per_subject,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return overall_mean


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
