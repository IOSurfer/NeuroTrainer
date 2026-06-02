"""
3D U-Net training framework — entry point.

Train from scratch:
    python train.py --data_root /data/brain --num_classes 3 \\
        --architecture unet3d --patch_based \\
        --patch_size 128 128 128 --epochs 200 --experiment_name exp_001

Reproduce a past experiment exactly:
    python train.py --reproduce output/exp_001
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import (
    ConfigManager,
    DataConfig, PatchConfig, AugmentConfig, LossConfig,
    OptimizerConfig, SchedulerConfig, TrainingConfig, InfraConfig,
    build_model_config, available_architectures,
    UNet3DConfig,
)
from config.model.base import EncoderDecoderModelConfig
from dataset import create_data_loaders, create_datasets
from losses import DiceCELoss, DiceLoss
from metrics import compute_dice, compute_iou
from model import UNet3D


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='3D U-Net training framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--reproduce', default=None, metavar='EXP_DIR',
                   help='Load all configs from a saved experiment directory '
                        'and skip all other flags.')

    # ── Data ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Data')
    g.add_argument('--data_root',     default='')
    g.add_argument('--modalities',    nargs='+', default=None,
                   help='Modality folder names (auto-detected when omitted)')
    g.add_argument('--label_name',    default='label')
    g.add_argument('--num_classes',   type=int, default=2)
    g.add_argument('--target_spacing', type=float, nargs=3, default=None,
                   metavar=('X','Y','Z'))
    g.add_argument('--target_shape',  type=int, nargs=3, default=None,
                   metavar=('D','H','W'))
    g.add_argument('--normalization', default='znorm',
                   choices=['znorm', 'rescale', 'none'])

    # ── Patch ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Patch-based training')
    g.add_argument('--patch_based',       action='store_true')
    g.add_argument('--patch_size',        type=int, nargs=3, default=[128, 128, 128],
                   metavar=('D','H','W'))
    g.add_argument('--patch_overlap',     type=int, nargs=3, default=[64, 64, 64],
                   metavar=('D','H','W'), help='Overlap for GridSampler (test)')
    g.add_argument('--samples_per_volume',type=int, default=4)
    g.add_argument('--queue_max_length',  type=int, default=256)
    g.add_argument('--weighted_sampling', action='store_true')

    # ── Augmentation ──────────────────────────────────────────────────────────
    g = p.add_argument_group('Augmentation')
    g.add_argument('--no_augment',          action='store_true')
    g.add_argument('--elastic_deformation', action='store_true')

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Model')
    g.add_argument('--architecture', default='unet3d',
                   help=f'Model architecture. Registered: {available_architectures()}')
    g.add_argument('--base_features',type=int, default=32,
                   help='Encoder base feature channels (for encoder-decoder models)')
    g.add_argument('--trilinear',    action='store_true', default=True)
    g.add_argument('--no_trilinear', action='store_false', dest='trilinear')

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Loss')
    g.add_argument('--loss',        default='dice_ce', choices=['dice', 'dice_ce', 'ce'])
    g.add_argument('--dice_weight', type=float, default=0.5)
    g.add_argument('--ce_weight',   type=float, default=0.5)

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    g = p.add_argument_group('Optimiser / Scheduler')
    g.add_argument('--epochs',             type=int,   default=200)
    g.add_argument('--batch_size',         type=int,   default=2)
    g.add_argument('--lr',                 type=float, default=1e-4)
    g.add_argument('--weight_decay',       type=float, default=1e-5)
    g.add_argument('--optimizer',          default='adamw', choices=['adam', 'adamw', 'sgd'])
    g.add_argument('--scheduler',          default='cosine',
                   choices=['cosine', 'plateau', 'step', 'none'])
    g.add_argument('--warmup_epochs',      type=int,   default=5)
    g.add_argument('--scheduler_patience', type=int,   default=10)
    g.add_argument('--scheduler_factor',   type=float, default=0.5)
    g.add_argument('--grad_clip',          type=float, default=1.0)
    g.add_argument('--amp',                action='store_true')

    # ── Early stopping ────────────────────────────────────────────────────────
    g = p.add_argument_group('Early stopping')
    g.add_argument('--early_stopping',          action='store_true')
    g.add_argument('--early_stopping_patience', type=int, default=30)

    # ── Infrastructure ────────────────────────────────────────────────────────
    g = p.add_argument_group('Infrastructure')
    g.add_argument('--output_dir',          default='./output')
    g.add_argument('--experiment_name',     default='unet3d')
    g.add_argument('--num_workers',         type=int, default=4)
    g.add_argument('--device',              default='auto', choices=['auto', 'cpu', 'cuda'])
    g.add_argument('--resume',              default=None, metavar='CHECKPOINT')
    g.add_argument('--seed',                type=int, default=42)
    g.add_argument('--val_interval',        type=int, default=1)
    g.add_argument('--save_interval',       type=int, default=10)
    g.add_argument('--log_interval',        type=int, default=10)
    g.add_argument('--log_images',          action='store_true')
    g.add_argument('--log_images_interval', type=int, default=10)

    return p.parse_args()


# ── Config builder ─────────────────────────────────────────────────────────────

def build_manager_from_args(args) -> ConfigManager:
    """
    Populate the ConfigManager singleton from argparse values.
    Each config type is registered under its well-known type name.
    """
    m = ConfigManager.get()

    # Data
    dcfg = DataConfig()
    dcfg.data_root      = args.data_root
    dcfg.modalities     = args.modalities
    dcfg.label_name     = args.label_name
    dcfg.num_classes    = args.num_classes
    dcfg.target_spacing = tuple(args.target_spacing) if args.target_spacing else None
    dcfg.target_shape   = tuple(args.target_shape)   if args.target_shape   else None
    dcfg.normalization  = args.normalization
    m.register(ConfigManager.DATA, dcfg)

    # Patch
    pcfg = PatchConfig()
    pcfg.enabled            = args.patch_based
    pcfg.size               = tuple(args.patch_size)
    pcfg.overlap            = tuple(args.patch_overlap)
    pcfg.samples_per_volume = args.samples_per_volume
    pcfg.queue_max_length   = args.queue_max_length
    pcfg.weighted_sampling  = args.weighted_sampling
    m.register(ConfigManager.PATCH, pcfg)

    # Augment
    acfg = AugmentConfig()
    acfg.enabled             = not args.no_augment
    acfg.elastic_deformation = args.elastic_deformation
    m.register(ConfigManager.AUGMENT, acfg)

    # Model — dynamic from architecture registry
    model_cfg = build_model_config(args.architecture)
    if hasattr(model_cfg, 'num_classes'):
        model_cfg.num_classes = args.num_classes
    if isinstance(model_cfg, EncoderDecoderModelConfig):
        model_cfg.encoder.base_features = args.base_features
        model_cfg.decoder.trilinear     = args.trilinear
        # in_channels resolved later (after modality auto-detection)
    m.register(ConfigManager.MODEL, model_cfg)

    # Loss
    lcfg = LossConfig()
    lcfg.type        = args.loss
    lcfg.dice_weight = args.dice_weight
    lcfg.ce_weight   = args.ce_weight
    m.register(ConfigManager.LOSS, lcfg)

    # Optimizer
    ocfg = OptimizerConfig()
    ocfg.type         = args.optimizer
    ocfg.lr           = args.lr
    ocfg.weight_decay = args.weight_decay
    ocfg.grad_clip    = args.grad_clip
    m.register(ConfigManager.OPTIMIZER, ocfg)

    # Scheduler
    scfg = SchedulerConfig()
    scfg.type          = args.scheduler
    scfg.warmup_epochs = args.warmup_epochs
    scfg.patience      = args.scheduler_patience
    scfg.factor        = args.scheduler_factor
    m.register(ConfigManager.SCHEDULER, scfg)

    # Training
    tcfg = TrainingConfig()
    tcfg.epochs                  = args.epochs
    tcfg.batch_size              = args.batch_size
    tcfg.amp                     = args.amp
    tcfg.early_stopping          = args.early_stopping
    tcfg.early_stopping_patience = args.early_stopping_patience
    m.register(ConfigManager.TRAINING, tcfg)

    # Infra
    icfg = InfraConfig()
    icfg.output_dir          = args.output_dir
    icfg.experiment_name     = args.experiment_name
    icfg.num_workers         = args.num_workers
    icfg.device              = args.device
    icfg.seed                = args.seed
    icfg.resume              = args.resume
    icfg.val_interval        = args.val_interval
    icfg.save_interval       = args.save_interval
    icfg.log_interval        = args.log_interval
    icfg.log_images          = args.log_images
    icfg.log_images_interval = args.log_images_interval
    m.register(ConfigManager.INFRA, icfg)

    return m


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training / validation / test loop.

    Reads all parameters from the global :class:`~config.ConfigManager`;
    no args object is passed after construction.
    """

    def __init__(self) -> None:
        m                = ConfigManager.get()
        self.m           = m
        self.data_cfg:   DataConfig      = m.get_config(ConfigManager.DATA)
        self.patch_cfg:  PatchConfig     = m.get_config(ConfigManager.PATCH)
        self.aug_cfg:    AugmentConfig   = m.get_config(ConfigManager.AUGMENT)
        self.loss_cfg:   LossConfig      = m.get_config(ConfigManager.LOSS)
        self.opt_cfg:    OptimizerConfig  = m.get_config(ConfigManager.OPTIMIZER)
        self.sched_cfg:  SchedulerConfig = m.get_config(ConfigManager.SCHEDULER)
        self.train_cfg:  TrainingConfig  = m.get_config(ConfigManager.TRAINING)
        self.infra_cfg:  InfraConfig     = m.get_config(ConfigManager.INFRA)

        self.device = self._resolve_device()
        self._seed_everything()

        # ── Directories ───────────────────────────────────────────────────────
        self.exp_dir  = Path(self.infra_cfg.output_dir) / self.infra_cfg.experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.tb_dir   = self.exp_dir / 'tensorboard'
        for d in (self.ckpt_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # ── Save all configs to experiment dir immediately ────────────────────
        m.save_all(str(self.exp_dir))
        self.log.info(f'Configs saved → {self.exp_dir}/')

        # ── Datasets ──────────────────────────────────────────────────────────
        self.log.info('Building datasets…')
        self.train_ds, self.val_ds, self.test_ds = create_datasets()
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_ds, self.val_ds
        )
        self.log.info(
            f'Subjects — train:{len(self.train_ds)}  '
            f'val:{len(self.val_ds)}  test:{len(self.test_ds)}'
        )

        if not self.data_cfg.modalities:
            raise RuntimeError('data_cfg.modalities is empty after create_datasets().')

        # Resolve encoder in_channels now that modalities are known
        model_cfg = m.get_model_config()
        if isinstance(model_cfg, EncoderDecoderModelConfig):
            model_cfg.encoder.in_channels = len(self.data_cfg.modalities)

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = self._build_model().to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info(f'UNet3D — trainable parameters: {n_params:,}')

        # ── Loss / optimiser / scheduler ──────────────────────────────────────
        self.criterion  = self._build_loss()
        self.optimizer  = self._build_optimizer()
        self.scheduler  = self._build_scheduler()
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler('cuda')
            if self.train_cfg.amp and self.device.type == 'cuda' else None
        )

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self._log_model_graph()

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch   = 0
        self.best_val_dice = -float('inf')
        self.stale_epochs  = 0

        if self.infra_cfg.resume:
            self._load_checkpoint(self.infra_cfg.resume)

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        spec = self.infra_cfg.device
        if spec == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(spec)

    def _seed_everything(self) -> None:
        import random
        s = self.infra_cfg.seed
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.exp_dir / 'training.log'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.log = logging.getLogger('trainer')

    # ── Component builders ────────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        """Build the neural network from the registered ModelConfig."""
        model_cfg = self.m.get_model_config()
        arch = getattr(model_cfg, 'architecture', 'unet3d')

        if arch == 'unet3d':
            assert isinstance(model_cfg, UNet3DConfig)
            return UNet3D(
                in_channels=model_cfg.encoder.in_channels,
                num_classes=model_cfg.num_classes,
                base_features=model_cfg.encoder.base_features,
                trilinear=model_cfg.decoder.trilinear,
            )
        raise NotImplementedError(
            f'No model builder for architecture {arch!r}. '
            f'Registered: {available_architectures()}'
        )

    def _build_loss(self) -> nn.Module:
        lc = self.loss_cfg
        nc = self.data_cfg.num_classes
        if lc.type == 'dice':
            return DiceLoss(nc)
        if lc.type == 'dice_ce':
            return DiceCELoss(nc, lc.dice_weight, lc.ce_weight)
        return nn.BCEWithLogitsLoss() if nc == 1 else nn.CrossEntropyLoss()

    def _build_optimizer(self) -> optim.Optimizer:
        oc = self.opt_cfg
        params = self.model.parameters()
        if oc.type == 'adam':
            return optim.Adam(params,  lr=oc.lr, weight_decay=oc.weight_decay)
        if oc.type == 'adamw':
            return optim.AdamW(params, lr=oc.lr, weight_decay=oc.weight_decay)
        return optim.SGD(params, lr=oc.lr, momentum=0.9,
                         weight_decay=oc.weight_decay, nesterov=True)

    def _build_scheduler(self):
        sc = self.sched_cfg
        if sc.type == 'cosine':
            T = max(self.train_cfg.epochs - sc.warmup_epochs, 1)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T)
        if sc.type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=sc.factor, patience=sc.patience)
        if sc.type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sc.patience, gamma=sc.factor)
        return None

    # ── TensorBoard helpers ───────────────────────────────────────────────────

    def _log_model_graph(self) -> None:
        try:
            shape = (
                self.patch_cfg.size if self.patch_cfg.enabled
                else self.data_cfg.target_shape if self.data_cfg.target_shape
                else (64, 64, 64)
            )
            dummy = torch.zeros(
                1, len(self.data_cfg.modalities), *shape, device=self.device
            )
            self.writer.add_graph(self.model, dummy)
        except Exception as exc:
            self.log.warning(f'Model graph logging skipped: {exc}')

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict) -> None:
        nc = self.data_cfg.num_classes
        self.writer.add_scalars(
            'epoch/loss',
            {'train': train_m['loss'], 'val': val_m.get('loss', 0)}, epoch)
        self.writer.add_scalars(
            'epoch/mean_dice',
            {'train': train_m['mean_dice'], 'val': val_m.get('mean_dice', 0)}, epoch)
        for c in range(1, nc):
            key = f'dice_class_{c}'
            if key in val_m:
                self.writer.add_scalars(
                    f'epoch/dice_class_{c}',
                    {'train': train_m.get(key, 0), 'val': val_m[key]}, epoch)
            iou_key = f'iou_class_{c}'
            if iou_key in val_m:
                self.writer.add_scalar(f'epoch/val_iou_class_{c}', val_m[iou_key], epoch)
        self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'weights/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, epoch)

    def _log_images(self, epoch: int) -> None:
        nc = self.data_cfg.num_classes
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            x, y  = self._batch_to_tensors(batch)
            with torch.no_grad():
                logits = self.model(x)
                pred   = (
                    (torch.sigmoid(logits) > 0.5).float()
                    if nc == 1
                    else logits.argmax(dim=1, keepdim=True).float() / max(nc - 1, 1)
                )
            img = x[0, 0];  lbl = y[0, 0].float() / max(nc - 1, 1);  prd = pred[0, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            D, H, W = img.shape

            def _row(*slices):
                return torch.stack([s.unsqueeze(0) for s in slices])

            self.writer.add_images('slices/axial',    _row(img[D//2],       lbl[D//2],       prd[D//2]),       epoch)
            self.writer.add_images('slices/coronal',  _row(img[:, H//2, :], lbl[:, H//2, :], prd[:, H//2, :]), epoch)
            self.writer.add_images('slices/sagittal', _row(img[:, :, W//2], lbl[:, :, W//2], prd[:, :, W//2]), epoch)
        except Exception as exc:
            self.log.warning(f'Image logging failed: {exc}')

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _batch_to_tensors(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.cat(
            [batch[m][tio.DATA] for m in self.data_cfg.modalities], dim=1
        ).to(self.device)
        y = (batch[self.data_cfg.label_name][tio.DATA].to(self.device)
             if self.data_cfg.label_name in batch else None)
        return x, y

    def _forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = self.model(x); loss = self.criterion(logits, y)
        else:
            logits = self.model(x); loss = self.criterion(logits, y)
        return logits, loss

    # ── LR warmup ─────────────────────────────────────────────────────────────

    def _apply_warmup(self, epoch: int) -> None:
        sc = self.sched_cfg
        if epoch < sc.warmup_epochs:
            factor = (epoch + 1) / sc.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.opt_cfg.lr * factor

    def _step_scheduler(self, val_dice: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_dice)
        else:
            self.scheduler.step()

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self._apply_warmup(epoch)
        nc = self.data_cfg.num_classes
        accum_loss = accum_dice = n = 0.0

        for i, batch in enumerate(self.train_loader):
            x, y = self._batch_to_tensors(batch)
            self.optimizer.zero_grad()
            logits, loss = self._forward(x, y)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.opt_cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.opt_cfg.grad_clip)
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                loss.backward()
                if self.opt_cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.opt_cfg.grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                dice_scores = compute_dice(logits, y, nc)

            accum_loss += loss.item(); accum_dice += dice_scores['mean_dice']; n += 1

            if i % self.infra_cfg.log_interval == 0:
                lr   = self.optimizer.param_groups[0]['lr']
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
                self.writer.add_scalar('train/batch_dice', dice_scores['mean_dice'], step)
                self.writer.add_scalar('train/lr', lr, step)
                self.log.info(
                    f'Epoch {epoch:04d} | Batch {i}/{len(self.train_loader)} | '
                    f'loss {loss.item():.4f} | dice {dice_scores["mean_dice"]:.4f} | '
                    f'lr {lr:.2e}'
                )

        return {'loss': accum_loss / n, 'mean_dice': accum_dice / n}

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self) -> Dict:
        self.model.eval()
        nc = self.data_cfg.num_classes
        accum: Dict[str, float] = {}
        n = 0

        for batch in self.val_loader:
            x, y = self._batch_to_tensors(batch)
            logits, loss = self._forward(x, y)
            dice = compute_dice(logits, y, nc)
            iou  = compute_iou(logits,  y, nc)

            if n == 0:
                accum = {'loss': 0.0, **{k: 0.0 for k in dice}, **{k: 0.0 for k in iou}}
            accum['loss'] += loss.item()
            for k, v in {**dice, **iou}.items():
                accum[k] += v
            n += 1

        return {k: v / n for k, v in accum.items()} if n else accum

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_m: Dict, is_best: bool) -> None:
        ckpt = {
            'epoch':         epoch,
            'model':         self.model.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'val_metrics':   val_m,
            'best_val_dice': self.best_val_dice,
            'config_types':  self.m.registered_types(),
        }
        if self.scheduler:  ckpt['scheduler'] = self.scheduler.state_dict()
        if self.scaler:     ckpt['scaler']    = self.scaler.state_dict()

        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if epoch % self.infra_cfg.save_interval == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:04d}.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            self.log.info(f'  ↳ New best — val_dice {val_m["mean_dice"]:.4f}')

    def _load_checkpoint(self, path: str) -> None:
        self.log.info(f'Resuming from {path}')
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch   = ckpt['epoch'] + 1
        self.best_val_dice = ckpt.get('best_val_dice', -float('inf'))
        if 'scheduler' in ckpt and self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt['scaler'])

    # ── Test evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self) -> float:
        best_ckpt = self.ckpt_dir / 'best.pth'
        if best_ckpt.exists():
            self.log.info('Loading best checkpoint for test evaluation…')
            self._load_checkpoint(str(best_ckpt))

        self.model.eval()
        nc = self.data_cfg.num_classes
        all_dice: list = []
        all_iou:  list = []

        for subject in self.test_ds:
            subject_id = getattr(subject, 'subject_id', 'unknown')

            if self.patch_cfg.enabled:
                grid = tio.GridSampler(subject, self.patch_cfg.size, self.patch_cfg.overlap)
                agg  = tio.data.GridAggregator(grid, overlap_mode='average')
                for pb in DataLoader(grid, batch_size=self.train_cfg.batch_size, num_workers=0):
                    imgs = torch.cat(
                        [pb[m][tio.DATA] for m in self.data_cfg.modalities], dim=1
                    ).to(self.device)
                    agg.add_batch(self.model(imgs), pb[tio.LOCATION])
                logits = agg.get_output_tensor().unsqueeze(0).to(self.device)
                label  = subject[self.data_cfg.label_name][tio.DATA].unsqueeze(0).to(self.device)
            else:
                imgs = torch.cat(
                    [subject[m][tio.DATA] for m in self.data_cfg.modalities], dim=0
                ).unsqueeze(0).to(self.device)
                logits = self.model(imgs)
                label  = subject[self.data_cfg.label_name][tio.DATA].unsqueeze(0).to(self.device)

            dice = compute_dice(logits, label, nc)
            iou  = compute_iou(logits, label, nc)
            all_dice.append(dice['mean_dice']); all_iou.append(iou['mean_iou'])
            self.log.info(
                f'  {subject_id}: dice {dice["mean_dice"]:.4f}  iou {iou["mean_iou"]:.4f}'
            )

        mean_dice = float(np.mean(all_dice))
        std_dice  = float(np.std(all_dice))
        mean_iou  = float(np.mean(all_iou))
        self.log.info(
            f'Test — mean Dice: {mean_dice:.4f} ± {std_dice:.4f}  mean IoU: {mean_iou:.4f}'
        )
        self.writer.add_scalar('test/mean_dice', mean_dice)
        self.writer.add_scalar('test/std_dice',  std_dice)
        self.writer.add_scalar('test/mean_iou',  mean_iou)

        results = {
            'mean_dice': mean_dice, 'std_dice': std_dice, 'mean_iou': mean_iou,
            'per_subject': [
                {'subject': s.subject_id if hasattr(s, 'subject_id') else f'sub_{i}',
                 'dice': float(d), 'iou': float(u)}
                for i, (s, d, u) in enumerate(zip(self.test_ds, all_dice, all_iou))
            ],
        }
        (self.exp_dir / 'test_results.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8'
        )
        return mean_dice

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self) -> None:
        self.log.info(f'Device: {self.device}')
        self.log.info(f'Experiment dir: {self.exp_dir}')

        # Write config summary to TensorBoard
        self.writer.add_text(
            'config/manager_summary',
            f'```\n{self.m.__repr__()}\n```',
        )
        for type_name, config in self.m._registry.items():
            self.writer.add_text(
                f'config/{type_name}',
                f'```\n{config.summary()}\n```',
            )

        tc = self.train_cfg; ic = self.infra_cfg; sc = self.sched_cfg
        for epoch in range(self.start_epoch, tc.epochs):
            t0 = time.perf_counter()
            train_m = self.train_epoch(epoch)

            val_m: Dict = {}
            if (epoch + 1) % ic.val_interval == 0:
                val_m = self.val_epoch()
                self._log_epoch(epoch, train_m, val_m)

                if ic.log_images and (epoch + 1) % ic.log_images_interval == 0:
                    self._log_images(epoch)

                if epoch >= sc.warmup_epochs:
                    self._step_scheduler(val_m.get('mean_dice', 0.0))

                current_dice = val_m.get('mean_dice', 0.0)
                is_best = current_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = current_dice; self.stale_epochs = 0
                else:
                    self.stale_epochs += 1

                self._save_checkpoint(epoch, val_m, is_best)
                elapsed = time.perf_counter() - t0
                self.log.info(
                    f'Epoch {epoch:04d}/{tc.epochs} | '
                    f'train_loss {train_m["loss"]:.4f}  train_dice {train_m["mean_dice"]:.4f} | '
                    f'val_loss {val_m.get("loss",0):.4f}  val_dice {current_dice:.4f} | '
                    f'best {self.best_val_dice:.4f} | stale {self.stale_epochs} | {elapsed:.0f}s'
                )

                if tc.early_stopping and self.stale_epochs >= tc.early_stopping_patience:
                    self.log.info(
                        f'Early stopping after {self.stale_epochs} epochs without improvement.'
                    )
                    break

        self.writer.add_hparams(
            hparam_dict={
                'lr':            self.opt_cfg.lr,
                'batch_size':    tc.batch_size,
                'base_features': (self.m.get_model_config().encoder.base_features
                                  if isinstance(self.m.get_model_config(), EncoderDecoderModelConfig)
                                  else 32),
                'optimizer':     self.opt_cfg.type,
                'loss':          self.loss_cfg.type,
                'patch_based':   self.patch_cfg.enabled,
            },
            metric_dict={'hparam/best_val_dice': self.best_val_dice},
        )
        self.writer.close()
        self.log.info('Training complete. Running test evaluation…')
        self.test()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.reproduce:
        m = ConfigManager.from_directory(args.reproduce)
        print(f'Reproducing experiment from: {args.reproduce}')
    else:
        m = build_manager_from_args(args)

    m.print_all()
    Trainer().run()


if __name__ == '__main__':
    main()
