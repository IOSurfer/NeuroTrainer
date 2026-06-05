"""
LabelMap Segmentation -- training entry point.

Usage
-----
Train from scratch:
    python -m apps.labelmap_segmentation.train \\
        --data_root /data/brain_mri --num_classes 3 \\
        --patch_based --patch_size 128 128 128 \\
        --experiment_name exp_001 --log_images

Reproduce a past experiment exactly:
    python -m apps.labelmap_segmentation.train \\
        --reproduce output/exp_001/

See --help for all options.
"""
from __future__ import annotations

import argparse
import contextlib
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

from configuration.manager import ConfigManager
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
from apps.labelmap_segmentation.dataset import (
    create_data_loaders,
    create_labelmap_segmentation_datasets,
)
from apps.labelmap_segmentation.losses import DiceCELoss, DiceLoss
from apps.labelmap_segmentation.metrics import compute_dice, compute_iou
from apps.labelmap_segmentation.model import UNet3D


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='LabelMap Segmentation -- 3D U-Net trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--reproduce', default=None, metavar='EXP_DIR',
                   help='Reproduce an experiment by loading its saved configs. '
                        'All other flags are ignored when this is set.')

    g = p.add_argument_group('Data')
    g.add_argument('--data_root')
    g.add_argument('--modalities',     nargs='+', default=None)
    g.add_argument('--label_name',     default='label')
    g.add_argument('--num_classes',    type=int)
    g.add_argument('--target_spacing', type=float, nargs=3,
                   default=None, metavar=('X', 'Y', 'Z'))
    g.add_argument('--target_shape',   type=int, nargs=3,
                   default=None, metavar=('D', 'H', 'W'))
    g.add_argument('--normalization',  default='znorm',
                   choices=['znorm', 'rescale', 'none'])

    g = p.add_argument_group('Patch sampling')
    g.add_argument('--patch_based',        action='store_true')
    g.add_argument('--patch_size',         type=int,
                   nargs=3, default=[128, 128, 128])
    g.add_argument('--patch_overlap',      type=int,
                   nargs=3, default=[64, 64, 64])
    g.add_argument('--samples_per_volume', type=int, default=4)
    g.add_argument('--queue_max_length',   type=int, default=256)
    g.add_argument('--weighted_sampling',  action='store_true')

    g = p.add_argument_group('Augmentation')
    g.add_argument('--no_augment',     action='store_true',
                   help='Disable all augmentation (overrides individual toggles)')
    g.add_argument('--no_flip',        action='store_true', help='Disable random flip')
    g.add_argument('--no_affine',      action='store_true', help='Disable random affine')
    g.add_argument('--no_noise',       action='store_true', help='Disable random noise')
    g.add_argument('--no_blur',        action='store_true', help='Disable random blur')
    g.add_argument('--no_gamma',       action='store_true', help='Disable random gamma')
    g.add_argument('--elastic',        action='store_true',
                   help='Enable random elastic deformation (disabled by default)')

    g = p.add_argument_group('Model')
    g.add_argument('--base_features', type=int, default=32)
    g.add_argument('--trilinear',     action='store_true', default=True)
    g.add_argument('--no_trilinear',  action='store_false', dest='trilinear')

    g = p.add_argument_group('Loss')
    g.add_argument('--loss',        default='dice_ce',
                   choices=['dice', 'dice_ce', 'ce'])
    g.add_argument('--dice_weight', type=float, default=0.5)
    g.add_argument('--ce_weight',   type=float, default=0.5)

    g = p.add_argument_group('Optimiser / Scheduler')
    g.add_argument('--epochs',              type=int,   default=200)
    g.add_argument('--batch_size',          type=int,   default=2)
    g.add_argument('--gradient_accumulation', type=bool, default=True)
    g.add_argument('--lr',                  type=float, default=1e-4)
    g.add_argument('--weight_decay',        type=float, default=1e-5)
    g.add_argument('--optimizer',           default='adamw',
                   choices=['adam', 'adamw', 'muon', 'sgd'])
    g.add_argument('--scheduler',           default='cosine',
                   choices=['cosine', 'plateau', 'step', 'none'])
    g.add_argument('--warmup_epochs',       type=int,   default=5)
    g.add_argument('--scheduler_patience',  type=int,   default=10)
    g.add_argument('--scheduler_factor',    type=float, default=0.5)
    g.add_argument('--grad_clip',           type=float, default=1.0)
    g.add_argument('--amp',                 action='store_true')
    g.add_argument('--ema',                 action='store_true',
                   help='Enable EMA of model weights')
    g.add_argument('--ema_decay',           type=float, default=0.999,
                   help='EMA decay factor')

    g = p.add_argument_group('Early stopping')
    g.add_argument('--early_stopping',         action='store_true')
    g.add_argument('--early_stopping_patience', type=int, default=30)

    g = p.add_argument_group('Infrastructure')
    g.add_argument('--output_dir',          default='./output')
    g.add_argument('--experiment_name',     default='labelmap_seg')
    g.add_argument('--num_workers',         type=int, default=4)
    g.add_argument('--device',              default='auto',
                   choices=['auto', 'cpu', 'cuda'])
    g.add_argument('--resume',              default=None)
    g.add_argument('--seed',                type=int, default=42)
    g.add_argument('--val_interval',        type=int, default=1)
    g.add_argument('--save_interval',       type=int, default=10)
    g.add_argument('--log_interval',        type=int, default=10)
    g.add_argument('--log_images',          action='store_true')
    g.add_argument('--log_images_interval', type=int, default=10)

    return p.parse_args()


# ── ConfigManager setup ───────────────────────────────────────────────────────

def setup_manager(args: argparse.Namespace) -> ConfigManager:
    """Build and register all configs from parsed CLI arguments."""

    if args.data_root is None or args.num_classes:
        print("Error: the following arguments are required: --data_root, --num_classes")

    m = ConfigManager.get()

    dc = DataConfig()
    dc.data_root = args.data_root
    dc.modalities = args.modalities
    dc.label_name = args.label_name
    dc.num_classes = args.num_classes
    dc.target_spacing = tuple(
        args.target_spacing) if args.target_spacing else None
    dc.target_shape = tuple(args.target_shape) if args.target_shape else None
    dc.normalization = args.normalization

    pc = PatchConfig()
    pc.enabled = args.patch_based
    pc.size = tuple(args.patch_size)
    pc.overlap = tuple(args.patch_overlap)
    pc.samples_per_volume = args.samples_per_volume
    pc.queue_max_length = args.queue_max_length
    pc.weighted_sampling = args.weighted_sampling

    ac = AugmentConfig()
    ac.enabled = not args.no_augment
    ac.flip    = not args.no_flip
    ac.affine  = not args.no_affine
    ac.noise   = not args.no_noise
    ac.blur    = not args.no_blur
    ac.gamma   = not args.no_gamma
    ac.elastic = args.elastic

    mc = UNet3DConfig()
    mc.encoder.base_features = args.base_features
    mc.decoder.trilinear = args.trilinear
    # encoder.in_channels is resolved in Trainer after modality auto-discovery

    lc = LossConfig()
    lc.type = args.loss
    lc.dice_weight = args.dice_weight
    lc.ce_weight = args.ce_weight

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
    ic.val_interval = args.val_interval
    ic.save_interval = args.save_interval
    ic.log_interval = args.log_interval
    ic.log_images = args.log_images
    ic.log_images_interval = args.log_images_interval

    m.register(ConfigManager.DATA,      dc)
    m.register(ConfigManager.PATCH,     pc)
    m.register(ConfigManager.AUGMENT,   ac)
    m.register(ConfigManager.MODEL,     mc)
    m.register(ConfigManager.LOSS,      lc)
    m.register(ConfigManager.OPTIMIZER, oc)
    m.register(ConfigManager.SCHEDULER, sc)
    m.register(ConfigManager.TRAINING,  tc)
    m.register(ConfigManager.INFRA,     ic)

    return m


def load_manager_from_directory(directory: str) -> ConfigManager:
    """Restore the ConfigManager from a previously saved experiment directory."""
    m = ConfigManager.get()
    dir_path = Path(directory)

    type_map = {
        ConfigManager.DATA:      DataConfig,
        ConfigManager.PATCH:     PatchConfig,
        ConfigManager.AUGMENT:   AugmentConfig,
        ConfigManager.MODEL:     UNet3DConfig,
        ConfigManager.LOSS:      LossConfig,
        ConfigManager.OPTIMIZER: OptimizerConfig,
        ConfigManager.SCHEDULER: SchedulerConfig,
        ConfigManager.TRAINING:  TrainingConfig,
        ConfigManager.INFRA:     InfraConfig,
    }
    for type_name, cfg_cls in type_map.items():
        cfg = cfg_cls()
        path = dir_path / f'{type_name.lower()}.json'
        if path.exists():
            cfg.load(str(path))
        m.register(type_name, cfg)

    return m


# ── EMA ────────────────────────────────────────────────────────────────────────

class _EMA:
    """Shadow-parameter EMA for any nn.Module."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.data.clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> dict:
        return {
            'decay':  self.decay,
            'shadow': {n: v.cpu() for n, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay = state['decay']
        self.shadow = {n: v.to(device) for n, v in state['shadow'].items()}


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    LabelMap Segmentation training loop.

    All behaviour is driven by configs registered in the ConfigManager;
    no arguments are passed after construction.
    """

    def __init__(self) -> None:
        m = ConfigManager.get()
        self.data_cfg:  DataConfig = m.get_config(ConfigManager.DATA)
        self.patch_cfg: PatchConfig = m.get_config(ConfigManager.PATCH)
        self.aug_cfg:   AugmentConfig = m.get_config(ConfigManager.AUGMENT)
        self.model_cfg: UNet3DConfig = m.get_config(ConfigManager.MODEL)
        self.loss_cfg:  LossConfig = m.get_config(ConfigManager.LOSS)
        self.opt_cfg:   OptimizerConfig = m.get_config(ConfigManager.OPTIMIZER)
        self.sched_cfg: SchedulerConfig = m.get_config(ConfigManager.SCHEDULER)
        self.train_cfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
        self.infra_cfg: InfraConfig = m.get_config(ConfigManager.INFRA)

        self.device = self._resolve_device()
        self._seed_everything()

        self.exp_dir = Path(self.infra_cfg.output_dir) / \
            self.infra_cfg.experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.tb_dir = self.exp_dir / 'tensorboard'
        for d in (self.ckpt_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        ConfigManager.get().save_all(str(self.exp_dir))
        self.log.info(f'Configs saved to {self.exp_dir}')

        self.log.info('Building datasets...')
        self.train_ds, self.val_ds, self.test_ds = create_labelmap_segmentation_datasets()
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_ds, self.val_ds)
        self.log.info(
            f'Subjects -- train: {len(self.train_ds)}  '
            f'val: {len(self.val_ds)}  test: {len(self.test_ds)}'
        )

        if not self.data_cfg.modalities:
            raise RuntimeError(
                'data_cfg.modalities is empty after dataset creation.')

        # Resolve in_channels now that modalities are known, then persist
        self.model_cfg.encoder.in_channels = len(self.data_cfg.modalities)

        self.model = UNet3D(
            in_channels=self.model_cfg.encoder.in_channels,
            num_classes=self.data_cfg.num_classes,
            base_features=self.model_cfg.encoder.base_features,
            trilinear=self.model_cfg.decoder.trilinear,
        ).to(self.device)

        n_params = sum(p.numel()
                       for p in self.model.parameters() if p.requires_grad)
        self.log.info(f'UNet3D -- trainable parameters: {n_params:,}')

        self.criterion = self._build_loss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler('cuda')
            if self.train_cfg.amp and self.device.type == 'cuda'
            else None
        )

        self.ema: Optional[_EMA] = (
            _EMA(self.model, self.train_cfg.ema_decay)
            if self.train_cfg.ema else None
        )
        if self.ema:
            self.log.info(f'EMA enabled -- decay {self.train_cfg.ema_decay}')

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self._log_model_graph()

        self.start_epoch = 0
        self.best_val_dice = -float('inf')
        self.stale_epochs = 0

        if self.infra_cfg.resume:
            self._load_checkpoint(self.infra_cfg.resume)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        spec = self.infra_cfg.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if spec == 'auto' else torch.device(spec)

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
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.exp_dir / 'training.log'),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.log = logging.getLogger('trainer')

    def _build_loss(self) -> nn.Module:
        c = self.loss_cfg
        n = self.data_cfg.num_classes
        if c.type == 'dice':
            return DiceLoss(n)
        if c.type == 'dice_ce':
            return DiceCELoss(n, c.dice_weight, c.ce_weight)
        return nn.BCEWithLogitsLoss() if n == 1 else nn.CrossEntropyLoss()

    def _build_optimizer(self) -> optim.Optimizer:
        c, p = self.opt_cfg, self.model.parameters()
        if c.type == 'adam':
            return optim.Adam(p, lr=c.lr, weight_decay=c.weight_decay)
        if c.type == 'adamw':
            return optim.AdamW(p, lr=c.lr, weight_decay=c.weight_decay)
        if c.type == 'muon':
            return optim.Muon(p, lr=c.lr, weight_decay=c.weight_decay)
        return optim.SGD(p, lr=c.lr, momentum=0.9, weight_decay=c.weight_decay, nesterov=True)

    def _build_scheduler(self):
        c, o, e = self.sched_cfg, self.optimizer, self.train_cfg.epochs
        if c.type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(e - c.warmup_epochs, 1))
        if c.type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(o, mode='max',
                                                        factor=c.factor, patience=c.patience)
        if c.type == 'step':
            return optim.lr_scheduler.StepLR(o, step_size=c.patience, gamma=c.factor)
        return None

    @contextlib.contextmanager
    def _ema_scope(self):
        """Temporarily swap EMA shadow weights into the model, then restore."""
        if self.ema is None:
            yield
            return
        backup = {n: p.data.clone()
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.ema.apply_shadow(self.model)
        try:
            yield
        finally:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])

    def _log_model_graph(self) -> None:
        try:
            shape = (self.patch_cfg.size if self.patch_cfg.enabled
                     else self.data_cfg.target_shape or (64, 64, 64))
            dummy = torch.zeros(
                1, len(self.data_cfg.modalities), *shape, device=self.device)
            self.writer.add_graph(self.model, dummy)
        except Exception as exc:
            self.log.warning(f'Model graph logging skipped: {exc}')

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
                logits = self.model(x)
                loss = self.criterion(logits, y)
        else:
            logits = self.model(x)
            loss = self.criterion(logits, y)
        return logits, loss

    # ── LR warmup / scheduler step ────────────────────────────────────────────

    def _apply_warmup(self, epoch: int) -> None:
        c = self.sched_cfg
        if epoch < c.warmup_epochs:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.opt_cfg.lr * (epoch + 1) / c.warmup_epochs

    def _step_scheduler(self, val_dice: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_dice)
        else:
            self.scheduler.step()

    # ── Epoch loops ───────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        self._apply_warmup(epoch)
        accum_loss = accum_dice = n = 0.0
        accum_steps = self.train_cfg.batch_size if self.train_cfg.gradient_accumulation else 1
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            x, y = self._batch_to_tensors(batch)
            self.optimizer.zero_grad()
            logits, loss = self._forward(x, y)

            loss_scaled = loss / accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            with torch.no_grad():
                dice = compute_dice(logits, y, self.data_cfg.num_classes)

            accum_loss += loss.item()
            accum_dice += dice['mean_dice']
            n += 1

            do_step = ((i + 1) % accum_steps == 0) or (i == len(self.train_loader) - 1)

            if do_step:
                # grad clip after unscale
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                if self.opt_cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.opt_cfg.grad_clip
                    )

                # optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # reset grad
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(self.model)

            if i % self.infra_cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
                self.writer.add_scalar(
                    'train/batch_dice', dice['mean_dice'], step)
                self.writer.add_scalar('train/lr', lr, step)
                self.log.info(
                    f'Epoch {epoch:04d} | Batch {i}/{len(self.train_loader)} | '
                    f'loss {loss.item():.4f} | dice {dice["mean_dice"]:.4f} | lr {lr:.2e}'
                )

        return {'loss': accum_loss / n, 'mean_dice': accum_dice / n}

    @torch.no_grad()
    def val_epoch(self) -> Dict:
        self.model.eval()
        accum: Dict[str, float] = {}
        n = 0

        for batch in self.val_loader:
            x, y = self._batch_to_tensors(batch)
            logits, loss = self._forward(x, y)
            dice = compute_dice(logits, y, self.data_cfg.num_classes)
            iou = compute_iou(logits,  y, self.data_cfg.num_classes)

            if n == 0:
                accum = {'loss': 0.0, **{k: 0.0 for k in dice},
                         **{k: 0.0 for k in iou}}
            accum['loss'] += loss.item()
            for k, v in {**dice, **iou}.items():
                accum[k] += v
            n += 1

        return {k: v / n for k, v in accum.items()} if n else accum

    # ── TensorBoard logging ───────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict) -> None:
        n = self.data_cfg.num_classes
        self.writer.add_scalars('epoch/loss',
                                {'train': train_m['loss'], 'val': val_m.get('loss', 0)}, epoch)
        self.writer.add_scalars('epoch/mean_dice',
                                {'train': train_m['mean_dice'],
                                 'val': val_m.get('mean_dice', 0)}, epoch)
        for c in range(1, n):
            k = f'dice_class_{c}'
            if k in val_m:
                self.writer.add_scalars(f'epoch/dice_class_{c}',
                                        {'train': train_m.get(k, 0), 'val': val_m[k]}, epoch)
            ki = f'iou_class_{c}'
            if ki in val_m:
                self.writer.add_scalar(
                    f'epoch/val_iou_class_{c}', val_m[ki], epoch)
        self.writer.add_scalar(
            'train/lr', self.optimizer.param_groups[0]['lr'], epoch)

        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(
                        f'weights/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(
                            f'grads/{name}', param.grad, epoch)

    def _log_images(self, epoch: int) -> None:
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            x, y = self._batch_to_tensors(batch)
            with torch.no_grad():
                logits = self.model(x)
                n = self.data_cfg.num_classes
                pred = ((torch.sigmoid(logits) > 0.5).float() if n == 1
                        else logits.argmax(dim=1, keepdim=True).float() / max(n - 1, 1))

            img = x[0, 0]
            lbl = y[0, 0].float() / max(self.data_cfg.num_classes - 1, 1)
            prd = pred[0, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            D, H, W = img.shape

            def _row(*slices):
                return torch.stack([s.unsqueeze(0) for s in slices])

            self.writer.add_images('slices/axial',
                                   _row(img[D//2], lbl[D//2], prd[D//2]), epoch)
            self.writer.add_images('slices/coronal',
                                   _row(img[:, H//2, :], lbl[:, H//2, :], prd[:, H//2, :]), epoch)
            self.writer.add_images('slices/sagittal',
                                   _row(img[:, :, W//2], lbl[:, :, W//2], prd[:, :, W//2]), epoch)
        except Exception as exc:
            self.log.warning(f'Image logging failed: {exc}')

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_m: Dict, is_best: bool) -> None:
        ckpt = {
            'epoch':         epoch,
            'model':         self.model.state_dict(),
            'optimizer':     self.optimizer.state_dict(),
            'val_metrics':   val_m,
            'best_val_dice': self.best_val_dice,
        }
        if self.scheduler:
            ckpt['scheduler'] = self.scheduler.state_dict()
        if self.scaler:
            ckpt['scaler'] = self.scaler.state_dict()
        if self.ema:
            ckpt['ema'] = self.ema.state_dict()

        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if epoch % self.infra_cfg.save_interval == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:04d}.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            self.log.info(f'  -> New best -- val_dice {val_m["mean_dice"]:.4f}')

    def _load_checkpoint(self, path: str) -> None:
        self.log.info(f'Resuming from {path}')
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_dice = ckpt.get('best_val_dice', -float('inf'))
        if 'scheduler' in ckpt and self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and self.scaler:
            self.scaler.load_state_dict(ckpt['scaler'])
        if 'ema' in ckpt and self.ema:
            self.ema.load_state_dict(ckpt['ema'], self.device)

    # ── Test evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self) -> float:
        best = self.ckpt_dir / 'best.pth'
        if best.exists():
            self.log.info('Loading best checkpoint for test evaluation…')
            self._load_checkpoint(str(best))

        with self._ema_scope():
            return self._run_test()

    def _run_test(self) -> float:
        self.model.eval()
        all_dice, all_iou = [], []

        for subject in self.test_ds:
            subject_id = getattr(subject, 'subject_id', 'unknown')

            if self.patch_cfg.enabled:
                grid = tio.GridSampler(
                    subject, self.patch_cfg.size, self.patch_cfg.overlap)
                loader = DataLoader(
                    grid, batch_size=self.train_cfg.batch_size, num_workers=0)
                aggr = tio.data.GridAggregator(grid, overlap_mode='average')
                for pb in loader:
                    imgs = torch.cat(
                        [pb[m][tio.DATA] for m in self.data_cfg.modalities], dim=1
                    ).to(self.device)
                    aggr.add_batch(self.model(imgs), pb[tio.LOCATION])
                logits = aggr.get_output_tensor().unsqueeze(0).to(self.device)
                label = subject[self.data_cfg.label_name][tio.DATA].unsqueeze(
                    0).to(self.device)
            else:
                imgs = torch.cat(
                    [subject[m][tio.DATA] for m in self.data_cfg.modalities], dim=0
                ).unsqueeze(0).to(self.device)
                logits = self.model(imgs)
                label = subject[self.data_cfg.label_name][tio.DATA].unsqueeze(
                    0).to(self.device)

            dice = compute_dice(logits, label, self.data_cfg.num_classes)
            iou = compute_iou(logits, label,  self.data_cfg.num_classes)
            all_dice.append(dice['mean_dice'])
            all_iou.append(iou['mean_iou'])
            self.log.info(f'  {subject_id}: dice {dice["mean_dice"]:.4f}  '
                          f'iou {iou["mean_iou"]:.4f}')

        mean_dice = float(np.mean(all_dice))
        std_dice = float(np.std(all_dice))
        mean_iou = float(np.mean(all_iou))
        self.log.info(f'Test -- mean Dice: {mean_dice:.4f} ± {std_dice:.4f}  '
                      f'mean IoU: {mean_iou:.4f}')

        self.writer.add_scalar('test/mean_dice', mean_dice)
        self.writer.add_scalar('test/std_dice',  std_dice)
        self.writer.add_scalar('test/mean_iou',  mean_iou)

        (self.exp_dir / 'test_results.json').write_text(
            json.dumps({'mean_dice': mean_dice, 'std_dice': std_dice, 'mean_iou': mean_iou,
                        'per_subject': [{'subject': s.subject_id if hasattr(s, 'subject_id')
                                         else f'sub_{i}', 'dice': float(d), 'iou': float(u)}
                                        for i, (s, d, u) in enumerate(
                                            zip(self.test_ds, all_dice, all_iou))]},
                       indent=2),
            encoding='utf-8',
        )
        return mean_dice

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.log.info(f'Device: {self.device}')
        self.log.info(f'Experiment: {self.exp_dir}')

        self.writer.add_text('config/summary',
                             f'```\n{ConfigManager.get().print_all}\n```')

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            t0 = time.perf_counter()
            train_m = self.train_epoch(epoch)
            val_m: Dict = {}

            if (epoch + 1) % self.infra_cfg.val_interval == 0:
                with self._ema_scope():
                    val_m = self.val_epoch()
                    if (self.infra_cfg.log_images
                            and (epoch + 1) % self.infra_cfg.log_images_interval == 0):
                        self._log_images(epoch)
                self._log_epoch(epoch, train_m, val_m)

                if epoch >= self.sched_cfg.warmup_epochs:
                    self._step_scheduler(val_m.get('mean_dice', 0.0))

                current_dice = val_m.get('mean_dice', 0.0)
                is_best = current_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = current_dice
                    self.stale_epochs = 0
                else:
                    self.stale_epochs += 1

                self._save_checkpoint(epoch, val_m, is_best)

                self.log.info(
                    f'Epoch {epoch:04d}/{self.train_cfg.epochs} | '
                    f'train_loss {train_m["loss"]:.4f}  '
                    f'train_dice {train_m["mean_dice"]:.4f} | '
                    f'val_loss {val_m.get("loss", 0):.4f}  '
                    f'val_dice {current_dice:.4f} | '
                    f'best {self.best_val_dice:.4f} | '
                    f'stale {self.stale_epochs} | '
                    f'{time.perf_counter() - t0:.0f}s'
                )

                if (self.train_cfg.early_stopping
                        and self.stale_epochs >= self.train_cfg.early_stopping_patience):
                    self.log.info(
                        f'Early stopping after {self.stale_epochs} epochs without improvement.'
                    )
                    break

        self.writer.add_hparams(
            {'lr': self.opt_cfg.lr, 'batch_size': self.train_cfg.batch_size,
             'base_features': self.model_cfg.encoder.base_features,
             'optimizer': self.opt_cfg.type, 'loss': self.loss_cfg.type,
             'patch_based': self.patch_cfg.enabled,
             'ema': self.train_cfg.ema,
             'ema_decay': self.train_cfg.ema_decay if self.train_cfg.ema else 0.0},
            {'hparam/best_val_dice': self.best_val_dice},
        )
        self.writer.close()
        self.log.info('Training complete. Running test evaluation…')
        self.test()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.reproduce:
        load_manager_from_directory(args.reproduce)
        print(f'Reproducing experiment from: {args.reproduce}')
    else:
        setup_manager(args)

    ConfigManager.get().print_all()
    Trainer().run()


if __name__ == '__main__':
    main()
