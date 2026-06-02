"""
3D U-Net training framework for medical image segmentation.

Usage
-----
Train from scratch:
    python train.py --data_root /data/my_dataset --num_classes 3 \\
        --patch_based --patch_size 128 128 128 --epochs 200 \\
        --experiment_name exp_001 --log_images --amp

Reproduce a past experiment exactly:
    python train.py --reproduce output/exp_001/config.json

See --help for all options.
"""

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

from config import Config
from dataset import create_data_loaders, create_datasets
from losses import DiceCELoss, DiceLoss
from metrics import compute_dice, compute_iou
from model import UNet3D


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='3D U-Net training framework (PyTorch + TorchIO)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Reproduce shortcut — load a saved config.json and skip all other flags
    p.add_argument('--reproduce', default=None, metavar='CONFIG_JSON',
                   help='Path to a saved config.json to reproduce an experiment exactly. '
                        'All other flags are ignored when this is set.')

    # ── Data ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Data')
    g.add_argument('--data_root', default='',
                   help='Root folder containing train / validation / test sub-directories')
    g.add_argument('--modalities', nargs='+', default=None,
                   help='Modality folder names (auto-detected when omitted)')
    g.add_argument('--label_name', default='label',
                   help='Segmentation mask folder name inside each subject')
    g.add_argument('--num_classes', type=int, default=2,
                   help='Number of output classes (include background for multi-class)')
    g.add_argument('--target_spacing', type=float, nargs=3, default=None,
                   metavar=('X', 'Y', 'Z'), help='Resample to this voxel spacing (mm)')
    g.add_argument('--target_shape', type=int, nargs=3, default=None,
                   metavar=('D', 'H', 'W'), help='Crop or pad to this shape')
    g.add_argument('--normalization', default='znorm',
                   choices=['znorm', 'rescale', 'none'])

    # ── Patch-based training ───────────────────────────────────────────────────
    g = p.add_argument_group('Patch-based training')
    g.add_argument('--patch_based', action='store_true')
    g.add_argument('--patch_size', type=int, nargs=3, default=[128, 128, 128],
                   metavar=('D', 'H', 'W'))
    g.add_argument('--patch_overlap', type=int, nargs=3, default=[64, 64, 64],
                   metavar=('D', 'H', 'W'), help='Overlap for GridSampler (test inference)')
    g.add_argument('--samples_per_volume', type=int, default=4)
    g.add_argument('--queue_max_length', type=int, default=256)
    g.add_argument('--weighted_sampling', action='store_true')

    # ── Augmentation ──────────────────────────────────────────────────────────
    g = p.add_argument_group('Augmentation')
    g.add_argument('--no_augment', action='store_true', help='Disable augmentation')
    g.add_argument('--elastic_deformation', action='store_true')

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Model')
    g.add_argument('--base_features', type=int, default=32)
    g.add_argument('--trilinear', action='store_true', default=True)
    g.add_argument('--no_trilinear', action='store_false', dest='trilinear')

    # ── Loss ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group('Loss')
    g.add_argument('--loss', default='dice_ce', choices=['dice', 'dice_ce', 'ce'])
    g.add_argument('--dice_weight', type=float, default=0.5)
    g.add_argument('--ce_weight', type=float, default=0.5)

    # ── Optimiser / Scheduler ─────────────────────────────────────────────────
    g = p.add_argument_group('Optimiser / Scheduler')
    g.add_argument('--epochs', type=int, default=200)
    g.add_argument('--batch_size', type=int, default=2)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--weight_decay', type=float, default=1e-5)
    g.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw', 'sgd'])
    g.add_argument('--scheduler', default='cosine',
                   choices=['cosine', 'plateau', 'step', 'none'])
    g.add_argument('--warmup_epochs', type=int, default=5)
    g.add_argument('--scheduler_patience', type=int, default=10)
    g.add_argument('--scheduler_factor', type=float, default=0.5)
    g.add_argument('--grad_clip', type=float, default=1.0, help='0 = disabled')
    g.add_argument('--amp', action='store_true', help='Automatic mixed precision (CUDA only)')

    # ── Early stopping ────────────────────────────────────────────────────────
    g = p.add_argument_group('Early stopping')
    g.add_argument('--early_stopping', action='store_true')
    g.add_argument('--early_stopping_patience', type=int, default=30)

    # ── Infrastructure ────────────────────────────────────────────────────────
    g = p.add_argument_group('Infrastructure')
    g.add_argument('--output_dir', default='./output')
    g.add_argument('--experiment_name', default='unet3d')
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    g.add_argument('--resume', default=None, metavar='CHECKPOINT')
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--val_interval', type=int, default=1)
    g.add_argument('--save_interval', type=int, default=10)
    g.add_argument('--log_interval', type=int, default=10)
    g.add_argument('--log_images', action='store_true')
    g.add_argument('--log_images_interval', type=int, default=10)

    return p.parse_args()


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training / validation / test loop.

    Reads all parameters from the global :class:`~config.Config` singleton;
    no 'args' object is passed after initialization.
    """

    def __init__(self) -> None:
        self.cfg = Config.get()
        cfg = self.cfg

        self.device = self._resolve_device()
        self._seed_everything()

        # ── Directories ───────────────────────────────────────────────────────
        self.exp_dir  = Path(cfg.infra.output_dir) / cfg.infra.experiment_name
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.tb_dir   = self.exp_dir / 'tensorboard'
        for d in (self.ckpt_dir, self.tb_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Persist config as JSON immediately so the experiment is reproducible
        # even if it crashes before completion.
        cfg_path = self.exp_dir / 'config.json'
        cfg.to_json(str(cfg_path))
        self.log.info(f"Config saved → {cfg_path}")

        # ── Datasets ──────────────────────────────────────────────────────────
        self.log.info("Building datasets…")
        self.train_ds, self.val_ds, self.test_ds = create_datasets()
        self.train_loader, self.val_loader = create_data_loaders(
            self.train_ds, self.val_ds
        )
        self.log.info(
            f"Subjects — train: {len(self.train_ds)}  "
            f"val: {len(self.val_ds)}  test: {len(self.test_ds)}"
        )

        if not cfg.data.modalities:
            raise RuntimeError("cfg.data.modalities is empty after dataset creation.")

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = UNet3D(
            in_channels=len(cfg.data.modalities),
            num_classes=cfg.data.num_classes,
            base_features=cfg.model.base_features,
            trilinear=cfg.model.trilinear,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info(f"UNet3D — trainable parameters: {n_params:,}")

        # ── Loss / optimiser / scheduler ──────────────────────────────────────
        self.criterion  = self._build_loss()
        self.optimizer  = self._build_optimizer()
        self.scheduler  = self._build_scheduler()
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler('cuda') if cfg.training.amp and self.device.type == 'cuda' else None
        )

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self._log_model_graph()

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch    = 0
        self.best_val_dice  = -float('inf')
        self.stale_epochs   = 0

        if cfg.infra.resume:
            self._load_checkpoint(cfg.infra.resume)

    # ── Device / seed ─────────────────────────────────────────────────────────

    def _resolve_device(self) -> torch.device:
        spec = self.cfg.infra.device
        if spec == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(spec)

    def _seed_everything(self) -> None:
        import random
        s = self.cfg.infra.seed
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

    # ── Component builders ────────────────────────────────────────────────────

    def _build_loss(self) -> nn.Module:
        cfg = self.cfg
        if cfg.loss.type == 'dice':
            return DiceLoss(cfg.data.num_classes)
        if cfg.loss.type == 'dice_ce':
            return DiceCELoss(cfg.data.num_classes, cfg.loss.dice_weight, cfg.loss.ce_weight)
        # plain CE
        return (nn.BCEWithLogitsLoss() if cfg.data.num_classes == 1
                else nn.CrossEntropyLoss())

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.cfg
        params = self.model.parameters()
        if cfg.optimizer.type == 'adam':
            return optim.Adam(params, lr=cfg.optimizer.lr,
                              weight_decay=cfg.optimizer.weight_decay)
        if cfg.optimizer.type == 'adamw':
            return optim.AdamW(params, lr=cfg.optimizer.lr,
                               weight_decay=cfg.optimizer.weight_decay)
        return optim.SGD(params, lr=cfg.optimizer.lr, momentum=0.9,
                         weight_decay=cfg.optimizer.weight_decay, nesterov=True)

    def _build_scheduler(self):
        cfg = self.cfg
        if cfg.scheduler.type == 'cosine':
            T = max(cfg.training.epochs - cfg.scheduler.warmup_epochs, 1)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T)
        if cfg.scheduler.type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max',
                factor=cfg.scheduler.factor, patience=cfg.scheduler.patience,
            )
        if cfg.scheduler.type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.scheduler.patience,
                gamma=cfg.scheduler.factor,
            )
        return None

    # ── TensorBoard helpers ───────────────────────────────────────────────────

    def _log_model_graph(self) -> None:
        cfg = self.cfg
        try:
            shape = (
                cfg.patch.size if cfg.patch.enabled
                else cfg.data.target_shape if cfg.data.target_shape
                else (64, 64, 64)
            )
            dummy = torch.zeros(
                1, len(cfg.data.modalities), *shape, device=self.device
            )
            self.writer.add_graph(self.model, dummy)
        except Exception as exc:
            self.log.warning(f"Model graph logging skipped: {exc}")

    def _log_epoch(self, epoch: int, train_m: Dict, val_m: Dict) -> None:
        cfg = self.cfg
        self.writer.add_scalars(
            'epoch/loss',
            {'train': train_m['loss'], 'val': val_m.get('loss', 0)},
            epoch,
        )
        self.writer.add_scalars(
            'epoch/mean_dice',
            {'train': train_m['mean_dice'], 'val': val_m.get('mean_dice', 0)},
            epoch,
        )
        for c in range(1, cfg.data.num_classes):
            key = f'dice_class_{c}'
            if key in val_m:
                self.writer.add_scalars(
                    f'epoch/dice_class_{c}',
                    {'train': train_m.get(key, 0), 'val': val_m[key]},
                    epoch,
                )
            iou_key = f'iou_class_{c}'
            if iou_key in val_m:
                self.writer.add_scalar(
                    f'epoch/val_iou_class_{c}', val_m[iou_key], epoch
                )
        self.writer.add_scalar(
            'train/lr', self.optimizer.param_groups[0]['lr'], epoch
        )
        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'weights/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, epoch)

    def _log_images(self, epoch: int) -> None:
        """Write axial / coronal / sagittal slices (input | label | pred) to TensorBoard."""
        cfg = self.cfg
        self.model.eval()
        try:
            batch = next(iter(self.val_loader))
            x, y = self._batch_to_tensors(batch)
            with torch.no_grad():
                logits = self.model(x)
                if cfg.data.num_classes == 1:
                    pred = (torch.sigmoid(logits) > 0.5).float()
                else:
                    pred = (logits.argmax(dim=1, keepdim=True).float()
                            / max(cfg.data.num_classes - 1, 1))

            img = x[0, 0]                                               # [D, H, W]
            lbl = y[0, 0].float() / max(cfg.data.num_classes - 1, 1)
            prd = pred[0, 0]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)   # normalise

            D, H, W = img.shape

            def _row(*slices):
                return torch.stack([s.unsqueeze(0) for s in slices])    # [3, 1, H, W]

            self.writer.add_images(
                'slices/axial',    _row(img[D//2],       lbl[D//2],       prd[D//2]),       epoch)
            self.writer.add_images(
                'slices/coronal',  _row(img[:, H//2, :], lbl[:, H//2, :], prd[:, H//2, :]), epoch)
            self.writer.add_images(
                'slices/sagittal', _row(img[:, :, W//2], lbl[:, :, W//2], prd[:, :, W//2]), epoch)
        except Exception as exc:
            self.log.warning(f"Image logging failed: {exc}")

    # ── Batch helpers ─────────────────────────────────────────────────────────

    def _batch_to_tensors(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        x = torch.cat(
            [batch[m][tio.DATA] for m in cfg.data.modalities], dim=1
        ).to(self.device)                                               # [B, C, D, H, W]
        y = None
        if cfg.data.label_name in batch:
            y = batch[cfg.data.label_name][tio.DATA].to(self.device)   # [B, 1, D, H, W]
        return x, y

    def _forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = self.model(x)
                loss   = self.criterion(logits, y)
        else:
            logits = self.model(x)
            loss   = self.criterion(logits, y)
        return logits, loss

    # ── LR warmup / scheduler ─────────────────────────────────────────────────

    def _apply_warmup(self, epoch: int) -> None:
        cfg = self.cfg
        if epoch < cfg.scheduler.warmup_epochs:
            factor = (epoch + 1) / cfg.scheduler.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = cfg.optimizer.lr * factor

    def _step_scheduler(self, val_dice: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_dice)
        else:
            self.scheduler.step()

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        cfg = self.cfg
        self.model.train()
        self._apply_warmup(epoch)

        accum_loss = accum_dice = n = 0.0

        for i, batch in enumerate(self.train_loader):
            x, y = self._batch_to_tensors(batch)
            self.optimizer.zero_grad()
            logits, loss = self._forward(x, y)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if cfg.optimizer.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if cfg.optimizer.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)
                self.optimizer.step()

            with torch.no_grad():
                dice_scores = compute_dice(logits, y, cfg.data.num_classes)

            accum_loss += loss.item()
            accum_dice += dice_scores['mean_dice']
            n += 1

            if i % cfg.infra.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
                self.writer.add_scalar('train/batch_dice', dice_scores['mean_dice'], step)
                self.writer.add_scalar('train/lr', lr, step)
                self.log.info(
                    f"Epoch {epoch:04d} | Batch {i}/{len(self.train_loader)} | "
                    f"loss {loss.item():.4f} | dice {dice_scores['mean_dice']:.4f} | "
                    f"lr {lr:.2e}"
                )

        return {'loss': accum_loss / n, 'mean_dice': accum_dice / n}

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self) -> Dict:
        cfg = self.cfg
        self.model.eval()
        accum: Dict[str, float] = {}
        n = 0

        for batch in self.val_loader:
            x, y = self._batch_to_tensors(batch)
            logits, loss = self._forward(x, y)
            dice_scores = compute_dice(logits, y, cfg.data.num_classes)
            iou_scores  = compute_iou(logits, y, cfg.data.num_classes)

            if n == 0:
                accum = {'loss': 0.0,
                         **{k: 0.0 for k in dice_scores},
                         **{k: 0.0 for k in iou_scores}}

            accum['loss'] += loss.item()
            for k, v in {**dice_scores, **iou_scores}.items():
                accum[k] += v
            n += 1

        return {k: v / n for k, v in accum.items()} if n else accum

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_m: Dict, is_best: bool) -> None:
        cfg = self.cfg
        ckpt = {
            'epoch':          epoch,
            'model':          self.model.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'val_metrics':    val_m,
            'best_val_dice':  self.best_val_dice,
            'config':         cfg.to_dict(),          # full config stored for reference
        }
        if self.scheduler:
            ckpt['scheduler'] = self.scheduler.state_dict()
        if self.scaler:
            ckpt['scaler'] = self.scaler.state_dict()

        torch.save(ckpt, self.ckpt_dir / 'latest.pth')

        if epoch % cfg.infra.save_interval == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{epoch:04d}.pth')

        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            self.log.info(f"  ↳ New best — val_dice {val_m['mean_dice']:.4f}")

    def _load_checkpoint(self, path: str) -> None:
        self.log.info(f"Resuming from {path}")
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
        """Evaluate the best checkpoint on the test split."""
        cfg = self.cfg
        best_ckpt = self.ckpt_dir / 'best.pth'
        if best_ckpt.exists():
            self.log.info("Loading best checkpoint for test evaluation…")
            self._load_checkpoint(str(best_ckpt))

        self.model.eval()
        all_dice: list = []
        all_iou:  list = []

        for subject in self.test_ds:
            subject_id = getattr(subject, 'subject_id', 'unknown')

            if cfg.patch.enabled:
                grid_sampler = tio.GridSampler(
                    subject, cfg.patch.size, cfg.patch.overlap
                )
                patch_loader = DataLoader(
                    grid_sampler, batch_size=cfg.training.batch_size, num_workers=0
                )
                aggregator = tio.data.GridAggregator(
                    grid_sampler, overlap_mode='average'
                )
                for pb in patch_loader:
                    imgs = torch.cat(
                        [pb[m][tio.DATA] for m in cfg.data.modalities], dim=1
                    ).to(self.device)
                    aggregator.add_batch(self.model(imgs), pb[tio.LOCATION])
                logits = aggregator.get_output_tensor().unsqueeze(0).to(self.device)
                label  = subject[cfg.data.label_name][tio.DATA].unsqueeze(0).to(self.device)
            else:
                imgs = torch.cat(
                    [subject[m][tio.DATA] for m in cfg.data.modalities], dim=0
                ).unsqueeze(0).to(self.device)
                logits = self.model(imgs)
                label  = subject[cfg.data.label_name][tio.DATA].unsqueeze(0).to(self.device)

            dice = compute_dice(logits, label, cfg.data.num_classes)
            iou  = compute_iou(logits, label, cfg.data.num_classes)
            all_dice.append(dice['mean_dice'])
            all_iou.append(iou['mean_iou'])
            self.log.info(
                f"  {subject_id}: dice {dice['mean_dice']:.4f}  iou {iou['mean_iou']:.4f}"
            )

        mean_dice = float(np.mean(all_dice))
        std_dice  = float(np.std(all_dice))
        mean_iou  = float(np.mean(all_iou))

        self.log.info(
            f"Test — mean Dice: {mean_dice:.4f} ± {std_dice:.4f}  mean IoU: {mean_iou:.4f}"
        )
        self.writer.add_scalar('test/mean_dice', mean_dice)
        self.writer.add_scalar('test/std_dice',  std_dice)
        self.writer.add_scalar('test/mean_iou',  mean_iou)

        results = {
            'mean_dice': mean_dice, 'std_dice': std_dice, 'mean_iou': mean_iou,
            'per_subject': [
                {
                    'subject': s.subject_id if hasattr(s, 'subject_id') else f'sub_{i}',
                    'dice': float(d), 'iou': float(u),
                }
                for i, (s, d, u) in enumerate(zip(self.test_ds, all_dice, all_iou))
            ],
        }
        (self.exp_dir / 'test_results.json').write_text(
            json.dumps(results, indent=2), encoding='utf-8'
        )
        return mean_dice

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self.cfg
        self.log.info(f"Device: {self.device}")
        self.log.info(f"Experiment dir: {self.exp_dir}")

        # Log full config as formatted text in TensorBoard
        self.writer.add_text(
            'config/summary',
            f"```\n{cfg.summary()}\n```",
        )
        self.writer.add_text(
            'config/json',
            f"```json\n{cfg.to_json()}\n```",
        )

        for epoch in range(self.start_epoch, cfg.training.epochs):
            t0 = time.perf_counter()
            train_m = self.train_epoch(epoch)

            val_m: Dict = {}
            if (epoch + 1) % cfg.infra.val_interval == 0:
                val_m = self.val_epoch()
                self._log_epoch(epoch, train_m, val_m)

                if cfg.infra.log_images and (epoch + 1) % cfg.infra.log_images_interval == 0:
                    self._log_images(epoch)

                if epoch >= cfg.scheduler.warmup_epochs:
                    self._step_scheduler(val_m.get('mean_dice', 0.0))

                current_dice = val_m.get('mean_dice', 0.0)
                is_best = current_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = current_dice
                    self.stale_epochs  = 0
                else:
                    self.stale_epochs += 1

                self._save_checkpoint(epoch, val_m, is_best)

                elapsed = time.perf_counter() - t0
                self.log.info(
                    f"Epoch {epoch:04d}/{cfg.training.epochs} | "
                    f"train_loss {train_m['loss']:.4f}  train_dice {train_m['mean_dice']:.4f} | "
                    f"val_loss {val_m.get('loss', 0):.4f}  val_dice {current_dice:.4f} | "
                    f"best {self.best_val_dice:.4f} | stale {self.stale_epochs} | {elapsed:.0f}s"
                )

                if (cfg.training.early_stopping
                        and self.stale_epochs >= cfg.training.early_stopping_patience):
                    self.log.info(
                        f"Early stopping after {self.stale_epochs} epochs without improvement."
                    )
                    break

        # Hyper-parameter / metric summary in TensorBoard
        self.writer.add_hparams(
            hparam_dict={
                'lr':            cfg.optimizer.lr,
                'batch_size':    cfg.training.batch_size,
                'base_features': cfg.model.base_features,
                'optimizer':     cfg.optimizer.type,
                'loss':          cfg.loss.type,
                'patch_based':   cfg.patch.enabled,
            },
            metric_dict={'hparam/best_val_dice': self.best_val_dice},
        )
        self.writer.close()
        self.log.info("Training complete. Running test evaluation…")
        self.test()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.reproduce:
        # Exact reproduction: load the saved config.json, ignore all other flags
        cfg = Config.from_json(args.reproduce)
        print(f"Reproducing experiment from: {args.reproduce}")
    else:
        cfg = Config.from_args(args)

    cfg.print_summary()
    Trainer().run()


if __name__ == '__main__':
    main()
