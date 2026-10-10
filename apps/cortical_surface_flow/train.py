"""
Cortical Surface Flow -- training entry point.

Learns to deform one FreeSurfer surface of a subject onto another, conditioned on the
T1 (see model.py).  One experiment = one hemisphere + one (source, target) pair.

Usage
-----
Train the validated setting (lh white -> pial.T1, closed loop, Kalman setpoint)::

    python -m apps.cortical_surface_flow.train ^
        --cache_dir input/cortical_surface_flow/cache ^
        --experiment_name lh_white_pial_001

Other tasks are configuration only::

    --hemi rh --source_surface white --target_surface pial.T1
    --source_surface orig     --target_surface white
    --source_surface white    --target_surface inflated  --select_on corr
    --source_surface inflated --target_surface sphere    --select_on corr --no_t1

Reproduce a past experiment exactly (every setting comes from its saved JSONs)::

    python -m apps.cortical_surface_flow.train --reproduce output/lh_white_pial_001/

Resume an interrupted run::

    python -m apps.cortical_surface_flow.train --reproduce output/lh_white_pial_001/ ^
        --resume output/lh_white_pial_001/checkpoints/latest.pth

Notes
-----
- The split and the perturbation scale are resolved on the first run and written back
  into data.json / control.json, so --reproduce re-runs the same experiment.
- best.pth is selected on the validation split only (TrainingConfig.select_on); the test
  split is evaluated once, after training (test.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from configuration.manager import ConfigManager
from apps.cortical_surface_flow.dataset import Augmenter, SurfacePairData, resolve_split
from apps.cortical_surface_flow.geometry import GridFrame, utf8_stdout
from apps.cortical_surface_flow.losses import closed_loop_loss, flow_loss
from apps.cortical_surface_flow.metrics import fold_detail, nn_surface_distance
from apps.cortical_surface_flow.model import ClosedLoopFlow, build_model, check_grid_sample, set_kalman
from apps.cortical_surface_flow.surface_flow_config import (
    AugmentConfig, ControlConfig, DataConfig, InfraConfig, LossConfig, OptimizerConfig,
    SchedulerConfig, SurfaceFlowModelConfig, TrainingConfig)
from apps.cortical_surface_flow.visualize import (
    alignment_figure, error_figure, fig_to_array, sweep_figure)

CONTROL = "Control"
TYPE_MAP = {
    ConfigManager.DATA: DataConfig,
    ConfigManager.AUGMENT: AugmentConfig,
    ConfigManager.MODEL: SurfaceFlowModelConfig,
    ConfigManager.LOSS: LossConfig,
    CONTROL: ControlConfig,
    ConfigManager.OPTIMIZER: OptimizerConfig,
    ConfigManager.SCHEDULER: SchedulerConfig,
    ConfigManager.TRAINING: TrainingConfig,
    ConfigManager.INFRA: InfraConfig,
}


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    D, A, M, L, C = DataConfig(), AugmentConfig(), SurfaceFlowModelConfig(), LossConfig(), ControlConfig()
    O, S, T, I = OptimizerConfig(), SchedulerConfig(), TrainingConfig(), InfraConfig()
    p = argparse.ArgumentParser(description="Cortical Surface Flow -- training",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reproduce", default=None,
                   help="Experiment directory: load every config from its JSONs (other flags ignored "
                        "except --resume)")

    g = p.add_argument_group("Data")
    g.add_argument("--cache_dir", default=D.cache_dir)
    g.add_argument("--hemi", default=D.hemi, choices=["lh", "rh"])
    g.add_argument("--source_surface", default=D.source_surface)
    g.add_argument("--target_surface", default=D.target_surface)
    g.add_argument("--x0_mode", default=D.x0_mode, choices=["subject", "template"])
    g.add_argument("--train_subjects", nargs="*", default=None)
    g.add_argument("--val_subjects", nargs="*", default=None)
    g.add_argument("--test_subjects", nargs="*", default=None)
    g.add_argument("--n_val", type=int, default=D.n_val)
    g.add_argument("--n_test", type=int, default=D.n_test)
    g.add_argument("--split_seed", type=int, default=D.split_seed)
    g.add_argument("--cache_device", default=D.cache_device, choices=["cpu", "cuda"])

    g = p.add_argument_group("Augmentation")
    g.add_argument("--no_augment", action="store_true")
    g.add_argument("--no_rotation", action="store_true")
    g.add_argument("--rotation_p", type=float, default=A.rotation_p)
    g.add_argument("--rotation_degrees", type=float, default=A.rotation_degrees)
    g.add_argument("--no_noise", action="store_true")
    g.add_argument("--noise_p", type=float, default=A.noise_p)
    g.add_argument("--noise_std", type=float, nargs=2, default=list(A.noise_std))
    g.add_argument("--no_gamma", action="store_true")
    g.add_argument("--gamma_p", type=float, default=A.gamma_p)
    g.add_argument("--gamma_log_gamma", type=float, nargs=2, default=list(A.gamma_log_gamma))

    g = p.add_argument_group("Model")
    g.add_argument("--architecture", default=M.architecture, choices=["closed_loop", "flow"])
    g.add_argument("--no_t1", action="store_true", help="Geometry only (e.g. inflated -> sphere)")
    g.add_argument("--base_features", type=int, default=M.encoder.base_features)
    g.add_argument("--depth", type=int, default=M.encoder.depth)
    g.add_argument("--pe_channels", type=int, default=M.encoder.pe_channels)
    g.add_argument("--pe_stride", type=int, default=M.encoder.pe_stride)
    g.add_argument("--encoder_type", default=M.encoder.type, choices=["plain", "residual", "residual_unet"])
    g.add_argument("--blocks_per_level", type=int, default=M.encoder.blocks_per_level)
    g.add_argument("--full_res_blocks", action="store_true")
    g.add_argument("--encoder_checkpoint", action="store_true")
    g.add_argument("--hidden", type=int, default=M.head.hidden)
    g.add_argument("--layers", type=int, default=M.head.layers)
    g.add_argument("--n_freq_x", type=int, default=M.head.n_freq_x)
    g.add_argument("--n_freq_t", type=int, default=M.head.n_freq_t)

    g = p.add_argument_group("Loss")
    g.add_argument("--beta_nll", type=float, default=L.beta_nll)
    g.add_argument("--normal_weight", type=float, default=L.normal_weight)
    g.add_argument("--tangential_weight", type=float, default=L.tangential_weight)
    g.add_argument("--no_uncertainty", action="store_true",
                   help="closed_loop: fixed-variance MSE instead of the beta-NLL posterior (ablation)")

    g = p.add_argument_group("Closed loop / integration")
    g.add_argument("--perturb_mm", type=float, default=C.perturb_mm,
                   help="None = perturb_rel x mean |X1 - X0| of the training set")
    g.add_argument("--perturb_rel", type=float, default=C.perturb_rel)
    g.add_argument("--onpolicy_p", type=float, default=C.onpolicy_p)
    g.add_argument("--onpolicy_warmup", type=int, default=C.onpolicy_warmup)
    g.add_argument("--rollout_steps", type=int, default=C.rollout_steps)
    g.add_argument("--euler_steps", type=int, default=C.euler_steps)
    g.add_argument("--val_euler_steps", type=int, default=C.val_euler_steps)
    g.add_argument("--kalman_q_mm", type=float, default=C.kalman_q_mm)

    g = p.add_argument_group("Optimiser / Scheduler")
    g.add_argument("--optimizer", default=O.type, choices=["adam", "adamw", "sgd"])
    g.add_argument("--lr", type=float, default=O.lr)
    g.add_argument("--weight_decay", type=float, default=O.weight_decay)
    g.add_argument("--grad_clip", type=float, default=O.grad_clip)
    g.add_argument("--scheduler", default=S.type, choices=["cosine", "none"])
    g.add_argument("--warmup_steps", type=int, default=S.warmup_steps)
    g.add_argument("--min_lr_ratio", type=float, default=S.min_lr_ratio)

    g = p.add_argument_group("Training")
    g.add_argument("--steps", type=int, default=T.steps)
    g.add_argument("--gradient_accumulation", type=int, default=T.gradient_accumulation)
    g.add_argument("--points", type=int, default=T.points)
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--select_on", default=T.select_on, choices=["fold", "surface", "corr"])
    g.add_argument("--eval_train_subjects", type=int, default=T.eval_train_subjects)

    g = p.add_argument_group("Infrastructure")
    g.add_argument("--output_dir", default=I.output_dir)
    g.add_argument("--experiment_name", default=I.experiment_name)
    g.add_argument("--device", default=I.device, choices=["auto", "cpu", "cuda"])
    g.add_argument("--seed", type=int, default=I.seed)
    g.add_argument("--resume", default=None)
    g.add_argument("--no_resume_state", action="store_true",
                   help="With --resume: load model weights only (fine-tuning)")
    g.add_argument("--val_interval", type=int, default=I.val_interval)
    g.add_argument("--save_interval", type=int, default=I.save_interval)
    g.add_argument("--log_interval", type=int, default=I.log_interval)
    g.add_argument("--no_log_images", action="store_true")
    g.add_argument("--log_images_interval", type=int, default=I.log_images_interval)
    g.add_argument("--no_test", action="store_true", help="Skip the test evaluation after training")
    return p.parse_args(argv)


def setup_manager(args: argparse.Namespace) -> ConfigManager:
    m = ConfigManager.get()

    dc = DataConfig()
    for k in ("cache_dir", "hemi", "source_surface", "target_surface", "x0_mode",
              "n_val", "n_test", "split_seed", "cache_device"):
        setattr(dc, k, getattr(args, k))
    dc.train_subjects, dc.val_subjects, dc.test_subjects = \
        args.train_subjects, args.val_subjects, args.test_subjects

    ac = AugmentConfig()
    ac.enabled = not args.no_augment
    ac.rotation, ac.rotation_p, ac.rotation_degrees = not args.no_rotation, args.rotation_p, args.rotation_degrees
    ac.noise, ac.noise_p, ac.noise_std = not args.no_noise, args.noise_p, tuple(args.noise_std)
    ac.gamma, ac.gamma_p, ac.gamma_log_gamma = not args.no_gamma, args.gamma_p, tuple(args.gamma_log_gamma)

    mc = SurfaceFlowModelConfig()
    mc.architecture, mc.use_t1 = args.architecture, not args.no_t1
    mc.encoder.base_features, mc.encoder.depth = args.base_features, args.depth
    mc.encoder.pe_channels, mc.encoder.pe_stride = args.pe_channels, args.pe_stride
    mc.encoder.checkpoint = args.encoder_checkpoint
    mc.encoder.type, mc.encoder.blocks_per_level = args.encoder_type, args.blocks_per_level
    mc.encoder.full_res_blocks = args.full_res_blocks
    mc.head.hidden, mc.head.layers = args.hidden, args.layers
    mc.head.n_freq_x, mc.head.n_freq_t = args.n_freq_x, args.n_freq_t

    lc = LossConfig()
    lc.beta_nll, lc.normal_weight, lc.tangential_weight = args.beta_nll, args.normal_weight, args.tangential_weight
    lc.uncertainty = not args.no_uncertainty

    cc = ControlConfig()
    for k in ("perturb_mm", "perturb_rel", "onpolicy_p", "onpolicy_warmup", "rollout_steps",
              "euler_steps", "val_euler_steps", "kalman_q_mm"):
        setattr(cc, k, getattr(args, k))

    oc = OptimizerConfig()
    oc.type, oc.lr, oc.weight_decay, oc.grad_clip = args.optimizer, args.lr, args.weight_decay, args.grad_clip

    sc = SchedulerConfig()
    sc.type, sc.warmup_steps, sc.min_lr_ratio = args.scheduler, args.warmup_steps, args.min_lr_ratio

    tc = TrainingConfig()
    tc.steps, tc.gradient_accumulation, tc.points = args.steps, args.gradient_accumulation, args.points
    tc.amp, tc.select_on, tc.eval_train_subjects = not args.no_amp, args.select_on, args.eval_train_subjects

    ic = InfraConfig()
    for k in ("output_dir", "experiment_name", "device", "seed", "resume", "val_interval",
              "save_interval", "log_interval", "log_images_interval"):
        setattr(ic, k, getattr(args, k))
    ic.resume_state = not args.no_resume_state
    ic.log_images = not args.no_log_images
    ic.run_test = not args.no_test

    for type_name, cfg in ((ConfigManager.DATA, dc), (ConfigManager.AUGMENT, ac), (ConfigManager.MODEL, mc),
                           (ConfigManager.LOSS, lc), (CONTROL, cc), (ConfigManager.OPTIMIZER, oc),
                           (ConfigManager.SCHEDULER, sc), (ConfigManager.TRAINING, tc),
                           (ConfigManager.INFRA, ic)):
        m.register(type_name, cfg)
    return m


def load_manager_from_directory(directory: str) -> ConfigManager:
    """Restore the ConfigManager from a saved experiment directory."""
    m = ConfigManager.get()
    for type_name, cls in TYPE_MAP.items():
        cfg = cls()
        path = Path(directory) / ("%s.json" % type_name.lower())
        if path.exists():
            cfg.load(str(path))
        m.register(type_name, cfg)
    return m


def configs_snapshot() -> dict:
    m = ConfigManager.get()
    return {t: m.get_config(t).to_dict() for t in m.registered_types()}


# ── Trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    """All behaviour is driven by the configs registered in the ConfigManager."""

    def __init__(self) -> None:
        m = ConfigManager.get()
        self.data_cfg: DataConfig = m.get_config(ConfigManager.DATA)
        self.aug_cfg: AugmentConfig = m.get_config(ConfigManager.AUGMENT)
        self.model_cfg: SurfaceFlowModelConfig = m.get_config(ConfigManager.MODEL)
        self.loss_cfg: LossConfig = m.get_config(ConfigManager.LOSS)
        self.ctl_cfg: ControlConfig = m.get_config(CONTROL)
        self.opt_cfg: OptimizerConfig = m.get_config(ConfigManager.OPTIMIZER)
        self.sched_cfg: SchedulerConfig = m.get_config(ConfigManager.SCHEDULER)
        self.train_cfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
        self.infra_cfg: InfraConfig = m.get_config(ConfigManager.INFRA)

        self.exp_dir = Path(self.infra_cfg.output_dir) / self.infra_cfg.experiment_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.tb_dir = self.exp_dir / "tensorboard"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.device = self._resolve_device()
        self._seed_everything()

        # ── frame, split, data ────────────────────────────────────────────────
        self.frame = GridFrame(self.data_cfg.cache_dir, self.data_cfg.hemi)
        self.hf = self.frame.half_fov
        self.log.info(repr(self.frame))
        err, rt = check_grid_sample(self.frame, self.device, self.model_cfg.encoder.depth)
        self.log.info("coordinate self-test: max error %.2e voxels, mm->norm->mm %.1e mm" % (err, rt))
        assert err < 1e-3 and rt < 1e-3, "coordinate frames disagree"

        self.train_names, self.val_names, self.test_names = resolve_split(self.data_cfg)
        if not self.data_cfg.train_subjects:            # persist the resolved split
            self.data_cfg.train_subjects = self.train_names
            self.data_cfg.val_subjects = self.val_names
            self.data_cfg.test_subjects = self.test_names
        t0 = time.time()
        self.data = SurfacePairData(self.data_cfg, self.frame, self.device)
        self.data.add(self.train_names + self.val_names)
        self.log.info("task %s.%s -> %s.%s (x0 %s) | %d train / %d val / %d test subjects loaded in %.0f s"
                      % (self.data_cfg.hemi, self.data_cfg.source_surface, self.data_cfg.hemi,
                         self.data_cfg.target_surface, self.data_cfg.x0_mode, len(self.train_names),
                         len(self.val_names), len(self.test_names), time.time() - t0))
        disp = self.data.mean_displacement_mm(self.train_names)
        out = max(self.frame.outside_grid(self.data.x1_mm(self.train_names[0])),
                  self.frame.outside_grid(self.data.x0_mm(self.train_names[0])))
        self.log.info("mean |X1 - X0| over the training set: %.2f mm | outside the T1 grid: %.1f%%"
                      % (disp, 100 * out))
        if out > 0 and self.model_cfg.use_t1:
            self.log.warning("part of the surfaces lies outside the T1 grid; features there are the "
                             "border values (consider --no_t1 for non-anatomical surfaces)")
        if self.ctl_cfg.perturb_mm is None:             # persist the resolved scale
            self.ctl_cfg.perturb_mm = round(self.ctl_cfg.perturb_rel * disp, 4)
            self.log.info("perturb_mm = %.2f x %.2f mm = %.3f mm"
                          % (self.ctl_cfg.perturb_rel, disp, self.ctl_cfg.perturb_mm))
        ConfigManager.get().save_all(str(self.exp_dir))
        self.log.info("Configs saved to %s" % self.exp_dir)

        # ── model, optimisation ───────────────────────────────────────────────
        self.model = build_model(self.model_cfg, self.frame).to(self.device)
        self.closed = isinstance(self.model, ClosedLoopFlow)
        set_kalman(self.model, self.ctl_cfg.kalman_q_mm)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.log.info("model %s, encoder %s, T1 %s, uncertainty %s, %.2fM parameters"
                      % (self.model_cfg.architecture, self.model_cfg.encoder.type, self.model_cfg.use_t1,
                         self.loss_cfg.uncertainty, self.n_params / 1e6))
        if self.closed and not self.loss_cfg.uncertainty and self.ctl_cfg.kalman_q_mm:
            self.log.warning("uncertainty=False: the variance head is untrained, Kalman setpoint disabled")
            self.ctl_cfg.kalman_q_mm = 0.0
            set_kalman(self.model, 0.0)
            ConfigManager.get().save_all(str(self.exp_dir))      # test.py must see it too
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.amp = self.train_cfg.amp and self.device.type == "cuda"
        self.augmenter = (Augmenter(self.aug_cfg, self.frame, self.device, self.data_cfg.x0_mode == "subject")
                          if self.aug_cfg.enabled else None)

        self.step, self.best_score = 0, float("inf")
        if self.infra_cfg.resume:
            self._resume(self.infra_cfg.resume)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir), purge_step=self.step + 1 if self.step else None,
                                    flush_secs=30)
        self.faces_t = torch.as_tensor(self.frame.faces, device=self.device)

    # ── setup helpers ─────────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(self.exp_dir / "training.log", encoding="utf-8"),
                                      logging.StreamHandler(sys.stdout)], force=True)
        self.log = logging.getLogger("trainer")

    def _resolve_device(self) -> torch.device:
        d = self.infra_cfg.device
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        if d == "cuda":
            assert torch.cuda.is_available(), "device cuda requested but CUDA is not available"
        return torch.device(d)

    def _seed_everything(self) -> None:
        s = self.infra_cfg.seed
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    def _build_optimizer(self):
        c, p = self.opt_cfg, self.model.parameters()
        if c.type == "adam":
            return torch.optim.Adam(p, lr=c.lr, weight_decay=c.weight_decay)
        if c.type == "adamw":
            return torch.optim.AdamW(p, lr=c.lr, weight_decay=c.weight_decay)
        return torch.optim.SGD(p, lr=c.lr, momentum=0.9, weight_decay=c.weight_decay, nesterov=True)

    def _build_scheduler(self):
        c, total = self.sched_cfg, self.train_cfg.steps

        def factor(i):                                   # i = scheduler steps taken
            if c.warmup_steps and i < c.warmup_steps:
                return (i + 1) / c.warmup_steps
            if c.type == "none":
                return 1.0
            x = min(1.0, (i - c.warmup_steps) / max(1, total - c.warmup_steps))
            return c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * x))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _resume(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if self.infra_cfg.resume_state and "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
            self.scheduler.load_state_dict(ck["scheduler"])
            self.step, self.best_score = ck["step"], ck.get("best_score", float("inf"))
            torch.set_rng_state(ck["rng_cpu"].cpu())
            np.random.set_state(ck["rng_np"])
            if self.device.type == "cuda" and ck.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
            self.log.info("resumed %s at step %d (best val %s %.4f)"
                          % (path, self.step, self.train_cfg.select_on, abs(self.best_score)))
        else:
            self.log.info("loaded model weights from %s (fresh optimiser)" % path)

    def save(self, name: str) -> None:
        """Atomic, self-contained (carries every config)."""
        ck = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
              "scheduler": self.scheduler.state_dict(), "step": self.step, "best_score": self.best_score,
              "configs": configs_snapshot(), "rng_cpu": torch.get_rng_state(), "rng_np": np.random.get_state(),
              "rng_cuda": torch.cuda.get_rng_state() if self.device.type == "cuda" else None}
        p = self.ckpt_dir / name
        torch.save(ck, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)

    # ── training ──────────────────────────────────────────────────────────────

    def _encode(self, vol):
        if not self.model.use_t1:
            return None
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            feats = self.model.encode(vol)
        return [f.float() for f in feats]

    def micro_batch(self, name: str):
        r = self.data.subjects[name]
        t1, mask = self.data.volume(name)
        if self.augmenter is not None:
            vol, x0, x1, n1, _ = self.augmenter(t1, mask, r["x0"], r["x1"], r["n1"])
        else:
            vol, x0, x1, n1 = self.data.network_input(name), r["x0"], r["x1"], r["n1"]
        P = self.train_cfg.points
        idx = torch.randint(0, x0.shape[0], (P,), device=self.device)
        p0, p1, nrm = x0[idx], x1[idx], n1[idx]
        t = torch.rand(P, 1, device=self.device) * 0.999
        feats = self._encode(vol)
        if self.closed:
            c = self.ctl_cfg
            xt = (1.0 - t) * p0 + t * p1 + torch.randn_like(p0) * (c.perturb_mm / self.hf) * t
            on = torch.zeros(P, dtype=torch.bool, device=self.device)
            if c.onpolicy_p > 0 and self.step > c.onpolicy_warmup:
                on = torch.rand(P, device=self.device) < c.onpolicy_p
                if on.any():
                    fd = [f.detach() for f in feats] if feats is not None else None
                    xt[on] = self.model.rollout(p0[on], t[on], fd, c.rollout_steps)
            mu, lvn, lvt, nh = self.model.estimate(xt, t, feats)
            loss, st = closed_loop_loss(mu, lvn, lvt, nh, p1, nrm, self.hf, self.loss_cfg.beta_nll,
                                        self.loss_cfg.normal_weight, self.loss_cfg.uncertainty,
                                        self.loss_cfg.tangential_weight)
            st["onpolicy_frac"] = on.float().mean()
        else:
            xt = (1.0 - t) * p0 + t * p1
            v = self.model.velocity(xt, t, feats)
            loss, st = flow_loss(v, p1 - p0, nrm, self.hf, self.loss_cfg.tangential_weight)
        (loss / self.train_cfg.gradient_accumulation).backward()
        return loss.item(), {k: float(v) for k, v in st.items()}

    @torch.no_grad()
    def evaluate(self, names, n_steps):
        """Per subject: correspondence error, nearest-vertex surface distance (mm),
        correspondence-free fold r4 / r16, median predicted sigma."""
        self.model.eval()
        out = {}
        for n in names:
            r = self.data.subjects[n]
            feats = self.model.encode(self.data.network_input(n)) if self.model.use_t1 else None
            pred, _ = self.model.integrate(r["x0"], feats, n_steps=n_steps)
            p = pred.cpu().numpy().astype(np.float64) * self.hf
            q = r["x1"].cpu().numpy().astype(np.float64) * self.hf
            fd = fold_detail(p, q, self.frame.faces)
            sig = self.model.last_sigma
            out[n] = {"corr": float(np.linalg.norm(p - q, axis=1).mean()), "surface": nn_surface_distance(p, q),
                      "fold_r4": fd["fold_r4"], "fold_r16": fd["fold_r16"],
                      "sigma_n": float(sig[0].median()) if sig is not None else float("nan"),
                      "sigma_t": float(sig[1].median()) if sig is not None else float("nan")}
        self.model.train()
        return out

    def score(self, m) -> float:
        """Lower is better."""
        sel = self.train_cfg.select_on
        return -0.5 * (m["fold_r4"] + m["fold_r16"]) if sel == "fold" else m[sel]

    @torch.no_grad()
    def log_images(self) -> None:
        self.model.eval()
        for tag, names in (("train", self.train_names), ("val", self.val_names)):
            if not names:
                continue
            n = names[0]
            r = self.data.subjects[n]
            vol = self.data.network_input(n)
            feats = self.model.encode(vol) if self.model.use_t1 else None
            pred, snaps = self.model.integrate(r["x0"], feats, n_steps=self.ctl_cfg.val_euler_steps,
                                               return_at=(0.25, 0.5, 0.75))
            mm = lambda x: self.frame.to_mm(x.cpu().numpy())             # noqa: E731
            x0, x1, pr = mm(r["x0"]), mm(r["x1"]), mm(pred)
            meshes = [("source", x0)] + [("t=%.2f" % t, mm(snaps[t])) for t in (0.25, 0.5, 0.75)] + \
                     [("prediction", pr), ("target", x1)]
            title = "%s  %s.%s -> %s  step %d" % (n, self.data_cfg.hemi, self.data_cfg.source_surface,
                                                  self.data_cfg.target_surface, self.step)
            self.writer.add_image("flow_%s/%s" % (tag, n), fig_to_array(
                sweep_figure(meshes, self.frame.faces, self.data_cfg.hemi, 300, title)), self.step, dataformats="HWC")
            self.writer.add_image("error_%s/%s" % (tag, n), fig_to_array(
                error_figure(pr, x1, self.frame.faces, self.data_cfg.hemi, 300)), self.step, dataformats="HWC")
            if self.model_cfg.use_t1:
                self.writer.add_image("alignment_%s/%s" % (tag, n), fig_to_array(
                    alignment_figure(self.frame, vol.cpu().numpy(), x1, pr, "%s step %d" % (n, self.step))),
                    self.step, dataformats="HWC")
        self.model.train()

    def run(self) -> None:
        tc, ic = self.train_cfg, self.infra_cfg
        if self.step == 0:
            self.writer.add_text("configs", "```\n%s\n```" % json.dumps(configs_snapshot(), indent=1))
            self.writer.add_text("model", "%s, %.2fM parameters, perturb %.3f mm"
                                 % (self.model_cfg.architecture, self.n_params / 1e6, self.ctl_cfg.perturb_mm))
        names = self.train_names
        run_loss, t0, start = None, time.time(), self.step
        self.model.train()
        while self.step < tc.steps:
            self.step += 1
            self.optimizer.zero_grad(set_to_none=True)
            stats, losses = {}, []
            try:
                for _ in range(tc.gradient_accumulation):
                    l, st = self.micro_batch(names[np.random.randint(len(names))])
                    losses.append(l)
                    for k, v in st.items():
                        stats[k] = stats.get(k, 0.0) + v / tc.gradient_accumulation
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise SystemExit("CUDA OOM at step %d: lower --points, raise --gradient_accumulation, "
                                 "add --encoder_checkpoint or lower --base_features" % self.step)
            gn = (torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt_cfg.grad_clip)
                  if self.opt_cfg.grad_clip > 0 else torch.tensor(0.0))
            self.optimizer.step()
            self.scheduler.step()
            loss = float(np.mean(losses))
            run_loss = loss if run_loss is None else 0.99 * run_loss + 0.01 * loss

            if self.step % ic.log_interval == 0 or self.step == start + 1:
                w = self.writer
                w.add_scalar("train/loss", run_loss, self.step)
                w.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.step)
                w.add_scalar("train/grad_norm", float(gn), self.step)
                w.add_scalar("train/steps_per_sec", (self.step - start) / (time.time() - t0), self.step)
                for k, v in stats.items():
                    if np.isfinite(v):
                        w.add_scalar("train_%s/%s" % ("cl" if self.closed else "flow", k), v, self.step)
                if self.device.type == "cuda":
                    w.add_scalar("train/gpu_mem_peak_mb", torch.cuda.max_memory_allocated() / 2 ** 20, self.step)
                if self.step % (ic.log_interval * 10) == 0 or self.step == start + 1:
                    self.log.info("step %6d/%d | loss %.4e | %s | lr %.2e | %.1f min"
                                  % (self.step, tc.steps, run_loss,
                                     "  ".join("%s %.3f" % (k.replace("_mm", ""), v) for k, v in stats.items()),
                                     self.scheduler.get_last_lr()[0], (time.time() - t0) / 60))

            if self.step % ic.val_interval == 0 or self.step == tc.steps:
                self.validate()
            if ic.log_images and (self.step % ic.log_images_interval == 0 or self.step == tc.steps):
                self.log_images()
            if self.step % ic.save_interval == 0 or self.step == tc.steps:
                self.save("latest.pth")

        self.writer.add_hparams(
            {"architecture": self.model_cfg.architecture, "task": "%s.%s->%s" % (
                self.data_cfg.hemi, self.data_cfg.source_surface, self.data_cfg.target_surface),
             "lr": self.opt_cfg.lr, "perturb_mm": self.ctl_cfg.perturb_mm, "beta_nll": self.loss_cfg.beta_nll,
             "kalman_q_mm": self.ctl_cfg.kalman_q_mm, "pe_channels": self.model_cfg.encoder.pe_channels},
            {"hparam/best_val_%s" % tc.select_on: abs(self.best_score)})
        self.writer.close()
        self.log.info("training complete (%.1f min)" % ((time.time() - t0) / 60))
        if ic.run_test and self.test_names:
            from apps.cortical_surface_flow.test import run_test
            self.log.info("running test evaluation on best.pth")
            run_test(str(self.exp_dir), str(self.ckpt_dir / "best.pth"), log=self.log)

    def validate(self) -> None:
        sub = self.train_names[:self.train_cfg.eval_train_subjects]
        for tag, names in (("train", sub), ("val", self.val_names)):
            if not names:
                continue
            e = self.evaluate(names, self.ctl_cfg.val_euler_steps)
            mean = {k: float(np.mean([v[k] for v in e.values()])) for k in next(iter(e.values()))}
            for k, v in mean.items():
                if np.isfinite(v):
                    self.writer.add_scalar("%s/%s" % (tag, k), v, self.step)
            self.log.info("  [%-5s] corr %.3f mm  surface %.3f mm  fold r4 %.3f r16 %.3f  sigma n %.2f t %.2f mm"
                          % (tag, mean["corr"], mean["surface"], mean["fold_r4"], mean["fold_r16"],
                             mean["sigma_n"], mean["sigma_t"]))
            if tag == "val":
                s = self.score(mean)
                if s < self.best_score:
                    self.best_score = s
                    self.save("best.pth")
                    self.writer.add_scalar("val/best_%s" % self.train_cfg.select_on, abs(s), self.step)
                    self.log.info("  [best ] val %s %.4f at step %d -> best.pth"
                                  % (self.train_cfg.select_on, abs(s), self.step))


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv=None) -> None:
    utf8_stdout()
    args = parse_args(argv)
    if args.reproduce:
        load_manager_from_directory(args.reproduce)
        if args.resume:
            ConfigManager.get().get_config(ConfigManager.INFRA).resume = args.resume
        print("Reproducing experiment from: %s" % args.reproduce)
    else:
        setup_manager(args)
    ConfigManager.get().print_all()
    Trainer().run()


if __name__ == "__main__":
    main()
