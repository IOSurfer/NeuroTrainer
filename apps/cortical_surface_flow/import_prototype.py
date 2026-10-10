"""
Import a run of the research prototype (cortical_flow/) as an experiment of this app.

    python -m apps.cortical_surface_flow.import_prototype ^
        --checkpoint cortical_flow/runs/own_s2_t1/ckpt_best.pt ^
        --experiment_name lh_white_pial_own_s2_t1

Writes ``output/<experiment_name>/`` with every config JSON (taken from the run's
saved arguments and split) and ``checkpoints/best.pth``, so ``test.py`` and
``train.py --reproduce`` work on it.  Parameter names are shared, so the weights load
unchanged.  Only aligned-sampling runs that start from a surface on the fsaverage
vertices (own surface or fsaverage template) can be imported.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from configuration.manager import ConfigManager
from apps.cortical_surface_flow.geometry import utf8_stdout
from apps.cortical_surface_flow.train import CONTROL, TYPE_MAP, configs_snapshot


def main(argv=None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="prototype ckpt_best.pt / ckpt.pt")
    p.add_argument("--prototype_root", default=None, help="Default: two levels above runs/<run>/")
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--kalman_q_mm", type=float, default=1.0, help="Inference setting to record")
    a = p.parse_args(argv)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    g = ck["args"]
    if g.get("sampling", "legacy") != "aligned":
        raise SystemExit("legacy-sampling prototype runs cannot be imported")
    root = a.prototype_root or str(Path(a.checkpoint).resolve().parents[2])
    cache = ck["cache"] if os.path.isabs(ck["cache"]) else os.path.join(root, ck["cache"])
    src = str(ck.get("x0_source") or "")
    if src.startswith("subject:"):
        x0_mode, source = "subject", src.split(":", 1)[1]
    elif "tpl-fsaverage" in src:
        x0_mode, source = "template", os.path.basename(src).split("_")[-1].split(".")[0]
    else:
        raise SystemExit("unsupported X0 %r (sphere-start runs cannot be imported)" % src)
    canon = np.load(os.path.join(cache, "canonical.npz"))

    ConfigManager.reset()
    m = ConfigManager.get()
    c = {t: cls() for t, cls in TYPE_MAP.items()}
    d = c[ConfigManager.DATA]
    d.cache_dir, d.hemi, d.target_surface = cache, str(canon["hemi"]), str(canon["surface"])
    d.source_surface, d.x0_mode = source, x0_mode
    d.train_subjects, d.val_subjects, d.test_subjects = list(ck["train"]), list(ck["val"]), list(ck["test"])
    d.n_val, d.n_test = len(ck["val"]), len(ck["test"])

    au = c[ConfigManager.AUGMENT]
    au.enabled = not g.get("no_augment", False)
    au.rotation_p, au.rotation_degrees = g.get("rot_p", 0.3), g.get("rot_deg", 15.0)
    au.noise_p, au.noise_std = g.get("noise_p", 0.3), tuple(g.get("noise_std", (0.0, 0.1)))
    au.gamma_p, au.gamma_log_gamma = g.get("gamma_p", 0.3), tuple(g.get("log_gamma", (-0.3, 0.3)))

    mo = c[ConfigManager.MODEL]
    mo.architecture = "closed_loop" if g.get("model") == "closed_loop" else "flow"
    mo.use_t1 = bool(ck["use_t1"])
    mo.encoder.type, mo.encoder.blocks_per_level = "plain", 1
    mo.encoder.base_features, mo.encoder.depth = g["base"], g["levels"]
    mo.encoder.pe_channels = g.get("pe_ch", 8) if mo.architecture == "closed_loop" else 0
    mo.encoder.pe_stride = g.get("pe_stride", 8)
    mo.head.hidden, mo.head.layers = g["hidden"], g["layers"]
    mo.head.n_freq_x, mo.head.n_freq_t = g.get("n_freq_x", 6), g.get("n_freq_t", 4)

    lo = c[ConfigManager.LOSS]
    lo.beta_nll, lo.normal_weight = g.get("beta_nll", 0.5), g.get("normal_weight", 0.1)
    lo.tangential_weight = g.get("tangential_weight", 0.1)

    co = c[CONTROL]
    co.perturb_mm, co.onpolicy_p = g.get("perturb_mm", 0.5), g.get("onpolicy_p", 0.5)
    co.onpolicy_warmup, co.rollout_steps = g.get("onpolicy_warmup", 1000), g.get("rollout_steps", 6)
    co.val_euler_steps, co.kalman_q_mm = g.get("eval_steps", 50), a.kalman_q_mm

    op = c[ConfigManager.OPTIMIZER]
    op.lr, op.weight_decay = g.get("lr", 1e-3), g.get("wd", 0.0)
    tr = c[ConfigManager.TRAINING]
    tr.steps, tr.gradient_accumulation, tr.points = g["steps"], g.get("accum", 1), g.get("points", 16384)
    tr.amp, tr.select_on = not g.get("no_amp", False), g.get("select_on", "surface")
    tr.eval_train_subjects = g.get("eval_subjects", 8)
    inf = c[ConfigManager.INFRA]
    inf.output_dir, inf.experiment_name, inf.seed = a.output_dir, a.experiment_name, g.get("seed", 0)
    inf.val_interval, inf.save_interval = g.get("eval_every", 2000), g.get("save_every", 1000)
    for t, cfg in c.items():
        m.register(t, cfg)

    exp = Path(a.output_dir) / a.experiment_name
    (exp / "checkpoints").mkdir(parents=True, exist_ok=True)
    m.save_all(str(exp))
    best = ck.get("best_val", float("inf"))
    torch.save({"model": ck["model"], "step": ck["step"], "best_score": best, "configs": configs_snapshot(),
                "imported_from": os.path.abspath(a.checkpoint)}, exp / "checkpoints" / "best.pth")
    m.print_all()
    print("imported %s (step %d) -> %s" % (a.checkpoint, ck["step"], exp))


if __name__ == "__main__":
    main()
