"""
Run inference and evaluation for a Cortical Surface Flow experiment.

Usage
-----
    python -m apps.cortical_surface_flow.test ^
        --config_dir output/lh_white_pial_001 ^
        --checkpoint output/lh_white_pial_001/checkpoints/best.pth

Outputs (under --output_dir, default <config_dir>/test_eval)
--------------------------------------------------------------
    predictions/<subject>_<hemi>.<target>.pred.surf.gii          prediction, MNI RAS mm
    predictions/<subject>_<hemi>.<target>.pred_scanner.surf.gii  prediction, the subject's
                                                                 scanner RAS (overlays the raw T1)
    predictions/<subject>_<hemi>.<target>.sigma.func.gii         closed loop: predicted posterior
                                                                 sigma (normal, tangential), mm
    metrics.json   per subject and mean, model and no-T1 references (see metrics.py)
    summary.md     the same as tables
    renders/       sweep, error and T1-alignment figures for the first --render subjects

The prediction is on the fsaverage 164k vertices of the hemisphere (the cache's
topology), not on the subject's native FreeSurfer mesh.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from configuration.manager import ConfigManager
from apps.cortical_surface_flow.dataset import SurfacePairData
from apps.cortical_surface_flow.geometry import (
    GridFrame, apply_affine, subject_to_scanner, utf8_stdout, write_func_gifti, write_gifti)
from apps.cortical_surface_flow.metrics import REFERENCES, ReferencePriors, evaluate_pred
from apps.cortical_surface_flow.model import build_model, set_kalman

COLS = [("sym", "surface (sym)"), ("t2p95", "target→pred p95"), ("corr", "corr"), ("c_p95", "corr p95"),
        ("fold4", "fold r4"), ("fold16", "fold r16"), ("famp4", "fold amp"), ("flip", "flipped"),
        ("contact", "self-contact"), ("c_tgt", "self-contact (target)")]


def _row(r) -> dict:
    sm, fd = r["smoothness"], r.get("fold_detail", {})
    return {"sym": r["d_surface_symmetric_mean"], "t2p95": r["d_surface_target_to_pred"]["p95"],
            "corr": r["correspondence_error"]["mean"], "c_p95": r["correspondence_error"]["p95"],
            "fold4": fd.get("fold_r4", float("nan")), "fold16": fd.get("fold_r16", float("nan")),
            "famp4": fd.get("fold_amp_ratio4", float("nan")), "flip": sm["flipped_faces_vs_target"],
            "contact": sm["self_contact_frac_pred"], "c_tgt": sm["self_contact_frac_target"]}


def _table(rows: List[List]) -> str:
    out = ["| variant | " + " | ".join(n for _, n in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for label, r in rows:
        out.append("| %s | " % label + " | ".join("%.3f" % r[k] for k, _ in COLS) + " |")
    return "\n".join(out)


def run_test(config_dir: str, checkpoint: Optional[str] = None, output_dir: Optional[str] = None,
             split: str = "test", subjects: Optional[List[str]] = None, limit: Optional[int] = None,
             euler_steps: Optional[int] = None, kalman_q_mm: Optional[float] = None,
             references: bool = True, save_surfaces: bool = True, render: int = 3,
             device: str = "auto", log: Optional[logging.Logger] = None) -> dict:
    from apps.cortical_surface_flow.train import CONTROL, load_manager_from_directory

    log = log or logging.getLogger("test")
    ConfigManager.reset()
    m = load_manager_from_directory(config_dir)
    dc, mc, cc = m.get_config(ConfigManager.DATA), m.get_config(ConfigManager.MODEL), m.get_config(CONTROL)
    checkpoint = checkpoint or os.path.join(config_dir, "checkpoints", "best.pth")
    out = Path(output_dir or os.path.join(config_dir, "test_eval"))
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       ("cpu" if device == "auto" else device))
    euler_steps = euler_steps or cc.euler_steps
    kalman_q_mm = cc.kalman_q_mm if kalman_q_mm is None else kalman_q_mm

    frame = GridFrame(dc.cache_dir, dc.hemi)
    model = build_model(mc, frame).to(dev)
    ck = torch.load(checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    set_kalman(model, kalman_q_mm)

    if subjects:
        groups = {"eval": list(subjects)}
    else:
        pick = {"test": ["test"], "val": ["val"], "train": ["train"], "all": ["train", "val", "test"]}[split]
        groups = {g: list(getattr(dc, "%s_subjects" % g) or [])[:limit] for g in pick}
    data = SurfacePairData(dc, frame, dev)
    priors = None
    if references and dc.train_subjects:
        t0 = time.time()
        priors = ReferencePriors(((data.x0_mm(s), data.x1_mm(s)) for s in dc.train_subjects), frame.faces)
        log.info("reference priors from %d training subjects (%.0f s)" % (priors.n, time.time() - t0))
    variants = ("model",) + (REFERENCES if priors is not None else ())
    task = "%s.%s -> %s.%s" % (dc.hemi, dc.source_surface, dc.hemi, dc.target_surface)
    log.info("evaluating %s on %s | %s | %d Euler steps | Kalman %s"
             % (checkpoint, {g: len(v) for g, v in groups.items()}, task, euler_steps,
                ("q=%.2f mm" % kalman_q_mm) if kalman_q_mm else "off"))

    results = {"checkpoint": checkpoint, "step": ck.get("step"), "task": task, "x0_mode": dc.x0_mode,
               "euler_steps": euler_steps, "kalman_q_mm": kalman_q_mm, "groups": {}, "summary": {}}
    rendered = 0
    for g, names in groups.items():
        results["groups"][g] = {}
        for s in names:
            t0 = time.time()
            data.add([s])
            r = data.subjects[s]
            vol = data.network_input(s)
            with torch.no_grad():
                feats = model.encode(vol) if model.use_t1 else None
                pred_n, snaps = model.integrate(r["x0"], feats, n_steps=euler_steps,
                                                return_at=(0.25, 0.5, 0.75) if rendered < render else ())
            pred = frame.to_mm(pred_n.cpu().numpy())
            x0, x1 = data.x0_mm(s), data.x1_mm(s)
            res = {"model": evaluate_pred(pred, x1, frame.faces, frame.edges, x0)}
            if priors is not None:
                for k in REFERENCES:
                    res[k] = evaluate_pred(priors.predict(k, x0, frame.faces), x1, frame.faces, frame.edges, x0)
            results["groups"][g][s] = res
            if save_surfaces:
                stem = out / "predictions" / ("%s_%s.%s" % (s, dc.hemi, dc.target_surface))
                write_gifti(str(stem) + ".pred.surf.gii", pred, frame.faces, space_code=4)
                M = subject_to_scanner(dc.cache_dir, s)
                if M is not None:
                    write_gifti(str(stem) + ".pred_scanner.surf.gii", apply_affine(M, pred.astype(np.float64)),
                                frame.faces, space_code=1)    # NIFTI_XFORM_SCANNER_ANAT
                if model.last_sigma is not None:
                    write_func_gifti(str(stem) + ".sigma.func.gii",
                                     {"sigma_normal_mm": model.last_sigma[0].cpu().numpy(),
                                      "sigma_tangential_mm": model.last_sigma[1].cpu().numpy()})
            if rendered < render:
                _render(out / "renders", s, frame, dc, vol.cpu().numpy(), x0, x1, pred,
                        {t: frame.to_mm(v.cpu().numpy()) for t, v in snaps.items()})
                rendered += 1
            row = _row(res["model"])
            log.info("  %-5s %-18s surface %.3f mm  corr %.3f mm  fold r4 %.3f  (%.0f s)"
                     % (g, s, row["sym"], row["corr"], row["fold4"], time.time() - t0))
            data.subjects.pop(s)                         # one subject in memory at a time

    md = ["# %s -- %s" % (os.path.basename(os.path.normpath(config_dir)), task), "",
          "checkpoint `%s` (step %s) | %d Euler steps | Kalman %s | x0 %s" % (
              checkpoint, ck.get("step"), euler_steps, ("q=%.2f mm" % kalman_q_mm) if kalman_q_mm else "off",
              dc.x0_mode), "",
          "Distances in mm (exact point-to-triangle); fold r: correspondence-free fold-detail "
          "correlation; flipped / self-contact: fractions.  References use no T1.", ""]
    for g, subj in results["groups"].items():
        if not subj:
            continue
        summ = {}
        for v in variants:
            rows = [_row(r[v]) for r in subj.values()]
            summ[v] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
        results["summary"][g] = summ
        md += ["## %s (mean over %d subjects)" % (g, len(subj)), "",
               _table([(v, summ[v]) for v in variants]), ""]
        md += ["### per subject (model)", "", _table([(s, _row(r["model"])) for s, r in subj.items()]), ""]
    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    for g, summ in results["summary"].items():
        for v in variants:
            r = summ[v]
            log.info("%-5s %-24s surface %.3f  corr %.3f  fold r4/r16 %.3f/%.3f  flipped %.3f  contact %.3f"
                     % (g, v, r["sym"], r["corr"], r["fold4"], r["fold16"], r["flip"], r["contact"]))
    log.info("wrote %s" % out)
    return results


def _render(d: Path, s, frame, dc, vol, x0, x1, pred, snaps) -> None:
    from apps.cortical_surface_flow.visualize import alignment_figure, error_figure, sweep_figure
    d.mkdir(parents=True, exist_ok=True)
    meshes = [("source", x0)] + [("t=%.2f" % t, snaps[t]) for t in sorted(snaps)] + \
             [("prediction", pred), ("target", x1)]
    title = "%s  %s.%s -> %s" % (s, dc.hemi, dc.source_surface, dc.target_surface)
    for name, fig in (("sweep", sweep_figure(meshes, frame.faces, dc.hemi, 420, title)),
                      ("error", error_figure(pred, x1, frame.faces, dc.hemi, 420)),
                      ("alignment", alignment_figure(frame, vol, x1, pred, s))):
        fig.savefig(str(d / ("%s_%s.png" % (s, name))), dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)


def main(argv=None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description="Cortical Surface Flow -- evaluation / inference",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config_dir", required=True, help="Experiment directory (its *.json configs)")
    p.add_argument("--checkpoint", default=None, help="Default: <config_dir>/checkpoints/best.pth")
    p.add_argument("--output_dir", default=None, help="Default: <config_dir>/test_eval")
    p.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    p.add_argument("--subjects", nargs="*", default=None, help="Explicit subjects instead of a split")
    p.add_argument("--limit", type=int, default=None, help="First N subjects per split")
    p.add_argument("--euler_steps", type=int, default=None, help="Default: ControlConfig.euler_steps")
    p.add_argument("--kalman_q_mm", type=float, default=None,
                   help="Default: ControlConfig.kalman_q_mm; 0 = plain closed loop")
    p.add_argument("--no_references", action="store_true", help="Skip the no-T1 reference rows")
    p.add_argument("--no_save_surfaces", action="store_true")
    p.add_argument("--render", type=int, default=3, help="Render figures for the first N subjects")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    run_test(a.config_dir, a.checkpoint, a.output_dir, a.split, a.subjects, a.limit, a.euler_steps,
             a.kalman_q_mm, not a.no_references, not a.no_save_surfaces, a.render, a.device)


if __name__ == "__main__":
    main()
