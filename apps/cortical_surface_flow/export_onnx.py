"""
Export trained Cortical Surface Flow experiments to ONNX.

Two layouts (``--layout``)

split (default) -- two graphs, the integration loop stays outside
    <name>.encoder.onnx    t1 (1, 1, D, H, W)  network input on the cache's MNI grid: T1 z-scored inside
                                               the Brainmask, ``z_outside`` elsewhere (dataset.normalize_t1_torch)
                           -> f0 .. f4         the feature pyramid (run once per subject)
    <name>.estimator.onnx  x_mm (N, 3)         vertices, MNI RAS mm
                           t (1,)              path time in [0, 1)
                           f0 .. f4            the encoder's outputs
                           -> mu_mm (N, 3)     posterior mean of the end point
                              var_normal_mm2 (N,), var_tangential_mm2 (N,)   posterior variances
                              n_hat (N, 3)     end-point normal
    The caller integrates (reference: ``integrate_split`` below, ClosedLoopFlow.integrate in model.py):
        x = x0; dt = 1 / steps
        for i in range(steps): estimate at (x, i dt); fuse with the Kalman setpoint; x += dt (m - x) / (1 - i dt)
    ``euler_steps`` and ``kalman_q_mm`` are default run time settings in the estimator's metadata_props, so the
    caller can change them; vertices can be processed in chunks (features are shared), which bounds the memory.

unrolled -- one graph, ``<name>.onnx``: (t1, x0_mm) -> (pred_mm, sigma_normal_mm, sigma_tangential_mm)
    holds ``euler_steps`` unrolled copies of the estimator (weights shared, ~22k nodes), Euler steps and Kalman q
    are fixed at export time and onnxruntime needs ~13 GB for 164k vertices.

N is dynamic (fsaverage 164k vertices); the T1 grid is fixed (static shape).  mm <-> normalised
coordinates (GridFrame) are inside the graphs.  Next to the .onnx files a ``<name>.json`` holds the grid,
the vertex / face counts and the task; the same text is in the models' metadata_props["surface_flow"].

Usage
-----
    python -m apps.cortical_surface_flow.export_onnx ^
        --config_dir output/ixi_rigid_lh_white_pial output/ixi_rigid_rh_white_pial

    -> output/<experiment>/onnx/   (best.pth; --checkpoint for another one)

After the export the graphs are run with onnxruntime on the first test subject and compared
with the PyTorch model (``--no_verify`` skips this; needs the cache).

Notes
-----
* GridSample on 5-D volumes needs opset >= 20 (default); onnxruntime >= 1.17.
* nn.GroupNorm is replaced by StableGroupNorm for the export, see there.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from configuration.manager import ConfigManager
from apps.cortical_surface_flow.geometry import GridFrame, utf8_stdout
from apps.cortical_surface_flow.model import build_model, set_kalman

log = logging.getLogger("export_onnx")

INPUTS = ["t1", "x0_mm"]
OUTPUTS = ["pred_mm", "sigma_normal_mm", "sigma_tangential_mm"]
LEVELS = 5
FEATURES = ["f%d" % i for i in range(LEVELS)]
EST_INPUTS = ["x_mm", "t"] + FEATURES
EST_OUTPUTS = ["mu_mm", "var_normal_mm2", "var_tangential_mm2", "n_hat"]
CHUNK = 32768


class StableGroupNorm(nn.Module):
    """GroupNorm as explicit ops whose means run axis by axis (every sum has at most one
    image dimension of terms, equal sizes make the mean of means exact).  The exporter turns
    nn.GroupNorm into InstanceNormalization, and onnxruntime's float32 sum over the ~14 M
    voxels of one full-resolution group is off by ~1 % of the activation scale (a one-pass
    or two-pass form with a plain ReduceMean has the same error); this matches PyTorch to ~1e-5."""

    def __init__(self, gn: nn.GroupNorm) -> None:
        super().__init__()
        self.groups, self.eps, self.weight, self.bias = gn.num_groups, gn.eps, gn.weight, gn.bias

    @staticmethod
    def _mean(h):
        for d in range(h.dim() - 1, 1, -1):
            h = h.mean(d, keepdim=True)
        return h

    def forward(self, x):
        n, c = x.shape[0], x.shape[1]
        h = x.reshape(n, self.groups, c // self.groups, *x.shape[2:])
        h = h - self._mean(h)
        h = h * torch.rsqrt(self._mean(h * h) + self.eps)
        h = h.reshape(x.shape)
        affine = (1, c) + (1,) * (x.dim() - 2)
        return h * self.weight.reshape(affine) + self.bias.reshape(affine)


def _stable_norms(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, StableGroupNorm(child))
        else:
            _stable_norms(child)
    return module


class SurfaceFlowONNX(nn.Module):
    """(T1 network input, X0 in mm) -> (prediction in mm, sigma_normal, sigma_tangential).
    Calls the model's own ``integrate``, so the graph and test.py cannot drift apart."""

    def __init__(self, model: nn.Module, frame: GridFrame, euler_steps: int) -> None:
        super().__init__()
        self.model, self.euler_steps = model, int(euler_steps)
        for name, T in (("mm2norm", frame.mm2norm), ("norm2mm", frame.norm2mm)):
            self.register_buffer(name + "_A", torch.tensor(T[:3, :3], dtype=torch.float32))
            self.register_buffer(name + "_b", torch.tensor(T[:3, 3], dtype=torch.float32))

    def forward(self, t1, x0_mm):
        feats = self.model.encode(t1[0, 0]) if self.model.use_t1 else None
        x0 = x0_mm @ self.mm2norm_A.T + self.mm2norm_b
        x, _ = self.model.integrate(x0, feats, n_steps=self.euler_steps)
        sn, st = self.model.last_sigma
        return x @ self.norm2mm_A.T + self.norm2mm_b, sn, st


class EncoderONNX(nn.Module):
    """T1 network input -> the feature pyramid (one tensor per level)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, t1):
        return tuple(self.model.encode(t1[0, 0]))


class EstimatorONNX(nn.Module):
    """(vertices in mm, time, feature pyramid) -> posterior mean in mm, variances in mm^2, end-point normal:
    ``ClosedLoopFlow.estimate`` with the mm <-> normalised conversion around it."""

    def __init__(self, model: nn.Module, frame: GridFrame) -> None:
        super().__init__()
        self.model = model
        for name, T in (("mm2norm", frame.mm2norm), ("norm2mm", frame.norm2mm)):
            self.register_buffer(name + "_A", torch.tensor(T[:3, :3], dtype=torch.float32))
            self.register_buffer(name + "_b", torch.tensor(T[:3, 3], dtype=torch.float32))

    def forward(self, x_mm, t, *feats):
        x = x_mm @ self.mm2norm_A.T + self.mm2norm_b
        mu, lv_n, lv_t, n_hat = self.model.estimate(x, torch.ones_like(x[:, :1]) * t.reshape(1, 1), list(feats))
        return mu @ self.norm2mm_A.T + self.norm2mm_b, lv_n.exp(), lv_t.exp(), n_hat   # exp(lv): mm^2


def integrate_split(estimate, x0, steps: int, kalman_q_mm: Optional[float]) -> np.ndarray:
    """The closed-loop Euler integration of ClosedLoopFlow.integrate on numpy arrays, in mm, for the split
    layout.  ``estimate(x, t) -> (mu, var_n, var_t, n_hat)``.  The C++ implementation (CorticalSurfaceFlow.cpp)
    does the same per vertex."""
    x = np.array(x0, np.float32)
    dt = 1.0 / steps
    m = pn = pt = None
    for i in range(steps):
        mu, vn, vt, nh = estimate(x, i * dt)
        if not kalman_q_mm or m is None:
            m, pn, pt = mu, vn, vt
        else:
            q = dt * kalman_q_mm ** 2
            pn, pt = pn + q, pt + q
            kn, kt = pn / (pn + vn), pt / (pt + vt)
            d = mu - m
            dn = (d * nh).sum(-1, keepdims=True) * nh
            m = m + kn[:, None] * dn + kt[:, None] * (d - dn)
            pn, pt = (1 - kn) * pn, (1 - kt) * pt
        x = x + dt * (m - x) / (1.0 - i * dt)
    return x


def _load(config_dir: str, checkpoint: Optional[str], euler_steps: Optional[int],
          kalman_q_mm: Optional[float], device: torch.device):
    from apps.cortical_surface_flow.train import CONTROL, load_manager_from_directory
    ConfigManager.reset()
    m = load_manager_from_directory(config_dir)
    dc, mc, cc = m.get_config(ConfigManager.DATA), m.get_config(ConfigManager.MODEL), m.get_config(CONTROL)
    checkpoint = checkpoint or os.path.join(config_dir, "checkpoints", "best.pth")
    if not os.path.isfile(checkpoint):
        sys.exit("checkpoint not found: %s" % checkpoint)
    frame = GridFrame(dc.cache_dir, dc.hemi)
    model = build_model(mc, frame).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    steps = euler_steps or cc.euler_steps
    q = cc.kalman_q_mm if kalman_q_mm is None else kalman_q_mm
    set_kalman(model, q)
    for p in model.parameters():
        p.requires_grad_(False)
    return dc, mc, frame, model, ck, checkpoint, steps, q


def _metadata(name, dc, mc, frame, ck, checkpoint, steps, q, opset) -> dict:
    return {
        "experiment": name, "checkpoint": os.path.abspath(checkpoint), "step": ck.get("step"),
        "task": "%s.%s -> %s.%s" % (dc.hemi, dc.source_surface, dc.hemi, dc.target_surface),
        "hemi": dc.hemi, "source_surface": dc.source_surface, "target_surface": dc.target_surface,
        "x0_mode": dc.x0_mode, "architecture": mc.architecture, "encoder": mc.encoder.type,
        "euler_steps": steps, "kalman_q_mm": q or 0.0, "opset": opset,
        "topology": "fsaverage 164k", "n_vertices": frame.n_vertices, "n_faces": int(len(frame.faces)),
        "grid": {"shape": list(frame.shape), "spacing_mm": [round(float(s), 6) for s in frame.spacing],
                 "vox2mm": frame.vox2mm.tolist(), "z_outside": frame.z_outside},
        "inputs": {"t1": "(1, 1, %d, %d, %d) float32, z-score inside the Brainmask, %g outside"
                         % (*frame.shape, frame.z_outside),
                   "x0_mm": "(N, 3) float32, MNI RAS mm, fsaverage 164k vertex order"},
        "outputs": {"pred_mm": "(N, 3) MNI RAS mm", "sigma_normal_mm": "(N,)", "sigma_tangential_mm": "(N,)"},
    }


def _verify(path: str, wrapper: nn.Module, dc, frame, subject: Optional[str], ref_device: torch.device) -> dict:
    import onnxruntime as ort
    from apps.cortical_surface_flow.dataset import SurfacePairData
    subject = subject or (dc.test_subjects or dc.val_subjects or dc.train_subjects)[0]
    data = SurfacePairData(dc, frame, torch.device("cpu"))
    data.add([subject])
    t1 = data.network_input(subject).view(1, 1, *frame.shape).numpy().astype(np.float32)
    x0, x1 = data.x0_mm(subject).astype(np.float32), data.x1_mm(subject)

    t0 = time.time()
    ref = copy.deepcopy(wrapper).to(ref_device)
    with torch.no_grad():
        want = [o.cpu().numpy() for o in ref(torch.from_numpy(t1).to(ref_device), torch.from_numpy(x0).to(ref_device))]
    t_ref = time.time() - t0
    del ref

    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    sess = ort.InferenceSession(path, providers=providers)
    t0 = time.time()
    got = sess.run(OUTPUTS, {"t1": t1, "x0_mm": x0})
    t_ort = time.time() - t0

    corr = lambda p: float(np.linalg.norm(p - x1, axis=1).mean())
    res = {"subject": subject, "provider": sess.get_providers()[0],
           "pred_max_abs_mm": float(np.abs(got[0] - want[0]).max()),
           "pred_mean_abs_mm": float(np.abs(got[0] - want[0]).mean()),
           "sigma_max_abs_mm": float(max(np.abs(got[1] - want[1]).max(), np.abs(got[2] - want[2]).max())),
           "mean_dist_to_target_mm": {"onnx": corr(got[0]), "torch": corr(want[0]), "source": corr(x0)},
           "seconds": {"torch": round(t_ref, 1), "onnx": round(t_ort, 1)}}
    log.info("verify %s: |onnx - torch| max %.2e mm (mean %.2e) | vertex-wise distance to target: "
             "onnx %.3f, torch %.3f, source %.3f mm | %s %.0f s, torch %s %.0f s"
             % (subject, res["pred_max_abs_mm"], res["pred_mean_abs_mm"], res["mean_dist_to_target_mm"]["onnx"],
                res["mean_dist_to_target_mm"]["torch"], res["mean_dist_to_target_mm"]["source"],
                res["provider"].replace("ExecutionProvider", ""), t_ort, ref_device.type, t_ref))
    return res


def _verify_split(encoder_path: str, estimator_path: str, wrapper: nn.Module, dc, frame, steps: int,
                  q: Optional[float], subject: Optional[str], ref_device: torch.device) -> dict:
    import onnxruntime as ort
    from apps.cortical_surface_flow.dataset import SurfacePairData
    subject = subject or (dc.test_subjects or dc.val_subjects or dc.train_subjects)[0]
    data = SurfacePairData(dc, frame, torch.device("cpu"))
    data.add([subject])
    t1 = data.network_input(subject).view(1, 1, *frame.shape).numpy().astype(np.float32)
    x0, x1 = data.x0_mm(subject).astype(np.float32), data.x1_mm(subject)

    t0 = time.time()
    ref = copy.deepcopy(wrapper).to(ref_device)
    with torch.no_grad():
        want = ref(torch.from_numpy(t1).to(ref_device), torch.from_numpy(x0).to(ref_device))[0].cpu().numpy()
    t_ref = time.time() - t0
    del ref

    avail = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
    enc = ort.InferenceSession(encoder_path, providers=providers)
    est = ort.InferenceSession(estimator_path, providers=providers)
    t0 = time.time()
    feats = dict(zip(FEATURES, enc.run(FEATURES, {"t1": t1})))
    t_enc = time.time() - t0

    def estimate(x, t):                                       # one chunk of vertices, like the C++ implementation
        return tuple(est.run(EST_OUTPUTS, {"x_mm": x, "t": np.array([t], np.float32), **feats}))

    got = np.concatenate([integrate_split(estimate, x0[a:a + CHUNK], steps, q) for a in range(0, len(x0), CHUNK)])
    t_all = time.time() - t0

    corr = lambda p: float(np.linalg.norm(p - x1, axis=1).mean())
    res = {"subject": subject, "provider": est.get_providers()[0],
           "pred_max_abs_mm": float(np.abs(got - want).max()), "pred_mean_abs_mm": float(np.abs(got - want).mean()),
           "mean_dist_to_target_mm": {"onnx": corr(got), "torch": corr(want), "source": corr(x0)},
           "seconds": {"torch": round(t_ref, 1), "encoder": round(t_enc, 1), "onnx": round(t_all, 1)}}
    log.info("verify %s: |onnx - torch| max %.2e mm (mean %.2e) | vertex-wise distance to target: "
             "onnx %.3f, torch %.3f, source %.3f mm | %s encoder %.0f s, total %.0f s (chunks of %d), torch %s %.0f s"
             % (subject, res["pred_max_abs_mm"], res["pred_mean_abs_mm"], res["mean_dist_to_target_mm"]["onnx"],
                res["mean_dist_to_target_mm"]["torch"], res["mean_dist_to_target_mm"]["source"],
                res["provider"].replace("ExecutionProvider", ""), t_enc, t_all, CHUNK, ref_device.type, t_ref))
    return res


def _finish(path, meta, extra=None) -> None:
    import onnx
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    for k, v in {"surface_flow": json.dumps(meta), **(extra or {})}.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, str(v)
    onnx.save(proto, str(path))
    log.info("onnx check ok: %s, %d nodes, %.1f MB" % (Path(path).name, len(proto.graph.node), Path(path).stat().st_size / 1e6))


def export_split(config_dir: str, checkpoint: Optional[str] = None, output_dir: Optional[str] = None,
                 opset: int = 20, euler_steps: Optional[int] = None, kalman_q_mm: Optional[float] = None,
                 device: str = "cpu", verify: bool = True, verify_subject: Optional[str] = None,
                 verify_device: str = "auto") -> str:
    dev = torch.device(device)
    dc, mc, frame, model, ck, checkpoint, steps, q = _load(config_dir, checkpoint, euler_steps, kalman_q_mm, dev)
    name = os.path.basename(os.path.normpath(config_dir))
    out = Path(output_dir or os.path.join(config_dir, "onnx"))
    out.mkdir(parents=True, exist_ok=True)
    enc_path, est_path = out / ("%s.encoder.onnx" % name), out / ("%s.estimator.onnx" % name)
    log.info("%s | %s | step %s | split layout, %d Euler steps | Kalman %s | opset %d | %s"
             % (name, "%s.%s -> %s" % (dc.hemi, dc.source_surface, dc.target_surface), ck.get("step"), steps,
                ("q=%.2f mm" % q) if q else "off", opset, frame))
    if not model.use_t1:
        raise SystemExit("%s was trained without the T1 (--no_t1): nothing to feed the t1 input" % name)

    wrapper = SurfaceFlowONNX(model, frame, steps).to(dev).eval()              # PyTorch reference
    stable = _stable_norms(copy.deepcopy(model))
    encoder = EncoderONNX(stable).to(dev).eval()
    estimator = EstimatorONNX(stable, frame).to(dev).eval()
    t1 = torch.zeros(1, 1, *frame.shape, device=dev)
    t0 = time.time()
    with torch.no_grad():
        feats = encoder(t1)
        if len(feats) != LEVELS:
            raise SystemExit("the estimator export expects %d pyramid levels, the encoder has %d" % (LEVELS, len(feats)))
        torch.onnx.export(encoder, (t1,), str(enc_path), export_params=True, opset_version=opset,
                          do_constant_folding=True, input_names=["t1"], output_names=FEATURES, dynamo=False)
        x = torch.zeros(4096, 3, device=dev)                  # trace size only; N is dynamic
        torch.onnx.export(estimator, (x, torch.zeros(1, device=dev), *feats), str(est_path), export_params=True,
                          opset_version=opset, do_constant_folding=True, input_names=EST_INPUTS,
                          output_names=EST_OUTPUTS, dynamo=False,
                          dynamic_axes={"x_mm": {0: "n_vertices"}, **{o: {0: "n_vertices"} for o in EST_OUTPUTS}})
    log.info("traced and wrote %s, %s (%.0f s)" % (enc_path.name, est_path.name, time.time() - t0))

    meta = _metadata(name, dc, mc, frame, ck, checkpoint, steps, q, opset)
    meta["layout"] = "split"
    meta["inputs"] = {"encoder.t1": meta["inputs"]["t1"],
                      "estimator": "x_mm (N, 3) MNI RAS mm, t (1,), f0..f4 of the encoder"}
    meta["outputs"] = {"encoder": "f0..f4, the feature pyramid",
                       "estimator": "mu_mm (N, 3), var_normal_mm2 (N,), var_tangential_mm2 (N,), n_hat (N, 3)"}
    _finish(enc_path, meta)
    _finish(est_path, meta, {"euler_steps": steps, "kalman_q_mm": q or 0.0})

    if verify:
        rd = torch.device("cuda" if verify_device == "auto" and torch.cuda.is_available() else
                          ("cpu" if verify_device == "auto" else verify_device))
        meta["verification"] = _verify_split(str(enc_path), str(est_path), wrapper, dc, frame, steps, q,
                                             verify_subject, rd)
    with open(out / ("%s.json" % name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return str(est_path)


def export(config_dir: str, checkpoint: Optional[str] = None, output_dir: Optional[str] = None,
           opset: int = 20, euler_steps: Optional[int] = None, kalman_q_mm: Optional[float] = None,
           device: str = "cpu", verify: bool = True, verify_subject: Optional[str] = None,
           verify_device: str = "auto") -> str:
    import onnx

    dev = torch.device(device)
    dc, mc, frame, model, ck, checkpoint, steps, q = _load(config_dir, checkpoint, euler_steps, kalman_q_mm, dev)
    name = os.path.basename(os.path.normpath(config_dir))
    out = Path(output_dir or os.path.join(config_dir, "onnx"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("%s.onnx" % name)
    log.info("%s | %s | step %s | %d Euler steps | Kalman %s | opset %d | %s"
             % (name, "%s.%s -> %s" % (dc.hemi, dc.source_surface, dc.target_surface), ck.get("step"), steps,
                ("q=%.2f mm" % q) if q else "off", opset, frame))
    if not model.use_t1:
        raise SystemExit("%s was trained without the T1 (--no_t1): nothing to feed the t1 input" % name)

    wrapper = SurfaceFlowONNX(model, frame, steps).to(dev).eval()              # PyTorch reference
    exported = SurfaceFlowONNX(_stable_norms(copy.deepcopy(model)), frame, steps).to(dev).eval()
    t1 = torch.zeros(1, 1, *frame.shape, device=dev)
    x0 = torch.zeros(4096, 3, device=dev)                    # trace size only; N is dynamic
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(exported, (t1, x0), str(path), export_params=True, opset_version=opset,
                          do_constant_folding=True, input_names=INPUTS, output_names=OUTPUTS,
                          dynamic_axes={"x0_mm": {0: "n_vertices"}, "pred_mm": {0: "n_vertices"},
                                        "sigma_normal_mm": {0: "n_vertices"},
                                        "sigma_tangential_mm": {0: "n_vertices"}},
                          dynamo=False)
    log.info("traced and wrote %s (%.0f s)" % (path, time.time() - t0))

    meta = _metadata(name, dc, mc, frame, ck, checkpoint, steps, q, opset)
    proto = onnx.load(str(path))
    onnx.checker.check_model(proto)
    entry = proto.metadata_props.add()
    entry.key, entry.value = "surface_flow", json.dumps(meta)
    onnx.save(proto, str(path))
    log.info("onnx check ok: %d nodes, %.1f MB" % (len(proto.graph.node), path.stat().st_size / 1e6))

    if verify:
        rd = torch.device("cuda" if verify_device == "auto" and torch.cuda.is_available() else
                          ("cpu" if verify_device == "auto" else verify_device))
        meta["verification"] = _verify(str(path), wrapper, dc, frame, verify_subject, rd)
    with open(out / ("%s.json" % name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return str(path)


def main(argv: Optional[List[str]] = None) -> None:
    utf8_stdout()
    p = argparse.ArgumentParser(description="Cortical Surface Flow -- ONNX export",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config_dir", nargs="+", required=True, help="Experiment directories (their *.json configs)")
    p.add_argument("--checkpoint", default=None, help="Default: <config_dir>/checkpoints/best.pth (one experiment only)")
    p.add_argument("--output_dir", default=None, help="Default: <config_dir>/onnx")
    p.add_argument("--layout", default="split", choices=["split", "unrolled"],
                   help="split: encoder + estimator graphs, the caller integrates; unrolled: one graph, steps fixed")
    p.add_argument("--opset", type=int, default=20, help="GridSample on 5-D input needs >= 20")
    p.add_argument("--euler_steps", type=int, default=None, help="Default: ControlConfig.euler_steps")
    p.add_argument("--kalman_q_mm", type=float, default=None,
                   help="Default: ControlConfig.kalman_q_mm; 0 = plain closed loop")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device used for tracing")
    p.add_argument("--no_verify", action="store_true", help="Skip the onnxruntime vs PyTorch comparison")
    p.add_argument("--verify_subject", default=None, help="Default: first test subject")
    p.add_argument("--verify_device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device of the PyTorch reference")
    a = p.parse_args(argv)
    if a.checkpoint and len(a.config_dir) > 1:
        p.error("--checkpoint is for a single --config_dir")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for d in a.config_dir:
        (export_split if a.layout == "split" else export)(d, a.checkpoint, a.output_dir, a.opset, a.euler_steps, a.kalman_q_mm, a.device,
               not a.no_verify, a.verify_subject, a.verify_device)


if __name__ == "__main__":
    main()
