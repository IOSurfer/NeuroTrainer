"""
Figures for TensorBoard and the test report.  A small numpy splat rasteriser, so no
OpenGL dependency.

    sweep_figure      source -> t = .25, .5, .75 -> prediction -> target, two views
    error_figure      per-vertex correspondence error on prediction and target
    alignment_figure  orthogonal T1 slices with target and predicted cross-sections
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from apps.cortical_surface_flow.geometry import vertex_normals  # noqa: E402

VIEWS = {"lateral": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         "medial": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         "superior": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
         "anterior": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))}
LIGHT = np.array([-0.4, 0.5, 0.85])
LIGHT /= np.linalg.norm(LIGHT)


def hemi_views(hemi: str):
    """Lateral looks along -x for lh and +x for rh."""
    if hemi == "rh":
        return {"lateral": VIEWS["medial"], "medial": VIEWS["lateral"]}
    return {"lateral": VIEWS["lateral"], "medial": VIEWS["medial"]}


def render(v, f, cam_dir, up, size=420, ss=2, scale=1.0, center=None,
           color=None, cmap="inferno", clim=None):
    """Orthographic splat render with a depth buffer -> (size, size, 3) float image."""
    S = size * ss
    fwd = np.array(cam_dir, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array(up, float), fwd)
    right /= np.linalg.norm(right)
    upv = np.cross(fwd, right)
    R = np.stack([right, upv, fwd])
    p = (v - center) @ R.T
    n = vertex_normals(v, f) @ R.T
    px = p[:, 0] * scale * ss + S / 2.0
    py = p[:, 1] * scale * ss + S / 2.0
    shade = 0.18 + 0.82 * np.clip(n @ LIGHT, 0.0, 1.0)
    if color is None:
        rgb = np.repeat((0.80 * shade)[:, None], 3, 1)
    else:
        lo, hi = clim if clim else (float(color.min()), float(color.max()))
        cm = plt.get_cmap(cmap)(np.clip((color - lo) / max(hi - lo, 1e-9), 0, 1))[:, :3]
        rgb = cm * (0.35 + 0.65 * shade)[:, None]
    scr = np.stack([px, py], 1)
    front = n[f].mean(1)[:, 2] > 0.0
    e = np.linalg.norm(scr[f[:, 0]] - scr[f[:, 1]], axis=1)[front if front.any() else slice(None)]
    r = int(max(1, np.ceil(np.percentile(e, 80))))
    off = np.array([(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                    if dx * dx + dy * dy <= r * r], np.int64)
    X = np.round(px)[:, None].astype(np.int64) + off[None, :, 0]
    Y = np.round(py)[:, None].astype(np.int64) + off[None, :, 1]
    D = np.repeat(p[:, 2][:, None], len(off), 1)
    C = np.repeat(rgb[:, None, :], len(off), 1)
    m = (X >= 0) & (X < S) & (Y >= 0) & (Y < S)
    X, Y, D, C = X[m], Y[m], D[m], C[m]
    o = np.argsort(D, kind="stable")                      # far first, near overwrites
    img = np.ones((S, S, 3), np.float32)
    img[S - 1 - Y[o], X[o]] = C[o]
    return img.reshape(size, ss, size, ss, 3).mean((1, 3))


def frame_scale(frames, size, margin=0.92):
    allv = np.concatenate(frames, 0)
    c = 0.5 * (allv.min(0) + allv.max(0))
    return c, margin * size / (allv.max(0) - allv.min(0)).max()


def fig_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def sweep_figure(meshes, faces, hemi="lh", size=380, title=""):
    """meshes: [(label, vertices mm)] in display order."""
    views = hemi_views(hemi)
    center, scale = frame_scale([m[1] for m in meshes], size)
    fig, ax = plt.subplots(len(views), len(meshes), figsize=(2.4 * len(meshes), 2.5 * len(views)),
                           squeeze=False)
    for i, (view, (cam, up)) in enumerate(views.items()):
        for j, (name, v) in enumerate(meshes):
            ax[i][j].imshow(render(v, faces, cam, up, size, scale=scale, center=center))
            ax[i][j].axis("off")
            if i == 0:
                ax[i][j].set_title(name, fontsize=11)
            if j == 0:
                ax[i][j].text(-0.04, 0.5, view, rotation=90, va="center", ha="center",
                              transform=ax[i][j].transAxes, fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    return fig


def error_figure(pred, target, faces, hemi="lh", size=380):
    err = np.linalg.norm(pred - target, axis=1)
    hi = float(np.percentile(err, 95))
    views = hemi_views(hemi)
    center, scale = frame_scale([pred, target], size)
    fig, ax = plt.subplots(1, 2 * len(views), figsize=(2.4 * 2 * len(views), 2.9), squeeze=False)
    for i, (view, (cam, up)) in enumerate(views.items()):
        for j, (v, lab) in enumerate([(pred, "prediction"), (target, "target")]):
            a = ax[0][2 * i + j]
            a.imshow(render(v, faces, cam, up, size, scale=scale, center=center,
                            color=err, cmap="inferno", clim=(0.0, hi)))
            a.axis("off")
            a.set_title("%s %s" % (view, lab), fontsize=10)
    fig.suptitle("per-vertex correspondence error, 0 - %.2f mm (mean %.2f)" % (hi, err.mean()), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def alignment_figure(frame, vol, x1, pred=None, title=""):
    """Orthogonal T1 slices with the target (red) and prediction (cyan) cross-sections."""
    idx = frame.to_idx(x1)
    pidx = frame.to_idx(pred) if pred is not None else None
    c = np.clip(np.round(frame.to_idx(x1.mean(0)[None])[0]).astype(int), 0, np.array(frame.shape) - 1)
    fg = vol[vol > frame.z_outside + 0.5]
    lo, hi = np.percentile(fg, [0.5, 99.5]) if fg.size else (vol.min(), vol.max())
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.0))
    for k, (axis, name, (u, w)) in enumerate([(0, "sagittal", (1, 2)), (1, "coronal", (0, 2)),
                                              (2, "axial", (0, 1))]):
        ax[k].imshow(np.take(vol, c[axis], axis=axis).T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        m = np.abs(idx[:, axis] - c[axis]) < 0.6
        ax[k].scatter(idx[m, u], idx[m, w], s=0.5, c="red", label="target")
        if pidx is not None:
            m = np.abs(pidx[:, axis] - c[axis]) < 0.6
            ax[k].scatter(pidx[m, u], pidx[m, w], s=0.5, c="cyan", label="prediction")
        ax[k].set_title("%s (slice %d)" % (name, c[axis]), fontsize=10)
        ax[k].axis("off")
    if pidx is not None:
        ax[0].legend(loc="lower left", fontsize=8, markerscale=8)
    fig.suptitle(title or "T1 / surface alignment", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig
