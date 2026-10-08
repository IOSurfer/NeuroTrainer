"""
Surface metrics (all distances in mm) and the no-T1 reference predictions.

surface distance      exact point-to-triangle, both directions and their mean
correspondence error  |pred_i - target_i| through the fsaverage vertex correspondence
fold detail           correspondence-free: each target vertex is matched to the nearest
                      predicted vertex by position and their signed fold heights
                      (v - smooth^k v) . n are correlated, at bands k = 4 and 16.  A
                      smooth sheet threaded through the folds scores well on surface
                      distance but not here; sliding along the surface is not counted.
mesh quality          flipped faces vs the target, self-contact, edge stretch

References (what a predictor that ignores the T1 achieves):
    ref_x0                 the source surface itself
    ref_mean_normal_offset source + the training set's mean per-vertex offset along the
                           source normal (white -> pial: the population thickness)
    ref_mean_displacement  source + the training set's mean per-vertex displacement
                           vector (better for large deformations, e.g. -> inflated)
    ref_train_mean         the training set's mean target surface
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from apps.cortical_surface_flow.geometry import (
    face_normals, mesh_edges, vertex_faces, vertex_normals)


def stats(d: np.ndarray) -> Dict[str, float]:
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max())}


# ── Point to mesh ─────────────────────────────────────────────────────────────


def _pt_tri_dist2(p, a, b, c):
    """Squared distance from p to triangle (a, b, c); broadcasting, last axis = 3."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def div(n, d):
        return n / np.where(np.abs(d) < 1e-20, 1e-20, d)

    v, w = div(vb, va + vb + vc), div(vc, va + vb + vc)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    tt = div(d4 - d3, (d4 - d3) + (d5 - d6))
    v, w = np.where(m, 1 - tt, v), np.where(m, tt, w)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    tt = div(d2, d2 - d6)
    v, w = np.where(m, 0.0, v), np.where(m, tt, w)
    m = (d6 >= 0) & (d5 <= d6)
    v, w = np.where(m, 0.0, v), np.where(m, 1.0, w)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    tt = div(d1, d1 - d3)
    v, w = np.where(m, tt, v), np.where(m, 0.0, w)
    m = (d3 >= 0) & (d4 <= d3)
    v, w = np.where(m, 1.0, v), np.where(m, 0.0, w)
    m = (d1 <= 0) & (d2 <= 0)
    v, w = np.where(m, 0.0, v), np.where(m, 0.0, w)
    q = a + v[..., None] * ab + w[..., None] * ac
    return ((p - q) ** 2).sum(-1)


def point_mesh_distance(pts, verts, faces, k=8, chunk=4096) -> np.ndarray:
    """Exact point-to-triangle distance, over the faces around the k nearest vertices."""
    tree = cKDTree(verts)
    adj = vertex_faces(faces, len(verts))
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        _, nn = tree.query(q, k=k)
        cand = adj[nn.reshape(len(q), k)].reshape(len(q), -1)
        valid = cand >= 0
        tri = verts[faces[np.where(valid, cand, 0)]]
        d2 = _pt_tri_dist2(q[:, None, :], tri[..., 0, :], tri[..., 1, :], tri[..., 2, :])
        out[s:s + chunk] = np.sqrt(np.where(valid, d2, np.inf).min(1))
    return out


def nn_surface_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric nearest-vertex distance: fast stand-in used during training."""
    return float(0.5 * (cKDTree(q).query(p)[0].mean() + cKDTree(p).query(q)[0].mean()))


# ── Fold detail ───────────────────────────────────────────────────────────────

_SMOOTHER: Dict[tuple, sparse.csr_matrix] = {}


def _smoother(faces: np.ndarray, n: int) -> sparse.csr_matrix:
    """Row-normalised vertex adjacency (uniform Laplacian), cached per mesh."""
    key = (faces.shape, int(faces[:16].sum()), n)
    if key not in _SMOOTHER:
        e = mesh_edges(faces)
        A = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                              shape=(n, n)).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        _SMOOTHER[key] = sparse.diags(1.0 / np.maximum(deg, 1)) @ A
    return _SMOOTHER[key]


def fold_height(v: np.ndarray, faces: np.ndarray, bands: Iterable[int] = (4, 16)) -> Dict[int, np.ndarray]:
    """Signed fold height per vertex and band: (v - smooth^k(v)) . n -- positive on gyral
    crowns, negative in sulcal fundi."""
    W = _smoother(faces, len(v))
    n = vertex_normals(v, faces)
    out, s, done = {}, v.copy(), 0
    for k in sorted(bands):
        for _ in range(k - done):
            s = s + 0.5 * (W @ s - s)
        done = k
        out[k] = ((v - s) * n).sum(1)
    return out


def fold_detail(pred: np.ndarray, target: np.ndarray, faces: np.ndarray,
                bands: Iterable[int] = (4, 16)) -> Dict[str, float]:
    hp, ht = fold_height(pred, faces, bands), fold_height(target, faces, bands)
    nn = cKDTree(pred).query(target)[1]
    out = {}
    for k in bands:
        out["fold_r%d" % k] = float(np.corrcoef(hp[k][nn], ht[k])[0, 1])
        out["fold_r%d_vertexwise" % k] = float(np.corrcoef(hp[k], ht[k])[0, 1])
        out["fold_amp_ratio%d" % k] = float(hp[k].std() / ht[k].std())
    return out


# ── Mesh quality ──────────────────────────────────────────────────────────────


def self_contact(v: np.ndarray, edges: np.ndarray, k: int = 16) -> float:
    """Fraction of vertices whose nearest non-1-ring vertex is closer than half a local
    edge length: the sheet has folded onto itself."""
    n = len(v)
    l = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    s = np.zeros(n)
    np.add.at(s, edges[:, 0], l)
    np.add.at(s, edges[:, 1], l)
    cnt = np.bincount(edges.reshape(-1), minlength=n).astype(float)
    local = s / np.maximum(cnt, 1.0)
    key = np.sort(np.concatenate([edges[:, 0] * n + edges[:, 1],
                                  edges[:, 1] * n + edges[:, 0]]).astype(np.int64))
    d, idx = cKDTree(v).query(v, k=k)
    rows = np.arange(n, dtype=np.int64)
    near = np.full(n, np.inf)
    for j in range(1, k):
        q = rows * n + idx[:, j]
        pos = np.clip(np.searchsorted(key, q), 0, len(key) - 1)
        near = np.minimum(near, np.where(key[pos] == q, np.inf, d[:, j]))
    return float(np.mean(near < 0.5 * local))


def smoothness(pred, target, faces, edges, x0) -> Dict[str, object]:
    lp = np.linalg.norm(pred[edges[:, 0]] - pred[edges[:, 1]], axis=1)
    lt = np.linalg.norm(target[edges[:, 0]] - target[edges[:, 1]], axis=1)
    l0 = np.linalg.norm(x0[edges[:, 0]] - x0[edges[:, 1]], axis=1)
    n_pred, a_pred = face_normals(pred, faces)
    n_tgt, a_tgt = face_normals(target, faces)
    return {
        "edge_log_ratio_vs_target": stats(np.abs(np.log(np.maximum(lp, 1e-9) / np.maximum(lt, 1e-9)))),
        "edge_stretch_vs_x0_pred": stats(lp / np.maximum(l0, 1e-9)),
        "flipped_faces_vs_target": float(((n_pred * n_tgt).sum(1) < 0).mean()),
        "degenerate_faces_pred": float((a_pred < 1e-6).mean()),
        "area_ratio_pred_over_target": float(a_pred.sum() / a_tgt.sum()),
        "self_contact_frac_pred": self_contact(pred, edges),
        "self_contact_frac_target": self_contact(target, edges),
    }


def evaluate_pred(pred, target, faces, edges, x0, fold: bool = True) -> Dict[str, object]:
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    d_pt = point_mesh_distance(pred, target, faces)
    d_tp = point_mesh_distance(target, pred, faces)
    out = {
        "d_surface_pred_to_target": stats(d_pt),
        "d_surface_target_to_pred": stats(d_tp),
        "d_surface_symmetric_mean": float(0.5 * (d_pt.mean() + d_tp.mean())),
        "correspondence_error": stats(np.linalg.norm(pred - target, axis=1)),
        "smoothness": smoothness(pred, target, faces, edges, np.asarray(x0, np.float64)),
    }
    if fold:
        out["fold_detail"] = fold_detail(pred, target, faces)
    return out


# ── References ────────────────────────────────────────────────────────────────


class ReferencePriors:
    """Population statistics of the training set that the no-T1 references use."""

    def __init__(self, pairs, faces) -> None:
        """pairs: iterable of (x0, x1) in mm for the training subjects."""
        off = disp = mean_x1 = None
        n = 0
        for x0, x1 in pairs:
            x0, x1 = np.asarray(x0, np.float64), np.asarray(x1, np.float64)
            o = ((x1 - x0) * vertex_normals(x0, faces)).sum(1)
            off = o if off is None else off + o
            disp = (x1 - x0) if disp is None else disp + (x1 - x0)
            mean_x1 = x1 if mean_x1 is None else mean_x1 + x1
            n += 1
        self.normal_offset, self.displacement, self.mean_x1 = off / n, disp / n, mean_x1 / n
        self.n = n

    def predict(self, kind: str, x0: np.ndarray, faces) -> np.ndarray:
        x0 = np.asarray(x0, np.float64)
        if kind == "ref_x0":
            return x0
        if kind == "ref_mean_normal_offset":
            return x0 + self.normal_offset[:, None] * vertex_normals(x0, faces)
        if kind == "ref_mean_displacement":
            return x0 + self.displacement
        if kind == "ref_train_mean":
            return self.mean_x1
        raise KeyError(kind)


REFERENCES = ("ref_x0", "ref_mean_normal_offset", "ref_mean_displacement", "ref_train_mean")
