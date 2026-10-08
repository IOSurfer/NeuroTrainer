"""
Coordinate frames, mesh utilities and surface I/O.

Frames (all related by 4x4 affines, applied to a whole vertex array in one matmul):

    mm    world millimetres: MNI RAS (the cache's T1 grid and every cached surface)
    norm  u = (mm - centre) / half_fov, isotropic -- what the network and loss use
    vox   voxel index into the T1 grid (the MNI152 grid has a flipped x axis)

Cache layout (written by preprocess.py)::

    <cache>/preprocess.json
    <cache>/canonical_<hemi>.npz          fsaverage faces of the hemisphere + T1 grid
    <cache>/<subject>/T1_N4_mni.nii.gz    bias-corrected T1 on the MNI152 grid
    <cache>/<subject>/Brainmask_mni.nii.gz
    <cache>/<subject>/T1_input_mni.nii.gz z-scored in the mask, z_outside elsewhere
    <cache>/<subject>/<hemi>.<surface>.surf.gii   fsaverage 164k vertices, MNI mm
    <cache>/<subject>/qc.json             registration (fixed_to_moving_ras) + QC

A single-hemisphere ``canonical.npz`` (earlier caches) is read as well.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.spatial import cKDTree

FSAVERAGE_NAME = "tpl-fsaverage_den-164k_hemi-{H}_{surf}.surf.gii"


def utf8_stdout() -> None:
    """ConfigManager.print_all draws box characters; a GBK Windows console (or a
    redirected stdout) cannot encode them."""
    import sys
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


# ── Affines ──────────────────────────────────────────────────────────────────


def apply_affine(T, V):
    """Batched affine transform of a whole vertex array: (4, 4) x (N, 3) -> (N, 3).
    Works for numpy and torch alike."""
    return V @ T[:3, :3].T + T[:3, 3]


def scale_translate(scale: float, center) -> np.ndarray:
    """4x4 taking mm to u = (mm - center) / scale."""
    T = np.eye(4)
    T[:3, :3] = np.eye(3) / scale
    T[:3, 3] = -np.asarray(center, np.float64) / scale
    return T


# ── Surface I/O ──────────────────────────────────────────────────────────────


def read_gifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    g = nib.load(path)
    return (g.agg_data("NIFTI_INTENT_POINTSET").astype(np.float64),
            g.agg_data("NIFTI_INTENT_TRIANGLE").astype(np.int64))


def write_gifti(path: str, verts, faces, space_code: int = 4) -> None:
    """Surface GIFTI, coordinates in RAS mm (NIfTI xform code: 4 = MNI152, 1 = scanner)."""
    cs = nib.gifti.GiftiCoordSystem(dataspace=space_code, xformspace=space_code, xform=np.eye(4))
    g = nib.gifti.GiftiImage()
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(
        np.asarray(verts, np.float32), intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32", coordsys=cs))
    g.add_gifti_data_array(nib.gifti.GiftiDataArray(
        np.asarray(faces, np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"))
    nib.save(g, path)


def write_func_gifti(path: str, columns: Dict[str, np.ndarray]) -> None:
    """Per-vertex scalar maps (e.g. predicted sigma), one data array per column."""
    g = nib.gifti.GiftiImage()
    for name, v in columns.items():
        da = nib.gifti.GiftiDataArray(np.asarray(v, np.float32), intent="NIFTI_INTENT_NONE",
                                      datatype="NIFTI_TYPE_FLOAT32")
        da.meta = nib.gifti.GiftiMetaData({"Name": name})
        g.add_gifti_data_array(da)
    nib.save(g, path)


def fsaverage_path(fsaverage_dir: str, hemi: str, surf: str) -> str:
    return os.path.join(fsaverage_dir, FSAVERAGE_NAME.format(H=hemi[0].upper(), surf=surf))


def subject_surface_path(cache: str, subject: str, hemi: str, surface: str) -> str:
    return os.path.join(cache, subject, "%s.%s.surf.gii" % (hemi, surface))


def load_subject_surface(cache: str, subject: str, hemi: str, surface: str) -> np.ndarray:
    p = subject_surface_path(cache, subject, hemi, surface)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            "%s missing -- add it with: python -m apps.cortical_surface_flow.preprocess "
            "--cache_dir %s --hemis %s --surfaces %s" % (p, cache, hemi, surface))
    return read_gifti(p)[0].astype(np.float32)


def list_subjects(cache: str, hemi: str, surfaces) -> list:
    """Cached subjects that have the T1 and every requested surface."""
    out = []
    for d in sorted(glob.glob(os.path.join(cache, "*", "T1_N4_mni.nii.gz"))):
        s = os.path.basename(os.path.dirname(d))
        if all(os.path.isfile(subject_surface_path(cache, s, hemi, x)) for x in surfaces):
            out.append(s)
    return out


# ── Mesh utilities ───────────────────────────────────────────────────────────


def vertex_faces(faces: np.ndarray, n: int) -> np.ndarray:
    """(n, max_valence) table of incident face ids, padded with -1."""
    vv = faces.reshape(-1)
    ff = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(vv, kind="stable")
    vv, ff = vv[order], ff[order]
    counts = np.bincount(vv, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(len(vv)) - np.repeat(starts, counts)
    adj = np.full((n, int(counts.max())), -1, np.int64)
    adj[vv, pos] = ff
    return adj


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    return np.unique(np.sort(e, 1), axis=0)


def face_normals(v: np.ndarray, f: np.ndarray):
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / (a[:, None] + 1e-12), 0.5 * a


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    t = v[f]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def sphere_barycentric(dirs: np.ndarray, sph_v: np.ndarray, faces: np.ndarray, k: int = 3):
    """For each unit direction, the containing triangle of the sphere `sph_v` and its
    barycentric weights (ray cast from the origin).  -> (idx (N,3), w (N,3), inside)"""
    u = sph_v / np.linalg.norm(sph_v, axis=1, keepdims=True)
    _, nn = cKDTree(u).query(dirs, k=k)
    nn = nn.reshape(len(dirs), k)
    adj = vertex_faces(faces, len(u))
    cand = adj[nn].reshape(len(dirs), -1)
    valid = cand >= 0
    tri = u[faces[np.where(valid, cand, 0)]]                  # (N, C, 3, 3)
    M = np.swapaxes(tri, -1, -2)                              # columns = a, b, c
    rhs = np.broadcast_to(dirs[:, None, :, None], M.shape[:-1] + (1,))
    w = np.linalg.solve(M, rhs)[..., 0]
    s = w.sum(-1)
    bary = w / np.where(np.abs(s) < 1e-12, 1e-12, s)[..., None]
    score = np.where(valid & (s > 0), bary.min(-1), -np.inf)
    best = score.argmax(-1)
    rows = np.arange(len(dirs))
    idx = faces[cand[rows, best]]
    w = np.clip(bary[rows, best], 0.0, None)
    w /= w.sum(1, keepdims=True)
    inside = float((score.max(-1) >= -1e-6).mean())
    return idx, w, inside


def similarity_residual(a: np.ndarray, b: np.ndarray):
    """Per-point distance after the best similarity transform a -> b, and its scale."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    U, S, Vt = np.linalg.svd(b0.T @ a0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (a0 ** 2).sum())
    return np.linalg.norm(s * a0 @ R.T - b0, axis=1), s


def vertex_normals_torch(v: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    t = v[faces]
    fn = torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)
    vn = torch.zeros_like(v)
    for k in range(3):
        vn.index_add_(0, faces[:, k], fn)
    return torch.nn.functional.normalize(vn, dim=1, eps=1e-12)


# ── Grid frame ───────────────────────────────────────────────────────────────


class GridFrame:
    """The T1 grid shared by every subject of a cache plus one hemisphere's fsaverage
    faces, with the mm / norm / vox affines between them."""

    def __init__(self, cache: str, hemi: str) -> None:
        p = os.path.join(cache, "canonical_%s.npz" % hemi)
        if not os.path.isfile(p):
            p = os.path.join(cache, "canonical.npz")             # earlier single-hemi caches
            if not os.path.isfile(p):
                raise FileNotFoundError("no canonical_%s.npz in %s -- run preprocess.py" % (hemi, cache))
        z = np.load(p)
        if str(z["hemi"]) != hemi:
            raise ValueError("%s holds hemisphere %s, not %s" % (p, z["hemi"], hemi))
        self.cache, self.hemi, self.source = cache, hemi, p
        self.faces = z["faces"].astype(np.int64)
        self.shape = tuple(int(s) for s in z["shape"])
        self.vox2mm = z["affine"].astype(np.float64)
        self.z_outside = float(z["z_outside"]) if "z_outside" in z else -10.0
        self.template = str(z["template"]) if "template" in z else ""
        self.sphere_source = str(z["sphere_source"]) if "sphere_source" in z else ""
        self.mm2vox = np.linalg.inv(self.vox2mm)
        self.spacing = np.linalg.norm(self.vox2mm[:3, :3], axis=0)
        n = np.array(self.shape, np.float64)
        self.center = apply_affine(self.vox2mm, ((n - 1) / 2.0)[None])[0]
        self.half_fov = float((n * self.spacing).max() / 2.0)
        self.mm2norm = scale_translate(self.half_fov, self.center)
        self.norm2mm = np.linalg.inv(self.mm2norm)
        self.norm2vox = self.mm2vox @ self.norm2mm
        self._edges = None

    @property
    def edges(self) -> np.ndarray:
        if self._edges is None:
            self._edges = mesh_edges(self.faces)
        return self._edges

    @property
    def n_vertices(self) -> int:
        return int(self.faces.max()) + 1

    def fsaverage_dir(self) -> str:
        pp = os.path.join(self.cache, "preprocess.json")
        if os.path.isfile(pp):
            d = json.load(open(pp, encoding="utf-8")).get("default", {}).get("fsaverage_dir")
            if d:
                return d
        return os.path.dirname(self.sphere_source)

    def to_norm(self, mm):
        return apply_affine(self.mm2norm.astype(np.float32), np.asarray(mm, np.float32))

    def to_mm(self, norm):
        return apply_affine(self.norm2mm.astype(np.float32), np.asarray(norm, np.float32))

    def to_idx(self, mm):
        return apply_affine(self.mm2vox.astype(np.float32), np.asarray(mm, np.float32))

    def outside_grid(self, mm) -> float:
        """Fraction of points outside the T1 grid."""
        i = self.to_idx(mm)
        return float(((i < 0) | (i > np.array(self.shape) - 1)).any(1).mean())

    def __repr__(self) -> str:
        return ("GridFrame(hemi=%s, grid=%s, spacing=%s mm, centre=%s, half_fov=%.2f mm, "
                "%d vertices, %d faces)"
                % (self.hemi, self.shape, np.round(self.spacing, 3).tolist(),
                   np.round(self.center, 2).tolist(), self.half_fov, self.n_vertices, len(self.faces)))


def load_subject_images(cache: str, subject: str) -> Tuple[np.ndarray, np.ndarray]:
    """(T1_N4 on the MNI grid, Brainmask as bool)."""
    d = os.path.join(cache, subject)
    t1 = np.asarray(nib.load(os.path.join(d, "T1_N4_mni.nii.gz")).dataobj, np.float32)
    mask = np.asarray(nib.load(os.path.join(d, "Brainmask_mni.nii.gz")).dataobj) > 0
    return t1, mask


def subject_to_scanner(cache: str, subject: str) -> Optional[np.ndarray]:
    """4x4 MNI RAS -> the subject's scanner RAS (elastix's fixed -> moving)."""
    p = os.path.join(cache, subject, "qc.json")
    if not os.path.isfile(p):
        return None
    m = json.load(open(p)).get("fixed_to_moving_ras")
    return np.asarray(m, np.float64) if m is not None else None
