"""
Subjects, split, in-memory store and GPU augmentation.

A subject is (T1_N4, Brainmask, X0, X1, normals of X1) on the cache's MNI grid:
X0 = the subject's own ``source_surface`` (or an fsaverage template), X1 = its
``target_surface``, both on the hemisphere's fsaverage 164k vertices.  Surfaces are
kept in the normalised frame on the training device; volumes wait as float16 on
``cache_device`` and move one subject per micro-batch.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from apps.cortical_surface_flow.geometry import (
    GridFrame, apply_affine, fsaverage_path, list_subjects, load_subject_images,
    load_subject_surface, read_gifti, vertex_normals)


def make_split(names: List[str], n_val: int, n_test: int, seed: int = 0):
    """Deterministic split (identical to the research prototype's)."""
    names = sorted(names)
    order = random.Random(seed).sample(range(len(names)), len(names))
    val = sorted(names[i] for i in order[:n_val])
    test = sorted(names[i] for i in order[n_val:n_val + n_test])
    train = sorted(names[i] for i in order[n_val + n_test:])
    return train, val, test


def resolve_split(data_cfg) -> Tuple[List[str], List[str], List[str]]:
    """Explicit lists from the config, else the deterministic split over every cached
    subject that has both surfaces.  The caller saves the resolved lists back."""
    if data_cfg.train_subjects:
        return (list(data_cfg.train_subjects), list(data_cfg.val_subjects or []),
                list(data_cfg.test_subjects or []))
    surfs = [data_cfg.target_surface] + ([data_cfg.source_surface] if data_cfg.x0_mode == "subject" else [])
    names = list_subjects(data_cfg.cache_dir, data_cfg.hemi, surfs)
    if not names:
        raise RuntimeError("no subject in %s has %s.%s" % (data_cfg.cache_dir, data_cfg.hemi, surfs))
    return make_split(names, data_cfg.n_val, data_cfg.n_test, data_cfg.split_seed)


def normalize_t1_torch(t1, mask, outside):
    """z-score inside the Brainmask, constant outside."""
    v = t1[mask]
    return torch.where(mask, (t1 - v.mean()) / v.std(), torch.full_like(t1, outside))


class SurfacePairData:
    """In-memory subjects of one hemisphere / surface pair."""

    def __init__(self, data_cfg, frame: GridFrame, device, load_volumes: bool = True) -> None:
        self.cfg, self.frame, self.device = data_cfg, frame, device
        self.mm2norm = torch.tensor(frame.mm2norm, dtype=torch.float32, device=device)
        self.subjects: Dict[str, dict] = {}
        self.load_volumes = load_volumes
        self.template_x0 = None
        if data_cfg.x0_mode == "template":
            path = fsaverage_path(frame.fsaverage_dir(), frame.hemi, data_cfg.source_surface)
            v, f = read_gifti(path)
            assert np.array_equal(f, frame.faces), "%s: faces differ from the cache" % path
            self.template_x0 = v.astype(np.float32)
        elif data_cfg.x0_mode != "subject":
            raise ValueError("x0_mode must be subject | template, got %r" % data_cfg.x0_mode)

    def x0_mm(self, name: str) -> np.ndarray:
        if self.template_x0 is not None:
            return self.template_x0
        return load_subject_surface(self.cfg.cache_dir, name, self.cfg.hemi, self.cfg.source_surface)

    def x1_mm(self, name: str) -> np.ndarray:
        return load_subject_surface(self.cfg.cache_dir, name, self.cfg.hemi, self.cfg.target_surface)

    def to_norm(self, mm) -> torch.Tensor:
        return apply_affine(self.mm2norm, torch.as_tensor(np.asarray(mm, np.float32), device=self.device)).contiguous()

    def add(self, names, store: Optional[str] = None) -> None:
        store = store or self.cfg.cache_device
        for n in names:
            if n in self.subjects:
                continue
            x0, x1 = self.x0_mm(n), self.x1_mm(n)
            rec = {"x0": self.to_norm(x0), "x1": self.to_norm(x1),
                   "n1": torch.tensor(vertex_normals(x1.astype(np.float64), self.frame.faces),
                                      dtype=torch.float32, device=self.device)}
            if self.load_volumes:
                t1, mask = load_subject_images(self.cfg.cache_dir, n)
                t1, mask = torch.from_numpy(t1).half(), torch.from_numpy(mask)
                if store == "cpu":
                    t1, mask = t1.pin_memory(), mask.pin_memory()
                else:
                    t1, mask = t1.to(store), mask.to(store)
                rec["t1"], rec["mask"] = t1, mask
            self.subjects[n] = rec

    def volume(self, name: str):
        """float32 T1 and bool mask on the training device."""
        r = self.subjects[name]
        return (r["t1"].to(self.device, non_blocking=True).float(),
                r["mask"].to(self.device, non_blocking=True))

    def network_input(self, name: str) -> torch.Tensor:
        t1, mask = self.volume(name)
        return normalize_t1_torch(t1, mask, self.frame.z_outside)

    def mean_displacement_mm(self, names) -> float:
        hf = self.frame.half_fov
        return float(np.mean([(self.subjects[n]["x1"] - self.subjects[n]["x0"]).norm(dim=1).mean().item() * hf
                              for n in names]))


def rotation_matrix(rad) -> np.ndarray:
    cx, cy, cz = np.cos(rad)
    sx, sy, sz = np.sin(rad)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


class Augmenter:
    """GPU augmentation of one subject (AugmentConfig): rotation of the T1, mask,
    target, its normals and an own-surface X0 together; noise; gamma."""

    def __init__(self, aug_cfg, frame: GridFrame, device, rotate_x0: bool) -> None:
        self.c, self.device, self.shape, self.rotate_x0 = aug_cfg, device, frame.shape, rotate_x0
        self.z_outside = frame.z_outside
        ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32, device=device)
                                           for n in frame.shape], indexing="ij"), -1).view(-1, 3)
        vox2norm = torch.tensor(frame.mm2norm @ frame.vox2mm, dtype=torch.float32, device=device)
        self.u = apply_affine(vox2norm, ijk)
        del ijk
        self.norm2vox = torch.tensor(frame.norm2vox, dtype=torch.float32, device=device)
        self.size = torch.tensor(frame.shape, dtype=torch.float32, device=device)

    def _rotate(self, t1, mask, R):
        """Output voxel u samples the input at R^T u (rotation about the grid centre)."""
        idx = apply_affine(self.norm2vox, self.u @ R)
        g = (2.0 * idx / (self.size - 1) - 1.0)[:, [2, 1, 0]].view(1, *self.shape, 3)
        t1 = F.grid_sample(t1[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        m = F.grid_sample(mask[None, None].float(), g, mode="bilinear", padding_mode="zeros",
                          align_corners=True)[0, 0] >= 0.5
        return t1, m

    def __call__(self, t1, mask, x0, x1, n1, force_rotation: bool = False):
        """-> (network input, x0, x1, n1, info)"""
        c, info = self.c, {}
        if c.enabled and c.rotation and (force_rotation or np.random.rand() < c.rotation_p):
            ang = np.random.uniform(-c.rotation_degrees, c.rotation_degrees, 3)
            R = torch.tensor(rotation_matrix(np.radians(ang)), dtype=torch.float32, device=self.device)
            t1, mask = self._rotate(t1, mask, R)
            x1, n1 = x1 @ R.T, n1 @ R.T
            if self.rotate_x0:
                x0 = x0 @ R.T
            info["rot_deg"] = ang.round(2).tolist()
        vol = normalize_t1_torch(t1, mask, self.z_outside)
        if c.enabled and c.noise and np.random.rand() < c.noise_p:
            std = float(np.random.uniform(*c.noise_std))
            vol = vol + torch.randn_like(vol) * std
            info["noise_std"] = round(std, 4)
        if c.enabled and c.gamma and np.random.rand() < c.gamma_p:
            g = float(np.exp(np.random.uniform(*c.gamma_log_gamma)))
            vol = vol.sign() * vol.abs().pow(g)
            info["gamma"] = round(g, 4)
        vol = torch.where(mask, vol, torch.full_like(vol, self.z_outside))
        return vol, x0, x1, n1, info
