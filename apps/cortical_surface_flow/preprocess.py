"""
Cortical Surface Flow -- preprocessing: NeuroPreprocess output -> training cache.

Usage
-----
Build (or extend) the cache::

    python -m apps.cortical_surface_flow.preprocess ^
        --raw_root D:\\NeuroPreprocess\\output ^
        --cache_dir input/cortical_surface_flow/cache ^
        --hemis lh rh --surfaces orig white pial.T1 inflated sphere --workers 6

Re-run with the saved settings::

    python -m apps.cortical_surface_flow.preprocess --config input/cortical_surface_flow/cache/preprocess.json

Per subject
-----------
1. N4 bias correction of the raw T1 (SimpleITK; field estimated on a shrunk grid).
2. FreeSurfer Brainmask resampled onto the T1 grid through world coordinates.
3. Affine elastix registration (no deformable stage) of the masked T1 to the MNI152
   brain.  The 4x4 RAS matrix (fixed -> moving) is stored in qc.json.
4. T1, mask and network input written on the exact MNI152 grid with the template's own
   header (asserted on re-read).
5. Every requested surface of every requested hemisphere: surface RAS -> scanner RAS
   (the cras stored in the surface file) -> MNI (inverse registration) -> resampled
   through ``<hemi>.sphere.reg`` onto the fsaverage 164k vertices.  All these surfaces
   share sphere.reg's vertex set, so one set of barycentric weights carries them all
   and vertex i is the same point of cortex in every cached surface.

Incremental: a subject whose registration is already cached is not re-registered;
only missing surfaces / hemispheres are added.  Any surface already in the cache is
re-derived and must match it (max |diff| < 1e-3 mm).  Subjects without FreeSurfer
output (T1 only) are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry
from scipy.ndimage import gaussian_gradient_magnitude, map_coordinates

from apps.cortical_surface_flow.geometry import (
    apply_affine, fsaverage_path, read_gifti, sphere_barycentric, subject_surface_path, utf8_stdout,
    write_gifti)
from apps.cortical_surface_flow.surface_flow_config import PreprocessConfig

HERE = os.path.dirname(os.path.abspath(__file__))
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


# ── Images ────────────────────────────────────────────────────────────────────


def n4_correct(raw, mask, shrink=4, iters=(50, 50, 50, 50)):
    """N4 bias correction; the log field is estimated on a shrunk grid and applied at
    full resolution."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(raw.transpose(2, 1, 0)))
    msk = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0).astype(np.uint8)))
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([int(i) for i in iters])
    n4.Execute(sitk.Cast(sitk.Shrink(img, [int(shrink)] * 3), sitk.sitkFloat32),
               sitk.Cast(sitk.Shrink(msk, [int(shrink)] * 3), sitk.sitkUInt8))
    bias = np.exp(sitk.GetArrayFromImage(n4.GetLogBiasFieldAsImage(img)).transpose(2, 1, 0))
    return (raw / bias).astype(np.float32), float(bias.min()), float(bias.max())


def load_native(t1_path, mask_path, cfg):
    """N4-corrected raw T1 and the conformed Brainmask on the T1 grid (scanner RAS)."""
    t1 = nib.load(t1_path)
    raw = np.asarray(t1.dataobj, np.float32)
    bm = nib.load(mask_path)
    ijk = np.argwhere(np.ones(t1.shape, bool)).astype(np.float64)
    vox = apply_affine(np.linalg.inv(bm.affine) @ t1.affine, ijk).T
    mask = map_coordinates(np.asarray(bm.dataobj, np.float32), vox, order=0,
                           mode="constant", cval=0.0).reshape(t1.shape) > 0.5
    nd, lo, hi = n4_correct(raw, mask, cfg["n4_shrink"], cfg["n4_iterations"])
    info = {"bias_min": lo, "bias_max": hi,
            "mask_ml": float(mask.sum() * np.prod(t1.header.get_zooms()) / 1000.0)}
    return nd, mask, t1.affine, info


def normalize_t1(t1, mask, outside):
    v = t1[mask]
    return np.where(mask, (t1 - v.mean()) / v.std(), outside).astype(np.float32)


def template_nifti(data, tpl):
    """Image on the template grid carrying the template's own header."""
    hdr = tpl.header.copy()
    hdr.set_data_dtype(data.dtype)
    hdr["cal_min"], hdr["cal_max"] = 0, 0
    img = nib.Nifti1Image(data, tpl.affine, hdr)
    img.set_qform(tpl.header.get_qform(), code=int(tpl.header["qform_code"]))
    img.set_sform(tpl.header.get_sform(), code=int(tpl.header["sform_code"]))
    return img


def same_geometry(img, tpl):
    h, t = img.header, tpl.header
    return bool(img.shape == tpl.shape and np.allclose(img.affine, tpl.affine)
                and np.allclose(h.get_qform(), t.get_qform()) and np.allclose(h.get_sform(), t.get_sform())
                and int(h["qform_code"]) == int(t["qform_code"]) and int(h["sform_code"]) == int(t["sform_code"])
                and np.allclose(h["pixdim"][1:4], t["pixdim"][1:4]))


# ── Registration ──────────────────────────────────────────────────────────────


def parse_elastix_affine(tp_path):
    """TransformParameters.0.txt (AffineTransform) -> 4x4 RAS matrix taking fixed
    (MNI) world points to moving (subject scanner) world points.
    elastix / ITK, in LPS: p_moving = A (p_fixed - c) + t + c."""
    txt = open(tp_path).read()
    kind = re.search(r'\(Transform "(\w+)"\)', txt).group(1)
    if kind != "AffineTransform":
        raise ValueError("%s: expected AffineTransform, got %s" % (tp_path, kind))
    p = np.array(re.search(r"\(TransformParameters ([^)]*)\)", txt).group(1).split(), float)
    c = np.array(re.search(r"\(CenterOfRotationPoint ([^)]*)\)", txt).group(1).split(), float)
    A, t = p[:9].reshape(3, 3), p[9:]
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = t + c - A @ c
    return LPS @ M @ LPS


def register_to_mni(nd, md, affine, work, fixed_mask, cfg):
    """Affine elastix registration of the masked T1 to the MNI brain (cached in work)."""
    tp = os.path.join(work, "TransformParameters.0.txt")
    if not os.path.isfile(tp):
        os.makedirs(work, exist_ok=True)
        moving = os.path.join(work, "moving.nii.gz")
        moving_mask = os.path.join(work, "moving_mask.nii.gz")
        nib.save(nib.Nifti1Image(nd * md, affine), moving)
        nib.save(nib.Nifti1Image(md.astype(np.uint8), affine), moving_mask)
        params = cfg["elastix_params"] or os.path.join(HERE, "elastix_affine.txt")
        with open(os.path.join(work, "elastix_stdout.txt"), "w") as f:
            subprocess.run([cfg["elastix"], "-f", cfg["mni_template"], "-m", moving, "-fMask", fixed_mask,
                            "-mMask", moving_mask, "-p", params, "-out", work],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
    metric = re.findall(r"Final metric value\s*=\s*([-\d.eE+]+)",
                        open(os.path.join(work, "elastix.log")).read())
    return parse_elastix_affine(tp), float(metric[-1]) if metric else float("nan")


# ── Surfaces ──────────────────────────────────────────────────────────────────


def surface_cras(surf_path):
    """Surface RAS -> scanner RAS: a translation by the cras stored in the surface file
    (these FreeSurfer outputs come without an mri/ folder)."""
    _, _, meta = read_geometry(surf_path, read_metadata=True)
    if not meta or "cras" not in meta:
        raise ValueError("%s carries no volume geometry (cras)" % surf_path)
    return np.asarray(meta["cras"], np.float64)


def grid_equal(img, tpl):
    """Same voxel grid (shape + affine); qform / sform codes may differ."""
    return bool(img.shape == tpl.shape and np.allclose(img.affine, tpl.affine, atol=1e-4))


def sitk_to_ras(t):
    """SimpleITK transform (fixed LPS point -> moving LPS point) -> 4x4 RAS matrix."""
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float) * 100.0
    Q = np.array([t.TransformPoint(tuple(p)) for p in P])
    M = np.eye(4)
    M[:3, :3] = (Q[1:] - Q[0]).T / 100.0
    M[:3, 3] = Q[0]
    return LPS @ M @ LPS


def _principal_axes(pts):
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    return c, v[:, ::-1]


def _cortex_points_lps(tar_path):
    """Both pial surfaces in native scanner space as LPS points (SimpleITK's physical space)."""
    tmp = tempfile.mkdtemp(prefix="pial_")
    try:
        extract_surfaces(tar_path, ("lh", "rh"), ("pial.T1",), tmp)
        pts = []
        for h in ("lh", "rh"):
            path = os.path.join(tmp, "%s.pial.T1" % h)
            v, _ = read_geometry(path)
            pts.append(v + surface_cras(path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    P = np.concatenate(pts)
    P[:, :2] *= -1
    return P


def register_rigid(fixed_path, moving_path, mask_path, cfg, tar_path=None):
    """Rigid (Euler) registration of the native T1 (moving) to T1_Rigid (fixed): same
    image under a rigid transform, so a scale-free correlation metric on a masked, seeded
    random sample converges to the exact transform.

    Start 1: centre-of-mass alignment (a few seconds; right for ~91 % of the subjects).
    Start 2, if the correlation stays below cfg['rigid_ncc_min']: the head can be rotated by up
    to ~90 deg, beyond any image-based capture range, so the principal axes of the two
    cortical surfaces (native space) are aligned with those of the rigid brain mask; the four
    sign choices that keep a proper rotation are each refined (4-level pyramid) and the first
    that reaches the threshold wins (else the best).
    -> M (4x4 RAS, MNI point -> native scanner point), metric, NCC of the resampled native
    T1 against T1_Rigid in the dilated brain mask, how the start was found."""
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(cfg["threads"]))
    fixed = sitk.Cast(sitk.ReadImage(fixed_path), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(moving_path), sitk.sitkFloat32)
    mask_img = sitk.ReadImage(mask_path)
    mask = sitk.BinaryDilate(mask_img > 0, [8, 8, 8])
    inside = sitk.GetArrayFromImage(mask).ravel() > 0
    fx = sitk.GetArrayFromImage(fixed).ravel()[inside].astype(np.float64)
    fx -= fx.mean()

    def run(init, shrink, sigma):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsCorrelation()
        reg.SetMetricFixedMask(mask)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(float(cfg["rigid_sampling"]), 1)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-5, numberOfIterations=300,
                                                     relaxationFactor=0.5, gradientMagnitudeTolerance=1e-8)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrink)
        reg.SetSmoothingSigmasPerLevel(sigma)
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init, inPlace=False)
        t = reg.Execute(fixed, moving)
        res = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, t, sitk.sitkLinear, 0.0)).ravel()[inside]
        a = res.astype(np.float64) - res.mean()
        ncc = float((a * fx).sum() / np.sqrt((a * a).sum() * (fx * fx).sum()))
        return t, float(reg.GetMetricValue()), ncc

    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    t, metric, ncc = run(init, [4, 2, 1], [2, 1, 0])
    how = "centre of mass"
    if ncc < cfg["rigid_ncc_min"] and tar_path is not None:
        mimg = nib.load(mask_path)
        ijk = np.argwhere(np.asarray(mimg.dataobj) > 0)[::50]
        ras = ijk @ mimg.affine[:3, :3].T + mimg.affine[:3, 3]
        ras[:, :2] *= -1
        cr, vr = _principal_axes(ras)
        cn, vn = _principal_axes(_cortex_points_lps(tar_path))
        best = (ncc, t, metric)
        for sx in (1, -1):
            for sy in (1, -1):
                R = vn @ np.diag([sx, sy, sx * sy]) @ vr.T          # rigid-space axes -> native axes
                st = sitk.Euler3DTransform()
                st.SetMatrix(tuple(R.ravel()))
                st.SetCenter(tuple(cr))
                st.SetTranslation(tuple(cn - cr))
                t2, m2, n2 = run(st, [8, 4, 2, 1], [4, 2, 1, 0])
                if n2 > best[0]:
                    best, how = (n2, t2, m2), "surface principal axes (start rotation %.0f deg)" % np.degrees(
                        np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
                if best[0] >= cfg["rigid_ncc_min"]:
                    break
            if best[0] >= cfg["rigid_ncc_min"]:
                break
        ncc, t, metric = best
    return sitk_to_ras(t), metric, ncc, how


def extract_surfaces(tar_path, hemis, surfaces, dest):
    """Only the needed members of surf.tar.gz (+ sphere.reg), flattened into `dest`."""
    want = {"%s.%s" % (h, n) for h in hemis for n in tuple(surfaces) + ("sphere.reg",)}
    with tarfile.open(tar_path) as t:
        members = [m for m in t.getmembers() if m.isfile() and os.path.basename(m.name) in want]
        missing = want - {os.path.basename(m.name) for m in members}
        if missing:
            raise FileNotFoundError("%s lacks %s" % (tar_path, sorted(missing)))
        for m in members:
            m.name = os.path.basename(m.name)
        try:
            t.extractall(dest, members=members, filter="data")
        except TypeError:                                   # Python < 3.12
            t.extractall(dest, members=members)
    return dest


def white_boundary_check(t1_path, x, mni2vox):
    """Registration QC independent of the transform's own metric: the white surface sits on
    the strongest T1 gradient, so the mean gradient magnitude at its vertices must peak at
    zero shift (+-0.5 mm along each axis)."""
    vol = np.asarray(nib.load(t1_path).dataobj, np.float32)
    grad = gaussian_gradient_magnitude(vol, 0.7)
    idx = apply_affine(mni2vox, x)
    prof = {}
    for ax, name in enumerate("xyz"):
        prof[name] = [float(map_coordinates(grad, (idx + np.eye(3)[ax] * sh).T, order=1, mode="nearest").mean())
                      for sh in (-0.5, 0.0, 0.5)]
    centre = prof["x"][1]
    return {"strength": centre, "centre_over_best": centre / max(max(v) for v in prof.values()),
            "peak_at_zero": bool(all(v[1] >= max(v) - 1e-9 for v in prof.values())), "profiles": prof}


def find_subjects_rigid(cfg):
    """rigid_dataset layout: every subject folder that has the T1_Rigid, its mask, the native
    T1 and the surface archive.  The 4th tuple element is the archive."""
    out, skipped = [], []
    for d in sorted(glob.glob(os.path.join(cfg["raw_root"], "*"))):
        if not os.path.isdir(d):
            continue
        s = os.path.basename(d)
        f = lambda pat: os.path.join(d, pat.format(subject=s))              # noqa: E731
        t1, mk, nat, tar = f(cfg["t1_pattern"]), f(cfg["mask_pattern"]), f(cfg["native_t1_pattern"]), \
            f(cfg["surf_archive_pattern"])
        if all(os.path.isfile(p) for p in (t1, mk, nat, tar)):
            out.append((s, t1, mk, tar))
        else:
            skipped.append(s)
    return out, skipped


def find_subjects(raw_root, hemis):
    """NeuroPreprocess layout: <subject>/{T1/*.nii.gz, freesurfer_brainmask/, surf/}."""
    out, skipped = [], []
    for d in sorted(glob.glob(os.path.join(raw_root, "*"))):
        if not os.path.isdir(d):
            continue
        t1 = sorted(glob.glob(os.path.join(d, "T1", "*.nii.gz")))
        mask = os.path.join(d, "freesurfer_brainmask", "freesurfer_brainmask.nii.gz")
        surf = os.path.join(d, "surf")
        if t1 and os.path.isfile(mask) and all(os.path.isfile(os.path.join(surf, h + ".sphere.reg"))
                                               for h in hemis):
            out.append((os.path.basename(d), t1[0], mask, surf))
        else:
            skipped.append(os.path.basename(d))
    return out, skipped


def write_canonical(cfg, hemi):
    """canonical_<hemi>.npz: the hemisphere's fsaverage faces + the cache's T1 grid."""
    out = os.path.join(cfg["cache_dir"], "canonical_%s.npz" % hemi)
    sph_path = fsaverage_path(cfg["fsaverage_dir"], hemi, "sphere")
    sph, faces = read_gifti(sph_path)
    tpl = nib.load(cfg["mni_template"])
    if os.path.isfile(out):
        z = np.load(out)
        assert np.array_equal(z["faces"], faces) and tuple(z["shape"]) == tpl.shape, \
            "%s disagrees with %s / %s" % (out, sph_path, cfg["mni_template"])
        return
    np.savez_compressed(out, faces=faces.astype(np.int32), hemi=hemi, shape=np.array(tpl.shape),
                        affine=tpl.affine, space="mni", template=cfg["mni_template"],
                        sphere_source=sph_path, z_outside=cfg["z_outside"])
    print("wrote %s (%d vertices, %d faces)" % (out, len(sph), len(faces)))


def process_subject(job):
    """One subject: registration + images (unless cached), then every missing surface."""
    sess, t1_path, mask_path, surf_dir, cfg = job
    t0 = time.time()
    cache = cfg["cache_dir"]
    d = os.path.join(cache, sess)
    os.makedirs(d, exist_ok=True)
    tpl = nib.load(cfg["mni_template"])
    qc_path = os.path.join(d, "qc.json")
    qc = json.load(open(qc_path)) if os.path.isfile(qc_path) else {}
    rigid = cfg["layout"] == "rigid_dataset"
    images = ("T1_mni.nii.gz", "Brainmask_mni.nii.gz") if rigid else \
        ("T1_N4_mni.nii.gz", "Brainmask_mni.nii.gz", "T1_input_mni.nii.gz")
    log = []

    if "fixed_to_moving_ras" in qc and all(os.path.isfile(os.path.join(d, n)) for n in images):
        M = np.asarray(qc["fixed_to_moving_ras"], np.float64)
    elif rigid:
        t1_img, mk_img = nib.load(t1_path), nib.load(mask_path)
        assert grid_equal(t1_img, tpl) and grid_equal(mk_img, tpl), \
            "%s: T1 / mask are not on the grid of %s" % (sess, cfg["mni_template"])
        native = os.path.join(cfg["raw_root"], sess, cfg["native_t1_pattern"].format(subject=sess))
        M, metric, ncc, how = register_rigid(t1_path, native, mask_path, cfg, surf_dir)
        assert ncc >= cfg["rigid_ncc_min"], \
            "%s: rigid registration did not converge (NCC %.5f < %.3f)" % (sess, ncc, cfg["rigid_ncc_min"])
        t1 = np.asarray(t1_img.dataobj, np.float32)
        mask = np.asarray(mk_img.dataobj) > 0
        for name, arr in (("T1_mni.nii.gz", t1), ("Brainmask_mni.nii.gz", mask.astype(np.uint8))):
            nib.save(template_nifti(arr, tpl), os.path.join(d, name))
            assert same_geometry(nib.load(os.path.join(d, name)), tpl), "%s: %s geometry" % (sess, name)
        brain = np.asarray(tpl.dataobj) > 0
        qc.update({"rigid_metric": metric, "rigid_ncc": ncc, "rigid_start": how,
                   "rigid_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))),
                   "rigid_translation_mm": float(np.linalg.norm(M[:3, 3])),
                   "fixed_to_moving_ras": M.tolist(), "geometry_equals_template": True,
                   "brainmask_dice_mni_brain": float(2 * (mask & brain).sum() / (mask.sum() + brain.sum()))})
        log.append("rigid (NCC %.5f, %.1f deg, %.1f mm%s)" % (
            ncc, qc["rigid_rotation_deg"], qc["rigid_translation_mm"],
            "" if how == "centre of mass" else "; start: " + how))
    else:
        nd, md, aff, info = load_native(t1_path, mask_path, cfg)
        fixed_mask = os.path.join(cache, "elastix", "mni_brain_mask.nii.gz")
        M, metric = register_to_mni(nd, md, aff, os.path.join(cache, "elastix", sess), fixed_mask, cfg)
        grid_mm = apply_affine(tpl.affine, np.argwhere(np.ones(tpl.shape, bool)).astype(np.float64))
        vox = apply_affine(np.linalg.inv(aff) @ M, grid_mm).T
        t1 = map_coordinates(nd, vox, order=1, mode="constant", cval=0.0).reshape(tpl.shape).astype(np.float32)
        mask = map_coordinates(md.astype(np.float32), vox, order=1, mode="constant",
                               cval=0.0).reshape(tpl.shape) >= 0.5
        out = {"T1_N4_mni.nii.gz": t1, "Brainmask_mni.nii.gz": mask.astype(np.uint8),
               "T1_input_mni.nii.gz": normalize_t1(t1, mask, cfg["z_outside"])}
        for name, arr in out.items():
            nib.save(template_nifti(arr, tpl), os.path.join(d, name))
        geometry_ok = all(same_geometry(nib.load(os.path.join(d, n)), tpl) for n in out)
        assert geometry_ok, "%s: output geometry differs from %s" % (sess, cfg["mni_template"])
        brain = np.asarray(tpl.dataobj) > 0
        qc.update({"elastix_mi": metric, "bias_min": info["bias_min"], "bias_max": info["bias_max"],
                   "mask_ml": info["mask_ml"], "geometry_equals_template": geometry_ok,
                   "subject_to_mni_scales": (1.0 / np.linalg.svd(M[:3, :3], compute_uv=False)).tolist(),
                   "fixed_to_moving_ras": M.tolist(),
                   "brainmask_dice_mni_brain": float(2 * (mask & brain).sum() / (mask.sum() + brain.sum()))})
        log.append("registered (MI %.4f, Dice %.3f)" % (metric, qc["brainmask_dice_mni_brain"]))

    mask = np.asarray(nib.load(os.path.join(d, "Brainmask_mni.nii.gz")).dataobj) > 0
    mni2vox = np.linalg.inv(tpl.affine)
    sqc = qc.setdefault("surfaces", {})
    tmp = None
    if rigid and any(not os.path.isfile(subject_surface_path(cache, sess, h, s)) or
                     s not in sqc.get(h, {}) for h in cfg["hemis"] for s in cfg["surfaces"]):
        tmp = tempfile.mkdtemp(prefix="surf_", dir=os.path.join(cache, "elastix"))
        surf_dir = extract_surfaces(surf_dir, cfg["hemis"], cfg["surfaces"], tmp)
    for hemi in cfg["hemis"]:
        todo = [s for s in cfg["surfaces"]
                if not os.path.isfile(subject_surface_path(cache, sess, hemi, s))]
        check = [s for s in cfg["surfaces"] if s not in todo]
        if not todo and all(s in sqc.get(hemi, {}) for s in cfg["surfaces"]):
            continue
        sph_t, _ = read_gifti(fsaverage_path(cfg["fsaverage_dir"], hemi, "sphere"))
        faces = np.load(os.path.join(cache, "canonical_%s.npz" % hemi))["faces"].astype(np.int64)
        dirs = sph_t / np.linalg.norm(sph_t, axis=1, keepdims=True)
        sph, sph_f = read_geometry(os.path.join(surf_dir, hemi + ".sphere.reg"))
        bidx, w, inside = sphere_barycentric(dirs, sph, sph_f)
        cras = None
        hq = sqc.setdefault(hemi, {})
        for s in todo + check:
            path = os.path.join(surf_dir, hemi + "." + s)
            cor, f = read_geometry(path)
            if len(cor) != len(sph) or not np.array_equal(f, sph_f):
                raise ValueError("%s: not on %s.sphere.reg's vertex set (%d vs %d vertices) -- "
                                 "cannot be put into correspondence" % (path, hemi, len(cor), len(sph)))
            c = surface_cras(path)
            if cras is None:
                cras = c
            assert np.allclose(c, cras, atol=1e-4), "%s: cras differs between surfaces" % path
            T = np.linalg.inv(M)
            T[:3, 3] += T[:3, :3] @ c                     # surface RAS -> scanner (+cras) -> MNI
            x = (apply_affine(T, cor)[bidx] * w[..., None]).sum(1)
            assert np.isfinite(x).all(), "%s: non-finite vertices" % path
            vi = np.clip(np.round(apply_affine(mni2vox, x)).astype(int), 0, np.array(tpl.shape) - 1)
            idx = apply_affine(mni2vox, x)
            entry = {"in_brainmask_frac": float(mask[tuple(vi.T)].mean()),
                     "outside_grid_frac": float(((idx < 0) | (idx > np.array(tpl.shape) - 1)).any(1).mean()),
                     "extent_mm": np.round(x.max(0) - x.min(0), 2).tolist(),
                     "sphere_barycentric_inside": inside}
            p_out = subject_surface_path(cache, sess, hemi, s)
            if s in check:
                diff = float(np.abs(read_gifti(p_out)[0] - x).max())
                assert diff < 1e-3, "%s: re-derived %s.%s differs from the cache by %.3g mm" % (sess, hemi, s, diff)
                entry["rederived_max_diff_mm"] = diff
            else:
                write_gifti(p_out, x, faces)
            hq[s] = entry
            if s == "white" and "white_boundary" not in qc:
                qc["white_boundary"] = white_boundary_check(
                    os.path.join(d, images[0]), x, mni2vox)
                assert qc["white_boundary"]["centre_over_best"] >= 0.95, \
                    "%s: white surface is not on the T1 gradient maximum (%s)" % (sess, qc["white_boundary"])
        log.append("%s: +%s%s" % (hemi, ",".join(todo) or "-",
                                  (" (re-derived %s)" % ",".join(check)) if check else ""))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    with open(qc_path + ".tmp", "w") as fh:
        json.dump(qc, fh, indent=1)
    os.replace(qc_path + ".tmp", qc_path)
    return sess, "; ".join(log) or "up to date", time.time() - t0


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    d = PreprocessConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=None, help="Load all settings from a saved preprocess.json")
    p.add_argument("--layout", default=d.layout, choices=["neuropreprocess", "rigid_dataset"])
    p.add_argument("--raw_root", default=d.raw_root)
    p.add_argument("--t1_pattern", default=d.t1_pattern)
    p.add_argument("--mask_pattern", default=d.mask_pattern)
    p.add_argument("--native_t1_pattern", default=d.native_t1_pattern)
    p.add_argument("--surf_archive_pattern", default=d.surf_archive_pattern)
    p.add_argument("--rigid_sampling", type=float, default=d.rigid_sampling)
    p.add_argument("--rigid_ncc_min", type=float, default=d.rigid_ncc_min)
    p.add_argument("--threads", type=int, default=d.threads)
    p.add_argument("--cache_dir", default=d.cache_dir)
    p.add_argument("--mni_template", default=d.mni_template)
    p.add_argument("--fsaverage_dir", default=d.fsaverage_dir)
    p.add_argument("--elastix", default=d.elastix)
    p.add_argument("--elastix_params", default=d.elastix_params)
    p.add_argument("--hemis", nargs="+", default=list(d.hemis), choices=["lh", "rh"])
    p.add_argument("--surfaces", nargs="+", default=list(d.surfaces))
    p.add_argument("--n4_shrink", type=int, default=d.n4_shrink)
    p.add_argument("--n4_iterations", type=int, nargs="+", default=list(d.n4_iterations))
    p.add_argument("--z_outside", type=float, default=d.z_outside)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--limit", type=int, default=d.limit)
    return p.parse_args(argv)


FIELDS = ("layout", "raw_root", "t1_pattern", "mask_pattern", "native_t1_pattern", "surf_archive_pattern",
          "cache_dir", "mni_template", "fsaverage_dir", "elastix", "elastix_params",
          "hemis", "surfaces", "n4_shrink", "n4_iterations", "z_outside", "workers", "limit",
          "rigid_sampling", "rigid_ncc_min", "threads")


def build_config(args) -> PreprocessConfig:
    """Fields set on a config with no file attached (no autosave); saved by run()."""
    cfg = PreprocessConfig()
    if args.config:
        if not cfg.load(args.config):
            sys.exit("cannot read %s" % args.config)
        return cfg
    for k in FIELDS:
        v = getattr(args, k)
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def run(cfg: PreprocessConfig) -> None:
    c = {k: getattr(cfg, k) for k in FIELDS}
    cache = c["cache_dir"]
    os.makedirs(os.path.join(cache, "elastix"), exist_ok=True)

    # the cache's own record, merged with what an earlier run already put there, so
    # a surface / hemisphere added later does not drop the earlier ones
    rec_path = os.path.join(cache, "preprocess.json")
    if os.path.isfile(rec_path):
        old = PreprocessConfig()
        old.load(rec_path)
        c["hemis"] = tuple(dict.fromkeys(list(old.hemis) + list(c["hemis"])))
        c["surfaces"] = tuple(dict.fromkeys(list(old.surfaces) + list(c["surfaces"])))
    rec = PreprocessConfig()
    for k in FIELDS:
        setattr(rec, k, c[k])
    rec.save(rec_path)
    print(rec.summary())

    tpl = nib.load(c["mni_template"])
    if c["layout"] != "rigid_dataset":                 # the elastix fixed mask is only used there
        fixed_mask = os.path.join(cache, "elastix", "mni_brain_mask.nii.gz")
        if not os.path.isfile(fixed_mask):
            nib.save(template_nifti((np.asarray(tpl.dataobj) > 0).astype(np.uint8), tpl), fixed_mask)
    for hemi in c["hemis"]:
        write_canonical(c, hemi)

    if c["layout"] == "rigid_dataset":
        subjects, skipped = find_subjects_rigid(c)
    else:
        subjects, skipped = find_subjects(c["raw_root"], c["hemis"])
    print("%d subjects with all required files (layout %s), %d skipped"
          % (len(subjects), c["layout"], len(skipped)))
    if c["limit"]:
        subjects = subjects[:c["limit"]]
    jobs = [(s, t1, m, sd, c) for s, t1, m, sd in subjects]
    t0, done, failed = time.time(), 0, []
    pool = Pool(c["workers"]) if c["workers"] > 1 else None
    it = pool.imap_unordered(_safe, jobs) if pool else map(_safe, jobs)
    for sess, msg, dt in it:
        done += 1
        if msg.startswith("FAILED"):
            failed.append(sess)
        print("[%3d/%3d] %s: %s  (%.0f s)" % (done, len(jobs), sess, msg, dt), flush=True)
    if pool:
        pool.close()
        pool.join()
    summary = {"subjects": len(jobs), "failed": failed, "skipped_raw": skipped,
               "hemis": list(c["hemis"]), "surfaces": list(c["surfaces"]),
               "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(cache, "preprocess_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done: %d subjects, %d failed%s, %.1f min"
          % (len(jobs), len(failed), (" " + str(failed)) if failed else "", summary["minutes"]))


def _safe(job):
    try:
        return process_subject(job)
    except Exception as e:                                   # one bad subject must not stop the rest
        return job[0], "FAILED %s: %s" % (type(e).__name__, e), 0.0


def main(argv=None) -> None:
    utf8_stdout()
    run(build_config(parse_args(argv)))


if __name__ == "__main__":
    main()
