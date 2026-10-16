"""
Build the IXI hemisphere (left/right) segmentation dataset under ``input/``.

Source (one folder per subject)::

    <src>/<subject>/T1_Rigid/NIFTI/*.nii.gz
    <src>/<subject>/FreeSurfer_Brainmask_Rigid/NIFTI/*.nii.gz
    <src>/<subject>/FreeSurfer_Brainmask_LR_Rigid/NIFTI/*.nii.gz

Output (layout expected by ``dataset.py``; files are hard-linked, no copy)::

    <dst>/{train,validation,test}/<subject>/{T1,freesurfer_brainmask,LR}/*.nii.gz

The split is stratified by acquisition site (the ``Guys`` / ``HH`` / ``IOP`` token in
the subject folder name) so every split sees all scanners, and is deterministic for a
given ``--seed``. The split is also written to ``<dst>/split.json``.

Usage:
    python tools/prepare_ixi_hemisphere.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# source folder -> destination folder name
FOLDER_MAP = {
    "T1_Rigid": "T1",
    "FreeSurfer_Brainmask_Rigid": "freesurfer_brainmask",
    "FreeSurfer_Brainmask_LR_Rigid": "LR",
}


def _nifti(subject_dir: Path, src_folder: str) -> Path:
    hits = sorted((subject_dir / src_folder / "NIFTI").glob("*.nii.gz"))
    if not hits:
        raise FileNotFoundError(f"{subject_dir.name}: no NIfTI in {src_folder}/NIFTI")
    return hits[0]


def split_subjects(
    subjects: List[str], val_frac: float, test_frac: float, seed: int
) -> Dict[str, List[str]]:
    by_site: Dict[str, List[str]] = defaultdict(list)
    for s in subjects:
        by_site[s.split("-")[1]].append(s)

    rng = random.Random(seed)
    out: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    for site in sorted(by_site):
        group = sorted(by_site[site])
        rng.shuffle(group)
        n_test = round(len(group) * test_frac)
        n_val = round(len(group) * val_frac)
        out["test"] += group[:n_test]
        out["validation"] += group[n_test : n_test + n_val]
        out["train"] += group[n_test + n_val :]
    return {k: sorted(v) for k, v in out.items()}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:  # different volume / filesystem without hard links
        shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", default="D:/DATASET-IXI")
    p.add_argument("--dst", default="./input/hemisphere_IXI")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    subjects = sorted(d.name for d in src.iterdir() if d.is_dir())
    # fail early if any subject is incomplete
    for s in subjects:
        for folder in FOLDER_MAP:
            _nifti(src / s, folder)

    split = split_subjects(subjects, args.val_frac, args.test_frac, args.seed)
    for name, ids in split.items():
        for s in ids:
            for folder, new_name in FOLDER_MAP.items():
                f = _nifti(src / s, folder)
                _link(f, dst / name / s / new_name / f.name)
        sites = defaultdict(int)
        for s in ids:
            sites[s.split("-")[1]] += 1
        print(f"{name:<10} {len(ids):>4} subjects  {dict(sorted(sites.items()))}")

    (dst / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Dataset written to {dst.resolve()}")


if __name__ == "__main__":
    main()
