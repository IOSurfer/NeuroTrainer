"""
Dataset inspection and verification tool for LabelMap Segmentation.

Run as a script (no training setup required):
    python -m apps.labelmap_segmentation.inspect_dataset \\
        --data_root /data/brain_mri \\
        --num_classes 3

Use programmatically:
    from apps.labelmap_segmentation.inspect_dataset import DatasetInspector
    inspector = DatasetInspector('/data/brain_mri', num_classes=3)
    reports   = inspector.run()
    inspector.print_report(reports)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SubjectVerification:
    """Structural and load check for a single subject."""
    subject_id: str
    valid:      bool
    issues:     List[str]        = field(default_factory=list)
    shapes:     Dict[str, tuple] = field(default_factory=dict)   # channel → shape
    spacings:   Dict[str, tuple] = field(default_factory=dict)   # channel → spacing


@dataclass
class IntensityStats:
    """Per-modality intensity statistics aggregated across subjects."""
    modality:    str
    n_subjects:  int   = 0
    mean_of_means: float = 0.0
    std_of_means:  float = 0.0
    mean_of_stds:  float = 0.0
    global_min:    float = 0.0
    global_max:    float = 0.0
    mean_p1:       float = 0.0   # mean 1st-percentile across subjects
    mean_p99:      float = 0.0   # mean 99th-percentile across subjects


@dataclass
class LabelClassStats:
    """Aggregate label statistics for one class across subjects."""
    class_id:          int
    n_subjects_present: int   = 0   # subjects where this class appears
    mean_vox_fraction:  float = 0.0  # mean voxel fraction
    std_vox_fraction:   float = 0.0


@dataclass
class SplitReport:
    split:         str
    verifications: List[SubjectVerification] = field(default_factory=list)
    intensity:     List[IntensityStats]      = field(default_factory=list)
    label_classes: List[LabelClassStats]     = field(default_factory=list)
    shape_counts:  Dict[str, Counter]        = field(default_factory=dict)   # modality → Counter
    spacing_data:  Dict[str, List[tuple]]    = field(default_factory=dict)   # modality → list

    @property
    def n_total(self) -> int:
        return len(self.verifications)

    @property
    def n_valid(self) -> int:
        return sum(1 for v in self.verifications if v.valid)

    @property
    def invalid_subjects(self) -> List[SubjectVerification]:
        return [v for v in self.verifications if not v.valid]


# ── Core inspector ─────────────────────────────────────────────────────────────

class DatasetInspector:
    """
    Inspect and validate a LabelMap Segmentation dataset.

    Parameters
    ----------
    data_root : str
        Root directory containing ``train/``, ``validation/``, ``test/`` splits.
    label_name : str
        Name of the label sub-folder inside each subject directory.
    num_classes : int
        Total number of segmentation classes (background included).
    modalities : list of str, optional
        Modality folder names.  Auto-detected from the first valid subject
        in the ``train`` split when ``None``.
    splits : list of str, optional
        Which splits to inspect.  Defaults to all three.
    skip_intensity : bool
        When ``True``, skip voxel-level intensity and label statistics
        (much faster for large datasets — structural checks only).
    """

    def __init__(
        self,
        data_root:       str,
        label_name:      str = 'label',
        num_classes:     int = 2,
        modalities:      Optional[List[str]] = None,
        splits:          Optional[List[str]] = None,
        skip_intensity:  bool = False,
    ) -> None:
        self.root         = Path(data_root)
        self.label_name   = label_name
        self.num_classes  = num_classes
        self.modalities   = modalities
        self.splits       = splits or ['train', 'validation', 'test']
        self.skip_intensity = skip_intensity

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, SplitReport]:
        """
        Run the full inspection and return a per-split report dictionary.
        """
        if self.modalities is None:
            self.modalities = self._auto_discover_modalities()
            if not self.modalities:
                raise RuntimeError(
                    f'Cannot detect modalities under {self.root / "train"}. '
                    'Provide --modalities explicitly.'
                )
            print(f'[inspect] Auto-detected modalities: {self.modalities}')

        reports: Dict[str, SplitReport] = {}
        for split in self.splits:
            split_dir = self.root / split
            if not split_dir.exists():
                print(f'[inspect] Split directory not found: {split_dir} — skipped')
                continue
            print(f'[inspect] Inspecting split: {split} …', end='  ', flush=True)
            reports[split] = self._inspect_split(split_dir, split)
            print(f'done  ({reports[split].n_valid}/{reports[split].n_total} valid)')

        return reports

    def print_report(self, reports: Dict[str, SplitReport]) -> None:
        """Print a formatted inspection report to stdout."""
        W = 68
        thick = '=' * W
        thin  = '-' * W

        def _header(title: str) -> None:
            print(f'\n+{thick}+')
            print(f'|{title:^{W}}|')
            print(f'+{thick}+')

        def _section(title: str) -> None:
            print(f'\n  {title}')
            print(f'  {thin}')

        _header('  LabelMap Segmentation -- Dataset Inspection  ')
        print(f'\n  Root       : {self.root}')
        print(f'  Label      : {self.label_name}   Classes: {self.num_classes}')
        print(f'  Modalities : {", ".join(self.modalities or [])}')

        # ── Split summary ────────────────────────────────────────────────────
        _section('SPLIT SUMMARY')
        print(f'  {"split":<14} {"subjects":>10} {"valid":>8} {"invalid":>9}')
        print(f'  {thin}')
        for name, rpt in reports.items():
            flag = '  [OK]' if rpt.n_valid == rpt.n_total else '  [!!]'
            print(f'  {name:<14} {rpt.n_total:>10} {rpt.n_valid:>8} '
                  f'{rpt.n_total - rpt.n_valid:>9}{flag}')

        for name, rpt in reports.items():
            if rpt.invalid_subjects:
                print(f'\n  Invalid subjects in [{name}]:')
                for v in rpt.invalid_subjects:
                    for issue in v.issues:
                        print(f'    * {v.subject_id}: {issue}')

        # ── Shape distribution ───────────────────────────────────────────────
        for name, rpt in reports.items():
            for mod, counter in rpt.shape_counts.items():
                _section(f'[{name.upper()}]  SHAPE DISTRIBUTION - {mod}')
                for shape, count in counter.most_common():
                    bar = '#' * min(count, 40)
                    print(f'    {str(shape):<28}  {count:>5} subjects  {bar}')

        # ── Spacing distribution ─────────────────────────────────────────────
        for name, rpt in reports.items():
            for mod, spacings in rpt.spacing_data.items():
                if not spacings:
                    continue
                arr = np.array(spacings)  # [N, 3]
                _section(f'[{name.upper()}]  VOXEL SPACING - {mod}  (mm)')
                for ax, axis_name in zip(range(arr.shape[1]), ('x', 'y', 'z')):
                    col = arr[:, ax]
                    print(f'    {axis_name}  mean {col.mean():.3f} +/- {col.std():.3f}'
                          f'   range [{col.min():.3f}, {col.max():.3f}]')

        # ── Intensity statistics ─────────────────────────────────────────────
        for name, rpt in reports.items():
            if not rpt.intensity:
                continue
            _section(f'[{name.upper()}]  INTENSITY STATISTICS')
            hdr = f'  {"modality":<10}{"mean +/- std":>20}{"min":>10}{"max":>10}{"p1":>9}{"p99":>9}'
            print(hdr)
            print(f'  {thin}')
            for st in rpt.intensity:
                print(
                    f'  {st.modality:<10}'
                    f'  {st.mean_of_means:>7.3f} +/- {st.mean_of_stds:<7.3f}'
                    f'  {st.global_min:>8.3f}'
                    f'  {st.global_max:>8.3f}'
                    f'  {st.mean_p1:>7.3f}'
                    f'  {st.mean_p99:>7.3f}'
                )

        # ── Label class distribution ─────────────────────────────────────────
        for name, rpt in reports.items():
            if not rpt.label_classes:
                continue
            n = rpt.n_valid
            _section(f'[{name.upper()}]  LABEL CLASS DISTRIBUTION  ({n} subjects)')
            print(f'  {"class":>6}  {"present":>8}  {"vox frac (mean+/-std)":>24}')
            print(f'  {thin}')
            for lc in rpt.label_classes:
                pct = f'{lc.mean_vox_fraction*100:.2f}% +/- {lc.std_vox_fraction*100:.2f}%'
                pres = f'{lc.n_subjects_present}/{n}'
                print(f'  {lc.class_id:>6}  {pres:>8}  {pct:>24}')

        print(f'\n  {thick}\n')

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _auto_discover_modalities(self) -> List[str]:
        train_dir = self.root / 'train'
        if not train_dir.exists():
            return []
        for subj_dir in sorted(train_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            mods = [
                d.name for d in sorted(subj_dir.iterdir())
                if d.is_dir()
                and d.name != self.label_name
                and _first_nifti(d) is not None
            ]
            if mods:
                return mods
        return []

    def _inspect_split(self, split_dir: Path, split_name: str) -> SplitReport:
        rpt = SplitReport(split=split_name)
        rpt.shape_counts = {m: Counter() for m in self.modalities}  # type: ignore[arg-type]
        rpt.spacing_data = {m: [] for m in self.modalities}          # type: ignore[arg-type]

        # Per-modality accumulators for intensity stats
        int_acc: Dict[str, _IntAcc] = {m: _IntAcc(m) for m in self.modalities}  # type: ignore[arg-type]
        # Per-class accumulators for label stats
        lbl_acc: Dict[int, _LblAcc] = {c: _LblAcc(c) for c in range(self.num_classes)}

        for subj_dir in sorted(split_dir.iterdir()):
            if not subj_dir.is_dir():
                continue

            # Structural + load verification
            ver = self._verify_subject(subj_dir)
            rpt.verifications.append(ver)
            if not ver.valid:
                continue

            # Accumulate shape / spacing
            for mod in self.modalities:  # type: ignore[union-attr]
                shape   = ver.shapes.get(mod)
                spacing = ver.spacings.get(mod)
                if shape:
                    rpt.shape_counts[mod][shape] += 1
                if spacing:
                    rpt.spacing_data[mod].append(spacing)

            if self.skip_intensity:
                continue

            # Intensity stats
            for mod in self.modalities:  # type: ignore[union-attr]
                nii_path = _first_nifti(subj_dir / mod)
                if nii_path:
                    data = nib.load(str(nii_path)).get_fdata(dtype=np.float32).ravel()
                    int_acc[mod].add(data)

            # Label stats
            label_path = _first_nifti(subj_dir / self.label_name)
            if label_path:
                lbl_data = np.asarray(
                    nib.load(str(label_path)).get_fdata(), dtype=np.int32
                ).ravel()
                n_vox = lbl_data.size
                for c in range(self.num_classes):
                    count = int((lbl_data == c).sum())
                    lbl_acc[c].add(count, n_vox)

        # Finalise intensity stats
        if not self.skip_intensity:
            rpt.intensity = [acc.finalise() for acc in int_acc.values()]
            rpt.label_classes = [acc.finalise() for acc in lbl_acc.values()]

        return rpt

    def _verify_subject(self, subj_dir: Path) -> SubjectVerification:
        ver = SubjectVerification(subject_id=subj_dir.name, valid=True)

        for mod in self.modalities:  # type: ignore[union-attr]
            mod_dir = subj_dir / mod
            if not mod_dir.exists():
                ver.valid = False
                ver.issues.append(f"missing modality folder '{mod}'")
                continue
            nii = _first_nifti(mod_dir)
            if nii is None:
                ver.valid = False
                ver.issues.append(f"no NIfTI file in '{mod}'")
                continue
            try:
                img = nib.load(str(nii))
                ver.shapes[mod]   = tuple(img.shape[:3])
                ver.spacings[mod] = tuple(float(z) for z in img.header.get_zooms()[:3])
            except Exception as exc:
                ver.valid = False
                ver.issues.append(f"cannot load '{mod}': {exc}")

        label_dir = subj_dir / self.label_name
        if not label_dir.exists():
            ver.valid = False
            ver.issues.append(f"missing label folder '{self.label_name}'")
        else:
            nii = _first_nifti(label_dir)
            if nii is None:
                ver.valid = False
                ver.issues.append(f"no NIfTI file in '{self.label_name}'")
            else:
                try:
                    nib.load(str(nii))
                except Exception as exc:
                    ver.valid = False
                    ver.issues.append(f"cannot load label: {exc}")

        return ver


# ── Private accumulators ───────────────────────────────────────────────────────

class _IntAcc:
    """Online accumulator for intensity statistics."""

    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.means: List[float] = []
        self.stds:  List[float] = []
        self.p1s:   List[float] = []
        self.p99s:  List[float] = []
        self.gmin   = float('inf')
        self.gmax   = float('-inf')

    def add(self, data: np.ndarray) -> None:
        self.means.append(float(data.mean()))
        self.stds.append(float(data.std()))
        p1, p99 = float(np.percentile(data, 1)), float(np.percentile(data, 99))
        self.p1s.append(p1)
        self.p99s.append(p99)
        self.gmin = min(self.gmin, float(data.min()))
        self.gmax = max(self.gmax, float(data.max()))

    def finalise(self) -> IntensityStats:
        if not self.means:
            return IntensityStats(modality=self.modality)
        return IntensityStats(
            modality      = self.modality,
            n_subjects    = len(self.means),
            mean_of_means = float(np.mean(self.means)),
            std_of_means  = float(np.std(self.means)),
            mean_of_stds  = float(np.mean(self.stds)),
            global_min    = self.gmin,
            global_max    = self.gmax,
            mean_p1       = float(np.mean(self.p1s)),
            mean_p99      = float(np.mean(self.p99s)),
        )


class _LblAcc:
    """Online accumulator for per-class label statistics."""

    def __init__(self, class_id: int) -> None:
        self.class_id   = class_id
        self.fractions: List[float] = []
        self.n_present  = 0

    def add(self, vox_count: int, total_vox: int) -> None:
        frac = vox_count / total_vox if total_vox > 0 else 0.0
        self.fractions.append(frac)
        if vox_count > 0:
            self.n_present += 1

    def finalise(self) -> LabelClassStats:
        fracs = np.array(self.fractions) if self.fractions else np.array([0.0])
        return LabelClassStats(
            class_id           = self.class_id,
            n_subjects_present = self.n_present,
            mean_vox_fraction  = float(fracs.mean()),
            std_vox_fraction   = float(fracs.std()),
        )


# ── Utility ────────────────────────────────────────────────────────────────────

def _first_nifti(folder: Path) -> Optional[Path]:
    for pattern in ('*.nii.gz', '*.nii'):
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='LabelMap Segmentation — dataset inspection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data_root',    required=True,
                   help='Root directory with train/validation/test splits')
    p.add_argument('--label_name',   default='label',
                   help='Segmentation mask sub-folder name')
    p.add_argument('--num_classes',  type=int, required=True,
                   help='Number of label classes (background included)')
    p.add_argument('--modalities',   nargs='+', default=None,
                   help='Modality folder names (auto-detected when omitted)')
    p.add_argument('--splits',       nargs='+',
                   default=['train', 'validation', 'test'],
                   help='Splits to inspect')
    p.add_argument('--skip_intensity', action='store_true',
                   help='Skip voxel-level statistics (structural check only — fast)')
    p.add_argument('--output',       default=None,
                   help='Write the text report to this file in addition to stdout')
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    inspector = DatasetInspector(
        data_root      = args.data_root,
        label_name     = args.label_name,
        num_classes    = args.num_classes,
        modalities     = args.modalities,
        splits         = args.splits,
        skip_intensity = args.skip_intensity,
    )

    reports = inspector.run()

    if args.output:
        # Redirect stdout to both the file and the terminal
        import io
        buf = io.StringIO()
        _orig = sys.stdout
        sys.stdout = _Tee(_orig, buf)

    inspector.print_report(reports)

    if args.output:
        sys.stdout = _orig  # type: ignore[assignment]
        Path(args.output).write_text(buf.getvalue(), encoding='utf-8')  # type: ignore[possibly-undefined]
        print(f'[inspect] Report written to {args.output}')


class _Tee:
    """Write to two streams simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()


if __name__ == '__main__':
    main()
