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
    shapes:     Dict[str, tuple] = field(default_factory=dict)   # channel -> shape
    spacings:   Dict[str, tuple] = field(default_factory=dict)   # channel -> spacing


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
    shape_counts:  Dict[str, Counter]        = field(default_factory=dict)   # modality -> Counter
    spacing_data:  Dict[str, List[tuple]]    = field(default_factory=dict)   # modality -> list

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
        (much faster for large datasets -- structural checks only).
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
                print(f'[inspect] Split directory not found: {split_dir} -- skipped')
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


# ── Config generation + loader test ───────────────────────────────────────────

def generate_configs(inspector: DatasetInspector, reports: Dict[str, SplitReport]):
    """
    Build ``DataConfig`` and ``AugmentConfig`` from the inspection results.

    Returns ``(DataConfig, AugmentConfig)``.

    ``target_shape`` and ``target_spacing`` are left ``None`` -- the function
    prints a suggestion based on the training-split shape distribution, but
    only the user can decide whether to crop/pad or resample.
    """
    from apps.labelmap_segmentation.segmentation_config import DataConfig, AugmentConfig

    dc = DataConfig()
    dc.data_root   = str(inspector.root)
    dc.modalities  = inspector.modalities
    dc.label_name  = inspector.label_name
    dc.num_classes = inspector.num_classes

    ac = AugmentConfig()   # all defaults

    W    = 68
    thin = '-' * W
    print(f'\n+{"=" * W}+')
    print(f'|{"  GENERATED CONFIGS":^{W}}|')
    print(f'+{"=" * W}+')
    print(f'\n  DataConfig')
    print(f'  {thin}')
    print(f'    data_root   = {dc.data_root!r}')
    print(f'    modalities  = {dc.modalities}')
    print(f'    label_name  = {dc.label_name!r}')
    print(f'    num_classes = {dc.num_classes}')

    # Suggest target_shape / target_spacing from training set
    train = reports.get('train')
    if train:
        for mod in (inspector.modalities or []):
            counter = train.shape_counts.get(mod, {})
            if counter:
                most_common_shape, cnt = counter.most_common(1)[0]
                pct = cnt / train.n_valid * 100 if train.n_valid else 0
                print(f'\n  Shape suggestion  ({mod})')
                print(f'  {thin}')
                print(f'    Most common : {most_common_shape}  ({pct:.0f}% of train subjects)')
                if len(counter) > 1:
                    print(f'    Other shapes: {list(counter.keys())[1:]}')
                    print(f'    -> Consider setting target_shape={most_common_shape}')
                else:
                    print(f'    All subjects share the same shape; no CropOrPad needed.')

            spacings = train.spacing_data.get(mod, [])
            if spacings:
                arr = np.array(spacings)
                med = tuple(float(np.median(arr[:, i])) for i in range(arr.shape[1]))
                std = arr.std(axis=0)
                if std.max() > 1e-3:
                    print(f'    Spacing varies (std {std}) -- consider resampling '
                          f'to target_spacing={med}')
                else:
                    print(f'    Spacing uniform: {med}; no Resample needed.')
                break  # show for first modality only

    print(f'\n  AugmentConfig  (all defaults; edit via JSON or CLI flags)')
    print(f'  {thin}')
    print(f'    flip={ac.flip}  affine={ac.affine}  noise={ac.noise}  '
          f'blur={ac.blur}  gamma={ac.gamma}  elastic={ac.elastic}')
    print()

    return dc, ac


def run_loader_test(
    inspector: DatasetInspector,
    data_cfg=None,
    augment_cfg=None,
    num_batches: int = 2,
) -> None:
    """
    Populate ConfigManager, create datasets and DataLoaders,
    then load *num_batches* training batches to verify the pipeline end-to-end.

    Parameters
    ----------
    inspector    : DatasetInspector
        The inspector that was already run (provides modalities / label_name).
    data_cfg     : DataConfig, optional
        Pre-built config; created from *inspector* when ``None``.
    augment_cfg  : AugmentConfig, optional
        Pre-built config; defaults when ``None``.
    num_batches  : int
        How many training batches to load.
    """
    try:
        from configuration.manager import ConfigManager
        from apps.labelmap_segmentation.segmentation_config import (
            DataConfig, AugmentConfig, PatchConfig, TrainingConfig, InfraConfig,
        )
        from apps.labelmap_segmentation.dataset import (
            create_labelmap_segmentation_datasets,
            create_data_loaders,
        )
        import torchio as tio
    except ImportError as exc:
        print(f'\n[loader test] Import error -- {exc}')
        return

    W    = 68
    thin = '-' * W
    print(f'\n+{"=" * W}+')
    print(f'|{"  LOADER TEST":^{W}}|')
    print(f'+{"=" * W}+')

    # Build configs if not provided
    if data_cfg is None:
        dc = DataConfig()
        dc.data_root   = str(inspector.root)
        dc.modalities  = inspector.modalities
        dc.label_name  = inspector.label_name
        dc.num_classes = inspector.num_classes
    else:
        dc = data_cfg

    ac = augment_cfg if augment_cfg is not None else AugmentConfig()
    ac.enabled = False   # disable augmentation for the loader test

    pc = PatchConfig()
    pc.enabled = False

    tc = TrainingConfig()
    tc.batch_size = 1

    ic = InfraConfig()
    ic.num_workers = 0   # safe for testing on any platform

    ConfigManager.reset()
    m = ConfigManager.get()
    m.register(ConfigManager.DATA,     dc)
    m.register(ConfigManager.PATCH,    pc)
    m.register(ConfigManager.AUGMENT,  ac)
    m.register(ConfigManager.TRAINING, tc)
    m.register(ConfigManager.INFRA,    ic)

    # ── Dataset creation ─────────────────────────────────────────────────────
    print('\n  Creating datasets ...')
    try:
        train_ds, val_ds, test_ds = create_labelmap_segmentation_datasets()
    except Exception as exc:
        print(f'  [FAIL] Dataset creation: {exc}')
        return

    print(f'  {thin}')
    print(f'  {"split":<14}  {"subjects":>10}')
    print(f'  {thin}')
    for name, ds in [('train', train_ds), ('validation', val_ds), ('test', test_ds)]:
        print(f'  {name:<14}  {len(ds):>10}')

    # ── DataLoader creation ───────────────────────────────────────────────────
    print('\n  Creating DataLoaders ...')
    try:
        train_loader, val_loader = create_data_loaders(train_ds, val_ds)
    except Exception as exc:
        print(f'  [FAIL] DataLoader creation: {exc}')
        return
    print(f'  train batch_size={train_loader.batch_size}  '
          f'val batch_size={val_loader.batch_size}')

    # ── Batch loading ─────────────────────────────────────────────────────────
    print(f'\n  Loading {num_batches} train batch(es) ...')
    print(f'  {thin}')
    errors = []
    try:
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            print(f'\n  batch {i}')
            for mod in inspector.modalities or []:
                if mod not in batch:
                    continue
                t = batch[mod][tio.DATA]                  # [B, 1, D, H, W]
                print(f'    {mod:<14}  shape={tuple(t.shape)}'
                      f'  dtype={t.dtype}'
                      f'  min={t.min().item():.3f}'
                      f'  max={t.max().item():.3f}'
                      f'  mean={t.mean().item():.3f}')
            lbl_key = inspector.label_name
            if lbl_key in batch:
                lbl = batch[lbl_key][tio.DATA]            # [B, 1, D, H, W]
                unique = sorted(lbl.unique().long().tolist())
                print(f'    {lbl_key:<14}  shape={tuple(lbl.shape)}'
                      f'  dtype={lbl.dtype}'
                      f'  classes present={unique}')
    except Exception as exc:
        errors.append(str(exc))
        print(f'  [FAIL] Batch loading: {exc}')

    print(f'\n  {thin}')
    if errors:
        print('  Result: FAIL')
    else:
        print(f'  Result: OK  ({num_batches} batch(es) loaded successfully)')
    print(f'+{"=" * W}+\n')


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='LabelMap Segmentation -- dataset inspection',
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
                   help='Skip voxel-level statistics (structural check only, fast)')
    p.add_argument('--test_loader',  action='store_true',
                   help='After inspection: generate configs and test the DataLoader pipeline')
    p.add_argument('--test_batches', type=int, default=2,
                   help='Number of training batches to load during --test_loader')
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

    if args.test_loader:
        data_cfg, augment_cfg = generate_configs(inspector, reports)
        run_loader_test(
            inspector,
            data_cfg=data_cfg,
            augment_cfg=augment_cfg,
            num_batches=args.test_batches,
        )


class _Tee:
    """Write to two streams simultaneously."""
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()


if __name__ == '__main__':
    main()
