"""
Dataset utilities for 3D medical image segmentation.

Expected directory layout:
    <data_root>/
    ├── train/
    │   ├── subject_001/
    │   │   ├── T1/           <- modality folders (ScalarImage)
    │   │   │   └── *.nii.gz
    │   │   ├── T2/
    │   │   │   └── *.nii.gz
    │   │   └── label/        <- segmentation mask (LabelMap)
    │   │       └── *.nii.gz
    │   └── subject_002/ ...
    ├── validation/
    └── test/

All functions read dataset parameters from the global Config singleton;
no 'args' object is passed around.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torchio as tio
from torch.utils.data import DataLoader

from config import Config


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _find_nifti(folder: Path) -> Optional[Path]:
    """Return the first NIfTI file in *folder* (.nii.gz preferred)."""
    for pattern in ('*.nii.gz', '*.nii'):
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


def discover_modalities(split_dir: Path, label_name: str) -> List[str]:
    """
    Scan the first valid subject in *split_dir* and return modality names.

    Modality names are folder names that contain at least one NIfTI file
    and are not the label folder.
    """
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        modalities = [
            d.name
            for d in sorted(subj_dir.iterdir())
            if d.is_dir() and d.name != label_name and _find_nifti(d) is not None
        ]
        if modalities:
            return modalities
    return []


# ── Subject building ───────────────────────────────────────────────────────────

def build_subjects(
    split_dir: Path,
    modalities: List[str],
    label_name: str,
    require_label: bool = True,
) -> List[tio.Subject]:
    """Create a list of :class:`tio.Subject` from a split directory."""
    split_dir = Path(split_dir)
    subjects: List[tio.Subject] = []

    if not split_dir.exists():
        print(f"[dataset] Split directory not found: {split_dir}")
        return subjects

    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        kwargs: dict = {'subject_id': subj_dir.name}
        skip = False

        for mod in modalities:
            mod_dir = subj_dir / mod
            if not mod_dir.exists():
                print(f"[dataset] Missing modality '{mod}' for {subj_dir.name} — skipped")
                skip = True
                break
            nii = _find_nifti(mod_dir)
            if nii is None:
                print(f"[dataset] No NIfTI in {mod_dir} — skipped")
                skip = True
                break
            kwargs[mod] = tio.ScalarImage(str(nii))

        if skip:
            continue

        label_dir = subj_dir / label_name
        if label_dir.exists():
            nii = _find_nifti(label_dir)
            if nii:
                kwargs[label_name] = tio.LabelMap(str(nii))
        elif require_label:
            print(f"[dataset] Missing label folder for {subj_dir.name} — skipped")
            continue

        subjects.append(tio.Subject(**kwargs))

    print(f"[dataset] Loaded {len(subjects)} subjects from {split_dir}")
    return subjects


# ── Transforms ────────────────────────────────────────────────────────────────

def get_preprocessing_transform() -> tio.Compose:
    """Build the preprocessing pipeline from the global Config."""
    cfg = Config.get()
    transforms = []
    if cfg.data.target_spacing is not None:
        transforms.append(tio.Resample(cfg.data.target_spacing))
    if cfg.data.target_shape is not None:
        transforms.append(tio.CropOrPad(cfg.data.target_shape))
    if cfg.data.normalization == 'znorm':
        transforms.append(tio.ZNormalization(masking_method=tio.ZNormalization.mean))
    elif cfg.data.normalization == 'rescale':
        transforms.append(tio.RescaleIntensity(out_min_max=(0.0, 1.0)))
    return tio.Compose(transforms)


def get_augmentation_transform() -> tio.Compose:
    """Build the augmentation pipeline from the global Config."""
    cfg = Config.get()
    transforms = [
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=10, p=0.5),
        tio.RandomNoise(std=(0.0, 0.1), p=0.3),
        tio.RandomBlur(std=(0.0, 1.0), p=0.3),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
    ]
    if cfg.augment.elastic_deformation:
        transforms.insert(2, tio.RandomElasticDeformation(p=0.3))
    return tio.Compose(transforms)


# ── Dataset / DataLoader factory ───────────────────────────────────────────────

def create_datasets() -> Tuple[
    tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset
]:
    """
    Build train, validation, and test datasets from the global Config.

    When ``cfg.data.modalities`` is ``None`` the modality names are
    auto-detected from the first subject in the train split, and the
    Config singleton is updated in-place so that other modules see the
    resolved list.
    """
    cfg = Config.get()
    data_root = Path(cfg.data.data_root)

    # Auto-detect modalities and persist back into the singleton
    if cfg.data.modalities is None:
        cfg.data.modalities = discover_modalities(
            data_root / 'train', cfg.data.label_name
        )
        if not cfg.data.modalities:
            raise RuntimeError(
                f"Could not discover modalities under {data_root / 'train'}. "
                "Check data_root and label_name."
            )
        print(f"[dataset] Auto-discovered modalities: {cfg.data.modalities}")

    modalities  = cfg.data.modalities
    label_name  = cfg.data.label_name

    preprocess = get_preprocessing_transform()
    augment    = get_augmentation_transform()
    train_transform = tio.Compose([preprocess, augment]) if cfg.augment.enabled else preprocess

    train_subjects = build_subjects(data_root / 'train',      modalities, label_name)
    val_subjects   = build_subjects(data_root / 'validation', modalities, label_name)
    test_subjects  = build_subjects(data_root / 'test',       modalities, label_name,
                                    require_label=True)

    return (
        tio.SubjectsDataset(train_subjects, transform=train_transform),
        tio.SubjectsDataset(val_subjects,   transform=preprocess),
        tio.SubjectsDataset(test_subjects,  transform=preprocess),
    )


def create_data_loaders(
    train_dataset: tio.SubjectsDataset,
    val_dataset: tio.SubjectsDataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader)`` configured from the global Config.

    Patch-based mode (``cfg.patch.enabled``) uses :class:`tio.Queue`;
    otherwise plain :class:`~torch.utils.data.DataLoader` is used.
    """
    cfg = Config.get()

    if cfg.patch.enabled:
        patch_size = cfg.patch.size

        sampler = (
            tio.data.WeightedSampler(patch_size, cfg.data.label_name)
            if cfg.patch.weighted_sampling
            else tio.data.UniformSampler(patch_size)
        )

        train_queue = tio.Queue(
            subjects_dataset=train_dataset,
            max_length=cfg.patch.queue_max_length,
            samples_per_volume=cfg.patch.samples_per_volume,
            sampler=sampler,
            num_workers=cfg.infra.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        val_queue = tio.Queue(
            subjects_dataset=val_dataset,
            max_length=cfg.patch.queue_max_length,
            samples_per_volume=cfg.patch.samples_per_volume,
            sampler=tio.data.UniformSampler(patch_size),
            num_workers=cfg.infra.num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        # num_workers must be 0 on the DataLoader when using a Queue
        return (
            DataLoader(train_queue, batch_size=cfg.training.batch_size, num_workers=0),
            DataLoader(val_queue,   batch_size=cfg.training.batch_size, num_workers=0),
        )

    return (
        DataLoader(train_dataset,
                   batch_size=cfg.training.batch_size,
                   shuffle=True,
                   num_workers=cfg.infra.num_workers,
                   pin_memory=True),
        DataLoader(val_dataset,
                   batch_size=cfg.training.batch_size,
                   shuffle=False,
                   num_workers=cfg.infra.num_workers,
                   pin_memory=True),
    )
