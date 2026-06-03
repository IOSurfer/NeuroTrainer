"""
Dataset utilities for the LabelMap Segmentation application.

Expected directory layout:
    <data_root>/
    ├── train/
    │   ├── subject_001/
    │   │   ├── T1/   *.nii.gz   →  tio.ScalarImage
    │   │   ├── T2/   *.nii.gz   →  tio.ScalarImage
    │   │   └── label/ *.nii.gz  →  tio.LabelMap
    │   └── subject_002/ ...
    ├── validation/
    └── test/
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torchio as tio
from torch.utils.data import DataLoader

from config.manager import ConfigManager
from apps.labelmap_segmentation.config import (
    AugmentConfig,
    DataConfig,
    InfraConfig,
    PatchConfig,
    TrainingConfig,
)


# ── File helpers ───────────────────────────────────────────────────────────────

def _find_nifti(folder: Path) -> Optional[Path]:
    """Return the first NIfTI file in *folder* (.nii.gz preferred over .nii)."""
    for pattern in ('*.nii.gz', '*.nii'):
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    return None


def discover_modalities(split_dir: Path, label_name: str) -> List[str]:
    """
    Scan the first valid subject in *split_dir* and return modality names.
    A folder qualifies if it is not *label_name* and contains at least one NIfTI file.
    """
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        mods = [
            d.name
            for d in sorted(subj_dir.iterdir())
            if d.is_dir() and d.name != label_name and _find_nifti(d) is not None
        ]
        if mods:
            return mods
    return []


# ── Subject building ───────────────────────────────────────────────────────────

def build_labelmap_segmentation_subjects(
    split_dir: Path,
    modalities: List[str],
    label_name: str,
    require_label: bool = True,
) -> List[tio.Subject]:
    """
    Load all valid subjects for label-map segmentation.

    Subjects missing a modality folder, a NIfTI file, or (when
    *require_label* is ``True``) a label folder are silently skipped.

    Returns a list of :class:`tio.Subject` where:
    - Each modality → :class:`tio.ScalarImage`
    - Label mask   → :class:`tio.LabelMap`
    - ``subject_id`` attribute set to the folder name
    """
    split_dir = Path(split_dir)
    subjects: List[tio.Subject] = []

    if not split_dir.exists():
        print(f'[dataset] Split directory not found: {split_dir}')
        return subjects

    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        kwargs: dict = {'subject_id': subj_dir.name}
        skip = False

        for mod in modalities:
            mod_dir = subj_dir / mod
            if not mod_dir.exists():
                print(
                    f'[dataset] Missing modality {mod!r} for {subj_dir.name} — skipped')
                skip = True
                break
            nii = _find_nifti(mod_dir)
            if nii is None:
                print(f'[dataset] No NIfTI in {mod_dir} — skipped')
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
            print(
                f'[dataset] Missing label folder for {subj_dir.name} — skipped')
            continue

        subjects.append(tio.Subject(**kwargs))

    print(f'[dataset] Loaded {len(subjects)} subjects from {split_dir}')
    return subjects


# ── Transforms ────────────────────────────────────────────────────────────────

def get_preprocessing_transform() -> tio.Compose:
    """
    Build the preprocessing pipeline from ``DataConfig``.
    Order: Resample → CropOrPad → intensity normalisation.
    """
    dcfg: DataConfig = ConfigManager.get().get_config(ConfigManager.DATA)
    transforms = []
    if dcfg.target_spacing is not None:
        transforms.append(tio.Resample(dcfg.target_spacing))
    if dcfg.target_shape is not None:
        transforms.append(tio.CropOrPad(dcfg.target_shape))
    if dcfg.normalization == 'znorm':
        transforms.append(tio.ZNormalization(
            masking_method=tio.ZNormalization.mean))
    elif dcfg.normalization == 'rescale':
        transforms.append(tio.RescaleIntensity(out_min_max=(0.0, 1.0)))
    return tio.Compose(transforms)


def get_augmentation_transform() -> tio.Compose:
    """Build the spatial and intensity augmentation pipeline from ``AugmentConfig``."""
    acfg: AugmentConfig = ConfigManager.get().get_config(ConfigManager.AUGMENT)
    transforms = [
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=15, translation=10, p=0.5),
        tio.RandomNoise(std=(0.0, 0.1), p=0.3),
        tio.RandomBlur(std=(0.0, 1.0), p=0.3),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
    ]
    if acfg.elastic_deformation:
        transforms.insert(2, tio.RandomElasticDeformation(p=0.3))
    return tio.Compose(transforms)


# ── Dataset / DataLoader factory ───────────────────────────────────────────────

def create_labelmap_segmentation_datasets() -> Tuple[
    tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset
]:
    """
    Build ``(train_dataset, val_dataset, test_dataset)`` from the ConfigManager.

    When ``DataConfig.modalities`` is ``None`` the names are auto-detected from
    the first subject and written back into the ConfigManager so that the
    Trainer and model builder see the resolved list.
    """
    m = ConfigManager.get()
    dcfg: DataConfig = m.get_config(ConfigManager.DATA)
    acfg: AugmentConfig = m.get_config(ConfigManager.AUGMENT)

    data_root = Path(dcfg.data_root)

    if dcfg.modalities is None:
        dcfg.modalities = discover_modalities(
            data_root / 'train', dcfg.label_name)
        if not dcfg.modalities:
            raise RuntimeError(
                f'Could not discover modalities under {data_root / "train"}. '
                'Check data_root and label_name.'
            )
        print(f'[dataset] Auto-discovered modalities: {dcfg.modalities}')

    modalities = dcfg.modalities
    label_name = dcfg.label_name

    preprocess = get_preprocessing_transform()
    augment = get_augmentation_transform()
    train_tf = tio.Compose([preprocess, augment]
                           ) if acfg.enabled else preprocess

    return (
        tio.SubjectsDataset(
            build_labelmap_segmentation_subjects(
                data_root / 'train', modalities, label_name),
            transform=train_tf,
        ),
        tio.SubjectsDataset(
            build_labelmap_segmentation_subjects(
                data_root / 'validation', modalities, label_name),
            transform=preprocess,
        ),
        tio.SubjectsDataset(
            build_labelmap_segmentation_subjects(data_root / 'test', modalities, label_name,
                                                 require_label=True),
            transform=preprocess,
        ),
    )


def create_data_loaders(
    train_dataset: tio.SubjectsDataset,
    val_dataset:   tio.SubjectsDataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader)``.

    Patch mode (``PatchConfig.enabled``) wraps datasets in :class:`tio.Queue`.
    """
    m = ConfigManager.get()
    pcfg: PatchConfig = m.get_config(ConfigManager.PATCH)
    dcfg: DataConfig = m.get_config(ConfigManager.DATA)
    tcfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
    icfg: InfraConfig = m.get_config(ConfigManager.INFRA)

    if pcfg.enabled:
        patch_size = pcfg.size
        sampler = (
            tio.data.WeightedSampler(patch_size, dcfg.label_name)
            if pcfg.weighted_sampling
            else tio.data.UniformSampler(patch_size)
        )
        train_q = tio.Queue(
            subjects_dataset=train_dataset,
            max_length=pcfg.queue_max_length,
            samples_per_volume=pcfg.samples_per_volume,
            sampler=sampler,
            num_workers=icfg.num_workers,
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        val_q = tio.Queue(
            subjects_dataset=val_dataset,
            max_length=pcfg.queue_max_length,
            samples_per_volume=pcfg.samples_per_volume,
            sampler=tio.data.UniformSampler(patch_size),
            num_workers=icfg.num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        return (
            DataLoader(train_q, batch_size=tcfg.batch_size, num_workers=0),
            DataLoader(val_q,   batch_size=tcfg.batch_size, num_workers=0),
        )

    return (
        DataLoader(train_dataset,
                   batch_size=tcfg.batch_size, shuffle=True,
                   num_workers=icfg.num_workers, pin_memory=True),
        DataLoader(val_dataset,
                   batch_size=tcfg.batch_size, shuffle=False,
                   num_workers=icfg.num_workers, pin_memory=True),
    )
