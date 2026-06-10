"""
Dataset utilities for the SDF Estimation application.

Expected directory layout::

    <data_root>/
    ├── train/
    │   ├── subject_001/
    │   │   ├── T1/          *.nii.gz   ->  tio.ScalarImage  (input modality)
    │   │   ├── T2/          *.nii.gz   ->  tio.ScalarImage  (input modality)
    │   │   ├── sdf_bone/    *.nii.gz   ->  tio.ScalarImage  (float32 SDF field)
    │   │   └── sdf_muscle/  *.nii.gz   ->  tio.ScalarImage  (float32 SDF field)
    │   └── subject_002/ ...
    ├── validation/
    └── test/

SDF fields are loaded as :class:`tio.ScalarImage` with float32 data and are
**never** intensity-normalised or affected by intensity augmentations (Noise,
Blur, Gamma).  Spatial augmentations (Flip, Affine, Elastic) apply to all
images (modalities + SDF fields).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torchio as tio
from torch.utils.data import DataLoader

from configuration.manager import ConfigManager
from apps.sdf_estimation.sdf_config import (
    AugmentConfig,
    DataConfig,
    InfraConfig,
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


# ── Discovery helpers ──────────────────────────────────────────────────────────

def discover_modalities(split_dir: Path, exclude: List[str]) -> List[str]:
    """
    Scan the first valid subject in *split_dir* and return modality names.

    A folder qualifies if its name is not in *exclude* and it contains at
    least one NIfTI file.  *exclude* should contain all ``sdf_names`` so that
    SDF folders are not mistaken for input modalities.
    """
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        mods = [
            d.name
            for d in sorted(subj_dir.iterdir())
            if d.is_dir() and d.name not in exclude and _find_nifti(d) is not None
        ]
        if mods:
            return mods
    return []


# ── Subject building ───────────────────────────────────────────────────────────

def build_sdf_subjects(
    split_dir: Path,
    modalities: List[str],
    sdf_names: List[str],
    require_sdf: bool = True,
) -> List[tio.Subject]:
    """
    Load all valid subjects for SDF estimation.

    Each input modality becomes a :class:`tio.ScalarImage`.
    Each SDF field becomes a :class:`tio.ScalarImage` with float32 data.

    Subjects missing any required modality or (when *require_sdf* is True)
    any SDF field folder are silently skipped.
    """
    split_dir = Path(split_dir)
    subjects: List[tio.Subject] = []

    if not split_dir.exists():
        print(f'[dataset] Split directory not found: {split_dir}')
        return subjects

    ref_img = None
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        kwargs: dict = {'subject_id': subj_dir.name}
        skip = False

        # ── Input modalities ───────────────────────────────────────────────
        for mod in modalities:
            mod_dir = subj_dir / mod
            if not mod_dir.exists():
                print(f'[dataset] Missing modality {mod!r} for {subj_dir.name} -- skipped')
                skip = True
                break
            nii = _find_nifti(mod_dir)
            if nii is None:
                print(f'[dataset] No NIfTI in {mod_dir} -- skipped')
                skip = True
                break

            img = tio.ScalarImage(str(nii))
            if ref_img is None:
                ref_img = img
                kwargs[mod] = img
            else:
                img = tio.Resample(ref_img)(img)
                kwargs[mod] = img

        if skip:
            continue

        # ── SDF fields ────────────────────────────────────────────────────
        for sdf in sdf_names:
            sdf_dir = subj_dir / sdf
            if not sdf_dir.exists():
                if require_sdf:
                    print(f'[dataset] Missing SDF folder {sdf!r} for {subj_dir.name} -- skipped')
                    skip = True
                    break
                continue
            nii = _find_nifti(sdf_dir)
            if nii is None:
                if require_sdf:
                    print(f'[dataset] No NIfTI in {sdf_dir} -- skipped')
                    skip = True
                    break
                continue
            # Load SDF as ScalarImage; TorchIO reads float32 NIfTI natively.
            sdf_img = tio.ScalarImage(str(nii))
            sdf_img = tio.Resample(ref_img)(sdf_img)
            kwargs[sdf] = sdf_img

        if skip:
            continue

        subjects.append(tio.Subject(**kwargs))

    print(f'[dataset] Loaded {len(subjects)} subjects from {split_dir}')
    return subjects


# ── Transforms ────────────────────────────────────────────────────────────────

def get_preprocessing_transform(modalities: List[str]) -> tio.Compose:
    """
    Build the preprocessing pipeline from ``DataConfig``.

    Order: Resample -> CropOrPad -> Intensity normalisation.
    Normalisation uses ``include=modalities`` so SDF fields are never scaled.
    """
    dcfg: DataConfig = ConfigManager.get().get_config(ConfigManager.DATA)
    transforms = []
    if dcfg.target_spacing is not None:
        transforms.append(tio.Resample(dcfg.target_spacing))
    if dcfg.target_shape is not None:
        transforms.append(tio.CropOrPad(dcfg.target_shape))
    if dcfg.normalization == 'znorm':
        transforms.append(tio.ZNormalization(
            masking_method=tio.ZNormalization.mean,
            include=list(modalities),
        ))
    elif dcfg.normalization == 'rescale':
        transforms.append(tio.RescaleIntensity(
            out_min_max=(0.0, 1.0),
            include=list(modalities),
        ))
    return tio.Compose(transforms)


def get_augmentation_transform(modalities: List[str]) -> tio.Compose:
    """
    Build the augmentation pipeline from ``AugmentConfig``.

    Spatial transforms (Flip, Affine, Elastic) have no ``include`` restriction
    and are applied to both modalities and SDF fields.

    Intensity transforms (Noise, Blur, Gamma) use ``include=modalities`` so
    they are **never** applied to SDF fields.

    Returns an empty :class:`tio.Compose` when ``AugmentConfig.enabled`` is False.
    """
    a: AugmentConfig = ConfigManager.get().get_config(ConfigManager.AUGMENT)
    transforms = []

    if not a.enabled:
        return tio.Compose(transforms)

    # ── Spatial (all images including SDF fields) ──────────────────────────
    if a.flip:
        transforms.append(tio.RandomFlip(
            axes=a.flip_axes,
            flip_probability=a.flip_p,
        ))

    if a.affine:
        transforms.append(tio.RandomAffine(
            scales=a.affine_scales,
            degrees=a.affine_degrees,
            translation=a.affine_translation,
            p=a.affine_p,
        ))

    if a.elastic:
        transforms.append(tio.RandomElasticDeformation(p=a.elastic_p))

    # ── Intensity (modalities only -- SDF fields excluded) ─────────────────
    if a.noise:
        transforms.append(tio.RandomNoise(
            std=a.noise_std,
            p=a.noise_p,
            include=list(modalities),
        ))

    if a.blur:
        transforms.append(tio.RandomBlur(
            std=a.blur_std,
            p=a.blur_p,
            include=list(modalities),
        ))

    if a.gamma:
        transforms.append(tio.RandomGamma(
            log_gamma=a.gamma_log_gamma,
            p=a.gamma_p,
            include=list(modalities),
        ))

    return tio.Compose(transforms)


# ── Dataset / DataLoader factory ───────────────────────────────────────────────

def create_sdf_datasets() -> Tuple[
    tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset
]:
    """
    Build ``(train_dataset, val_dataset, test_dataset)`` from the ConfigManager.

    ``sdf_names`` must be explicitly set in DataConfig before calling this.
    When ``DataConfig.modalities`` is ``None`` it is auto-detected from the
    first training subject (excluding all sdf_names folders) and written back
    into the ConfigManager so that the Trainer and model builder see the
    resolved list.
    """
    m = ConfigManager.get()
    dcfg: DataConfig = m.get_config(ConfigManager.DATA)
    acfg: AugmentConfig = m.get_config(ConfigManager.AUGMENT)

    if not dcfg.sdf_names:
        raise RuntimeError(
            'DataConfig.sdf_names must be set explicitly before building datasets. '
            'Example: sdf_names = ["sdf_bone", "sdf_muscle"]'
        )

    data_root = Path(dcfg.data_root)
    sdf_names: List[str] = list(dcfg.sdf_names)

    if dcfg.modalities is None:
        dcfg.modalities = discover_modalities(
            data_root / 'train', exclude=sdf_names)
        if not dcfg.modalities:
            raise RuntimeError(
                f'Could not discover modalities under {data_root / "train"}. '
                'Check data_root and ensure modality folders are present.'
            )
        print(f'[dataset] Auto-discovered modalities: {dcfg.modalities}')

    # Resolve num_sdf_fields from sdf_names and write back to config
    dcfg.num_sdf_fields = len(sdf_names)

    modalities: List[str] = list(dcfg.modalities)

    preprocess = get_preprocessing_transform(modalities)
    augment = get_augmentation_transform(modalities)
    train_tf = tio.Compose([preprocess, augment]) if acfg.enabled else preprocess

    return (
        tio.SubjectsDataset(
            build_sdf_subjects(data_root / 'train', modalities, sdf_names),
            transform=train_tf,
        ),
        tio.SubjectsDataset(
            build_sdf_subjects(data_root / 'validation', modalities, sdf_names),
            transform=preprocess,
        ),
        tio.SubjectsDataset(
            build_sdf_subjects(data_root / 'test', modalities, sdf_names,
                               require_sdf=True),
            transform=preprocess,
        ),
    )


def create_data_loaders(
    train_dataset: tio.SubjectsDataset,
    val_dataset: tio.SubjectsDataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader)``.

    SDF estimation always uses full-volume (global) training -- patch mode is
    not supported.  Batch size is read from TrainingConfig; gradient
    accumulation keeps effective batch size intact.
    """
    m = ConfigManager.get()
    tcfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
    icfg: InfraConfig = m.get_config(ConfigManager.INFRA)

    bs = 1 if tcfg.gradient_accumulation else tcfg.batch_size

    return (
        DataLoader(train_dataset, batch_size=bs, shuffle=True,
                   num_workers=icfg.num_workers, pin_memory=False),
        DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                   num_workers=icfg.num_workers, pin_memory=False),
    )
