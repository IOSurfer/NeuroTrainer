"""
Dataset utilities for the Multi-Head Segmentation application.

Expected directory layout (one label folder per task, analogous to
sdf_estimation's multiple SDF-field folders):

    <data_root>/
    ├── train/
    │   ├── subject_001/
    │   │   ├── T1/        *.nii.gz   ->  tio.ScalarImage  (modality, shared)
    │   │   ├── T2/        *.nii.gz   ->  tio.ScalarImage  (modality, shared)
    │   │   ├── organs/    *.nii.gz   ->  tio.LabelMap     (task "organs")
    │   │   └── tumor/     *.nii.gz   ->  tio.LabelMap     (task "tumor")
    │   └── subject_002/ ...
    ├── validation/
    └── test/

Preprocessing / augmentation pipelines and mask transforms are reused
unchanged from apps.labelmap_segmentation.dataset -- they only consult
DataConfig fields that are identical between the two apps (target_spacing,
target_shape, normalization, znorm_mask_name, foreground_mask_name) and
AugmentConfig (unchanged).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torchio as tio
from torch.utils.data import DataLoader

from configuration.manager import ConfigManager
from apps.labelmap_segmentation.dataset import (
    _find_nifti,
    get_augmentation_transform,
    get_preprocessing_transform,
)
from apps.labelmap_segmentation.segmentation_config import (
    AugmentConfig,
    InfraConfig,
    PatchConfig,
    TrainingConfig,
)
from apps.multihead_segmentation.multihead_config import DataConfig

# ── Discovery helpers ──────────────────────────────────────────────────────────


def discover_modalities(split_dir: Path, task_names: List[str]) -> List[str]:
    """
    Scan the first valid subject in *split_dir* and return modality names.

    A folder qualifies if its name is not one of *task_names* and it
    contains at least one NIfTI file.
    """
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        mods = [
            d.name
            for d in sorted(subj_dir.iterdir())
            if d.is_dir() and d.name not in task_names and _find_nifti(d) is not None
        ]
        if mods:
            return mods
    return []


# ── Subject building ───────────────────────────────────────────────────────────


def build_multihead_subjects(
    split_dir: Path,
    modalities: List[str],
    task_names: List[str],
    require_labels: bool = True,
    mask_names: Optional[List[str]] = None,
) -> List[tio.Subject]:
    """
    Load all valid subjects for multi-head segmentation.

    Each input modality becomes a :class:`tio.ScalarImage`. Each task becomes
    a :class:`tio.LabelMap` keyed by its task name (label folder name).

    Subjects missing a modality folder, a NIfTI file, or (when
    *require_labels* is True) any task's label folder are silently skipped.

    *mask_names* are additional labelmap folders loaded for use as
    normalization / foreground masks (see ``DataConfig.znorm_mask_name`` and
    ``DataConfig.foreground_mask_name``). Names already present in
    *task_names* are ignored -- that task's label doubles as the mask.
    """
    split_dir = Path(split_dir)
    subjects: List[tio.Subject] = []

    if not split_dir.exists():
        print(f"[dataset] Split directory not found: {split_dir}")
        return subjects

    extra_masks = [
        name for name in dict.fromkeys(mask_names or []) if name and name not in task_names
    ]

    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        kwargs: dict = {"subject_id": subj_dir.name}
        skip = False

        # ── Input modalities ───────────────────────────────────────────────
        for mod in modalities:
            mod_dir = subj_dir / mod
            if not mod_dir.exists():
                print(
                    f"[dataset] Missing modality {mod!r} for {subj_dir.name} -- skipped"
                )
                skip = True
                break
            nii = _find_nifti(mod_dir)
            if nii is None:
                print(f"[dataset] No NIfTI in {mod_dir} -- skipped")
                skip = True
                break
            kwargs[mod] = tio.ScalarImage(str(nii))

        if skip:
            continue

        # ── Task labels ───────────────────────────────────────────────────
        for task in task_names:
            task_dir = subj_dir / task
            if not task_dir.exists():
                if require_labels:
                    print(
                        f"[dataset] Missing task folder {task!r} for {subj_dir.name} -- skipped"
                    )
                    skip = True
                    break
                continue
            nii = _find_nifti(task_dir)
            if nii is None:
                if require_labels:
                    print(f"[dataset] No NIfTI in {task_dir} -- skipped")
                    skip = True
                    break
                continue
            kwargs[task] = tio.LabelMap(str(nii))

        if skip:
            continue

        # ── Extra masks (znorm / foreground) ─────────────────────────────
        for mask_name in extra_masks:
            mask_dir = subj_dir / mask_name
            nii = _find_nifti(mask_dir) if mask_dir.exists() else None
            if nii is None:
                print(
                    f"[dataset] Missing mask {mask_name!r} for {subj_dir.name} -- skipped"
                )
                skip = True
                break
            kwargs[mask_name] = tio.LabelMap(str(nii))

        if skip:
            continue

        subjects.append(tio.Subject(**kwargs))

    print(f"[dataset] Loaded {len(subjects)} subjects from {split_dir}")
    return subjects


# ── Dataset / DataLoader factory ───────────────────────────────────────────────


def create_multihead_datasets() -> (
    Tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]
):
    """
    Build ``(train_dataset, val_dataset, test_dataset)`` from the ConfigManager.

    ``DataConfig.tasks`` must be set explicitly before calling this. When
    ``DataConfig.modalities`` is ``None`` it is auto-detected from the first
    training subject (excluding all task folders) and written back into the
    ConfigManager so that the Trainer and model builder see the resolved list.
    """
    m = ConfigManager.get()
    dcfg: DataConfig = m.get_config(ConfigManager.DATA)
    acfg: AugmentConfig = m.get_config(ConfigManager.AUGMENT)

    if not dcfg.tasks:
        raise RuntimeError(
            "DataConfig.tasks must be set explicitly before building datasets. "
            'Example: tasks = {"organs": 14, "tumor": 2}'
        )

    data_root = Path(dcfg.data_root)
    task_names: List[str] = list(dcfg.tasks.keys())

    if dcfg.modalities is None:
        dcfg.modalities = discover_modalities(data_root / "train", task_names)
        if not dcfg.modalities:
            raise RuntimeError(
                f'Could not discover modalities under {data_root / "train"}. '
                "Check data_root and tasks."
            )
        print(f"[dataset] Auto-discovered modalities: {dcfg.modalities}")

    modalities: List[str] = list(dcfg.modalities)
    mask_names = [dcfg.znorm_mask_name, dcfg.foreground_mask_name]

    preprocess = get_preprocessing_transform()
    augment = get_augmentation_transform()
    train_tf = tio.Compose([preprocess, augment]) if acfg.enabled else preprocess

    return (
        tio.SubjectsDataset(
            build_multihead_subjects(
                data_root / "train", modalities, task_names, mask_names=mask_names
            ),
            transform=train_tf,
        ),
        tio.SubjectsDataset(
            build_multihead_subjects(
                data_root / "validation", modalities, task_names, mask_names=mask_names
            ),
            transform=preprocess,
        ),
        tio.SubjectsDataset(
            build_multihead_subjects(
                data_root / "test",
                modalities,
                task_names,
                require_labels=True,
                mask_names=mask_names,
            ),
            transform=preprocess,
        ),
    )


def create_data_loaders(
    train_dataset: tio.SubjectsDataset,
    val_dataset: tio.SubjectsDataset,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader)``.

    Patch mode (``PatchConfig.enabled``) wraps datasets in :class:`tio.Queue`.
    Weighted patch sampling uses the first task in ``DataConfig.tasks`` as the
    sampling-density reference (dict insertion order is preserved).
    """
    m = ConfigManager.get()
    pcfg: PatchConfig = m.get_config(ConfigManager.PATCH)
    dcfg: DataConfig = m.get_config(ConfigManager.DATA)
    tcfg: TrainingConfig = m.get_config(ConfigManager.TRAINING)
    icfg: InfraConfig = m.get_config(ConfigManager.INFRA)

    if pcfg.enabled:
        patch_size = pcfg.size
        primary_task = next(iter(dcfg.tasks))
        sampler = (
            tio.data.WeightedSampler(patch_size, primary_task)
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
            DataLoader(
                train_q,
                batch_size=1 if tcfg.gradient_accumulation else tcfg.batch_size,
                num_workers=0,
            ),
            DataLoader(
                val_q,
                batch_size=1 if tcfg.gradient_accumulation else tcfg.batch_size,
                num_workers=0,
            ),
        )

    return (
        DataLoader(
            train_dataset,
            batch_size=1 if tcfg.gradient_accumulation else tcfg.batch_size,
            shuffle=True,
            num_workers=icfg.num_workers,
            pin_memory=False,
        ),
        DataLoader(
            val_dataset,
            batch_size=1 if tcfg.gradient_accumulation else tcfg.batch_size,
            shuffle=False,
            num_workers=icfg.num_workers,
            pin_memory=False,
        ),
    )
