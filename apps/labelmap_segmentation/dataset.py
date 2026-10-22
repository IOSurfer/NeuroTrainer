"""
Dataset utilities for the LabelMap Segmentation application.

Expected directory layout:
    <data_root>/
    ├── train/
    │   ├── subject_001/
    │   │   ├── T1/   *.nii.gz   ->  tio.ScalarImage
    │   │   ├── T2/   *.nii.gz   ->  tio.ScalarImage
    │   │   └── label/ *.nii.gz  ->  tio.LabelMap
    │   └── subject_002/ ...
    ├── validation/
    └── test/
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchio as tio
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from configuration.manager import ConfigManager
from apps.labelmap_segmentation.segmentation_config import (
    AugmentConfig,
    DataConfig,
    InfraConfig,
    PatchConfig,
    TrainingConfig,
)

# ── File helpers ───────────────────────────────────────────────────────────────


def _find_nifti(folder: Path) -> Optional[Path]:
    """Return the first NIfTI file in *folder* (.nii.gz preferred over .nii)."""
    for pattern in ("*.nii.gz", "*.nii"):
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
    mask_names: Optional[List[str]] = None,
) -> List[tio.Subject]:
    """
    Load all valid subjects for label-map segmentation.

    Subjects missing a modality folder, a NIfTI file, or (when
    *require_label* is ``True``) a label folder are silently skipped.

    *mask_names* are additional labelmap folders loaded for use as
    normalization / foreground masks (see ``DataConfig.znorm_mask_name`` and
    ``DataConfig.foreground_mask_name``). Subjects missing one of these
    folders are skipped. Names equal to *label_name* are ignored -- the
    segmentation label doubles as that mask.

    Returns a list of :class:`tio.Subject` where:
    - Each modality -> :class:`tio.ScalarImage`
    - Label mask   -> :class:`tio.LabelMap`
    - Extra masks  -> :class:`tio.LabelMap`, keyed by folder name
    - ``subject_id`` attribute set to the folder name
    """
    split_dir = Path(split_dir)
    subjects: List[tio.Subject] = []

    if not split_dir.exists():
        print(f"[dataset] Split directory not found: {split_dir}")
        return subjects

    extra_masks = [
        name for name in dict.fromkeys(mask_names or []) if name and name != label_name
    ]

    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        kwargs: dict = {"subject_id": subj_dir.name}
        skip = False

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

            img = tio.ScalarImage(str(nii))
            kwargs[mod] = img

        if skip:
            continue

        label_dir = subj_dir / label_name
        if label_dir.exists():
            nii = _find_nifti(label_dir)
            if nii:
                label = tio.LabelMap(str(nii))
                kwargs[label_name] = label
        elif require_label:
            print(f"[dataset] Missing label folder for {subj_dir.name} -- skipped")
            continue

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


# ── Transforms ────────────────────────────────────────────────────────────────


class ZeroOutsideMask(tio.Transform):
    """
    Fill intensity-image voxels outside a labelmap's non-zero region with a
    constant value.

    Voxels where ``subject[mask_name]`` is non-zero keep their (normalized)
    value; all other voxels are set to ``fill_value`` (default 0). Intended
    to run after intensity normalization (see ``DataConfig.foreground_mask_name``
    and ``DataConfig.background_fill_value``).
    """

    def __init__(self, mask_name: str, fill_value: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_name = mask_name
        self.fill_value = fill_value
        self.args_names = ["mask_name", "fill_value"]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        mask = subject[self.mask_name].data != 0
        for image in subject.get_images(intensity_only=True):
            image.set_data(torch.where(mask, image.data, torch.full_like(image.data, self.fill_value)))
        return subject


def soft_label_name(label_name: str) -> str:
    """Name of the soft-label image that ``SoftLabel`` adds next to the hard label ``label_name``."""
    return f"{label_name}_soft"


class SoftLabel(tio.Transform):
    """
    Add a soft version of a labelmap as an extra *intensity* image of shape ``[num_classes, D, H, W]``.

    Channel ``c`` is the one-hot mask of class ``c`` smoothed by a Gaussian (``sigma_mm`` in mm, converted
    to voxels with the label's spacing, kernel radius ``truncate * sigma``); channels sum to 1 everywhere.
    The hard labelmap is kept untouched (metrics, foreground masks).

    The soft label is a ScalarImage on purpose: torchio resamples ScalarImages linearly and LabelMaps with
    nearest neighbour, so the spatial augmentations (affine, elastic) interpolate the soft label linearly
    and keep it a probability map. Intensity augmentations must exclude it (see ``get_augmentation_transform``).
    Run it after the intensity normalisation / masking steps, which would otherwise modify it too.
    """

    def __init__(
        self,
        label_name: str,
        num_classes: int,
        sigma_mm: Union[float, Sequence[float]],
        truncate: float = 4.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label_name = label_name
        self.num_classes = num_classes
        self.sigma_mm = sigma_mm
        self.truncate = truncate
        self.args_names = ["label_name", "num_classes", "sigma_mm", "truncate"]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        label = subject[self.label_name]
        hard = label.data[0].long()
        sigma_mm = np.broadcast_to(np.asarray(self.sigma_mm, dtype=float), (3,))
        sigma_vox = sigma_mm / np.asarray(label.spacing, dtype=float)

        soft = torch.zeros((self.num_classes, *hard.shape), dtype=torch.float32)
        foreground = hard > 0
        if foreground.any():
            # only the neighbourhood of the foreground can differ from 0: smooth inside its bounding box plus the kernel radius
            idx = foreground.nonzero()
            margin = [int(math.ceil(self.truncate * s)) + 1 for s in sigma_vox]
            lo = [max(int(idx[:, a].min()) - margin[a], 0) for a in range(3)]
            hi = [min(int(idx[:, a].max()) + 1 + margin[a], hard.shape[a]) for a in range(3)]
            box = tuple(slice(l, h) for l, h in zip(lo, hi))
            crop = hard[box].numpy()
            for c in range(1, self.num_classes):
                if (crop == c).any():
                    smoothed = gaussian_filter((crop == c).astype(np.float32), sigma_vox, truncate=self.truncate, mode="constant")
                    soft[(c, *box)] = torch.from_numpy(smoothed)
        soft[0] = (1.0 - soft[1:].sum(dim=0)).clamp_(min=0.0)  # the smoothed background mask
        subject.add_image(tio.ScalarImage(tensor=soft, affine=label.affine), soft_label_name(self.label_name))
        return subject


def get_preprocessing_transform() -> tio.Compose:
    """
    Build the preprocessing pipeline from ``DataConfig``.
    Order: Resample -> CropOrPad -> Intensity Normalization -> Foreground masking.
    """
    dcfg: DataConfig = ConfigManager.get().get_config(ConfigManager.DATA)
    transforms = []
    if dcfg.target_spacing is not None:
        transforms.append(tio.Resample(dcfg.target_spacing))
    if dcfg.target_shape is not None:
        transforms.append(
            tio.CropOrPad(dcfg.target_shape, mask_name=dcfg.crop_mask_name)
            if dcfg.crop_mask_name
            else tio.CropOrPad(dcfg.target_shape)
        )
    if dcfg.normalization == "znorm":
        transforms.append(tio.ZNormalization(masking_method=dcfg.znorm_mask_name))
    elif dcfg.normalization == "rescale":
        transforms.append(tio.RescaleIntensity(out_min_max=(0.0, 1.0)))
    if dcfg.foreground_mask_name:
        transforms.append(ZeroOutsideMask(dcfg.foreground_mask_name, fill_value=dcfg.background_fill_value))
    if dcfg.soft_label_sigma:
        transforms.append(SoftLabel(dcfg.label_name, dcfg.num_classes, dcfg.soft_label_sigma, dcfg.soft_label_truncate))
    return tio.Compose(transforms)


def get_augmentation_transform() -> tio.Compose:
    """
    Build the augmentation pipeline from ``AugmentConfig``.

    Each transform is included only when its individual toggle is ``True``.
    When the global ``enabled`` flag is ``False`` the result is an empty
    :class:`tio.Compose` (all transforms skipped).

    Transform order: spatial (Flip -> Affine -> Elastic) then
    intensity (Noise -> Blur -> Gamma).
    """
    a: AugmentConfig = ConfigManager.get().get_config(ConfigManager.AUGMENT)
    d: DataConfig = ConfigManager.get().get_config(ConfigManager.DATA)
    transforms = []
    # intensity transforms must leave the soft label alone; the spatial ones should treat it like an image (linear interpolation)
    keep_soft = {"exclude": [soft_label_name(d.label_name)]} if d.soft_label_sigma else {}

    if a.flip:
        transforms.append(
            tio.RandomFlip(
                axes=a.flip_axes,
                flip_probability=a.flip_p,
            )
        )

    if a.affine:
        transforms.append(
            tio.RandomAffine(
                scales=a.affine_scales,
                degrees=a.affine_degrees,
                translation=a.affine_translation,
                p=a.affine_p,
            )
        )

    if a.elastic:
        transforms.append(tio.RandomElasticDeformation(p=a.elastic_p))

    if a.noise:
        transforms.append(tio.RandomNoise(std=a.noise_std, p=a.noise_p, **keep_soft))

    if a.blur:
        transforms.append(tio.RandomBlur(std=a.blur_std, p=a.blur_p, **keep_soft))

    if a.gamma:
        transforms.append(tio.RandomGamma(log_gamma=a.gamma_log_gamma, p=a.gamma_p, **keep_soft))

    return tio.Compose(transforms)


# ── Dataset / DataLoader factory ───────────────────────────────────────────────


def create_labelmap_segmentation_datasets() -> (
    Tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]
):
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
        dcfg.modalities = discover_modalities(data_root / "train", dcfg.label_name)
        if not dcfg.modalities:
            raise RuntimeError(
                f'Could not discover modalities under {data_root / "train"}. '
                "Check data_root and label_name."
            )
        print(f"[dataset] Auto-discovered modalities: {dcfg.modalities}")

    modalities = dcfg.modalities
    label_name = dcfg.label_name
    mask_names = [dcfg.znorm_mask_name, dcfg.foreground_mask_name, dcfg.crop_mask_name]

    preprocess = get_preprocessing_transform()
    augment = get_augmentation_transform()
    train_tf = tio.Compose([preprocess, augment]) if acfg.enabled else preprocess

    train_subjects = build_labelmap_segmentation_subjects(
        data_root / "train", modalities, label_name, mask_names=mask_names
    )
    return (
        tio.SubjectsDataset(train_subjects, transform=train_tf),
        tio.SubjectsDataset(
            build_labelmap_segmentation_subjects(
                data_root / "validation", modalities, label_name, mask_names=mask_names
            ),
            transform=preprocess,
        ),
        tio.SubjectsDataset(
            build_labelmap_segmentation_subjects(
                data_root / "test",
                modalities,
                label_name,
                require_label=True,
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
