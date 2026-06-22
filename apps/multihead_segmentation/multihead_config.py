"""
Configuration classes specific to the Multi-Head Segmentation application.

Only ``DataConfig`` and ``LossConfig`` differ from
``apps.labelmap_segmentation.segmentation_config`` -- every other config
(``PatchConfig``, ``AugmentConfig``, ``UNet3DEncoderConfig``,
``UNet3DDecoderConfig``, ``UNet3DConfig``, ``OptimizerConfig``,
``SchedulerConfig``, ``TrainingConfig``, ``InfraConfig``) has no
task-specific coupling and is imported directly from there.

Key differences from labelmap_segmentation
-------------------------------------------
- A single label_name + num_classes is replaced by ``tasks``: a mapping of
  label-folder name -> num_classes, one entry per independent segmentation
  head (e.g. ``{"organs": 14, "tumor": 2}``).
- ``LossConfig`` gains ``task_weights`` to control each task's contribution
  to the combined loss.
"""

from configuration.base import AbstractConfig, ConfigField


class DataConfig(AbstractConfig):
    config_type = "Data"

    data_root = ConfigField("", doc="Root dir with train/validation/test splits")
    modalities = ConfigField(None, doc="Modality folder names (None = auto-detect)")
    tasks = ConfigField(
        None,
        doc="Mapping of task name (label folder name) -> num_classes "
        '(background included), e.g. {"organs": 14, "tumor": 2}. Must be set '
        "explicitly -- one independent segmentation head is created per task.",
    )
    target_spacing = ConfigField(None, doc="Resample spacing (x, y, z) in mm")
    target_shape = ConfigField(None, doc="Crop/pad target shape (D, H, W)")
    normalization = ConfigField("znorm", doc="znorm | rescale | none")
    znorm_mask_name = ConfigField(
        None,
        doc="Optional labelmap folder name; its non-zero voxels define the "
        "ZNormalization mask (None = normalize over the whole volume). "
        "May be set to one of the task names to reuse that task's mask.",
    )
    foreground_mask_name = ConfigField(
        None,
        doc="Optional labelmap folder name; applied after normalization, "
        "intensity values are kept where this labelmap is non-zero and "
        "set to 0 elsewhere (None = disabled). May be set to one of the "
        "task names or znorm_mask_name to reuse that mask.",
    )


class LossConfig(AbstractConfig):
    config_type = "Loss"

    type = ConfigField("dice_ce", doc="dice | dice_ce | ce -- shared by every task head")
    dice_weight = ConfigField(0.5, doc="Dice term weight in combined loss")
    ce_weight = ConfigField(0.5, doc="Cross-entropy term weight")
    task_weights = ConfigField(
        None,
        doc="Optional mapping task_name -> weight in the cross-task loss sum "
        "(None = equal weight 1.0 for every task).",
    )
