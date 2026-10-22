"""
All configuration classes for the LabelMap Segmentation application.

Each class covers exactly one concern.  Instances are registered with
ConfigManager under the well-known type-name constants and persist to
``<exp_dir>/<type_name>.json`` at the start of every run.

Model config hierarchy
----------------------
UNet3DConfig (EncoderDecoderModelConfig)
├── encoder: UNet3DEncoderConfig   in_channels, base_features, depth
└── decoder: UNet3DDecoderConfig   trilinear
"""

from configuration.base import AbstractConfig, ConfigField
from configuration.model.base import (
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderModelConfig,
)


class DataConfig(AbstractConfig):
    config_type = "Data"

    data_root = ConfigField("", doc="Root dir with train/validation/test splits")
    modalities = ConfigField(None, doc="Modality folder names (None = auto-detect)")
    label_name = ConfigField("label", doc="Segmentation mask folder name")
    num_classes = ConfigField(2, doc="Output classes (background included)")
    target_spacing = ConfigField(None, doc="Resample spacing (x, y, z) in mm")
    target_shape = ConfigField(None, doc="Crop/pad target shape (D, H, W)")
    crop_mask_name = ConfigField(
        None,
        doc="Optional labelmap folder name; when set together with target_shape the "
        "volume is cropped (or padded) to a target_shape window centred on the "
        "bounding box of this labelmap's non-zero voxels, instead of the image centre. "
        "May equal znorm_mask_name / foreground_mask_name.",
    )
    normalization = ConfigField("znorm", doc="znorm | rescale | none")
    soft_label_sigma = ConfigField(
        None,
        doc="Soft labels: standard deviation in mm of the Gaussian that smooths the one-hot label (a single value, or one value per "
        "axis). The smoothed multi-channel label is the training target (the hard label is still used for the Dice / IoU metrics), and it "
        "is interpolated linearly -- not nearest -- by the spatial augmentations. None or 0 = hard labels.",
    )
    soft_label_truncate = ConfigField(
        4.0, doc="Soft labels: Gaussian kernel radius in units of sigma (the kernel is cut off beyond truncate * sigma)"
    )
    znorm_mask_name = ConfigField(
        None,
        doc="Optional labelmap folder name; its non-zero voxels define the "
        "ZNormalization mask (None = normalize over the whole volume). "
        "May be set to the same value as label_name to reuse the "
        "segmentation mask.",
    )
    foreground_mask_name = ConfigField(
        None,
        doc="Optional labelmap folder name; applied after normalization, "
        "intensity values are kept where this labelmap is non-zero and "
        "set to background_fill_value elsewhere (None = disabled). May be "
        "set to the same value as label_name or znorm_mask_name to reuse "
        "that mask.",
    )
    background_fill_value = ConfigField(
        0.0,
        doc="Fill value written to voxels outside the foreground mask "
        "(only used when foreground_mask_name is set).",
    )


class PatchConfig(AbstractConfig):
    config_type = "Patch"

    enabled = ConfigField(False, doc="Enable patch-based sampling")
    size = ConfigField((128, 128, 128), doc="Patch size (D, H, W)")
    overlap = ConfigField((64, 64, 64), doc="GridSampler overlap (inference)")
    samples_per_volume = ConfigField(4, doc="Patches per volume per queue epoch")
    queue_max_length = ConfigField(256, doc="Max patches kept in the queue")
    weighted_sampling = ConfigField(False, doc="Weight sampling by label density")


class AugmentConfig(AbstractConfig):
    """
    Augmentation pipeline configuration.

    Master switch ``enabled`` overrides every individual toggle.
    Each transform has its own on/off flag and parameter fields so that
    any combination can be expressed in the saved JSON without rerunning
    from the CLI.

    Naming convention
    -----------------
    <transform>          bool  -- whether the transform is active
    <transform>_p        float -- apply probability (0 = never, 1 = always)
    <transform>_<param>  any   -- transform-specific parameter
    """

    config_type = "Augment"

    # Master switch -- overrides all individual toggles when False
    enabled = ConfigField(True, doc="Global switch; False disables all augmentation")

    # ── Random Flip ───────────────────────────────────────────────────────────
    flip = ConfigField(False, doc="Enable random axis flipping")
    flip_axes = ConfigField((0, 1, 2), doc="Axes to consider for flipping")
    flip_p = ConfigField(0.3, doc="Per-axis flip probability")

    # ── Random Affine ─────────────────────────────────────────────────────────
    affine = ConfigField(False, doc="Enable random affine transform")
    affine_p = ConfigField(0.3, doc="Apply probability")
    affine_scales = ConfigField((1.0, 1.0), doc="Scale range (min, max)")
    affine_degrees = ConfigField(15, doc="Max rotation in degrees")
    affine_translation = ConfigField(10, doc="Max translation in mm")

    # ── Random Elastic Deformation ────────────────────────────────────────────
    elastic = ConfigField(True, doc="Enable random elastic deformation")
    elastic_p = ConfigField(0.3, doc="Apply probability")

    # ── Random Noise ──────────────────────────────────────────────────────────
    noise = ConfigField(True, doc="Enable additive Gaussian noise")
    noise_p = ConfigField(0.3, doc="Apply probability")
    noise_std = ConfigField((0.0, 0.1), doc="Noise std range (min, max)")

    # ── Random Blur ───────────────────────────────────────────────────────────
    blur = ConfigField(False, doc="Enable random Gaussian blur")
    blur_p = ConfigField(0.3, doc="Apply probability")
    blur_std = ConfigField((0.0, 1.0), doc="Blur std range (min, max)")

    # ── Random Gamma ──────────────────────────────────────────────────────────
    gamma = ConfigField(True, doc="Enable random gamma correction")
    gamma_p = ConfigField(0.3, doc="Apply probability")
    gamma_log_gamma = ConfigField((-0.3, 0.3), doc="Log-gamma range (min, max)")


class UNet3DEncoderConfig(EncoderConfig):
    """
    Encoder configuration for the 3D U-Net.

    Inherits from :class:`~config.model.base.EncoderConfig`:
        in_channels   -- set at runtime from len(DataConfig.modalities)
        base_features -- feature channels at the first encoding level
        depth         -- number of encoding / downsampling levels
    """

    config_type = "UNet3DEncoder"


class UNet3DDecoderConfig(DecoderConfig):
    """
    Decoder configuration for the 3D U-Net.

    Inherits from :class:`~config.model.base.DecoderConfig`:
        trilinear -- True = trilinear upsampling, False = transposed conv
    """

    config_type = "UNet3DDecoder"


class UNet3DConfig(EncoderDecoderModelConfig):
    """
    Full model configuration for the 3D U-Net architecture.

    Sub-configs are serialized / restored automatically via
    :meth:`~config.base.AbstractConfig.sub_configs`.

    Access pattern::

        cfg = UNet3DConfig()
        cfg.encoder.base_features      = 32
        cfg.encoder.in_channels        = 2    # set after modality discovery
        cfg.decoder.trilinear          = True
        cfg.num_supervision_levels     = 1    # >1 enables deep supervision
    """

    config_type = "Model"

    encoder_type = ConfigField("conv3d", doc="conv3d | res | convnext")
    decoder_type = ConfigField("conv3d", doc="conv3d | res | effidec")

    num_supervision_levels = ConfigField(
        1, doc="1 = single output; >1 = deep supervision with auxiliary heads"
    )

    def __init__(self, file_path: str = "") -> None:
        super().__init__(
            encoder=UNet3DEncoderConfig(),
            decoder=UNet3DDecoderConfig(),
            file_path=file_path,
        )


class LossConfig(AbstractConfig):
    config_type = "Loss"

    type = ConfigField("dice_ce", doc="dice | dice_ce | ce | focal | dice_focal")
    dice_weight = ConfigField(0.5, doc="Dice term weight in dice_ce / dice_focal")
    ce_weight = ConfigField(0.5, doc="Cross-entropy term weight in dice_ce")
    focal_gamma = ConfigField(2.0, doc="Focal loss focusing parameter γ (focal / dice_focal only)")
    focal_weight = ConfigField(0.5, doc="Focal term weight in dice_focal")
    focal_ignore_background = ConfigField(
        True,
        doc="True: the focal term only looks at foreground voxels (and so does Dice), nothing supervises the background output -- fine when "
        "fine-tuning a model whose head already handles the background, but a randomly initialised head then predicts arbitrary classes "
        "for the background. False: the focal term also covers the background voxels (normalised by the foreground voxel count).",
    )
    class_weights = ConfigField(
        None,
        doc="Per-class loss weights as a list [w0, w1, ...] in class-index order "
        "(None = uniform). Applied to ce, focal, and dice_focal.",
    )


class OptimizerConfig(AbstractConfig):
    config_type = "Optimizer"

    type = ConfigField("adamw", doc="adam | adamw | muon | sgd")
    lr = ConfigField(1e-4, doc="Peak learning rate")
    weight_decay = ConfigField(1e-5)
    grad_clip = ConfigField(1.0, doc="Max gradient L2-norm (0 = disabled)")


class SchedulerConfig(AbstractConfig):
    config_type = "Scheduler"

    type = ConfigField("cosine", doc="cosine | plateau | step | none")
    warmup_epochs = ConfigField(30, doc="Linear LR warm-up epochs")
    patience = ConfigField(10, doc="ReduceLROnPlateau patience / StepLR step")
    factor = ConfigField(0.5, doc="LR decay multiplier")


class TrainingConfig(AbstractConfig):
    config_type = "Training"

    epochs = ConfigField(200)
    batch_size = ConfigField(2)
    gradient_accumulation = ConfigField(True)
    amp = ConfigField(False, doc="Automatic mixed precision (CUDA only)")
    early_stopping = ConfigField(False)
    early_stopping_patience = ConfigField(30)
    ema = ConfigField(False, doc="Exponential moving average of model weights")
    ema_decay = ConfigField(0.99, doc="EMA decay (closer to 1 = slower shadow update)")
    gradient_checkpointing = ConfigField(
        False,
        doc="Recompute the activations of the conv3d encoder / decoder blocks in the backward pass instead of storing them: "
        "~30% slower, far less memory (needed to train the full-resolution decoder on 182x218x182 volumes on a 16 GB GPU)",
    )


class InfraConfig(AbstractConfig):
    config_type = "Infra"

    output_dir = ConfigField("./output")
    experiment_name = ConfigField("labelmap_seg")
    num_workers = ConfigField(4)
    prefetch_factor = ConfigField(
        2,
        doc="Batches each training DataLoader worker prepares ahead. The loader hands out batches in order, so one slow sample (e.g. a 25 s "
        "elastic deformation of a multi-channel soft label) stalls the GPU unless the buffer is deeper than that delay; raise it for that case "
        "(a prefetched batch is ~140 MB of shared memory)",
    )
    device = ConfigField("auto", doc="auto | cpu | cuda")
    seed = ConfigField(42)
    resume = ConfigField(None, doc="Checkpoint file path to resume from")
    resume_state = ConfigField(
        True,
        doc="When resuming, also restore optimizer / scheduler / scaler / EMA / epoch "
        "state. Set False for transfer-learning or fine-tuning from a pre-trained "
        "checkpoint where only model weights should be carried over.",
    )
    partial_load = ConfigField(
        "full",
        doc="Which model components to load from the checkpoint: "
        "full | encoder | encoder_decoder. "
        "'encoder' loads encoder + bottleneck only (head is re-initialised). "
        "'encoder_decoder' loads encoder + bottleneck + decoder (head is re-initialised). "
        "Optimizer / scheduler / EMA state is always skipped when partial_load != 'full'.",
    )
    val_interval = ConfigField(1)
    save_interval = ConfigField(10)
    log_interval = ConfigField(10)
    log_images = ConfigField(False)
    log_images_interval = ConfigField(10)
    torch_compile = ConfigField(
        False,
        doc="torch.compile(model, dynamic=False) for faster training (PyTorch >= 2.0)",
    )
