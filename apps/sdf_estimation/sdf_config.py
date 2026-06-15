"""
All configuration classes for the SDF Estimation application.

Each class covers exactly one concern.  Instances are registered with
ConfigManager under the well-known type-name constants and persist to
``<exp_dir>/<type_name>.json`` at the start of every run.

Key differences from labelmap_segmentation
-------------------------------------------
- Labels are float32 signed-distance-field NIfTI files, not integer label maps.
- Multiple SDF fields can be predicted simultaneously (``sdf_names`` / ``num_sdf_fields``).
- Patch-based training is not supported (SDF estimation is a global problem).
- Intensity augmentations (Noise, Blur, Gamma) are applied to input modalities only.
- Loss is MSE + Eikonal (``|Nabla SDF| = 1`` constraint).
"""

from configuration.base import AbstractConfig, ConfigField
from configuration.model.base import (
    DecoderConfig,
    EncoderConfig,
    EncoderDecoderModelConfig,
)


class DataConfig(AbstractConfig):
    config_type = "Data"

    data_root = ConfigField(
        "", doc="Root directory containing train / validation / test splits"
    )
    modalities = ConfigField(
        None,
        doc="Input modality folder names (None = auto-detect, excluding sdf_names)",
    )
    sdf_names = ConfigField(
        None,
        doc="SDF field folder names, one subfolder per field "
        '(e.g. ["sdf_bone", "sdf_muscle"]). Must be set explicitly.',
    )
    num_sdf_fields = ConfigField(
        1, doc="Number of SDF output channels; resolved from len(sdf_names) at runtime"
    )
    target_spacing = ConfigField(
        None, doc="Resample spacing (x, y, z) in mm; None = keep original"
    )
    target_shape = ConfigField(
        None, doc="Crop/pad target shape (D, H, W); None = keep original"
    )
    normalization = ConfigField(
        "znorm",
        doc="Intensity normalisation applied to modalities only: znorm | rescale | none",
    )


class AugmentConfig(AbstractConfig):
    """
    Augmentation pipeline configuration.

    Spatial transforms (Flip, Affine, Elastic) are applied to ALL images
    (modalities + SDF fields).

    Intensity transforms (Noise, Blur, Gamma) are applied to **input modalities
    only** -- never to SDF fields, whose values encode physical distances.
    """

    config_type = "Augment"

    enabled = ConfigField(True, doc="Global switch; False disables all augmentation")

    # ── Spatial (all images) ──────────────────────────────────────────────────
    flip = ConfigField(False, doc="Enable random axis flipping")
    flip_axes = ConfigField((0, 1, 2), doc="Axes to consider for flipping")
    flip_p = ConfigField(0.3, doc="Per-axis flip probability")

    affine = ConfigField(False, doc="Enable random affine transform")
    affine_p = ConfigField(0.3, doc="Apply probability")
    affine_scales = ConfigField((1.0, 1.0), doc="Scale range (min, max)")
    affine_degrees = ConfigField(15, doc="Max rotation in degrees")
    affine_translation = ConfigField(10, doc="Max translation in mm")

    elastic = ConfigField(True, doc="Enable random elastic deformation")
    elastic_p = ConfigField(0.3, doc="Apply probability")

    # ── Intensity (modalities only) ───────────────────────────────────────────
    noise = ConfigField(True, doc="Enable additive Gaussian noise (modalities only)")
    noise_p = ConfigField(0.3, doc="Apply probability")
    noise_std = ConfigField((0.0, 0.1), doc="Noise std range (min, max)")

    blur = ConfigField(False, doc="Enable random Gaussian blur (modalities only)")
    blur_p = ConfigField(0.3, doc="Apply probability")
    blur_std = ConfigField((0.0, 1.0), doc="Blur std range (min, max)")

    gamma = ConfigField(True, doc="Enable random gamma correction (modalities only)")
    gamma_p = ConfigField(0.3, doc="Apply probability")
    gamma_log_gamma = ConfigField((-0.3, 0.3), doc="Log-gamma range (min, max)")


class UNet3DEncoderConfig(EncoderConfig):
    """
    Encoder configuration for the 3D U-Net.

    Inherits from :class:`~configuration.model.base.EncoderConfig`:
        in_channels   -- set at runtime from len(DataConfig.modalities)
        base_features -- feature channels at the first encoding level
        depth         -- number of encoding / downsampling levels
    """

    config_type = "UNet3DEncoder"


class UNet3DDecoderConfig(DecoderConfig):
    """
    Decoder configuration for the 3D U-Net.

    Inherits from :class:`~configuration.model.base.DecoderConfig`:
        trilinear -- True = trilinear upsampling, False = transposed conv
    """

    config_type = "UNet3DDecoder"


class UNet3DConfig(EncoderDecoderModelConfig):
    """
    Full model configuration for the 3D U-Net SDF estimation backbone.

    Access pattern::

        cfg = UNet3DConfig()
        cfg.encoder.base_features = 32
        cfg.encoder.in_channels   = 2     # set after modality discovery
        cfg.decoder.trilinear     = True
    """

    config_type = "Model"

    def __init__(self, file_path: str = "") -> None:
        super().__init__(
            encoder=UNet3DEncoderConfig(),
            decoder=UNet3DDecoderConfig(),
            file_path=file_path,
        )


class LossConfig(AbstractConfig):
    config_type = "Loss"

    recon_weight = ConfigField(
        1.0,
        doc="Weight of the combined Smooth-L1 reconstruction term "
        "(boosted near the zero level-set)",
    )
    eikonal_weight = ConfigField(
        0.1, doc="Weight of the Eikonal constraint (|Nabla SDF| approximate 1)"
    )
    normal_weight = ConfigField(
        0.0,
        doc="Weight of the gradient-direction (normal) consistency term "
        "(weighted near the zero level-set)",
    )
    boundary_sigma = ConfigField(
        1.0,
        doc="Gaussian width (in voxels) for the boundary weighting, "
        "shared by the reconstruction and normal terms",
    )


class OptimizerConfig(AbstractConfig):
    config_type = "Optimizer"

    type = ConfigField("adamw", doc="adam | adamw | sgd")
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
    batch_size = ConfigField(1, doc="Global-volume mode: batch_size=1 is typical")
    gradient_accumulation = ConfigField(True)
    amp = ConfigField(False, doc="Automatic mixed precision (CUDA only)")
    early_stopping = ConfigField(False)
    early_stopping_patience = ConfigField(30)
    ema = ConfigField(False, doc="Exponential moving average of model weights")
    ema_decay = ConfigField(0.99, doc="EMA decay (closer to 1 = slower shadow update)")


class InfraConfig(AbstractConfig):
    config_type = "Infra"

    output_dir = ConfigField("./output")
    experiment_name = ConfigField("sdf_estimation")
    num_workers = ConfigField(4)
    device = ConfigField("auto", doc="auto | cpu | cuda")
    seed = ConfigField(42)
    resume = ConfigField(None, doc="Checkpoint file path to resume from")
    val_interval = ConfigField(1)
    save_interval = ConfigField(10)
    log_interval = ConfigField(10)
    log_images = ConfigField(False)
    log_images_interval = ConfigField(10)
    torch_compile = ConfigField(
        False,
        doc="torch.compile(model, dynamic=False) for faster training (PyTorch >= 2.0)",
    )
