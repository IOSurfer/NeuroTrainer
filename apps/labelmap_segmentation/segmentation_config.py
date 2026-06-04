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
from configuration.model.base import DecoderConfig, EncoderConfig, EncoderDecoderModelConfig


class DataConfig(AbstractConfig):
    config_type = 'Data'

    data_root = ConfigField(
        '',       doc='Root dir with train/validation/test splits')
    modalities = ConfigField(
        None,     doc='Modality folder names (None = auto-detect)')
    label_name = ConfigField('label',  doc='Segmentation mask folder name')
    num_classes = ConfigField(
        2,        doc='Output classes (background included)')
    target_spacing = ConfigField(
        None,     doc='Resample spacing (x, y, z) in mm')
    target_shape = ConfigField(None,     doc='Crop/pad target shape (D, H, W)')
    normalization = ConfigField('znorm',  doc='znorm | rescale | none')


class PatchConfig(AbstractConfig):
    config_type = 'Patch'

    enabled = ConfigField(False,          doc='Enable patch-based sampling')
    size = ConfigField((128, 128, 128), doc='Patch size (D, H, W)')
    overlap = ConfigField(
        (64,  64,  64),  doc='GridSampler overlap (inference)')
    samples_per_volume = ConfigField(
        4,               doc='Patches per volume per queue epoch')
    queue_max_length = ConfigField(
        256,             doc='Max patches kept in the queue')
    weighted_sampling = ConfigField(
        False,           doc='Weight sampling by label density')


class AugmentConfig(AbstractConfig):
    """
    Augmentation pipeline configuration.

    Master switch ``enabled`` overrides every individual toggle.
    Each transform has its own on/off flag and parameter fields so that
    any combination can be expressed in the saved JSON without rerunning
    from the CLI.

    Naming convention
    -----------------
    <transform>          bool  — whether the transform is active
    <transform>_p        float — apply probability (0 = never, 1 = always)
    <transform>_<param>  any   — transform-specific parameter
    """

    config_type = 'Augment'

    # Master switch — overrides all individual toggles when False
    enabled = ConfigField(
        True, doc='Global switch; False disables all augmentation')

    # ── Random Flip ───────────────────────────────────────────────────────────
    flip = ConfigField(False,      doc='Enable random axis flipping')
    flip_axes = ConfigField((0, 1, 2), doc='Axes to consider for flipping')
    flip_p = ConfigField(0.3,       doc='Per-axis flip probability')

    # ── Random Affine ─────────────────────────────────────────────────────────
    affine = ConfigField(False,       doc='Enable random affine transform')
    affine_p = ConfigField(0.3,        doc='Apply probability')
    affine_scales = ConfigField((1.0, 1.0), doc='Scale range (min, max)')
    affine_degrees = ConfigField(15,          doc='Max rotation in degrees')
    affine_translation = ConfigField(10,          doc='Max translation in mm')

    # ── Random Elastic Deformation ────────────────────────────────────────────
    elastic = ConfigField(True, doc='Enable random elastic deformation')
    elastic_p = ConfigField(0.3,   doc='Apply probability')

    # ── Random Noise ──────────────────────────────────────────────────────────
    noise = ConfigField(True,       doc='Enable additive Gaussian noise')
    noise_p = ConfigField(0.3,        doc='Apply probability')
    noise_std = ConfigField((0.0, 0.1), doc='Noise std range (min, max)')

    # ── Random Blur ───────────────────────────────────────────────────────────
    blur = ConfigField(False,       doc='Enable random Gaussian blur')
    blur_p = ConfigField(0.3,        doc='Apply probability')
    blur_std = ConfigField((0.0, 1.0), doc='Blur std range (min, max)')

    # ── Random Gamma ──────────────────────────────────────────────────────────
    gamma = ConfigField(True,        doc='Enable random gamma correction')
    gamma_p = ConfigField(0.3,         doc='Apply probability')
    gamma_log_gamma = ConfigField(
        (-0.3, 0.3), doc='Log-gamma range (min, max)')


class UNet3DEncoderConfig(EncoderConfig):
    """
    Encoder configuration for the 3D U-Net.

    Inherits from :class:`~config.model.base.EncoderConfig`:
        in_channels   — set at runtime from len(DataConfig.modalities)
        base_features — feature channels at the first encoding level
        depth         — number of encoding / downsampling levels
    """

    config_type = 'UNet3DEncoder'


class UNet3DDecoderConfig(DecoderConfig):
    """
    Decoder configuration for the 3D U-Net.

    Inherits from :class:`~config.model.base.DecoderConfig`:
        trilinear — True = trilinear upsampling, False = transposed conv
    """

    config_type = 'UNet3DDecoder'


class UNet3DConfig(EncoderDecoderModelConfig):
    """
    Full model configuration for the 3D U-Net architecture.

    Sub-configs are serialized / restored automatically via
    :meth:`~config.base.AbstractConfig.sub_configs`.

    Access pattern::

        cfg = UNet3DConfig()
        cfg.encoder.base_features = 32
        cfg.encoder.in_channels   = 2    # set after modality discovery
        cfg.decoder.trilinear     = True
    """

    config_type = 'Model'

    def __init__(self, file_path: str = '') -> None:
        super().__init__(
            encoder=UNet3DEncoderConfig(),
            decoder=UNet3DDecoderConfig(),
            file_path=file_path,
        )


class LossConfig(AbstractConfig):
    config_type = 'Loss'

    type = ConfigField('dice_ce', doc='dice | dice_ce | ce')
    dice_weight = ConfigField(
        0.5,       doc='Dice term weight in combined loss')
    ce_weight = ConfigField(0.5,       doc='Cross-entropy term weight')


class OptimizerConfig(AbstractConfig):
    config_type = 'Optimizer'

    type = ConfigField('adamw', doc='adam | adamw | muon | sgd')
    lr = ConfigField(1e-4,    doc='Peak learning rate')
    weight_decay = ConfigField(1e-5)
    grad_clip = ConfigField(1.0,     doc='Max gradient L2-norm (0 = disabled)')


class SchedulerConfig(AbstractConfig):
    config_type = 'Scheduler'

    type = ConfigField('cosine', doc='cosine | plateau | step | none')
    warmup_epochs = ConfigField(5,        doc='Linear LR warm-up epochs')
    patience = ConfigField(
        10,       doc='ReduceLROnPlateau patience / StepLR step')
    factor = ConfigField(0.5,      doc='LR decay multiplier')


class TrainingConfig(AbstractConfig):
    config_type = 'Training'

    epochs = ConfigField(200)
    batch_size = ConfigField(2)
    amp = ConfigField(False, doc='Automatic mixed precision (CUDA only)')
    early_stopping = ConfigField(False)
    early_stopping_patience = ConfigField(30)


class InfraConfig(AbstractConfig):
    config_type = 'Infra'

    output_dir = ConfigField('./output')
    experiment_name = ConfigField('labelmap_seg')
    num_workers = ConfigField(4)
    device = ConfigField('auto',  doc='auto | cpu | cuda')
    seed = ConfigField(42)
    resume = ConfigField(None,    doc='Checkpoint file path to resume from')
    val_interval = ConfigField(1)
    save_interval = ConfigField(10)
    log_interval = ConfigField(10)
    log_images = ConfigField(False)
    log_images_interval = ConfigField(10)
