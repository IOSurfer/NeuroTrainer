"""
Training-specific configuration classes.

Each class is a thin AbstractConfig subclass whose fields are declared via
ConfigField descriptors.  Register instances with ConfigManager:

    manager.register(ConfigManager.DATA,      DataConfig())
    manager.register(ConfigManager.PATCH,     PatchConfig())
    manager.register(ConfigManager.AUGMENT,   AugmentConfig())
    manager.register(ConfigManager.LOSS,      LossConfig())
    manager.register(ConfigManager.OPTIMIZER, OptimizerConfig())
    manager.register(ConfigManager.SCHEDULER, SchedulerConfig())
    manager.register(ConfigManager.TRAINING,  TrainingConfig())
    manager.register(ConfigManager.INFRA,     InfraConfig())
"""
from config.base import AbstractConfig, ConfigField


class DataConfig(AbstractConfig):
    """Data paths, modality names, and preprocessing parameters."""

    config_type = 'Data'

    data_root     = ConfigField('',       doc='Root dir with train/validation/test splits')
    modalities    = ConfigField(None,     doc='Modality folder names (None = auto-detect)')
    label_name    = ConfigField('label',  doc='Segmentation mask folder name')
    num_classes   = ConfigField(2,        doc='Output classes (include background)')
    target_spacing = ConfigField(None,    doc='Resample voxel spacing (x, y, z) in mm')
    target_shape   = ConfigField(None,    doc='Crop / pad to shape (D, H, W)')
    normalization  = ConfigField('znorm', doc='znorm | rescale | none')


class PatchConfig(AbstractConfig):
    """Patch-based sampling parameters for training and inference."""

    config_type = 'Patch'

    enabled            = ConfigField(False, doc='Enable patch-based sampling')
    size               = ConfigField((128, 128, 128), doc='Patch size  (D, H, W)')
    overlap            = ConfigField((64,  64,  64),  doc='GridSampler overlap (test)')
    samples_per_volume = ConfigField(4,     doc='Patches drawn per volume per queue epoch')
    queue_max_length   = ConfigField(256,   doc='Maximum patches kept in the queue')
    weighted_sampling  = ConfigField(False, doc='Weight sampling by label voxel count')


class AugmentConfig(AbstractConfig):
    """Augmentation pipeline switches."""

    config_type = 'Augment'

    enabled             = ConfigField(True,  doc='Enable all augmentation during training')
    elastic_deformation = ConfigField(False, doc='Include RandomElasticDeformation')


class LossConfig(AbstractConfig):
    """Loss function selection and blending weights."""

    config_type = 'Loss'

    type        = ConfigField('dice_ce', doc='dice | dice_ce | ce')
    dice_weight = ConfigField(0.5,       doc='Dice term weight in combined loss')
    ce_weight   = ConfigField(0.5,       doc='Cross-entropy term weight')


class OptimizerConfig(AbstractConfig):
    """Optimizer type and hyper-parameters."""

    config_type = 'Optimizer'

    type         = ConfigField('adamw', doc='adam | adamw | sgd')
    lr           = ConfigField(1e-4,    doc='Peak learning rate')
    weight_decay = ConfigField(1e-5)
    grad_clip    = ConfigField(1.0,     doc='Max gradient L2-norm (0 = disabled)')


class SchedulerConfig(AbstractConfig):
    """Learning-rate scheduler parameters."""

    config_type = 'Scheduler'

    type          = ConfigField('cosine', doc='cosine | plateau | step | none')
    warmup_epochs = ConfigField(5,        doc='Linear LR warm-up duration')
    patience      = ConfigField(10,       doc='ReduceLROnPlateau patience / StepLR step size')
    factor        = ConfigField(0.5,      doc='LR decay multiplier')


class TrainingConfig(AbstractConfig):
    """Core training loop parameters."""

    config_type = 'Training'

    epochs                  = ConfigField(200)
    batch_size              = ConfigField(2)
    amp                     = ConfigField(False, doc='Automatic mixed precision (CUDA only)')
    early_stopping          = ConfigField(False)
    early_stopping_patience = ConfigField(30)


class InfraConfig(AbstractConfig):
    """Infrastructure: paths, device, logging intervals."""

    config_type = 'Infra'

    output_dir          = ConfigField('./output')
    experiment_name     = ConfigField('unet3d')
    num_workers         = ConfigField(4)
    device              = ConfigField('auto',  doc='auto | cpu | cuda')
    seed                = ConfigField(42)
    resume              = ConfigField(None,    doc='Checkpoint file path to resume from')
    val_interval        = ConfigField(1,       doc='Validate every N epochs')
    save_interval       = ConfigField(10,      doc='Save periodic checkpoint every N epochs')
    log_interval        = ConfigField(10,      doc='Log batch stats every N batches')
    log_images          = ConfigField(False,   doc='Write slice images to TensorBoard')
    log_images_interval = ConfigField(10)
