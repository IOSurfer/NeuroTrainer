"""
UNet3D configuration — encoder + decoder composition.

Architecture tree:
    UNet3DConfig (EncoderDecoderModelConfig)
    ├── encoder: UNet3DEncoderConfig
    └── decoder: UNet3DDecoderConfig

Adding a new architecture is as simple as:

    @register_model_config('my_arch')
    class MyArchConfig(EncoderDecoderModelConfig):   # or ModelConfig directly
        def __init__(self, file_path=''):
            super().__init__(
                encoder=MyEncoderConfig(),
                decoder=MyDecoderConfig(),
                file_path=file_path,
            )
            self._set('architecture', 'my_arch')
"""
from config.base import ConfigField
from config.model.base import DecoderConfig, EncoderConfig, EncoderDecoderModelConfig
from config.model.registry import register_model_config


class UNet3DEncoderConfig(EncoderConfig):
    """
    Encoder config for the 3D U-Net.

    Inherits from :class:`~config.model.base.EncoderConfig`:
        in_channels, base_features, depth
    """
    config_type = 'UNet3DEncoder'


class UNet3DDecoderConfig(DecoderConfig):
    """
    Decoder config for the 3D U-Net.

    Inherits from :class:`~config.model.base.DecoderConfig`:
        base_features, trilinear
    """
    config_type = 'UNet3DDecoder'


@register_model_config('unet3d')
class UNet3DConfig(EncoderDecoderModelConfig):
    """
    Full configuration for the 3D U-Net architecture.

    Example
    -------
    >>> cfg = UNet3DConfig()
    >>> cfg.encoder.in_channels  = 2     # two MRI modalities
    >>> cfg.encoder.base_features = 32
    >>> cfg.decoder.trilinear    = True
    >>> cfg.num_classes          = 3
    """

    num_classes = ConfigField(2, doc='Number of segmentation output classes')

    def __init__(self, file_path: str = '') -> None:
        super().__init__(
            encoder=UNet3DEncoderConfig(),
            decoder=UNet3DDecoderConfig(),
            file_path=file_path,
        )
        # Set fixed fields without triggering save (no file path yet)
        with self._lock:
            self._props['default']['architecture']  = 'unet3d'
            self._props['default']['encoder_type']  = 'UNet3DEncoderConfig'
            self._props['default']['decoder_type']  = 'UNet3DDecoderConfig'
