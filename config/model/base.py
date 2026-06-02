"""
Model configuration base classes.

ModelConfig       — base for any segmentation architecture
EncoderConfig     — base for encoder blocks
DecoderConfig     — base for decoder blocks
EncoderDecoderModelConfig — composition of encoder + decoder

Non-encoder-decoder architectures (e.g. SegResNet) sub-class
ModelConfig directly and declare their own flat fields.
"""
from __future__ import annotations

from typing import Dict

from config.base import AbstractConfig, ConfigField


class ModelConfig(AbstractConfig):
    """Base for all model configurations."""

    config_type = 'Model'

    architecture = ConfigField('', doc='Registry key that identifies this architecture')


class EncoderConfig(AbstractConfig):
    """Base for encoder (downsampling) block configurations."""

    config_type = 'Encoder'

    in_channels   = ConfigField(1,  doc='Input channels (= number of modalities)')
    base_features = ConfigField(32, doc='Feature channels at the first encoding level')
    depth         = ConfigField(4,  doc='Number of encoding / downsampling levels')


class DecoderConfig(AbstractConfig):
    """Base for decoder (upsampling) block configurations."""

    config_type = 'Decoder'

    base_features = ConfigField(32,   doc='Feature channels at the first decoding level')
    trilinear     = ConfigField(True, doc='Use trilinear upsampling; False = transposed conv')


class EncoderDecoderModelConfig(ModelConfig):
    """
    Config for architectures assembled from a separate encoder and decoder.

    The encoder and decoder are independent :class:`~config.base.AbstractConfig`
    instances exposed via :meth:`sub_configs` so they are transparently
    serialized and restored as part of the parent JSON.

    Sub-classes supply concrete encoder / decoder types in ``__init__``::

        @register_model_config('my_arch')
        class MyArchConfig(EncoderDecoderModelConfig):
            def __init__(self, file_path=''):
                super().__init__(
                    encoder=MyEncoderConfig(),
                    decoder=MyDecoderConfig(),
                    file_path=file_path,
                )
    """

    encoder_type = ConfigField('', doc='Encoder class name (informational)')
    decoder_type = ConfigField('', doc='Decoder class name (informational)')

    def __init__(
        self,
        encoder: EncoderConfig = None,
        decoder: DecoderConfig = None,
        file_path: str = '',
    ) -> None:
        super().__init__(file_path=file_path)
        self._encoder: EncoderConfig = encoder if encoder is not None else EncoderConfig()
        self._decoder: DecoderConfig = decoder if decoder is not None else DecoderConfig()

    @property
    def encoder(self) -> EncoderConfig:
        return self._encoder

    @property
    def decoder(self) -> DecoderConfig:
        return self._decoder

    def sub_configs(self) -> Dict[str, AbstractConfig]:
        return {'encoder': self._encoder, 'decoder': self._decoder}
