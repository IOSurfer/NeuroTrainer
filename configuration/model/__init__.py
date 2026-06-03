# Architecture-agnostic model config building blocks.
# App-specific classes (UNet3DConfig, etc.) live in apps/<app>/config.py.
from configuration.model.base import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    EncoderDecoderModelConfig,
)

__all__ = [
    'ModelConfig',
    'EncoderConfig',
    'DecoderConfig',
    'EncoderDecoderModelConfig',
]
