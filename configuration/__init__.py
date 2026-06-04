"""
config -- base configuration infrastructure.

Exports only the building blocks shared by all applications.
Application-specific config classes live in apps/<app_name>/config.py.
"""
from configuration.base import AbstractConfig, ConfigField
from configuration.manager import ConfigManager
from configuration.model.base import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    EncoderDecoderModelConfig,
)

__all__ = [
    # Primitive building blocks
    'AbstractConfig', 'ConfigField', 'ConfigManager',
    # Model-config composition pattern
    'ModelConfig', 'EncoderConfig', 'DecoderConfig', 'EncoderDecoderModelConfig',
]
