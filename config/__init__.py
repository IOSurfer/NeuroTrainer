"""
config — unified configuration management package.

Public API
----------
Core:
    AbstractConfig, ConfigField   — base class and field descriptor
    ConfigManager                 — singleton registry (mirrors qLarmorConfigManager)

Training configs (register with ConfigManager.DATA/PATCH/… keys):
    DataConfig, PatchConfig, AugmentConfig, LossConfig,
    OptimizerConfig, SchedulerConfig, TrainingConfig, InfraConfig

Model configs (register with ConfigManager.MODEL):
    ModelConfig, EncoderConfig, DecoderConfig, EncoderDecoderModelConfig
    UNet3DConfig, UNet3DEncoderConfig, UNet3DDecoderConfig

Model registry helpers:
    register_model_config(architecture)  — decorator
    build_model_config(architecture)     — factory
    available_architectures()            — list all registered names
"""

from config.base import AbstractConfig, ConfigField
from config.manager import ConfigManager
from config.training import (
    DataConfig,
    PatchConfig,
    AugmentConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    InfraConfig,
)
from config.model import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    EncoderDecoderModelConfig,
    UNet3DConfig,
    UNet3DEncoderConfig,
    UNet3DDecoderConfig,
    register_model_config,
    build_model_config,
    available_architectures,
    load_model_config_from_file,
)

__all__ = [
    # Core
    'AbstractConfig', 'ConfigField', 'ConfigManager',
    # Training
    'DataConfig', 'PatchConfig', 'AugmentConfig', 'LossConfig',
    'OptimizerConfig', 'SchedulerConfig', 'TrainingConfig', 'InfraConfig',
    # Model
    'ModelConfig', 'EncoderConfig', 'DecoderConfig', 'EncoderDecoderModelConfig',
    'UNet3DConfig', 'UNet3DEncoderConfig', 'UNet3DDecoderConfig',
    # Registry
    'register_model_config', 'build_model_config',
    'available_architectures', 'load_model_config_from_file',
]
