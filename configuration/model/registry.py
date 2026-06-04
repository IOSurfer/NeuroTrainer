"""
Model-config registry.

register_model_config  -- decorator to register a ModelConfig subclass
build_model_config     -- factory: instantiate by architecture name
available_architectures -- list all registered names
load_model_config_from_file -- load from JSON, inferring the right class
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from configuration.model.base import ModelConfig

_REGISTRY: Dict[str, Type['ModelConfig']] = {}


def register_model_config(architecture: str):
    """
    Class decorator that registers a :class:`~config.model.base.ModelConfig`
    subclass under *architecture*.

    Example
    -------
    >>> @register_model_config('unet3d')
    ... class UNet3DConfig(EncoderDecoderModelConfig):
    ...     ...
    """
    def decorator(cls):
        _REGISTRY[architecture] = cls
        return cls
    return decorator


def build_model_config(architecture: str, **kwargs) -> 'ModelConfig':
    """
    Instantiate the registered config class for *architecture*.

    Raises
    ------
    KeyError
        When *architecture* is not in the registry.
    """
    cls = _REGISTRY.get(architecture)
    if cls is None:
        raise KeyError(
            f"Unknown architecture {architecture!r}. "
            f"Available: {available_architectures()}"
        )
    return cls(**kwargs)


def available_architectures() -> List[str]:
    """Return all registered architecture names, sorted."""
    return sorted(_REGISTRY)


def load_model_config_from_file(path: str) -> 'ModelConfig':
    """
    Load a ModelConfig from a JSON file, choosing the right concrete
    class from the saved ``architecture`` field.

    Falls back to a plain :class:`~config.model.base.ModelConfig` when
    the architecture is unknown or missing.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    architecture = data.get('default', {}).get('architecture', '')

    if architecture and architecture in _REGISTRY:
        cfg: 'ModelConfig' = _REGISTRY[architecture]()
    else:
        from configuration.model.base import ModelConfig
        cfg = ModelConfig()

    cfg.from_dict(data)
    return cfg
