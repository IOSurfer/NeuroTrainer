"""
ConfigManager — singleton registry of named AbstractConfig instances.
Mirrors qLarmorConfigManager.

Usage
-----
Initialize (once at startup):
    manager = ConfigManager.get()
    manager.register(ConfigManager.DATA,  DataConfig())
    manager.register(ConfigManager.MODEL, UNet3DConfig())

Access anywhere:
    cfg  = ConfigManager.get().get_config(ConfigManager.DATA)
    lr   = ConfigManager.get().get_config(ConfigManager.OPTIMIZER).lr

Persist at experiment start:
    ConfigManager.get().save_all('output/exp_001')

Reproduce a past run:
    manager = ConfigManager.from_directory('output/exp_001')
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from config.base import AbstractConfig
    from config.model.base import ModelConfig


class ConfigManager:
    """
    Singleton registry.  Equivalent to qLarmorConfigManager.

    Type name constants mirror the C++ class constants so that callers
    never use raw strings:

        manager.get_config(ConfigManager.OPTIMIZER)
    """

    _instance: Optional['ConfigManager'] = None

    # Well-known config type names (mirrors C++ class constants)
    DATA      = 'Data'
    PATCH     = 'Patch'
    AUGMENT   = 'Augment'
    MODEL     = 'Model'
    LOSS      = 'Loss'
    OPTIMIZER = 'Optimizer'
    SCHEDULER = 'Scheduler'
    TRAINING  = 'Training'
    INFRA     = 'Infra'

    def __init__(self) -> None:
        self._registry: Dict[str, 'AbstractConfig'] = {}

    # ── Singleton ────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> 'ConfigManager':
        """Return the global singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful in unit tests)."""
        cls._instance = None

    # ── Registration (mirrors registerConfig / getConfig) ────────────────────

    def register(self, type_name: str, config: 'AbstractConfig') -> 'ConfigManager':
        """
        Register *config* under *type_name*.

        Subsequent calls with the same name overwrite the previous entry.
        Returns *self* for method chaining::

            manager.register(ConfigManager.DATA, DataConfig()) \\
                   .register(ConfigManager.MODEL, UNet3DConfig())
        """
        self._registry[type_name] = config
        return self

    def get_config(self, type_name: str) -> Optional['AbstractConfig']:
        """Return the registered config for *type_name*, or ``None``."""
        return self._registry.get(type_name)

    def get_model_config(self) -> Optional['ModelConfig']:
        """Type-safe convenience accessor for the model config."""
        return self._registry.get(self.MODEL)  # type: ignore[return-value]

    def registered_types(self) -> List[str]:
        """All registered type names in insertion order."""
        return list(self._registry)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_all(self, directory: str) -> None:
        """
        Save every registered config to ``<directory>/<type_name>.json``.

        Call this once at the start of training so the experiment is
        reproducible even if it crashes before completion.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        for type_name, config in self._registry.items():
            config.save(str(dir_path / f'{type_name.lower()}.json'))

    def load_all(self, directory: str) -> None:
        """
        Load every registered config from ``<directory>/<type_name>.json``.
        Missing files are silently ignored.
        """
        dir_path = Path(directory)
        for type_name, config in self._registry.items():
            path = dir_path / f'{type_name.lower()}.json'
            if path.exists():
                config.load(str(path))

    @classmethod
    def from_directory(cls, directory: str) -> 'ConfigManager':
        """
        Reconstruct a full ConfigManager from a saved experiment directory.

        The model config class is inferred from its saved ``architecture``
        field so the correct encoder / decoder sub-configs are restored.

        Example
        -------
        >>> manager = ConfigManager.from_directory('output/exp_001')
        >>> manager.print_all()
        """
        from config.training import (
            DataConfig, PatchConfig, AugmentConfig, LossConfig,
            OptimizerConfig, SchedulerConfig, TrainingConfig, InfraConfig,
        )
        from config.model.registry import load_model_config_from_file

        manager = cls()
        cls._instance = manager

        _type_map: Dict[str, Type['AbstractConfig']] = {
            cls.DATA:      DataConfig,
            cls.PATCH:     PatchConfig,
            cls.AUGMENT:   AugmentConfig,
            cls.LOSS:      LossConfig,
            cls.OPTIMIZER: OptimizerConfig,
            cls.SCHEDULER: SchedulerConfig,
            cls.TRAINING:  TrainingConfig,
            cls.INFRA:     InfraConfig,
        }
        dir_path = Path(directory)
        for type_name, cfg_cls in _type_map.items():
            cfg = cfg_cls()
            cfg.load(str(dir_path / f'{type_name.lower()}.json'))
            manager.register(type_name, cfg)

        # Model config — architecture-aware factory load
        model_path = dir_path / 'model.json'
        if model_path.exists():
            manager.register(cls.MODEL, load_model_config_from_file(str(model_path)))

        return manager

    # ── Display (mirrors printAll) ────────────────────────────────────────────

    def print_all(self) -> None:
        """Print a formatted summary of every registered config."""
        infra = self.get_config(self.INFRA)
        exp   = getattr(infra, 'experiment_name', '') if infra else ''

        width = 66
        title = f' ConfigManager — {exp} ' if exp else ' ConfigManager '
        thick = '═' * width

        print(f'╔{thick}╗')
        print(f'║{title:^{width}}║')
        print(f'╚{thick}╝')

        for type_name, config in self._registry.items():
            print(f'\n  ▶  {type_name}')
            for line in config.summary().splitlines():
                print(f'    {line}')
        print()

    def __repr__(self) -> str:
        return f'ConfigManager(registered={list(self._registry)})'
