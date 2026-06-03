"""
ConfigManager — singleton registry of named AbstractConfig instances.
Mirrors qLarmorConfigManager.

This module is pure infrastructure: it knows nothing about specific config
types (DataConfig, ModelConfig, etc.).  Those live in each application under
apps/<app_name>/config.py.

Usage
-----
Register configs (once, at startup — done in the app's train.py):
    manager = ConfigManager.get()
    manager.register(ConfigManager.DATA,  MyDataConfig())
    manager.register(ConfigManager.MODEL, MyModelConfig())

Access from anywhere:
    cfg = ConfigManager.get().get_config(ConfigManager.DATA)

Persist at experiment start:
    ConfigManager.get().save_all('output/exp_001')
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.base import AbstractConfig


class ConfigManager:
    """
    Singleton registry mapping type-name strings to AbstractConfig instances.

    Well-known type-name constants are provided for convenience; apps may
    also use their own arbitrary string keys.
    """

    _instance: Optional['ConfigManager'] = None

    # Well-known type-name constants (plain strings — not tied to any class)
    DATA = 'Data'
    AUGMENT = 'Augment'
    PATCH = 'Patch'
    MODEL = 'Model'
    LOSS = 'Loss'
    OPTIMIZER = 'Optimizer'
    SCHEDULER = 'Scheduler'
    TRAINING = 'Training'
    INFRA = 'Infra'

    def __init__(self) -> None:
        self._registry: Dict[str, 'AbstractConfig'] = {}

    @classmethod
    def get(cls) -> 'ConfigManager':
        """Return the global singleton, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful in tests)."""
        cls._instance = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, type_name: str, config: 'AbstractConfig') -> 'ConfigManager':
        """
        Register *config* under *type_name*.  Chaining is supported:
            manager.register('Data', dc).register('Model', mc)
        """
        self._registry[type_name] = config
        return self

    def get_config(self, type_name: str) -> Optional['AbstractConfig']:
        """Return the registered config for *type_name*, or ``None``."""
        return self._registry.get(type_name)

    def registered_types(self) -> List[str]:
        """Return all registered type names in registration order."""
        return list(self._registry)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_all(self, directory: str) -> None:
        """Write every registered config to ``<directory>/<type_name>.json``."""
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

    # ── Display ───────────────────────────────────────────────────────────────

    def print_all(self) -> None:
        """Print a formatted summary of all registered configs to stdout."""
        width = 64
        thick = '═' * width
        print(f'╔{thick}╗')
        print(f'║{"  ConfigManager":^{width}}║')
        print(f'╚{thick}╝')
        for type_name, config in self._registry.items():
            print(f'\n  ▶ {type_name}')
            for line in config.summary().splitlines():
                print(f'    {line}')
        print()

    def __repr__(self) -> str:
        return f'ConfigManager(registered={list(self._registry)})'
