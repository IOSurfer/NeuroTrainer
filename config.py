"""
Singleton configuration manager for the 3D U-Net training framework.

Quick reference
---------------
Initialize (once, at startup):
    cfg = Config.from_args(args)          # from argparse
    cfg = Config.from_json('cfg.json')    # reproduce a past experiment

Access from anywhere (no need to pass args around):
    cfg = Config.get()
    lr  = cfg.optimizer.lr

Inspect / persist:
    cfg.print_summary()                   # formatted table to stdout
    cfg.to_json('output/exp/config.json') # save for reproduction
    s   = cfg.summary()                   # formatted string

Compare two experiments:
    delta = cfg_a.diff(cfg_b)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Parameter groups (one dataclass per concern) ───────────────────────────────

@dataclass
class DataConfig:
    data_root: str = ''
    modalities: Optional[List[str]] = None      # None → auto-detected
    label_name: str = 'label'
    num_classes: int = 2
    target_spacing: Optional[Tuple[float, float, float]] = None   # mm
    target_shape: Optional[Tuple[int, int, int]] = None           # D H W
    normalization: str = 'znorm'                # znorm | rescale | none


@dataclass
class PatchConfig:
    enabled: bool = False
    size: Tuple[int, int, int] = (128, 128, 128)
    overlap: Tuple[int, int, int] = (64, 64, 64)
    samples_per_volume: int = 4
    queue_max_length: int = 256
    weighted_sampling: bool = False             # weight by label presence


@dataclass
class AugmentConfig:
    enabled: bool = True
    elastic_deformation: bool = False


@dataclass
class ModelConfig:
    base_features: int = 32
    trilinear: bool = True                      # False → transposed conv


@dataclass
class LossConfig:
    type: str = 'dice_ce'                       # dice | dice_ce | ce
    dice_weight: float = 0.5
    ce_weight: float = 0.5


@dataclass
class OptimizerConfig:
    type: str = 'adamw'                         # adam | adamw | sgd
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0                      # 0 = disabled


@dataclass
class SchedulerConfig:
    type: str = 'cosine'                        # cosine | plateau | step | none
    warmup_epochs: int = 5
    patience: int = 10                          # ReduceLROnPlateau / StepLR step
    factor: float = 0.5                         # decay multiplier


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 2
    amp: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 30


@dataclass
class InfraConfig:
    output_dir: str = './output'
    experiment_name: str = 'unet3d'
    num_workers: int = 4
    device: str = 'auto'                        # auto | cpu | cuda
    seed: int = 42
    resume: Optional[str] = None
    val_interval: int = 1
    save_interval: int = 10
    log_interval: int = 10
    log_images: bool = False
    log_images_interval: int = 10


# ── Internal constants ─────────────────────────────────────────────────────────

_GROUPS: Tuple[str, ...] = (
    'data', 'patch', 'augment', 'model',
    'loss', 'optimizer', 'scheduler', 'training', 'infra',
)

# Tuple-typed fields that become lists in JSON and must be restored on load
_TUPLE_FIELDS: Dict[str, set] = {
    'data':  {'target_spacing', 'target_shape'},
    'patch': {'size', 'overlap'},
}


# ── Singleton ──────────────────────────────────────────────────────────────────

class Config:
    """
    Singleton configuration manager.

    All parameter groups are exposed as typed dataclass attributes so that
    IDEs can auto-complete and type-checkers can validate field access.

    Examples
    --------
    >>> cfg = Config.from_args(parsed_args)
    >>> cfg.print_summary()
    >>> cfg.to_json('output/exp_001/config.json')

    >>> # In any other module — no need to pass anything around:
    >>> from config import Config
    >>> lr = Config.get().optimizer.lr
    """

    _instance: Optional['Config'] = None

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self.data      = DataConfig()
        self.patch     = PatchConfig()
        self.augment   = AugmentConfig()
        self.model     = ModelConfig()
        self.loss      = LossConfig()
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()
        self.training  = TrainingConfig()
        self.infra     = InfraConfig()

    # ── Singleton access ───────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> 'Config':
        """Return the global singleton, creating a default instance if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton (useful in unit tests)."""
        cls._instance = None

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_args(cls, args) -> 'Config':
        """
        Create (or replace) the singleton from an :class:`argparse.Namespace`.

        This is the canonical entry-point when launching via CLI.
        """
        cfg = cls()
        cls._instance = cfg

        # data
        cfg.data.data_root      = args.data_root
        cfg.data.modalities     = args.modalities
        cfg.data.label_name     = args.label_name
        cfg.data.num_classes    = args.num_classes
        cfg.data.target_spacing = tuple(args.target_spacing) if args.target_spacing else None
        cfg.data.target_shape   = tuple(args.target_shape)   if args.target_shape   else None
        cfg.data.normalization  = args.normalization

        # patch
        cfg.patch.enabled            = args.patch_based
        cfg.patch.size               = tuple(args.patch_size)
        cfg.patch.overlap            = tuple(args.patch_overlap)
        cfg.patch.samples_per_volume = args.samples_per_volume
        cfg.patch.queue_max_length   = args.queue_max_length
        cfg.patch.weighted_sampling  = args.weighted_sampling

        # augment
        cfg.augment.enabled             = not args.no_augment
        cfg.augment.elastic_deformation = args.elastic_deformation

        # model
        cfg.model.base_features = args.base_features
        cfg.model.trilinear     = args.trilinear

        # loss
        cfg.loss.type        = args.loss
        cfg.loss.dice_weight = args.dice_weight
        cfg.loss.ce_weight   = args.ce_weight

        # optimizer
        cfg.optimizer.type         = args.optimizer
        cfg.optimizer.lr           = args.lr
        cfg.optimizer.weight_decay = args.weight_decay
        cfg.optimizer.grad_clip    = args.grad_clip

        # scheduler
        cfg.scheduler.type          = args.scheduler
        cfg.scheduler.warmup_epochs = args.warmup_epochs
        cfg.scheduler.patience      = args.scheduler_patience
        cfg.scheduler.factor        = args.scheduler_factor

        # training
        cfg.training.epochs                  = args.epochs
        cfg.training.batch_size              = args.batch_size
        cfg.training.amp                     = args.amp
        cfg.training.early_stopping          = args.early_stopping
        cfg.training.early_stopping_patience = args.early_stopping_patience

        # infra
        cfg.infra.output_dir          = args.output_dir
        cfg.infra.experiment_name     = args.experiment_name
        cfg.infra.num_workers         = args.num_workers
        cfg.infra.device              = args.device
        cfg.infra.seed                = args.seed
        cfg.infra.resume              = args.resume
        cfg.infra.val_interval        = args.val_interval
        cfg.infra.save_interval       = args.save_interval
        cfg.infra.log_interval        = args.log_interval
        cfg.infra.log_images          = args.log_images
        cfg.infra.log_images_interval = args.log_images_interval

        return cfg

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """
        Reload a previously saved config to reproduce an experiment exactly.

        Parameters
        ----------
        path : str
            Path to a JSON file produced by :meth:`to_json`.
        """
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create (or replace) the singleton from a nested dict."""
        cfg = cls()
        cls._instance = cfg

        for group_name in _GROUPS:
            group_data = data.get(group_name, {})
            if not isinstance(group_data, dict):
                continue
            group = getattr(cfg, group_name)
            tuple_fields = _TUPLE_FIELDS.get(group_name, set())

            for f in fields(group):
                if f.name not in group_data:
                    continue
                val = group_data[f.name]
                # JSON arrays → Python tuples for spatial / size fields
                if f.name in tuple_fields and isinstance(val, list) and val is not None:
                    val = tuple(val)
                setattr(group, f.name, val)

        return cfg

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialize all parameters to a JSON-ready nested dict.

        Tuples are converted to lists so the output is valid JSON.
        """
        result: dict = {}
        for group_name in _GROUPS:
            raw = asdict(getattr(self, group_name))
            result[group_name] = {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in raw.items()
            }
        return result

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """
        Serialize the config to a JSON string.

        Parameters
        ----------
        path : str, optional
            When provided, also write to this file (parent dirs are created).
        indent : int
            JSON indentation width.

        Returns
        -------
        str
            The complete JSON representation.

        Example
        -------
        >>> cfg.to_json('output/exp_001/config.json')
        >>> # Reproduce later:
        >>> cfg2 = Config.from_json('output/exp_001/config.json')
        """
        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding='utf-8')
        return text

    # ── Human-readable summary ─────────────────────────────────────────────────

    def summary(self) -> str:
        """
        Return a formatted, human-readable table of every parameter.

        Columns: parameter name (left-aligned) | current value (right side).
        Groups are separated by headers.
        """
        # Determine column widths
        key_col = max(len(f.name) for g in _GROUPS for f in fields(getattr(self, g))) + 4
        val_col = 38
        total   = key_col + val_col + 5   # 5 = "│ " + " │ " + " │"

        title  = f' 3D U-Net Config — {self.infra.experiment_name} '
        border = '═' * total

        lines: List[str] = [
            f'╔{border}╗',
            f'║{title:^{total}}║',
            f'╠{border}╣',
        ]

        for i, group_name in enumerate(_GROUPS):
            group = getattr(self, group_name)
            group_title = f' [{group_name.upper()}] '
            lines.append(f'║{group_title:<{total}}║')
            lines.append(f'║{"─" * total}║')

            for f in fields(group):
                val = getattr(group, f.name)
                key_str = f'  {f.name}'
                val_str = '—' if val is None else str(val)
                # Truncate long values
                if len(val_str) > val_col:
                    val_str = val_str[:val_col - 3] + '...'
                lines.append(f'║{key_str:<{key_col}}  {val_str:<{val_col}}║')

            if i < len(_GROUPS) - 1:
                lines.append(f'╠{"─" * total}╣')

        lines.append(f'╚{border}╝')
        return '\n'.join(lines)

    def print_summary(self) -> None:
        """Print the formatted parameter table to stdout."""
        print(self.summary())

    # ── Experiment comparison ──────────────────────────────────────────────────

    def diff(self, other: 'Config') -> Dict[str, Dict[str, Tuple[Any, Any]]]:
        """
        Compare this config against *other* and return differing fields.

        Parameters
        ----------
        other : Config
            Config to compare against (e.g., a loaded past experiment).

        Returns
        -------
        dict
            ``{group: {field: (self_value, other_value), ...}, ...}``
            Only groups and fields where values differ are included.

        Example
        -------
        >>> delta = cfg_new.diff(cfg_old)
        >>> for group, changes in delta.items():
        ...     for field, (new_val, old_val) in changes.items():
        ...         print(f'{group}.{field}: {old_val!r} → {new_val!r}')
        """
        delta: Dict[str, Dict[str, Tuple[Any, Any]]] = {}
        for group_name in _GROUPS:
            g_self  = getattr(self, group_name)
            g_other = getattr(other, group_name)
            diffs = {
                f.name: (getattr(g_self, f.name), getattr(g_other, f.name))
                for f in fields(g_self)
                if getattr(g_self, f.name) != getattr(g_other, f.name)
            }
            if diffs:
                delta[group_name] = diffs
        return delta

    def __repr__(self) -> str:
        return (
            f"Config("
            f"experiment='{self.infra.experiment_name}', "
            f"num_classes={self.data.num_classes}, "
            f"lr={self.optimizer.lr}, "
            f"epochs={self.training.epochs}"
            f")"
        )
