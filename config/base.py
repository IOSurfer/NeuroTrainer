"""
AbstractConfig and ConfigField — base building blocks for all configuration classes.

Design mirrors qLarmorAbstractConfig:
  - _props[group][key] flat storage per group
  - Dot-path access for nested values (get_value / set_value)
  - JSON load / save
  - Observer callbacks  (on_change)
  - Batch mode         (begin_batch / end_batch)
  - Sub-config composition via sub_configs()

ConfigField is the Python equivalent of larmorConfigSetMacro / larmorConfigGetMacro.
Declare parameters as class-level descriptors:

    class MyConfig(AbstractConfig):
        config_type = 'MyConfig'
        lr     = ConfigField(1e-4,  doc='Learning rate')
        epochs = ConfigField(200)
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_MISSING = object()   # sentinel meaning "key not present"


# ── Field descriptor ───────────────────────────────────────────────────────────

class ConfigField:
    """
    Descriptor for a typed configuration parameter.

    Equivalent to the C++ larmorConfigSetMacro / larmorConfigGetMacro pair.
    Values are stored inside the owning :class:`AbstractConfig`'s backing
    store so that changes are automatically notified and persisted.
    """

    def __init__(self, default: Any = None, *, doc: str = '') -> None:
        self.default = default
        self.doc     = doc
        self.name    = ''        # populated by __set_name__

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: 'AbstractConfig', objtype=None) -> Any:
        if obj is None:
            return self
        return obj._get(self.name, self.default)

    def __set__(self, obj: 'AbstractConfig', value: Any) -> None:
        obj._set(self.name, value)

    def __repr__(self) -> str:
        return f'ConfigField(default={self.default!r}, name={self.name!r})'


# ── Abstract base ──────────────────────────────────────────────────────────────

class AbstractConfig:
    """
    Base configuration with dict-backed storage, JSON I/O, and change
    notification. Mirrors qLarmorAbstractConfig.

    Storage layout
    --------------
    ``_props[group][key] = value`` — each group is an independent flat
    key-value namespace.  The active group is ``'default'`` unless
    changed with :meth:`set_group`.

    Sub-config composition
    ----------------------
    Override :meth:`sub_configs` to expose child :class:`AbstractConfig`
    instances (e.g. encoder / decoder).  They are transparently included
    in :meth:`to_dict` / :meth:`from_dict` under ``__<name>__`` keys.
    """

    config_type: str = ''

    def __init__(self, file_path: str = '') -> None:
        self._file_path: Optional[Path] = Path(file_path) if file_path else None
        self._group     = 'default'
        self._props: Dict[str, Dict[str, Any]] = {'default': {}}
        self._lock      = threading.Lock()
        self._batch     = False
        self._cbs: List[Callable[[str, str, Any], None]] = []

        # Seed defaults from every ConfigField in the MRO
        self._init_defaults()

    def _init_defaults(self) -> None:
        seen: set = set()
        for cls in type(self).__mro__:
            for attr_val in vars(cls).values():
                if isinstance(attr_val, ConfigField):
                    key = attr_val.name
                    if key and key not in seen:
                        seen.add(key)
                        val = attr_val.default
                        if isinstance(val, (list, dict)):
                            val = copy.deepcopy(val)
                        self._props['default'][key] = val

    # ── Group management ────────────────────────────────────────────────────

    def set_group(self, group: str) -> None:
        with self._lock:
            self._group = group
            self._props.setdefault(group, {})

    @property
    def group(self) -> str:
        return self._group

    # ── Core get / set (used by ConfigField descriptors) ────────────────────

    def _get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._props.get(self._group, {}).get(key, default)

    def _set(self, key: str, value: Any) -> None:
        with self._lock:
            self._props.setdefault(self._group, {})[key] = value
        self._notify(self._group, key, value)
        if not self._batch:
            self.save()

    # ── Dot-path access ─────────────────────────────────────────────────────

    def get_value(self, path: str, default: Any = None) -> Any:
        """Read by dot-path, e.g. ``'optimizer.lr'``."""
        with self._lock:
            parts = path.split('.')
            node: Any = self._props.get(self._group, {}).get(parts[0], _MISSING)
            if node is _MISSING:
                return default
            for part in parts[1:]:
                if not isinstance(node, dict):
                    return default
                node = node.get(part, _MISSING)
                if node is _MISSING:
                    return default
            return node

    def set_value(self, path: str, value: Any) -> None:
        """Write by dot-path, e.g. ``'optimizer.lr'``."""
        with self._lock:
            parts  = path.split('.')
            group  = self._props.setdefault(self._group, {})
            if len(parts) == 1:
                group[parts[0]] = value
            else:
                node = group.setdefault(parts[0], {})
                if not isinstance(node, dict):
                    node = {}
                    group[parts[0]] = node
                for part in parts[1:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
        self._notify(self._group, path, value)
        if not self._batch:
            self.save()

    def exist(self, key: str) -> bool:
        with self._lock:
            return key in self._props.get(self._group, {})

    # ── Observers ───────────────────────────────────────────────────────────

    def on_change(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register ``callback(group, key, new_value)`` for any value change."""
        self._cbs.append(callback)

    def _notify(self, group: str, key: str, value: Any) -> None:
        for cb in self._cbs:
            try:
                cb(group, key, value)
            except Exception:
                pass

    # ── Batch mode (mirrors beginBatch / endBatch) ───────────────────────────

    def begin_batch(self) -> None:
        """Acquire lock and defer all saves until :meth:`end_batch`."""
        self._lock.acquire()
        self._batch = True

    def end_batch(self) -> None:
        """Release batch mode and flush to disk once."""
        self._batch = False
        self._lock.release()
        self.save()

    # ── Sub-config composition ───────────────────────────────────────────────

    def sub_configs(self) -> Dict[str, 'AbstractConfig']:
        """
        Return named child :class:`AbstractConfig` instances.

        Override in subclasses that compose an encoder, decoder, etc.
        Children are included in :meth:`to_dict` / :meth:`from_dict`
        under keys of the form ``__<name>__``.
        """
        return {}

    # ── Serialization ────────────────────────────────────────────────────────

    def _tuple_fields(self) -> set:
        """Names of all ConfigField attributes whose default is a tuple."""
        names: set = set()
        for cls in type(self).__mro__:
            for attr_val in vars(cls).values():
                if isinstance(attr_val, ConfigField) and isinstance(attr_val.default, tuple):
                    names.add(attr_val.name)
        return names

    def to_dict(self) -> dict:
        """
        Serialize to a JSON-compatible nested dict.

        Structure: ``{group: {key: value, ...}, __child__: {...}, ...}``
        Python tuples are converted to lists for JSON compatibility.
        """
        with self._lock:
            result: dict = {
                group: {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in props.items()
                }
                for group, props in self._props.items()
            }
        for name, child in self.sub_configs().items():
            result[f'__{name}__'] = child.to_dict()
        return result

    def from_dict(self, data: dict) -> None:
        """Restore state from a dict produced by :meth:`to_dict`."""
        tuple_fields = self._tuple_fields()
        sub = self.sub_configs()
        with self._lock:
            for group, props in data.items():
                if group.startswith('__') and group.endswith('__'):
                    name = group[2:-2]
                    if name in sub:
                        sub[name].from_dict(props)
                elif isinstance(props, dict):
                    target = self._props.setdefault(group, {})
                    for k, v in props.items():
                        if k in tuple_fields and isinstance(v, list) and v is not None:
                            v = tuple(v)
                        target[k] = v

    def save(self, path: Optional[str] = None) -> bool:
        """Write JSON to *path* (or the file path set at construction)."""
        target = Path(path) if path else self._file_path
        if target is None:
            return False
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding='utf-8',
            )
            return True
        except OSError:
            return False

    def load(self, path: Optional[str] = None) -> bool:
        """Read and restore from a JSON file."""
        target = Path(path) if path else self._file_path
        if target is None or not target.exists():
            return False
        try:
            self.from_dict(json.loads(target.read_text(encoding='utf-8')))
            return True
        except (OSError, json.JSONDecodeError):
            return False

    # ── Human-readable summary ───────────────────────────────────────────────

    def summary(self, _indent: int = 0) -> str:
        """Return a formatted, indented summary of all fields."""
        pad = '  ' * _indent
        lines = [f'{pad}[{type(self).__name__}]  type={self.config_type!r}']
        with self._lock:
            for group, props in self._props.items():
                if not props:
                    continue
                for k, v in props.items():
                    val_str = repr(v)
                    if len(val_str) > 52:
                        val_str = val_str[:49] + '...'
                    lines.append(f'{pad}  {k:<28} {val_str}')
        for name, child in self.sub_configs().items():
            lines.append(f'{pad}  ┌─ {name}:')
            lines.append(child.summary(_indent + 2))
        return '\n'.join(lines)

    def __repr__(self) -> str:
        n = len(self._props.get(self._group, {}))
        return f'{type(self).__name__}(type={self.config_type!r}, fields={n})'
