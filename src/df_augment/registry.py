"""Global plug-in registry: maps technique name to implementation class."""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, type[Any]] = {}


def register(cls: type[Any]) -> type[Any]:
    """Class decorator: register *cls* under its ``.name`` attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Any]:
    """Return the plug-in class registered under *name*.

    Raises
    ------
    KeyError
        If no plug-in with that name has been registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown technique: {name!r}. Available: {sorted(_REGISTRY)}.")
    return _REGISTRY[name]
