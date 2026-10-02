from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    'spatial',
    'maze',
    'figure_panels',
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals().keys()) | _SUBMODULES)
