from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    'io',
    'spatial',
    'similarity',
    'stats',
    'compat',
    'config',
    'maze',
    'trajectory',
    'mm_maze_utils',
    'mm_plot_utils',
    'data_loader_utils',
    'query_data_utils',
    'meta_data_loader_utils',
    'mm_traj',
    'jw_utils',
    'figure_panels',
    'session_loader_tools',
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
