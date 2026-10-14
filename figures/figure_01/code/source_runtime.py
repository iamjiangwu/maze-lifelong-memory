"""Verified runtime imports and browser-export helpers for Figure 1.

The current Figure 1 builder binds the included tables directly. These helpers
load its authenticated shared Python runtime, isolate imports from external
research sources and locate the browser used for vector export.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import types

# Dynamic imports must never write cache files into immutable selected records.
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]


def _load_module(name: str, path: Path, *, synthetic_file: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(synthetic_file or path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    # A few historically frozen helpers retain a UTF-8 BOM.  ``utf-8-sig``
    # consumes it while remaining identical to UTF-8 for every other source.
    source = path.read_text(encoding="utf-8-sig")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _runtime_root() -> Path:
    # Hash verification is performed by reproduce.py before staging this tree.
    for parent in (VOR_ROOT, *VOR_ROOT.parents):
        root = parent / "shared/runtime"
        if (root / "ca_analysis/__init__.py").is_file():
            return root
    raise RuntimeError("Run this figure through reproduce.py to stage its shared runtime")


def _remove_live_source_path() -> None:
    live_src = (REPO_ROOT / "src").resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == live_src:
                continue
        except (OSError, RuntimeError):
            pass
        retained.append(entry)
    sys.path[:] = retained


def _assert_isolated_ca_analysis(record: Path, runtime_root: Path) -> None:
    allowed_roots = (runtime_root.resolve(), (record / "source_code").resolve())
    live_src = (REPO_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "ca_analysis" and not name.startswith("ca_analysis."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        resolved = Path(module_file).resolve()
        if resolved == live_src or live_src in resolved.parents:
            raise RuntimeError(f"Live repository dependency imported during source rebuild: {resolved}")
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Unfrozen ca_analysis dependency imported: {resolved}")


def _install_frozen_panel_modules(record: Path):
    runtime_root = _runtime_root()
    _remove_live_source_path()
    runtime_text = str(runtime_root)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_text]
    sys.path.insert(0, runtime_text)
    for name in tuple(sys.modules):
        if name == "ca_analysis" or name.startswith("ca_analysis."):
            del sys.modules[name]
    ca_analysis = importlib.import_module("ca_analysis")

    frozen_panels = _load_module(
        "ca_analysis.figure_panels",
        record / "source_code" / "figure_panels.py",
    )
    setattr(ca_analysis, "figure_panels", frozen_panels)
    _assert_isolated_ca_analysis(record, runtime_root)
    return frozen_panels


def _find_chrome() -> Path:
    from ca_analysis.browser_export import find_browser

    return find_browser(preferred=("chrome", "edge", "chromium"))


def _wait_for_file(path: Path, *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable = 0
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.is_file() else -1
        if size > 0 and size == previous_size:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        previous_size = size
        time.sleep(0.15)
    raise RuntimeError(f"Export did not settle: {path}")
