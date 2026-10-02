"""Verified runtime imports and browser-export helpers for Figure 1.

The current Figure 1 builder binds the included tables directly. These helpers
verify and extract its frozen Python runtime, isolate imports from external
research sources and locate the browser used for vector export.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import types
from typing import Any
import zipfile

# Dynamic imports must never write cache files into immutable selected records.
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.ticker import FormatStrFormatter


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]
FORBIDDEN_SOURCE_PARTS = (
    "figure_version_records_archive",
    "figure_version_candidates",
    "parent_record_history",
    "parent_record_snapshot",
    "round11_parent_snapshot",
    "desktop",
    "ltm-figs",
)
MASTER_EXPORT_RE = re.compile(
    r"figure[1-5]_version_record(?:_300dpi|_600dpi)?\.(?:svg|pdf|png|tiff|html)$",
    re.IGNORECASE,
)
RUNTIME_ARCHIVE = "SOURCE_REBUILD_MAIN_RUNTIME.zip"
RUNTIME_ARCHIVE_SHA256 = "ac0d1efc9ce078f3b42d16fdcaf48b6446fb42c9acc986b83d20fbd9d0cab4cc"
RUNTIME_ARCHIVE_BYTES = 472999

_RUNTIME_TEMP: tempfile.TemporaryDirectory[str] | None = None
_RUNTIME_ROOT: Path | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_vor_source(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != VOR_ROOT.resolve():
        raise RuntimeError(f"VOR-root sidecar must be an immediate child of {VOR_ROOT}: {resolved}")
    if not resolved.name.startswith("SOURCE_REBUILD_MAIN_"):
        raise RuntimeError(f"Unrecognized VOR-root sidecar name: {resolved.name}")
    lowered = resolved.as_posix().lower()
    if any(part in lowered for part in FORBIDDEN_SOURCE_PARTS):
        raise RuntimeError(f"Forbidden source tree referenced: {resolved}")
    if MASTER_EXPORT_RE.search(resolved.name):
        raise RuntimeError(f"Current master export cannot be used as a source: {resolved}")


def _verify_vor_source(
    filename: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    kind: str,
    role: str,
) -> dict[str, Any]:
    path = VOR_ROOT / filename
    _assert_vor_source(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    observed_hash = _sha256(path)
    if observed_bytes != int(expected_bytes) or observed_hash.lower() != expected_sha256.lower():
        raise RuntimeError(f"VOR-root source hash drift: {path}")
    if path.suffix.lower() == ".zip" and not zipfile.is_zipfile(path):
        raise RuntimeError(f"VOR-root source archive is not a valid ZIP: {path}")
    return {
        "path": filename,
        "path_base": "vor_root",
        "kind": kind,
        "role": role,
        "sha256": observed_hash,
        "bytes": observed_bytes,
    }


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


def _extract_verified_archive(
    filename: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    prefix: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    _verify_vor_source(
        filename,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        kind="input",
        role="runtime extraction",
    )
    archive = VOR_ROOT / filename
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    root = Path(temporary.name).resolve()
    try:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                member = info.filename
                parts = PurePosixPath(member).parts
                if (
                    not parts
                    or PurePosixPath(member).is_absolute()
                    or "\\" in member
                    or any(part in {"", ".", ".."} for part in parts)
                    or ":" in parts[0]
                ):
                    raise RuntimeError(f"Unsafe member in {archive.name}: {member!r}")
                destination = root.joinpath(*parts).resolve()
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Archive member escapes the extraction root: {member!r}"
                    ) from exc
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info, "r") as input_stream, destination.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
    except Exception:
        temporary.cleanup()
        raise
    return temporary, root


def _runtime_root() -> Path:
    global _RUNTIME_TEMP, _RUNTIME_ROOT
    if _RUNTIME_ROOT is None:
        _RUNTIME_TEMP, _RUNTIME_ROOT = _extract_verified_archive(
            RUNTIME_ARCHIVE,
            expected_sha256=RUNTIME_ARCHIVE_SHA256,
            expected_bytes=RUNTIME_ARCHIVE_BYTES,
            prefix="source_rebuild_main_runtime_",
        )
        if not (_RUNTIME_ROOT / "ca_analysis" / "__init__.py").is_file():
            raise RuntimeError("The source-rebuild runtime lacks ca_analysis/__init__.py")
    return _RUNTIME_ROOT


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
    os.environ["CA_COMPAT_LINK_ARTIFACTS"] = "0"
    ca_analysis = importlib.import_module("ca_analysis")
    compat = importlib.import_module("ca_analysis.compat")

    def frozen_bootstrap_environment(repo_root: str | os.PathLike | None = None):
        return types.SimpleNamespace(
            repo_root=str(Path(repo_root).resolve() if repo_root is not None else REPO_ROOT),
            added_paths=[],
            missing_paths=[],
        )

    compat.bootstrap_environment = frozen_bootstrap_environment

    # ``jw_utils.py`` is a large legacy utility module with costly optional
    # imports.  None of these five rebuilds calls it: Main Figure 5 freezes a
    # precomputed exact seriation, so the on-demand hierarchical fallback in
    # ``figure_panels.py`` is never reached.  Install only the lightweight
    # frozen helper that is actually executed.
    for stem in ("level_heatmap_utils",):
        frozen = record / "source_code" / f"{stem}.py"
        if frozen.is_file():
            loaded = _load_module(f"ca_analysis.{stem}", frozen)
            setattr(ca_analysis, stem, loaded)
    frozen_panels = _load_module(
        "ca_analysis.figure_panels",
        record / "source_code" / "figure_panels.py",
    )
    setattr(ca_analysis, "figure_panels", frozen_panels)
    _assert_isolated_ca_analysis(record, runtime_root)
    return frozen_panels


def _find_chrome() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Chrome/Edge is required for vector SVG-to-PDF export")


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
