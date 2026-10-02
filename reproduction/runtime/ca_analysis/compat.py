from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

@dataclass
class CompatPathReport:
    repo_root: str
    added_paths: list[str]
    missing_paths: list[str]

def _add_path(path: Path, added: list[str], missing: list[str]) -> None:
    if path.exists():
        p = str(path.resolve())
        if p in sys.path:
            sys.path.remove(p)
        # Preserve candidate order by appending in sequence.
        sys.path.append(p)
        added.append(p)
    else:
        missing.append(str(path))

def _ensure_artifact_dirs(repo: Path) -> None:
    """
    Ensure `outdata/`, `figs/`, and `mm-trajectories/` are reachable from repo root.

    Default behavior is standalone-safe: create local directories in the repo.
    If `CA_COMPAT_ARTIFACT_SOURCE_ROOT` is set, it may link to that source root first.
    """
    if os.environ.get("CA_COMPAT_LINK_ARTIFACTS", "1").strip().lower() in {"0", "false", "no"}:
        return

    source_root_env = os.environ.get("CA_COMPAT_ARTIFACT_SOURCE_ROOT", "").strip()
    source_root = Path(source_root_env).resolve() if source_root_env else None
    for name in ("outdata", "figs", "mm-trajectories"):
        dst = repo / name
        if dst.exists():
            continue
        src = (source_root / name) if source_root is not None else None
        if src is None or not src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            continue

        # Try native symlink first.
        try:
            os.symlink(src, dst, target_is_directory=True)
            continue
        except Exception:
            pass

        # On Windows, directory junction usually works without admin rights.
        if os.name == "nt":
            try:
                cmd = ["cmd", "/c", "mklink", "/J", str(dst), str(src)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode == 0 and dst.exists():
                    continue
            except Exception:
                pass

        # Last resort: create empty directory to keep save paths valid.
        dst.mkdir(parents=True, exist_ok=True)

def bootstrap_environment(repo_root: str | os.PathLike | None = None) -> CompatPathReport:
    # Configure import paths for legacy modules and vendored external code.
    if repo_root is None:
        repo = Path(__file__).resolve().parents[2]
    else:
        repo = Path(repo_root).resolve()
    added: list[str] = []
    missing: list[str] = []
    candidates = [
        repo / 'src',
        repo / 'external' / 'GLM_Tensorflow_2-main' / 'code',
        repo / 'external' / 'mouse-maze-analysis-master',
        repo / 'external' / 'cscg',
    ]
    for c in candidates:
        _add_path(c, added, missing)
    _ensure_artifact_dirs(repo)
    return CompatPathReport(repo_root=str(repo), added_paths=added, missing_paths=missing)

def check_required_modules(modules: Iterable[str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for m in modules:
        try:
            importlib.import_module(m)
            status[m] = True
        except Exception:
            status[m] = False
    return status
