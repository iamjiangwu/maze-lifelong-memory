"""Report the source-package root without changing paths or linking raw data."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompatPathReport:
    repo_root: str
    added_paths: list[str]
    missing_paths: list[str]


def bootstrap_environment(repo_root: str | os.PathLike | None = None) -> CompatPathReport:
    """Return the package root without consulting an external workspace."""
    repo = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root).resolve()
    return CompatPathReport(repo_root=str(repo), added_paths=[], missing_paths=[])
