"""Paths to files generated during a Source Data build."""
from pathlib import Path
from typing import Iterable


def unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Keep each absolute path once, in the supplied order."""
    return tuple(dict.fromkeys(Path(path).resolve() for path in paths))


def repository_relative_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    """Resolve runtime output paths, including a user-selected output folder.

    These paths are transient result records, not stored data dependencies.
    Absolute paths allow assembly when outputs are on another drive.
    """
    return tuple(Path(path).resolve() for path in paths)
