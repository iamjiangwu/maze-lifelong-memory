"""Small output records used when panels are assembled into a figure."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

RunMode = Literal['full']
PanelKind = Literal['quantitative', 'example', 'schematic', 'table']


@dataclass(frozen=True)
class PanelResult:
    """Identify the files just generated for one panel."""

    panel_key: str
    kind: PanelKind
    run_mode: RunMode
    input_paths: tuple[Path, ...]
    output_paths: tuple[Path, ...]
    runtime_seconds: float
    plotted_counts: Mapping[str, int] = field(default_factory=dict)
    statistical_unit: str | None = None
    cluster_unit: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureAssemblyResult:
    """Identify the complete page generated from this build's panels."""

    artifact_key: str
    run_mode: RunMode
    output_paths: tuple[Path, ...]
