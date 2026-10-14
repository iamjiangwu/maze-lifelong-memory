"""Small final-output helpers shared by figure modules.

:func:`save_panel_output` is the one function every figure notebook calls to
write a finished panel: its plotted values as CSV and the figure as SVG, PDF
and PNG.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
from matplotlib.figure import Figure

from .config import REPOSITORY_ROOT
from .paths import repository_relative_paths, unique_paths
from .plotting import close_figure, save_figure, version_record_style
from .types import PanelKind, PanelResult, RunMode


_SVG_REFERENCE_PATTERN = re.compile(r"#([A-Za-z_][\w:.-]*)")


def _result_file(result: PanelResult, suffix: str) -> Path:
    """Resolve one generated file with ``suffix`` from a completed result."""

    matches = [
        (REPOSITORY_ROOT / path).resolve()
        for path in result.output_paths
        if path.suffix.casefold() == suffix
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise ValueError(f"{result.panel_key} must provide one existing {suffix} file")
    return matches[0]


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    """Keep SVG identifiers unique when generated files are combined."""

    identifiers: dict[str, str] = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier:
            identifiers[identifier] = f"{prefix}_{identifier}"
            element.attrib["id"] = identifiers[identifier]
    for element in root.iter():
        for name, value in tuple(element.attrib.items()):
            if "#" in value:
                element.attrib[name] = _SVG_REFERENCE_PATTERN.sub(
                    lambda match: f"#{identifiers.get(match.group(1), match.group(1))}",
                    value,
                )


def _svg_viewbox(path: Path) -> tuple[ET.Element, float, float]:
    """Read a generated SVG and return its positive view-box dimensions."""

    root = ET.parse(path).getroot()
    values = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError(f"SVG has no four-value viewBox: {path}")
    width, height = float(values[2]), float(values[3])
    if width <= 0 or height <= 0:
        raise ValueError(f"SVG viewBox is not positive: {path}")
    return root, width, height


def validate_output_request(
    *, run_mode: RunMode, output_dir: Path, random_seed: int
) -> Path:
    """Validate the common arguments used when saving one panel output."""

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")
    return Path(output_dir)


def significance_text(p_value: float, *, nonsignificant: str = "n.s.") -> str:
    """Translate one displayed p value into the manuscript star convention.

    The thresholds are p < 0.001 (``***``), p < 0.01 (``**``) and p < 0.05
    (``*``).  Any other value, including NaN, returns ``nonsignificant``.
    """

    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return nonsignificant


def write_panel(
    *,
    figure: Figure,
    values: pd.DataFrame,
    output_dir: Path,
    file_stem: str,
) -> tuple[Path, ...]:
    """Write one panel and its exact plotted values, then release the figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{file_stem}_values.csv"
    values.to_csv(table_path, index=False)
    figure_paths = save_figure(
        figure,
        output_dir / file_stem,
        formats=("svg", "pdf", "png"),
        raster_dpi=600,
    )
    close_figure(figure)
    return (table_path, *figure_paths)


def completed_panel(
    *,
    panel_key: str,
    kind: PanelKind,
    run_mode: RunMode,
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    started_at: float,
    plotted_counts: Mapping[str, int] | None = None,
    statistical_unit: str | None = None,
    cluster_unit: str | None = None,
    notes: Sequence[str] = (),
) -> PanelResult:
    """Create the portable result record returned to a notebook cell."""

    return PanelResult(
        panel_key=panel_key,
        kind=kind,
        run_mode=run_mode,
        input_paths=repository_relative_paths(input_paths),
        output_paths=repository_relative_paths(output_paths),
        runtime_seconds=time.perf_counter() - started_at,
        plotted_counts={} if plotted_counts is None else dict(plotted_counts),
        statistical_unit=statistical_unit,
        cluster_unit=cluster_unit,
        notes=tuple(notes),
    )


def finish_panel(
    *,
    artifact_key: str,
    panel: str,
    kind: PanelKind,
    run_mode: RunMode,
    output_dir: Path,
    figure: Figure,
    values: pd.DataFrame,
    input_paths: Sequence[Path],
    started_at: float,
    plotted_counts: Mapping[str, int] | None = None,
    statistical_unit: str | None = None,
    cluster_unit: str | None = None,
    notes: Sequence[str] = (),
    file_stem: str | None = None,
) -> PanelResult:
    """Write one standard figure panel and return its portable result record.

    Figure modules supply only the artifact-specific values and labels.  Keeping
    file writing and :class:`PanelResult` construction here avoids repeating the
    same notebook-facing plumbing in every figure module.
    """

    stem = file_stem or f"{artifact_key}_panel_{panel}"
    output_paths = write_panel(
        figure=figure,
        values=values,
        output_dir=output_dir,
        file_stem=stem,
    )
    return completed_panel(
        panel_key=f"{artifact_key}:{panel}",
        kind=kind,
        run_mode=run_mode,
        input_paths=input_paths,
        output_paths=output_paths,
        started_at=started_at,
        plotted_counts=plotted_counts,
        statistical_unit=statistical_unit,
        cluster_unit=cluster_unit,
        notes=notes,
    )


def save_panel_output(
    *,
    artifact_key: str,
    panel: str,
    kind: PanelKind,
    run_mode: RunMode,
    output_dir: Path,
    random_seed: int,
    figure: Figure,
    values: pd.DataFrame,
    input_paths: Iterable[Path] = (),
    started_at: float | None = None,
    plotted_counts: Mapping[str, int] | None = None,
    statistical_unit: str | None = None,
    cluster_unit: str | None = None,
    notes: Iterable[str] = (),
    transparent_background: bool = False,
    record_style: Mapping[str, object] | None = None,
) -> PanelResult:
    """Save one calculated and drawn panel and return its result record.

    The files are written to ``output_dir`` as
    ``<artifact_key>_panel_<panel>_values.csv`` (exactly ``values``) and
    ``<artifact_key>_panel_<panel>.svg``, ``.pdf`` and ``.png``.  The figure is
    closed afterwards.

    ``input_paths`` are the scientific inputs of the panel; repeated paths are
    listed once.  ``started_at`` is the :func:`time.perf_counter` value when
    the panel's calculation began; without it only the saving is timed.
    ``plotted_counts`` defaults to the number of rows in ``values``.

    Two settings change how the figure files are written:

    - ``transparent_background`` makes the figure and axes backgrounds
      transparent first (Extended Data Figure 8, whose panel slots overlap);
    - ``record_style`` is a set of Matplotlib settings applied with
      :func:`release_rendering.plotting.version_record_style` while the
      files are written (Figure 5).
    """

    started = time.perf_counter() if started_at is None else started_at
    destination = validate_output_request(
        run_mode=run_mode, output_dir=output_dir, random_seed=random_seed
    )
    if transparent_background:
        figure.patch.set_alpha(0.0)
        for axis in figure.axes:
            axis.patch.set_alpha(0.0)
    counts = (
        {"value_rows": int(len(values))} if plotted_counts is None else plotted_counts
    )
    style = nullcontext() if record_style is None else version_record_style(record_style)
    with style:
        return finish_panel(
            artifact_key=artifact_key,
            panel=panel,
            kind=kind,
            run_mode=run_mode,
            output_dir=destination,
            figure=figure,
            values=values,
            input_paths=unique_paths(input_paths),
            started_at=started,
            plotted_counts=counts,
            statistical_unit=statistical_unit,
            cluster_unit=cluster_unit,
            notes=tuple(notes),
        )


__all__ = [
    "completed_panel",
    "finish_panel",
    "save_panel_output",
    "significance_text",
    "validate_output_request",
    "write_panel",
]
