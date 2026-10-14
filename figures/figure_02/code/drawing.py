"""Accepted presentation grammar for the longitudinal representation summary.

The notebook calculates every example, null distribution, gap summary, and
recurrence statistic before calling this module.  This file contains only the
fixed drawing and page placement used by the immutable publication record.
"""

from __future__ import annotations

import copy
import io
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Polygon, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image
from release_rendering.plotting import (
    chromium_export_browser,
    figure_font_family,
    fix_pdf_dates,
    report_skipped_browser_export,
    savefig_reproducibly,
    wait_for_browser_file,
)
from types import SimpleNamespace
ContextImages = MatchedNeuronExamples = SimpleNamespace


_CANVAS_SIZE_INCHES = (6.08, 7.058683289588801)
_EXPORT_WIDTH_MM = 180.0
_TOP_ROW_SHIFT_INCHES = 7.0 / 300.0
_TOP_SLOTS = {
    "context": (0.048486439195100615, 0.04500218722659668 + _TOP_ROW_SHIFT_INCHES, 1.4323129921259843, 2.1070395888014),
    "example_labels": (1.4199792213473317, _TOP_ROW_SHIFT_INCHES, 0.125, 2.3166218285214346),
    "example_1": (1.5450, _TOP_ROW_SHIFT_INCHES, 1.0791458880139984, 2.3166218285214346),
    "example_2": (2.7072189413823273, _TOP_ROW_SHIFT_INCHES, 1.0791458880139984, 2.3166218285214346),
    "example_3": (3.8293285214348205, _TOP_ROW_SHIFT_INCHES, 1.0791458880139984, 2.3166218285214346),
    "example_4": (4.946116579177603, _TOP_ROW_SHIFT_INCHES, 1.0791458880139984, 2.3166218285214346),
}
_EXAMPLE_SLOT = (
    _TOP_SLOTS["example_labels"][0],
    _TOP_SLOTS["example_labels"][1],
    _TOP_SLOTS["example_4"][0] + _TOP_SLOTS["example_4"][2] - _TOP_SLOTS["example_labels"][0],
    _TOP_SLOTS["example_4"][3],
)
_ORIGINAL_LOWER_COLUMNS = {
    0: (0.060494313210848645, 1.5728860454943132),
    1: (1.621017060367454, 0.8354002624671916),
    2: (2.570269028871391, 1.9038823272090988),
    3: (4.538460192475941, 1.4868022747156606),
}
_LOWER_WIDTH_SCALES = {0: 0.8, 1: 1.0, 2: 0.5, 3: 0.8}
_LOWER_GAP_INCHES = 0.08
_LOWER_LEFT = _ORIGINAL_LOWER_COLUMNS[0][0]
_LOWER_RIGHT = _ORIGINAL_LOWER_COLUMNS[3][0] + _ORIGINAL_LOWER_COLUMNS[3][1]
_SCHEMATIC_WIDTH = (
    _LOWER_RIGHT
    - _LOWER_LEFT
    - sum(_ORIGINAL_LOWER_COLUMNS[index][1] * _LOWER_WIDTH_SCALES[index] for index in range(4))
    - 4 * _LOWER_GAP_INCHES
)
_LOWER_COLUMNS: dict[int, tuple[float, float]] = {}
_next_column_x = _LOWER_LEFT + _SCHEMATIC_WIDTH + _LOWER_GAP_INCHES
for _column_index in range(4):
    _column_width = _ORIGINAL_LOWER_COLUMNS[_column_index][1] * _LOWER_WIDTH_SCALES[_column_index]
    _LOWER_COLUMNS[_column_index] = (_next_column_x, _column_width)
    _next_column_x += _column_width + _LOWER_GAP_INCHES
del _column_index, _column_width, _next_column_x

_LOWER_FIRST_Y = 2.565062335958005
_LOWER_HEIGHT = 1.398365048118985
_LOWER_ROW_GAP = 0.12
_LOWER_ROWS = {
    row: (_LOWER_FIRST_Y + row * (_LOWER_HEIGHT + _LOWER_ROW_GAP), _LOWER_HEIGHT)
    for row in range(3)
}
_METRICS = ("pv", "place", "rate")
_METRIC_COLORS = {"place": "#B95A58", "pv": "#E29957", "rate": "#4A5E65"}
_METRIC_TITLES = {"place": "Place map", "pv": "Population vector (PV)", "rate": "Ensemble rate"}
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_SVG_TAG = f"{{{_SVG_NAMESPACE}}}"


@dataclass(frozen=True, slots=True)
class RepresentationStabilitySummaryResult:
    """Paths written for the complete source rendered summary."""

    output_paths: tuple[Path, ...]
    export_renderer: Path | None
    """Browser that printed the PDF and PNG, or None when they were skipped."""

    def show(self) -> None:
        """Display the PNG and link the editable vector outputs in a notebook."""

        from IPython.display import FileLink, Image as NotebookImage, display

        if self.export_renderer is None:
            print("No browser export: only the SVG page was written.")
        else:
            print(f"Accepted export renderer: {self.export_renderer}")
        for path in self.output_paths:
            if path.suffix.casefold() == ".png":
                display(NotebookImage(filename=str(path)))
            else:
                display(FileLink(str(path)))


@dataclass(slots=True)
class _PanelAsset:
    """One freshly rendered panel held in memory for three-format placement."""

    children: list[ET.Element]
    width_points: float
    height_points: float
    png: bytes
    pdf: bytes


def _style_context() -> dict[str, object]:
    """Return the typography and line settings for the accepted page."""

    return {
        "font.family": figure_font_family(),
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "axes.titlesize": 6.2,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
        "legend.fontsize": 4.9,
        "axes.linewidth": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _style_axis(axis: plt.Axes) -> None:
    """Apply the shared compact axis styling."""

    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)
    axis.tick_params(axis="both", width=0.55, length=1.8, labelsize=5.4, pad=1.5)
    axis.xaxis.label.set_size(6.0)
    axis.yaxis.label.set_size(6.0)
    axis.title.set_size(6.2)
    axis.grid(False)


def _star_label(p_value: float) -> str:
    """Convert one adjusted probability to its displayed symbol."""

    if not np.isfinite(p_value) or p_value > 0.05:
        return "n.s."
    if p_value > 0.01:
        return "*"
    if p_value > 0.001:
        return "**"
    return "***"


def _render_context_images(data: ContextImages) -> plt.Figure:
    """Draw the accepted two-session local-correlation example."""

    figure = plt.figure(figsize=(_TOP_SLOTS["context"][2], _TOP_SLOTS["context"][3]), facecolor="none")
    axes = (
        figure.add_axes([0.055, 0.506, 0.940, 0.492]),
        figure.add_axes([0.055, 0.006, 0.940, 0.492]),
    )
    images = [source.values for source in data.sources]
    vmin = min(float(np.nanmin(image)) for image in images)
    vmax = max(float(np.nanmax(image)) for image in images)
    for axis, image, label, source in zip(axes, images, ("Day 1", "Day 91"), data.sources, strict=True):
        axis.imshow(
            image,
            cmap=plt.matplotlib.colors.ListedColormap(source.colormap),
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=(1.0, 360.0, 288.0, 1.0),
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_axis_off()
        axis.text(-0.060, 0.50, label, transform=axis.transAxes, rotation=90, ha="center", va="center", fontsize=8.0, clip_on=False)
    axes[1].plot((295.0, 355.0), (278.0, 278.0), color="white", linewidth=2.4, solid_capstyle="butt", clip_on=False)
    axes[1].text(312.0, 268.0, "100 µm", color="white", fontsize=6.4, ha="center", va="bottom", clip_on=False)
    return figure


def _draw_direction_marker(
    axis: plt.Axes,
    body_x: float,
    body_y: float,
    nose_x: float,
    nose_y: float,
) -> None:
    """Draw one accepted red triangular head from body to nose direction."""

    direction = 0.25 * np.asarray((nose_x - body_x, nose_y - body_y), dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0:
        return
    unit = direction / norm
    normal = np.asarray((-unit[1], unit[0]))
    tip = np.asarray((body_x, body_y)) + direction
    base = tip - 0.34 * unit
    vertices = np.vstack((tip, base + 0.07 * normal, base - 0.07 * normal))
    axis.add_patch(Polygon(vertices, closed=True, facecolor="red", edgecolor="red", alpha=0.42, linewidth=0.01, zorder=2))


def _render_matched_neurons(data: MatchedNeuronExamples) -> plt.Figure:
    """Draw the four accepted matched-neuron trajectory examples."""

    figure, axes = plt.subplots(2, 4, figsize=(_EXAMPLE_SLOT[2], _EXAMPLE_SLOT[3]), squeeze=False)
    figure.subplots_adjust(left=0.038, right=0.999, top=0.996, bottom=0.070, wspace=-0.005, hspace=0.010)
    for column, (source, target, correlation, _support) in enumerate(data.examples):
        for axis, arrows in zip(axes[:, column], (source, target), strict=True):
            axis.plot(arrows.trajectory_x, arrows.trajectory_y, color="0.55", linewidth=0.26, alpha=0.28, rasterized=True, zorder=1)
            for body_x, body_y, nose_x, nose_y in zip(arrows.body_x, arrows.body_y, arrows.nose_x, arrows.nose_y, strict=True):
                _draw_direction_marker(axis, body_x, body_y, nose_x, nose_y)
            axis.set_aspect("equal")
            axis.set_xlim(-0.5, 14.5)
            axis.set_ylim(-0.5, 14.5)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[1, column].text(0.5, -0.075, f"{correlation:.2f}", transform=axes[1, column].transAxes, ha="center", va="top", fontsize=6.2, clip_on=False)
    axes[0, 0].set_ylabel("Day i", fontsize=8.0, labelpad=2)
    axes[1, 0].set_ylabel("Day i+90", fontsize=8.0, labelpad=2)
    for axis in axes[1, :]:
        position = axis.get_position()
        axis.set_position((position.x0, position.y0 - 0.010, position.width, position.height))
        for text in axis.texts:
            x_value, y_value = text.get_position()
            text.set_position((x_value, y_value + 0.010 / position.height))
    for axis in axes.flat:
        axis.add_patch(Rectangle((-0.5, -0.5), 15.0, 15.0, fill=False, edgecolor="black", linewidth=1.15, joinstyle="miter", capstyle="butt", zorder=5, clip_on=False))
        axis.set_xlim(-0.5, 14.5)
        axis.set_ylim(-0.5, 14.5)
    return figure


def _schematic_values(metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the two small arrays drawn in a metric schematic."""

    first = np.asarray(
        [
            [0.08, 0.36, 0.82, 0.97, 0.99, 1.00],
            [0.88, 0.52, 0.09, 0.82, 0.96, 0.99],
            [0.99, 0.88, 0.54, 0.12, 0.77, 0.94],
            [1.00, 0.98, 0.88, 0.58, 0.11, 0.72],
            [1.00, 1.00, 0.98, 0.86, 0.57, 0.08],
            [1.00, 1.00, 1.00, 0.96, 0.83, 0.49],
        ]
    )
    second = np.asarray(
        [
            [0.06, 0.33, 0.78, 0.96, 0.99, 1.00],
            [0.91, 0.56, 0.13, 0.79, 0.95, 0.99],
            [0.98, 0.84, 0.60, 0.20, 0.71, 0.89],
            [1.00, 0.98, 0.90, 0.65, 0.15, 0.69],
            [1.00, 1.00, 0.98, 0.88, 0.61, 0.10],
            [1.00, 1.00, 1.00, 0.94, 0.86, 0.52],
        ]
    )
    second[:, 3] = (0.93, 0.76, 0.20, 0.65, 0.82, 0.94)
    if metric == "rate":
        return (
            np.asarray((0.08, 0.47, 0.62, 0.78, 0.90, 0.55)),
            np.asarray((0.11, 0.42, 0.67, 0.73, 0.85, 0.60)),
        )
    return first, second


def representation_metric_schematic_values(metric: str) -> pd.DataFrame:
    """Return the fixed values displayed in one metric-definition schematic."""

    if metric not in _METRICS:
        raise ValueError("metric must be 'place', 'pv', or 'rate'")
    first, second = _schematic_values(metric)
    rows: list[dict[str, object]] = []
    for session_number, values in enumerate((first, second), start=1):
        if values.ndim == 1:
            for neuron_index, value in enumerate(values):
                rows.append({"session": session_number, "neuron_index": neuron_index, "position_index": np.nan, "schematic_value": float(value), "highlighted_for_correlation": True})
        else:
            for neuron_index, position_index in np.ndindex(values.shape):
                rows.append(
                    {
                        "session": session_number,
                        "neuron_index": neuron_index,
                        "position_index": position_index,
                        "schematic_value": float(values[neuron_index, position_index]),
                        "highlighted_for_correlation": bool(neuron_index == 2 if metric == "place" else position_index == 3),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _schematic_arrays(values: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the two displayed arrays from the notebook table."""

    arrays: list[np.ndarray] = []
    for session_number in (1, 2):
        session = values.loc[values["session"].eq(session_number)]
        if metric == "rate":
            arrays.append(session.sort_values("neuron_index")["schematic_value"].to_numpy(float))
        else:
            matrix = session.pivot(index="neuron_index", columns="position_index", values="schematic_value").sort_index().sort_index(axis=1)
            arrays.append(matrix.to_numpy(float))
    return arrays[0], arrays[1]


def _schematic_header_bounds(size: tuple[float, float]) -> tuple[float, float, float, float]:
    """Return the centered title box for a schematic of the given size."""

    font = FontProperties(family=figure_font_family(), size=6.2)
    title_width = float(TextPath((0.0, 0.0), _METRIC_TITLES["pv"], prop=font, usetex=False).get_extents().width)
    width = min(0.94, (title_width + 10.0) / 72.0 / size[0])
    return (1.0 - width) / 2.0, 0.830, width, 0.14


def _render_metric_schematic(values: pd.DataFrame, metric: str, size: tuple[float, float]) -> plt.Figure:
    """Draw one accepted row, column, or vector comparison schematic."""

    first, second = _schematic_arrays(values, metric)
    color = _METRIC_COLORS[metric]
    figure = plt.figure(figsize=size, facecolor="none")
    axis = figure.add_axes((0.025, 0.040, 0.95, 0.73))
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.set_axis_off()
    content_width = size[0] * 0.95
    content_height = size[1] * 0.73

    if metric in {"place", "pv"}:
        matrix_width = 0.34
        matrix_height = matrix_width * content_width / content_height
        matrix_y = 0.56 - matrix_height / 2.0
        for session_index, (x_start, matrix) in enumerate(zip((0.12, 0.57), (first, second), strict=True), start=1):
            cell_width = matrix_width / matrix.shape[1]
            cell_height = matrix_height / matrix.shape[0]
            for row, column in np.ndindex(matrix.shape):
                axis.add_patch(Rectangle((x_start + column * cell_width, matrix_y + (matrix.shape[0] - row - 1) * cell_height), cell_width, cell_height, facecolor=plt.cm.Greys(1.0 - matrix[row, column]), edgecolor="#D2D5D6", linewidth=0.45))
            axis.text(x_start + matrix_width / 2.0, 0.93, f"Session {session_index}", ha="center", va="center", fontsize=5.4)
            axis.annotate("", xy=(x_start + matrix_width, 0.27), xytext=(x_start, 0.27), arrowprops={"arrowstyle": "-|>", "color": "#34383A", "linewidth": 0.75, "mutation_scale": 7.0, "shrinkA": 0.0, "shrinkB": 0.0})
            axis.text(x_start + matrix_width / 2.0, 0.10, "Position", ha="center", va="center", fontsize=5.4)
            if metric == "place":
                highlight = (x_start, matrix_y + 3 * cell_height, matrix_width, cell_height)
            else:
                highlight = (x_start + 3 * cell_width, matrix_y, cell_width, matrix_height)
            axis.add_patch(Rectangle(highlight[:2], highlight[2], highlight[3], facecolor=color, edgecolor=color, alpha=0.18, linewidth=1.0))
            axis.add_patch(Rectangle(highlight[:2], highlight[2], highlight[3], fill=False, edgecolor=color, linewidth=1.05))
        axis.text(0.515, matrix_y + matrix_height / 2.0, "vs", ha="center", va="center", fontsize=5.4, color="#6C7072")
        axis.annotate("", xy=(0.070, matrix_y + matrix_height), xytext=(0.070, matrix_y), arrowprops={"arrowstyle": "-|>", "color": "#34383A", "linewidth": 0.75, "mutation_scale": 7.0, "shrinkA": 0.0, "shrinkB": 0.0})
        axis.text(0.024, matrix_y + matrix_height / 2.0, "Neuron", ha="center", va="center", rotation=90, fontsize=5.4)
    else:
        vector_height = 0.60
        cell_height = vector_height / 6.0
        vector_width = cell_height * content_height / content_width
        vector_y = 0.55 - vector_height / 2.0
        for session_index, (x_start, vector) in enumerate(zip((0.27 - vector_width / 2.0, 0.75 - vector_width / 2.0), (first, second), strict=True), start=1):
            for row, value in enumerate(vector):
                axis.add_patch(Rectangle((x_start, vector_y + (len(vector) - row - 1) * cell_height), vector_width, cell_height, facecolor=plt.cm.Greys(1.0 - value), edgecolor="#D2D5D6", linewidth=0.45))
            axis.add_patch(Rectangle((x_start, vector_y), vector_width, vector_height, fill=False, edgecolor=color, linewidth=1.05))
            axis.text(x_start + vector_width / 2.0, 0.93, f"Session {session_index}", ha="center", va="center", fontsize=5.4)
            axis.text(x_start + vector_width / 2.0, 0.12, "Mean rate", ha="center", va="center", fontsize=5.4)
        axis.text(0.510, vector_y + vector_height / 2.0, "vs", ha="center", va="center", fontsize=5.4, color="#6C7072")
        axis.annotate("", xy=(0.155, vector_y + vector_height), xytext=(0.155, vector_y), arrowprops={"arrowstyle": "-|>", "color": "#34383A", "linewidth": 0.75, "mutation_scale": 7.0, "shrinkA": 0.0, "shrinkB": 0.0})
        axis.text(0.100, vector_y + vector_height / 2.0, "Neuron", ha="center", va="center", rotation=90, fontsize=5.4)

    header = figure.add_axes((0.0, 0.0, 1.0, 1.0), zorder=10)
    header.set_axis_off()
    x_start, y_start, width, height = _schematic_header_bounds(size)
    header.add_patch(Rectangle((x_start, y_start), width, height, facecolor=color, edgecolor="none"))
    header.text(x_start + width / 2.0, y_start + height / 2.0, _METRIC_TITLES[metric], ha="center", va="center", color="white", fontsize=6.2)
    return figure


def plot_representation_metric_schematic(
    values: pd.DataFrame,
    metric: str,
) -> plt.Figure:
    """Draw the accepted metric schematic from its visible notebook table."""

    if metric not in _METRICS:
        raise ValueError("metric must be 'place', 'pv', or 'rate'")
    with plt.rc_context(_style_context()):
        return _render_metric_schematic(values, metric, (_SCHEMATIC_WIDTH, _LOWER_HEIGHT))


def _histogram_rows(values: pd.DataFrame) -> pd.DataFrame:
    """Select and order the precomputed distribution histogram rows."""

    rows = values.loc[values["record_type"].eq("histogram_bin")].sort_values("bin_center")
    if rows.empty:
        raise ValueError("A correlation distribution is missing histogram bins")
    return rows


def _draw_distribution(axis: plt.Axes, values: pd.DataFrame, metric: str) -> None:
    """Draw observed and shuffled pair-correlation distributions."""

    bins = _histogram_rows(values)
    widths = bins["bin_right"].to_numpy(float) - bins["bin_left"].to_numpy(float)
    centers = bins["bin_center"].to_numpy(float)
    axis.bar(centers, bins["shuffle_density"].to_numpy(float), width=widths, color="#8a8a8a", alpha=0.62, edgecolor="none", align="center", zorder=1)
    axis.bar(centers, bins["observed_density"].to_numpy(float), width=widths, color=_METRIC_COLORS[metric], alpha=0.72, edgecolor="none", align="center", zorder=2)
    axis.set_xlim(-0.45, 1.0)
    axis.set_ylim(bottom=0)
    if metric == "place":
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    axis.set_xlabel("Correlation")
    axis.set_ylabel("Density")
    shuffle_xy = {"place": (0.42, 0.89), "pv": (0.05, 0.83), "rate": (0.05, 0.83)}[metric]
    data_xy = {"place": (0.74, 0.33), "pv": (0.61, 0.30), "rate": (0.42, 0.32)}[metric]
    axis.text(*shuffle_xy, "Shuffle", transform=axis.transAxes, ha="left", va="top", fontsize=5.4, color="#6f6f6f", fontweight="bold")
    axis.text(*data_xy, "Data", transform=axis.transAxes, ha="left", va="bottom" if metric == "rate" else "center", fontsize=5.4, color=_METRIC_COLORS[metric], fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4} if metric == "rate" else None)
    _style_axis(axis)


def _draw_mean_shuffle(axis: plt.Axes, values: pd.DataFrame, metric: str) -> None:
    """Draw the shuffled mean distribution and observed mean."""

    null = values.loc[values["record_type"].eq("shuffle_repeat"), "value"].to_numpy(float)
    observed = float(values.loc[values["record_type"].eq("observed_mean"), "value"].iloc[0])
    axis.hist(null[np.isfinite(null)], bins=60, color="#6f6f6f", alpha=0.78, edgecolor="none")
    axis.axvline(observed, color=_METRIC_COLORS[metric], linestyle=(0, (4, 2)), linewidth=1.60)
    axis.set_xlim(*{"place": (-0.012, 0.13), "pv": (-0.025, 0.25), "rate": (-0.055, 0.62)}[metric])
    axis.set_xlabel("Mean correlation")
    axis.set_ylabel("Count")
    axis.text(0.13, 0.78, "Shuffle", transform=axis.transAxes, ha="left", va="top", fontsize=5.4, color="#6f6f6f", fontweight="bold")
    axis.text(0.58, 0.52, "Data", transform=axis.transAxes, ha="left", va="center", fontsize=5.4, color=_METRIC_COLORS[metric], fontweight="bold")
    _style_axis(axis)


def _draw_gap_curve(axis: plt.Axes, values: pd.DataFrame, metric: str) -> None:
    """Draw correlation and shuffled expectation across recording gaps."""

    summary = values.loc[values["record_type"].eq("summary")].sort_values("time_days")
    x = summary["time_days"].to_numpy(float)
    y = summary["observed"].to_numpy(float)
    low = summary["ci95_low"].to_numpy(float)
    high = summary["ci95_high"].to_numpy(float)
    shuffle = summary["shuffle_mean"].to_numpy(float)
    color = _METRIC_COLORS[metric]
    axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    axis.plot(x, y, color=color, marker="o", markersize=2.4, linewidth=1.25, label="Data")
    axis.plot(x, shuffle, color="#777777", marker="s", markersize=1.9, linestyle=(0, (2, 1.4)), linewidth=0.75, label="Shuffle")
    axis.legend(loc="upper right", frameon=False, fontsize=5.4, handlelength=1.2, handletextpad=0.35, borderaxespad=0.10, labelspacing=0.20)
    axis.set_xlim(-8, 370)
    axis.set_ylim(-0.16 if metric == "rate" else -0.08, 0.84)
    axis.set_xticks((1, 90, 180, 270, 360), ("1", "90", "180", "270", "360"))
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.set_xlabel("Days apart")
    axis.set_ylabel("Correlation")
    _style_axis(axis)


def _draw_recurrence(
    axis: plt.Axes,
    points: pd.DataFrame,
    statistic: pd.Series,
    display_values: pd.DataFrame,
    metric: str,
) -> None:
    """Draw one successive-gap recurrence relationship."""

    x = points["previous_correlation"].to_numpy(float)
    y = points["next_correlation"].to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    color = _METRIC_COLORS[metric]
    limits = {"place": (-0.50, 1.0), "pv": (-0.05, 0.85), "rate": (-0.10, 0.80)}[metric]
    if metric == "rate":
        axis.scatter(x, y, s=15.75, facecolor=color, edgecolor="#2f4146", linewidth=0.45, alpha=0.72, zorder=2)
        if not display_values.empty:
            axis.fill_between(display_values["previous_correlation"], display_values["ci95_low"], display_values["ci95_high"], color=color, alpha=0.16, linewidth=0, zorder=1)
            axis.plot(display_values["previous_correlation"], display_values["next_correlation"], color=color, linewidth=1.50, zorder=3)
        axis.text(0.94, 0.08, f"r = {float(statistic['pearson_r']):.2f}", transform=axis.transAxes, ha="right", va="bottom", fontsize=5.4, color=color, fontweight="bold")
        axis.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))
        axis.set_yticks((-0.1, 0.0, 0.2, 0.4, 0.6, 0.8))
    else:
        many_points = x.size > 5000
        axis.scatter(x, y, s=4.0 if many_points else 7.0, color=color, alpha=0.035 if many_points else 0.115, edgecolors="none", rasterized=True)
        if len(display_values) >= 2:
            axis.plot(display_values["previous_correlation"], display_values["next_correlation"], color=color, linewidth=1.50)
        if metric == "place":
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    q_value = float(statistic["p_bh4"])
    axis.text(0.08, 0.91, _star_label(q_value), transform=axis.transAxes, ha="left", va="top", fontsize=11.745, color="black", fontweight="bold")
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_xlabel("Previous gap correlation")
    axis.set_ylabel("Next gap correlation")
    _style_axis(axis)


def _render_quantitative_panel(
    *,
    metric: str,
    column: int,
    size: tuple[float, float],
    distributions: Mapping[str, pd.DataFrame],
    mean_shuffles: Mapping[str, pd.DataFrame],
    gap_curves: Mapping[str, pd.DataFrame],
    recurrence_points: Mapping[str, pd.DataFrame],
    recurrence_statistics: Mapping[str, pd.Series],
    recurrence_displays: Mapping[str, pd.DataFrame],
) -> plt.Figure:
    """Render one metric and quantitative-column combination."""

    figure, axis = plt.subplots(figsize=size)
    if column == 0:
        _draw_distribution(axis, distributions[metric], metric)
    elif column == 1:
        _draw_mean_shuffle(axis, mean_shuffles[metric], metric)
    elif column == 2:
        _draw_gap_curve(axis, gap_curves[metric], metric)
    elif column == 3:
        _draw_recurrence(axis, recurrence_points[metric], recurrence_statistics[metric], recurrence_displays[metric], metric)
    else:
        raise ValueError(f"Unknown quantitative column: {column}")
    # The immutable panel SVGs have 1.2 pt of outer padding.  With the current
    # Matplotlib layout engine this corresponds to 0.20 font-size units.
    figure.tight_layout(pad=0.20)
    if column < 3:
        # Fix accepted frame margins; tight_layout left/bottom vary with the
        # renderer's text metrics.  Right/top already reproduce the page.
        left_pt, bottom_pt = {
            0: (16.13875, 17.14),
            1: (13.5, 17.14),
            2: (22.45, 23.30),
        }[column]
        width_pt, height_pt = np.asarray(size) * 72.0
        figure.subplots_adjust(left=left_pt / width_pt, bottom=bottom_pt / height_pt)
    if column == 3:
        # Shared data frame for g/l/q; margins are points within the fixed slot.
        width_pt, height_pt = np.asarray(size) * 72.0
        figure.subplots_adjust(
            left=25.6 / width_pt, right=1.0 - 4.935 / width_pt,
            bottom=19.3 / height_pt, top=1.0 - 3.0 / height_pt,
        )
    return figure


def _svg_viewbox(root: ET.Element) -> tuple[float, float]:
    """Return an SVG root's width and height in viewBox units."""

    values = root.attrib["viewBox"].replace(",", " ").split()
    return float(values[2]), float(values[3])


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    """Prefix SVG identifiers and update internal references."""

    identifiers: dict[str, str] = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier:
            renamed = f"{prefix}_{identifier}"
            identifiers[identifier] = renamed
            element.attrib["id"] = renamed
    for element in root.iter():
        for name, value in tuple(element.attrib.items()):
            for original, renamed in identifiers.items():
                value = value.replace(f"#{original}", f"#{renamed}")
            element.attrib[name] = value


def _capture_panel(figure: plt.Figure, identifier: str, *, raster_dpi: int) -> _PanelAsset:
    """Capture one fresh panel as SVG children plus PNG and PDF bytes."""

    figure.set_dpi(raster_dpi)
    svg_buffer = io.StringIO()
    png_buffer = io.BytesIO()
    pdf_buffer = io.BytesIO()
    save_options = {"bbox_inches": None, "pad_inches": 0.0, "transparent": True}
    savefig_reproducibly(figure, svg_buffer, format="svg", dpi=raster_dpi, **save_options)
    savefig_reproducibly(figure, png_buffer, format="png", dpi=raster_dpi, **save_options)
    savefig_reproducibly(figure, pdf_buffer, format="pdf", dpi=raster_dpi, **save_options)
    plt.close(figure)
    root = ET.fromstring(svg_buffer.getvalue())
    _prefix_svg_ids(root, identifier)
    width, height = _svg_viewbox(root)
    children = [child for child in list(root) if child.tag != f"{_SVG_TAG}metadata"]
    return _PanelAsset(children=children, width_points=width, height_points=height, png=png_buffer.getvalue(), pdf=pdf_buffer.getvalue())


def _placed_geometry(asset: _PanelAsset, slot: Sequence[float]) -> tuple[float, float, float, float, float]:
    """Scale one panel asset to fit its inch-based page slot."""

    x, y, width, height = (float(value) * 72.0 for value in slot)
    scale = min(width / asset.width_points, height / asset.height_points)
    return x, y, asset.width_points * scale, asset.height_points * scale, scale


def _append_svg_panel(root: ET.Element, asset: _PanelAsset, identifier: str, slot: Sequence[float]) -> None:
    """Append one scaled panel asset to the assembled SVG page."""

    x, y, _width, _height, scale = _placed_geometry(asset, slot)
    group = ET.SubElement(root, f"{_SVG_TAG}g", {"id": f"panel_{identifier}", "transform": f"translate({x:.6f} {y:.6f}) scale({scale:.8f})"})
    for child in asset.children:
        group.append(copy.deepcopy(child))


def _add_svg_label(root: ET.Element, label: str, x_inches: float, y_inches: float, *, anchor: str | None = None) -> None:
    """Add one panel letter to the assembled SVG page."""

    attributes = {"x": f"{x_inches * 72.0:.6f}", "y": f"{y_inches * 72.0:.6f}", "font-family": figure_font_family(), "font-size": "10.5", "font-weight": "bold", "fill": "black"}
    if anchor is not None:
        attributes["text-anchor"] = anchor
    text = ET.SubElement(root, f"{_SVG_TAG}text", attributes)
    text.text = label


def _panel_slots() -> tuple[dict[str, tuple[float, float, float, float]], dict[str, tuple[float, float, str | None]]]:
    """Return accepted page slots and panel-letter positions."""

    slots: dict[str, tuple[float, float, float, float]] = {"context": _TOP_SLOTS["context"], "examples": _EXAMPLE_SLOT}
    labels: dict[str, tuple[float, float, str | None]] = {
        "a": (0.015, 0.17 + _TOP_ROW_SHIFT_INCHES, None),
        "b": (1.475, 0.17 + _TOP_ROW_SHIFT_INCHES, None),
    }
    letters = iter("cdefghijklmnopq")
    for row, metric in enumerate(_METRICS):
        row_y, row_height = _LOWER_ROWS[row]
        schematic_id = f"{metric}_schematic"
        slots[schematic_id] = (_LOWER_LEFT, row_y, _SCHEMATIC_WIDTH, row_height)
        schematic_letter = next(letters)
        labels[schematic_letter] = (_LOWER_LEFT + 0.010, row_y + 0.030, None)
        for column in range(4):
            x_start, width = _LOWER_COLUMNS[column]
            identifier = f"{metric}_{column}"
            slots[identifier] = (x_start, row_y, width, row_height)
            letter = next(letters)
            if column == 1:
                labels[letter] = (x_start - 0.015, row_y + 0.030, "end")
            else:
                labels[letter] = (x_start + 0.010, row_y + 0.030, None)
    return slots, labels


def _write_svg(assets: Mapping[str, _PanelAsset], destination: Path) -> None:
    """Assemble panel assets and labels into the final SVG page."""

    ET.register_namespace("", _SVG_NAMESPACE)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    width_points = _CANVAS_SIZE_INCHES[0] * 72.0
    height_points = _CANVAS_SIZE_INCHES[1] * 72.0
    height_mm = _EXPORT_WIDTH_MM * _CANVAS_SIZE_INCHES[1] / _CANVAS_SIZE_INCHES[0]
    root = ET.Element(f"{_SVG_TAG}svg", {"width": f"{_EXPORT_WIDTH_MM:.3f}mm", "height": f"{height_mm:.3f}mm", "viewBox": f"0 0 {width_points:.6f} {height_points:.6f}", "version": "1.1"})
    ET.SubElement(root, f"{_SVG_TAG}rect", {"width": "100%", "height": "100%", "fill": "white"})
    slots, labels = _panel_slots()
    for identifier, slot in slots.items():
        _append_svg_panel(root, assets[identifier], identifier, slot)
    for label, (x_inches, y_inches, anchor) in labels.items():
        _add_svg_label(root, label, x_inches, y_inches, anchor=anchor)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def _export_svg_with_browser(
    svg_path: Path,
    png_path: Path,
    pdf_path: Path,
    *,
    browser: Path,
    raster_dpi: int,
) -> None:
    """Rasterize and print the assembled SVG with the accepted browser path.

    The HTML wrapper and browser profile are temporary, so the repository keeps
    only the requested SVG, PDF, and PNG outputs. The accepted page was exported
    through Chromium; without a browser the caller skips this export rather
    than substituting a visually different renderer.
    """

    width_mm = _EXPORT_WIDTH_MM
    height_mm = width_mm * _CANVAS_SIZE_INCHES[1] / _CANVAS_SIZE_INCHES[0]
    css_width = max(1, int(round(width_mm / 25.4 * 96.0)))
    css_height = max(1, int(round(height_mm / 25.4 * 96.0)))
    device_scale = max(0.1, float(raster_dpi) / 96.0)
    svg_text = svg_path.read_text(encoding="utf-8")
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: {width_mm:.6f}mm {height_mm:.6f}mm; margin: 0; }}
html, body {{ width: {width_mm:.6f}mm; height: {height_mm:.6f}mm; margin: 0; padding: 0; background: white; overflow: hidden; }}
svg {{ display: block; width: {width_mm:.6f}mm !important; height: {height_mm:.6f}mm !important; }}
* {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
</style></head><body>{svg_text}</body></html>
"""
    with tempfile.TemporaryDirectory(
        prefix=".maze_figure_2_export_", dir=png_path.parent
    ) as temporary:
        temporary_path = Path(temporary)
        html_path = temporary_path / "Figure_2.html"
        temporary_png = temporary_path / "Figure_2.png"
        temporary_pdf = temporary_path / "Figure_2.pdf"
        profile_path = temporary_path / "browser_profile"
        html_path.write_text(html_text, encoding="utf-8")
        common = (
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--user-data-dir={profile_path}",
        )
        subprocess.run(
            (
                *common,
                f"--window-size={css_width},{css_height}",
                f"--force-device-scale-factor={device_scale:.6f}",
                f"--screenshot={temporary_png}",
                html_path.as_uri(),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        wait_for_browser_file(temporary_png)
        expected_width = max(1, round(4250 * raster_dpi / 600))
        expected_height = max(1, round(4934 * raster_dpi / 600))
        with Image.open(temporary_png) as screenshot:
            screenshot = screenshot.convert("RGB")
            if screenshot.width < expected_width or screenshot.height < expected_height:
                raise RuntimeError(
                    "Browser screenshot is smaller than the accepted export canvas"
                )
            screenshot.crop((0, 0, expected_width, expected_height)).save(
                temporary_png,
                dpi=(raster_dpi, raster_dpi),
            )
        subprocess.run(
            (
                *common,
                "--no-pdf-header-footer",
                f"--print-to-pdf={temporary_pdf}",
                html_path.as_uri(),
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        wait_for_browser_file(temporary_pdf)
        fix_pdf_dates(temporary_pdf)
        os.replace(temporary_png, png_path)
        os.replace(temporary_pdf, pdf_path)


def render_representation_stability_summary(
    *,
    context_images: ContextImages,
    matched_neurons: MatchedNeuronExamples,
    schematic_values: Mapping[str, pd.DataFrame],
    distributions: Mapping[str, pd.DataFrame],
    mean_shuffles: Mapping[str, pd.DataFrame],
    gap_curves: Mapping[str, pd.DataFrame],
    recurrence_points: Mapping[str, pd.DataFrame],
    recurrence_statistics: Mapping[str, pd.Series],
    recurrence_displays: Mapping[str, pd.DataFrame],
    output_dir: Path,
    raster_dpi: int = 600,
) -> RepresentationStabilitySummaryResult:
    """Render the complete accepted page from calculations exposed in the notebook."""

    if set(schematic_values) != set(_METRICS):
        raise ValueError("schematic_values must contain place, pv, and rate")
    for name, tables in (
        ("distributions", distributions),
        ("mean_shuffles", mean_shuffles),
        ("gap_curves", gap_curves),
        ("recurrence_points", recurrence_points),
        ("recurrence_statistics", recurrence_statistics),
        ("recurrence_displays", recurrence_displays),
    ):
        if set(tables) != set(_METRICS):
            raise ValueError(f"{name} must contain place, pv, and rate")
    if raster_dpi <= 0:
        raise ValueError("raster_dpi must be positive")

    with plt.rc_context(_style_context()):
        assets: dict[str, _PanelAsset] = {
            "context": _capture_panel(_render_context_images(context_images), "context", raster_dpi=raster_dpi),
            "examples": _capture_panel(_render_matched_neurons(matched_neurons), "examples", raster_dpi=raster_dpi),
        }
        for row, metric in enumerate(_METRICS):
            row_height = _LOWER_ROWS[row][1]
            assets[f"{metric}_schematic"] = _capture_panel(
                _render_metric_schematic(schematic_values[metric], metric, (_SCHEMATIC_WIDTH, row_height)),
                f"{metric}_schematic",
                raster_dpi=raster_dpi,
            )
            for column in range(4):
                panel_width = _LOWER_COLUMNS[column][1]
                figure = _render_quantitative_panel(
                    metric=metric,
                    column=column,
                    size=(panel_width, row_height),
                    distributions=distributions,
                    mean_shuffles=mean_shuffles,
                    gap_curves=gap_curves,
                    recurrence_points=recurrence_points,
                    recurrence_statistics=recurrence_statistics,
                    recurrence_displays=recurrence_displays,
                )
                identifier = f"{metric}_{column}"
                assets[identifier] = _capture_panel(
                    figure, identifier, raster_dpi=raster_dpi
                )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "Figure_2.svg"
    pdf_path = output_dir / "Figure_2.pdf"
    png_path = output_dir / "Figure_2.png"
    with tempfile.TemporaryDirectory(
        prefix=".representation_summary_export_", dir=output_dir
    ) as temporary:
        temporary_dir = Path(temporary)
        temporary_svg = temporary_dir / svg_path.name
        temporary_pdf = temporary_dir / pdf_path.name
        temporary_png = temporary_dir / png_path.name
        _write_svg(assets, temporary_svg)
        export_renderer = chromium_export_browser()
        if export_renderer is None:
            os.replace(temporary_svg, svg_path)
            report_skipped_browser_export(
                "Figure 2", skipped=(pdf_path, png_path), written=(svg_path,)
            )
            return RepresentationStabilitySummaryResult((svg_path,), None)
        _export_svg_with_browser(
            temporary_svg,
            temporary_png,
            temporary_pdf,
            browser=export_renderer,
            raster_dpi=raster_dpi,
        )
        for source, destination in (
            (temporary_svg, svg_path),
            (temporary_pdf, pdf_path),
            (temporary_png, png_path),
        ):
            os.replace(source, destination)
    return RepresentationStabilitySummaryResult(
        (svg_path, pdf_path, png_path), export_renderer
    )


__all__ = [
    "RepresentationStabilitySummaryResult",
    "plot_representation_metric_schematic",
    "render_representation_stability_summary",
    "representation_metric_schematic_values",
]
