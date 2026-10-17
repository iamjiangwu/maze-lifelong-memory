"""Figure 3 drawing from saved plot values and example coordinates."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Sequence
from types import SimpleNamespace
import math
import itertools
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from release_rendering.statistics import cr1_ols, linear_contrast, benjamini_hochberg, sem
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from PIL import Image
from release_rendering.plotting import close_figure, declare_svg_canvas_size, despine, export_svg_with_chromium, publication_style, save_figure, version_record_style, draw_water_port_icon
from release_rendering.common import significance_text, validate_output_request as _validate_output_request
from release_rendering.types import FigureAssemblyResult, RunMode
from release_rendering.maze import build_maze
REPOSITORY_ROOT=Path(__file__).resolve().parents[1]
GRID_SIZE=15
_significance_mark=significance_text
DISPLAY_METRICS = (
    "place_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)

def fit_band(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an ordinary least-squares line and its descriptive 95% band."""

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    if x.size < 3 or np.ptp(x) <= 0:
        return line_x, np.full(100, np.nan), np.full(100, np.nan)
    design = np.column_stack((np.ones(x.size), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    line_design = np.column_stack((np.ones(line_x.size), line_x))
    line_y = line_design @ beta
    residual = y - design @ beta
    variance = float(np.sum(residual**2) / (x.size - 2))
    covariance = variance * np.linalg.inv(design.T @ design)
    line_se = np.sqrt(np.einsum("ij,jk,ik->i", line_design, covariance, line_design))
    critical = float(stats.t.ppf(0.975, df=x.size - 2))
    return line_x, line_y, critical * line_se
SIMILARITY_METRIC_LABELS = {
    "place_map_correlation": "Place-map correlation",
    "level_map_correlation": "Maze-depth-map correlation",
    "pv_correlation": "Population-vector correlation",
    "ensemble_rate_correlation": "Ensemble-rate correlation",
}

EXPERIENCE_GROUP_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")

EXPERIENCE_GROUP_COLORS = {
    "place_map_correlation": {
        "naive": "#D89C98",
        "3mo": "#CB817D",
        "6mo": "#BF6965",
        "9mo": "#A84E4B",
        "12mo": "#78312F",
    },
    "pv_correlation": {
        "naive": "#EDBD8F",
        "3mo": "#E8A76F",
        "6mo": "#E29957",
        "9mo": "#BD7137",
        "12mo": "#8A4A1F",
    },
    "ensemble_rate_correlation": {
        "naive": "#9FB0B5",
        "3mo": "#7D9198",
        "6mo": "#637980",
        "9mo": "#4A5E65",
        "12mo": "#2C3A3F",
    },
}

def experience_group(cycle_index: int) -> str:
    """Convert recording cycle index to its reader facing label."""

    labels = {0: "naive", 1: "3mo", 2: "6mo", 3: "9mo", 4: "12mo"}
    try:
        return labels[int(cycle_index)]
    except KeyError as error:
        raise ValueError(f"Unsupported Figure 3 cycle index: {cycle_index}") from error

def group_curve_values(curves, metric):
    """Select the saved group curve; no recording processing is needed."""
    return curves.loc[curves.metric.eq(metric)].copy()

_METRIC_COLORS = {
    "place_map_correlation": "#B95A58",
    "pv_correlation": "#E29957",
    "ensemble_rate_correlation": "#4A5E65",
}

_Y_LIMITS = {
    "place_map_correlation": (0.0, 0.65),
    "pv_correlation": (0.0, 0.75),
    "ensemble_rate_correlation": (0.0, 0.90),
}

_Y_TICKS = {
    "place_map_correlation": (0.0, 0.2, 0.4, 0.6),
    "pv_correlation": (0.0, 0.2, 0.4, 0.6),
    "ensemble_rate_correlation": (0.2, 0.4, 0.6, 0.8),
}

_PRESENTATION_VIEWBOX_POINTS = (353.006063, 486.8)

_PRESENTATION_SIZE_INCHES = (
    _PRESENTATION_VIEWBOX_POINTS[0] / 72.0,
    _PRESENTATION_VIEWBOX_POINTS[1] / 72.0,
)

_PRESENTATION_EXPORT_SIZE_MM = (136.0, 187.545787)

_PRESENTATION_PDF_SIZE_POINTS = (385.91998291015625, 532.0800170898438)

_PRESENTATION_RASTER_SIZE_600 = (3216, 4434)

_PRESENTATION_SLOTS = {
    "registered_activity": (0.0, 0.0, 1.0, 0.20092133114215283),
    "moving_window_definition": (
        0.005510823195124555,
        0.20589338126540674,
        0.32594095699710407,
        0.1068200493015612,
    ),
    "early_alignment_definition": (
        0.3184353408683522,
        0.20589338126540674,
        0.32594095699710407,
        0.1068200493015612,
    ),
    "successive_cycle_definition": (
        0.6633478898632968,
        0.20589338126540674,
        0.32594095699710407,
        0.1068200493015612,
    ),
    "place_map_convergence": (
        0.005510823195124555,
        0.5507988866064092,
        0.32594095699710407,
        0.2111192173377157,
    ),
    "place_map_early_alignment": (
        0.3184353408683522,
        0.5507988866064092,
        0.32594095699710407,
        0.2111192173377157,
    ),
    "place_map_successive_cycle": (
        0.6633478898632968,
        0.5507988866064092,
        0.32594095699710407,
        0.2111192173377157,
    ),
    "population_vector_convergence": (
        0.0029106582228872364,
        0.32401170501232535,
        0.33114106314938846,
        0.2144875102711586,
    ),
    "population_vector_early_alignment": (
        0.3184353408683522,
        0.32401170501232535,
        0.33114106314938846,
        0.2144875102711586,
    ),
    "population_vector_successive_cycle": (
        0.6633478898632968,
        0.32401170501232535,
        0.33114106314938846,
        0.2144875102711586,
    ),
    "ensemble_rate_convergence": (
        0.0017764199137848802,
        0.781283685291701,
        0.33227530429130336,
        0.21663959942481512,
    ),
    "ensemble_rate_early_alignment": (
        0.3173011025592498,
        0.781283685291701,
        0.33227530429130336,
        0.21663959942481512,
    ),
    "ensemble_rate_successive_cycle": (
        0.6633478898632968,
        0.781283685291701,
        0.33227530429130336,
        0.21663959942481512,
    ),
}

_PRESENTATION_LABELS = {
    "registered_activity": "a",
    "moving_window_definition": "b",
    "early_alignment_definition": "c",
    "successive_cycle_definition": "d",
    "place_map_convergence": "h",
    "place_map_early_alignment": "i",
    "place_map_successive_cycle": "j",
    "population_vector_convergence": "e",
    "population_vector_early_alignment": "f",
    "population_vector_successive_cycle": "g",
    "ensemble_rate_convergence": "k",
    "ensemble_rate_early_alignment": "l",
    "ensemble_rate_successive_cycle": "m",
}

_PRESENTATION_METRIC_ROWS = {
    "place_map_correlation": (
        "place_map_convergence",
        "place_map_early_alignment",
        "place_map_successive_cycle",
    ),
    "pv_correlation": (
        "population_vector_convergence",
        "population_vector_early_alignment",
        "population_vector_successive_cycle",
    ),
    "ensemble_rate_correlation": (
        "ensemble_rate_convergence",
        "ensemble_rate_early_alignment",
        "ensemble_rate_successive_cycle",
    ),
}

_EXPERIENCE_DISPLAY = {
    "naive": "naive",
    "3mo": "3 mo",
    "6mo": "6 mo",
    "9mo": "9 mo",
    "12mo": "12 mo",
}

_PRESENTATION_LABEL_OFFSETS_POINTS = {
    "a": (-7.0, -5.0),
    "b": (-7.0, -2.0),
    "c": (-1.8, -2.0),
    "d": (-7.0, -2.0),
    "e": (-7.0, -5.0),
    "f": (-1.8, -5.0),
    "g": (-7.0, -5.0),
    "h": (-2.0, -5.0),
    "i": (3.2, -5.0),
    "j": (-7.0, -5.0),
    "k": (-7.0, -5.0),
    "l": (3.2, -5.0),
    "m": (-7.0, -5.0),
}

@dataclass(frozen=True, slots=True)
class ReturnSessionPresentationResult:
    """Paths produced by the semantic return-session presentation."""

    run_mode: RunMode
    output_paths: tuple[Path, ...]

    def show(self) -> None:
        """Display the raster preview and link the editable exports."""

        try:
            from IPython.display import FileLink, Image as NotebookImage, display
        except ImportError as error:
            raise RuntimeError(
                "Notebook display requires the optional notebook dependencies"
            ) from error
        for relative_path in self.output_paths:
            absolute_path = (REPOSITORY_ROOT / relative_path).resolve()
            if absolute_path.suffix.casefold() == ".png":
                display(NotebookImage(filename=str(absolute_path)))
            elif absolute_path.suffix.casefold() in {".svg", ".pdf", ".tiff"}:
                display(FileLink(str(absolute_path)))

def _draw_supported_curve_runs(
    axis: plt.Axes,
    selected: pd.DataFrame,
    *,
    color: str,
    linewidth: float,
    band_alpha: float,
    label: str | None = None,
) -> None:
    """Join adjacent supported time points without bridging a missing section."""

    ordered = selected.sort_values("x_minutes", kind="stable")
    x_values = ordered["x_minutes"].to_numpy(dtype=float)
    means = ordered["plot_mean"].to_numpy(dtype=float)
    errors = ordered["plot_sem"].to_numpy(dtype=float)
    retained = ordered["retained_for_plot"].to_numpy(dtype=bool)
    supported = retained & np.isfinite(x_values) & np.isfinite(means)
    starts = np.flatnonzero(supported & np.r_[True, ~supported[:-1]])
    stops = np.flatnonzero(supported & np.r_[~supported[1:], True]) + 1
    first_run = True
    for start, stop in zip(starts, stops, strict=True):
        indices = np.arange(int(start), int(stop))
        if np.isfinite(errors[indices]).any():
            axis.fill_between(
                x_values[indices],
                means[indices] - errors[indices],
                means[indices] + errors[indices],
                color=color,
                alpha=band_alpha,
                linewidth=0,
            )
        segments = [
            np.asarray(
                [
                    [x_values[index], means[index]],
                    [x_values[index + 1], means[index + 1]],
                ],
                dtype=float,
            )
            for index in indices[:-1]
        ]
        if segments:
            lines = LineCollection(
                segments,
                colors=[color],
                linewidths=linewidth,
                label=label if first_run else "_nolegend_",
                zorder=3,
            )
            lines.set_capstyle("round")
            axis.add_collection(lines)
            first_run = False

def _triangle_head(
    axis: plt.Axes,
    body_x: float,
    body_y: float,
    nose_x: float,
    nose_y: float,
) -> None:
    """Draw one compact triangular activity-direction marker."""

    vector = 0.25 * np.asarray([nose_x - body_x, nose_y - body_y], dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return
    direction = vector / norm
    normal = np.asarray([-direction[1], direction[0]])
    tip = np.asarray([body_x, body_y]) + vector
    base = tip - 0.34 * direction
    points = np.vstack((tip, base + 0.07 * normal, base - 0.07 * normal))
    axis.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor="red",
            edgecolor="red",
            alpha=0.42,
            linewidth=0.01,
            zorder=2,
        )
    )

def _presentation_axis(
    figure: Figure,
    semantic_slot: str,
    *,
    left_fraction: float = 0.0,
    right_fraction: float = 1.0,
    bottom_fraction: float = 0.0,
    top_fraction: float = 1.0,
) -> plt.Axes:
    """Add an axis inside one immutable semantic presentation slot."""

    x_position, top_position, width, height = _PRESENTATION_SLOTS[semantic_slot]
    return figure.add_axes(
        [
            x_position + width * left_fraction,
            1.0 - top_position - height + height * bottom_fraction,
            width * (right_fraction - left_fraction),
            height * (top_fraction - bottom_fraction),
        ]
    )

def _draw_registered_activity_presentation(
    figure: Figure,
    activity: Iterable[RegisteredNeuronActivity],
) -> None:
    """Draw the fixed registered-neuron example directly on the final canvas."""

    activity = tuple(activity)
    if len(activity) != 4:
        raise ValueError("The registered neuron example requires four sessions")
    x_position, top_position, width, height = _PRESENTATION_SLOTS["registered_activity"]
    map_left = width * 0.005
    map_span = width * (0.995 - 0.005)
    subgrid_wspace = 0.04
    map_width = map_span / (4.0 + 3.0 * subgrid_wspace)
    gap = map_width * subgrid_wspace
    map_bottom = 1.0 - top_position - height * 0.98
    map_height = height * 0.86
    for index, values in enumerate(activity):
        axis = figure.add_axes(
            [
                x_position + map_left + index * (map_width + gap),
                map_bottom,
                map_width,
                map_height,
            ]
        )
        axis.plot(
            values.trajectory_x,
            values.trajectory_y,
            color="0.55",
            linewidth=0.22,
            alpha=0.24,
            rasterized=True,
            zorder=1,
        )
        for body_x, body_y, nose_x, nose_y in zip(
            values.body_x,
            values.body_y,
            values.nose_x,
            values.nose_y,
            strict=True,
        ):
            _triangle_head(axis, body_x, body_y, nose_x, nose_y)
        axis.add_patch(
            Rectangle(
                (-0.5, -0.5),
                15.0,
                15.0,
                fill=False,
                edgecolor="black",
                linewidth=1.15,
                clip_on=False,
                zorder=5,
            )
        )
        axis.set_xlim(-0.5, 14.5)
        axis.set_ylim(-0.5, 14.5)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(values.display_day, pad=3.5, fontsize=6.2)
        despine(axis, keep=())

def _draw_method_definition(axis: plt.Axes, semantic_definition: str) -> None:
    """Draw one accepted compact method definition without a result curve."""

    accepted_titles = {
        "moving_window_definition": "Moving 30-min windows",
        "early_alignment_definition": "First 30 min across cycles",
        "successive_cycle_definition": "Day 1 across long gaps",
    }
    if semantic_definition not in accepted_titles:
        raise ValueError(f"Unknown method definition: {semantic_definition}")
    colors = {
        "ink": "#30363A",
        "muted": "#667168",
        "bar_fill": "#F1F6F1",
        "bar_edge": "#91A493",
        "whole_fill": "#75A86F",
        "whole_edge": "#376F3E",
        "window_edge": "#3F8048",
        "arrow": "#3F8048",
        "connector": "#B3BEB4",
    }
    window_colors = ("#DCEBDC", "#B8D3B8", "#8FB791")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    linewidth = 0.72

    def add_bar(
        x_position: float,
        y_position: float,
        width: float,
        height: float,
        *,
        complete_session: bool = False,
    ) -> None:
        """Draw one compact day or complete-session bar."""

        axis.add_patch(
            Rectangle(
                (x_position, y_position),
                width,
                height,
                facecolor=(
                    colors["whole_fill"] if complete_session else colors["bar_fill"]
                ),
                edgecolor=(
                    colors["whole_edge"] if complete_session else colors["bar_edge"]
                ),
                linewidth=linewidth,
                zorder=2,
            )
        )

    def add_comparison_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        curvature: float,
        mutation_scale: float = 5.8,
    ) -> None:
        """Draw one curved comparison arrow between schematic bars."""

        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=linewidth,
                color=colors["arrow"],
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
            )
        )

    axis.text(
        0.5,
        0.965,
        accepted_titles[semantic_definition],
        ha="center",
        va="top",
        fontsize=6.2,
        color=colors["ink"],
        fontweight="bold",
    )
    if semantic_definition == "moving_window_definition":
        source_x, source_width = 0.07, 0.45
        reference_x, reference_width = 0.69, 0.24
        bar_y, bar_height = 0.34, 0.17
        add_bar(source_x, bar_y, source_width, bar_height)
        add_bar(
            reference_x,
            bar_y,
            reference_width,
            bar_height,
            complete_session=True,
        )
        axis.text(
            source_x,
            0.61,
            "Day i",
            ha="left",
            va="bottom",
            fontsize=4.9,
            color=colors["ink"],
        )
        axis.text(
            reference_x + reference_width,
            0.61,
            "Day i + 1",
            ha="right",
            va="bottom",
            fontsize=4.9,
            color=colors["ink"],
        )
        axis.add_patch(
            FancyArrowPatch(
                (source_x, 0.245),
                (source_x + source_width, 0.245),
                arrowstyle="-|>",
                mutation_scale=4.5,
                linewidth=0.62,
                color=colors["ink"],
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=4,
            )
        )
        axis.text(
            source_x + source_width / 2.0,
            0.145,
            "time",
            ha="center",
            va="top",
            fontsize=4.7,
            color=colors["muted"],
        )
        for start, curvature, color in zip(
            (0.115, 0.255, 0.395),
            (-0.32, -0.22, -0.12),
            window_colors,
            strict=True,
        ):
            axis.add_patch(
                Rectangle(
                    (start, bar_y),
                    0.105,
                    bar_height,
                    facecolor=color,
                    edgecolor=colors["window_edge"],
                    linewidth=linewidth,
                    zorder=3,
                )
            )
            add_comparison_arrow(
                (start + 0.0525, bar_y + bar_height + 0.005),
                (0.81, 0.525),
                curvature=curvature,
            )
        return

    group_positions = np.linspace(0.10, 0.815, 5)
    bar_y, bar_height, bar_width, gap = 0.36, 0.17, 0.060, 0.012
    for experience_index, x_position in enumerate(group_positions):
        if semantic_definition == "early_alignment_definition":
            add_bar(float(x_position), bar_y, bar_width, bar_height)
            add_bar(
                float(x_position + bar_width + gap),
                bar_y,
                bar_width,
                bar_height,
                complete_session=True,
            )
            highlight_width = bar_width * 0.32
            axis.add_patch(
                Rectangle(
                    (float(x_position), bar_y),
                    highlight_width,
                    bar_height,
                    facecolor=window_colors[min(experience_index // 2, 2)],
                    edgecolor=colors["window_edge"],
                    linewidth=linewidth,
                    zorder=3,
                )
            )
            add_comparison_arrow(
                (float(x_position + highlight_width / 2.0), bar_y + bar_height),
                (
                    float(x_position + bar_width + gap + bar_width / 2.0),
                    bar_y + bar_height,
                ),
                curvature=-0.43,
                mutation_scale=5.2,
            )
        else:
            add_bar(
                float(x_position),
                bar_y,
                bar_width,
                bar_height,
                complete_session=True,
            )
            add_bar(
                float(x_position + bar_width + gap),
                bar_y,
                bar_width,
                bar_height,
            )
        axis.text(
            float(x_position + bar_width + gap / 2.0),
            0.25,
            str(experience_index),
            ha="center",
            va="top",
            fontsize=4.7,
            color=colors["muted"],
        )
        if experience_index < len(group_positions) - 1:
            next_position = float(group_positions[experience_index + 1])
            axis.plot(
                [float(x_position + 2.0 * bar_width + gap), next_position],
                [bar_y + bar_height / 2.0] * 2,
                color=colors["connector"],
                linewidth=linewidth,
                linestyle=(0, (2.2, 2.2)),
                zorder=1,
            )
            if semantic_definition == "successive_cycle_definition":
                add_comparison_arrow(
                    (float(x_position + bar_width / 2.0), bar_y + bar_height),
                    (next_position + bar_width / 2.0, bar_y + bar_height),
                    curvature=-0.34,
                    mutation_scale=5.4,
                )
    if semantic_definition == "early_alignment_definition":
        first_position = float(group_positions[0])
        axis.text(
            first_position + bar_width / 2.0,
            0.62,
            "Day 1",
            ha="right",
            va="bottom",
            fontsize=4.7,
            color=colors["ink"],
        )
        axis.text(
            first_position + bar_width + gap + bar_width / 2.0,
            0.62,
            "Day 2",
            ha="left",
            va="bottom",
            fontsize=4.7,
            color=colors["ink"],
        )
        # Identify the first pair without crossing its curved comparison
        # arrow or moving either label or rectangle.
        for start, end in (
            ((first_position - 0.010, 0.605), (first_position, bar_y + bar_height)),
            ((first_position + 0.150, 0.605), (first_position + 2.0 * bar_width + gap, bar_y + bar_height)),
        ):
            axis.plot(
                [start[0], end[0]], [start[1], end[1]],
                color=colors["ink"], linewidth=0.45,
                solid_capstyle="butt", zorder=4,
            )
    axis.text(
        0.5,
        0.10,
        "No. of cycles" if semantic_definition == "early_alignment_definition" else "Cycle",
        ha="center",
        va="top",
        fontsize=4.9,
        color=colors["muted"],
    )

def _draw_convergence_presentation(
    axis: plt.Axes,
    within_cycle_stability: WithinResults,
    metric: str,
    *,
    aggregate_groups: bool,
) -> None:
    """Draw the accepted fixed-reference moving-window presentation."""

    if aggregate_groups:
        summary = group_curve_values(within_cycle_stability.curves, metric)
        for group in EXPERIENCE_GROUP_ORDER:
            selected = summary.loc[
                summary["experience_group"].eq(group)
            ]
            if selected.empty:
                continue
            _draw_supported_curve_runs(
                axis,
                selected,
                color=EXPERIENCE_GROUP_COLORS[metric][group],
                linewidth=1.25,
                band_alpha=0.07,
            )
    else:
        selected = within_cycle_stability.curves.loc[
            within_cycle_stability.curves["metric"].eq(metric)
        ].sort_values("x_value", kind="stable")
        axis.plot(
            selected["x_value"].to_numpy(dtype=float) / 60.0,
            selected["value"].to_numpy(dtype=float),
            color=_METRIC_COLORS[metric],
            linewidth=1.25,
        )
    axis.set_xlim(25.0, 185.0)
    axis.set_xticks((50, 100, 150))
    axis.set_ylim(*_Y_LIMITS[metric])
    axis.set_yticks(_Y_TICKS[metric])
    axis.set_xlabel("Time in maze (min)")
    axis.set_ylabel("PV correlation" if metric == "pv_correlation" else SIMILARITY_METRIC_LABELS[metric].replace("-", " "))
    despine(axis)

def _significance_symbol(adjusted_probability: float) -> str:
    """Return the accepted correlation-family significance display."""

    if not np.isfinite(adjusted_probability) or adjusted_probability >= 0.05:
        return "n.s."
    if adjusted_probability < 0.001:
        return "***"
    if adjusted_probability < 0.01:
        return "**"
    return "*"

def _draw_experience_presentation(
    axis: plt.Axes,
    points: pd.DataFrame,
    statistics: pd.DataFrame,
    metric: str,
    *,
    first_window: bool,
) -> None:
    """Draw one accepted experience association from calculated pair values."""

    selected = points.loc[points["metric"].eq(metric)].copy()
    selected["_animal_order"] = pd.to_numeric(
        selected["animal_id"], errors="raise"
    )
    if first_window:
        selected = selected.sort_values(
            ["cycle_index", "_animal_order"], kind="stable"
        )
        x_column = "cycle_index"
    else:
        selected = selected.sort_values(
            ["_animal_order", "experience"], kind="stable"
        )
        x_column = "experience"
    x_values = selected[x_column].to_numpy(dtype=float)
    y_values = selected["value"].to_numpy(dtype=float)
    metric_offset = DISPLAY_METRICS.index(metric)
    jitter = np.random.default_rng(240624 + metric_offset).normal(
        0.0, 0.045, size=len(selected)
    )
    point_colors = [
        EXPERIENCE_GROUP_COLORS[metric][
            experience_group(int(value))
            if first_window
            else {1: "3mo", 2: "6mo", 3: "9mo", 4: "12mo"}[int(value)]
        ]
        for value in x_values
    ]
    axis.scatter(
        x_values + jitter,
        y_values,
        color=point_colors,
        s=15.0,
        alpha=0.72,
        edgecolors="none",
        zorder=3,
    )
    if len(selected) >= 3 and np.unique(x_values[np.isfinite(x_values)]).size >= 2:
        line_x, line_y, band = fit_band(x_values, y_values)
        finite_band = np.isfinite(line_y) & np.isfinite(band)
        if finite_band.any():
            axis.fill_between(
                line_x[finite_band],
                line_y[finite_band] - band[finite_band],
                line_y[finite_band] + band[finite_band],
                color=_METRIC_COLORS[metric],
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        axis.plot(
            line_x,
            line_y,
            color=_METRIC_COLORS[metric],
            linewidth=1.10,
            zorder=2,
        )
    if first_window:
        axis.set_xlim(-0.45, 4.45)
        axis.set_xticks((0, 1, 2, 3, 4))
    else:
        axis.set_xlim(0.55, 4.45)
        axis.set_xticks((1, 2, 3, 4))
    axis.set_ylim(*_Y_LIMITS[metric])
    axis.set_yticks(_Y_TICKS[metric])
    axis.set_xlabel("No. of cycles" if first_window else "Cycle")
    axis.set_ylabel("")
    despine(axis)

def _add_presentation_group_legend(
    figure: Figure,
    semantic_slot: str,
    metric: str,
) -> None:
    """Draw the accepted slot-relative two-column experience legend."""

    x_position, top_position, width, height = _PRESENTATION_SLOTS[semantic_slot]
    page_width_points = figure.get_figwidth() * 72.0
    page_height_points = figure.get_figheight() * 72.0
    x0_points = (x_position + width) * page_width_points - 63.0
    y0_points = (top_position + height) * page_height_points - 50.0
    for column, groups in enumerate((("naive", "3mo", "6mo"), ("9mo", "12mo"))):
        for row, group in enumerate(groups):
            x_points = x0_points + column * 27.0
            y_points = y0_points + row * 6.4
            y_figure = 1.0 - y_points / page_height_points
            figure.add_artist(
                Line2D(
                    (
                        x_points / page_width_points,
                        (x_points + 5.5) / page_width_points,
                    ),
                    (y_figure, y_figure),
                    transform=figure.transFigure,
                    color=EXPERIENCE_GROUP_COLORS[metric][group],
                    linewidth=2.1,
                    solid_capstyle="round",
                    clip_on=False,
                )
            )
            figure.text(
                (x_points + 6.6) / page_width_points,
                y_figure,
                _EXPERIENCE_DISPLAY[group],
                ha="left",
                va="center",
                fontsize=5.6,
                color="black",
            )

def _presentation_stat_values(
    statistics: pd.DataFrame,
    metric: str,
) -> tuple[str, float] | None:
    """Return the existing significance call and correlation for one overlay."""

    stat_row = statistics.loc[statistics["metric"].eq(metric)]
    if stat_row.empty:
        return None
    correlation = float(stat_row["pearson_r"].iloc[0])
    adjusted_probability = float(stat_row["p_bh4"].iloc[0])
    if not np.isfinite(correlation) or not np.isfinite(adjusted_probability):
        return None
    return _significance_symbol(adjusted_probability), correlation

def _add_presentation_stat_overlays(
    figure: Figure,
    first_window_statistics: pd.DataFrame,
    successive_cycle_statistics: pd.DataFrame,
) -> None:
    """Place accepted page-level stars and correlations over scatter slots."""

    column_specs = (
        (1, first_window_statistics),
        (2, successive_cycle_statistics),
    )
    for column, statistics in column_specs:
        slots = [
            semantic_row[column] for semantic_row in _PRESENTATION_METRIC_ROWS.values()
        ]
        star_x = float(
            np.mean(
                [
                    _PRESENTATION_SLOTS[semantic_slot][0]
                    + _PRESENTATION_SLOTS[semantic_slot][2] * 0.315
                    for semantic_slot in slots
                ]
            )
        )
        for metric, semantic_row in _PRESENTATION_METRIC_ROWS.items():
            values = _presentation_stat_values(statistics, metric)
            if values is None:
                continue
            star, correlation = values
            x_position, top_position, width, height = _PRESENTATION_SLOTS[
                semantic_row[column]
            ]
            figure.text(
                star_x,
                1.0 - top_position - height * 0.035,
                star,
                ha="left",
                va="top",
                fontsize=6.0 if star == "n.s." else 11.7,
                color="black",
                fontweight="bold",
            )
            figure.text(
                x_position + width * 0.905,
                1.0 - top_position - height * 0.050,
                f"r={correlation:.2f}",
                ha="right",
                va="top",
                fontsize=5.4,
                color="black",
            )

def _add_presentation_label(figure: Figure, semantic_slot: str) -> None:
    """Place one accepted lower-case label beside its semantic slot."""

    x_position, top_position, _, _ = _PRESENTATION_SLOTS[semantic_slot]
    label = _PRESENTATION_LABELS[semantic_slot]
    x_offset_points, y_offset_points = _PRESENTATION_LABEL_OFFSETS_POINTS[label]
    page_width_points = figure.get_figwidth() * 72.0
    page_height_points = figure.get_figheight() * 72.0
    label_x_points = max(0.0, x_position * page_width_points + x_offset_points)
    if label in {"f", "i", "l"}:
        anchor_x = _PRESENTATION_SLOTS["place_map_early_alignment"][0]
        label_x_points = anchor_x * page_width_points + _PRESENTATION_LABEL_OFFSETS_POINTS["f"][0]
    label_y_points = max(0.0, top_position * page_height_points + y_offset_points)
    figure.text(
        label_x_points / page_width_points,
        1.0 - label_y_points / page_height_points,
        label,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="black",
    )

def present_return_session_reconstruction(
    *,
    registered_activity: Iterable[RegisteredNeuronActivity],
    within_cycle_stability: WithinResults,
    successive_cycle_stability: _LongResults,
    run_mode: RunMode,
    output_dir: Path,
    random_seed: int,
    raster_dpi: int = 600,
) -> ReturnSessionPresentationResult:
    """Compose the accepted semantic presentation from calculated stages."""

    output_dir = _validate_output_request(
        run_mode=run_mode,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    if isinstance(raster_dpi, bool) or not isinstance(raster_dpi, int):
        raise TypeError("raster_dpi must be an integer")
    if raster_dpi <= 0:
        raise ValueError("raster_dpi must be positive")
    accepted_style = {
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "axes.titlesize": 6.2,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
        "legend.fontsize": 4.9,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
    }
    with version_record_style(accepted_style):
        presentation = plt.figure(
            figsize=_PRESENTATION_SIZE_INCHES,
            facecolor="white",
            dpi=raster_dpi,
        )
        _draw_registered_activity_presentation(presentation, registered_activity)
        for semantic_definition in (
            "moving_window_definition",
            "early_alignment_definition",
            "successive_cycle_definition",
        ):
            definition_axis = _presentation_axis(presentation, semantic_definition)
            _draw_method_definition(definition_axis, semantic_definition)
        for metric, semantic_row in _PRESENTATION_METRIC_ROWS.items():
            convergence_axis = _presentation_axis(
                presentation,
                semantic_row[0],
                left_fraction=0.22,
                right_fraction=0.985,
                bottom_fraction=0.255,
                top_fraction=0.98,
            )
            _draw_convergence_presentation(
                convergence_axis,
                within_cycle_stability,
                metric,
                aggregate_groups=True,
            )
            early_alignment_axis = _presentation_axis(
                presentation,
                semantic_row[1],
                left_fraction=0.17,
                right_fraction=0.985,
                bottom_fraction=0.255,
                top_fraction=0.98,
            )
            _draw_experience_presentation(
                early_alignment_axis,
                within_cycle_stability.first_points,
                within_cycle_stability.first_stats,
                metric,
                first_window=True,
            )
            successive_cycle_axis = _presentation_axis(
                presentation,
                semantic_row[2],
                left_fraction=0.17,
                right_fraction=0.985,
                bottom_fraction=0.255,
                top_fraction=0.98,
            )
            _draw_experience_presentation(
                successive_cycle_axis,
                successive_cycle_stability.points,
                successive_cycle_stability.statistics,
                metric,
                first_window=False,
            )
            _add_presentation_group_legend(
                presentation,
                semantic_row[0],
                metric,
            )
        _add_presentation_stat_overlays(
            presentation,
            within_cycle_stability.first_stats,
            successive_cycle_stability.statistics,
        )
        for semantic_slot in _PRESENTATION_LABELS:
            _add_presentation_label(presentation, semantic_slot)
        output_stem = output_dir / "Figure_3"
        try:
            svg_path = save_figure(
                presentation,
                output_stem,
                formats=("svg",),
                raster_dpi=raster_dpi,
                transparent=False,
                preserve_canvas=True,
            )[0]
            declare_svg_canvas_size(
                svg_path,
                width_mm=_PRESENTATION_EXPORT_SIZE_MM[0],
                height_mm=_PRESENTATION_EXPORT_SIZE_MM[1],
                view_box=(0.0, 0.0, *_PRESENTATION_VIEWBOX_POINTS),
            )
            pdf_path = output_stem.with_suffix(".pdf")
            png_path = output_stem.with_suffix(".png")
            tiff_path = output_stem.with_suffix(".tiff")
            browser = export_svg_with_chromium(
                svg_path,
                png_path,
                pdf_path,
                width_mm=_PRESENTATION_EXPORT_SIZE_MM[0],
                height_mm=_PRESENTATION_EXPORT_SIZE_MM[1],
                raster_dpi=raster_dpi,
                tiff_path=tiff_path,
                expected_pdf_size_points=_PRESENTATION_PDF_SIZE_POINTS,
                expected_raster_size=tuple(
                    round(value * raster_dpi / 600.0)
                    for value in _PRESENTATION_RASTER_SIZE_600
                ),
                page_name="Figure 3",
            )
            output_paths = (
                (svg_path,)
                if browser is None
                else (svg_path, pdf_path, png_path, tiff_path)
            )
        finally:
            close_figure(presentation)
    root = REPOSITORY_ROOT.resolve()
    return ReturnSessionPresentationResult(
        run_mode=run_mode,
        output_paths=tuple(path.resolve().relative_to(root) for path in output_paths),
    )
