"""Assemble Extended Data Figure 6 (within-session convergence and local stability).

Every panel is redrawn from the included plot-ready tables. Time-window rows
require 300 s of selected-data coverage (7,500 frames at DT = 0.04 s).
Tile-change windows have no fractional time-coverage gate. The panels show
data, axes and labels; statistical results are exported in the caption notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ed_supplement_common import (  # noqa: E402
    AXIS_COLOR,
    MUTED_COLOR,
    SCHEMATIC_COLORS,
    Canvas,
    Checks,
    sha256,
)

# --------------------------------------------------------------------------- sources

COVERAGE_ROOT = REPO_ROOT / "docs" / "coverage5min_window_reanalysis_2026-08-05"
FIXED_DIR = COVERAGE_ROOT / "rendered" / "ed07_fixed_time_tile"
PIVOT_DIR = COVERAGE_ROOT / "rendered" / "ed07_pivot"
FIXED_CURVES = FIXED_DIR / "ed07_current59_group_curve_summary.csv"
FIXED_STATS = FIXED_DIR / "ed07_current59_clustered_stats.csv"
FIXED_SUPPORT = FIXED_DIR / "ed07_current59_curve_support_mask.csv"
PIVOT_CURVES = PIVOT_DIR / "ed07_pivot_experience_group_curve_summary_current59.csv"
PIVOT_STATS = (
    REPO_ROOT / "docs/figure_version_records/extended_data_fig06_2026-08-16_round7_available_curve_all59"
    / "source_data/available_curve_support_2026_08_16/ed06_available_curve_clustered_stats.csv"
)
PIVOT_LIMITS = PIVOT_DIR / "ed07_pivot_experience_display_limits_current59.csv"

FIGURE_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed06_within_session_convergence"
)
DEFAULT_OUT = FIGURE_ROOT / "assembly"
STEM = "ed06_within_session_convergence_assembly"

# --------------------------------------------------------------------------- contract

COVERAGE_GATE_SEC = 300.0
COVERAGE_GATE_FRAMES = 7500
EXPECTED_PAIRS = 59
EXPECTED_ANIMALS = 22
PIVOT_WINDOW_MIN = 30.0
PIVOT_EXPECTED_PAIR_ROWS = 59
FIXED_BH_FAMILY = 12
PIVOT_BH_FAMILY = 3

EXPERIENCE_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")
EXPERIENCE_LABEL = {
    "naive": "naive",
    "3mo": "3 mo",
    "6mo": "6 mo",
    "9mo": "9 mo",
    "12mo": "12 mo",
}
# Maintained semantic ramps from ca_analysis.figure_colors.get_correlation_experience_colors.
EXPERIENCE_COLORS = {
    "place_map_correlation": {
        "naive": "#d89c98",
        "3mo": "#cb817d",
        "6mo": "#bf6965",
        "9mo": "#a84e4b",
        "12mo": "#78312f",
    },
    "pv_correlation": {
        "naive": "#edbd8f",
        "3mo": "#e8a76f",
        "6mo": "#e29957",
        "9mo": "#bd7137",
        "12mo": "#8a4a1f",
    },
    "ensemble_rate_correlation": {
        "naive": "#9fb0b5",
        "3mo": "#7d9198",
        "6mo": "#637980",
        "9mo": "#4a5e65",
        "12mo": "#2c3a3f",
    },
}

# One y-axis title per metric, shared by that metric's fixed-template panel (c/e/g) and
# its pivot panel (d/f/h). The "(r)" qualifier is dropped: every displayed quantity on this
# page is a correlation, so the letter added nothing. What the two blocks correlate does
# differ -- c/e/g against the Day-2 template, d/f/h across a pivot -- and that distinction is
# carried by the block titles and the caption rather than by the axis title.
METRICS = (
    {
        "metric": "place_map_correlation",
        "metric_key": "place",
        "y_label": "Place map correlation",
    },
    {
        "metric": "pv_correlation",
        "metric_key": "pv",
        "y_label": "PV correlation",
    },
    {
        "metric": "ensemble_rate_correlation",
        "metric_key": "ensemble_rate",
        "y_label": "Ensemble rate correlation",
    },
)

# Axis titles follow Extended Data Figure 1's vocabulary so the two pages read as one
# supplement: elapsed maze time is "Time in maze (min)" and tile progress is
# "Maze tile transitions".
X_LABEL_TIME = "Time in maze (min)"
X_LABEL_TILES = "Maze tile transitions"

# Displayed fixed-template columns, in slide order.
FIXED_COLUMNS = (
    {
        "window_type": "maze_time",
        "window_size": 10.0,
        "title": "10-min window",
        "x_label": X_LABEL_TIME,
        "x_display_scale": 1.0,
        "slope_unit": "r/h",
    },
    {
        "window_type": "maze_time",
        "window_size": 20.0,
        "title": "20-min window",
        "x_label": X_LABEL_TIME,
        "x_display_scale": 1.0,
        "slope_unit": "r/h",
    },
    {
        "window_type": "tile_change",
        "window_size": 3000.0,
        "title": "3,000-tile window",
        # Adopting Extended Data Figure 1's "Maze tile transitions" means the axis has to
        # carry raw transitions: leaving x_plot in thousands under that title would read
        # "3" where the data say 3,000. The source column stays in thousands and is scaled
        # here, at the point of drawing, so the conversion is visible.
        "x_label": X_LABEL_TILES,
        "x_display_scale": 1000.0,
        "slope_unit": "r/1,000 tiles",
    },
)
# ``x_plot`` arrives in the source's own unit; ``x_display_scale`` converts it to the
# drawn axis unit (identity for maze time, thousands to transitions for tile progress).
EXPECTED_X_UNITS = {
    "maze_time": "in-maze minute",
    "tile_change": "thousand tile changes",
}
DRAWN_X_UNITS = {
    "maze_time": "in-maze minute",
    "tile_change": "maze tile transition",
}
DAY_I_LABEL = r"Day $\mathrm{i}$"
DAY_IP1_LABEL = r"Day $\mathrm{i}+1$"
PIVOT_COLUMNS = (
    {"day": "day1", "title": DAY_I_LABEL},
    {"day": "day2", "title": DAY_IP1_LABEL},
)

# --------------------------------------------------------------------------- geometry

FIGURE_WIDTH_IN = 7.5
FONT_REFERENCE_WIDTH_IN = 7.5

LEFT_MARGIN = 0.52
RIGHT_MARGIN = 0.10
SUBPANEL_W = 1.21
INNER_GAP = 0.10
BLOCK_GAP = 0.55

SCHEMATIC_H = 0.74
# The schematic box is filled to its lower edge, and the row legend carries its own
# padding, so the gap between them only needs to clear descenders.
SCHEMATIC_TO_ROWS = 0.05

N_METRIC_ROWS = 3
ROW_AXES_H = 1.00
ROW_LEGEND_H = 0.15
# Two lines, and **only on the top row**: the day identity sits above the window title so
# "30-min window" lands at the same height as "10-min window". Rows 2 and 3 repeat neither,
# so they reserve no title band at all — reserving one is what left the figure airy.
ROW_TITLE_H_TOP = 0.27
ROW_DAY_LINE_OFFSET = 0.155
# Keeps the row-wide legend clear of the panel letters, which hang from the axes top. The
# top row does not need it: its title band already separates the legend from the letters,
# and the clearance there is what left a visible band under the schematics.
ROW_LEGEND_GAP_TOP = 0.02
ROW_LEGEND_GAP_OTHER = 0.09
INTER_ROW_GAP = 0.16
BOTTOM_XLABEL_H = 0.30
TOP_PAD = 0.18


def _row_title_height(row_index: int) -> float:
    return ROW_TITLE_H_TOP if row_index == 0 else 0.0


def _row_legend_gap(row_index: int) -> float:
    return ROW_LEGEND_GAP_TOP if row_index == 0 else ROW_LEGEND_GAP_OTHER


_ROW_BLOCKS = tuple(
    ROW_LEGEND_H + _row_legend_gap(index) + _row_title_height(index) + ROW_AXES_H
    for index in range(N_METRIC_ROWS)
)
FIGURE_HEIGHT_IN = (
    TOP_PAD
    + SCHEMATIC_H
    + SCHEMATIC_TO_ROWS
    + sum(_ROW_BLOCKS)
    + (N_METRIC_ROWS - 1) * INTER_ROW_GAP
    + BOTTOM_XLABEL_H
)

FIXED_BLOCK_LEFT = LEFT_MARGIN
FIXED_BLOCK_W = 3.0 * SUBPANEL_W + 2.0 * INNER_GAP
PIVOT_BLOCK_LEFT = FIXED_BLOCK_LEFT + FIXED_BLOCK_W + BLOCK_GAP
PIVOT_BLOCK_W = 2.0 * SUBPANEL_W + INNER_GAP


def _fixed_column_left(index: int) -> float:
    return FIXED_BLOCK_LEFT + index * (SUBPANEL_W + INNER_GAP)


def _pivot_column_left(index: int) -> float:
    return PIVOT_BLOCK_LEFT + index * (SUBPANEL_W + INNER_GAP)


def _row_axes_bottom(row_index: int) -> float:
    """Bottom edge, in inches from the canvas bottom, of a metric row's axes."""

    row_top = FIGURE_HEIGHT_IN - TOP_PAD - SCHEMATIC_H - SCHEMATIC_TO_ROWS
    for index in range(row_index):
        row_top -= _ROW_BLOCKS[index] + INTER_ROW_GAP
    return (
        row_top
        - ROW_LEGEND_H
        - _row_legend_gap(row_index)
        - _row_title_height(row_index)
        - ROW_AXES_H
    )


# --------------------------------------------------------------------------- loading


def _repo_relative(path: Path) -> str:
    """Repo-relative when the path is inside the checkout, absolute otherwise (tests)."""

    try:
        return str(Path(path).relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(Path(path))


def _load_frames() -> dict[str, pd.DataFrame]:
    frames = {
        "fixed_curves": pd.read_csv(FIXED_CURVES),
        "fixed_stats": pd.read_csv(FIXED_STATS),
        "fixed_support": pd.read_csv(FIXED_SUPPORT),
        "pivot_curves": pd.read_csv(PIVOT_CURVES),
        "pivot_stats": pd.read_csv(PIVOT_STATS),
        "pivot_limits": pd.read_csv(PIVOT_LIMITS),
    }
    return frames


def _fixed_slope(stats: pd.DataFrame, metric: str, window_type: str, window_size: float):
    rows = stats[
        stats["metric"].eq(metric)
        & stats["window_type"].eq(window_type)
        & np.isclose(stats["window_size"], window_size)
        & stats["estimand"].eq("mean_pairwise_within_day1_slope")
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one slope row for {metric}/{window_type}/{window_size}, got {len(rows)}"
        )
    return rows.iloc[0]


def _pivot_slope(stats: pd.DataFrame, metric: str, day: str):
    estimand = {"day1": "day_i_linear_slope", "day2": "day_i1_linear_slope"}[day]
    rows = stats[
        stats["metric"].eq(metric)
        & stats["estimand"].eq(estimand)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one pivot slope row for {metric}/{day}, got {len(rows)}")
    return rows.iloc[0]


def _curve_slice(
    curves: pd.DataFrame, metric: str, window_type: str, window_size: float
) -> pd.DataFrame:
    frame = curves[
        curves["metric"].eq(metric)
        & curves["window_type"].eq(window_type)
        & np.isclose(curves["window_size"], window_size)
        & curves["passes_support_mask"].astype(bool)
    ]
    return frame


def _pivot_slice(curves: pd.DataFrame, metric: str, day: str) -> pd.DataFrame:
    frame = curves[
        curves["metric"].eq(metric)
        & curves["day"].eq(day)
        & np.isclose(curves["window_size"], PIVOT_WINDOW_MIN)
        & curves["passes_support_mask"].astype(bool)
    ]
    return frame


# --------------------------------------------------------------------------- drawing


def _style_axes(canvas: Canvas, ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(canvas.axes_linewidth)
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=1.9,
        width=canvas.axes_linewidth,
        colors=AXIS_COLOR,
        labelsize=canvas.tick_label,
        pad=1.4,
    )


def _draw_curve_axes(
    canvas: Canvas,
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
) -> tuple[float, float, float, float]:
    """Draw the five experience curves with SEM bands; return the data extent."""

    colors = EXPERIENCE_COLORS[metric]
    x_lo, x_hi = np.inf, -np.inf
    y_lo, y_hi = np.inf, -np.inf
    for experience in EXPERIENCE_ORDER:
        group = frame[frame["experience_group"].eq(experience)].sort_values("x_plot")
        if group.empty:
            continue
        x = group["x_plot"].to_numpy(float)
        mean = group["mean"].to_numpy(float)
        sem = group["sem"].to_numpy(float)
        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=colors[experience],
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=colors[experience],
            linewidth=0.95,
            solid_capstyle="round",
            zorder=2,
            label=EXPERIENCE_LABEL[experience],
        )
        x_lo, x_hi = min(x_lo, float(x.min())), max(x_hi, float(x.max()))
        finite = np.isfinite(mean - sem) & np.isfinite(mean + sem)
        if finite.any():
            y_lo = min(y_lo, float((mean - sem)[finite].min()))
            y_hi = max(y_hi, float((mean + sem)[finite].max()))
    return x_lo, x_hi, y_lo, y_hi


def _row_legend(canvas: Canvas, metric: str, y_in: float) -> None:
    """One experience legend per metric row, centred across the whole row.

    The ramp applies to every subpanel in the row, including the pivot pair, so the legend
    is centred over the full row width rather than sitting above the left block only.
    """

    colors = EXPERIENCE_COLORS[metric]
    row_width = PIVOT_BLOCK_LEFT + PIVOT_BLOCK_W - FIXED_BLOCK_LEFT
    ax = canvas.fig.add_axes(
        canvas.rect(FIXED_BLOCK_LEFT, y_in, row_width, ROW_LEGEND_H)
    )
    ax.set_axis_off()
    handles = [
        plt.Line2D([0.0], [0.0], color=colors[experience], linewidth=1.15)
        for experience in EXPERIENCE_ORDER
    ]
    ax.legend(
        handles,
        [EXPERIENCE_LABEL[experience] for experience in EXPERIENCE_ORDER],
        loc="center",
        ncol=len(EXPERIENCE_ORDER),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.95,
        fontsize=canvas.tick_label,
        borderpad=0.0,
    )


# --------------------------------------------------------------------------- schematics


def _schematic_axes(canvas: Canvas, left: float, bottom: float, width: float, height: float):
    ax = canvas.fig.add_axes(canvas.rect(left, bottom, width, height))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    return ax


def _schematic_bar(ax, x, y, width, height, *, whole: bool, linewidth: float) -> None:
    ax.add_patch(
        plt.Rectangle(
            (x, y),
            width,
            height,
            facecolor=SCHEMATIC_COLORS["whole_fill"] if whole else SCHEMATIC_COLORS["bar_fill"],
            edgecolor=SCHEMATIC_COLORS["whole_edge"] if whole else SCHEMATIC_COLORS["bar_edge"],
            linewidth=linewidth,
            zorder=2,
        )
    )


def _schematic_arrow(
    ax, start, end, *, rad: float, linewidth: float, both: bool = False, color: str | None = None
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="<|-|>" if both else "-|>",
            mutation_scale=4.6,
            linewidth=linewidth,
            color=color or SCHEMATIC_COLORS["arrow"],
            shrinkA=0.0,
            shrinkB=0.0,
            clip_on=False,
            zorder=5,
        )
    )


def draw_convergence_schematic(canvas: Canvas, bottom: float) -> dict[str, Any]:
    """Sliding Day-i window compared with the whole of Day i+1 (many-to-one)."""

    ax = _schematic_axes(canvas, FIXED_BLOCK_LEFT, bottom, FIXED_BLOCK_W, SCHEMATIC_H)
    linewidth = 0.62
    ink = SCHEMATIC_COLORS["ink"]
    muted = SCHEMATIC_COLORS["muted"]
    label = max(canvas.tick_label, 4.6)

    ax.text(
        0.5,
        0.985,
        f"{DAY_I_LABEL} window vs whole of {DAY_IP1_LABEL}",
        ha="center",
        va="top",
        fontsize=canvas.base_font,
        color=ink,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
        zorder=10,
    )

    day1_x, day1_w = 0.055, 0.50
    day2_x, day2_w = 0.735, 0.21
    bar_y, bar_h = 0.30, 0.20
    _schematic_bar(ax, day1_x, bar_y, day1_w, bar_h, whole=False, linewidth=linewidth)
    _schematic_bar(ax, day2_x, bar_y, day2_w, bar_h, whole=True, linewidth=linewidth)
    ax.text(
        day1_x,
        bar_y + bar_h + 0.045,
        DAY_I_LABEL,
        ha="left",
        va="bottom",
        fontsize=label,
        color=ink,
    )
    ax.text(
        day2_x + day2_w,
        bar_y + bar_h + 0.045,
        DAY_IP1_LABEL,
        ha="right",
        va="bottom",
        fontsize=label,
        color=ink,
    )
    ax.text(
        day2_x + day2_w / 2.0,
        bar_y - 0.055,
        "whole session",
        ha="center",
        va="top",
        fontsize=label,
        color=muted,
    )

    window_w = 0.115
    starts = (0.085, 0.255, 0.425)
    radii = (-0.34, -0.24, -0.14)
    target = (day2_x + day2_w / 2.0, bar_y + bar_h + 0.01)
    for start, rad, color in zip(starts, radii, SCHEMATIC_COLORS["window_colors"], strict=True):
        ax.add_patch(
            plt.Rectangle(
                (start, bar_y),
                window_w,
                bar_h,
                facecolor=color,
                edgecolor=SCHEMATIC_COLORS["window_edge"],
                linewidth=linewidth,
                zorder=3,
            )
        )
        _schematic_arrow(
            ax,
            (start + window_w / 2.0, bar_y + bar_h + 0.01),
            target,
            rad=rad,
            linewidth=linewidth,
        )
    _schematic_arrow(
        ax,
        (starts[0], bar_y - 0.10),
        (starts[-1] + window_w, bar_y - 0.10),
        rad=0.0,
        linewidth=linewidth,
        color=SCHEMATIC_COLORS["connector"],
    )
    ax.text(
        (starts[0] + starts[-1] + window_w) / 2.0,
        bar_y - 0.145,
        f"window slides through {DAY_I_LABEL}",
        ha="center",
        va="top",
        fontsize=label,
        color=muted,
    )
    return {
        "role": f"fixed {DAY_IP1_LABEL}-template convergence design",
        "n_windows_drawn": len(starts),
        "whole_session_reference": DAY_IP1_LABEL,
    }


def draw_pivot_schematic(canvas: Canvas, bottom: float) -> dict[str, Any]:
    """Pre/post windows straddling a sliding pivot, within each day (pairwise, local)."""

    ax = _schematic_axes(canvas, PIVOT_BLOCK_LEFT, bottom, PIVOT_BLOCK_W, SCHEMATIC_H)
    linewidth = 0.62
    ink = SCHEMATIC_COLORS["ink"]
    muted = SCHEMATIC_COLORS["muted"]
    label = max(canvas.tick_label, 4.6)
    pale, _, dark = SCHEMATIC_COLORS["window_colors"]

    ax.text(
        0.5,
        0.985,
        "Windows across a sliding pivot",
        ha="center",
        va="top",
        fontsize=canvas.base_font,
        color=ink,
        fontweight="bold",
    )

    bar_y, bar_h = 0.30, 0.20
    bar_w = 0.40
    window_w = 0.105
    for bar_x, day_label in ((0.045, DAY_I_LABEL), (0.545, DAY_IP1_LABEL)):
        _schematic_bar(ax, bar_x, bar_y, bar_w, bar_h, whole=False, linewidth=linewidth)
        pivot_x = bar_x + bar_w * 0.52
        for offset, color in ((-window_w, pale), (0.0, dark)):
            ax.add_patch(
                plt.Rectangle(
                    (pivot_x + offset, bar_y),
                    window_w,
                    bar_h,
                    facecolor=color,
                    edgecolor=SCHEMATIC_COLORS["window_edge"],
                    linewidth=linewidth,
                    zorder=3,
                )
            )
        ax.plot(
            [pivot_x, pivot_x],
            [bar_y - 0.115, bar_y + bar_h + 0.035],
            color=ink,
            linewidth=0.85,
            zorder=6,
        )
        _schematic_arrow(
            ax,
            (pivot_x - window_w / 2.0, bar_y + bar_h + 0.01),
            (pivot_x + window_w / 2.0, bar_y + bar_h + 0.01),
            rad=-0.55,
            linewidth=linewidth,
            both=True,
        )
        ax.text(
            bar_x,
            bar_y + bar_h + 0.045,
            day_label,
            ha="left",
            va="bottom",
            fontsize=label,
            color=ink,
        )
        ax.text(
            pivot_x + 0.012,
            bar_y - 0.135,
            "pivot",
            ha="left",
            va="top",
            fontsize=label,
            color=muted,
        )
        _schematic_arrow(
            ax,
            (bar_x + bar_w * 0.18, bar_y - 0.115),
            (bar_x + bar_w * 0.86, bar_y - 0.115),
            rad=0.0,
            linewidth=linewidth,
            color=SCHEMATIC_COLORS["connector"],
        )
    ax.text(
        0.5,
        bar_y - 0.20,
        "pivot slides within each day",
        ha="center",
        va="top",
        fontsize=label,
        color=muted,
    )
    return {
        "role": "within-session sliding-pivot local-stability design",
        "days_drawn": ["day1", "day2"],
        "window_pair": "pre / post, abutting the pivot",
    }


# --------------------------------------------------------------------------- assembly


def render(out_dir: Path, round_number: int = 1) -> dict[str, Any]:
    frames = _load_frames()
    checks = Checks()
    suffix = f"_round{round_number}"

    fixed_curves = frames["fixed_curves"]
    fixed_stats = frames["fixed_stats"]
    pivot_curves = frames["pivot_curves"]
    pivot_stats = frames["pivot_stats"]
    pivot_limits = frames["pivot_limits"]

    # ---- scientific contract, asserted from the tables ------------------------
    slope_rows = fixed_stats[fixed_stats["estimand"].eq("mean_pairwise_within_day1_slope")]
    checks.record(
        "fixed_template_scope_59_pairs_22_mice",
        set(slope_rows["n_pair_rows"]) == {EXPECTED_PAIRS}
        and set(slope_rows["n_animal_clusters"]) == {EXPECTED_ANIMALS},
        f"{sorted(set(slope_rows['n_pair_rows']))} pair rows / "
        f"{sorted(set(slope_rows['n_animal_clusters']))} animal clusters",
    )
    checks.record(
        "fixed_template_bh12_family",
        len(slope_rows) == FIXED_BH_FAMILY
        and "q_bh12_two_sided" in fixed_stats.columns,
        f"{len(slope_rows)} slope estimands in the BH{FIXED_BH_FAMILY} family",
    )
    checks.record(
        "pivot_available_curve_scope_59_pairs_22_animals",
        set(pivot_stats["n_recording_cycles"]) == {PIVOT_EXPECTED_PAIR_ROWS}
        and set(pivot_stats["n_animals"]) == {EXPECTED_ANIMALS},
        f"{PIVOT_EXPECTED_PAIR_ROWS} recording cycles retained, "
        f"{EXPECTED_ANIMALS} animal clusters",
    )
    checks.record(
        "pivot_bh3_family_per_estimand",
        len(pivot_stats) == 9
        and pivot_stats.groupby("estimand").size().eq(PIVOT_BH_FAMILY).all()
        and "p_bh3_two_sided" in pivot_stats.columns,
        "BH3 across three neural measures separately for Day i slope, Day i+1 slope and within-curve SD difference",
    )
    checks.record(
        "pivot_available_curve_estimands",
        set(pivot_stats["value_column"]) == {
            "day_i_slope_r_per_hour", "day_i1_slope_r_per_hour", "day_i1_minus_day_i_sd"
        },
        "slopes and SD differences use each recording cycle's complete finite-in-both-days pivot curve",
    )

    # ---- canvas -------------------------------------------------------------
    canvas = Canvas(
        FIGURE_WIDTH_IN,
        FIGURE_HEIGHT_IN,
        font_reference_width_in=FONT_REFERENCE_WIDTH_IN,
    )

    schematic_bottom = FIGURE_HEIGHT_IN - TOP_PAD - SCHEMATIC_H
    schematic_a = draw_convergence_schematic(canvas, schematic_bottom)
    schematic_b = draw_pivot_schematic(canvas, schematic_bottom)
    canvas.panel_letter(0.03, FIGURE_HEIGHT_IN - 0.04, "a")
    canvas.panel_letter(PIVOT_BLOCK_LEFT - 0.34, FIGURE_HEIGHT_IN - 0.04, "b")

    panel_letters = ["a", "b"]
    panel_audit: dict[str, Any] = {
        "a": schematic_a,
        "b": schematic_b,
    }
    row_letters = (("c", "d"), ("e", "f"), ("g", "h"))
    row_axes: dict[str, list[plt.Axes]] = {}
    displayed_stats: list[dict[str, Any]] = []
    n_axes = 2

    for row_index, (metric_spec, (fixed_letter, pivot_letter)) in enumerate(
        zip(METRICS, row_letters, strict=True)
    ):
        metric = metric_spec["metric"]
        axes_bottom = _row_axes_bottom(row_index)
        legend_bottom = (
            axes_bottom
            + ROW_AXES_H
            + _row_title_height(row_index)
            + _row_legend_gap(row_index)
        )
        is_bottom_row = row_index == len(METRICS) - 1
        is_top_row = row_index == 0

        _row_legend(canvas, metric, legend_bottom)

        # ---- fixed-template block -----------------------------------------
        fixed_axes: list[plt.Axes] = []
        extents: list[tuple[float, float, float, float]] = []
        for column_index, column in enumerate(FIXED_COLUMNS):
            ax = canvas.fig.add_axes(
                canvas.rect(
                    _fixed_column_left(column_index), axes_bottom, SUBPANEL_W, ROW_AXES_H
                )
            )
            _style_axes(canvas, ax)
            frame = _curve_slice(
                fixed_curves, metric, str(column["window_type"]), float(column["window_size"])
            )
            units = set(frame["x_plot_unit"].unique())
            if units != {EXPECTED_X_UNITS[str(column["window_type"])]}:
                raise ValueError(
                    f"unexpected x unit for {column['title']}: {sorted(units)}; "
                    "x_display_scale assumes the maintained source unit"
                )
            scale = float(column["x_display_scale"])
            if scale != 1.0:
                frame = frame.assign(x_plot=frame["x_plot"].astype(float) * scale)
            extents.append(_draw_curve_axes(canvas, ax, frame, metric))
            if scale != 1.0:
                # Raw transition counts need a locator that stays legible in a 1.21 in
                # subpanel, and a separator so five-digit ticks stay readable.
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}")
                )
            stats_row = _fixed_slope(
                fixed_stats, metric, str(column["window_type"]), float(column["window_size"])
            )
            # Recorded for the caption only; nothing statistical is drawn on the panel.
            stars = canvas.significance_stars(float(stats_row["q_bh12_two_sided"]))
            displayed_stats.append(
                {
                    "panel": fixed_letter,
                    "metric": metric,
                    "column": column["title"],
                    "estimand": "mean_pairwise_within_day1_slope",
                    "estimate": float(stats_row["estimate"]),
                    "unit": str(stats_row["estimate_unit"]),
                    "q_bh12_two_sided": float(stats_row["q_bh12_two_sided"]),
                    "stars": stars,
                    "n_pair_rows": int(stats_row["n_pair_rows"]),
                    "n_animal_clusters": int(stats_row["n_animal_clusters"]),
                }
            )
            if is_top_row:
                ax.set_title(str(column["title"]), fontsize=canvas.base_font, color=AXIS_COLOR, pad=2.4)
            if is_bottom_row:
                ax.set_xlabel(str(column["x_label"]), fontsize=canvas.axis_label, color=AXIS_COLOR, labelpad=1.6)
            if column_index == 0:
                ax.set_ylabel(
                    str(metric_spec["y_label"]),
                    fontsize=canvas.axis_label,
                    color=AXIS_COLOR,
                    labelpad=1.8,
                )
            fixed_axes.append(ax)
            n_axes += 1

        # x limits are per column; the y range is settled once both blocks are measured.
        for ax, extent in zip(fixed_axes, extents, strict=True):
            ax.set_xlim(extent[0], extent[1])
        for ax in fixed_axes[1:]:
            ax.tick_params(axis="y", labelleft=False)

        # ---- pivot block ---------------------------------------------------
        limits = pivot_limits[
            pivot_limits["metric"].eq(metric)
            & np.isclose(pivot_limits["window_size_min"], PIVOT_WINDOW_MIN)
        ]
        if len(limits) != 1:
            raise ValueError(f"expected one pivot limit row for {metric}, got {len(limits)}")
        limit = limits.iloc[0]
        pivot_axes: list[plt.Axes] = []
        pivot_extents: list[tuple[float, float]] = []
        for column_index, column in enumerate(PIVOT_COLUMNS):
            ax = canvas.fig.add_axes(
                canvas.rect(
                    _pivot_column_left(column_index), axes_bottom, SUBPANEL_W, ROW_AXES_H
                )
            )
            _style_axes(canvas, ax)
            frame = _pivot_slice(pivot_curves, metric, str(column["day"]))
            frame = frame.rename(columns={"pair_experience_group": "experience_group", "x_min": "x_plot"})
            _draw_curve_axes(canvas, ax, frame, metric)
            ax.set_xlim(float(limit["x_min"]), float(limit["x_max_common_supported"]))
            # Measure the y extent over the drawn x window only: curve tails beyond
            # x_max_common_supported are clipped away and must not stretch the row.
            visible = frame[
                frame["x_plot"].between(
                    float(limit["x_min"]), float(limit["x_max_common_supported"])
                )
            ]
            pivot_extents.append(
                (
                    float((visible["mean"] - visible["sem"]).min()),
                    float((visible["mean"] + visible["sem"]).max()),
                )
            )
            stats_row = _pivot_slope(pivot_stats, metric, str(column["day"]))
            # Recorded for the caption only; nothing statistical is drawn on the panel.
            stars = canvas.significance_stars(float(stats_row["p_bh3_two_sided"]))
            displayed_stats.append(
                {
                    "panel": pivot_letter,
                    "metric": metric,
                    "column": column["title"],
                    "estimand": str(stats_row["estimand"]),
                    "estimate": float(stats_row["estimate"]),
                    "unit": "r per hour",
                    "p_bh3_two_sided": float(stats_row["p_bh3_two_sided"]),
                    "stars": stars,
                    "n_pair_rows": int(stats_row["n_recording_cycles"]),
                    "n_animal_clusters": int(stats_row["n_animals"]),
                }
            )
            if is_top_row:
                # Day identity sits on the upper line; the window title drops to the
                # same height as "10-min window" in the fixed-template block.
                canvas.text(
                    _pivot_column_left(column_index) + SUBPANEL_W / 2.0,
                    axes_bottom + ROW_AXES_H + ROW_DAY_LINE_OFFSET,
                    str(column["title"]),
                    ha="center",
                    va="bottom",
                    fontsize=canvas.base_font,
                    color=AXIS_COLOR,
                )
            if is_bottom_row:
                # The pivot x axis is the in-maze time at which the pivot sits, so it takes
                # the same title as the fixed-template time columns.
                ax.set_xlabel(
                    X_LABEL_TIME,
                    fontsize=canvas.axis_label,
                    color=AXIS_COLOR,
                    labelpad=1.6,
                )
            if column_index == 0:
                ax.set_ylabel(
                    str(metric_spec["y_label"]),
                    fontsize=canvas.axis_label,
                    color=AXIS_COLOR,
                    labelpad=1.8,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
            pivot_axes.append(ax)
            n_axes += 1

        # ---- one y range across the whole metric row ------------------------
        # Both blocks plot the same correlation for the same mice, so the row shares a
        # single y axis and the eye can carry a value straight across it. The range is
        # the union of every drawn curve band in the row, padded by the same 6% the
        # fixed-template block used on its own. This supersedes the exported canonical
        # pivot range, whose job -- making Day 1 and Day 2 comparable -- a row-wide range
        # already does, and which left a lot of empty axis above the ensemble-rate data.
        row_lo = min(
            min(extent[2] for extent in extents),
            min(extent[0] for extent in pivot_extents),
        )
        row_hi = max(
            max(extent[3] for extent in extents),
            max(extent[1] for extent in pivot_extents),
        )
        pad = 0.06 * (row_hi - row_lo)
        row_limits = [float(row_lo - pad), float(row_hi + pad)]
        for ax in (*fixed_axes, *pivot_axes):
            ax.set_ylim(*row_limits)

        if is_top_row:
            canvas.text(
                PIVOT_BLOCK_LEFT + PIVOT_BLOCK_W / 2.0,
                axes_bottom + ROW_AXES_H + 0.035,
                f"{PIVOT_WINDOW_MIN:.0f}-min window",
                ha="center",
                va="bottom",
                fontsize=canvas.base_font,
                color=AXIS_COLOR,
            )

        # Letters hang from the axes top, in the left gutter. Sitting them *above* the axes
        # would put them inside the row legend, which now spans the whole row width.
        canvas.panel_letter(FIXED_BLOCK_LEFT - 0.46, axes_bottom + ROW_AXES_H, fixed_letter)
        canvas.panel_letter(PIVOT_BLOCK_LEFT - 0.46, axes_bottom + ROW_AXES_H, pivot_letter)
        panel_letters.extend([fixed_letter, pivot_letter])
        row_axes[fixed_letter] = fixed_axes
        row_axes[pivot_letter] = pivot_axes
        panel_audit[fixed_letter] = {
            "role": f"{metric} fixed {DAY_IP1_LABEL}-template convergence",
            "columns": [column["title"] for column in FIXED_COLUMNS],
            "y_label": str(metric_spec["y_label"]),
            "x_labels": [str(column["x_label"]) for column in FIXED_COLUMNS],
            "x_display_units": [
                DRAWN_X_UNITS[str(column["window_type"])] for column in FIXED_COLUMNS
            ],
            "shared_y_limits": list(row_limits),
            "y_range_scope": "whole metric row (panels c-h), union of both blocks",
        }
        panel_audit[pivot_letter] = {
            "role": f"{metric} within-session local stability across a sliding pivot",
            "columns": [column["title"] for column in PIVOT_COLUMNS],
            "y_label": str(metric_spec["y_label"]),
            "x_labels": [X_LABEL_TIME for _ in PIVOT_COLUMNS],
            "x_display_units": [DRAWN_X_UNITS["maze_time"] for _ in PIVOT_COLUMNS],
            "shared_y_limits": list(row_limits),
            "y_range_scope": "whole metric row (panels c-h), union of both blocks",
            # Kept for provenance: the maintained export this display range supersedes.
            "superseded_canonical_y_limits": [
                float(limit["y_min_shared_days"]),
                float(limit["y_max_shared_days"]),
            ],
            "shared_x_limits": [
                float(limit["x_min"]),
                float(limit["x_max_common_supported"]),
            ],
        }

    # ---- display contract ---------------------------------------------------
    checks.check_typography(canvas)
    checks.check_no_grid_and_upright_labels(canvas.fig)
    checks.record(
        "panel_letters_a_to_h",
        panel_letters == list("abcdefgh"),
        f"{len(panel_letters)} panels a-h over {n_axes} axes",
    )
    checks.record(
        "axes_count_two_schematics_plus_fifteen",
        n_axes == 2 + 15,
        f"{n_axes} axes: 2 schematics + 9 fixed-template + 6 pivot subpanels",
    )
    checks.record(
        "displayed_statistics_count",
        len(displayed_stats) == 15,
        f"{len(displayed_stats)} slope annotations, one per quantitative subpanel",
    )
    null_claims = [row for row in displayed_stats if row["stars"] == "n.s."]
    checks.record(
        "nulls_recorded_for_the_caption",
        all(row["column"] == DAY_IP1_LABEL for row in null_claims),
        f"{len(null_claims)} null claim(s) carried to the caption, not the panel: "
        + ", ".join(f"{row['metric']} {row['column']}" for row in null_claims),
    )
    checks.record(
        "y_limits_shared_across_the_whole_metric_row",
        all(
            panel_audit[fixed_letter]["shared_y_limits"]
            == panel_audit[pivot_letter]["shared_y_limits"]
            for fixed_letter, pivot_letter in zip("ceg", "dfh", strict=True)
        ),
        "; ".join(
            f"{fixed_letter}/{pivot_letter} "
            f"[{panel_audit[fixed_letter]['shared_y_limits'][0]:.3f}, "
            f"{panel_audit[fixed_letter]['shared_y_limits'][1]:.3f}]"
            for fixed_letter, pivot_letter in zip("ceg", "dfh")
        ),
    )
    # A shared range must not hide data: every drawn band has to fall inside it.
    clipped = [
        f"{letter}:{axis.get_title() or index}"
        for letter, axes_group in row_axes.items()
        for index, axis in enumerate(axes_group)
        for lo, hi in [axis.get_ylim()]
        if any(
            float(np.nanmin(collection.get_paths()[0].vertices[:, 1])) < lo - 1e-9
            or float(np.nanmax(collection.get_paths()[0].vertices[:, 1])) > hi + 1e-9
            for collection in axis.collections
            if collection.get_paths()
        )
    ]
    checks.record(
        "shared_y_range_clips_no_curve_band",
        not clipped,
        "every SEM band in c-h lies inside its row's shared range"
        if not clipped
        else f"clipped: {clipped}",
    )
    checks.record(
        "schematic_row_added_as_requested",
        panel_audit["a"]["n_windows_drawn"] == 3
        and panel_audit["b"]["days_drawn"] == ["day1", "day2"],
        "slide note 'add a schematic?' answered with the two design diagrams",
    )
    checks.record(
        "x_axes_declare_their_drawn_unit",
        set(fixed_curves["x_plot_unit"].unique())
        == {EXPECTED_X_UNITS["maze_time"], EXPECTED_X_UNITS["tile_change"]},
        "x_plot arrives as in-maze minutes (drawn as-is) and thousand tile changes "
        "(scaled x1,000 so the ED1 title 'Maze tile transitions' names the drawn unit)",
    )
    # The pivot panels must not silently drift away from their metric row's y title.
    shared_y = [
        (
            panel_audit[fixed_letter]["y_label"],
            panel_audit[pivot_letter]["y_label"],
            metric_spec["y_label"],
        )
        for fixed_letter, pivot_letter, metric_spec in zip(
            "ceg", "dfh", METRICS, strict=True
        )
    ]
    checks.record(
        "pivot_panels_share_the_metric_row_y_title",
        all(fixed == pivot == expected for fixed, pivot, expected in shared_y),
        "; ".join(f"{fixed} == {pivot}" for fixed, pivot, _ in shared_y),
    )
    checks.record(
        "y_titles_carry_no_r_qualifier",
        not any("(r)" in spec["y_label"] for spec in METRICS),
        ", ".join(str(spec["y_label"]) for spec in METRICS),
    )
    drawn_x_labels = {
        label
        for letter in "cdefgh"
        for label in panel_audit[letter]["x_labels"]
    }
    checks.record(
        "x_titles_match_extended_data_figure1_vocabulary",
        drawn_x_labels == {X_LABEL_TIME, X_LABEL_TILES},
        "c-h use only " + " and ".join(sorted(drawn_x_labels)),
    )
    checks.record(
        "no_panel_image_embedded",
        not any(
            isinstance(artist, matplotlib.image.AxesImage)
            for axis in canvas.fig.axes
            for artist in axis.get_children()
        ),
        "every panel drawn from plot-ready tables; no maintained panel raster placed",
    )

    figure_text = [
        artist.get_text()
        for artist in canvas.fig.findobj(match=matplotlib.text.Text)
        if artist.get_text()
    ]
    stale_day_text = [
        value for value in figure_text if "Day 1" in value or "Day 2" in value
    ]
    checks.record(
        "displayed_day_labels_use_generic_cycle_notation",
        any(DAY_I_LABEL in value for value in figure_text)
        and any(DAY_IP1_LABEL in value for value in figure_text)
        and not stale_day_text,
        f"visible day roles use {DAY_I_LABEL} and {DAY_IP1_LABEL}; "
        + ("no legacy labels" if not stale_day_text else f"stale labels: {stale_day_text}"),
    )

    outputs = canvas.save(out_dir / f"{STEM}{suffix}")
    checks.check_outputs(outputs)
    svg_text = (out_dir / f"{STEM}{suffix}.svg").read_text(encoding="utf-8", errors="ignore")
    checks.record(
        "svg_editable_text",
        "<text" in svg_text,
        f"{svg_text.count('<text')} live <text> elements",
    )
    forbidden = {
        "q=": "exact q value",
        "slope": "slope estimate",
        "***": "significance stars",
        "n.s.": "null marker",
        "r/h": "slope unit",
    }
    present = sorted(
        description for token, description in forbidden.items() if token in svg_text
    )
    checks.record(
        "no_statistics_on_panels",
        not present,
        "panels carry data, axes and labels only; every statistic lives in the caption"
        if not present
        else f"still drawn on the panels: {present}",
    )
    canvas.close()

    manifest = {
        "figure": "Extended Data Figure 6",
        "round": round_number,
        "canvas_inches": [FIGURE_WIDTH_IN, round(FIGURE_HEIGHT_IN, 4)],
        "status": checks.status,
        "data_authority": {
            "coverage_gate_sec": COVERAGE_GATE_SEC,
            "coverage_gate_frames_at_dt_0p04": COVERAGE_GATE_FRAMES,
        },
        "figure_size_inches": [FIGURE_WIDTH_IN, round(FIGURE_HEIGHT_IN, 4)],
        "renderer": "scripts/render_ed06_within_session_convergence_assembly.py",
        "sources": {
            name: {
                "path": _repo_relative(path),
                "sha256": sha256(path),
                "rows": int(len(frames[key])),
            }
            for name, key, path in (
                ("fixed_group_curves", "fixed_curves", FIXED_CURVES),
                ("fixed_clustered_stats", "fixed_stats", FIXED_STATS),
                ("fixed_curve_support_mask", "fixed_support", FIXED_SUPPORT),
                ("pivot_group_curves", "pivot_curves", PIVOT_CURVES),
                ("pivot_available_curve_stats", "pivot_stats", PIVOT_STATS),
                ("pivot_display_limits", "pivot_limits", PIVOT_LIMITS),
            )
        },
        "panels": panel_audit,
        "displayed_statistics": displayed_stats,
        "outputs": [
            {
                "path": _repo_relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "validation": checks.rows,
        "validation_status": checks.status,
    }
    (out_dir / f"ED06_ASSEMBLY_MANIFEST_ROUND{round_number}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(checks.rows).to_csv(
        out_dir / f"ED06_VALIDATION_ROUND{round_number}.csv", index=False
    )
    pd.DataFrame(displayed_stats).to_csv(
        out_dir / f"ED06_DISPLAYED_STATISTICS_ROUND{round_number}.csv", index=False
    )
    caption_dir = FIGURE_ROOT if out_dir == DEFAULT_OUT else out_dir
    _write_caption_notes(displayed_stats, round_number, caption_dir)
    return manifest


def _write_caption_notes(
    displayed_stats: list[dict[str, Any]],
    round_number: int,
    caption_dir: Path,
) -> None:
    """Everything the panels deliberately do not carry: exact q, counts, policies."""

    lines = [
        f"# Extended Data Figure 6 — caption notes (round {round_number})",
        "",
        f"Within-session convergence to the {DAY_IP1_LABEL} template and local stability across a",
        "sliding pivot. **The panels carry no statistics at all** — no slope estimates, no",
        "q values, no significance stars. Every number below belongs in the caption.",
        "",
        "## Support and gate",
        "",
        f"- Coverage gate: each window, and each side of a pivot, requires a fixed"
        f" {COVERAGE_GATE_SEC:.0f} s of selected data ({COVERAGE_GATE_FRAMES:,} frames at"
        " DT = 0.04 s). This supersedes the previous 50%-of-window rule.",
        f"- Fixed {DAY_IP1_LABEL}-template panels (c, e, g): {EXPECTED_PAIRS} same-stage pairs from"
        f" {EXPECTED_ANIMALS} animals; standard errors clustered by animal; BH across the"
        f" {FIXED_BH_FAMILY}-member window family.",
        f"- Pivot panels (d, f, h): {PIVOT_EXPECTED_PAIR_ROWS} of {EXPECTED_PAIRS} eligible"
        f" pairs retained at the {PIVOT_WINDOW_MIN:.0f}-min window, {EXPECTED_ANIMALS} mice;"
        f" BH{PIVOT_BH_FAMILY} across three neural measures separately for the within-curve"
        " SD difference, Day i slope and Day i+1 slope.",
        "- Pivot windows are placed on cumulative TimeInMaze, not on selected-sample time.",
        "- Curves are drawn only where the maintained support mask passes; shading is ± SEM"
        " across animals.",
        "",
        "## Exact statistics",
        "",
        "| Panel | Metric | Column | Estimand | Estimate | Unit | q (BH) | Stars | Pairs | Mice |",
        "|---|---|---|---|---:|---|---:|---|---:|---:|",
    ]
    for row in displayed_stats:
        q_value = row.get("q_bh12_two_sided", row.get("p_bh3_two_sided"))
        lines.append(
            f"| {row['panel']} | {row['metric']} | {row['column']} | {row['estimand']} | "
            f"{row['estimate']:.6f} | {row['unit']} | {q_value:.3g} | {row['stars']} | "
            f"{row['n_pair_rows']} | {row['n_animal_clusters']} |"
        )
    lines += [
        "",
        "`n.s.` marks a deliberately null claim rather than an untested one.",
        "",
        "## Scope notes",
        "",
        "- Each metric row carries one y title across both of its panels, and the title names",
        "  the metric rather than the comparison. The caption therefore has to say what is",
        f"  correlated: c/e/g correlate each {DAY_I_LABEL} window against the full {DAY_IP1_LABEL} template,",
        "  while d/f/h correlate the window before a pivot against the window after it.",
        "- Axis titles follow Extended Data Figure 1: `Time in maze (min)` and",
        "  `Maze tile transitions`. The tile axis is drawn in raw transitions, so its source",
        "  column (thousands) is scaled x1,000 at the point of drawing.",
        "- Each metric row shares one y range across all five of its subpanels, taken as the",
        f"  union of every drawn band in the row and padded by 6%. {DAY_I_LABEL} and {DAY_IP1_LABEL} are",
        "  therefore comparable, and so are the pivot and fixed-template blocks. This",
        "  supersedes the exported canonical pivot range, which is retained in the manifest",
        "  as `superseded_canonical_y_limits`.",
        "- `selected_time` and `absolute_time` windows are not displayed in the active ED6",
        "  selection; the 30-min maze-time fixed-template window is the Main Figure 3 panel and",
        "  is deliberately not repeated here.",
        "",
    ]
    (caption_dir / f"CAPTION_NOTES_ROUND{round_number}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Review round; every round writes round-suffixed files into the same folder.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = render(out_dir, int(args.round))
    failed = [row for row in manifest["validation"] if row["status"] == "FAIL"]
    print(
        json.dumps(
            {
                "out_dir": _repo_relative(out_dir),
                "figure_size_inches": manifest["figure_size_inches"],
                "checks": len(manifest["validation"]),
                "status": manifest["validation_status"],
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
