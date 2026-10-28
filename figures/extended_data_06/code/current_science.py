from __future__ import annotations
from collections.abc import Mapping, Sequence, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final
from types import SimpleNamespace
from io import BytesIO
import itertools
import hashlib
import copy
import re
import xml.etree.ElementTree as ET
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from PIL import Image
from release_rendering.plotting import *
from release_rendering.statistics import *
from release_rendering.maze import build_maze, run_path_between
from release_rendering.types import FigureAssemblyResult, PanelResult, RunMode
from release_rendering.common import validate_output_request as _validate_output_request
from release_rendering.common import significance_text
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GRID_SIZE = 15
CANONICAL_MAZE_DEPTH = 6
REWARD_NODE = 94
TARGET_NEURONS = 181
COORDINATED_DRAWS = 50
DECODER_CONTROL_BINS = (91, 121, 133)
DECODER_TARGET_BIN = 103


FIXED_REFERENCE_METRICS = (
    "place_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)


FIXED_REFERENCE_TIME_WINDOWS_MINUTES = (10.0, 20.0, 30.0)


FIXED_REFERENCE_TILE_WINDOWS = (3000,)


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


def _fixed_reference_definitions() -> set[tuple[str, str, float]]:
    """Return the 12 displayed metric and window combinations."""

    return {
        (metric, "maze_time", float(size))
        for metric in FIXED_REFERENCE_METRICS
        for size in FIXED_REFERENCE_TIME_WINDOWS_MINUTES
    } | {
        (metric, "tile_change", float(size))
        for metric in FIXED_REFERENCE_METRICS
        for size in FIXED_REFERENCE_TILE_WINDOWS
    }


def calculate_fixed_reference_slope_statistics(
    slopes: pd.DataFrame,
) -> pd.DataFrame:
    """Test three neural measures in four displayed fixed reference windows."""

    expected_definitions = _fixed_reference_definitions()
    observed_definitions = set(
        zip(
            slopes["metric"].astype(str),
            slopes["window_axis"].astype(str),
            slopes["window_size"].astype(float),
            strict=True,
        )
    )
    if observed_definitions != expected_definitions:
        raise ValueError("Fixed reference slopes do not contain all 12 displayed tests")
    pair_count = int(slopes["pair_id"].astype(str).nunique())
    support = slopes.groupby(
        ["metric", "window_axis", "window_size"], observed=True
    )["pair_id"].nunique()
    if not support.eq(pair_count).all():
        raise ValueError("At least one fixed reference test is missing a cycle")

    stats_rows: list[dict[str, object]] = []
    for key, selected in slopes.groupby(
        ["metric", "window_axis", "window_size"],
        sort=False,
    ):
        metric, axis, size = key
        stats_rows.append(
            {
                "metric": str(metric),
                "window_axis": str(axis),
                "window_size": float(size),
                **clustered_intercept_test(selected),
            }
        )
    statistics = pd.DataFrame(stats_rows)
    if len(statistics) != 12:
        raise AssertionError("The fixed reference correction family must contain 12 tests")
    statistics["p_bh12"] = benjamini_hochberg(
        statistics["p_two_sided"].to_numpy(dtype=float)
    )
    validate_fixed_reference_slope_statistics(slopes, statistics)
    return statistics


def validate_fixed_reference_slope_statistics(
    slopes: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Require one complete statistics row for each planned slope test."""

    required_columns = {
        "metric",
        "window_axis",
        "window_size",
        "mean_slope",
        "cluster_se",
        "cluster_t",
        "cluster_df",
        "p_two_sided",
        "n_rows",
        "n_animals",
        "p_bh12",
    }
    missing_columns = required_columns.difference(statistics.columns)
    if missing_columns:
        raise ValueError(
            "Fixed reference statistics are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    definition_columns = ["metric", "window_axis", "window_size"]
    if len(statistics) != 12 or statistics.duplicated(definition_columns).any():
        raise ValueError("Fixed reference statistics must contain 12 unique tests")
    observed_definitions = set(
        zip(
            statistics["metric"].astype(str),
            statistics["window_axis"].astype(str),
            statistics["window_size"].astype(float),
            strict=True,
        )
    )
    if observed_definitions != _fixed_reference_definitions():
        raise ValueError("Fixed reference statistics do not match the 12 tests")

    expected_support = (
        slopes.dropna(subset=["slope_per_hour_or_ktile"])
        .groupby(definition_columns, observed=True)
        .agg(
            n_rows=("pair_id", "size"),
            n_animals=("animal_id", "nunique"),
        )
        .reset_index()
    )
    observed_support = statistics[
        [*definition_columns, "n_rows", "n_animals"]
    ].copy()
    for frame in (expected_support, observed_support):
        frame["metric"] = frame["metric"].astype(str)
        frame["window_axis"] = frame["window_axis"].astype(str)
        frame["window_size"] = frame["window_size"].astype(float)
        frame["n_rows"] = frame["n_rows"].astype(int)
        frame["n_animals"] = frame["n_animals"].astype(int)
    support_check = expected_support.merge(
        observed_support,
        on=definition_columns,
        how="outer",
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
        indicator=True,
    )
    if (
        not support_check["_merge"].eq("both").all()
        or not support_check["n_rows_expected"].eq(
            support_check["n_rows_observed"]
        ).all()
        or not support_check["n_animals_expected"].eq(
            support_check["n_animals_observed"]
        ).all()
    ):
        raise ValueError("Fixed reference statistics do not match slope support")


def clustered_intercept_test(frame: pd.DataFrame) -> dict[str, object]:
    """Test the mean pair slope with animal-clustered CR1 covariance."""

    use = frame[["slope_per_hour_or_ktile", "animal_id"]].dropna()
    if len(use) <= 1 or use["animal_id"].nunique() < 2:
        return {
            "mean_slope": np.nan,
            "cluster_se": np.nan,
            "cluster_t": np.nan,
            "cluster_df": np.nan,
            "p_two_sided": np.nan,
            "n_rows": int(len(use)),
            "n_animals": int(use["animal_id"].nunique()),
        }
    fit = cr1_ols(
        use["slope_per_hour_or_ktile"].to_numpy(dtype=float),
        np.ones((len(use), 1), dtype=float),
        use["animal_id"].tolist(),
    )
    contrast = linear_contrast(fit, [1.0])
    return {
        "mean_slope": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "p_two_sided": contrast.p_two_sided,
        "n_rows": fit.n_observations,
        "n_animals": fit.n_clusters,
    }


_METRICS = (
    "place_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)


PIVOT_WINDOW_MINUTES = 30.0


def calculate_sliding_pivot_statistics(
    pair_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Test the three pair-level local-stability estimands with CR1 errors."""

    estimands = {
        "day_i1_minus_day_i_within_curve_sd": "day_i1_minus_day_i_sd",
        "day_i_linear_slope": "day_i_slope_r_per_hour",
        "day_i1_linear_slope": "day_i1_slope_r_per_hour",
    }
    rows: list[dict[str, object]] = []
    for estimand, value_column in estimands.items():
        family: list[dict[str, object]] = []
        for metric in _METRICS:
            selected = pair_summaries.loc[pair_summaries["metric"].eq(metric)].copy()
            renamed = selected.rename(columns={value_column: "slope_per_hour_or_ktile"})
            test = clustered_intercept_test(renamed)
            record: dict[str, object] = {
                "estimand": estimand,
                "metric": metric,
                "value_column": value_column,
                **test,
            }
            if estimand == "day_i1_minus_day_i_within_curve_sd":
                record["mean_day_i_within_curve_sd"] = float(
                    selected["day_i_within_curve_sd"].mean()
                )
                record["mean_day_i1_within_curve_sd"] = float(
                    selected["day_i1_within_curve_sd"].mean()
                )
            family.append(record)
        adjusted = benjamini_hochberg([float(row["p_two_sided"]) for row in family])
        for row, adjusted_p in zip(family, adjusted, strict=True):
            row["p_bh3"] = float(adjusted_p)
            rows.append(row)
    return pd.DataFrame(rows)


_ARTIFACT_KEY = "extended_data_figure_6"


_PAGE_WIDTH_INCHES = 7.5


_PAGE_HEIGHT_INCHES = 5.51


_LEFT_MARGIN_INCHES = 0.52


_SUBPANEL_WIDTH_INCHES = 1.21


_INNER_GAP_INCHES = 0.10


_BLOCK_GAP_INCHES = 0.55


_SCHEMATIC_HEIGHT_INCHES = 0.74


_SCHEMATIC_TO_ROWS_INCHES = 0.05


_ROW_AXES_HEIGHT_INCHES = 1.00


_ROW_LEGEND_HEIGHT_INCHES = 0.15


_ROW_TITLE_HEIGHT_TOP_INCHES = 0.27


_ROW_DAY_LINE_OFFSET_INCHES = 0.155


_ROW_LEGEND_GAP_TOP_INCHES = 0.02


_ROW_LEGEND_GAP_OTHER_INCHES = 0.09


_INTER_ROW_GAP_INCHES = 0.16


_TOP_PAD_INCHES = 0.18


_FIXED_BLOCK_WIDTH_INCHES = (
    3.0 * _SUBPANEL_WIDTH_INCHES + 2.0 * _INNER_GAP_INCHES
)


_PIVOT_BLOCK_LEFT_INCHES = (
    _LEFT_MARGIN_INCHES + _FIXED_BLOCK_WIDTH_INCHES + _BLOCK_GAP_INCHES
)


_PIVOT_BLOCK_WIDTH_INCHES = (
    2.0 * _SUBPANEL_WIDTH_INCHES + _INNER_GAP_INCHES
)


_AXIS_COLOR = "#202020"


_SCHEMATIC_COLORS = {
    "ink": "#30363A",
    "muted": "#667168",
    "bar_fill": "#F1F6F1",
    "bar_edge": "#91A493",
    "whole_fill": "#75A86F",
    "whole_edge": "#376F3E",
    "window_edge": "#3F8048",
    "window_colors": ("#DCEBDC", "#B8D3B8", "#8FB791"),
    "arrow": "#3F8048",
    "connector": "#B3BEB4",
}


_EXPERIENCE_LABELS = {
    "naive": "naive",
    "3mo": "3 mo",
    "6mo": "6 mo",
    "9mo": "9 mo",
    "12mo": "12 mo",
}


_METRIC_DISPLAY = (
    ("place_map_correlation", "Place map correlation"),
    ("pv_correlation", "PV correlation"),
    ("ensemble_rate_correlation", "Ensemble rate correlation"),
)


_FIXED_DISPLAY_COLUMNS = (
    ("maze_time", 10.0, "10-min window", "Time in maze (min)", 1.0),
    ("maze_time", 20.0, "20-min window", "Time in maze (min)", 1.0),
    (
        "tile_change",
        3000.0,
        "3,000-step window",
        "Maze tile steps",
        1000.0,
    ),
)


_DAY_I_MATH_LABEL = r"Day $\mathrm{i}$"


_DAY_I1_MATH_LABEL = r"Day $\mathrm{i}+1$"


_ACCEPTED_PIVOT_X_LIMITS = (30.0, 137.0)


_ACCEPTED_PIVOT_X_TICKS = (50.0, 75.0, 100.0, 125.0)


_ACCEPTED_ROW_Y_LIMITS = {
    "place_map_correlation": (0.06550722276756327, 0.7237064239433842),
    "pv_correlation": (0.18751834895277414, 0.8409209598434568),
    "ensemble_rate_correlation": (0.38504326794244936, 0.959228075138198),
}


_ACCEPTED_ROW_Y_TICKS = {
    "place_map_correlation": (0.2, 0.4, 0.6),
    "pv_correlation": (0.2, 0.4, 0.6, 0.8),
    "ensemble_rate_correlation": (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
}


class _WithinCycleCanvas:
    """Small inch-positioned canvas for the accepted within-cycle page."""

    width = _PAGE_WIDTH_INCHES
    height = _PAGE_HEIGHT_INCHES
    base_font = 6.0
    axis_label = 6.0
    tick_label = 5.25
    panel_label_size = 9.75
    axes_linewidth = 0.60

    def __init__(self) -> None:
        """Create the fixed-size white page used by the accepted layout."""

        self.figure = plt.figure(figsize=(self.width, self.height))
        self.figure.patch.set_facecolor("white")

    def rectangle(
        self,
        left: float,
        bottom: float,
        width: float,
        height: float,
    ) -> list[float]:
        """Convert an inch rectangle to Matplotlib figure coordinates."""

        return [
            left / self.width,
            bottom / self.height,
            width / self.width,
            height / self.height,
        ]

    def text(self, x: float, y: float, value: str, **kwargs: Any) -> None:
        """Write figure text at an inch coordinate."""

        self.figure.text(x / self.width, y / self.height, value, **kwargs)

    def panel_label(self, x: float, y: float, value: str) -> None:
        """Write one accepted lowercase panel label."""

        self.text(
            x,
            y,
            value,
            ha="left",
            va="top",
            fontsize=self.panel_label_size,
            fontweight="bold",
        )


def _fixed_column_left(index: int) -> float:
    """Return the left edge of one fixed-reference column in inches."""

    return _LEFT_MARGIN_INCHES + index * (
        _SUBPANEL_WIDTH_INCHES + _INNER_GAP_INCHES
    )


def _pivot_column_left(index: int) -> float:
    """Return the left edge of one sliding-pivot column in inches."""

    return _PIVOT_BLOCK_LEFT_INCHES + index * (
        _SUBPANEL_WIDTH_INCHES + _INNER_GAP_INCHES
    )


def _row_title_height(index: int) -> float:
    """Return the title space reserved above one metric row."""

    return _ROW_TITLE_HEIGHT_TOP_INCHES if index == 0 else 0.0


def _row_legend_gap(index: int) -> float:
    """Return the vertical legend gap for one metric row."""

    return (
        _ROW_LEGEND_GAP_TOP_INCHES
        if index == 0
        else _ROW_LEGEND_GAP_OTHER_INCHES
    )


def _row_axes_bottom(index: int) -> float:
    """Return the lower edge of one metric row in inches."""

    row_top = (
        _PAGE_HEIGHT_INCHES
        - _TOP_PAD_INCHES
        - _SCHEMATIC_HEIGHT_INCHES
        - _SCHEMATIC_TO_ROWS_INCHES
    )
    for earlier in range(index):
        row_top -= (
            _ROW_LEGEND_HEIGHT_INCHES
            + _row_legend_gap(earlier)
            + _row_title_height(earlier)
            + _ROW_AXES_HEIGHT_INCHES
            + _INTER_ROW_GAP_INCHES
        )
    return (
        row_top
        - _ROW_LEGEND_HEIGHT_INCHES
        - _row_legend_gap(index)
        - _row_title_height(index)
        - _ROW_AXES_HEIGHT_INCHES
    )


def _normalise_fixed_curve_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return fixed-reference group curves with one stable display schema."""

    frame = summary.copy()
    if "window_axis" in frame.columns and "window_type" not in frame.columns:
        frame = frame.rename(columns={"window_axis": "window_type"})
    if "x_plot" not in frame.columns:
        if "x_value" not in frame.columns:
            raise ValueError("Fixed-reference curves need x_plot or x_value")
        frame["x_plot"] = np.where(
            frame["window_type"].eq("maze_time"),
            pd.to_numeric(frame["x_value"], errors="coerce") / 60.0,
            pd.to_numeric(frame["x_value"], errors="coerce"),
        )
    required = {
        "metric",
        "window_type",
        "window_size",
        "experience_group",
        "x_plot",
        "mean",
        "sem",
        "passes_support_mask",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Fixed-reference curve summary is missing {missing}")
    return frame


def _normalise_pivot_curve_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return sliding-pivot group curves with one stable display schema."""

    frame = summary.copy()
    frame = frame.rename(
        columns={
            "experience_group": "pair_experience_group",
            "pivot_minutes": "x_min",
        }
    )
    if "window_size" not in frame.columns:
        frame["window_size"] = PIVOT_WINDOW_MINUTES
    frame["day"] = frame["day"].replace(
        {"Day i": "day1", "Day i+1": "day2"}
    )
    required = {
        "metric",
        "day",
        "window_size",
        "pair_experience_group",
        "x_min",
        "mean",
        "sem",
        "passes_support_mask",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Sliding-pivot curve summary is missing {missing}")
    return frame


def _normalise_pivot_display_limits(limits: pd.DataFrame) -> pd.DataFrame:
    """Return one shared 30-minute horizontal range per metric."""

    frame = limits.copy()
    if "window_size_min" in frame.columns:
        frame = frame.loc[
            np.isclose(frame["window_size_min"], PIVOT_WINDOW_MINUTES)
        ].copy()
        return frame[
            ["metric", "x_min", "x_max_common_supported"]
        ].reset_index(drop=True)
    if "day" in frame.columns:
        frame = frame.loc[frame["day"].eq("shared")].copy()
    frame = frame.rename(
        columns={
            "minimum_pivot_minutes": "x_min",
            "maximum_pivot_minutes": "x_max_common_supported",
        }
    )
    required = {"metric", "x_min", "x_max_common_supported"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Sliding-pivot display limits are missing {missing}")
    return frame[list(required)].reset_index(drop=True)


def _style_version_record_axis(
    canvas: _WithinCycleCanvas,
    axis: plt.Axes,
) -> None:
    """Apply the accepted line and tick treatment to one quantitative axis."""

    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(canvas.axes_linewidth)
        axis.spines[side].set_color(_AXIS_COLOR)
    axis.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=1.9,
        width=canvas.axes_linewidth,
        colors=_AXIS_COLOR,
        labelsize=canvas.tick_label,
        pad=1.4,
    )


def _group_curve_x_extent(
    frame: pd.DataFrame,
    metric: str,
) -> tuple[float, float]:
    """Return the finite horizontal extent used by a fixed-reference axis."""

    x_low, x_high = np.inf, -np.inf
    for experience in EXPERIENCE_GROUP_ORDER:
        selected = frame.loc[
            frame["experience_group"].eq(experience)
        ].sort_values("x_plot", kind="stable")
        if selected.empty:
            continue
        x = selected["x_plot"].to_numpy(dtype=float)
        x_low = min(x_low, float(np.min(x)))
        x_high = max(x_high, float(np.max(x)))
    if not np.all(np.isfinite((x_low, x_high))):
        raise ValueError(f"No finite supported curves were available for {metric}")
    return x_low, x_high


def _draw_group_curves(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
) -> tuple[float, float]:
    """Draw experience means and SEM bands and return the horizontal extent."""

    for experience in EXPERIENCE_GROUP_ORDER:
        selected = frame.loc[
            frame["experience_group"].eq(experience)
        ].sort_values("x_plot", kind="stable")
        if selected.empty:
            continue
        x = selected["x_plot"].to_numpy(dtype=float)
        mean = selected["mean"].to_numpy(dtype=float)
        error = selected["sem"].to_numpy(dtype=float)
        color = EXPERIENCE_GROUP_COLORS[metric][experience]
        axis.fill_between(
            x,
            mean - error,
            mean + error,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=0.95,
            solid_capstyle="round",
            zorder=2,
        )
    return _group_curve_x_extent(frame, metric)


def _draw_experience_legend(
    canvas: _WithinCycleCanvas,
    metric: str,
    bottom: float,
) -> None:
    """Draw one experience legend centered across a complete metric row."""

    width = (
        _PIVOT_BLOCK_LEFT_INCHES
        + _PIVOT_BLOCK_WIDTH_INCHES
        - _LEFT_MARGIN_INCHES
    )
    axis = canvas.figure.add_axes(
        canvas.rectangle(
            _LEFT_MARGIN_INCHES,
            bottom,
            width,
            _ROW_LEGEND_HEIGHT_INCHES,
        )
    )
    axis.set_axis_off()
    handles = [
        plt.Line2D(
            [0.0],
            [0.0],
            color=EXPERIENCE_GROUP_COLORS[metric][experience],
            linewidth=1.15,
        )
        for experience in EXPERIENCE_GROUP_ORDER
    ]
    axis.legend(
        handles,
        [_EXPERIENCE_LABELS[experience] for experience in EXPERIENCE_GROUP_ORDER],
        loc="center",
        ncol=len(EXPERIENCE_GROUP_ORDER),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.95,
        fontsize=canvas.tick_label,
        borderpad=0.0,
    )


def _schematic_axis(
    canvas: _WithinCycleCanvas,
    left: float,
    bottom: float,
    width: float,
) -> plt.Axes:
    """Add a blank axis for one design schematic."""

    axis = canvas.figure.add_axes(
        canvas.rectangle(left, bottom, width, _SCHEMATIC_HEIGHT_INCHES)
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()
    return axis


def _draw_schematic_bar(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    whole_session: bool,
) -> None:
    """Draw one day or whole-session bar in a design schematic."""

    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=(
                _SCHEMATIC_COLORS["whole_fill"]
                if whole_session
                else _SCHEMATIC_COLORS["bar_fill"]
            ),
            edgecolor=(
                _SCHEMATIC_COLORS["whole_edge"]
                if whole_session
                else _SCHEMATIC_COLORS["bar_edge"]
            ),
            linewidth=0.62,
            zorder=2,
        )
    )


def _draw_schematic_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    curvature: float,
    both_directions: bool = False,
    color: str | None = None,
) -> None:
    """Draw one connector arrow in a design schematic."""

    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="<|-|>" if both_directions else "-|>",
            mutation_scale=4.6,
            linewidth=0.62,
            color=color or _SCHEMATIC_COLORS["arrow"],
            shrinkA=0.0,
            shrinkB=0.0,
            clip_on=False,
            zorder=5,
        )
    )


def _draw_fixed_reference_design(
    canvas: _WithinCycleCanvas,
    bottom: float,
) -> None:
    """Draw moving Day i windows compared with whole Day i+1."""

    axis = _schematic_axis(
        canvas,
        _LEFT_MARGIN_INCHES,
        bottom,
        _FIXED_BLOCK_WIDTH_INCHES,
    )
    label_size = max(canvas.tick_label, 4.6)
    ink = _SCHEMATIC_COLORS["ink"]
    muted = _SCHEMATIC_COLORS["muted"]
    axis.text(
        0.5,
        0.985,
        f"{_DAY_I_MATH_LABEL} window vs whole of {_DAY_I1_MATH_LABEL}",
        ha="center",
        va="top",
        fontsize=canvas.base_font,
        color=ink,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
        zorder=10,
    )
    day_i_left, day_i_width = 0.055, 0.50
    day_i1_left, day_i1_width = 0.735, 0.21
    bar_bottom, bar_height = 0.30, 0.20
    _draw_schematic_bar(
        axis,
        day_i_left,
        bar_bottom,
        day_i_width,
        bar_height,
        whole_session=False,
    )
    _draw_schematic_bar(
        axis,
        day_i1_left,
        bar_bottom,
        day_i1_width,
        bar_height,
        whole_session=True,
    )
    axis.text(
        day_i_left,
        bar_bottom + bar_height + 0.045,
        _DAY_I_MATH_LABEL,
        ha="left",
        va="bottom",
        fontsize=label_size,
        color=ink,
    )
    axis.text(
        day_i1_left + day_i1_width,
        bar_bottom + bar_height + 0.045,
        _DAY_I1_MATH_LABEL,
        ha="right",
        va="bottom",
        fontsize=label_size,
        color=ink,
    )
    axis.text(
        day_i1_left + day_i1_width / 2.0,
        bar_bottom - 0.055,
        "whole session",
        ha="center",
        va="top",
        fontsize=label_size,
        color=muted,
    )
    window_width = 0.115
    starts = (0.085, 0.255, 0.425)
    target = (day_i1_left + day_i1_width / 2.0, bar_bottom + bar_height + 0.01)
    for start, curvature, color in zip(
        starts,
        (-0.34, -0.24, -0.14),
        _SCHEMATIC_COLORS["window_colors"],
        strict=True,
    ):
        axis.add_patch(
            Rectangle(
                (start, bar_bottom),
                window_width,
                bar_height,
                facecolor=color,
                edgecolor=_SCHEMATIC_COLORS["window_edge"],
                linewidth=0.62,
                zorder=3,
            )
        )
        _draw_schematic_arrow(
            axis,
            (start + window_width / 2.0, bar_bottom + bar_height + 0.01),
            target,
            curvature=curvature,
        )
    _draw_schematic_arrow(
        axis,
        (starts[0], bar_bottom - 0.10),
        (starts[-1] + window_width, bar_bottom - 0.10),
        curvature=0.0,
        color=_SCHEMATIC_COLORS["connector"],
    )
    axis.text(
        (starts[0] + starts[-1] + window_width) / 2.0,
        bar_bottom - 0.145,
        f"window slides through {_DAY_I_MATH_LABEL}",
        ha="center",
        va="top",
        fontsize=label_size,
        color=muted,
    )


def _draw_sliding_pivot_design(
    canvas: _WithinCycleCanvas,
    bottom: float,
) -> None:
    """Draw adjacent windows around a moving pivot within each day."""

    axis = _schematic_axis(
        canvas,
        _PIVOT_BLOCK_LEFT_INCHES,
        bottom,
        _PIVOT_BLOCK_WIDTH_INCHES,
    )
    label_size = max(canvas.tick_label, 4.6)
    ink = _SCHEMATIC_COLORS["ink"]
    muted = _SCHEMATIC_COLORS["muted"]
    pale, _, dark = _SCHEMATIC_COLORS["window_colors"]
    axis.text(
        0.5,
        0.985,
        "Windows across a sliding pivot",
        ha="center",
        va="top",
        fontsize=canvas.base_font,
        color=ink,
        fontweight="bold",
    )
    bar_bottom, bar_height = 0.30, 0.20
    bar_width, window_width = 0.40, 0.105
    for bar_left, day_label in (
        (0.045, _DAY_I_MATH_LABEL),
        (0.545, _DAY_I1_MATH_LABEL),
    ):
        _draw_schematic_bar(
            axis,
            bar_left,
            bar_bottom,
            bar_width,
            bar_height,
            whole_session=False,
        )
        pivot = bar_left + bar_width * 0.52
        for offset, color in ((-window_width, pale), (0.0, dark)):
            axis.add_patch(
                Rectangle(
                    (pivot + offset, bar_bottom),
                    window_width,
                    bar_height,
                    facecolor=color,
                    edgecolor=_SCHEMATIC_COLORS["window_edge"],
                    linewidth=0.62,
                    zorder=3,
                )
            )
        axis.plot(
            [pivot, pivot],
            [bar_bottom - 0.115, bar_bottom + bar_height + 0.035],
            color=ink,
            linewidth=0.85,
            zorder=6,
        )
        _draw_schematic_arrow(
            axis,
            (pivot - window_width / 2.0, bar_bottom + bar_height + 0.01),
            (pivot + window_width / 2.0, bar_bottom + bar_height + 0.01),
            curvature=-0.55,
            both_directions=True,
        )
        axis.text(
            bar_left,
            bar_bottom + bar_height + 0.045,
            day_label,
            ha="left",
            va="bottom",
            fontsize=label_size,
            color=ink,
        )
        axis.text(
            pivot + 0.012,
            bar_bottom - 0.135,
            "pivot",
            ha="left",
            va="top",
            fontsize=label_size,
            color=muted,
        )
        _draw_schematic_arrow(
            axis,
            (bar_left + bar_width * 0.18, bar_bottom - 0.115),
            (bar_left + bar_width * 0.86, bar_bottom - 0.115),
            curvature=0.0,
            color=_SCHEMATIC_COLORS["connector"],
        )
    axis.text(
        0.5,
        bar_bottom - 0.20,
        "pivot slides within each day",
        ha="center",
        va="top",
        fontsize=label_size,
        color=muted,
    )


def _draw_fixed_metric_block(
    canvas: _WithinCycleCanvas,
    summary: pd.DataFrame,
    metric: str,
    y_label: str,
    row_index: int,
) -> list[plt.Axes]:
    """Draw the three fixed-reference windows for one neural measure."""

    axes: list[plt.Axes] = []
    bottom = _row_axes_bottom(row_index)
    top_row = row_index == 0
    bottom_row = row_index == len(_METRIC_DISPLAY) - 1
    for index, (axis_name, window_size, title, x_label, scale) in enumerate(
        _FIXED_DISPLAY_COLUMNS
    ):
        axis = canvas.figure.add_axes(
            canvas.rectangle(
                _fixed_column_left(index),
                bottom,
                _SUBPANEL_WIDTH_INCHES,
                _ROW_AXES_HEIGHT_INCHES,
            )
        )
        _style_version_record_axis(canvas, axis)
        selected = summary.loc[
            summary["metric"].eq(metric)
            & summary["window_type"].eq(axis_name)
            & np.isclose(summary["window_size"], window_size)
            & summary["passes_support_mask"].astype(bool)
        ].copy()
        selected["x_plot"] = selected["x_plot"].astype(float) * scale
        x_limits = _draw_group_curves(axis, selected, metric)
        axis.set_xlim(*x_limits)
        if scale != 1.0:
            axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
            axis.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _position: f"{value:,.0f}")
            )
        if top_row:
            axis.set_title(
                title,
                fontsize=canvas.base_font,
                color=_AXIS_COLOR,
                pad=2.4,
            )
        if bottom_row:
            axis.set_xlabel(
                x_label,
                fontsize=canvas.axis_label,
                color=_AXIS_COLOR,
                labelpad=1.6,
            )
        if index == 0:
            axis.set_ylabel(
                y_label,
                fontsize=canvas.axis_label,
                color=_AXIS_COLOR,
                labelpad=1.8,
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        axes.append(axis)
    return axes


def _draw_pivot_metric_block(
    canvas: _WithinCycleCanvas,
    summary: pd.DataFrame,
    limits: pd.DataFrame,
    metric: str,
    y_label: str,
    row_index: int,
) -> list[plt.Axes]:
    """Validate support and draw both pivot curves on the accepted range."""

    rows = limits.loc[limits["metric"].eq(metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one 30-minute support row for {metric}")
    x_low, x_high = _ACCEPTED_PIVOT_X_LIMITS
    bottom = _row_axes_bottom(row_index)
    top_row = row_index == 0
    bottom_row = row_index == len(_METRIC_DISPLAY) - 1
    axes: list[plt.Axes] = []
    for index, (day, title) in enumerate(
        (("day1", _DAY_I_MATH_LABEL), ("day2", _DAY_I1_MATH_LABEL))
    ):
        axis = canvas.figure.add_axes(
            canvas.rectangle(
                _pivot_column_left(index),
                bottom,
                _SUBPANEL_WIDTH_INCHES,
                _ROW_AXES_HEIGHT_INCHES,
            )
        )
        _style_version_record_axis(canvas, axis)
        selected = summary.loc[
            summary["metric"].eq(metric)
            & summary["day"].eq(day)
            & np.isclose(summary["window_size"], PIVOT_WINDOW_MINUTES)
            & summary["passes_support_mask"].astype(bool)
        ].copy()
        selected = selected.rename(
            columns={
                "pair_experience_group": "experience_group",
                "x_min": "x_plot",
            }
        )
        _draw_group_curves(axis, selected, metric)
        axis.set_xlim(x_low, x_high)
        axis.set_xticks(_ACCEPTED_PIVOT_X_TICKS)
        if top_row:
            canvas.text(
                _pivot_column_left(index) + _SUBPANEL_WIDTH_INCHES / 2.0,
                bottom + _ROW_AXES_HEIGHT_INCHES + _ROW_DAY_LINE_OFFSET_INCHES,
                title,
                ha="center",
                va="bottom",
                fontsize=canvas.base_font,
                color=_AXIS_COLOR,
            )
        if bottom_row:
            axis.set_xlabel(
                "Time in maze (min)",
                fontsize=canvas.axis_label,
                color=_AXIS_COLOR,
                labelpad=1.6,
            )
        if index == 0:
            axis.set_ylabel(
                y_label,
                fontsize=canvas.axis_label,
                color=_AXIS_COLOR,
                labelpad=1.8,
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        axes.append(axis)
    if top_row:
        canvas.text(
            _PIVOT_BLOCK_LEFT_INCHES + _PIVOT_BLOCK_WIDTH_INCHES / 2.0,
            bottom + _ROW_AXES_HEIGHT_INCHES + 0.035,
            "30-min window",
            ha="center",
            va="bottom",
            fontsize=canvas.base_font,
            color=_AXIS_COLOR,
        )
    return axes


def assemble_within_cycle_stability(
    fixed_curve_summary: pd.DataFrame,
    pivot_curve_summary: pd.DataFrame,
    pivot_display_limits: pd.DataFrame,
    *,
    run_mode: RunMode,
    output_dir: Path,
) -> FigureAssemblyResult:
    """Draw and save the accepted within-cycle stability figure.

    The calculation tables are created explicitly in the notebook.  This
    function only applies the publication page geometry: two design
    schematics, three fixed-reference columns, and two sliding-pivot columns
    for each neural measure.
    """

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    fixed = _normalise_fixed_curve_summary(fixed_curve_summary)
    pivot = _normalise_pivot_curve_summary(pivot_curve_summary)
    limits = _normalise_pivot_display_limits(pivot_display_limits)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = REPOSITORY_ROOT / destination

    style = {
        "font.size": 6.0,
        "axes.labelsize": 6.0,
        "axes.titlesize": 6.0,
        "xtick.labelsize": 5.25,
        "ytick.labelsize": 5.25,
        "legend.fontsize": 5.25,
        "axes.linewidth": 0.60,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
    with version_record_style(style):
        canvas = _WithinCycleCanvas()
        schematic_bottom = (
            _PAGE_HEIGHT_INCHES
            - _TOP_PAD_INCHES
            - _SCHEMATIC_HEIGHT_INCHES
        )
        _draw_fixed_reference_design(canvas, schematic_bottom)
        _draw_sliding_pivot_design(canvas, schematic_bottom)
        canvas.panel_label(0.03, _PAGE_HEIGHT_INCHES - 0.04, "a")
        canvas.panel_label(
            _PIVOT_BLOCK_LEFT_INCHES - 0.34,
            _PAGE_HEIGHT_INCHES - 0.04,
            "b",
        )

        for row_index, ((metric, y_label), fixed_letter, pivot_letter) in enumerate(
            zip(_METRIC_DISPLAY, "ceg", "dfh", strict=True)
        ):
            bottom = _row_axes_bottom(row_index)
            legend_bottom = (
                bottom
                + _ROW_AXES_HEIGHT_INCHES
                + _row_title_height(row_index)
                + _row_legend_gap(row_index)
            )
            _draw_experience_legend(canvas, metric, legend_bottom)
            fixed_axes = _draw_fixed_metric_block(
                canvas,
                fixed,
                metric,
                y_label,
                row_index,
            )
            pivot_axes = _draw_pivot_metric_block(
                canvas,
                pivot,
                limits,
                metric,
                y_label,
                row_index,
            )
            row_y_limits = _ACCEPTED_ROW_Y_LIMITS[metric]
            for axis in (*fixed_axes, *pivot_axes):
                axis.set_ylim(*row_y_limits)
                axis.set_yticks(_ACCEPTED_ROW_Y_TICKS[metric])
            canvas.panel_label(
                _LEFT_MARGIN_INCHES - 0.46,
                bottom + _ROW_AXES_HEIGHT_INCHES,
                fixed_letter,
            )
            canvas.panel_label(
                _PIVOT_BLOCK_LEFT_INCHES - 0.46,
                bottom + _ROW_AXES_HEIGHT_INCHES,
                pivot_letter,
            )

        absolute_outputs = save_figure(
            canvas.figure,
            destination / "Extended_Data_Figure_6",
            formats=("svg", "pdf", "png", "tiff"),
            raster_dpi=600,
            preserve_canvas=True,
        )
        plt.close(canvas.figure)

    relative_outputs = tuple(
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
        for path in absolute_outputs
    )
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT_KEY,
        run_mode=run_mode,
        output_paths=relative_outputs,
    )
