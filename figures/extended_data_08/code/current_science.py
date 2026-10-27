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
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch, Polygon, Ellipse
from matplotlib.path import Path as MatplotlibPath
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from PIL import Image
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from release_rendering.plotting import *
from release_rendering.plotting import draw_water_port_icon
from release_rendering.statistics import *
from release_rendering.maze import build_maze, run_path_between
from release_rendering.types import FigureAssemblyResult, PanelResult, RunMode
from release_rendering.common import validate_output_request as _validate_output_request
from release_rendering.common import significance_text
_significance_text = significance_text
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


def summarize_balanced_decoder_pairs(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Average decoder controls within animals and run the four planned tests."""

    animal = pair_frame.groupby(
        ["animal_id", "source_variant", "balance_method"], as_index=False
    ).agg(
        reward_error=("reward_error", "mean"),
        control_error=("control_error", "mean"),
        control_minus_reward=("control_minus_reward", "mean"),
        n_pairs=("pair_key", "nunique"),
    )
    animal.insert(0, "row_type", "animal")
    stats_rows = balanced_contrast_tests(animal)
    display_rows = _paired_display_rows(
        animal,
        group_columns=("source_variant", "balance_method"),
        measurement_columns=("reward_error", "control_error"),
    )
    return pd.concat(
        (pair_frame, animal, stats_rows, display_rows),
        ignore_index=True,
        sort=False,
    )


def summarize_pv_recurrence_pairs(pair_frame: pd.DataFrame) -> pd.DataFrame:
    """Test the two source-session variants with animal-clustered errors."""

    stat_rows: list[dict[str, object]] = []
    raw_p: list[float] = []
    interim: list[tuple[str, object]] = []
    for variant in ("day_i", "day_i_plus_1"):
        frame = pair_frame.loc[pair_frame["source_variant"].eq(variant)]
        if frame["animal_id"].nunique() >= 2:
            fit = cr1_ols(
                frame["reward_location_recurrence"].to_numpy(float),
                np.ones((len(frame), 1), dtype=float),
                frame["animal_id"].astype(str).tolist(),
            )
            result = linear_contrast(fit, [1.0])
            raw_p.append(result.p_two_sided)
            interim.append((variant, result))
        else:
            raw_p.append(np.nan)
            interim.append((variant, None))
    adjusted = benjamini_hochberg(np.asarray(raw_p, dtype=float))
    for index, (variant, result) in enumerate(interim):
        stat_rows.append(
            {
                "row_type": "statistics",
                "source_variant": variant,
                "estimate": np.nan if result is None else result.estimate,
                "ci95_low": np.nan if result is None else result.confidence_low,
                "ci95_high": np.nan if result is None else result.confidence_high,
                "p_two_sided": raw_p[index],
                "p_bh_two": adjusted[index],
            }
        )
    display_rows = _paired_display_rows(
        pair_frame,
        group_columns=("source_variant",),
        measurement_columns=(
            "reward_target_matched_source",
            "control_targets_matched_sources",
        ),
    )
    return pd.concat(
        (pair_frame, pd.DataFrame(stat_rows), display_rows),
        ignore_index=True,
        sort=False,
    )


def _paired_display_rows(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    measurement_columns: tuple[str, str],
) -> pd.DataFrame:
    """Calculate the plotted mean and 95% interval for paired observations."""

    selected_columns = [*group_columns, *measurement_columns]
    selected = frame.loc[:, selected_columns].replace([np.inf, -np.inf], np.nan)
    selected = selected.loc[selected[list(measurement_columns)].notna().all(axis=1)]
    if group_columns:
        groups = selected.groupby(list(group_columns), sort=False, dropna=False)
    else:
        groups = [((), selected)]

    rows: list[dict[str, object]] = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(group_columns, keys, strict=True))
        for column in measurement_columns:
            mean, low, high = mean_t_interval(group[column].to_numpy(float))
            rows.append(
                {
                    "row_type": "display_summary",
                    **identifiers,
                    "display_group": column,
                    "estimate": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_observations": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


_PHYSICAL_REWARD_GRID_BIN = 104


_ALTERNATIVE_ROUTE_BINS = {"2_to_water": 19, "3_to_water": 15}


_PV_COLOR = "#E29957"


_CONTROL_COLOR = "#8DAFC0"


_HOME_COLOR = "#587785"


_NODE2_COLOR = "#5B8E7D"


_NODE3_COLOR = "#9C8060"


_TEXT_COLOR = "#262626"


_VOR_GRAY_BLOCKS = (
    (-0.5, 0.5, 1.0, 1.0),
    (-0.5, 4.5, 1.0, 1.0),
    (-0.5, 8.5, 1.0, 1.0),
    (-0.5, 12.5, 1.0, 1.0),
    (2.5, 13.5, 1.0, 1.0),
    (6.5, 13.5, 1.0, 1.0),
    (10.5, 13.5, 1.0, 1.0),
    (13.5, 12.5, 1.0, 1.0),
    (13.5, 8.5, 1.0, 1.0),
    (13.5, 4.5, 1.0, 1.0),
    (13.5, 0.5, 1.0, 1.0),
    (10.5, -0.5, 1.0, 1.0),
    (6.5, -0.5, 1.0, 1.0),
    (2.5, -0.5, 1.0, 1.0),
    (6.5, 1.5, 1.0, 1.0),
    (6.5, 11.5, 1.0, 1.0),
    (10.5, 5.5, 1.0, 1.0),
    (10.5, 7.5, 1.0, 1.0),
    (5.5, 4.5, 1.0, 1.0),
    (5.5, 8.5, 1.0, 1.0),
    (7.5, 4.5, 1.0, 1.0),
    (7.5, 8.5, 1.0, 1.0),
    (2.5, 5.5, 1.0, 1.0),
    (2.5, 7.5, 1.0, 1.0),
    (-0.5, 2.5, 3.0, 1.0),
    (-0.5, 10.5, 3.0, 1.0),
    (11.5, 10.5, 3.0, 1.0),
    (11.5, 2.5, 3.0, 1.0),
    (5.5, 0.5, 3.0, 1.0),
    (5.5, 12.5, 3.0, 1.0),
    (7.5, 6.5, 7.0, 1.0),
)


_PAGE_WIDTH_IN = 7.5


_PAGE_HEIGHT_IN = 6.04


_PANEL_LABEL_SIZE_PT = 8.0


_SVG_NAMESPACE = "http://www.w3.org/2000/svg"


_SVG_TAG = f"{{{_SVG_NAMESPACE}}}"


_SVG_REFERENCE = re.compile(r"#([A-Za-z_][\w:.-]*)")


@dataclass(frozen=True, slots=True)
class _PagePlacement:
    """One accepted panel rectangle measured from the page top left."""

    panel: str
    left_in: float
    top_in: float
    width_in: float
    height_in: float


_PAGE_PLACEMENTS = (
    _PagePlacement("a", .15, .10, 2.15, 1.95),
    _PagePlacement("b", 2.40, .10, 2.75, 1.95),
    _PagePlacement("c", 5.30, .10, 2.05, 1.95),
    _PagePlacement("d", .15, 1.94, 2.36, 2.00),
    _PagePlacement("e", 2.61, 1.94, 1.55, 2.00),
    _PagePlacement("f", 4.26, 1.94, 1.55, 2.00),
    _PagePlacement("g", 5.91, 1.94, 1.44, 2.00),
    _PagePlacement("h", .15, 3.81, 1.90, 2.08),
    _PagePlacement("i", 2.15, 3.81, 2.97, 2.08),
    _PagePlacement("j", 5.22, 3.81, 2.13, 2.08),
)


def _route_bin_count(route: str) -> int:
    """Return the declared number of spatial bins for one route."""

    if route in {"home_to_water", "water_to_home"}:
        return ROUTE_BINS
    try:
        return _ALTERNATIVE_ROUTE_BINS[route]
    except KeyError as error:
        raise ValueError(f"Unknown ED8 route {route!r}") from error


def _axes_in_inches(
    figure: Figure,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> Axes:
    """Add an axis using the accepted ED8 physical geometry."""

    figure_width, figure_height = figure.get_size_inches()
    return figure.add_axes(
        (
            left / figure_width,
            bottom / figure_height,
            width / figure_width,
            height / figure_height,
        )
    )


def _style_vor_axis(axis: Axes) -> None:
    """Apply the accepted ED8 quantitative-axis typography and line weights."""

    despine(axis)
    axis.tick_params(labelsize=5.4, width=0.55, length=1.8, pad=1.2)
    axis.xaxis.label.set_size(6.0)
    axis.yaxis.label.set_size(6.0)
    axis.title.set_size(6.2)
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#111111")


def _finalize_vor_figure(figure: Figure) -> Figure:
    """Set every text artist to the accepted editable figure font (Arial)."""

    family = figure_font_family()
    for artist in figure.findobj(match=mpl.text.Text):
        artist.set_fontfamily(family)
        artist.set_color(artist.get_color() or _TEXT_COLOR)
    return figure


def _add_vor_panel_label(
    figure: Figure,
    label: str,
    *,
    top_in: float,
) -> None:
    """Place one panel letter at its accepted local physical anchor."""

    width, height = figure.get_size_inches()
    figure.text(
        0.01 / float(width),
        1.0 - top_in / float(height),
        label,
        fontsize=_PANEL_LABEL_SIZE_PT,
        fontweight="bold",
        fontfamily=figure_font_family(),
        color="#111111",
        ha="left",
        va="top",
    )


def _summary_interval(
    display_summary: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    """Read one already calculated display estimate and 95% interval."""

    summary = display_summary.loc[display_summary["display_group"].eq(column)]
    if len(summary) != 1:
        raise ValueError(f"Expected one display summary for {column!r}")
    row = summary.iloc[0]
    return float(row["estimate"]), float(row["ci95_low"]), float(row["ci95_high"])


def _draw_summary_point(
    axis: Axes,
    *,
    x: float,
    estimate: float,
    low: float,
    high: float,
    color: str,
    marker: str,
    open_marker: bool = False,
    marker_size: float = 4.2,
    marker_edge_color: str | None = None,
    marker_edge_width: float = 0.55,
    error_color: str | None = None,
    error_line_width: float = 0.7,
    capsize: float = 1.7,
) -> None:
    """Draw one accepted ED8 mean and 95% interval."""

    yerr = None
    if np.isfinite(low) and np.isfinite(high):
        yerr = np.asarray([[estimate - low], [high - estimate]])
    axis.errorbar(
        x,
        estimate,
        yerr=yerr,
        fmt=marker,
        ms=marker_size,
        mfc="white" if open_marker else color,
        mec=(
            marker_edge_color
            if marker_edge_color is not None
            else (color if open_marker else "#111111")
        ),
        mew=marker_edge_width,
        ecolor=(
            error_color
            if error_color is not None
            else (color if open_marker else "#111111")
        ),
        elinewidth=error_line_width,
        capsize=capsize,
        zorder=5,
    )


def _statistic_value(
    values: pd.DataFrame,
    column: str,
    **selectors: str,
) -> float:
    """Read one named inferential value without recalculating the test."""

    selected = values.loc[values["row_type"].eq("statistics")]
    for name, value in selectors.items():
        selected = selected.loc[selected[name].eq(value)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one statistics row for {selectors}, found {len(selected)}"
        )
    return float(selected.iloc[0][column])


def _draw_significance_tick(
    axis: Axes,
    *,
    x1: float,
    x2: float,
    y: float,
    height: float,
    p_value: float,
) -> None:
    """Keep the comparison height and token, centered in a compact gap."""
    _draw_compact_tick(axis,x1,x2,y+height,_significance_text(p_value))


def _draw_compact_tick(axis: Axes, left: float, right: float, height: float, text: str) -> None:
    """Short physical caps and a true text gap never mask observations."""
    bounds=axis.get_window_extent()
    width_mm,height_mm=np.array([bounds.width,bounds.height])/axis.figure.dpi*25.4
    low,high=axis.get_ylim()
    y=(height-low)/(high-low)
    weight="normal" if text=="n.s." else "bold"
    fontsize=5.4 if text=="n.s." else 10.0
    padding_mm=.25
    glyph=TextPath((0,0),text,size=fontsize,prop=FontProperties(family="Arial",weight=weight)).get_extents()
    mm_per_pt=25.4/72.0
    half_gap=(glyph.width*mm_per_pt/2+padding_mm)/width_mm*np.ptp(axis.get_xlim())
    assert half_gap < (right-left)/2, "Comparison span cannot contain its label"
    center=(left+right)/2
    cap=.20/height_mm
    key=f"compact_{left:g}_{right:g}"
    axis.plot([left,left,center-half_gap],[y-cap,y,y],transform=axis.get_xaxis_transform(),
              color="black",lw=.6,clip_on=False,zorder=8,gid=key+"_left")
    axis.plot([center+half_gap,right,right],[y,y,y-cap],transform=axis.get_xaxis_transform(),
              color="black",lw=.6,clip_on=False,zorder=8,gid=key+"_right")
    baseline=y-(glyph.y0+glyph.height/2)*mm_per_pt/height_mm
    axis.text(center,baseline,text,transform=axis.get_xaxis_transform(),ha="center",va="baseline",
              fontsize=fontsize,fontweight=weight,color="black",clip_on=False,zorder=9,gid=key+"_text")


def _vor_route_polyline(route: str) -> FloatArray:
    """Return VOR display coordinates from E's canonical maze graph."""

    maze = build_maze()
    points: list[tuple[float, float]] = []
    for node in route_path_nodes(route):
        if int(node) == maze.exit_run:
            root_end = int(maze.runs[0][-1])
            points.append((-0.5, float(maze.cell_y[root_end])))
        else:
            end_cell = int(maze.runs[int(node)][-1])
            points.append((float(maze.cell_x[end_cell]), float(maze.cell_y[end_cell])))
    return np.asarray(points, dtype=float)


def plot_decoder_error_map(
    values: pd.DataFrame, *, control_color: str = _CONTROL_COLOR
) -> Figure:
    """Draw the equal-animal error map in the accepted maze-view grammar."""
    import matplotlib.patheffects as path_effects

    cohort = values.loc[values["row_type"].eq("cohort_bin")]
    image = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=float)
    image[cohort["by"].to_numpy(dtype=int), cohort["bx"].to_numpy(dtype=int),] = cohort[
        "decoding_error"
    ].to_numpy(float)
    figure = plt.figure(figsize=(2.15, 1.95), facecolor="white")
    axis = _axes_in_inches(figure, left=.47, bottom=.25, width=1.51, height=1.51)
    color_axis = _axes_in_inches(
        figure, left=.25, bottom=.25, width=.065, height=1.51
    )
    colormap = mpl.colormaps["cividis"].copy()
    colormap.set_bad("white")
    artist = axis.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        extent=(-0.5, 14.5, -0.5, 14.5),
        interpolation="none",
        cmap=colormap,
        vmin=1.0,
        vmax=14.0,
        alpha=0.95,
        zorder=1,
    )
    maze = build_maze()
    axis.plot(
        maze.wall_xy[:, 0],
        maze.wall_xy[:, 1],
        color="#000000",
        linewidth=0.85,
        solid_joinstyle="miter",
        solid_capstyle="projecting",
        zorder=5,
    )
    outline_size = 1.10
    outline_offset = outline_size / 2.0
    for spatial_bin in DECODER_CONTROL_BINS:
        axis.add_patch(
            Rectangle(
                (
                    spatial_bin % GRID_SIZE - outline_offset,
                    spatial_bin // GRID_SIZE - outline_offset,
                ),
                outline_size,
                outline_size,
                fill=False,
                edgecolor=control_color,
                linewidth=2.00,
                path_effects=[
                    path_effects.Stroke(linewidth=3.21, foreground="white"),
                    path_effects.Normal(),
                ],
                zorder=7,
            )
        )
    axis.add_patch(
        Rectangle(
            (
                DECODER_TARGET_BIN % GRID_SIZE - outline_offset,
                DECODER_TARGET_BIN // GRID_SIZE - outline_offset,
            ),
            outline_size,
            outline_size,
            fill=False,
            edgecolor=_PV_COLOR,
            linewidth=1.76,
            path_effects=[
                path_effects.Stroke(linewidth=2.73, foreground="white"),
                path_effects.Normal(),
            ],
            zorder=8,
        )
    )
    axis.set_xlim(-0.5, 14.5)
    axis.set_ylim(14.5, -0.5)
    axis.set_aspect("equal")
    axis.axis("off")
    reward_axes_y = axis.transAxes.inverted().transform(
        axis.transData.transform(
            (float(DECODER_TARGET_BIN % GRID_SIZE), float(DECODER_TARGET_BIN // GRID_SIZE))
        )
    )[1]
    draw_water_port_icon(axis, y_axes=float(reward_axes_y))
    colorbar = figure.colorbar(artist, cax=color_axis, orientation="vertical")
    colorbar.set_ticks([2, 6, 10, 14])
    colorbar.set_label("Decoding error (tiles)", fontsize=6.0, labelpad=1.0)
    colorbar.ax.tick_params(labelsize=5.4, width=0.55, length=1.8, pad=1.0)
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.outline.set_linewidth(0.55)
    _add_vor_panel_label(figure, "a", top_in=.12)
    return _finalize_vor_figure(figure)


def plot_balanced_decoder(values: pd.DataFrame) -> Figure:
    """Draw the four accepted balanced-decoder cells on one shared axis."""

    animal = values.loc[values["row_type"].eq("animal")]
    display_rows = values.loc[values["row_type"].eq("display_summary")]
    figure = plt.figure(figsize=(2.75, 1.95), facecolor="white")
    axis = _axes_in_inches(figure, left=.40, bottom=.30, width=2.25, height=1.35)
    # Set the unchanged final limits before laying out physical-size brackets.
    axis.set_xlim(-0.55, 3.80)
    axis.set_ylim(0.0, 13.1)
    order = (
        ("day_i", "equal_frames", 0.0, "Day i\nequal frames"),
        ("day_i", "equal_visits", 1.0, "Day i\nequal visits"),
        ("day_i_plus_1", "equal_frames", 2.25, "Day i+1\nequal frames"),
        ("day_i_plus_1", "equal_visits", 3.25, "Day i+1\nequal visits"),
    )
    for variant, method, center, _label in order:
        frame = animal.loc[
            animal["source_variant"].eq(variant) & animal["balance_method"].eq(method)
        ]
        summary = display_rows.loc[
            display_rows["source_variant"].eq(variant)
            & display_rows["balance_method"].eq(method)
        ]
        # More within-pair space fits 10 pt stars while group centers stay fixed.
        reward_x, control_x = center - 0.20, center + 0.20
        for row in frame.itertuples(index=False):
            axis.plot(
                [reward_x, control_x],
                [row.reward_error, row.control_error],
                color="#CFCFCF",
                lw=0.5,
                zorder=1,
            )
        axis.scatter(
            np.full(len(frame), reward_x),
            frame["reward_error"],
            s=5,
            color=_PV_COLOR,
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )
        axis.scatter(
            np.full(len(frame), control_x),
            frame["control_error"],
            s=5,
            color=_CONTROL_COLOR,
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )
        for x, column, color, marker in (
            (reward_x, "reward_error", _PV_COLOR, "o"),
            (control_x, "control_error", _CONTROL_COLOR, "s"),
        ):
            estimate, low, high = _summary_interval(summary, column)
            _draw_summary_point(
                axis,
                x=x,
                estimate=estimate,
                low=low,
                high=high,
                color=color,
                marker=marker,
            )
        _draw_significance_tick(
            axis,
            x1=reward_x,
            x2=control_x,
            y=10.25,
            height=0.18,
            p_value=_statistic_value(
                values,
                "p_bh_four",
                source_variant=variant,
                balance_method=method,
            ),
        )
    axis.set_xlim(-0.55, 3.80)
    axis.set_ylim(0.0, 13.1)
    axis.set_xticks([item[2] for item in order], [item[3] for item in order])
    axis.set_ylabel("Decoding error (tiles)", labelpad=2.0)
    _style_vor_axis(axis)
    legend_handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_PV_COLOR,
            markeredgecolor="#111111",
            markersize=4.0,
            label="Reward",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=_CONTROL_COLOR,
            markeredgecolor="#111111",
            markersize=4.0,
            label="Matched control",
        ),
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(.5, 1.08),
        ncol=2,
        columnspacing=0.75,
        handletextpad=0.3,
        fontsize=4.9,
        frameon=False,
    )
    figure.text((.40+2.25/2)/2.75,.98,"Uniform-prior decoding error",fontsize=6.7,fontweight="bold",va="top",ha="center")
    _add_vor_panel_label(figure, "b", top_in=.12)
    return _finalize_vor_figure(figure)


def plot_pv_recurrence(values: pd.DataFrame) -> Figure:
    """Draw matched and cross-location PV recurrence in the accepted facets."""

    pairs = values.loc[values["row_type"].eq("pair")]
    display_rows = values.loc[values["row_type"].eq("display_summary")]
    figure = plt.figure(figsize=(2.05, 1.95), facecolor="white")
    axes = (
        _axes_in_inches(figure, left=.33, bottom=.25, width=.72, height=1.15),
        _axes_in_inches(figure, left=1.17, bottom=.25, width=.72, height=1.15),
    )
    for index, (axis, (variant, title, color)) in enumerate(
        zip(
            axes,
            (("day_i", "Day i", _PV_COLOR), ("day_i_plus_1", "Day i+1", _PV_COLOR)),
            strict=True,
        )
    ):
        frame = pairs.loc[pairs["source_variant"].eq(variant)]
        summary = display_rows.loc[display_rows["source_variant"].eq(variant)]
        matched = frame[
            ["reward_target_matched_source", "control_targets_matched_sources"]
        ].to_numpy(float)
        cross = frame[
            ["reward_target_control_sources", "control_targets_reward_source"]
        ].to_numpy(float)
        for row in matched:
            axis.plot([0, 1], row, color=color, alpha=0.20, lw=0.45, zorder=1)
        for row in cross:
            axis.plot(
                [0, 1],
                row,
                color="#C4C4C4",
                alpha=0.16,
                lw=0.42,
                ls="--",
                zorder=1,
            )
        axis.scatter(
            np.tile([0, 1], len(frame)),
            matched.ravel(),
            s=6,
            color=color,
            alpha=0.28,
            edgecolor="none",
            zorder=2,
        )
        axis.scatter(
            np.tile([0, 1], len(frame)),
            cross.ravel(),
            s=6,
            facecolor="white",
            edgecolor=_CONTROL_COLOR,
            linewidth=0.35,
            alpha=0.55,
            zorder=2,
        )
        layers = (
            (0, "reward_target_matched_source", matched[:, 0], True),
            (1, "control_targets_matched_sources", matched[:, 1], True),
            (0, "reward_target_control_sources", cross[:, 0], False),
            (1, "control_targets_reward_source", cross[:, 1], False),
        )
        for x, column, layer_values, filled in layers:
            if filled:
                estimate, low, high = _summary_interval(summary, column)
            else:
                estimate, low, high = mean_t_interval(layer_values)
            _draw_summary_point(
                axis,
                x=x + (-0.035 if filled else 0.035),
                estimate=estimate,
                low=low,
                high=high,
                color=color if filled else _CONTROL_COLOR,
                marker="o",
                open_marker=not filled,
                marker_size=3.7,
                marker_edge_color=color if filled else _CONTROL_COLOR,
                marker_edge_width=0.65,
                error_color=color if filled else _CONTROL_COLOR,
                error_line_width=0.65,
                capsize=1.4,
            )
        axis.set_title(
            title,
            color="black",
            pad=2.0,
            fontweight="bold",
            fontstyle="normal",
            fontsize=6.2,
        )
        axis.set_xlim(-0.30, 1.30)
        axis.set_ylim(-0.10, 0.74)
        axis.set_xticks([0, 1], ["Reward\ntarget", "Control\ntargets"])
        if index == 0:
            axis.set_ylabel("PV correlation", labelpad=1.5)
        else:
            axis.set_yticklabels([])
        _style_vor_axis(axis)
        axis.tick_params(axis="x", labelsize=4.7)
        _draw_significance_tick(
            axis,
            x1=-0.035,
            x2=1.035,
            y=0.655,
            height=0.015,
            p_value=_statistic_value(values, "p_bh_two", source_variant=variant),
        )
    legend_handles = (
        Line2D(
            [0],
            [0],
            marker="o",
            color=_PV_COLOR,
            linestyle="",
            markerfacecolor=_PV_COLOR,
            markersize=3.0,
            label="Matched source",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=_CONTROL_COLOR,
            linestyle="",
            markerfacecolor="white",
            markersize=3.0,
            label="Cross-location",
        ),
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.91),
        ncol=1,
        handletextpad=0.3,
        labelspacing=0.25,
        fontsize=4.8,
        frameon=False,
    )
    _add_vor_panel_label(figure, "c", top_in=0.12)
    return _finalize_vor_figure(figure)


def draw_route_schematic() -> tuple[Figure, pd.DataFrame]:
    """Draw the three physical routes in the accepted maze-view grammar."""

    figure = plt.figure(figsize=(1.90, 2.08), facecolor="white")
    axis = _axes_in_inches(
        figure, left=.06, bottom=.25, width=1.78, height=1.58
    )
    maze = build_maze()
    for x, y, width, height in _VOR_GRAY_BLOCKS:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor="#D9D9D9",
                edgecolor="#D9D9D9",
                linewidth=0.2633,
                zorder=1,
            )
        )
    axis.plot(
        maze.wall_xy[:, 0],
        maze.wall_xy[:, 1],
        color="#111111",
        linewidth=0.79,
        zorder=2,
    )
    route_specs = (
        ("home_to_water", "Home", _HOME_COLOR),
        ("2_to_water", "Node 2", _NODE2_COLOR),
        ("3_to_water", "Node 3", _NODE3_COLOR),
    )
    legend_handles: list[Line2D] = []
    for route, label, color in route_specs:
        points = _vor_route_polyline(route)
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            lw=1.55,
            solid_capstyle="round",
            zorder=6,
        )
        axis.scatter(
            [points[0, 0]],
            [points[0, 1]],
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=7,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=1.55,
                marker="o",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=5.0,
                label=label,
            )
        )
    draw_house_icon(axis, -1.30, 7.0, 0.78, zorder=20, door_zorder=21)
    draw_water_drop_icon(
        axis, 15.18, 6.12, 0.75, linewidth=0.75, zorder=20, highlight_zorder=21
    )
    axis.set_xlim(-1.90, 16.25)
    axis.set_ylim(15.35, -0.75)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.02),
        ncol=3,
        columnspacing=0.55,
        handlelength=1.4,
        handletextpad=0.25,
        fontsize=4.8,
        frameon=False,
    )
    values = pd.DataFrame(
        [
            {
                "route": route,
                "start_node": start,
                "end_node": REWARD_NODE,
                "n_route_bins": _route_bin_count(route),
                "physical_reward_grid_bin": _PHYSICAL_REWARD_GRID_BIN,
                "decoder_target_bin": DECODER_TARGET_BIN,
            }
            for route, start in (
                ("home_to_water", 127),
                ("2_to_water", 2),
                ("3_to_water", 3),
            )
        ]
    )
    _add_vor_panel_label(figure, "h", top_in=0.12)
    return _finalize_vor_figure(figure), values


def _svg_canvas_size(root: ET.Element) -> tuple[float, float]:
    """Read a positive four-value SVG view box in points."""

    values = root.attrib.get("viewBox", "").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError("ED8 panel SVG is missing a four-value viewBox")
    width, height = float(values[2]), float(values[3])
    if width <= 0 or height <= 0:
        raise ValueError("ED8 panel SVG viewBox must have positive dimensions")
    return width, height


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    """Make SVG identifiers unique before placing the nine panels."""

    identifiers: dict[str, str] = {}
    for element in root.iter():
        identifier = element.attrib.get("id")
        if identifier:
            renamed = f"{prefix}_{identifier}"
            identifiers[identifier] = renamed
            element.attrib["id"] = renamed
    if not identifiers:
        return

    def replace(value: str) -> str:
        """Replace local SVG references with their prefixed identifiers."""

        return _SVG_REFERENCE.sub(
            lambda match: f"#{identifiers.get(match.group(1), match.group(1))}",
            value,
        )

    for element in root.iter():
        for name, value in tuple(element.attrib.items()):
            if "#" in value:
                element.attrib[name] = replace(value)
        if element.text and "#" in element.text:
            element.text = replace(element.text)
        if element.tail and "#" in element.tail:
            element.tail = replace(element.tail)


def _panel_placement_scale(
    *,
    source_width: float,
    source_height: float,
    placement: _PagePlacement,
    units_per_inch: float,
) -> float:
    """Validate one panel aspect ratio and return its exact slot scale."""

    width_scale = placement.width_in * units_per_inch / source_width
    height_scale = placement.height_in * units_per_inch / source_height
    if not np.isclose(width_scale, height_scale, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"ED8 panel {placement.panel} canvas does not match the accepted "
            f"slot: source={source_width}x{source_height}, "
            f"slot={placement.width_in}x{placement.height_in} inches"
        )
    return float((width_scale + height_scale) / 2.0)


def _assemble_page_svg(
    *,
    sources: Mapping[str, Mapping[str, Path]],
    destination: Path,
) -> None:
    """Compose the accepted editable ED8 SVG page."""

    ET.register_namespace("", _SVG_NAMESPACE)
    width_points = _PAGE_WIDTH_IN * 72.0
    height_points = _PAGE_HEIGHT_IN * 72.0
    root = ET.Element(
        f"{_SVG_TAG}svg",
        {
            "width": f"{width_points:.6f}pt",
            "height": f"{height_points:.6f}pt",
            "viewBox": f"0 0 {width_points:.6f} {height_points:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        f"{_SVG_TAG}rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{width_points:.6f}",
            "height": f"{height_points:.6f}",
            "fill": "#FFFFFF",
        },
    )
    for placement in _PAGE_PLACEMENTS:
        panel_root = ET.parse(sources[placement.panel][".svg"]).getroot()
        _prefix_svg_ids(panel_root, f"ed08_{placement.panel}")
        source_width, source_height = _svg_canvas_size(panel_root)
        scale = _panel_placement_scale(
            source_width=source_width,
            source_height=source_height,
            placement=placement,
            units_per_inch=72.0,
        )
        group = ET.SubElement(
            root,
            f"{_SVG_TAG}g",
            {
                "id": f"panel_{placement.panel}",
                "transform": (
                    f"translate({placement.left_in * 72.0:.6f} "
                    f"{placement.top_in * 72.0:.6f}) scale({scale:.9f})"
                ),
            },
        )
        for child in panel_root:
            if child.tag != f"{_SVG_TAG}metadata":
                group.append(copy.deepcopy(child))
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def _assemble_page_pdf(
    *,
    sources: Mapping[str, Mapping[str, Path]],
    destination: Path,
) -> None:
    """Compose the accepted vector ED8 PDF page."""

    width_points = _PAGE_WIDTH_IN * 72.0
    height_points = _PAGE_HEIGHT_IN * 72.0
    page = PageObject.create_blank_page(width=width_points, height=height_points)
    for placement in _PAGE_PLACEMENTS:
        panel_page = PdfReader(sources[placement.panel][".pdf"]).pages[0]
        source_width = float(panel_page.mediabox.width)
        source_height = float(panel_page.mediabox.height)
        scale = _panel_placement_scale(
            source_width=source_width,
            source_height=source_height,
            placement=placement,
            units_per_inch=72.0,
        )
        left = placement.left_in * 72.0
        bottom = height_points - (placement.top_in + placement.height_in) * 72.0
        page.merge_transformed_page(
            panel_page,
            Transformation().scale(scale).translate(tx=left, ty=bottom),
            over=True,
        )
    writer = PdfWriter()
    writer.add_page(page)
    with destination.open("wb") as stream:
        writer.write(stream)


def _assemble_page_raster(
    *,
    sources: Mapping[str, Mapping[str, Path]],
    destination: Path,
    dpi: int,
) -> None:
    """Compose one exact-size ED8 PNG or LZW TIFF page."""

    canvas = Image.new(
        "RGB",
        (
            round(_PAGE_WIDTH_IN * dpi),
            round(_PAGE_HEIGHT_IN * dpi),
        ),
        "white",
    )
    for placement in _PAGE_PLACEMENTS:
        with Image.open(sources[placement.panel][".png"]) as opened:
            panel = opened.convert("RGBA")
        target_size = (
            round(placement.width_in * dpi),
            round(placement.height_in * dpi),
        )
        if (
            abs(panel.width - target_size[0]) > 1
            or abs(panel.height - target_size[1]) > 1
        ):
            source_ratio = panel.width / panel.height
            target_ratio = target_size[0] / target_size[1]
            if not np.isclose(source_ratio, target_ratio, rtol=0.0, atol=1e-3):
                raise ValueError(
                    f"ED8 panel {placement.panel} raster canvas does not match "
                    f"the accepted slot: source={panel.size}, target={target_size}"
                )
        if panel.size != target_size:
            panel = panel.resize(target_size, Image.Resampling.LANCZOS)
        canvas.paste(
            panel,
            (
                round(placement.left_in * dpi),
                round(placement.top_in * dpi),
            ),
            panel,
        )
    save_kwargs: dict[str, object] = {"dpi": (dpi, dpi)}
    if destination.suffix.casefold() in {".tif", ".tiff"}:
        save_kwargs["compression"] = "tiff_lzw"
    canvas.save(destination, **save_kwargs)


def _assemble_page_from_sources(
    *,
    sources: Mapping[str, Mapping[str, Path]],
    output_dir: Path,
    output_stem: str,
) -> tuple[Path, ...]:
    """Write the accepted page bundle from already rendered panels."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{output_stem}.svg"
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_300_path = output_dir / f"{output_stem}_300dpi.png"
    png_600_path = output_dir / f"{output_stem}_600dpi.png"
    tiff_600_path = output_dir / f"{output_stem}_600dpi.tiff"
    _assemble_page_svg(sources=sources, destination=svg_path)
    _assemble_page_pdf(sources=sources, destination=pdf_path)
    _assemble_page_raster(sources=sources, destination=png_300_path, dpi=300)
    _assemble_page_raster(sources=sources, destination=png_600_path, dpi=600)
    _assemble_page_raster(sources=sources, destination=tiff_600_path, dpi=600)
    return svg_path, pdf_path, png_300_path, png_600_path, tiff_600_path


@lru_cache(maxsize=4)
def route_path_nodes(route: str) -> IntArray:
    """Return ordered maze-run nodes for one physical route."""

    maze = build_maze()
    home_to_water = np.concatenate(
        (
            np.asarray([maze.exit_run], dtype=np.int64),
            run_path_between(0, REWARD_NODE, maze),
        )
    )
    if route == "home_to_water":
        return home_to_water
    if route == "water_to_home":
        return home_to_water[::-1].copy()
    if route == "2_to_water":
        return run_path_between(2, REWARD_NODE, maze)
    if route == "3_to_water":
        return run_path_between(3, REWARD_NODE, maze)
    raise ValueError(f"Unknown physical route {route!r}")


ROUTE_BINS = 22


def draw_house_icon(
    axis: Axes,
    x: float,
    y: float,
    size: float,
    *,
    zorder: float,
    door_zorder: float,
) -> None:
    """Draw the black home glyph with a white door, centred at ``(x, y)``.

    ``size`` scales the glyph in data units.  The chimney, roof and walls are
    drawn at ``zorder`` and the door above them at ``door_zorder``.
    """

    axis.add_patch(
        Rectangle(
            (x + 0.23 * size, y - 0.48 * size),
            0.14 * size,
            0.28 * size,
            facecolor="#111111",
            edgecolor="#111111",
            linewidth=0.0,
            clip_on=False,
            zorder=zorder,
        )
    )
    axis.add_patch(
        Polygon(
            (
                (x - 0.56 * size, y - 0.03 * size),
                (x, y - 0.62 * size),
                (x + 0.56 * size, y - 0.03 * size),
                (x + 0.40 * size, y - 0.03 * size),
                (x + 0.40 * size, y + 0.48 * size),
                (x - 0.40 * size, y + 0.48 * size),
                (x - 0.40 * size, y - 0.03 * size),
            ),
            closed=True,
            facecolor="#111111",
            edgecolor="#111111",
            linewidth=0.0,
            joinstyle="miter",
            clip_on=False,
            zorder=zorder,
        )
    )
    axis.add_patch(
        Rectangle(
            (x - 0.10 * size, y + 0.19 * size),
            0.20 * size,
            0.29 * size,
            facecolor="white",
            edgecolor="white",
            linewidth=0.0,
            clip_on=False,
            zorder=door_zorder,
        )
    )


def draw_water_drop_icon(
    axis: Axes,
    x: float,
    y: float,
    size: float,
    *,
    linewidth: float,
    zorder: float,
    highlight_zorder: float,
) -> None:
    """Draw the blue water-port drop with a pale highlight, centred at ``(x, y)``.

    ``size`` scales the glyph in data units and ``linewidth`` sets its dark
    blue outline.  The highlight is drawn at ``highlight_zorder``.
    """

    vertices = [
        (x, y - 0.66 * size),
        (x - 0.60 * size, y - 0.05 * size),
        (x - 0.48 * size, y + 0.52 * size),
        (x, y + 0.58 * size),
        (x + 0.48 * size, y + 0.52 * size),
        (x + 0.60 * size, y - 0.05 * size),
        (x, y - 0.66 * size),
        (x, y - 0.66 * size),
    ]
    codes = [
        MatplotlibPath.MOVETO,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CURVE4,
        MatplotlibPath.CLOSEPOLY,
    ]
    axis.add_patch(
        PathPatch(
            MatplotlibPath(vertices, codes),
            facecolor="#36A3FF",
            edgecolor="#0866B2",
            linewidth=linewidth,
            joinstyle="round",
            clip_on=False,
            zorder=zorder,
        )
    )
    axis.add_patch(
        Ellipse(
            (x - 0.13 * size, y - 0.02 * size),
            width=0.15 * size,
            height=0.30 * size,
            angle=-28,
            facecolor="#C9ECFF",
            edgecolor="none",
            alpha=0.9,
            clip_on=False,
            zorder=highlight_zorder,
        )
    )


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Equal-observation mean and 95% Student-t interval over finite values."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    half = float(stats.sem(values) * stats.t.ppf(0.975, len(values) - 1))
    return mean, mean - half, mean + half


def exact_sign_flip(values: np.ndarray) -> float:
    """Two-sided exact test of mouse paired differences, without Monte Carlo."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    return float(np.mean(np.abs(signs @ values) >= abs(values.sum()) - 1e-12))


def summarize_reward_contrasts(animal_values: pd.DataFrame) -> pd.DataFrame:
    """Paired reward-minus-elsewhere tests with BH across the supplied groups."""

    rows = []
    for group, frame in animal_values.groupby("group", sort=False):
        paired = frame.loc[np.isfinite(frame[["port", "away"]]).all(axis=1)]
        delta = (paired.port - paired.away).to_numpy()
        mean, lo, hi = mean_interval(delta)
        rows.append(
            dict(
                group=group,
                mean=mean,
                lo=lo,
                hi=hi,
                n_mice=len(delta),
                p_exact=exact_sign_flip(delta),
            )
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result.p_exact.to_numpy())
    return result


def balanced_contrast_tests(animal: pd.DataFrame) -> pd.DataFrame:
    """Test mean control minus reward error per cell, with four-test BH adjustment.

    Input rows are animal means, with equal pair weights within each animal.
    Each cell uses its own available animals; no cross-cell complete-case mask
    is imposed. Exact sign flips use the mean contrast, not a t statistic.
    """

    records = []
    for variant in ("day_i", "day_i_plus_1"):
        for method in ("equal_frames", "equal_visits"):
            values = animal.loc[
                animal.source_variant.eq(variant) & animal.balance_method.eq(method),
                "control_minus_reward",
            ].to_numpy(float)
            n = len(values)
            if not 2 <= n <= 20 or not np.isfinite(values).all():
                raise ValueError("Each balanced contrast requires 2 to 20 finite animal means")
            estimate = float(values.mean())
            error = float(values.std(ddof=1) / np.sqrt(n))
            margin = float(stats.t.ppf(0.975, n - 1)) * error
            patterns = 1 << n
            signs = 2 * ((np.arange(patterns)[:, None] >> np.arange(n)) & 1) - 1
            p = float(np.mean(signs @ values / n >= estimate - 1e-12))
            records.append({
                "row_type": "statistics", "source_variant": variant,
                "balance_method": method, "estimate": estimate,
                "ci95_low": estimate - margin, "ci95_high": estimate + margin,
                "n_animals": n, "p_exact_one_sided": p, "sign_patterns": patterns,
            })
    result = pd.DataFrame(records)
    result["p_bh_four"] = benjamini_hochberg(result.p_exact_one_sided.to_numpy())
    return result


ORANGE, BROWN, GRAY = "#E29957", "#B96B2C", _CONTROL_COLOR


STYLE = {
    "font.family": "Arial",
    "font.size": 6.0,
    "axes.labelsize": 6.0,
    "axes.titlesize": 6.2,
    "xtick.labelsize": 5.4,
    "ytick.labelsize": 5.4,
    "legend.fontsize": 4.9,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
}


def _figure(letter: str, size: tuple[float, float]) -> Figure:
    figure = plt.figure(figsize=size, facecolor="white")
    figure.text(
        0.01, 1 - 0.12 / size[1], letter, fontsize=8, fontweight="bold", va="top"
    )
    return figure


def _axis(
    figure: Figure, left: float, bottom: float, width: float, height: float
) -> Axes:
    fw, fh = figure.get_size_inches()
    axis = figure.add_axes([left / fw, bottom / fh, width / fw, height / fh])
    axis.tick_params(
        labelsize=STYLE["xtick.labelsize"], length=2.0, width=0.5, pad=1.2, direction="out"
    )
    return axis


def _bracket(
    axis: Axes, left: float, right: float, height: float, probability: float
) -> None:
    _draw_compact_tick(axis,left,right,height,significance_text(probability))


def _paired(
    axis: Axes,
    rows: pd.DataFrame,
    columns: tuple[str, str],
    positions: tuple[float, float],
    colors: tuple[str, str],
    markers: tuple[str, str] = ("s", "o"),
) -> None:
    if not np.isfinite(rows[list(columns)].to_numpy(float)).all():
        raise ValueError(
            "Paired display requires identical finite animals for both means"
        )
    for first, second in rows[list(columns)].itertuples(index=False, name=None):
        axis.plot(positions, [first, second], color="#CFCFCF", lw=0.5, zorder=1)
    for x, column, color, marker in zip(
        positions, columns, colors, markers, strict=True
    ):
        axis.scatter(
            np.full(len(rows), x),
            rows[column],
            s=5,
            color=color,
            alpha=0.45,
            lw=0,
            zorder=2,
        )
        mean, low, high = _interval(rows[column])
        axis.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt=marker,
            ms=4.2,
            color="black",
            mfc=color,
            mew=0.6,
            lw=0.8,
            capsize=1.6,
            zorder=3,
        )


def _probability(
    statistics: pd.DataFrame, group: str, column: str = "adjusted_p"
) -> float:
    return float(statistics.loc[statistics.group.eq(group), column].iloc[0])


def plot_day_pair_route_advantages(
    first_animals: pd.DataFrame, second_animals: pd.DataFrame
) -> Figure:
    """Show direction effects for the same ordered mice on both day pairs."""
    first = first_animals.pivot(index="animal", columns="group", values="advantage")
    first = first.loc[first.mean(axis=1).sort_values().index]
    second = second_animals.pivot(
        index="animal", columns="group", values="advantage"
    ).reindex(first.index)
    if first.notna().sum().to_dict() != second.notna().sum().to_dict():
        raise ValueError("Day-pair displays require the same animal support")
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("d", (2.36, 2.00))
        for column, (frame, title) in enumerate(
            [(first, "Day i → i + 90"), (second, "Day i + 1 → i + 91")]
        ):
            axis = _axis(figure, 0.38 + column * 1.02, 0.40, 0.82, 1.20)
            for group, color, marker, offset in [
                ("home_to_reward", ORANGE, "D", 0.17),
                ("reward_to_home", BROWN, "s", -0.17),
            ]:
                axis.scatter(
                    frame[group],
                    np.arange(len(first)) + offset,
                    marker=marker,
                    s=10,
                    color=color,
                    lw=0,
                    zorder=2,
                )
                axis.axvline(frame[group].mean(), color=color, lw=0.8)
            axis.set_yticks(np.arange(len(first)), first.index)
            axis.set(
                ylim=(-0.7, len(first) - 0.3),
                xlim=(-0.13, 0.52),
                ylabel="Animal ID" if not column else "",
            )
            axis.set_xticks([0, 0.2, 0.4])
            axis.tick_params(labelsize=6, pad=1.2)
            if column:
                axis.tick_params(labelleft=False)
            axis.set_title(title, fontsize=6, pad=2, fontweight="bold")
        figure.legend(
            handles=[
                Line2D([], [], marker=marker, ls="", color=color, ms=3.1, label=label)
                for marker, color, label in [
                            ("D", ORANGE, "Toward reward"),
                            ("s", BROWN, "Away from reward"),
                ]
            ],
            loc="upper center",
            bbox_to_anchor=(0.57, 0.99),
            ncol=2,
            fontsize=6,
            handletextpad=0.25,
            columnspacing=0.6,
            borderaxespad=0.2,
        )
        figure.text(
            0.55,
            0.025,
            "Reward − elsewhere\n(Standardized PV correlation)",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    return figure


def _route_pairs(
    figure: Figure,
    animal_rows: pd.DataFrame,
    statistics: pd.DataFrame,
    groups: list[tuple[str, str, str]],
    rectangle: tuple[float, float, float, float],
    ylabel: str,
    limits: tuple[float, float],
    star_y: float,
    title: str,
) -> Axes:
    # When a plotted mouse value or 95% interval reaches the bracket feet, raise
    # the bracket and the upper axis limit together in steps of 5% of the
    # original span (at most six). Data below the bracket draw as before.
    values: list[float] = []
    for group, _, _ in groups:
        rows = animal_rows.loc[animal_rows.group.eq(group)]
        for column in ("away", "port"):
            values += list(rows[column].to_numpy(float)) + [_interval(rows[column])[2]]
    top, step, steps = float(np.nanmax(values)), 0.05 * (limits[1] - limits[0]), 0
    while top >= star_y - 0.023 * (limits[1] - limits[0]):
        steps += 1
        if steps > 6:
            raise ValueError(f"The bracket cannot clear the data in {title}")
        limits, star_y = (limits[0], limits[1] + step), star_y + step
    axis = _axis(figure, *rectangle)
    axis.set(ylim=limits, xlim=(-0.65, 3.65), ylabel=ylabel)
    for index, (group, color, label) in enumerate(groups):
        rows = animal_rows.loc[animal_rows.group.eq(group)]
        _paired(axis, rows, ("away", "port"), (2 * index, 2 * index + 1), (GRAY, color))
        _bracket(
            axis, 2 * index, 2 * index + 1, star_y, _probability(statistics, group)
        )
        axis.text(
            2 * index + 0.5,
            -0.18,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=4.7,
        )
    axis.set_xticks([0, 1, 2, 3], ["Else-\nwhere", "Reward", "Else-\nwhere", "Reward"])
    axis.tick_params(axis="x", labelsize=4.6, pad=1.2)
    axis.set_title(title, fontsize=6.2, fontweight="bold", pad=4)
    axis.yaxis.label.set_fontsize(5.6)
    return axis


def _direction_groups() -> list[tuple[str, str, str]]:
    return [
        ("home_to_reward", ORANGE, "Toward\nreward"),
        ("reward_to_home", BROWN, "Away from\nreward"),
    ]


def plot_reliability_adjusted_routes(
    animal_rows: pd.DataFrame, statistics: pd.DataFrame
) -> Figure:
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("e", (1.55, 2.00))
        _route_pairs(
            figure,
            animal_rows,
            statistics,
            _direction_groups(),
            (0.40, 0.43, 1.09, 1.20),
            "Standardized PV correlation\n(across-gap / within-day)",
            (-0.35, 0.85),
            0.77,
            "Reliability-adjusted",
        )
    return figure


def plot_equal_frame_route_control(
    animal_rows: pd.DataFrame, statistics: pd.DataFrame
) -> Figure:
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("f", (1.55, 2.00))
        _route_pairs(
            figure,
            animal_rows,
            statistics,
            _direction_groups(),
            (0.39, 0.43, 1.10, 1.20),
            "Standardized PV correlation",
            (-0.12, 0.62),
            0.56,
            "Equal frame control",
        )
    return figure


def plot_reward_delivery_route_control(
    animal_rows: pd.DataFrame, statistics: pd.DataFrame
) -> Figure:
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("g", (1.44, 2.00))
        axis = _axis(figure, 0.40, 0.43, 0.98, 1.20)
        rows = animal_rows.pivot(
            index="animal", columns="group", values="advantage"
        ).reset_index()
        _paired(
            axis, rows, ("rewarded", "timeout"), (0, 1), (ORANGE, ORANGE), ("o", "s")
        )
        axis.set(
            xlim=(-0.5, 1.5),
            ylim=(-0.3, 0.7),
            ylabel="Reward − elsewhere\n(Standardized PV correlation)",
        )
        axis.set_xticks([0, 1], ["Rewarded", "Timeout"])
        axis.yaxis.label.set_fontsize(5.5)
        axis.set_title("Reward delivery", fontsize=6.2, fontweight="bold", pad=4)
        _bracket(
            axis,
            0,
            1,
            0.62,
            _probability(statistics, "rewarded minus timeout", "p_exact"),
        )
    return figure


def plot_node_route_controls(
    animal_rows: pd.DataFrame, statistics: pd.DataFrame
) -> Figure:
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("i", (2.97, 2.08))
        for index, node in enumerate((2, 3)):
            groups = [
                (f"node{node}_to_reward", ORANGE, "Toward\nreward"),
                (f"reward_to_node{node}", BROWN, "Away from\nreward"),
            ]
            axis = _route_pairs(
                figure,
                animal_rows,
                statistics,
                groups,
                (0.37 + index * 1.30, 0.40, 1.14, 1.30),
                "Standardized PV correlation" if not index else "",
                (-0.12, 0.62),
                0.56,
                f"Node {node}",
            )
            axis.title.set_color("#5B8E7D" if node == 2 else "#9C8060")
            if index:
                axis.tick_params(labelleft=False)
    return figure


def plot_cross_direction_route_profiles(
    profile_rows: pd.DataFrame, statistics: pd.DataFrame
) -> Figure:
    with plt.rc_context(with_figure_font(STYLE)):
        figure = _figure("j", (2.13, 2.08))
        axis = _axis(figure, 0.38, 0.38, 1.65, 1.08)
        for measure, color, linestyle, width, label in [
            (
                "same direction across gap: home to reward",
                ORANGE,
                "-",
                0.9,
                "Toward vs toward",
            ),
            (
                "same direction across gap: reward to home",
                BROWN,
                "--",
                0.9,
                "Away vs away",
            ),
            ("cross direction across gap", "black", "-", 1.2, "Toward vs away"),
            (
                "cross direction across gap, shuffle",
                GRAY,
                ":",
                0.9,
                "Neuron ID shuffle",
            ),
        ]:
            rows = profile_rows.loc[profile_rows.measure.eq(measure)]
            profile = np.asarray(
                [_interval(rows.loc[rows.bin.eq(index), "pv"]) for index in range(22)]
            )
            if measure == "cross direction across gap":
                axis.fill_between(
                    range(22),
                    profile[:, 1],
                    profile[:, 2],
                    color="black",
                    alpha=0.12,
                    lw=0,
                )
                port_mean = profile[21, 0]
            axis.plot(
                range(22),
                profile[:, 0],
                color=color,
                ls=linestyle,
                lw=width,
                label=label,
            )
        axis.set(
            xlim=(-0.5, 22.5),
            ylim=(-0.08, 0.45),
            ylabel="Standardized PV correlation",
            xlabel="Position along route",
        )
        axis.set_xticks([0, 21], ["Home", "Reward"])
        axis.yaxis.label.set_fontsize(5.6)
        figure.text(
            0.57,
            0.96,
            "Approach vs departure",
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="top",
        )
        axis.legend(
            loc="lower left",
            bbox_to_anchor=(-0.02, 1),
            ncol=2,
            fontsize=4.9,
            handlelength=1.2,
            columnspacing=0.5,
            handletextpad=0.35,
        )
        axis.text(
            21.35,
            port_mean,
            significance_text(_probability(statistics, "cross direction across gap", "p_exact")),
            fontsize=5.6,
            fontweight="bold",
            va="center",
        )
    return figure

_interval = mean_interval
