"""Figure 4 drawing from decoder pair values, route profiles and illustrative arrays."""
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
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.textpath import TextPath
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
DECODER_SPEC=SimpleNamespace(window_rows=48,window_bounds=(-47,0),is_default=False)
DECODER_WINDOW_SECONDS=1.92
FloatArray = NDArray[np.float64]

SCHEMATIC_SAMPLE_ROWS = 4

SCHEMATIC_SPAN_ROWS = 1_250

def schematic_event_raster(
    activity: FloatArray, window_rows: int
) -> tuple[NDArray[np.bool_], int, int, int]:
    """Return schematic event samples, the shown samples and the window end.

    The diagram shows binarized activity: one tick for each 6.25-Hz sample
    with an event, nothing otherwise. Events are placed at the rising samples
    of the illustrative traces. The decoding window ends at the sample where
    most of the five neurons, then most events, fall inside it.
    """

    samples = np.asarray(activity, dtype=float)[::SCHEMATIC_SAMPLE_ROWS]
    events = np.diff(samples, axis=0, prepend=samples[:1]) > 0.03
    span = SCHEMATIC_SPAN_ROWS // SCHEMATIC_SAMPLE_ROWS
    before = int(round(0.8 * span))
    window = max(1, int(math.ceil(window_rows / SCHEMATIC_SAMPLE_ROWS)))
    best_end, best_key = before, (-1, -1)
    for end in range(before, len(samples) - (span - before)):
        inside = events[end - window + 1:end + 1]
        key = (int(inside.any(axis=0).sum()), int(inside.sum()))
        if key > best_key:
            best_end, best_key = end, key
    start = best_end - before
    return events, start, start + span, best_end
_ARTIFACT = "main_figure_4"

_RECORD_COLOR = "#e29957"

_RECORD_REFERENCE_COLOR = "#C4C4C4"

_RECORD_FIGURE_SIZE = (7.50, 4.00)

_RECORD_OUTPUT_WIDTH_MM = 180.0

_RECORD_OUTPUT_SIZE = (
    _RECORD_OUTPUT_WIDTH_MM / 25.4,
    (_RECORD_OUTPUT_WIDTH_MM / 25.4)
    * _RECORD_FIGURE_SIZE[1]
    / _RECORD_FIGURE_SIZE[0],
)

_RECORD_RASTER_SIZE = (4_250, 2_269)

_RECORD_SLOTS = {
    "a": (0.056, 0.127, 3.822, 1.646),
    "b": (3.878, 0.000, 0.749, 2.089),
    "c": (4.760, 0.205, 2.005, 1.490),
    "d": (6.550, 0.000, 0.749, 2.089),
    "e": (-0.030, 1.724, 4.899, 2.126),
    "f": (4.720, 1.773, 2.650, 2.226),
}

_RECORD_LABEL_POSITIONS = {
    "a": (0.02, 0.12),
    "b": (3.80, 0.12),
    "c": (4.60, 0.12),
    "d": (6.48, 0.12),
    "e": (0.02, 1.82),
    "f": (4.86, 1.87),
}

def _draw_event_raster(
    axis: plt.Axes,
    data: DecoderSchematicData,
    left: float,
    right: float,
    centers: Sequence[float],
) -> int:
    """Draw event ticks and the configured decoding window; return shown rows."""

    events, start, stop, end = schematic_event_raster(data.activity, DECODER_SPEC.window_rows)
    shown = events[start:stop]
    x = np.linspace(left, right, len(shown))
    step = (right - left) / max(len(shown) - 1, 1)
    first, last = DECODER_SPEC.window_bounds
    window_first = end - start + int(math.floor(first / SCHEMATIC_SAMPLE_ROWS))
    window_last = end - start + int(math.ceil(last / SCHEMATIC_SAMPLE_ROWS))
    axis.axvspan(
        max(left, x[window_first] - step / 2.0),
        min(right, x[window_last] + step / 2.0),
        ymin=0.174,
        ymax=0.872,
        color="#e5e5e5",
        alpha=0.85,
        linewidth=0,
        zorder=0,
    )
    for index, center in enumerate(centers):
        baseline = center - 0.040
        ticks = x[shown[:, index]]
        axis.vlines(ticks, baseline, baseline + 0.070, color="#202020", linewidth=0.60)
    return len(shown) * SCHEMATIC_SAMPLE_ROWS

def _draw_decoder_schematic(
    axis: plt.Axes,
    data: DecoderSchematicData,
    *,
    panel_size: tuple[float, float],
) -> None:
    """Draw the accepted stacked-map, activity-trace, and posterior grammar."""

    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    title_font = 6.0
    tiny_font = 4.5
    maze = build_maze(6)
    map_cmap = plt.get_cmap("cividis").copy()
    map_cmap.set_bad("#f4f4f4")
    posterior_cmap = plt.get_cmap("cividis").copy()
    posterior_cmap.set_bad("#f7f7f7")
    trace_centers = (0.825, 0.690, 0.500, 0.365, 0.230)

    map_x, map_width, map_height, shear = 0.012, 0.235, 0.128, 0.105
    map_y = [center - map_height / 2.0 for center in trace_centers]
    for layer, (bottom, place_map) in enumerate(zip(map_y, data.place_maps, strict=True)):
        finite = np.isfinite(place_map)
        baseline = float(np.nanpercentile(place_map[finite], 55.0))
        image = np.where(finite, np.clip(place_map - baseline, 0.0, None), 0.0)
        positive = image[image > 0]
        vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
        sheet = Polygon(
            (
                (map_x, bottom),
                (map_x + map_width, bottom),
                (map_x + map_width + shear, bottom + map_height),
                (map_x + shear, bottom + map_height),
            ),
            closed=True,
            facecolor="#020204",
            edgecolor="none",
            transform=axis.transAxes,
            zorder=1 + layer * 0.1,
        )
        axis.add_patch(sheet)
        tile = image
        transform = (
            matplotlib.transforms.Affine2D()
            .skew_deg(math.degrees(math.atan2(shear, map_height)), 0)
            .translate(map_x, bottom)
            + axis.transAxes
        )
        axis.imshow(
            tile,
            extent=(0.0, map_width, 0.0, map_height),
            origin="upper",
            cmap=map_cmap,
            norm=Normalize(0.0, max(vmax, 1e-12)),
            interpolation="none",
            aspect="auto",
            transform=transform,
            zorder=1.05 + layer * 0.1,
            clip_on=False,
        )
    stack_center = map_x + map_width / 2.0 + shear / 2.0
    gap_mid = (map_y[1] + map_y[2] + map_height) / 2.0
    axis.scatter(
        np.full(3, stack_center),
        gap_mid + np.asarray((0.009, 0.0, -0.009)),
        s=0.12,
        color="#202020",
        transform=axis.transAxes,
        clip_on=False,
        zorder=6,
    )
    axis.text(
        stack_center,
        0.025,
        "Day i place maps",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )

    trace_left, trace_right = 0.405, 0.635
    if DECODER_SPEC.is_default:
        trace_x = np.linspace(trace_left, trace_right, len(data.activity))
        vline_x = float(trace_x[data.highlighted_row])
        frame_dx = (trace_right - trace_left) / max(len(trace_x) - 1, 1)
        axis.axvspan(
            max(trace_left, vline_x - max(frame_dx / 2.0, 0.0014)),
            min(trace_right, vline_x + max(frame_dx / 2.0, 0.0014)),
            ymin=0.174,
            ymax=0.872,
            color="#e5e5e5",
            alpha=0.85,
            linewidth=0,
            zorder=0,
        )
        for index, center in enumerate(trace_centers):
            baseline = center - 0.050
            trace = 0.010 + 0.073 * np.clip(data.activity[:, index], 0.0, 1.0)
            axis.plot(trace_x, baseline + trace, color="#202020", linewidth=0.70)
        displayed_rows = len(trace_x)
    else:
        displayed_rows = _draw_event_raster(
            axis, data, trace_left, trace_right, trace_centers
        )
    trace_gap = (trace_centers[1] + trace_centers[2]) / 2.0
    axis.scatter(
        np.full(3, (trace_left + trace_right) / 2.0),
        trace_gap + np.asarray((0.009, 0.0, -0.009)),
        s=0.12,
        color="#202020",
        transform=axis.transAxes,
        clip_on=False,
        zorder=6,
    )
    scale_length = (10.0 / 0.04) / max(displayed_rows - 1, 1) * (
        trace_right - trace_left
    )
    axis.plot(
        (trace_left, trace_left + scale_length),
        (0.145, 0.145),
        color="#202020",
        linewidth=0.90,
        solid_capstyle="butt",
    )
    axis.text(
        trace_left + scale_length / 2.0,
        0.115,
        "10 s",
        ha="center",
        va="top",
        fontsize=tiny_font,
    )
    axis.text(
        (trace_left + trace_right) / 2.0,
        0.025,
        "Day i + 90 activity",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )
    axis.text(0.377, 0.500, "x", ha="center", va="center", fontsize=6.0)

    posterior_height = 0.710
    posterior_width = posterior_height * panel_size[1] / panel_size[0]
    posterior_x, posterior_y = 0.675, 0.150
    axis.plot(
        (trace_right + 0.010, posterior_x - 0.010),
        (0.500, 0.500),
        color="0.25",
        linewidth=0.65,
        marker=">",
        markevery=[-1],
        markersize=1.25,
        markeredgewidth=0,
    )
    posterior_axis = axis.inset_axes(
        (posterior_x, posterior_y, posterior_width, posterior_height)
    )
    image = gaussian_filter(np.nan_to_num(data.posterior), sigma=0.55)
    posterior_axis.imshow(
        image,
        origin="upper",
        interpolation="nearest",
        cmap=posterior_cmap,
        vmin=0.0,
        vmax=max(float(np.nanmax(image)), 1e-12),
    )
    posterior_axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            GRID_SIZE,
            GRID_SIZE,
            fill=False,
            edgecolor="0.15",
            linewidth=0.60,
        )
    )
    posterior_axis.plot(
        maze.wall_xy[:, 0],
        maze.wall_xy[:, 1],
        color="0.15",
        linewidth=0.42,
        zorder=3,
    )
    posterior_axis.scatter(
        [data.decoded_xy[0]],
        [data.decoded_xy[1]],
        s=2.8,
        marker="o",
        facecolor="none",
        edgecolor="#00A6D6",
        linewidths=0.65,
        zorder=5,
    )
    posterior_axis.scatter(
        [data.observed_xy[0]],
        [data.observed_xy[1]],
        s=2.8,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.65,
        zorder=5,
    )
    posterior_axis.set(xlim=(-0.5, 14.5), ylim=(14.5, -0.5))
    posterior_axis.set_aspect("equal")
    posterior_axis.axis("off")
    colorbar_x = posterior_x + posterior_width * 0.18
    colorbar_y = posterior_y + posterior_height + 0.025
    colorbar_width = posterior_width * 0.64
    colorbar_axis = axis.inset_axes(
        (colorbar_x, colorbar_y, colorbar_width, 0.014)
    )
    colorbar_axis.imshow(
        np.linspace(0.0, 1.0, 128)[None, :],
        aspect="auto",
        cmap=posterior_cmap,
        interpolation="nearest",
    )
    colorbar_axis.axis("off")
    axis.text(
        colorbar_x + colorbar_width / 2.0,
        colorbar_y + 0.020,
        "P(position)",
        fontsize=tiny_font * 0.92,
        ha="center",
        va="bottom",
    )
    legend_y = posterior_y - 0.035
    axis.scatter(
        [posterior_x + 0.030],
        [legend_y],
        s=4.0,
        marker="o",
        facecolor="none",
        edgecolor="#00A6D6",
        linewidths=0.65,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        posterior_x + 0.042,
        legend_y,
        "Decoded",
        fontsize=5.0,
        fontweight="bold",
        color="#00A6D6",
        va="center",
    )
    axis.scatter(
        [posterior_x + 0.150],
        [legend_y],
        s=4.0,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.65,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.text(
        posterior_x + 0.162,
        legend_y,
        "Real",
        fontsize=5.0,
        fontweight="bold",
        color="#F04E37",
        va="center",
    )
    axis.text(
        posterior_x + posterior_width / 2.0,
        0.025,
        "Decoded position",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )

def _record_style_settings() -> dict[str, object]:
    """Return the typography and line weights of the accepted figure."""

    return {
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

def _draw_paired_decoder_comparison(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
    left_label: str,
    right_label: str,
) -> None:
    """Draw outline bars and paired observations with compact significance."""

    left = values[left_column].to_numpy(float)
    right = values[right_column].to_numpy(float)
    for first, second in zip(left, right, strict=True):
        axis.plot(
            (0, 1),
            (first, second),
            color="#8a8a8a",
            linewidth=0.55,
            alpha=0.30,
            zorder=1,
        )
    jitter = np.linspace(-0.045, 0.045, len(left))
    axis.scatter(
        jitter,
        left,
        s=11,
        color=_RECORD_COLOR,
        alpha=0.72,
        edgecolors="none",
        zorder=3,
    )
    axis.scatter(
        1 + jitter,
        right,
        s=11,
        color=_RECORD_REFERENCE_COLOR,
        alpha=0.72,
        edgecolors="none",
        zorder=3,
    )
    for x, observations, color in (
        (0, left, _RECORD_COLOR),
        (1, right, _RECORD_REFERENCE_COLOR),
    ):
        axis.bar(
            x,
            float(np.mean(observations)),
            yerr=float(stats.sem(observations)),
            width=0.68,
            facecolor="none",
            edgecolor=color,
            alpha=1.0,
            linewidth=0.8,
            capsize=2.0,
            error_kw={"zorder": 4},
            zorder=2,
        )
    p_value = float(values["stat_cluster_p"].iloc[0])
    axis.set_xticks((0, 1), (left_label, right_label))
    axis.set_ylabel("Decoding error (bins)", labelpad=0.0)
    # Keep both decoder comparisons on the same scale. The highest displayed
    # observation is 10.897 bins, so a limit of 10 would hide a control value.
    axis.set(xlim=(-0.55, 1.45), ylim=(0.0, 11.5))
    axis.set_yticks((0.0, 5.0, 10.0))
    bracket_y = 11.32 / 11.5
    width_mm, height_mm = axis.get_window_extent().size / axis.figure.dpi * 25.4
    glyph = TextPath((0, 0), _significance_mark(p_value), size=6.5,
                     prop=FontProperties(family="Arial", weight="bold")).get_extents()
    points_per_mm = 72.0 / 25.4
    half_gap = (glyph.width / points_per_mm / 2 + 0.25) / width_mm * 2.0
    cap = 0.20 / height_mm
    assert bracket_y * height_mm + glyph.height / points_per_mm / 2 < height_mm
    # A real gap in the line leaves the star legible without masking data.
    axis.plot(
        (0, 0, 0.5-half_gap),
        (bracket_y-cap, bracket_y, bracket_y),
        transform=axis.get_xaxis_transform(),
        color="black",
        linewidth=0.6,
        gid=f"decoder_comparison_{left_column}_left",
    )
    axis.plot(
        (0.5+half_gap, 1, 1),
        (bracket_y, bracket_y, bracket_y-cap),
        transform=axis.get_xaxis_transform(),
        color="black",
        linewidth=0.6,
        gid=f"decoder_comparison_{left_column}_right",
    )
    axis.text(
        0.5,
        bracket_y - (glyph.y0 + glyph.height/2) / points_per_mm / height_mm,
        _significance_mark(p_value),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="baseline",
        fontsize=6.5,
        fontweight="bold",
        gid=f"decoder_comparison_{left_column}_star",
    )
    despine(axis)
    axis.tick_params(direction="out", pad=1.2)

def _draw_error_map(
    figure: Figure,
    map_axis: plt.Axes,
    colorbar_axis: plt.Axes,
    values: pd.DataFrame,
) -> None:
    """Draw the accepted maze outline and supported per-bin decoder errors."""

    grid = np.full((GRID_SIZE, GRID_SIZE), np.nan, dtype=float)
    supported = values.loc[values["n_frames"].to_numpy(float) >= 5].copy()
    grid[
        supported["by"].to_numpy(int), supported["bx"].to_numpy(int)
    ] = supported["mean_decoding_error"].to_numpy(float)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("white")
    image = map_axis.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=(-0.5, 14.5, -0.5, 14.5),
        interpolation="none",
        cmap=cmap,
        vmin=1.0,
        vmax=14.0,
        alpha=0.95,
        zorder=1,
    )
    maze = build_maze(6)
    map_axis.plot(
        maze.wall_xy[:, 0],
        maze.wall_xy[:, 1],
        color="black",
        linewidth=0.85,
        zorder=5,
    )
    for _, row in values.loc[values["is_control_bin"].astype(bool)].iterrows():
        map_axis.add_patch(
            Rectangle(
                (float(row["bx"]) - 0.55, float(row["by"]) - 0.55),
                1.10,
                1.10,
                fill=False,
                edgecolor=_RECORD_REFERENCE_COLOR,
                linewidth=1.45,
                path_effects=[path_effects.Stroke(linewidth=2.25, foreground="white"), path_effects.Normal()],
                zorder=6,
            )
        )
    reward_rows = values.loc[values["is_decoder_target_bin"].astype(bool)]
    for _, row in reward_rows.iterrows():
        map_axis.add_patch(
            Rectangle(
                (float(row["bx"]) - 0.55, float(row["by"]) - 0.55),
                1.10,
                1.10,
                fill=False,
                edgecolor=_RECORD_COLOR,
                linewidth=1.45,
                path_effects=[path_effects.Stroke(linewidth=2.25, foreground="white"), path_effects.Normal()],
                zorder=7,
            )
        )
    map_axis.set(xlim=(-0.5, 14.5), ylim=(14.5, -0.5))
    map_axis.set_aspect("equal")
    map_axis.axis("off")
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Decoding error (bins)", labelpad=1.0)
    colorbar.set_ticks((2, 4, 6, 8, 10, 12, 14))
    colorbar_axis.yaxis.set_ticks_position("left")
    colorbar_axis.yaxis.set_label_position("left")
    colorbar_axis.tick_params(axis="y", pad=1.0, length=2.0)
    if not reward_rows.empty:
        row = reward_rows.iloc[0]
        reward_axes_y = map_axis.transAxes.inverted().transform(
            map_axis.transData.transform((float(row["bx"]), float(row["by"])))
        )[1]
        draw_water_port_icon(map_axis, y_axes=float(reward_axes_y))

def _axis_in_bounds(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    local: tuple[float, float, float, float],
) -> plt.Axes:
    """Add an axis at local panel coordinates inside one figure rectangle."""

    left, bottom, width, height = bounds
    x, y, local_width, local_height = local
    return figure.add_axes(
        (
            left + x * width,
            bottom + y * height,
            local_width * width,
            local_height * height,
        )
    )

def _draw_standardized_route_results(
    figure: Figure, bounds: tuple[float, float, float, float], bundle: RouteDisplay
) -> None:
    """Draw saved route maps and animal-level standardized PV profiles."""
    for group, title, x0, reverse in (
        ("home_to_reward", "Home to reward", 0.075, False),
        ("reward_to_home", "Reward to home", 0.555, True),
    ):
        axes = [
            _axis_in_bounds(figure, bounds, (x0 + i * 0.130, 0.20, 0.110, 0.64))
            for i in range(2)
        ]
        for axis, matrix, label in zip(
            axes, bundle.heatmaps[group], ("Day i", "Day i + 90"), strict=True
        ):
            image = axis.imshow(
                matrix[:, ::-1] if reverse else matrix,
                aspect="auto",
                origin="lower",
                cmap="cividis",
                vmin=0,
                vmax=0.6,
                interpolation="antialiased",
                rasterized=True,
            )
            axis.set_title(label, fontsize=5.8, pad=3)
            bins = [21, 15, 10, 5, 0] if reverse else [0, 5, 10, 15, 21]
            axis.set_xticks(
                [21 - b if reverse else b for b in bins],
                [str(b) for b in bins],
                fontsize=4.8,
            )
            # Two short lines keep the enlarged labels inside each map's width.
            axis.set_xlabel(title.rsplit(" ", 1)[0] + "\n" + title.rsplit(" ", 1)[1], fontsize=6)
            axis.tick_params(labelsize=4.8, pad=1)
        axes[0].set_ylabel("Neurons", fontsize=5.2)
        axes[1].tick_params(labelleft=False)
        axis = _axis_in_bounds(
            figure, bounds, (x0 + 2 * 0.130 + 0.009, 0.20, 0.100, 0.64)
        )
        curve = bundle.curves.loc[bundle.curves.group.eq(group)].sort_values(
            "bin", ascending=False
        )
        position = 21 - curve.bin.to_numpy()
        axis.fill_betweenx(
            position, curve.lo, curve.hi, color=_RECORD_COLOR, alpha=0.22, lw=0
        )
        axis.plot(curve["mean"], position, color=_RECORD_COLOR, lw=1.3, label="Data")
        axis.plot(
            curve.shuffle,
            position,
            color=_RECORD_REFERENCE_COLOR,
            lw=0.9,
            ls=":",
            label="Neuron ID shuffle",
        )
        axis.set(xlim=(-0.04, 0.36), ylim=(-0.5, 21.5))
        axis.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"], fontsize=4.8)
        axis.set_yticks([21, 0], ["Home", "Reward"], fontsize=4.8)
        axis.yaxis.tick_right()
        axis.tick_params(axis="y", length=0, pad=1.5)
        axis.set_xlabel("Standardized PV\ncorrelation", fontsize=6)
        axis.spines[["top", "left"]].set_visible(False)
        axis.legend(
            frameon=False,
            fontsize=4.6,
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            handlelength=1.6,
        )
        p = float(
            bundle.statistics.loc[bundle.statistics.group.eq(group), "adjusted_p"].iloc[
                0
            ]
        )
        star = _significance_mark(p)
        # Right-aligned just inside the right spine, beside the reward end.
        axis.text(
            0.985,
            1.4,
            star,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#202020",
        )
    colorbar_axis = _axis_in_bounds(figure, bounds, (0.955, 0.34, 0.008, 0.4))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Peak-normalized activity", fontsize=5, labelpad=2)
    colorbar.ax.tick_params(labelsize=4.6, length=1.5, pad=1)

def _draw_reward_decoding_association(
    axis: plt.Axes,
    values: pd.DataFrame,
    display_values: pd.DataFrame,
    *,
    marker_area: float = 15.0,
) -> None:
    """Draw neural x, behavior y, their fit/bootstrap band and CR1 test."""

    x = values["reward_decoding_error"].to_numpy(float)
    y = values["first_long_bout_time_s"].to_numpy(float)
    axis.fill_between(
        display_values["reward_decoding_error"],
        display_values["ci95_low"],
        display_values["ci95_high"],
        color=_RECORD_COLOR,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        display_values["reward_decoding_error"],
        display_values["fitted_first_long_bout_time_s"],
        color=_RECORD_COLOR,
        linewidth=1.1,
    )
    axis.scatter(
        x,
        y,
        s=marker_area,
        color=_RECORD_COLOR,
        alpha=0.75,
        edgecolors="none",
        zorder=3,
    )
    p_value = float(values["stat_cluster_p"].iloc[0])
    axis.text(
        0.04,
        0.95,
        _significance_mark(p_value),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.75,
        color="#202020",
        fontweight="bold",
    )
    axis.text(
        0.96,
        0.08,
        f"r={float(values['stat_pearson_r'].iloc[0]):.2f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
    )
    axis.set_xlabel("Reward decoding error (bins)")
    axis.set_ylabel("First long bout time (s)", labelpad=1.0)
    # Start the error axis at 0 unless every error lies well away from it.
    x_low = 0.0 if x.min() <= 0.15 * x.max() else float(np.floor(x.min() - 0.05 * (x.max() - x.min())))
    axis.set(xlim=(x_low, float(x.max() * 1.08)), ylim=(0.0, float(y.max() * 1.10)))
    despine(axis)
    axis.tick_params(direction="out", pad=1.2)

def _record_slot_bounds(panel: str) -> tuple[float, float, float, float]:
    """Convert one accepted top-left slot from inches to figure coordinates."""

    x, y, width, height = _RECORD_SLOTS[panel]
    return (
        x / _RECORD_FIGURE_SIZE[0],
        1.0 - (y + height) / _RECORD_FIGURE_SIZE[1],
        width / _RECORD_FIGURE_SIZE[0],
        height / _RECORD_FIGURE_SIZE[1],
    )

def _normalize_record_raster_exports(output_paths: Sequence[Path]) -> None:
    """Match the accepted Chrome-rendered PNG and TIFF canvas exactly."""

    for path in output_paths:
        suffix = path.suffix.lower()
        if suffix not in {".png", ".tiff"}:
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != _RECORD_RASTER_SIZE:
                image = image.resize(_RECORD_RASTER_SIZE, Image.Resampling.LANCZOS)
        if suffix == ".tiff":
            image.save(path, format="TIFF", compression="tiff_lzw", dpi=(600, 600))
        else:
            image.save(path, dpi=(600, 600))

def assemble_spatial_decoder_results(
    schematic: DecoderSchematicData,
    whole_maze_values: pd.DataFrame,
    error_map_values: pd.DataFrame,
    reward_control_values: pd.DataFrame,
    route_values: RouteDisplay,
    behavior_values: pd.DataFrame,
    behavior_display: pd.DataFrame,
    *,
    run_mode: RunMode,
    output_dir: Path,
    random_seed: int,
) -> FigureAssemblyResult:
    """Assemble the six calculated results with the accepted figure geometry."""

    destination = _validate_output_request(
        run_mode=run_mode,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    with version_record_style(_record_style_settings()):
        figure = plt.figure(figsize=_RECORD_OUTPUT_SIZE, facecolor="white", dpi=600)

        schematic_bounds = _record_slot_bounds("a")
        schematic_axis = figure.add_axes(schematic_bounds)
        _draw_decoder_schematic(
            schematic_axis,
            schematic,
            panel_size=(_RECORD_SLOTS["a"][2], _RECORD_SLOTS["a"][3]),
        )

        for panel, values, left_column, right_column, labels in (
            (
                "b",
                whole_maze_values,
                "data_error",
                "shuffle_error",
                ("Data", "Shuffle"),
            ),
            (
                "d",
                reward_control_values,
                "reward_error",
                "control_error",
                ("Reward", "Ctrls"),
            ),
        ):
            bounds = _record_slot_bounds(panel)
            axis = _axis_in_bounds(figure, bounds, (0.33, 0.25, 0.65, 0.61))
            _draw_paired_decoder_comparison(
                axis,
                values,
                left_column=left_column,
                right_column=right_column,
                left_label=labels[0],
                right_label=labels[1],
            )

        map_bounds = _record_slot_bounds("c")
        map_height = 0.885
        map_width = map_height * _RECORD_SLOTS["c"][3] / _RECORD_SLOTS["c"][2]
        map_axis = _axis_in_bounds(
            figure, map_bounds, (0.125, 0.072, map_width, map_height)
        )
        colorbar_axis = _axis_in_bounds(
            figure, map_bounds, (0.060, 0.072, 0.032, map_height)
        )
        _draw_error_map(figure, map_axis, colorbar_axis, error_map_values)

        _draw_standardized_route_results(figure, _record_slot_bounds("e"), route_values)

        behavior_bounds = _record_slot_bounds("f")
        behavior_axis = _axis_in_bounds(
            figure, behavior_bounds, (0.21, 0.20, 0.775, 0.775)
        )
        _draw_reward_decoding_association(
            behavior_axis, behavior_values, behavior_display,
            # Account for the accepted page reduction.
            marker_area=15.0 * (_RECORD_OUTPUT_SIZE[0] / _RECORD_FIGURE_SIZE[0]) ** 2,
        )

        for panel, (x, y) in _RECORD_LABEL_POSITIONS.items():
            figure.text(
                x / _RECORD_FIGURE_SIZE[0],
                1.0 - y / _RECORD_FIGURE_SIZE[1],
                panel,
                ha="left",
                va="top",
                fontsize=10.5,
                fontweight="bold",
            )
        output_paths = save_figure(
            figure,
            destination / "Figure_4",
            formats=("svg", "pdf", "png", "tiff"),
            raster_dpi=600,
            preserve_canvas=True,
        )
        _normalize_record_raster_exports(output_paths)
        plt.close(figure)

    root = REPOSITORY_ROOT.resolve()
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT,
        run_mode=run_mode,
        output_paths=tuple(path.resolve().relative_to(root) for path in output_paths),
    )
