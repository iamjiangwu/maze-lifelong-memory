"""Plots and page assembly for Main Figure 5.

The calculations and statistics that this module draws are in
:mod:`maze_lifelong_memory.analyses.main_figure_5`; the tree depth
calculations that other figures share are in
:mod:`maze_lifelong_memory.tree_depth_representation`.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PageObject, PdfReader, PdfWriter, Transformation

HISTOGRAM_EDGES = np.linspace(-1.0, 1.0, 101)
from release_rendering.config import CANONICAL_MAZE_DEPTH, REPOSITORY_ROOT
from release_rendering.maze import build_maze
from release_rendering.plotting import (
    despine,
    figure_font_family,
    savefig_reproducibly,
    version_record_style,
)
from release_rendering.statistics import significance_stars
from release_rendering.types import FigureAssemblyResult, PanelResult, RunMode
from release_rendering.common import _prefix_svg_ids, _result_file, _svg_viewbox


_ARTIFACT_KEY = "main_figure_5"


_LEVEL_COLOR = "#7976A2"
_SHUFFLE_COLOR = "#7F7F7F"

_HEATMAP_CMAP = "magma"

_RECORD_LOGICAL_SIZE = (8.65, 5.97)
_RECORD_OUTPUT_WIDTH_MM = 180.0
_RECORD_OUTPUT_HEIGHT_MM = 124.23121387283236
_RECORD_RASTER_SIZE = (4_250, 2_938)
_RECORD_PANEL_SLOTS = {
    "a": (0.03, 0.17, 3.37, 2.58),
    "b": (3.44, 0.17, 5.20, 2.78),
    "c": (0.03, 3.17, 2.72, 2.62),
    "d": (2.86, 3.17, 1.30, 2.62),
    "e": (4.27, 3.17, 1.10, 2.62),
    "f": (5.48, 3.17, 3.16, 2.62),
}
_RECORD_PANEL_LABEL_BASELINES = {
    "a": (0.03, 0.12 + 8.0 / 72.0),
    "b": (3.44, 0.12 + 8.0 / 72.0),
    "c": (0.03, 0.12 + 3.05 - 3.0 / 72.0),
    "d": (2.86, 0.12 + 3.05 - 3.0 / 72.0),
    "e": (4.27, 0.12 + 3.05 - 3.0 / 72.0),
    "f": (5.48, 0.12 + 3.05 - 3.0 / 72.0),
}
_CURRENT_PANEL_SLOTS = {
    "a": _RECORD_PANEL_SLOTS["a"], "b": _RECORD_PANEL_SLOTS["b"],
    "c": _RECORD_PANEL_SLOTS["c"], "d": (2.86, 3.17, 2.51, 2.62),
    "e": _RECORD_PANEL_SLOTS["f"],
}
_CURRENT_PANEL_LABELS = {
    "a": _RECORD_PANEL_LABEL_BASELINES["a"], "b": _RECORD_PANEL_LABEL_BASELINES["b"],
    "c": _RECORD_PANEL_LABEL_BASELINES["c"], "d": _RECORD_PANEL_LABEL_BASELINES["d"],
    "e": _RECORD_PANEL_LABEL_BASELINES["f"],
}
_DIRECT_LABEL_FONT_SIZE = 6.8


def record_style_settings() -> dict[str, object]:
    """Return the typography and line weights stored in the accepted figure."""

    return {
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "savefig.dpi": 600.0,
    }


def _axes_dx_for_chars(
    panel_size: tuple[float, float], *, left: float, right: float, chars: float
) -> float:
    """Convert an accepted label offset from characters to axis coordinates."""

    axis_width_pt = panel_size[0] * 72.0 * (right - left)
    return chars * (0.55 * _DIRECT_LABEL_FONT_SIZE) / axis_width_pt


def _axes_dy_for_text_heights(
    panel_size: tuple[float, float], *, bottom: float, top: float, units: float
) -> float:
    """Convert an accepted label offset from text heights to axis coordinates."""

    axis_height_pt = panel_size[1] * 72.0 * (top - bottom)
    return units * _DIRECT_LABEL_FONT_SIZE / axis_height_pt


def _add_direct_label(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    va: str = "top",
    fontsize: float = _DIRECT_LABEL_FONT_SIZE,
) -> None:
    """Draw one label at the position used by the accepted renderer."""

    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        ha="left",
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def _horizontal_tree_positions(
    *, depth: int, tree_width: float, tree_height: float
) -> dict[tuple[int, int], tuple[float, float]]:
    """Place a complete binary tree in the accepted horizontal geometry."""

    vertical: dict[tuple[int, int], tuple[float, float]] = {}
    leaf_count = 2 ** (depth - 1)
    leaf_x = np.linspace(0.0, tree_width, leaf_count)
    y_step = tree_height / (depth - 1) if depth > 1 else 0.0
    for index in range(leaf_count):
        vertical[(depth - 1, index)] = (float(leaf_x[index]), 0.0)
    for level in range(depth - 2, -1, -1):
        for index in range(2**level):
            left = vertical[(level + 1, 2 * index)]
            right = vertical[(level + 1, 2 * index + 1)]
            vertical[(level, index)] = (
                0.5 * (left[0] + right[0]),
                left[1] + y_step,
            )
    center = 0.5 * tree_width
    return {
        key: (tree_height - point[1], center - point[0])
        for key, point in vertical.items()
    }


def draw_tree_depth_schematic() -> tuple[Figure, pd.DataFrame]:
    """Draw the seven-level tree and the same level colors on the maze."""

    maze = build_maze(CANONICAL_MAZE_DEPTH)
    color_map = plt.get_cmap(_HEATMAP_CMAP)
    level_count = CANONICAL_MAZE_DEPTH + 1
    level_colors = {
        level: to_hex(color_map(level / (level_count - 1)))
        for level in range(level_count)
    }
    # Distinguish level 1 on the tree and maze without changing the heatmap palette.
    level_colors[1] = "#442067"
    cell_colors = {
        int(cell): level_colors[
            int(np.floor(np.log2(int(maze.run_for_cell[int(cell)]) + 1)))
        ]
        for cell in maze.coordinate_to_cell.values()
    }
    panel_size = (_RECORD_PANEL_SLOTS["a"][2], _RECORD_PANEL_SLOTS["a"][3])
    row_scale = panel_size[1] / 4.4
    # Native panel points, decreasing from root (0) to terminal level (6).
    branch_widths = [3.20, 2.60, 2.00, 1.40, 0.95, 0.65, 0.50]
    positions = _horizontal_tree_positions(
        depth=level_count,
        tree_width=12.55,
        tree_height=3.6,
    )
    with version_record_style(record_style_settings()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=panel_size,
            gridspec_kw={"width_ratios": (0.45, 1.0), "wspace": 0.02},
        )
        tree_axis, maze_axis = axes
        for level in range(level_count - 1):
            for node in range(2**level):
                for child in (2 * node, 2 * node + 1):
                    parent_point = positions[(level, node)]
                    child_point = positions[(level + 1, child)]
                    tree_axis.plot(
                        [parent_point[0], child_point[0]],
                        [parent_point[1], child_point[1]],
                        color=level_colors[level + 1],
                        linewidth=branch_widths[level + 1],
                        solid_capstyle="round",
                    )
        leaf_points = np.asarray(
            [
                positions[(level_count - 1, index)]
                for index in range(2 ** (level_count - 1))
            ],
            dtype=float,
        )
        tree_axis.scatter(
            leaf_points[:, 0],
            leaf_points[:, 1],
            s=2.0 * row_scale,
            color=level_colors[level_count - 1],
            zorder=4,
        )
        root_x, root_y = positions[(0, 0)]
        tree_axis.plot(
            [root_x - 1.15, root_x],
            [root_y, root_y],
            color=level_colors[0],
            linewidth=branch_widths[0],
            solid_capstyle="round",
        )
        label_size = 12.0 * row_scale
        root_label_x = root_x - (1.0 - 0.52) * 1.15
        root_label_y = root_y - 0.40
        tree_axis.text(
            root_label_x,
            root_label_y - 0.18,
            "0",
            ha="center",
            va="center",
            fontsize=label_size,
        )
        tree_axis.text(
            root_label_x + 0.5,
            root_label_y - 5.0,
            "level",
            ha="center",
            va="top",
            fontsize=label_size,
        )
        label_t = {4: 0.50, 5: 0.44, 6: 0.36}
        for level in range(1, level_count):
            parent = positions[(level - 1, 2 ** (level - 1) - 1)]
            child = positions[(level, 2**level - 1)]
            fraction = label_t.get(level, 0.56)
            label_x = (1.0 - fraction) * parent[0] + fraction * child[0]
            label_y = (1.0 - fraction) * parent[1] + fraction * child[1]
            edge = np.asarray(child, dtype=float) - np.asarray(parent, dtype=float)
            normal = np.asarray([-edge[1], edge[0]], dtype=float)
            normal /= np.linalg.norm(normal)
            if normal[1] > 0.0:
                normal *= -1.0
            label_x += 0.40 * normal[0]
            label_y += 0.40 * normal[1]
            tree_axis.text(
                label_x,
                label_y,
                str(level),
                ha="center",
                va="center",
                fontsize=label_size,
            )
        tree_axis.set(xlim=(-1.95, 3.70), ylim=(-6.375, 6.375))
        tree_axis.set_aspect("equal")
        tree_axis.set_anchor("E")
        tree_axis.axis("off")
        for cell, color in cell_colors.items():
            maze_axis.add_patch(
                Rectangle(
                    (float(maze.cell_x[cell]) - 0.5, float(maze.cell_y[cell]) - 0.5),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="none",
                )
            )
        maze_axis.plot(
            maze.wall_xy[:, 0],
            maze.wall_xy[:, 1],
            color="black",
            linewidth=1.55 * row_scale,
            solid_capstyle="round",
        )
        maze_axis.set(xlim=(-0.65, 14.65), ylim=(-0.65, 14.65))
        maze_axis.set_aspect("equal")
        maze_axis.set_anchor("W")
        maze_axis.axis("off")
        figure.subplots_adjust(
            left=0.03, right=0.995, bottom=0.08, top=0.94, wspace=0.02
        )
    values = pd.DataFrame(
        {
            "level": np.arange(level_count, dtype=int),
            "tree_nodes": [2**level for level in range(level_count)],
            "maze_cells": [
                sum(color == level_colors[level] for color in cell_colors.values())
                for level in range(level_count)
            ],
            "color": [level_colors[level] for level in range(level_count)],
        }
    )
    return figure, values


def plot_level_heatmaps(values: pd.DataFrame) -> Figure:
    """Plot already ordered source and target normalized level maps."""

    source = values[
        [f"source_normalized_level_{level}" for level in range(7)]
    ].to_numpy(float)
    target = values[
        [f"target_normalized_level_{level}" for level in range(7)]
    ].to_numpy(float)
    panel_size = (_RECORD_PANEL_SLOTS["b"][2], _RECORD_PANEL_SLOTS["b"][3])
    with version_record_style(record_style_settings()):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=panel_size,
            sharey=True,
            gridspec_kw={"wspace": 0.06},
        )
        image = axes[0].imshow(
            source,
            aspect="auto",
            cmap=_HEATMAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="antialiased",
        )
        axes[1].imshow(
            target,
            aspect="auto",
            cmap=_HEATMAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="antialiased",
        )
        for axis, title in zip(axes, ("Day i", "Day i+90"), strict=True):
            axis.set_title(title, fontsize=8.0, pad=3.0)
            axis.set_xlabel("Levels", fontsize=8.0)
            axis.set_xticks(range(7))
        axes[0].set_ylabel("Neuron", fontsize=8.0)
        # Preserve the accepted full-cohort ticks without stretching a small
        # heatmap to the full 16,000-neuron display range.
        if source.shape[0] >= 16_001:
            axes[0].set_yticks(np.arange(0, 16_001, 2_000))
        axes[1].tick_params(labelleft=False)
        # Heatmaps keep the accepted 2.58 in geometry; the extra 0.20 in below holds "Levels".
        figure.subplots_adjust(
            left=0.11,
            right=0.91,
            bottom=(0.08 * 2.58 + 0.20) / 2.78,
            top=(0.94 * 2.58 + 0.20) / 2.78,
            wspace=0.06,
        )
        colorbar = figure.colorbar(
            image, ax=list(axes), location="right", fraction=0.028, pad=0.018
        )
        colorbar.set_label("Normalized rate", fontsize=7.0)
        colorbar.set_ticks((0.0, 0.2, 0.4, 0.6, 0.8))
        colorbar.ax.tick_params(labelsize=6.5, length=2.0)
    return figure


def plot_level_population_vectors(matrix: pd.DataFrame) -> Figure:
    """Draw the seven by seven pair-mean Level PV matrix, without refitting maps."""
    from matplotlib.colors import LinearSegmentedColormap

    with version_record_style(record_style_settings()):
        figure = plt.figure(figsize=(2.72, 2.62))
        side = 0.72 * 2.62
        axis = figure.add_axes([0.42 / 2.72, 0.22, side / 2.72, 0.72])
        image = axis.pcolormesh(
            np.arange(8) - 0.5,
            np.arange(8) - 0.5,
            matrix.to_numpy(float),
            vmin=0,
            vmax=0.65,
            shading="flat",
            rasterized=False,
            cmap=LinearSegmentedColormap.from_list(
                "level_pv", ["#ffffff", "#7976A2", "#4E4A75"]
            ),
        )
        for level in range(7):
            axis.add_patch(
                Rectangle(
                    (level - 0.5, level - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#333333",
                    lw=0.55,
                )
            )
        axis.set(
            xlim=(-0.5, 6.5),
            ylim=(6.5, -0.5),
            xticks=np.arange(7),
            yticks=np.arange(7),
            xlabel="Later level",
            ylabel="Earlier level",
        )
        axis.set_aspect("equal")
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title("Level PV correlation", fontsize=8, pad=3)
        colorbar_axis = figure.add_axes(
            [2.42 / 2.72, 0.22 + 0.15 / 2.62, 0.07 / 2.72, (side - 0.30) / 2.62]
        )
        colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=[0, 0.3, 0.6])
        colorbar.solids.set_rasterized(False)
        colorbar.ax.tick_params(length=1.5, pad=1)
        colorbar.outline.set_linewidth(0.45)
    return figure


def plot_level_correlation_distribution(values: pd.DataFrame) -> Figure:
    """Plot already computed observed and shuffled level map densities."""

    panel_size = (2.51, 2.62)
    with version_record_style(record_style_settings()):
        figure, axis = plt.subplots(figsize=panel_size)
        width = float(np.diff(HISTOGRAM_EDGES)[0])
        axis.bar(
            values["bin_center"],
            values["shuffle_density"],
            width=width,
            color=_SHUFFLE_COLOR,
            alpha=0.40,
            linewidth=1.0,
            edgecolor="none",
        )
        axis.bar(
            values["bin_center"],
            values["data_density"],
            width=width,
            color=_LEVEL_COLOR,
            alpha=0.72,
            linewidth=1.0,
            edgecolor="none",
        )
        label_y = 0.90 - _axes_dy_for_text_heights(
            panel_size, bottom=0.24, top=0.94, units=3
        )
        data_x = 0.52 + _axes_dx_for_chars(panel_size, left=0.16, right=0.965, chars=4)
        _add_direct_label(axis, 0.30, label_y, "Shuffle", color=_SHUFFLE_COLOR)
        _add_direct_label(axis, data_x, label_y, "Data", color=_LEVEL_COLOR)
        axis.set(xlim=(-1.0, 1.0), xlabel="Level map correlation", ylabel="Density")
        despine(axis)
        figure.subplots_adjust(left=0.16, right=0.965, bottom=0.22, top=0.94)
    return figure


def plot_level_stability_vs_efficiency(
    values: pd.DataFrame,
    inference: dict[str, float | int],
    display_values: pd.DataFrame,
) -> Figure:
    """Plot prepared stability-efficiency points, fit rows, and test result."""

    x = values["level_map_correlation"].to_numpy(float)
    y = values["exploration_efficiency"].to_numpy(float)
    panel_size = (_RECORD_PANEL_SLOTS["f"][2], _RECORD_PANEL_SLOTS["f"][3])
    with version_record_style(record_style_settings()):
        figure, axis = plt.subplots(figsize=panel_size)
        # Match a 3.66 pt dot diameter after full page scaling.
        axis.scatter(
            x, y, s=19.952666666666666, color=_LEVEL_COLOR, alpha=0.75, edgecolors="none", zorder=3
        )
        if not display_values.empty:
            axis.plot(
                display_values["level_map_correlation"],
                display_values["fitted_exploration_efficiency"],
                color=_LEVEL_COLOR,
                linewidth=1.1,
                zorder=2,
            )
            axis.fill_between(
                display_values["level_map_correlation"],
                display_values["ci95_low"],
                display_values["ci95_high"],
                color=_LEVEL_COLOR,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            axis.text(
                0.08,
                0.92,
                significance_stars(
                    float(inference["cluster_p"]),
                    thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")),
                    nonsignificant="n.s.",
                ),
                transform=axis.transAxes,
                color="black",
                fontweight="bold",
                fontsize=12.0,
                ha="left",
                va="top",
            )
            axis.text(
                0.94,
                0.08,
                f"r={float(inference['pearson_r']):.2f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
            )
        axis.set(xlabel="Level map correlation", ylabel="Exploration efficiency")
        axis.set_ylim(bottom=0.0)
        despine(axis)
        figure.subplots_adjust(left=0.18, right=0.95, bottom=0.22, top=0.94)
    return figure


_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_SVG_TAG = f"{{{_SVG_NAMESPACE}}}"


def _compose_tree_depth_svg(results: dict[str, PanelResult], destination: Path) -> None:
    """Compose editable panel SVGs on the accepted publication page."""

    ET.register_namespace("", _SVG_NAMESPACE)
    logical_width = _RECORD_LOGICAL_SIZE[0] * 72.0
    logical_height = _RECORD_LOGICAL_SIZE[1] * 72.0
    root = ET.Element(
        f"{_SVG_TAG}svg",
        {
            "width": f"{_RECORD_OUTPUT_WIDTH_MM:.6f}mm",
            "height": f"{_RECORD_OUTPUT_HEIGHT_MM:.6f}mm",
            "viewBox": f"0 0 {logical_width:.6f} {logical_height:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        f"{_SVG_TAG}rect",
        {"width": "100%", "height": "100%", "fill": "white"},
    )
    for label, result in results.items():
        panel_root, width, height = _svg_viewbox(_result_file(result, ".svg"))
        _prefix_svg_ids(panel_root, f"tree_depth_{label}")
        left, top, slot_width, slot_height = _CURRENT_PANEL_SLOTS[label]
        scale = min(slot_width * 72.0 / width, slot_height * 72.0 / height)
        group = ET.SubElement(
            root,
            f"{_SVG_TAG}g",
            {
                "id": f"tree_depth_{label}",
                "transform": (
                    f"translate({left * 72.0:.6f} {top * 72.0:.6f}) "
                    f"scale({scale:.9f})"
                ),
            },
        )
        for child in panel_root:
            if child.tag != f"{_SVG_TAG}metadata":
                group.append(copy.deepcopy(child))
    for label, (left, baseline) in _CURRENT_PANEL_LABELS.items():
        text = ET.SubElement(
            root,
            f"{_SVG_TAG}text",
            {
                "x": f"{left * 72.0:.6f}",
                "y": f"{baseline * 72.0:.6f}",
                "font-family": figure_font_family(),
                "font-size": "13.0",
                "font-weight": "700",
                "fill": "black",
            },
        )
        text.text = label
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def _label_overlay(format_name: str) -> io.BytesIO:
    """Render the six accepted panel labels as a transparent page."""

    output_size = (
        _RECORD_OUTPUT_WIDTH_MM / 25.4,
        _RECORD_OUTPUT_HEIGHT_MM / 25.4,
    )
    physical_scale = output_size[0] / _RECORD_LOGICAL_SIZE[0]
    stream = io.BytesIO()
    with version_record_style(record_style_settings()):
        figure = plt.figure(figsize=output_size)
        figure.patch.set_alpha(0.0)
        for label, (left, baseline) in _CURRENT_PANEL_LABELS.items():
            figure.text(
                left / _RECORD_LOGICAL_SIZE[0],
                1.0 - baseline / _RECORD_LOGICAL_SIZE[1],
                label,
                ha="left",
                va="baseline",
                fontsize=13.0 * physical_scale,
                fontweight="bold",
            )
        savefig_reproducibly(figure, stream, format=format_name, transparent=True, dpi=600)
        plt.close(figure)
    stream.seek(0)
    return stream


def _compose_tree_depth_pdf(results: dict[str, PanelResult], destination: Path) -> None:
    """Compose the accepted page from freshly generated vector PDF panels."""

    page_width = _RECORD_OUTPUT_WIDTH_MM / 25.4 * 72.0
    page_height = _RECORD_OUTPUT_HEIGHT_MM / 25.4 * 72.0
    x_scale = page_width / _RECORD_LOGICAL_SIZE[0]
    y_scale = page_height / _RECORD_LOGICAL_SIZE[1]
    assembled = PageObject.create_blank_page(width=page_width, height=page_height)
    for label, result in results.items():
        source = PdfReader(_result_file(result, ".pdf")).pages[0]
        left, top, width, height = _CURRENT_PANEL_SLOTS[label]
        scale = min(
            width * x_scale / float(source.mediabox.width),
            height * y_scale / float(source.mediabox.height),
        )
        rendered_height = float(source.mediabox.height) * scale
        assembled.merge_transformed_page(
            source,
            Transformation()
            .scale(scale)
            .translate(
                tx=left * x_scale,
                ty=page_height - top * y_scale - rendered_height,
            ),
            over=True,
        )
    assembled.merge_page(PdfReader(_label_overlay("pdf")).pages[0], over=True)
    writer = PdfWriter()
    writer.add_page(assembled)
    with destination.open("wb") as handle:
        writer.write(handle)


def _compose_tree_depth_png(results: dict[str, PanelResult], destination: Path) -> None:
    """Compose the exact 4250 by 2938 accepted raster canvas."""

    canvas = Image.new("RGBA", _RECORD_RASTER_SIZE, "white")
    x_scale = _RECORD_RASTER_SIZE[0] / _RECORD_LOGICAL_SIZE[0]
    y_scale = _RECORD_RASTER_SIZE[1] / _RECORD_LOGICAL_SIZE[1]
    for label, result in results.items():
        left, top, width, height = _CURRENT_PANEL_SLOTS[label]
        with Image.open(_result_file(result, ".png")) as opened:
            panel = opened.convert("RGBA")
        target_size = (
            max(1, round(width * x_scale)),
            max(1, round(height * y_scale)),
        )
        panel = panel.resize(target_size, Image.Resampling.LANCZOS)
        canvas.alpha_composite(panel, (round(left * x_scale), round(top * y_scale)))
    with Image.open(_label_overlay("png")) as opened:
        labels = opened.convert("RGBA").resize(
            _RECORD_RASTER_SIZE, Image.Resampling.LANCZOS
        )
    canvas.alpha_composite(labels, (0, 0))
    canvas.convert("RGB").save(destination, dpi=(600, 600))


def assemble_tree_depth_results(
    *,
    run_mode: RunMode,
    output_dir: Path,
    depth_definition: PanelResult,
    ordered_level_maps: PanelResult,
    level_population_vectors: PanelResult,
    neuron_identity_distribution: PanelResult,
    stability_efficiency_association: PanelResult,
) -> FigureAssemblyResult:
    """Place the five current calculated results on the publication page."""

    results = {
        "a": depth_definition,
        "b": ordered_level_maps,
        "c": level_population_vectors,
        "d": neuron_identity_distribution,
        "e": stability_efficiency_association,
    }
    for label, result in results.items():
        if result.panel_key != f"{_ARTIFACT_KEY}:{label}":
            raise ValueError(
                f"Expected {_ARTIFACT_KEY}:{label}, found {result.panel_key}"
            )
        if result.run_mode != run_mode:
            raise ValueError(f"Panel {label} run mode differs from {run_mode}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        ".svg": output_dir / "Figure_5.svg",
        ".pdf": output_dir / "Figure_5.pdf",
        ".png": output_dir / "Figure_5.png",
    }
    _compose_tree_depth_svg(results, destinations[".svg"])
    _compose_tree_depth_pdf(results, destinations[".pdf"])
    _compose_tree_depth_png(results, destinations[".png"])
    root = REPOSITORY_ROOT.resolve()
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT_KEY,
        run_mode=run_mode,
        output_paths=tuple(
            path.resolve().relative_to(root) for path in destinations.values()
        ),
    )


__all__ = [
    "assemble_tree_depth_results",
    "draw_tree_depth_schematic",
    "plot_level_stability_vs_efficiency",
    "plot_level_correlation_distribution",
    "plot_level_heatmaps",
    "plot_level_population_vectors",
    "record_style_settings",
]
