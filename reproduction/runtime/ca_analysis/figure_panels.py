"""Shared plotting helpers for the included derived Source Data.

Only the functions required by current figure builders are included. Inputs are
saved tables, image arrays, map vectors, or display coordinates; this module
does not load raw recording sessions or calculate place fields.
"""

from __future__ import annotations

import hashlib

from copy import deepcopy

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib import patches

from matplotlib.colors import ListedColormap

from matplotlib.lines import Line2D

from matplotlib.offsetbox import AnnotationBbox, DrawingArea

from matplotlib.path import Path as MplPath

from matplotlib.patches import FancyArrowPatch

from scipy import stats

from .figure_colors import CORRELATION_MEASURE_COLORS

from .level_heatmap_utils import normalize_maps, stack_level_maps

from .maze import NewMaze

from .spatial import get_level_list

DEFAULT_MAZE_ID = 6

DEFAULT_PANEL_OUTPUT_DIR = Path("figs/figure_panels")

DEFAULT_EXPORT_FORMATS = ("svg",)

VECTOR_EXPORT_FORMATS = {"svg", "pdf", "eps", "ps"}

LINEAR_SCALE_KEYS = (
    "maze_wall_linewidth",
    "gray_block_linewidth",
    "path_linewidth",
    "timeline_reference_linewidth",
)

AREA_SCALE_KEYS = ("path_marker_size", "timeline_scatter_marker_size")

RCPARAM_LINEAR_SCALE_KEYS = ("font.size", "axes.linewidth")


END_NODE_LOWER_THRESHOLD = 63

END_NODE_UPPER_THRESHOLD = 126

PANEL_RCPARAMS = {
    "font.family": "Arial",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "savefig.bbox": "tight",
}

PANEL_LAYOUT = {
    "reference_panel_size": (6, 6),
    "panel_size": (6, 6),
    "maze_wall_linewidth": 3,
    "gray_block_linewidth": 1,
    "quadrant_alpha": 0.2,
    "path_linewidth": 6,
    "path_marker_size": 80,
    "timeline_reference_linewidth": 2,
    "timeline_reference_alpha": 0.3,
    "timeline_scatter_marker_size": 7,
    "timeline_scatter_alpha": 0.5,
}

END_NODE_COLOR_CODE_OPTION_1 = {
    "lower_left": "#e9a752",
    "lower_right": "#accab2",
    "upper_left": "#d44720",
    "upper_right": "#78614d",
}

END_NODE_COLOR_CODE = deepcopy(END_NODE_COLOR_CODE_OPTION_1)

DEFAULT_COLOR_PALETTE = deepcopy(END_NODE_COLOR_CODE)

PANEL_COLORS = {
    "wall": "k",
    "gray_block_face": "lightgray",
    "gray_block_edge": "lightgray",
    "path": "tab:red",
    "reward_reference": DEFAULT_COLOR_PALETTE["lower_right"],
    "quadrants": deepcopy(DEFAULT_COLOR_PALETTE),
}

MAZE_SCHEMATIC_LANDMARKS = {
    "show": True,
    "home": {
        "xy": (-1.33, 6.86),
        "size": 0.88,
        "facecolor": "#111111",
        "edgecolor": "#111111",
        "door_color": "white",
        "linewidth": 0.0,
        "zorder": 8,
    },
    "reward": {
        "xy": (15.18, 6.12),
        "size": 0.82,
        "facecolor": "#36a3ff",
        "edgecolor": "#0866b2",
        "highlight_color": "#c9ecff",
        "linewidth": 0.9,
        "zorder": 8,
    },
}

GRAY_BLOCKS = [
    [-0.5, 0.5, 1, 1],
    [-0.5, 4.5, 1, 1],
    [-0.5, 8.5, 1, 1],
    [-0.5, 12.5, 1, 1],
    [2.5, 13.5, 1, 1],
    [6.5, 13.5, 1, 1],
    [10.5, 13.5, 1, 1],
    [13.5, 12.5, 1, 1],
    [13.5, 8.5, 1, 1],
    [13.5, 4.5, 1, 1],
    [13.5, 0.5, 1, 1],
    [10.5, -0.5, 1, 1],
    [6.5, -0.5, 1, 1],
    [2.5, -0.5, 1, 1],
    [6.5, 1.5, 1, 1],
    [6.5, 11.5, 1, 1],
    [10.5, 5.5, 1, 1],
    [10.5, 7.5, 1, 1],
    [5.5, 4.5, 1, 1],
    [5.5, 8.5, 1, 1],
    [7.5, 4.5, 1, 1],
    [7.5, 8.5, 1, 1],
    [2.5, 5.5, 1, 1],
    [2.5, 7.5, 1, 1],
    [-0.5, 2.5, 3, 1],
    [-0.5, 10.5, 3, 1],
    [11.5, 10.5, 3, 1],
    [11.5, 2.5, 3, 1],
    [5.5, 0.5, 3, 1],
    [5.5, 12.5, 3, 1],
    [7.5, 6.5, 7, 1],
]

EXAMPLE_PATH_POINTS = np.array(
    [
        [3, 11],
        [7, 11],
        [7, 7],
        [7, 3],
        [11, 3],
        [11, 5],
        [13, 5],
        [13, 6],
        [14, 6],
    ],
    dtype=float,
)


GRID_ROUTE_CHOICE_SCHEMATIC_LAYOUT = {
    "reference_panel_size": (2.8, 3.95),
    "panel_size": (2.8, 3.95),
    "xlim": (0.0, 2.8),
    "ylim": (0.0, 3.95),
    "tight_layout": True,
}

GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY = {
    "cell_size": 0.58,
    "cells": [(-1, 1), (0, 0), (0, 1), (0, 2)],
    "rows": {
        "stem_to_branch": {
            "origin": (1.44, 2.12),
            "cells": [(0, 0), (0, 1), (0, 2), (1, 1)],
        },
        "stem_forward": {
            "origin": (0.76, 2.12),
        },
        "branch_to_stem": {
            "origin": (1.44, 0.08),
            "cells": [(0, 0), (0, 1), (0, 2), (1, 1)],
        },
        "branch_forward": {
            "origin": (0.76, 0.08),
        },
    },
}

GRID_ROUTE_CHOICE_SCHEMATIC_COLORS = {
    "cell_edge": "#001b2a",
    "cell_face": "none",
    "active_arrow": "#129bd7",
    "previous_arrow": "#a7a7a7",
    "or": "#129bd7",
    "title": "black",
    "annotation": "#a7a7a7",
}

GRID_ROUTE_CHOICE_SCHEMATIC_STYLE = {
    "cell_linewidth": 1.85,
    "arrow_linewidth": 2.85,
    "previous_arrow_linewidth": 2.85,
    "arrow_mutation_scale": 14,
    "previous_head_length": 0.20,
    "title_fontsize": 12.0,
    "title_fontweight": "bold",
    "or_fontsize": 12.8,
    "annotation_fontsize": 11.5,
    "annotation_lineheight": 1.05,
}

GRID_ROUTE_CHOICE_SCHEMATIC_TEXT = {
    "rows": [
        {
            "key": "stem_to_branch",
            "title": "SA",
            "title_xy": (2.31, 2.42),
            "title_ha": "center",
            "annotations": [
                {
                    "text": "preceding\nturn",
                    "xy": (2.76, 3.66),
                    "ha": "right",
                    "va": "center",
                },
            ],
            "or_xy_units": (0.12, 1.50),
            "or_ha": "left",
            "arrows": [
                {
                    "points_units": [(1.72, 1.60), (0.50, 1.60), (0.50, 2.72)],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(1.72, 1.40), (0.50, 1.40), (0.50, 0.28)],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(1.72, 1.78), (0.76, 1.78), (0.76, 2.56)],
                    "color_key": "previous_arrow",
                    "linestyle": "--",
                    "linewidth_key": "previous_arrow_linewidth",
                },
            ],
        },
        {
            "key": "stem_forward",
            "title": "SF",
            "title_xy": (0.47, 2.42),
            "title_ha": "center",
            "or_xy_units": (0.78, 1.16),
            "or_ha": "center",
            "or_va": "center",
            "arrows": [
                {
                    "points_units": [(-0.72, 1.62), (0.62, 1.62)],
                    "color_key": "active_arrow",
                    "arrowstyle": "-",
                },
                {
                    "points_units": [(0.62, 1.62), (0.62, 1.98)],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(0.62, 1.62), (0.62, 1.30)],
                    "color_key": "active_arrow",
                },
                {
                    "path_units": [
                        (-0.72, 1.42),
                        (-0.08, 1.42),
                        (0.48, 1.42),
                        (0.48, 1.32),
                        (0.48, 1.22),
                        (-0.08, 1.22),
                        (-0.72, 1.22),
                    ],
                    "path_codes": [
                        "MOVETO",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                    ],
                    "color_key": "active_arrow",
                },
            ],
        },
        {
            "key": "branch_to_stem",
            "title": "BS",
            "title_xy": (2.31, 0.42),
            "title_ha": "center",
            "or_xy_units": (0.66, 1.10),
            "or_ha": "left",
            "or_va": "bottom",
            "arrows": [
                {
                    "points_units": [(0.60, 2.72), (0.60, 1.58), (1.72, 1.58)],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(0.40, 2.72), (0.40, 0.28)],
                    "color_key": "active_arrow",
                },
            ],
        },
        {
            "key": "branch_forward",
            "title": "BF",
            "title_xy": (0.47, 0.42),
            "title_ha": "center",
            "or_xy_units": (0.70, 1.08),
            "or_ha": "center",
            "or_va": "bottom",
            "arrows": [
                {
                    "path_units": [
                        (0.60, 2.72),
                        (0.60, 2.10),
                        (0.60, 1.34),
                        (0.70, 1.34),
                        (0.80, 1.34),
                        (0.80, 2.10),
                        (0.80, 2.72),
                    ],
                    "path_codes": [
                        "MOVETO",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                        "CURVE4",
                    ],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(0.40, 2.72), (0.40, 1.56)],
                    "color_key": "active_arrow",
                    "arrowstyle": "-",
                },
                {
                    "points_units": [(0.40, 1.56), (0.05, 1.56)],
                    "color_key": "active_arrow",
                },
                {
                    "points_units": [(0.40, 1.56), (0.40, 1.15)],
                    "color_key": "active_arrow",
                },
            ],
        },
    ],
}

LEVEL_ASSIGNMENT_SCHEMATIC_LAYOUT = {
    "reference_panel_size": (10.8, 4.4),
    "panel_size": (10.8, 4.4),
    "width_ratios": (1.0, 1.28),
    "grid_wspace": 0.16,
}

LEVEL_ASSIGNMENT_SCHEMATIC_GEOMETRY = {
    "tree_depth": 7,
    "tree_width": 8.0,
    "tree_height": 5.0,
    "root_line_length": 2.0,
    "tree_left_pad": 1.3,
    "tree_right_pad": 1.0,
    "tree_vertical_pad": 0.35,
    "maze_margin": 0.45,
}

LEVEL_ASSIGNMENT_SCHEMATIC_COLORS = {
    "cmap_name": "magma",
    "cell_edge": "none",
    "maze_outline": "black",
    "tree_background": "white",
    "maze_background": "white",
    "annotation": "black",
}

LEVEL_ASSIGNMENT_SCHEMATIC_STYLE = {
    "tree_linewidth": 2.1,
    "root_linewidth": 2.1,
    "cell_linewidth": 0.0,
    "maze_outline_linewidth": 1.6,
    "show_tree_nodes": False,
    "node_size": 38,
    "show_leaf_markers": True,
    "leaf_marker_size": 18,
    "label_fontsize": 12,
}

LEVEL_ASSIGNMENT_SCHEMATIC_TEXT = {
    "level_label": {
        "text": "level",
        "position": "left_of_root",
        "dx": -1.15,
        "dy": 0.0,
        "gap_below_zero": 0.18,
        "ha": "right",
        "va": "center",
    },
    "edge_label_t": 0.62,
    "edge_label_t_by_level": {},
    "edge_label_offset": 0.34,
    "root_label_t": 0.55,
    "root_label_offset": 0.34,
    "leaf_label_dx": 0.0,
    "leaf_label_dy": -0.40,
}


def get_panel_scale(panel_size=None, layout: dict | None = None):
    panel_layout = get_panel_layout(layout)
    target_size = tuple(panel_size or panel_layout["panel_size"])
    ref_size = tuple(
        panel_layout.get("reference_panel_size", PANEL_LAYOUT["reference_panel_size"])
    )
    ref_w, ref_h = float(ref_size[0]), float(ref_size[1])
    if ref_w <= 0 or ref_h <= 0:
        return 1.0
    target_w, target_h = float(target_size[0]), float(target_size[1])
    return min(target_w / ref_w, target_h / ref_h)


def get_effective_panel_layout(
    panel_size=None, layout: dict | None = None, *, scale_visuals: bool = True
):
    panel_layout = get_panel_layout(layout)
    if panel_size is not None:
        panel_layout["panel_size"] = tuple(panel_size)
    if not scale_visuals:
        return panel_layout
    scale = get_panel_scale(panel_size=panel_layout["panel_size"], layout=panel_layout)
    panel_layout["visual_scale"] = scale
    for key in LINEAR_SCALE_KEYS:
        panel_layout[key] = panel_layout[key] * scale
    for key in AREA_SCALE_KEYS:
        panel_layout[key] = panel_layout[key] * (scale**2)
    return panel_layout


def get_effective_rcparams(
    overrides: dict | None = None, *, panel_size=None, layout: dict | None = None
):
    rcparams = dict(PANEL_RCPARAMS)
    if overrides:
        rcparams.update(overrides)
    if panel_size is not None:
        scale = get_panel_scale(panel_size=panel_size, layout=layout)
        for key in RCPARAM_LINEAR_SCALE_KEYS:
            if key in rcparams:
                rcparams[key] = rcparams[key] * scale
    return rcparams


def apply_panel_rcparams(
    overrides: dict | None = None, *, panel_size=None, layout: dict | None = None
):
    rcparams = get_effective_rcparams(overrides, panel_size=panel_size, layout=layout)
    plt.rcParams.update(rcparams)
    return rcparams


def get_panel_layout(overrides: dict | None = None):
    layout = dict(PANEL_LAYOUT)
    if overrides:
        layout.update(overrides)
    return layout


def get_panel_colors(overrides: dict | None = None):
    colors = deepcopy(PANEL_COLORS)
    if not overrides:
        return colors
    for key, value in overrides.items():
        if key == "quadrants" and isinstance(value, dict):
            colors["quadrants"].update(value)
        else:
            colors[key] = value
    return colors


def get_panel_maze(maze_id: int = DEFAULT_MAZE_ID):
    return NewMaze(int(maze_id))


def get_maze_schematic_landmarks(overrides: dict | None = None):
    return _merged_dict(MAZE_SCHEMATIC_LANDMARKS, overrides)


def _merged_dict(defaults: dict, overrides: dict | None = None):
    merged = deepcopy(defaults)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def get_grid_route_choice_schematic_layout(overrides: dict | None = None):
    return _merged_dict(GRID_ROUTE_CHOICE_SCHEMATIC_LAYOUT, overrides)


def get_grid_route_choice_schematic_geometry(overrides: dict | None = None):
    return _merged_dict(GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY, overrides)


def get_grid_route_choice_schematic_colors(overrides: dict | None = None):
    return _merged_dict(GRID_ROUTE_CHOICE_SCHEMATIC_COLORS, overrides)


def get_grid_route_choice_schematic_style(overrides: dict | None = None):
    return _merged_dict(GRID_ROUTE_CHOICE_SCHEMATIC_STYLE, overrides)


def get_grid_route_choice_schematic_text(overrides: dict | None = None):
    return _merged_dict(GRID_ROUTE_CHOICE_SCHEMATIC_TEXT, overrides)


def get_level_assignment_schematic_layout(overrides: dict | None = None):
    return _merged_dict(LEVEL_ASSIGNMENT_SCHEMATIC_LAYOUT, overrides)


def get_level_assignment_schematic_geometry(overrides: dict | None = None):
    return _merged_dict(LEVEL_ASSIGNMENT_SCHEMATIC_GEOMETRY, overrides)


def get_level_assignment_schematic_colors(overrides: dict | None = None):
    return _merged_dict(LEVEL_ASSIGNMENT_SCHEMATIC_COLORS, overrides)


def get_level_assignment_schematic_style(overrides: dict | None = None):
    return _merged_dict(LEVEL_ASSIGNMENT_SCHEMATIC_STYLE, overrides)


def get_level_assignment_schematic_text(overrides: dict | None = None):
    return _merged_dict(LEVEL_ASSIGNMENT_SCHEMATIC_TEXT, overrides)


def _unit_vector(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length vector.")
    return v / norm


def _add_styled_poly_arrow(
    ax,
    pts,
    *,
    linewidth,
    color,
    mutation_scale,
    arrowstyle="->",
    linestyle="-",
):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
    path = MplPath(pts, codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        fill=False,
        zorder=3,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def _add_separate_head_poly_arrow(
    ax,
    pts,
    *,
    linewidth,
    color,
    mutation_scale,
    linestyle="--",
    head_length=0.18,
):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        raise ValueError("Arrow paths need at least two points.")
    direction = _unit_vector(pts[-1] - pts[-2])
    head_base = pts[-1] - float(head_length) * direction
    shaft_pts = np.vstack([pts[:-1], head_base])
    ax.plot(
        shaft_pts[:, 0],
        shaft_pts[:, 1],
        color=color,
        lw=linewidth,
        linestyle=linestyle,
        solid_capstyle="butt",
        dash_capstyle="butt",
        solid_joinstyle="miter",
        zorder=2,
    )
    head = FancyArrowPatch(
        posA=head_base,
        posB=pts[-1],
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        fill=True,
        zorder=4,
        joinstyle="miter",
        capstyle="butt",
    )
    ax.add_patch(head)
    return head


def _add_curved_arrow(ax, start, end, *, linewidth, color, mutation_scale, rad):
    patch = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="->",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        fill=False,
        zorder=3,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def _add_styled_path_arrow(
    ax,
    pts,
    codes,
    *,
    linewidth,
    color,
    mutation_scale,
    linestyle="-",
):
    path = MplPath(np.asarray(pts, dtype=float), codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="->",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        fill=False,
        zorder=3,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(patch)
    return patch


def make_panel_figure(panel_size=None, layout: dict | None = None):
    panel_layout = get_effective_panel_layout(
        panel_size=panel_size, layout=layout, scale_visuals=False
    )
    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    return fig, ax


def draw_maze_outline(
    ax, maze, *, panel_size=None, layout: dict | None = None, colors: dict | None = None
):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout)
    panel_colors = get_panel_colors(colors)
    ax.plot(
        maze.wa[:, 0],
        maze.wa[:, 1],
        "-",
        color=panel_colors["wall"],
        linewidth=panel_layout["maze_wall_linewidth"],
    )
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


def draw_gray_blocks(
    ax,
    *,
    panel_size=None,
    blocks=GRAY_BLOCKS,
    layout: dict | None = None,
    colors: dict | None = None,
):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout)
    panel_colors = get_panel_colors(colors)
    for x, y, width, height in blocks:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=panel_layout["gray_block_linewidth"],
                edgecolor=panel_colors["gray_block_edge"],
                facecolor=panel_colors["gray_block_face"],
            )
        )
    return ax


def get_quadrant_specs(colors: dict | None = None):
    panel_colors = get_panel_colors(colors)
    return [
        (
            np.arange(-0.5, 7.5, 2),
            np.arange(-0.5, 7.5, 2),
            panel_colors["quadrants"]["lower_left"],
        ),
        (
            np.arange(7.5, 16.5, 2),
            np.arange(-0.5, 7.5, 2),
            panel_colors["quadrants"]["lower_right"],
        ),
        (
            np.arange(-0.5, 7.5, 2),
            np.arange(7.5, 16.5, 2),
            panel_colors["quadrants"]["upper_left"],
        ),
        (
            np.arange(7.5, 16.5, 2),
            np.arange(7.5, 16.5, 2),
            panel_colors["quadrants"]["upper_right"],
        ),
    ]


def draw_quadrant_overlays(
    ax,
    maze,
    *,
    panel_size=None,
    quadrants=None,
    layout: dict | None = None,
    colors: dict | None = None,
):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout)
    quadrant_specs = quadrants if quadrants is not None else get_quadrant_specs(colors)
    valid_cells = set(maze.ce.keys())
    for xs, ys, color in quadrant_specs:
        for x in xs:
            for y in ys:
                # Shade only real maze cells so schematic panels align cleanly during assembly.
                cell_xy = (int(x + 0.5), int(y + 0.5))
                if cell_xy not in valid_cells:
                    continue
                ax.add_patch(
                    patches.Rectangle(
                        (x, y),
                        1,
                        1,
                        facecolor=color,
                        edgecolor=color,
                        alpha=panel_layout["quadrant_alpha"],
                    )
                )
    return ax


def draw_connected_path(
    ax,
    points,
    *,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    show_vertices: bool = True,
):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout)
    panel_colors = get_panel_colors(colors)
    points = np.asarray(points, dtype=float)
    ax.plot(
        points[:, 0],
        points[:, 1],
        "-",
        linewidth=panel_layout["path_linewidth"],
        color=panel_colors["path"],
    )
    if show_vertices:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=panel_layout["path_marker_size"],
            color=panel_colors["path"],
            zorder=5,
        )
    return ax


def _draw_house_icon(ax, spec: dict):
    x, y = map(float, spec["xy"])
    size = float(spec["size"])
    facecolor = spec.get("facecolor", "#111111")
    edgecolor = spec.get("edgecolor", facecolor)
    linewidth = float(spec.get("linewidth", 0.0))
    zorder = float(spec.get("zorder", 8))

    chimney = patches.Rectangle(
        (x + 0.23 * size, y - 0.48 * size),
        0.14 * size,
        0.28 * size,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
    )
    silhouette = patches.Polygon(
        [
            (x - 0.56 * size, y - 0.03 * size),
            (x, y - 0.62 * size),
            (x + 0.56 * size, y - 0.03 * size),
            (x + 0.40 * size, y - 0.03 * size),
            (x + 0.40 * size, y + 0.48 * size),
            (x - 0.40 * size, y + 0.48 * size),
            (x - 0.40 * size, y - 0.03 * size),
        ],
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
        joinstyle="miter",
    )
    door = patches.Rectangle(
        (x - 0.10 * size, y + 0.19 * size),
        0.20 * size,
        0.29 * size,
        facecolor=spec.get("door_color", "white"),
        edgecolor=spec.get("door_color", "white"),
        linewidth=0.0,
        zorder=zorder + 0.1,
        clip_on=False,
    )
    for patch in (chimney, silhouette, door):
        ax.add_patch(patch)


def _draw_water_drop_icon(ax, spec: dict):
    x, y = map(float, spec["xy"])
    size = float(spec["size"])
    facecolor = spec.get("facecolor", "#36a3ff")
    edgecolor = spec.get("edgecolor", "#0866b2")
    linewidth = float(spec.get("linewidth", 0.9))
    zorder = float(spec.get("zorder", 8))

    verts = [
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
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    drop = patches.PathPatch(
        MplPath(verts, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
        joinstyle="round",
    )
    highlight = patches.Ellipse(
        (x - 0.13 * size, y - 0.02 * size),
        width=0.15 * size,
        height=0.30 * size,
        angle=-28,
        facecolor=spec.get("highlight_color", "#c9ecff"),
        edgecolor="none",
        alpha=0.9,
        zorder=zorder + 0.1,
        clip_on=False,
    )
    ax.add_patch(drop)
    ax.add_patch(highlight)


def draw_maze_landmarks(ax, landmarks: dict | None = None):
    panel_landmarks = get_maze_schematic_landmarks(landmarks)
    if not panel_landmarks.get("show", True):
        return ax
    home = panel_landmarks.get("home")
    reward = panel_landmarks.get("reward")
    if home:
        _draw_house_icon(ax, home)
    if reward:
        _draw_water_drop_icon(ax, reward)
    return ax


def finish_panel(ax):
    ax.axis("off")
    return ax


def format_panel_title(nickname: str, display_title: str | None = None):
    if display_title is not None:
        return str(display_title)
    nickname = str(nickname)
    if nickname.endswith("1stdose"):
        return nickname[:-8]
    return nickname


def get_end_node_color(node, colors: dict | None = None):
    palette = get_panel_colors(colors)["quadrants"]
    node = float(node)
    if 62 < node <= 78:
        return palette["lower_left"]
    if 78 <= node <= 94:
        return palette["lower_right"]
    if 94 <= node <= 110:
        return palette["upper_left"]
    return palette["upper_right"]


def get_end_node_colors(nodes, colors: dict | None = None):
    return [get_end_node_color(node, colors=colors) for node in nodes]


def extract_endnode_timeline(*args, **kwargs):
    """Callback supplied by Figure 1's saved endpoint-table loader."""
    raise RuntimeError("Configure the saved endpoint-table loader before rendering")


def _data_y_to_axes_fraction(ax, y_value: float) -> float:
    ymin, ymax = ax.get_ylim()
    if np.isclose(ymin, ymax):
        return 0.5
    return float((float(y_value) - ymin) / (ymax - ymin))


def _add_timeline_water_drop_icon(ax, xy_axes, marker_spec: dict):
    size = float(marker_spec.get("size", 18.0))
    facecolor = marker_spec.get("facecolor", marker_spec.get("color", "#36a3ff"))
    edgecolor = marker_spec.get("edgecolor", "#0866b2")
    linewidth = float(marker_spec.get("linewidth", 0.9))
    zorder = float(marker_spec.get("zorder", 12))

    drawing = DrawingArea(size, size, 0, 0)
    cx = cy = size / 2.0
    scale = size * 0.74
    vertices = [
        (cx, cy + 0.66 * scale),
        (cx - 0.60 * scale, cy + 0.05 * scale),
        (cx - 0.48 * scale, cy - 0.52 * scale),
        (cx, cy - 0.58 * scale),
        (cx + 0.48 * scale, cy - 0.52 * scale),
        (cx + 0.60 * scale, cy + 0.05 * scale),
        (cx, cy + 0.66 * scale),
        (cx, cy + 0.66 * scale),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    drop = patches.PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        joinstyle="round",
    )
    drawing.add_artist(drop)

    highlight = patches.Ellipse(
        (cx - 0.13 * scale, cy + 0.02 * scale),
        width=0.15 * scale,
        height=0.30 * scale,
        angle=28,
        facecolor=marker_spec.get("highlight_color", "#c9ecff"),
        edgecolor="none",
        alpha=float(marker_spec.get("highlight_alpha", 0.9)),
    )
    drawing.add_artist(highlight)

    annotation_kwargs = {}
    if (
        marker_spec.get("x_offset_points") is not None
        or marker_spec.get("y_offset_points") is not None
    ):
        annotation_kwargs["xybox"] = (
            float(marker_spec.get("x_offset_points", 0.0)),
            float(marker_spec.get("y_offset_points", 0.0)),
        )
        annotation_kwargs["boxcoords"] = "offset points"
    annotation = AnnotationBbox(
        drawing,
        xy_axes,
        xycoords=ax.transAxes,
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        annotation_clip=False,
        zorder=zorder,
        **annotation_kwargs,
    )
    ax.add_artist(annotation)
    return annotation


def _add_timeline_no_water_marker(ax, xy_axes, marker_spec: dict):
    size = float(marker_spec.get("size", 18.0))
    color = marker_spec.get("color", "#D62728")
    linewidth = float(marker_spec.get("linewidth", 2.2))
    zorder = float(marker_spec.get("zorder", 12))
    pad = size * 0.24
    drawing = DrawingArea(size, size, 0, 0)
    drawing.add_artist(
        Line2D(
            [pad, size - pad],
            [pad, size - pad],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
        )
    )
    drawing.add_artist(
        Line2D(
            [pad, size - pad],
            [size - pad, pad],
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
        )
    )
    annotation_kwargs = {}
    if (
        marker_spec.get("x_offset_points") is not None
        or marker_spec.get("y_offset_points") is not None
    ):
        annotation_kwargs["xybox"] = (
            float(marker_spec.get("x_offset_points", 0.0)),
            float(marker_spec.get("y_offset_points", 0.0)),
        )
        annotation_kwargs["boxcoords"] = "offset points"
    annotation = AnnotationBbox(
        drawing,
        xy_axes,
        xycoords=ax.transAxes,
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        annotation_clip=False,
        zorder=zorder,
        **annotation_kwargs,
    )
    ax.add_artist(annotation)
    return annotation


def draw_endnode_timeline_marker(ax, marker: dict | None):
    """Draw a fixed-size endpoint availability marker beside a timeline axis."""
    if marker is None or not marker.get("show", True):
        return None
    node = marker.get("node")
    if node is None:
        return None

    kind = str(marker.get("kind", "water")).lower()
    xy_axes = (
        float(marker.get("x_axes", -0.055)),
        _data_y_to_axes_fraction(ax, float(node))
        + float(marker.get("y_axes_offset", 0.0)),
    )
    if kind in {"water", "drop", "reward"}:
        return _add_timeline_water_drop_icon(ax, xy_axes, marker)
    if kind in {"no_water", "no-water", "x"}:
        return _add_timeline_no_water_marker(ax, xy_axes, marker)
    raise ValueError(
        f"Unsupported endnode timeline marker kind: {marker.get('kind')!r}"
    )


def draw_endnode_timeline_axes(
    ax,
    nickname: str,
    *,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    lower_threshold: int = END_NODE_LOWER_THRESHOLD,
    upper_threshold: int = END_NODE_UPPER_THRESHOLD,
    reward_node: int | None = 94,
    xlim=(0, 3600),
    ylim=None,
    xticks=(0, 600, 1200, 1800, 2400, 3000, 3600),
    ylabel: str | None = "End nodes",
    xlabel: str | None = "Time in maze (s)",
    display_title: str | None = None,
    endnode_marker: dict | None = None,
):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout)
    panel_colors = get_panel_colors(colors)
    endnodes, times = extract_endnode_timeline(
        nickname,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    node_colors = get_end_node_colors(endnodes, colors=panel_colors)

    if reward_node is not None:
        ax.axhline(
            reward_node,
            color=panel_colors["reward_reference"],
            linestyle="-",
            linewidth=panel_layout["timeline_reference_linewidth"],
            alpha=panel_layout["timeline_reference_alpha"],
        )
    ax.scatter(
        times,
        endnodes,
        color=node_colors,
        s=panel_layout["timeline_scatter_marker_size"],
        alpha=panel_layout["timeline_scatter_alpha"],
        label="Node",
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    ax.set_yticks([])
    draw_endnode_timeline_marker(ax, endnode_marker)
    ax.set_xlabel("" if xlabel is None else xlabel)
    ax.set_ylabel("" if ylabel is None else ylabel)
    ax.set_title(format_panel_title(nickname, display_title=display_title))
    return ax


def render_endnode_timeline_grid_panel(
    session_rows,
    *,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    rcparams: dict | None = None,
    lower_threshold: int = END_NODE_LOWER_THRESHOLD,
    upper_threshold: int = END_NODE_UPPER_THRESHOLD,
    reward_node: int | None = 94,
    xlim=(0, 3600),
    ylim=None,
    xticks=(0, 600, 1200, 1800, 2400, 3000, 3600),
    endnode_marker: dict | None = None,
    sharex: bool = True,
    sharey: bool = True,
    save: bool = False,
    panel_name: str = "endnode_timeline_grid",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    panel_layout = get_panel_layout(layout)
    resolved_panel_size = tuple(panel_size or panel_layout["panel_size"])
    n_rows = len(session_rows)
    n_cols = max(len(row) for row in session_rows)
    subplot_size = (resolved_panel_size[0] / n_cols, resolved_panel_size[1] / n_rows)

    apply_panel_rcparams(rcparams, panel_size=subplot_size, layout=panel_layout)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=resolved_panel_size,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    fig.subplots_adjust(
        wspace=panel_layout.get("grid_wspace", 0.18),
        hspace=panel_layout.get("grid_hspace", 0.28),
    )

    summaries = []
    for row_idx, session_row in enumerate(session_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(session_row):
                ax.axis("off")
                continue

            session_spec = session_row[col_idx]
            nickname = session_spec["nickname"]
            endnodes, times = extract_endnode_timeline(
                nickname,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            summaries.append(
                {
                    "nickname": nickname,
                    "display_title": session_spec.get("display_title"),
                    "n_endnodes": int(len(endnodes)),
                    "max_time": float(np.nanmax(times)) if len(times) else np.nan,
                }
            )
            draw_endnode_timeline_axes(
                ax,
                nickname,
                panel_size=subplot_size,
                layout=panel_layout,
                colors=colors,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                reward_node=reward_node,
                xlim=xlim,
                ylim=ylim,
                xticks=xticks,
                ylabel="End nodes" if col_idx == 0 else None,
                xlabel="Time in maze (s)" if row_idx == n_rows - 1 else None,
                display_title=session_spec.get("display_title"),
                endnode_marker=session_spec.get("endnode_marker", endnode_marker),
            )
            ax.tick_params(labelbottom=row_idx == n_rows - 1)

    if n_rows:
        fig.align_ylabels(axes[:, 0])
        fig.align_xlabels(axes[-1, :])

    panel_data = {
        "session_rows": deepcopy(session_rows),
        "summary": pd.DataFrame(summaries),
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, axes, panel_data, saved_paths


def _prepare_level_code_heatmap_data(
    *,
    plot_ready_csv=None,
    variant: str = "d1v1",
    pair_order: list[tuple[str, str]] | None = None,
    style: dict | None = None,
):
    style = {} if style is None else dict(style)
    if pair_order is None:
        raise ValueError("A saved display pair order is required")
    requested_pair_keys = [f"{src}→{dst}" for (src, dst) in pair_order]
    if plot_ready_csv is None:
        raise ValueError("A saved level-map Source Data CSV is required")
    plot_ready_path = Path(plot_ready_csv)
    expected_sha256 = style.get("plot_ready_sha256")
    if expected_sha256 is not None:
        digest = hashlib.sha256(plot_ready_path.read_bytes()).hexdigest()
        if digest != str(expected_sha256):
            raise ValueError(
                f"Level-code plot-ready CSV hash drift: {digest} != {expected_sha256}"
            )
    rows = pd.read_csv(plot_ready_path)
    if "variant" in rows.columns:
        rows = rows.loc[rows["variant"].astype(str).eq(str(variant))].copy()
    level_indices = sorted(
        (
            int(column.removeprefix("train_level_"))
            for column in rows.columns
            if column.startswith("train_level_")
        )
    )
    required_columns = {
        "pair_key",
        "level_corr",
        *(f"train_level_{index}" for index in level_indices),
        *(f"test_level_{index}" for index in level_indices),
    }
    missing_columns = sorted(required_columns.difference(rows.columns))
    if missing_columns or not level_indices:
        raise ValueError(
            f"Invalid level-code plot-ready CSV; missing columns={missing_columns!r}"
        )
    train_maps = {}
    test_maps = {}
    similarities = {}
    if {"src_session", "dst_session"}.issubset(rows.columns):
        pair_key_values = pd.Series(
            [
                f"{src}→{dst}"
                for (src, dst) in zip(
                    rows["src_session"].astype(str),
                    rows["dst_session"].astype(str),
                    strict=False,
                )
            ],
            index=rows.index,
        )
    else:
        pair_key_values = (
            rows["pair_key"].astype(str).str.replace("->", "→", regex=False)
        )
    for key in requested_pair_keys:
        pair_rows = rows.loc[pair_key_values.eq(key)]
        if pair_rows.empty:
            continue
        train_maps[key] = (
            pair_rows[[f"train_level_{index}" for index in level_indices]]
            .to_numpy(dtype=float)
            .T
        )
        test_maps[key] = (
            pair_rows[[f"test_level_{index}" for index in level_indices]]
            .to_numpy(dtype=float)
            .T
        )
        similarities[key] = pd.to_numeric(
            pair_rows["level_corr"], errors="coerce"
        ).to_numpy(dtype=float)
    registry = {
        "source_kind": "plot_ready_csv",
        "path": str(plot_ready_path),
        "sha256": expected_sha256,
        "row_count": int(len(rows)),
    }
    if not train_maps or not test_maps:
        raise ValueError(
            f"No level-code train/test maps were found for variant={variant!r} in {plot_ready_csv!r}."
        )
    if train_maps.keys() != test_maps.keys():
        raise ValueError("Level-code train_map and test_map keys did not match.")
    missing_train_keys = [key for key in requested_pair_keys if key not in train_maps]
    missing_test_keys = [key for key in requested_pair_keys if key not in test_maps]
    missing_similarity_keys = [
        key for key in requested_pair_keys if key not in similarities
    ]
    if missing_train_keys or missing_test_keys or missing_similarity_keys:
        raise ValueError(
            f"Level-code heatmap inputs did not contain the full requested pair set. missing_train={missing_train_keys!r}, missing_test={missing_test_keys!r}, missing_similarity={missing_similarity_keys!r}"
        )
    train_maps = {key: train_maps[key] for key in requested_pair_keys}
    test_maps = {key: test_maps[key] for key in requested_pair_keys}
    similarities = {key: similarities[key] for key in requested_pair_keys}
    n_levels = int(np.asarray(next(iter(train_maps.values()))).shape[0])
    (train_maps_stacked, pair_slices) = stack_level_maps(
        train_maps, return_slices=True, expected_n_levels=n_levels
    )
    (test_maps_stacked, _) = stack_level_maps(
        test_maps, return_slices=True, expected_n_levels=n_levels
    )
    similarities_original = (
        np.concatenate(list(similarities.values())) if similarities else None
    )
    pair_slices_original = dict(pair_slices)
    similarities_for_order = similarities_original
    nonfinite_similarity_count = 0
    retained_similarity_mask = None
    nonfinite_similarity_counts_by_pair = {key: 0 for key in pair_slices_original}
    retained_neuron_counts_by_pair = {
        key: int(end - start) for (key, (start, end)) in pair_slices_original.items()
    }
    if similarities_original is not None:
        finite_similarity_mask = np.isfinite(similarities_original)
        retained_similarity_mask = finite_similarity_mask.astype(bool, copy=True)
        nonfinite_similarity_count = int(
            np.size(similarities_original) - np.sum(retained_similarity_mask)
        )
        for (key, (start, end)) in pair_slices_original.items():
            pair_mask = retained_similarity_mask[start:end]
            retained_neuron_counts_by_pair[key] = int(np.sum(pair_mask))
            nonfinite_similarity_counts_by_pair[key] = int(
                pair_mask.size - np.sum(pair_mask)
            )
        if nonfinite_similarity_count:
            if not np.any(retained_similarity_mask):
                raise ValueError(
                    "All level-code similarity entries were non-finite; cannot build a filtered neuron ordering."
                )
            train_maps_stacked = train_maps_stacked[:, retained_similarity_mask]
            test_maps_stacked = test_maps_stacked[:, retained_similarity_mask]
            similarities_for_order = similarities_original[retained_similarity_mask]
            filtered_pair_slices: dict[str, tuple[int, int]] = {}
            next_start = 0
            for (key, (start, end)) in pair_slices_original.items():
                n_keep = retained_neuron_counts_by_pair[key]
                filtered_pair_slices[key] = (next_start, next_start + n_keep)
                next_start += n_keep
            pair_slices = filtered_pair_slices
    normalize_mode = str(style.get("normalize_mode", "l2"))
    train_maps_norm = normalize_maps(train_maps_stacked, mode=normalize_mode)
    test_maps_norm = normalize_maps(test_maps_stacked, mode=normalize_mode)
    refined_order_path = style.get("refined_order_path")
    if refined_order_path is None:
        raise ValueError("The saved neuron display order is required")
    refined_order_path = Path(refined_order_path)
    if not refined_order_path.is_file():
        raise FileNotFoundError(
            f"Precomputed refined level-code order does not exist: {refined_order_path}"
        )
    expected_order_sha256 = style.get("refined_order_sha256")
    if expected_order_sha256 is not None:
        actual_order_sha256 = hashlib.sha256(
            refined_order_path.read_bytes()
        ).hexdigest()
        if actual_order_sha256 != str(expected_order_sha256):
            raise ValueError(
                f"Precomputed refined level-code order hash mismatch: expected {expected_order_sha256}, observed {actual_order_sha256}"
            )
    refined_order = np.asarray(
        np.load(refined_order_path, allow_pickle=False), dtype=int
    ).reshape(-1)
    expected_indices = np.arange(train_maps_norm.shape[1], dtype=int)
    if refined_order.shape != expected_indices.shape or not np.array_equal(
        np.sort(refined_order), expected_indices
    ):
        raise ValueError(
            f"Precomputed refined level-code order must be an exact permutation of 0..{train_maps_norm.shape[1] - 1}."
        )
    ordering_method = "precomputed_refined_order"
    order_hier = None
    train_maps_refined = train_maps_norm[:, refined_order]
    test_maps_refined = test_maps_norm[:, refined_order]
    return {
        "registry": registry,
        "pair_slices": pair_slices,
        "pair_slices_original": pair_slices_original,
        "train_maps_stacked": train_maps_stacked,
        "test_maps_stacked": test_maps_stacked,
        "train_maps_norm": train_maps_norm,
        "test_maps_norm": test_maps_norm,
        "train_maps_refined": train_maps_refined,
        "test_maps_refined": test_maps_refined,
        "similarities_original": similarities_original,
        "similarities_for_order": similarities_for_order,
        "nonfinite_similarity_count": nonfinite_similarity_count,
        "retained_similarity_mask": retained_similarity_mask,
        "nonfinite_similarity_counts_by_pair": nonfinite_similarity_counts_by_pair,
        "retained_neuron_counts_by_pair": retained_neuron_counts_by_pair,
        "order_hier": order_hier,
        "refined_order": refined_order,
        "ordering_method": ordering_method,
        "n_levels": int(train_maps_refined.shape[0]),
        "n_neurons": int(train_maps_refined.shape[1]),
    }


def render_level_code_heatmap_panel(
    *,
    plot_ready_csv=None,
    variant: str = "d1v1",
    pair_order: list[tuple[str, str]] | None = None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    xticks=None,
    save: bool = False,
    panel_name: str = "level_code_heatmap",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    panel_layout = get_panel_layout(layout)
    colors = {} if colors is None else dict(colors)
    style = {} if style is None else dict(style)
    labels = {} if labels is None else dict(labels)
    xticks = list(xticks) if xticks is not None else None
    panel_data = _prepare_level_code_heatmap_data(
        plot_ready_csv=plot_ready_csv,
        variant=variant,
        pair_order=pair_order,
        style=style,
    )
    (fig, axes) = plt.subplots(
        1,
        2,
        figsize=panel_size or panel_layout["panel_size"],
        sharey=bool(style.get("sharey", False)),
    )
    axes = np.atleast_1d(axes)
    if "grid_wspace" in panel_layout:
        fig.subplots_adjust(wspace=float(panel_layout["grid_wspace"]))
    heatmap_cmap = colors.get("heatmap", "hot")
    use_shared_scale = bool(style.get("shared_color_scale", False))
    heatmap_kwargs = {"aspect": style.get("aspect", "auto"), "cmap": heatmap_cmap}
    interpolation = style.get("interpolation")
    if interpolation is not None:
        heatmap_kwargs["interpolation"] = interpolation
    if use_shared_scale:
        finite_values = np.concatenate(
            [
                np.ravel(panel_data["train_maps_refined"]),
                np.ravel(panel_data["test_maps_refined"]),
            ]
        )
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            heatmap_kwargs["vmin"] = float(style.get("vmin", np.min(finite_values)))
            heatmap_kwargs["vmax"] = float(style.get("vmax", np.max(finite_values)))
    im_train = axes[0].imshow(panel_data["train_maps_refined"].T, **heatmap_kwargs)
    im_test = axes[1].imshow(panel_data["test_maps_refined"].T, **heatmap_kwargs)
    level_xticklabels = style.get("xticklabels")
    tick_labelsize = float(style.get("tick_labelsize", 16))
    label_fontsize = float(style.get("label_fontsize", 24))
    title_fontsize = float(style.get("title_fontsize", label_fontsize))
    for (idx, ax) in enumerate(axes):
        ax.set_xlabel(labels.get("xlabel", "Levels"), fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        if xticks is not None:
            ax.set_xticks(list(xticks))
        if level_xticklabels is not None:
            if xticks is None:
                ax.set_xticks(list(range(len(level_xticklabels))))
            ax.set_xticklabels(level_xticklabels)
        if idx == 1 and bool(style.get("hide_right_yticklabels", False)):
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(labels.get("ylabel", "Neuron"), fontsize=label_fontsize)
    left_title = labels.get("left_title", "")
    right_title = labels.get("right_title", "")
    if left_title:
        axes[0].set_title(left_title, fontsize=title_fontsize)
    if right_title:
        axes[1].set_title(right_title, fontsize=title_fontsize)
    show_colorbar = bool(style.get("show_colorbar", False))
    if show_colorbar:
        fig.colorbar(
            im_test if use_shared_scale else im_train,
            ax=list(axes),
            location=style.get("colorbar_location", "right"),
            fraction=float(style.get("colorbar_fraction", 0.046)),
            pad=float(style.get("colorbar_pad", 0.04)),
            label=labels.get("colorbar_label", ""),
        )
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return (fig, axes, panel_data, saved_paths)


def _significance_text(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n.s."
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def render_reward_acquisition_month_panel(
    *,
    points_csv,
    summary_csv=None,
    contrasts_csv=None,
    metric: str = "lb0",
    month_order=(0, 3, 6, 9, 12),
    month_labels=None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "reward_acquisition_lb0",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render the Figure 1D reward-acquisition month panel from saved CSVs."""
    panel_layout = {
        "panel_size": (4.2, 3.6),
    }
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        0: "#8BBDEB",
        3: "#F5A18E",
        6: "#F37B6D",
        9: "#DB4B4B",
        12: "#9B0E13",
        "edge": "black",
        "error": "black",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "bar_width": 0.6,
        "bar_alpha": 0.75,
        "bar_linewidth": 1.0,
        "errorbar_capsize": 4,
        "errorbar_linewidth": 1.1,
        "scatter_size": 30,
        "scatter_alpha": 0.85,
        "scatter_jitter": 0.06,
        "scatter_seed": 0,
        "annotation_source": "paired_wilcoxon",
        "annotation_p_col": None,
        "annotation_show_ns": False,
        "annotation_linewidth": 1.0,
        "annotation_fontsize": None,
        "annotation_fontweight": None,
        "annotation_y_start": None,
        "annotation_step": None,
        "annotation_offset_fraction": 0.06,
        "annotation_step_fraction": 0.08,
        "annotation_tick_fraction": 0.25,
        "annotation_text_offset_fraction": 0.30,
        "axis_label_fontsize": None,
        "axis_label_fontweight": None,
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": "Time (months)",
        "ylabel": "First long bout time (s)",
        "title": "",
    }
    if labels:
        panel_labels.update(labels)

    month_order = [int(m) for m in month_order]
    month_labels = month_labels or {
        0: "Naive",
        3: "3 mo",
        6: "6 mo",
        9: "9 mo",
        12: "12 mo",
    }
    month_to_x = {month: idx for idx, month in enumerate(month_order)}

    points_df = pd.read_csv(Path(points_csv))
    if metric not in points_df.columns:
        raise KeyError(f"Metric column {metric!r} not found in {points_csv!r}.")
    points_df = points_df.copy()
    points_df["month"] = pd.to_numeric(points_df["month"], errors="coerce").astype(
        "Int64"
    )
    points_df[metric] = pd.to_numeric(points_df[metric], errors="coerce")
    points_df = points_df[
        np.isfinite(points_df[metric]) & points_df["month"].isin(month_order)
    ].copy()

    if summary_csv is not None:
        summary_df = pd.read_csv(Path(summary_csv))
    else:
        rows = []
        for month in month_order:
            vals = points_df.loc[
                points_df["month"].astype(int) == month, metric
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rows.append(
                    {
                        "month": month,
                        "mean": float(np.nanmean(vals)),
                        "sem": float(stats.sem(vals, nan_policy="omit"))
                        if vals.size > 1
                        else 0.0,
                    }
                )
        summary_df = pd.DataFrame(rows)
    summary_df = summary_df.copy()
    summary_df["month"] = pd.to_numeric(summary_df["month"], errors="coerce").astype(
        "Int64"
    )

    contrasts_df = (
        pd.read_csv(Path(contrasts_csv))
        if contrasts_csv is not None
        else pd.DataFrame()
    )

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    for month in month_order:
        row = summary_df.loc[summary_df["month"].astype(int) == month]
        if row.empty:
            continue
        x = month_to_x[month]
        mean = float(row.iloc[0]["mean"])
        sem = float(row.iloc[0].get("sem", 0.0))
        ax.bar(
            x,
            mean,
            width=float(panel_style["bar_width"]),
            color=panel_colors.get(month, "#8BBDEB"),
            edgecolor=panel_colors.get("edge", "black"),
            linewidth=float(panel_style["bar_linewidth"]),
            alpha=float(panel_style["bar_alpha"]),
            zorder=1,
        )
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            color=panel_colors.get("error", "black"),
            capsize=float(panel_style["errorbar_capsize"]),
            linewidth=float(panel_style["errorbar_linewidth"]),
            zorder=2,
        )

    rng = np.random.default_rng(int(panel_style["scatter_seed"]))
    for month in month_order:
        vals = points_df.loc[points_df["month"].astype(int) == month, metric].to_numpy(
            dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x = month_to_x[month]
        jitter = rng.normal(0.0, float(panel_style["scatter_jitter"]), size=vals.size)
        ax.scatter(
            np.full(vals.size, x, dtype=float) + jitter,
            vals,
            s=float(panel_style["scatter_size"]),
            alpha=float(panel_style["scatter_alpha"]),
            color=panel_colors.get(month, "#8BBDEB"),
            edgecolors="none",
            zorder=3,
        )

    if not contrasts_df.empty:
        source = str(panel_style.get("annotation_source", "paired_wilcoxon")).lower()
        p_col = panel_style.get("annotation_p_col")
        if p_col is None:
            p_col = (
                "paired_wilcoxon_q_bh"
                if source.startswith("paired")
                else "mannwhitney_q_bh"
            )
        ymax = float(np.nanmax(points_df[metric].to_numpy(dtype=float)))
        annotation_y_start = panel_style.get("annotation_y_start")
        annotation_step = panel_style.get("annotation_step")
        y = (
            float(annotation_y_start)
            if annotation_y_start is not None
            else ymax * (1.0 + float(panel_style["annotation_offset_fraction"]))
        )
        step = (
            max(float(annotation_step), 0.1)
            if annotation_step is not None
            else max(ymax * float(panel_style["annotation_step_fraction"]), 0.1)
        )
        tick_height = step * float(panel_style["annotation_tick_fraction"])
        text_offset = step * float(panel_style["annotation_text_offset_fraction"])
        text_kwargs = {"ha": "center", "va": "bottom"}
        if panel_style.get("annotation_fontsize") is not None:
            text_kwargs["fontsize"] = float(panel_style["annotation_fontsize"])
        if panel_style.get("annotation_fontweight") is not None:
            text_kwargs["fontweight"] = panel_style["annotation_fontweight"]
        for _, row in contrasts_df.iterrows():
            if p_col not in row or not np.isfinite(float(row[p_col])):
                continue
            group_a = int(row.get("group_a", 0))
            group_b = int(row.get("group_b"))
            if group_a not in month_to_x or group_b not in month_to_x:
                continue
            text = _significance_text(float(row[p_col]))
            if text == "n.s." and not bool(panel_style["annotation_show_ns"]):
                continue
            x1 = month_to_x[group_a]
            x2 = month_to_x[group_b]
            ax.plot(
                [x1, x1, x2, x2],
                [y, y + tick_height, y + tick_height, y],
                color="black",
                linewidth=float(panel_style["annotation_linewidth"]),
                clip_on=False,
            )
            ax.text((x1 + x2) / 2.0, y + text_offset, text, **text_kwargs)
            y += step

    ax.set_xticks([month_to_x[m] for m in month_order])
    ax.set_xticklabels([month_labels.get(m, str(m)) for m in month_order])
    label_kwargs = {}
    if panel_style.get("axis_label_fontsize") is not None:
        label_kwargs["fontsize"] = float(panel_style["axis_label_fontsize"])
    if panel_style.get("axis_label_fontweight") is not None:
        label_kwargs["fontweight"] = panel_style["axis_label_fontweight"]
    ax.set_xlabel(str(panel_labels["xlabel"]), **label_kwargs)
    ax.set_ylabel(str(panel_labels["ylabel"]), **label_kwargs)
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
    ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
    fig.tight_layout()

    panel_data = {
        "points": points_df,
        "summary": summary_df,
        "contrasts": contrasts_df,
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, panel_data, saved_paths


def _draw_grouped_bar_scatter_axes(
    ax,
    *,
    points_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    contrasts_df: pd.DataFrame,
    value_col: str,
    group_order,
    panel_colors: dict,
    panel_style: dict,
    panel_labels: dict,
    pair_id_cols=None,
):
    group_order = [str(group) for group in group_order]
    group_to_x = {group: idx for idx, group in enumerate(group_order)}
    summary_df = summary_df.copy()
    if "group" in summary_df.columns:
        summary_df["group"] = summary_df["group"].astype(str)

    if pair_id_cols:
        valid_pair_id_cols = [col for col in pair_id_cols if col in points_df.columns]
        if valid_pair_id_cols and "group" in points_df.columns:
            complete_pairs = (
                points_df.pivot_table(
                    index=valid_pair_id_cols,
                    columns="group",
                    values=value_col,
                    aggfunc="mean",
                )
                .reindex(columns=group_order)
                .dropna()
            )
            if not complete_pairs.empty:
                pair_line_color = panel_colors.get("pair_line", "#8f8f8f")
                for _, row in complete_pairs.iterrows():
                    ax.plot(
                        [group_to_x[group] for group in group_order],
                        [float(row[group]) for group in group_order],
                        color=pair_line_color,
                        alpha=float(panel_style["pair_line_alpha"]),
                        linewidth=float(panel_style["pair_line_width"]),
                        zorder=0.8,
                    )

    for group in group_order:
        row = summary_df.loc[summary_df["group"].astype(str) == group]
        if row.empty:
            continue
        x = group_to_x[group]
        mean = float(row.iloc[0]["mean"])
        sem = float(row.iloc[0].get("sem", 0.0))
        color = panel_colors.get(group, panel_colors.get(x, "#8BBDEB"))
        ax.bar(
            x,
            mean,
            width=float(panel_style["bar_width"]),
            color=color,
            edgecolor=panel_colors.get("edge", "black"),
            linewidth=float(panel_style["bar_linewidth"]),
            alpha=float(panel_style["bar_alpha"]),
            zorder=1,
        )
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            color=panel_colors.get("error", "black"),
            capsize=float(panel_style["errorbar_capsize"]),
            linewidth=float(panel_style["errorbar_linewidth"]),
            zorder=2,
        )

    rng = np.random.default_rng(int(panel_style["scatter_seed"]))
    for group in group_order:
        vals = points_df.loc[
            points_df["group"].astype(str) == group, value_col
        ].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x = group_to_x[group]
        color = panel_colors.get(group, panel_colors.get(x, "#8BBDEB"))
        jitter = rng.normal(0.0, float(panel_style["scatter_jitter"]), size=vals.size)
        ax.scatter(
            np.full(vals.size, x, dtype=float) + jitter,
            vals,
            s=float(panel_style["scatter_size"]),
            alpha=float(panel_style["scatter_alpha"]),
            color=color,
            edgecolors="none",
            zorder=3,
        )

    if not contrasts_df.empty:
        p_col = str(panel_style["annotation_p_col"])
        metric_values = points_df[value_col].to_numpy(dtype=float)
        metric_values = metric_values[np.isfinite(metric_values)]
        if metric_values.size:
            ymax = float(np.nanmax(metric_values))
            ymin = float(np.nanmin(metric_values))
            span = ymax - ymin
            if span <= 0.0:
                span = max(abs(ymax), abs(ymin), 1.0)
            annotation_y_start = panel_style.get("annotation_y_start")
            annotation_step = panel_style.get("annotation_step")
            y = (
                float(annotation_y_start)
                if annotation_y_start is not None
                else ymax + span * float(panel_style["annotation_offset_fraction"])
            )
            step = (
                max(float(annotation_step), 0.05)
                if annotation_step is not None
                else max(span * float(panel_style["annotation_step_fraction"]), 0.05)
            )
            for _, row in contrasts_df.iterrows():
                if p_col not in row or not np.isfinite(float(row[p_col])):
                    continue
                group_a = int(row.get("group_a"))
                group_b = int(row.get("group_b"))
                if (
                    group_a < 0
                    or group_b < 0
                    or group_a >= len(group_order)
                    or group_b >= len(group_order)
                ):
                    continue
                text = _significance_text(float(row[p_col]))
                if text == "n.s." and not bool(panel_style["annotation_show_ns"]):
                    continue
                x1 = group_a
                x2 = group_b
                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + step * 0.25, y + step * 0.25, y],
                    color="black",
                    linewidth=float(panel_style["annotation_linewidth"]),
                    clip_on=False,
                )
                text_kwargs = {
                    "ha": "center",
                    "va": "bottom",
                }
                if "annotation_fontsize" in panel_style:
                    text_kwargs["fontsize"] = float(panel_style["annotation_fontsize"])
                if "annotation_fontweight" in panel_style:
                    text_kwargs["fontweight"] = panel_style["annotation_fontweight"]
                text_offset = step * float(
                    panel_style.get("annotation_text_offset_fraction", 0.30)
                )
                ax.text((x1 + x2) / 2.0, y + text_offset, text, **text_kwargs)
                y += step

    group_labels = panel_labels.get("group_labels", {})
    ax.set_xticks([group_to_x[group] for group in group_order])
    ax.set_xticklabels(
        [group_labels.get(group, group) for group in group_order],
        rotation=float(panel_style["xtick_rotation"]),
    )
    ax.set_xlabel(str(panel_labels["xlabel"]))
    ax.set_ylabel(str(panel_labels["ylabel"]))
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
    ax.spines["right"].set_visible(bool(panel_style["spine_right"]))


def render_grouped_bar_scatter_panel(
    *,
    points_csv,
    summary_csv=None,
    contrasts_csv=None,
    metric: str,
    group_order=None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "grouped_bar_scatter",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    panel_layout = {"panel_size": (5.0, 4.0)}
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        "edge": "black",
        "error": "black",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "bar_width": 0.6,
        "bar_alpha": 0.75,
        "bar_linewidth": 1.0,
        "errorbar_capsize": 4,
        "errorbar_linewidth": 1.1,
        "scatter_size": 30,
        "scatter_alpha": 0.85,
        "scatter_jitter": 0.06,
        "scatter_seed": 0,
        "annotation_p_col": "cluster_ols_q_bh",
        "annotation_show_ns": False,
        "annotation_linewidth": 1.0,
        "annotation_y_start": None,
        "annotation_step": None,
        "annotation_offset_fraction": 0.06,
        "annotation_step_fraction": 0.08,
        "annotation_text_offset_fraction": 0.30,
        "pair_line_alpha": 0.0,
        "pair_line_width": 0.8,
        "spine_top": False,
        "spine_right": False,
        "xtick_rotation": 0,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": "",
        "ylabel": metric,
        "title": "",
        "group_labels": {},
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(Path(points_csv)).copy()
    if metric not in points_df.columns:
        raise KeyError(f"Metric column {metric!r} not found in {points_csv!r}.")
    points_df[metric] = pd.to_numeric(points_df[metric], errors="coerce")
    points_df = points_df[np.isfinite(points_df[metric])].copy()
    if group_order is None:
        if "group_index" in points_df.columns:
            group_order = (
                points_df.sort_values("group_index")["group"]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )
        else:
            group_order = points_df["group"].drop_duplicates().astype(str).tolist()
    group_order = [str(group) for group in group_order]
    points_df = points_df.loc[points_df["group"].astype(str).isin(group_order)].copy()

    if summary_csv is not None:
        summary_df = pd.read_csv(Path(summary_csv)).copy()
    else:
        rows = []
        for group in group_order:
            vals = points_df.loc[
                points_df["group"].astype(str) == group, metric
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rows.append(
                    {
                        "group": group,
                        "mean": float(np.nanmean(vals)),
                        "sem": float(stats.sem(vals, nan_policy="omit"))
                        if vals.size > 1
                        else 0.0,
                    }
                )
        summary_df = pd.DataFrame(rows)
    contrasts_df = (
        pd.read_csv(Path(contrasts_csv))
        if contrasts_csv is not None
        else pd.DataFrame()
    )

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    _draw_grouped_bar_scatter_axes(
        ax,
        points_df=points_df,
        summary_df=summary_df,
        contrasts_df=contrasts_df,
        value_col=metric,
        group_order=group_order,
        panel_colors=panel_colors,
        panel_style=panel_style,
        panel_labels=panel_labels,
        pair_id_cols=None,
    )
    fig.tight_layout()

    panel_data = {
        "points": points_df,
        "summary": summary_df,
        "contrasts": contrasts_df,
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, panel_data, saved_paths


def render_exploration_efficiency_histogram_panel(
    *,
    points_csv,
    group_order=None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "exploration_efficiency_histogram",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a first-chunk exploration-efficiency histogram from saved CSV rows."""
    panel_layout = {
        "panel_size": (5.4, 3.6),
    }
    if layout:
        panel_layout.update(layout)

    default_color_cycle = [
        "#8BBDEB",
        "#F5A18E",
        "#F37B6D",
        "#DB4B4B",
        "#9B0E13",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    panel_colors = dict(colors or {})

    panel_style = {
        "n_bins": 20,
        "alpha": 0.6,
        "histtype": "bar",
        "edgecolor": "none",
        "linewidth": 1.0,
        "show_legend": False,
        "show_counts_in_legend": True,
        "legend_loc": "best",
        "spine_top": False,
        "spine_right": False,
        "density": False,
        "yticks": None,
        "xticks": None,
        "xlim": None,
        "ylim": None,
        "bins": None,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": "Exploration efficiency",
        "ylabel": "# of Animals",
        "title": "",
        "group_labels": {},
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(Path(points_csv))
    required_cols = {"group", "nickname", "first_chunk_efficiency"}
    missing = required_cols.difference(points_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {points_csv!r}: {sorted(missing)}")

    points_df = points_df.copy()
    points_df["first_chunk_efficiency"] = pd.to_numeric(
        points_df["first_chunk_efficiency"], errors="coerce"
    )
    points_df = points_df[np.isfinite(points_df["first_chunk_efficiency"])].copy()
    if points_df.empty:
        raise ValueError(
            f"No finite first_chunk_efficiency rows found in {points_csv!r}."
        )

    if group_order is None:
        if "group_index" in points_df.columns:
            points_df["group_index"] = pd.to_numeric(
                points_df["group_index"], errors="coerce"
            )
            ordered = (
                points_df[["group", "group_index"]]
                .drop_duplicates()
                .sort_values(["group_index", "group"], kind="stable")
            )
            group_order = ordered["group"].astype(str).tolist()
        else:
            group_order = points_df["group"].astype(str).drop_duplicates().tolist()
    else:
        group_order = [str(group) for group in group_order]

    selected_df = points_df.loc[points_df["group"].astype(str).isin(group_order)].copy()
    if selected_df.empty:
        raise ValueError(
            "No rows remain after applying group_order to exploration histogram data."
        )

    values = selected_df["first_chunk_efficiency"].to_numpy(dtype=float)
    bins = panel_style.get("bins")
    if bins is not None:
        bins = np.asarray(bins, dtype=float)
    else:
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if np.isclose(value_min, value_max):
            pad = max(abs(value_min) * 0.05, 0.01)
            value_min -= pad
            value_max += pad
        bins = np.linspace(
            value_min, value_max, int(panel_style["n_bins"]) + 1, dtype=float
        )

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    group_label_map = panel_labels.get("group_labels", {})

    histogram_rows = []
    for group_idx, group_name in enumerate(group_order):
        sub = selected_df.loc[
            selected_df["group"].astype(str) == str(group_name)
        ].copy()
        if sub.empty:
            continue
        display_name = str(group_label_map.get(group_name, group_name))
        legend_label = (
            f"{display_name} (n={len(sub)})"
            if bool(panel_style.get("show_counts_in_legend", True))
            else display_name
        )
        color = panel_colors.get(
            group_name, default_color_cycle[group_idx % len(default_color_cycle)]
        )
        ax.hist(
            sub["first_chunk_efficiency"].to_numpy(dtype=float),
            bins=bins,
            alpha=float(panel_style["alpha"]),
            label=legend_label,
            color=color,
            edgecolor=str(panel_style.get("edgecolor", "none")),
            linewidth=float(panel_style.get("linewidth", 1.0)),
            histtype=str(panel_style.get("histtype", "bar")),
            density=bool(panel_style.get("density", False)),
        )
        counts, edges = np.histogram(
            sub["first_chunk_efficiency"].to_numpy(dtype=float), bins=bins
        )
        total = int(np.sum(counts))
        for bin_index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "group": str(group_name),
                    "bin_index": int(bin_index),
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "count": int(count),
                    "fraction_within_group": float(count / total)
                    if total > 0
                    else np.nan,
                }
            )

    ax.set_xlabel(str(panel_labels["xlabel"]))
    ax.set_ylabel(str(panel_labels["ylabel"]))
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    if panel_style.get("xticks") is not None:
        ax.set_xticks(list(panel_style["xticks"]))
    if panel_style.get("yticks") is not None:
        ax.set_yticks(list(panel_style["yticks"]))
    if panel_style.get("xlim") is not None:
        ax.set_xlim(*tuple(panel_style["xlim"]))
    if panel_style.get("ylim") is not None:
        ax.set_ylim(*tuple(panel_style["ylim"]))
    ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
    ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
    if bool(panel_style.get("show_legend", False)):
        ax.legend(frameon=False, loc=str(panel_style.get("legend_loc", "best")))
    fig.tight_layout()

    panel_data = {
        "points": selected_df,
        "histogram_counts": pd.DataFrame(histogram_rows),
        "bins": np.asarray(bins, dtype=float),
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, panel_data, saved_paths


def render_spatial_decoding_error_map(
    *,
    map_csv,
    maze=None,
    maze_id: int = DEFAULT_MAZE_ID,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "spatial_decoding_error_map",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a saved per-bin spatial decoding-error map with square location overlays."""
    panel_layout = {
        **PANEL_LAYOUT,
        "panel_size": (5.4, 4.2),
        "n_bins_x": 15,
        "n_bins_y": 15,
        "colorbar_fraction": 0.045,
        "colorbar_pad": 0.025,
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "cmap": "viridis",
        "bad": "white",
        "wall": "black",
        "reward_square": "#DA2323",
        "control_square": "#8f8f8f",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "vmin": None,
        "vmax": None,
        "image_alpha": 0.95,
        "maze_wall_linewidth": panel_layout.get("maze_wall_linewidth", 3),
        "square_linewidth": 2.4,
        "square_size": 1.12,
        "reward_zorder": 7,
        "control_zorder": 6,
        "axis_off": True,
        "invert_yaxis": True,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "colorbar": "Decoding Error (bins)",
        "title": "",
    }
    if labels:
        panel_labels.update(labels)

    map_df = pd.read_csv(Path(map_csv)).copy()
    required = {"bin_index", "bx", "by", "decoding_error", "is_reward", "is_control"}
    missing = required.difference(map_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {map_csv!r}: {sorted(missing)}")
    for flag_col in ("is_reward", "is_control"):
        if map_df[flag_col].dtype == object:
            map_df[flag_col] = (
                map_df[flag_col].astype(str).str.lower().isin({"true", "1", "yes"})
            )
        else:
            map_df[flag_col] = map_df[flag_col].astype(bool)

    n_bins_x = int(panel_layout["n_bins_x"])
    n_bins_y = int(panel_layout["n_bins_y"])
    error_grid = np.full((n_bins_y, n_bins_x), np.nan, dtype=float)
    for _, row in map_df.iterrows():
        bx = int(row["bx"])
        by = int(row["by"])
        if 0 <= bx < n_bins_x and 0 <= by < n_bins_y:
            error_grid[by, bx] = float(row["decoding_error"])

    cmap = plt.get_cmap(panel_colors["cmap"]).copy()
    cmap.set_bad(panel_colors["bad"])
    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    image = ax.imshow(
        np.ma.masked_invalid(error_grid),
        origin="lower",
        extent=(-0.5, n_bins_x - 0.5, -0.5, n_bins_y - 0.5),
        interpolation="none",
        cmap=cmap,
        vmin=panel_style["vmin"],
        vmax=panel_style["vmax"],
        alpha=float(panel_style["image_alpha"]),
        zorder=1,
    )

    maze = get_panel_maze(maze_id) if maze is None else maze
    outline_layout = {
        **panel_layout,
        "maze_wall_linewidth": panel_style["maze_wall_linewidth"],
    }
    draw_maze_outline(
        ax,
        maze,
        panel_size=panel_size,
        layout=outline_layout,
        colors={"wall": panel_colors["wall"]},
    )

    square_size = float(panel_style["square_size"])
    square_offset = square_size / 2.0
    for _, row in map_df.loc[map_df["is_control"].astype(bool)].iterrows():
        ax.add_patch(
            patches.Rectangle(
                (float(row["bx"]) - square_offset, float(row["by"]) - square_offset),
                square_size,
                square_size,
                fill=False,
                edgecolor=panel_colors["control_square"],
                linewidth=float(panel_style["square_linewidth"]),
                zorder=int(panel_style["control_zorder"]),
            )
        )
    for _, row in map_df.loc[map_df["is_reward"].astype(bool)].iterrows():
        ax.add_patch(
            patches.Rectangle(
                (float(row["bx"]) - square_offset, float(row["by"]) - square_offset),
                square_size,
                square_size,
                fill=False,
                edgecolor=panel_colors["reward_square"],
                linewidth=float(panel_style["square_linewidth"]),
                zorder=int(panel_style["reward_zorder"]),
            )
        )

    ax.set_xlim(-0.5, n_bins_x - 0.5)
    if bool(panel_style["invert_yaxis"]):
        ax.set_ylim(n_bins_y - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, n_bins_y - 0.5)
    ax.set_aspect("equal")
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    if bool(panel_style["axis_off"]):
        ax.axis("off")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=float(panel_layout["colorbar_fraction"]),
        pad=float(panel_layout["colorbar_pad"]),
        location=str(panel_layout.get("colorbar_location", "right")),
    )
    colorbar.set_label(str(panel_labels["colorbar"]))
    fig.tight_layout()

    panel_data = {
        "map": map_df,
        "error_grid": error_grid,
        "colorbar": colorbar,
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, panel_data, saved_paths


def _add_distribution_text_label(
    ax, text, xy, *, color, style, default_ha="center", default_va="center"
):
    if xy is None:
        return
    transform = (
        ax.transAxes
        if str(style.get("direct_label_transform", "data")) == "axes"
        else ax.transData
    )
    text_kwargs = {
        "color": color,
        "alpha": float(style.get("direct_label_alpha", 0.95)),
        "fontsize": style.get("direct_label_fontsize", None),
        "fontweight": style.get("direct_label_fontweight", "bold"),
        "ha": style.get("direct_label_ha", default_ha),
        "va": style.get("direct_label_va", default_va),
        "transform": transform,
        "zorder": int(style.get("direct_label_zorder", 5)),
    }
    if style.get("direct_label_bbox") is not None:
        text_kwargs["bbox"] = style["direct_label_bbox"]
    ax.text(float(xy[0]), float(xy[1]), str(text), **text_kwargs)


def render_correlation_shuffle_distribution_panel(
    *,
    npz_path,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "correlation_shuffle_distribution",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a saved observed-correlation distribution over a saved shuffle histogram."""
    panel_layout = {"panel_size": (4.2, 3.6)}
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "data": "#e15759",
        "shuffle": "#bdbdbd",
        "shuffle_edge": "none",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "data_alpha": 0.78,
        "shuffle_alpha": 0.85,
        "data_edgecolor": "none",
        "linewidth": 1.0,
        "legend": True,
        "legend_loc": "upper left",
        "show_counts": True,
        "shuffle_label_suffix": "saved",
        "direct_labels": False,
        "data_label_xy": None,
        "shuffle_label_xy": None,
        "direct_label_alpha": 0.95,
        "direct_label_fontsize": None,
        "direct_label_fontweight": "bold",
        "direct_label_transform": "data",
        "direct_label_zorder": 5,
        "xlim": None,
        "ylim": None,
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "xlabel": "Level map correlation",
        "ylabel": "Density",
        "title": "",
        "data": "Data",
        "shuffle": "Shuffle",
    }
    if labels:
        panel_labels.update(labels)

    data = np.load(Path(npz_path), allow_pickle=False)
    obs = np.asarray(data["obs"], dtype=float)
    obs = obs[np.isfinite(obs)]
    hist_edges = np.asarray(data["hist_edges"], dtype=float)
    if "shuffle_hist" in data.files:
        shuffle_hist = np.asarray(data["shuffle_hist"], dtype=float)
    elif "shuffle_values" in data.files:
        shuffle_values = np.asarray(data["shuffle_values"], dtype=float)
        shuffle_values = shuffle_values[np.isfinite(shuffle_values)]
        shuffle_hist, _ = np.histogram(shuffle_values, bins=hist_edges)
        shuffle_hist = shuffle_hist.astype(float)
    else:
        raise KeyError(
            f"{npz_path!s} must contain either 'shuffle_hist' or 'shuffle_values'."
        )
    bin_widths = np.diff(hist_edges)
    total_shuffle = float(np.sum(shuffle_hist))
    shuffle_density = np.divide(
        shuffle_hist,
        total_shuffle * bin_widths,
        out=np.zeros_like(shuffle_hist, dtype=float),
        where=(total_shuffle > 0) & (bin_widths > 0),
    )
    centers = hist_edges[:-1] + bin_widths / 2.0
    data_label = str(panel_labels["data"])
    shuffle_label = str(panel_labels["shuffle"])
    if bool(panel_style["show_counts"]):
        data_label = f"{data_label} (n={len(obs)})"
        shuffle_suffix = str(panel_style.get("shuffle_label_suffix", "")).strip()
        if shuffle_suffix:
            shuffle_label = f"{shuffle_label} ({shuffle_suffix})"

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    ax.bar(
        centers,
        shuffle_density,
        width=bin_widths,
        align="center",
        color=panel_colors["shuffle"],
        edgecolor=panel_colors["shuffle_edge"],
        linewidth=float(panel_style["linewidth"]),
        alpha=float(panel_style["shuffle_alpha"]),
        label=shuffle_label,
        zorder=1,
    )
    ax.hist(
        obs,
        bins=hist_edges,
        density=True,
        color=panel_colors["data"],
        edgecolor=panel_style["data_edgecolor"],
        linewidth=float(panel_style["linewidth"]),
        alpha=float(panel_style["data_alpha"]),
        label=data_label,
        zorder=2,
    )
    if panel_style["xlim"] is not None:
        ax.set_xlim(*panel_style["xlim"])
    if panel_style["ylim"] is not None:
        ax.set_ylim(*panel_style["ylim"])
    ax.set_xlabel(str(panel_labels["xlabel"]))
    ax.set_ylabel(str(panel_labels["ylabel"]))
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    if bool(panel_style["direct_labels"]):
        _add_distribution_text_label(
            ax,
            panel_labels["shuffle"],
            panel_style.get("shuffle_label_xy"),
            color=panel_colors["shuffle"],
            style=panel_style,
        )
        _add_distribution_text_label(
            ax,
            panel_labels["data"],
            panel_style.get("data_label_xy"),
            color=panel_colors["data"],
            style=panel_style,
        )
    if bool(panel_style["legend"]):
        ax.legend(loc=str(panel_style["legend_loc"]), frameon=True)
    ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
    ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
    fig.tight_layout()

    panel_data = {
        "obs": obs,
        "shuffle_hist": shuffle_hist,
        "hist_edges": hist_edges,
        "shuffle_density": shuffle_density,
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, panel_data, saved_paths


def _load_local_correlation_image(fig_path) -> dict:
    """Load the included local-correlation image and display-coordinate arrays."""
    with np.load(fig_path, allow_pickle=False) as source:
        required = {"image", "xdata", "ydata", "clim", "colormap"}
        if not required.issubset(source.files):
            raise KeyError(
                f"Local-correlation source is missing {sorted(required - set(source.files))}"
            )
        data = {key: np.asarray(source[key]).copy() for key in required}
    data.update(
        xticks=[],
        yticks=[],
        xticklabels=[],
        yticklabels=[],
        colorbar_ticks=[],
        colorbar_ticklabels=[],
        title_text=None,
    )
    return data


def render_stacked_local_correlation_panel(
    *,
    top_fig_path,
    bottom_fig_path,
    panel_size=None,
    layout: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "stacked_local_correlation",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    panel_layout = {"panel_size": (6.2, 9.2), "grid_hspace": 0.32}
    if layout:
        panel_layout.update(layout)
    panel_style = {
        "origin": "upper",
        "interpolation": "nearest",
        "aspect": "equal",
        "colorbar_fraction": 0.046,
        "colorbar_pad": 0.018,
        "suptitle_y": 0.995,
        "shared_clim": False,
        "shared_vmin": None,
        "shared_vmax": None,
        "show_axes": True,
        "show_colorbars": True,
        "show_panel_title": True,
        "show_subplot_titles": True,
        "side_label_transform": "axes",
        "side_label_color": "black",
        "side_label_fontsize": None,
        "side_label_fontweight": "bold",
        "side_label_rotation": 0,
        "side_label_ha": "right",
        "side_label_va": "center",
        "side_label_clip_on": False,
        "top_side_label_xy": (-0.08, 0.5),
        "bottom_side_label_xy": (-0.08, 0.5),
        "show_scale_bar": False,
        "scale_bar_panel": "bottom",
        "scale_bar_xy": (280.0, 270.0),
        "scale_bar_length_px": 60.0,
        "scale_bar_linewidth": 3.0,
        "scale_bar_color": "white",
        "scale_bar_text_color": None,
        "scale_bar_fontsize": None,
        "scale_bar_fontweight": "bold",
        "scale_bar_text_offset_px": -8.0,
        "scale_bar_ha": "center",
        "scale_bar_va": "bottom",
        "scale_bar_clip_on": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "panel_title": "",
        "top_title": "Top",
        "bottom_title": "Bottom",
        "top_side_label": "",
        "bottom_side_label": "",
        "xlabel": "x position (px)",
        "ylabel": "y position (px)",
        "scale_bar_label": "",
    }
    if labels:
        panel_labels.update(labels)
    top_panel = _load_local_correlation_image(top_fig_path)
    bottom_panel = _load_local_correlation_image(bottom_fig_path)
    panel_data = [top_panel, bottom_panel]
    shared_vmin = None
    shared_vmax = None
    if bool(panel_style["shared_clim"]):
        shared_vmin = panel_style["shared_vmin"]
        shared_vmax = panel_style["shared_vmax"]
        if shared_vmin is None or shared_vmax is None:
            combined_values = np.concatenate(
                [
                    np.asarray(fig_data["image"], dtype=float)[
                        np.isfinite(np.asarray(fig_data["image"], dtype=float))
                    ]
                    for fig_data in panel_data
                ]
            )
            if combined_values.size == 0:
                raise ValueError(
                    "Local correlation figures do not contain finite values for shared color limits."
                )
            if shared_vmin is None:
                shared_vmin = float(np.min(combined_values))
            if shared_vmax is None:
                shared_vmax = float(np.max(combined_values))
    (fig, axes) = plt.subplots(
        2, 1, figsize=panel_size or panel_layout["panel_size"], squeeze=False
    )
    fig.subplots_adjust(hspace=float(panel_layout["grid_hspace"]))
    for (ax, fig_data, subplot_key, subplot_title) in zip(
        axes[:, 0],
        panel_data,
        ["top", "bottom"],
        [panel_labels["top_title"], panel_labels["bottom_title"]],
    ):
        image = ax.imshow(
            fig_data["image"],
            cmap=ListedColormap(fig_data["colormap"]),
            vmin=float(shared_vmin if shared_vmin is not None else fig_data["clim"][0]),
            vmax=float(shared_vmax if shared_vmax is not None else fig_data["clim"][1]),
            origin=str(panel_style["origin"]),
            extent=[
                float(fig_data["xdata"][0]),
                float(fig_data["xdata"][1]),
                float(fig_data["ydata"][1]),
                float(fig_data["ydata"][0]),
            ],
            interpolation=str(panel_style["interpolation"]),
            aspect=str(panel_style["aspect"]),
        )
        if bool(panel_style["show_subplot_titles"]) and subplot_title:
            ax.set_title(str(subplot_title))
        if bool(panel_style["show_axes"]):
            ax.set_xlabel(str(panel_labels["xlabel"]))
            ax.set_ylabel(str(panel_labels["ylabel"]))
            ax.set_xticks(fig_data["xticks"])
            ax.set_yticks(fig_data["yticks"])
            ax.set_xticklabels(fig_data["xticklabels"])
            ax.set_yticklabels(fig_data["yticklabels"])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()
        if bool(panel_style["show_colorbars"]):
            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=float(panel_style["colorbar_fraction"]),
                pad=float(panel_style["colorbar_pad"]),
            )
            colorbar.set_ticks(fig_data["colorbar_ticks"])
            colorbar.set_ticklabels(fig_data["colorbar_ticklabels"])
        side_label_text = str(panel_labels.get(f"{subplot_key}_side_label", "")).strip()
        side_label_xy = panel_style.get(f"{subplot_key}_side_label_xy")
        if side_label_text and side_label_xy is not None:
            side_label_transform = (
                ax.transAxes
                if str(panel_style.get("side_label_transform", "axes")) == "axes"
                else ax.transData
            )
            ax.text(
                float(side_label_xy[0]),
                float(side_label_xy[1]),
                side_label_text,
                transform=side_label_transform,
                color=panel_style["side_label_color"],
                fontsize=panel_style["side_label_fontsize"],
                fontweight=panel_style["side_label_fontweight"],
                rotation=float(panel_style["side_label_rotation"]),
                ha=str(panel_style["side_label_ha"]),
                va=str(panel_style["side_label_va"]),
                clip_on=bool(panel_style["side_label_clip_on"]),
            )
    if bool(panel_style["show_scale_bar"]):
        scale_bar_ax = (
            axes[1, 0]
            if str(panel_style["scale_bar_panel"]) == "bottom"
            else axes[0, 0]
        )
        scale_bar_xy = panel_style["scale_bar_xy"]
        scale_bar_length = float(panel_style["scale_bar_length_px"])
        scale_bar_color = str(panel_style["scale_bar_color"])
        scale_bar_text_color = (
            str(panel_style["scale_bar_text_color"])
            if panel_style["scale_bar_text_color"] is not None
            else scale_bar_color
        )
        scale_bar_ax.plot(
            [float(scale_bar_xy[0]), float(scale_bar_xy[0]) + scale_bar_length],
            [float(scale_bar_xy[1]), float(scale_bar_xy[1])],
            color=scale_bar_color,
            linewidth=float(panel_style["scale_bar_linewidth"]),
            solid_capstyle="butt",
            clip_on=bool(panel_style["scale_bar_clip_on"]),
        )
        if str(panel_labels.get("scale_bar_label", "")).strip():
            scale_bar_ax.text(
                float(scale_bar_xy[0]) + scale_bar_length / 2.0,
                float(scale_bar_xy[1]) + float(panel_style["scale_bar_text_offset_px"]),
                str(panel_labels["scale_bar_label"]),
                color=scale_bar_text_color,
                fontsize=panel_style["scale_bar_fontsize"],
                fontweight=panel_style["scale_bar_fontweight"],
                ha=str(panel_style["scale_bar_ha"]),
                va=str(panel_style["scale_bar_va"]),
                clip_on=bool(panel_style["scale_bar_clip_on"]),
            )
    if bool(panel_style["show_panel_title"]) and panel_labels.get("panel_title"):
        fig.suptitle(
            str(panel_labels["panel_title"]), y=float(panel_style["suptitle_y"])
        )
    if bool(panel_style["show_axes"]):
        fig.align_ylabels(axes[:, 0])
    panel_data = {
        "top": top_panel,
        "bottom": bottom_panel,
        "layout": panel_layout,
        "style": panel_style,
        "labels": panel_labels,
        "paths": {"top": str(top_fig_path), "bottom": str(bottom_fig_path)},
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return (fig, axes[:, 0], panel_data, saved_paths)


def _grid_units_to_data(origin, cell_size, point_units):
    return np.asarray(origin, dtype=float) + float(cell_size) * np.asarray(
        point_units, dtype=float
    )


def _draw_grid_route_cells(ax, *, origin, cell_size, cells, colors, style):
    for col, row in cells:
        xy = _grid_units_to_data(origin, cell_size, (col, row))
        ax.add_patch(
            patches.Rectangle(
                xy,
                cell_size,
                cell_size,
                linewidth=style["cell_linewidth"],
                edgecolor=colors["cell_edge"],
                facecolor=colors["cell_face"],
                joinstyle="miter",
                zorder=1,
            )
        )


def _normalize_route_points(point_spec):
    arr = np.asarray(point_spec, dtype=float)
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError("Flat route point specs must contain x0, y0, x1, y1.")
        return arr.reshape(2, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Route point specs must be a 2-column coordinate array.")
    return arr


def _draw_grid_route_arrow(ax, arrow_spec, *, origin, cell_size, colors, style):
    linewidth_key = arrow_spec.get("linewidth_key", "arrow_linewidth")
    color = colors[arrow_spec.get("color_key", "active_arrow")]

    if "curve_units" in arrow_spec:
        curve_spec = arrow_spec["curve_units"]
        start = _grid_units_to_data(origin, cell_size, curve_spec["start"])
        end = _grid_units_to_data(origin, cell_size, curve_spec["end"])
        return _add_curved_arrow(
            ax,
            start,
            end,
            linewidth=style[linewidth_key],
            color=color,
            mutation_scale=style["arrow_mutation_scale"],
            rad=curve_spec.get("rad", 0.0),
        )

    if "path_units" in arrow_spec:
        path_codes = arrow_spec.get("path_codes")
        if path_codes is None:
            raise ValueError("Curved path arrows require path_codes.")
        code_lookup = {
            "MOVETO": MplPath.MOVETO,
            "LINETO": MplPath.LINETO,
            "CURVE3": MplPath.CURVE3,
            "CURVE4": MplPath.CURVE4,
        }
        codes = [code_lookup[code] for code in path_codes]
        pts = np.vstack(
            [
                _grid_units_to_data(origin, cell_size, point)
                for point in arrow_spec["path_units"]
            ]
        )
        return _add_styled_path_arrow(
            ax,
            pts,
            codes,
            linewidth=style[linewidth_key],
            color=color,
            mutation_scale=style["arrow_mutation_scale"],
            linestyle=arrow_spec.get("linestyle", "-"),
        )

    points_units = _normalize_route_points(arrow_spec["points_units"])
    pts = np.vstack(
        [_grid_units_to_data(origin, cell_size, point) for point in points_units]
    )
    if arrow_spec.get("separate_head", False):
        return _add_separate_head_poly_arrow(
            ax,
            pts,
            linewidth=style[linewidth_key],
            color=color,
            mutation_scale=style["arrow_mutation_scale"],
            linestyle=arrow_spec.get("linestyle", "--"),
            head_length=style.get("previous_head_length", 0.18),
        )
    return _add_styled_poly_arrow(
        ax,
        pts,
        linewidth=style[linewidth_key],
        color=color,
        mutation_scale=style["arrow_mutation_scale"],
        arrowstyle=arrow_spec.get("arrowstyle", "->"),
        linestyle=arrow_spec.get("linestyle", "-"),
    )


def _draw_grid_route_choice_row(ax, row_spec, *, geometry, colors, style):
    row_geometry = geometry["rows"][row_spec["key"]]
    origin = row_geometry["origin"]
    cell_size = geometry["cell_size"]
    cells = row_geometry.get("cells", geometry["cells"])

    _draw_grid_route_cells(
        ax,
        origin=origin,
        cell_size=cell_size,
        cells=cells,
        colors=colors,
        style=style,
    )

    title_xy = row_spec["title_xy"]
    ax.text(
        title_xy[0],
        title_xy[1],
        row_spec["title"],
        fontsize=style["title_fontsize"],
        fontweight=style["title_fontweight"],
        color=colors["title"],
        ha=row_spec.get("title_ha", "left"),
        va=row_spec.get("title_va", "center"),
    )

    for annotation in row_spec.get("annotations", []):
        ax.text(
            annotation["xy"][0],
            annotation["xy"][1],
            annotation["text"],
            fontsize=style["annotation_fontsize"],
            color=colors["annotation"],
            ha=annotation.get("ha", "center"),
            va=annotation.get("va", "center"),
            linespacing=style.get("annotation_lineheight", 1.1),
        )

    if "or_xy_units" in row_spec:
        or_xy = _grid_units_to_data(origin, cell_size, row_spec["or_xy_units"])
        ax.text(
            or_xy[0],
            or_xy[1],
            row_spec.get("or_text", "or"),
            fontsize=style["or_fontsize"],
            color=colors["or"],
            ha=row_spec.get("or_ha", "center"),
            va=row_spec.get("or_va", "center"),
        )

    for arrow_spec in row_spec.get("arrows", []):
        _draw_grid_route_arrow(
            ax,
            arrow_spec,
            origin=origin,
            cell_size=cell_size,
            colors=colors,
            style=style,
        )


def render_grid_route_choice_schematic(
    *,
    layout: dict | None = None,
    geometry: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    text: dict | None = None,
    save: bool = False,
    export_name: str = "grid_route_choice_schematic",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    resolved_layout = get_grid_route_choice_schematic_layout(layout)
    resolved_geometry = get_grid_route_choice_schematic_geometry(geometry)
    resolved_colors = get_grid_route_choice_schematic_colors(colors)
    resolved_style = get_grid_route_choice_schematic_style(style)
    resolved_text = get_grid_route_choice_schematic_text(text)

    fig, ax = plt.subplots(figsize=resolved_layout["panel_size"])
    ax.set_xlim(*resolved_layout["xlim"])
    ax.set_ylim(*resolved_layout["ylim"])
    ax.set_aspect("equal")
    ax.axis("off")

    for row_spec in resolved_text.get("rows", []):
        _draw_grid_route_choice_row(
            ax,
            row_spec,
            geometry=resolved_geometry,
            colors=resolved_colors,
            style=resolved_style,
        )

    if resolved_layout.get("tight_layout", False):
        fig.tight_layout(pad=0)

    schematic_data = {
        "layout": resolved_layout,
        "geometry": resolved_geometry,
        "colors": resolved_colors,
        "style": resolved_style,
        "text": resolved_text,
    }
    saved_paths = (
        save_panel(fig, export_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, schematic_data, saved_paths


def save_panel(
    fig,
    panel_name: str,
    *,
    formats=DEFAULT_EXPORT_FORMATS,
    dpi: int | None = None,
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for ext in formats:
        path = output_dir / f"{panel_name}.{ext}"
        save_kwargs = {"bbox_inches": plt.rcParams["savefig.bbox"]}
        if ext.lower() not in VECTOR_EXPORT_FORMATS and dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    return saved_paths


def _compute_complete_binary_tree_positions(
    depth: int, tree_width: float, tree_height: float
):
    pos = {}
    n_leaves = 2 ** (depth - 1)
    xs_leaves = np.linspace(0.0, tree_width, n_leaves)
    y_step = tree_height / (depth - 1) if depth > 1 else 0.0
    for i in range(n_leaves):
        pos[(depth - 1, i)] = (xs_leaves[i], 0.0)
    for level in range(depth - 2, -1, -1):
        n_nodes = 2**level
        for i in range(n_nodes):
            left_child = (level + 1, 2 * i)
            right_child = (level + 1, 2 * i + 1)
            x = 0.5 * (pos[left_child][0] + pos[right_child][0])
            y = pos[left_child][1] + y_step
            pos[(level, i)] = (x, y)
    return pos


def _horizontal_binary_tree_positions(
    depth: int, tree_width: float, tree_height: float
):
    vertical = _compute_complete_binary_tree_positions(depth, tree_width, tree_height)
    y_center = 0.5 * tree_width
    return {
        key: (tree_height - xy[1], y_center - xy[0]) for key, xy in vertical.items()
    }


def _level_assignment_palette(level_count: int, colors: dict):
    cmap = plt.get_cmap(colors["cmap_name"])
    denom = max(int(level_count) - 1, 1)
    return [cmap(i / denom) for i in range(int(level_count))]


def _maze_level_grid(maze, *, grid_size: int = 15, level_definition: str | None = None):
    level_masks = get_level_list(
        level_definition, maze=maze
    )
    level_by_cell = np.full(len(maze.xc), -1, dtype=int)
    for level_idx, mask in enumerate(level_masks):
        level_by_cell[np.asarray(mask, dtype=bool)] = level_idx
    level_grid = np.full((grid_size, grid_size), -1, dtype=int)
    for (x, y), cell_id in maze.ce.items():
        level_grid[int(y), int(x)] = int(level_by_cell[int(cell_id)])
    return level_grid, level_by_cell


def _draw_level_assignment_tree(
    ax, *, geometry: dict, colors: dict, style: dict, text: dict
):
    depth = int(geometry["tree_depth"])
    level_count = int(geometry.get("level_count", depth + 1))
    positions = _horizontal_binary_tree_positions(
        depth,
        float(geometry["tree_width"]),
        float(geometry["tree_height"]),
    )
    palette = _level_assignment_palette(level_count, colors)
    for level in range(depth - 1):
        color = palette[min(level + 1, level_count - 1)]
        for i in range(2**level):
            parent = (level, i)
            for child in ((level + 1, 2 * i), (level + 1, 2 * i + 1)):
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=color,
                    linewidth=style["tree_linewidth"],
                    solid_capstyle="round",
                )

    if style.get("show_tree_nodes", False):
        for level in range(depth):
            node_color = palette[min(level + 1, level_count - 1)]
            xs = []
            ys = []
            for i in range(2**level):
                x, y = positions[(level, i)]
                xs.append(x)
                ys.append(y)
            ax.scatter(xs, ys, s=style["node_size"], color=node_color, zorder=3)

    leaf_x = []
    leaf_y = []
    for i in range(2 ** (depth - 1)):
        x, y = positions[(depth - 1, i)]
        leaf_x.append(x)
        leaf_y.append(y)
    if style.get("show_leaf_markers", True):
        ax.scatter(
            leaf_x,
            leaf_y,
            s=style["leaf_marker_size"],
            color=palette[-1],
            zorder=4,
        )

    root_x, root_y = positions[(0, 0)]
    ax.plot(
        [root_x - float(geometry["root_line_length"]), root_x],
        [root_y, root_y],
        color=palette[0],
        linewidth=style["root_linewidth"],
        solid_capstyle="round",
    )

    root_t = float(text.get("root_label_t", 0.55))
    root_mid_x = root_x - (1.0 - root_t) * float(geometry["root_line_length"])
    root_mid_y = root_y - float(text.get("root_label_offset", 0.34))
    level_label = text["level_label"]
    if level_label.get("position", "left_of_root") == "below_zero":
        level_label_x = root_mid_x + float(level_label.get("dx", 0.0))
        level_label_y = (
            root_mid_y
            - float(level_label.get("gap_below_zero", 0.18))
            + float(level_label.get("dy", 0.0))
        )
    else:
        level_label_x = (
            root_x
            - float(geometry["root_line_length"])
            + float(level_label.get("dx", 0.0))
        )
        level_label_y = root_y + float(level_label.get("dy", 0.0))
    ax.text(
        level_label_x,
        level_label_y,
        level_label["text"],
        fontsize=style["label_fontsize"],
        color=colors["annotation"],
        ha=level_label.get("ha", "right"),
        va=level_label.get("va", "center"),
    )

    ax.text(
        root_mid_x,
        root_mid_y,
        "0",
        fontsize=style["label_fontsize"],
        color=colors["annotation"],
        ha="center",
        va="center",
    )

    for level_idx in range(1, depth):
        parent = (level_idx - 1, 2 ** (level_idx - 1) - 1)
        child = (level_idx, 2**level_idx - 1)
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        edge_t = float(
            text.get("edge_label_t_by_level", {}).get(
                level_idx, text.get("edge_label_t", 0.72)
            )
        )
        label_x = (1.0 - edge_t) * x0 + edge_t * x1
        label_y = (1.0 - edge_t) * y0 + edge_t * y1
        edge_vec = np.array([x1 - x0, y1 - y0], dtype=float)
        norm = np.linalg.norm(edge_vec)
        if norm > 0:
            normals = (
                np.array([-edge_vec[1], edge_vec[0]], dtype=float) / norm,
                np.array([edge_vec[1], -edge_vec[0]], dtype=float) / norm,
            )
            down_normal = normals[0] if normals[0][1] < normals[1][1] else normals[1]
            label_x += down_normal[0] * float(text.get("edge_label_offset", 0.34))
            label_y += down_normal[1] * float(text.get("edge_label_offset", 0.34))
        ax.text(
            label_x,
            label_y,
            f"{level_idx}",
            fontsize=style["label_fontsize"],
            color=colors["annotation"],
            ha="center",
            va="center",
        )

    leaf_x, leaf_y = positions[(depth - 1, 2 ** (depth - 1) - 1)]
    show_leaf_label = bool(text.get("show_leaf_label", level_count > depth))
    if show_leaf_label:
        ax.text(
            leaf_x + float(text.get("leaf_label_dx", 0.08)),
            leaf_y + float(text.get("leaf_label_dy", -0.40)),
            f"{level_count - 1}",
            fontsize=style["label_fontsize"],
            color=colors["annotation"],
            ha="center",
            va="center",
        )

    ax.set_aspect("equal")
    ax.set_facecolor(colors["tree_background"])
    ax.set_xlim(
        -float(geometry["root_line_length"]) - float(geometry["tree_left_pad"]),
        float(geometry["tree_height"]) + float(geometry["tree_right_pad"]),
    )
    y_extent = 0.5 * float(geometry["tree_width"]) + float(
        geometry["tree_vertical_pad"]
    )
    ax.set_ylim(-y_extent, y_extent)
    return positions, palette


def _draw_level_assignment_maze(
    ax,
    maze,
    *,
    colors: dict,
    style: dict,
    grid_size: int = 15,
    level_definition: str | None = None,
):
    level_grid, level_by_cell = _maze_level_grid(
        maze, grid_size=grid_size, level_definition=level_definition
    )
    level_count = int(level_by_cell.max()) + 1 if level_by_cell.size else 1
    palette = _level_assignment_palette(level_count, colors)
    for (x, y), _cell_id in maze.ce.items():
        level_idx = int(level_grid[int(y), int(x)])
        ax.add_patch(
            patches.Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                facecolor=palette[level_idx],
                edgecolor=colors["cell_edge"],
                linewidth=style["cell_linewidth"],
            )
        )
    ax.plot(
        maze.wa[:, 0],
        maze.wa[:, 1],
        color=colors["maze_outline"],
        linewidth=style["maze_outline_linewidth"],
        solid_capstyle="round",
    )
    ax.set_aspect("equal")
    ax.set_facecolor(colors["maze_background"])
    return level_grid, palette


def render_level_assignment_schematic(
    *,
    maze=None,
    maze_id: int = DEFAULT_MAZE_ID,
    grid_size: int = 15,
    level_definition: str | None = None,
    panel_size=None,
    layout: dict | None = None,
    geometry: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    text: dict | None = None,
    save: bool = False,
    panel_name: str = "level_assignment_schematic",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    maze = get_panel_maze(maze_id) if maze is None else maze
    resolved_layout = get_level_assignment_schematic_layout(layout)
    resolved_geometry = get_level_assignment_schematic_geometry(geometry)
    resolved_colors = get_level_assignment_schematic_colors(colors)
    resolved_style = get_level_assignment_schematic_style(style)
    resolved_text = get_level_assignment_schematic_text(text)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=panel_size or resolved_layout["panel_size"],
        squeeze=False,
        gridspec_kw={
            "width_ratios": tuple(resolved_layout.get("width_ratios", (1.0, 1.28)))
        },
    )
    fig.subplots_adjust(wspace=float(resolved_layout.get("grid_wspace", 0.16)))
    ax_tree, ax_maze = axes[0]

    tree_positions, level_colors = _draw_level_assignment_tree(
        ax_tree,
        geometry=resolved_geometry,
        colors=resolved_colors,
        style=resolved_style,
        text=resolved_text,
    )
    level_grid, _ = _draw_level_assignment_maze(
        ax_maze,
        maze,
        colors=resolved_colors,
        style=resolved_style,
        grid_size=grid_size,
        level_definition=level_definition,
    )
    maze_margin = float(resolved_geometry["maze_margin"])
    ax_maze.set_xlim(-0.5 - maze_margin, grid_size - 0.5 + maze_margin)
    ax_maze.set_ylim(-0.5 - maze_margin, grid_size - 0.5 + maze_margin)
    ax_tree.set_anchor("E")
    ax_maze.set_anchor("W")

    finish_panel(ax_tree)
    finish_panel(ax_maze)

    panel_data = {
        "layout": resolved_layout,
        "geometry": resolved_geometry,
        "colors": resolved_colors,
        "style": resolved_style,
        "text": resolved_text,
        "tree_positions": tree_positions,
        "level_grid": level_grid,
        "level_colors": level_colors,
    }
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, axes[0], panel_data, saved_paths


def render_maze_schematic_panel(
    *,
    maze=None,
    maze_id: int = DEFAULT_MAZE_ID,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    gray_blocks=GRAY_BLOCKS,
    quadrants=None,
    path_points=None,
    landmarks: dict | None = None,
    show_path_vertices: bool = True,
    save: bool = False,
    panel_name: str = "maze_schematic",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    maze = get_panel_maze(maze_id) if maze is None else maze
    fig, ax = make_panel_figure(panel_size=panel_size, layout=layout)
    draw_maze_outline(ax, maze, panel_size=panel_size, layout=layout, colors=colors)
    draw_gray_blocks(
        ax, panel_size=panel_size, blocks=gray_blocks, layout=layout, colors=colors
    )
    draw_quadrant_overlays(
        ax,
        maze,
        panel_size=panel_size,
        quadrants=quadrants,
        layout=layout,
        colors=colors,
    )
    if path_points is not None:
        draw_connected_path(
            ax,
            path_points,
            panel_size=panel_size,
            layout=layout,
            colors=colors,
            show_vertices=show_path_vertices,
        )
    draw_maze_landmarks(ax, landmarks)
    panel_layout = get_effective_panel_layout(
        panel_size=panel_size, layout=layout, scale_visuals=False
    )
    if panel_layout.get("maze_xlim") is not None:
        ax.set_xlim(*panel_layout["maze_xlim"])
    if panel_layout.get("maze_ylim") is not None:
        ax.set_ylim(*panel_layout["maze_ylim"])
    finish_panel(ax)
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, ax, saved_paths


def load_neuron_data_bundle(path):
    """Load a saved example-cell ``.npz`` bundle.

    Scalars stored as 0-d arrays are unwrapped via ``.item()``; arrays are
    returned as ``numpy.ndarray``. Use alongside
    :func:`resolve_neuron_data_path` when only ``(pair_key, row_idx)``
    are known.
    """
    with np.load(Path(path), allow_pickle=False) as z:
        return {
            key: (z[key].item() if z[key].shape == () else np.asarray(z[key]))
            for key in z.files
        }


def resolve_neuron_data_path(pair_key, row_idx, saved_dir):
    """Find the ``.npz`` bundle for ``(pair_key, row_idx)``.

    Files follow the pattern produced by
    ``scripts/save_example_neuron_picks.py``:
    ``<pair_safe>__row<NNNN>_A<cell>_B<cell>.npz``.
    Raises ``FileNotFoundError`` if no match exists.
    """
    safe = str(pair_key).replace("->", "_to_").replace("/", "_")
    pattern = f"{safe}__row{int(row_idx):04d}_*.npz"
    matches = sorted(Path(saved_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no saved bundle matching {pattern!r} under {saved_dir}; "
            f"run scripts/save_example_neuron_picks.py first."
        )
    return matches[0]


def load_example_cell_arrow_data(pack, side):
    """Read saved trajectory and arrow coordinates, without raw-data processing."""
    keys = ("traj_x", "traj_y", "px", "py", "ex", "ey")
    return {key: np.asarray(pack[f"{side}_{key}"]) for key in keys}


def _add_triangle_arrow_head(
    ax,
    x: float,
    y: float,
    dx: float,
    dy: float,
    *,
    head_width: float,
    head_length: float,
    facecolor,
    edgecolor,
    alpha: float,
    linewidth: float,
    zorder: int,
) -> None:
    """Draw a direction marker using only a triangular arrow head."""
    vector = np.array([dx, dy], dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return
    direction = vector / norm
    normal = np.array([-direction[1], direction[0]], dtype=float)
    tip = np.array([x + dx, y + dy], dtype=float)
    base = tip - float(head_length) * direction
    half_width = 0.5 * float(head_width)
    points = np.vstack(
        [
            tip,
            base + half_width * normal,
            base - half_width * normal,
        ]
    )
    ax.add_patch(
        patches.Polygon(
            points,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def render_example_cell_grid_panel(
    *,
    picks,
    saved_dir,
    mode: str = "arrow",
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "example_cell_grid",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Draw the saved trajectory and arrow coordinates for paired example neurons. No activity threshold, place field or occupancy map is computed here."""
    mode = str(mode).lower()
    if mode != "arrow":
        raise ValueError("Only saved arrow geometry is supported")
    picks = list(picks)
    n_cols = len(picks)
    if n_cols == 0:
        raise ValueError("picks must be non-empty")
    panel_layout = {
        "panel_size": (2.3 * max(n_cols, 1), 5.4),
        "grid_wspace": 0.02,
        "grid_hspace": 0.05,
        "margins": {"left": 0.04, "right": 0.995, "top": 0.86, "bottom": 0.02},
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "maze_line": "k",
        "trajectory": "0.55",
        "arrow_face": "red",
        "arrow_edge": "red",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "show_trajectory": True,
        "trajectory_alpha": 0.28,
        "trajectory_linewidth": 0.3,
        "trajectory_zorder": 1,
        "arrow_head_width": 0.15,
        "arrow_head_length": 0.4,
        "arrow_linewidth": 0.01,
        "arrow_alpha": 0.4,
        "arrow_scale": 0.25,
        "arrow_zorder": 2,
        "axis_lim": (-0.5, 14.5),
        "show_frame": False,
        "title_fontsize": None,
        "title_pad": 4,
        "title_location": "top",
        "title_y": -0.1,
        "row_label_fontsize": None,
        "row_label_pad": None,
        "tight_layout": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "row_top": "Day i",
        "row_bottom": "Day i + 90",
        "title_template": "{place_r:.2f}",
    }
    if labels:
        panel_labels.update(labels)
    saved_dir = Path(saved_dir)
    (fig, axes) = plt.subplots(
        2,
        n_cols,
        figsize=panel_size or panel_layout["panel_size"],
        gridspec_kw=dict(
            wspace=panel_layout["grid_wspace"], hspace=panel_layout["grid_hspace"]
        ),
        squeeze=False,
    )
    panel_data = []
    show_frame = bool(panel_style["show_frame"])
    title_template = panel_labels.get("title_template")
    _meta_keys = ("pair_key", "row_idx", "place_r")

    def _add_column_title(col: int, text: str) -> None:
        title_kwargs = {"pad": int(panel_style["title_pad"])}
        if panel_style.get("title_fontsize") is not None:
            title_kwargs["fontsize"] = int(panel_style["title_fontsize"])
        title_location = str(panel_style.get("title_location", "top")).lower()
        if title_location == "bottom":
            axes[-1, col].text(
                0.5,
                float(panel_style.get("title_y", -0.1)),
                text,
                transform=axes[-1, col].transAxes,
                ha="center",
                va="top",
                clip_on=False,
                **{k: v for (k, v) in title_kwargs.items() if k != "pad"},
            )
        elif title_location == "top":
            axes[0, col].set_title(text, **title_kwargs)
        else:
            raise ValueError("title_location must be 'top' or 'bottom'.")

    axis_lim = panel_style["axis_lim"]
    show_trajectory = bool(panel_style["show_trajectory"])
    arrow_scale = float(panel_style["arrow_scale"])
    arrow_head_kwargs = dict(
        head_width=float(panel_style["arrow_head_width"]),
        head_length=float(panel_style["arrow_head_length"]),
        facecolor=str(panel_colors["arrow_face"]),
        edgecolor=str(panel_colors["arrow_edge"]),
        alpha=float(panel_style["arrow_alpha"]),
        linewidth=float(panel_style["arrow_linewidth"]),
        zorder=int(panel_style["arrow_zorder"]),
    )
    for (col, pick) in enumerate(picks):
        pick = tuple(pick)
        if len(pick) != 2:
            raise ValueError(f"each pick must be (pair_key, row_idx); got {pick!r}")
        (pair_key, row_idx) = pick
        bundle_path = resolve_neuron_data_path(pair_key, row_idx, saved_dir)
        pack = load_neuron_data_bundle(bundle_path)
        arrows_a = load_example_cell_arrow_data(pack, "A")
        arrows_b = load_example_cell_arrow_data(pack, "B")
        for (ax, arrows) in zip(axes[:, col], (arrows_a, arrows_b)):
            if show_trajectory:
                ax.plot(
                    arrows["traj_x"],
                    arrows["traj_y"],
                    color=str(panel_colors["trajectory"]),
                    linewidth=float(panel_style["trajectory_linewidth"]),
                    alpha=float(panel_style["trajectory_alpha"]),
                    rasterized=True,
                    zorder=int(panel_style["trajectory_zorder"]),
                )
            for (a, b, e, f) in zip(
                arrows["px"], arrows["py"], arrows["ex"], arrows["ey"]
            ):
                dx = (e - a) * arrow_scale
                dy = (f - b) * arrow_scale
                _add_triangle_arrow_head(ax, a, b, dx, dy, **arrow_head_kwargs)
            ax.set_aspect("equal")
            ax.set_xlim(axis_lim)
            ax.set_ylim(axis_lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if not show_frame:
                for spine in ax.spines.values():
                    spine.set_visible(False)
        entry = {
            "pair_key": str(pair_key),
            "row_idx": int(row_idx),
            "place_r": float(pack.get("place_r", np.nan)),
            "arrows_a": arrows_a,
            "arrows_b": arrows_b,
            "bundle_path": str(bundle_path),
        }
        panel_data.append(entry)
        if title_template:
            fmt_ns = {k: pack[k] for k in _meta_keys if k in pack}
            fmt_ns.setdefault("pair_key", str(pair_key))
            fmt_ns.setdefault("row_idx", int(row_idx))
            _add_column_title(col, title_template.format(**fmt_ns))
    row_label_kwargs = {}
    if panel_style.get("row_label_fontsize") is not None:
        row_label_kwargs["fontsize"] = int(panel_style["row_label_fontsize"])
    if panel_style.get("row_label_pad") is not None:
        row_label_kwargs["labelpad"] = int(panel_style["row_label_pad"])
    if panel_labels.get("row_top"):
        axes[0, 0].set_ylabel(str(panel_labels["row_top"]), **row_label_kwargs)
    if panel_labels.get("row_bottom"):
        axes[1, 0].set_ylabel(str(panel_labels["row_bottom"]), **row_label_kwargs)
    if bool(panel_style["tight_layout"]):
        fig.tight_layout()
    else:
        margins = dict(panel_layout.get("margins", {}))
        fig.subplots_adjust(
            left=float(margins.get("left", 0.04)),
            right=float(margins.get("right", 0.995)),
            top=float(margins.get("top", 0.86)),
            bottom=float(margins.get("bottom", 0.02)),
            wspace=float(panel_layout["grid_wspace"]),
            hspace=float(panel_layout["grid_hspace"]),
        )
    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return (fig, axes, panel_data, saved_paths)
