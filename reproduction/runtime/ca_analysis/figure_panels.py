"""Shared plotting helpers for the included derived Source Data.

Only the functions required by current figure builders are included. Inputs are
saved tables, image arrays, map vectors, or display coordinates; this module
does not load raw recording sessions or calculate place fields.
"""

from __future__ import annotations


from copy import deepcopy

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib import patches


from matplotlib.lines import Line2D

from matplotlib.offsetbox import AnnotationBbox, DrawingArea

from matplotlib.path import Path as MplPath

from matplotlib.patches import FancyArrowPatch

from scipy import stats



from .maze import NewMaze


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
    color = "#000000"
    linewidth = 1.0
    zorder = float(marker_spec.get("zorder", 12))
    pad = size * 0.24
    drawing = DrawingArea(size, size, 0, 0)
    drawing.add_artist(patches.Circle(
        (size / 2, size / 2), size * 0.24,
        facecolor="none", edgecolor=color, linewidth=linewidth,
    ))
    drawing.add_artist(
        Line2D(
            [pad, size - pad],
            [pad, size - pad],
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
