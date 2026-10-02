from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import io as sio
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

from .figure_colors import (
    CORRELATION_EXPERIENCE_COLORS,
    CORRELATION_MEASURE_COLORS,
    REPRESENTATION_COLOR_ALIASES,
    REPRESENTATION_COLORS,
    get_correlation_experience_colors,
    get_correlation_measure_colors,
    get_representation_colors,
)
from .linearized_pf_utils import STABLE_THR, WP_BINS, get_data_root, load_pair_data
from .level_heatmap_utils import (
    collect_var_from_variant,
    normalize_maps,
    refine_order_by_argmax,
    stack_level_maps,
)
from . import session_loader_tools as sl
from .maze import NewMaze
from .mm_plot_utils import plot
from .similarity import compute_tuning_and_prior
from .spatial import discretize_positions, generate_level_list, get_level_list
from .trajectory import LoadTraj, TimeInMaze

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
CYCLE_DIAGRAM_LINEAR_SCALE_KEYS = (
    "box_linewidth",
    "rounding_size",
    "loop_linewidth",
    "arrow_linewidth",
    "arrow_mutation_scale",
)
END_NODE_LOWER_THRESHOLD = 63
END_NODE_UPPER_THRESHOLD = 126

DEFAULT_LEVEL_CODE_DAY1_DAY1_GROUPS = {
    "once": [
        ("1616day1", "1616bday1"),
        ("418RRday1", "418-3day1"),
        ("434RRday1", "434-3day1"),
        ("451Rday1", "451-3day1"),
        ("488Rday1", "488-3day1"),
        ("489Lday1", "489-3day1"),
        ("579RLday1", "579-3day1"),
        ("468Lday1", "468-3day1"),
        ("422RRday1", "422-3day1"),
    ],
    "twice": [
        ("1616bday1", "1616b-3day1"),
        ("418-3day1", "418-6day1"),
        ("451-3day1", "451-6day1"),
        ("488-3day1", "488-6day1"),
        ("489-3day1", "489-6day1"),
        ("468-3day1", "468-6day1"),
        ("422-3day1", "422-6day1"),
    ],
    "three times": [
        ("418-6day1", "418-9day1"),
        ("451-6day1", "451-9day1"),
        ("468-6day1", "468-9day1"),
    ],
    "four times": [
        ("468-9day1", "468-12day1"),
    ],
}

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

END_NODE_COLOR_CODE_ALTERNATIVES = {
    "option_4": {
        "lower_left": "#ead7e4",
        "lower_right": "#818b70",
        "upper_left": "#fcc5d9",
        "upper_right": "#ff8486",
    },
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

QUADRANT_SPECS = [
    (np.arange(-0.5, 7.5, 2), np.arange(-0.5, 7.5, 2), PANEL_COLORS["quadrants"]["lower_left"]),
    (np.arange(7.5, 16.5, 2), np.arange(-0.5, 7.5, 2), PANEL_COLORS["quadrants"]["lower_right"]),
    (np.arange(-0.5, 7.5, 2), np.arange(7.5, 16.5, 2), PANEL_COLORS["quadrants"]["upper_left"]),
    (np.arange(7.5, 16.5, 2), np.arange(7.5, 16.5, 2), PANEL_COLORS["quadrants"]["upper_right"]),
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

CYCLE_DIAGRAM_LAYOUT = {
    "reference_panel_size": (4.07, 1.82),
    "panel_size": (4.07, 1.82),
    "coordinate_width": 407,
    "coordinate_height": 182,
}

CYCLE_DIAGRAM_COLORS = {
    "background": "#ececec",
    "left_box_edge": "#0b7a5d",
    "left_box_face": "#cfe2dc",
    "left_text": "#0a5f4e",
    "right_box_edge": "#6f6f6f",
    "right_box_face": "#ece9e3",
    "right_text": "#4f4f4f",
    "line": "#6b6b6b",
    "bottom_text": "#666666",
}

CYCLE_DIAGRAM_STYLE = {
    "box_linewidth": 1.5,
    "rounding_size": 15,
    "loop_linewidth": 3.2,
    "arrow_linewidth": 2.0,
    "arrow_mutation_scale": 14,
}

CYCLE_DIAGRAM_BOXES = {
    "recording": {
        "xy": (48, 11),
        "width": 143,
        "height": 109,
        "edgecolor_key": "left_box_edge",
        "facecolor_key": "left_box_face",
    },
    "rest": {
        "xy": (232, 11),
        "width": 142,
        "height": 109,
        "edgecolor_key": "right_box_edge",
        "facecolor_key": "right_box_face",
    },
}

CYCLE_DIAGRAM_LOOP = {
    "x": [374, 395, 395, 7, 7],
    "y": [67, 67, 179, 179, 67],
}

CYCLE_DIAGRAM_ARROWS = [
    {
        "start": (7, 67),
        "end": (48, 67),
    },
    {
        "start": (191, 67),
        "end": (232, 67),
    },
]

CYCLE_DIAGRAM_TEXT = [
    {
        "x": 119.5,
        "y": 50,
        "text": "Day 1–2",
        "fontsize": 18,
        "fontweight": "bold",
        "color_key": "left_text",
    },
    {
        "x": 119.5,
        "y": 82,
        "text": "Recording",
        "fontsize": 15,
        "color_key": "left_text",
    },
    {
        "x": 303,
        "y": 50,
        "text": "Day 3–90",
        "fontsize": 17,
        "fontweight": "bold",
        "color_key": "right_text",
    },
    {
        "x": 303,
        "y": 82,
        "text": "Rest",
        "fontsize": 15,
        "color_key": "right_text",
    },
    {
        "x": 202,
        "y": 151,
        "text": "Repeats every 3 months",
        "fontsize": 12.5,
        "color_key": "bottom_text",
    },
]

SPLIT_DAY_CYCLE_DIAGRAM_LAYOUT = {
    "reference_panel_size": (6.0, 2.0),
    "panel_size": (6.0, 2.0),
    "coordinate_width": 600,
    "coordinate_height": 200,
}

SPLIT_DAY_CYCLE_DIAGRAM_COLORS = {
    "background": "#ffffff",
    "day1_box_edge": "#0b7a5d",
    "day1_box_face": "#cfe2dc",
    "day1_text": "#0a5f4e",
    "day2_box_edge": "#0b7a5d",
    "day2_box_face": "#cfe2dc",
    "day2_text": "#0a5f4e",
    "rest_box_edge": "#6f6f6f",
    "rest_box_face": "#ece9e3",
    "rest_text": "#4f4f4f",
}

SPLIT_DAY_CYCLE_DIAGRAM_STYLE = {
    **CYCLE_DIAGRAM_STYLE,
    "loop_linewidth": 2.4,
    "arrow_linewidth": 2.4,
}

SPLIT_DAY_CYCLE_DIAGRAM_BOXES = {
    "day1": {
        "xy": (36, 18),
        "width": 170,
        "height": 116,
        "edgecolor_key": "day1_box_edge",
        "facecolor_key": "day1_box_face",
    },
    "day2": {
        "xy": (248, 18),
        "width": 170,
        "height": 116,
        "edgecolor_key": "day2_box_edge",
        "facecolor_key": "day2_box_face",
    },
    "rest": {
        "xy": (460, 18),
        "width": 108,
        "height": 116,
        "edgecolor_key": "rest_box_edge",
        "facecolor_key": "rest_box_face",
    },
}

SPLIT_DAY_CYCLE_DIAGRAM_LOOP = {
    "x": [568, 590, 590, 10, 10],
    "y": [76, 76, 190, 190, 76],
}

SPLIT_DAY_CYCLE_DIAGRAM_ARROWS = [
    {
        "start": (10, 76),
        "end": (36, 76),
    },
    {
        "start": (206, 76),
        "end": (248, 76),
    },
    {
        "start": (418, 76),
        "end": (460, 76),
    },
]

SPLIT_DAY_CYCLE_DIAGRAM_TEXT = [
    {
        "x": 121,
        "y": 42,
        "text": "Day $i$",
        "fontsize": 17,
        "fontweight": "bold",
        "color_key": "day1_text",
    },
    {
        "x": 121,
        "y": 78,
        "text": "$i = 1$: Naive day",
        "fontsize": 11.5,
        "color_key": "day1_text",
    },
    {
        "x": 121,
        "y": 115,
        "text": "$i > 1$: Return day",
        "fontsize": 11.5,
        "color_key": "day1_text",
    },
    {
        "x": 333,
        "y": 48,
        "text": "Day $i+1$",
        "fontsize": 17,
        "fontweight": "bold",
        "color_key": "day2_text",
    },
    {
        "x": 333,
        "y": 88,
        "text": "Second day",
        "fontsize": 12.5,
        "color_key": "day2_text",
    },
    {
        "x": 514,
        "y": 48,
        "text": "Rest",
        "fontsize": 16,
        "fontweight": "bold",
        "color_key": "rest_text",
    },
    {
        "x": 514,
        "y": 88,
        "text": "~3 months",
        "fontsize": 12.5,
        "color_key": "rest_text",
    },
    {
        "x": 228,
        "y": 180,
        "text": "Next cycle:",
        "fontsize": 12.5,
        "color_key": "bottom_text",
    },
    {
        "x": 300,
        "y": 180,
        "text": "$i$",
        "fontsize": 12.5,
        "color_key": "bottom_text",
    },
    {
        "x": 328,
        "y": 180,
        "text": r"$\rightarrow$",
        "fontsize": 12.5,
        "color_key": "bottom_text",
    },
    {
        "x": 370,
        "y": 180,
        "text": "$i+90$",
        "fontsize": 12.5,
        "color_key": "bottom_text",
    },
]

ROUTE_CHOICE_SCHEMATIC_LAYOUT = {
    "reference_panel_size": (9, 11),
    "panel_size": (9, 11),
    "xlim": (-4.4, 4.2),
    "ylim": (-2.2, 2.8),
    "subplots": (2, 1),
    "subplots_adjust": {"hspace": 0.02},
    "tight_layout": True,
}

ROUTE_CHOICE_SCHEMATIC_GEOMETRY = {
    "center": (1.2, 0.0),
    "top": (1.2, 1.55),
    "left": (-0.25, -1.0),
    "right": (2.65, -1.0),
}

ROUTE_CHOICE_SCHEMATIC_COLORS = {
    "tree": "black",
    "solid_arrow": "#8ecae6",
    "dotted_arrow": "#8a8a8a",
}

ROUTE_CHOICE_SCHEMATIC_STYLE = {
    "tree_linewidth": 6.2,
    "arrow_linewidth": 5.8,
    "title_fontsize": 30,
    "label_fontsize": 33,
    "or_fontsize": 27,
    "annotation_fontsize": 23,
    "arrow_mutation_scale": 22,
    "dotted_head_length": 0.22,
}

ROUTE_CHOICE_SCHEMATIC_TEXT = {
    "node_labels": {"top": "S", "left": "B", "right": "B"},
    "node_label_offsets": {"top": (0.0, 0.20), "left": (-0.14, -0.20), "right": (0.14, -0.20)},
    "rows": [
        {
            "title": "Stem to Branch\nAlternation Bias\n(SA)",
            "title_xy": (-4.1, 0.5),
            "or_xy": (1.2, -0.47),
            "annotation": "preceding\nturn",
            "annotation_xy": (1.92, 0.56),
            "arrows": [
                {"start": "top", "corner": "center", "end": "left", "offset": -0.20, "trim_start": 0.12, "trim_end": 0.22},
                {"start": "top", "corner": "center", "end": "right", "offset": 0.20, "trim_start": 0.12, "trim_end": 0.22},
                {
                    "start": "top",
                    "corner": "center",
                    "end": "right",
                    "offset": 0.43,
                    "trim_start": 0.16,
                    "trim_end": 0.28,
                    "dotted": True,
                },
            ],
        },
        {
            "title": "Branch to Stem\nProbability (BS)",
            "title_xy": (-4.1, 0.2),
            "or_sector": ("top", "right"),
            "or_distance": 0.42,
            "arrows": [
                {"start": "left", "corner": "center", "end": "top", "offset": 0.17, "trim_start": 0.18, "trim_end": 0.18},
                {"start": "left", "corner": "center", "end": "right", "offset": -0.17, "trim_start": 0.18, "trim_end": 0.22},
            ],
        },
    ],
}

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
                {"text": "preceding\nturn", "xy": (2.76, 3.66), "ha": "right", "va": "center"},
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
    ref_size = tuple(panel_layout.get("reference_panel_size", PANEL_LAYOUT["reference_panel_size"]))
    ref_w, ref_h = float(ref_size[0]), float(ref_size[1])
    if ref_w <= 0 or ref_h <= 0:
        return 1.0
    target_w, target_h = float(target_size[0]), float(target_size[1])
    return min(target_w / ref_w, target_h / ref_h)


def get_effective_panel_layout(panel_size=None, layout: dict | None = None, *, scale_visuals: bool = True):
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


def get_effective_rcparams(overrides: dict | None = None, *, panel_size=None, layout: dict | None = None):
    rcparams = dict(PANEL_RCPARAMS)
    if overrides:
        rcparams.update(overrides)
    if panel_size is not None:
        scale = get_panel_scale(panel_size=panel_size, layout=layout)
        for key in RCPARAM_LINEAR_SCALE_KEYS:
            if key in rcparams:
                rcparams[key] = rcparams[key] * scale
    return rcparams


def apply_panel_rcparams(overrides: dict | None = None, *, panel_size=None, layout: dict | None = None):
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


def get_cycle_diagram_layout(overrides: dict | None = None):
    layout = dict(CYCLE_DIAGRAM_LAYOUT)
    if overrides:
        layout.update(overrides)
    return layout


def get_cycle_diagram_colors(overrides: dict | None = None):
    colors = deepcopy(CYCLE_DIAGRAM_COLORS)
    if overrides:
        colors.update(overrides)
    return colors


def get_cycle_diagram_style(overrides: dict | None = None, *, panel_size=None, layout: dict | None = None):
    style = dict(CYCLE_DIAGRAM_STYLE)
    if overrides:
        style.update(overrides)
    cycle_layout = get_cycle_diagram_layout(layout)
    target_panel_size = tuple(panel_size or cycle_layout["panel_size"])
    scale = get_panel_scale(panel_size=target_panel_size, layout=cycle_layout)
    for key in CYCLE_DIAGRAM_LINEAR_SCALE_KEYS:
        style[key] = style[key] * scale
    return style


def resolve_cycle_diagram_boxes(boxes: dict, colors: dict | None = None):
    cycle_colors = get_cycle_diagram_colors(colors)
    resolved = deepcopy(boxes)
    for spec in resolved.values():
        if "edgecolor_key" in spec:
            spec["edgecolor"] = cycle_colors[spec.pop("edgecolor_key")]
        if "facecolor_key" in spec:
            spec["facecolor"] = cycle_colors[spec.pop("facecolor_key")]
    return resolved


def get_cycle_diagram_boxes(colors: dict | None = None):
    return resolve_cycle_diagram_boxes(CYCLE_DIAGRAM_BOXES, colors=colors)


def resolve_cycle_diagram_text_specs(
    text_specs: list[dict],
    colors: dict | None = None,
    *,
    panel_size=None,
    layout: dict | None = None,
):
    cycle_colors = get_cycle_diagram_colors(colors)
    cycle_layout = get_cycle_diagram_layout(layout)
    target_panel_size = tuple(panel_size or cycle_layout["panel_size"])
    scale = get_panel_scale(panel_size=target_panel_size, layout=cycle_layout)
    resolved = deepcopy(text_specs)
    for spec in resolved:
        if "color_key" in spec:
            spec["color"] = cycle_colors[spec.pop("color_key")]
        spec["fontsize"] = spec["fontsize"] * scale
    return resolved


def get_cycle_diagram_text_specs(colors: dict | None = None, *, panel_size=None, layout: dict | None = None):
    return resolve_cycle_diagram_text_specs(
        CYCLE_DIAGRAM_TEXT,
        colors=colors,
        panel_size=panel_size,
        layout=layout,
    )


def get_split_day_cycle_diagram_specs(
    *,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    boxes: dict | None = None,
    loop: dict | None = None,
    arrows: list[dict] | None = None,
    text_specs: list[dict] | None = None,
):
    cycle_layout = get_cycle_diagram_layout(SPLIT_DAY_CYCLE_DIAGRAM_LAYOUT)
    if layout:
        cycle_layout.update(layout)

    cycle_colors = get_cycle_diagram_colors(SPLIT_DAY_CYCLE_DIAGRAM_COLORS)
    if colors:
        cycle_colors.update(colors)

    cycle_style = dict(SPLIT_DAY_CYCLE_DIAGRAM_STYLE)
    if style:
        cycle_style.update(style)

    return {
        "layout": cycle_layout,
        "colors": cycle_colors,
        "style": cycle_style,
        "boxes": resolve_cycle_diagram_boxes(
            SPLIT_DAY_CYCLE_DIAGRAM_BOXES if boxes is None else boxes,
            colors=cycle_colors,
        ),
        "loop": deepcopy(SPLIT_DAY_CYCLE_DIAGRAM_LOOP if loop is None else loop),
        "arrows": deepcopy(SPLIT_DAY_CYCLE_DIAGRAM_ARROWS if arrows is None else arrows),
        "text_specs": resolve_cycle_diagram_text_specs(
            SPLIT_DAY_CYCLE_DIAGRAM_TEXT if text_specs is None else text_specs,
            colors=cycle_colors,
            panel_size=cycle_layout["panel_size"],
            layout=cycle_layout,
        ),
    }


def get_cycle_diagram_loop():
    return deepcopy(CYCLE_DIAGRAM_LOOP)


def get_cycle_diagram_arrows():
    return deepcopy(CYCLE_DIAGRAM_ARROWS)


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


def get_route_choice_schematic_layout(overrides: dict | None = None):
    return _merged_dict(ROUTE_CHOICE_SCHEMATIC_LAYOUT, overrides)


def get_route_choice_schematic_geometry(overrides: dict | None = None):
    return _merged_dict(ROUTE_CHOICE_SCHEMATIC_GEOMETRY, overrides)


def get_route_choice_schematic_colors(overrides: dict | None = None):
    return _merged_dict(ROUTE_CHOICE_SCHEMATIC_COLORS, overrides)


def get_route_choice_schematic_style(overrides: dict | None = None):
    return _merged_dict(ROUTE_CHOICE_SCHEMATIC_STYLE, overrides)


def get_route_choice_schematic_text(overrides: dict | None = None):
    return _merged_dict(ROUTE_CHOICE_SCHEMATIC_TEXT, overrides)


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


def _normal_from_direction(v):
    v = _unit_vector(v)
    return np.array([-v[1], v[0]])


def _line_intersection(p1, d1, p2, d2):
    matrix = np.column_stack([d1, -d2])
    rhs = p2 - p1
    t, _ = np.linalg.solve(matrix, rhs)
    return p1 + t * d1


def _offset_polyline(p_start, p_corner, p_end, offset, trim_start=0.1, trim_end=0.1):
    p_start = np.asarray(p_start, dtype=float)
    p_corner = np.asarray(p_corner, dtype=float)
    p_end = np.asarray(p_end, dtype=float)

    d1 = _unit_vector(p_corner - p_start)
    d2 = _unit_vector(p_end - p_corner)
    n1 = _normal_from_direction(d1)
    n2 = _normal_from_direction(d2)

    a1 = p_start + offset * n1
    a2 = p_corner + offset * n2
    elbow = _line_intersection(a1, d1, a2, d2)
    q_start = p_start + offset * n1 + trim_start * d1
    q_end = p_end + offset * n2 - trim_end * d2
    return np.vstack([q_start, elbow, q_end])


def _add_poly_arrow(ax, pts, *, linewidth, color, mutation_scale):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
    path = MplPath(pts, codes)
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="->",
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


def _add_dotted_poly_arrow(ax, pts, *, linewidth, color, mutation_scale, head_length):
    pts = np.asarray(pts, dtype=float)
    direction = _unit_vector(pts[-1] - pts[-2])
    head_base = pts[-1] - head_length * direction
    shaft_pts = np.vstack([pts[:-1], head_base])
    ax.plot(
        shaft_pts[:, 0],
        shaft_pts[:, 1],
        color=color,
        lw=linewidth,
        linestyle=":",
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=2,
    )
    head = FancyArrowPatch(
        posA=head_base,
        posB=pts[-1],
        arrowstyle="->",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        fill=False,
        zorder=4,
        joinstyle="round",
        capstyle="round",
    )
    ax.add_patch(head)
    return head


def _draw_route_tree(ax, points, *, linewidth, color):
    center = points["center"]
    for key in ("top", "left", "right"):
        point = points[key]
        ax.plot(
            [center[0], point[0]],
            [center[1], point[1]],
            color=color,
            lw=linewidth,
            solid_capstyle="round",
            zorder=1,
        )


def _resolve_route_points(geometry):
    return {key: np.asarray(value, dtype=float) for key, value in geometry.items()}


def _resolve_or_position(row_spec, points):
    if "or_xy" in row_spec:
        return np.asarray(row_spec["or_xy"], dtype=float)
    first_key, second_key = row_spec.get("or_sector", ("top", "right"))
    sector_dir = _unit_vector((points[first_key] - points["center"]) + (points[second_key] - points["center"]))
    return points["center"] + float(row_spec.get("or_distance", 0.42)) * sector_dir


def _draw_route_choice_row(ax, row_spec, *, points, labels, colors, style):
    _draw_route_tree(ax, points, linewidth=style["tree_linewidth"], color=colors["tree"])

    title_xy = row_spec["title_xy"]
    ax.text(
        title_xy[0],
        title_xy[1],
        row_spec["title"],
        fontsize=style["title_fontsize"],
        ha="left",
        va="center",
    )

    node_labels = labels.get("node_labels", {})
    node_offsets = labels.get("node_label_offsets", {})
    for key, label in node_labels.items():
        xy = points[key] + np.asarray(node_offsets.get(key, (0.0, 0.0)), dtype=float)
        va = "bottom" if key == "top" else "top"
        ax.text(xy[0], xy[1], label, fontsize=style["label_fontsize"], ha="center", va=va)

    or_xy = _resolve_or_position(row_spec, points)
    ax.text(
        or_xy[0],
        or_xy[1],
        row_spec.get("or_text", "or"),
        fontsize=style["or_fontsize"],
        color=colors["solid_arrow"],
        ha="center",
        va="center",
    )

    for arrow_spec in row_spec.get("arrows", []):
        pts = _offset_polyline(
            points[arrow_spec["start"]],
            points[arrow_spec["corner"]],
            points[arrow_spec["end"]],
            arrow_spec["offset"],
            trim_start=arrow_spec.get("trim_start", 0.1),
            trim_end=arrow_spec.get("trim_end", 0.1),
        )
        if arrow_spec.get("dotted", False):
            _add_dotted_poly_arrow(
                ax,
                pts,
                linewidth=style["arrow_linewidth"],
                color=colors["dotted_arrow"],
                mutation_scale=style["arrow_mutation_scale"],
                head_length=style["dotted_head_length"],
            )
        else:
            _add_poly_arrow(
                ax,
                pts,
                linewidth=style["arrow_linewidth"],
                color=colors["solid_arrow"],
                mutation_scale=style["arrow_mutation_scale"],
            )

    if row_spec.get("annotation"):
        annotation_xy = row_spec["annotation_xy"]
        ax.text(
            annotation_xy[0],
            annotation_xy[1],
            row_spec["annotation"],
            fontsize=style["annotation_fontsize"],
            ha="left",
            va="center",
            color=colors["dotted_arrow"],
        )


def make_panel_figure(panel_size=None, layout: dict | None = None):
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout, scale_visuals=False)
    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    return fig, ax


def draw_maze_outline(ax, maze, *, panel_size=None, layout: dict | None = None, colors: dict | None = None):
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


def draw_gray_blocks(ax, *, panel_size=None, blocks=GRAY_BLOCKS, layout: dict | None = None, colors: dict | None = None):
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
        (np.arange(-0.5, 7.5, 2), np.arange(-0.5, 7.5, 2), panel_colors["quadrants"]["lower_left"]),
        (np.arange(7.5, 16.5, 2), np.arange(-0.5, 7.5, 2), panel_colors["quadrants"]["lower_right"]),
        (np.arange(-0.5, 7.5, 2), np.arange(7.5, 16.5, 2), panel_colors["quadrants"]["upper_left"]),
        (np.arange(7.5, 16.5, 2), np.arange(7.5, 16.5, 2), panel_colors["quadrants"]["upper_right"]),
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


def extract_endnode_timeline(
    nickname: str,
    *,
    lower_threshold: int = END_NODE_LOWER_THRESHOLD,
    upper_threshold: int = END_NODE_UPPER_THRESHOLD,
):
    tf = LoadTraj(f"{nickname}-tf")
    endnodes = []
    times = []
    for bout_idx in range(len(tf.no)):
        sel_tj = np.asarray(tf.no[bout_idx])
        if sel_tj.ndim != 2 or sel_tj.shape[1] < 2:
            continue
        sel_no = sel_tj[:, 0]
        sel_times = sel_tj[:, 1]
        start_time = tf.fr[bout_idx][0]
        abs_times = sel_times + start_time
        maze_times = [TimeInMaze(t, tf) for t in abs_times]
        for node, maze_time in zip(sel_no, maze_times):
            if lower_threshold <= node <= upper_threshold:
                endnodes.append(int(node))
                times.append(float(maze_time))
    return np.asarray(endnodes, dtype=int), np.asarray(times, dtype=float)


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
    if marker_spec.get("x_offset_points") is not None or marker_spec.get("y_offset_points") is not None:
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
    if marker_spec.get("x_offset_points") is not None or marker_spec.get("y_offset_points") is not None:
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
        _data_y_to_axes_fraction(ax, float(node)) + float(marker.get("y_axes_offset", 0.0)),
    )
    if kind in {"water", "drop", "reward"}:
        return _add_timeline_water_drop_icon(ax, xy_axes, marker)
    if kind in {"no_water", "no-water", "x"}:
        return _add_timeline_no_water_marker(ax, xy_axes, marker)
    raise ValueError(f"Unsupported endnode timeline marker kind: {marker.get('kind')!r}")


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
    ylabel: str | None = "End Nodes",
    xlabel: str | None = "Time in Maze (s)",
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


def render_endnode_timeline_panel(
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
    ylabel: str = "End Nodes",
    xlabel: str = "Time in Maze (s)",
    display_title: str | None = None,
    endnode_marker: dict | None = None,
    save: bool = False,
    panel_name: str | None = None,
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    fig, ax = make_panel_figure(panel_size=panel_size, layout=layout)
    draw_endnode_timeline_axes(
        ax,
        nickname,
        panel_size=panel_size,
        layout=layout,
        colors=colors,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        reward_node=reward_node,
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        ylabel=ylabel,
        xlabel=xlabel,
        display_title=display_title,
        endnode_marker=endnode_marker,
    )

    safe_name = panel_name or f"endnode_timeline_{str(nickname).replace('/', '_')}"
    saved_paths = save_panel(fig, safe_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, saved_paths


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
                ylabel="End Nodes" if col_idx == 0 else None,
                xlabel="Time in Maze (s)" if row_idx == n_rows - 1 else None,
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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def _centered_jitter_offsets(n_points: int, span: float, seed: int) -> np.ndarray:
    if n_points <= 1:
        return np.array([0.0], dtype=float)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-span / 2.0, span / 2.0, size=n_points)
    offsets -= offsets.mean()
    max_abs = np.max(np.abs(offsets))
    if max_abs > span / 2.0:
        offsets *= (span / 2.0) / max_abs
    return offsets.astype(float)


def render_variant_correlation_panel(
    *,
    data_dir,
    variant_order,
    panel_specs: list[dict] | None = None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    ylim=None,
    annotations: dict[str, dict[str, str]] | None = None,
    save: bool = False,
    panel_name: str = "variant_correlations",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    data_dir = Path(data_dir)
    variant_order = list(variant_order)
    colors = {} if colors is None else dict(colors)
    style = {} if style is None else dict(style)
    labels = {} if labels is None else dict(labels)
    annotations = {} if annotations is None else dict(annotations)
    default_titles = {
        "pv": labels.get("pv", "PV correlation"),
        "place_map": labels.get("place_map", "Place-map correlation"),
        "firing_rate": labels.get("firing_rate", "Firing-rate correlation"),
    }
    default_colors = get_representation_colors()
    if panel_specs is None:
        panel_specs = [
            {
                "panel_key": "pv",
                "sharey_group": "correlation",
                "title": default_titles["pv"],
                "mean_file": "panel_variant_means.csv",
                "mean_col": "pv_mean",
                "rows_file": "panel_variant_pv_rows.csv",
                "value_col": "pv_value",
                "color": colors.get("pv", default_colors["pv"]),
            },
            {
                "panel_key": "place_map",
                "sharey_group": "correlation",
                "title": default_titles["place_map"],
                "mean_file": "panel_variant_means.csv",
                "mean_col": "place_map_mean",
                "rows_file": "panel_variant_place_rows.csv",
                "value_col": "place_value",
                "color": colors.get("place_map", default_colors["place_map"]),
            },
        ]

    resolved_panel_specs = []
    for spec in panel_specs:
        panel_key = str(spec["panel_key"])
        spec_data_dir = Path(spec.get("data_dir", data_dir))
        rows_frame = pd.read_csv(spec_data_dir / str(spec["rows_file"]))
        mean_col = str(spec["mean_col"])
        if bool(spec.get("mean_from_rows", False)):
            mean_frame = (
                rows_frame.groupby("variant", observed=True)[str(spec["value_col"])]
                .mean()
                .rename(mean_col)
                .reindex(variant_order)
                .reset_index()
            )
        else:
            mean_frame = (
                pd.read_csv(spec_data_dir / str(spec.get("mean_file", "panel_variant_means.csv")))
                .set_index("variant")
                .reindex(variant_order)
                .reset_index()
            )
        resolved_panel_specs.append(
            {
                "panel_key": panel_key,
                "title": spec.get("title", default_titles.get(panel_key, panel_key)),
                "mean_col": mean_col,
                "value_col": str(spec["value_col"]),
                "rows": rows_frame,
                "color": spec.get("color", colors.get(panel_key, default_colors.get(panel_key, "#666666"))),
                "variant_means": mean_frame,
                "sharey_group": spec.get("sharey_group", panel_key),
            }
        )

    fig, axes = plt.subplots(
        1,
        len(resolved_panel_specs),
        figsize=panel_size or get_panel_layout(layout)["panel_size"],
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=float(get_panel_layout(layout).get("grid_wspace", 0.18)))

    shared_y_group_refs: dict[object, object] = {}
    for ax, panel_spec in zip(axes, resolved_panel_specs, strict=False):
        group_key = panel_spec.get("sharey_group")
        if group_key is None:
            continue
        if group_key not in shared_y_group_refs:
            shared_y_group_refs[group_key] = ax
            continue
        ax.sharey(shared_y_group_refs[group_key])

    variant_x = np.arange(len(variant_order), dtype=float)
    point_edge = colors.get("point_edge", "white")
    bar_width = float(style.get("bar_width", 0.72))
    bar_alpha = float(style.get("bar_alpha", 0.9))
    scatter_size = float(style.get("scatter_size", 28))
    scatter_alpha = float(style.get("scatter_alpha", 0.5))
    scatter_span = float(style.get("scatter_span", 0.26))
    scatter_seed = int(style.get("scatter_seed", 0))
    point_edge_linewidth = float(style.get("point_edge_linewidth", 0.35))
    annotation_offset_points = float(style.get("annotation_offset_points", 8.0))
    annotation_fontsize = float(style.get("annotation_fontsize", plt.rcParams.get("font.size", 12)))
    annotation_fontweight = style.get("annotation_fontweight", "bold")
    annotation_color = style.get("annotation_color", "black")

    for panel_idx, (ax, panel_spec) in enumerate(zip(axes, resolved_panel_specs, strict=False)):
        color = panel_spec["color"]
        variant_means = panel_spec["variant_means"]
        ax.bar(
            variant_x,
            variant_means[panel_spec["mean_col"]].to_numpy(dtype=float),
            width=bar_width,
            color=color,
            alpha=bar_alpha,
            zorder=1,
        )
        for variant_idx, variant in enumerate(variant_order):
            variant_rows = panel_spec["rows"].loc[panel_spec["rows"]["variant"] == variant].copy()
            variant_rows = variant_rows.sort_values(panel_spec["value_col"], kind="mergesort")
            yvals = variant_rows[panel_spec["value_col"]].to_numpy(dtype=float)
            xvals = variant_x[variant_idx] + _centered_jitter_offsets(
                yvals.size,
                scatter_span,
                scatter_seed + panel_idx * 100 + variant_idx,
            )
            ax.scatter(
                xvals,
                yvals,
                color=color,
                edgecolors=point_edge,
                linewidths=point_edge_linewidth,
                s=scatter_size,
                alpha=scatter_alpha,
                zorder=2,
            )
        for variant, text in annotations.get(panel_spec["panel_key"], {}).items():
            if variant not in variant_order:
                continue
            variant_idx = variant_order.index(variant)
            mean_series = variant_means.loc[variant_means["variant"] == variant, panel_spec["mean_col"]]
            if mean_series.empty:
                continue
            ax.annotate(
                str(text),
                xy=(variant_x[variant_idx], float(mean_series.iloc[0])),
                xytext=(0, annotation_offset_points),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=annotation_fontsize,
                fontweight=annotation_fontweight,
                color=annotation_color,
                clip_on=False,
                zorder=3,
            )
        ax.set_xticks(variant_x, variant_order)
        ax.set_title(panel_spec["title"])
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel(labels.get("ylabel", "Aligned mean correlation"))

    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    else:
        group_values: dict[object, list[np.ndarray]] = {}
        for panel_spec in resolved_panel_specs:
            group_key = panel_spec.get("sharey_group")
            group_values.setdefault(group_key, [])
            group_values[group_key].append(panel_spec["variant_means"][panel_spec["mean_col"]].to_numpy(dtype=float))
            group_values[group_key].append(
                pd.to_numeric(panel_spec["rows"][panel_spec["value_col"]], errors="coerce").to_numpy(dtype=float)
            )
        for ax, panel_spec in zip(axes, resolved_panel_specs, strict=False):
            group_key = panel_spec.get("sharey_group")
            if group_key not in group_values:
                continue
            finite_values = np.concatenate(group_values[group_key])
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                ymin = float(np.min(finite_values))
                ymax = float(np.max(finite_values))
                yrange = max(ymax - ymin, 1e-6)
                top_pad = float(style.get("ylim_top_pad_fraction", 0.16)) * yrange
                bottom_pad = float(style.get("ylim_bottom_pad_fraction", 0.05)) * yrange
                ax.set_ylim(ymin - bottom_pad, ymax + top_pad)

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, saved_paths


def _prepare_level_code_heatmap_data(
    *,
    data_roots=None,
    plot_ready_csv=None,
    base_groups: dict[str, list[tuple[str, str]]] | None = None,
    variant: str = "d1v1",
    pair_order: list[tuple[str, str]] | None = None,
    pattern_template: str = "tuning_{src}_to_{dst}.pkl",
    workers: int = 8,
    style: dict | None = None,
):
    style = {} if style is None else dict(style)
    resolved_data_roots = (
        []
        if data_roots is None
        else list(data_roots)
    )
    resolved_groups = deepcopy(
        DEFAULT_LEVEL_CODE_DAY1_DAY1_GROUPS if base_groups is None else base_groups
    )
    variant_groups = sl.build_variant_groups(resolved_groups)
    if variant not in variant_groups:
        raise KeyError(f"Unknown level-code variant: {variant!r}")
    if pair_order is None:
        requested_pair_keys = [
            sl.canonical_pair_key(pair)
            for group_pairs in variant_groups[variant].values()
            for pair in group_pairs
        ]
    else:
        requested_pair_keys = [sl.canonical_pair_key(pair) for pair in pair_order]

    if plot_ready_csv is None:
        registry = sl.load_cross_session_variants(
            base_groups=resolved_groups,
            data_roots=resolved_data_roots,
            pattern_template=pattern_template,
            workers=int(workers),
        )
        train_maps = collect_var_from_variant(registry, variant, "train_map", which="data")
        test_maps = collect_var_from_variant(registry, variant, "test_map", which="data")
        similarities = collect_var_from_variant(registry, variant, "similarity", which="data")
    else:
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
            int(column.removeprefix("train_level_"))
            for column in rows.columns
            if column.startswith("train_level_")
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
                    sl.canonical_pair_key((src, dst))
                    for src, dst in zip(
                        rows["src_session"].astype(str),
                        rows["dst_session"].astype(str),
                        strict=False,
                    )
                ],
                index=rows.index,
            )
        else:
            pair_key_values = rows["pair_key"].astype(str).str.replace(
                "->", "→", regex=False
            )
        for key in requested_pair_keys:
            pair_rows = rows.loc[pair_key_values.eq(key)]
            if pair_rows.empty:
                continue
            train_maps[key] = pair_rows[
                [f"train_level_{index}" for index in level_indices]
            ].to_numpy(dtype=float).T
            test_maps[key] = pair_rows[
                [f"test_level_{index}" for index in level_indices]
            ].to_numpy(dtype=float).T
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
            f"No level-code train/test maps were found for variant={variant!r} in {resolved_data_roots!r}."
        )
    if train_maps.keys() != test_maps.keys():
        raise ValueError("Level-code train_map and test_map keys did not match.")

    # Freeze pair ordering before stacking so the clustering input is stable.
    # Some legacy figures also depend on a specific historical pre-clustering
    # pair order to break ties consistently during hierarchical ordering.
    missing_train_keys = [key for key in requested_pair_keys if key not in train_maps]
    missing_test_keys = [key for key in requested_pair_keys if key not in test_maps]
    missing_similarity_keys = [key for key in requested_pair_keys if key not in similarities]
    if missing_train_keys or missing_test_keys or missing_similarity_keys:
        raise ValueError(
            "Level-code heatmap inputs did not contain the full requested pair set. "
            f"missing_train={missing_train_keys!r}, "
            f"missing_test={missing_test_keys!r}, "
            f"missing_similarity={missing_similarity_keys!r}"
        )
    train_maps = {key: train_maps[key] for key in requested_pair_keys}
    test_maps = {key: test_maps[key] for key in requested_pair_keys}
    similarities = {key: similarities[key] for key in requested_pair_keys}

    n_levels = int(np.asarray(next(iter(train_maps.values()))).shape[0])
    train_maps_stacked, pair_slices = stack_level_maps(
        train_maps, return_slices=True, expected_n_levels=n_levels
    )
    test_maps_stacked, _ = stack_level_maps(
        test_maps, return_slices=True, expected_n_levels=n_levels
    )
    similarities_original = np.concatenate(list(similarities.values())) if similarities else None
    pair_slices_original = dict(pair_slices)
    similarities_for_order = similarities_original
    nonfinite_similarity_count = 0
    retained_similarity_mask = None
    nonfinite_similarity_counts_by_pair = {
        key: 0 for key in pair_slices_original
    }
    retained_neuron_counts_by_pair = {
        key: int(end - start) for key, (start, end) in pair_slices_original.items()
    }
    if similarities_original is not None:
        finite_similarity_mask = np.isfinite(similarities_original)
        retained_similarity_mask = finite_similarity_mask.astype(bool, copy=True)
        nonfinite_similarity_count = int(
            np.size(similarities_original) - np.sum(retained_similarity_mask)
        )
        for key, (start, end) in pair_slices_original.items():
            pair_mask = retained_similarity_mask[start:end]
            retained_neuron_counts_by_pair[key] = int(np.sum(pair_mask))
            nonfinite_similarity_counts_by_pair[key] = int(pair_mask.size - np.sum(pair_mask))
        if nonfinite_similarity_count:
            if not np.any(retained_similarity_mask):
                raise ValueError(
                    "All level-code similarity entries were non-finite; "
                    "cannot build a filtered neuron ordering."
                )
            train_maps_stacked = train_maps_stacked[:, retained_similarity_mask]
            test_maps_stacked = test_maps_stacked[:, retained_similarity_mask]
            similarities_for_order = similarities_original[retained_similarity_mask]

            filtered_pair_slices: dict[str, tuple[int, int]] = {}
            next_start = 0
            for key, (start, end) in pair_slices_original.items():
                n_keep = retained_neuron_counts_by_pair[key]
                filtered_pair_slices[key] = (next_start, next_start + n_keep)
                next_start += n_keep
            pair_slices = filtered_pair_slices

    normalize_mode = str(style.get("normalize_mode", "l2"))
    train_maps_norm = normalize_maps(train_maps_stacked, mode=normalize_mode)
    test_maps_norm = normalize_maps(test_maps_stacked, mode=normalize_mode)
    refined_order_path = style.get("refined_order_path")
    if refined_order_path is not None:
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
                    "Precomputed refined level-code order hash mismatch: "
                    f"expected {expected_order_sha256}, observed {actual_order_sha256}"
                )
        refined_order = np.asarray(
            np.load(refined_order_path, allow_pickle=False), dtype=int
        ).reshape(-1)
        expected_indices = np.arange(train_maps_norm.shape[1], dtype=int)
        if (
            refined_order.shape != expected_indices.shape
            or not np.array_equal(np.sort(refined_order), expected_indices)
        ):
            raise ValueError(
                "Precomputed refined level-code order must be an exact permutation "
                f"of 0..{train_maps_norm.shape[1] - 1}."
            )
        ordering_method = "precomputed_refined_order"
        order_hier = None
    else:
        ordering_method = str(style.get("ordering_method", "hierarchical")).strip().lower()
    if refined_order_path is None and ordering_method == "hierarchical":
        # Preserve the legacy renderer for smaller exploratory pools without
        # imposing its large import cost on the scalable publication path.
        from .level_ordering import order_hier_shape_index

        order_hier = order_hier_shape_index(
            train_maps_stacked,
            similarity=similarities_for_order,
            lam_index=float(style.get("lam_index", 1.0)),
            lam_sim=float(style.get("lam_sim", 1.0)),
        )
    elif refined_order_path is None and ordering_method == "preferred_level_similarity":
        # The legacy hierarchical path materializes an N x N distance matrix.
        # That is inappropriate for the maintained 16,901-neuron Figure 5 pool.
        # Use a stable one-dimensional base order by cross-session similarity,
        # then group rows by their preferred level below. This changes display
        # seriation only; maps, neuron support, and every statistic are intact.
        if similarities_for_order is None:
            order_hier = np.arange(train_maps_norm.shape[1], dtype=int)
        else:
            similarity_values = np.asarray(similarities_for_order, dtype=float)
            order_hier = np.argsort(similarity_values, kind="stable")
    elif refined_order_path is None and ordering_method == "preferred_level_hierarchical":
        from .level_ordering import order_hier_shape_index

        preferred_levels = np.argmax(train_maps_norm, axis=0)
        ordered_groups = []
        for level_index in range(train_maps_norm.shape[0]):
            group_indices = np.flatnonzero(preferred_levels == level_index)
            if group_indices.size <= 1:
                ordered_groups.append(group_indices)
                continue
            group_similarity = (
                None
                if similarities_for_order is None
                else np.asarray(similarities_for_order, dtype=float)[group_indices]
            )
            group_order = order_hier_shape_index(
                train_maps_stacked[:, group_indices],
                similarity=group_similarity,
                lam_index=float(style.get("lam_index", 1.0)),
                lam_sim=float(style.get("lam_sim", 1.0)),
            )
            ordered_groups.append(group_indices[group_order])
        order_hier = np.concatenate(ordered_groups)
    elif refined_order_path is None and ordering_method == "input_order":
        order_hier = np.arange(train_maps_norm.shape[1], dtype=int)
    elif refined_order_path is None:
        raise ValueError(
            "ordering_method must be 'hierarchical', 'preferred_level_similarity', "
            "'preferred_level_hierarchical', or 'input_order'"
        )
    if refined_order_path is None:
        refined_order = refine_order_by_argmax(
            train_maps_norm,
            base_order=order_hier,
            eps=float(style.get("refine_eps", 1e-6)),
        )
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
    data_roots=None,
    plot_ready_csv=None,
    base_groups: dict[str, list[tuple[str, str]]] | None = None,
    variant: str = "d1v1",
    pair_order: list[tuple[str, str]] | None = None,
    pattern_template: str = "tuning_{src}_to_{dst}.pkl",
    workers: int = 8,
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
        data_roots=data_roots,
        plot_ready_csv=plot_ready_csv,
        base_groups=base_groups,
        variant=variant,
        pair_order=pair_order,
        pattern_template=pattern_template,
        workers=workers,
        style=style,
    )

    fig, axes = plt.subplots(
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
    heatmap_kwargs = {
        "aspect": style.get("aspect", "auto"),
        "cmap": heatmap_cmap,
    }
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
    for idx, ax in enumerate(axes):
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

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def _select_linearized_matched_place_fields(pair_rows, neuron_set: str):
    pf1_blocks = []
    pf2_blocks = []
    for pair_row in pair_rows:
        stable_mask = np.asarray(pair_row["stable"], dtype=bool)
        if neuron_set == "stable":
            mask = stable_mask
        elif neuron_set == "unstable":
            mask = ~stable_mask
        else:
            mask = np.ones(np.asarray(pair_row["pf1"]).shape[0], dtype=bool)
        if not np.any(mask):
            continue
        pf1_blocks.append(np.asarray(pair_row["pf1"], dtype=float)[mask])
        pf2_blocks.append(np.asarray(pair_row["pf2"], dtype=float)[mask])
    if not pf1_blocks:
        raise ValueError("No matched place-field rows were available for the requested neuron set.")
    return np.vstack(pf1_blocks), np.vstack(pf2_blocks)


def _order_linearized_heatmaps_by_earlier_peak(pf1_all, pf2_all, max_display=None):
    order = np.argsort(np.nanargmax(pf1_all, axis=1))[::-1]
    if max_display is not None and int(max_display) > 0 and len(order) > int(max_display):
        order = order[np.linspace(0, len(order) - 1, int(max_display), dtype=int)]
    return pf1_all[order], pf2_all[order]


def _prepare_linearized_matched_panel_data(
    *,
    data_root=None,
    run_label: str = "home_to_water",
    stable_threshold: float = STABLE_THR,
    pair_filter: str = "day1_day1",
    heatmap_neuron_set: str = "all",
    max_display=None,
    style: dict | None = None,
):
    if heatmap_neuron_set not in {"all", "stable", "unstable"}:
        raise ValueError("heatmap_neuron_set must be one of: 'all', 'stable', 'unstable'.")

    resolved_data_root = Path(get_data_root() if data_root is None else data_root)
    style = {} if style is None else dict(style)

    pair_rows, n_bins = load_pair_data(
        str(resolved_data_root),
        run_label,
        stable_thr=float(stable_threshold),
        include_place_fields=True,
        pair_filter=pair_filter,
    )
    exclude_pair_keys = style.get("exclude_pair_keys", ())
    if exclude_pair_keys:
        excluded_pairs: set[tuple[str, str]] = set()
        for pair in exclude_pair_keys:
            if isinstance(pair, str):
                if "->" in pair:
                    src, dst = pair.split("->", 1)
                elif "_to_" in pair:
                    src, dst = pair.split("_to_", 1)
                else:
                    continue
            else:
                src, dst = pair
            excluded_pairs.add((str(src), str(dst)))
        pair_rows = [
            row
            for row in pair_rows
            if (str(row.get("id1", "")), str(row.get("id2", ""))) not in excluded_pairs
        ]
    if not pair_rows:
        raise ValueError(f"No matched precomputed pair rows were found for run_label={run_label!r}.")

    pf1_all, pf2_all = _select_linearized_matched_place_fields(pair_rows, heatmap_neuron_set)
    pf1_sorted, pf2_sorted = _order_linearized_heatmaps_by_earlier_peak(
        pf1_all,
        pf2_all,
        max_display=max_display,
    )
    display_nan_as_zero = bool(style.get("display_nan_as_zero", True))
    nan_fill_value = float(style.get("nan_fill_value", 0.0))
    if display_nan_as_zero:
        pf1_display = np.nan_to_num(pf1_sorted, nan=nan_fill_value)
        pf2_display = np.nan_to_num(pf2_sorted, nan=nan_fill_value)
    else:
        pf1_display = pf1_sorted
        pf2_display = pf2_sorted

    stable_peak_blocks = [
        np.asarray(pair_row["stable_peaks"], dtype=int)
        for pair_row in pair_rows
        if np.asarray(pair_row["stable_peaks"]).size > 0
    ]
    unstable_peak_blocks = [
        np.asarray(pair_row["unstable_peaks"], dtype=int)
        for pair_row in pair_rows
        if np.asarray(pair_row["unstable_peaks"]).size > 0
    ]
    if not stable_peak_blocks or not unstable_peak_blocks:
        raise ValueError("Stable and unstable peak arrays are required for the density comparison.")
    stable_peaks = np.concatenate(stable_peak_blocks)
    unstable_peaks = np.concatenate(unstable_peak_blocks)

    stable_fraction = np.bincount(stable_peaks, minlength=n_bins).astype(float)
    stable_fraction /= stable_fraction.sum()
    unstable_fraction = np.bincount(unstable_peaks, minlength=n_bins).astype(float)
    unstable_fraction /= unstable_fraction.sum()

    density_sigma = float(style.get("density_sigma", 1.5))
    stable_density = gaussian_filter1d(stable_fraction, sigma=density_sigma, mode="nearest")
    unstable_density = gaussian_filter1d(unstable_fraction, sigma=density_sigma, mode="nearest")
    if bool(style.get("reverse_position_axis", False)):
        pf1_display = np.flip(pf1_display, axis=1)
        pf2_display = np.flip(pf2_display, axis=1)
        stable_density = stable_density[::-1]
        unstable_density = unstable_density[::-1]

    vmax = np.nanpercentile(
        np.concatenate([pf1_sorted.ravel(), pf2_sorted.ravel()]),
        float(style.get("heatmap_percentile_vmax", 99.0)),
    )
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    return {
        "n_bins": n_bins,
        "pf1_sorted": pf1_sorted,
        "pf2_sorted": pf2_sorted,
        "pf1_display": pf1_display,
        "pf2_display": pf2_display,
        "stable_peaks": stable_peaks,
        "unstable_peaks": unstable_peaks,
        "stable_density": stable_density,
        "unstable_density": unstable_density,
        "vmax": float(vmax),
    }


def draw_linearized_matched_panel_axes(
    ax_early,
    ax_late,
    ax_density,
    *,
    colorbar_ax=None,
    data_root=None,
    run_label: str = "home_to_water",
    stable_threshold: float = STABLE_THR,
    pair_filter: str = "day1_day1",
    heatmap_neuron_set: str = "all",
    max_display=None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    xticks=None,
):
    colors = {} if colors is None else dict(colors)
    style = {} if style is None else dict(style)
    labels = {} if labels is None else dict(labels)
    xticks = list(xticks) if xticks is not None else None

    panel_data = _prepare_linearized_matched_panel_data(
        data_root=data_root,
        run_label=run_label,
        stable_threshold=stable_threshold,
        pair_filter=pair_filter,
        heatmap_neuron_set=heatmap_neuron_set,
        max_display=max_display,
        style=style,
    )

    display_vmax = style.get("heatmap_vmax")
    if display_vmax is None:
        display_vmax = float(panel_data["vmax"])
    else:
        display_vmax = float(display_vmax)
        if not np.isfinite(display_vmax) or display_vmax <= 0:
            raise ValueError("heatmap_vmax must be a positive finite number")
    heatmap_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "cmap": colors.get("heatmap", "viridis"),
        "vmin": 0,
        "vmax": display_vmax,
    }
    im_early = ax_early.imshow(panel_data["pf1_display"], **heatmap_kwargs)
    im_late = ax_late.imshow(panel_data["pf2_display"], **heatmap_kwargs)

    n_bins = int(panel_data["n_bins"])
    axis_xticks = xticks if xticks is not None else [0, 5, 10, 15, 20, n_bins - 1]
    ax_early.set_title(labels.get("early", "3 months ago") + f" (n={len(panel_data['pf1_sorted'])} matched)")
    ax_late.set_title(labels.get("late", "3 months later"))
    ax_early.set_xlabel(labels.get("heatmap_xlabel", "Linearized position (0=home, 21=water port)"))
    ax_late.set_xlabel(labels.get("heatmap_xlabel", "Linearized position (0=home, 21=water port)"))
    ax_early.set_ylabel(labels.get("heatmap_ylabel", "Coregistered neurons (sorted by earlier peak)"))
    ax_late.tick_params(labelleft=False)
    for heatmap_ax in (ax_early, ax_late):
        heatmap_ax.set_xticks(axis_xticks)
    if bool(style.get("reverse_position_axis", False)):
        reversed_tick_labels = [str(tick) for tick in reversed(axis_xticks)]
        for heatmap_ax in (ax_early, ax_late):
            heatmap_ax.set_xticklabels(reversed_tick_labels)

    colorbar_location = style.get("colorbar_location", "bottom")
    colorbar_orientation = style.get("colorbar_orientation")
    if colorbar_orientation is None:
        colorbar_orientation = "horizontal" if colorbar_location in {"top", "bottom"} else "vertical"
    show_colorbar = bool(style.get("show_colorbar", True))
    if show_colorbar and colorbar_ax is not None:
        ax_early.figure.colorbar(
            im_late,
            cax=colorbar_ax,
            orientation=colorbar_orientation,
            label=labels.get("colorbar_label", "Normalized rate"),
        )
    elif show_colorbar:
        ax_early.figure.colorbar(
            im_late,
            ax=[ax_early, ax_late],
            location=colorbar_location,
            orientation=colorbar_orientation,
            fraction=float(style.get("colorbar_fraction", 0.046)),
            pad=float(style.get("colorbar_pad", 0.025)),
            label=labels.get("colorbar_label", "Normalized rate"),
        )

    peak_bins = np.arange(n_bins)
    unstable_color = colors.get("unstable", "gray")
    stable_color = colors.get("stable", "salmon")
    density_alpha = float(style.get("density_alpha", 0.22))
    density_linewidth = float(style.get("density_linewidth", 2.5))
    ax_density.fill_betweenx(
        peak_bins,
        0.0,
        color=unstable_color,
        x2=panel_data["unstable_density"],
        alpha=density_alpha,
        label=f"Unstable (n={len(panel_data['unstable_peaks'])})",
    )
    ax_density.plot(panel_data["unstable_density"], peak_bins, color=unstable_color, linewidth=density_linewidth)
    ax_density.fill_betweenx(
        peak_bins,
        0.0,
        color=stable_color,
        x2=panel_data["stable_density"],
        alpha=density_alpha,
        label=f"Stable (corr >= {float(stable_threshold):.1f}, n={len(panel_data['stable_peaks'])})",
    )
    ax_density.plot(panel_data["stable_density"], peak_bins, color=stable_color, linewidth=density_linewidth)
    if bool(style.get("show_water_port_band", False)):
        if bool(style.get("reverse_position_axis", False)):
            band_ymin = -0.5
            band_ymax = WP_BINS - 0.5
        else:
            band_ymin = n_bins - WP_BINS - 0.5
            band_ymax = n_bins - 0.5
        ax_density.axhspan(
            band_ymin,
            band_ymax,
            color=colors.get("water_port_band", "#d95f02"),
            alpha=float(style.get("water_port_band_alpha", 0.08)),
        )
    ax_density.set_title(labels.get("density_title", "Smoothed peak density"))
    ax_density.set_xlabel(labels.get("density_xlabel", "Density"))
    ax_density.set_ylabel(labels.get("density_ylabel", "Peak bin (0=home, 21=water port)"))
    density_xlim_max = style.get("density_xlim_max")
    if density_xlim_max is None:
        ax_density.set_xlim(left=0)
    else:
        ax_density.set_xlim(0.0, float(density_xlim_max))
    ax_density.set_ylim(-0.5, n_bins - 0.5)
    ax_density.set_yticks(axis_xticks)
    ax_density.legend(frameon=False, loc=style.get("legend_loc", "upper left"))
    ax_density.grid(alpha=float(style.get("grid_alpha", 0.2)), axis="x")

    return {
        "earlier": ax_early,
        "later": ax_late,
        "density": ax_density,
        "colorbar": colorbar_ax,
        "heatmap_vmax": display_vmax,
        "source_heatmap_vmax": float(panel_data["vmax"]),
    }


def render_linearized_home_to_water_matched_panel(
    *,
    data_root=None,
    run_label: str = "home_to_water",
    stable_threshold: float = STABLE_THR,
    pair_filter: str = "day1_day1",
    heatmap_neuron_set: str = "all",
    max_display=None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    xticks=None,
    save: bool = False,
    panel_name: str = "linearized_home_to_water_matched_heatmaps",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render the precomputed home-to-water matched heatmaps plus density summary.

    This helper reads saved matched arrays (`__pf1_matched.npy`, `__pf2_matched.npy`,
    and `__corrs.npy`) from the linearized place-field export directory. It does not
    rerun trajectory linearization or place-field computation.
    """
    panel_layout = get_panel_layout(layout)
    style = {} if style is None else dict(style)
    fig = plt.figure(figsize=panel_size or panel_layout["panel_size"])
    width_ratios = tuple(panel_layout.get("width_ratios", (1.0, 1.0, 0.92)))
    use_bottom_colorbar_row = style.get("colorbar_location", "bottom") == "bottom" and bool(
        style.get("dedicated_colorbar_axis", True)
    )
    if use_bottom_colorbar_row:
        grid = fig.add_gridspec(
            2,
            3,
            width_ratios=width_ratios,
            height_ratios=(1.0, float(style.get("colorbar_height_ratio", 0.08))),
            wspace=float(panel_layout.get("grid_wspace", 0.18)),
            hspace=float(style.get("colorbar_hspace", 0.08)),
        )
        ax_early = fig.add_subplot(grid[0, 0])
        ax_late = fig.add_subplot(grid[0, 1], sharey=ax_early)
        ax_density = fig.add_subplot(grid[0, 2])
        colorbar_ax = fig.add_subplot(grid[1, 0:2])
    else:
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=width_ratios,
            wspace=float(panel_layout.get("grid_wspace", 0.18)),
        )
        ax_early = fig.add_subplot(grid[0, 0])
        ax_late = fig.add_subplot(grid[0, 1], sharey=ax_early)
        ax_density = fig.add_subplot(grid[0, 2])
        colorbar_ax = None

    axes_dict = draw_linearized_matched_panel_axes(
        ax_early,
        ax_late,
        ax_density,
        colorbar_ax=colorbar_ax,
        data_root=data_root,
        run_label=run_label,
        stable_threshold=stable_threshold,
        pair_filter=pair_filter,
        heatmap_neuron_set=heatmap_neuron_set,
        max_display=max_display,
        colors=colors,
        style=style,
        labels=labels,
        xticks=xticks,
    )

    saved_paths = (
        save_panel(fig, panel_name, output_dir=output_dir, formats=formats)
        if save
        else []
    )
    return fig, axes_dict, saved_paths


def render_cycle_diagram_panel(
    *,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    boxes: dict | None = None,
    loop: dict | None = None,
    arrows: list[dict] | None = None,
    text_specs: list[dict] | None = None,
    save: bool = False,
    panel_name: str = "cycle_diagram",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    cycle_layout = get_cycle_diagram_layout(layout)
    if panel_size is not None:
        cycle_layout["panel_size"] = tuple(panel_size)
    cycle_colors = get_cycle_diagram_colors(colors)
    cycle_style = get_cycle_diagram_style(style, panel_size=cycle_layout["panel_size"], layout=cycle_layout)
    box_specs = get_cycle_diagram_boxes(colors=cycle_colors) if boxes is None else deepcopy(boxes)
    loop_spec = get_cycle_diagram_loop() if loop is None else deepcopy(loop)
    arrow_specs = get_cycle_diagram_arrows() if arrows is None else deepcopy(arrows)
    rendered_text_specs = (
        get_cycle_diagram_text_specs(
            colors=cycle_colors,
            panel_size=cycle_layout["panel_size"],
            layout=cycle_layout,
        )
        if text_specs is None
        else deepcopy(text_specs)
    )

    fig, ax = make_panel_figure(panel_size=cycle_layout["panel_size"], layout=cycle_layout)
    fig.patch.set_facecolor(cycle_colors["background"])
    ax.set_facecolor(cycle_colors["background"])
    ax.set_xlim(0, cycle_layout["coordinate_width"])
    ax.set_ylim(cycle_layout["coordinate_height"], 0)
    ax.axis("off")

    for box_spec in box_specs.values():
        ax.add_patch(
            FancyBboxPatch(
                box_spec["xy"],
                box_spec["width"],
                box_spec["height"],
                boxstyle=f"round,pad=0.02,rounding_size={cycle_style['rounding_size']}",
                linewidth=cycle_style["box_linewidth"],
                edgecolor=box_spec["edgecolor"],
                facecolor=box_spec["facecolor"],
            )
        )

    ax.plot(
        loop_spec["x"],
        loop_spec["y"],
        color=cycle_colors["line"],
        lw=cycle_style["loop_linewidth"],
        solid_capstyle="butt",
        solid_joinstyle="miter",
        antialiased=True,
        zorder=1,
    )

    for arrow_spec in arrow_specs:
        ax.add_patch(
            FancyArrowPatch(
                arrow_spec["start"],
                arrow_spec["end"],
                arrowstyle="-|>",
                mutation_scale=cycle_style["arrow_mutation_scale"],
                linewidth=cycle_style["arrow_linewidth"],
                color=cycle_colors["line"],
                shrinkA=0,
                shrinkB=0,
                capstyle="butt",
                joinstyle="miter",
                zorder=3,
            )
        )

    for text_spec in rendered_text_specs:
        text_kwargs = {
            "ha": "center",
            "va": "center",
            "fontsize": text_spec["fontsize"],
            "color": text_spec["color"],
        }
        if "fontweight" in text_spec:
            text_kwargs["fontweight"] = text_spec["fontweight"]
        ax.text(text_spec["x"], text_spec["y"], text_spec["text"], **text_kwargs)

    plt.tight_layout(pad=0)
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, saved_paths


def render_turning_bias_three_group_panel(
    *,
    csv_path,
    fraction: str = "full",
    x_col: str = "Ba",
    y_col: str = "Lo",
    reference_xy: tuple[float, float] = (0.5, 0.5),
    min_axis_support: int | None = None,
    support_columns: tuple[str, str] | None = None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "turning_bias_full_session_three_groups",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render the full-session three-group turning-bias scatter panel.

    This helper is designed to consume the CSV written by:

        python scripts/plot_legacy_turning_bias_scatter.py --group-preset three_groups

    The notebook can then stay configuration-driven while still using the same
    maintained legacy computation path for the source values. By default it
    renders the historical ``Ba``/``Lo`` scatter. Set ``x_col``, ``y_col``,
    labels, and ``reference_xy`` to render the same panel for other bias pairs.
    Set ``min_axis_support`` to require the corresponding ``n_<metric>``
    columns to meet a minimum support threshold before plotting.
    """
    panel_layout = {
        "panel_size": (6.8, 3.4),
        "grid_wspace": 0.32,
    }
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        "naive": "blue",
        "LTM": "red",
        "experienced": "green",
        "reference": "black",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "marker": "o",
        "marker_size": 5.0,
        "marker_alpha": 0.55,
        "reference_marker": "+",
        "reference_marker_size": 8.0,
        "reference_markeredgewidth": 1.5,
        "legend_loc": "lower left",
        "show_legend": True,
        "show_counts_in_legend": True,
        "equal_aspect": True,
        "xlim": (0.0, 1.0),
        "ylim": (0.0, 1.0),
    }
    provided_style_keys = set(style) if style else set()
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": r"$P_{\mathrm{SA}}$",
        "ylabel": r"$P_{\mathrm{BS}}$",
        "reference": "random",
        "left_title": "naive vs LTM",
        "right_title": "LTM vs experienced",
        "group_labels": {
            "naive": "naive",
            "LTM": "LTM",
            "experienced": "experienced",
        },
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(Path(csv_path))
    requested_fraction = str(fraction)
    requested_fraction_aliases = {requested_fraction}
    if requested_fraction == "full":
        requested_fraction_aliases.update({"1/1", "1", "full session"})
    if requested_fraction.startswith("first_1_"):
        denom = requested_fraction.replace("first_1_", "", 1)
        requested_fraction_aliases.add(f"1/{denom}")
    if requested_fraction.startswith("1/"):
        denom = requested_fraction.split("/", 1)[1]
        requested_fraction_aliases.add(f"first_1_{denom}")
    if "fraction" in points_df.columns:
        fraction_values = points_df["fraction"].astype(str).str.strip()
        points_df = points_df.loc[fraction_values.isin(requested_fraction_aliases)].copy()
        if points_df.empty and requested_fraction == "full" and "n_chunks" in points_df.columns:
            points_df = pd.read_csv(Path(csv_path))
            points_df = points_df.loc[pd.to_numeric(points_df["n_chunks"], errors="coerce") == 1].copy()
    elif requested_fraction == "full" and "n_chunks" in points_df.columns:
        points_df = points_df.loc[pd.to_numeric(points_df["n_chunks"], errors="coerce") == 1].copy()

    if points_df.empty:
        raise ValueError(f"No turning-bias rows found for fraction={fraction!r} in {csv_path!r}.")

    missing_columns = [col for col in (x_col, y_col) if col not in points_df.columns]
    if missing_columns:
        raise ValueError(f"Turning-bias CSV is missing required columns: {missing_columns}")

    if min_axis_support is not None:
        min_support = int(min_axis_support)
        if min_support < 0:
            raise ValueError("min_axis_support must be non-negative.")
        x_support_col, y_support_col = support_columns or (f"n_{x_col}", f"n_{y_col}")
        missing_support_columns = [
            col for col in (x_support_col, y_support_col) if col not in points_df.columns
        ]
        if missing_support_columns:
            raise ValueError(
                "Turning-bias CSV is missing required support columns: "
                f"{missing_support_columns}"
            )
        x_support = pd.to_numeric(points_df[x_support_col], errors="coerce")
        y_support = pd.to_numeric(points_df[y_support_col], errors="coerce")
        points_df = points_df.loc[(x_support >= min_support) & (y_support >= min_support)].copy()
        if points_df.empty:
            raise ValueError(
                "No turning-bias rows remained after support filtering "
                f"({x_support_col}, {y_support_col} >= {min_support})."
            )

    required_groups = {"naive", "LTM", "experienced"}
    missing_groups = required_groups.difference(set(points_df["group"].astype(str).unique()))
    if missing_groups:
        raise ValueError(f"Turning-bias CSV is missing required groups: {sorted(missing_groups)}")

    fig, axes = plt.subplots(1, 2, figsize=panel_size or panel_layout["panel_size"], squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(wspace=float(panel_layout.get("grid_wspace", 0.32)))

    panels = [
        (axes[0], panel_labels.get("left_title", "naive vs LTM"), ("LTM", "naive")),
        (axes[1], panel_labels.get("right_title", "LTM vs experienced"), ("experienced", "LTM")),
    ]
    group_label_map = panel_labels.get("group_labels", {})
    xlim = tuple(panel_style.get("xlim", (0.0, 1.0)))
    ylim = tuple(panel_style.get("ylim", (0.0, 1.0)))

    for ax, title, groups in panels:
        for group_name in groups:
            sub = points_df.loc[points_df["group"].astype(str) == str(group_name)]
            display_name = str(group_label_map.get(group_name, group_name))
            legend_label = (
                f"{display_name} (n={len(sub)})"
                if bool(panel_style.get("show_counts_in_legend", True))
                else display_name
            )
            ax.plot(
                sub[x_col],
                sub[y_col],
                linestyle="None",
                marker=str(panel_style.get("marker", "o")),
                color=panel_colors[group_name],
                markersize=float(panel_style.get("marker_size", 5.0)),
                alpha=float(panel_style.get("marker_alpha", 0.55)),
                label=legend_label,
            )

        ax.plot(
            [float(reference_xy[0])],
            [float(reference_xy[1])],
            linestyle="None",
            marker=str(panel_style.get("reference_marker", "+")),
            color=panel_colors.get("reference", "black"),
            markersize=float(panel_style.get("reference_marker_size", 8.0)),
            markeredgewidth=float(panel_style.get("reference_markeredgewidth", 1.5)),
            label=str(panel_labels.get("reference", "random")),
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if bool(panel_style.get("equal_aspect", True)):
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(panel_labels.get("xlabel", r"$P_{\mathrm{SA}}$"))
        ax.set_ylabel(panel_labels.get("ylabel", r"$P_{\mathrm{BS}}$"))
        ax.set_title(str(title))
        if bool(panel_style.get("show_legend", True)):
            ax.legend(frameon=False, loc=str(panel_style.get("legend_loc", "lower left")))

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, points_df, saved_paths


def _filter_turning_bias_fraction(points_df: pd.DataFrame, fraction: str) -> pd.DataFrame:
    requested_fraction = str(fraction).strip()
    requested_fraction_aliases = {requested_fraction}
    if requested_fraction == "full":
        requested_fraction_aliases.update({"1/1", "1", "full session"})
    if requested_fraction.startswith("first_1_"):
        denom = requested_fraction.replace("first_1_", "", 1)
        requested_fraction_aliases.add(f"1/{denom}")
    if requested_fraction.startswith("1/"):
        denom = requested_fraction.split("/", 1)[1]
        requested_fraction_aliases.add(f"first_1_{denom}")

    if "fraction" in points_df.columns:
        fraction_values = points_df["fraction"].astype(str).str.strip()
        filtered = points_df.loc[fraction_values.isin(requested_fraction_aliases)].copy()
        if filtered.empty and requested_fraction == "full" and "n_chunks" in points_df.columns:
            filtered = points_df.loc[
                pd.to_numeric(points_df["n_chunks"], errors="coerce") == 1
            ].copy()
        return filtered
    if requested_fraction == "full" and "n_chunks" in points_df.columns:
        return points_df.loc[pd.to_numeric(points_df["n_chunks"], errors="coerce") == 1].copy()
    return points_df.copy()


def _default_turning_bias_four_bias_metric_pairs() -> list[dict]:
    return [
        {
            "x_col": "Bf",
            "y_col": "Lf",
            "xlabel": r"$P_{\mathrm{SF}}$",
            "ylabel": r"$P_{\mathrm{BF}}$",
            "title": "",
            "reference_xy": (2.0 / 3.0, 2.0 / 3.0),
        },
        {
            "x_col": "Ba",
            "y_col": "Lo",
            "xlabel": r"$P_{\mathrm{SA}}$",
            "ylabel": r"$P_{\mathrm{BS}}$",
            "title": "",
            "reference_xy": (0.5, 0.5),
        },
    ]


def prepare_turning_bias_four_bias_points(
    *,
    csv_path,
    fraction: str = "1/32",
    groups: tuple[str, ...] = ("naive", "LTM"),
    metric_pairs: list[dict] | None = None,
    min_axis_support: int | None = 10,
):
    """Load and support-filter the four-bias Panel F source rows.

    Returns ``(candidate_points, kept_points, excluded_points, metric_pairs,
    support_columns)`` so the notebook can keep the plotted rows and the
    support-filter audit together.
    """
    rendered_metric_pairs = deepcopy(metric_pairs) if metric_pairs is not None else (
        _default_turning_bias_four_bias_metric_pairs()
    )
    points_df = pd.read_csv(Path(csv_path))
    points_df = _filter_turning_bias_fraction(points_df, fraction)
    if points_df.empty:
        raise ValueError(f"No turning-bias rows found for fraction={fraction!r} in {csv_path!r}.")

    group_values = points_df["group"].astype(str)
    requested_groups = [str(group) for group in groups]
    points_df = points_df.loc[group_values.isin(requested_groups)].copy()
    if points_df.empty:
        raise ValueError(f"No turning-bias rows found for groups={groups!r}.")

    missing_groups = set(requested_groups).difference(set(points_df["group"].astype(str)))
    if missing_groups:
        raise ValueError(f"Turning-bias CSV is missing required groups: {sorted(missing_groups)}")

    required_columns = []
    support_columns = []
    for pair in rendered_metric_pairs:
        required_columns.extend([str(pair["x_col"]), str(pair["y_col"])])
        support_columns.extend(
            [
                str(pair.get("x_support_col", f"n_{pair['x_col']}")),
                str(pair.get("y_support_col", f"n_{pair['y_col']}")),
            ]
        )
    required_columns = list(dict.fromkeys(required_columns))
    support_columns = list(dict.fromkeys(support_columns))

    missing_columns = [col for col in required_columns if col not in points_df.columns]
    if missing_columns:
        raise ValueError(f"Turning-bias CSV is missing required columns: {missing_columns}")

    candidate_points_df = points_df.copy()
    excluded_points_df = points_df.iloc[0:0].copy()
    if min_axis_support is not None:
        min_support = int(min_axis_support)
        if min_support < 0:
            raise ValueError("min_axis_support must be non-negative.")
        missing_support_columns = [col for col in support_columns if col not in points_df.columns]
        if missing_support_columns:
            raise ValueError(
                "Turning-bias CSV is missing required support columns: "
                f"{missing_support_columns}"
            )

        support_values = points_df[support_columns].apply(pd.to_numeric, errors="coerce")
        keep = (support_values >= min_support).all(axis=1).to_numpy()
        min_support_by_row = support_values.min(axis=1)
        excluded_points_df = points_df.loc[~keep].copy()
        if not excluded_points_df.empty:
            excluded_points_df["min_display_support"] = min_support_by_row.loc[~keep].to_numpy()
            excluded_points_df["support_threshold"] = min_support
            excluded_points_df["exclusion_reason"] = (
                f"any of {', '.join(support_columns)} < {min_support}"
            )
        points_df = points_df.loc[keep].copy()
        if points_df.empty:
            raise ValueError(
                "No turning-bias rows remained after support filtering "
                f"({support_columns} >= {min_support})."
            )

    return candidate_points_df, points_df, excluded_points_df, rendered_metric_pairs, support_columns


def render_turning_bias_four_bias_panel(
    *,
    csv_path,
    fraction: str = "1/32",
    groups: tuple[str, ...] = ("naive", "LTM"),
    metric_pairs: list[dict] | None = None,
    min_axis_support: int | None = 10,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "turning_bias_four_bias_first_1_32",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a two-axis four-bias turning panel for a selected chunk fraction.

    The first axis uses the historical ``Bf``/``Lf`` columns labelled as
    ``P_SF``/``P_BF``. The second uses ``Ba``/``Lo`` labelled as
    ``P_SA``/``P_BS``. By default this shows only naive and LTM Day 1 rows,
    leaving the old Day 2 comparison to ``render_turning_bias_three_group_panel``.
    """
    panel_layout = {
        "panel_size": (6.2, 3.1),
        "grid_wspace": 0.34,
    }
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        "naive": "#1f77b4",
        "LTM": "#d62728",
        "reference": "black",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "marker": "o",
        "marker_size": 4.4,
        "marker_alpha": 0.58,
        "reference_marker": "+",
        "reference_marker_size": 8.2,
        "reference_markeredgewidth": 1.5,
        "reference_halo": True,
        "reference_halo_marker_size": 10.0,
        "reference_halo_markeredgewidth": 3.0,
        "reference_halo_color": "white",
        "reference_linewidth": 0.55,
        "reference_alpha": 0.12,
        "show_reference_lines": True,
        "title_fontsize": 9.0,
        "axis_label_fontsize": 9.5,
        "tick_labelsize": 9.0,
        "legend_fontsize": 6.8,
        "legend_markerscale": 0.80,
        "legend_handletextpad": 0.35,
        "legend_labelspacing": 0.25,
        "legend_loc": "lower left",
        "show_legend": True,
        "legend_ax_index": 0,
        "show_counts_in_legend": False,
        "equal_aspect": True,
        "xlim": (0.0, 1.0),
        "ylim": (0.0, 1.0),
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "reference": "random",
        "group_labels": {
            "naive": "Naive day",
            "LTM": "Return day",
        },
    }
    if labels:
        panel_labels.update(labels)

    _, points_df, _, rendered_metric_pairs, _ = prepare_turning_bias_four_bias_points(
        csv_path=csv_path,
        fraction=fraction,
        groups=groups,
        metric_pairs=metric_pairs,
        min_axis_support=min_axis_support,
    )

    n_pairs = len(rendered_metric_pairs)
    fig, axes = plt.subplots(
        1,
        n_pairs,
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes.ravel()
    fig.subplots_adjust(wspace=float(panel_layout.get("grid_wspace", 0.34)))

    group_label_map = panel_labels.get("group_labels", {})
    xlim = tuple(panel_style.get("xlim", (0.0, 1.0)))
    ylim = tuple(panel_style.get("ylim", (0.0, 1.0)))
    reference_color = panel_colors.get("reference", "black")

    for ax_index, (ax, pair) in enumerate(zip(axes, rendered_metric_pairs)):
        ref_x, ref_y = tuple(pair.get("reference_xy", (0.5, 0.5)))
        if bool(panel_style.get("show_reference_lines", True)):
            ax.axvline(
                float(ref_x),
                color=reference_color,
                linewidth=float(panel_style.get("reference_linewidth", 0.8)),
                alpha=float(panel_style.get("reference_alpha", 0.30)),
                linestyle="--",
                zorder=0,
            )
            ax.axhline(
                float(ref_y),
                color=reference_color,
                linewidth=float(panel_style.get("reference_linewidth", 0.8)),
                alpha=float(panel_style.get("reference_alpha", 0.30)),
                linestyle="--",
                zorder=0,
            )

        for group_name in groups:
            group_key = str(group_name)
            sub = points_df.loc[points_df["group"].astype(str) == group_key]
            display_name = str(group_label_map.get(group_key, group_key))
            legend_label = (
                f"{display_name} (n={len(sub)})"
                if bool(panel_style.get("show_counts_in_legend", True))
                else display_name
            )
            ax.plot(
                sub[pair["x_col"]],
                sub[pair["y_col"]],
                linestyle="None",
                marker=str(panel_style.get("marker", "o")),
                color=panel_colors.get(group_key, "gray"),
                markersize=float(panel_style.get("marker_size", 5.0)),
                alpha=float(panel_style.get("marker_alpha", 0.62)),
                label=legend_label,
                zorder=2,
            )

        if bool(panel_style.get("reference_halo", True)):
            ax.plot(
                [float(ref_x)],
                [float(ref_y)],
                linestyle="None",
                marker=str(panel_style.get("reference_marker", "+")),
                color=str(panel_style.get("reference_halo_color", "white")),
                markersize=float(panel_style.get("reference_halo_marker_size", 10.0)),
                markeredgewidth=float(panel_style.get("reference_halo_markeredgewidth", 3.0)),
                zorder=2.8,
            )
        ax.plot(
            [float(ref_x)],
            [float(ref_y)],
            linestyle="None",
            marker=str(panel_style.get("reference_marker", "+")),
            color=reference_color,
            markersize=float(panel_style.get("reference_marker_size", 8.0)),
            markeredgewidth=float(panel_style.get("reference_markeredgewidth", 1.5)),
            label=str(panel_labels.get("reference", "random")),
            zorder=3,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if bool(panel_style.get("equal_aspect", True)):
            ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(
            pair.get("xlabel", pair["x_col"]),
            fontsize=panel_style.get("axis_label_fontsize", None),
        )
        ax.set_ylabel(
            pair.get("ylabel", pair["y_col"]),
            fontsize=panel_style.get("axis_label_fontsize", None),
        )
        if panel_style.get("tick_labelsize", None) is not None:
            ax.tick_params(labelsize=float(panel_style["tick_labelsize"]))
        title = pair.get("title", "")
        if title:
            ax.set_title(str(title), fontsize=panel_style.get("title_fontsize", None))
        show_axis_legend = bool(panel_style.get("show_legend", True)) and (
            ax_index == int(panel_style.get("legend_ax_index", 0))
        )
        if show_axis_legend:
            ax.legend(
                frameon=False,
                loc=str(panel_style.get("legend_loc", "lower left")),
                fontsize=panel_style.get("legend_fontsize", None),
                markerscale=float(panel_style.get("legend_markerscale", 1.0)),
                handletextpad=float(panel_style.get("legend_handletextpad", 0.8)),
                labelspacing=float(panel_style.get("legend_labelspacing", 0.5)),
            )

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, points_df, saved_paths


def _default_turning_bias_policy_summary_metrics() -> list[dict[str, object]]:
    """Return the maintained source-column to policy-label mapping."""
    return [
        {"column": "Bf", "label": r"$P_{\mathrm{SF}}$", "reference": 2.0 / 3.0},
        {"column": "Ba", "label": r"$P_{\mathrm{SA}}$", "reference": 0.5},
        {"column": "Lf", "label": r"$P_{\mathrm{BF}}$", "reference": 2.0 / 3.0},
        {"column": "Lo", "label": r"$P_{\mathrm{BS}}$", "reference": 0.5},
    ]


def _turning_bias_cr1_mean(values: np.ndarray, clusters: np.ndarray) -> dict[str, float | int]:
    """Intercept-only CR1 mean and t(G-1) interval for session rows."""
    numeric = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(clusters).astype(str)
    finite = np.isfinite(numeric)
    numeric = numeric[finite]
    cluster_ids = cluster_ids[finite]
    unique_clusters = np.unique(cluster_ids)
    n_rows = int(numeric.size)
    n_clusters = int(unique_clusters.size)
    if n_rows < 2 or n_clusters < 3:
        raise ValueError("Turning-policy CR1 intervals require at least three mouse clusters.")

    estimate = float(np.mean(numeric))
    residual = numeric - estimate
    cluster_scores = np.asarray(
        [float(np.sum(residual[cluster_ids == cluster])) for cluster in unique_clusters],
        dtype=float,
    )
    # This is the intercept-only specialization of the repository's CR1 OLS
    # estimator: (G/(G-1))*((N-1)/(N-K)), with K=1.
    correction = n_clusters / float(n_clusters - 1)
    variance = correction * float(np.sum(cluster_scores**2)) / float(n_rows**2)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    critical = float(stats.t.ppf(0.975, df=n_clusters - 1))
    return {
        "n_sessions": n_rows,
        "n_mice": n_clusters,
        "mean": estimate,
        "se_cr1": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "df": n_clusters - 1,
    }


def prepare_turning_bias_policy_summary(
    *,
    csv_path,
    window_minutes: int = 30,
    groups: tuple[str, ...] = ("naive", "LTM"),
    metric_specs: list[dict[str, object]] | None = None,
    expected_group_rows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Load a fixed-clock policy table and calculate group means with CR1 CIs.

    The input is expected to contain one session per row. The point estimate is
    therefore the equal-session mean; repeated sessions from a mouse are retained
    and handled by the mouse-clustered CR1 covariance.
    """
    metrics = deepcopy(metric_specs) if metric_specs is not None else (
        _default_turning_bias_policy_summary_metrics()
    )
    points = pd.read_csv(Path(csv_path), dtype={"animal_id": str})
    required = {"window_minutes", "group", "animal_id"}
    required.update(str(metric["column"]) for metric in metrics)
    missing = sorted(required.difference(points.columns))
    if missing:
        raise ValueError(f"Fixed-clock turning-policy CSV is missing columns: {missing}")

    numeric_window = pd.to_numeric(points["window_minutes"], errors="coerce")
    points = points.loc[numeric_window.eq(int(window_minutes))].copy()
    if points.empty:
        raise ValueError(
            f"No turning-policy rows found for window_minutes={int(window_minutes)} "
            f"in {csv_path!r}."
        )
    requested_groups = tuple(str(group) for group in groups)
    points = points.loc[points["group"].astype(str).isin(requested_groups)].copy()
    observed_groups = set(points["group"].astype(str))
    missing_groups = sorted(set(requested_groups).difference(observed_groups))
    if missing_groups:
        raise ValueError(f"Fixed-clock turning-policy CSV is missing groups: {missing_groups}")

    counts = points["group"].astype(str).value_counts().to_dict()
    if expected_group_rows is not None:
        expected = {str(group): int(count) for group, count in expected_group_rows.items()}
        observed = {group: int(counts.get(group, 0)) for group in expected}
        if observed != expected:
            raise ValueError(
                "Fixed-clock turning-policy cohort drifted: "
                f"expected {expected}, observed {observed}."
            )

    summary_rows: list[dict[str, object]] = []
    for metric in metrics:
        column = str(metric["column"])
        numeric = pd.to_numeric(points[column], errors="coerce")
        if numeric.isna().any() or ((numeric < 0.0) | (numeric > 1.0)).any():
            raise ValueError(
                f"Fixed-clock turning-policy column {column!r} must be finite and within [0, 1]."
            )
        points[column] = numeric
        for group in requested_groups:
            group_rows = points.loc[points["group"].astype(str).eq(group)]
            cr1 = _turning_bias_cr1_mean(
                group_rows[column].to_numpy(dtype=float),
                group_rows["animal_id"].to_numpy(),
            )
            summary_rows.append(
                {
                    "window_minutes": int(window_minutes),
                    "metric": column,
                    "policy_label": str(metric["label"]),
                    "random_reference": float(metric["reference"]),
                    "group": group,
                    **cr1,
                }
            )
    return points.reset_index(drop=True), pd.DataFrame(summary_rows), metrics


def render_turning_bias_policy_summary_panel(
    *,
    csv_path,
    window_minutes: int = 30,
    groups: tuple[str, ...] = ("naive", "LTM"),
    metric_specs: list[dict[str, object]] | None = None,
    expected_group_rows: dict[str, int] | None = None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "turning_bias_policy_summary_fixed30min",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render four fixed-clock policies as raw sessions plus CR1 group summaries."""
    panel_layout = {
        "panel_size": (4.25, 2.58),
        "left": 0.16,
        "right": 0.985,
        "bottom": 0.18,
        "top": 0.79,
    }
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        "naive": "#1F90D5",
        "LTM": "#D82929",
        "reference": "#6F6F6F",
        "row_band": "#F6F6F4",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "raw_point_size": 8.0,
        "raw_alpha": 0.24,
        "raw_jitter": 0.035,
        "group_offset": 0.135,
        "mean_marker_size": 4.1,
        "mean_marker_edgewidth": 0.45,
        "ci_linewidth": 1.15,
        "ci_capsize": 2.0,
        "reference_linewidth": 0.75,
        "reference_alpha": 0.70,
        "axis_label_fontsize": 8.0,
        "tick_labelsize": 7.0,
        "title_fontsize": 8.4,
        "subtitle_fontsize": 6.8,
        "legend_fontsize": 6.4,
        "xlim": (0.0, 1.0),
        "xticks": (0.0, 0.5, 1.0),
        "seed": 20260804,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "title": f"First {int(window_minutes)} min in maze",
        "subtitle": "Explore decisions only",
        "xlabel": "Policy probability",
        "group_labels": {"naive": "Naive", "LTM": "Return"},
        "reference": "Random",
    }
    if labels:
        panel_labels.update(labels)

    points, summary, metrics = prepare_turning_bias_policy_summary(
        csv_path=csv_path,
        window_minutes=window_minutes,
        groups=groups,
        metric_specs=metric_specs,
        expected_group_rows=expected_group_rows,
    )
    figsize = panel_size or panel_layout["panel_size"]
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=float(panel_layout["left"]),
        right=float(panel_layout["right"]),
        bottom=float(panel_layout["bottom"]),
        top=float(panel_layout["top"]),
    )

    n_metrics = len(metrics)
    row_positions = np.arange(n_metrics, dtype=float)[::-1]
    group_offsets = np.linspace(
        float(panel_style["group_offset"]),
        -float(panel_style["group_offset"]),
        num=len(groups),
    )
    rng = np.random.default_rng(int(panel_style["seed"]))
    group_label_map = dict(panel_labels.get("group_labels", {}))
    for metric_index, (metric, row_y) in enumerate(zip(metrics, row_positions)):
        if metric_index % 2 == 0:
            ax.axhspan(
                row_y - 0.43,
                row_y + 0.43,
                color=panel_colors["row_band"],
                zorder=-4,
            )
        reference = float(metric["reference"])
        ax.plot(
            [reference, reference],
            [row_y - 0.38, row_y + 0.38],
            color=panel_colors["reference"],
            linewidth=float(panel_style["reference_linewidth"]),
            alpha=float(panel_style["reference_alpha"]),
            linestyle=(0, (2.2, 2.2)),
            zorder=0,
        )
        column = str(metric["column"])
        for group_index, (group, offset) in enumerate(zip(groups, group_offsets)):
            group_key = str(group)
            group_rows = points.loc[points["group"].astype(str).eq(group_key)]
            y_jitter = rng.uniform(
                -float(panel_style["raw_jitter"]),
                float(panel_style["raw_jitter"]),
                size=len(group_rows),
            )
            ax.scatter(
                group_rows[column].to_numpy(dtype=float),
                row_y + offset + y_jitter,
                s=float(panel_style["raw_point_size"]),
                color=panel_colors[group_key],
                alpha=float(panel_style["raw_alpha"]),
                linewidths=0,
                zorder=1,
            )
            stat_row = summary.loc[
                summary["metric"].eq(column) & summary["group"].eq(group_key)
            ].iloc[0]
            estimate = float(stat_row["mean"])
            low = float(stat_row["ci_low"])
            high = float(stat_row["ci_high"])
            ax.errorbar(
                estimate,
                row_y + offset,
                xerr=np.asarray([[estimate - low], [high - estimate]]),
                fmt="o",
                color=panel_colors[group_key],
                markerfacecolor=panel_colors[group_key],
                markeredgecolor="white",
                markeredgewidth=float(panel_style["mean_marker_edgewidth"]),
                markersize=float(panel_style["mean_marker_size"]),
                elinewidth=float(panel_style["ci_linewidth"]),
                capsize=float(panel_style["ci_capsize"]),
                capthick=float(panel_style["ci_linewidth"]),
                zorder=3 + group_index,
            )

    ax.set_xlim(*tuple(panel_style["xlim"]))
    ax.set_xticks(tuple(panel_style["xticks"]))
    ax.set_ylim(-0.55, n_metrics - 0.45)
    ax.set_yticks(row_positions, [str(metric["label"]) for metric in metrics])
    ax.set_xlabel(
        str(panel_labels["xlabel"]), fontsize=float(panel_style["axis_label_fontsize"])
    )
    ax.tick_params(
        axis="both",
        labelsize=float(panel_style["tick_labelsize"]),
        width=0.6,
        length=2.3,
    )
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.text(
        0.0,
        1.20,
        str(panel_labels["title"]),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=float(panel_style["title_fontsize"]),
        fontweight="bold",
    )
    counts = points["group"].astype(str).value_counts().to_dict()
    subtitle = str(panel_labels["subtitle"])
    if bool(panel_labels.get("show_session_counts", True)):
        count_text = ", ".join(
            f"{int(counts[str(group)])} {str(group_label_map.get(str(group), group)).lower()}"
            for group in groups
        )
        subtitle = f"{subtitle}; {count_text} sessions"
    ax.text(
        0.0,
        1.095,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=float(panel_style["subtitle_fontsize"]),
        color="#4F4F4F",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=panel_colors[str(group)],
            markerfacecolor=panel_colors[str(group)],
            markeredgecolor="white",
            markeredgewidth=float(panel_style["mean_marker_edgewidth"]),
            linewidth=float(panel_style["ci_linewidth"]),
            markersize=float(panel_style["mean_marker_size"]),
            label=str(group_label_map.get(str(group), group)),
        )
        for group in groups
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=panel_colors["reference"],
            linewidth=float(panel_style["reference_linewidth"]),
            linestyle=(0, (2.2, 2.2)),
            label=str(panel_labels["reference"]),
        )
    )
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.155),
        ncol=len(legend_handles),
        fontsize=float(panel_style["legend_fontsize"]),
        handlelength=1.5,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0,
    )

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, points, summary, saved_paths


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


def _bh_adjust_pvalues(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full(pvalues.shape, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return qvalues
    finite_p = pvalues[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = float(len(ranked))
    adjusted = ranked * m / (np.arange(len(ranked), dtype=float) + 1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    qvalues[finite] = restored
    return qvalues


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
    month_labels = month_labels or {0: "Naive", 3: "3 mo", 6: "6 mo", 9: "9 mo", 12: "12 mo"}
    month_to_x = {month: idx for idx, month in enumerate(month_order)}

    points_df = pd.read_csv(Path(points_csv))
    if metric not in points_df.columns:
        raise KeyError(f"Metric column {metric!r} not found in {points_csv!r}.")
    points_df = points_df.copy()
    points_df["month"] = pd.to_numeric(points_df["month"], errors="coerce").astype("Int64")
    points_df[metric] = pd.to_numeric(points_df[metric], errors="coerce")
    points_df = points_df[np.isfinite(points_df[metric]) & points_df["month"].isin(month_order)].copy()

    if summary_csv is not None:
        summary_df = pd.read_csv(Path(summary_csv))
    else:
        rows = []
        for month in month_order:
            vals = points_df.loc[points_df["month"].astype(int) == month, metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rows.append(
                    {
                        "month": month,
                        "mean": float(np.nanmean(vals)),
                        "sem": float(stats.sem(vals, nan_policy="omit")) if vals.size > 1 else 0.0,
                    }
                )
        summary_df = pd.DataFrame(rows)
    summary_df = summary_df.copy()
    summary_df["month"] = pd.to_numeric(summary_df["month"], errors="coerce").astype("Int64")

    contrasts_df = pd.read_csv(Path(contrasts_csv)) if contrasts_csv is not None else pd.DataFrame()

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
        vals = points_df.loc[points_df["month"].astype(int) == month, metric].to_numpy(dtype=float)
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
            p_col = "paired_wilcoxon_q_bh" if source.startswith("paired") else "mannwhitney_q_bh"
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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
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
        vals = points_df.loc[points_df["group"].astype(str) == group, value_col].to_numpy(dtype=float)
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
                if group_a < 0 or group_b < 0 or group_a >= len(group_order) or group_b >= len(group_order):
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
                text_offset = step * float(panel_style.get("annotation_text_offset_fraction", 0.30))
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
                points_df.sort_values("group_index")["group"].drop_duplicates().astype(str).tolist()
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
            vals = points_df.loc[points_df["group"].astype(str) == group, metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rows.append(
                    {
                        "group": group,
                        "mean": float(np.nanmean(vals)),
                        "sem": float(stats.sem(vals, nan_policy="omit")) if vals.size > 1 else 0.0,
                    }
                )
        summary_df = pd.DataFrame(rows)
    contrasts_df = pd.read_csv(Path(contrasts_csv)) if contrasts_csv is not None else pd.DataFrame()

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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, panel_data, saved_paths


def render_refiner_symmetry_preservation_panel(
    *,
    points_csv,
    summary_csv,
    contrasts_csv,
    panel_specs,
    group_order=None,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "refiner_symmetry_preservation",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    panel_layout = {"panel_size": (11.4, 4.0), "grid_wspace": 0.35}
    if layout:
        panel_layout.update(layout)

    panel_colors = {
        "high_symmetry": "#c44e52",
        "other": "#2f6fb0",
        "pair_line": "#b3b3b3",
        "edge": "black",
        "error": "black",
    }
    if colors:
        panel_colors.update(colors)

    panel_style = {
        "bar_width": 0.58,
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
        "annotation_offset_fraction": 0.06,
        "annotation_step_fraction": 0.08,
        "pair_line_alpha": 0.32,
        "pair_line_width": 0.8,
        "spine_top": False,
        "spine_right": False,
        "xtick_rotation": 0,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": "",
        "group_labels": {
            "high_symmetry": "High-sym",
            "other": "Other",
        },
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(Path(points_csv)).copy()
    summary_df = pd.read_csv(Path(summary_csv)).copy()
    contrasts_df = pd.read_csv(Path(contrasts_csv)).copy()
    points_df["panel_key"] = points_df["panel_key"].astype(str)
    summary_df["panel_key"] = summary_df["panel_key"].astype(str)
    contrasts_df["panel_key"] = contrasts_df["panel_key"].astype(str)
    points_df["group"] = points_df["group"].astype(str)
    summary_df["group"] = summary_df["group"].astype(str)
    if group_order is None:
        if "group_index" in points_df.columns:
            group_order = (
                points_df.sort_values("group_index")["group"].drop_duplicates().astype(str).tolist()
            )
        else:
            group_order = points_df["group"].drop_duplicates().astype(str).tolist()

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes.ravel()
    fig.subplots_adjust(wspace=float(panel_layout.get("grid_wspace", 0.35)))

    panel_data = {}
    for ax, spec in zip(axes, panel_specs, strict=False):
        panel_key = str(spec["panel_key"])
        sub_points = points_df.loc[points_df["panel_key"] == panel_key].copy()
        sub_points["value"] = pd.to_numeric(sub_points["value"], errors="coerce")
        sub_points = sub_points[np.isfinite(sub_points["value"])].copy()
        sub_summary = summary_df.loc[summary_df["panel_key"] == panel_key].copy()
        sub_contrasts = contrasts_df.loc[contrasts_df["panel_key"] == panel_key].copy()
        sub_colors = dict(panel_colors)
        spec_group_colors = spec.get("group_colors")
        if isinstance(spec_group_colors, dict):
            sub_colors.update(spec_group_colors)
        if spec.get("pair_line_color") is not None:
            sub_colors["pair_line"] = spec["pair_line_color"]
        sub_labels = {
            "xlabel": str(panel_labels.get("xlabel", "")),
            "ylabel": str(spec.get("ylabel", "")),
            "title": str(spec.get("title", "")),
            "group_labels": dict(panel_labels.get("group_labels", {})),
        }
        _draw_grouped_bar_scatter_axes(
            ax,
            points_df=sub_points,
            summary_df=sub_summary,
            contrasts_df=sub_contrasts,
            value_col="value",
            group_order=group_order,
            panel_colors=sub_colors,
            panel_style=panel_style,
            panel_labels=sub_labels,
            pair_id_cols=("animal_id", "session_pair_id"),
        )
        if spec.get("ylim") is not None:
            ax.set_ylim(*spec["ylim"])
        panel_data[panel_key] = {
            "points": sub_points,
            "summary": sub_summary,
            "contrasts": sub_contrasts,
        }

    fig.tight_layout()
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


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
        raise ValueError(f"No finite first_chunk_efficiency rows found in {points_csv!r}.")

    if group_order is None:
        if "group_index" in points_df.columns:
            points_df["group_index"] = pd.to_numeric(points_df["group_index"], errors="coerce")
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
        raise ValueError("No rows remain after applying group_order to exploration histogram data.")

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
        bins = np.linspace(value_min, value_max, int(panel_style["n_bins"]) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    group_label_map = panel_labels.get("group_labels", {})

    histogram_rows = []
    for group_idx, group_name in enumerate(group_order):
        sub = selected_df.loc[selected_df["group"].astype(str) == str(group_name)].copy()
        if sub.empty:
            continue
        display_name = str(group_label_map.get(group_name, group_name))
        legend_label = (
            f"{display_name} (n={len(sub)})"
            if bool(panel_style.get("show_counts_in_legend", True))
            else display_name
        )
        color = panel_colors.get(group_name, default_color_cycle[group_idx % len(default_color_cycle)])
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
        counts, edges = np.histogram(sub["first_chunk_efficiency"].to_numpy(dtype=float), bins=bins)
        total = int(np.sum(counts))
        for bin_index, count in enumerate(counts):
            histogram_rows.append(
                {
                    "group": str(group_name),
                    "bin_index": int(bin_index),
                    "bin_left": float(edges[bin_index]),
                    "bin_right": float(edges[bin_index + 1]),
                    "count": int(count),
                    "fraction_within_group": float(count / total) if total > 0 else np.nan,
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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, panel_data, saved_paths


def _nested_get(mapping, path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def render_paired_bar_scatter_panel(
    *,
    points_csv,
    summary_csv=None,
    stats_json=None,
    value_col: str = "euclidean_error",
    group_order=("Data", "Shuffle"),
    pair_col: str = "pair_id",
    group_col: str = "group",
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "paired_bar_scatter",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render paired bar/scatter rows from saved long-format data."""
    panel_layout = {"panel_size": (2.6, 4.0)}
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "Data": "#DA2323",
        "Shuffle": "#8f8f8f",
        "pair_line": "#8f8f8f",
        "bar_edge": "none",
        "point_edge": "none",
        "annotation": "black",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "bar_width": 0.72,
        "bar_alpha": 0.78,
        "point_size": 26,
        "point_alpha": 0.72,
        "point_jitter": 0.035,
        "point_seed": 20260420,
        "pair_line_alpha": 0.34,
        "pair_line_width": 0.8,
        "summary_error": None,
        "error_linewidth": 1.0,
        "annotation": True,
        "annotation_p_path": ("cluster_ols", "p_less"),
        "annotation_text": None,
        "annotation_y": None,
        "annotation_offset_fraction": 0.06,
        "annotation_height_fraction": 0.025,
        "annotation_linewidth": 1.0,
        "annotation_fontsize": None,
        "annotation_show_ns": True,
        "ylim": None,
        "yticks": None,
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "xlabel": "",
        "ylabel": "Value",
        "title": "",
        "group_labels": {},
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(Path(points_csv)).copy()
    required = {pair_col, group_col, value_col}
    missing = required.difference(points_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {points_csv!r}: {sorted(missing)}")
    points_df[value_col] = pd.to_numeric(points_df[value_col], errors="coerce")
    points_df = points_df[np.isfinite(points_df[value_col])].copy()
    group_order = [str(group) for group in group_order]
    points_df[group_col] = points_df[group_col].astype(str)
    points_df = points_df.loc[points_df[group_col].isin(group_order)].copy()
    if points_df.empty:
        raise ValueError("No finite paired rows remain after filtering group_order.")

    if summary_csv:
        summary_df = pd.read_csv(Path(summary_csv)).copy()
        summary_df["group"] = summary_df["group"].astype(str)
    else:
        summary_df = (
            points_df.groupby(group_col, as_index=False)
            .agg(n=(value_col, "size"), mean=(value_col, "mean"), sd=(value_col, "std"))
            .rename(columns={group_col: "group"})
        )
        summary_df["sem"] = summary_df["sd"] / np.sqrt(summary_df["n"])

    stats_data = {}
    if stats_json:
        stats_data = json.loads(Path(stats_json).read_text(encoding="utf-8"))

    group_to_x = {group: idx for idx, group in enumerate(group_order)}
    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])

    complete_pairs = (
        points_df.pivot_table(index=pair_col, columns=group_col, values=value_col, aggfunc="mean")
        .reindex(columns=group_order)
        .dropna()
    )
    for _, row in complete_pairs.iterrows():
        ax.plot(
            [group_to_x[group] for group in group_order],
            [float(row[group]) for group in group_order],
            color=panel_colors["pair_line"],
            alpha=float(panel_style["pair_line_alpha"]),
            linewidth=float(panel_style["pair_line_width"]),
            zorder=1,
        )

    for group in group_order:
        x = group_to_x[group]
        sub_summary = summary_df.loc[summary_df["group"].astype(str) == group]
        if sub_summary.empty:
            continue
        mean = float(sub_summary.iloc[0]["mean"])
        ax.bar(
            x,
            mean,
            width=float(panel_style["bar_width"]),
            color=panel_colors.get(group, "#888888"),
            edgecolor=panel_colors["bar_edge"],
            alpha=float(panel_style["bar_alpha"]),
            zorder=2,
        )
        err_col = panel_style.get("summary_error")
        if err_col and err_col in sub_summary.columns:
            err = float(sub_summary.iloc[0][err_col])
            if np.isfinite(err):
                ax.errorbar(
                    x,
                    mean,
                    yerr=err,
                    color="black",
                    linewidth=float(panel_style["error_linewidth"]),
                    capsize=0,
                    zorder=4,
                )

    rng = np.random.default_rng(int(panel_style["point_seed"]))
    for group in group_order:
        sub = points_df.loc[points_df[group_col] == group]
        if sub.empty:
            continue
        x = group_to_x[group]
        jitter = rng.normal(0.0, float(panel_style["point_jitter"]), size=len(sub))
        ax.scatter(
            np.full(len(sub), x, dtype=float) + jitter,
            sub[value_col].to_numpy(dtype=float),
            s=float(panel_style["point_size"]),
            color=panel_colors.get(group, "#888888"),
            edgecolors=panel_colors["point_edge"],
            alpha=float(panel_style["point_alpha"]),
            zorder=3,
        )

    if bool(panel_style["annotation"]) and len(group_order) >= 2:
        p_value = _nested_get(stats_data, tuple(panel_style["annotation_p_path"]), np.nan)
        text = panel_style.get("annotation_text")
        if text is None:
            text = _significance_text(float(p_value)) if p_value is not None else "n.s."
        if text != "n.s." or bool(panel_style["annotation_show_ns"]):
            y_values = points_df[value_col].to_numpy(dtype=float)
            data_top = float(np.nanmax(y_values))
            y_min = float(np.nanmin(y_values))
            span = max(data_top - y_min, 1.0)
            y = panel_style["annotation_y"]
            if y is None:
                y = data_top + span * float(panel_style["annotation_offset_fraction"])
            h = span * float(panel_style["annotation_height_fraction"])
            x1 = group_to_x[group_order[0]]
            x2 = group_to_x[group_order[1]]
            ax.plot(
                [x1, x1, x2, x2],
                [y, y + h, y + h, y],
                color=panel_colors["annotation"],
                linewidth=float(panel_style["annotation_linewidth"]),
                clip_on=False,
            )
            ax.text(
                (x1 + x2) / 2.0,
                y + h,
                str(text),
                ha="center",
                va="bottom",
                fontsize=panel_style["annotation_fontsize"],
                color=panel_colors["annotation"],
            )

    group_labels = panel_labels.get("group_labels", {})
    ax.set_xticks([group_to_x[group] for group in group_order])
    ax.set_xticklabels([group_labels.get(group, group) for group in group_order])
    ax.set_xlabel(str(panel_labels["xlabel"]))
    ax.set_ylabel(str(panel_labels["ylabel"]))
    if panel_labels.get("title"):
        ax.set_title(str(panel_labels["title"]))
    if panel_style["yticks"] is not None:
        ax.set_yticks(list(panel_style["yticks"]))
    if panel_style["ylim"] is not None:
        ax.set_ylim(*panel_style["ylim"])
    ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
    ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
    fig.tight_layout()

    panel_data = {
        "points": points_df,
        "summary": summary_df,
        "stats": stats_data,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
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
            map_df[flag_col] = map_df[flag_col].astype(str).str.lower().isin({"true", "1", "yes"})
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
    outline_layout = {**panel_layout, "maze_wall_linewidth": panel_style["maze_wall_linewidth"]}
    draw_maze_outline(ax, maze, panel_size=panel_size, layout=outline_layout, colors={"wall": panel_colors["wall"]})

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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, panel_data, saved_paths


def render_level_exploration_panel(
    *,
    behavior_rows_csv,
    decoding_rows_csv,
    panel_size=None,
    layout: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "level_exploration_summary",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render level-correlation and level-decoding scatter plots vs exploration efficiency.

    The notebook keeps figure sizing / font / style configuration local, while this
    helper owns the CSV loading, merge logic, regression overlay, and annotation
    formatting.
    """
    panel_layout = {
        "panel_size": (11.5, 4.6),
        "grid_wspace": 0.32,
    }
    if layout:
        panel_layout.update(layout)

    panel_style = {
        "point_color": "#DA2323",
        "point_size": 28,
        "point_alpha": 0.75,
        "show_ci": False,
        "ci_panels": ("level_correlation",),
        "ci_level": 0.95,
        "ci_alpha": 0.18,
        "ci_edge_alpha": 0.35,
        "ci_line_width": 0.8,
        "line_color": "#555555",
        "line_style": "--",
        "line_width": 1.6,
        "annotation_fontsize": 10,
        "annotation_x": 0.04,
        "annotation_y": 0.95,
        "grid_alpha": 0.18,
    }
    if style:
        panel_style.update(style)

    panel_labels = {
        "xlabel": "Exploration efficiency",
        "left_ylabel": "Level correlation",
        "right_ylabel": "Level decoding accuracy",
        "left_title": "Level correlation vs exploration efficiency",
        "right_title": "Level decoding accuracy vs exploration efficiency",
        "annotation_template": "Pearson r={r:.2f}, p={p:.2g}",
    }
    if labels:
        panel_labels.update(labels)

    behavior_rows = pd.read_csv(Path(behavior_rows_csv))
    decoding_rows = pd.read_csv(Path(decoding_rows_csv)).rename(
        columns={
            "src": "src_session",
            "dst": "dst_session",
            "animal": "animal_id",
        }
    )
    decoding_rows = decoding_rows.merge(
        behavior_rows[["src_session", "dst_session", "exploration_efficiency"]],
        on=["src_session", "dst_session"],
        how="left",
    )

    level_correlation_rows = behavior_rows[["exploration_efficiency", "level_mean_all"]].dropna().copy()
    level_decoding_rows = decoding_rows[["exploration_efficiency", "level_acc"]].dropna().copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes.ravel()
    fig.subplots_adjust(wspace=float(panel_layout.get("grid_wspace", 0.32)))

    plot_specs = [
        {
            "rows": level_correlation_rows,
            "y_col": "level_mean_all",
            "title": panel_labels["left_title"],
            "ylabel": panel_labels["left_ylabel"],
            "stats_key": "level_correlation",
        },
        {
            "rows": level_decoding_rows,
            "y_col": "level_acc",
            "title": panel_labels["right_title"],
            "ylabel": panel_labels["right_ylabel"],
            "stats_key": "level_decoding",
        },
    ]

    summary_stats: dict[str, dict[str, float | int]] = {}
    ci_panels_raw = panel_style.get("ci_panels", ("level_correlation",))
    ci_panels = (
        {str(ci_panels_raw)}
        if isinstance(ci_panels_raw, str)
        else {str(panel_name) for panel_name in ci_panels_raw}
    )
    for ax, plot_spec in zip(axes, plot_specs, strict=False):
        rows_df = plot_spec["rows"]
        x = rows_df["exploration_efficiency"].to_numpy(dtype=float)
        y = rows_df[plot_spec["y_col"]].to_numpy(dtype=float)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        fit = stats.linregress(x, y)

        ax.scatter(
            x,
            y,
            color=panel_style["point_color"],
            s=float(panel_style["point_size"]),
            alpha=float(panel_style["point_alpha"]),
            zorder=3,
        )

        x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        y_fit = fit.slope * x_fit + fit.intercept
        if bool(panel_style.get("show_ci", False)) and plot_spec["stats_key"] in ci_panels and x.size >= 3:
            x_mean = float(np.mean(x))
            sxx = float(np.sum((x - x_mean) ** 2))
            residuals = y - (fit.slope * x + fit.intercept)
            dof = int(x.size - 2)
            rss = float(np.sum(residuals**2))
            if dof > 0 and sxx > 0.0:
                mse = rss / dof
                t_crit = float(stats.t.ppf((1.0 + float(panel_style.get("ci_level", 0.95))) / 2.0, dof))
                se_fit = np.sqrt(mse * (1.0 / x.size + ((x_fit - x_mean) ** 2) / sxx))
                ci_half_width = t_crit * se_fit
                ci_color = panel_style.get("ci_color", panel_style["point_color"])
                ax.fill_between(
                    x_fit,
                    y_fit - ci_half_width,
                    y_fit + ci_half_width,
                    color=ci_color,
                    alpha=float(panel_style.get("ci_alpha", 0.18)),
                    linewidth=0.0,
                    zorder=1,
                )
                ax.plot(
                    x_fit,
                    y_fit - ci_half_width,
                    color=ci_color,
                    alpha=float(panel_style.get("ci_edge_alpha", 0.35)),
                    linewidth=float(panel_style.get("ci_line_width", 0.8)),
                    zorder=1.2,
                )
                ax.plot(
                    x_fit,
                    y_fit + ci_half_width,
                    color=ci_color,
                    alpha=float(panel_style.get("ci_edge_alpha", 0.35)),
                    linewidth=float(panel_style.get("ci_line_width", 0.8)),
                    zorder=1.2,
                )
        ax.plot(
            x_fit,
            y_fit,
            color=panel_style["line_color"],
            linestyle=panel_style["line_style"],
            linewidth=float(panel_style["line_width"]),
            zorder=2,
        )
        ax.text(
            float(panel_style["annotation_x"]),
            float(panel_style["annotation_y"]),
            str(panel_labels["annotation_template"]).format(r=pearson_r, p=pearson_p),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(panel_style["annotation_fontsize"]),
        )
        ax.set_title(str(plot_spec["title"]))
        ax.set_xlabel(str(panel_labels["xlabel"]))
        ax.set_ylabel(str(plot_spec["ylabel"]))
        ax.grid(alpha=float(panel_style["grid_alpha"]))

        summary_stats[plot_spec["stats_key"]] = {
            "n": int(rows_df.shape[0]),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "fit_slope": float(fit.slope),
            "fit_intercept": float(fit.intercept),
        }

    panel_data = {
        "behavior_rows": behavior_rows,
        "decoding_rows": decoding_rows,
        "level_correlation_rows": level_correlation_rows,
        "level_decoding_rows": level_decoding_rows,
        "stats": summary_stats,
    }

    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def _add_distribution_text_label(ax, text, xy, *, color, style, default_ha="center", default_va="center"):
    if xy is None:
        return
    transform = ax.transAxes if str(style.get("direct_label_transform", "data")) == "axes" else ax.transData
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

    data = np.load(Path(npz_path), allow_pickle=True)
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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, panel_data, saved_paths


def render_shuffle_mean_distribution_panel(
    *,
    npz_path,
    null_key: str = "null",
    obs_key: str = "obs",
    p_key: str = "p_right",
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "shuffle_mean_distribution",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a saved shuffle-mean null distribution with a real-data line."""
    panel_layout = {"panel_size": (4.2, 3.6)}
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "shuffle": "#4d4d4d",
        "shuffle_edge": "#222222",
        "data": "#e41a1c",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "bins": 80,
        "hist_alpha": 0.75,
        "hist_linewidth": 0.5,
        "line_width": 2.0,
        "line_style": "--",
        "legend": True,
        "legend_loc": "best",
        "show_pvalue": True,
        "show_counts": True,
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
        "xlabel": "Level map correlation (across-pair mean)",
        "ylabel": "Density",
        "title": "",
        "shuffle": "Shuffle",
        "data": "Real data",
    }
    if labels:
        panel_labels.update(labels)

    data = np.load(Path(npz_path), allow_pickle=True)
    null = np.asarray(data[null_key], dtype=float)
    null = null[np.isfinite(null)]
    obs = float(np.asarray(data[obs_key], dtype=float).ravel()[0])
    p_value = float(np.asarray(data[p_key], dtype=float).ravel()[0]) if p_key in data.files else np.nan

    shuffle_label = str(panel_labels["shuffle"])
    data_label = str(panel_labels["data"])
    if bool(panel_style["show_counts"]):
        shuffle_label = f"{shuffle_label} (K={len(null)})"
    if bool(panel_style["show_pvalue"]) and np.isfinite(p_value):
        data_label = f"{data_label} (r={obs:.4f}, p={p_value:.1e})"
    else:
        data_label = f"{data_label} (r={obs:.4f})"

    fig, ax = plt.subplots(figsize=panel_size or panel_layout["panel_size"])
    ax.hist(
        null,
        bins=panel_style["bins"],
        density=True,
        color=panel_colors["shuffle"],
        edgecolor=panel_colors["shuffle_edge"],
        linewidth=float(panel_style["hist_linewidth"]),
        alpha=float(panel_style["hist_alpha"]),
        label=shuffle_label,
        zorder=1,
    )
    ax.axvline(
        obs,
        color=panel_colors["data"],
        linestyle=str(panel_style["line_style"]),
        linewidth=float(panel_style["line_width"]),
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
        "null": null,
        "obs": obs,
        "p_value": p_value,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, ax, panel_data, saved_paths


def _load_matlab_local_correlation_figure(fig_path) -> dict:
    matlab_fig = sio.loadmat(str(fig_path), squeeze_me=True, struct_as_record=False)["hgS_070000"]
    figure_children = np.atleast_1d(matlab_fig.children)
    axes_obj = next(child for child in figure_children if getattr(child, "type", None) == "axes")
    colorbar_obj = next(
        child for child in figure_children if "colorbar" in str(getattr(child, "type", "")).lower()
    )
    axes_children = np.atleast_1d(axes_obj.children)
    image_obj = next(child for child in axes_children if getattr(child, "type", None) == "image")
    text_children = [child for child in axes_children if getattr(child, "type", None) == "text"]
    title_text = text_children[0].properties.String if text_children else None
    return {
        "image": np.asarray(image_obj.properties.CData),
        "xdata": np.asarray(image_obj.properties.XData, dtype=float),
        "ydata": np.asarray(image_obj.properties.YData, dtype=float),
        "xticks": np.asarray(axes_obj.properties.XTick, dtype=float),
        "yticks": np.asarray(axes_obj.properties.YTick, dtype=float),
        "xticklabels": [str(v) for v in np.atleast_1d(axes_obj.properties.XTickLabel).tolist()],
        "yticklabels": [str(v) for v in np.atleast_1d(axes_obj.properties.YTickLabel).tolist()],
        "clim": np.asarray(axes_obj.properties.CLim, dtype=float),
        "colormap": np.asarray(colorbar_obj.properties.Colormap, dtype=float),
        "colorbar_ticks": np.asarray(colorbar_obj.properties.Ticks, dtype=float),
        "colorbar_ticklabels": [str(v) for v in np.atleast_1d(colorbar_obj.properties.TickLabels).tolist()],
        "title_text": title_text,
    }


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
    panel_layout = {
        "panel_size": (6.2, 9.2),
        "grid_hspace": 0.32,
    }
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

    top_panel = _load_matlab_local_correlation_figure(top_fig_path)
    bottom_panel = _load_matlab_local_correlation_figure(bottom_fig_path)
    panel_data = [top_panel, bottom_panel]

    shared_vmin = None
    shared_vmax = None
    if bool(panel_style["shared_clim"]):
        shared_vmin = panel_style["shared_vmin"]
        shared_vmax = panel_style["shared_vmax"]
        if shared_vmin is None or shared_vmax is None:
            combined_values = np.concatenate(
                [
                    np.asarray(fig_data["image"], dtype=float)[np.isfinite(np.asarray(fig_data["image"], dtype=float))]
                    for fig_data in panel_data
                ]
            )
            if combined_values.size == 0:
                raise ValueError("Local correlation figures do not contain finite values for shared color limits.")
            if shared_vmin is None:
                shared_vmin = float(np.min(combined_values))
            if shared_vmax is None:
                shared_vmax = float(np.max(combined_values))

    fig, axes = plt.subplots(2, 1, figsize=panel_size or panel_layout["panel_size"], squeeze=False)
    fig.subplots_adjust(hspace=float(panel_layout["grid_hspace"]))

    for ax, fig_data, subplot_key, subplot_title in zip(
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
        scale_bar_ax = axes[1, 0] if str(panel_style["scale_bar_panel"]) == "bottom" else axes[0, 0]
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
        fig.suptitle(str(panel_labels["panel_title"]), y=float(panel_style["suptitle_y"]))
    if bool(panel_style["show_axes"]):
        fig.align_ylabels(axes[:, 0])

    panel_data = {
        "top": top_panel,
        "bottom": bottom_panel,
        "layout": panel_layout,
        "style": panel_style,
        "labels": panel_labels,
        "paths": {
            "top": str(top_fig_path),
            "bottom": str(bottom_fig_path),
        },
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes[:, 0], panel_data, saved_paths


def render_route_choice_schematic(
    *,
    layout: dict | None = None,
    geometry: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    text: dict | None = None,
    save: bool = False,
    export_name: str = "route_choice_schematic",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    resolved_layout = get_route_choice_schematic_layout(layout)
    resolved_geometry = get_route_choice_schematic_geometry(geometry)
    resolved_colors = get_route_choice_schematic_colors(colors)
    resolved_style = get_route_choice_schematic_style(style)
    resolved_text = get_route_choice_schematic_text(text)
    points = _resolve_route_points(resolved_geometry)

    nrows, ncols = resolved_layout.get("subplots", (2, 1))
    fig, axes = plt.subplots(nrows, ncols, figsize=resolved_layout["panel_size"])
    axes = np.ravel(np.asarray(axes))

    for ax in axes:
        ax.set_xlim(*resolved_layout["xlim"])
        ax.set_ylim(*resolved_layout["ylim"])
        ax.set_aspect("equal")
        ax.axis("off")

    row_specs = resolved_text.get("rows", [])
    for ax, row_spec in zip(axes, row_specs, strict=False):
        _draw_route_choice_row(
            ax,
            row_spec,
            points=points,
            labels=resolved_text,
            colors=resolved_colors,
            style=resolved_style,
        )

    if "subplots_adjust" in resolved_layout:
        fig.subplots_adjust(**resolved_layout["subplots_adjust"])
    if resolved_layout.get("tight_layout", False):
        fig.tight_layout()

    schematic_data = {
        "layout": resolved_layout,
        "geometry": resolved_geometry,
        "colors": resolved_colors,
        "style": resolved_style,
        "text": resolved_text,
    }
    saved_paths = save_panel(fig, export_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, schematic_data, saved_paths


def _grid_units_to_data(origin, cell_size, point_units):
    return np.asarray(origin, dtype=float) + float(cell_size) * np.asarray(point_units, dtype=float)


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
            [_grid_units_to_data(origin, cell_size, point) for point in arrow_spec["path_units"]]
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
    pts = np.vstack([_grid_units_to_data(origin, cell_size, point) for point in points_units])
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
    saved_paths = save_panel(fig, export_name, formats=formats, output_dir=output_dir) if save else []
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


def _compute_complete_binary_tree_positions(depth: int, tree_width: float, tree_height: float):
    pos = {}
    n_leaves = 2 ** (depth - 1)
    xs_leaves = np.linspace(0.0, tree_width, n_leaves)
    y_step = tree_height / (depth - 1) if depth > 1 else 0.0
    for i in range(n_leaves):
        pos[(depth - 1, i)] = (xs_leaves[i], 0.0)
    for level in range(depth - 2, -1, -1):
        n_nodes = 2 ** level
        for i in range(n_nodes):
            left_child = (level + 1, 2 * i)
            right_child = (level + 1, 2 * i + 1)
            x = 0.5 * (pos[left_child][0] + pos[right_child][0])
            y = pos[left_child][1] + y_step
            pos[(level, i)] = (x, y)
    return pos


def _horizontal_binary_tree_positions(depth: int, tree_width: float, tree_height: float):
    vertical = _compute_complete_binary_tree_positions(depth, tree_width, tree_height)
    y_center = 0.5 * tree_width
    return {
        key: (tree_height - xy[1], y_center - xy[0])
        for key, xy in vertical.items()
    }


def _level_assignment_palette(level_count: int, colors: dict):
    cmap = plt.get_cmap(colors["cmap_name"])
    denom = max(int(level_count) - 1, 1)
    return [cmap(i / denom) for i in range(int(level_count))]


def _maze_level_grid(maze, *, grid_size: int = 15, level_definition: str | None = None):
    level_masks = get_level_list(level_definition, maze=maze, ma_ce=maze.ce, grid_size=grid_size)
    level_by_cell = np.full(len(maze.xc), -1, dtype=int)
    for level_idx, mask in enumerate(level_masks):
        level_by_cell[np.asarray(mask, dtype=bool)] = level_idx
    level_grid = np.full((grid_size, grid_size), -1, dtype=int)
    for (x, y), cell_id in maze.ce.items():
        level_grid[int(y), int(x)] = int(level_by_cell[int(cell_id)])
    return level_grid, level_by_cell


def _draw_level_assignment_tree(ax, *, geometry: dict, colors: dict, style: dict, text: dict):
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
        level_label_x = root_x - float(geometry["root_line_length"]) + float(level_label.get("dx", 0.0))
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
        edge_t = float(text.get("edge_label_t_by_level", {}).get(level_idx, text.get("edge_label_t", 0.72)))
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
    y_extent = 0.5 * float(geometry["tree_width"]) + float(geometry["tree_vertical_pad"])
    ax.set_ylim(-y_extent, y_extent)
    return positions, palette


def _draw_level_assignment_maze(ax, maze, *, colors: dict, style: dict, grid_size: int = 15, level_definition: str | None = None):
    level_grid, level_by_cell = _maze_level_grid(maze, grid_size=grid_size, level_definition=level_definition)
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
        gridspec_kw={"width_ratios": tuple(resolved_layout.get("width_ratios", (1.0, 1.28)))},
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
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes[0], panel_data, saved_paths


def render_within_session_place_map_correlation_panel(
    *,
    curves_csv,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "within_session_place_map_correlation",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render saved within-session place-map correlation curves."""
    curves_csv = Path(curves_csv)
    if not curves_csv.exists():
        raise FileNotFoundError(f"Curve table not found: {curves_csv}")

    panel_layout = {
        "panel_size": (10.8, 4.0),
        "days": ("day1", "day2"),
        "sharey": True,
        "grid_wspace": 0.25,
        "title_y": 1.02,
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "cmap": "Reds",
        "levels": None,
        "groups": {},
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "line_style": "--",
        "line_width": 2.5,
        "error_alpha": 0.20,
        "legend": True,
        "legend_loc": "best",
        "legend_fontsize": 9,
        "xlim": None,
        "ylim": None,
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "title": "Within-session place map correlation",
        "day_titles": {"day1": "day1", "day2": "day2"},
        "xlabel": "Time in maze (s)",
        "ylabel": "Place map correlation",
        "groups": {},
    }
    if labels:
        panel_labels.update(labels)

    curves = pd.read_csv(curves_csv)
    days = tuple(panel_layout["days"])
    fig, axes = plt.subplots(
        1,
        len(days),
        figsize=panel_size or panel_layout["panel_size"],
        sharey=bool(panel_layout["sharey"]),
        squeeze=False,
    )
    axes = axes[0]
    cmap = plt.get_cmap(str(panel_colors["cmap"]))

    plotted_groups = {}
    for ax, day in zip(axes, days):
        day_curves = curves[curves["day"].astype(str) == str(day)].copy()
        groups = list(dict.fromkeys(day_curves["group"].astype(str)))
        levels = panel_colors["levels"]
        if levels is None:
            levels = np.linspace(0.25, 0.90, max(len(groups), 1)) if len(groups) > 1 else [0.60]
        group_color_lookup = {
            group: panel_colors["groups"].get(group, cmap(float(level)))
            for group, level in zip(groups, levels)
        }
        plotted_groups[str(day)] = groups
        for group in groups:
            subset = day_curves[day_curves["group"].astype(str) == group].sort_values("time_in_maze_s")
            x = subset["time_in_maze_s"].to_numpy(dtype=float)
            y = subset["place_map_correlation_mean"].to_numpy(dtype=float)
            err = subset["place_map_correlation_sem"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue
            color = group_color_lookup[group]
            display_group = panel_labels["groups"].get(group, group)
            ax.plot(
                x[finite],
                y[finite],
                ls=panel_style["line_style"],
                lw=float(panel_style["line_width"]),
                color=color,
                label=display_group,
            )
            band = finite & np.isfinite(err)
            if band.any():
                ax.fill_between(
                    x[band],
                    y[band] - err[band],
                    y[band] + err[band],
                    color=color,
                    alpha=float(panel_style["error_alpha"]),
                    linewidth=0,
                )
        day_titles = panel_labels.get("day_titles", {})
        if str(day) in day_titles:
            ax.set_title(str(day_titles[str(day)]))
        ax.set_xlabel(str(panel_labels["xlabel"]))
        ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
        ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
        if panel_style["xlim"] is not None:
            ax.set_xlim(*panel_style["xlim"])
        if panel_style["ylim"] is not None:
            ax.set_ylim(*panel_style["ylim"])
        if bool(panel_style["legend"]):
            ax.legend(
                loc=str(panel_style["legend_loc"]),
                fontsize=panel_style["legend_fontsize"],
                frameon=False,
            )
    if len(axes):
        axes[0].set_ylabel(str(panel_labels["ylabel"]))
    if panel_labels.get("title"):
        fig.suptitle(str(panel_labels["title"]), y=float(panel_layout["title_y"]))
    fig.subplots_adjust(wspace=float(panel_layout["grid_wspace"]))
    if bool(panel_layout.get("tight_layout", True)):
        fig.tight_layout()

    panel_data = {
        "curves": curves,
        "curves_csv": curves_csv,
        "plotted_groups": plotted_groups,
        "layout": panel_layout,
        "colors": panel_colors,
        "style": panel_style,
        "labels": panel_labels,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def render_experience_trend_panel(
    *,
    points_csv,
    stats_csv,
    metric_specs,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "experience_trends",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render metric-vs-experience scatter plots from saved point/stat tables."""
    points_csv = Path(points_csv)
    stats_csv = Path(stats_csv)
    if not points_csv.exists():
        raise FileNotFoundError(f"Experience trend point table not found: {points_csv}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Experience trend stats table not found: {stats_csv}")

    panel_layout = {
        "panel_size": (16.0, 3.4),
        "grid_wspace": 0.35,
        "ncols": len(metric_specs),
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "points": "#DA2323",
        "fit": "black",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "point_size": 38,
        "point_alpha": 0.70,
        "point_rasterized": False,
        "line_width": 2.0,
        "line_style": "--",
        "line_alpha": 0.70,
        "annotation": True,
        "annotation_x": 0.05,
        "annotation_y": 0.95,
        "annotation_fontsize": 9,
        "annotation_p_col": "selected_q_bh_two_sided",
        "annotation_p_label": "cluster q",
        "annotation_r2_col": "r2",
        "annotation_r2_label": "OLS R2",
        "annotation_show_stars": True,
        "xlim": None,
        "ylims": {},
        "xticks": [1, 2, 3, 4],
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "xlabel": "# of experiences",
        "metric_labels": {},
        "titles": {},
    }
    if labels:
        panel_labels.update(labels)

    points = pd.read_csv(points_csv)
    stats_df = pd.read_csv(stats_csv)
    selected_metrics = [spec["metric"] if isinstance(spec, dict) else spec[0] for spec in metric_specs]
    selected_p = []
    for metric in selected_metrics:
        metric_stats = stats_df.loc[stats_df["metric"].astype(str).eq(str(metric))]
        if metric_stats.empty or "p_two_sided" not in metric_stats:
            selected_p.append(np.nan)
        else:
            selected_p.append(float(metric_stats.iloc[0]["p_two_sided"]))
    selected_q_lookup = dict(zip(selected_metrics, _bh_adjust_pvalues(selected_p)))
    stats_df = stats_df.copy()
    stats_df["selected_q_bh_two_sided"] = [
        selected_q_lookup.get(str(metric), np.nan) for metric in stats_df["metric"].astype(str)
    ]
    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes[0]

    for ax, spec in zip(axes, metric_specs):
        spec_dict = spec if isinstance(spec, dict) else {}
        metric = spec_dict["metric"] if spec_dict else spec[0]
        ylabel = (
            spec_dict.get("ylabel")
            if spec_dict.get("ylabel") is not None
            else panel_labels["metric_labels"].get(metric, metric)
        )
        title = spec_dict.get("title") if spec_dict else panel_labels["titles"].get(metric, "")
        point_color = spec_dict.get("point_color") or spec_dict.get("color") or panel_colors["points"]
        fit_color = spec_dict.get("fit_color") or panel_colors["fit"]
        metric_points = points.loc[points["metric"].astype(str).eq(str(metric))].copy()
        metric_stats = stats_df.loc[stats_df["metric"].astype(str).eq(str(metric))]
        if metric_points.empty or metric_stats.empty:
            raise ValueError(f"No saved experience-trend rows/statistics for metric {metric!r}")
        metric_stats = metric_stats.iloc[0]

        x = metric_points["experience"].to_numpy(dtype=float)
        y = metric_points["value"].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        ax.scatter(
            x,
            y,
            c=point_color,
            s=float(panel_style["point_size"]),
            alpha=float(panel_style["point_alpha"]),
            zorder=3,
        )

        slope = float(metric_stats["slope"])
        intercept = float(metric_stats["intercept"])
        if x.size and np.isfinite(slope) and np.isfinite(intercept):
            x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ax.plot(
                x_fit,
                intercept + slope * x_fit,
                color=fit_color,
                linestyle=panel_style["line_style"],
                linewidth=float(panel_style["line_width"]),
                alpha=float(panel_style["line_alpha"]),
            )

        if bool(panel_style["annotation"]):
            p_col = str(panel_style["annotation_p_col"])
            r2_col = str(panel_style["annotation_r2_col"])
            p_value = float(metric_stats[p_col]) if p_col in metric_stats and pd.notna(metric_stats[p_col]) else np.nan
            r2_value = float(metric_stats[r2_col]) if r2_col in metric_stats and pd.notna(metric_stats[r2_col]) else np.nan
            stars = f" {_significance_text(p_value)}" if bool(panel_style["annotation_show_stars"]) else ""
            ax.text(
                float(panel_style["annotation_x"]),
                float(panel_style["annotation_y"]),
                f"{panel_style['annotation_p_label']} = {p_value:.2g}{stars}\n"
                f"{panel_style['annotation_r2_label']} = {r2_value:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=panel_style["annotation_fontsize"],
            )

        if title:
            ax.set_title(str(title))
        ax.set_xlabel(str(panel_labels["xlabel"]))
        ax.set_ylabel(str(ylabel))
        if panel_style["xticks"] is not None:
            ax.set_xticks(panel_style["xticks"])
        if panel_style["xlim"] is not None:
            ax.set_xlim(*panel_style["xlim"])
        ylims = panel_style.get("ylims", {})
        if metric in ylims:
            ax.set_ylim(*ylims[metric])
        ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
        ax.spines["right"].set_visible(bool(panel_style["spine_right"]))

    fig.subplots_adjust(wspace=float(panel_layout["grid_wspace"]))
    if bool(panel_layout.get("tight_layout", True)):
        fig.tight_layout()

    panel_data = {
        "points": points,
        "stats": stats_df,
        "points_csv": points_csv,
        "stats_csv": stats_csv,
        "metric_specs": metric_specs,
        "layout": panel_layout,
        "colors": panel_colors,
        "style": panel_style,
        "labels": panel_labels,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def render_support_matched_sensitivity_panel(
    *,
    points_csv,
    stats_csv,
    metric_specs,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "support_matched_sensitivity",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render support-matched sensitivity summaries from saved point/stat tables."""
    points_csv = Path(points_csv)
    stats_csv = Path(stats_csv)
    if not points_csv.exists():
        raise FileNotFoundError(f"Support-matched point table not found: {points_csv}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Support-matched stats table not found: {stats_csv}")

    panel_layout = {
        "panel_size": (13.5, 3.6),
        "grid_wspace": 0.25,
        "ncols": len(metric_specs),
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "metrics": get_correlation_measure_colors(),
        "individual_line": "#d0d0d0",
        "individual_edge": "white",
        "mean": "black",
        "descriptive_band": "#f0f0f0",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "line_width": 1.0,
        "line_alpha": 0.6,
        "point_size": 36,
        "point_alpha": 0.95,
        "point_edge_width": 0.5,
        "mean_line_width": 1.6,
        "mean_marker_size": 10.0,
        "capsize": 3.0,
        "shade_descriptive": True,
        "descriptive_xspan": (3.5, 4.5),
        "annotation": True,
        "annotation_x": 0.04,
        "annotation_y": 0.96,
        "annotation_fontsize": 8.5,
        "annotation_p_col": "p_two_sided",
        "annotation_q_col": "q_bh_two_sided",
        "annotation_show_q": True,
        "annotation_use_q_for_stars": True,
        "xticks": [0, 1, 2, 3, 4],
        "xticklabels": ["Exp0", "Exp1", "Exp2", "Exp3", "Exp4"],
        "spine_top": False,
        "spine_right": False,
        "grid_alpha": 0.2,
        "ylims": {},
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "xlabel": "",
        "ylabel": "Correlation",
        "titles": {},
    }
    if labels:
        panel_labels.update(labels)

    points_df = pd.read_csv(points_csv)
    stats_df = pd.read_csv(stats_csv)
    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes[0]

    for idx, (ax, spec) in enumerate(zip(axes, metric_specs)):
        metric = spec["metric"] if isinstance(spec, dict) else spec[0]
        metric_title = (
            spec.get("title")
            if isinstance(spec, dict) and spec.get("title") is not None
            else panel_labels.get("titles", {}).get(metric, "")
        )
        ylabel = (
            spec.get("ylabel")
            if isinstance(spec, dict) and spec.get("ylabel") is not None
            else panel_labels.get("ylabel", "Correlation")
        )
        metric_points = points_df.loc[points_df["metric"].astype(str).eq(str(metric))].copy()
        if metric_points.empty:
            raise ValueError(f"No support-matched rows found for metric {metric!r}")
        metric_points["experience_index"] = metric_points["experience_index"].astype(int)
        metric_points["animal_id"] = metric_points["animal_id"].astype(str)
        metric_stats = stats_df.loc[stats_df["metric"].astype(str).eq(str(metric))]
        metric_stats_row = metric_stats.iloc[0] if not metric_stats.empty else pd.Series(dtype=float)

        metric_color = (
            spec.get("color")
            if isinstance(spec, dict) and spec.get("color") is not None
            else panel_colors["metrics"].get(str(metric), REPRESENTATION_COLORS["ensemble_rate"])
        )

        if bool(panel_style["shade_descriptive"]) and (
            "descriptive_only" in metric_points.columns and metric_points["descriptive_only"].fillna(False).astype(bool).any()
        ):
            ax.axvspan(
                float(panel_style["descriptive_xspan"][0]),
                float(panel_style["descriptive_xspan"][1]),
                color=str(panel_colors["descriptive_band"]),
                alpha=0.65,
                zorder=0,
            )

        for _, animal_points in metric_points.groupby("animal_id", sort=False):
            animal_points = animal_points.sort_values("experience_index")
            x = animal_points["experience_index"].to_numpy(dtype=float)
            y = animal_points["value"].to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            if not keep.any():
                continue
            ax.plot(
                x[keep],
                y[keep],
                color=str(panel_colors["individual_line"]),
                alpha=float(panel_style["line_alpha"]),
                linewidth=float(panel_style["line_width"]),
                zorder=1,
            )
            ax.scatter(
                x[keep],
                y[keep],
                s=float(panel_style["point_size"]),
                color=metric_color,
                edgecolor=str(panel_colors["individual_edge"]),
                linewidth=float(panel_style["point_edge_width"]),
                alpha=float(panel_style["point_alpha"]),
                zorder=2,
            )

        means = metric_points.groupby("experience_index")["value"].mean()
        sems = metric_points.groupby("experience_index")["value"].sem()
        xvals = means.index.to_numpy(dtype=float)
        yvals = means.to_numpy(dtype=float)
        yerr = sems.reindex(means.index).to_numpy(dtype=float)
        ax.errorbar(
            xvals,
            yvals,
            yerr=yerr,
            color=str(panel_colors["mean"]),
            marker="o",
            linewidth=float(panel_style["mean_line_width"]),
            markersize=float(panel_style["mean_marker_size"]),
            capsize=float(panel_style["capsize"]),
            zorder=3,
        )

        if bool(panel_style["annotation"]) and not metric_stats.empty:
            p_col = str(panel_style["annotation_p_col"])
            q_col = str(panel_style["annotation_q_col"])
            p_value = float(metric_stats_row[p_col]) if p_col in metric_stats_row and pd.notna(metric_stats_row[p_col]) else np.nan
            q_value = float(metric_stats_row[q_col]) if q_col in metric_stats_row and pd.notna(metric_stats_row[q_col]) else np.nan
            star_value = q_value if bool(panel_style["annotation_use_q_for_stars"]) and np.isfinite(q_value) else p_value
            lines = [f"p = {p_value:.2g}{' ' + _significance_text(star_value) if np.isfinite(star_value) else ''}"]
            if bool(panel_style["annotation_show_q"]):
                lines.append(f"q = {q_value:.2g}" if np.isfinite(q_value) else "q = nan")
            ax.text(
                float(panel_style["annotation_x"]),
                float(panel_style["annotation_y"]),
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=float(panel_style["annotation_fontsize"]),
            )

        ax.set_title(str(metric_title) if metric_title else str(metric))
        ax.set_xlabel(str(panel_labels["xlabel"]))
        if idx == 0:
            ax.set_ylabel(str(ylabel))
        else:
            ax.set_ylabel("")
        if panel_style["xticks"] is not None:
            ax.set_xticks(panel_style["xticks"], panel_style.get("xticklabels"))
        ylims = panel_style.get("ylims", {})
        if metric in ylims:
            ax.set_ylim(*ylims[metric])
        ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
        ax.spines["right"].set_visible(bool(panel_style["spine_right"]))
        ax.grid(alpha=float(panel_style["grid_alpha"]))

    fig.subplots_adjust(wspace=float(panel_layout["grid_wspace"]))
    if bool(panel_layout.get("tight_layout", True)):
        fig.tight_layout()

    panel_data = {
        "points": points_df,
        "stats": stats_df,
        "metric_specs": metric_specs,
        "layout": panel_layout,
        "colors": panel_colors,
        "style": panel_style,
        "labels": panel_labels,
        "points_csv": points_csv,
        "stats_csv": stats_csv,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


def render_consecutive_gap_correlation_persistence_panel(
    *,
    points_csv,
    stats_csv,
    metric_specs,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "consecutive_gap_correlation_persistence",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a saved 1xN persistence row for consecutive-gap metrics."""
    points_csv = Path(points_csv)
    stats_csv = Path(stats_csv)
    if not points_csv.exists():
        raise FileNotFoundError(f"Consecutive-gap persistence point table not found: {points_csv}")
    if not stats_csv.exists():
        raise FileNotFoundError(f"Consecutive-gap persistence stats table not found: {stats_csv}")

    panel_layout = {
        "panel_size": (16.0, 3.4),
        "grid_wspace": 0.35,
        "ncols": len(metric_specs),
    }
    if layout:
        panel_layout.update(layout)
    panel_colors = {
        "points": "#DA2323",
        "fit": "black",
        "identity": "#999999",
    }
    if colors:
        panel_colors.update(colors)
    panel_style = {
        "point_size": 38,
        "point_alpha": 0.70,
        "plot_kind": "scatter",
        "density_gridsize": 42,
        "density_min_count": 1,
        "density_norm": "linear",
        "density_power_gamma": 0.5,
        "density_vmax_quantile": None,
        "density_alpha": 1.0,
        "density_linewidth": 0.0,
        "show_fit_line": True,
        "line_width": 2.0,
        "line_style": "--",
        "line_alpha": 0.70,
        "show_identity_line": False,
        "identity_line_style": ":",
        "identity_line_width": 1.0,
        "identity_line_alpha": 0.7,
        "auto_limit_padding_fraction": 0.05,
        "annotation": True,
        "annotation_x": 0.05,
        "annotation_y": 0.95,
        "annotation_fontsize": 9,
        "annotation_p_col": "q_bh_two_sided",
        "annotation_p_label": "cluster q",
        "annotation_r2_col": "r2",
        "annotation_r2_label": "OLS R2",
        "annotation_show_stars": True,
        "spine_top": False,
        "spine_right": False,
    }
    if style:
        panel_style.update(style)
    panel_labels = {
        "xlabel": "Previous 3-month gap correlation",
        "ylabel": "Next 3-month gap correlation",
        "metric_labels": {},
        "titles": {},
    }
    if labels:
        panel_labels.update(labels)

    points = pd.read_csv(points_csv)
    stats_df = pd.read_csv(stats_csv)
    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=panel_size or panel_layout["panel_size"],
        squeeze=False,
    )
    axes = axes[0]

    def _padded_limits(values, *, pad_fraction: float) -> tuple[float, float]:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return (-1.0, 1.0)
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if np.isclose(lo, hi):
            center = lo
            pad = max(abs(center) * float(pad_fraction), 0.05)
            return (center - pad, center + pad)
        pad = (hi - lo) * float(pad_fraction)
        return (lo - pad, hi + pad)

    for ax, spec in zip(axes, metric_specs, strict=False):
        metric = spec["metric"] if isinstance(spec, dict) else spec[0]
        title = spec.get("title") if isinstance(spec, dict) else panel_labels["titles"].get(metric, "")
        xlabel = (
            spec.get("xlabel")
            if isinstance(spec, dict) and spec.get("xlabel") is not None
            else panel_labels["xlabel"]
        )
        ylabel = (
            spec.get("ylabel")
            if isinstance(spec, dict) and spec.get("ylabel") is not None
            else panel_labels["metric_labels"].get(metric, panel_labels["ylabel"])
        )
        x_col = (
            spec.get("x_col")
            if isinstance(spec, dict) and spec.get("x_col") is not None
            else "prev_gap_correlation"
        )
        y_col = (
            spec.get("y_col")
            if isinstance(spec, dict) and spec.get("y_col") is not None
            else "next_gap_correlation"
        )
        point_color = (
            spec.get("point_color")
            if isinstance(spec, dict) and spec.get("point_color") is not None
            else panel_colors["points"]
        )
        fit_color = (
            spec.get("fit_color")
            if isinstance(spec, dict) and spec.get("fit_color") is not None
            else panel_colors["fit"]
        )
        identity_color = (
            spec.get("identity_color")
            if isinstance(spec, dict) and spec.get("identity_color") is not None
            else panel_colors["identity"]
        )
        plot_kind = (
            str(spec.get("plot_kind"))
            if isinstance(spec, dict) and spec.get("plot_kind") is not None
            else str(panel_style.get("plot_kind", "scatter"))
        ).lower()
        density_gridsize = (
            spec.get("density_gridsize")
            if isinstance(spec, dict) and spec.get("density_gridsize") is not None
            else panel_style.get("density_gridsize", 42)
        )
        density_min_count = (
            spec.get("density_min_count")
            if isinstance(spec, dict) and spec.get("density_min_count") is not None
            else panel_style.get("density_min_count", 1)
        )
        density_norm = (
            str(spec.get("density_norm"))
            if isinstance(spec, dict) and spec.get("density_norm") is not None
            else str(panel_style.get("density_norm", "linear"))
        ).lower()
        density_power_gamma = (
            spec.get("density_power_gamma")
            if isinstance(spec, dict) and spec.get("density_power_gamma") is not None
            else panel_style.get("density_power_gamma", 0.5)
        )
        density_vmax_quantile = (
            spec.get("density_vmax_quantile")
            if isinstance(spec, dict) and spec.get("density_vmax_quantile") is not None
            else panel_style.get("density_vmax_quantile")
        )

        metric_points = points.loc[points["metric"].astype(str).eq(str(metric))].copy()
        metric_stats = stats_df.loc[stats_df["metric"].astype(str).eq(str(metric))]
        if metric_points.empty or metric_stats.empty:
            raise ValueError(f"No saved persistence rows/statistics for metric {metric!r}")
        metric_stats = metric_stats.iloc[0]

        x = metric_points[x_col].to_numpy(dtype=float)
        y = metric_points[y_col].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        if plot_kind in {"hexbin", "density"}:
            density_cmap = LinearSegmentedColormap.from_list(
                f"persistence_density_{metric}",
                ["#ffffff", point_color],
            )
            hexbin_kwargs = {}
            if density_norm in {"log", "log10", "logarithmic"}:
                hexbin_kwargs["norm"] = LogNorm(vmin=1.0)
            elif density_norm in {"sqrt", "power", "powernorm"}:
                hexbin_kwargs["norm"] = PowerNorm(gamma=float(density_power_gamma), vmin=1.0)
            hexbin = ax.hexbin(
                x,
                y,
                gridsize=int(density_gridsize),
                mincnt=int(density_min_count),
                cmap=density_cmap,
                linewidths=float(panel_style.get("density_linewidth", 0.0)),
                alpha=float(panel_style.get("density_alpha", 1.0)),
                zorder=1.5,
                rasterized=bool(panel_style.get("point_rasterized", False)),
                **hexbin_kwargs,
            )
            if density_vmax_quantile is not None:
                counts = np.asarray(hexbin.get_array(), dtype=float)
                counts = counts[np.isfinite(counts)]
                if counts.size:
                    vmax = float(np.nanquantile(counts, float(density_vmax_quantile)))
                    if np.isfinite(vmax) and vmax > 1.0:
                        hexbin.set_clim(vmin=1.0, vmax=vmax)
        else:
            ax.scatter(
                x,
                y,
                c=point_color,
                s=float(panel_style["point_size"]),
                alpha=float(panel_style["point_alpha"]),
                rasterized=bool(panel_style.get("point_rasterized", False)),
                zorder=3,
            )

        slope = float(metric_stats["slope"])
        intercept = float(metric_stats["intercept"])
        if bool(panel_style.get("show_fit_line", True)) and x.size and np.isfinite(slope) and np.isfinite(intercept):
            x_fit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ax.plot(
                x_fit,
                intercept + slope * x_fit,
                color=fit_color,
                linestyle=panel_style["line_style"],
                linewidth=float(panel_style["line_width"]),
                alpha=float(panel_style["line_alpha"]),
                zorder=2,
            )

        x_limits = (
            tuple(spec["xlim"])
            if isinstance(spec, dict) and spec.get("xlim") is not None
            else _padded_limits(x, pad_fraction=float(panel_style["auto_limit_padding_fraction"]))
        )
        y_limits = (
            tuple(spec["ylim"])
            if isinstance(spec, dict) and spec.get("ylim") is not None
            else _padded_limits(y, pad_fraction=float(panel_style["auto_limit_padding_fraction"]))
        )

        if bool(panel_style.get("show_identity_line", False)) and x.size:
            lo = max(float(x_limits[0]), float(y_limits[0]))
            hi = min(float(x_limits[1]), float(y_limits[1]))
            if hi > lo:
                ax.plot(
                    [lo, hi],
                    [lo, hi],
                    color=identity_color,
                    linestyle=str(panel_style["identity_line_style"]),
                    linewidth=float(panel_style["identity_line_width"]),
                    alpha=float(panel_style["identity_line_alpha"]),
                    zorder=1,
                )

        if bool(panel_style["annotation"]):
            p_col = str(panel_style["annotation_p_col"])
            r2_col = str(panel_style["annotation_r2_col"])
            p_value = float(metric_stats[p_col]) if p_col in metric_stats and pd.notna(metric_stats[p_col]) else np.nan
            r2_value = float(metric_stats[r2_col]) if r2_col in metric_stats and pd.notna(metric_stats[r2_col]) else np.nan
            stars = f" {_significance_text(p_value)}" if bool(panel_style["annotation_show_stars"]) else ""
            ax.text(
                float(panel_style["annotation_x"]),
                float(panel_style["annotation_y"]),
                f"{panel_style['annotation_p_label']} = {p_value:.2g}{stars}\n"
                f"{panel_style['annotation_r2_label']} = {r2_value:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=panel_style["annotation_fontsize"],
            )

        if title:
            ax.set_title(str(title))
        ax.set_xlabel(str(xlabel))
        ax.set_ylabel(str(ylabel))
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        if isinstance(spec, dict) and spec.get("xticks") is not None:
            ax.set_xticks(spec["xticks"])
        if isinstance(spec, dict) and spec.get("yticks") is not None:
            ax.set_yticks(spec["yticks"])
        ax.spines["top"].set_visible(bool(panel_style["spine_top"]))
        ax.spines["right"].set_visible(bool(panel_style["spine_right"]))

    fig.subplots_adjust(wspace=float(panel_layout["grid_wspace"]))
    if bool(panel_layout.get("tight_layout", True)):
        fig.tight_layout()

    panel_data = {
        "points": points,
        "stats": stats_df,
        "points_csv": points_csv,
        "stats_csv": stats_csv,
        "metric_specs": metric_specs,
        "layout": panel_layout,
        "colors": panel_colors,
        "style": panel_style,
        "labels": panel_labels,
    }
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
    return fig, axes, panel_data, saved_paths


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
    draw_gray_blocks(ax, panel_size=panel_size, blocks=gray_blocks, layout=layout, colors=colors)
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
    panel_layout = get_effective_panel_layout(panel_size=panel_size, layout=layout, scale_visuals=False)
    if panel_layout.get("maze_xlim") is not None:
        ax.set_xlim(*panel_layout["maze_xlim"])
    if panel_layout.get("maze_ylim") is not None:
        ax.set_ylim(*panel_layout["maze_ylim"])
    finish_panel(ax)
    saved_paths = save_panel(fig, panel_name, formats=formats, output_dir=output_dir) if save else []
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


def compute_example_cell_placemap(
    pack,
    side,
    *,
    n_bins=15,
    sigma=1.0,
    signal="Craw",
    dt=0.04,
    min_time_base=5.0,
    filter_ratemap=True,
):
    """Rebuild a smoothed placemap and occupancy (in seconds) for one side.

    ``pack`` is a dict as returned by :func:`load_neuron_data_bundle`.
    ``side`` is ``"A"`` (earlier session) or ``"B"`` (later session).
    ``signal`` selects the per-frame trace: ``"Craw"`` (raw calcium),
    ``"C"`` (denoised), or ``"S"`` (deconvolved spikes).

    The occupancy threshold is ``min_time_base * (15/n_bins)**2`` seconds,
    scaling the pipeline's 5 s default at 15x15 to finer grids. Bins with
    occupancy below the threshold are zeroed so ``occ_sec > 0`` gives the
    retained-bin mask. Outputs have shape ``(n_bins, n_bins)`` with
    rows=iy, cols=ix (matches ``imshow(origin='upper')``).
    """
    trace_key = f"{side}_{signal}"
    if trace_key not in pack:
        raise KeyError(
            f"signal {signal!r} not present in bundle (missing {trace_key!r})"
        )
    frame = np.asarray(pack[f"{side}_frame"], dtype=int)
    pos = np.column_stack(
        [np.asarray(pack[f"{side}_mass_x"], dtype=float),
         np.asarray(pack[f"{side}_mass_y"], dtype=float)]
    )
    labels, _, _ = discretize_positions(pos, int(n_bins), int(n_bins))
    counts = np.asarray(pack[trace_key], dtype=float)[frame]
    ok = ~np.isnan(counts)
    counts = counts[ok]
    labels = labels[ok].astype(int)
    eff_min_time = float(min_time_base) * (15.0 / float(n_bins)) ** 2
    lambdas, _ = compute_tuning_and_prior(
        counts[:, None], labels,
        n_bins=int(n_bins) * int(n_bins),
        dt=float(dt), filter_ratemap=bool(filter_ratemap), sigma=float(sigma),
        min_time=eff_min_time,
        spatial_shape=(int(n_bins), int(n_bins)),
    )
    occ_sec = np.bincount(labels, minlength=int(n_bins) * int(n_bins)).astype(float) * float(dt)
    occ_sec[occ_sec < eff_min_time] = 0.0
    pm = lambdas[:, 0].reshape(int(n_bins), int(n_bins))
    occ_sec = occ_sec.reshape(int(n_bins), int(n_bins))
    return pm, occ_sec


def _welch_sn_from_trace(y, range_ff=(0.25, 0.5)):
    """Welch-PSD noise estimate used to threshold deconvolved spikes.

    Mirrors the ``_welch_sn`` helper in ``example_neuron_detail.ipynb`` so
    the arrow panel matches the notebook's spike picks.
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)].reshape(-1, 1)
    if y.size == 0:
        return float("nan")
    ff, psd = welch(y, nperseg=None, axis=0, fs=1.0)
    m = (ff >= range_ff[0]) & (ff <= range_ff[1])
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.exp(np.mean(np.log(psd[m, :] / 2)))))


def compute_example_cell_arrow_data(
    pack,
    side,
    *,
    threshold_sn: float = 5.0,
    dedupe_frames: int = 4,
    spike_signal: str = "S",
    noise_from: str = "Craw",
):
    """Return trajectory + arrow endpoints for one session side.

    ``pack`` is a dict as returned by :func:`load_neuron_data_bundle`.
    Positions are converted to grid-bin coordinates with ``y`` flipped
    (``(1 - y) * 15 - 0.5``) so they overlay the maze walls drawn in the
    canonical ``MAZE.wa`` frame (matplotlib y-up, no ``origin='upper'``).

    Spike frames are selected by thresholding ``{side}_{spike_signal}`` at
    ``threshold_sn * sn`` where ``sn`` is Welch-PSD noise on
    ``{side}_{noise_from}``. Consecutive spikes within ``dedupe_frames``
    are collapsed to one (matches the per-spike draw the detail notebook
    uses).

    Returns a dict with ``traj_x, traj_y``; arrow endpoints ``px, py``
    (body centroid) and ``ex, ey`` (nose); ``n_spikes``, ``sn``,
    ``threshold``.
    """
    frame = np.asarray(pack[f"{side}_frame"], dtype=int)
    x = np.asarray(pack[f"{side}_x"], dtype=float)
    y = np.asarray(pack[f"{side}_y"], dtype=float)
    mass_x = np.asarray(pack[f"{side}_mass_x"], dtype=float)
    mass_y = np.asarray(pack[f"{side}_mass_y"], dtype=float)

    traj_x = x * 15.0 - 0.5
    traj_y = (1.0 - y) * 15.0 - 0.5

    spike_key = f"{side}_{spike_signal}"
    noise_key = f"{side}_{noise_from}"
    if spike_key not in pack or noise_key not in pack:
        raise KeyError(
            f"bundle is missing {spike_key!r} or {noise_key!r} for the arrow panel"
        )
    s_row = np.asarray(pack[spike_key], dtype=float)
    noise_row = np.asarray(pack[noise_key], dtype=float)
    sn = _welch_sn_from_trace(noise_row)
    threshold = float(threshold_sn) * sn if np.isfinite(sn) else float("inf")
    spike_all_frames = np.where(s_row > threshold)[0]

    # Keep only spikes that occurred during the speed-thresholded window
    # for which we saved positions, preserving order.
    sel_set = dict(zip(frame.tolist(), np.arange(frame.size, dtype=int)))
    sel_idx = [sel_set[fr] for fr in spike_all_frames.tolist() if fr in sel_set]
    sel_idx = np.asarray(sel_idx, dtype=int)

    # Dedupe consecutive spikes falling within ``dedupe_frames``.
    if sel_idx.size:
        frame_hits = frame[sel_idx]
        keep = np.concatenate(([True], np.diff(frame_hits) >= int(dedupe_frames)))
        sel_idx = sel_idx[keep]

    px = mass_x[sel_idx] * 15.0 - 0.5
    py = (1.0 - mass_y[sel_idx]) * 15.0 - 0.5
    ex = x[sel_idx] * 15.0 - 0.5
    ey = (1.0 - y[sel_idx]) * 15.0 - 0.5

    return {
        "traj_x": traj_x, "traj_y": traj_y,
        "px": px, "py": py, "ex": ex, "ey": ey,
        "n_spikes": int(sel_idx.size),
        "sn": float(sn) if np.isfinite(sn) else float("nan"),
        "threshold": float(threshold) if np.isfinite(threshold) else float("nan"),
    }


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
    mode: str = "placemap",
    n_bins: int = 15,
    sigma: float = 1.0,
    signal: str = "Craw",
    dt: float = 0.04,
    min_time_base: float = 5.0,
    maze=None,
    maze_id: int = DEFAULT_MAZE_ID,
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
    """Render a ``2 x N`` grid of example-cell panels from saved bundles.

    Set ``mode="placemap"`` (the default) for rate-map pairs, or
    ``mode="arrow"`` for body-to-nose arrow maps on the nose trajectory.
    Columns correspond to ``picks`` (left-to-right); top row is session A
    (earlier), bottom row session B (later).

    Placemap mode uses :func:`compute_example_cell_placemap` (same engine
    as the cross-session correlation pipeline). At 15x15 the mask defaults
    to the pipeline's cached pair mask when present in the bundle;
    otherwise it falls back to the per-session occupancy intersection.

    Arrow mode uses :func:`compute_example_cell_arrow_data`: it thresholds
    the deconvolved-spike trace against Welch-PSD noise on the raw-calcium
    trace (thresholds configurable via ``style``), plots one short arrow
    per retained spike from the animal's body centroid to its nose, and
    underlays the full selected-frame nose trajectory. None of
    ``n_bins, sigma, signal, dt, min_time_base, prefer_cached_mask`` apply
    to arrow mode -- they are simply ignored.

    Parameters
    ----------
    picks : iterable of ``(pair_key, row_idx)``
        Columns in render order.
    saved_dir : str or Path
        Directory containing the saved ``.npz`` bundles.
    mode : ``"placemap"`` | ``"arrow"``
        Panel style. Defaults to ``"placemap"``.
    n_bins, sigma, signal, dt, min_time_base
        Forwarded to :func:`compute_example_cell_placemap` in placemap
        mode only.
    maze, maze_id
        Maze used for the wall overlay. Pass ``maze`` to reuse an existing
        instance; otherwise ``NewMaze(maze_id)`` is built.

    ``colors``
        Placemap: ``{"cmap", "maze_line"}``.
        Arrow:    ``{"maze_line", "trajectory", "arrow_face", "arrow_edge"}``.
        Entries are merged over mode-specific defaults.

    ``style``
        Placemap knobs: ``normalize`` (``"per_panel"`` | ``"per_column"``
        | ``"global"`` | ``"none"``, default ``"per_panel"``),
        ``interpolation``, ``vmin``, ``maze_linewidth``, ``show_maze``,
        ``prefer_cached_mask``, and the shared title / row-label / layout
        knobs below.
        Arrow knobs: ``threshold_sn`` (default 5.0), ``dedupe_frames``
        (default 4), ``spike_signal`` (default ``"S"``), ``noise_from``
        (default ``"Craw"``), ``show_trajectory``, ``trajectory_alpha``,
        ``trajectory_linewidth``, ``trajectory_zorder``,
        ``arrow_head_width``, ``arrow_head_length``, ``arrow_linewidth``,
        ``arrow_alpha``, ``arrow_scale`` (fraction of the body->nose
        vector drawn; default 0.25), ``arrow_zorder``,
        ``axis_lim`` (grid-bin x/y limits), ``maze_linewidth``,
        ``show_maze``, plus the shared knobs.
        Shared knobs (both modes): ``show_frame``, ``title_fontsize``,
        ``title_pad``, ``title_location`` (``"top"`` or ``"bottom"``),
        ``title_y`` for bottom titles in axes coordinates,
        ``row_label_fontsize``, ``row_label_pad``, ``tight_layout``.

    ``labels``
        ``row_top`` / ``row_bottom`` (``None`` to suppress),
        ``title_template`` -- a ``str.format`` template resolved against
        the bundle's metadata scalars (``pair_key``, ``row_idx``, ``src``,
        ``dst``, ``animal``, ``cohort``, ``cell_a``, ``cell_b``,
        ``place_r``, ``level_r``, ``bc``, ``overlap``, ``logratio``,
        ``sn_a``, ``sn_b``, ``dt``). Pass ``None`` to suppress titles.

    Returns
    -------
    fig, axes, panel_data, saved_paths
        ``axes`` has shape ``(2, N)``. In placemap mode each ``panel_data``
        entry carries ``pm_a``, ``pm_b``, ``mask``, ``vmax``, plus
        metrics and ``bundle_path``. In arrow mode each entry carries
        ``arrows_a`` and ``arrows_b`` dicts (trajectory + arrow endpoints
        from :func:`compute_example_cell_arrow_data`), plus metrics and
        ``bundle_path``.
    """
    mode = str(mode).lower()
    if mode not in ("placemap", "arrow"):
        raise ValueError(
            f"mode must be 'placemap' or 'arrow'; got {mode!r}"
        )

    picks = list(picks)
    n_cols = len(picks)
    if n_cols == 0:
        raise ValueError("picks must be non-empty")

    panel_layout = {
        "panel_size": (2.3 * max(n_cols, 1), 5.4),
        "grid_wspace": 0.02,
        "grid_hspace": 0.05,
        # Explicit outer margins used when ``style['tight_layout']`` is False
        # (the default). Kept tight so the place maps dominate the panel.
        "margins": {"left": 0.04, "right": 0.995, "top": 0.86, "bottom": 0.02},
    }
    if layout:
        panel_layout.update(layout)

    if mode == "placemap":
        panel_colors = {"cmap": "viridis", "maze_line": "w"}
    else:  # arrow
        panel_colors = {
            "maze_line": "k",
            "trajectory": "0.55",
            "arrow_face": "red",
            "arrow_edge": "red",
        }
    if colors:
        panel_colors.update(colors)

    if mode == "placemap":
        panel_style = {
            # Normalization mode for the color axis:
            #   "per_panel"  - each of the 2*N maps scaled to its own max
            #   "per_column" - A and B of a column share the column max
            #   "global"     - all panels share the grid-wide max
            #   "none"       - raw rates (matplotlib autoscales each panel)
            # ``per_column_vmax`` is accepted as a legacy alias (True -> per_column).
            "normalize": "per_panel",
            "interpolation": "nearest",
            "vmin": 0.0,
            "maze_linewidth": 0.9,
            "show_maze": True,
            "show_frame": False,   # hide the axes spines (the black square)
            "prefer_cached_mask": True,
            "title_fontsize": None,
            "title_pad": 4,
            "title_location": "top",
            "title_y": -0.10,
            "row_label_fontsize": None,
            "row_label_pad": None,
            # When False (default), uses ``layout['margins']`` + wspace/hspace
            # via ``fig.subplots_adjust`` so gaps are honored exactly. Set to
            # True to let matplotlib's ``fig.tight_layout()`` auto-balance.
            "tight_layout": False,
        }
    else:  # arrow
        panel_style = {
            "threshold_sn": 5.0,
            "dedupe_frames": 4,
            "spike_signal": "S",       # saved-bundle key suffix: "S" | "C" | "Craw"
            "noise_from": "Craw",      # saved-bundle key suffix for Welch
            "show_trajectory": True,
            "trajectory_alpha": 0.28,
            "trajectory_linewidth": 0.3,
            "trajectory_zorder": 1,
            "arrow_head_width": 0.15,
            "arrow_head_length": 0.4,
            "arrow_linewidth": 0.01,
            "arrow_alpha": 0.4,
            "arrow_scale": 0.25,       # fraction of body->nose vector drawn as arrow
            "arrow_zorder": 2,
            "axis_lim": (-0.5, 14.5),  # grid-bin limits; matches the *15-0.5 mapping
            "maze_linewidth": 1.5,
            "show_maze": True,
            "show_frame": False,
            "prefer_cached_mask": False,  # unused here; kept for API parity
            "title_fontsize": None,
            "title_pad": 4,
            "title_location": "top",
            "title_y": -0.10,
            "row_label_fontsize": None,
            "row_label_pad": None,
            "tight_layout": False,
        }
    if style:
        panel_style.update(style)
    panel_labels = {
        "row_top": "Day i",
        "row_bottom": "Day i + 90",
        "title_template": (
            "{src} → {dst}\n"
            "row {row_idx}  A={cell_a} B={cell_b}\n"
            "place={place_r:.2f}  lvl={level_r:.2f}"
        ),
    }
    if labels:
        panel_labels.update(labels)

    if maze is None:
        maze = get_panel_maze(maze_id)
    saved_dir = Path(saved_dir)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=panel_size or panel_layout["panel_size"],
        gridspec_kw=dict(
            wspace=panel_layout["grid_wspace"],
            hspace=panel_layout["grid_hspace"],
        ),
        squeeze=False,
    )

    panel_data = []
    show_frame = bool(panel_style["show_frame"])
    title_template = panel_labels.get("title_template")
    _meta_keys = (
        "pair_key", "row_idx", "src", "dst", "animal", "cohort",
        "cell_a", "cell_b",
        "place_r", "level_r", "bc", "overlap", "logratio",
        "sn_a", "sn_b", "dt",
    )

    def _add_column_title(col: int, text: str) -> None:
        title_kwargs = {"pad": int(panel_style["title_pad"])}
        if panel_style.get("title_fontsize") is not None:
            title_kwargs["fontsize"] = int(panel_style["title_fontsize"])
        title_location = str(panel_style.get("title_location", "top")).lower()
        if title_location == "bottom":
            axes[-1, col].text(
                0.5,
                float(panel_style.get("title_y", -0.10)),
                text,
                transform=axes[-1, col].transAxes,
                ha="center",
                va="top",
                clip_on=False,
                **{k: v for k, v in title_kwargs.items() if k != "pad"},
            )
        elif title_location == "top":
            axes[0, col].set_title(text, **title_kwargs)
        else:
            raise ValueError("title_location must be 'top' or 'bottom'.")

    if mode == "placemap":
        # Pass 1: compute masked placemaps and collect metadata per column.
        for col, pick in enumerate(picks):
            pick = tuple(pick)
            if len(pick) != 2:
                raise ValueError(f"each pick must be (pair_key, row_idx); got {pick!r}")
            pair_key, row_idx = pick
            bundle_path = resolve_neuron_data_path(pair_key, row_idx, saved_dir)
            pack = load_neuron_data_bundle(bundle_path)

            pm_a, oc_a = compute_example_cell_placemap(
                pack, "A",
                n_bins=n_bins, sigma=sigma, signal=signal,
                dt=dt, min_time_base=min_time_base,
            )
            pm_b, oc_b = compute_example_cell_placemap(
                pack, "B",
                n_bins=n_bins, sigma=sigma, signal=signal,
                dt=dt, min_time_base=min_time_base,
            )

            if bool(panel_style["prefer_cached_mask"]) and int(n_bins) == 15 and "mask_15" in pack:
                mask = np.asarray(pack["mask_15"], dtype=bool).reshape(int(n_bins), int(n_bins))
            else:
                mask = (oc_a > 0) & (oc_b > 0)
            pm_a = np.where(mask, pm_a, np.nan)
            pm_b = np.where(mask, pm_b, np.nan)

            panel_data.append({
                "pair_key": str(pair_key), "row_idx": int(row_idx),
                "src": str(pack.get("src", "")),
                "dst": str(pack.get("dst", "")),
                "cell_a": int(pack.get("cell_a", -1)),
                "cell_b": int(pack.get("cell_b", -1)),
                "place_r": float(pack.get("place_r", np.nan)),
                "level_r": float(pack.get("level_r", np.nan)),
                "bc": float(pack.get("bc", np.nan)),
                "overlap": float(pack.get("overlap", np.nan)),
                "logratio": float(pack.get("logratio", np.nan)),
                "pm_a": pm_a, "pm_b": pm_b, "mask": mask,
                "bundle_path": str(bundle_path),
                "pack": pack,
            })

        # Resolve color-axis scaling per axis from the ``normalize`` mode,
        # with a legacy fallback to ``per_column_vmax``.
        if "normalize" in provided_style_keys:
            norm_mode = str(panel_style["normalize"]).lower()
        elif "per_column_vmax" in provided_style_keys:
            norm_mode = "per_column" if bool(panel_style["per_column_vmax"]) else "none"
        else:
            norm_mode = str(panel_style["normalize"]).lower()

        def _nanmax_or_eps(arr):
            return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1e-12

        if norm_mode == "global":
            global_max = max(
                (max(_nanmax_or_eps(d["pm_a"]), _nanmax_or_eps(d["pm_b"]))
                 for d in panel_data),
                default=1e-12,
            )

        for d in panel_data:
            max_a = _nanmax_or_eps(d["pm_a"])
            max_b = _nanmax_or_eps(d["pm_b"])
            if norm_mode == "per_panel":
                d["vmax_a"], d["vmax_b"] = max(max_a, 1e-12), max(max_b, 1e-12)
            elif norm_mode == "per_column":
                shared = max(max_a, max_b, 1e-12)
                d["vmax_a"] = d["vmax_b"] = shared
            elif norm_mode == "global":
                d["vmax_a"] = d["vmax_b"] = float(global_max)
            elif norm_mode == "none":
                d["vmax_a"] = d["vmax_b"] = None
            else:
                raise ValueError(
                    f"unknown normalize mode {norm_mode!r}; "
                    f"expected 'per_panel' | 'per_column' | 'global' | 'none'."
                )
            d["vmax"] = d["vmax_a"] if d["vmax_a"] == d["vmax_b"] else max(
                d["vmax_a"] or 0.0, d["vmax_b"] or 0.0
            )

        # Pass 2: draw each panel with its resolved vmax and optional maze
        # overlay; hide axes spines unless the caller asks for a frame.
        scale = float(n_bins) / 15.0
        for col, d in enumerate(panel_data):
            for ax, pm, vmax_ax in zip(
                axes[:, col],
                (d["pm_a"], d["pm_b"]),
                (d["vmax_a"], d["vmax_b"]),
            ):
                ax.imshow(
                    pm, origin="upper", aspect="equal",
                    vmin=float(panel_style["vmin"]), vmax=vmax_ax,
                    cmap=str(panel_colors["cmap"]),
                    interpolation=str(panel_style["interpolation"]),
                )
                if bool(panel_style["show_maze"]):
                    plot(
                        maze.wa[:, 0] * scale, maze.wa[:, 1] * scale,
                        fmts=[f"{panel_colors['maze_line']}-"],
                        linewidth=float(panel_style["maze_linewidth"]),
                        axes=ax,
                    )
                ax.set_xticks([]); ax.set_yticks([])
                if not show_frame:
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            if title_template:
                pack = d["pack"]
                fmt_ns = {k: pack[k] for k in _meta_keys if k in pack}
                fmt_ns.setdefault("pair_key", d["pair_key"])
                fmt_ns.setdefault("row_idx", d["row_idx"])
                _add_column_title(col, title_template.format(**fmt_ns))

        for d in panel_data:
            d.pop("pack", None)

    else:  # mode == "arrow"
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

        for col, pick in enumerate(picks):
            pick = tuple(pick)
            if len(pick) != 2:
                raise ValueError(f"each pick must be (pair_key, row_idx); got {pick!r}")
            pair_key, row_idx = pick
            bundle_path = resolve_neuron_data_path(pair_key, row_idx, saved_dir)
            pack = load_neuron_data_bundle(bundle_path)

            arrows_a = compute_example_cell_arrow_data(
                pack, "A",
                threshold_sn=float(panel_style["threshold_sn"]),
                dedupe_frames=int(panel_style["dedupe_frames"]),
                spike_signal=str(panel_style["spike_signal"]),
                noise_from=str(panel_style["noise_from"]),
            )
            arrows_b = compute_example_cell_arrow_data(
                pack, "B",
                threshold_sn=float(panel_style["threshold_sn"]),
                dedupe_frames=int(panel_style["dedupe_frames"]),
                spike_signal=str(panel_style["spike_signal"]),
                noise_from=str(panel_style["noise_from"]),
            )

            for ax, arrows in zip(axes[:, col], (arrows_a, arrows_b)):
                if show_trajectory:
                    ax.plot(
                        arrows["traj_x"], arrows["traj_y"],
                        color=str(panel_colors["trajectory"]),
                        linewidth=float(panel_style["trajectory_linewidth"]),
                        alpha=float(panel_style["trajectory_alpha"]),
                        rasterized=True,
                        zorder=int(panel_style["trajectory_zorder"]),
                    )
                for a, b, e, f in zip(arrows["px"], arrows["py"], arrows["ex"], arrows["ey"]):
                    dx = (e - a) * arrow_scale
                    dy = (f - b) * arrow_scale
                    _add_triangle_arrow_head(ax, a, b, dx, dy, **arrow_head_kwargs)
                if bool(panel_style["show_maze"]):
                    plot(
                        maze.wa[:, 0], maze.wa[:, 1],
                        fmts=[f"{panel_colors['maze_line']}-"],
                        linewidth=float(panel_style["maze_linewidth"]),
                        axes=ax,
                    )
                ax.set_aspect("equal")
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_xticks([]); ax.set_yticks([])
                if not show_frame:
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            entry = {
                "pair_key": str(pair_key), "row_idx": int(row_idx),
                "src": str(pack.get("src", "")),
                "dst": str(pack.get("dst", "")),
                "cell_a": int(pack.get("cell_a", -1)),
                "cell_b": int(pack.get("cell_b", -1)),
                "place_r": float(pack.get("place_r", np.nan)),
                "level_r": float(pack.get("level_r", np.nan)),
                "bc": float(pack.get("bc", np.nan)),
                "overlap": float(pack.get("overlap", np.nan)),
                "logratio": float(pack.get("logratio", np.nan)),
                "arrows_a": arrows_a, "arrows_b": arrows_b,
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
    return fig, axes, panel_data, saved_paths


def render_example_cell_day12_placemap_panel(
    *,
    pick,
    saved_dir,
    mode: str = "placemap",
    n_bins: int = 15,
    sigma: float = 1.0,
    signal: str = "Craw",
    dt: float = 0.04,
    min_time_base: float = 5.0,
    maze=None,
    maze_id: int = DEFAULT_MAZE_ID,
    panel_size=None,
    layout: dict | None = None,
    colors: dict | None = None,
    style: dict | None = None,
    labels: dict | None = None,
    save: bool = False,
    panel_name: str = "example_cell_day12_placemap",
    output_dir=DEFAULT_PANEL_OUTPUT_DIR,
    formats=DEFAULT_EXPORT_FORMATS,
):
    """Render a ``1 x 4`` per-neuron panel across day1/day2 x session1/session2.

    Set ``mode="placemap"`` (default) for rate-map panels or ``mode="arrow"``
    for body-to-nose arrow panels on the nose trajectory, matching
    :func:`render_example_cell_grid_panel`. Columns left-to-right are
    ``SRC day1 | SRC day2 | DST day1 | DST day2``. Because day1 <-> day2
    share a 1:1 neuron ID within the same session, the same ``cell_a``
    indexes both ``A_*`` and ``A2_*`` in the bundle (likewise for
    ``cell_b`` in ``B_*`` / ``B2_*``).

    Requires a saved bundle that includes the day-2 arrays
    (``A2_*``, ``B2_*``), produced by
    ``scripts/save_example_neuron_picks.py`` (updated to write the
    extended format on 2026-04-23). In placemap mode the mask is the
    4-way intersection of per-session occupancy masks. Arrow mode uses
    :func:`compute_example_cell_arrow_data` per side and does not apply
    a spatial mask.

    Parameters
    ----------
    pick : ``(pair_key, row_idx)``
        Single neuron to display. Used to locate the saved ``.npz``
        bundle under ``saved_dir``.
    saved_dir : str or Path
        Directory containing the saved bundles.
    n_bins, sigma, signal, dt, min_time_base
        Forwarded to :func:`compute_example_cell_placemap`.
    maze, maze_id
        Maze for the wall overlay.

    ``colors`` (merged over defaults)
        ``cmap`` (matplotlib colormap), ``maze_line``.

    ``style`` (merged over defaults)
        ``normalize`` (``"per_panel"`` | ``"per_column"`` -- shared across
        day1+day2 of the same session -- | ``"global"`` -- shared across
        all four panels -- | ``"none"`` -- matplotlib autoscale),
        ``vmin``, ``interpolation``, ``maze_linewidth``, ``show_maze``,
        ``show_frame``, ``title_fontsize``, ``title_pad``,
        ``suptitle_fontsize``, ``tight_layout``.

    ``labels`` (merged over defaults)
        ``col_titles`` (4-tuple of ``str.format`` templates or ``None`` to
        suppress), ``suptitle_template`` (a ``str.format`` template for a
        panel-wide title; ``None`` to suppress). Templates resolve against
        the bundle's scalars plus derived keys ``src``, ``src_d2``,
        ``dst``, ``dst_d2``, ``cell_a``, ``cell_b``, ``place_r``,
        ``level_r``, etc.

    Returns
    -------
    fig, axes, panel_data, saved_paths
        ``axes`` is a length-4 array of axes. ``panel_data`` is a dict
        with ``pm_by_side``, ``occ_by_side``, ``mask``, ``vmax_by_side``,
        plus metrics and ``bundle_path``.
    """
    mode = str(mode).lower()
    if mode not in ("placemap", "arrow"):
        raise ValueError(f"mode must be 'placemap' or 'arrow'; got {mode!r}")

    pick = tuple(pick)
    if len(pick) != 2:
        raise ValueError(f"pick must be (pair_key, row_idx); got {pick!r}")
    pair_key, row_idx = pick

    panel_layout = {
        "panel_size": (13.0, 3.7),
        "grid_wspace": 0.04,
        "margins": {"left": 0.03, "right": 0.995, "top": 0.84, "bottom": 0.03},
    }
    if layout:
        panel_layout.update(layout)

    if mode == "placemap":
        panel_colors = {"cmap": "viridis", "maze_line": "w"}
    else:  # arrow
        panel_colors = {
            "maze_line": "k",
            "trajectory": "0.55",
            "arrow_face": "red",
            "arrow_edge": "red",
        }
    if colors:
        panel_colors.update(colors)

    if mode == "placemap":
        panel_style = {
            "normalize": "per_panel",
            "vmin": 0.0,
            "interpolation": "nearest",
            "maze_linewidth": 0.9,
            "show_maze": True,
            "show_frame": False,
            "title_fontsize": None,
            "title_pad": 4,
            "suptitle_fontsize": None,
            "tight_layout": False,
        }
    else:  # arrow
        panel_style = {
            "threshold_sn": 5.0,
            "dedupe_frames": 4,
            "spike_signal": "S",      # bundle key suffix: "S" | "C" | "Craw"
            "noise_from": "Craw",     # bundle key suffix for Welch PSD
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
            "maze_linewidth": 1.5,
            "show_maze": True,
            "show_frame": False,
            "title_fontsize": None,
            "title_pad": 4,
            "suptitle_fontsize": None,
            "tight_layout": False,
        }
    if style:
        panel_style.update(style)

    panel_labels = {
        "col_titles": (
            "{src}  day 1",
            "{src_d2}  day 2",
            "{dst}  day 1",
            "{dst_d2}  day 2",
        ),
        "suptitle_template": (
            "{src} ↔ {dst}   cell A={cell_a}  B={cell_b}   "
            "place_r={place_r:.2f}  level_r={level_r:.2f}"
        ),
    }
    if labels:
        panel_labels.update(labels)

    if maze is None:
        maze = get_panel_maze(maze_id)
    saved_dir = Path(saved_dir)

    bundle_path = resolve_neuron_data_path(pair_key, row_idx, saved_dir)
    pack = load_neuron_data_bundle(bundle_path)

    sides = ("A", "A2", "B", "B2")
    missing = [f"{s}_frame" for s in sides if f"{s}_frame" not in pack]
    if missing:
        raise KeyError(
            f"bundle {bundle_path.name} is missing day-2 arrays "
            f"{missing!r}; re-run scripts/save_example_neuron_picks.py "
            f"to write the extended format."
        )

    fig, axes_arr = plt.subplots(
        1, 4,
        figsize=panel_size or panel_layout["panel_size"],
        gridspec_kw=dict(wspace=float(panel_layout["grid_wspace"])),
        squeeze=False,
    )
    axes = axes_arr[0]

    show_frame = bool(panel_style["show_frame"])

    meta_keys = (
        "pair_key", "row_idx",
        "src", "dst", "src_d2", "dst_d2",
        "animal", "cohort",
        "cell_a", "cell_b",
        "place_r", "level_r", "bc", "overlap", "logratio",
        "sn_a", "sn_b", "sn_a2", "sn_b2", "dt",
    )
    fmt_ns = {k: pack[k] for k in meta_keys if k in pack}
    fmt_ns.setdefault("pair_key", str(pair_key))
    fmt_ns.setdefault("row_idx", int(row_idx))

    col_titles = panel_labels.get("col_titles")
    if col_titles is not None and len(col_titles) != 4:
        raise ValueError(
            f"labels['col_titles'] must have 4 entries; got {len(col_titles)}"
        )

    if mode == "placemap":
        pms = []
        occs = []
        for side in sides:
            pm, oc = compute_example_cell_placemap(
                pack, side,
                n_bins=n_bins, sigma=sigma, signal=signal,
                dt=dt, min_time_base=min_time_base,
            )
            pms.append(pm)
            occs.append(oc)

        mask = (occs[0] > 0) & (occs[1] > 0) & (occs[2] > 0) & (occs[3] > 0)
        pms = [np.where(mask, pm, np.nan) for pm in pms]

        def _nanmax_or_eps(arr):
            return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else 1e-12

        norm_mode = str(panel_style["normalize"]).lower()
        if norm_mode == "per_panel":
            vmaxes = [max(_nanmax_or_eps(pm), 1e-12) for pm in pms]
        elif norm_mode == "per_column":  # "column" = one session, i.e. d1+d2 pair
            ma = max(_nanmax_or_eps(pms[0]), _nanmax_or_eps(pms[1]), 1e-12)
            mb = max(_nanmax_or_eps(pms[2]), _nanmax_or_eps(pms[3]), 1e-12)
            vmaxes = [ma, ma, mb, mb]
        elif norm_mode == "global":
            gm = max((_nanmax_or_eps(pm) for pm in pms), default=1e-12)
            gm = max(gm, 1e-12)
            vmaxes = [gm, gm, gm, gm]
        elif norm_mode == "none":
            vmaxes = [None, None, None, None]
        else:
            raise ValueError(
                f"unknown normalize mode {norm_mode!r}; expected "
                f"'per_panel' | 'per_column' | 'global' | 'none'."
            )

        scale = float(n_bins) / 15.0
        for col, (ax, pm, vmax_ax) in enumerate(zip(axes, pms, vmaxes)):
            ax.imshow(
                pm, origin="upper", aspect="equal",
                vmin=float(panel_style["vmin"]), vmax=vmax_ax,
                cmap=str(panel_colors["cmap"]),
                interpolation=str(panel_style["interpolation"]),
            )
            if bool(panel_style["show_maze"]):
                plot(
                    maze.wa[:, 0] * scale, maze.wa[:, 1] * scale,
                    fmts=[f"{panel_colors['maze_line']}-"],
                    linewidth=float(panel_style["maze_linewidth"]),
                    axes=ax,
                )
            ax.set_xticks([]); ax.set_yticks([])
            if not show_frame:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if col_titles and col_titles[col]:
                title_kwargs = {"pad": int(panel_style["title_pad"])}
                if panel_style.get("title_fontsize") is not None:
                    title_kwargs["fontsize"] = int(panel_style["title_fontsize"])
                ax.set_title(str(col_titles[col]).format(**fmt_ns), **title_kwargs)
    else:  # mode == "arrow"
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

        arrows_by_side = {}
        for side in sides:
            arrows_by_side[side] = compute_example_cell_arrow_data(
                pack, side,
                threshold_sn=float(panel_style["threshold_sn"]),
                dedupe_frames=int(panel_style["dedupe_frames"]),
                spike_signal=str(panel_style["spike_signal"]),
                noise_from=str(panel_style["noise_from"]),
            )

        for col, (ax, side) in enumerate(zip(axes, sides)):
            arrows = arrows_by_side[side]
            if show_trajectory:
                ax.plot(
                    arrows["traj_x"], arrows["traj_y"],
                    color=str(panel_colors["trajectory"]),
                    linewidth=float(panel_style["trajectory_linewidth"]),
                    alpha=float(panel_style["trajectory_alpha"]),
                    rasterized=True,
                    zorder=int(panel_style["trajectory_zorder"]),
                )
            for a, b, e, f in zip(arrows["px"], arrows["py"], arrows["ex"], arrows["ey"]):
                dx = (e - a) * arrow_scale
                dy = (f - b) * arrow_scale
                _add_triangle_arrow_head(ax, a, b, dx, dy, **arrow_head_kwargs)
            if bool(panel_style["show_maze"]):
                plot(
                    maze.wa[:, 0], maze.wa[:, 1],
                    fmts=[f"{panel_colors['maze_line']}-"],
                    linewidth=float(panel_style["maze_linewidth"]),
                    axes=ax,
                )
            ax.set_aspect("equal")
            ax.set_xlim(axis_lim)
            ax.set_ylim(axis_lim)
            ax.set_xticks([]); ax.set_yticks([])
            if not show_frame:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if col_titles and col_titles[col]:
                title_kwargs = {"pad": int(panel_style["title_pad"])}
                if panel_style.get("title_fontsize") is not None:
                    title_kwargs["fontsize"] = int(panel_style["title_fontsize"])
                ax.set_title(str(col_titles[col]).format(**fmt_ns), **title_kwargs)

    suptitle_template = panel_labels.get("suptitle_template")
    if suptitle_template:
        suptitle_kwargs = {}
        if panel_style.get("suptitle_fontsize") is not None:
            suptitle_kwargs["fontsize"] = int(panel_style["suptitle_fontsize"])
        fig.suptitle(suptitle_template.format(**fmt_ns), **suptitle_kwargs)

    if bool(panel_style["tight_layout"]):
        fig.tight_layout()
    else:
        margins = dict(panel_layout.get("margins", {}))
        fig.subplots_adjust(
            left=float(margins.get("left", 0.03)),
            right=float(margins.get("right", 0.995)),
            top=float(margins.get("top", 0.84)),
            bottom=float(margins.get("bottom", 0.03)),
            wspace=float(panel_layout["grid_wspace"]),
        )

    panel_data = {
        "pair_key": str(pair_key), "row_idx": int(row_idx),
        "src": str(pack.get("src", "")),
        "src_d2": str(pack.get("src_d2", "")),
        "dst": str(pack.get("dst", "")),
        "dst_d2": str(pack.get("dst_d2", "")),
        "cell_a": int(pack.get("cell_a", -1)),
        "cell_b": int(pack.get("cell_b", -1)),
        "place_r": float(pack.get("place_r", np.nan)),
        "level_r": float(pack.get("level_r", np.nan)),
        "bc": float(pack.get("bc", np.nan)),
        "overlap": float(pack.get("overlap", np.nan)),
        "logratio": float(pack.get("logratio", np.nan)),
        "mode": mode,
        "bundle_path": str(bundle_path),
    }
    if mode == "placemap":
        panel_data["pm_by_side"] = dict(zip(sides, pms))
        panel_data["occ_by_side"] = dict(zip(sides, occs))
        panel_data["vmax_by_side"] = dict(zip(sides, vmaxes))
        panel_data["mask"] = mask
    else:  # arrow
        panel_data["arrows_by_side"] = arrows_by_side

    saved_paths = (
        save_panel(fig, panel_name, formats=formats, output_dir=output_dir)
        if save
        else []
    )
    return fig, axes, panel_data, saved_paths


__all__ = [
    "DEFAULT_EXPORT_FORMATS",
    "DEFAULT_MAZE_ID",
    "DEFAULT_PANEL_OUTPUT_DIR",
    "END_NODE_LOWER_THRESHOLD",
    "END_NODE_UPPER_THRESHOLD",
    "CORRELATION_MEASURE_COLORS",
    "CORRELATION_EXPERIENCE_COLORS",
    "REPRESENTATION_COLOR_ALIASES",
    "REPRESENTATION_COLORS",
    "DEFAULT_COLOR_PALETTE",
    "END_NODE_COLOR_CODE_ALTERNATIVES",
    "END_NODE_COLOR_CODE",
    "END_NODE_COLOR_CODE_OPTION_1",
    "EXAMPLE_PATH_POINTS",
    "GRAY_BLOCKS",
    "PANEL_COLORS",
    "PANEL_LAYOUT",
    "PANEL_RCPARAMS",
    "QUADRANT_SPECS",
    "CYCLE_DIAGRAM_ARROWS",
    "CYCLE_DIAGRAM_BOXES",
    "CYCLE_DIAGRAM_COLORS",
    "CYCLE_DIAGRAM_LAYOUT",
    "CYCLE_DIAGRAM_LOOP",
    "CYCLE_DIAGRAM_STYLE",
    "CYCLE_DIAGRAM_TEXT",
    "GRID_ROUTE_CHOICE_SCHEMATIC_COLORS",
    "GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY",
    "GRID_ROUTE_CHOICE_SCHEMATIC_LAYOUT",
    "GRID_ROUTE_CHOICE_SCHEMATIC_STYLE",
    "GRID_ROUTE_CHOICE_SCHEMATIC_TEXT",
    "SPLIT_DAY_CYCLE_DIAGRAM_ARROWS",
    "SPLIT_DAY_CYCLE_DIAGRAM_BOXES",
    "SPLIT_DAY_CYCLE_DIAGRAM_COLORS",
    "SPLIT_DAY_CYCLE_DIAGRAM_LAYOUT",
    "SPLIT_DAY_CYCLE_DIAGRAM_LOOP",
    "SPLIT_DAY_CYCLE_DIAGRAM_STYLE",
    "SPLIT_DAY_CYCLE_DIAGRAM_TEXT",
    "LEVEL_ASSIGNMENT_SCHEMATIC_COLORS",
    "LEVEL_ASSIGNMENT_SCHEMATIC_GEOMETRY",
    "LEVEL_ASSIGNMENT_SCHEMATIC_LAYOUT",
    "LEVEL_ASSIGNMENT_SCHEMATIC_STYLE",
    "LEVEL_ASSIGNMENT_SCHEMATIC_TEXT",
    "MAZE_SCHEMATIC_LANDMARKS",
    "ROUTE_CHOICE_SCHEMATIC_COLORS",
    "ROUTE_CHOICE_SCHEMATIC_GEOMETRY",
    "ROUTE_CHOICE_SCHEMATIC_LAYOUT",
    "ROUTE_CHOICE_SCHEMATIC_STYLE",
    "ROUTE_CHOICE_SCHEMATIC_TEXT",
    "apply_panel_rcparams",
    "get_cycle_diagram_arrows",
    "get_cycle_diagram_boxes",
    "get_cycle_diagram_colors",
    "get_cycle_diagram_layout",
    "get_cycle_diagram_style",
    "get_cycle_diagram_text_specs",
    "get_grid_route_choice_schematic_colors",
    "get_grid_route_choice_schematic_geometry",
    "get_grid_route_choice_schematic_layout",
    "get_grid_route_choice_schematic_style",
    "get_grid_route_choice_schematic_text",
    "get_split_day_cycle_diagram_specs",
    "get_level_assignment_schematic_colors",
    "get_level_assignment_schematic_geometry",
    "get_level_assignment_schematic_layout",
    "get_level_assignment_schematic_style",
    "get_level_assignment_schematic_text",
    "get_maze_schematic_landmarks",
    "get_route_choice_schematic_colors",
    "get_route_choice_schematic_geometry",
    "get_route_choice_schematic_layout",
    "get_route_choice_schematic_style",
    "get_route_choice_schematic_text",
    "prepare_turning_bias_four_bias_points",
    "prepare_turning_bias_policy_summary",
    "draw_connected_path",
    "draw_endnode_timeline_axes",
    "draw_endnode_timeline_marker",
    "draw_linearized_matched_panel_axes",
    "draw_gray_blocks",
    "draw_maze_landmarks",
    "draw_maze_outline",
    "draw_quadrant_overlays",
    "extract_endnode_timeline",
    "finish_panel",
    "format_panel_title",
    "get_end_node_color",
    "get_end_node_colors",
    "get_effective_panel_layout",
    "get_effective_rcparams",
    "get_correlation_measure_colors",
    "get_correlation_experience_colors",
    "get_panel_colors",
    "get_panel_layout",
    "get_panel_maze",
    "get_panel_scale",
    "get_quadrant_specs",
    "get_representation_colors",
    "make_panel_figure",
    "render_cycle_diagram_panel",
    "render_correlation_shuffle_distribution_panel",
    "render_consecutive_gap_correlation_persistence_panel",
    "render_endnode_timeline_panel",
    "render_endnode_timeline_grid_panel",
    "render_example_cell_day12_placemap_panel",
    "render_example_cell_grid_panel",
    "render_experience_trend_panel",
    "render_exploration_efficiency_histogram_panel",
    "render_grouped_bar_scatter_panel",
    "render_grid_route_choice_schematic",
    "render_level_assignment_schematic",
    "render_level_code_heatmap_panel",
    "render_level_exploration_panel",
    "render_linearized_home_to_water_matched_panel",
    "render_maze_schematic_panel",
    "render_paired_bar_scatter_panel",
    "render_refiner_symmetry_preservation_panel",
    "render_reward_acquisition_month_panel",
    "render_route_choice_schematic",
    "render_stacked_local_correlation_panel",
    "resolve_cycle_diagram_boxes",
    "resolve_cycle_diagram_text_specs",
    "render_spatial_decoding_error_map",
    "render_support_matched_sensitivity_panel",
    "render_shuffle_mean_distribution_panel",
    "render_turning_bias_four_bias_panel",
    "render_turning_bias_policy_summary_panel",
    "render_turning_bias_three_group_panel",
    "render_variant_correlation_panel",
    "render_within_session_place_map_correlation_panel",
    "save_panel",
    "load_neuron_data_bundle",
    "resolve_neuron_data_path",
    "compute_example_cell_placemap",
    "compute_example_cell_arrow_data",
]
