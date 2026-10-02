from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis.figure_panels import (
    END_NODE_LOWER_THRESHOLD,
    END_NODE_UPPER_THRESHOLD,
    EXAMPLE_PATH_POINTS,
    MAZE_SCHEMATIC_LANDMARKS,
    PANEL_COLORS,
    PANEL_LAYOUT,
    get_split_day_cycle_diagram_specs,
    render_cycle_diagram_panel,
    render_endnode_timeline_grid_panel,
    render_exploration_efficiency_histogram_panel,
    render_grouped_bar_scatter_panel,
    render_maze_schematic_panel,
    render_reward_acquisition_month_panel,
    get_panel_maze,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = Path("docs/figure_panel_previews")
DEFAULT_BASENAME = "figure1_assembly_preview"

FONT_FAMILY = "Arial"
EXPORT_WIDTH_MM = 180.0
EXPORT_MAX_HEIGHT_MM = 215.0
CANVAS_SIZE = (10.0, 9.35)
EXPORT_CANVAS_SIZE = (8.0, 8.48)
EXPORT_HEIGHT_MM = min(EXPORT_WIDTH_MM * EXPORT_CANVAS_SIZE[1] / EXPORT_CANVAS_SIZE[0], EXPORT_MAX_HEIGHT_MM)

BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
TITLE_SIZE = 8.0
PANEL_ID_SIZE = 13.0
FIGURE1_MOSAIC = [
    "aaaa.ccccccccc",
    "aaaa.ccccccccc",
    "aaaa.ccccccccc",
    "aaaa.ccccccccc",
    "bbbb.ccccccccc",
    "bbbb.ccccccccc",
    "dddd.hhhhhhhhh",
    "dddd.hhhhhhhhh",
    "dddd.hhhhhhhhh",
    "ggggiiiiiiijjj",
    "ggggiiiiiiijjj",
    "ggggiiiiiiijjj",
]
MOSAIC_WIDTH_RATIOS = [1.0, 1.0, 1.0, 1.0, 0.10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
MOSAIC_HEIGHT_RATIOS = [0.64, 0.64, 0.64, 0.64, 0.66, 0.66, 0.90, 0.90, 0.90, 0.82, 0.82, 0.82]
PANEL_SIZES = {
    "a": (2.60, 2.60),
    "b": (3.05, 0.92),
    "c": (5.00, 4.10),
    "d": (3.15, 2.58),
    "g": (2.55, 2.30),
    "h": (5.00, 2.55),
    "i": (2.55, 2.58),
    "j": (2.55, 2.58),
}
FIGURE1_PANEL_C_AXES_BOTTOM = 0.12
FIGURE1_PANEL_C_AXES_TOP = 0.90
FIGURE1_PANEL_C_AXES_HSPACE = 0.26
FIGURE1_PANEL_D_AXES_BOTTOM = 0.28
FIGURE1_PANEL_D_AXES_TOP = 0.96
FIGURE1_PANEL_F_AXES_BOTTOM = 0.24
FIGURE1_PANEL_F_AXES_TOP = FIGURE1_PANEL_F_AXES_BOTTOM + (
    (FIGURE1_PANEL_C_AXES_TOP - FIGURE1_PANEL_C_AXES_BOTTOM)
    * PANEL_SIZES["c"][1]
    / (2.0 + FIGURE1_PANEL_C_AXES_HSPACE)
    / PANEL_SIZES["h"][1]
)
PANEL_ID_POSITIONS = {
    "h": (-0.020, 0.96),
    "i": (-0.08, 0.99),
    "j": (-0.08, 0.99),
}
PANEL_DISPLAY_LABELS = {
    "g": "e",
    "h": "f",
    "i": "g",
    "j": "h",
}
LEFT_ANCHORED_PANELS = {"a", "b"}
RIGHT_ANCHORED_PANELS: set[str] = set()
PLACED_BOUNDS_PANEL_IDS = {"i", "j"}
OFFSET_Y_PANEL_IDS = {"d", "g", "h", "i", "j"}
PLACED_LABEL_Y_PANEL_IDS = {"d"}
COMPACT_ENDNODE_PANELS = {"c", "h"}
FIGURE1_LEFT_PANELS_VERTICAL_OFFSET_POINTS = 6.64
FIGURE1_DF_ROW_VERTICAL_OFFSET_POINTS = 6.0
FIGURE1_BOTTOM_ROW_VERTICAL_OFFSET_POINTS = -24.0
PANEL_OFFSETS_POINTS = {
    "a": (0.0, FIGURE1_LEFT_PANELS_VERTICAL_OFFSET_POINTS),
    "b": (0.0, FIGURE1_LEFT_PANELS_VERTICAL_OFFSET_POINTS),
    "d": (0.0, FIGURE1_DF_ROW_VERTICAL_OFFSET_POINTS),
    "g": (0.0, FIGURE1_BOTTOM_ROW_VERTICAL_OFFSET_POINTS),
    "h": (0.0, FIGURE1_DF_ROW_VERTICAL_OFFSET_POINTS),
    "i": (0.0, FIGURE1_BOTTOM_ROW_VERTICAL_OFFSET_POINTS),
    "j": (0.0, FIGURE1_BOTTOM_ROW_VERTICAL_OFFSET_POINTS),
}
PANEL_SCALE_ADJUST = {
    "g": 0.92,
    "h": 1.00,
    "i": 1.03,
    "j": 1.00,
}
PANEL_FONT_COMPENSATION = {
    "d": 1.00,
    "g": 0.95,
    "h": 1.00,
    "i": 1.10,
    "j": 1.10,
}
PANEL_STAR_FONT_SIZE = {
    "d": 7.90,
    "i": 8.80,
    "j": 8.95,
}
FIGURE1_CURRENT_PLUS_NEW_DIR = (
    REPO_ROOT / "docs" / "new_12mo_session_fit_2026-06-02" / "panel_by_panel_current_vs_new"
)
FIGURE1_REWARD_ACQUISITION_POINTS = (
    FIGURE1_CURRENT_PLUS_NEW_DIR / "reward_acquisition_day1_month_points_current_plus_new12.csv"
)
FIGURE1_REWARD_ACQUISITION_SUMMARY = (
    FIGURE1_CURRENT_PLUS_NEW_DIR / "reward_acquisition_day1_month_summary_current_plus_new12.csv"
)
FIGURE1_REWARD_ACQUISITION_CONTRASTS = (
    FIGURE1_CURRENT_PLUS_NEW_DIR / "reward_acquisition_day1_month_clustered_contrasts_current_plus_new12.csv"
)
FIGURE1_TURNING_POLICY_FIXED30_ROWS = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed30min_promoted_2026-08-04"
    / "source_tables"
    / "primary_30min_strict_joint_equal_junction_policy_rows.csv"
)
FIGURE1_EXPLORATION_POINTS = FIGURE1_CURRENT_PLUS_NEW_DIR / "exploration_first_chunk_current_plus_new_points.csv"
FIGURE1_LONG_BOUT_POINTS = FIGURE1_CURRENT_PLUS_NEW_DIR / "long_bout_group_points_current_plus_new12.csv"
FIGURE1_LONG_BOUT_SUMMARY = FIGURE1_CURRENT_PLUS_NEW_DIR / "long_bout_group_summary_current_plus_new12.csv"
FIGURE1_LONG_BOUT_CONTRASTS = (
    FIGURE1_CURRENT_PLUS_NEW_DIR / "long_bout_group_clustered_contrasts_current_plus_new12.csv"
)
FIGURE1_ED04_SOURCE_TABLE_DIR = (
    REPO_ROOT / "docs" / "extended_data_figure_panel_refresh_2026-07-12" / "source_tables"
)
FIGURE1_ROUTE_SCORE_POINTS = (
    FIGURE1_ED04_SOURCE_TABLE_DIR / "ed04_session_route_current_plus_new12.csv"
)
FIGURE1_ROUTE_SCORE_SUMMARY = (
    FIGURE1_ED04_SOURCE_TABLE_DIR / "ed04_route_group_stats_current_plus_new12.csv"
)
FIGURE1_ROUTE_SCORE_CONTRASTS = (
    FIGURE1_ED04_SOURCE_TABLE_DIR / "ed04_route_stats_current_plus_new12.csv"
)
FIGURE1_END_NODE_COLOR_CODE = {
    # Maze coordinates use an inverted y-axis: lower_* are the visual top row.
    "lower_left": "#E0B27D",   # visual top-left: ochre
    "lower_right": "#9FBF96",  # visual top-right/reward: sage
    "upper_left": "#A38FBF",   # visual bottom-left: heather
    "upper_right": "#CC9A87",  # visual bottom-right: clay
}
FIGURE1_NAIVE_COLOR = "#8BBDEB"
FIGURE1_NAIVE_MARK_COLOR = "#1F90D5"
FIGURE1_RETURN_COLOR = "#DA2323"
FIGURE1_RETURN_MARK_COLOR = "#D82929"
FIGURE1_SECOND_DAY_COLOR = "#5F8C62"
FIGURE1_PATH_COLOR = "#D82929"
FIGURE1_LATENT_DAY2_COLOR = "#B3A99B"
FIGURE1_REWARD_PATH_RUNS = (0, 1, 4, 10, 22, 46, 94)
# Round 2 scales the accepted round-1 circle diameter, outline, and numeral
# size linearly by 1.4. Matplotlib scatter ``s`` is area, so it scales by 1.4^2.
FIGURE1_DECISION_SCALE = 1.4
FIGURE1_DECISION_MARKER_AREA_POINTS2 = 34.0 * FIGURE1_DECISION_SCALE**2
FIGURE1_DECISION_MARKER_LINEWIDTH = 0.65 * FIGURE1_DECISION_SCALE
FIGURE1_DECISION_LABEL_SIZE = 4.0 * FIGURE1_DECISION_SCALE
FIGURE1_DECISION_COLOR = "#202020"
FIGURE1_MONTH_COLORS = {
    0: FIGURE1_NAIVE_COLOR,
    3: "#F5A18E",
    6: "#F37B6D",
    9: "#DB4B4B",
    12: "#9B0E13",
}
FIGURE1_PANEL_B_DAY1_TITLE_CENTER_X = 129
FIGURE1_PANEL_B_CONDITION_CENTER = (129, 66)
FIGURE1_PANEL_B_DAY2_TITLE_CENTER_X = 347
FIGURE1_PANEL_B_NEXT_CYCLE_CENTER_X = 300
FIGURE1_PANEL_B_NEXT_CYCLE_CENTER_Y = 138
FIGURE1_PANEL_B_DAY1_TITLE_TEXT = "Day i"
FIGURE1_PANEL_B_CONDITION_TEXT = "i = 1: Naive day\ni > 1: Return day"
FIGURE1_PANEL_B_DAY2_TITLE_TEXT = "Day i + 1"
FIGURE1_PANEL_B_NEXT_CYCLE_TEXT = "Next cycle: i \u2192 i + 90"
FIGURE1_PANEL_B_ANNOTATION_FONT_SCALE = 1.20


def _panel_size(panel_id: str) -> tuple[float, float]:
    return PANEL_SIZES[panel_id]


def _set_common_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.8,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "svg.fonttype": "none",
        }
    )


def _standardize_axes_text(fig, *, font_scale: float = 1.0) -> None:
    axis_label_size = AXIS_LABEL_SIZE * font_scale
    tick_label_size = TICK_LABEL_SIZE * font_scale
    title_size = TITLE_SIZE * font_scale
    for ax in fig.axes:
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontsize(axis_label_size)
        ax.yaxis.label.set_fontsize(axis_label_size)
        ax.tick_params(labelsize=tick_label_size)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(FONT_FAMILY)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(FONT_FAMILY)
                text.set_fontsize(tick_label_size)


def _fill_single_axis_figure(fig) -> None:
    """Use the whole offscreen panel for schematic-only figures."""
    if len(fig.axes) != 1:
        return
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.axes[0].set_position([0, 0, 1, 1])


def get_figure1_reward_path_decisions() -> list[dict[str, int]]:
    """Resolve the six scored first-visit decisions from canonical maze runs."""
    maze = get_panel_maze(6)
    decisions: list[dict[str, int]] = []
    for depth, run in enumerate(FIGURE1_REWARD_PATH_RUNS[:-1]):
        cell = int(maze.ru[run][-1])
        decisions.append(
            {
                "depth": int(depth),
                "run": int(run),
                "cell": cell,
                "x": int(maze.xc[cell]),
                "y": int(maze.yc[cell]),
            }
        )
    return decisions


def _draw_figure1_decision_points(ax) -> list[dict[str, int]]:
    """Overlay the ED3e decision grammar at the scaled Main Figure size."""
    decisions = get_figure1_reward_path_decisions()
    ax.scatter(
        [decision["x"] for decision in decisions],
        [decision["y"] for decision in decisions],
        s=FIGURE1_DECISION_MARKER_AREA_POINTS2,
        facecolor="white",
        edgecolor=FIGURE1_DECISION_COLOR,
        linewidth=FIGURE1_DECISION_MARKER_LINEWIDTH,
        zorder=8,
    )
    for decision in decisions:
        ax.text(
            decision["x"],
            decision["y"],
            str(decision["depth"]),
            ha="center",
            va="center",
            fontsize=FIGURE1_DECISION_LABEL_SIZE,
            fontfamily=FONT_FAMILY,
            color=FIGURE1_DECISION_COLOR,
            zorder=9,
        )
    return decisions


def _render_panel_a(size: tuple[float, float]):
    _set_common_rc()
    layout = {
        **PANEL_LAYOUT,
        "panel_size": size,
        "quadrant_alpha": 1,
        "maze_xlim": (-1.85, 15.95),
        "maze_ylim": (15.0, -1.0),
    }
    colors = {
        **PANEL_COLORS,
        "path": FIGURE1_PATH_COLOR,
        "quadrants": {**FIGURE1_END_NODE_COLOR_CODE},
    }
    landmarks = {
        "show": True,
        "home": {**MAZE_SCHEMATIC_LANDMARKS["home"]},
        "reward": {**MAZE_SCHEMATIC_LANDMARKS["reward"]},
    }
    landmarks["reward"].update({"size": 1.0, "linewidth": 1.0})
    fig, _ax, _paths = render_maze_schematic_panel(
        panel_size=size,
        layout=layout,
        colors=colors,
        path_points=EXAMPLE_PATH_POINTS,
        landmarks=landmarks,
        save=False,
    )
    _draw_figure1_decision_points(_ax)
    _fill_single_axis_figure(fig)
    return fig


def _render_panel_b(size: tuple[float, float]):
    _set_common_rc()
    specs = get_split_day_cycle_diagram_specs(
        layout={
            "panel_size": size,
            "background": "none",
        },
        colors={"background": "none"},
    )
    specs["layout"]["coordinate_height"] = 150
    specs["boxes"]["day1"].update({"xy": (34, 12), "width": 190, "height": 88})
    specs["boxes"]["day2"].update({"xy": (252, 12), "width": 190, "height": 88})
    specs["boxes"]["rest"].update({"xy": (472, 12), "width": 108, "height": 88})
    specs["loop"]["x"] = [580, 592, 592, 6, 6]
    specs["loop"]["y"] = [56, 56, 122, 122, 56]
    specs["arrows"] = [
        {"start": (6, 56), "end": (34, 56)},
        {"start": (224, 56), "end": (252, 56)},
        {"start": (442, 56), "end": (472, 56)},
    ]
    next_cycle_color = "#666666"
    day1_title_spec = None
    condition_spec = None
    day2_title_spec = None
    cleaned_text_specs = []
    for original_text_spec in specs["text_specs"]:
        text_spec = {
            **original_text_spec,
            "fontsize": float(original_text_spec["fontsize"])
            * FIGURE1_PANEL_B_ANNOTATION_FONT_SCALE,
        }
        text = str(text_spec.get("text", ""))
        if text == "Day $i$":
            day1_title_spec = {**text_spec, "x": 129, "y": 30}
            continue
        elif text.startswith("$i = 1$:") and "Naive" in text:
            condition_spec = {**text_spec}
            continue
        elif text.startswith("$i > 1$:") and "Return" in text:
            continue
        elif text == "Day $i+1$":
            day2_title_spec = {**text_spec, "x": 347, "y": 32}
            continue
        elif text.startswith("Second"):
            text_spec.update({"x": 347, "y": 64, "text": "Second day"})
        elif text == "Rest":
            text_spec.update({"x": 526, "y": 32})
        elif text == "~3 months":
            text_spec.update({"x": 526, "y": 64})
        elif text in {"Next cycle:", "$i$", r"$\rightarrow$", "$i+90$"}:
            next_cycle_color = str(text_spec.get("color", next_cycle_color))
            continue
        cleaned_text_specs.append(text_spec)
    specs["text_specs"] = cleaned_text_specs
    fig, _ax, _paths = render_cycle_diagram_panel(
        panel_size=size,
        layout=specs["layout"],
        colors=specs["colors"],
        style=specs["style"],
        boxes=specs["boxes"],
        loop=specs["loop"],
        arrows=specs["arrows"],
        text_specs=specs["text_specs"],
        save=False,
    )
    if condition_spec is not None:
        _ax.text(
            FIGURE1_PANEL_B_CONDITION_CENTER[0],
            FIGURE1_PANEL_B_CONDITION_CENTER[1],
            FIGURE1_PANEL_B_CONDITION_TEXT,
            ha="center",
            va="center",
            multialignment="center",
            linespacing=1.35,
            fontsize=condition_spec["fontsize"],
            color=condition_spec["color"],
            fontfamily=FONT_FAMILY,
        )
    if day1_title_spec is not None:
        _ax.text(
            FIGURE1_PANEL_B_DAY1_TITLE_CENTER_X,
            30,
            FIGURE1_PANEL_B_DAY1_TITLE_TEXT,
            ha="center",
            va="center",
            fontsize=day1_title_spec["fontsize"],
            color=day1_title_spec["color"],
            fontfamily=FONT_FAMILY,
            fontweight=day1_title_spec.get("fontweight"),
        )
    if day2_title_spec is not None:
        _ax.text(
            FIGURE1_PANEL_B_DAY2_TITLE_CENTER_X,
            32,
            FIGURE1_PANEL_B_DAY2_TITLE_TEXT,
            ha="center",
            va="center",
            fontsize=day2_title_spec["fontsize"],
            color=day2_title_spec["color"],
            fontfamily=FONT_FAMILY,
            fontweight=day2_title_spec.get("fontweight"),
        )
    _ax.text(
        FIGURE1_PANEL_B_NEXT_CYCLE_CENTER_X,
        FIGURE1_PANEL_B_NEXT_CYCLE_CENTER_Y,
        FIGURE1_PANEL_B_NEXT_CYCLE_TEXT,
        ha="center",
        va="center",
        fontsize=6.2 * FIGURE1_PANEL_B_ANNOTATION_FONT_SCALE,
        color=next_cycle_color,
        fontfamily=FONT_FAMILY,
    )
    _fill_single_axis_figure(fig)
    return fig


def _render_panel_c(size: tuple[float, float]):
    _set_common_rc()
    session_rows = [
        [
            {"nickname": "434RRday1", "display_title": "Day1"},
            {"nickname": "434-3day1", "display_title": "Day91"},
        ],
        [
            {"nickname": "434RRday2", "display_title": "Day2"},
            {"nickname": "434-6day1", "display_title": "Day181"},
        ],
    ]
    subplot_size = (size[0] / 2.0, size[1] / 2.0)
    layout = {
        **PANEL_LAYOUT,
        "reference_panel_size": subplot_size,
        "panel_size": size,
        "timeline_scatter_alpha": 1,
        "timeline_reference_alpha": 0.3,
        "timeline_scatter_marker_size": 2.2,
        "grid_wspace": 0.10,
        "grid_hspace": 0.16,
    }
    marker = {
        "kind": "water",
        "node": 94,
        "x_axes": 1.0,
        "x_offset_points": 6,
        "size": 15,
    }
    fig, axes, _data, _paths = render_endnode_timeline_grid_panel(
        session_rows,
        panel_size=size,
        layout=layout,
        colors={
            **PANEL_COLORS,
            "reward_reference": FIGURE1_END_NODE_COLOR_CODE["lower_right"],
            "quadrants": {**FIGURE1_END_NODE_COLOR_CODE},
        },
        rcparams={
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.linewidth": 0.8,
        },
        lower_threshold=END_NODE_LOWER_THRESHOLD,
        upper_threshold=END_NODE_UPPER_THRESHOLD,
        reward_node=94,
        xlim=(0, 6500),
        ylim=(END_NODE_LOWER_THRESHOLD - 2, END_NODE_UPPER_THRESHOLD + 2),
        xticks=(0, 1000, 2000, 3000, 4000, 5000, 6000),
        endnode_marker=marker,
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(
        left=0.11,
        right=0.91,
        bottom=FIGURE1_PANEL_C_AXES_BOTTOM,
        top=FIGURE1_PANEL_C_AXES_TOP,
        wspace=0.12,
        hspace=FIGURE1_PANEL_C_AXES_HSPACE,
    )
    return fig


def _render_panel_d(size: tuple[float, float]):
    _set_common_rc()
    font_scale = PANEL_FONT_COMPENSATION["d"]
    fig, _ax, _data, _paths = render_reward_acquisition_month_panel(
        points_csv=FIGURE1_REWARD_ACQUISITION_POINTS,
        summary_csv=FIGURE1_REWARD_ACQUISITION_SUMMARY,
        contrasts_csv=FIGURE1_REWARD_ACQUISITION_CONTRASTS,
        metric="lb0",
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            **FIGURE1_MONTH_COLORS,
            "edge": "none",
            "error": "black",
        },
        style={
            "bar_width": 0.6,
            "bar_alpha": 0.75,
            "bar_linewidth": 0.0,
            "scatter_size": 15,
            "scatter_alpha": 0.85,
            "scatter_jitter": 0.06,
            "scatter_seed": 260113,
            "annotation_source": "cluster_ols",
            "annotation_p_col": "cluster_ols_q_bh",
            "annotation_show_ns": False,
            "annotation_y_start": 3000,
            "annotation_step": 420,
            "annotation_text_offset_fraction": 0.10,
            "annotation_fontsize": PANEL_STAR_FONT_SIZE["d"],
            "annotation_fontweight": "bold",
            "axis_label_fontsize": AXIS_LABEL_SIZE * font_scale,
        },
        labels={
            "xlabel": "Time",
            "ylabel": "First long bout time (s)",
        },
        save=False,
    )
    _standardize_axes_text(fig, font_scale=font_scale)
    fig.subplots_adjust(
        left=0.24,
        right=0.98,
        bottom=FIGURE1_PANEL_D_AXES_BOTTOM,
        top=FIGURE1_PANEL_D_AXES_TOP,
    )
    return fig






def _render_panel_exploration(size: tuple[float, float]):
    _set_common_rc()
    font_scale = PANEL_FONT_COMPENSATION["g"]
    group_order = ("Naive D1 (rewarded)", "LTM D1 (all)")
    fig, _ax, panel_data, _paths = render_exploration_efficiency_histogram_panel(
        points_csv=FIGURE1_EXPLORATION_POINTS,
        group_order=group_order,
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "Naive D1 (rewarded)": FIGURE1_NAIVE_COLOR,
            "LTM D1 (all)": FIGURE1_RETURN_COLOR,
        },
        style={
            "n_bins": 20,
            "alpha": 0.6,
            "edgecolor": "none",
            "show_legend": True,
            "show_counts_in_legend": False,
            "legend_loc": "upper right",
            "yticks": None,
        },
        labels={
            "xlabel": "Exploration Efficiency",
            "ylabel": "Fraction of sessions",
            "group_labels": {
                "Naive D1 (rewarded)": "Naive day",
                "LTM D1 (all)": "Return day",
            },
        },
        save=False,
    )
    histogram_rows = panel_data["histogram_counts"]
    patch_offset = 0
    for group_name in group_order:
        group_histogram = histogram_rows.loc[
            histogram_rows["group"].astype(str).eq(group_name)
        ].sort_values("bin_index")
        n_bins = int(len(group_histogram))
        group_patches = list(_ax.patches[patch_offset : patch_offset + n_bins])
        patch_offset += n_bins
        n_sessions = int(
            panel_data["points"]["group"].astype(str).eq(group_name).sum()
        )
        if len(group_patches) != n_bins or n_sessions <= 0:
            raise RuntimeError(
                f"Histogram fraction normalization failed for {group_name!r}."
            )
        for patch, row in zip(
            group_patches, group_histogram.itertuples(index=False), strict=True
        ):
            patch.set_height(float(row.count) / float(n_sessions))

    _ax.set_ylim(0.0, 0.30)
    _ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    _standardize_axes_text(fig, font_scale=font_scale)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.95)
    return fig


def _render_panel_h(size: tuple[float, float]):
    _set_common_rc()
    font_scale = PANEL_FONT_COMPENSATION["h"]
    session_rows = [
        [
            {
                "nickname": "1636Rday1",
                "display_title": "Latent day1",
                "endnode_marker": {
                    "kind": "no_water",
                    "node": 94,
                    "x_axes": 1.0,
                    "x_offset_points": 8,
                    "size": 13,
                    "linewidth": 1.4,
                },
            },
            {
                "nickname": "1636Rday2",
                "display_title": "Latent day2",
                "endnode_marker": {
                    "kind": "water",
                    "node": 94,
                    "x_axes": 1.0,
                    "x_offset_points": 6,
                    "size": 15,
                },
            },
        ],
    ]
    subplot_size = (size[0] / 2.0, size[1])
    fig, _axes, _data, _paths = render_endnode_timeline_grid_panel(
        session_rows,
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "reference_panel_size": subplot_size,
            "panel_size": size,
            "timeline_scatter_alpha": 1,
            "timeline_reference_alpha": 0.0,
            "timeline_scatter_marker_size": 2.2,
            "grid_wspace": 0.10,
            "grid_hspace": 0.28,
        },
        colors={**PANEL_COLORS, "quadrants": {**FIGURE1_END_NODE_COLOR_CODE}},
        rcparams={
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE * font_scale,
            "axes.linewidth": 0.8,
        },
        lower_threshold=END_NODE_LOWER_THRESHOLD,
        upper_threshold=END_NODE_UPPER_THRESHOLD,
        reward_node=None,
        xlim=(0, 6500),
        ylim=(END_NODE_LOWER_THRESHOLD - 2, END_NODE_UPPER_THRESHOLD + 2),
        xticks=(0, 1000, 2000, 3000, 4000, 5000, 6000),
        save=False,
    )
    _standardize_axes_text(fig, font_scale=font_scale)
    # Panel f retains its original narrower width; panel c is reduced to this
    # same horizontal geometry so all six end-node axes align.
    fig.subplots_adjust(
        left=0.11,
        right=0.91,
        bottom=FIGURE1_PANEL_F_AXES_BOTTOM,
        top=FIGURE1_PANEL_F_AXES_TOP,
        wspace=0.12,
    )
    return fig


def _render_panel_i(size: tuple[float, float]):
    _set_common_rc()
    font_scale = PANEL_FONT_COMPENSATION["i"]
    fig, _ax, _data, _paths = render_grouped_bar_scatter_panel(
        points_csv=FIGURE1_LONG_BOUT_POINTS,
        summary_csv=FIGURE1_LONG_BOUT_SUMMARY,
        contrasts_csv=FIGURE1_LONG_BOUT_CONTRASTS,
        metric="lb0",
        group_order=("Rewarded D1 (all)", "Unrewarded D2 (all)", "LTM D1 (all)"),
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "Rewarded D1 (all)": FIGURE1_NAIVE_COLOR,
            "Unrewarded D2 (all)": FIGURE1_LATENT_DAY2_COLOR,
            "LTM D1 (all)": FIGURE1_RETURN_COLOR,
            "edge": "none",
            "error": "black",
        },
        style={
            "bar_width": 0.6,
            "bar_alpha": 0.78,
            "bar_linewidth": 0.0,
            "scatter_size": 15,
            "scatter_alpha": 0.72,
            "scatter_jitter": 0.06,
            "scatter_seed": 260113,
            "annotation_p_col": "cluster_ols_q_bh",
            "annotation_show_ns": False,
            "annotation_y_start": 3000,
            "annotation_step": 430,
            "annotation_text_offset_fraction": 0.10,
            "annotation_fontsize": PANEL_STAR_FONT_SIZE["i"],
            "annotation_fontweight": "bold",
            "annotation_offset_fraction": 0.06,
            "annotation_step_fraction": 0.08,
        },
        labels={
            "xlabel": "",
            "ylabel": "First long bout time (s)",
            "group_labels": {
                "Rewarded D1 (all)": "Naive\nday",
                "Unrewarded D2 (all)": "Latent\nday2",
                "LTM D1 (all)": "Return\nday",
            },
        },
        save=False,
    )
    _standardize_axes_text(fig, font_scale=font_scale)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.25, top=0.96)
    return fig


def _route_score_stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _render_panel_route_score(size: tuple[float, float]):
    """Render the current-plus-new12 first-visit route-score control."""
    _set_common_rc()
    font_scale = PANEL_FONT_COMPENSATION["j"]
    points = pd.read_csv(FIGURE1_ROUTE_SCORE_POINTS)
    summary = pd.read_csv(FIGURE1_ROUTE_SCORE_SUMMARY).set_index("condition")
    contrasts = pd.read_csv(FIGURE1_ROUTE_SCORE_CONTRASTS).set_index("comparison")

    group_order = ("naive", "return", "experienced_day2", "latent_day2")
    group_labels = {
        "naive": "Naive\nday",
        "return": "Return\nday",
        "experienced_day2": "Second\nday",
        "latent_day2": "Latent\nday2",
    }
    point_colors = {
        "naive": FIGURE1_NAIVE_COLOR,
        "return": FIGURE1_RETURN_COLOR,
        "experienced_day2": FIGURE1_SECOND_DAY_COLOR,
        "latent_day2": FIGURE1_LATENT_DAY2_COLOR,
    }
    expected_support = {
        "naive": (23, 23),
        "return": (56, 17),
        "experienced_day2": (74, 24),
        "latent_day2": (9, 9),
    }
    observed_support = {
        group: (
            int(points["condition"].eq(group).sum()),
            int(points.loc[points["condition"].eq(group), "animal"].astype(str).nunique()),
        )
        for group in group_order
    }
    if observed_support != expected_support:
        raise ValueError(
            f"Figure 1 route-score support drifted: {observed_support}; expected {expected_support}"
        )

    supported = set(
        contrasts.loc[contrasts["q_bh6"].lt(0.05)].index.astype(str)
    )
    expected_supported = {
        "experienced_day2_vs_naive",
        "experienced_day2_vs_return",
        "experienced_day2_vs_latent_day2",
    }
    if supported != expected_supported:
        raise ValueError(
            f"Unexpected supported Figure 1 route-score contrasts: {sorted(supported)}"
        )

    fig, ax = plt.subplots(figsize=size)
    rng = np.random.default_rng(20260716)
    tick_labels: list[str] = []
    for index, group in enumerate(group_order):
        group_rows = points.loc[points["condition"].eq(group)]
        values = pd.to_numeric(
            group_rows["route_fraction_correct"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        row = summary.loc[group]
        mean = float(row["mean_route_fraction_correct"])
        low = float(row["cluster_ci95_low"])
        high = float(row["cluster_ci95_high"])
        ax.bar(
            index,
            mean,
            width=0.60,
            color=point_colors[group],
            edgecolor="none",
            alpha=0.78,
            zorder=1,
        )
        jitter = rng.normal(0.0, 0.06, len(values))
        ax.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=15,
            facecolor=point_colors[group],
            edgecolor="none",
            alpha=0.72,
            zorder=2,
        )
        ax.errorbar(
            index,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            color="black",
            fmt="none",
            capsize=4.0,
            elinewidth=1.1,
            capthick=1.1,
            zorder=4,
        )
        tick_labels.append(group_labels[group])

    bracket_specs = (
        ("experienced_day2_vs_return", 1, 2, 1.025),
        ("experienced_day2_vs_latent_day2", 2, 3, 1.080),
        ("experienced_day2_vs_naive", 0, 2, 1.135),
    )
    for comparison, x_left, x_right, y_base in bracket_specs:
        q_value = float(contrasts.loc[comparison, "q_bh6"])
        annotation_step = 0.055
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [
                y_base,
                y_base + annotation_step * 0.25,
                y_base + annotation_step * 0.25,
                y_base,
            ],
            color="#202020",
            linewidth=1.0,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (x_left + x_right) / 2.0,
            y_base + annotation_step * 0.10,
            _route_score_stars(q_value),
            ha="center",
            va="bottom",
            fontsize=PANEL_STAR_FONT_SIZE["j"],
            fontweight="bold",
            fontfamily=FONT_FAMILY,
            color="#202020",
            clip_on=False,
            zorder=6,
        )

    ax.axhline(0.5, color="#8F8F8F", linewidth=0.65, linestyle=(0, (3, 2)), zorder=3)
    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(0.22, 1.19)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_xticks(range(len(group_order)), tick_labels)
    ax.set_ylabel("First visit route score")
    ax.tick_params(axis="x", direction="out", length=3.5, width=0.8, pad=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _standardize_axes_text(fig, font_scale=font_scale)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.25, top=0.96)
    return fig


PANEL_RENDERERS = {
    "a": _render_panel_a,
    "b": _render_panel_b,
    "c": _render_panel_c,
    "d": _render_panel_d,
    "g": _render_panel_exploration,
    "h": _render_panel_h,
    "i": _render_panel_i,
    "j": _render_panel_route_score,
}
PANEL_EXPORT_STEMS = {
    "a": "figure1_panel_a_maze_schematic",
    "b": "figure1_panel_b_cycle_design",
    "c": "figure1_panel_c_rewarded_endnode_timelines",
    "d": "figure1_panel_d_reward_acquisition",
    "g": "figure1_panel_e_exploration_efficiency",
    "h": "figure1_panel_f_latent_endnode_timelines",
    "i": "figure1_panel_g_latent_return_long_bout",
    "j": "figure1_panel_h_first_visit_route_score",
}


def export_individual_panels(
    output_dir: Path,
    *,
    dpi: int = 300,
    formats: tuple[str, ...] = ("svg", "pdf", "png", "tiff"),
) -> dict[str, dict[str, str]]:
    """Export every active panel as a selectable source-native artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, dict[str, str]] = {}
    for panel_id, renderer in PANEL_RENDERERS.items():
        fig = renderer(_panel_size(panel_id))
        stem = PANEL_EXPORT_STEMS[panel_id]
        panel_paths: dict[str, str] = {}
        for ext in formats:
            ext = ext.lower()
            path = output_dir / f"{stem}.{ext}"
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.01,
                "facecolor": "white",
            }
            if ext == "tiff":
                save_kwargs.update(
                    {
                        "dpi": max(600, int(dpi)),
                        "pil_kwargs": {"compression": "tiff_lzw"},
                    }
                )
            elif ext == "png":
                save_kwargs["dpi"] = int(dpi)
            elif ext == "svg":
                save_kwargs["metadata"] = {"Date": None}
            elif ext == "pdf":
                pass
            else:
                raise ValueError(f"Unsupported individual-panel format: {ext}")
            fig.savefig(path, format=ext, **save_kwargs)
            panel_paths[ext] = str(path)
        plt.close(fig)
        exported[panel_id] = panel_paths
    return exported


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(value) for value in re.split(r"[\s,]+", view_box.strip()) if value]
        if len(values) == 4:
            return values[2], values[3]
    return _parse_svg_length(root.attrib["width"]), _parse_svg_length(root.attrib["height"])


def _parse_svg_length(value: str) -> float:
    match = re.match(r"^\s*([0-9.]+)\s*([a-z]*)\s*$", value)
    if not match:
        return float(value)
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"", "pt"}:
        return number
    if unit == "in":
        return number * 72.0
    if unit == "px":
        return number * 72.0 / 96.0
    if unit == "cm":
        return number * 72.0 / 2.54
    if unit == "mm":
        return number * 72.0 / 25.4
    return number


def _replace_id_references(value: str, id_map: dict[str, str]) -> str:
    for old_id, new_id in id_map.items():
        value = value.replace(f"url(#{old_id})", f"url(#{new_id})")
        if value == f"#{old_id}":
            value = f"#{new_id}"
        else:
            value = value.replace(f"#{old_id};", f"#{new_id};")
            value = value.replace(f"#{old_id})", f"#{new_id})")
            value = value.replace(f"#{old_id} ", f"#{new_id} ")
            value = value.replace(f"#{old_id}\"", f"#{new_id}\"")
            value = value.replace(f"#{old_id}'", f"#{new_id}'")
    return value


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for element in root.iter():
        old_id = element.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            element.attrib["id"] = new_id
    if not id_map:
        return
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if "#" in value:
                element.attrib[key] = _replace_id_references(value, id_map)
        if element.text and "#" in element.text:
            element.text = _replace_id_references(element.text, id_map)
        if element.tail and "#" in element.tail:
            element.tail = _replace_id_references(element.tail, id_map)


def _render_panel_svg(panel_id: str, renderer, size: tuple[float, float], *, dpi: int) -> dict:
    panel_fig = renderer(size)
    panel_fig.set_dpi(dpi)
    panel_fig.canvas.draw()
    canvas_renderer = panel_fig.canvas.get_renderer()
    tight_bbox_inches = panel_fig.get_tightbbox(canvas_renderer)
    pad_points = 0.01 * 72.0
    inner_axes_bounds_points = []
    for ax in panel_fig.axes:
        if not ax.get_visible():
            continue
        axes_bbox_inches = ax.get_window_extent(canvas_renderer).transformed(
            panel_fig.dpi_scale_trans.inverted()
        )
        inner_axes_bounds_points.append(
            [
                float((axes_bbox_inches.x0 - tight_bbox_inches.x0) * 72.0 + pad_points),
                float((tight_bbox_inches.y1 - axes_bbox_inches.y1) * 72.0 + pad_points),
                float(axes_bbox_inches.width * 72.0),
                float(axes_bbox_inches.height * 72.0),
            ]
        )
    buffer = io.StringIO()
    panel_fig.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=True,
        metadata={"Date": None},
    )
    plt.close(panel_fig)
    root = ET.fromstring(buffer.getvalue())
    _prefix_svg_ids(root, f"{panel_id}_")
    width, height = _parse_svg_viewbox(root)
    children = [child for child in list(root) if child.tag != _svg_tag("metadata")]
    return {
        "children": children,
        "viewbox_width": width,
        "viewbox_height": height,
        "size_inches": size,
        "inner_axes_width_points": [bounds[2] for bounds in inner_axes_bounds_points],
        "inner_axes_bounds_points": inner_axes_bounds_points,
    }


def _make_layout_axes(*, dpi: int):
    _set_common_rc()
    fig, axes = plt.subplot_mosaic(
        [list(row) for row in FIGURE1_MOSAIC],
        figsize=CANVAS_SIZE,
        dpi=dpi,
        empty_sentinel=".",
        gridspec_kw={
            "width_ratios": MOSAIC_WIDTH_RATIOS,
            "height_ratios": MOSAIC_HEIGHT_RATIOS,
            "wspace": 0.10,
            "hspace": 0.06,
        },
    )
    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.04, top=0.975)
    fig.canvas.draw()
    return fig, axes


def _slot_from_axes(ax) -> dict[str, float]:
    bbox = ax.get_position()
    figure_width = CANVAS_SIZE[0] * 72.0
    figure_height = CANVAS_SIZE[1] * 72.0
    x = bbox.x0 * figure_width
    y = (1.0 - bbox.y1) * figure_height
    width = bbox.width * figure_width
    height = bbox.height * figure_height
    return {"x": x, "y": y, "width": width, "height": height}


def _panel_id_position(
    panel_id: str,
    slot: dict[str, float],
    placed: dict[str, float],
) -> tuple[float, float]:
    x_rel, y_rel = PANEL_ID_POSITIONS.get(panel_id, (-0.055, 0.99))
    x_bounds = placed if panel_id in PLACED_BOUNDS_PANEL_IDS else slot
    x = max(2.0, x_bounds["x"] + x_rel * x_bounds["width"])
    y_base = placed["y"] if panel_id in PLACED_LABEL_Y_PANEL_IDS else slot["y"]
    if panel_id in OFFSET_Y_PANEL_IDS and panel_id not in PLACED_LABEL_Y_PANEL_IDS:
        y_base += PANEL_OFFSETS_POINTS.get(panel_id, (0.0, 0.0))[1]
    y_height = placed["height"] if panel_id in PLACED_LABEL_Y_PANEL_IDS else slot["height"]
    y = max(8.0, y_base + (1.0 - y_rel) * y_height)
    return x, y


def render_assembly_svg(*, dpi: int = 300) -> tuple[ET.ElementTree, dict]:
    _set_common_rc()
    layout_fig, axes = _make_layout_axes(dpi=dpi)
    figure_width = EXPORT_CANVAS_SIZE[0] * 72.0
    figure_height = EXPORT_CANVAS_SIZE[1] * 72.0
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{EXPORT_WIDTH_MM:.3f}mm",
            "height": f"{EXPORT_HEIGHT_MM:.3f}mm",
            "viewBox": f"0 0 {figure_width:.6f} {figure_height:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
    rendered_sizes = {}
    endnode_reference_scale = None
    compact_right_edge = None
    compact_left_edge = _slot_from_axes(axes["c"])["x"]
    original_panel_i_target_width = _slot_from_axes(axes["j"])["width"] * 1.03
    panel_svgs = {
        panel_id: _render_panel_svg(
            panel_id,
            renderer,
            _panel_size(panel_id),
            dpi=dpi,
        )
        for panel_id, renderer in PANEL_RENDERERS.items()
    }
    for panel_id, renderer in PANEL_RENDERERS.items():
        size = _panel_size(panel_id)
        panel_svg = panel_svgs[panel_id]
        slot = _slot_from_axes(axes[panel_id])
        scale = min(
            slot["width"] / panel_svg["viewbox_width"],
            slot["height"] / panel_svg["viewbox_height"],
        )
        scale *= float(PANEL_SCALE_ADJUST.get(panel_id, 1.0))
        if panel_id == "c":
            endnode_reference_scale = scale
        elif panel_id == "h" and endnode_reference_scale is not None:
            scale = endnode_reference_scale
        elif panel_id == "i":
            # Preserve the old panel i footprint after moving it into the
            # former panel h slot; the wider slot must not enlarge the bars.
            scale = min(
                original_panel_i_target_width / panel_svg["viewbox_width"],
                slot["height"] / panel_svg["viewbox_height"],
            )
        dx, dy = PANEL_OFFSETS_POINTS.get(panel_id, (0.0, 0.0))
        x = slot["x"] + dx
        y = slot["y"] + dy
        if panel_id in COMPACT_ENDNODE_PANELS:
            x = compact_left_edge + dx
        elif panel_id == "i":
            x = compact_left_edge + dx
        elif panel_id == "j":
            if compact_right_edge is None:
                raise RuntimeError("Panel c must be placed before panel h route score.")
            x = compact_right_edge - panel_svg["viewbox_width"] * scale + dx
        elif panel_id in RIGHT_ANCHORED_PANELS:
            x += slot["width"] - panel_svg["viewbox_width"] * scale
        elif panel_id not in LEFT_ANCHORED_PANELS:
            x += (slot["width"] - panel_svg["viewbox_width"] * scale) / 2.0
        if panel_id == "d":
            if endnode_reference_scale is None:
                raise RuntimeError("Panel c must be placed before panel d.")
            f_svg = panel_svgs["h"]
            f_slot = _slot_from_axes(axes["h"])
            _f_dx, f_dy = PANEL_OFFSETS_POINTS.get("h", (0.0, 0.0))
            f_axis = f_svg["inner_axes_bounds_points"][0]
            f_axis_bottom = (
                f_slot["y"]
                + f_dy
                + float(f_axis[1] + f_axis[3]) * endnode_reference_scale
            )
            d_axis = panel_svg["inner_axes_bounds_points"][0]
            d_axis_bottom = float(d_axis[1] + d_axis[3]) * scale
            y = f_axis_bottom - d_axis_bottom
        if panel_id == "c":
            compact_right_edge = x + panel_svg["viewbox_width"] * scale
        group = ET.SubElement(
            root,
            _svg_tag("g"),
            {
                "id": f"panel_{panel_id}",
                "transform": f"translate({x:.6f} {y:.6f}) scale({scale:.8f})",
            },
        )
        for child in panel_svg["children"]:
            group.append(child)
        placed = {
            "x": x,
            "y": y,
            "width": panel_svg["viewbox_width"] * scale,
            "height": panel_svg["viewbox_height"] * scale,
        }
        label_x, label_y = _panel_id_position(panel_id, slot, placed)
        label = ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{label_x:.6f}",
                "y": f"{label_y:.6f}",
                "font-family": FONT_FAMILY,
                "font-size": f"{PANEL_ID_SIZE:.1f}",
                "font-weight": "700",
                "fill": "black",
            },
        )
        label.text = PANEL_DISPLAY_LABELS.get(panel_id, panel_id)
        rendered_sizes[panel_id] = {
            "size_inches": size,
            "viewbox_points": [panel_svg["viewbox_width"], panel_svg["viewbox_height"]],
            "slot_points": [slot["width"], slot["height"]],
            "slot_bounds_points": [slot["x"], slot["y"], slot["width"], slot["height"]],
            "placed_bounds_points": [
                placed["x"],
                placed["y"],
                placed["width"],
                placed["height"],
            ],
            "inner_axes_width_points": [
                width * scale for width in panel_svg["inner_axes_width_points"]
            ],
            "placed_inner_axes_bounds_points": [
                [
                    x + bounds[0] * scale,
                    y + bounds[1] * scale,
                    bounds[2] * scale,
                    bounds[3] * scale,
                ]
                for bounds in panel_svg["inner_axes_bounds_points"]
            ],
            "panel_id_position_points": [label_x, label_y],
            "scale": scale,
        }
    plt.close(layout_fig)
    return ET.ElementTree(root), rendered_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("svg",),
        help="Output formats to write. The assembly path is vector-only and currently supports svg.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_tree, rendered_sizes = render_assembly_svg(dpi=int(args.dpi))
    saved = []
    for ext in tuple(args.formats):
        if ext.lower() != "svg":
            raise ValueError("The vector assembly path only writes SVG. Use --formats svg.")
        path = args.output_dir / f"{args.basename}.{ext}"
        svg_tree.write(path, encoding="utf-8", xml_declaration=True)
        saved.append(path)
    layout_path = args.output_dir / f"{args.basename}_layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "assembly": "vector_svg_composition",
                "export_width_mm": EXPORT_WIDTH_MM,
                "export_height_mm": EXPORT_HEIGHT_MM,
                "export_max_height_mm": EXPORT_MAX_HEIGHT_MM,
                "canvas_size": CANVAS_SIZE,
                "export_canvas_size": EXPORT_CANVAS_SIZE,
                "mosaic": FIGURE1_MOSAIC,
                "mosaic_width_ratios": MOSAIC_WIDTH_RATIOS,
                "mosaic_height_ratios": MOSAIC_HEIGHT_RATIOS,
                "panel_sizes": PANEL_SIZES,
                "panel_display_labels": PANEL_DISPLAY_LABELS,
                "rendered_sizes": rendered_sizes,
                "font_family": FONT_FAMILY,
                "base_font_size": BASE_FONT_SIZE,
                "axis_label_size": AXIS_LABEL_SIZE,
                "tick_label_size": TICK_LABEL_SIZE,
                "title_size": TITLE_SIZE,
                "panel_id_size": PANEL_ID_SIZE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        {
            "saved_paths": [str(path) for path in saved],
            "layout_path": str(layout_path),
        }
    )


if __name__ == "__main__":
    main()
