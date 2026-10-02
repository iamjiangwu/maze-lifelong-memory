from __future__ import annotations

import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis.compat import bootstrap_environment
from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    DEFAULT_PANEL_OUTPUT_DIR,
    PANEL_LAYOUT,
    apply_panel_rcparams,
    draw_linearized_matched_panel_axes,
    render_correlation_shuffle_distribution_panel,
    render_example_cell_grid_panel,
    render_paired_bar_scatter_panel,
    render_shuffle_mean_distribution_panel,
    render_spatial_decoding_error_map,
    render_stacked_local_correlation_panel,
)
import ca_analysis.figure_panels as figure_panels_module


_COMPAT = bootstrap_environment()
REPO_ROOT = Path(_COMPAT.repo_root)

DEFAULT_OUTPUT_DIR = Path("docs/figure_panel_previews")
DEFAULT_BASENAME = "figure2_assembly_preview_current"
DEFAULT_LAYOUT_SPEC = DEFAULT_OUTPUT_DIR / "figure2_layout_controls.json"
LAYOUT_SPEC_VERSION = 2
DEFAULT_LINEAR_PF_DATA_ROOT = None  # Supplied explicitly for the optional linearized panel.
LINEAR_PF_DATA_ROOT = DEFAULT_LINEAR_PF_DATA_ROOT

FONT_FAMILY = "Arial"
EXPORT_WIDTH_MM = 180.0
BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
TITLE_SIZE = 8.0
PANEL_ID_SIZE = 12.0
HEATMAP_CMAP = os.environ.get("FIGURE2_HEATMAP_CMAP", "magma")

CANVAS_SIZE = (11.8, 8.0)
FIGURE2_MOSAIC = [
    ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"],
    ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"],
    ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"],
    ["a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"],
    ["c", "c", "c", "c", "c", "d", "d", "e", "e", "e", "e", "e", "f", "f", "g", "g", "g", "g", "g", "g", "h", "h", "h", "h"],
    ["c", "c", "c", "c", "c", "d", "d", "e", "e", "e", "e", "e", "f", "f", "g", "g", "g", "g", "g", "g", "h", "h", "h", "h"],
    ["c", "c", "c", "c", "c", "d", "d", "e", "e", "e", "e", "e", "f", "f", "g", "g", "g", "g", "g", "g", "h", "h", "h", "h"],
    ["i", "i", "j", "j", "j", "j", "j", "k", "k", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l"],
    ["i", "i", "j", "j", "j", "j", "j", "k", "k", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l"],
    ["i", "i", "j", "j", "j", "j", "j", "k", "k", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l"],
    ["i", "i", "j", "j", "j", "j", "j", "k", "k", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l", "l"],
]
MOSAIC_WIDTH_RATIOS = [1.0] * 24
MOSAIC_HEIGHT_RATIOS = [0.80, 0.80, 0.80, 0.80, 0.57, 0.57, 0.57, 0.82, 0.82, 0.82, 0.82]
PANEL_SIZES = {
    "a": (2.20, 3.08),
    "b": (9.90, 3.08),
    "c": (2.25, 1.92),
    "d": (0.98, 2.14),
    "e": (2.25, 1.92),
    "f": (0.98, 2.14),
    "g": (2.60, 1.92),
    "h": (1.20, 2.14),
    "i": (1.12, 2.70),
    "j": (2.60, 2.70),
    "k": (1.12, 2.70),
    "l": (7.15, 2.70),
}
PANEL_ID_POSITIONS = {
    "b": (0.006, 0.99),
    "d": (0.00, 0.99),
    "f": (0.00, 0.99),
    "h": (0.00, 0.99),
    "l": (-0.035, 0.99),
}
PANEL_OFFSETS_POINTS = {
    "a": (-8.0, 0.0),
    "b": (0.0, 0.0),
    "i": (0.0, 18.0),
    "j": (0.0, 18.0),
    "k": (0.0, 18.0),
    "l": (-1.0, 18.0),
}
PANEL_ANCHORS = {
    "a": ("left", "center"),
    "b": ("left", "center"),
    "c": ("center", "top"),
    "d": ("center", "top"),
    "e": ("center", "top"),
    "f": ("center", "top"),
    "g": ("center", "top"),
    "h": ("center", "top"),
    "i": ("center", "top"),
    "j": ("center", "top"),
    "k": ("center", "top"),
    "l": ("center", "top"),
}
PANEL_SCALE_ADJUST = {
    "a": 1.00,
    "b": 1.00,
    "l": 1.00,
}
FIXED_BBOX_PANELS = set("cdefghijkl")
FULL_FRAME_EXPORT_PANELS = set("ijkl")

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
            "axes.linewidth": 0.75,
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


def _render_panel_a(size: tuple[float, float]):
    _set_common_rc()
    layout = {
        **PANEL_LAYOUT,
        "panel_size": size,
        "grid_hspace": 0.05,
    }
    style = {
        "origin": "upper",
        "shared_clim": True,
        "shared_vmin": None,
        "shared_vmax": None,
        "show_axes": False,
        "show_colorbars": False,
        "show_panel_title": False,
        "show_subplot_titles": False,
        "side_label_transform": "axes",
        "side_label_rotation": 90,
        "side_label_ha": "center",
        "side_label_va": "center",
        "side_label_fontsize": 8.0,
        "side_label_fontweight": "normal",
        "top_side_label_xy": (-0.060, 0.50),
        "bottom_side_label_xy": (-0.060, 0.50),
        "show_scale_bar": True,
        "scale_bar_panel": "bottom",
        "scale_bar_xy": (295.0, 278.0),
        "scale_bar_length_px": 60.0,
        "scale_bar_linewidth": 2.4,
        "scale_bar_color": "white",
        "scale_bar_text_color": "white",
        "scale_bar_text_offset_px": -10.0,
        "scale_bar_fontsize": 6.4,
        "scale_bar_fontweight": "normal",
    }
    fig, _axes, _data, _paths = render_stacked_local_correlation_panel(
        top_fig_path=Path(__file__).resolve().parents[1] / "panel_a_sources/422_corr.npz",
        bottom_fig_path=Path(__file__).resolve().parents[1] / "panel_a_sources/422-3_corr.npz",
        panel_size=size,
        layout=layout,
        style=style,
        labels={
            "top_side_label": "Day 1",
            "bottom_side_label": "Day 91",
            "scale_bar_label": "100 \u00b5m",
        },
        save=False,
    )
    axes_flat = list(_axes.ravel()) if hasattr(_axes, "ravel") else list(_axes)
    if len(axes_flat) >= 2:
        axes_flat[0].set_position([0.055, 0.506, 0.940, 0.492])
        axes_flat[1].set_position([0.055, 0.006, 0.940, 0.492])
    _standardize_axes_text(fig)
    return fig


def _render_panel_b(size: tuple[float, float]):
    _set_common_rc()
    picks = [
        ("468-9day1->468-12day1", 0),
        ("837-6day1->837-9day1", 588),
        ("451-6day1->451-9day1", 305),
        ("837-6day1->837-9day1", 175),
        ("418-3day1->418-6day1", 130),
        ("418-3day1->418-6day1", 62),
    ]
    fig, _axes, _data, _paths = render_example_cell_grid_panel(
        picks=picks,
        mode="arrow",
        saved_dir=REPO_ROOT / "docs" / "example_neuron_candidates" / "saved_picks",
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "grid_wspace": 0.012,
            "grid_hspace": 0.025,
            "margins": {"left": 0.042, "right": 0.998, "top": 0.990, "bottom": 0.078},
        },
        colors={
            "maze_line": "k",
            "trajectory": "0.55",
            "arrow_face": "red",
            "arrow_edge": "red",
        },
        style={
            "maze_linewidth": 1.15,
            "show_maze": True,
            "title_fontsize": 7.2,
            "title_location": "bottom",
            "title_y": -0.105,
            "row_label_fontsize": 7.4,
            "row_label_pad": 5,
            "trajectory_alpha": 0.28,
            "trajectory_linewidth": 0.26,
            "arrow_head_width": 0.14,
            "arrow_head_length": 0.34,
            "arrow_linewidth": 0.01,
            "arrow_alpha": 0.42,
            "arrow_scale": 0.25,
        },
        labels={
            "row_top": "Day i",
            "row_bottom": "Day i + 90",
            "title_template": "{place_r:.2f}",
        },
        save=False,
        n_bins=45,
        sigma=1.0,
        signal="C",
        dt=0.04,
        min_time_base=0.5,
    )
    _standardize_axes_text(fig)
    return fig


def _corr_distribution_panel(
    *,
    size: tuple[float, float],
    npz_name: str,
    color_key: str,
    xlabel: str,
    data_label_xy: tuple[float, float],
    shuffle_label_xy: tuple[float, float],
):
    _set_common_rc()
    fig, _ax, _data, _paths = render_correlation_shuffle_distribution_panel(
        npz_path=REPO_ROOT / "docs" / "three_month_correlation_2026-04-19" / npz_name,
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={"data": CORRELATION_MEASURE_COLORS[color_key], "shuffle": "#7f7f7f"},
        style={
            "data_alpha": 0.72,
            "shuffle_alpha": 0.75,
            "legend": False,
            "show_counts": False,
            "direct_labels": True,
            "data_label_xy": data_label_xy,
            "shuffle_label_xy": shuffle_label_xy,
            "direct_label_alpha": 0.98,
            "direct_label_fontsize": 8.2,
            "direct_label_fontweight": "bold",
            "direct_label_transform": "axes",
            "xlim": (-0.5, 1.0),
        },
        labels={
            "xlabel": xlabel,
            "ylabel": "Density",
            "data": "Data",
            "shuffle": "Shuffle",
        },
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.20, top=0.96)
    return fig


def _mean_shuffle_panel(
    *,
    size: tuple[float, float],
    npz_name: str,
    color_key: str,
    xlabel: str,
    xlim: tuple[float, float],
    data_label_xy: tuple[float, float],
    shuffle_label_xy: tuple[float, float],
):
    _set_common_rc()
    fig, _ax, _data, _paths = render_shuffle_mean_distribution_panel(
        npz_path=REPO_ROOT / "docs" / "three_month_correlation_2026-04-19" / npz_name,
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "shuffle": "#7f7f7f",
            "shuffle_edge": "#555555",
            "data": CORRELATION_MEASURE_COLORS[color_key],
        },
        style={
            "bins": 80,
            "hist_alpha": 0.75,
            "line_width": 1.7,
            "line_style": "--",
            "legend": False,
            "show_counts": False,
            "show_pvalue": False,
            "direct_labels": True,
            "data_label_xy": data_label_xy,
            "shuffle_label_xy": shuffle_label_xy,
            "direct_label_alpha": 0.98,
            "direct_label_fontsize": 7.2,
            "direct_label_fontweight": "bold",
            "direct_label_transform": "axes",
            "xlim": xlim,
        },
        labels={
            "xlabel": xlabel,
            "ylabel": "",
            "shuffle": "Shuffle",
            "data": "Data",
        },
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.20, top=0.96)
    return fig


def _render_panel_c(size: tuple[float, float]):
    return _corr_distribution_panel(
        size=size,
        npz_name="plot_data_cell228_place.npz",
        color_key="place_map_correlation",
        xlabel="",
        data_label_xy=(0.60, 0.48),
        shuffle_label_xy=(0.48, 0.82),
    )


def _render_panel_d(size: tuple[float, float]):
    return _mean_shuffle_panel(
        size=size,
        npz_name="plot_data_cell226_place.npz",
        color_key="place_map_correlation",
        xlabel="",
        xlim=(-0.005, 0.105),
        data_label_xy=(0.66, 0.56),
        shuffle_label_xy=(0.26, 0.78),
    )


def _render_panel_e(size: tuple[float, float]):
    return _corr_distribution_panel(
        size=size,
        npz_name="plot_data_cell228_pv.npz",
        color_key="pv_correlation",
        xlabel="",
        data_label_xy=(0.60, 0.38),
        shuffle_label_xy=(0.58, 0.86),
    )


def _render_panel_f(size: tuple[float, float]):
    return _mean_shuffle_panel(
        size=size,
        npz_name="plot_data_cell226_pv.npz",
        color_key="pv_correlation",
        xlabel="",
        xlim=(-0.02, 0.27),
        data_label_xy=(0.66, 0.56),
        shuffle_label_xy=(0.26, 0.78),
    )


def _render_panel_g(size: tuple[float, float]):
    return _corr_distribution_panel(
        size=size,
        npz_name="plot_data_cell228_rate.npz",
        color_key="ensemble_rate_correlation",
        xlabel="",
        data_label_xy=(0.72, 0.50),
        shuffle_label_xy=(0.58, 0.86),
    )


def _render_panel_h(size: tuple[float, float]):
    return _mean_shuffle_panel(
        size=size,
        npz_name="plot_data_cell226_rate.npz",
        color_key="ensemble_rate_correlation",
        xlabel="",
        xlim=(-0.06, 0.62),
        data_label_xy=(0.66, 0.56),
        shuffle_label_xy=(0.28, 0.78),
    )


def _render_panel_i(size: tuple[float, float]):
    _set_common_rc()
    fig, _ax, _data, _paths = render_paired_bar_scatter_panel(
        points_csv=REPO_ROOT / "docs" / "cross_session_decoding" / "cross_session_decoding_error_points.csv",
        summary_csv=REPO_ROOT / "docs" / "cross_session_decoding" / "cross_session_decoding_error_summary.csv",
        stats_json=REPO_ROOT / "docs" / "cross_session_decoding" / "cross_session_decoding_error_stats.json",
        value_col="euclidean_error",
        group_order=("Data", "Shuffle"),
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "Data": CORRELATION_MEASURE_COLORS["pv_correlation"],
            "Shuffle": "#7f7f7f",
            "pair_line": "#8a8a8a",
            "bar_edge": "none",
            "point_edge": "none",
        },
        style={
            "bar_width": 0.72,
            "bar_alpha": 0.78,
            "point_size": 12,
            "point_alpha": 0.65,
            "point_jitter": 0.03,
            "pair_line_alpha": 0.28,
            "pair_line_width": 0.65,
            "annotation": True,
            "annotation_p_path": ("cluster_ols", "p_less"),
            "annotation_y": 7.85,
            "annotation_fontsize": 8.0,
            "ylim": (0, 8.2),
            "yticks": [0, 2, 4, 6, 8],
        },
        labels={
            "ylabel": "Decoding error (bins)",
            "group_labels": {"Data": "Data", "Shuffle": "Shuffle"},
        },
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.42, right=0.98, bottom=0.155, top=0.900)
    return fig


def _render_panel_j(size: tuple[float, float]):
    _set_common_rc()
    fig, _ax, _data, _paths = render_spatial_decoding_error_map(
        map_csv=REPO_ROOT / "docs" / "cross_session_decoding" / "spatial_decoding_error_map.csv",
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "reference_panel_size": size,
            "colorbar_fraction": 0.060,
            "colorbar_pad": 0.020,
            "colorbar_location": "left",
        },
        colors={
            "cmap": HEATMAP_CMAP,
            "bad": "white",
            "wall": "black",
            "reward_square": CORRELATION_MEASURE_COLORS["pv_correlation"],
            "control_square": "#7f7f7f",
        },
        style={
            "vmin": 1.0,
            "vmax": 14.0,
            "image_alpha": 0.95,
            "maze_wall_linewidth": 0.85,
            "square_linewidth": 1.45,
            "square_size": 1.10,
        },
        labels={"colorbar": "Decoding error (bins)"},
        save=False,
    )
    _standardize_axes_text(fig)
    map_ax = fig.axes[0]
    colorbar_ax = fig.axes[-1]
    heatmap_height = 0.745
    heatmap_width = heatmap_height * size[1] / size[0]
    map_ax.set_position([0.140, 0.155, heatmap_width, heatmap_height])
    colorbar_ax.set_position([0.084, 0.155, 0.026, heatmap_height])
    colorbar_ax.yaxis.set_ticks_position("left")
    colorbar_ax.yaxis.set_label_position("left")
    colorbar_ax.yaxis.labelpad = 1.0
    colorbar_ax.tick_params(axis="y", pad=1.0, length=2.0)
    map_df = pd.read_csv(REPO_ROOT / "docs" / "cross_session_decoding" / "spatial_decoding_error_map.csv")
    reward_rows = map_df.loc[map_df["is_reward"].astype(bool)]
    if not reward_rows.empty:
        reward_row = reward_rows.iloc[0]
        reward_xy_axes = map_ax.transAxes.inverted().transform(
            map_ax.transData.transform((float(reward_row["bx"]), float(reward_row["by"])))
        )
        figure_panels_module._add_timeline_water_drop_icon(  # Same icon geometry used by Figure 1 timeline panels.
            map_ax,
            (1.0, float(reward_xy_axes[1])),
            {
                "kind": "water",
                "size": 12.0,
                "linewidth": 0.65,
                "x_offset_points": 6.0,
                "zorder": 20,
            },
        )
    return fig


def _render_panel_k(size: tuple[float, float]):
    _set_common_rc()
    fig, _ax, _data, _paths = render_paired_bar_scatter_panel(
        points_csv=REPO_ROOT / "docs" / "cross_session_decoding" / "reward_control_decoding_error_points.csv",
        summary_csv=REPO_ROOT / "docs" / "cross_session_decoding" / "reward_control_decoding_error_summary.csv",
        stats_json=REPO_ROOT / "docs" / "cross_session_decoding" / "reward_control_decoding_error_stats.json",
        value_col="decoding_error",
        group_order=("Reward", "Ctrls"),
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "Reward": CORRELATION_MEASURE_COLORS["pv_correlation"],
            "Ctrls": "#7f7f7f",
            "pair_line": "#8a8a8a",
            "bar_edge": "none",
            "point_edge": "none",
        },
        style={
            "bar_width": 0.72,
            "bar_alpha": 0.78,
            "point_size": 13,
            "point_alpha": 0.65,
            "point_jitter": 0.03,
            "pair_line_alpha": 0.30,
            "pair_line_width": 0.65,
            "annotation": True,
            "annotation_p_path": ("cluster_ols", "p_less"),
            "annotation_y": 9.55,
            "annotation_fontsize": 8.0,
            "ylim": (0, 10.2),
            "yticks": [0, 5, 10],
        },
        labels={
            "ylabel": "Decoding error (bins)",
            "group_labels": {"Reward": "Reward", "Ctrls": "Ctrls"},
        },
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.42, right=0.98, bottom=0.155, top=0.900)
    return fig


def _render_panel_l(
    size: tuple[float, float],
    *,
    heatmap_vmax: float | None = None,
):
    _set_common_rc()
    data_root = LINEAR_PF_DATA_ROOT
    layout = {
        **PANEL_LAYOUT,
        "panel_size": size,
        "grid_wspace": 0.10,
        "width_ratios": (0.5, 0.5, 0.62, 0.34, 0.5, 0.5, 0.62),
    }
    style = {
        "heatmap_percentile_vmax": 99.0,
        "heatmap_vmax": heatmap_vmax,
        "display_nan_as_zero": True,
        "nan_fill_value": 0.0,
        "density_sigma": 1.5,
        "density_xlim_max": 0.2,
        "density_linewidth": 1.1,
        "density_alpha": 0.22,
        "grid_alpha": 0.2,
        "legend_loc": "upper right",
        "show_colorbar": True,
        "colorbar_location": "right",
        "colorbar_orientation": "vertical",
        "show_water_port_band": False,
    }
    apply_panel_rcparams(
        {
            "font.family": FONT_FAMILY,
            "font.size": 7.2,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
        },
        panel_size=size,
        layout=layout,
    )
    fig = plt.figure(figsize=size)
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=layout["width_ratios"],
        wspace=float(layout["grid_wspace"]),
    )
    left_early_ax = fig.add_subplot(grid[0, 0])
    left_late_ax = fig.add_subplot(grid[0, 1], sharey=left_early_ax)
    left_density_ax = fig.add_subplot(grid[0, 2])
    right_early_ax = fig.add_subplot(grid[0, 4])
    right_late_ax = fig.add_subplot(grid[0, 5], sharey=right_early_ax)
    right_density_ax = fig.add_subplot(grid[0, 6])
    shared_colorbar_ax = fig.add_axes([0.955, 0.255, 0.018, 0.52])
    specs = [
        {
            "run_label": "home_to_water",
            "labels": {
                "early": "Day i",
                "late": "Day i + 90",
                "heatmap_xlabel": "Home to water",
                "heatmap_ylabel": "Neurons",
                "density_title": " ",
                "density_xlabel": "Density",
                "density_ylabel": "Home to water",
                "colorbar_label": "Normalized rate",
            },
        },
        {
            "run_label": "water_to_home",
            "style": {"reverse_position_axis": True, "show_colorbar": False},
            "labels": {
                "early": "Day i",
                "late": "Day i + 90",
                "heatmap_xlabel": "Water to home",
                "heatmap_ylabel": "Neurons",
                "density_title": " ",
                "density_xlabel": "Density",
                "density_ylabel": "Water to home",
                "colorbar_label": "Normalized rate",
            },
        },
    ]
    run_panel_axes = []

    def _detect_n_bins(run_label: str) -> int:
        files = sorted(Path(data_root).glob(f"*__{run_label}__pf1_matched.npy"))
        if not files:
            files = sorted(Path(data_root).glob(f"*__{run_label}__pf_all.npy"))
        if not files:
            return 22
        return int(np.load(files[0], mmap_mode="r").shape[1])

    for spec, early_ax, late_ax, density_ax, colorbar_ax in zip(
        specs,
        [left_early_ax, right_early_ax],
        [left_late_ax, right_late_ax],
        [left_density_ax, right_density_ax],
        [shared_colorbar_ax, None],
        strict=False,
    ):
        n_bins = _detect_n_bins(spec["run_label"])
        last_bin = n_bins - 1
        xticks = [tick for tick in (0, 5, 10, 15) if tick < last_bin] + [last_bin]
        panel_axes = draw_linearized_matched_panel_axes(
            early_ax,
            late_ax,
            density_ax,
            colorbar_ax=colorbar_ax,
            data_root=data_root,
            run_label=spec["run_label"],
            stable_threshold=figure_panels_module.STABLE_THR,
            pair_filter="day1_min_runs",
            heatmap_neuron_set="all",
            max_display=None,
            colors={
                "heatmap": HEATMAP_CMAP,
                "stable": "salmon",
                "unstable": "gray",
                "water_port_band": "#d95f02",
            },
            style={
                **style,
                **spec.get("style", {}),
                "exclude_pair_keys": [("1616day1", "1616bday1")],
            },
            labels=spec["labels"],
            xticks=xticks,
        )
        panel_axes["earlier"].set_title(spec["labels"]["early"])
        handles, labels = panel_axes["density"].get_legend_handles_labels()
        labels = ["Stable" if label.startswith("Stable") else "Unstable" for label in labels]
        panel_axes["density"].legend(handles, labels, frameon=False, loc="upper right")
        panel_axes["density"].yaxis.set_label_position("right")
        panel_axes["density"].yaxis.tick_right()
        panel_axes["density"].set_ylabel("")
        panel_axes["density"].tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
        panel_axes["density"].set_ylim(last_bin, 0)
        panel_axes["density"].set_yticks(xticks)
        panel_axes["density"].grid(False, axis="x")
        run_panel_axes.append(panel_axes)
    fig.subplots_adjust(left=0.060, right=0.930, bottom=0.155, top=0.900)
    shared_colorbar_ax.set_position([0.934, 0.270, 0.014, 0.500])
    shared_colorbar_ax.yaxis.set_ticks_position("right")
    shared_colorbar_ax.yaxis.set_label_position("right")
    _standardize_axes_text(fig, font_scale=0.86)
    right_density_ax = run_panel_axes[-1]["density"]
    right_density_ax.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    right_density_ax.set_yticklabels([])
    shared_colorbar_ax.set_ylabel("")
    shared_colorbar_ax.tick_params(axis="y", pad=1.0, length=2.0)
    fig.text(
        0.992,
        0.520,
        "Normalized rate",
        rotation=90,
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_SIZE * 0.86,
        fontfamily=FONT_FAMILY,
    )
    for panel_axes in run_panel_axes:
        density_ax = panel_axes["density"]
        legend = density_ax.get_legend()
        if legend is not None:
            legend.remove()
        handles, labels = density_ax.get_legend_handles_labels()
        labels = ["Stable" if label.startswith("Stable") else "Unstable" for label in labels]
        density_ax.legend(
            handles,
            labels,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            borderaxespad=0.0,
            fontsize=5.2,
            handlelength=1.0,
            handletextpad=0.28,
            labelspacing=0.22,
        )
    return fig


PANEL_RENDERERS = {
    "a": _render_panel_a,
    "b": _render_panel_b,
    "c": _render_panel_c,
    "d": _render_panel_d,
    "e": _render_panel_e,
    "f": _render_panel_f,
    "g": _render_panel_g,
    "h": _render_panel_h,
    "i": _render_panel_i,
    "j": _render_panel_j,
    "k": _render_panel_k,
    "l": _render_panel_l,
}

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


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


def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(value) for value in re.split(r"[\s,]+", view_box.strip()) if value]
        if len(values) == 4:
            return values[2], values[3]
    return _parse_svg_length(root.attrib["width"]), _parse_svg_length(root.attrib["height"])


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
    buffer = io.StringIO()
    fixed_bbox = panel_id in FIXED_BBOX_PANELS
    rc_overrides = {"savefig.bbox": "standard"} if panel_id in FULL_FRAME_EXPORT_PANELS else {}
    with matplotlib.rc_context(rc_overrides):
        panel_fig.savefig(
            buffer,
            format="svg",
            dpi=dpi,
            bbox_inches=None if fixed_bbox else "tight",
            pad_inches=0.0 if fixed_bbox else 0.01,
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
    }


def _make_layout_axes(*, dpi: int):
    _set_common_rc()
    fig, axes = plt.subplot_mosaic(
        FIGURE2_MOSAIC,
        figsize=CANVAS_SIZE,
        dpi=dpi,
        empty_sentinel=".",
        gridspec_kw={
            "width_ratios": MOSAIC_WIDTH_RATIOS,
            "height_ratios": MOSAIC_HEIGHT_RATIOS,
            "wspace": 0.045,
            "hspace": 0.085,
        },
    )
    fig.subplots_adjust(left=0.035, right=0.988, bottom=0.045, top=0.975)
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
    label_rel: tuple[float, float] | None = None,
) -> tuple[float, float]:
    x_rel, y_rel = label_rel if label_rel is not None else PANEL_ID_POSITIONS.get(panel_id, (-0.060, 0.99))
    x = max(2.0, slot["x"] + x_rel * slot["width"])
    y = max(8.0, slot["y"] + (1.0 - y_rel) * slot["height"])
    return x, y


def _default_layout_state(*, dpi: int) -> dict:
    layout_fig, axes = _make_layout_axes(dpi=dpi)
    try:
        slots = {panel_id: _slot_from_axes(axes[panel_id]) for panel_id in PANEL_RENDERERS}
    finally:
        plt.close(layout_fig)
    panels = {}
    for panel_id in PANEL_RENDERERS:
        anchor_x, anchor_y = PANEL_ANCHORS.get(panel_id, ("center", "center"))
        dx, dy = PANEL_OFFSETS_POINTS.get(panel_id, (0.0, 0.0))
        label_x_rel, label_y_rel = PANEL_ID_POSITIONS.get(panel_id, (-0.060, 0.99))
        label_x, label_y = _panel_id_position(panel_id, slots[panel_id], (label_x_rel, label_y_rel))
        panels[panel_id] = {
            "slot": slots[panel_id],
            "size": PANEL_SIZES[panel_id],
            "anchor": (anchor_x, anchor_y),
            "offset": (dx, dy),
            "scale_adjust": float(PANEL_SCALE_ADJUST.get(panel_id, 1.0)),
            "label_rel": (label_x_rel, label_y_rel),
            "label_abs": (label_x, label_y),
            "label_size": PANEL_ID_SIZE,
        }
    return {"canvas_size": CANVAS_SIZE, "panels": panels}


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _export_height_mm(canvas_size: tuple[float, float]) -> float:
    return EXPORT_WIDTH_MM * float(canvas_size[1]) / float(canvas_size[0])


def build_editable_layout_spec(*, dpi: int = 300) -> dict:
    state = _default_layout_state(dpi=dpi)
    panels = {}
    for panel_id, panel in state["panels"].items():
        slot = panel["slot"]
        size = panel["size"]
        dx, dy = panel["offset"]
        anchor_x, anchor_y = panel["anchor"]
        label_x_rel, label_y_rel = panel["label_rel"]
        label_x, label_y = panel["label_abs"]
        panels[panel_id] = {
            "x": _round_float(slot["x"] / 72.0),
            "y": _round_float(slot["y"] / 72.0),
            "w": _round_float(slot["width"] / 72.0),
            "h": _round_float(slot["height"] / 72.0),
            "render_w": _round_float(size[0]),
            "render_h": _round_float(size[1]),
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "dx_pt": _round_float(dx, 2),
            "dy_pt": _round_float(dy, 2),
            "scale_adjust": _round_float(panel["scale_adjust"], 3),
            "label_x_rel": _round_float(label_x_rel, 3),
            "label_y_rel": _round_float(label_y_rel, 3),
            "label_x": _round_float(label_x / 72.0),
            "label_y": _round_float(label_y / 72.0),
            "label_size": _round_float(panel["label_size"], 2),
        }
    return {
        "schema": "figure2_assembly_layout",
        "schema_version": LAYOUT_SPEC_VERSION,
        "units": "inches for x/y/w/h and render_w/render_h; points for dx_pt/dy_pt",
        "notes": [
            "x and y place the upper-left corner of the panel slot on the final canvas.",
            "w and h set the available slot in the final canvas.",
            "render_w and render_h set the native panel aspect before SVG composition.",
            "label_x and label_y place panel IDs independently from panel slots.",
            "Use draw-slots previews to check panel boxes before updating notebook code.",
        ],
        "canvas_size": list(state["canvas_size"]),
        "panels": panels,
    }


def _layout_state_from_spec(path: Path, *, dpi: int) -> dict:
    state = _default_layout_state(dpi=dpi)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "canvas_size" in spec:
        canvas = spec["canvas_size"]
        if len(canvas) != 2:
            raise ValueError(f"Expected canvas_size with 2 values in {path}")
        state["canvas_size"] = (float(canvas[0]), float(canvas[1]))
    panel_specs = spec.get("panels", {})
    if not isinstance(panel_specs, dict):
        raise ValueError(f"Expected a panels object in {path}")
    for panel_id, panel_spec in panel_specs.items():
        if panel_id not in state["panels"]:
            continue
        panel = state["panels"][panel_id]
        if all(key in panel_spec for key in ("x", "y", "w", "h")):
            panel["slot"] = {
                "x": float(panel_spec["x"]) * 72.0,
                "y": float(panel_spec["y"]) * 72.0,
                "width": float(panel_spec["w"]) * 72.0,
                "height": float(panel_spec["h"]) * 72.0,
            }
        render_w = float(panel_spec.get("render_w", panel["size"][0]))
        render_h = float(panel_spec.get("render_h", panel["size"][1]))
        panel["size"] = (render_w, render_h)
        panel["anchor"] = (
            str(panel_spec.get("anchor_x", panel["anchor"][0])),
            str(panel_spec.get("anchor_y", panel["anchor"][1])),
        )
        panel["offset"] = (
            float(panel_spec.get("dx_pt", panel["offset"][0])),
            float(panel_spec.get("dy_pt", panel["offset"][1])),
        )
        panel["scale_adjust"] = float(panel_spec.get("scale_adjust", panel["scale_adjust"]))
        panel["label_rel"] = (
            float(panel_spec.get("label_x_rel", panel["label_rel"][0])),
            float(panel_spec.get("label_y_rel", panel["label_rel"][1])),
        )
        if "label_x" in panel_spec and "label_y" in panel_spec:
            panel["label_abs"] = (float(panel_spec["label_x"]) * 72.0, float(panel_spec["label_y"]) * 72.0)
        else:
            panel["label_abs"] = _panel_id_position(panel_id, panel["slot"], panel["label_rel"])
        panel["label_size"] = float(panel_spec.get("label_size", panel.get("label_size", PANEL_ID_SIZE)))
    state["shared_xlabels"] = spec.get("shared_xlabels", [])
    return state


def _draw_slot_overlay(root: ET.Element, panel_id: str, slot: dict[str, float]) -> None:
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {
            "x": f"{slot['x']:.6f}",
            "y": f"{slot['y']:.6f}",
            "width": f"{slot['width']:.6f}",
            "height": f"{slot['height']:.6f}",
            "fill": "none",
            "stroke": "#2563eb",
            "stroke-width": "0.8",
            "stroke-dasharray": "3 2",
            "opacity": "0.70",
        },
    )
    text = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{slot['x'] + 2.5:.6f}",
            "y": f"{slot['y'] + 9.0:.6f}",
            "font-family": FONT_FAMILY,
            "font-size": "7",
            "font-weight": "700",
            "fill": "#2563eb",
        },
    )
    text.text = panel_id


def _draw_shared_xlabels(root: ET.Element, layout_state: dict) -> None:
    for item in layout_state.get("shared_xlabels", []):
        text_value = str(item.get("text", ""))
        if not text_value:
            continue
        text = ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{float(item['x']) * 72.0:.6f}",
                "y": f"{float(item['y']) * 72.0:.6f}",
                "font-family": FONT_FAMILY,
                "font-size": f"{float(item.get('fontsize', AXIS_LABEL_SIZE)):.1f}",
                "font-weight": str(item.get("fontweight", "400")),
                "fill": str(item.get("color", "black")),
                "text-anchor": "middle",
            },
        )
        text.text = text_value


def render_assembly_svg(
    *,
    dpi: int = 300,
    layout_spec: Path | None = None,
    draw_slots: bool = False,
) -> tuple[ET.ElementTree, dict, dict]:
    _set_common_rc()
    layout_state = _layout_state_from_spec(layout_spec, dpi=dpi) if layout_spec else _default_layout_state(dpi=dpi)
    canvas_size = tuple(layout_state["canvas_size"])
    figure_width = canvas_size[0] * 72.0
    figure_height = canvas_size[1] * 72.0
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{EXPORT_WIDTH_MM:.3f}mm",
            "height": f"{_export_height_mm(canvas_size):.3f}mm",
            "viewBox": f"0 0 {figure_width:.6f} {figure_height:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
    rendered_sizes = {}
    for panel_id, renderer in PANEL_RENDERERS.items():
        panel_layout = layout_state["panels"][panel_id]
        size = tuple(panel_layout["size"])
        panel_svg = _render_panel_svg(panel_id, renderer, size, dpi=dpi)
        slot = dict(panel_layout["slot"])
        scale = min(
            slot["width"] / panel_svg["viewbox_width"],
            slot["height"] / panel_svg["viewbox_height"],
        )
        scale *= float(panel_layout["scale_adjust"])
        dx, dy = panel_layout["offset"]
        anchor_x, anchor_y = panel_layout["anchor"]
        rendered_width = panel_svg["viewbox_width"] * scale
        rendered_height = panel_svg["viewbox_height"] * scale
        if anchor_x == "left":
            x = slot["x"] + dx
        elif anchor_x == "right":
            x = slot["x"] + slot["width"] - rendered_width + dx
        else:
            x = slot["x"] + (slot["width"] - rendered_width) / 2.0 + dx
        if anchor_y == "top":
            y = slot["y"] + dy
        elif anchor_y == "bottom":
            y = slot["y"] + slot["height"] - rendered_height + dy
        else:
            y = slot["y"] + (slot["height"] - rendered_height) / 2.0 + dy
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
        label_x, label_y = panel_layout.get("label_abs") or _panel_id_position(panel_id, slot, panel_layout["label_rel"])
        label = ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{label_x:.6f}",
                "y": f"{label_y:.6f}",
                "font-family": FONT_FAMILY,
                "font-size": f"{float(panel_layout.get('label_size', PANEL_ID_SIZE)):.1f}",
                "font-weight": "700",
                "fill": "black",
            },
        )
        label.text = panel_id
        rendered_sizes[panel_id] = {
            "size_inches": size,
            "viewbox_points": [panel_svg["viewbox_width"], panel_svg["viewbox_height"]],
            "slot_points": [slot["width"], slot["height"]],
            "scale": scale,
            "slot_inches": [slot["x"] / 72.0, slot["y"] / 72.0, slot["width"] / 72.0, slot["height"] / 72.0],
        }
    _draw_shared_xlabels(root, layout_state)
    if draw_slots:
        for panel_id, panel_layout in layout_state["panels"].items():
            _draw_slot_overlay(root, panel_id, panel_layout["slot"])
    return ET.ElementTree(root), rendered_sizes, layout_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--layout-spec",
        type=Path,
        default=None,
        help="Optional editable JSON layout spec with absolute panel slots.",
    )
    parser.add_argument(
        "--init-layout-spec",
        type=Path,
        nargs="?",
        const=DEFAULT_LAYOUT_SPEC,
        default=None,
        help="Write a default editable layout spec before rendering. Defaults to docs/figure_panel_previews/figure2_layout_controls.json.",
    )
    parser.add_argument(
        "--draw-slots",
        action="store_true",
        help="Overlay dashed panel slot boxes on the assembled SVG for layout debugging.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("svg",),
        help="Output formats to write. The vector assembly path currently supports svg.",
    )
    parser.add_argument(
        "--linear-pf-data-root",
        type=Path,
        default=DEFAULT_LINEAR_PF_DATA_ROOT,
        help="Data root for panel L linearized place-field arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global LINEAR_PF_DATA_ROOT
    LINEAR_PF_DATA_ROOT = args.linear_pf_data_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout_spec = args.layout_spec
    if args.init_layout_spec is not None:
        args.init_layout_spec.parent.mkdir(parents=True, exist_ok=True)
        args.init_layout_spec.write_text(
            json.dumps(build_editable_layout_spec(dpi=int(args.dpi)), indent=2),
            encoding="utf-8",
        )
        if layout_spec is None:
            layout_spec = args.init_layout_spec
    svg_tree, rendered_sizes, layout_state = render_assembly_svg(
        dpi=int(args.dpi),
        layout_spec=layout_spec,
        draw_slots=bool(args.draw_slots),
    )
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
                "canvas_size": layout_state["canvas_size"],
                "export_width_mm": EXPORT_WIDTH_MM,
                "export_height_mm": _export_height_mm(tuple(layout_state["canvas_size"])),
                "mosaic": FIGURE2_MOSAIC,
                "mosaic_width_ratios": MOSAIC_WIDTH_RATIOS,
                "mosaic_height_ratios": MOSAIC_HEIGHT_RATIOS,
                "layout_spec": str(layout_spec) if layout_spec is not None else None,
                "linear_pf_data_root": str(LINEAR_PF_DATA_ROOT),
                "panel_sizes": {
                    panel_id: tuple(panel["size"]) for panel_id, panel in layout_state["panels"].items()
                },
                "panel_slots": {
                    panel_id: {
                        "x": panel["slot"]["x"] / 72.0,
                        "y": panel["slot"]["y"] / 72.0,
                        "w": panel["slot"]["width"] / 72.0,
                        "h": panel["slot"]["height"] / 72.0,
                    }
                    for panel_id, panel in layout_state["panels"].items()
                },
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
    print({"saved_paths": [str(path) for path in saved], "layout_path": str(layout_path)})


if __name__ == "__main__":
    main()
