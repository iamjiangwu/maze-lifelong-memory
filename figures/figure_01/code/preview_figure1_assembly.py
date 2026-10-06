from __future__ import annotations

from pathlib import Path
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
    render_endnode_timeline_grid_panel,
    render_exploration_efficiency_histogram_panel,
    render_grouped_bar_scatter_panel,
    render_maze_schematic_panel,
    render_reward_acquisition_month_panel,
    get_panel_maze,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


FONT_FAMILY = "Arial"
EXPORT_WIDTH_MM = 180.0
CANVAS_SIZE = (10.0, 9.35)

BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
TITLE_SIZE = 8.0
PANEL_ID_SIZE = 13.0
PANEL_SIZES = {
    "a": (2.60, 2.60),
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
FIGURE1_RETURN_COLOR = "#DA2323"
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
            "xlabel": "Exploration efficiency",
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
    "c": _render_panel_c,
    "d": _render_panel_d,
    "g": _render_panel_exploration,
    "h": _render_panel_h,
    "i": _render_panel_i,
    "j": _render_panel_route_score,
}


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


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
