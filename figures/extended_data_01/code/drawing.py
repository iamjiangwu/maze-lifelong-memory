"""ED1 page assembly from derived plotting tables."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from release_rendering.statistics import sem, significance_stars, cr1_ols, linear_contrast, benjamini_hochberg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from release_rendering.plotting import with_figure_font, savefig_reproducibly
MONTH_ORDER = ("0mo", "3mo", "6mo", "9mo", "12mo", "experienced_day2")

MONTH_LABELS = {
    "0mo": "Naive day",
    "3mo": "3 mo return",
    "6mo": "6 mo return",
    "9mo": "9 mo return",
    "12mo": "12 mo return",
    "experienced_day2": "Second day",
}

GROUP_ORDER = ("naive", "return", "experienced_day2")

GROUP_LABELS = {
    "naive": "Naive day",
    "return": "Return day",
    "experienced_day2": "Second day",
}
_MONTH_COLORS = {
    "0mo": "#78ADD2",
    "3mo": "#F2B3AE",
    "6mo": "#E77D72",
    "9mo": "#C84B44",
    "12mo": "#8E2C2A",
    "experienced_day2": "#5B8C5A",
}

_GROUP_COLORS = {
    "naive": "#78ADD2",
    "return": "#C84B44",
    "experienced_day2": "#5B8C5A",
}

_AXIS_COLOR = "#303030"

_LIGHT_GREY = "#C9C9C9"

_RECORD_STYLE = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "font.size": 5.7,
    "axes.titlesize": 6.0,
    "axes.labelsize": 5.7,
    "xtick.labelsize": 5.0,
    "ytick.labelsize": 5.0,
    "legend.fontsize": 4.6,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
}

def _draw_record_cumulative_curve(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    y_label: str,
) -> None:
    """Draw one cumulative mean and s.e.m. curve on the publication canvas."""

    for group_key in MONTH_ORDER:
        group = values.loc[values["group_key"].eq(group_key)].sort_values(x_column)
        if group.empty:
            continue
        x = group[x_column].to_numpy(float)
        mean = group["mean"].to_numpy(float)
        error = group["sem"].to_numpy(float)
        axis.plot(
            x,
            mean,
            color=_MONTH_COLORS[group_key],
            linewidth=0.9,
            linestyle="--" if group_key == "experienced_day2" else "-",
            label=MONTH_LABELS[group_key],
        )
        finite = np.isfinite(error)
        axis.fill_between(
            x[finite],
            (mean - error)[finite],
            (mean + error)[finite],
            color=_MONTH_COLORS[group_key],
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.grid(False)

def _draw_record_acquisition_grid(
    subgrid: object,
    points: pd.DataFrame,
    statistics: pd.DataFrame,
    random_seed: int,
) -> list[plt.Axes]:
    """Draw the six acquisition endpoint comparisons."""

    axes: list[plt.Axes] = []
    rng = np.random.default_rng(random_seed)
    endpoints = (
        ("first_reward", "First reward"),
        ("tenth_reward", "Tenth reward"),
        ("first_long_bout", "First long bout"),
    )
    axes_to_draw = (
        ("maze_time_s", "Time in maze (s)"),
        ("maze_tiles", "Steps in maze"),
    )
    for row_index, (axis_key, y_label) in enumerate(axes_to_draw):
        # The three columns share one y range because only the first column has
        # tick labels.
        row_points = points.loc[
            points["axis_key"].eq(axis_key)
            & points["endpoint_key"].isin([key for key, _ in endpoints])
            & points["group_key"].isin(GROUP_ORDER)
        ]
        span = max(float(pd.to_numeric(row_points["value"], errors="coerce").max()), 1.0)
        row_axes: list[plt.Axes] = []
        row_top = 0.0
        for column_index, (endpoint_key, endpoint_label) in enumerate(endpoints):
            axis = plt.subplot(subgrid[row_index, column_index])
            axes.append(axis)
            panel_points = points.loc[
                points["endpoint_key"].eq(endpoint_key)
                & points["axis_key"].eq(axis_key)
            ]
            all_values: list[float] = []
            for x_position, group_key in enumerate(GROUP_ORDER):
                group_points = panel_points.loc[
                    panel_points["group_key"].eq(group_key)
                ].sort_values("session")
                values = pd.to_numeric(
                    group_points["value"],
                    errors="coerce",
                ).dropna().to_numpy(float)
                all_values.extend(values.tolist())
                axis.scatter(
                    x_position + rng.uniform(-0.13, 0.13, values.size),
                    values,
                    s=3.2,
                    facecolor="none",
                    edgecolor=_GROUP_COLORS[group_key],
                    linewidth=0.35,
                    alpha=0.55,
                    zorder=2,
                )
                mean_value = float(np.mean(values))
                error = sem(values)
                axis.bar(
                    x_position,
                    mean_value,
                    width=0.55,
                    color=_GROUP_COLORS[group_key],
                    alpha=0.55,
                    zorder=1,
                )
                axis.errorbar(
                    x_position,
                    mean_value,
                    yerr=error,
                    color=_AXIS_COLOR,
                    linewidth=0,
                    elinewidth=0.55,
                    capsize=1.2,
                    zorder=3,
                )
            axis.set_xlim(-0.55, 2.55)
            raw_max = max(all_values)
            base = raw_max + 0.055 * span
            step = 0.200 * span
            tick_height = 0.022 * span
            annotations = 0
            group_to_x = {key: index for index, key in enumerate(GROUP_ORDER)}
            panel_statistics = statistics.loc[
                statistics["endpoint_key"].eq(endpoint_key)
                & statistics["axis_key"].eq(axis_key)
            ]
            for contrast in panel_statistics.itertuples(index=False):
                stars = significance_stars(
                    float(contrast.p_bh18),
                    thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")),
                )
                if stars == "n.s.":
                    continue
                left = group_to_x[str(contrast.group_a)]
                right = group_to_x[str(contrast.group_b)]
                y = base + annotations * step
                axis.plot(
                    [left, left, right, right],
                    [y, y + tick_height, y + tick_height, y],
                    color=_AXIS_COLOR,
                    linewidth=0.48,
                    clip_on=False,
                )
                axis.text(
                    0.5 * (left + right),
                    y + tick_height + 0.006 * span,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=4.5,
                )
                annotations += 1
            annotation_top = (
                base + max(annotations - 1, 0) * step + tick_height + 0.22 * span
            )
            row_top = max(row_top, raw_max * 1.08, annotation_top)
            row_axes.append(axis)
            if row_index == 0:
                axis.set_title(endpoint_label, pad=4.0)
                axis.tick_params(axis="x", labelbottom=False)
            else:
                axis.set_xticks(
                    range(3),
                    ["Naive", "Return", "Second"],
                    rotation=28,
                    ha="right",
                )
            if column_index == 0:
                axis.set_ylabel(y_label)
            else:
                axis.tick_params(axis="y", labelleft=False)
            axis.grid(False)
        for row_axis in row_axes:
            row_axis.set_ylim(0, row_top)
    return axes

def _draw_record_long_bout_relation(
    axis: plt.Axes,
    points: pd.DataFrame,
    line: pd.DataFrame,
) -> None:
    """Draw the log relation and the fitted line used in the figure."""

    for group_key in GROUP_ORDER:
        group = points.loc[points["group_key"].eq(group_key)]
        axis.scatter(
            group["first_long_bout_time_s"],
            group["first_long_bout_tile_transitions"],
            s=4.2,
            facecolor="none",
            edgecolor=_GROUP_COLORS[group_key],
            linewidth=0.4,
            alpha=0.7,
            label=GROUP_LABELS[group_key],
        )
    axis.plot(
        line["first_long_bout_time_s"],
        line["first_long_bout_tile_transitions"],
        color=_AXIS_COLOR,
        linewidth=0.8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("First long bout time (s)")
    axis.set_ylabel("Steps in maze")
    axis.legend(
        loc="lower right",
        handletextpad=0.25,
        borderaxespad=0.2,
        labelspacing=0.2,
    )
    axis.grid(False)

def _draw_record_speed_axes(
    subgrid: object,
    summary: pd.DataFrame,
) -> list[plt.Axes]:
    """Draw nose speed with the shared scale and legend."""

    axes: list[plt.Axes] = []
    for index, (mode, title) in enumerate(
        (("explore", "Exploration"), ("reward", "Reward directed"))
    ):
        axis = plt.subplot(subgrid[0, index])
        axes.append(axis)
        day_one = summary.loc[
            summary["analysis_mode"].eq(mode) & summary["day"].eq(1)
        ].sort_values("month")
        day_two = summary.loc[
            summary["analysis_mode"].eq(mode) & summary["day"].eq(2)
        ].sort_values("month")
        axis.plot(
            day_one["month"],
            day_one["speed_mean_cm_s"],
            color=_AXIS_COLOR,
            linewidth=0.75,
            zorder=1,
        )
        for row in day_one.itertuples(index=False):
            color = (
                _GROUP_COLORS["naive"]
                if int(row.month) == 0
                else _GROUP_COLORS["return"]
            )
            axis.errorbar(
                row.month,
                row.speed_mean_cm_s,
                yerr=row.speed_sem_cm_s,
                color=color,
                marker="o",
                markersize=2.8,
                linewidth=0,
                elinewidth=0.55,
                capsize=1.1,
                zorder=3,
            )
        axis.errorbar(
            day_two["month"],
            day_two["speed_mean_cm_s"],
            yerr=day_two["speed_sem_cm_s"],
            color=_GROUP_COLORS["experienced_day2"],
            marker="o",
            markersize=2.8,
            linewidth=0.75,
            elinewidth=0.55,
            capsize=1.1,
            zorder=2,
        )
        axis.set_title(title, pad=1.5)
        axis.set_xticks([0, 3, 6, 9, 12])
        axis.set_xlim(-0.4, 12.4)
        axis.set_ylim(0, 14)
        axis.set_yticks(np.arange(0, 14.1, 2))
        axis.set_box_aspect(1)
        axis.set_anchor("C")
        axis.set_xlabel("Months since initial exposure")
        if index == 0:
            axis.set_ylabel("Speed (cm s$^{-1}$)")
        else:
            axis.tick_params(axis="y", labelleft=False)
            axis.legend(
                handles=(
                    Line2D(
                        [0], [0], color=_AXIS_COLOR, marker="o", markersize=2.6,
                        label="Day 1",
                    ),
                    Line2D(
                        [0], [0], color=_GROUP_COLORS["experienced_day2"],
                        marker="o", markersize=2.6, label="Day 2",
                    ),
                ),
                loc="lower right",
                handlelength=1.1,
                handletextpad=0.3,
                borderaxespad=0.2,
            )
        axis.grid(False)
    return axes

def _draw_record_coverage_examples(
    subgrid: object,
    sequence: pd.DataFrame,
    cycles: pd.DataFrame,
) -> list[plt.Axes]:
    """Draw the two-session coverage-round example."""

    axes: list[plt.Axes] = []
    conditions = (
        ("naive", "Naive example", _GROUP_COLORS["naive"]),
        ("return", "Return example", _GROUP_COLORS["return"]),
    )
    return_max_cycle = int(
        cycles.loc[cycles["condition"].eq("return"), "cycle_index"].max()
    )
    naive_two_cycle_limit = int(
        cycles.loc[
            cycles["condition"].eq("naive") & cycles["cycle_index"].eq(2),
            "cycle_end_endpoint_visit",
        ].iloc[0]
    )
    coverage_limit = max(
        float(
            cycles.loc[
                cycles["condition"].eq("naive") & cycles["cycle_index"].le(5),
                "cycle_end_endpoint_visit",
            ].max()
        ),
        float(
            cycles.loc[
                cycles["condition"].eq("return"),
                "cycle_end_endpoint_visit",
            ].max()
        ),
    )
    efficiency_max = float(
        cycles.loc[cycles["cycle_index"].le(5), "efficiency"].max()
    )
    efficiency_top = float(np.ceil((efficiency_max * 1.08) / 0.05) * 0.05)
    for row_index, (condition, label, color) in enumerate(conditions):
        condition_sequence = sequence.loc[sequence["condition"].eq(condition)]
        condition_cycles = cycles.loc[
            cycles["condition"].eq(condition)
        ].sort_values("cycle_index")
        end_node_axis = plt.subplot(subgrid[row_index, 0])
        coverage_axis = plt.subplot(subgrid[row_index, 1])
        efficiency_axis = plt.subplot(subgrid[row_index, 2])
        axes.extend((end_node_axis, coverage_axis, efficiency_axis))

        if condition == "naive":
            displayed_sequence = condition_sequence.loc[
                condition_sequence["cycle_index"].le(2)
            ]
            sequence_boundaries = condition_cycles.loc[
                condition_cycles["cycle_index"].le(2)
            ]
            displayed_coverage_cycles = 5
        else:
            displayed_sequence = condition_sequence.loc[
                condition_sequence["endpoint_visit_index"].le(
                    naive_two_cycle_limit
                )
            ]
            sequence_boundaries = condition_cycles.loc[
                condition_cycles["cycle_end_endpoint_visit"].le(
                    naive_two_cycle_limit
                )
            ]
            displayed_coverage_cycles = return_max_cycle
        displayed_coverage = condition_sequence.loc[
            condition_sequence["cycle_index"].le(displayed_coverage_cycles)
        ]
        displayed_efficiency = condition_cycles.loc[
            condition_cycles["cycle_index"].le(5)
        ]

        first_visit = displayed_sequence["is_first_visit_within_cycle"].astype(bool)
        end_node_axis.scatter(
            displayed_sequence.loc[~first_visit, "endpoint_visit_index"],
            displayed_sequence.loc[~first_visit, "end_node"],
            s=1.2,
            color=_LIGHT_GREY,
            alpha=0.55,
            linewidth=0,
        )
        end_node_axis.scatter(
            displayed_sequence.loc[first_visit, "endpoint_visit_index"],
            displayed_sequence.loc[first_visit, "end_node"],
            s=2.2,
            color=color,
            alpha=0.9,
            linewidth=0,
        )
        for _, one_cycle in displayed_coverage.groupby("cycle_index", sort=True):
            coverage_axis.plot(
                one_cycle["endpoint_visit_index"],
                one_cycle["cumulative_unique_within_cycle"],
                color=color,
                linewidth=0.55,
            )
        for axis, displayed_cycles in (
            (end_node_axis, sequence_boundaries),
            (
                coverage_axis,
                condition_cycles.loc[
                    condition_cycles["cycle_index"].le(displayed_coverage_cycles)
                ],
            ),
        ):
            for boundary in displayed_cycles["cycle_end_endpoint_visit"]:
                axis.axvline(
                    float(boundary) + 0.5,
                    color="#8A8A8A",
                    linewidth=0.45,
                    linestyle="--",
                    zorder=1,
                )
        efficiency_axis.plot(
            displayed_efficiency["cycle_index"],
            displayed_efficiency["efficiency"],
            color=color,
            linewidth=0.6,
        )
        efficiency_axis.scatter(
            displayed_efficiency["cycle_index"],
            displayed_efficiency["efficiency"],
            s=8,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
        )
        first_cycle = displayed_efficiency.loc[
            displayed_efficiency["cycle_index"].eq(1)
        ]
        efficiency_axis.scatter(
            first_cycle["cycle_index"],
            first_cycle["efficiency"],
            s=9,
            color=color,
            linewidth=0,
            zorder=4,
        )
        position = end_node_axis.get_position()
        end_node_axis.figure.text(
            0.014,
            0.5 * (position.y0 + position.y1),
            label,
            color=color,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
        end_node_axis.set_ylabel("End nodes", labelpad=1.5)
        end_node_axis.set_xlim(0.5, naive_two_cycle_limit + 0.5)
        end_node_axis.set_ylim(0.5, 64.5)
        coverage_axis.set_xlim(0.5, coverage_limit + 0.5)
        coverage_axis.set_ylim(0, 33)
        efficiency_axis.set_xlim(0.7, 5.3)
        efficiency_axis.set_ylim(0, efficiency_top)
        efficiency_axis.set_xticks(range(1, 6))
        if row_index == 0:
            end_node_axis.set_title("End node visits", pad=1.0)
            coverage_axis.set_title("Within round coverage", pad=1.0)
            efficiency_axis.set_title("Coverage round efficiency", pad=1.0)
            for axis in (end_node_axis, coverage_axis, efficiency_axis):
                axis.tick_params(axis="x", labelbottom=False)
        else:
            end_node_axis.set_xlabel("End node visit")
            coverage_axis.set_xlabel("End node visit")
            efficiency_axis.set_xlabel("Coverage round")
        for axis in (end_node_axis, coverage_axis, efficiency_axis):
            axis.grid(False)
    return axes

def _draw_record_efficiency_curve(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    x_column: str,
    x_label: str,
    legend: bool,
) -> None:
    """Draw one session-first exploration efficiency curve."""

    for group_key in GROUP_ORDER:
        group = values.loc[
            values["group_key"].eq(group_key) & values["plotted"].astype(bool)
        ].sort_values("bin_index")
        x = group[x_column].to_numpy(float)
        mean = group["mean"].to_numpy(float)
        error = group["sem"].to_numpy(float)
        axis.plot(
            x,
            mean,
            color=_GROUP_COLORS[group_key],
            linewidth=0.8,
            label=GROUP_LABELS[group_key],
        )
        axis.fill_between(
            x,
            mean - error,
            mean + error,
            color=_GROUP_COLORS[group_key],
            alpha=0.12,
            linewidth=0,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Exploration efficiency")
    axis.set_ylim(0, 0.55)
    axis.set_xlim(left=0)
    if legend:
        axis.legend(
            loc="upper right",
            handlelength=1.1,
            handletextpad=0.3,
            labelspacing=0.2,
        )
    axis.grid(False)

def _axis_group_bounds(axes: Sequence[plt.Axes]) -> tuple[float, float, float, float]:
    """Return the combined figure-coordinate bounds of an axis group."""

    boxes = [axis.get_position() for axis in axes]
    return (
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )

def _add_record_panel_letters(
    figure: plt.Figure,
    panel_axes: dict[str, list[plt.Axes]],
) -> None:
    """Place the a to j labels outside their plotting boxes."""

    for letter, axes in panel_axes.items():
        x0, _, _, y1 = _axis_group_bounds(axes)
        if letter == "h":
            label_x = 0.003
            label_y = min(0.994, y1 + 0.010)
            vertical_alignment = "bottom"
        elif letter in {"f", "g", "i", "j"}:
            label_x = max(0.006, x0 - 0.028)
            label_y = min(0.994, y1 + 0.008)
            vertical_alignment = "bottom"
        else:
            label_x = max(0.006, x0 - 0.022)
            label_y = min(0.994, y1 + 0.006)
            vertical_alignment = "bottom"
        figure.text(
            label_x,
            label_y,
            letter,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va=vertical_alignment,
        )

def compose_behavior_learning_overview(
    *,
    reward_time_curve: pd.DataFrame,
    long_bout_time_curve: pd.DataFrame,
    reward_transition_curve: pd.DataFrame,
    long_bout_transition_curve: pd.DataFrame,
    acquisition_endpoint_points: pd.DataFrame,
    acquisition_endpoint_statistics: pd.DataFrame,
    long_bout_relation_points: pd.DataFrame,
    long_bout_relation_line: pd.DataFrame,
    session_speed_summary: pd.DataFrame,
    coverage_visit_sequence: pd.DataFrame,
    coverage_cycles: pd.DataFrame,
    time_efficiency_summary: pd.DataFrame,
    transition_efficiency_summary: pd.DataFrame,
    random_seed: int = 20260805,
) -> plt.Figure:
    """Compose the publication page from explicit notebook tables.

    This function changes only presentation. Every argument is a table already
    calculated and displayed in the notebook sections above the final assembly.
    """

    with plt.rc_context(with_figure_font(_RECORD_STYLE)):
        figure = plt.figure(figsize=(7.5, 7.5), facecolor="white")
        outer = figure.add_gridspec(
            7,
            1,
            left=0.060,
            right=0.987,
            bottom=0.050,
            top=0.985,
            height_ratios=[1.42, 0.00, 1.00, 0.42, 1.30, 0.32, 1.28],
            hspace=0,
        )
        panel_axes: dict[str, list[plt.Axes]] = {}
        # These four column centers retain the physical panel geometry.
        column_grid = figure.add_gridspec(
            1, 5, left=0.025, right=0.955,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 0.045], wspace=0.012,
        )
        column_centers = [
            0.5 * (column_grid[0, column].get_position(figure).x0
                   + column_grid[0, column].get_position(figure).x1)
            for column in range(4)
        ]

        curve_row = outer[2].subgridspec(
            2,
            4,
            height_ratios=[0.17, 1.0],
            hspace=0.11,
            wspace=0.34,
        )
        legend_axis = figure.add_subplot(curve_row[0, :])
        legend_axis.axis("off")
        legend_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=_MONTH_COLORS[key],
                    linewidth=0.9,
                    linestyle="--" if key == "experienced_day2" else "-",
                    label=MONTH_LABELS[key],
                )
                for key in MONTH_ORDER
            ],
            loc="center",
            bbox_to_anchor=(0.5, 1.30),
            ncol=len(MONTH_ORDER),
            columnspacing=0.9,
            handlelength=1.35,
            handletextpad=0.3,
            borderaxespad=0,
        )
        b_axis = figure.add_subplot(curve_row[1, 0])
        c_axis = figure.add_subplot(curve_row[1, 1])
        d_axis = figure.add_subplot(curve_row[1, 2])
        e_axis = figure.add_subplot(curve_row[1, 3])
        _draw_record_cumulative_curve(
            b_axis,
            reward_time_curve,
            x_column="time_minutes",
            x_label="Time in maze (min)",
            y_label="Cumulative rewards",
        )
        _draw_record_cumulative_curve(
            c_axis,
            long_bout_time_curve,
            x_column="time_minutes",
            x_label="Time in maze (min)",
            y_label="Cumulative long bouts",
        )
        _draw_record_cumulative_curve(
            d_axis,
            reward_transition_curve,
            x_column="maze_tile_transitions",
            x_label="Steps in maze",
            y_label="Cumulative rewards",
        )
        _draw_record_cumulative_curve(
            e_axis,
            long_bout_transition_curve,
            x_column="maze_tile_transitions",
            x_label="Steps in maze",
            y_label="Cumulative long bouts",
        )
        panel_axes.update(
            {"a": [b_axis], "b": [c_axis], "c": [d_axis], "d": [e_axis]}
        )

        comparison_row = outer[4].subgridspec(
            1,
            3,
            width_ratios=[2.15, 0.92, 1.72],
            wspace=0.32,
        )
        endpoint_grid = comparison_row[0, 0].subgridspec(
            2, 3, hspace=0.48, wspace=0.30
        )
        acquisition_axes = _draw_record_acquisition_grid(
            endpoint_grid,
            acquisition_endpoint_points,
            acquisition_endpoint_statistics,
            random_seed,
        )
        relation_axis = figure.add_subplot(comparison_row[0, 1])
        _draw_record_long_bout_relation(
            relation_axis, long_bout_relation_points, long_bout_relation_line
        )
        relation_axis.set_box_aspect(1)
        relation_axis.set_anchor("C")
        speed_grid = comparison_row[0, 2].subgridspec(1, 2, wspace=0.16)
        speed_axes = _draw_record_speed_axes(speed_grid, session_speed_summary)
        panel_axes.update(
            {"e": acquisition_axes, "f": [relation_axis], "g": speed_axes}
        )

        efficiency_row = outer[6].subgridspec(
            1,
            3,
            width_ratios=[2.55, 1.0, 1.0],
            wspace=0.30,
        )
        example_grid = efficiency_row[0, 0].subgridspec(
            2, 3, hspace=0.28, wspace=0.38
        )
        example_axes = _draw_record_coverage_examples(
            example_grid, coverage_visit_sequence, coverage_cycles
        )
        time_axis = figure.add_subplot(efficiency_row[0, 1])
        transition_axis = figure.add_subplot(efficiency_row[0, 2])
        _draw_record_efficiency_curve(
            time_axis,
            time_efficiency_summary,
            x_column="completion_time_min",
            x_label="Time in maze (min)",
            legend=True,
        )
        _draw_record_efficiency_curve(
            transition_axis,
            transition_efficiency_summary,
            x_column="completion_tile_transitions",
            x_label="Steps in maze",
            legend=False,
        )
        transition_axis.tick_params(axis="y", labelleft=False)
        transition_axis.set_ylabel("")
        panel_axes.update(
            {"h": example_axes, "i": [time_axis], "j": [transition_axis]}
        )

        curve_width = b_axis.get_position().width
        for axis, center in zip(
            (b_axis, c_axis, d_axis, e_axis), column_centers, strict=True
        ):
            box = axis.get_position()
            axis.set_position((center - curve_width / 2, box.y0, curve_width, box.height))

        first_two_left = column_centers[0] - curve_width / 2
        first_two_right = column_centers[1] + curve_width / 2

        def align_group_to_first_two(axes: Sequence[plt.Axes]) -> None:
            """Scale an axis group to the width of the first two columns."""

            boxes = [axis.get_position() for axis in axes]
            source_left = min(box.x0 for box in boxes)
            source_right = max(box.x1 for box in boxes)
            source_width = source_right - source_left
            target_width = first_two_right - first_two_left
            for axis, box in zip(axes, boxes, strict=True):
                axis.set_position(
                    (
                        first_two_left
                        + (box.x0 - source_left) / source_width * target_width,
                        box.y0,
                        box.width / source_width * target_width,
                        box.height,
                    )
                )

        align_group_to_first_two(acquisition_axes)
        align_group_to_first_two(example_axes)
        for index, axis in enumerate(example_axes):
            reference = acquisition_axes[index % 3]
            reference_box = reference.get_position()
            box = axis.get_position()
            axis.set_position(
                (reference_box.x0, box.y0, reference_box.width, box.height)
            )
        for axis in (acquisition_axes[0], acquisition_axes[3]):
            axis.yaxis.set_label_coords(-0.269, 0.5)
            axis.tick_params(axis="y", pad=1.0)

        relation_box = relation_axis.get_position()
        relation_side = min(curve_width, relation_box.height)
        relation_axis.set_position(
            (
                column_centers[2] - relation_side / 2,
                0.5 * (relation_box.y0 + relation_box.y1) - relation_side / 2,
                relation_side,
                relation_side,
            )
        )
        relation_axis.set_box_aspect(1)
        relation_axis.set_anchor("C")

        speed_boxes = [axis.get_position() for axis in speed_axes]
        speed_center_y = 0.5 * (
            min(box.y0 for box in speed_boxes) + max(box.y1 for box in speed_boxes)
        )
        speed_gap = 0.012
        speed_width = (curve_width - speed_gap) / 2
        speed_left = column_centers[3] - curve_width / 2
        for index, axis in enumerate(speed_axes):
            axis.set_position(
                (
                    speed_left + index * (speed_width + speed_gap),
                    speed_center_y - relation_side / 2,
                    speed_width,
                    relation_side,
                )
            )
            axis.set_box_aspect(None)
            axis.set_xlabel("")
        figure.text(
            column_centers[3],
            min(axis.get_position().y0 for axis in speed_axes) - 0.024,
            "Months since initial exposure",
            fontsize=plt.rcParams["axes.labelsize"],
            ha="center",
            va="top",
        )

        for text in figure.texts:
            if text.get_text() in {"Naive example", "Return example"}:
                text.set_x(0.008)
        for axis in (example_axes[0], example_axes[3]):
            axis.yaxis.set_label_coords(-0.22, 0.5)

        for axis, center in (
            (time_axis, column_centers[2] + 0.016),
            (transition_axis, column_centers[3] + 0.016),
        ):
            box = axis.get_position()
            axis.set_position((center - curve_width / 2, box.y0, curve_width, box.height))

        for axis in figure.axes:
            axis.grid(False)
        _add_record_panel_letters(figure, panel_axes)
        # Retain physical artist positions while removing the unused top/right canvas.
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        visible = [axis.get_tightbbox(renderer) for axis in figure.axes if axis.axison]
        visible.extend(axis.get_legend().get_window_extent(renderer)
                       for axis in figure.axes if not axis.axison and axis.get_legend() is not None)
        visible.extend(text.get_window_extent(renderer) for text in figure.texts if text.get_visible())
        width, height = figure.get_size_inches()
        new_width = max(box.x1 for box in visible if box is not None) / figure.dpi + 4 / 72
        new_height = max(box.y1 for box in visible if box is not None) / figure.dpi + 4 / 72
        positions = {axis: axis.get_position().frozen() for axis in figure.axes}
        text_positions = {text: text.get_position() for text in figure.texts}
        figure.set_size_inches(new_width, new_height, forward=True)
        for axis, box in positions.items():
            axis.set_position((box.x0 * width / new_width, box.y0 * height / new_height,
                               box.width * width / new_width, box.height * height / new_height))
        for text, (x, y) in text_positions.items():
            text.set_position((x * width / new_width, y * height / new_height))
    return figure

def save_behavior_learning_overview(
    figure: plt.Figure,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save the complete publication page as SVG, PDF, and 600 dpi PNG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Extended_Data_Figure_1"
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    with plt.rc_context(with_figure_font(_RECORD_STYLE)):
        savefig_reproducibly(figure, svg_path)
        savefig_reproducibly(figure, pdf_path)
        savefig_reproducibly(figure, png_path, dpi=600)
    return svg_path, pdf_path, png_path
