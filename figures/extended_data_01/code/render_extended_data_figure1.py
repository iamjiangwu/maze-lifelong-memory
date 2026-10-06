"""Render Extended Data Figure 1 directly from maintained source tables.

The supplied slide fixes panel identity and evidence order, but not its
distorted picture-box transforms.  This renderer redraws every panel in one
matplotlib canvas, aligns intended row members, removes caption-level prose,
and never imports a previously rendered panel SVG or raster.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd



SPEED_WINDOW_ID = "first_30_min"
SPEED_SUMMARY_VARIANT = "exact_current_mode_rule"
TIME_CURVE_CAPTION_NOTE = (
    "Mean ± SEM across session rows. The common 80%-session time support "
    "window and initial group counts belong in the caption/source audit."
)
TILE_CURVE_CAPTION_NOTE = (
    "Mean ± SEM across all 153 sessions over the fully observed common "
    "0-5,200-transition range; group counts are 23, 17, 17, 12, 10 and 74."
)
EFFICIENCY_CAPTION_NOTE = (
    "Session-first mean ± SEM in fixed completion-coordinate bins; only bins "
    "with at least five sessions are drawn. The plotted y-axis wording is "
    "`Exploration efficiency`; the 32/N definition and dynamic support remain "
    "in the exported tables rather than below the axes."
)

# The supported launcher binds this mapping to included publication inputs
# before rendering; it supplies no raw trajectory paths.
SOURCE_FILES: dict[str, tuple[Path, str]]
PANEL_SCOPE = {
    "b-c": ("ED1", "maintained current153 primary evidence"),
    "d-e": ("ED1", "maintained current153 primary evidence"),
    "f-rows": ("ED1", "maintained current153 primary evidence"),
    "f-stats": ("ED1", "caption-only statistics"),
    "g-rows": ("ED2 retained on supplied slide", "maintained diagnostic"),
    "g-stats": ("ED2 retained on supplied slide", "caption-only statistics"),
    "h-rows": ("ED2 retained on supplied slide", "maintained locomotor control"),
    "h-reference": ("ED2 retained on supplied slide", "summary parity target"),
    "i-naive": ("ED2 retained on supplied slide", "raw selected trajectory"),
    "i-return": ("ED2 retained on supplied slide", "raw selected trajectory"),
    "j": ("ED2 retained on supplied slide", "candidate axis option; not promoted primary"),
    "k": ("ED2 retained on supplied slide", "candidate axis option; not promoted primary"),
}

MONTH_ORDER = ("0mo", "3mo", "6mo", "9mo", "12mo", "experienced_day2")
MONTH_COLORS = {
    "0mo": "#78ADD2",
    "3mo": "#F2B3AE",
    "6mo": "#E77D72",
    "9mo": "#C84B44",
    "12mo": "#8E2C2A",
    "experienced_day2": "#5B8C5A",
}
MONTH_LABELS = {
    "0mo": "Naive day",
    "3mo": "3 mo return",
    "6mo": "6 mo return",
    "9mo": "9 mo return",
    "12mo": "12 mo return",
    "experienced_day2": "Second day",
}
GROUP_ORDER = ("naive", "return", "experienced_day2")
GROUP_COLORS = {
    "naive": "#78ADD2",
    "return": "#C84B44",
    "experienced_day2": "#5B8C5A",
}
GROUP_LABELS = {
    "naive": "Naive day",
    "return": "Return day",
    "experienced_day2": "Second day",
}
AXIS = "#303030"
GRID = "#E5E5E5"
GREY = "#8A8A8A"
F_BRACKET_STEP_FRACTION = 0.200
F_STAR_OFFSET_FRACTION = 0.006
F_STAR_FONTSIZE_PT = 4.5


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
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
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sem(values: Iterable[float]) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    return (
        float(np.std(array, ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else 0.0
    )










def accumulation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["display_coverage_ok"].astype(bool)].copy()
    return (
        data.groupby(
            ["group_key", "event_type", "time_minutes"],
            observed=True,
            as_index=False,
        )
        .agg(
            mean=("cum_count", "mean"),
            sem=("cum_count", sem),
            n_sessions=("session", "nunique"),
        )
        .sort_values(["event_type", "group_key", "time_minutes"])
    )


def draw_month_curve(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    event_type: str,
    x: str,
    mean: str,
    error: str,
    xlabel: str,
    ylabel: str,
    legend: bool,
) -> None:
    subset = frame.loc[frame["event_type"].eq(event_type)].copy()
    for key in MONTH_ORDER:
        group = subset.loc[subset["group_key"].eq(key)].sort_values(x)
        if group.empty:
            continue
        xv = group[x].to_numpy(float)
        yv = group[mean].to_numpy(float)
        ev = group[error].to_numpy(float)
        ax.plot(
            xv,
            yv,
            color=MONTH_COLORS[key],
            linewidth=0.9,
            linestyle="--" if key == "experienced_day2" else "-",
            label=MONTH_LABELS[key],
        )
        ax.fill_between(
            xv,
            yv - ev,
            yv + ev,
            color=MONTH_COLORS[key],
            alpha=0.10,
            linewidth=0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(False)


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return "n.s."
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "n.s."


def draw_endpoint_grid(
    subgrid: Any,
    rows: pd.DataFrame,
    stats: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[list[plt.Axes], list[int]]:
    axes: list[plt.Axes] = []
    annotation_counts: list[int] = []
    endpoints = ("first_reward", "tenth_reward", "first_long_bout")
    endpoint_labels = ("First reward", "Tenth reward", "First long bout")
    axis_specs = (
        ("maze_time_s", "Time in maze (s)"),
        ("maze_tiles", "Maze tile transitions"),
    )
    for row_index, (axis_key, ylabel) in enumerate(axis_specs):
        # The three columns of a row share one y range because only the first
        # column has tick labels; bracket spacing follows the largest point in the row.
        row_values = pd.to_numeric(
            rows.loc[
                rows["endpoint_key"].isin(endpoints)
                & rows["group_key"].isin(GROUP_ORDER),
                axis_key,
            ],
            errors="coerce",
        ).dropna()
        span = max(float(row_values.max()), 1.0)
        row_axes: list[plt.Axes] = []
        row_top = 0.0
        for col_index, (endpoint, endpoint_label) in enumerate(
            zip(endpoints, endpoint_labels, strict=True)
        ):
            ax = plt.subplot(subgrid[row_index, col_index])
            axes.append(ax)
            panel = rows.loc[rows["endpoint_key"].eq(endpoint)].copy()
            panel_stats = stats.loc[
                stats["endpoint_key"].eq(endpoint)
                & stats["axis_key"].eq(axis_key)
            ].copy()
            all_values: list[float] = []
            for xpos, group_key in enumerate(GROUP_ORDER):
                values = pd.to_numeric(
                    panel.loc[panel["group_key"].eq(group_key), axis_key],
                    errors="coerce",
                ).dropna().to_numpy(float)
                all_values.extend(values.tolist())
                jitter = rng.uniform(-0.13, 0.13, values.size)
                ax.scatter(
                    xpos + jitter,
                    values,
                    s=3.2,
                    facecolor="none",
                    edgecolor=GROUP_COLORS[group_key],
                    linewidth=0.35,
                    alpha=0.55,
                    zorder=2,
                )
                mean_value = float(np.mean(values))
                error = sem(values)
                ax.bar(
                    xpos,
                    mean_value,
                    width=0.55,
                    color=GROUP_COLORS[group_key],
                    alpha=0.55,
                    zorder=1,
                )
                ax.errorbar(
                    xpos,
                    mean_value,
                    yerr=error,
                    color=AXIS,
                    linewidth=0,
                    elinewidth=0.55,
                    capsize=1.2,
                    zorder=3,
                )
            ax.set_xlim(-0.55, 2.55)
            raw_max = max(all_values)
            base = raw_max + 0.055 * span
            step = F_BRACKET_STEP_FRACTION * span
            tick_height = 0.022 * span
            group_to_x = {key: index for index, key in enumerate(GROUP_ORDER)}
            annotations = 0
            for _, contrast in panel_stats.iterrows():
                q_value = float(contrast["p_bh18"])
                stars = significance_stars(q_value)
                if stars == "n.s.":
                    continue
                x1 = group_to_x[str(contrast["group_a"])]
                x2 = group_to_x[str(contrast["group_b"])]
                y = base + annotations * step
                ax.plot(
                    [x1, x1, x2, x2],
                    [y, y + tick_height, y + tick_height, y],
                    color=AXIS,
                    linewidth=0.48,
                    clip_on=False,
                )
                ax.text(
                    (x1 + x2) / 2,
                    y + tick_height + F_STAR_OFFSET_FRACTION * span,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=F_STAR_FONTSIZE_PT,
                )
                annotations += 1
            annotation_counts.append(annotations)
            annotation_top = (
                base + max(annotations - 1, 0) * step + tick_height + 0.22 * span
            )
            row_top = max(row_top, raw_max * 1.08, annotation_top)
            row_axes.append(ax)
            ax.grid(False)
            if row_index == 0:
                ax.set_title(endpoint_label, pad=4.0)
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xticks(
                    range(3),
                    ["Naive", "Return", "Second"],
                    rotation=28,
                    ha="right",
                )
            if col_index == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", labelleft=False)
        for row_ax in row_axes:
            row_ax.set_ylim(0, row_top)
    return axes, annotation_counts


def draw_tile_time(
    ax: plt.Axes,
    rows: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:
    model = rows.loc[rows["included_in_loglog_model"].astype(bool)].copy()
    for group_key in GROUP_ORDER:
        subset = model.loc[model["group_key"].eq(group_key)]
        ax.scatter(
            subset["lb0_time_in_maze_s"],
            subset["nose_tiles_to_first_long_bout"],
            s=4.2,
            facecolor="none",
            edgecolor=GROUP_COLORS[group_key],
            linewidth=0.4,
            alpha=0.7,
            label=GROUP_LABELS[group_key],
        )
    stat = stats.iloc[0]
    x = np.geomspace(
        float(model["lb0_time_in_maze_s"].min()),
        float(model["lb0_time_in_maze_s"].max()),
        200,
    )
    y = np.exp(float(stat["intercept_log_scale"])) * x ** float(
        stat["slope_loglog"]
    )
    ax.plot(x, y, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("First long bout time (s)")
    ax.set_ylabel("Maze tile transitions")
    ax.grid(False)
    ax.legend(
        loc="lower right",
        handletextpad=0.25,
        borderaxespad=0.2,
        labelspacing=0.2,
    )


def speed_summary(rows: pd.DataFrame) -> pd.DataFrame:
    data = rows.loc[
        rows["analysis_mode"].isin(["explore", "reward"])
        & rows["day"].isin(["day1", "day2"])
    ].copy()
    if "window_id" in data.columns:
        data = data.loc[data["window_id"].eq(SPEED_WINDOW_ID)].copy()
    data["mode_time_distance_rate_cm_s"] = pd.to_numeric(
        data["mode_time_distance_rate_cm_s"],
        errors="coerce",
    )
    data = data.loc[np.isfinite(data["mode_time_distance_rate_cm_s"])].copy()
    data["month"] = pd.to_numeric(
        data["month_label"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="raise",
    ).astype(int)
    return (
        data.groupby(["analysis_mode", "day", "month"], as_index=False)
        .agg(
            speed_mean_cm_s=("mode_time_distance_rate_cm_s", "mean"),
            speed_sem_cm_s=("mode_time_distance_rate_cm_s", sem),
            n_sessions=("session_id", "nunique"),
            n_animals=("animal_id", "nunique"),
        )
        .sort_values(["analysis_mode", "day", "month"])
    )


def draw_speed_axes(
    subgrid: Any,
    summary: pd.DataFrame,
) -> list[plt.Axes]:
    axes: list[plt.Axes] = []
    upper_bound = float(
        np.nanmax(summary["speed_mean_cm_s"] + summary["speed_sem_cm_s"])
    )
    shared_ymax = max(2.0, 2.0 * np.ceil(upper_bound / 2.0))
    shared_yticks = np.arange(0.0, shared_ymax + 0.1, 2.0)
    for index, (mode, title) in enumerate(
        (("explore", "Exploration"), ("reward", "Reward directed"))
    ):
        ax = plt.subplot(subgrid[0, index])
        axes.append(ax)
        day1 = summary.loc[
            summary["analysis_mode"].eq(mode) & summary["day"].eq("day1")
        ].sort_values("month")
        day2 = summary.loc[
            summary["analysis_mode"].eq(mode) & summary["day"].eq("day2")
        ].sort_values("month")
        ax.plot(
            day1["month"],
            day1["speed_mean_cm_s"],
            color=AXIS,
            linewidth=0.75,
            zorder=1,
        )
        for row in day1.itertuples(index=False):
            color = GROUP_COLORS["naive"] if int(row.month) == 0 else GROUP_COLORS["return"]
            ax.errorbar(
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
        ax.errorbar(
            day2["month"],
            day2["speed_mean_cm_s"],
            yerr=day2["speed_sem_cm_s"],
            color=GROUP_COLORS["experienced_day2"],
            marker="o",
            markersize=2.8,
            linewidth=0.75,
            elinewidth=0.55,
            capsize=1.1,
            zorder=2,
        )
        ax.set_title(title, pad=1.5)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_xlim(-0.4, 12.4)
        ax.set_ylim(0, shared_ymax)
        ax.set_yticks(shared_yticks)
        ax.set_box_aspect(1)
        ax.set_anchor("C")
        ax.set_xlabel("Months since initial exposure")
        ax.grid(False)
        if index == 0:
            ax.set_ylabel("Speed (cm s$^{-1}$)")
        else:
            ax.tick_params(axis="y", labelleft=False)
            handles = [
                Line2D([0], [0], color=AXIS, marker="o", markersize=2.6, label="Day 1"),
                Line2D(
                    [0],
                    [0],
                    color=GROUP_COLORS["experienced_day2"],
                    marker="o",
                    markersize=2.6,
                    label="Day 2",
                ),
            ]
            ax.legend(
                handles=handles,
                loc="lower right",
                handlelength=1.1,
                handletextpad=0.3,
                borderaxespad=0.2,
            )
    return axes


def draw_efficiency_examples(
    subgrid: Any,
    sequence: pd.DataFrame,
    cycles: pd.DataFrame,
) -> tuple[list[plt.Axes], dict[str, Any]]:
    axes: list[plt.Axes] = []
    conditions = (
        ("redundant_naive", "Naive example", GROUP_COLORS["naive"]),
        ("efficient_return", "Return example", GROUP_COLORS["return"]),
    )
    return_max_cycle = int(
        cycles.loc[
            cycles["condition"].eq("efficient_return"),
            "cycle_index",
        ].max()
    )
    naive_two_cycle_visit_limit = int(
        cycles.loc[
            cycles["condition"].eq("redundant_naive")
            & cycles["cycle_index"].eq(2),
            "cycle_end_endpoint_visit_index",
        ].iloc[0]
    )
    coverage_xmax = max(
        float(
            cycles.loc[
                cycles["condition"].eq("redundant_naive")
                & cycles["cycle_index"].le(5),
                "cycle_end_endpoint_visit_index",
            ].max()
        ),
        float(
            cycles.loc[
                cycles["condition"].eq("efficient_return"),
                "cycle_end_endpoint_visit_index",
            ].max()
        ),
    )
    first_five_efficiency = cycles.loc[cycles["cycle_index"].le(5)]
    efficiency_max = float(
        first_five_efficiency["efficiency_32_over_steps"].max()
    )
    efficiency_ylim_top = float(
        np.ceil((efficiency_max * 1.08) / 0.05) * 0.05
    )
    display_contract = {
        "naive_sequence_cycles": 2,
        "return_sequence_max_endnode_visit": naive_two_cycle_visit_limit,
        "naive_coverage_cycles": 5,
        "return_coverage_cycles": return_max_cycle,
        "both_efficiency_cycles": 5,
    }

    boundary_counts: list[int] = []
    x_limits_by_column: dict[int, list[tuple[float, float]]] = {
        0: [],
        1: [],
        2: [],
    }
    for row_index, (condition, label, color) in enumerate(conditions):
        seq = sequence.loc[sequence["condition"].eq(condition)].copy()
        cyc = (
            cycles.loc[cycles["condition"].eq(condition)]
            .sort_values("cycle_index")
            .copy()
        )
        ax1 = plt.subplot(subgrid[row_index, 0])
        ax2 = plt.subplot(subgrid[row_index, 1])
        ax3 = plt.subplot(subgrid[row_index, 2])
        axes.extend([ax1, ax2, ax3])

        if condition == "redundant_naive":
            sequence_display = seq.loc[seq["cycle_index"].le(2)].copy()
            sequence_boundary_cycles = cyc.loc[cyc["cycle_index"].le(2)]
            coverage_limit = 5
        else:
            sequence_display = seq.loc[
                seq["endpoint_visit_index"].le(naive_two_cycle_visit_limit)
            ].copy()
            sequence_boundary_cycles = cyc.loc[
                cyc["cycle_end_endpoint_visit_index"].le(
                    naive_two_cycle_visit_limit
                )
            ]
            coverage_limit = return_max_cycle
        coverage_display = seq.loc[seq["cycle_index"].le(coverage_limit)].copy()
        efficiency_display = cyc.loc[cyc["cycle_index"].le(5)].copy()

        first = sequence_display["is_first_visit_within_cycle"].astype(bool)
        ax1.scatter(
            sequence_display.loc[~first, "endpoint_visit_index"],
            sequence_display.loc[~first, "end_node_display_id_1_to_64"],
            s=1.2,
            color="#C9C9C9",
            alpha=0.55,
            linewidth=0,
        )
        ax1.scatter(
            sequence_display.loc[first, "endpoint_visit_index"],
            sequence_display.loc[first, "end_node_display_id_1_to_64"],
            s=2.2,
            color=color,
            alpha=0.9,
            linewidth=0,
        )
        for _, cycle in coverage_display.groupby("cycle_index", sort=True):
            ax2.plot(
                cycle["endpoint_visit_index"],
                cycle["cumulative_unique_within_cycle"],
                color=color,
                linewidth=0.55,
            )
        for axis, displayed_cycles in (
            (ax1, sequence_boundary_cycles),
            (ax2, cyc.loc[cyc["cycle_index"].le(coverage_limit)]),
        ):
            for boundary in displayed_cycles["cycle_end_endpoint_visit_index"]:
                axis.axvline(
                    float(boundary) + 0.5,
                    color=GREY,
                    linewidth=0.45,
                    linestyle="--",
                    zorder=1,
                )
            boundary_counts.append(len(displayed_cycles))
        ax3.plot(
            efficiency_display["cycle_index"],
            efficiency_display["efficiency_32_over_steps"],
            color=color,
            linewidth=0.6,
        )
        ax3.scatter(
            efficiency_display["cycle_index"],
            efficiency_display["efficiency_32_over_steps"],
            s=8,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
        )
        first_cycle = efficiency_display.loc[efficiency_display["cycle_index"].eq(1)]
        ax3.scatter(
            first_cycle["cycle_index"],
            first_cycle["efficiency_32_over_steps"],
            s=9,
            color=color,
            linewidth=0,
            zorder=4,
        )

        ax1_position = ax1.get_position()
        ax1.figure.text(
            0.014,
            0.5 * (ax1_position.y0 + ax1_position.y1),
            label,
            color=color,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
        ax1.set_ylabel("End nodes", labelpad=1.5)
        ax2.set_ylabel("")
        ax3.set_ylabel("")
        ax1.set_ylim(0.5, 64.5)
        ax2.set_ylim(0, 33)
        ax3.set_ylim(0, efficiency_ylim_top)
        ax1.set_xlim(0.5, naive_two_cycle_visit_limit + 0.5)
        ax2.set_xlim(0.5, coverage_xmax + 0.5)
        ax3.set_xlim(0.7, 5.3)
        ax3.set_xticks(range(1, 6))
        for axis in (ax1, ax2, ax3):
            axis.grid(False)
        x_limits_by_column[0].append(ax1.get_xlim())
        x_limits_by_column[1].append(ax2.get_xlim())
        x_limits_by_column[2].append(ax3.get_xlim())
        if row_index == 0:
            ax1.set_title("End node visits", pad=1.0)
            ax2.set_title("Within round coverage", pad=1.0)
            ax3.set_title("Coverage round efficiency", pad=1.0)
            for axis in (ax1, ax2, ax3):
                axis.tick_params(axis="x", labelbottom=False)
        else:
            ax1.set_xlabel("End node visit")
            ax2.set_xlabel("End node visit")
            ax3.set_xlabel("Coverage round")
    audit = {
        "display_contract": display_contract,
        "shared_x_limits_by_column": x_limits_by_column,
        "boundary_counts": boundary_counts,
        "efficiency_ylim": [0.0, efficiency_ylim_top],
        "efficiency_display_max": efficiency_max,
        "condition_labels_vertical": True,
        "condition_label_x_fraction": 0.014,
        "return_sequence_rows_after_truncation": int(
            sequence.loc[
                sequence["condition"].eq("efficient_return")
                & sequence["endpoint_visit_index"].le(
                    naive_two_cycle_visit_limit
                )
            ].shape[0]
        ),
    }
    return axes, audit


def efficiency_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["group_key", "group_label", "bin_index", "axis_bin_center"],
            observed=True,
            as_index=False,
        )
        .agg(
            efficiency_mean=("session_efficiency", "mean"),
            efficiency_sem=("session_efficiency", sem),
            n_sessions=("nickname", "nunique"),
            n_animals=("animal_id", "nunique"),
        )
        .sort_values(["group_key", "bin_index"])
    )


def draw_efficiency_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    xlabel: str,
    legend: bool,
) -> None:
    for group_key in GROUP_ORDER:
        subset = summary.loc[
            summary["group_key"].eq(group_key) & summary["n_sessions"].ge(5)
        ].sort_values("axis_bin_center")
        x = subset["axis_bin_center"].to_numpy(float)
        y = subset["efficiency_mean"].to_numpy(float)
        error = subset["efficiency_sem"].to_numpy(float)
        ax.plot(
            x,
            y,
            color=GROUP_COLORS[group_key],
            linewidth=0.8,
            label=GROUP_LABELS[group_key],
        )
        ax.fill_between(
            x,
            y - error,
            y + error,
            color=GROUP_COLORS[group_key],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exploration efficiency")
    ax.set_ylim(0, 0.55)
    ax.set_xlim(left=0)
    ax.grid(False)
    if legend:
        ax.legend(
            loc="upper right",
            handlelength=1.1,
            handletextpad=0.3,
            labelspacing=0.2,
        )


def group_bbox(axes: list[plt.Axes]) -> tuple[float, float, float, float]:
    boxes = [axis.get_position() for axis in axes]
    return (
        min(box.x0 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.x1 for box in boxes),
        max(box.y1 for box in boxes),
    )


def add_panel_letters(
    fig: plt.Figure,
    panel_axes: dict[str, list[plt.Axes]],
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, tuple[float, float]],
]:
    fig.canvas.draw()
    bboxes: dict[str, tuple[float, float, float, float]] = {}
    label_positions: dict[str, tuple[float, float]] = {}
    for letter, axes in panel_axes.items():
        x0, y0, x1, y1 = group_bbox(axes)
        bboxes[letter] = (x0, y0, x1, y1)
        if letter == "a":
            label_x = max(0.006, x0 - 0.022)
            label_y = y1 - 0.004
            vertical_alignment = "top"
        elif letter == "i":
            label_x = 0.003
            label_y = min(0.994, y1 + 0.010)
            vertical_alignment = "bottom"
        elif letter in {"g", "h", "j", "k"}:
            # Quantitative-panel letters share one outside-top-left grammar.
            # Earlier rounds left g/h at different heights and put j/k inside
            # their axes, which made the lower rows look misregistered.
            label_x = max(0.006, x0 - 0.028)
            label_y = min(0.994, y1 + 0.008)
            vertical_alignment = "bottom"
        else:
            label_x = max(0.006, x0 - 0.022)
            label_y = min(0.994, y1 + 0.006)
            vertical_alignment = "bottom"
        label_positions[letter] = (label_x, label_y)
        fig.text(
            label_x,
            label_y,
            letter,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va=vertical_alignment,
        )
    return bboxes, label_positions


def build_figure(
    data: dict[str, pd.DataFrame],
) -> tuple[plt.Figure, dict[str, tuple[float, float, float, float]], dict[str, Any]]:
    rng = np.random.default_rng(20260805)
    fig = plt.figure(figsize=(7.5, 7.5), facecolor="white")
    outer = fig.add_gridspec(
        7,
        1,
        left=0.060,
        right=0.987,
        bottom=0.050,
        top=0.985,
        height_ratios=[1.42, 0.00, 1.00, 0.42, 1.30, 0.32, 1.28],
        hspace=0,
    )
    outer_a = fig.add_gridspec(
        7,
        1,
        left=0.025,
        right=0.955,
        bottom=0.050,
        top=0.985,
        height_ratios=[1.42, 0.00, 1.00, 0.42, 1.30, 0.32, 1.28],
        hspace=0,
    )
    panel_axes: dict[str, list[plt.Axes]] = {}

    row_a = outer_a[0].subgridspec(
        1,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.045],
        wspace=0.012,
    )
    trajectory_axes: list[plt.Axes] = []
    # Historical panel a is not part of the current exported ED1. Its empty
    # axes preserve the inherited column skeleton until the adapter removes it.
    shared_max = None
    for column, (nickname, title) in enumerate(
        (
            ("434RRday1", "Day 1"),
            ("434RRday2", "Day 2"),
            ("434-3day1", "Day 91"),
            ("434-6day1", "Day 181"),
        )
    ):
        axis = fig.add_subplot(row_a[0, column])
        trajectory_axes.append(axis)
        axis.set_xlim(-1.85, 15.95)
        axis.set_ylim(15.0, -1.0)
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_title(title, fontweight="normal", pad=1.0)
    color_axis = fig.add_subplot(row_a[0, 4])
    color_axis.axis("off")
    color_axis.set_ylabel("Time in maze (min)", fontsize=5.3, labelpad=1.0)
    panel_axes["a"] = [*trajectory_axes, color_axis]

    row_two = outer[2].subgridspec(
        2,
        4,
        height_ratios=[0.17, 1.0],
        hspace=0.11,
        wspace=0.34,
    )
    curve_legend_axis = fig.add_subplot(row_two[0, :])
    curve_legend_axis.axis("off")
    curve_legend_handles = [
        Line2D(
            [0],
            [0],
            color=MONTH_COLORS[key],
            linewidth=0.9,
            linestyle="--" if key == "experienced_day2" else "-",
            label=MONTH_LABELS[key],
        )
        for key in MONTH_ORDER
    ]
    curve_legend = curve_legend_axis.legend(
        handles=curve_legend_handles,
        loc="center",
        # Keep the shared condition row above the b-e panel-letter lane.  The
        # previous 0.78 anchor let its marker handles overlap the d label even
        # though the legend text itself appeared clear.
        bbox_to_anchor=(0.5, 1.30),
        ncol=len(MONTH_ORDER),
        columnspacing=0.9,
        handlelength=1.35,
        handletextpad=0.3,
        borderaxespad=0,
    )
    accumulation = accumulation_summary(data["accumulation"])
    b_axis = fig.add_subplot(row_two[1, 0])
    c_axis = fig.add_subplot(row_two[1, 1])
    d_axis = fig.add_subplot(row_two[1, 2])
    e_axis = fig.add_subplot(row_two[1, 3])
    draw_month_curve(
        b_axis,
        accumulation,
        event_type="reward",
        x="time_minutes",
        mean="mean",
        error="sem",
        xlabel="Time in maze (min)",
        ylabel="Cumulative rewards",
        legend=False,
    )
    draw_month_curve(
        c_axis,
        accumulation,
        event_type="long_path_reward",
        x="time_minutes",
        mean="mean",
        error="sem",
        xlabel="Time in maze (min)",
        ylabel="Cumulative long bouts",
        legend=False,
    )
    tile_curves = data["tile_curves"].rename(
        columns={
            "month_group_key": "group_key",
            "cumulative_mean": "mean",
            "cumulative_sem": "sem",
        }
    )
    tile_curves["group_key"] = tile_curves["group_key"].replace(
        {
            "naive_0mo": "0mo",
            "return_3mo": "3mo",
            "return_6mo": "6mo",
            "return_9mo": "9mo",
            "return_12mo": "12mo",
        }
    )
    draw_month_curve(
        d_axis,
        tile_curves,
        event_type="reward",
        x="axis_value_tiles",
        mean="mean",
        error="sem",
        xlabel="Maze tile transitions",
        ylabel="Cumulative rewards",
        legend=False,
    )
    draw_month_curve(
        e_axis,
        tile_curves,
        event_type="long_path_reward",
        x="axis_value_tiles",
        mean="mean",
        error="sem",
        xlabel="Maze tile transitions",
        ylabel="Cumulative long bouts",
        legend=False,
    )
    panel_axes.update(
        {
            "b": [b_axis],
            "c": [c_axis],
            "d": [d_axis],
            "e": [e_axis],
        }
    )

    row_three = outer[4].subgridspec(
        1,
        3,
        width_ratios=[2.15, 0.92, 1.72],
        wspace=0.32,
    )
    endpoint_grid = row_three[0, 0].subgridspec(2, 3, hspace=0.48, wspace=0.30)
    f_axes, f_annotation_counts = draw_endpoint_grid(
        endpoint_grid,
        data["endpoint_rows"],
        data["endpoint_stats"],
        rng,
    )
    g_axis = fig.add_subplot(row_three[0, 1])
    draw_tile_time(g_axis, data["tile_time_rows"], data["tile_time_stats"])
    g_axis.set_box_aspect(1)
    g_axis.set_anchor("C")
    speed_grid = row_three[0, 2].subgridspec(1, 2, wspace=0.16)
    speed = speed_summary(data["speed_rows"])
    h_axes = draw_speed_axes(speed_grid, speed)
    panel_axes.update({"f": f_axes, "g": [g_axis], "h": h_axes})

    row_four = outer[6].subgridspec(
        1,
        3,
        width_ratios=[2.55, 1.0, 1.0],
        wspace=0.30,
    )
    example_grid = row_four[0, 0].subgridspec(2, 3, hspace=0.28, wspace=0.38)
    i_axes, i_audit = draw_efficiency_examples(
        example_grid,
        data["example_sequence"],
        data["example_cycles"],
    )
    j_axis = fig.add_subplot(row_four[0, 1])
    k_axis = fig.add_subplot(row_four[0, 2])
    time_summary = efficiency_summary(data["efficiency_time"])
    tile_summary = efficiency_summary(data["efficiency_tile"])
    draw_efficiency_axis(
        j_axis,
        time_summary,
        xlabel="Time in maze (min)",
        legend=True,
    )
    draw_efficiency_axis(
        k_axis,
        tile_summary,
        xlabel="Maze tile transitions",
        legend=False,
    )
    k_axis.tick_params(axis="y", labelleft=False)
    k_axis.set_ylabel("")
    panel_axes.update({"i": i_axes, "j": [j_axis], "k": [k_axis]})

    # Use the four trajectory panels as the shared column skeleton.  The
    # quantitative rows previously used unrelated three- and four-column
    # grids, which made aligned panels drift progressively to the right.
    # Preserve each panel's scientific artists and y geometry; change only the
    # horizontal envelopes and the requested square plotting boxes.
    trajectory_boxes = [axis.get_position() for axis in trajectory_axes]
    column_centers = [0.5 * (box.x0 + box.x1) for box in trajectory_boxes]
    row_two_width = b_axis.get_position().width
    for axis, center in zip((b_axis, c_axis, d_axis, e_axis), column_centers, strict=True):
        box = axis.get_position()
        axis.set_position((center - row_two_width / 2, box.y0, row_two_width, box.height))

    first_two_left = column_centers[0] - row_two_width / 2
    first_two_right = column_centers[1] + row_two_width / 2

    def align_group_to_first_two(axes: list[plt.Axes]) -> None:
        boxes = [axis.get_position() for axis in axes]
        source_left = min(box.x0 for box in boxes)
        source_right = max(box.x1 for box in boxes)
        source_width = source_right - source_left
        target_width = first_two_right - first_two_left
        for axis, box in zip(axes, boxes, strict=True):
            relative_left = (box.x0 - source_left) / source_width
            relative_width = box.width / source_width
            axis.set_position(
                (
                    first_two_left + relative_left * target_width,
                    box.y0,
                    relative_width * target_width,
                    box.height,
                )
            )

    align_group_to_first_two(f_axes)
    align_group_to_first_two(i_axes)
    # Panel f spans the first two page-level columns but contains three
    # internal subplot columns.  Bind panel i to those exact three x slots;
    # preserve panel i's own two-row y geometry and scientific artists.
    for index, axis in enumerate(i_axes):
        reference = f_axes[index % 3]
        reference_box = reference.get_position()
        box = axis.get_position()
        axis.set_position(
            (
                reference_box.x0,
                box.y0,
                reference_box.width,
                box.height,
            )
        )
    # Keep the two left-column panel-f row labels inside the publication page
    # after the group is aligned to the shared four-column skeleton.  The
    # unadjusted geometry places their PDF word boxes about 2 pt beyond the
    # left page edge.
    for axis in (f_axes[0], f_axes[3]):
        axis.yaxis.set_label_coords(-0.269, 0.5)
        axis.tick_params(axis="y", pad=1.0)

    g_box = g_axis.get_position()
    g_side = min(row_two_width, g_box.height)
    g_axis.set_position(
        (
            column_centers[2] - g_side / 2,
            0.5 * (g_box.y0 + g_box.y1) - g_side / 2,
            g_side,
            g_side,
        )
    )
    g_axis.set_box_aspect(1)
    g_axis.set_anchor("C")

    h_boxes = [axis.get_position() for axis in h_axes]
    h_center_y = 0.5 * (
        min(box.y0 for box in h_boxes) + max(box.y1 for box in h_boxes)
    )
    h_gap = 0.012
    h_side = (row_two_width - h_gap) / 2
    h_left = column_centers[3] - row_two_width / 2
    for index, axis in enumerate(h_axes):
        axis.set_position(
            (
                h_left + index * (h_side + h_gap),
                h_center_y - g_side / 2,
                h_side,
                g_side,
            )
        )
        axis.set_box_aspect(None)
        axis.set_xlabel("")
    h_shared_xlabel = fig.text(
        column_centers[3],
        min(axis.get_position().y0 for axis in h_axes) - 0.024,
        "Months since initial exposure",
        fontsize=plt.rcParams["axes.labelsize"],
        ha="center",
        va="top",
    )

    # Give panel i three dedicated left-gutter lanes while keeping its three
    # data columns exactly aligned to panel f above.  The earlier extra 0.02
    # page-width right shift caused panel i to intrude into panel j.
    condition_label_x = 0.008
    for text in fig.texts:
        if text.get_text() in {"Naive example", "Return example"}:
            text.set_x(condition_label_x)
    for axis in (i_axes[0], i_axes[3]):
        axis.yaxis.set_label_coords(-0.22, 0.5)

    # Give panel j its own left-title gutter.  At the shared column-3 center,
    # the vertical `Exploration efficiency` title reached into panel i's
    # rightmost subplot.  Shift j/k together to preserve their spacing and
    # keep both plotting boxes unchanged.
    jk_right_shift = 0.016
    for axis, center in (
        (j_axis, column_centers[2] + jk_right_shift),
        (k_axis, column_centers[3] + jk_right_shift),
    ):
        box = axis.get_position()
        axis.set_position((center - row_two_width / 2, box.y0, row_two_width, box.height))

    for axis in fig.axes:
        axis.grid(False)
    visible_gridline_count = sum(
        line.get_visible()
        for axis in fig.axes
        for line in (*axis.get_xgridlines(), *axis.get_ygridlines())
    )
    trajectory_boxes = [axis.get_position() for axis in trajectory_axes]
    max_trajectory_gap = max(
        right.x0 - left.x1
        for left, right in zip(
            trajectory_boxes[:-1],
            trajectory_boxes[1:],
            strict=True,
        )
    )
    bboxes, panel_label_positions = add_panel_letters(fig, panel_axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    h_legend = h_axes[1].get_legend()
    if h_legend is None:
        raise RuntimeError("Panel h must contain the shared Day 1/Day 2 legend")
    h_legend_bbox_display = h_legend.get_window_extent(renderer)
    h_legend_bbox_axes = h_legend_bbox_display.transformed(
        h_axes[1].transAxes.inverted()
    ).bounds
    reward_speed = speed.loc[speed["analysis_mode"].eq("reward")]
    h_legend_data_overlap_count = 0
    for row in reward_speed.itertuples(index=False):
        for y_value in (
            row.speed_mean_cm_s - row.speed_sem_cm_s,
            row.speed_mean_cm_s,
            row.speed_mean_cm_s + row.speed_sem_cm_s,
        ):
            x_display, y_display = h_axes[1].transData.transform((row.month, y_value))
            h_legend_data_overlap_count += int(
                h_legend_bbox_display.contains(x_display, y_display)
            )
    colorbar_label_bbox = (
        color_axis.yaxis.label.get_window_extent(renderer)
        .transformed(fig.transFigure.inverted())
        .bounds
    )
    h_shared_xlabel_bbox = h_shared_xlabel.get_window_extent(renderer)
    h_axis_xlabel_count = sum(bool(axis.get_xlabel()) for axis in h_axes)
    i_sample_artists = [
        text
        for text in fig.texts
        if text.get_text() in {"Naive example", "Return example"}
    ]
    i_axis_label_artists = [i_axes[0].yaxis.label, i_axes[3].yaxis.label]
    i_tick_artists = [
        label
        for axis in (i_axes[0], i_axes[3])
        for label in axis.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    i_letter_artists = [text for text in fig.texts if text.get_text() == "i"]
    i_text_artists = [
        *i_sample_artists,
        *i_axis_label_artists,
        *i_tick_artists,
        *i_letter_artists,
    ]
    i_text_bboxes = [artist.get_window_extent(renderer) for artist in i_text_artists]

    def overlap_area(first: Any, second: Any) -> float:
        width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
        height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
        return float(width * height)

    i_sample_bboxes = [artist.get_window_extent(renderer) for artist in i_sample_artists]
    i_axis_role_bboxes = [
        artist.get_window_extent(renderer)
        for artist in (*i_axis_label_artists, *i_tick_artists)
    ]
    i_sample_to_axis_max_overlap_px2 = max(
        overlap_area(sample, axis_role)
        for sample in i_sample_bboxes
        for axis_role in i_axis_role_bboxes
    )
    i_letter_bbox = i_letter_artists[0].get_window_extent(renderer)
    i_letter_to_sample_max_overlap_px2 = max(
        overlap_area(i_letter_bbox, sample) for sample in i_sample_bboxes
    )
    figure_pixel_width, figure_pixel_height = fig.canvas.get_width_height()
    i_min_page_clearance_px = min(
        min(box.x0, box.y0, figure_pixel_width - box.x1, figure_pixel_height - box.y1)
        for box in i_text_bboxes
    )
    j_ylabel_bbox = j_axis.yaxis.label.get_window_extent(renderer)
    i_right_axis_bbox = i_axes[2].get_window_extent(renderer)
    i_to_j_ylabel_clearance_px = float(j_ylabel_bbox.x0 - i_right_axis_bbox.x1)
    f_internal_column_centers = [
        float(0.5 * (axis.get_position().x0 + axis.get_position().x1))
        for axis in f_axes[:3]
    ]
    i_internal_column_centers = [
        float(0.5 * (axis.get_position().x0 + axis.get_position().x1))
        for axis in i_axes[:3]
    ]
    i_to_f_internal_column_center_max_delta_fraction = max(
        abs(i_center - f_center)
        for i_center, f_center in zip(
            i_internal_column_centers,
            f_internal_column_centers,
            strict=True,
        )
    )
    i_to_f_internal_column_width_max_delta_fraction = max(
        abs(i_axes[index].get_position().width - f_axes[index].get_position().width)
        for index in range(3)
    )
    curve_legend_bbox_display = curve_legend.get_window_extent(renderer)
    d_letter_artists = [text for text in fig.texts if text.get_text() == "d"]
    if len(d_letter_artists) != 1:
        raise RuntimeError("Expected exactly one panel-d letter")
    d_letter_bbox = d_letter_artists[0].get_window_extent(renderer)
    curve_legend_to_d_clearance_px = float(
        curve_legend_bbox_display.y0 - d_letter_bbox.y1
    )
    curve_legend_to_d_overlap_px2 = overlap_area(
        curve_legend_bbox_display,
        d_letter_bbox,
    )
    a_to_legend_gap = max(
        0.0,
        bboxes["a"][1] - curve_legend_axis.get_position().y1,
    )
    f_annotation_headroom = min(
        (
            axis.get_ylim()[1]
            - max(
                text.get_position()[1]
                for text in axis.texts
                if text.get_text() in {"*", "**", "***"}
            )
        )
        / axis.get_ylim()[1]
        for axis in f_axes
    )
    figure_width, figure_height = fig.get_size_inches()

    def physical_box_aspect(axis: plt.Axes) -> float:
        box = axis.get_position()
        return float((box.height * figure_height) / (box.width * figure_width))

    f_row_center = 0.5 * (bboxes["f"][1] + bboxes["f"][3])
    gh_center_delta = max(
        abs(0.5 * (bboxes[letter][1] + bboxes[letter][3]) - f_row_center)
        for letter in ("g", "h")
    )
    axis_scale_rows: list[dict[str, Any]] = []
    for panel, axes in panel_axes.items():
        for axis_index, axis in enumerate(axes):
            collection_norms = [
                collection.norm
                for collection in axis.collections
                if getattr(collection, "norm", None) is not None
            ]
            axis_scale_rows.append(
                {
                    "panel": panel,
                    "axis_index": axis_index,
                    "title": axis.get_title(),
                    "xlabel": axis.get_xlabel(),
                    "ylabel": axis.get_ylabel(),
                    "xscale": axis.get_xscale(),
                    "yscale": axis.get_yscale(),
                    "xlim_low": float(axis.get_xlim()[0]),
                    "xlim_high": float(axis.get_xlim()[1]),
                    "ylim_low": float(axis.get_ylim()[0]),
                    "ylim_high": float(axis.get_ylim()[1]),
                    "xticks": json.dumps([float(value) for value in axis.get_xticks()]),
                    "yticks": json.dumps([float(value) for value in axis.get_yticks()]),
                    "visible_x_tick_labels": sum(
                        label.get_visible() and bool(label.get_text())
                        for label in axis.get_xticklabels()
                    ),
                    "visible_y_tick_labels": sum(
                        label.get_visible() and bool(label.get_text())
                        for label in axis.get_yticklabels()
                    ),
                    "collection_norm_vmin": (
                        float(collection_norms[0].vmin)
                        if collection_norms and collection_norms[0].vmin is not None
                        else np.nan
                    ),
                    "collection_norm_vmax": (
                        float(collection_norms[0].vmax)
                        if collection_norms and collection_norms[0].vmax is not None
                        else np.nan
                    ),
                }
            )
    derived = {
        "accumulation_summary": accumulation,
        "speed_summary": speed,
        "time_efficiency_summary": time_summary,
        "tile_efficiency_summary": tile_summary,
        "shared_trajectory_minutes": shared_max,
        "curve_legend_labels": [
            text.get_text() for text in curve_legend.get_texts()
        ],
        "f_annotation_counts": f_annotation_counts,
        "panel_f_row_ylims": [
            [list(axis.get_ylim()) for axis in f_axes[:3]],
            [list(axis.get_ylim()) for axis in f_axes[3:]],
        ],
        "panel_f_row_yticks": [
            [list(axis.get_yticks()) for axis in f_axes[:3]],
            [list(axis.get_yticks()) for axis in f_axes[3:]],
        ],
        "i_audit": i_audit,
        "visible_gridline_count": visible_gridline_count,
        "colorbar_orientation": "vertical",
        "max_trajectory_gap_fraction": max_trajectory_gap,
        "colorbar_label_bbox_fraction": colorbar_label_bbox,
        "panel_a_to_curve_legend_gap_fraction": a_to_legend_gap,
        "panel_f_annotation_headroom_fraction": f_annotation_headroom,
        "panel_f_bracket_step_fraction": F_BRACKET_STEP_FRACTION,
        "panel_f_star_offset_fraction": F_STAR_OFFSET_FRACTION,
        "panel_f_star_fontsize_pt": F_STAR_FONTSIZE_PT,
        "panel_label_positions": panel_label_positions,
        "panel_g_box_aspect": physical_box_aspect(g_axis),
        "panel_h_box_aspects": [physical_box_aspect(axis) for axis in h_axes],
        "panel_h_ylabel": h_axes[0].get_ylabel(),
        "panel_h_ylims": [list(axis.get_ylim()) for axis in h_axes],
        "panel_h_yticks": [list(axis.get_yticks()) for axis in h_axes],
        "axis_scale_audit": pd.DataFrame(axis_scale_rows),
        "panel_i_titles": [axis.get_title() for axis in i_axes[:3]],
        "panel_g_physical_height_in": g_axis.get_position().height * figure_height,
        "panel_h_physical_heights_in": [
            axis.get_position().height * figure_height for axis in h_axes
        ],
        "panel_fgh_center_alignment_delta_fraction": gh_center_delta,
        "panel_h_legend_location": h_legend._loc,
        "panel_h_legend_labels": [text.get_text() for text in h_legend.get_texts()],
        "panel_h_legend_bbox_axes_fraction": h_legend_bbox_axes,
        "panel_h_legend_data_overlap_count": h_legend_data_overlap_count,
        "panel_h_axis_xlabel_count": h_axis_xlabel_count,
        "panel_h_shared_xlabel_bbox_px": list(h_shared_xlabel_bbox.bounds),
        "panel_i_sample_to_axis_max_overlap_px2": i_sample_to_axis_max_overlap_px2,
        "panel_i_letter_to_sample_max_overlap_px2": i_letter_to_sample_max_overlap_px2,
        "panel_i_label_min_page_clearance_px": i_min_page_clearance_px,
        "panel_i_to_j_ylabel_clearance_px": i_to_j_ylabel_clearance_px,
        "panel_f_internal_column_centers_fraction": f_internal_column_centers,
        "panel_i_internal_column_centers_fraction": i_internal_column_centers,
        "panel_i_to_f_internal_column_center_max_delta_fraction": (
            i_to_f_internal_column_center_max_delta_fraction
        ),
        "panel_i_to_f_internal_column_width_max_delta_fraction": (
            i_to_f_internal_column_width_max_delta_fraction
        ),
        "curve_legend_to_d_clearance_px": curve_legend_to_d_clearance_px,
        "curve_legend_to_d_overlap_px2": curve_legend_to_d_overlap_px2,
    }
    return fig, bboxes, derived
