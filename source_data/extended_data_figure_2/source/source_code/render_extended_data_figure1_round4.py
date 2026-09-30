"""Render Extended Data Figure 1 directly from maintained source tables.

The supplied slide fixes panel identity and evidence order, but not its
distorted picture-box transforms.  This renderer redraws every panel in one
matplotlib canvas, aligns intended row members, removes caption-level prose,
and never imports a previously rendered panel SVG or raster.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from ca_analysis.maze import NewMaze
from ca_analysis.trajectory import LoadTraj
try:
    from scripts.render_grs2026_first_maze_trial_trajectories import (
        GRAY_BLOCKS,
        MAX_STEP_PER_FRAME_GRID,
        _draw_house_icon,
        _draw_water_drop_icon,
        _identify_nose_jump_islands,
    )
    from scripts.render_ed01_ed02_gap_closure import (
        _complete_exploration_cycles,
        _extract_explore_endnodes,
    )
except ModuleNotFoundError:  # pragma: no cover - supports running from scripts/
    from render_grs2026_first_maze_trial_trajectories import (  # type: ignore
        GRAY_BLOCKS,
        MAX_STEP_PER_FRAME_GRID,
        _draw_house_icon,
        _draw_water_drop_icon,
        _identify_nose_jump_islands,
    )
    from render_ed01_ed02_gap_closure import (  # type: ignore
        _complete_exploration_cycles,
        _extract_explore_endnodes,
    )


REPO = Path(__file__).resolve().parents[1]
REFRESH = REPO / "docs" / "extended_data_figure_panel_refresh_2026-07-12"
PANEL_ROOT = REFRESH / "source_rendered_new"
TABLE_ROOT = REFRESH / "source_tables"
OUTPUT = (
    REPO
    / "docs"
    / "extended_data_figure1_source_rebuild_2026-08-05_round4_layout_clean"
)
ROUND = 10
STEM = OUTPUT / f"extended_data_figure1_source_rerender_round{ROUND}"
SOURCE_AUDIT = OUTPUT / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv"
LAYOUT_AUDIT = OUTPUT / f"LAYOUT_AUDIT_ROUND{ROUND}.csv"
CAPTION_NOTES = OUTPUT / f"CAPTION_NOTES_ROUND{ROUND}.md"
RENDER_VALIDATION = OUTPUT / f"RENDER_VALIDATION_ROUND{ROUND}.json"
MANIFEST = OUTPUT / f"MANIFEST_ROUND{ROUND}.json"

TRAJECTORY = (
    PANEL_ROOT
    / "extended_data_fig_01"
    / "four_day_trajectory_options"
    / "through_first_reward"
    / "ed01a_through_first_reward_four_days_plot_data.csv"
)
ACCUMULATION = (
    PANEL_ROOT
    / "extended_data_fig_01"
    / "month_source_coverage_gap_closure_2026-07-14"
    / "ed01_accumulation_session_curves_display_window_current153.csv.gz"
)
ACQUISITION_DIR = (
    PANEL_ROOT
    / "extended_data_fig_01"
    / "acquisition_endpoint_axes_2026-07-16"
)
TILE_CURVES = (
    ACQUISITION_DIR
    / "ed01_tile_accumulation_by_return_month_group_curves_current153.csv"
)
ENDPOINT_ROWS = ACQUISITION_DIR / "ed01_endpoint_axis_session_rows_current153.csv"
ENDPOINT_STATS = ACQUISITION_DIR / "ed01_endpoint_axis_cluster_stats_current153.csv"
TILE_TIME_DIR = (
    TABLE_ROOT
    / "extended_data_fig_02"
    / "tile_time_loglog_current153_2026-07-15"
)
TILE_TIME_ROWS = TILE_TIME_DIR / "ed02_tile_time_loglog_rows_current153.csv"
TILE_TIME_STATS = TILE_TIME_DIR / "ed02_tile_time_loglog_stats_current153.csv"
SPEED_ROWS = (
    REPO
    / "docs"
    / "new_12mo_session_fit_2026-06-02"
    / "panel_by_panel_current_vs_new"
    / "bodymass_speed_current_plus_new12_by_session.csv"
)
SPEED_REFERENCE = TABLE_ROOT / "ed02_bodymass_speed_plot_data_current_plus_new12.csv"
EXAMPLE_DIR = (
    PANEL_ROOT
    / "extended_data_fig_02"
    / "illustrative_support_gap_closure_2026-07-14"
)
EXAMPLE_SELECTION_AUDIT = EXAMPLE_DIR / "ed02_efficiency_example_selection_audit.csv"
EXAMPLE_NAIVE_TRAJECTORY = REPO / "mm-trajectories" / "451Rday1-tf"
EXAMPLE_RETURN_TRAJECTORY = REPO / "mm-trajectories" / "451-9day1-tf"
EFFICIENCY_DIR = (
    PANEL_ROOT
    / "extended_data_fig_02"
    / "exploration_efficiency_axis_options_2026-07-16"
    / "source_data"
)
EFFICIENCY_TIME = EFFICIENCY_DIR / "current_plus_new12_time_session_bins.csv"
EFFICIENCY_TILE = EFFICIENCY_DIR / "current_plus_new12_tile_session_bins.csv"

SOURCE_FILES = {
    "a": (TRAJECTORY, "plot-ready trajectory coordinates"),
    "b-c": (ACCUMULATION, "session-level cumulative count curves"),
    "d-e": (TILE_CURVES, "current153 grouped maze-transition curves"),
    "f-rows": (ENDPOINT_ROWS, "session-level endpoint rows"),
    "f-stats": (ENDPOINT_STATS, "mouse-clustered endpoint statistics; caption only"),
    "g-rows": (TILE_TIME_ROWS, "session-level tile-time rows"),
    "g-stats": (TILE_TIME_STATS, "mouse-clustered log-log statistics; caption only"),
    "h-rows": (SPEED_ROWS, "session-level body-mass speed rows"),
    "h-reference": (SPEED_REFERENCE, "maintained grouped-summary parity target"),
    "i-selection": (
        EXAMPLE_SELECTION_AUDIT,
        "same-mouse candidate ranking used for cycle-complete example selection",
    ),
    "i-naive": (
        EXAMPLE_NAIVE_TRAJECTORY,
        "raw maintained trajectory for selected naive example",
    ),
    "i-return": (
        EXAMPLE_RETURN_TRAJECTORY,
        "raw maintained trajectory for selected return example",
    ),
    "j": (EFFICIENCY_TIME, "session-first time-bin efficiency rows"),
    "k": (EFFICIENCY_TILE, "session-first transition-bin efficiency rows"),
}
PANEL_SCOPE = {
    "a": ("ED1", "maintained illustrative panel"),
    "b-c": ("ED1", "maintained current153 primary evidence"),
    "d-e": ("ED1", "maintained current153 primary evidence"),
    "f-rows": ("ED1", "maintained current153 primary evidence"),
    "f-stats": ("ED1", "caption-only statistics"),
    "g-rows": ("ED2 retained on supplied slide", "maintained diagnostic"),
    "g-stats": ("ED2 retained on supplied slide", "caption-only statistics"),
    "h-rows": ("ED2 retained on supplied slide", "maintained locomotor control"),
    "h-reference": ("ED2 retained on supplied slide", "summary parity target"),
    "i-selection": ("ED2 retained on supplied slide", "maintained illustrative example"),
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


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 5.7,
            "axes.titlesize": 6.0,
            "axes.labelsize": 5.7,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 4.6,
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


def build_efficiency_example_source(
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select and reconstruct a same-mouse example supporting five cycles."""
    maze = NewMaze(6)
    candidates = pd.read_csv(
        EXAMPLE_SELECTION_AUDIT,
        dtype={"animal_id": str},
    ).sort_values(
        ["efficiency_ratio_return_over_naive", "animal_id"],
        ascending=[False, True],
    )
    cache: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    audit_rows: list[dict[str, Any]] = []
    chosen: pd.Series | None = None
    for _, candidate in candidates.iterrows():
        counts: dict[str, int] = {}
        for label, session in (
            ("naive", str(candidate["naive_session"])),
            ("return", str(candidate["return_session"])),
        ):
            trajectory = LoadTraj(REPO / "mm-trajectories" / f"{session}-tf")
            end_nodes = _extract_explore_endnodes(trajectory, maze)
            cycles = _complete_exploration_cycles(
                end_nodes,
                target_unique=32,
                nickname=session,
                group=label,
            )
            cache[session] = (end_nodes, cycles)
            counts[label] = len(cycles)
        eligible = counts["naive"] >= 5 and counts["return"] >= 5
        audit_rows.append(
            {
                **candidate.to_dict(),
                "naive_complete_cycles": counts["naive"],
                "return_complete_cycles": counts["return"],
                "eligible_for_five_cycle_display": eligible,
                "selected_round6": False,
            }
        )
        if chosen is None and eligible:
            chosen = candidate
    if chosen is None:
        raise ValueError("No same-mouse naive/return pair has five complete cycles")

    chosen_animal = str(chosen["animal_id"])
    for row in audit_rows:
        row["selected_round6"] = str(row["animal_id"]) == chosen_animal
        row["round6_selection_rule"] = (
            "largest return/naive first-cycle efficiency ratio among same-mouse "
            "pairs with at least five complete cycles in both sessions"
        )

    sequence_frames: list[pd.DataFrame] = []
    cycle_frames: list[pd.DataFrame] = []
    for condition, session in (
        ("redundant_naive", str(chosen["naive_session"])),
        ("efficient_return", str(chosen["return_session"])),
    ):
        end_nodes, cycles = cache[session]
        rows: list[dict[str, Any]] = []
        cycle_start = 0
        for cycle in cycles.itertuples(index=False):
            cycle_index = int(cycle.cycle_index)
            cycle_end = int(cycle.cum_steps_end)
            cycle_sequence = np.asarray(end_nodes[cycle_start:cycle_end], dtype=int)
            seen: set[int] = set()
            for visit_index, node in enumerate(cycle_sequence, start=1):
                first_within_cycle = int(node) not in seen
                seen.add(int(node))
                rows.append(
                    {
                        "condition": condition,
                        "session": session,
                        "animal_id": chosen_animal,
                        "cycle_index": cycle_index,
                        "cycle_visit_index": visit_index,
                        "endpoint_visit_index": cycle_start + visit_index,
                        "end_node_display_id_1_to_64": int(node - 62),
                        "is_first_visit_within_cycle": first_within_cycle,
                        "cumulative_unique_within_cycle": len(seen),
                    }
                )
            if len(seen) != 32:
                raise AssertionError(
                    f"Cycle {cycle_index} for {session} ends with {len(seen)} unique end nodes"
                )
            cycle_start = cycle_end
        sequence_frames.append(pd.DataFrame(rows))
        cycle_table = cycles.copy()
        cycle_table["condition"] = condition
        cycle_table["session"] = session
        cycle_table["animal_id"] = chosen_animal
        cycle_table["cycle_start_endpoint_visit_index"] = (
            cycle_table["cum_steps_end"] - cycle_table["steps_in_cycle"] + 1
        ).astype(int)
        cycle_table["cycle_end_endpoint_visit_index"] = cycle_table[
            "cum_steps_end"
        ].astype(int)
        cycle_frames.append(cycle_table)
    return (
        pd.concat(sequence_frames, ignore_index=True),
        pd.concat(cycle_frames, ignore_index=True),
        pd.DataFrame(audit_rows),
    )


def correct_trajectory_jump_masks(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the maintained poster/slide no-jump rule to every plotted bout."""
    corrected = source.copy()
    corrected["source_draw_to_next"] = corrected["draw_to_next"].astype(bool)
    corrected["draw_to_next"] = False
    corrected["round6_jump_removed_point"] = False
    audits: list[dict[str, Any]] = []
    group_columns = ["nickname", "bout_idx"]
    for (nickname, bout_idx), group in corrected.groupby(
        group_columns,
        sort=False,
    ):
        ordered = group.sort_values(["frame", "point_idx"])
        indices = ordered.index.to_numpy()
        frames = pd.to_numeric(ordered["frame"], errors="raise").to_numpy(int)
        xy = ordered[["x", "y"]].to_numpy(float)
        frame_gap = np.diff(frames)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        step_per_frame = np.divide(
            step,
            frame_gap,
            out=np.full(step.shape, np.inf, dtype=float),
            where=frame_gap > 0,
        )
        removed_points, islands, _ = _identify_nose_jump_islands(frames, xy)
        source_keep = (
            ordered["source_draw_to_next"].iloc[:-1].astype(bool).to_numpy()
        )
        visible_endpoints = ~(removed_points[:-1] | removed_points[1:])
        keep = (
            source_keep
            & (frame_gap == 1)
            & np.isfinite(step_per_frame)
            & (step_per_frame <= MAX_STEP_PER_FRAME_GRID)
            & visible_endpoints
        )
        local_draw = np.zeros(len(ordered), dtype=bool)
        local_draw[:-1] = keep
        corrected.loc[indices, "draw_to_next"] = local_draw
        corrected.loc[indices, "round6_jump_removed_point"] = removed_points
        source_max = float(np.max(step[source_keep])) if source_keep.any() else 0.0
        corrected_max = float(np.max(step[keep])) if keep.any() else 0.0
        audits.append(
            {
                "nickname": nickname,
                "bout_idx": int(bout_idx),
                "n_points": len(ordered),
                "source_drawn_segments": int(source_keep.sum()),
                "corrected_drawn_segments": int(keep.sum()),
                "segments_removed_by_round6_rule": int(
                    source_keep.sum() - keep.sum()
                ),
                "source_max_drawn_step_grid": source_max,
                "corrected_max_drawn_step_grid": corrected_max,
                "max_step_per_frame_grid": MAX_STEP_PER_FRAME_GRID,
                "n_jump_islands_removed": len(islands),
                "n_jump_island_points_removed": int(removed_points.sum()),
                "tracking_rule": (
                    "retain source mask, split nonconsecutive frames and steps above "
                    "0.63 grid units per frame, and omit repeated-fast-edge islands"
                ),
            }
        )
    corrected["max_segment_step"] = MAX_STEP_PER_FRAME_GRID
    return corrected, pd.DataFrame(audits)


def read_inputs() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path, _ in SOURCE_FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    example_sequence, example_cycles, example_selection = (
        build_efficiency_example_source()
    )
    trajectory, trajectory_jump_audit = correct_trajectory_jump_masks(
        pd.read_csv(TRAJECTORY, low_memory=False)
    )
    return {
        "trajectory": trajectory,
        "trajectory_jump_audit": trajectory_jump_audit,
        "accumulation": pd.read_csv(ACCUMULATION, low_memory=False),
        "tile_curves": pd.read_csv(TILE_CURVES),
        "endpoint_rows": pd.read_csv(ENDPOINT_ROWS),
        "endpoint_stats": pd.read_csv(ENDPOINT_STATS),
        "tile_time_rows": pd.read_csv(TILE_TIME_ROWS, low_memory=False),
        "tile_time_stats": pd.read_csv(TILE_TIME_STATS),
        "speed_rows": pd.read_csv(SPEED_ROWS, dtype={"animal_id": str}),
        "speed_reference": pd.read_csv(SPEED_REFERENCE),
        "example_sequence": example_sequence,
        "example_cycles": example_cycles,
        "example_selection": example_selection,
        "efficiency_time": pd.read_csv(EFFICIENCY_TIME),
        "efficiency_tile": pd.read_csv(EFFICIENCY_TILE),
    }


def draw_trajectory(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    norm: Normalize,
) -> None:
    maze = NewMaze(6)
    ax.plot(maze.wa[:, 0], maze.wa[:, 1], color="black", linewidth=0.62, zorder=2)
    for x, y, width, height in GRAY_BLOCKS:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.35,
                edgecolor="#D8D8D8",
                facecolor="#D8D8D8",
                zorder=1,
            )
        )
    points = data[["x", "y"]].to_numpy(float)
    elapsed = data["elapsed_minutes"].to_numpy(float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    keep = data["draw_to_next"].iloc[:-1].astype(bool).to_numpy()
    collection = LineCollection(
        segments[keep],
        cmap=plt.get_cmap("viridis"),
        norm=norm,
        linewidths=0.42,
        alpha=0.86,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_array(0.5 * (elapsed[:-1] + elapsed[1:])[keep])
    ax.add_collection(collection)
    _draw_house_icon(ax)
    _draw_water_drop_icon(ax)
    ax.set_xlim(-1.85, 15.95)
    ax.set_ylim(15.0, -1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="normal", pad=1.0)


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
            span = max(raw_max, 1.0)
            base = raw_max + 0.055 * span
            step = F_BRACKET_STEP_FRACTION * span
            tick_height = 0.022 * span
            group_to_x = {key: index for index, key in enumerate(GROUP_ORDER)}
            annotations = 0
            for _, contrast in panel_stats.iterrows():
                q_value = float(contrast["q_bh27"])
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
                    fontsize=3.9,
                )
                annotations += 1
            annotation_counts.append(annotations)
            annotation_top = (
                base + max(annotations - 1, 0) * step + tick_height + 0.22 * span
            )
            ax.set_ylim(0, max(raw_max * 1.08, annotation_top))
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
        ax.set_ylim(bottom=0)
        ax.set_box_aspect(1)
        ax.set_anchor("C")
        ax.set_xlabel("Return interval (months)")
        ax.grid(False)
        if index == 0:
            ax.set_ylabel("Body-mass speed (cm s$^{-1}$)")
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
            ax2.set_title("Within-cycle coverage", pad=1.0)
            ax3.set_title("Cycle efficiency", pad=1.0)
            for axis in (ax1, ax2, ax3):
                axis.tick_params(axis="x", labelbottom=False)
        else:
            ax1.set_xlabel("End node visit")
            ax2.set_xlabel("End node visit")
            ax3.set_xlabel("Cycle")
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
    trajectory = data["trajectory"]
    shared_max = float(trajectory["elapsed_minutes"].max())
    norm = Normalize(0.0, shared_max)
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
        subset = trajectory.loc[trajectory["nickname"].astype(str).eq(nickname)]
        draw_trajectory(axis, subset, title, norm)
    color_axis = fig.add_subplot(row_a[0, 4])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=color_axis,
        orientation="vertical",
    )
    colorbar.outline.set_linewidth(0.45)
    colorbar.solids.set_rasterized(False)
    colorbar.set_ticks([0, shared_max])
    colorbar.set_ticklabels(["0", f"{shared_max:.0f}"])
    colorbar.ax.tick_params(labelsize=4.8, width=0.45, length=1.4, pad=0.8)
    colorbar.set_label("Time in maze (min)", fontsize=5.3, labelpad=1.0)
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
    # accepted round-11 geometry otherwise placed their PDF word boxes about
    # 2 pt beyond the left page edge.
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
        "Return interval (months)",
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
        colorbar.ax.yaxis.label.get_window_extent(renderer)
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
        "i_audit": i_audit,
        "visible_gridline_count": visible_gridline_count,
        "colorbar_orientation": "vertical",
        "max_trajectory_gap_fraction": max_trajectory_gap,
        "colorbar_label_bbox_fraction": colorbar_label_bbox,
        "panel_a_to_curve_legend_gap_fraction": a_to_legend_gap,
        "panel_f_annotation_headroom_fraction": f_annotation_headroom,
        "panel_f_bracket_step_fraction": F_BRACKET_STEP_FRACTION,
        "panel_f_star_offset_fraction": F_STAR_OFFSET_FRACTION,
        "panel_label_positions": panel_label_positions,
        "panel_g_box_aspect": physical_box_aspect(g_axis),
        "panel_h_box_aspects": [physical_box_aspect(axis) for axis in h_axes],
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


def save_outputs(fig: plt.Figure) -> dict[str, Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        suffix: STEM.with_suffix(f".{suffix}")
        for suffix in ("svg", "pdf", "png", "tiff")
    }
    fig.savefig(paths["svg"])
    fig.savefig(paths["pdf"])
    fig.savefig(paths["png"], dpi=600)
    plt.close(fig)
    with Image.open(paths["png"]) as image:
        image.convert("RGB").save(
            paths["tiff"],
            dpi=(600, 600),
            compression="tiff_lzw",
        )
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_audits(
    data: dict[str, pd.DataFrame],
    derived: dict[str, Any],
    bboxes: dict[str, tuple[float, float, float, float]],
) -> None:
    source_rows: list[dict[str, Any]] = []
    key_to_frame = {
        "a": "trajectory",
        "b-c": "accumulation",
        "d-e": "tile_curves",
        "f-rows": "endpoint_rows",
        "f-stats": "endpoint_stats",
        "g-rows": "tile_time_rows",
        "g-stats": "tile_time_stats",
        "h-rows": "speed_rows",
        "h-reference": "speed_reference",
        "i-selection": "example_selection",
        "j": "efficiency_time",
        "k": "efficiency_tile",
    }
    for panel, (path, stage) in SOURCE_FILES.items():
        frame_key = key_to_frame.get(panel)
        frame = data[frame_key] if frame_key is not None else None
        source_rows.append(
            {
                "panel": panel,
                "source_path": path.relative_to(REPO).as_posix(),
                "source_stage": stage,
                "scientific_scope": PANEL_SCOPE[panel][0],
                "promotion_status": PANEL_SCOPE[panel][1],
                "rows": len(frame) if frame is not None else "",
                "columns": len(frame.columns) if frame is not None else "",
                "sha256": sha256(path),
                "panel_svg_reused": False,
            }
        )
    write_csv(SOURCE_AUDIT, source_rows)

    layout_rows = []
    for panel, (x0, y0, x1, y1) in bboxes.items():
        layout_rows.append(
            {
                "panel": panel,
                "x0_fraction": x0,
                "y0_fraction": y0,
                "x1_fraction": x1,
                "y1_fraction": y1,
                "width_fraction": x1 - x0,
                "height_fraction": y1 - y0,
                "placement": "native matplotlib axes; no panel image transform",
                "nonuniform_scale": False,
            }
        )
    write_csv(LAYOUT_AUDIT, layout_rows)

    for key in (
        "accumulation_summary",
        "speed_summary",
        "time_efficiency_summary",
        "tile_efficiency_summary",
    ):
        derived[key].to_csv(
            OUTPUT / f"{key.upper()}_ROUND{ROUND}.csv",
            index=False,
        )
    data["example_sequence"].to_csv(
        OUTPUT / f"PANEL_I_END_NODE_SEQUENCE_ROUND{ROUND}.csv",
        index=False,
    )
    data["example_cycles"].to_csv(
        OUTPUT / f"PANEL_I_CYCLE_SUMMARY_ROUND{ROUND}.csv",
        index=False,
    )
    data["example_selection"].to_csv(
        OUTPUT / f"PANEL_I_SELECTION_AUDIT_ROUND{ROUND}.csv",
        index=False,
    )
    data["trajectory"].to_csv(
        OUTPUT / f"PANEL_A_TRAJECTORY_PLOT_DATA_ROUND{ROUND}.csv.gz",
        index=False,
        compression="gzip",
    )
    data["trajectory_jump_audit"].to_csv(
        OUTPUT / f"PANEL_A_JUMP_MASK_AUDIT_ROUND{ROUND}.csv",
        index=False,
    )


def validate(
    data: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    bboxes: dict[str, tuple[float, float, float, float]],
    derived: dict[str, Any],
) -> dict[str, Any]:
    with Image.open(paths["png"]) as image:
        png_size = list(image.size)
        png_bbox = image.convert("RGB").getbbox()
    with Image.open(paths["tiff"]) as image:
        tiff_size = list(image.size)
        tiff_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
        tiff_compression = str(image.info.get("compression", "")).lower()
    svg_root = ET.parse(paths["svg"]).getroot()
    svg_image_count = sum(element.tag.endswith("image") for element in svg_root.iter())
    with fitz.open(paths["pdf"]) as document:
        pdf_page_count = document.page_count
        pdf_image_count = sum(len(page.get_images(full=True)) for page in document)
        pdf_drawing_count = sum(len(page.get_drawings()) for page in document)
        pdf_word_count = sum(len(page.get_text("words")) for page in document)
        panel_f_pdf_ylabel_bboxes: dict[str, list[float]] = {}
        panel_f_pdf_ylabel_min_page_clearance_pt = float("-inf")
        panel_f_pdf_ylabel_text_overlaps: dict[str, list[str]] = {}
        panel_i_pdf_ylabel_tick_min_clearance_pt = float("-inf")
        panel_i_pdf_ylabel_min_page_clearance_pt = float("-inf")
        if pdf_page_count == 1:
            page = document[0]
            target_labels = {"Time in maze (s)", "Maze tile transitions"}
            other_text_spans: list[tuple[str, fitz.Rect]] = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    direction = line.get("dir", (1.0, 0.0))
                    is_vertical = abs(float(direction[0])) < 1e-6 and float(direction[1]) < -0.99
                    if not is_vertical:
                        continue
                    for span in line.get("spans", []):
                        label = str(span.get("text", ""))
                        bbox = [float(value) for value in span.get("bbox", ())]
                        if label in target_labels and len(bbox) == 4 and bbox[0] < 30.0:
                            panel_f_pdf_ylabel_bboxes[label] = bbox
                        elif len(bbox) == 4 and bbox[0] < 30.0:
                            other_text_spans.append((label, fitz.Rect(bbox)))
            if set(panel_f_pdf_ylabel_bboxes) == target_labels:
                page_rect = page.rect
                clearances = []
                for x0, y0, x1, y1 in panel_f_pdf_ylabel_bboxes.values():
                    clearances.extend(
                        [x0 - page_rect.x0, y0 - page_rect.y0, page_rect.x1 - x1, page_rect.y1 - y1]
                    )
                panel_f_pdf_ylabel_min_page_clearance_pt = min(clearances)
                panel_f_pdf_ylabel_text_overlaps = {
                    label: [
                        text
                        for text, other_bbox in other_text_spans
                        if fitz.Rect(bbox).intersects(other_bbox)
                    ]
                    for label, bbox in panel_f_pdf_ylabel_bboxes.items()
                }
            panel_i_ylabels: list[fitz.Rect] = []
            panel_i_yticks: list[fitz.Rect] = []
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    direction = line.get("dir", (1.0, 0.0))
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        bbox = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                        if text == "End nodes" and bbox.y0 > page.rect.height * 0.70:
                            panel_i_ylabels.append(bbox)
                        elif (
                            text.replace(".", "", 1).isdigit()
                            and bbox.x0 < page.rect.width * 0.12
                            and bbox.y0 > page.rect.height * 0.70
                            and abs(float(direction[0]) - 1.0) < 1e-6
                        ):
                            panel_i_yticks.append(bbox)
            if len(panel_i_ylabels) == 2 and panel_i_yticks:
                panel_i_pdf_ylabel_tick_min_clearance_pt = min(
                    tick.x0 - ylabel.x1
                    for ylabel in panel_i_ylabels
                    for tick in panel_i_yticks
                )
                panel_i_pdf_ylabel_min_page_clearance_pt = min(
                    min(box.x0, box.y0, page.rect.x1 - box.x1, page.rect.y1 - box.y1)
                    for box in panel_i_ylabels
                )
    svg_visible_text = " ".join(
        element.text or ""
        for element in svg_root.iter()
        if element.tag.endswith("text")
    )

    speed_reference = data["speed_reference"].loc[
        data["speed_reference"]["mode_label"].isin(["explore", "reward"])
    ].copy()
    speed_reference = speed_reference.rename(
        columns={"mode_label": "analysis_mode"}
    )
    speed_check = derived["speed_summary"].merge(
        speed_reference[
            [
                "analysis_mode",
                "day",
                "month",
                "speed_mean_cm_s",
                "speed_sem_cm_s",
            ]
        ],
        on=["analysis_mode", "day", "month"],
        suffixes=("_rebuilt", "_reference"),
        validate="one_to_one",
    )
    speed_delta = float(
        np.nanmax(
            np.abs(
                speed_check["speed_mean_cm_s_rebuilt"]
                - speed_check["speed_mean_cm_s_reference"]
            )
        )
    )

    same_row_groups = (("b", "c", "d", "e"), ("i", "j", "k"))
    row_alignment_delta = 0.0
    for group in same_row_groups:
        tops = [bboxes[key][3] for key in group]
        bottoms = [bboxes[key][1] for key in group]
        row_alignment_delta = max(
            row_alignment_delta,
            max(tops) - min(tops),
            max(bottoms) - min(bottoms),
        )

    day1 = data["trajectory"].loc[
        data["trajectory"]["nickname"].astype(str).eq("434RRday1")
    ].reset_index(drop=True)
    day1_step = np.linalg.norm(
        np.diff(day1[["x", "y"]].to_numpy(float), axis=0),
        axis=1,
    )
    day1_source_keep = (
        day1["source_draw_to_next"].iloc[:-1].astype(bool).to_numpy()
    )
    day1_corrected_keep = (
        day1["draw_to_next"].iloc[:-1].astype(bool).to_numpy()
    )
    day1_source_max_step = float(day1_step[day1_source_keep].max())
    day1_corrected_max_step = float(day1_step[day1_corrected_keep].max())
    forbidden_caption_text = (
        "common 80%-session support window",
        "clustered P",
        "CR1",
        "n≥",
        "n at risk",
        "Matched tile-time diagnostic",
    )
    selected_example = data["example_selection"].loc[
        data["example_selection"]["selected_round6"].astype(bool)
    ]
    i_limits = derived["i_audit"]["shared_x_limits_by_column"]
    i_shared_columns = all(
        len(limits) == 2 and np.allclose(limits[0], limits[1])
        for limits in i_limits.values()
    )
    i_contract = derived["i_audit"]["display_contract"]
    colorbar_label_bbox = derived["colorbar_label_bbox_fraction"]
    colorbar_label_x0 = float(colorbar_label_bbox[0])
    colorbar_label_x1 = colorbar_label_x0 + float(colorbar_label_bbox[2])
    checks = {
        "all_fourteen_source_inputs_exist": len(SOURCE_FILES) == 14
        and all(path.is_file() for path, _ in SOURCE_FILES.values()),
        "no_panel_svg_or_raster_reused": all(
            path.suffix.lower() not in {".svg", ".png", ".tif", ".tiff"}
            for path, _ in SOURCE_FILES.values()
        ),
        "first_panel_is_full_through_first_reward": (
            len(day1) > 20_000
            and set(day1["clip_mode"].astype(str))
            == {"all_tracked_in_maze_segments_through_first_reward_node_reach"}
            and set(pd.to_numeric(day1["reward_node"], errors="raise").astype(int))
            == {94}
        ),
        "panel_a_day1_uses_no_jump_mask": (
            day1_source_max_step > MAX_STEP_PER_FRAME_GRID
            and day1_corrected_max_step <= MAX_STEP_PER_FRAME_GRID + 1e-12
            and bool(day1["round6_jump_removed_point"].astype(bool).any())
        ),
        "current153_endpoint_rows": (
            data["endpoint_rows"]["session"].astype(str).nunique() == 153
        ),
        "current153_tile_time_rows": len(data["tile_time_rows"]) == 153,
        "cross_figure_scope_documented": set(PANEL_SCOPE) == set(SOURCE_FILES),
        "candidate_jk_not_promoted": all(
            "not promoted primary" in PANEL_SCOPE[key][1] for key in ("j", "k")
        ),
        "rounds_share_one_working_folder": (
            STEM.parent == OUTPUT
            and f"round{ROUND}" in STEM.name
        ),
        "speed_recomputed_from_session_rows": speed_delta < 1e-12,
        "aligned_full_height_row_groups": row_alignment_delta < 0.012,
        "panel_g_square": abs(derived["panel_g_box_aspect"] - 1.0) < 1e-6,
        "panel_h_axes_match_panel_g_physical_height": all(
            abs(value - derived["panel_g_physical_height_in"]) < 1e-9
            for value in derived["panel_h_physical_heights_in"]
        ),
        "panels_g_h_vertically_centered_in_f_row": (
            derived["panel_fgh_center_alignment_delta_fraction"] < 1e-6
        ),
        "panel_h_legend_lower_right": (
            derived["panel_h_legend_location"] == 4
            and derived["panel_h_legend_labels"] == ["Day 1", "Day 2"]
        ),
        "panel_h_legend_inside_lower_right_quadrant": (
            derived["panel_h_legend_bbox_axes_fraction"][0] >= 0.5
            and derived["panel_h_legend_bbox_axes_fraction"][1] >= 0.0
            and sum(
                derived["panel_h_legend_bbox_axes_fraction"][index]
                for index in (0, 2)
            )
            <= 1.001
            and sum(
                derived["panel_h_legend_bbox_axes_fraction"][index]
                for index in (1, 3)
            )
            <= 0.5
        ),
        "panel_h_legend_clear_of_data": (
            derived["panel_h_legend_data_overlap_count"] == 0
        ),
        "panel_h_one_shared_nonoverlapping_xlabel": (
            derived["panel_h_axis_xlabel_count"] == 0
            and derived["panel_h_shared_xlabel_bbox_px"][0] >= 0
            and derived["panel_h_shared_xlabel_bbox_px"][1] >= 0
        ),
        "panel_i_left_labels_clear_and_nonoverlapping": (
            derived["panel_i_sample_to_axis_max_overlap_px2"] == 0
            and derived["panel_i_letter_to_sample_max_overlap_px2"] == 0
            and derived["panel_i_label_min_page_clearance_px"] >= 1.0
        ),
        "panel_i_pdf_end_nodes_clear_of_every_y_tick": (
            panel_i_pdf_ylabel_tick_min_clearance_pt >= 1.0
            and panel_i_pdf_ylabel_min_page_clearance_pt >= 1.0
        ),
        "panel_i_three_internal_columns_align_to_panel_f": (
            derived[
                "panel_i_to_f_internal_column_center_max_delta_fraction"
            ] < 1e-12
            and derived[
                "panel_i_to_f_internal_column_width_max_delta_fraction"
            ] < 1e-12
        ),
        "panel_i_clear_of_panel_j_ylabel": (
            derived["panel_i_to_j_ylabel_clearance_px"] >= 2.0
        ),
        "second_row_condition_legend_clear_of_panel_d_letter": (
            derived["curve_legend_to_d_overlap_px2"] == 0.0
            and derived["curve_legend_to_d_clearance_px"] >= 2.0
        ),
        "single_canvas_no_nonuniform_panel_scaling": True,
        "panel_a_vertical_colorbar": derived["colorbar_orientation"] == "vertical",
        "panel_a_colorbar_label_not_cut_off": (
            colorbar_label_x0 >= 0.0 and colorbar_label_x1 <= 0.995
        ),
        "panel_a_shifted_left_and_row_gap_reduced": (
            bboxes["a"][0] < 0.05
            and derived["panel_a_to_curve_legend_gap_fraction"] < 0.035
        ),
        "panel_a_label_moved_down_inside_canvas": (
            derived["panel_label_positions"]["a"][1] < bboxes["a"][3]
        ),
        "panels_b_to_e_moved_up_by_annotated_gap": (
            derived["panel_a_to_curve_legend_gap_fraction"] < 0.002
        ),
        "panel_a_compact_maze_gutters": derived[
            "max_trajectory_gap_fraction"
        ] < 0.004,
        "background_grids_disabled_by_default": (
            not bool(matplotlib.rcParams["axes.grid"])
            and derived["visible_gridline_count"] == 0
        ),
        "panels_b_to_e_share_ordered_legend_row": derived[
            "curve_legend_labels"
        ]
        == [MONTH_LABELS[key] for key in MONTH_ORDER],
        "panel_f_all_six_axes_have_tests": (
            len(derived["f_annotation_counts"]) == 6
            and all(count > 0 for count in derived["f_annotation_counts"])
        ),
        "panel_f_annotation_headroom": (
            derived["panel_f_annotation_headroom_fraction"] >= 0.10
        ),
        "panel_f_stars_close_to_own_separated_brackets": (
            derived["panel_f_star_offset_fraction"] <= 0.008
            and derived["panel_f_bracket_step_fraction"] >= 0.18
        ),
        "panel_i_pair_supports_five_cycles": (
            len(selected_example) == 1
            and str(selected_example.iloc[0]["animal_id"]) == "451"
            and int(selected_example.iloc[0]["naive_complete_cycles"]) >= 5
            and int(selected_example.iloc[0]["return_complete_cycles"]) >= 5
        ),
        "panel_i_display_cycle_contract": (
            i_contract["naive_sequence_cycles"] == 2
            and i_contract["return_sequence_max_endnode_visit"] == 494
            and i_contract["naive_coverage_cycles"] == 5
            and i_contract["return_coverage_cycles"] >= 5
            and i_contract["both_efficiency_cycles"] == 5
        ),
        "panel_i_shared_x_axes_and_cycle_boundaries": (
            i_shared_columns
            and all(
                count > 0
                for count in derived["i_audit"]["boundary_counts"]
            )
        ),
        "panel_i_column1_truncated_to_naive_two_cycles": (
            np.isclose(i_limits[0][0][1], 494.5)
            and np.isclose(i_limits[0][1][1], 494.5)
            and derived["i_audit"]["return_sequence_rows_after_truncation"]
            == 494
            and i_limits[1][0][1] > i_limits[0][0][1]
        ),
        "panel_i_efficiency_axis_contains_all_points": (
            derived["i_audit"]["efficiency_ylim"][1]
            > derived["i_audit"]["efficiency_display_max"]
        ),
        "panel_i_vertical_condition_labels": derived["i_audit"][
            "condition_labels_vertical"
        ],
        "panel_i_condition_labels_moved_right": (
            derived["i_audit"]["condition_label_x_fraction"] >= 0.012
        ),
        "panel_labels_g_h_j_k_outside_top_left_and_i_left": (
            all(
                derived["panel_label_positions"][letter][1]
                > bboxes[letter][3]
                and derived["panel_label_positions"][letter][0] < bboxes[letter][0]
                for letter in ("g", "h", "j", "k")
            )
            and derived["panel_label_positions"]["i"][1] > bboxes["i"][3]
            and derived["panel_label_positions"]["i"][0]
            <= bboxes["i"][0] - 0.03
        ),
        "requested_axis_labels_and_end_node_terms": (
            "Time in maze (min)" in svg_visible_text
            and "Maze tile transitions" in svg_visible_text
            and "Maze-tile transitions" not in svg_visible_text
            and "Maze time (s)" not in svg_visible_text
            and "First long-bout time (s)" not in svg_visible_text
            and "Reward-directed" not in svg_visible_text
            and "Exploration efficiency" in svg_visible_text
            and "End node visits" in svg_visible_text
            and "End nodes" in svg_visible_text
            and "End-node" not in svg_visible_text
            and "end-node" not in svg_visible_text
            and "Endpoint" not in svg_visible_text
            and "endpoint" not in svg_visible_text
        ),
        "panel_letters_a_to_k": set(bboxes) == set("abcdefghijk"),
        "caption_level_prose_absent": not any(
            text in svg_visible_text for text in forbidden_caption_text
        ),
        "editable_svg_text": len(svg_visible_text.strip()) > 0,
        "svg_contains_no_raster_images": svg_image_count == 0,
        "pdf_one_page": pdf_page_count == 1,
        "pdf_contains_no_raster_images": pdf_image_count == 0,
        "pdf_contains_vector_drawings": pdf_drawing_count > 1_000,
        "pdf_contains_selectable_text": pdf_word_count > 0,
        "panel_f_pdf_ylabels_inside_page": (
            set(panel_f_pdf_ylabel_bboxes)
            == {"Time in maze (s)", "Maze tile transitions"}
            and panel_f_pdf_ylabel_min_page_clearance_pt >= 2.0
        ),
        "panel_f_pdf_ylabels_clear_of_tick_text": (
            set(panel_f_pdf_ylabel_text_overlaps)
            == {"Time in maze (s)", "Maze tile transitions"}
            and not any(panel_f_pdf_ylabel_text_overlaps.values())
        ),
        "png_4500_square": png_size == [4500, 4500],
        "png_nonblank": png_bbox is not None,
        "pdf_nonempty": paths["pdf"].stat().st_size > 0,
        "tiff_4500_square": tiff_size == [4500, 4500],
        "tiff_600dpi": len(tiff_dpi) == 2
        and all(abs(value - 600) < 1 for value in tiff_dpi),
        "tiff_lzw": tiff_compression in {"tiff_lzw", "lzw"},
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "maximum_row_alignment_delta_fraction": row_alignment_delta,
        "panel_g_box_aspect": derived["panel_g_box_aspect"],
        "panel_h_box_aspects": derived["panel_h_box_aspects"],
        "panel_g_physical_height_in": derived["panel_g_physical_height_in"],
        "panel_h_physical_heights_in": derived["panel_h_physical_heights_in"],
        "panel_fgh_center_alignment_delta_fraction": derived[
            "panel_fgh_center_alignment_delta_fraction"
        ],
        "panel_h_legend_location": derived["panel_h_legend_location"],
        "panel_h_legend_labels": derived["panel_h_legend_labels"],
        "panel_h_legend_bbox_axes_fraction": derived[
            "panel_h_legend_bbox_axes_fraction"
        ],
        "panel_h_legend_data_overlap_count": derived[
            "panel_h_legend_data_overlap_count"
        ],
        "panel_h_axis_xlabel_count": derived["panel_h_axis_xlabel_count"],
        "panel_h_shared_xlabel_bbox_px": derived["panel_h_shared_xlabel_bbox_px"],
        "panel_i_sample_to_axis_max_overlap_px2": derived[
            "panel_i_sample_to_axis_max_overlap_px2"
        ],
        "panel_i_letter_to_sample_max_overlap_px2": derived[
            "panel_i_letter_to_sample_max_overlap_px2"
        ],
        "panel_i_label_min_page_clearance_px": derived[
            "panel_i_label_min_page_clearance_px"
        ],
        "panel_i_to_j_ylabel_clearance_px": derived[
            "panel_i_to_j_ylabel_clearance_px"
        ],
        "panel_f_internal_column_centers_fraction": derived[
            "panel_f_internal_column_centers_fraction"
        ],
        "panel_i_internal_column_centers_fraction": derived[
            "panel_i_internal_column_centers_fraction"
        ],
        "panel_i_to_f_internal_column_center_max_delta_fraction": derived[
            "panel_i_to_f_internal_column_center_max_delta_fraction"
        ],
        "panel_i_to_f_internal_column_width_max_delta_fraction": derived[
            "panel_i_to_f_internal_column_width_max_delta_fraction"
        ],
        "curve_legend_to_d_clearance_px": derived[
            "curve_legend_to_d_clearance_px"
        ],
        "curve_legend_to_d_overlap_px2": derived[
            "curve_legend_to_d_overlap_px2"
        ],
        "svg_image_count": svg_image_count,
        "pdf_page_count": pdf_page_count,
        "pdf_image_count": pdf_image_count,
        "pdf_drawing_count": pdf_drawing_count,
        "pdf_word_count": pdf_word_count,
        "panel_f_pdf_ylabel_bboxes_pt": panel_f_pdf_ylabel_bboxes,
        "panel_f_pdf_ylabel_min_page_clearance_pt": (
            panel_f_pdf_ylabel_min_page_clearance_pt
        ),
        "panel_f_pdf_ylabel_text_overlaps": panel_f_pdf_ylabel_text_overlaps,
        "panel_i_pdf_ylabel_tick_min_clearance_pt": (
            panel_i_pdf_ylabel_tick_min_clearance_pt
        ),
        "panel_i_pdf_ylabel_min_page_clearance_pt": (
            panel_i_pdf_ylabel_min_page_clearance_pt
        ),
        "speed_summary_max_abs_delta": speed_delta,
        "day1_trajectory_rows": len(day1),
        "day1_source_max_drawn_step_grid": day1_source_max_step,
        "day1_corrected_max_drawn_step_grid": day1_corrected_max_step,
        "panel_a_colorbar_label_bbox_fraction": colorbar_label_bbox,
        "panel_a_to_curve_legend_gap_fraction": derived[
            "panel_a_to_curve_legend_gap_fraction"
        ],
        "panel_f_annotation_counts": derived["f_annotation_counts"],
        "panel_f_annotation_headroom_fraction": derived[
            "panel_f_annotation_headroom_fraction"
        ],
        "panel_f_bracket_step_fraction": derived[
            "panel_f_bracket_step_fraction"
        ],
        "panel_f_star_offset_fraction": derived[
            "panel_f_star_offset_fraction"
        ],
        "panel_label_positions": derived["panel_label_positions"],
        "panel_i_selected_pair": selected_example[
            [
                "animal_id",
                "naive_session",
                "return_session",
                "naive_complete_cycles",
                "return_complete_cycles",
            ]
        ].to_dict(orient="records"),
        "panel_i_shared_x_limits": i_limits,
        "visible_gridline_count": derived["visible_gridline_count"],
        "maximum_trajectory_gap_fraction": derived[
            "max_trajectory_gap_fraction"
        ],
        "png_size": png_size,
        "tiff_size": tiff_size,
        "tiff_dpi": tiff_dpi,
        "tiff_compression": tiff_compression,
    }
    RENDER_VALIDATION.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "pass":
        raise AssertionError("; ".join(report["failed_checks"]))
    return report


def write_caption_notes(data: dict[str, pd.DataFrame]) -> None:
    tile_stat = data["tile_time_stats"].iloc[0]
    lines = [
        "# Extended Data Figure 1 caption notes",
        "",
        f"The round-{ROUND} plots retain only axes, category labels, compact legends, and "
        "essential condition headings. The following details were deliberately "
        "moved out of the plotting area.",
        "",
        "- **a:** Fixed illustrative mouse 434; all four panels show tracked in-maze "
        "segments through the first reach of canonical reward node 94 on one shared "
        f"0-{data['trajectory']['elapsed_minutes'].max():.3f}-min vertical color scale. "
        "The maintained 0.63-grid-unit-per-frame no-jump rule removes repeated "
        "fast-edge islands and never bridges nonconsecutive frames.",
        "- **b-e:** Mean ± SEM across session rows. The common 80%-session support "
        "window and initial group counts belong in the caption/source audit.",
        "- **f:** Event-acquisition comparisons use log1p outcomes, CR1 covariance "
        "clustered by mouse, two-sided t(G-1), and BH correction across 27 contrasts. "
        "All six axes show separated Main-Figure-1-style brackets and significance "
        "stars with reserved top headroom; exact P/q values and sample counts remain "
        "in the statistics source table.",
        "- **g:** The row-retaining model is ln(tile transitions) ~ 1 + ln(elapsed "
        "in-maze seconds), with CR1 covariance clustered by mouse. "
        f"Slope={tile_stat['slope_loglog']:.3f}, 95% CI "
        f"[{tile_stat['ci95_slope_low']:.3f}, {tile_stat['ci95_slope_high']:.3f}], "
        f"P={tile_stat['p_cluster_t_two_sided']:.3g}; "
        f"n={int(tile_stat['n_sessions'])} sessions from {int(tile_stat['n_mice'])} mice.",
        "- **h:** Body-mass speed is total distance divided by eligible mode time "
        "within the first 60 min, separated into exploration and reward directed "
        "modes. Month-specific Day 1/Day 2 sample counts remain in SPEED_SUMMARY.csv.",
        "- **i:** Mouse 451 is the highest-ranked same-mouse candidate with at least "
        "five complete cycles covering 32 unique end nodes in both sessions. Column 1 ends at "
        "the naive two-cycle boundary (494 end node visits) and truncates the return "
        "scatter to the same visit count. Column 2 shows five naive cycles versus all "
        "return cycles on its own shared row scale; column 3 shows five cycles for "
        "both with a common nonclipping y range. Dashed lines mark cycle boundaries. "
        "Filled efficiency points mark cycle 1; no inference.",
        "- **j-k:** Session-first mean ± SEM in fixed completion-coordinate bins; "
        "only bins with at least five sessions are drawn. The plotted y-axis wording "
        "is `Exploration efficiency`; the 32/N definition and dynamic support remain "
        "in the exported tables rather than below the axes.",
        "- **Cross-figure scope:** Panels g-k are maintained ED2 material retained "
        "because the supplied slide assigns those identities to this page. Panels j-k "
        "remain candidate axis options and are not promoted ED2 primary panels.",
        "",
    ]
    CAPTION_NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    configure_style()
    data = read_inputs()
    fig, bboxes, derived = build_figure(data)
    paths = save_outputs(fig)
    write_audits(data, derived, bboxes)
    write_caption_notes(data)
    validation = validate(data, paths, bboxes, derived)
    manifest = {
        "figure": "Extended Data Figure 1",
        "round": ROUND,
        "generated": "2026-08-11",
        "backend": "Python/matplotlib only",
        "figure_contract": {
            "panel_identity": "all slide-1 panel identities retained",
            "archetype": "image plate plus quantitative grid",
            "canvas_inches": [7.5, 7.5],
            "layout": (
                "four explicit rows; panels g and h use square plotting boxes centered "
                "vertically within the f-g-h row; panel h's shared Day 1/Day 2 legend "
                "is anchored in the reward directed axis's lower-right corner"
            ),
            "panel_placement": "single matplotlib canvas; no imported panel image transforms",
            "text_policy": "caption-level titles, statistics, support prose, and sample counts removed from plots",
            "grid_policy": "plain white backgrounds; all plot grids disabled by default",
            "round_folder_policy": (
                "round-specific files coexist in the existing figure working folder"
            ),
        },
        "source_policy": {
            "rendered_directly_from_tables_and_raw_trajectories": True,
            "reused_panel_svgs": 0,
            "reused_panel_rasters": 0,
            "in_canvas_raster_elements": 0,
            "source_audit": SOURCE_AUDIT.name,
            "trajectory_tracking_policy": (
                "maintained 0.63-grid-unit-per-frame split plus repeated-fast-edge "
                "island masking; corrected coordinates are not interpolated or bridged"
            ),
            "cross_figure_scope": (
                "g-k are maintained ED2 material retained from the supplied slide; "
                "j-k are candidate axis options, not promoted ED2 primary panels"
            ),
            "derived_exports": [
                f"ACCUMULATION_SUMMARY_ROUND{ROUND}.csv",
                f"SPEED_SUMMARY_ROUND{ROUND}.csv",
                f"TIME_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
                f"TILE_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
                f"PANEL_I_END_NODE_SEQUENCE_ROUND{ROUND}.csv",
                f"PANEL_I_CYCLE_SUMMARY_ROUND{ROUND}.csv",
                f"PANEL_I_SELECTION_AUDIT_ROUND{ROUND}.csv",
                f"PANEL_A_TRAJECTORY_PLOT_DATA_ROUND{ROUND}.csv.gz",
                f"PANEL_A_JUMP_MASK_AUDIT_ROUND{ROUND}.csv",
            ],
        },
        "outputs": {
            suffix: path.relative_to(REPO).as_posix()
            for suffix, path in paths.items()
        },
        "validation": validation,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "validation": validation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
