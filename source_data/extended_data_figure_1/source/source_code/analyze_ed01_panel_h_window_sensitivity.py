"""Recompute ED1 panel-h body-mass speed for 30 min and whole sessions.

The accepted panel uses the first 60 minutes of cumulative in-maze time.  This
task-scoped analysis freezes its session cohort and continuous-position speed
estimator, changes only the time window, and leaves the accepted figure record
untouched.  It also retains a reward-context audit because the historical
``MODE_DRINK`` label is operational and is not biologically a reward mode in
no-reward Day-1 sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from audit_continuous_position_speed import (
    MAZE_WIDTH_CM,
    _binned_metrics as speed_binned_metrics,
    _load_raw as load_speed_raw,
    _summarize_session_bins as summarize_speed_session_bins,
)
from ca_analysis.config import FPS, get_behav_wd
from ca_analysis.maze import NewMaze
from ca_analysis.mode_analysis import MODE_DRINK, MODE_EXPLORE
from ca_analysis.trajectory import LoadTraj, SplitModeClips


REPO = Path(__file__).resolve().parents[1]
ACCEPTED_RECORD = (
    REPO
    / "docs"
    / "figure_version_records"
    / "extended_data_fig01_2026-08-15_round17_whole_group_first30_shared_scale"
)
ACCEPTED_SESSION_ROWS = (
    ACCEPTED_RECORD / "plot_ready" / "bodymass_speed_current_plus_new12_by_session.csv"
)
ROUTE_CONTEXT = (
    REPO
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_tables"
    / "ed04_route_events_current_plus_new12.csv"
)
NO_REWARD_CONTEXT = (
    REPO
    / "docs"
    / "reward_port_introduction_no_waterport_2026-05-28"
    / "report"
    / "cohort_manifest.csv"
)
DEFAULT_OUT = REPO / "docs" / "ed01_panel_h_window_sensitivity_2026-08-15"
HISTORICAL_SPEED_ROOT = (
    REPO
    / "docs"
    / "within_session_level_map_correlation_2026-04-29"
    / "tile_speed_diagnostics_raw_trajectory_2026-05-05"
)
HISTORICAL_EXPLORE_BINS = (
    HISTORICAL_SPEED_ROOT
    / "continuous_position_speed_audit_explore_mode_long"
    / "continuous_position_speed_by_bin.csv"
)
HISTORICAL_REWARD_BINS = (
    HISTORICAL_SPEED_ROOT
    / "continuous_position_speed_audit_reward_mode_long"
    / "continuous_position_speed_by_bin.csv"
)

POSITION_SOURCE = "bodymass"
MODES = ("explore", "reward")
MODE_TITLES = {
    "explore": "Exploration",
    "reward": "Reward directed",
}
WINDOWS = (
    ("first_30_min", "First 30 min", 30.0),
    ("first_60_min", "First 60 min", 60.0),
    ("whole_session", "Whole session", None),
)
MONTHS = (0, 3, 6, 9, 12)
MEDIAN_WINDOW = 5
MAX_SEGMENT_SPEED_CM_S = 80.0
BIN_SIZE_MIN = 5
MAZE_ID = 6
AXIS = "#303030"
GROUP_COLORS = {
    "naive": "#78ADD2",
    "return": "#C84B44",
    "experienced_day2": "#5B8C5A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--behavior-root", type=Path, default=None)
    parser.add_argument("--reuse-computed", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def sem(values: Iterable[float]) -> float:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    if array.size <= 1:
        return np.nan
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def safe_divide(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0):
        return np.nan
    return float(numerator / denominator)


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
        }
    )


def load_registry() -> pd.DataFrame:
    accepted = pd.read_csv(ACCEPTED_SESSION_ROWS, dtype={"animal_id": str})
    accepted = accepted.loc[
        accepted["position_source"].astype(str).eq(POSITION_SOURCE)
        & accepted["analysis_mode"].astype(str).isin(MODES)
        & pd.to_numeric(accepted["scope_min"], errors="coerce").eq(60)
        & pd.to_numeric(accepted["median_window"], errors="coerce").eq(MEDIAN_WINDOW)
        & pd.to_numeric(accepted["max_segment_speed_cm_s"], errors="coerce").eq(
            MAX_SEGMENT_SPEED_CM_S
        )
        & pd.to_numeric(accepted["bin_size_min"], errors="coerce").eq(BIN_SIZE_MIN)
    ].copy()
    registry_columns = [
        "session_id",
        "animal_id",
        "day",
        "cohort",
        "cohort_simple",
        "group",
        "month_label",
        "source_set",
    ]
    registry = accepted[registry_columns].drop_duplicates("session_id").copy()
    registry["month"] = pd.to_numeric(
        registry["month_label"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="raise",
    ).astype(int)

    route = pd.read_csv(ROUTE_CONTEXT, dtype={"animal": str})
    route = route[
        ["session_id", "cohort_key", "reward_context"]
    ].drop_duplicates("session_id")
    registry = registry.merge(route, on="session_id", how="left", validate="one_to_one")

    context = pd.read_csv(NO_REWARD_CONTEXT, dtype={"animal_id": str})
    day1_context = context[
        [
            "session_day1",
            "cohort_group",
            "cohort_label",
            "day1_waterport_status",
        ]
    ].rename(columns={"session_day1": "session_id"})
    registry = registry.merge(
        day1_context,
        on="session_id",
        how="left",
        validate="one_to_one",
    )
    registry["reward_context_defined"] = registry["reward_context"].notna()
    registry["reward_context_note"] = np.where(
        registry["reward_context_defined"],
        registry["reward_context"].astype(str),
        np.where(
            registry["day1_waterport_status"].notna(),
            "day1_" + registry["day1_waterport_status"].astype(str),
            "not_resolved_by_reward_context_sources",
        ),
    )
    registry = registry.sort_values(["day", "month", "session_id"]).reset_index(drop=True)
    return registry


def aggregate_window(
    segments: pd.DataFrame,
    *,
    cutoff_min: float | None,
    total_maze_time_sec: float,
) -> dict[str, float | int]:
    time_sec = pd.to_numeric(segments["segment_time_sec"], errors="coerce").to_numpy(float)
    in_scope = np.isfinite(time_sec)
    if cutoff_min is not None:
        in_scope &= time_sec < float(cutoff_min) * 60.0
    eligible = segments["eligible_pair"].to_numpy(bool) & in_scope
    valid = segments["valid_continuous_pair"].to_numpy(bool) & in_scope
    kept = segments["kept_pair"].to_numpy(bool) & in_scope
    dt_sec = pd.to_numeric(segments["dt_sec"], errors="coerce").to_numpy(float)
    displacement = pd.to_numeric(segments["disp_cm"], errors="coerce").to_numpy(float)
    duration_sec = (
        float(total_maze_time_sec)
        if cutoff_min is None
        else min(float(total_maze_time_sec), float(cutoff_min) * 60.0)
    )
    distance_cm = float(np.nansum(displacement[kept])) if np.any(kept) else 0.0
    eligible_time_sec = float(np.nansum(dt_sec[eligible])) if np.any(eligible) else 0.0
    valid_time_sec = float(np.nansum(dt_sec[valid])) if np.any(valid) else 0.0
    observed_time_sec = float(np.nansum(dt_sec[kept])) if np.any(kept) else 0.0
    n_valid = int(np.sum(valid))
    n_kept = int(np.sum(kept))
    return {
        "duration_sec": duration_sec,
        "distance_cm": distance_cm,
        "eligible_time_sec": eligible_time_sec,
        "valid_time_sec": valid_time_sec,
        "observed_time_sec": observed_time_sec,
        "n_valid_continuous_pairs": n_valid,
        "n_kept_pairs": n_kept,
        "kept_pair_fraction": safe_divide(float(n_kept), float(n_valid)),
        "valid_pair_fraction_of_scope": safe_divide(float(n_valid) / float(FPS), duration_sec),
        "valid_pair_fraction_of_mode_time": safe_divide(valid_time_sec, eligible_time_sec),
        "mode_time_distance_rate_cm_s": safe_divide(distance_cm, eligible_time_sec),
        "observed_pair_speed_cm_s": safe_divide(distance_cm, observed_time_sec),
    }


def rolling_median_within_runs_fast(
    values: np.ndarray,
    frames: np.ndarray,
    bout_index: np.ndarray,
    window: int,
) -> np.ndarray:
    """Vectorize run discovery while preserving the maintained rolling median."""
    values = np.asarray(values, dtype=float)
    frames = np.asarray(frames, dtype=float)
    bout_index = np.asarray(bout_index, dtype=int)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(frames) & (bout_index >= 0)
    if not np.any(valid):
        return output
    linked_to_previous = np.zeros(valid.shape, dtype=bool)
    linked_to_previous[1:] = (
        valid[1:]
        & valid[:-1]
        & (bout_index[1:] == bout_index[:-1])
        & ((frames[1:] - frames[:-1]) == 1)
    )
    linked_to_next = np.zeros(valid.shape, dtype=bool)
    linked_to_next[:-1] = linked_to_previous[1:]
    starts = np.flatnonzero(valid & ~linked_to_previous)
    stops = np.flatnonzero(valid & ~linked_to_next) + 1
    if len(starts) != len(stops):
        raise RuntimeError("continuous-run boundary mismatch")
    for start, stop in zip(starts, stops, strict=True):
        smoothed = (
            pd.Series(values[start:stop])
            .rolling(window=int(window), center=True, min_periods=1)
            .median()
            .to_numpy(float)
        )
        smoothed[~np.isfinite(values[start:stop])] = np.nan
        output[start:stop] = smoothed
    return output


def build_segment_tables(
    raw: pd.DataFrame,
    trajectory: Any,
    *,
    maze: Any,
) -> dict[str, pd.DataFrame]:
    """Build both mode tables once using sorted-frame interval assignment.

    This is algebraically identical to the maintained helper but avoids
    rebuilding maze time, smoothed coordinates and trajectory modes separately
    for each requested mode.
    """
    required = {"frame", "mass_x", "mass_y"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing columns for bodymass: {sorted(missing)}")
    frames = pd.to_numeric(raw["frame"], errors="coerce").to_numpy(float)
    finite_frames = frames[np.isfinite(frames)]
    if finite_frames.size and np.any(np.diff(finite_frames) < 0):
        raise ValueError("raw frames must be sorted for exact interval assignment")
    x = pd.to_numeric(raw["mass_x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(raw["mass_y"], errors="coerce").to_numpy(float)

    maze_time = np.full(frames.shape, np.nan, dtype=float)
    bout_index = np.full(frames.shape, -1, dtype=int)
    frame_ranges = np.asarray(trajectory.fr, dtype=float)
    elapsed = 0.0
    for bout, (start, stop) in enumerate(frame_ranges[:, :2]):
        duration = max(0.0, float(stop) - float(start))
        lo = int(np.searchsorted(frames, float(start), side="left"))
        hi = int(np.searchsorted(frames, float(stop), side="left"))
        if hi > lo:
            local = frames[lo:hi]
            valid = np.isfinite(local) & (local >= float(start)) & (local < float(stop))
            indices = np.arange(lo, hi, dtype=int)[valid]
            maze_time[indices] = (elapsed + frames[indices] - float(start)) / float(FPS)
            bout_index[indices] = int(bout)
        elapsed += duration

    frame_mode = np.full(frames.shape, -1, dtype=int)
    clips = np.asarray(SplitModeClips(trajectory, maze), dtype=int)
    node_trajectories = getattr(trajectory, "no", []) or []
    if clips.ndim == 2 and clips.shape[1] >= 4:
        for bout, start, nstep, mode in clips[:, :4]:
            bout = int(bout)
            if bout < 0 or bout >= len(node_trajectories) or bout >= frame_ranges.shape[0]:
                continue
            nodes = np.asarray(node_trajectories[bout])
            if nodes.ndim != 2 or nodes.shape[1] < 2 or nodes.shape[0] == 0:
                continue
            lo_node = int(np.clip(int(start), 0, nodes.shape[0] - 1))
            hi_node = int(
                np.clip(int(start) + max(0, int(nstep)), 0, nodes.shape[0] - 1)
            )
            if hi_node <= lo_node:
                continue
            start_abs = float(frame_ranges[bout, 0]) + float(nodes[lo_node, 1])
            stop_abs = float(frame_ranges[bout, 0]) + float(nodes[hi_node, 1])
            if not (np.isfinite(start_abs) and np.isfinite(stop_abs)) or stop_abs <= start_abs:
                continue
            lo = int(np.searchsorted(frames, start_abs, side="left"))
            hi = int(np.searchsorted(frames, stop_abs, side="left"))
            if hi > lo:
                local = frames[lo:hi]
                valid = np.isfinite(local) & (local >= start_abs) & (local < stop_abs)
                indices = np.arange(lo, hi, dtype=int)[valid]
                frame_mode[indices] = int(mode)

    x = rolling_median_within_runs_fast(x, frames, bout_index, MEDIAN_WINDOW)
    y = rolling_median_within_runs_fast(y, frames, bout_index, MEDIAN_WINDOW)
    if frames.size < 2:
        return {mode: pd.DataFrame() for mode in MODES}
    same_bout = (bout_index[1:] >= 0) & (bout_index[1:] == bout_index[:-1])
    continuous = (
        np.isfinite(frames[1:])
        & np.isfinite(frames[:-1])
        & (np.diff(frames) == 1)
    )
    finite_xy = (
        np.isfinite(x[1:])
        & np.isfinite(x[:-1])
        & np.isfinite(y[1:])
        & np.isfinite(y[:-1])
    )
    finite_time = np.isfinite(maze_time[1:]) & np.isfinite(maze_time[:-1])
    dx_cm = np.diff(x) * MAZE_WIDTH_CM
    dy_cm = np.diff(y) * MAZE_WIDTH_CM
    displacement_cm = np.sqrt(dx_cm * dx_cm + dy_cm * dy_cm)
    dt_sec = np.diff(maze_time)
    segment_speed = displacement_cm / dt_sec
    outputs: dict[str, pd.DataFrame] = {}
    for mode, mode_code in (("explore", MODE_EXPLORE), ("reward", MODE_DRINK)):
        mode_ok = (frame_mode[1:] == int(mode_code)) & (
            frame_mode[:-1] == int(mode_code)
        )
        eligible = same_bout & continuous & finite_time & mode_ok
        valid = eligible & finite_xy
        kept = (
            valid
            & np.isfinite(segment_speed)
            & (dt_sec > 0)
            & (segment_speed <= MAX_SEGMENT_SPEED_CM_S)
        )
        outputs[mode] = pd.DataFrame(
            {
                "segment_time_sec": maze_time[1:],
                "dt_sec": dt_sec,
                "eligible_pair": eligible,
                "valid_continuous_pair": valid,
                "kept_pair": kept,
                "disp_cm": np.where(kept, displacement_cm, np.nan),
                "segment_speed_cm_s": np.where(kept, segment_speed, np.nan),
                "raw_segment_speed_cm_s": np.where(valid, segment_speed, np.nan),
            }
        )
    return outputs


def recompute(
    registry: pd.DataFrame,
    *,
    behavior_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maze = NewMaze(MAZE_ID)
    session_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    selected_root = behavior_root / "selected_data"
    trajectory_root = REPO / "mm-trajectories"

    for index, registry_row in registry.iterrows():
        session_id = str(registry_row["session_id"])
        raw_path = selected_root / f"{session_id}_nose_interp_data.csv"
        trajectory_path = trajectory_root / f"{session_id}-tf"
        try:
            raw = load_speed_raw(session_id, behavior_root)
            trajectory = LoadTraj(session_id + "-tf")
            frame_ranges = np.asarray(trajectory.fr, dtype=float)
            total_maze_time_sec = float(
                np.sum(np.clip(frame_ranges[:, 1] - frame_ranges[:, 0], 0, None)) / float(FPS)
            )
            base = registry_row.to_dict()
            source_rows.append(
                {
                    **base,
                    "raw_rows": int(len(raw)),
                    "trajectory_bouts": int(frame_ranges.shape[0]),
                    "whole_session_maze_time_sec": total_maze_time_sec,
                    "whole_session_maze_time_min": total_maze_time_sec / 60.0,
                    "raw_source_size_bytes": int(raw_path.stat().st_size),
                    "raw_source_sha256": sha256(raw_path),
                    "trajectory_source_size_bytes": int(trajectory_path.stat().st_size),
                    "trajectory_source_sha256": sha256(trajectory_path),
                }
            )
            segment_tables = build_segment_tables(raw, trajectory, maze=maze)
            for mode in MODES:
                segments = segment_tables[mode]
                for window_order, (window_id, window_label, cutoff_min) in enumerate(WINDOWS):
                    row = {
                        **base,
                        "analysis_mode": mode,
                        "position_source": POSITION_SOURCE,
                        "median_window": MEDIAN_WINDOW,
                        "max_segment_speed_cm_s": MAX_SEGMENT_SPEED_CM_S,
                        "window_id": window_id,
                        "window_label": window_label,
                        "window_order": window_order,
                        "scope_min": cutoff_min,
                        "whole_session_maze_time_sec": total_maze_time_sec,
                    }
                    row.update(
                        aggregate_window(
                            segments,
                            cutoff_min=cutoff_min,
                            total_maze_time_sec=total_maze_time_sec,
                        )
                    )
                    session_rows.append(row)

                max_time_min = max(
                    BIN_SIZE_MIN,
                    math.ceil((total_maze_time_sec / 60.0) / BIN_SIZE_MIN) * BIN_SIZE_MIN,
                )
                for metric_row in speed_binned_metrics(
                    segments,
                    bin_size_min=BIN_SIZE_MIN,
                    max_time_min=float(max_time_min),
                ):
                    bin_rows.append(
                        {
                            **base,
                            "analysis_mode": mode,
                            "position_source": POSITION_SOURCE,
                            "median_window": MEDIAN_WINDOW,
                            "max_segment_speed_cm_s": MAX_SEGMENT_SPEED_CM_S,
                            **metric_row,
                        }
                    )
            print(f"[{index + 1:03d}/{len(registry):03d}] {session_id}", flush=True)
        except Exception as exc:  # pragma: no cover - real-data audit path
            failure_rows.append({**registry_row.to_dict(), "error": repr(exc)})
            print(f"FAILED {session_id}: {exc!r}", flush=True)

    failures = pd.DataFrame(failure_rows)
    return (
        pd.DataFrame(session_rows),
        pd.DataFrame(bin_rows),
        pd.DataFrame(source_rows),
        failures,
    )


def build_accepted_vintage_rows(
    *,
    registry: pd.DataFrame,
    raw_session_rows: pd.DataFrame,
    raw_bin_rows: pd.DataFrame,
    source_qc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use preserved current-session conversions plus raw added-12mo rows."""
    current_ids = set(
        registry.loc[registry["source_set"].eq("current_saved"), "session_id"].astype(str)
    )
    new12_ids = set(
        registry.loc[registry["source_set"].eq("new_12mo"), "session_id"].astype(str)
    )
    historical_frames = []
    for path, expected_mode in (
        (HISTORICAL_EXPLORE_BINS, "explore"),
        (HISTORICAL_REWARD_BINS, "reward"),
    ):
        frame = pd.read_csv(path, dtype={"animal_id": str})
        keep = (
            frame["session_id"].astype(str).isin(current_ids)
            & frame["position_source"].astype(str).eq(POSITION_SOURCE)
            & frame["analysis_mode"].astype(str).eq(expected_mode)
            & pd.to_numeric(frame["median_window"], errors="coerce").eq(MEDIAN_WINDOW)
            & pd.to_numeric(frame["max_segment_speed_cm_s"], errors="coerce").eq(
                MAX_SEGMENT_SPEED_CM_S
            )
            & pd.to_numeric(frame["bin_size_min"], errors="coerce").eq(BIN_SIZE_MIN)
        )
        frame = frame.loc[keep].copy()
        frame["source_vintage"] = "preserved_accepted_current_conversion"
        historical_frames.append(frame)
    historical_bins = pd.concat(historical_frames, ignore_index=True, sort=False)
    raw_new12 = raw_bin_rows.loc[
        raw_bin_rows["session_id"].astype(str).isin(new12_ids)
    ].copy()
    raw_new12["source_vintage"] = "raw_added_12mo_recompute"
    primary_bins = pd.concat([historical_bins, raw_new12], ignore_index=True, sort=False)

    expected_pairs = len(registry) * len(MODES)
    observed_pairs = primary_bins[["session_id", "analysis_mode"]].drop_duplicates()
    if len(observed_pairs) != expected_pairs:
        raise RuntimeError(
            f"accepted-vintage bin coverage mismatch: {len(observed_pairs)}/{expected_pairs}"
        )
    summarized = summarize_speed_session_bins(primary_bins, (30, 60, 300))
    extras = registry[
        [
            "session_id",
            "month",
            "source_set",
            "cohort_key",
            "reward_context",
            "cohort_group",
            "cohort_label",
            "day1_waterport_status",
            "reward_context_defined",
            "reward_context_note",
        ]
    ]
    summarized = summarized.merge(extras, on="session_id", how="left", validate="many_to_one")
    duration = source_qc[["session_id", "whole_session_maze_time_sec"]]
    summarized = summarized.merge(duration, on="session_id", how="left", validate="many_to_one")
    window_map = {
        30: ("first_30_min", "First 30 min", 0),
        60: ("first_60_min", "First 60 min", 1),
        300: ("whole_session", "Whole session", 2),
    }
    summarized["window_id"] = summarized["scope_min"].map(
        lambda value: window_map[int(value)][0]
    )
    summarized["window_label"] = summarized["scope_min"].map(
        lambda value: window_map[int(value)][1]
    )
    summarized["window_order"] = summarized["scope_min"].map(
        lambda value: window_map[int(value)][2]
    )
    whole = summarized["window_id"].eq("whole_session")
    summarized.loc[whole, "duration_sec"] = summarized.loc[
        whole, "whole_session_maze_time_sec"
    ]
    summarized.loc[whole, "valid_pair_fraction_of_scope"] = (
        pd.to_numeric(
            summarized.loc[whole, "n_valid_continuous_pairs"], errors="coerce"
        )
        / float(FPS)
        / pd.to_numeric(summarized.loc[whole, "duration_sec"], errors="coerce")
    )
    summarized.loc[whole, "scope_min"] = np.nan
    summarized["source_vintage"] = np.where(
        summarized["source_set"].eq("current_saved"),
        "preserved_accepted_current_conversion",
        "raw_added_12mo_recompute",
    )
    summarized["position_source"] = POSITION_SOURCE
    summarized["median_window"] = MEDIAN_WINDOW
    summarized["max_segment_speed_cm_s"] = MAX_SEGMENT_SPEED_CM_S
    return (
        summarized.sort_values(
            ["day", "month", "session_id", "analysis_mode", "window_order"]
        ).reset_index(drop=True),
        primary_bins.sort_values(
            ["day", "month_label", "session_id", "analysis_mode", "bin_index"]
        ).reset_index(drop=True),
    )


def summarize_groups(session_rows: pd.DataFrame, *, strict_reward_context: bool) -> pd.DataFrame:
    work = session_rows.copy()
    work["included_in_summary"] = np.isfinite(
        pd.to_numeric(work["mode_time_distance_rate_cm_s"], errors="coerce")
    )
    if strict_reward_context:
        undefined = work["analysis_mode"].eq("reward") & ~work[
            "reward_context_defined"
        ].astype(bool)
        work.loc[undefined, "included_in_summary"] = False
    work["summary_variant"] = (
        "reward_context_strict" if strict_reward_context else "exact_current_mode_rule"
    )
    group_cols = [
        "summary_variant",
        "analysis_mode",
        "day",
        "month",
        "month_label",
        "window_id",
        "window_label",
        "window_order",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=False):
        included = group.loc[group["included_in_summary"]].copy()
        speeds = pd.to_numeric(included["mode_time_distance_rate_cm_s"], errors="coerce")
        eligible = pd.to_numeric(included["eligible_time_sec"], errors="coerce")
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_sessions_registry": int(group["session_id"].nunique()),
                "n_sessions": int(included["session_id"].nunique()),
                "n_animals": int(included["animal_id"].nunique()),
                "speed_mean_cm_s": float(speeds.mean()) if len(speeds) else np.nan,
                "speed_sem_cm_s": sem(speeds),
                "eligible_time_sec_mean": float(eligible.mean()) if len(eligible) else np.nan,
                "eligible_time_sec_min": float(eligible.min()) if len(eligible) else np.nan,
                "excluded_undefined_reward_context": int(
                    (
                        group["analysis_mode"].eq("reward")
                        & ~group["reward_context_defined"].astype(bool)
                    ).sum()
                    if strict_reward_context
                    else 0
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["summary_variant", "window_order", "analysis_mode", "day", "month"]
    )


def accepted_parity(session_rows: pd.DataFrame) -> pd.DataFrame:
    accepted = pd.read_csv(ACCEPTED_SESSION_ROWS, dtype={"animal_id": str})
    accepted = accepted.loc[
        accepted["position_source"].astype(str).eq(POSITION_SOURCE)
        & accepted["analysis_mode"].astype(str).isin(MODES)
        & pd.to_numeric(accepted["scope_min"], errors="coerce").eq(60)
        & pd.to_numeric(accepted["median_window"], errors="coerce").eq(MEDIAN_WINDOW)
        & pd.to_numeric(accepted["max_segment_speed_cm_s"], errors="coerce").eq(
            MAX_SEGMENT_SPEED_CM_S
        )
        & pd.to_numeric(accepted["bin_size_min"], errors="coerce").eq(BIN_SIZE_MIN)
    ].copy()
    expected = accepted[
        [
            "session_id",
            "analysis_mode",
            "mode_time_distance_rate_cm_s",
            "eligible_time_sec",
            "distance_cm",
        ]
    ].rename(
        columns={
            "mode_time_distance_rate_cm_s": "accepted_speed_cm_s",
            "eligible_time_sec": "accepted_eligible_time_sec",
            "distance_cm": "accepted_distance_cm",
        }
    )
    observed = session_rows.loc[
        session_rows["window_id"].eq("first_60_min"),
        [
            "session_id",
            "analysis_mode",
            "mode_time_distance_rate_cm_s",
            "eligible_time_sec",
            "distance_cm",
        ],
    ].rename(
        columns={
            "mode_time_distance_rate_cm_s": "recomputed_speed_cm_s",
            "eligible_time_sec": "recomputed_eligible_time_sec",
            "distance_cm": "recomputed_distance_cm",
        }
    )
    parity = expected.merge(
        observed,
        on=["session_id", "analysis_mode"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    for metric in ("speed_cm_s", "eligible_time_sec", "distance_cm"):
        parity[f"abs_delta_{metric}"] = (
            pd.to_numeric(parity[f"accepted_{metric}"], errors="coerce")
            - pd.to_numeric(parity[f"recomputed_{metric}"], errors="coerce")
        ).abs()
    parity["parity_pass_1e_9"] = (
        parity["_merge"].eq("both")
        & parity["abs_delta_speed_cm_s"].fillna(0).le(1e-9)
        & parity["abs_delta_eligible_time_sec"].fillna(0).le(1e-9)
        & parity["abs_delta_distance_cm"].fillna(0).le(1e-9)
    )
    return parity.sort_values(["analysis_mode", "session_id"])


def build_window_comparison(group_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["summary_variant", "analysis_mode", "day", "month", "month_label"]
    value_cols = ["speed_mean_cm_s", "speed_sem_cm_s", "n_sessions", "n_animals"]
    wide = group_summary.pivot(index=keys, columns="window_id", values=value_cols)
    wide.columns = [f"{value}_{window}" for value, window in wide.columns]
    wide = wide.reset_index()
    wide["whole_minus_first30_speed_cm_s"] = (
        wide["speed_mean_cm_s_whole_session"] - wide["speed_mean_cm_s_first_30_min"]
    )
    wide["first30_minus_first60_speed_cm_s"] = (
        wide["speed_mean_cm_s_first_30_min"] - wide["speed_mean_cm_s_first_60_min"]
    )
    wide["whole_minus_first60_speed_cm_s"] = (
        wide["speed_mean_cm_s_whole_session"] - wide["speed_mean_cm_s_first_60_min"]
    )
    return wide.sort_values(keys)


def draw_speed_axis(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    mode: str,
    show_ylabel: bool,
    show_xlabel: bool,
    legend: bool,
) -> None:
    mode_rows = summary.loc[summary["analysis_mode"].eq(mode)].copy()
    day1 = mode_rows.loc[mode_rows["day"].eq("day1")].sort_values("month")
    day2 = mode_rows.loc[mode_rows["day"].eq("day2")].sort_values("month")
    ax.plot(day1["month"], day1["speed_mean_cm_s"], color=AXIS, linewidth=0.9, zorder=1)
    for row in day1.itertuples(index=False):
        color = GROUP_COLORS["naive"] if int(row.month) == 0 else GROUP_COLORS["return"]
        ax.errorbar(
            row.month,
            row.speed_mean_cm_s,
            yerr=row.speed_sem_cm_s,
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=0,
            elinewidth=0.7,
            capsize=1.4,
            zorder=3,
        )
    ax.errorbar(
        day2["month"],
        day2["speed_mean_cm_s"],
        yerr=day2["speed_sem_cm_s"],
        color=GROUP_COLORS["experienced_day2"],
        marker="o",
        markersize=3.5,
        linewidth=0.9,
        elinewidth=0.7,
        capsize=1.4,
        zorder=2,
    )
    ax.set_title(MODE_TITLES[mode], pad=3)
    ax.set_xticks(MONTHS)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)
    ax.set_anchor("C")
    ax.grid(False)
    if show_xlabel:
        ax.set_xlabel("Return interval (months)")
    if show_ylabel:
        ax.set_ylabel("Body-mass speed (cm s$^{-1}$)")
    if legend:
        handles = [
            Line2D([0], [0], color=AXIS, marker="o", markersize=3.2, label="Day 1"),
            Line2D(
                [0],
                [0],
                color=GROUP_COLORS["experienced_day2"],
                marker="o",
                markersize=3.2,
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


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf", ".tiff")]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight")
    fig.savefig(outputs[3], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    return outputs


def render_single_window(
    group_summary: pd.DataFrame,
    *,
    window_id: str,
    variant: str,
    out_dir: Path,
) -> list[Path]:
    summary = group_summary.loc[
        group_summary["window_id"].eq(window_id)
        & group_summary["summary_variant"].eq(variant)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.25), sharey=False)
    for index, mode in enumerate(MODES):
        draw_speed_axis(
            axes[index],
            summary,
            mode=mode,
            show_ylabel=index == 0,
            show_xlabel=True,
            legend=index == 1,
        )
    window_label = summary["window_label"].iloc[0]
    fig.suptitle(f"Body-mass speed by mode: {window_label}", y=0.995, fontsize=8.0)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.84, wspace=0.12)
    paths = save_figure(fig, out_dir / f"ed01_panel_h_{window_id}_{variant}")
    plt.close(fig)
    return paths


def render_comparison(
    group_summary: pd.DataFrame,
    *,
    variant: str,
    out_dir: Path,
) -> list[Path]:
    requested_windows = ("first_30_min", "whole_session")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(4.8, 4.4),
        sharex="col",
        sharey="col",
    )
    for row_index, window_id in enumerate(requested_windows):
        summary = group_summary.loc[
            group_summary["window_id"].eq(window_id)
            & group_summary["summary_variant"].eq(variant)
        ]
        for col_index, mode in enumerate(MODES):
            draw_speed_axis(
                axes[row_index, col_index],
                summary,
                mode=mode,
                show_ylabel=col_index == 0,
                show_xlabel=row_index == 1,
                legend=row_index == 0 and col_index == 1,
            )
        axes[row_index, 0].text(
            -0.43,
            0.5,
            summary["window_label"].iloc[0],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.2,
        )
    fig.suptitle("Body-mass speed during exploration and reward-directed modes", y=0.995, fontsize=8.2)
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.12, top=0.90, wspace=0.12, hspace=0.24)
    paths = save_figure(fig, out_dir / f"ed01_panel_h_first30_vs_whole_{variant}")
    plt.close(fig)
    return paths


def validate_outputs(
    *,
    out_dir: Path,
    registry: pd.DataFrame,
    session_rows: pd.DataFrame,
    source_qc: pd.DataFrame,
    failures: pd.DataFrame,
    parity: pd.DataFrame,
    raw_parity: pd.DataFrame,
    group_summary: pd.DataFrame,
    figure_paths: list[Path],
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    expected_sessions = 120
    expected_session_rows = expected_sessions * len(MODES) * len(WINDOWS)
    check("accepted_registry_120_sessions", len(registry) == expected_sessions, len(registry))
    check("no_recompute_failures", failures.empty, len(failures))
    check(
        "complete_session_mode_window_grid",
        len(session_rows) == expected_session_rows,
        f"{len(session_rows)}/{expected_session_rows}",
    )
    check(
        "one_source_qc_row_per_session",
        len(source_qc) == expected_sessions,
        f"{len(source_qc)}/{expected_sessions}",
    )
    check(
        "exact_whole_session_not_fixed_300min",
        session_rows.loc[session_rows["window_id"].eq("whole_session"), "scope_min"].isna().all(),
        {
            "min_session_min": float(source_qc["whole_session_maze_time_min"].min()),
            "max_session_min": float(source_qc["whole_session_maze_time_min"].max()),
        },
    )
    parity_max = {
        column: float(pd.to_numeric(parity[column], errors="coerce").max())
        for column in [
            "abs_delta_speed_cm_s",
            "abs_delta_eligible_time_sec",
            "abs_delta_distance_cm",
        ]
    }
    check(
        "first60_recompute_matches_accepted_session_rows",
        len(parity) == expected_sessions * len(MODES) and parity["parity_pass_1e_9"].all(),
        parity_max,
    )
    raw_drift = raw_parity.loc[~raw_parity["parity_pass_1e_9"].astype(bool)].copy()
    check(
        "current_raw_source_drift_isolated_and_audited",
        set(raw_drift["session_id"].astype(str)) == {"1616bday1"}
        and set(raw_drift["analysis_mode"].astype(str)) == set(MODES),
        raw_drift[
            ["session_id", "analysis_mode", "abs_delta_speed_cm_s"]
        ].to_dict(orient="records"),
    )
    exact = group_summary.loc[group_summary["summary_variant"].eq("exact_current_mode_rule")]
    check("complete_exact_group_grid", len(exact) == 3 * 2 * 2 * 5, len(exact))
    strict = group_summary.loc[group_summary["summary_variant"].eq("reward_context_strict")]
    excluded = strict["excluded_undefined_reward_context"].sum()
    check("strict_reward_context_exclusions_are_explicit", excluded > 0, int(excluded))
    check(
        "all_figure_exports_exist",
        all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
        [rel(path) for path in figure_paths],
    )
    png_paths = [path for path in figure_paths if path.suffix.lower() == ".png"]
    tiff_paths = [path for path in figure_paths if path.suffix.lower() == ".tiff"]
    pdf_paths = [path for path in figure_paths if path.suffix.lower() == ".pdf"]
    svg_paths = [path for path in figure_paths if path.suffix.lower() == ".svg"]
    png_sizes = {path.name: list(Image.open(path).size) for path in png_paths}
    check("png_previews_have_publication_resolution", all(min(size) >= 1000 for size in png_sizes.values()), png_sizes)
    tiff_info = {}
    for path in tiff_paths:
        with Image.open(path) as image:
            tiff_info[path.name] = {
                "size": list(image.size),
                "dpi": (
                    [float(value) for value in image.info.get("dpi")]
                    if image.info.get("dpi") is not None
                    else None
                ),
                "compression": image.info.get("compression"),
            }
    check(
        "tiff_exports_are_600dpi_lzw",
        all(
            info["dpi"] is not None
            and min(info["dpi"]) >= 599
            and str(info["compression"]).lower() == "tiff_lzw"
            for info in tiff_info.values()
        ),
        tiff_info,
    )
    pdf_images = {}
    for path in pdf_paths:
        with fitz.open(path) as document:
            pdf_images[path.name] = sum(len(page.get_images(full=True)) for page in document)
    check("pdf_exports_contain_no_raster_images", all(value == 0 for value in pdf_images.values()), pdf_images)
    svg_audit = {}
    for path in svg_paths:
        text = path.read_text(encoding="utf-8")
        svg_audit[path.name] = {
            "has_image_element": "<image" in text,
            "has_editable_text": "<text" in text,
        }
    check(
        "svg_exports_have_editable_text_and_no_images",
        all(not row["has_image_element"] and row["has_editable_text"] for row in svg_audit.values()),
        svg_audit,
    )
    passed = all(row["pass"] for row in checks.values())
    validation = {
        "pass": passed,
        "passed": int(sum(row["pass"] for row in checks.values())),
        "total": len(checks),
        "checks": checks,
    }
    (out_dir / "VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def write_report(
    *,
    out_dir: Path,
    group_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    source_qc: pd.DataFrame,
    parity: pd.DataFrame,
    raw_parity: pd.DataFrame,
    validation: dict[str, Any],
) -> None:
    exact = group_summary.loc[
        group_summary["summary_variant"].eq("exact_current_mode_rule")
        & group_summary["window_id"].isin(["first_30_min", "whole_session"])
    ].copy()
    exact = exact.sort_values(["analysis_mode", "window_order", "day", "month"])
    lines = [
        "# ED1 panel-h 30-min and whole-session sensitivity",
        "",
        "## Estimand",
        "",
        "Session-level body-mass speed is total adjacent-frame distance divided by eligible behavioral-mode time. Raw continuous body-mass coordinates are median-smoothed within continuous trajectory runs (5 frames), segments above 80 cm/s are excluded, and cumulative in-maze time selects the first-30-min window. Whole session means all observed cumulative in-maze time for each session; it is not a fixed 300-min cap. Group symbols and error bars are session means +/- SEM.",
        "",
        "## Exact current-mode-rule results",
        "",
        "| mode | window | day | month | mean +/- SEM (cm/s) | n sessions | mean eligible mode time (min) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in exact.itertuples(index=False):
        lines.append(
            f"| {row.analysis_mode} | {row.window_label} | {row.day} | {row.month} | "
            f"{row.speed_mean_cm_s:.3f} +/- {row.speed_sem_cm_s:.3f} | {int(row.n_sessions)} | "
            f"{row.eligible_time_sec_mean / 60.0:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Window-change summary",
            "",
            "Positive values below mean that the whole-session group mean is faster than the first-30-min group mean. These are descriptive changes in the same session cohort; no new significance tests were added.",
            "",
            "| mode | day | month | whole - first30 (cm/s) |",
            "|---|---|---:|---:|",
        ]
    )
    exact_comparison = comparison.loc[
        comparison["summary_variant"].eq("exact_current_mode_rule")
    ].sort_values(["analysis_mode", "day", "month"])
    for row in exact_comparison.itertuples(index=False):
        lines.append(
            f"| {row.analysis_mode} | {row.day} | {row.month} | "
            f"{row.whole_minus_first30_speed_cm_s:+.3f} |"
        )
    undefined = group_summary.loc[
        group_summary["summary_variant"].eq("reward_context_strict")
        & group_summary["analysis_mode"].eq("reward")
        & group_summary["window_id"].isin(["first_30_min", "whole_session"])
        & group_summary["excluded_undefined_reward_context"].gt(0),
        [
            "window_label",
            "day",
            "month",
            "excluded_undefined_reward_context",
            "n_sessions",
        ],
    ].drop_duplicates()
    lines.extend(
        [
            "",
            "## Reward-context caveat",
            "",
            "The accepted panel's historical `reward` row is the operational trajectory `MODE_DRINK` classification. It is not biologically a reward mode in no-reward Day-1 sessions. The exact-current-rule exports retain those operational rows solely for parity with the accepted panel. A separate reward-context-strict summary and figure exclude unresolved/no-reward rows from the reward panel while leaving exploration unchanged.",
            "",
        ]
    )
    if not undefined.empty:
        for row in undefined.itertuples(index=False):
            lines.append(
                f"- {row.window_label}, {row.day}, {int(row.month)} months: excluded {int(row.excluded_undefined_reward_context)} operational reward-mode sessions; strict n={int(row.n_sessions)}."
            )
    parity_max = float(parity["abs_delta_speed_cm_s"].max())
    raw_drift = raw_parity.loc[~raw_parity["parity_pass_1e_9"].astype(bool)]
    lines.extend(
        [
            "",
            "## Alignment and validation",
            "",
            f"- Cohort: 120 accepted ED1 panel-h sessions; behavior-only analysis, so no neural eligibility filter was applied.",
            f"- Timebase: absolute raw frames mapped to cumulative in-maze time through half-open trajectory-bout intervals; only adjacent frames within the same bout contribute displacement.",
            f"- Session duration: {source_qc['whole_session_maze_time_min'].min():.1f}-{source_qc['whole_session_maze_time_min'].max():.1f} cumulative in-maze minutes.",
            f"- First-60-min parity after preserving the accepted source vintage: maximum absolute session-speed difference from the accepted plot-ready rows was {parity_max:.3g} cm/s.",
            f"- Current-local-source drift audit: {len(raw_drift)} mode rows, all from `1616bday1`, differed from the accepted source vintage; these rows are exported separately and were not allowed to redefine the same-plot sensitivity.",
            f"- Validation: {validation['passed']}/{validation['total']} checks passed.",
            "- The accepted ED1 figure record and manuscript caption were not modified.",
            "",
            "## Outputs",
            "",
            "- `speed_by_session_window.csv`: statistical-unit values and support/QC fields.",
            "- `speed_by_5min_bin.csv`: intermediate distance, time and mask-support conversions.",
            "- `session_source_qc.csv`: session coverage plus raw/trajectory source hashes.",
            "- `group_summary.csv`: exact-current-rule and reward-context-strict means/SEM.",
            "- `window_comparison.csv`: first-30, first-60 and whole-session group differences.",
            "- `accepted_first60_parity.csv`: session-level reproduction of the accepted estimator.",
            "- `current_raw_first60_drift.csv`: isolated current-local-source drift relative to the accepted panel.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_existing(out_dir: Path) -> int:
    manifest_path = out_dir / "MANIFEST.json"
    validation_path = out_dir / "VALIDATION.json"
    if not (manifest_path.is_file() and validation_path.is_file()):
        raise FileNotFoundError("Run the analysis before --validate-only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    hash_failures = []
    for item in manifest["output_files"]:
        path = REPO / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            hash_failures.append(item["path"])
    print(json.dumps({"validation": validation, "hash_failures": hash_failures}, indent=2))
    return 0 if validation.get("pass") and not hash_failures else 1


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if args.validate_only:
        return validate_existing(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_root = Path(args.behavior_root or get_behav_wd()).resolve()
    configure_style()
    if args.reuse_computed:
        registry = pd.read_csv(out_dir / "session_registry.csv", dtype={"animal_id": str})
        raw_session_path = out_dir / "speed_by_session_window_current_raw.csv"
        if not raw_session_path.is_file():
            raw_session_path = out_dir / "speed_by_session_window.csv"
        raw_bin_path = out_dir / "speed_by_5min_bin_current_raw.csv"
        if not raw_bin_path.is_file():
            raw_bin_path = out_dir / "speed_by_5min_bin.csv"
        raw_session_rows = pd.read_csv(
            raw_session_path, dtype={"animal_id": str}
        )
        raw_bin_rows = pd.read_csv(
            raw_bin_path, dtype={"animal_id": str}
        )
        source_qc = pd.read_csv(out_dir / "session_source_qc.csv", dtype={"animal_id": str})
        failures_path = out_dir / "failures.csv"
        try:
            failures = pd.read_csv(failures_path)
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame()
    else:
        registry = load_registry()
        raw_session_rows, raw_bin_rows, source_qc, failures = recompute(
            registry,
            behavior_root=behavior_root,
        )
        registry.to_csv(out_dir / "session_registry.csv", index=False)
        source_qc.to_csv(out_dir / "session_source_qc.csv", index=False)
        failures.to_csv(out_dir / "failures.csv", index=False)

    raw_session_rows.to_csv(
        out_dir / "speed_by_session_window_current_raw.csv", index=False
    )
    raw_bin_rows.to_csv(out_dir / "speed_by_5min_bin_current_raw.csv", index=False)
    raw_parity = accepted_parity(raw_session_rows)
    raw_parity.loc[~raw_parity["parity_pass_1e_9"].astype(bool)].to_csv(
        out_dir / "current_raw_first60_drift.csv", index=False
    )
    session_rows, bin_rows = build_accepted_vintage_rows(
        registry=registry,
        raw_session_rows=raw_session_rows,
        raw_bin_rows=raw_bin_rows,
        source_qc=source_qc,
    )
    parity = accepted_parity(session_rows)
    exact_summary = summarize_groups(session_rows, strict_reward_context=False)
    strict_summary = summarize_groups(session_rows, strict_reward_context=True)
    group_summary = pd.concat([exact_summary, strict_summary], ignore_index=True)
    comparison = build_window_comparison(group_summary)

    session_rows.to_csv(out_dir / "speed_by_session_window.csv", index=False)
    bin_rows.to_csv(out_dir / "speed_by_5min_bin.csv", index=False)
    parity.to_csv(out_dir / "accepted_first60_parity.csv", index=False)
    group_summary.to_csv(out_dir / "group_summary.csv", index=False)
    comparison.to_csv(out_dir / "window_comparison.csv", index=False)

    figure_paths: list[Path] = []
    for variant in ("exact_current_mode_rule", "reward_context_strict"):
        figure_paths.extend(
            render_single_window(
                group_summary,
                window_id="first_30_min",
                variant=variant,
                out_dir=out_dir,
            )
        )
        figure_paths.extend(
            render_single_window(
                group_summary,
                window_id="whole_session",
                variant=variant,
                out_dir=out_dir,
            )
        )
        figure_paths.extend(render_comparison(group_summary, variant=variant, out_dir=out_dir))

    validation = validate_outputs(
        out_dir=out_dir,
        registry=registry,
        session_rows=session_rows,
        source_qc=source_qc,
        failures=failures,
        parity=parity,
        raw_parity=raw_parity,
        group_summary=group_summary,
        figure_paths=figure_paths,
    )
    write_report(
        out_dir=out_dir,
        group_summary=group_summary,
        comparison=comparison,
        source_qc=source_qc,
        parity=parity,
        raw_parity=raw_parity,
        validation=validation,
    )

    output_paths = [
        path
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name not in {"MANIFEST.json"}
    ]
    manifest = {
        "analysis": "ED1 panel-h first-30-min and exact whole-session sensitivity",
        "generated": "2026-08-15",
        "backend": "Python/matplotlib only",
        "accepted_record_unchanged": True,
        "figure_contract": {
            "core_conclusion": "test whether the accepted 60-min month-by-day speed pattern depends on the time window",
            "statistical_unit": "session",
            "representation": "continuous normalized body-mass position converted to centimeters",
            "timebase": "cumulative in-maze time from trajectory bout frame intervals",
            "windows": ["first 30 min", "first 60 min parity", "all observed in-maze time"],
            "estimator": "total retained adjacent-frame distance divided by eligible mode time",
            "error_bars": "SEM across session-level estimates",
            "inference": "descriptive; no new significance tests",
            "source_vintage_policy": "preserved accepted conversions for 114 current sessions; raw recompute for six added 12-month sessions",
        },
        "fixed_settings": {
            "position_source": POSITION_SOURCE,
            "modes": list(MODES),
            "median_window_frames": MEDIAN_WINDOW,
            "max_segment_speed_cm_s": MAX_SEGMENT_SPEED_CM_S,
            "bin_size_min": BIN_SIZE_MIN,
            "maze_id": MAZE_ID,
            "fps": FPS,
        },
        "source_files": [
            {"path": rel(ACCEPTED_SESSION_ROWS), "sha256": sha256(ACCEPTED_SESSION_ROWS)},
            {"path": rel(ROUTE_CONTEXT), "sha256": sha256(ROUTE_CONTEXT)},
            {"path": rel(NO_REWARD_CONTEXT), "sha256": sha256(NO_REWARD_CONTEXT)},
            {"path": rel(HISTORICAL_EXPLORE_BINS), "sha256": sha256(HISTORICAL_EXPLORE_BINS)},
            {"path": rel(HISTORICAL_REWARD_BINS), "sha256": sha256(HISTORICAL_REWARD_BINS)},
        ],
        "validation": validation,
        "output_files": [
            {"path": rel(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in output_paths
        ],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": rel(out_dir), "validation": validation}, indent=2))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
