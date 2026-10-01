"""Build an unpromoted ED1 candidate with per-group 80% curve endpoints.

Panels b/c and d/e retain all 153 sessions. Cumulative counts are carried
forward after a shorter session ends, matching the maintained time-axis
conversion, and each group stops at the last grid point where at least 80% of
that group's sessions remain directly observed. Panels j/k use the same
group-specific recording-opportunity rule without terminal carry-forward;
their efficiency means remain session-first among sessions completing a cycle
in each retained bin. Panel f uses independent endpoint-specific y scales.
Panel h uses the matched-153 first-30-min source.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RENDERER_PATH = REPO / "scripts/render_extended_data_figure1_round4.py"
TILE_AUDIT_PATH = REPO / "scripts/audit_ed01_panel_d_whole_group.py"
TIME_DIR = (
    REPO
    / "docs/extended_data_figure_panel_refresh_2026-07-12/source_rendered_new"
    / "extended_data_fig_01/month_source_coverage_gap_closure_2026-07-14"
)
TIME_EVENTS = TIME_DIR / "ed01_accumulation_event_rows_current153.csv"
TIME_SESSIONS = TIME_DIR / "ed01_accumulation_session_inclusion_current153.csv"
PANEL_H_DIR = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_panel_h_matched153_window_sensitivity_2026-08-15"
)
OUTPUT = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_group_specific_80_support_round18_2026-08-15"
)
TIME_SESSION_CURVES = OUTPUT / "ed01_bc_group_specific_80_session_curves.csv.gz"
TIME_GROUP_CURVES = OUTPUT / "ed01_bc_group_specific_80_group_curves.csv"
TILE_SESSION_CURVES = OUTPUT / "ed01_de_group_specific_80_session_curves.csv.gz"
TILE_GROUP_CURVES = OUTPUT / "ed01_de_group_specific_80_group_curves.csv"
EFFICIENCY_DIR = (
    REPO
    / "docs/extended_data_figure_panel_refresh_2026-07-12/source_rendered_new"
    / "extended_data_fig_02/exploration_efficiency_axis_options_2026-07-16/source_data"
)
EFFICIENCY_TIME_SOURCE = EFFICIENCY_DIR / "current_plus_new12_time_session_bins.csv"
EFFICIENCY_TILE_SOURCE = EFFICIENCY_DIR / "current_plus_new12_tile_session_bins.csv"
EFFICIENCY_TIME_ROWS = OUTPUT / "ed01_j_group_specific_80_time_session_bins.csv"
EFFICIENCY_TILE_ROWS = OUTPUT / "ed01_k_group_specific_80_tile_session_bins.csv"
SUPPORT_ENDPOINTS = OUTPUT / "ed01_curve_group_specific_80_support_endpoints.csv"
FULL_STEM = OUTPUT / "ed01_group_specific_80_full_page_round18"
CLOSEUP_STEM = OUTPUT / "ed01_panels_b_to_e_group_specific_80_closeup_round18"
ROUND = 18
TIME_STEP_SEC = 10.0
TILE_STEP = 50.0
EVENT_TYPES = ("reward", "long_path_reward")

TIME_GROUP_ORDER = ("0mo", "3mo", "6mo", "9mo", "12mo", "experienced_day2")
TIME_TO_TILE_GROUP = {
    "0mo": "naive_0mo",
    "3mo": "return_3mo",
    "6mo": "return_6mo",
    "9mo": "return_9mo",
    "12mo": "return_12mo",
    "experienced_day2": "experienced_day2",
}
EXPECTED_COUNTS = {
    "0mo": 23,
    "3mo": 17,
    "6mo": 17,
    "9mo": 12,
    "12mo": 10,
    "experienced_day2": 74,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sem(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def maximal_grid_endpoint(durations: np.ndarray, step: float) -> tuple[float, int, int]:
    values = np.asarray(durations, dtype=float)
    values = values[np.isfinite(values)]
    n_total = int(values.size)
    n_required = int(math.ceil(0.8 * n_total))
    candidate = float(np.sort(values)[n_total - n_required])
    endpoint = math.floor((candidate + 1e-9) / step) * step
    n_observed = int(np.sum(values >= endpoint - 1e-9))
    n_next = int(np.sum(values >= endpoint + step - 1e-9))
    if n_observed < n_required or n_next >= n_required:
        raise RuntimeError(
            f"Endpoint is not maximal on the grid: {endpoint}, {n_observed}, {n_next}, {n_required}"
        )
    return float(endpoint), n_required, n_observed


def event_lookup(
    events: pd.DataFrame,
    *,
    coordinate: str,
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str, str], np.ndarray] = {}
    use = events.loc[events["event_type"].isin(EVENT_TYPES)].copy()
    use[coordinate] = pd.to_numeric(use[coordinate], errors="coerce")
    use = use.loc[np.isfinite(use[coordinate])]
    for (session, event_type), rows in use.groupby(["session", "event_type"], sort=False):
        out[(str(session), str(event_type))] = np.sort(rows[coordinate].to_numpy(float))
    return out


def build_time_curves() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    events = pd.read_csv(TIME_EVENTS, dtype={"session": str, "animal_id": str})
    sessions = pd.read_csv(TIME_SESSIONS, dtype={"session": str, "animal_id": str})
    if len(sessions) != 153 or sessions["session"].nunique() != 153:
        raise RuntimeError("Time source is not the exact 153-session registry")
    if not sessions["load_ok"].astype(bool).all():
        raise RuntimeError("Time source contains load failures")
    counts = sessions.groupby("group_key", observed=True).size().astype(int).to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected time group counts: {counts}")

    lookup = event_lookup(events, coordinate="maze_seconds")
    session_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    for group_key in TIME_GROUP_ORDER:
        group = sessions.loc[sessions["group_key"].eq(group_key)].copy()
        durations = pd.to_numeric(group["maze_duration_sec"], errors="raise").to_numpy(float)
        endpoint, n_required, n_observed = maximal_grid_endpoint(durations, TIME_STEP_SEC)
        grid = np.arange(0.0, endpoint + TIME_STEP_SEC / 2.0, TIME_STEP_SEC)
        support_rows.append(
            {
                "axis": "time_in_maze",
                "group_key": group_key,
                "group_label": str(group["group_label"].iloc[0]),
                "n_sessions": len(group),
                "minimum_required_for_80pct": n_required,
                "n_observed_at_endpoint": n_observed,
                "support_fraction_at_endpoint": n_observed / len(group),
                "n_terminal_carried_at_endpoint": len(group) - n_observed,
                "endpoint": endpoint / 60.0,
                "endpoint_unit": "min",
                "grid_step": TIME_STEP_SEC / 60.0,
            }
        )
        for event_type in EVENT_TYPES:
            matrix: list[np.ndarray] = []
            observed_matrix: list[np.ndarray] = []
            for row in group.itertuples(index=False):
                values = lookup.get((str(row.session), event_type), np.asarray([], dtype=float))
                cumulative = np.searchsorted(values, grid, side="right").astype(float)
                observed = grid <= float(row.maze_duration_sec) + 1e-9
                matrix.append(cumulative)
                observed_matrix.append(observed)
                session_frames.append(
                    pd.DataFrame(
                        {
                            "session": str(row.session),
                            "animal_id": str(row.animal_id),
                            "group_key": group_key,
                            "group_label": str(row.group_label),
                            "group_order": int(row.group_order),
                            "month": int(row.month),
                            "memory_stage": str(row.memory_stage),
                            "source_set": str(row.source_set),
                            "event_type": event_type,
                            "maze_seconds": grid,
                            "time_minutes": grid / 60.0,
                            "cum_count": cumulative,
                            "cum_count_active": np.where(observed, cumulative, np.nan),
                            "session_in_window": observed,
                            "terminal_count_carried_forward": ~observed,
                            "display_coverage_ok": True,
                            "group_80pct_endpoint_minutes": endpoint / 60.0,
                        }
                    )
                )
            values = np.asarray(matrix, dtype=float)
            observed_values = np.asarray(observed_matrix, dtype=bool)
            group_frames.append(
                pd.DataFrame(
                    {
                        "event_type": event_type,
                        "group_key": group_key,
                        "group_label": str(group["group_label"].iloc[0]),
                        "group_order": int(group["group_order"].iloc[0]),
                        "time_minutes": grid / 60.0,
                        "mean": values.mean(axis=0),
                        "sem": [sem(values[:, index]) for index in range(values.shape[1])],
                        "n_sessions_averaged": len(group),
                        "n_sessions_observed_at_x": observed_values.sum(axis=0).astype(int),
                        "n_sessions_terminal_carried_at_x": (~observed_values).sum(axis=0).astype(int),
                        "group_80pct_endpoint_minutes": endpoint / 60.0,
                    }
                )
            )
    session_curves = pd.concat(session_frames, ignore_index=True)
    group_curves = pd.concat(group_frames, ignore_index=True)
    session_curves.to_csv(TIME_SESSION_CURVES, index=False, compression="gzip")
    group_curves.to_csv(TIME_GROUP_CURVES, index=False)
    return session_curves, group_curves, support_rows


def build_tile_curves() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    audit = load_module(TILE_AUDIT_PATH, "ed01_tile_audit")
    events, sessions, _, coverage = audit.load_sources()
    lookup = audit.event_lookup(events)
    session_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    for group_key in audit.MONTH_GROUP_ORDER:
        group = sessions.loc[sessions["month_group_key"].eq(group_key)].copy()
        durations = pd.to_numeric(group["nose_tiles_max"], errors="raise").to_numpy(float)
        endpoint, n_required, n_observed = maximal_grid_endpoint(durations, TILE_STEP)
        grid = np.arange(0.0, endpoint + TILE_STEP / 2.0, TILE_STEP)
        source_row = coverage.loc[coverage["month_group_key"].eq(group_key)].iloc[0]
        source_endpoint = math.floor(
            float(source_row["group_coverage_window_tiles"]) / TILE_STEP
        ) * TILE_STEP
        if not np.isclose(endpoint, source_endpoint):
            raise RuntimeError(
                f"Tile endpoint differs from maintained 80% source for {group_key}: "
                f"{endpoint} vs {source_endpoint}"
            )
        support_fraction = n_observed / len(group)
        support_rows.append(
            {
                "axis": "maze_tile_transitions",
                "group_key": group_key,
                "group_label": audit.MONTH_GROUP_LABELS[group_key],
                "n_sessions": len(group),
                "minimum_required_for_80pct": n_required,
                "n_observed_at_endpoint": n_observed,
                "support_fraction_at_endpoint": support_fraction,
                "n_terminal_carried_at_endpoint": len(group) - n_observed,
                "endpoint": endpoint,
                "endpoint_unit": "transitions",
                "grid_step": TILE_STEP,
            }
        )
        for event_type in EVENT_TYPES:
            matrix: list[np.ndarray] = []
            observed_matrix: list[np.ndarray] = []
            for row in group.itertuples(index=False):
                values = lookup.get((str(row.session), event_type), np.asarray([], dtype=float))
                cumulative = np.searchsorted(values, grid, side="right").astype(float)
                observed = grid <= float(row.nose_tiles_max) + 1e-9
                matrix.append(cumulative)
                observed_matrix.append(observed)
                session_frames.append(
                    pd.DataFrame(
                        {
                            "event_type": event_type,
                            "session": str(row.session),
                            "animal_id": str(row.animal_id),
                            "month_group_key": group_key,
                            "month_group_label": audit.MONTH_GROUP_LABELS[group_key],
                            "axis_value_tiles": grid,
                            "cumulative_count": cumulative,
                            "observed_at_x": observed,
                            "terminal_count_carried_forward": ~observed,
                            "group_80pct_endpoint_tiles": endpoint,
                        }
                    )
                )
            values = np.asarray(matrix, dtype=float)
            observed_values = np.asarray(observed_matrix, dtype=bool)
            group_frames.append(
                pd.DataFrame(
                    {
                        "event_type": event_type,
                        "month_group_key": group_key,
                        "month_group_label": audit.MONTH_GROUP_LABELS[group_key],
                        "month_group_order": audit.MONTH_GROUP_ORDER.index(group_key),
                        "color": audit.MONTH_GROUP_COLORS[group_key],
                        "axis_value_tiles": grid,
                        "cumulative_mean": values.mean(axis=0),
                        "cumulative_sem": [sem(values[:, index]) for index in range(values.shape[1])],
                        "n_active_sessions": observed_values.sum(axis=0).astype(int),
                        "initial_n_sessions": len(group),
                        "n_averaged_sessions": len(group),
                        "minimum_coverage_n_sessions": n_required,
                        "coverage_fraction": observed_values.mean(axis=0),
                        "group_80pct_endpoint_tiles": endpoint,
                        "support_fraction_at_endpoint": support_fraction,
                        "support_policy": "group_specific_80pct_terminal_carry_forward",
                    }
                )
            )
    session_curves = pd.concat(session_frames, ignore_index=True)
    group_curves = pd.concat(group_frames, ignore_index=True)
    session_curves.to_csv(TILE_SESSION_CURVES, index=False, compression="gzip")
    group_curves.to_csv(TILE_GROUP_CURVES, index=False)
    return session_curves, group_curves, support_rows


def efficiency_group_from_month(group_key: str) -> str:
    if group_key in {"0mo", "naive_0mo"}:
        return "naive"
    if group_key == "experienced_day2":
        return "experienced_day2"
    return "return"


def build_efficiency_curves() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Restrict j/k by recording opportunity, not by cycle occurrence.

    A session contributes an efficiency datum only when a coverage cycle ends
    inside a bin, so per-bin metric counts are not a monotone censoring measure.
    The 80% endpoint is therefore calculated from recording duration/transition
    reach among sessions eligible for the efficiency analysis. No terminal
    efficiency value is carried after a session ends.
    """

    time_rows = pd.read_csv(
        EFFICIENCY_TIME_SOURCE, dtype={"nickname": str, "animal_id": str}
    )
    tile_rows = pd.read_csv(
        EFFICIENCY_TILE_SOURCE, dtype={"nickname": str, "animal_id": str}
    )
    support_rows: list[dict[str, Any]] = []

    time_sessions = pd.read_csv(
        TIME_SESSIONS, dtype={"session": str, "animal_id": str}
    )
    time_sessions["efficiency_group_key"] = time_sessions["group_key"].map(
        efficiency_group_from_month
    )

    audit = load_module(TILE_AUDIT_PATH, "ed01_efficiency_tile_audit")
    _, tile_sessions, _, _ = audit.load_sources()
    tile_sessions = tile_sessions.copy()
    tile_sessions["session"] = tile_sessions["session"].astype(str)
    tile_sessions["efficiency_group_key"] = tile_sessions["month_group_key"].map(
        efficiency_group_from_month
    )

    axis_specs = (
        (
            "exploration_efficiency_time",
            time_rows,
            time_sessions,
            "maze_duration_sec",
            600.0,
            60.0,
            "min",
        ),
        (
            "exploration_efficiency_transitions",
            tile_rows,
            tile_sessions,
            "nose_tiles_max",
            1000.0,
            1.0,
            "transitions",
        ),
    )
    filtered: dict[str, pd.DataFrame] = {}
    for axis, rows, sessions, duration_column, step, display_scale, unit in axis_specs:
        eligible_ids = set(rows["nickname"].astype(str).unique())
        eligible = sessions.loc[sessions["session"].isin(eligible_ids)].copy()
        missing = eligible_ids - set(eligible["session"])
        if missing:
            raise RuntimeError(
                f"Missing recording support rows for {axis}: {sorted(missing)}"
            )

        kept_frames: list[pd.DataFrame] = []
        for group_key, group_rows in rows.groupby(
            "group_key", observed=True, sort=False
        ):
            group_sessions = eligible.loc[
                eligible["efficiency_group_key"].eq(group_key)
            ].copy()
            if group_sessions["session"].nunique() != group_rows["nickname"].nunique():
                raise RuntimeError(f"Efficiency/session mismatch for {axis}, {group_key}")
            durations = pd.to_numeric(
                group_sessions[duration_column], errors="raise"
            ).to_numpy(float)
            endpoint_raw, n_required, n_observed = maximal_grid_endpoint(
                durations, step
            )
            endpoint = endpoint_raw / display_scale
            kept = group_rows.loc[
                pd.to_numeric(group_rows["axis_bin_end"], errors="raise").le(
                    endpoint + 1e-9
                )
            ].copy()
            if kept.empty:
                raise RuntimeError(f"No efficiency bins retained for {axis}, {group_key}")
            last_center = float(kept["axis_bin_center"].max())
            last_rows = kept.loc[kept["axis_bin_center"].eq(last_center)]
            n_contributors = int(last_rows["nickname"].nunique())
            kept["recording_support_policy"] = (
                "group_specific_80pct_recording_opportunity_no_terminal_carry"
            )
            kept["group_80pct_recording_endpoint"] = endpoint
            kept["group_recording_support_n_at_endpoint"] = n_observed
            kept["group_recording_support_n_total"] = len(group_sessions)
            kept_frames.append(kept)
            support_rows.append(
                {
                    "axis": axis,
                    "group_key": group_key,
                    "group_label": str(group_rows["group_label"].iloc[0]),
                    "n_sessions": len(group_sessions),
                    "minimum_required_for_80pct": n_required,
                    "n_observed_at_endpoint": n_observed,
                    "support_fraction_at_endpoint": n_observed / len(group_sessions),
                    "n_terminal_carried_at_endpoint": 0,
                    "endpoint": endpoint,
                    "endpoint_unit": unit,
                    "grid_step": step / display_scale,
                    "last_plotted_bin_center": last_center,
                    "n_metric_contributors_at_last_bin": n_contributors,
                    "metric_contributor_fraction_at_last_bin": n_contributors
                    / len(group_sessions),
                    "support_basis": "recording opportunity; no terminal carry-forward",
                }
            )
        filtered[axis] = pd.concat(kept_frames, ignore_index=True)

    filtered["exploration_efficiency_time"].to_csv(
        EFFICIENCY_TIME_ROWS, index=False
    )
    filtered["exploration_efficiency_transitions"].to_csv(
        EFFICIENCY_TILE_ROWS, index=False
    )
    return (
        filtered["exploration_efficiency_time"],
        filtered["exploration_efficiency_transitions"],
        support_rows,
    )


def configure_renderer(renderer: Any) -> None:
    renderer.OUTPUT = OUTPUT
    renderer.ROUND = ROUND
    renderer.GENERATED = "2026-08-15"
    renderer.STEM = FULL_STEM
    renderer.SOURCE_AUDIT = OUTPUT / "SOURCE_DATA_AUDIT_ROUND18.csv"
    renderer.LAYOUT_AUDIT = OUTPUT / "LAYOUT_AUDIT_ROUND18.csv"
    renderer.AXIS_SCALE_AUDIT = OUTPUT / "AXIS_SCALE_AUDIT_ROUND18.csv"
    renderer.CAPTION_NOTES = OUTPUT / "CAPTION_NOTES_ROUND18.md"
    renderer.RENDER_VALIDATION = OUTPUT / "RENDER_VALIDATION_ROUND18.json"
    renderer.MANIFEST = OUTPUT / "MANIFEST_ROUND18.json"
    renderer.ACCUMULATION = TIME_SESSION_CURVES
    renderer.TILE_CURVES = TILE_GROUP_CURVES
    renderer.EFFICIENCY_TIME = EFFICIENCY_TIME_ROWS
    renderer.EFFICIENCY_TILE = EFFICIENCY_TILE_ROWS
    renderer.SPEED_ROWS = PANEL_H_DIR / "speed_by_session_window.csv"
    renderer.SPEED_REFERENCE = PANEL_H_DIR / "group_summary.csv"
    renderer.SPEED_WINDOW_ID = "first_30_min"
    renderer.SPEED_SUMMARY_VARIANT = "exact_current_mode_rule"
    renderer.TIME_CURVE_CAPTION_NOTE = (
        "Mean ± SEM across all sessions with terminal cumulative counts carried "
        "after session end; each group stops at its own 80%-session-support endpoint."
    )
    renderer.TILE_CURVE_CAPTION_NOTE = renderer.TIME_CURVE_CAPTION_NOTE
    renderer.EFFICIENCY_CAPTION_NOTE = (
        "Session-first mean ± SEM in fixed completion-coordinate bins. Each "
        "group stops at its own 80%-recording-opportunity endpoint; no terminal "
        "efficiency value is carried forward. Bins still require at least five "
        "sessions completing a coverage cycle."
    )
    renderer.SOURCE_FILES["b-c"] = (
        TIME_SESSION_CURVES,
        "full current153 time curves through each group's own 80% support endpoint",
    )
    renderer.SOURCE_FILES["d-e"] = (
        TILE_GROUP_CURVES,
        "full current153 transition curves through each group's own 80% support endpoint",
    )
    renderer.SOURCE_FILES["h-rows"] = (
        renderer.SPEED_ROWS,
        "matched-153 first-30-min body-mass speed rows",
    )
    renderer.SOURCE_FILES["h-reference"] = (
        renderer.SPEED_REFERENCE,
        "matched-153 first-30-min grouped summary",
    )
    renderer.SOURCE_FILES["j"] = (
        EFFICIENCY_TIME_ROWS,
        "session-first time-bin efficiency rows through group-specific 80% recording support",
    )
    renderer.SOURCE_FILES["k"] = (
        EFFICIENCY_TILE_ROWS,
        "session-first transition-bin efficiency rows through group-specific 80% recording support",
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def draw_closeup(renderer: Any, time_rows: pd.DataFrame, tile_rows: pd.DataFrame) -> None:
    renderer.configure_style()
    time_summary = renderer.accumulation_summary(time_rows)
    tile_rows = tile_rows.rename(
        columns={
            "month_group_key": "group_key",
            "cumulative_mean": "mean",
            "cumulative_sem": "sem",
        }
    )
    tile_rows["group_key"] = tile_rows["group_key"].replace(
        {
            "naive_0mo": "0mo",
            "return_3mo": "3mo",
            "return_6mo": "6mo",
            "return_9mo": "9mo",
            "return_12mo": "12mo",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.4), constrained_layout=False)
    specs = (
        (axes[0, 0], time_summary, "reward", "time_minutes", "mean", "sem", "Time in maze (min)", "Cumulative rewards", "b"),
        (axes[0, 1], time_summary, "long_path_reward", "time_minutes", "mean", "sem", "Time in maze (min)", "Cumulative long bouts", "c"),
        (axes[1, 0], tile_rows, "reward", "axis_value_tiles", "mean", "sem", "Maze tile transitions", "Cumulative rewards", "d"),
        (axes[1, 1], tile_rows, "long_path_reward", "axis_value_tiles", "mean", "sem", "Maze tile transitions", "Cumulative long bouts", "e"),
    )
    for ax, frame, event_type, x, mean, error, xlabel, ylabel, letter in specs:
        renderer.draw_month_curve(
            ax,
            frame,
            event_type=event_type,
            x=x,
            mean=mean,
            error=error,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=False,
        )
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8.0, fontweight="bold")
    handles = [
        Line2D(
            [0],
            [0],
            color=renderer.MONTH_COLORS[key],
            lw=1.0,
            ls="--" if key == "experienced_day2" else "-",
        )
        for key in renderer.MONTH_ORDER
    ]
    fig.legend(
        handles,
        [renderer.MONTH_LABELS[key] for key in renderer.MONTH_ORDER],
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.9,
        handlelength=1.7,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.90, wspace=0.34, hspace=0.36)
    save_figure(fig, CLOSEUP_STEM)
    plt.close(fig)


def export_qa(support: pd.DataFrame) -> dict[str, Any]:
    render_validation = json.loads(
        (OUTPUT / "RENDER_VALIDATION_ROUND18.json").read_text(encoding="utf-8")
    )
    checks: dict[str, bool] = {
        "renderer_pass": render_validation.get("status") == "pass",
        "support_has_18_group_axis_rows": len(support) == 18,
        "all_endpoints_have_at_least_80pct_support": support[
            "support_fraction_at_endpoint"
        ].ge(0.8 - 1e-12).all(),
        "cumulative_axes_represent_all_153_sessions": all(
            int(rows["n_sessions"].sum()) == 153
            for axis, rows in support.groupby("axis", observed=True)
            if axis in {"time_in_maze", "maze_tile_transitions"}
        ),
        "efficiency_axes_represent_all_152_cycle_eligible_sessions": all(
            int(rows["n_sessions"].sum()) == 152
            for axis, rows in support.groupby("axis", observed=True)
            if axis.startswith("exploration_efficiency_")
        ),
        "time_endpoints_are_group_specific": support.loc[
            support["axis"].eq("time_in_maze"), "endpoint"
        ].nunique()
        > 1,
        "tile_endpoints_are_group_specific": support.loc[
            support["axis"].eq("maze_tile_transitions"), "endpoint"
        ].nunique()
        > 1,
        "efficiency_time_endpoints_are_group_specific": support.loc[
            support["axis"].eq("exploration_efficiency_time"), "endpoint"
        ].nunique()
        > 1,
        "efficiency_transition_endpoints_are_group_specific": support.loc[
            support["axis"].eq("exploration_efficiency_transitions"), "endpoint"
        ].nunique()
        > 1,
    }
    export_detail: dict[str, Any] = {}
    for stem in (FULL_STEM, CLOSEUP_STEM):
        with fitz.open(stem.with_suffix(".pdf")) as document:
            pdf_images = sum(len(page.get_images(full=True)) for page in document)
        root = ET.parse(stem.with_suffix(".svg")).getroot()
        svg_images = sum(element.tag.endswith("image") for element in root.iter())
        svg_text = sum(element.tag.endswith("text") for element in root.iter())
        with Image.open(stem.with_suffix(".png")) as image:
            png_size = list(image.size)
            png_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
        with Image.open(stem.with_suffix(".tiff")) as image:
            tiff_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
            tiff_compression = str(image.info.get("compression", "")).lower()
        export_detail[stem.name] = {
            "pdf_images": pdf_images,
            "svg_images": svg_images,
            "svg_text_elements": svg_text,
            "png_size": png_size,
            "png_dpi": png_dpi,
            "tiff_dpi": tiff_dpi,
            "tiff_compression": tiff_compression,
        }
        checks[f"{stem.name}_vector_editable"] = (
            pdf_images == 0 and svg_images == 0 and svg_text > 0
        )
        checks[f"{stem.name}_600dpi_rasters"] = (
            all(abs(value - 600.0) < 1.0 for value in png_dpi + tiff_dpi)
            and tiff_compression == "tiff_lzw"
        )
    checks = {key: bool(value) for key, value in checks.items()}
    result = {
        "schema": "ed01-group-specific-80-support-qa/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "export_detail": export_detail,
    }
    (OUTPUT / "QA_REPORT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"Candidate QA failed: {[key for key, value in checks.items() if not value]}")
    return result


def write_manifest() -> None:
    manifest_path = OUTPUT / "MANIFEST.json"
    rows = []
    for path in sorted(OUTPUT.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.name != manifest_path.name:
            rows.append(
                {
                    "path": path.relative_to(REPO).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ed01-group-specific-80-support-manifest/v1",
                "status": "PASS",
                "accepted_record_modified": False,
                "files": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(support: pd.DataFrame, qa: dict[str, Any]) -> None:
    lines = [
        "# ED1 group-specific 80%-support curve candidate",
        "",
        "This unpromoted candidate restores panel f's endpoint-specific y scales and lets each panel b--e and j-k group curve stop at its own 80%-recording-support endpoint.",
        "",
        "Panels b-e retain all 153 sessions in every cumulative group mean. After a shorter session ends, its terminal cumulative count is carried forward, matching the maintained panel-b/c conversion. Panels j-k contain the 152 sessions with at least one complete coverage cycle; 646-9day2 has no complete cycle and therefore no defined efficiency row. Their 80% endpoint is based on recording opportunity, while the plotted mean at each retained bin remains session-first among sessions completing a cycle in that bin. No efficiency value is carried forward.",
        "",
        "## Endpoints",
        "",
        "| axis | group | endpoint | recording support | metric n at last bin | carried |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in support.itertuples(index=False):
        metric_n = getattr(row, "n_metric_contributors_at_last_bin", np.nan)
        metric_display = "" if pd.isna(metric_n) else str(int(metric_n))
        lines.append(
            f"| {row.axis} | {row.group_label} | {row.endpoint:g} {row.endpoint_unit} | "
            f"{row.n_observed_at_endpoint}/{row.n_sessions} | {metric_display} | "
            f"{row.n_terminal_carried_at_endpoint} |"
        )
    lines.extend(
        [
            "",
            "## QA",
            "",
            f"Status: {qa['status']}. The full page and close-up are editable vector SVG/PDF with 600-dpi PNG/LZW-TIFF previews.",
            "",
            "The accepted ED1 record and manuscript caption were not changed.",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    time_sessions, time_groups, time_support = build_time_curves()
    _, tile_groups, tile_support = build_tile_curves()
    _, _, efficiency_support = build_efficiency_curves()
    support = pd.DataFrame(time_support + tile_support + efficiency_support)
    support.to_csv(SUPPORT_ENDPOINTS, index=False)

    renderer = load_module(RENDERER_PATH, "ed01_group80_renderer")
    configure_renderer(renderer)
    status = renderer.main()
    if status != 0:
        raise RuntimeError("Full ED1 candidate renderer failed")
    draw_closeup(renderer, time_sessions, tile_groups)
    qa = export_qa(support)
    write_report(support, qa)
    write_manifest()
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
