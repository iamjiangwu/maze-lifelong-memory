"""Audit and rebuild ED1 reward accumulation with the complete session cohort.

The accepted ED1 panel d uses a fixed subset of sessions that reach the common
80%-support endpoint on the corrected-nose maze-tile transition axis.  This
script freezes that estimator, reconstructs it for parity, and compares it with
two whole-group alternatives:

1. observed_common_support (primary candidate): retain every session and stop
   at the largest transition count observed in every session;
2. terminal_carry_forward (sensitivity): retain every session over the current
   11,050-transition range and carry the terminal cumulative count forward
   after a session ends, matching the time-axis panel's aggregation convention.

No accepted figure record, caption, or maintained source table is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_audit_2026-06-30"
    / "source_rendered_individual_panels"
)
EVENT_SOURCE = SOURCE_ROOT / "reward_accumulation_progress_axis_event_coordinates.csv"
SESSION_SOURCE = SOURCE_ROOT / "reward_accumulation_progress_axis_session_summary.csv"
CURRENT_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_01"
    / "acquisition_endpoint_axes_2026-07-16"
)
CURRENT_CURVES = (
    CURRENT_ROOT / "ed01_tile_accumulation_by_return_month_group_curves_current153.csv"
)
CURRENT_COVERAGE = (
    CURRENT_ROOT / "ed01_tile_accumulation_by_return_month_coverage_current153.csv"
)
DEFAULT_OUT = REPO_ROOT / "docs" / "ed01_panel_d_whole_group_2026-08-15"

MONTH_GROUP_ORDER = (
    "naive_0mo",
    "return_3mo",
    "return_6mo",
    "return_9mo",
    "return_12mo",
    "experienced_day2",
)
MONTH_GROUP_LABELS = {
    "naive_0mo": "Naive day",
    "return_3mo": "3 mo return",
    "return_6mo": "6 mo return",
    "return_9mo": "9 mo return",
    "return_12mo": "12 mo return",
    "experienced_day2": "Second day",
}
MONTH_GROUP_COLORS = {
    "naive_0mo": "#78ADD2",
    "return_3mo": "#F2B3AE",
    "return_6mo": "#E77D72",
    "return_9mo": "#C84B44",
    "return_12mo": "#8E2C2A",
    "experienced_day2": "#5B8C5A",
}
EXPECTED_GROUP_COUNTS = {
    "naive_0mo": 23,
    "return_3mo": 17,
    "return_6mo": 17,
    "return_9mo": 12,
    "return_12mo": 10,
    "experienced_day2": 74,
}
EVENT_TYPES = ("reward", "long_path_reward")
STEP_TILES = 50.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sem(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 0.0
    return float(np.std(finite, ddof=1) / np.sqrt(finite.size))


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def add_month_groups(sessions: pd.DataFrame) -> pd.DataFrame:
    out = sessions.copy()
    keys: list[str] = []
    for row in out.itertuples(index=False):
        group_key = str(row.group_key)
        if group_key == "naive":
            key = "naive_0mo"
        elif group_key == "return":
            key = f"return_{int(row.month)}mo"
        elif group_key == "experienced_day2":
            key = "experienced_day2"
        else:
            raise ValueError(f"Unexpected source group: {group_key}")
        keys.append(key)
    out["month_group_key"] = keys
    out["month_group_label"] = out["month_group_key"].map(MONTH_GROUP_LABELS)
    out["month_group_order"] = out["month_group_key"].map(
        {key: index for index, key in enumerate(MONTH_GROUP_ORDER)}
    )
    return out


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(EVENT_SOURCE, dtype={"session": str, "animal_id": str})
    sessions = pd.read_csv(SESSION_SOURCE, dtype={"session": str, "animal_id": str})
    current = pd.read_csv(CURRENT_CURVES)
    coverage = pd.read_csv(CURRENT_COVERAGE)

    if len(sessions) != 153 or sessions["session"].nunique() != 153:
        raise AssertionError("The ED1 session source is no longer current153")
    if not sessions["load_ok"].astype(bool).all():
        failed = sessions.loc[~sessions["load_ok"].astype(bool), ["session", "error"]]
        raise AssertionError(f"Source contains failed sessions:\n{failed}")
    sessions = add_month_groups(sessions)
    counts = sessions.groupby("month_group_key", observed=True).size().to_dict()
    if counts != EXPECTED_GROUP_COUNTS:
        raise AssertionError(f"Unexpected month-group counts: {counts}")
    if set(events["session"].astype(str)).difference(sessions["session"].astype(str)):
        raise AssertionError("Event source contains sessions absent from the session source")
    return events, sessions, current, coverage


def event_lookup(events: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    keep = events.loc[events["event_type"].astype(str).isin(EVENT_TYPES)].copy()
    keep["nose_tiles_at_event"] = pd.to_numeric(
        keep["nose_tiles_at_event"], errors="coerce"
    )
    keep = keep.loc[np.isfinite(keep["nose_tiles_at_event"])].copy()
    for (session, event_type), sub in keep.groupby(
        ["session", "event_type"], sort=False
    ):
        values = np.sort(sub["nose_tiles_at_event"].to_numpy(float))
        lookup[(str(session), str(event_type))] = values
    return lookup


def curve_grid(endpoint: float) -> np.ndarray:
    return np.arange(0.0, math.floor(endpoint / STEP_TILES) * STEP_TILES + STEP_TILES, STEP_TILES)


def build_curves(
    sessions: pd.DataFrame,
    lookup: dict[tuple[str, str], np.ndarray],
    *,
    estimator: str,
    endpoint: float,
    include_mask: pd.Series,
    carry_forward: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = curve_grid(endpoint)
    included = sessions.loc[include_mask].copy()
    session_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []

    for event_type in EVENT_TYPES:
        matrices: dict[str, list[np.ndarray]] = {key: [] for key in MONTH_GROUP_ORDER}
        observed_matrices: dict[str, list[np.ndarray]] = {
            key: [] for key in MONTH_GROUP_ORDER
        }
        for row in included.itertuples(index=False):
            session = str(row.session)
            values = lookup.get((session, event_type), np.asarray([], dtype=float))
            counts = np.searchsorted(values, grid, side="right").astype(float)
            observed = grid <= float(row.nose_tiles_max) + 1e-9
            if not carry_forward:
                counts[~observed] = np.nan
            matrices[str(row.month_group_key)].append(counts)
            observed_matrices[str(row.month_group_key)].append(observed.astype(float))
            session_frames.append(
                pd.DataFrame(
                    {
                        "estimator": estimator,
                        "event_type": event_type,
                        "session": session,
                        "animal_id": str(row.animal_id),
                        "month_group_key": str(row.month_group_key),
                        "month_group_label": str(row.month_group_label),
                        "axis_value_tiles": grid,
                        "cumulative_count": counts,
                        "observed_at_x": observed,
                        "terminal_count_carried_forward": (~observed) & carry_forward,
                    }
                )
            )

        for group_key in MONTH_GROUP_ORDER:
            matrix = np.asarray(matrices[group_key], dtype=float)
            observed_matrix = np.asarray(observed_matrices[group_key], dtype=float)
            if matrix.size == 0:
                raise AssertionError(f"No sessions for {estimator}/{group_key}")
            means = np.nanmean(matrix, axis=0)
            sems = np.asarray([sem(matrix[:, index]) for index in range(grid.size)])
            observed_n = np.sum(observed_matrix, axis=0).astype(int)
            group_frames.append(
                pd.DataFrame(
                    {
                        "estimator": estimator,
                        "event_type": event_type,
                        "month_group_key": group_key,
                        "month_group_label": MONTH_GROUP_LABELS[group_key],
                        "month_group_order": MONTH_GROUP_ORDER.index(group_key),
                        "color": MONTH_GROUP_COLORS[group_key],
                        "axis_value_tiles": grid,
                        "cumulative_mean": means,
                        "cumulative_sem": sems,
                        "n_sessions_averaged": matrix.shape[0],
                        "n_sessions_observed_at_x": observed_n,
                        "n_sessions_carried_at_x": matrix.shape[0] - observed_n,
                        "global_window_tiles": float(grid[-1]),
                    }
                )
            )
    return pd.concat(group_frames, ignore_index=True), pd.concat(
        session_frames, ignore_index=True
    )


def reconstruct_current(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    current: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = event_lookup(events)
    current = current.rename(columns={"n_averaged_sessions": "n_sessions_averaged"})
    endpoints = pd.to_numeric(
        coverage["global_coverage_window_tiles"], errors="raise"
    ).unique()
    if len(endpoints) != 1:
        raise AssertionError(f"Current coverage has multiple global endpoints: {endpoints}")
    current_endpoint = float(endpoints[0])
    fixed_mask = pd.to_numeric(sessions["nose_tiles_max"], errors="coerce").ge(
        current_endpoint - 1e-9
    )
    rebuilt, rebuilt_sessions = build_curves(
        sessions,
        lookup,
        estimator="current_fixed_80pct_subset",
        endpoint=current_endpoint,
        include_mask=fixed_mask,
        carry_forward=False,
    )

    compare_columns = [
        "event_type",
        "month_group_key",
        "axis_value_tiles",
        "cumulative_mean",
        "cumulative_sem",
        "n_sessions_averaged",
    ]
    parity = current[compare_columns].merge(
        rebuilt[compare_columns],
        on=["event_type", "month_group_key", "axis_value_tiles"],
        how="outer",
        suffixes=("_source", "_rebuilt"),
        indicator=True,
    )
    parity["abs_mean_delta"] = (
        parity["cumulative_mean_source"] - parity["cumulative_mean_rebuilt"]
    ).abs()
    parity["abs_sem_delta"] = (
        parity["cumulative_sem_source"] - parity["cumulative_sem_rebuilt"]
    ).abs()
    parity["n_match"] = parity["n_sessions_averaged_source"].eq(
        parity["n_sessions_averaged_rebuilt"]
    )
    parity["passes"] = (
        parity["_merge"].eq("both")
        & parity["abs_mean_delta"].le(1e-12)
        & parity["abs_sem_delta"].le(1e-12)
        & parity["n_match"]
    )
    if not parity["passes"].all():
        bad = parity.loc[~parity["passes"]].head(20)
        raise AssertionError(f"Current estimator reconstruction failed parity:\n{bad}")
    return rebuilt, rebuilt_sessions, parity


def build_all_estimators(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    current_rebuilt: pd.DataFrame,
    current_sessions: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    lookup = event_lookup(events)
    current_endpoint = float(coverage["global_coverage_window_tiles"].iloc[0])
    minimum_observed_max = float(
        pd.to_numeric(sessions["nose_tiles_max"], errors="raise").min()
    )
    observed_endpoint = float(curve_grid(minimum_observed_max)[-1])
    include_all = pd.Series(True, index=sessions.index)

    observed_curves, observed_sessions = build_curves(
        sessions,
        lookup,
        estimator="whole_group_observed_common_support",
        endpoint=observed_endpoint,
        include_mask=include_all,
        carry_forward=False,
    )
    carry_curves, carry_sessions = build_curves(
        sessions,
        lookup,
        estimator="whole_group_terminal_carry_forward",
        endpoint=current_endpoint,
        include_mask=include_all,
        carry_forward=True,
    )
    dynamic_curves, dynamic_sessions = build_curves(
        sessions,
        lookup,
        estimator="whole_group_dynamic_available_case",
        endpoint=current_endpoint,
        include_mask=include_all,
        carry_forward=False,
    )
    group_curves = pd.concat(
        [current_rebuilt, observed_curves, carry_curves, dynamic_curves],
        ignore_index=True,
        sort=False,
    )
    session_curves = pd.concat(
        [current_sessions, observed_sessions, carry_sessions, dynamic_sessions],
        ignore_index=True,
        sort=False,
    )
    endpoints = {
        "current_fixed_endpoint_tiles": current_endpoint,
        "whole_group_minimum_observed_nose_tiles_max": minimum_observed_max,
        "whole_group_observed_endpoint_tiles": observed_endpoint,
    }
    return group_curves, session_curves, endpoints


def build_inclusion_audit(
    sessions: pd.DataFrame, coverage: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_endpoint = float(coverage["global_coverage_window_tiles"].iloc[0])
    audit = sessions.copy()
    audit["nose_tiles_max"] = pd.to_numeric(audit["nose_tiles_max"], errors="raise")
    audit["current_fixed_subset_included"] = audit["nose_tiles_max"].ge(
        current_endpoint - 1e-9
    )
    audit["whole_group_included"] = True
    audit["shortfall_to_current_endpoint"] = np.maximum(
        0.0, current_endpoint - audit["nose_tiles_max"]
    )
    audit["current_drop_reason"] = np.where(
        audit["current_fixed_subset_included"],
        "included",
        "nose_tiles_max below fixed 80pct support endpoint",
    )
    keep_columns = [
        "session",
        "animal_id",
        "month_group_key",
        "month_group_label",
        "source_set",
        "load_ok",
        "nose_tiles_max",
        "n_reward_events",
        "n_long_path_reward_events",
        "current_fixed_subset_included",
        "whole_group_included",
        "shortfall_to_current_endpoint",
        "current_drop_reason",
    ]
    audit = audit[keep_columns].sort_values(
        ["month_group_key", "current_fixed_subset_included", "nose_tiles_max"],
        kind="mergesort",
    )

    support_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for group_key in MONTH_GROUP_ORDER:
        sub = audit.loc[audit["month_group_key"].eq(group_key)]
        retained = sub.loc[sub["current_fixed_subset_included"]]
        dropped = sub.loc[~sub["current_fixed_subset_included"]]
        source_coverage = coverage.loc[coverage["month_group_key"].eq(group_key)].iloc[0]
        support_rows.append(
            {
                "month_group_key": group_key,
                "month_group_label": MONTH_GROUP_LABELS[group_key],
                "whole_group_n": len(sub),
                "current_fixed_subset_n": len(retained),
                "dropped_n": len(dropped),
                "retained_fraction": len(retained) / len(sub),
                "minimum_required_for_80pct": int(
                    source_coverage["minimum_coverage_n_sessions"]
                ),
                "group_80pct_window_tiles": float(
                    source_coverage["group_coverage_window_tiles"]
                ),
                "global_current_window_tiles": float(
                    source_coverage["global_coverage_window_tiles"]
                ),
                "minimum_nose_tiles_max": float(sub["nose_tiles_max"].min()),
                "median_nose_tiles_max": float(sub["nose_tiles_max"].median()),
            }
        )
        for status, frame in (("retained", retained), ("dropped", dropped)):
            selection_rows.append(
                {
                    "month_group_key": group_key,
                    "status": status,
                    "n_sessions": len(frame),
                    "mean_nose_tiles_max": float(frame["nose_tiles_max"].mean())
                    if len(frame)
                    else np.nan,
                    "mean_total_rewards": float(frame["n_reward_events"].mean())
                    if len(frame)
                    else np.nan,
                    "mean_total_long_bouts": float(
                        frame["n_long_path_reward_events"].mean()
                    )
                    if len(frame)
                    else np.nan,
                }
            )
    return audit, pd.DataFrame(support_rows), pd.DataFrame(selection_rows)


def build_estimator_comparison(
    curves: pd.DataFrame, endpoints: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    curve_differences: list[dict[str, Any]] = []
    specs = (
        (
            "whole_observed_minus_current_fixed_on_observed_range",
            "whole_group_observed_common_support",
            "current_fixed_80pct_subset",
            endpoints["whole_group_observed_endpoint_tiles"],
        ),
        (
            "whole_carry_minus_current_fixed_on_current_range",
            "whole_group_terminal_carry_forward",
            "current_fixed_80pct_subset",
            endpoints["current_fixed_endpoint_tiles"],
        ),
    )
    for event_type in EVENT_TYPES:
        for comparison, target_name, reference_name, endpoint in specs:
            for group_key in MONTH_GROUP_ORDER:
                target = curves.loc[
                    curves["estimator"].eq(target_name)
                    & curves["event_type"].eq(event_type)
                    & curves["month_group_key"].eq(group_key)
                    & curves["axis_value_tiles"].le(endpoint + 1e-9),
                    ["axis_value_tiles", "cumulative_mean"],
                ].rename(columns={"cumulative_mean": "target_mean"})
                reference = curves.loc[
                    curves["estimator"].eq(reference_name)
                    & curves["event_type"].eq(event_type)
                    & curves["month_group_key"].eq(group_key)
                    & curves["axis_value_tiles"].le(endpoint + 1e-9),
                    ["axis_value_tiles", "cumulative_mean"],
                ].rename(columns={"cumulative_mean": "reference_mean"})
                merged = target.merge(reference, on="axis_value_tiles", validate="one_to_one")
                merged["delta_target_minus_reference"] = (
                    merged["target_mean"] - merged["reference_mean"]
                )
                for row in merged.itertuples(index=False):
                    curve_differences.append(
                        {
                            "comparison": comparison,
                            "event_type": event_type,
                            "month_group_key": group_key,
                            "axis_value_tiles": float(row.axis_value_tiles),
                            "target_mean": float(row.target_mean),
                            "reference_mean": float(row.reference_mean),
                            "delta_target_minus_reference": float(
                                row.delta_target_minus_reference
                            ),
                        }
                    )
                endpoint_row = merged.iloc[-1]
                comparisons.append(
                    {
                        "comparison": comparison,
                        "event_type": event_type,
                        "month_group_key": group_key,
                        "endpoint_tiles": float(endpoint_row["axis_value_tiles"]),
                        "target_endpoint_mean": float(endpoint_row["target_mean"]),
                        "reference_endpoint_mean": float(
                            endpoint_row["reference_mean"]
                        ),
                        "endpoint_delta_target_minus_reference": float(
                            endpoint_row["delta_target_minus_reference"]
                        ),
                        "max_abs_curve_delta": float(
                            merged["delta_target_minus_reference"].abs().max()
                        ),
                    }
                )
    return pd.DataFrame(comparisons), pd.DataFrame(curve_differences)


def build_monotonicity_audit(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (estimator, event_type, group_key), sub in curves.groupby(
        ["estimator", "event_type", "month_group_key"], sort=False
    ):
        ordered = sub.sort_values("axis_value_tiles")
        diffs = np.diff(ordered["cumulative_mean"].to_numpy(float))
        rows.append(
            {
                "estimator": estimator,
                "event_type": event_type,
                "month_group_key": group_key,
                "n_curve_steps": len(diffs),
                "n_negative_steps": int(np.sum(diffs < -1e-12)),
                "minimum_step_delta": float(np.min(diffs)) if len(diffs) else np.nan,
                "monotonic_non_decreasing": bool(np.all(diffs >= -1e-12)),
            }
        )
    return pd.DataFrame(rows)


def draw_curve_panel(
    ax: plt.Axes,
    curves: pd.DataFrame,
    *,
    estimator: str,
    title: str,
    subtitle: str,
    legend: bool,
) -> None:
    subset = curves.loc[
        curves["estimator"].eq(estimator) & curves["event_type"].eq("reward")
    ].copy()
    for group_key in MONTH_GROUP_ORDER:
        group = subset.loc[subset["month_group_key"].eq(group_key)].sort_values(
            "axis_value_tiles"
        )
        n = int(group["n_sessions_averaged"].iloc[0])
        x = group["axis_value_tiles"].to_numpy(float)
        mean = group["cumulative_mean"].to_numpy(float)
        error = group["cumulative_sem"].to_numpy(float)
        linestyle = "--" if group_key == "experienced_day2" else "-"
        ax.plot(
            x,
            mean,
            color=MONTH_GROUP_COLORS[group_key],
            lw=1.15,
            ls=linestyle,
            label=f"{MONTH_GROUP_LABELS[group_key]} (n={n})",
        )
        ax.fill_between(
            x,
            mean - error,
            mean + error,
            color=MONTH_GROUP_COLORS[group_key],
            alpha=0.11,
            lw=0,
        )
    ax.set_xlabel("Maze tile transitions")
    ax.set_ylabel("Cumulative rewards")
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )
    ax.set_xlim(0, float(subset["axis_value_tiles"].max()))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E5E5E5", lw=0.4)
    if legend:
        ax.legend(
            loc="upper left",
            ncol=2,
            columnspacing=0.7,
            handlelength=1.4,
            handletextpad=0.35,
            borderaxespad=0.2,
        )


def save_figure(
    fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int = 600
) -> list[Path]:
    paths: list[Path] = []
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = out_dir / f"{stem}{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = dpi
        if suffix == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def render_figures(curves: pd.DataFrame, out_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    primary = "whole_group_observed_common_support"
    fig, ax = plt.subplots(figsize=(3.55, 2.45), facecolor="white")
    draw_curve_panel(
        ax,
        curves,
        estimator=primary,
        title="All-session reward accumulation",
        subtitle="Fully observed common range; no session exclusions or imputation",
        legend=True,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.85)
    outputs.extend(
        save_figure(fig, out_dir, "ed01d_reward_accumulation_whole_group_observed")
    )

    carry = "whole_group_terminal_carry_forward"
    fig, ax = plt.subplots(figsize=(3.55, 2.45), facecolor="white")
    draw_curve_panel(
        ax,
        curves,
        estimator=carry,
        title="All-session reward accumulation",
        subtitle="Original range; terminal counts carried forward after session end",
        legend=True,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.20, top=0.85)
    outputs.extend(
        save_figure(
            fig, out_dir, "ed01d_reward_accumulation_whole_group_carry_forward"
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55), facecolor="white", sharey=True)
    specs = (
        (
            "current_fixed_80pct_subset",
            "Current fixed subset",
            "11,050 transitions; n=20/17/16/10/8/60",
        ),
        (
            primary,
            "Whole group: observed",
            "5,200 transitions; n=23/17/17/12/10/74",
        ),
        (
            carry,
            "Whole group: carry forward",
            "11,050 transitions; n=23/17/17/12/10/74",
        ),
    )
    for index, (axis, (estimator, title, subtitle)) in enumerate(
        zip(axes, specs, strict=True)
    ):
        draw_curve_panel(
            axis,
            curves,
            estimator=estimator,
            title=title,
            subtitle=subtitle,
            legend=index == 2,
        )
        if index:
            axis.set_ylabel("")
    axes[2].legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        ncol=1,
        handlelength=1.4,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.075, right=0.83, bottom=0.20, top=0.84, wspace=0.20)
    outputs.extend(save_figure(fig, out_dir, "ed01d_estimator_comparison"))
    return outputs


def write_report(
    out_dir: Path,
    support: pd.DataFrame,
    selection: pd.DataFrame,
    comparisons: pd.DataFrame,
    monotonicity: pd.DataFrame,
    endpoints: dict[str, float],
    parity: pd.DataFrame,
) -> None:
    reward_compare = comparisons.loc[comparisons["event_type"].eq("reward")].copy()
    observed_compare = reward_compare.loc[
        reward_compare["comparison"].eq(
            "whole_observed_minus_current_fixed_on_observed_range"
        )
    ]
    carry_compare = reward_compare.loc[
        reward_compare["comparison"].eq(
            "whole_carry_minus_current_fixed_on_current_range"
        )
    ]
    dynamic = monotonicity.loc[
        monotonicity["estimator"].eq("whole_group_dynamic_available_case")
        & monotonicity["event_type"].eq("reward")
    ]

    def markdown_table(frame: pd.DataFrame) -> str:
        def cell(value: Any) -> str:
            if pd.isna(value):
                return "NA"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4g}"
            return str(value).replace("|", "\\|")

        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(cell(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# ED1 panel-d whole-group correction audit",
        "",
        "## Result",
        "",
        "The accepted panel-d source did not drop sessions because of load failures or biological exclusions. "
        "It selected a fixed subset of sessions reaching the common 11,050-transition endpoint after defining "
        "that endpoint from an 80%-within-group support rule. This changed the displayed cohort from 153 to "
        f"{int(support['current_fixed_subset_n'].sum())} sessions.",
        "",
        "The primary whole-group replacement retains all 153 sessions and uses the largest transition range "
        f"observed in every session: 0-{endpoints['whole_group_observed_endpoint_tiles']:.0f} corrected-nose "
        "maze-tile transitions. It uses no carry-forward or changing-composition averaging.",
        "",
        "## Cohort accounting",
        "",
        markdown_table(
            support[
                [
                    "month_group_label",
                    "whole_group_n",
                    "current_fixed_subset_n",
                    "dropped_n",
                    "retained_fraction",
                    "minimum_nose_tiles_max",
                ]
            ]
        ),
        "",
        "## Reward-curve changes",
        "",
        f"At the fully observed {endpoints['whole_group_observed_endpoint_tiles']:.0f}-transition endpoint, "
        "the whole-group estimate differs from the current fixed-subset estimate as follows:",
        "",
        markdown_table(
            observed_compare[
                [
                    "month_group_key",
                    "target_endpoint_mean",
                    "reference_endpoint_mean",
                    "endpoint_delta_target_minus_reference",
                    "max_abs_curve_delta",
                ]
            ]
        ),
        "",
        f"At the original {endpoints['current_fixed_endpoint_tiles']:.0f}-transition endpoint, retaining all "
        "sessions requires terminal-count carry-forward. That sensitivity differs from the current fixed subset by:",
        "",
        markdown_table(
            carry_compare[
                [
                    "month_group_key",
                    "target_endpoint_mean",
                    "reference_endpoint_mean",
                    "endpoint_delta_target_minus_reference",
                    "max_abs_curve_delta",
                ]
            ]
        ),
        "",
        "## Why the sessions were dropped",
        "",
        "The renderer first calculated an 80%-support transition endpoint within each group, used the smallest "
        "group endpoint as the shared 11,050-transition range, and then excluded every session below that shared "
        "endpoint from every plotted x value. The stated reason was to prevent non-monotonic cumulative curves "
        "caused by a changing available-session composition. The dynamic available-case audit confirms whether "
        "that artifact occurs, but the fixed-subset solution introduced a different problem: the curve no longer "
        "represents the complete behavioral groups.",
        "",
        markdown_table(
            dynamic[
                [
                    "month_group_key",
                    "n_negative_steps",
                    "minimum_step_delta",
                    "monotonic_non_decreasing",
                ]
            ]
        ),
        "",
        "## Selection diagnostic",
        "",
        markdown_table(selection),
        "",
        "## Recommendation",
        "",
        "Use the all-session fully observed common-support curve as the primary panel-d candidate. It matches "
        "panel b's nominal group membership while avoiding unobserved-transition imputation. Keep the full-range "
        "terminal-carry-forward curve only as a sensitivity because its right-hand portion treats session-end "
        "counts as if no additional rewards would occur over transitions the animal never completed.",
        "",
        "Do not promote either replacement until the author reviews the shortened 5,200-transition primary range.",
        "",
        "## Validation",
        "",
        f"- Current fixed-subset parity: {int(parity['passes'].sum())}/{len(parity)} rows pass; "
        f"maximum mean delta {parity['abs_mean_delta'].max():.3g}; maximum SEM delta {parity['abs_sem_delta'].max():.3g}.",
        "- Every source session has `load_ok=True`; the drop reason is transition support only.",
        "- Means are session-row means and bands are SEM across session rows; curves are descriptive.",
        "- The accepted ED1 record and manuscript caption were not modified.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def validate(
    out_dir: Path,
    curves: pd.DataFrame,
    sessions: pd.DataFrame,
    parity: pd.DataFrame,
    outputs: list[Path],
    endpoints: dict[str, float],
) -> dict[str, Any]:
    failures: list[str] = []
    expected_estimators = {
        "current_fixed_80pct_subset",
        "whole_group_observed_common_support",
        "whole_group_terminal_carry_forward",
        "whole_group_dynamic_available_case",
    }
    if set(curves["estimator"].astype(str)) != expected_estimators:
        failures.append("estimator inventory mismatch")
    if len(sessions) != 153 or sessions["session"].nunique() != 153:
        failures.append("source session count is not 153")
    if not parity["passes"].all():
        failures.append("current fixed-subset parity failed")
    observed = curves.loc[
        curves["estimator"].eq("whole_group_observed_common_support")
    ]
    observed_counts = (
        observed.groupby("month_group_key", observed=True)["n_sessions_averaged"]
        .first()
        .to_dict()
    )
    if observed_counts != EXPECTED_GROUP_COUNTS:
        failures.append(f"whole-group counts mismatch: {observed_counts}")
    if not np.isclose(
        observed["axis_value_tiles"].max(),
        endpoints["whole_group_observed_endpoint_tiles"],
    ):
        failures.append("whole-group observed endpoint mismatch")
    for (estimator, event_type, group_key), sub in curves.loc[
        ~curves["estimator"].eq("whole_group_dynamic_available_case")
    ].groupby(["estimator", "event_type", "month_group_key"], sort=False):
        values = sub.sort_values("axis_value_tiles")["cumulative_mean"].to_numpy(float)
        if np.any(np.diff(values) < -1e-12):
            failures.append(f"nonmonotonic curve: {estimator}/{event_type}/{group_key}")
    for path in outputs:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty figure export: {path.name}")
        if path.suffix.lower() == ".svg":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "<text" not in text:
                failures.append(f"SVG lacks editable text: {path.name}")
            if "<image" in text:
                failures.append(f"SVG unexpectedly contains raster image: {path.name}")
    required_tables = (
        "source_session_inclusion_audit.csv",
        "dropped_sessions.csv",
        "group_support_summary.csv",
        "selection_diagnostic.csv",
        "plot_ready_group_curves.csv",
        "plot_ready_session_curves.csv.gz",
        "current_fixed_subset_parity.csv",
        "estimator_endpoint_comparison.csv",
        "estimator_curve_differences.csv",
        "monotonicity_audit.csv",
        "REPORT.md",
    )
    for name in required_tables:
        path = out_dir / name
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing output artifact: {name}")
    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "source_sessions": len(sessions),
        "source_animals": int(sessions["animal_id"].nunique()),
        "whole_group_counts": observed_counts,
        "current_fixed_parity_rows": len(parity),
        "current_fixed_parity_passes": int(parity["passes"].sum()),
        "max_current_mean_delta": float(parity["abs_mean_delta"].max()),
        "max_current_sem_delta": float(parity["abs_sem_delta"].max()),
        "endpoints": endpoints,
        "figure_exports": [path.name for path in outputs],
    }
    (out_dir / "QA_REPORT.json").write_text(
        json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise AssertionError("; ".join(failures))
    return report


def write_manifest(out_dir: Path, outputs: list[Path], qa: dict[str, Any]) -> None:
    artifact_paths = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name not in {"MANIFEST.json"}
    )
    manifest = {
        "analysis": "ED1 panel-d whole-group correction audit",
        "status": "preview_only_not_promoted",
        "figure_contract": {
            "core_conclusion": "Panel d should retain the complete behavioral groups used by panel b.",
            "archetype": "quantitative comparison audit",
            "backend": "Python/matplotlib only",
            "primary_candidate": "whole_group_observed_common_support",
            "primary_tradeoff": "all sessions retained; x range ends at fully observed common support",
        },
        "sources": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (EVENT_SOURCE, SESSION_SOURCE, CURRENT_CURVES, CURRENT_COVERAGE)
        },
        "qa": qa,
        "artifacts": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_style()

    events, sessions, current, coverage = load_sources()
    current_rebuilt, current_session_curves, parity = reconstruct_current(
        events, sessions, current, coverage
    )
    curves, session_curves, endpoints = build_all_estimators(
        events,
        sessions,
        current_rebuilt,
        current_session_curves,
        coverage,
    )
    inclusion, support, selection = build_inclusion_audit(sessions, coverage)
    comparisons, curve_differences = build_estimator_comparison(curves, endpoints)
    monotonicity = build_monotonicity_audit(curves)

    inclusion.to_csv(out_dir / "source_session_inclusion_audit.csv", index=False)
    inclusion.loc[~inclusion["current_fixed_subset_included"]].to_csv(
        out_dir / "dropped_sessions.csv", index=False
    )
    support.to_csv(out_dir / "group_support_summary.csv", index=False)
    selection.to_csv(out_dir / "selection_diagnostic.csv", index=False)
    curves.to_csv(out_dir / "plot_ready_group_curves.csv", index=False)
    session_curves.to_csv(
        out_dir / "plot_ready_session_curves.csv.gz", index=False, compression="gzip"
    )
    parity.to_csv(out_dir / "current_fixed_subset_parity.csv", index=False)
    comparisons.to_csv(out_dir / "estimator_endpoint_comparison.csv", index=False)
    curve_differences.to_csv(out_dir / "estimator_curve_differences.csv", index=False)
    monotonicity.to_csv(out_dir / "monotonicity_audit.csv", index=False)

    outputs = render_figures(curves, out_dir)
    write_report(
        out_dir,
        support,
        selection,
        comparisons,
        monotonicity,
        endpoints,
        parity,
    )
    qa = validate(out_dir, curves, sessions, parity, outputs, endpoints)
    write_manifest(out_dir, outputs, qa)
    print(json.dumps(json_ready(qa), indent=2))


if __name__ == "__main__":
    main()
