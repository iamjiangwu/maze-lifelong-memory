"""Close remaining Extended Data 1/2 render, source-audit, and example gaps.

This script is deliberately table-first.  It rebuilds the exact per-session
conversion behind the current 153-session month-resolved accumulation curves,
exports the inclusion/support masks, adds row-retaining clustered inference,
computes the distinct unique-coverage-before-first-reward control, and renders
real examples for the 32/N32 and first-long-bout-onset estimands.

No notebook is edited and no legacy panel is used as a numeric source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ca_analysis.maze import NewMaze
from ca_analysis.trajectory import FindPathsToNode, LoadTraj, SplitModeClips


REPO = Path(__file__).resolve().parents[1]
REFRESH = REPO / "docs" / "extended_data_figure_panel_refresh_2026-07-12"
SOURCE = REPO / "docs" / "new_12mo_session_fit_2026-06-02" / "panel_by_panel_current_vs_new"
CURRENT_PREVIEW = REPO / "docs" / "figure_panel_previews"
CURRENT_BEHAVIOR = REPO / "docs" / "behavior_panel_sensitivity_840_841_with_new12_baseline_2026-06-06"
EXACT_ED2 = REFRESH / "rerun_current" / "ed02_lb0_tiles_current_plus_new12"

OUT1 = (
    REFRESH
    / "source_rendered_new"
    / "extended_data_fig_01"
    / "month_source_coverage_gap_closure_2026-07-14"
)
OUT2 = (
    REFRESH
    / "source_rendered_new"
    / "extended_data_fig_02"
    / "illustrative_support_gap_closure_2026-07-14"
)

CURRENT_EVENTS = CURRENT_PREVIEW / "whole_night_behavior_accumulation_month_day2_v2_events.csv"
CURRENT_SESSIONS = CURRENT_PREVIEW / "whole_night_behavior_accumulation_month_day2_v2_session_summary.csv"
NEW_EVENTS = SOURCE / "whole_night_behavior_month_new12_events.csv"
NEW_SESSIONS = SOURCE / "whole_night_behavior_month_new12_session_summary.csv"
SAVED_GROUP_CURVES = SOURCE / "whole_night_behavior_month_current_plus_new12_group_curves.csv"
REWARD_METRICS = SOURCE / "reward_performance_current_plus_new12_per_session_metrics.csv"
FIRST_CHUNK_POINTS = SOURCE / "exploration_first_chunk_current_plus_new_points.csv"
CYCLE_SUMMARY = CURRENT_BEHAVIOR / "exploration_efficiency_cycles_pooled_current_plus_new12_summary.csv"
EXACT_LB0 = EXACT_ED2 / "ed02_lb0_matched_tile_transitions_per_session_current_plus_new12.csv"

MONTH_ORDER = [0, 3, 6, 9, 12]
MONTH_KEYS = ["0mo", "3mo", "6mo", "9mo", "12mo"]
MONTH_COLORS = {
    "0mo": "#78ADD2",
    "3mo": "#F2B3AE",
    "6mo": "#E77D72",
    "9mo": "#C84B44",
    "12mo": "#8E2C2A",
    "experienced_day2": "#5B8C5A",
}
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
GRID = "#E2E2E2"
GREY = "#8A8A8A"
ACCENT = "#C84B44"
BLUE = "#2E6F9E"


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.4,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 5.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str, *, dpi: int = 600) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = root / f"{stem}{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def _sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0


def _bh(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    out = np.full(p.size, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if not finite.size:
        return out
    order = finite[np.argsort(p[finite], kind="mergesort")]
    ranked = p[order] * float(order.size) / np.arange(1, order.size + 1, dtype=float)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out


def _p_text(value: float, *, prefix: str = "q") -> str:
    if not np.isfinite(value):
        return f"{prefix}=NA"
    if value < 0.001:
        return f"{prefix}={value:.1e}"
    return f"{prefix}={value:.3f}"


def _cluster_fit(
    frame: pd.DataFrame,
    *,
    y: str,
    target_col: str,
    cluster_col: str,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    covariates = list(covariates or [])
    columns = [y, target_col, cluster_col, *covariates]
    data = frame.loc[:, columns].copy()
    for column in [y, target_col, *covariates]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.loc[np.isfinite(data[[y, target_col, *covariates]]).all(axis=1)].copy()
    data[cluster_col] = data[cluster_col].astype(str)
    x_parts = [np.ones(len(data), dtype=float), data[target_col].to_numpy(float)]
    x_parts.extend(data[column].to_numpy(float) for column in covariates)
    x = np.column_stack(x_parts)
    yv = data[y].to_numpy(float)
    groups = data[cluster_col].to_numpy(str)
    g = int(pd.Series(groups).nunique())
    if len(data) <= x.shape[1] or g < 3:
        raise ValueError(f"Insufficient support for clustered fit: n={len(data)}, G={g}, k={x.shape[1]}")
    fit = sm.OLS(yv, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    beta = float(fit.params[1])
    se = float(fit.bse[1])
    df = int(g - 1)
    t_value = float(beta / se)
    p_value = float(2.0 * stats.t.sf(abs(t_value), df=df))
    critical = float(stats.t.ppf(0.975, df=df))
    return {
        "n_rows": int(len(data)),
        "n_clusters": g,
        "df_t": df,
        "beta_target_minus_baseline": beta,
        "cluster_se": se,
        "t_cluster": t_value,
        "p_cluster": p_value,
        "ci95_low": float(beta - critical * se),
        "ci95_high": float(beta + critical * se),
        "covariates": "+".join(covariates) if covariates else "none",
        "covariance": "CR1 clustered by animal; two-sided Student t(G-1)",
    }


def _cluster_mean_log(frame: pd.DataFrame, value_col: str, cluster_col: str) -> dict[str, float]:
    data = frame.loc[:, [value_col, cluster_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.loc[np.isfinite(data[value_col])].copy()
    data[cluster_col] = data[cluster_col].astype(str)
    x = np.ones((len(data), 1), dtype=float)
    fit = sm.OLS(data[value_col].to_numpy(float), x).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col].to_numpy(str), "use_correction": True},
    )
    g = int(data[cluster_col].nunique())
    critical = float(stats.t.ppf(0.975, df=g - 1))
    mean = float(fit.params[0])
    se = float(fit.bse[0])
    return {
        "mean_log": mean,
        "se_log": se,
        "ci95_low_log": mean - critical * se,
        "ci95_high_log": mean + critical * se,
        "n_rows": int(len(data)),
        "n_clusters": g,
    }


def _load_accumulation_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_events = pd.read_csv(CURRENT_EVENTS)
    current_events["source_set"] = "current_saved"
    current_events["source_event_table"] = str(CURRENT_EVENTS)
    new_events = pd.read_csv(NEW_EVENTS)
    new_events["source_set"] = "new_12mo"
    new_events["source_event_table"] = str(NEW_EVENTS)
    events = pd.concat([current_events, new_events], ignore_index=True, sort=False)

    current_sessions = pd.read_csv(CURRENT_SESSIONS, dtype={"animal_id": str})
    current_sessions["source_set"] = "current_saved"
    current_sessions["source_session_table"] = str(CURRENT_SESSIONS)
    new_sessions = pd.read_csv(NEW_SESSIONS, dtype={"animal_id": str})
    new_sessions["source_set"] = "new_12mo"
    new_sessions["source_session_table"] = str(NEW_SESSIONS)
    sessions = pd.concat([current_sessions, new_sessions], ignore_index=True, sort=False)
    sessions["animal_id"] = sessions["animal_id"].astype(str)
    saved = pd.read_csv(SAVED_GROUP_CURVES)

    if len(sessions) != 153 or sessions["session"].nunique() != 153:
        raise ValueError("Accumulation session source is not 153 unique sessions")
    if not sessions["load_ok"].astype(bool).all():
        failures = sessions.loc[~sessions["load_ok"].astype(bool), ["session", "error"]]
        raise ValueError(f"Accumulation source has failed sessions:\n{failures}")
    if len(events) != 33001:
        raise ValueError(f"Expected 33,001 current event rows, found {len(events):,}")
    if set(sessions["session"].astype(str)) != set(saved_group_session_scope(sessions)):
        raise AssertionError("Internal session-scope check failed")
    return events, sessions, saved


def saved_group_session_scope(sessions: pd.DataFrame) -> list[str]:
    return sessions["session"].astype(str).tolist()


def _rebuild_accumulation_conversion(
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    saved: pd.DataFrame,
) -> dict[str, Any]:
    durations = pd.to_numeric(sessions["maze_duration_sec"], errors="coerce").to_numpy(float)
    max_duration = float(np.nanmax(durations))
    grid = np.arange(0.0, max_duration + 10.0, 10.0, dtype=float)
    windows: dict[str, dict[str, Any]] = {}
    for group_key, sub in sessions.groupby("group_key", sort=False):
        group_durations = pd.to_numeric(sub["maze_duration_sec"], errors="coerce").to_numpy(float)
        initial_n = int(len(group_durations))
        min_n = int(math.ceil(0.80 * initial_n))
        n_active = np.asarray([np.sum(group_durations >= t - 1e-9) for t in grid], dtype=int)
        ok = n_active >= min_n
        windows[str(group_key)] = {
            "initial_n_sessions": initial_n,
            "min_coverage_n_sessions": min_n,
            "coverage_window_sec": float(np.max(grid[ok])),
            "n_active": n_active,
        }
    global_window = float(min(item["coverage_window_sec"] for item in windows.values()))
    display_grid = grid[grid <= global_window + 1e-9]

    event_lookup: dict[tuple[str, str], np.ndarray] = {}
    for (session, event_type), sub in events.groupby(["session", "event_type"], sort=False):
        times = pd.to_numeric(sub["maze_seconds"], errors="coerce").to_numpy(float)
        event_lookup[(str(session), str(event_type))] = np.sort(times[np.isfinite(times)])

    session_curve_frames: list[pd.DataFrame] = []
    group_curve_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    for event_type in ("reward", "long_path_reward"):
        for group_key, sub in sessions.sort_values(["group_order", "group_key", "session"]).groupby(
            "group_key", sort=False
        ):
            group_key = str(group_key)
            vals: list[np.ndarray] = []
            display_vals: list[np.ndarray] = []
            display_active: list[np.ndarray] = []
            for _, row in sub.iterrows():
                session = str(row["session"])
                duration = float(row["maze_duration_sec"])
                times = event_lookup.get((session, event_type), np.asarray([], dtype=float))
                times = times[(times >= 0.0) & (times <= duration + 1e-9)]
                cumulative = np.searchsorted(times, grid, side="right").astype(float)
                vals.append(cumulative)
                dvals = cumulative[: display_grid.size]
                active = display_grid <= duration + 1e-9
                display_vals.append(dvals)
                display_active.append(active)
                session_curve_frames.append(
                    pd.DataFrame(
                        {
                            "session": session,
                            "animal_id": str(row["animal_id"]),
                            "group_key": group_key,
                            "group_label": str(row["group_label"]),
                            "group_order": int(row["group_order"]),
                            "month": int(row["month"]),
                            "memory_stage": str(row["memory_stage"]),
                            "source_set": str(row["source_set"]),
                            "event_type": event_type,
                            "maze_seconds": display_grid,
                            "time_minutes": display_grid / 60.0,
                            "cum_count": dvals,
                            "cum_count_active": np.where(active, dvals, np.nan),
                            "session_in_window": active,
                            "display_coverage_ok": True,
                        }
                    )
                )
            matrix = np.vstack(vals)
            mean = matrix.mean(axis=0)
            sem = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0]) if matrix.shape[0] > 1 else np.zeros(grid.size)
            n_active = windows[group_key]["n_active"]
            min_n = int(windows[group_key]["min_coverage_n_sessions"])
            group_curve_frames.append(
                pd.DataFrame(
                    {
                        "group_key": group_key,
                        "group_label": str(sub["group_label"].iloc[0]),
                        "group_order": int(sub["group_order"].iloc[0]),
                        "color_key": str(sub["color_key"].iloc[0]),
                        "event_type": event_type,
                        "maze_seconds": grid,
                        "time_minutes": grid / 60.0,
                        "cum_mean_rebuilt": mean,
                        "cum_sem_rebuilt": sem,
                        "n_sessions_active": n_active,
                        "n_sessions_averaged": matrix.shape[0],
                        "initial_n_sessions": int(matrix.shape[0]),
                        "min_coverage_n_sessions": min_n,
                        "coverage_ok_rebuilt": (n_active >= min_n) & (grid <= global_window + 1e-9),
                        "plot_window_sec_rebuilt": global_window,
                    }
                )
            )
            coverage_frames.append(
                pd.DataFrame(
                    {
                        "group_key": group_key,
                        "event_type": event_type,
                        "maze_seconds": display_grid,
                        "time_minutes": display_grid / 60.0,
                        "n_active_sessions": n_active[: display_grid.size],
                        "n_total_sessions": matrix.shape[0],
                        "min_coverage_n_sessions": min_n,
                        "active_fraction": n_active[: display_grid.size] / float(matrix.shape[0]),
                        "coverage_ok": True,
                        "global_plot_window_sec": global_window,
                    }
                )
            )

    session_curves = pd.concat(session_curve_frames, ignore_index=True)
    rebuilt = pd.concat(group_curve_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    # ``time_minutes`` is a derived display column whose final decimal can
    # differ by one floating-point representation step after CSV round-trip.
    # The exact time key is the integer-like 10-s ``maze_seconds`` grid.
    parity = saved.merge(
        rebuilt.drop(columns=["time_minutes"]),
        on=["group_key", "group_label", "group_order", "color_key", "event_type", "maze_seconds"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    parity["abs_mean_delta"] = np.abs(parity["cum_mean"] - parity["cum_mean_rebuilt"])
    parity["abs_sem_delta"] = np.abs(parity["cum_sem"] - parity["cum_sem_rebuilt"])
    parity["coverage_match"] = parity["coverage_ok"].astype(bool).eq(parity["coverage_ok_rebuilt"].astype(bool))
    parity["n_active_match"] = pd.to_numeric(parity["n_sessions"], errors="coerce").eq(
        pd.to_numeric(parity["n_sessions_active"], errors="coerce")
    )
    parity["n_averaged_match"] = pd.to_numeric(parity["n_averaged_sessions"], errors="coerce").eq(
        pd.to_numeric(parity["n_sessions_averaged"], errors="coerce")
    )
    parity["passes"] = (
        parity["_merge"].eq("both")
        & parity["abs_mean_delta"].le(1e-12)
        & parity["abs_sem_delta"].le(1e-12)
        & parity["coverage_match"]
        & parity["n_active_match"]
        & parity["n_averaged_match"]
    )
    if not parity["passes"].all():
        OUT1.mkdir(parents=True, exist_ok=True)
        parity.to_csv(OUT1 / "_debug_accumulation_parity_failure.csv", index=False)
        bad = parity.loc[~parity["passes"]].head(10)
        raise AssertionError(f"Rebuilt accumulation conversion differs from maintained group curves:\n{bad}")

    endpoint = session_curves.loc[np.isclose(session_curves["maze_seconds"], global_window)].copy()
    endpoint["log1p_cum_count"] = np.log1p(endpoint["cum_count"])
    stats_rows: list[dict[str, Any]] = []
    for event_type in ("reward", "long_path_reward"):
        for month in (3, 6, 9, 12):
            target = f"{month}mo"
            for specification, mask in (
                ("all_rows_carried_forward_as_plotted", np.ones(len(endpoint), dtype=bool)),
                ("active_at_endpoint_sensitivity", endpoint["session_in_window"].astype(bool).to_numpy()),
            ):
                sub = endpoint.loc[
                    mask
                    & endpoint["event_type"].eq(event_type)
                    & endpoint["group_key"].isin(["0mo", target])
                ].copy()
                sub["target"] = sub["group_key"].eq(target).astype(int)
                result = _cluster_fit(
                    sub,
                    y="log1p_cum_count",
                    target_col="target",
                    cluster_col="animal_id",
                )
                result.update(
                    {
                        "event_type": event_type,
                        "month": month,
                        "contrast": f"{target}_minus_0mo",
                        "specification": specification,
                        "analysis_scale": "log1p cumulative count at common 80%-coverage endpoint",
                        "endpoint_seconds": global_window,
                        "target_n_rows": int(sub["target"].sum()),
                        "baseline_n_rows": int((sub["target"] == 0).sum()),
                        "ratio_one_plus_count": float(np.exp(result["beta_target_minus_baseline"])),
                    }
                )
                stats_rows.append(result)
    endpoint_stats = pd.DataFrame(stats_rows)
    endpoint_stats["q_bh8_within_specification"] = np.nan
    for specification, indices in endpoint_stats.groupby("specification").groups.items():
        endpoint_stats.loc[list(indices), "q_bh8_within_specification"] = _bh(
            endpoint_stats.loc[list(indices), "p_cluster"]
        )

    return {
        "session_curves": session_curves,
        "rebuilt_group_curves": rebuilt,
        "coverage": coverage,
        "parity": parity,
        "endpoint_rows": endpoint,
        "endpoint_stats": endpoint_stats,
        "global_window": global_window,
    }


def _render_month_curve(saved: pd.DataFrame, event_type: str, ylabel: str, title: str, stem: str) -> list[str]:
    subset = saved.loc[saved["event_type"].eq(event_type) & saved["coverage_ok"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(3.55, 2.35))
    order = [*MONTH_KEYS, "experienced_day2"]
    for key in order:
        group = subset.loc[subset["group_key"].eq(key)].sort_values("time_minutes")
        if group.empty:
            continue
        x = group["time_minutes"].to_numpy(float)
        mean = group["cum_mean"].to_numpy(float)
        sem = group["cum_sem"].to_numpy(float)
        n = int(group["initial_n_sessions"].iloc[0])
        label = GROUP_LABELS.get(key, str(group["group_label"].iloc[0])) + f" (n={n})"
        linestyle = "--" if key == "experienced_day2" else "-"
        ax.plot(x, mean, color=MONTH_COLORS[key], lw=1.15, ls=linestyle, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=MONTH_COLORS[key], alpha=0.10, lw=0)
    ax.set_xlabel("Maze time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=2.0)
    ax.set_xlim(0, float(subset["time_minutes"].max()))
    ax.legend(loc="upper left", ncol=2, columnspacing=0.7, handlelength=1.5, handletextpad=0.4)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.text(
        0.99,
        0.02,
        "Mean ± SEM; common 80%-session support window",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.5,
        color=GREY,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.90)
    return _save(fig, OUT1, stem)


def _render_row_retaining_lb0(metrics: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    data = metrics.loc[metrics["load_ok"].astype(bool)].copy()
    data["animal_id"] = data["animal_id"].astype(str)
    data["lb0"] = pd.to_numeric(data["lb0"], errors="coerce")
    data = data.loc[np.isfinite(data["lb0"]) & data["lb0"].gt(0)].copy()
    data["log_lb0_s"] = np.log(data["lb0"])
    order = ["naive", "return", "experienced_day2"]
    comparisons = [("return", "naive"), ("experienced_day2", "naive"), ("return", "experienced_day2")]
    rows: list[dict[str, Any]] = []
    for target, baseline in comparisons:
        sub = data.loc[data["group_key"].isin([target, baseline])].copy()
        sub["target"] = sub["group_key"].eq(target).astype(int)
        result = _cluster_fit(sub, y="log_lb0_s", target_col="target", cluster_col="animal_id")
        result.update(
            {
                "target": target,
                "baseline": baseline,
                "contrast": f"{target}_minus_{baseline}",
                "analysis_scale": "log seconds to first long bout",
                "fold_change_target_over_baseline": float(np.exp(result["beta_target_minus_baseline"])),
                "target_n_rows": int(sub["target"].sum()),
                "baseline_n_rows": int((sub["target"] == 0).sum()),
            }
        )
        rows.append(result)
    pair_stats = pd.DataFrame(rows)
    pair_stats["q_bh3"] = _bh(pair_stats["p_cluster"])

    summary_rows: list[dict[str, Any]] = []
    for key in order:
        row = _cluster_mean_log(data.loc[data["group_key"].eq(key)], "log_lb0_s", "animal_id")
        row.update(
            {
                "group_key": key,
                "geometric_mean_s": float(np.exp(row["mean_log"])),
                "ci95_low_s": float(np.exp(row["ci95_low_log"])),
                "ci95_high_s": float(np.exp(row["ci95_high_log"])),
            }
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    rng = np.random.default_rng(20260714)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for xpos, key in enumerate(order):
        sub = data.loc[data["group_key"].eq(key)]
        y = sub["lb0"].to_numpy(float) / 60.0
        jitter = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(
            xpos + jitter,
            y,
            s=6.5,
            facecolor="white",
            edgecolor=GROUP_COLORS[key],
            linewidth=0.45,
            alpha=0.68,
            zorder=2,
        )
        sr = summary.loc[summary["group_key"].eq(key)].iloc[0]
        center = float(sr["geometric_mean_s"]) / 60.0
        low = float(sr["ci95_low_s"]) / 60.0
        high = float(sr["ci95_high_s"]) / 60.0
        ax.errorbar(
            xpos,
            center,
            yerr=np.asarray([[center - low], [high - center]]),
            color=GROUP_COLORS[key],
            marker="o",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=1.8,
            elinewidth=0.8,
            zorder=3,
        )
        ax.text(
            xpos,
            0.02,
            f"{len(sub)} sessions\n{str(sub['animal_id'].nunique())} mice",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=4.4,
        )
    ax.set_yscale("log")
    ax.set_xticks(range(3), [GROUP_LABELS[key] for key in order])
    ax.set_ylabel("Time to first long bout (min)")
    ax.set_title("Session rows retained; inference clustered by mouse", loc="left", fontweight="bold", pad=2.0)
    q_lines = []
    for _, row in pair_stats.iterrows():
        q_lines.append(
            f"{GROUP_LABELS[row['target']]} − {GROUP_LABELS[row['baseline']]}: {_p_text(float(row['q_bh3']))}"
        )
    ax.text(0.02, 0.98, "\n".join(q_lines), transform=ax.transAxes, ha="left", va="top", fontsize=4.5)
    ax.grid(axis="y", color=GRID, lw=0.4, which="both")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.89)
    outputs = _save(fig, OUT1, "ed01_row_retaining_first_long_bout_current153")
    return outputs, pair_stats, summary


def _legacy_binned_xy(x_raw: pd.Series, y_raw: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    source_x = pd.to_numeric(x_raw, errors="coerce").to_numpy(dtype=float)
    source_y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    x = 1.0 - source_y
    y = 1.0 - source_x
    bx = np.full(x.size, np.nan, dtype=float)
    by = np.full(y.size, np.nan, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    step = -1.0 / 15.0
    bx[finite] = np.floor((x[finite] - 1.0) / step)
    by[finite] = np.floor((y[finite] - 1.0) / step)
    bad = (~finite) | (bx < 0) | (bx > 14) | (by < 0) | (by > 14)
    bx[bad] = np.nan
    by[bad] = np.nan
    return bx, by


def _cell_labels_from_bins(maze, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    labels = np.full(len(bx), -1, dtype=int)
    finite = np.isfinite(bx) & np.isfinite(by)
    lookup = np.full((15, 15), -1, dtype=int)
    for (row, column), label in maze.ce.items():
        lookup[int(row), int(column)] = int(label)
    labels[finite] = lookup[by[finite].astype(int), bx[finite].astype(int)]
    return labels


def _corrected_nose_cells(frame: pd.DataFrame, maze) -> np.ndarray:
    nose_bx, nose_by = _legacy_binned_xy(frame["x"], frame["y"])
    mass_bx, mass_by = _legacy_binned_xy(frame["mass_x"], frame["mass_y"])
    nose = _cell_labels_from_bins(maze, nose_bx, nose_by)
    mass = _cell_labels_from_bins(maze, mass_bx, mass_by)
    corrected = np.full(len(frame), -1, dtype=int)
    nv = nose >= 0
    mv = mass >= 0
    corrected[nv & ~mv] = nose[nv & ~mv]
    corrected[mv & ~nv] = mass[mv & ~nv]
    both = nv & mv
    corrected[both] = nose[both]
    indices = np.flatnonzero(both)
    use_mass = maze.di[mass[both], nose[both]] > 2.0
    corrected[indices[use_mass]] = mass[indices[use_mass]]
    return corrected


def _read_nose_prefix(path: Path, stop_frame: float, *, chunksize: int = 50000) -> pd.DataFrame:
    columns = ["frame", "x", "y", "mass_x", "mass_y"]
    parts: list[pd.DataFrame] = []
    prior_max = -np.inf
    monotonic = True
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        chunk["frame"] = pd.to_numeric(chunk["frame"], errors="coerce")
        chunk = chunk.loc[np.isfinite(chunk["frame"])].copy()
        if chunk.empty:
            continue
        values = chunk["frame"].to_numpy(float)
        monotonic = monotonic and bool(np.all(np.diff(values) >= 0)) and float(values[0]) >= prior_max
        prior_max = float(values[-1])
        keep = chunk.loc[chunk["frame"].le(float(stop_frame))].copy()
        if not keep.empty:
            parts.append(keep)
        if monotonic and float(values[-1]) > float(stop_frame):
            break
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("frame").reset_index(drop=True)


def _dedupe_valid_with_frames(cells: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=int)
    frames = np.asarray(frames, dtype=float)
    valid = cells >= 0
    cells = cells[valid]
    frames = frames[valid]
    if cells.size == 0:
        return cells, frames
    keep = np.insert(np.diff(cells) != 0, 0, True)
    return cells[keep], frames[keep]


def _frame_to_maze_seconds(frames: np.ndarray, trajectory, fps: float = 25.0) -> np.ndarray:
    values = np.asarray(frames, dtype=float)
    bouts = np.asarray(trajectory.fr, dtype=float)
    durations = bouts[:, 1] - bouts[:, 0]
    cumulative_before = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    indices = np.searchsorted(bouts[:, 1], values, side="left")
    output = np.empty(values.size, dtype=float)
    after = indices >= len(bouts)
    output[after] = float(np.sum(durations)) / fps
    active = ~after
    output[active] = (cumulative_before[indices[active]] + values[active] - bouts[indices[active], 0]) / fps
    return output


def _compute_unique_coverage(
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    maze,
    *,
    recompute: bool,
) -> pd.DataFrame:
    cache = OUT1 / "ed01_unique_coverage_before_first_reward_current153.csv"
    if cache.exists() and not recompute:
        cached = pd.read_csv(cache, dtype={"animal_id": str})
        if len(cached) == 153 and cached["session"].nunique() == 153:
            return cached
    first_reward = (
        events.loc[events["event_type"].eq("reward")]
        .sort_values(["session", "maze_seconds", "abs_frame"])
        .groupby("session", as_index=False)
        .first()[["session", "abs_frame", "maze_seconds"]]
        .rename(columns={"abs_frame": "first_reward_abs_frame", "maze_seconds": "first_reward_maze_s"})
    )
    joined = metrics.merge(first_reward, left_on="nickname", right_on="session", how="inner", validate="one_to_one")
    if len(joined) != 153:
        raise ValueError(f"First-reward join should contain 153 sessions, found {len(joined)}")
    terminal_cells = {int(maze.ru[run][-1]) for run in range(63, 127)}
    rows: list[dict[str, Any]] = []
    for index, row in joined.iterrows():
        nickname = str(row["nickname"])
        print(f"ED1 unique coverage {index + 1:03d}/153: {nickname}", flush=True)
        source = Path(str(row["nose_traversal_source"]))
        prefix = _read_nose_prefix(source, float(row["first_reward_abs_frame"]))
        cells = _corrected_nose_cells(prefix, maze)
        valid_cells = cells[cells >= 0]
        deduped, deduped_frames = _dedupe_valid_with_frames(cells, prefix["frame"].to_numpy(float))
        transitions = float(max(0, deduped.size - 1)) if deduped.size else np.nan
        unique_cells = int(np.unique(valid_cells).size) if valid_cells.size else 0
        unique_terminal_cells = int(np.isin(np.unique(valid_cells), list(terminal_cells)).sum()) if valid_cells.size else 0
        saved_tiles = float(row["nose_tiles_to_first_reward"])
        rows.append(
            {
                "session": nickname,
                "animal_id": str(row["animal_id"]),
                "month": int(row["month"]),
                "day": int(row["day"]),
                "group_key": str(row["group_key"]),
                "month_stage_key": str(row["month_stage_key"]),
                "source_set": str(row["source_set"]),
                "first_reward_abs_frame": float(row["first_reward_abs_frame"]),
                "first_reward_maze_s": float(row["first_reward_maze_s"]),
                "saved_ti1_s": float(row["ti1"]),
                "nose_prefix_rows": int(len(prefix)),
                "nose_prefix_valid_rows": int(valid_cells.size),
                "nose_prefix_valid_fraction": float(valid_cells.size / len(prefix)) if len(prefix) else np.nan,
                "nose_tile_transitions_recomputed": transitions,
                "nose_tile_transitions_saved": saved_tiles,
                "transition_count_delta": transitions - saved_tiles,
                "unique_physical_maze_cells_before_first_reward": unique_cells,
                "unique_terminal_endpoint_cells_before_first_reward": unique_terminal_cells,
                "n_transition_events": int(deduped.size),
                "nose_traversal_source": str(source),
                "source_size_bytes": int(source.stat().st_size),
                "terminal_definition": "physical endpoint cell ma.ru[run][-1] for tree runs 63..126",
                "maze_cell_definition": "corrected nose/mass physical cell label from NewMaze(6).ce",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    return result


def _unique_coverage_stats(coverage: pd.DataFrame) -> pd.DataFrame:
    day1 = coverage.loc[coverage["day"].eq(1) & coverage["month"].isin(MONTH_ORDER)].copy()
    day1["log1p_first_reward_s"] = np.log1p(day1["first_reward_maze_s"])
    metrics = [
        "unique_physical_maze_cells_before_first_reward",
        "unique_terminal_endpoint_cells_before_first_reward",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        day1[f"log1p_{metric}"] = np.log1p(pd.to_numeric(day1[metric], errors="coerce"))
        for month in (3, 6, 9, 12):
            sub = day1.loc[day1["month"].isin([0, month])].copy()
            sub["target"] = sub["month"].eq(month).astype(int)
            for specification, covariates in (
                ("raw_count", []),
                ("time_adjusted", ["log1p_first_reward_s"]),
            ):
                result = _cluster_fit(
                    sub,
                    y=f"log1p_{metric}",
                    target_col="target",
                    cluster_col="animal_id",
                    covariates=covariates,
                )
                result.update(
                    {
                        "metric": metric,
                        "month": month,
                        "contrast": f"{month}mo_minus_naive",
                        "specification": specification,
                        "analysis_scale": f"log1p({metric})",
                        "ratio_one_plus_count": float(np.exp(result["beta_target_minus_baseline"])),
                        "target_n_rows": int(sub["target"].sum()),
                        "baseline_n_rows": int((sub["target"] == 0).sum()),
                    }
                )
                rows.append(result)
    output = pd.DataFrame(rows)
    output["q_bh8_within_specification"] = np.nan
    for specification, indices in output.groupby("specification").groups.items():
        output.loc[list(indices), "q_bh8_within_specification"] = _bh(output.loc[list(indices), "p_cluster"])
    return output


def _render_unique_coverage(coverage: pd.DataFrame, stats_table: pd.DataFrame) -> list[str]:
    day1 = coverage.loc[coverage["day"].eq(1) & coverage["month"].isin(MONTH_ORDER)].copy()
    specs = [
        ("unique_physical_maze_cells_before_first_reward", "Unique physical maze cells"),
        ("unique_terminal_endpoint_cells_before_first_reward", "Unique terminal endpoints"),
    ]
    rng = np.random.default_rng(20260714)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.40))
    for ax, (metric, ylabel) in zip(axes, specs):
        for xpos, month in enumerate(MONTH_ORDER):
            sub = day1.loc[day1["month"].eq(month)]
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
            color = MONTH_COLORS[f"{month}mo"]
            ax.scatter(
                xpos + rng.uniform(-0.09, 0.09, len(values)),
                values,
                s=7,
                facecolor="white",
                edgecolor=color,
                linewidth=0.45,
                alpha=0.72,
            )
            ax.errorbar(
                xpos,
                float(np.mean(values)),
                yerr=_sem(values),
                color=color,
                marker="o",
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.35,
                capsize=1.6,
                elinewidth=0.8,
            )
            ax.text(xpos, 0.02, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4.4)
            if month != 0:
                q = float(
                    stats_table.loc[
                        stats_table["metric"].eq(metric)
                        & stats_table["month"].eq(month)
                        & stats_table["specification"].eq("time_adjusted"),
                        "q_bh8_within_specification",
                    ].iloc[0]
                )
                ax.text(xpos, 0.98, _p_text(q), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.3)
        ax.set_xticks(range(5), ["Naive", "3 mo", "6 mo", "9 mo", "12 mo"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0].set_title("Spatial coverage before first reward", loc="left", fontweight="bold", pad=2.0)
    axes[1].set_title("Endpoint coverage before first reward", loc="left", fontweight="bold", pad=2.0)
    fig.text(
        0.99,
        0.01,
        "Session rows; labels: time-adjusted CR1 mouse-clustered q (BH8)",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color=GREY,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.21, top=0.90, wspace=0.28)
    return _save(fig, OUT1, "ed01_unique_spatial_coverage_before_first_reward_current153")


def _render_unique_coverage_forest(stats_table: pd.DataFrame) -> list[str]:
    data = stats_table.loc[stats_table["specification"].eq("time_adjusted")].copy()
    specs = [
        ("unique_physical_maze_cells_before_first_reward", "Unique physical maze cells"),
        ("unique_terminal_endpoint_cells_before_first_reward", "Unique terminal endpoints"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.15), sharex=True)
    for ax, (metric, title) in zip(axes, specs):
        sub = data.loc[data["metric"].eq(metric)].sort_values("month", ascending=False)
        y = np.arange(len(sub))
        for yi, (_, row) in zip(y, sub.iterrows()):
            color = MONTH_COLORS[f"{int(row['month'])}mo"]
            ax.plot([row["ci95_low"], row["ci95_high"]], [yi, yi], color=color, lw=1.0)
            ax.plot(row["beta_target_minus_baseline"], yi, "o", color=color, ms=3.6)
            ax.text(
                row["ci95_high"] + 0.025,
                yi,
                _p_text(float(row["q_bh8_within_specification"])),
                va="center",
                ha="left",
                fontsize=4.4,
            )
        ax.axvline(0, color=GREY, lw=0.7, ls="--")
        ax.set_yticks(y, [f"{int(month)} mo − naive" for month in sub["month"]])
        ax.set_title(title, loc="left", fontweight="bold", pad=2.0)
        ax.set_xlabel("Adjusted difference in log(1 + count)")
        ax.grid(axis="x", color=GRID, lw=0.4)
    fig.text(
        0.99,
        0.01,
        "Adjusted for log(1 + time to first reward); CR1 by mouse; 95% CI; BH8",
        ha="right",
        va="bottom",
        fontsize=4.6,
        color=GREY,
    )
    fig.subplots_adjust(left=0.20, right=0.93, bottom=0.23, top=0.88, wspace=0.34)
    return _save(fig, OUT1, "ed01_unique_coverage_time_adjusted_current153")


def _extract_explore_endnodes(trajectory, maze, *, end_level: int = 6) -> np.ndarray:
    clips = np.asarray(SplitModeClips(trajectory, maze))
    if clips.size == 0:
        return np.asarray([], dtype=int)
    explore = clips[np.where(clips[:, 3] == 2)]
    parts: list[np.ndarray] = []
    for clip in explore:
        bout, start, nstep = int(clip[0]), int(clip[1]), int(clip[2])
        segment = np.asarray(trajectory.no[bout][start : start + nstep])
        if segment.ndim == 2 and segment.shape[1] >= 1:
            parts.append(segment[:, 0].astype(int, copy=False))
    if not parts:
        return np.asarray([], dtype=int)
    nodes = np.concatenate(parts)
    endpoints = np.arange(2**end_level - 1, 2 ** (end_level + 1) - 1, dtype=int)
    return nodes[np.isin(nodes, endpoints)]


def _complete_exploration_cycles(
    sequence: np.ndarray,
    *,
    target_unique: int = 32,
    nickname: str,
    group: str,
) -> pd.DataFrame:
    """Segment one endpoint sequence with the maintained reset-after-32 rule."""
    seq = np.asarray(sequence, dtype=int).ravel()
    rows: list[dict[str, Any]] = []
    cycle_start = 0
    seen: set[int] = set()
    cycle_index = 0
    for endpoint_index, node in enumerate(seq):
        seen.add(int(node))
        if len(seen) < int(target_unique):
            continue
        cycle_index += 1
        steps = int(endpoint_index - cycle_start + 1)
        rows.append(
            {
                "cycle_index": cycle_index,
                "steps_in_cycle": steps,
                "cum_steps_end": int(endpoint_index + 1),
                "efficiency_32_over_steps": float(target_unique / steps),
                "cover_target_states": int(target_unique),
                "states_covered": int(target_unique),
                "complete_cycle": True,
                "nickname": nickname,
                "group": group,
            }
        )
        cycle_start = endpoint_index + 1
        seen = set()
    return pd.DataFrame(rows)


def _build_efficiency_examples(
    maze,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(FIRST_CHUNK_POINTS, dtype={"animal_id": str})
    naive = points.loc[points["group"].str.startswith("Naive")].copy()
    returns = points.loc[points["group"].str.startswith("LTM")].copy()
    candidates: list[dict[str, Any]] = []
    for animal, nsub in naive.groupby("animal_id"):
        rsub = returns.loc[returns["animal_id"].eq(animal)]
        if len(nsub) != 1 or rsub.empty:
            continue
        best = rsub.sort_values(["first_chunk_efficiency", "nickname"], ascending=[False, True]).iloc[0]
        nrow = nsub.iloc[0]
        candidates.append(
            {
                "animal_id": animal,
                "naive_session": str(nrow["nickname"]),
                "naive_efficiency": float(nrow["first_chunk_efficiency"]),
                "return_session": str(best["nickname"]),
                "return_efficiency": float(best["first_chunk_efficiency"]),
                "efficiency_ratio_return_over_naive": float(best["first_chunk_efficiency"] / nrow["first_chunk_efficiency"]),
            }
        )
    selection_audit = pd.DataFrame(candidates).sort_values(
        ["efficiency_ratio_return_over_naive", "animal_id"], ascending=[False, True]
    )
    chosen = selection_audit.iloc[0]
    sequence_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    cycle_frames: list[pd.DataFrame] = []
    for condition, session, expected in (
        ("redundant_naive", str(chosen["naive_session"]), float(chosen["naive_efficiency"])),
        ("efficient_return", str(chosen["return_session"]), float(chosen["return_efficiency"])),
    ):
        path = REPO / "mm-trajectories" / f"{session}-tf"
        trajectory = LoadTraj(path)
        full = _extract_explore_endnodes(trajectory, maze)
        cycles = _complete_exploration_cycles(
            full,
            target_unique=32,
            nickname=session,
            group=condition,
        )
        if cycles.empty:
            raise ValueError(f"Selected example {session} never reaches 32 unique endpoints")
        first_cycle = cycles.iloc[0]
        computed = float(first_cycle["efficiency_32_over_steps"])
        if not np.isclose(computed, expected, atol=1e-12, rtol=0):
            raise AssertionError(f"Example parity failed for {session}: {computed} versus {expected}")

        rows: list[dict[str, Any]] = []
        cycle_start = 0
        for cycle in cycles.itertuples(index=False):
            cycle_index = int(cycle.cycle_index)
            cycle_end = int(cycle.cum_steps_end)
            cycle_steps = int(cycle.steps_in_cycle)
            cycle_efficiency = float(cycle.efficiency_32_over_steps)
            cycle_sequence = np.asarray(full[cycle_start:cycle_end], dtype=int)
            if cycle_sequence.size != cycle_steps:
                raise AssertionError(
                    f"Cycle length mismatch for {session} cycle {cycle_index}: "
                    f"{cycle_sequence.size} versus {cycle_steps}"
                )

            seen: set[int] = set()
            for cycle_visit_index, node in enumerate(cycle_sequence, start=1):
                first_within_cycle = int(node) not in seen
                seen.add(int(node))
                cell = int(maze.ru[int(node)][-1])
                rows.append(
                    {
                        "condition": condition,
                        "session": session,
                        "animal_id": str(chosen["animal_id"]),
                        "cycle_index": cycle_index,
                        "cycle_visit_index": cycle_visit_index,
                        "endpoint_visit_index": cycle_start + cycle_visit_index,
                        "endpoint_run_node": int(node),
                        "endpoint_display_id_1_to_64": int(node - 62),
                        "physical_endpoint_cell": cell,
                        "endpoint_x": int(maze.xc[cell]),
                        "endpoint_y": int(maze.yc[cell]),
                        "is_first_visit_within_cycle": first_within_cycle,
                        "cumulative_unique_within_cycle": int(len(seen)),
                        "cycle_start_endpoint_visit_index": cycle_start + 1,
                        "cycle_end_endpoint_visit_index": cycle_end,
                        "cycle_steps": cycle_steps,
                        "cycle_efficiency_32_over_steps": cycle_efficiency,
                        "trajectory_source": str(path),
                    }
                )
            if len(seen) != 32:
                raise AssertionError(
                    f"Cycle {cycle_index} for {session} ends with {len(seen)} unique endpoints"
                )
            cycle_start = cycle_end

        sequence_frames.append(pd.DataFrame(rows))
        cycle_table = cycles.copy()
        cycle_table["condition"] = condition
        cycle_table["session"] = session
        cycle_table["animal_id"] = str(chosen["animal_id"])
        cycle_table["cycle_start_endpoint_visit_index"] = (
            cycle_table["cum_steps_end"] - cycle_table["steps_in_cycle"] + 1
        ).astype(int)
        cycle_table["cycle_end_endpoint_visit_index"] = cycle_table["cum_steps_end"].astype(int)
        cycle_frames.append(cycle_table)
        selected_rows.append(
            {
                "condition": condition,
                "session": session,
                "animal_id": str(chosen["animal_id"]),
                "n32": int(first_cycle["steps_in_cycle"]),
                "efficiency_32_over_n32": computed,
                "saved_efficiency": expected,
                "parity_delta": computed - expected,
                "n_complete_cycles": int(len(cycles)),
                "selection_rule": "same mouse with largest best-return / naive first-cycle efficiency ratio",
                "selection_ratio": float(chosen["efficiency_ratio_return_over_naive"]),
            }
        )
    selection_audit["selected"] = selection_audit["animal_id"].eq(str(chosen["animal_id"]))
    selection_audit.to_csv(OUT2 / "ed02_efficiency_example_selection_audit.csv", index=False)
    return (
        pd.concat(sequence_frames, ignore_index=True),
        pd.DataFrame(selected_rows),
        pd.concat(cycle_frames, ignore_index=True),
    )


def _render_efficiency_examples(
    sequence: pd.DataFrame,
    selection: pd.DataFrame,
    cycle_summary: pd.DataFrame,
) -> list[str]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.4, 3.8),
        gridspec_kw={"width_ratios": [1.55, 1.15, 0.82]},
    )
    conditions = ["redundant_naive", "efficient_return"]
    titles = ["Naive: redundant endpoint sampling", "Return: efficient endpoint sampling"]
    row_colors = [GROUP_COLORS["naive"], GROUP_COLORS["return"]]
    for row_index, (condition, title) in enumerate(zip(conditions, titles)):
        sub = sequence.loc[sequence["condition"].eq(condition)].copy()
        meta = selection.loc[selection["condition"].eq(condition)].iloc[0]
        cycles = (
            cycle_summary.loc[cycle_summary["condition"].eq(condition)]
            .sort_values("cycle_index")
            .reset_index(drop=True)
        )
        row_color = row_colors[row_index]
        ax = axes[row_index, 0]
        ax2 = axes[row_index, 1]
        for cycle in cycles.itertuples(index=False):
            start = float(cycle.cycle_start_endpoint_visit_index) - 0.5
            end = float(cycle.cycle_end_endpoint_visit_index) + 0.5
            if int(cycle.cycle_index) % 2 == 0:
                for band_ax in (ax, ax2):
                    band_ax.axvspan(start, end, color="#F1F1F1", zorder=0)
            if int(cycle.cycle_index) < int(cycles["cycle_index"].max()):
                for boundary_ax in (ax, ax2):
                    boundary_ax.axvline(end, color=GREY, lw=0.55, ls="--", zorder=1)

        repeat = sub.loc[~sub["is_first_visit_within_cycle"]]
        first = sub.loc[sub["is_first_visit_within_cycle"]]
        ax.scatter(
            repeat["endpoint_visit_index"],
            repeat["endpoint_display_id_1_to_64"],
            s=3.5,
            color="#B9B9B9",
            alpha=0.48,
            linewidth=0,
            label="repeat within cycle",
        )
        ax.scatter(
            first["endpoint_visit_index"],
            first["endpoint_display_id_1_to_64"],
            s=7.5,
            color=BLUE,
            edgecolor="white",
            linewidth=0.25,
            label="first visit within cycle",
        )
        ax.set_xlabel("Endpoint visit index (whole session)")
        ax.set_ylabel("Terminal endpoint (1–64)")
        ax.set_ylim(0, 65)
        ax.set_xlim(0.5, float(sub["endpoint_visit_index"].max()) + 0.5)
        ax.set_title(f"{title}\n{meta['session']} (same mouse)", loc="left", fontweight="bold", pad=2.0)
        ax.grid(color=GRID, lw=0.35)
        if row_index == 0:
            ax.legend(
                loc="lower right",
                bbox_to_anchor=(1.0, 1.005),
                ncol=2,
                handletextpad=0.35,
                columnspacing=0.8,
                borderaxespad=0,
            )

        for cycle in cycles.itertuples(index=False):
            cycle_sub = sub.loc[sub["cycle_index"].eq(int(cycle.cycle_index))]
            ax2.step(
                cycle_sub["endpoint_visit_index"],
                cycle_sub["cumulative_unique_within_cycle"],
                where="post",
                color=row_color,
                lw=1.05,
            )
            midpoint = (
                float(cycle.cycle_start_endpoint_visit_index)
                + float(cycle.cycle_end_endpoint_visit_index)
            ) / 2.0
            ax2.text(
                midpoint,
                33.1,
                f"C{int(cycle.cycle_index)}",
                ha="center",
                va="bottom",
                fontsize=4.8,
                color=AXIS,
            )
            if int(cycle.cycle_index) < int(cycles["cycle_index"].max()):
                boundary = float(cycle.cycle_end_endpoint_visit_index) + 0.5
                ax2.plot([boundary, boundary], [32, 0], color=row_color, lw=0.9, zorder=3)
                ax2.text(
                    boundary,
                    1.0,
                    "reset",
                    ha="left",
                    va="bottom",
                    rotation=90,
                    fontsize=4.3,
                    color=GREY,
                )
        ax2.axhline(32, color=GREY, lw=0.6, ls="--")
        ax2.set_xlabel("Endpoint visit index (whole session)")
        ax2.set_ylabel("Unique endpoints in current cycle")
        ax2.set_xlim(0.5, float(sub["endpoint_visit_index"].max()) + 0.5)
        ax2.set_ylim(0, 35.5)
        ax2.set_title("Coverage resets after 32 unique endpoints", loc="left", pad=2.0)
        ax2.grid(color=GRID, lw=0.35)

        ax3 = axes[row_index, 2]
        x = cycles["cycle_index"].to_numpy(float)
        y = cycles["efficiency_32_over_steps"].to_numpy(float)
        ax3.plot(x, y, color=row_color, lw=0.9, alpha=0.72, zorder=1)
        ax3.scatter(
            x[1:],
            y[1:],
            s=16,
            facecolor="white",
            edgecolor=row_color,
            linewidth=0.8,
            zorder=2,
            label="Cycles 2+ (Extended Data)",
        )
        ax3.scatter(
            x[:1],
            y[:1],
            s=22,
            facecolor=row_color,
            edgecolor=AXIS,
            linewidth=0.7,
            zorder=3,
            label="Cycle 1 (main text)",
        )
        for cycle_index, efficiency, steps in zip(
            cycles["cycle_index"],
            cycles["efficiency_32_over_steps"],
            cycles["steps_in_cycle"],
        ):
            ax3.annotate(
                f"N={int(steps)}",
                (float(cycle_index), float(efficiency)),
                xytext=(0, 5 if int(cycle_index) % 2 else -8),
                textcoords="offset points",
                ha="center",
                va="bottom" if int(cycle_index) % 2 else "top",
                fontsize=4.4,
                color=AXIS,
            )
        ax3.set_xlabel("32-unique-endpoint cycle")
        ax3.set_ylabel(r"Efficiency, $32/N_c$")
        ax3.set_xticks(x.astype(int))
        ax3.set_xlim(0.65, float(x.max()) + 0.35)
        ax3.set_ylim(0, 0.56)
        ax3.set_title(r"Repeated-cycle efficiency: $E_c=32/N_c$", loc="left", pad=2.0)
        ax3.grid(color=GRID, lw=0.35)
    fig.text(
        0.50,
        0.995,
        "Same estimator in every non-overlapping cycle: reach 32 unique endpoints, reset, and repeat",
        ha="center",
        va="top",
        fontsize=6.2,
        fontweight="bold",
        color=AXIS,
    )
    fig.text(
        0.995,
        0.955,
        "Filled point = cycle 1 (main text); open points = subsequent cycles (Extended Data)",
        ha="right",
        va="top",
        fontsize=4.8,
        color=AXIS,
    )
    fig.text(
        0.995,
        0.008,
        "Real same-mouse examples selected by a prespecified rule; all complete cycles shown; no inference",
        ha="right",
        va="bottom",
        fontsize=4.7,
        color=GREY,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.14, top=0.90, hspace=0.56, wspace=0.34)
    return _save(fig, OUT2, "ed02_real_32_over_n32_efficiency_examples")


def _render_cycle_support() -> tuple[list[str], pd.DataFrame, int]:
    summary = pd.read_csv(CYCLE_SUMMARY)
    summary = summary.loc[summary["cycle_index"].le(15)].copy()
    first = summary.loc[summary["cycle_index"].eq(1)].set_index("group_key")["n_sessions"].astype(int)
    summary["initial_n_sessions"] = summary["group_key"].map(first)
    summary["support_fraction"] = summary["n_sessions"] / summary["initial_n_sessions"]
    summary["min_20pct_n"] = np.ceil(0.20 * summary["initial_n_sessions"]).astype(int)
    summary["passes_20pct_support"] = summary["n_sessions"].ge(summary["min_20pct_n"])
    group_keys = ["naive", "return", "experienced_day2"]
    supported_cycles: list[int] = []
    for cycle in sorted(summary["cycle_index"].unique()):
        sub = summary.loc[summary["cycle_index"].eq(cycle) & summary["group_key"].isin(group_keys)]
        if len(sub) == 3 and sub["passes_20pct_support"].all():
            supported_cycles.append(int(cycle))
    cutoff = max(supported_cycles)
    summary["inside_common_20pct_window"] = summary["cycle_index"].le(cutoff)

    fig = plt.figure(figsize=(3.75, 2.75))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    label_map = {"naive": "Naive day", "return": "Return day", "experienced_day2": "Second day"}
    for group in group_keys:
        sub = summary.loc[summary["group_key"].eq(group) & summary["cycle_index"].le(cutoff)].sort_values("cycle_index")
        x = sub["cycle_index"].to_numpy(float)
        y = sub["eff_mean"].to_numpy(float)
        sem = sub["eff_sem"].to_numpy(float)
        ax.plot(x, y, color=GROUP_COLORS[group], marker="o", ms=2.3, lw=1.0, label=f"{label_map[group]} (n1={int(first[group])})")
        ax.fill_between(x, y - sem, y + sem, color=GROUP_COLORS[group], alpha=0.11, lw=0)
    ax.set_xlim(1, cutoff)
    ax.set_ylabel(r"Exploration efficiency ($32/N_{32}$)")
    ax.set_title("Successive coverage cycles with explicit support", loc="left", fontweight="bold", pad=2.0)
    ax.legend(loc="best")
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.tick_params(labelbottom=False)

    tax = fig.add_subplot(grid[1, 0], sharex=ax)
    tax.set_ylim(-0.5, 2.5)
    tax.set_yticks([2, 1, 0], [label_map[group] for group in group_keys])
    tax.set_xlabel("Successive 32-endpoint coverage cycle")
    tax.set_xticks(range(1, cutoff + 1))
    tax.spines[["top", "right", "left"]].set_visible(False)
    tax.tick_params(axis="y", length=0)
    for row_index, group in enumerate(group_keys[::-1]):
        sub = summary.loc[summary["group_key"].eq(group) & summary["cycle_index"].le(cutoff)].set_index("cycle_index")
        for cycle in range(1, cutoff + 1):
            tax.text(cycle, row_index, str(int(sub.loc[cycle, "n_sessions"])), ha="center", va="center", fontsize=4.8, color=GROUP_COLORS[group])
    tax.set_title("n at risk", loc="left", fontsize=4.8, fontweight="bold", pad=1.0)
    fig.text(0.99, 0.01, f"Displayed through cycle {cutoff}: every group retains ≥20% of cycle-1 sessions", ha="right", va="bottom", fontsize=4.5, color=GREY)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.17, top=0.91)
    outputs = _save(fig, OUT2, "ed02_exploration_cycles_with_n_at_risk_current_plus_new12")
    return outputs, summary, cutoff


def _select_lb0_example(exact: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    eligible = exact.loc[exact["group_key"].eq("return") & exact["load_ok"].astype(bool)].copy()
    eligible["log_tiles"] = np.log1p(pd.to_numeric(eligible["nose_tiles_to_first_long_bout"], errors="coerce"))
    eligible["log_time"] = np.log1p(pd.to_numeric(eligible["lb0_time_in_maze_s"], errors="coerce"))
    for column in ("log_tiles", "log_time"):
        median = float(eligible[column].median())
        scale = float(eligible[column].std(ddof=0))
        eligible[f"z_{column}"] = (eligible[column] - median) / scale
    eligible["selection_distance"] = np.hypot(eligible["z_log_tiles"], eligible["z_log_time"])
    eligible = eligible.sort_values(["selection_distance", "nickname"]).reset_index(drop=True)
    eligible["selected"] = False
    eligible.loc[0, "selected"] = True
    eligible["selection_rule"] = "return session closest to bivariate median of log1p(lb0 tiles) and log1p(lb0 time)"
    return eligible.iloc[0], eligible


def _path_xy_with_gaps(cells: np.ndarray, maze, max_step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cells, dtype=int)
    xs: list[float] = []
    ys: list[float] = []
    prior: int | None = None
    for cell in cells:
        if prior is not None and float(maze.di[int(prior), int(cell)]) > max_step:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(float(maze.xc[int(cell)]))
        ys.append(float(maze.yc[int(cell)]))
        prior = int(cell)
    return np.asarray(xs), np.asarray(ys)


def _build_lb0_example(exact: pd.DataFrame, events: pd.DataFrame, metrics: pd.DataFrame, maze) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, audit = _select_lb0_example(exact)
    session = str(selected["nickname"])
    first_reward = (
        events.loc[events["session"].astype(str).eq(session) & events["event_type"].eq("reward")]
        .sort_values(["maze_seconds", "abs_frame"])
        .iloc[0]
    )
    metric_row = metrics.loc[metrics["nickname"].astype(str).eq(session)].iloc[0]
    trajectory_path = REPO / "mm-trajectories" / f"{session}-tf"
    trajectory = LoadTraj(trajectory_path)
    paths = np.asarray(FindPathsToNode(94, trajectory, maze), dtype=float)
    paths = paths[paths[:, 2] >= 7]
    if paths.size == 0:
        raise ValueError(f"Selected LB0 example {session} has no qualifying long path")
    start_times = _frame_to_maze_seconds(paths[:, 3], trajectory)
    path_row = paths[int(np.argmin(start_times))]
    bout_index = int(path_row[0])
    start_index = int(path_row[1])
    node_distance = int(path_row[2])
    long_bout_start_frame = float(path_row[3])
    bout_nodes = np.asarray(trajectory.no[bout_index])
    end_index = start_index + node_distance
    if end_index >= len(bout_nodes):
        raise IndexError(f"Qualifying long-bout end index is outside trajectory bout for {session}")
    long_bout_end_frame = float(bout_nodes[end_index, 1] + trajectory.fr[bout_index, 0])
    long_bout_end_s = float(_frame_to_maze_seconds(np.asarray([long_bout_end_frame]), trajectory)[0])
    if not np.isclose(long_bout_start_frame, float(selected["first_long_bout_abs_frame"]), atol=0, rtol=0):
        raise AssertionError("Recovered first-long-bout start frame differs from the exact ED2 table")
    stop_frame = max(float(first_reward["abs_frame"]), long_bout_end_frame)
    prefix = _read_nose_prefix(Path(str(selected["nose_traversal_source"])), stop_frame)
    cells = _corrected_nose_cells(prefix, maze)
    deduped, frames = _dedupe_valid_with_frames(cells, prefix["frame"].to_numpy(float))
    maze_seconds = _frame_to_maze_seconds(frames, trajectory)
    trace = pd.DataFrame(
        {
            "session": session,
            "transition_event_index": np.arange(deduped.size, dtype=int),
            "physical_maze_cell": deduped,
            "x_grid": maze.xc[deduped],
            "y_grid": maze.yc[deduped],
            "abs_frame": frames,
            "maze_seconds": maze_seconds,
            "cumulative_tile_transitions": np.arange(deduped.size, dtype=int),
            "before_first_long_bout_onset": frames <= long_bout_start_frame,
            "during_first_long_bout": (frames >= long_bout_start_frame) & (frames <= long_bout_end_frame),
            "first_long_bout_abs_frame": float(selected["first_long_bout_abs_frame"]),
            "first_long_bout_maze_s": float(selected["lb0_time_in_maze_s"]),
            "first_long_bout_end_abs_frame": long_bout_end_frame,
            "first_long_bout_end_maze_s": long_bout_end_s,
            "first_reward_abs_frame": float(first_reward["abs_frame"]),
            "first_reward_maze_s": float(first_reward["maze_seconds"]),
            "nose_source": str(selected["nose_traversal_source"]),
            "trajectory_source": str(trajectory_path),
        }
    )
    onset_count = int(trace.loc[trace["abs_frame"].le(float(selected["first_long_bout_abs_frame"])), "cumulative_tile_transitions"].max())
    reward_count = int(
        trace.loc[
            trace["abs_frame"].le(float(first_reward["abs_frame"])),
            "cumulative_tile_transitions",
        ].max()
    )
    long_bout_end_count = int(
        trace.loc[
            trace["abs_frame"].le(long_bout_end_frame),
            "cumulative_tile_transitions",
        ].max()
    )
    expected_onset = int(float(selected["nose_tiles_to_first_long_bout"]))
    expected_reward = int(float(metric_row["nose_tiles_to_first_reward"]))
    if onset_count != expected_onset or reward_count != expected_reward:
        raise AssertionError(
            f"LB0 example parity failed for {session}: onset {onset_count}/{expected_onset}, reward {reward_count}/{expected_reward}"
        )
    selection = pd.DataFrame(
        [
            {
                "session": session,
                "animal_id": str(selected["animal_id"]),
                "month": int(selected["month"]),
                "group_key": str(selected["group_key"]),
                "first_long_bout_abs_frame": float(selected["first_long_bout_abs_frame"]),
                "first_long_bout_maze_s": float(selected["lb0_time_in_maze_s"]),
                "tile_transitions_to_first_long_bout": onset_count,
                "first_reward_abs_frame": float(first_reward["abs_frame"]),
                "first_reward_maze_s": float(first_reward["maze_seconds"]),
                "tile_transitions_to_first_reward": reward_count,
                "first_long_bout_end_abs_frame": long_bout_end_frame,
                "first_long_bout_end_maze_s": long_bout_end_s,
                "tile_transitions_to_first_long_bout_end": long_bout_end_count,
                "long_path_min": 7,
                "canonical_reward_run_node": 94,
                "selection_rule": audit.loc[audit["selected"], "selection_rule"].iloc[0],
                "selection_distance": float(selected["selection_distance"]),
                "illustrative_only": True,
                "nose_source": str(selected["nose_traversal_source"]),
                "trajectory_source": str(trajectory_path),
            }
        ]
    )
    audit.to_csv(OUT2 / "ed02_first_long_bout_example_selection_audit.csv", index=False)
    return trace, selection


def _render_lb0_example(trace: pd.DataFrame, selection: pd.DataFrame, maze) -> list[str]:
    meta = selection.iloc[0]
    onset = float(meta["first_long_bout_maze_s"])
    reward = float(meta["first_reward_maze_s"])
    bout_end = float(meta["first_long_bout_end_maze_s"])
    before = trace.loc[trace["before_first_long_bout_onset"]]
    during = trace.loc[trace["during_first_long_bout"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
    ax = axes[0]
    wall = np.asarray(maze.wa, dtype=float)
    ax.plot(wall[:, 0], wall[:, 1], color="#C7C7C7", lw=0.45, zorder=0)
    x1, y1 = _path_xy_with_gaps(before["physical_maze_cell"].to_numpy(int), maze)
    x2, y2 = _path_xy_with_gaps(during["physical_maze_cell"].to_numpy(int), maze)
    ax.plot(x1, y1, color="#7F7F7F", lw=0.55, alpha=0.80, label="before long-bout onset")
    if len(during):
        ax.plot(x2, y2, color=ACCENT, lw=0.85, alpha=0.90, label="first long bout to reward")
    first_cell = int(trace.iloc[0]["physical_maze_cell"])
    onset_row = trace.loc[trace["before_first_long_bout_onset"]].iloc[-1]
    first_reward_row = trace.loc[trace["abs_frame"].le(float(meta["first_reward_abs_frame"]))].iloc[-1]
    bout_end_row = trace.loc[trace["abs_frame"].le(float(meta["first_long_bout_end_abs_frame"]))].iloc[-1]
    ax.plot(maze.xc[first_cell], maze.yc[first_cell], "o", ms=3.8, color=BLUE, markeredgecolor="white", markeredgewidth=0.35, label="start")
    ax.plot(onset_row["x_grid"], onset_row["y_grid"], "D", ms=3.8, color=ACCENT, markeredgecolor="white", markeredgewidth=0.35, label="long-bout onset")
    ax.plot(first_reward_row["x_grid"], first_reward_row["y_grid"], "*", ms=5.2, color="#D98C00", markeredgecolor="white", markeredgewidth=0.35, label="first reward")
    ax.plot(bout_end_row["x_grid"], bout_end_row["y_grid"], "P", ms=4.4, color=ACCENT, markeredgecolor="white", markeredgewidth=0.35, label="reward completing long bout")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_title(f"Real session {meta['session']}: spatial trace", loc="left", fontweight="bold", pad=2.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        fontsize=4.5,
        columnspacing=0.6,
        handletextpad=0.3,
    )

    ax = axes[1]
    ax.step(trace["maze_seconds"] / 60.0, trace["cumulative_tile_transitions"], where="post", color=AXIS, lw=1.0)
    ax.axvline(onset / 60.0, color=ACCENT, lw=0.9, ls="--")
    ax.axvline(reward / 60.0, color="#D98C00", lw=0.9, ls=":")
    ax.axvline(bout_end / 60.0, color=ACCENT, lw=0.8, ls=":")
    ax.plot(onset / 60.0, int(meta["tile_transitions_to_first_long_bout"]), "D", ms=3.8, color=ACCENT)
    ax.plot(reward / 60.0, int(meta["tile_transitions_to_first_reward"]), "*", ms=5.2, color="#D98C00")
    ax.plot(bout_end / 60.0, int(meta["tile_transitions_to_first_long_bout_end"]), "P", ms=4.4, color=ACCENT)
    ax.annotate(
        f"first long-bout onset\n{onset / 60.0:.1f} min, {int(meta['tile_transitions_to_first_long_bout'])} transitions",
        (onset / 60.0, int(meta["tile_transitions_to_first_long_bout"])),
        xytext=(4, -5),
        textcoords="offset points",
        fontsize=4.6,
        color=ACCENT,
        va="top",
    )
    ax.annotate(
        f"first reward\n{reward / 60.0:.1f} min, {int(meta['tile_transitions_to_first_reward'])} transitions",
        (reward / 60.0, int(meta["tile_transitions_to_first_reward"])),
        xytext=(-4, 5),
        textcoords="offset points",
        fontsize=4.6,
        color="#9A6500",
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("Maze time (min)")
    ax.set_ylabel("Cumulative corrected-nose tile transitions")
    ax.set_title("First-long-bout endpoint is not first reward", loc="left", fontweight="bold", pad=2.0)
    ax.grid(color=GRID, lw=0.4)
    fig.text(0.99, 0.01, "Illustrative only; onset defined by first reward-directed path of ≥7 nodes toward canonical reward run node 94", ha="right", va="bottom", fontsize=4.5, color=GREY)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.24, top=0.90, wspace=0.24)
    return _save(fig, OUT2, "ed02_real_first_long_bout_onset_trace")


def _write_reports(
    *,
    accumulation: dict[str, Any],
    endpoint_stats: pd.DataFrame,
    lb0_stats: pd.DataFrame,
    coverage: pd.DataFrame,
    coverage_stats: pd.DataFrame,
    cycle_cutoff: int,
    efficiency_selection: pd.DataFrame,
    efficiency_cycles: pd.DataFrame,
    lb0_selection: pd.DataFrame,
) -> None:
    ed1_lines = [
        "# ED1 month/source/coverage gap closure",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: current return-month reward and long-bout accumulation can be inspected without pooling 3, 6, 9, and 12 months, while the event-to-session-to-group conversion and its coverage mask remain auditable.",
        "- Archetype: quantitative grid with source-conversion audit.",
        "- Backend/export: Python/matplotlib only; Arial-compatible editable SVG/PDF plus 600-dpi PNG/TIFF.",
        "- Statistics: eligible session rows retained; CR1 covariance clustered by mouse; two-sided Student t with G-1 degrees of freedom; BH within explicitly named families.",
        "- Reviewer risk addressed: return months are not pooled; common time support is visible; transitions are not relabeled as unique coverage; exposure-time adjustment is shown separately.",
        "",
        "## Current scope and conversion parity",
        "",
        f"- Sessions: 153; event rows: 33,001; common 80%-coverage display window: {accumulation['global_window'] / 60.0:.1f} min.",
        f"- Rebuilt group-curve rows: {len(accumulation['parity']):,}; every mean, SEM, active-session count, averaged-session count, and coverage flag matches the maintained current153 table to 1e-12.",
        "- The exported per-session curve table contains every 10-s display-window conversion, the carried-forward cumulative count, the duration-masked active count, and the session-in-window mask.",
        "",
        "## Statistical policy correction",
        "",
        "The replacement first-long-bout panel displays session rows and uses mouse-clustered inference. Complete-animal compression is not the primary estimand.",
        "",
    ]
    for _, row in lb0_stats.iterrows():
        ed1_lines.append(
            f"- {row['contrast']}: beta(log s)={row['beta_target_minus_baseline']:.4f}, p={row['p_cluster']:.4g}, q(BH3)={row['q_bh3']:.4g}, n={int(row['n_rows'])} rows/{int(row['n_clusters'])} mice."
        )
    ed1_lines.extend(
        [
            "",
            "## Unique coverage before first reward",
            "",
            f"- Recomputed sessions: {len(coverage)}; exact corrected-nose transition-count parity: {(coverage['transition_count_delta'].abs() <= 1e-9).sum()}/{len(coverage)}; exact first-reward time parity: {(np.abs(coverage['first_reward_maze_s'] - coverage['saved_ti1_s']) <= 1e-9).sum()}/{len(coverage)}.",
            "- Unique physical maze cells and unique terminal endpoint cells are distinct from tile-transition counts. Raw and log-time-adjusted month-versus-naive models are exported separately.",
            "- The time-adjusted specification conditions on log(1 + time to first reward); it is an exposure diagnostic, not a causal mediation model.",
            "",
            "## Files",
            "",
            "- `ed01_reward_accumulation_by_return_month_current153.*`",
            "- `ed01_long_bout_accumulation_by_return_month_current153.*`",
            "- `ed01_row_retaining_first_long_bout_current153.*`",
            "- `ed01_unique_spatial_coverage_before_first_reward_current153.*`",
            "- `ed01_unique_coverage_time_adjusted_current153.*`",
            "- `ed01_accumulation_event_rows_current153.csv`",
            "- `ed01_accumulation_session_inclusion_current153.csv`",
            "- `ed01_accumulation_session_curves_display_window_current153.csv.gz`",
            "- `ed01_accumulation_group_curve_parity_current153.csv.gz`",
            "",
        ]
    )
    (OUT1 / "ED01_GAP_CLOSURE_REPORT.md").write_text("\n".join(ed1_lines), encoding="utf-8")

    eff_naive = efficiency_selection.loc[efficiency_selection["condition"].eq("redundant_naive")].iloc[0]
    eff_return = efficiency_selection.loc[efficiency_selection["condition"].eq("efficient_return")].iloc[0]
    example = lb0_selection.iloc[0]
    ed2_lines = [
        "# ED2 illustrative/support gap closure",
        "",
        "## Figure contract",
        "",
        "- Core conclusion: the reader can see how the main-text 32/N estimator is reapplied after each 32-unique-endpoint reset, how repeated visits lower each cycle's efficiency, how the first-long-bout onset differs from first reward, and where population cycle curves lose support.",
        "- Archetype: schematic-led real-data examples plus a support-aware quantitative curve.",
        "- Backend/export: Python/matplotlib only; Arial-compatible editable SVG/PDF plus 600-dpi PNG/TIFF.",
        "- Examples are qualitative and provenance-linked; no inferential claim is made from their selection.",
        "",
        "## 32/N32 worked example",
        "",
        f"- Same mouse {eff_naive['animal_id']} was selected by the prespecified largest best-return/naive efficiency ratio.",
        f"- Redundant session {eff_naive['session']}: N32={int(eff_naive['n32'])}, 32/N32={eff_naive['efficiency_32_over_n32']:.3f}.",
        f"- Efficient session {eff_return['session']}: N32={int(eff_return['n32'])}, 32/N32={eff_return['efficiency_32_over_n32']:.3f}.",
        "- Both recomputed values exactly match the current first-chunk source table.",
        "- The schematic then continues through every complete non-overlapping cycle; endpoint novelty is reset within each cycle, not accumulated globally.",
        "- Cycle-level N and 32/Nc values are exported in `ed02_efficiency_example_cycle_summary.csv`.",
        "",
        "## First-long-bout onset example",
        "",
        f"- Deterministically selected median-like return session: {example['session']} ({int(example['month'])} months).",
        f"- First long-bout onset: {example['first_long_bout_maze_s']:.2f} s and {int(example['tile_transitions_to_first_long_bout'])} tile transitions.",
        f"- First reward: {example['first_reward_maze_s']:.2f} s and {int(example['tile_transitions_to_first_reward'])} tile transitions.",
        "- Both transition counts exactly match the current first-long-bout and first-reward tables.",
        "",
        "## Cycle support",
        "",
        f"- The support-aware curve stops at cycle {cycle_cutoff}, the last cycle at which every group retains at least 20% of its cycle-1 sessions.",
        "- The exact n at risk is printed under every displayed cycle; later saved cycles remain in the source table but are not foregrounded.",
        "",
        "## Files",
        "",
        "- `ed02_real_32_over_n32_efficiency_examples.*`",
        "- `ed02_real_first_long_bout_onset_trace.*`",
        "- `ed02_exploration_cycles_with_n_at_risk_current_plus_new12.*`",
        "- source and selection-audit CSVs in this folder.",
        "",
    ]
    (OUT2 / "ED02_GAP_CLOSURE_REPORT.md").write_text("\n".join(ed2_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute-unique-coverage",
        action="store_true",
        help="Ignore a complete 153-row unique-coverage cache and reread the external nose sources.",
    )
    parser.add_argument(
        "--only-efficiency-example",
        action="store_true",
        help="Rebuild only the repeated-cycle 32/Nc worked-example panel and its source tables.",
    )
    args = parser.parse_args()
    _style()
    OUT2.mkdir(parents=True, exist_ok=True)
    maze = NewMaze(6)

    if args.only_efficiency_example:
        sequence, efficiency_selection, efficiency_cycles = _build_efficiency_examples(maze)
        sequence_path = OUT2 / "ed02_efficiency_example_endpoint_sequence.csv"
        selection_path = OUT2 / "ed02_efficiency_example_selected_sessions.csv"
        cycles_path = OUT2 / "ed02_efficiency_example_cycle_summary.csv"
        sequence.to_csv(sequence_path, index=False)
        efficiency_selection.to_csv(selection_path, index=False)
        efficiency_cycles.to_csv(cycles_path, index=False)
        outputs = _render_efficiency_examples(sequence, efficiency_selection, efficiency_cycles)
        manifest = {
            "analysis": "ED2 real repeated-cycle 32/Nc estimator schematic",
            "scope": "current_plus_new12 real same-mouse examples",
            "figure_contract": {
                "core_conclusion": (
                    "The Extended Data curve reapplies the main-text 32/N estimator to consecutive "
                    "non-overlapping cycles, resetting after each set of 32 unique endpoints."
                ),
                "evidence_chain": (
                    "full endpoint sequence -> within-cycle first/repeat labels -> reset-aware coverage "
                    "trace -> per-cycle 32/Nc"
                ),
                "archetype": "schematic-led real-data composite",
                "backend": "Python/matplotlib only",
                "exports": "editable Arial-compatible SVG/PDF plus 600-dpi PNG/TIFF and source CSVs",
                "review_risks": [
                    "global cumulative-unique interpretation",
                    "global-versus-within-cycle new-endpoint ambiguity",
                    "hidden reset boundaries",
                    "first-cycle metric drift from the main-text source",
                ],
            },
            "efficiency_examples": efficiency_selection.to_dict(orient="records"),
            "cycle_rows": efficiency_cycles.to_dict(orient="records"),
            "inputs": {str(FIRST_CHUNK_POINTS): _sha256(FIRST_CHUNK_POINTS)},
            "source_outputs": [str(sequence_path), str(selection_path), str(cycles_path)],
            "figure_outputs": outputs,
        }
        _write_json(manifest, OUT2 / "ed02_efficiency_example_manifest.json")
        print(json.dumps(manifest, indent=2, default=str))
        return

    OUT1.mkdir(parents=True, exist_ok=True)

    events, sessions, saved = _load_accumulation_sources()
    accumulation = _rebuild_accumulation_conversion(events, sessions, saved)
    events.to_csv(OUT1 / "ed01_accumulation_event_rows_current153.csv", index=False)
    sessions.to_csv(OUT1 / "ed01_accumulation_session_inclusion_current153.csv", index=False)
    accumulation["session_curves"].to_csv(
        OUT1 / "ed01_accumulation_session_curves_display_window_current153.csv.gz",
        index=False,
        compression="gzip",
    )
    accumulation["coverage"].to_csv(OUT1 / "ed01_accumulation_coverage_masks_current153.csv", index=False)
    accumulation["parity"].to_csv(
        OUT1 / "ed01_accumulation_group_curve_parity_current153.csv.gz",
        index=False,
        compression="gzip",
    )
    accumulation["endpoint_rows"].to_csv(OUT1 / "ed01_accumulation_common_endpoint_session_rows_current153.csv", index=False)
    accumulation["endpoint_stats"].to_csv(OUT1 / "ed01_accumulation_common_endpoint_cluster_stats_current153.csv", index=False)

    outputs1: list[str] = []
    outputs1.extend(
        _render_month_curve(
            saved,
            "reward",
            "Cumulative rewards",
            "Reward accumulation by return interval",
            "ed01_reward_accumulation_by_return_month_current153",
        )
    )
    outputs1.extend(
        _render_month_curve(
            saved,
            "long_path_reward",
            "Cumulative long reward-path bouts",
            "Long-bout accumulation by return interval",
            "ed01_long_bout_accumulation_by_return_month_current153",
        )
    )

    metrics = pd.read_csv(REWARD_METRICS, dtype={"animal_id": str})
    lb0_outputs, lb0_stats, lb0_summary = _render_row_retaining_lb0(metrics)
    outputs1.extend(lb0_outputs)
    metrics.to_csv(OUT1 / "ed01_first_long_bout_session_rows_current153.csv", index=False)
    lb0_stats.to_csv(OUT1 / "ed01_first_long_bout_cluster_stats_current153.csv", index=False)
    lb0_summary.to_csv(OUT1 / "ed01_first_long_bout_cluster_summary_current153.csv", index=False)

    coverage = _compute_unique_coverage(
        metrics,
        events,
        maze,
        recompute=bool(args.recompute_unique_coverage),
    )
    coverage_stats = _unique_coverage_stats(coverage)
    coverage_stats.to_csv(OUT1 / "ed01_unique_coverage_cluster_stats_current153.csv", index=False)
    outputs1.extend(_render_unique_coverage(coverage, coverage_stats))
    outputs1.extend(_render_unique_coverage_forest(coverage_stats))

    sequence, efficiency_selection, efficiency_cycles = _build_efficiency_examples(maze)
    sequence.to_csv(OUT2 / "ed02_efficiency_example_endpoint_sequence.csv", index=False)
    efficiency_selection.to_csv(OUT2 / "ed02_efficiency_example_selected_sessions.csv", index=False)
    efficiency_cycles.to_csv(OUT2 / "ed02_efficiency_example_cycle_summary.csv", index=False)
    outputs2 = _render_efficiency_examples(sequence, efficiency_selection, efficiency_cycles)

    cycle_outputs, cycle_support, cycle_cutoff = _render_cycle_support()
    outputs2.extend(cycle_outputs)
    cycle_support.to_csv(OUT2 / "ed02_cycle_n_at_risk_current_plus_new12.csv", index=False)

    exact = pd.read_csv(EXACT_LB0, dtype={"animal_id": str})
    lb0_trace, lb0_selection = _build_lb0_example(exact, events, metrics, maze)
    lb0_trace.to_csv(OUT2 / "ed02_first_long_bout_example_transition_trace.csv", index=False)
    lb0_selection.to_csv(OUT2 / "ed02_first_long_bout_example_selected_session.csv", index=False)
    outputs2.extend(_render_lb0_example(lb0_trace, lb0_selection, maze))

    _write_reports(
        accumulation=accumulation,
        endpoint_stats=accumulation["endpoint_stats"],
        lb0_stats=lb0_stats,
        coverage=coverage,
        coverage_stats=coverage_stats,
        cycle_cutoff=cycle_cutoff,
        efficiency_selection=efficiency_selection,
        efficiency_cycles=efficiency_cycles,
        lb0_selection=lb0_selection,
    )

    manifest1 = {
        "analysis": "ED1 month-resolved accumulation, source conversion, row-retaining stats, and unique coverage",
        "scope": "current_plus_new12 current153",
        "figure_contract": {
            "core_conclusion": "Return-month curves and unique pre-reward coverage are shown with exact source conversion and support masks.",
            "archetype": "quantitative grid with source-conversion audit",
            "backend": "Python/matplotlib only",
            "exports": "editable Arial-compatible SVG/PDF plus 600-dpi PNG/TIFF",
        },
        "global_80pct_coverage_window_seconds": accumulation["global_window"],
        "n_sessions": len(sessions),
        "n_event_rows": len(events),
        "n_session_curve_rows_display_window": len(accumulation["session_curves"]),
        "group_curve_parity_all_pass": bool(accumulation["parity"]["passes"].all()),
        "unique_coverage_transition_parity_n": int(coverage["transition_count_delta"].abs().le(1e-9).sum()),
        "unique_coverage_n": len(coverage),
        "statistics": "session rows; CR1 mouse clusters; two-sided t(G-1); BH within stated families",
        "inputs": {str(path): _sha256(path) for path in [CURRENT_EVENTS, CURRENT_SESSIONS, NEW_EVENTS, NEW_SESSIONS, SAVED_GROUP_CURVES, REWARD_METRICS]},
        "figure_outputs": outputs1,
    }
    _write_json(manifest1, OUT1 / "ed01_gap_closure_manifest.json")

    manifest2 = {
        "analysis": "ED2 real estimand examples and cycle support",
        "scope": "current_plus_new12",
        "figure_contract": {
            "core_conclusion": (
                "Real sequences show that the main-text 32/N estimator is reset and reapplied across "
                "consecutive non-overlapping cycles; first-long-bout onset is separately defined, and "
                "n at risk bounds late-cycle interpretation."
            ),
            "evidence_chain": (
                "full endpoint sequence -> within-cycle first/repeat labels -> reset-aware coverage "
                "trace -> per-cycle 32/Nc -> population support curve"
            ),
            "archetype": "schematic-led real-data examples plus support-aware quantitative curve",
            "backend": "Python/matplotlib only",
            "exports": "editable Arial-compatible SVG/PDF plus 600-dpi PNG/TIFF",
            "review_risks": [
                "global cumulative-unique interpretation",
                "global-versus-within-cycle new-endpoint ambiguity",
                "hidden reset boundaries",
                "first-cycle metric drift from the main-text source",
            ],
        },
        "cycle_common_20pct_support_cutoff": cycle_cutoff,
        "efficiency_examples": efficiency_selection.to_dict(orient="records"),
        "efficiency_example_cycles": efficiency_cycles.to_dict(orient="records"),
        "first_long_bout_example": lb0_selection.to_dict(orient="records"),
        "inputs": {str(path): _sha256(path) for path in [FIRST_CHUNK_POINTS, CYCLE_SUMMARY, EXACT_LB0]},
        "figure_outputs": outputs2,
    }
    _write_json(manifest2, OUT2 / "ed02_gap_closure_manifest.json")
    print(json.dumps({"ed01": manifest1, "ed02": manifest2}, indent=2, default=str))


if __name__ == "__main__":
    main()
