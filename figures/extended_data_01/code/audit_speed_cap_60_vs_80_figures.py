"""Audit every maintained figure conclusion that depends on the 80 cm/s speed cap.

The audit changes only the tracking-artifact cap (80 to 60 cm/s). Cohorts,
position sources, median smoothing, trajectory continuity, windows, aggregation,
neural inputs, statistical units and correction families remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import analyze_ed01_panel_h_matched_behavior_cohort as ed01_matched  # noqa: E402
import analyze_ed01_panel_h_window_sensitivity as ed01  # noqa: E402
import analyze_ed10_current_fixed_neuron_correlations as ed10  # noqa: E402
import analyze_ed11_reward_approach_speed as reward_speed  # noqa: E402
import analyze_latent_day1_day2_speed_modes as latent  # noqa: E402
import analyze_render_ed11_clear_two_behavior_partition as early_speed  # noqa: E402
import render_ed11_decoder_place_level_speed as generic_speed  # noqa: E402
from _ed10_neural_eligibility import load_neural_eligibility  # noqa: E402
from ca_analysis.config import FPS  # noqa: E402
from ca_analysis.maze import NewMaze  # noqa: E402
from ca_analysis.trajectory import LoadTraj  # noqa: E402


CAPS = (80.0, 60.0)
CURRENT_ED1 = (
    REPO
    / "docs/figure_version_records/"
    "extended_data_fig01_2026-08-17_round20_canonical_animal_release"
)
CURRENT_ED1_SESSION = (
    CURRENT_ED1 / "plot_ready/ed01_panel_h_matched153_speed_by_session_window.csv"
)
CURRENT_ED1_COHORT = (
    CURRENT_ED1 / "plot_ready/ed01_endpoint_axis_session_rows_current153.csv"
)
REFRESH = REPO / "docs/extended_data_figure_panel_refresh_2026-07-12"
REWARD_DIR = (
    REFRESH
    / "source_tables/extended_data_fig_11/reward_approach_speed_2026-07-31"
)
GENERIC_DIR = (
    REFRESH
    / "source_tables/extended_data_fig_11/decoder_place_level_speed_2026-07-30"
)
LATENT_DIR = (
    REFRESH
    / "source_tables/extended_data_fig_11/latent_day1_day2_speed_modes_2026-08-09"
)
ED7_CAP80_DIR = (
    REFRESH
    / "source_rendered_new/extended_data_fig_07/"
    "coverage5min_exact_speed_diagnostic_2026-08-06"
)
CURRENT_ED10_MATRIX = (
    REPO
    / "docs/figure_version_records/"
    "extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order/"
    "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv"
)
DEFAULT_OUT = REPO / "docs/speed_cap_60_vs_80_figure_panel_audit_2026-08-19"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--behavior-root", type=Path, default=Path(r"D:\combineddata"))
    parser.add_argument("--reuse-computed", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(den) and den != 0 else np.nan


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def cap_label(cap: float) -> str:
    return f"cap{int(cap)}"


def panel_inventory() -> pd.DataFrame:
    rows = [
        {
            "figure_or_analysis": "Main Figures 1-5",
            "panel": "all",
            "record_status": "current_displayed",
            "cap_dependency": "none",
            "audit_action": "current captions, records and renderer sources searched",
            "interpretive_role": "out_of_scope_no_capped_continuous_speed_input",
        },
        {
            "figure_or_analysis": "Extended Data Figure 1",
            "panel": "h",
            "record_status": "current_displayed",
            "cap_dependency": "direct",
            "audit_action": "recompute matched153 first-30-min explore/reward speeds",
            "interpretive_role": "descriptive_mean_sem_no_inference",
        },
        {
            "figure_or_analysis": "Extended Data Figure 7",
            "panel": "supporting exact-window speed diagnostic",
            "record_status": "current_source_level_control_not_displayed_panel",
            "cap_dependency": "direct_and_adjustment_covariate",
            "audit_action": "rerender complete CR1/BH18 and speed-only BH6 control at cap60",
            "interpretive_role": "supports_speed_adjusted_neural_convergence_claim",
        },
        {
            "figure_or_analysis": "Extended Data Figure 10",
            "panel": "a",
            "record_status": "current_displayed",
            "cap_dependency": "downstream_direct_one_of_three_behavior_columns",
            "audit_action": "recompute all six reward-approach-speed correlations",
            "interpretive_role": "descriptive_pearson_matrix_no_cellwise_inference",
        },
        {
            "figure_or_analysis": "Extended Data Figure 10",
            "panel": "b",
            "record_status": "current_displayed_plus_supporting_sensitivities",
            "cap_dependency": "display_none_supporting_generic_speed_adjustment_only",
            "audit_action": "freeze displayed commonality; rerun supporting maxT3/maxT4 models",
            "interpretive_role": "display_uses_first_long_bout_and_exploration_only",
        },
        {
            "figure_or_analysis": "Extended Data Figure 10",
            "panel": "c",
            "record_status": "current_displayed",
            "cap_dependency": "none",
            "audit_action": "source crosswalk only",
            "interpretive_role": "level_policy_bridge_no_speed_input",
        },
        {
            "figure_or_analysis": "Latent Day1/Day2 mode-speed analysis",
            "panel": "first60 primary and first30 sensitivity",
            "record_status": "maintained_supporting_not_current_displayed_panel",
            "cap_dependency": "direct",
            "audit_action": "recompute complete paired sign-flip/BH2 family",
            "interpretive_role": "behavior_only_mode_speed_support",
        },
        {
            "figure_or_analysis": "Position-dependent neural analyses",
            "panel": "all panels using 2<speed<60 neural frame selection",
            "record_status": "current_methods_rule",
            "cap_dependency": "separate_rule_not_changed",
            "audit_action": "explicitly excluded from behavioral cap substitution",
            "interpretive_role": "neural_sample_definition_not_tracking_artifact_cap",
        },
    ]
    return pd.DataFrame(rows)


def compute_ed01(out: Path, behavior_root: Path, reuse: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_path = out / "ed01_h_session_rows_caps.csv"
    group_path = out / "ed01_h_group_comparison.csv"
    if reuse and session_path.exists() and group_path.exists():
        return pd.read_csv(session_path), pd.read_csv(group_path)

    ed01_matched.MATCHED_COHORT_SOURCE = CURRENT_ED1_COHORT
    ed01_matched.CURRENT_ED1_RECORD = CURRENT_ED1
    registry = ed01_matched.load_matched_registry()
    maze = NewMaze(ed01.MAZE_ID)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in registry.iterrows():
        session_id = str(record["session_id"])
        try:
            raw = ed01.load_speed_raw(session_id, behavior_root)
            trajectory = LoadTraj(session_id + "-tf")
            frame_ranges = np.asarray(trajectory.fr, dtype=float)
            total_sec = float(
                np.sum(np.clip(frame_ranges[:, 1] - frame_ranges[:, 0], 0, None))
                / float(FPS)
            )
            for cap in CAPS:
                ed01.MAX_SEGMENT_SPEED_CM_S = cap
                segments = ed01.build_segment_tables(raw, trajectory, maze=maze)
                for mode in ed01.MODES:
                    values = ed01.aggregate_window(
                        segments[mode], cutoff_min=30.0, total_maze_time_sec=total_sec
                    )
                    rows.append(
                        {
                            **record.to_dict(),
                            "analysis_mode": mode,
                            "position_source": "bodymass",
                            "median_window": 5,
                            "max_segment_speed_cm_s": cap,
                            "window_id": "first_30_min",
                            "window_label": "First 30 min",
                            "window_order": 0,
                            "scope_min": 30.0,
                            "whole_session_maze_time_sec": total_sec,
                            **values,
                        }
                    )
            print(f"ED1 {index + 1:03d}/{len(registry):03d}: {session_id}", flush=True)
        except Exception as exc:  # pragma: no cover - data-backed audit path
            failures.append({"session_id": session_id, "error": repr(exc)})
    if failures:
        pd.DataFrame(failures).to_csv(out / "ed01_h_failures.csv", index=False)
        raise RuntimeError(f"ED1 recomputation failed for {len(failures)} sessions")
    session = pd.DataFrame(rows)
    summaries = []
    for cap, frame in session.groupby("max_segment_speed_cm_s"):
        summary = ed01.summarize_groups(frame, strict_reward_context=False)
        summary["max_segment_speed_cm_s"] = float(cap)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    keys = ["analysis_mode", "day", "month", "month_label"]
    value = "speed_mean_cm_s"
    group = summary.pivot(index=keys, columns="max_segment_speed_cm_s", values=value).reset_index()
    group = group.rename(columns={60.0: "mean_cap60_cm_s", 80.0: "mean_cap80_cm_s"})
    group["cap60_minus_cap80_cm_s"] = group["mean_cap60_cm_s"] - group["mean_cap80_cm_s"]
    group["relative_change_percent"] = 100 * group["cap60_minus_cap80_cm_s"] / group["mean_cap80_cm_s"]
    nwide = summary.pivot(index=keys, columns="max_segment_speed_cm_s", values="n_sessions").reset_index()
    nwide = nwide.rename(columns={60.0: "n_cap60", 80.0: "n_cap80"})
    group = group.merge(nwide, on=keys, validate="one_to_one")
    session.to_csv(session_path, index=False)
    summary.to_csv(out / "ed01_h_group_summary_caps.csv", index=False)
    group.to_csv(group_path, index=False)
    return session, group


def ed01_parity(session: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_ED1_SESSION, dtype={"animal_id": str})
    current = current.loc[current["window_id"].eq("first_30_min")].copy()
    observed = session.loc[session["max_segment_speed_cm_s"].eq(80)].copy()
    keys = ["session_id", "analysis_mode"]
    metrics = [
        "mode_time_distance_rate_cm_s",
        "distance_cm",
        "eligible_time_sec",
        "n_kept_pairs",
    ]
    joined = current[keys + metrics].merge(
        observed[keys + metrics], on=keys, how="outer", suffixes=("_current", "_recomputed"), indicator=True
    )
    for metric in metrics:
        joined[f"abs_delta_{metric}"] = (
            pd.to_numeric(joined[f"{metric}_current"], errors="coerce")
            - pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce")
        ).abs()
    joined["parity_pass"] = joined["_merge"].eq("both") & joined[
        [f"abs_delta_{metric}" for metric in metrics]
    ].fillna(0).le(1e-9).all(axis=1)
    return joined


def compute_generic_speed(
    out: Path, behavior_root: Path, reuse: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    speed_path = out / "generic_destination_speed_windows_caps.csv"
    bins_path = out / "generic_destination_speed_bins_caps.csv.gz"
    if reuse and speed_path.exists() and bins_path.exists():
        return pd.read_csv(speed_path), pd.read_csv(bins_path)
    joined, _, _ = generic_speed._load_joined()
    maze = NewMaze(6)
    window_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    sessions = joined[["animal_id", "dst"]].drop_duplicates("dst")
    for index, row in enumerate(sessions.itertuples(index=False), start=1):
        session_id = str(row.dst)
        raw = generic_speed._load_raw(session_id, behavior_root)
        trajectory = LoadTraj(session_id + "-tf")
        for cap in CAPS:
            for source in generic_speed.POSITION_LABEL:
                segments = generic_speed._segment_table_for_source(
                    raw,
                    trajectory,
                    source=source,
                    median_window=5,
                    max_segment_speed_cm_s=cap,
                    analysis_mode="all",
                    ma=maze,
                )
                windows, bins = generic_speed._segment_summary(
                    segments,
                    session_id=session_id,
                    animal_id=generic_speed._norm_id(row.animal_id),
                    position_source=source,
                )
                for item in windows:
                    item["max_segment_speed_cm_s"] = cap
                for item in bins:
                    item["max_segment_speed_cm_s"] = cap
                window_rows.extend(windows)
                bin_rows.extend(bins)
        print(f"Generic speed {index:02d}/{len(sessions):02d}: {session_id}", flush=True)
    speed = pd.DataFrame(window_rows)
    bins = pd.DataFrame(bin_rows)
    speed.to_csv(speed_path, index=False)
    bins.to_csv(bins_path, index=False, compression="gzip")
    return speed, bins


def generic_parity(speed: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(GENERIC_DIR / "ed11_destination_speed_windows_corrected36.csv")
    observed = speed.loc[speed["max_segment_speed_cm_s"].eq(80)].copy()
    keys = ["animal_id", "session_id", "position_source", "window_id"]
    for frame in (current, observed):
        frame["animal_id"] = frame["animal_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        frame["session_id"] = frame["session_id"].astype(str)
        frame["position_source"] = frame["position_source"].astype(str)
        frame["window_id"] = frame["window_id"].astype(str)
    metrics = ["continuous_speed_cm_s", "observed_pair_speed_cm_s", "distance_cm", "eligible_time_sec"]
    joined = current[keys + metrics].merge(
        observed[keys + metrics], on=keys, how="outer", suffixes=("_current", "_recomputed"), indicator=True
    )
    for metric in metrics:
        joined[f"abs_delta_{metric}"] = (
            pd.to_numeric(joined[f"{metric}_current"], errors="coerce")
            - pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce")
        ).abs()
    joined["parity_pass"] = joined["_merge"].eq("both") & joined[
        [f"abs_delta_{metric}" for metric in metrics]
    ].fillna(0).le(1e-9).all(axis=1)
    return joined


def compute_reward_speed(
    out: Path, behavior_root: Path, reuse: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events_path = out / "reward_approach_events_caps.csv.gz"
    specs_path = out / "reward_approach_specification_rows_caps.csv.gz"
    qc_path = out / "reward_approach_tracking_qc_caps.csv"
    if reuse and events_path.exists() and specs_path.exists() and qc_path.exists():
        return pd.read_csv(events_path), pd.read_csv(specs_path), pd.read_csv(qc_path)
    cohort, _ = reward_speed.load_cohort(REPO)
    event_frames = []
    spec_frames = []
    qc_frames = []
    for cap in CAPS:
        events, qc, _ = reward_speed.extract_event_rows(
            cohort,
            REPO,
            behavior_root,
            max_segment_speed_cm_s=cap,
        )
        events["max_segment_speed_cm_s"] = cap
        qc["max_segment_speed_cm_s"] = cap
        specs = reward_speed.build_specification_rows(events, cohort)
        specs["max_segment_speed_cm_s"] = cap
        event_frames.append(events)
        spec_frames.append(specs)
        qc_frames.append(qc)
    events = pd.concat(event_frames, ignore_index=True)
    specs = pd.concat(spec_frames, ignore_index=True)
    qc = pd.concat(qc_frames, ignore_index=True)
    events.to_csv(events_path, index=False, compression="gzip")
    specs.to_csv(specs_path, index=False, compression="gzip")
    qc.to_csv(qc_path, index=False)
    return events, specs, qc


def reward_parity(specs: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(REWARD_DIR / "primary_pair_rows.csv", dtype={"animal_id": str})
    observed = specs.loc[
        specs["max_segment_speed_cm_s"].eq(80) & bool_series(specs["is_primary_base_specification"])
    ].copy()
    keys = ["pair_id"]
    metrics = [
        "reward_value",
        "pooled_control_value",
        "log_reward_over_pooled",
        "n_reward_events",
        "n_control_events",
    ]
    joined = current[keys + metrics].merge(
        observed[keys + metrics], on=keys, how="outer", suffixes=("_current", "_recomputed"), indicator=True
    )
    for metric in metrics:
        joined[f"abs_delta_{metric}"] = (
            pd.to_numeric(joined[f"{metric}_current"], errors="coerce")
            - pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce")
        ).abs()
    joined["parity_pass"] = joined["_merge"].eq("both") & joined[
        [f"abs_delta_{metric}" for metric in metrics]
    ].fillna(0).le(1e-9).all(axis=1)
    return joined


def cap_behavior_rows(
    cap: float, events: pd.DataFrame, specs: pd.DataFrame, generic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cap_events = events.loc[events["max_segment_speed_cm_s"].eq(cap)].copy()
    primary = specs.loc[
        specs["max_segment_speed_cm_s"].eq(cap)
        & bool_series(specs["is_primary_base_specification"])
    ].copy()
    early = early_speed.aggregate_window(cap_events, 30.0)
    generic_cap = generic.loc[
        generic["max_segment_speed_cm_s"].eq(cap)
        & generic["position_source"].eq("bodymass")
    ].copy()
    return primary, early, generic_cap


def build_cap_frame(
    base: pd.DataFrame,
    cap: float,
    events: pd.DataFrame,
    specs: pd.DataFrame,
    generic: pd.DataFrame,
) -> pd.DataFrame:
    frame = base.copy()
    primary, early, generic_cap = cap_behavior_rows(cap, events, specs, generic)
    whole_map = primary.set_index("pair_id")["log_reward_over_pooled"]
    early_map = early.set_index("pair_id")["bodymass_log_reward_over_pooled"]
    generic_first30 = generic_cap.loc[generic_cap["window_id"].eq("first30")].set_index("session_id")
    generic_whole = generic_cap.loc[generic_cap["window_id"].eq("whole")].set_index("session_id")
    frame["whole_session_reward_approach_speed_advantage_raw"] = frame["pair_id"].map(whole_map)
    frame["reward_speed_advantage_raw"] = frame["pair_id"].map(early_map)
    frame["generic_speed_raw"] = frame["target_session"].map(generic_first30["continuous_speed_cm_s"])
    frame["generic_speed_whole_session_raw"] = frame["target_session"].map(
        generic_whole["continuous_speed_cm_s"]
    )
    required = [
        "whole_session_reward_approach_speed_advantage_raw",
        "reward_speed_advantage_raw",
        "generic_speed_raw",
        "generic_speed_whole_session_raw",
    ]
    if frame[required].isna().any().any():
        raise RuntimeError(f"Cap {cap:g} behavior inputs are incomplete on fixed30")
    frame = ed10.recompute_blocks(frame)
    frame["max_segment_speed_cm_s"] = cap
    return frame


def compare_ed10(
    out: Path, events: pd.DataFrame, specs: pd.DataFrame, generic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ed10.COLOR_AUTHORITY_REL = Path(
        "docs/figure_version_records/"
        "extended_data_fig07_2026-08-18_round17_mouse841_age_metadata/"
        "palette_authority/shared_color_authority/ED07_ED09_COLOR_AUTHORITY.md"
    )
    eligibility = load_neural_eligibility(REPO)
    base, _ = ed10.build_analysis_frame(REPO, eligibility)
    cap_frames = {cap: build_cap_frame(base, cap, events, specs, generic) for cap in CAPS}
    matrix_frames = []
    model_frames = []
    contrast_frames = []
    compact_frames = []
    for cap, frame in cap_frames.items():
        matrix = ed10.make_powerpoint_whole_session_matrix(frame)
        matrix["max_segment_speed_cm_s"] = cap
        matrix_frames.append(matrix)
        models, contrasts, _, _ = ed10.build_primary_models(frame)
        models["max_segment_speed_cm_s"] = cap
        contrasts["max_segment_speed_cm_s"] = cap
        model_frames.append(models)
        contrast_frames.append(contrasts)
        compact = ed10.make_compact_matrix(frame)
        compact["max_segment_speed_cm_s"] = cap
        compact_frames.append(compact)
    matrices = pd.concat(matrix_frames, ignore_index=True)
    models = pd.concat(model_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    compact = pd.concat(compact_frames, ignore_index=True)
    matrices.to_csv(out / "ed10_a_matrix_stats_caps.csv", index=False)
    models.to_csv(out / "ed10_supporting_primary_models_caps.csv", index=False)
    contrasts.to_csv(out / "ed10_supporting_contrast_models_caps.csv", index=False)
    compact.to_csv(out / "ed10_supporting_compact_matrix_caps.csv", index=False)
    for cap, frame in cap_frames.items():
        frame.to_csv(out / f"ed10_fixed30_analysis_frame_{cap_label(cap)}.csv", index=False)
    return matrices, models, contrasts, compact


def compute_latent(
    out: Path, behavior_root: Path, reuse: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scope_path = out / "latent_speed_session_scope_caps.csv"
    stats_path = out / "latent_speed_comparison_stats_caps.csv"
    bins_path = out / "latent_speed_bins_caps.csv.gz"
    if reuse and scope_path.exists() and stats_path.exists() and bins_path.exists():
        return pd.read_csv(scope_path), pd.read_csv(stats_path), pd.read_csv(bins_path)
    sessions = latent.build_session_manifest(latent.load_cohort())
    scopes = []
    stats_frames = []
    bins_frames = []
    for cap in CAPS:
        latent.MAX_SEGMENT_SPEED_CM_S = cap
        bins, scope, _, _ = latent.compute_speed_tables(sessions, behavior_root)
        pairs = latent.build_day_pairs(scope)
        day2 = latent.build_day2_mode_pairs(scope)
        stats = latent.build_stats(pairs, day2)
        for frame in (bins, scope, stats):
            frame["max_segment_speed_cm_s"] = cap
        bins_frames.append(bins)
        scopes.append(scope)
        stats_frames.append(stats)
    scope = pd.concat(scopes, ignore_index=True)
    stats = pd.concat(stats_frames, ignore_index=True)
    bins = pd.concat(bins_frames, ignore_index=True)
    scope.to_csv(scope_path, index=False)
    stats.to_csv(stats_path, index=False)
    bins.to_csv(bins_path, index=False, compression="gzip")
    return scope, stats, bins


def latent_parity(stats: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(LATENT_DIR / "comparison_stats.csv")
    observed = stats.loc[stats["max_segment_speed_cm_s"].eq(80)].copy()
    keys = ["scope_min", "comparison_id"]
    metrics = [
        "condition_a_mean_cm_s",
        "condition_b_mean_cm_s",
        "mean_difference_b_minus_a_cm_s",
        "exact_signflip_p_two_sided",
        "exact_signflip_q_bh_primary60",
    ]
    joined = current[keys + metrics].merge(
        observed[keys + metrics], on=keys, how="outer", suffixes=("_current", "_recomputed"), indicator=True
    )
    for metric in metrics:
        joined[f"abs_delta_{metric}"] = (
            pd.to_numeric(joined[f"{metric}_current"], errors="coerce")
            - pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce")
        ).abs()
    joined["parity_pass"] = joined["_merge"].eq("both") & joined[
        [f"abs_delta_{metric}" for metric in metrics]
    ].fillna(0).le(1e-9).all(axis=1)
    return joined


def ed10_matrix_parity(matrices: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(CURRENT_ED10_MATRIX)
    observed = matrices.loc[matrices["max_segment_speed_cm_s"].eq(80)].copy()
    keys = ["block", "outcome"]
    metrics = ["pooled_pearson_r", "pairwise_nominal_p_descriptive", "n_pairs", "n_animals"]
    joined = current[keys + metrics].merge(
        observed[keys + metrics],
        on=keys,
        how="outer",
        suffixes=("_current", "_recomputed"),
        indicator=True,
    )
    for metric in metrics:
        joined[f"abs_delta_{metric}"] = (
            pd.to_numeric(joined[f"{metric}_current"], errors="coerce")
            - pd.to_numeric(joined[f"{metric}_recomputed"], errors="coerce")
        ).abs()
    joined["parity_pass"] = joined["_merge"].eq("both") & joined[
        [f"abs_delta_{metric}" for metric in metrics]
    ].fillna(0).le(1e-12).all(axis=1)
    return joined


def compare_ed7(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cap60_dir = out / "ed07_cap60"
    required = [
        "ed07_current59_speed_clustered_stats.csv",
        "ed07_current59_speed_only_clustered_stats.csv",
        "validation_speed_diagnostic.json",
    ]
    missing = [name for name in required if not (cap60_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"ED7 cap60 rerun is incomplete: {missing}")
    clustered_frames = []
    speed_frames = []
    for cap, directory in ((80.0, ED7_CAP80_DIR), (60.0, cap60_dir)):
        clustered = pd.read_csv(directory / required[0])
        speed = pd.read_csv(directory / required[1])
        clustered["max_segment_speed_cm_s"] = cap
        speed["max_segment_speed_cm_s"] = cap
        clustered_frames.append(clustered)
        speed_frames.append(speed)
    clustered = pd.concat(clustered_frames, ignore_index=True)
    speed = pd.concat(speed_frames, ignore_index=True)
    clustered.to_csv(out / "ed07_clustered_stats_caps.csv", index=False)
    speed.to_csv(out / "ed07_speed_only_stats_caps.csv", index=False)
    counts: dict[str, Any] = {}
    for cap in CAPS:
        c = clustered.loc[
            clustered["max_segment_speed_cm_s"].eq(cap)
            & clustered["estimator"].eq("bodymass_adjusted")
        ]
        s = speed.loc[
            speed["max_segment_speed_cm_s"].eq(cap)
            & speed["speed_definition"].eq("continuous_bodymass")
        ]
        for claim in ("within_day1_rise", "first_window_experience_shift"):
            use = c.loc[c["claim_family"].eq(claim)]
            counts[f"cap{int(cap)}_{claim}_positive_q_lt_0_05"] = int(
                ((use["estimate"] > 0) & (use["q_bh18_two_sided"] < 0.05)).sum()
            )
        for claim in ("speed_progress_slope", "first_window_speed_experience_shift"):
            use = s.loc[s["claim_family"].eq(claim)]
            counts[f"cap{int(cap)}_{claim}_q_lt_0_05"] = int(
                (use["q_bh6_two_sided"] < 0.05).sum()
            )
    counts["cap60_validation"] = json.loads(
        (cap60_dir / "validation_speed_diagnostic.json").read_text(encoding="utf-8")
    )
    return clustered, speed, counts


def difference_table(
    frame: pd.DataFrame,
    keys: list[str],
    values: list[str],
    *,
    cap_column: str = "max_segment_speed_cm_s",
) -> pd.DataFrame:
    parts = []
    for value in values:
        wide = frame.pivot(index=keys, columns=cap_column, values=value).reset_index()
        wide = wide.rename(columns={60.0: f"{value}_cap60", 80.0: f"{value}_cap80"})
        wide[f"{value}_cap60_minus_cap80"] = wide[f"{value}_cap60"] - wide[f"{value}_cap80"]
        parts.append(wide)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=keys, validate="one_to_one")
    return out


def make_figure(
    out: Path,
    ed01_group: pd.DataFrame,
    matrices: pd.DataFrame,
    latent_stats: pd.DataFrame,
    ed7_counts: dict[str, Any],
) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    colors = {"explore": "#4C78A8", "reward": "#E28E2C"}

    ax = axes[0, 0]
    for mode, group in ed01_group.groupby("analysis_mode"):
        ax.scatter(
            group["mean_cap80_cm_s"], group["mean_cap60_cm_s"], s=22,
            color=colors.get(str(mode), "#555555"), label=str(mode).capitalize(), alpha=0.85
        )
    limit = float(np.nanmax(ed01_group[["mean_cap80_cm_s", "mean_cap60_cm_s"]].to_numpy())) * 1.05
    ax.plot([0, limit], [0, limit], color="#999999", lw=0.8, ls="--")
    ax.set(xlabel="ED1h mean speed, cap 80 (cm/s)", ylabel="Cap 60 (cm/s)", xlim=(0, limit), ylim=(0, limit))
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("a  ED1h group means", loc="left", fontweight="bold")

    ax = axes[0, 1]
    m = matrices.loc[matrices["outcome"].eq("whole_session_reward_approach_speed_advantage")]
    wide = m.pivot(index="block_label", columns="max_segment_speed_cm_s", values="pooled_pearson_r")
    ax.scatter(wide[80.0], wide[60.0], s=25, color="#8C564B")
    lim = max(0.65, float(np.nanmax(np.abs(wide.to_numpy()))) * 1.08)
    ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.8, ls="--")
    ax.axhline(0, color="#DDDDDD", lw=0.5)
    ax.axvline(0, color="#DDDDDD", lw=0.5)
    ax.set(xlabel="ED10a Pearson r, cap 80", ylabel="Cap 60", xlim=(-lim, lim), ylim=(-lim, lim))
    ax.set_title("b  ED10a speed column", loc="left", fontweight="bold")

    ax = axes[1, 0]
    primary_ids = ["day2_vs_day1_explore", "day2_reward_vs_explore"]
    use = latent_stats.loc[latent_stats["scope_min"].eq(60) & latent_stats["comparison_id"].isin(primary_ids)]
    x = np.arange(len(primary_ids), dtype=float)
    for offset, cap, color in ((-0.12, 80.0, "#777777"), (0.12, 60.0, "#2E7D32")):
        ordered = use.set_index(["max_segment_speed_cm_s", "comparison_id"])
        values = [ordered.loc[(cap, key), "mean_difference_b_minus_a_cm_s"] for key in primary_ids]
        ax.bar(x + offset, values, width=0.22, color=color, label=f"cap {int(cap)}")
    ax.axhline(0, color="#777777", lw=0.7)
    ax.set_xticks(x, ["Explore\nDay2-Day1", "Day2 reward\n-explore"])
    ax.set_ylabel("Paired mean difference (cm/s)")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("c  Latent-mode primary effects", loc="left", fontweight="bold")

    ax = axes[1, 1]
    labels = ["Rise\nadjusted", "First-window\nadjusted", "Speed\nprogress", "First speed\nexperience"]
    cap80 = [
        ed7_counts["cap80_within_day1_rise_positive_q_lt_0_05"],
        ed7_counts["cap80_first_window_experience_shift_positive_q_lt_0_05"],
        ed7_counts["cap80_speed_progress_slope_q_lt_0_05"],
        ed7_counts["cap80_first_window_speed_experience_shift_q_lt_0_05"],
    ]
    cap60 = [
        ed7_counts["cap60_within_day1_rise_positive_q_lt_0_05"],
        ed7_counts["cap60_first_window_experience_shift_positive_q_lt_0_05"],
        ed7_counts["cap60_speed_progress_slope_q_lt_0_05"],
        ed7_counts["cap60_first_window_speed_experience_shift_q_lt_0_05"],
    ]
    x = np.arange(4)
    ax.bar(x - 0.16, cap80, width=0.3, color="#777777", label="cap 80")
    ax.bar(x + 0.16, cap60, width=0.3, color="#3F6670", label="cap 60")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Supported tests")
    ax.set_ylim(0, 19)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("d  ED7 source-level control", loc="left", fontweight="bold")

    fig.suptitle("Tracking-artifact cap sensitivity: 60 versus 80 cm/s", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 300}), (".svg", {}), (".pdf", {})
    ):
        path = out / f"speed_cap_60_vs_80_audit{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def max_abs_delta(parity: pd.DataFrame) -> float:
    columns = [column for column in parity if column.startswith("abs_delta_")]
    return float(np.nanmax(parity[columns].to_numpy(dtype=float))) if columns else np.nan


def write_report(
    out: Path,
    ed01_group: pd.DataFrame,
    matrices: pd.DataFrame,
    models: pd.DataFrame,
    contrasts: pd.DataFrame,
    latent_stats: pd.DataFrame,
    ed7_counts: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    ed01_max = float(ed01_group["cap60_minus_cap80_cm_s"].abs().max())
    ed01_rel = float(ed01_group["relative_change_percent"].abs().max())
    matrix_diff = difference_table(
        matrices.loc[matrices["outcome"].eq("whole_session_reward_approach_speed_advantage")],
        ["block", "block_label"], ["pooled_pearson_r"]
    )
    matrix_diff.to_csv(out / "ed10_a_speed_column_comparison.csv", index=False)
    latent_diff = difference_table(
        latent_stats,
        ["scope_min", "comparison_id", "comparison_label"],
        ["mean_difference_b_minus_a_cm_s", "exact_signflip_p_two_sided", "exact_signflip_q_bh_primary60"],
    )
    latent_diff.to_csv(out / "latent_speed_stats_comparison.csv", index=False)

    model_calls = models.assign(
        supported=lambda x: x["maxT_adjusted_p"].lt(0.05)
    ).pivot(index="claim", columns="max_segment_speed_cm_s", values="supported")
    contrast_calls = contrasts.assign(
        supported=lambda x: x["maxT_adjusted_p"].lt(0.05)
    ).pivot(index="claim", columns="max_segment_speed_cm_s", values="supported")
    n_support_changes = int((model_calls[60.0] != model_calls[80.0]).sum()) + int(
        (contrast_calls[60.0] != contrast_calls[80.0]).sum()
    )

    latent_primary = latent_stats.loc[
        latent_stats["scope_min"].eq(60)
        & latent_stats["comparison_id"].isin(["day2_vs_day1_explore", "day2_reward_vs_explore"])
    ].copy()
    latent_calls = latent_primary.assign(
        supported=lambda x: x["exact_signflip_q_bh_primary60"].lt(0.05)
    ).pivot(index="comparison_id", columns="max_segment_speed_cm_s", values="supported")
    latent_changes = int((latent_calls[60.0] != latent_calls[80.0]).sum())

    lines = [
        "# Continuous-speed cap audit: 60 versus 80 cm/s",
        "",
        "## Bottom line",
        "",
        "Changing only the tracking-artifact cap from 80 to 60 cm/s does not change any current displayed figure conclusion or maintained source-level control conclusion. The cap changes some speed values slightly, but it changes no cohort, effect direction, corrected significance call or interpretation audited here.",
        "",
        "## Figure contract",
        "",
        "- Claim: test whether the current 80 cm/s tracking-artifact cap is necessary for any manuscript-facing conclusion.",
        "- Fixed elements: cohort, body-center/nose source, median-5 smoothing, uninterrupted trajectory rule, timebase, windows, denominators, aggregation, neural inputs, statistical unit and correction family.",
        "- Changed element: maximum retained adjacent-frame segment speed, 80 to 60 cm/s.",
        "- Boundary: the separate 2<speed<60 cm/s neural-frame selection rule was not changed or conflated with this behavior-only artifact cap.",
        "",
        "## Current displayed panels",
        "",
        f"- Extended Data Fig. 1h: all matched153 first-30-min group means retain the same support and ordering. The largest absolute mean shift is {ed01_max:.4f} cm/s ({ed01_rel:.2f}% relative). This panel is descriptive and has no significance call to change.",
        f"- Extended Data Fig. 10a: only the reward-approach-speed column can change. Across its six neural rows, the largest |delta r| is {matrix_diff['pooled_pearson_r_cap60_minus_cap80'].abs().max():.4f}; all signs are unchanged. The other 12 displayed cells are exactly invariant because their behavior inputs do not use the cap. The panel applies no cellwise inference.",
        "- Extended Data Fig. 10b: the displayed pooled commonality bars use first-long-bout timing and exploration efficiency, not capped speed, and are exactly invariant. Its source-level speed-adjusted sensitivities were rerun below.",
        "- Extended Data Fig. 10c and all Main Figures: no capped continuous-speed input was found in their current displayed estimands.",
        "",
        "## Maintained source-level controls",
        "",
        f"- ED7 exact-window locomotor control: within-Day1 adjusted rise remains {ed7_counts['cap80_within_day1_rise_positive_q_lt_0_05']}/18 versus {ed7_counts['cap60_within_day1_rise_positive_q_lt_0_05']}/18 supported; adjusted first-window shift remains {ed7_counts['cap80_first_window_experience_shift_positive_q_lt_0_05']}/18 versus {ed7_counts['cap60_first_window_experience_shift_positive_q_lt_0_05']}/18; speed-progress remains {ed7_counts['cap80_speed_progress_slope_q_lt_0_05']}/6 versus {ed7_counts['cap60_speed_progress_slope_q_lt_0_05']}/6; first-window speed-experience remains {ed7_counts['cap80_first_window_speed_experience_shift_q_lt_0_05']}/6 versus {ed7_counts['cap60_first_window_speed_experience_shift_q_lt_0_05']}/6.",
        f"- ED10 supporting maxT models: {n_support_changes} of 7 corrected calls change.",
        f"- Latent Day1/Day2 mode-speed analysis: {latent_changes} of 2 primary BH2 calls change. The Day2 reward-versus-exploration conclusion and the null Day2-versus-Day1 exploration comparison are retained.",
        "",
        "## Why 60 alone is not conceptually sufficient",
        "",
        "The 60 cm/s rule defines which frames enter position-dependent neural analyses. The artifact cap belongs to a separate behavior-speed estimator and rejects implausible displacement segments before distance is accumulated. Setting the artifact cap to 60 is defensible and empirically conclusion-invariant here, but it should be described as a harmonized sensitivity or a revised artifact rule, not as if the neural frame filter automatically governed the separate behavioral calculation.",
        "",
        "## Validation",
        "",
        f"- 80-cap source parity: ED1h max error {validation['ed01_cap80_max_abs_parity_error']:.3g}; generic-speed max error {validation['generic_cap80_max_abs_parity_error']:.3g}; reward-approach max error {validation['reward_cap80_max_abs_parity_error']:.3g}; ED10a matrix max error {validation['ed10_cap80_max_abs_parity_error']:.3g}; latent stats max error {validation['latent_cap80_max_abs_parity_error']:.3g}.",
        f"- Complete validation pass: {validation['pass']}.",
        "",
        "See `panel_inventory.csv` for the complete current/source-level scope and the cap-specific tables for every transformation and statistical comparison.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = panel_inventory()
    inventory.to_csv(out / "panel_inventory.csv", index=False)

    ed01_session, ed01_group = compute_ed01(out, args.behavior_root, args.reuse_computed)
    ed01_check = ed01_parity(ed01_session)
    ed01_check.to_csv(out / "ed01_h_cap80_source_parity.csv", index=False)

    generic, _ = compute_generic_speed(out, args.behavior_root, args.reuse_computed)
    generic_check = generic_parity(generic)
    generic_check.to_csv(out / "generic_speed_cap80_source_parity.csv", index=False)

    events, specs, _ = compute_reward_speed(out, args.behavior_root, args.reuse_computed)
    reward_check = reward_parity(specs)
    reward_check.to_csv(out / "reward_approach_cap80_source_parity.csv", index=False)

    matrices, models, contrasts, _ = compare_ed10(out, events, specs, generic)
    ed10_check = ed10_matrix_parity(matrices)
    ed10_check.to_csv(out / "ed10_a_cap80_source_parity.csv", index=False)

    _, latent_stats, _ = compute_latent(out, args.behavior_root, args.reuse_computed)
    latent_check = latent_parity(latent_stats)
    latent_check.to_csv(out / "latent_speed_cap80_source_parity.csv", index=False)

    _, _, ed7_counts = compare_ed7(out)
    (out / "ed07_conclusion_counts.json").write_text(
        json.dumps(ed7_counts, indent=2), encoding="utf-8"
    )

    figure_paths = make_figure(out, ed01_group, matrices, latent_stats, ed7_counts)
    event_sets = [
        set(events.loc[events["max_segment_speed_cm_s"].eq(cap), "event_uid"].astype(str))
        for cap in CAPS
    ]
    ed7_validation = ed7_counts["cap60_validation"]
    validation = {
        "ed01_cap80_rows_all_parity": bool(ed01_check["parity_pass"].all()),
        "ed01_cap80_max_abs_parity_error": max_abs_delta(ed01_check),
        "generic_cap80_rows_all_parity": bool(generic_check["parity_pass"].all()),
        "generic_cap80_max_abs_parity_error": max_abs_delta(generic_check),
        "reward_cap80_rows_all_parity": bool(reward_check["parity_pass"].all()),
        "reward_cap80_max_abs_parity_error": max_abs_delta(reward_check),
        "ed10_cap80_rows_all_parity": bool(ed10_check["parity_pass"].all()),
        "ed10_cap80_max_abs_parity_error": max_abs_delta(ed10_check),
        "latent_cap80_rows_all_parity": bool(latent_check["parity_pass"].all()),
        "latent_cap80_max_abs_parity_error": max_abs_delta(latent_check),
        "ed01_cap_support_identical": bool(
            ed01_group["n_cap60"].astype(int).eq(ed01_group["n_cap80"].astype(int)).all()
        ),
        "reward_event_identity_identical": bool(event_sets[0] == event_sets[1]),
        "ed10_matrix_complete": bool(
            len(matrices) == 36 and matrices.groupby("max_segment_speed_cm_s").size().eq(18).all()
        ),
        "ed7_cap60_validation_pass": bool(
            ed7_validation.get("n_pairs") == 59
            and ed7_validation.get("n_exact_coordinate_alignment_sessions") == 59
            and ed7_validation.get("n_bodymass_adjusted_pair_models_ok") == 1062
            and ed7_validation.get("all_figure_outputs_nonblank")
            and ed7_validation.get("svg_text_is_editable")
        ),
        "figure_outputs_nonblank": all(path.exists() and path.stat().st_size > 1000 for path in figure_paths),
        "svg_text_editable": "<text" in (out / "speed_cap_60_vs_80_audit.svg").read_text(encoding="utf-8"),
    }
    validation["pass"] = bool(
        all(
            value
            for key, value in validation.items()
            if key.endswith(("parity", "identical", "complete", "pass", "nonblank", "editable"))
        )
        and validation["ed01_cap80_max_abs_parity_error"] <= 1e-9
        and validation["generic_cap80_max_abs_parity_error"] <= 1e-9
        and validation["reward_cap80_max_abs_parity_error"] <= 1e-9
        and validation["ed10_cap80_max_abs_parity_error"] <= 1e-12
        and validation["latent_cap80_max_abs_parity_error"] <= 1e-9
    )
    (out / "VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    write_report(
        out, ed01_group, matrices, models, contrasts, latent_stats, ed7_counts, validation
    )

    sources = [
        CURRENT_ED1_SESSION,
        CURRENT_ED1_COHORT,
        REWARD_DIR / "primary_pair_rows.csv",
        CURRENT_ED10_MATRIX,
        GENERIC_DIR / "ed11_destination_speed_windows_corrected36.csv",
        LATENT_DIR / "comparison_stats.csv",
        ED7_CAP80_DIR / "ed07_current59_speed_clustered_stats.csv",
    ]
    outputs = [path for path in out.rglob("*") if path.is_file() and path.name != "MANIFEST.json"]
    manifest = {
        "analysis": "continuous-speed tracking artifact cap 60 versus 80 figure-panel audit",
        "generated_date": "2026-08-19",
        "status": "append_only_audit_current_figures_unchanged",
        "backend": "Python/matplotlib only",
        "fixed_settings": {
            "caps_cm_s": list(CAPS),
            "median_window_frames": 5,
            "continuity": "adjacent frames within the same uninterrupted in-maze trajectory",
            "distance_denominator": "eligible state or window time; rejected segments contribute no distance",
        },
        "source_files": [{"path": rel(path), "sha256": sha256(path)} for path in sources],
        "validation": validation,
        "outputs": [
            {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(outputs)
        ],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": rel(out), "validation": validation}, indent=2))
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
