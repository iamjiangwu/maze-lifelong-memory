"""Promoted ED3 fixed-30-minute Explore-only turning-policy analysis.

This analysis consumes the finalized fixed-time policy tables from the audited
20/30-minute rerun, freezes a single 30-minute TimeInMaze estimand, and rebuilds
the biased-walk simulations after applying one strict joint support gate.  The
primary cohort must pass the equal-junction four-policy and depth gates *and*
the event-pooled four-policy and depth gates at 30 minutes.

The primary equal-junction estimator and event-pooled sensitivity use the same
22 Naive and 58 Return sessions.  Only Explore-mode decisions contribute policy
events; all behavioral modes consume the cumulative TimeInMaze clock.  The
whole-session half-coverage efficiency remains a reconstruction target, not a
prospective or causal outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_ed03_turning_bias_fixed_time_20_30min as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed_time_20_30min_explore_2026-08-04"
    / "source_tables"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "ed03_turning_bias_fixed30min_promoted_2026-08-04"
)
WINDOW_MINUTES = 30
WINDOWS_MINUTES = (WINDOW_MINUTES,)
WINDOW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "window_id": "time_30m",
        "family": "clock",
        "quota": WINDOW_MINUTES,
        "label": "30 min",
    },
)
EXPECTED_PRIMARY_COUNTS = {"LTM": 58, "naive": 22}
EXPECTED_PRIMARY_EXCLUSIONS = {"640day1"}
EXPECTED_OVERLAP = {
    "n_naive_mice": 22,
    "n_return_mice": 18,
    "n_overlapping_mice": 11,
    "n_union_mice": 29,
}
POLICY_PARAMETERS = ("Bf", "Ba", "Lf", "Lo")
DEPTH_PARAMETERS = ("Lo_shallow", "Lo_mid", "Lo_deep")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def configure_modules(output: Path) -> None:
    """Freeze inherited maintained helpers to the promoted 30-minute window."""
    base.WINDOWS_MINUTES = WINDOWS_MINUTES
    base.WINDOW_SPECS = WINDOW_SPECS
    base.configure_modules(output)
    base.equal.WINDOWS_MINUTES = WINDOWS_MINUTES


def load_input_policies(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    equal_path = input_dir / "fixed_time_policy_rows.csv"
    event_path = input_dir / "event_pooled_policy_rows.csv"
    equal_policy = pd.read_csv(equal_path)
    event_policy = pd.read_csv(event_path)
    equal_policy = equal_policy.loc[
        pd.to_numeric(equal_policy["window_minutes"], errors="raise").eq(WINDOW_MINUTES)
    ].copy()
    event_policy = event_policy.loc[
        pd.to_numeric(event_policy["quota"], errors="raise").eq(WINDOW_MINUTES)
    ].copy()
    for frame in (equal_policy, event_policy):
        frame["animal_id"] = frame["animal_id"].astype(str)
    if len(equal_policy) != 155 or len(event_policy) != 155:
        raise ValueError(
            "Expected one 30-minute row for each of 155 sessions; "
            f"equal={len(equal_policy)}, event={len(event_policy)}"
        )
    keys = ["nickname", "group", "animal_id"]
    if equal_policy.duplicated(keys).any() or event_policy.duplicated(keys).any():
        raise ValueError("The finalized 30-minute input has duplicate session keys")
    return equal_policy.reset_index(drop=True), event_policy.reset_index(drop=True)


def _failed_gate_names(row: pd.Series) -> str:
    gates = (
        ("equal_four", "equal_primary_policy_support"),
        ("equal_strict_depth", "equal_primary_depth_support"),
        ("event_four", "event_four_bias_event_pooled_support"),
        ("event_strict_depth", "event_depth_stratified_event_pooled_support"),
    )
    failed = [label for label, column in gates if not bool(row[column])]
    return ";".join(failed)


def build_strict_joint_ledger(
    equal_policy: pd.DataFrame,
    event_policy: pd.DataFrame,
    *,
    validate_expected: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the four-gate 30-minute cohort and cross-group mouse audit."""
    keys = ["nickname", "group", "animal_id"]
    equal_flags = equal_policy[
        [*keys, "primary_policy_support", "primary_depth_support"]
    ].rename(
        columns={
            "primary_policy_support": "equal_primary_policy_support",
            "primary_depth_support": "equal_primary_depth_support",
        }
    )
    event_flags = event_policy[
        [
            *keys,
            "four_bias_event_pooled_support",
            "depth_stratified_event_pooled_support",
        ]
    ].rename(
        columns={
            "four_bias_event_pooled_support": "event_four_bias_event_pooled_support",
            "depth_stratified_event_pooled_support": (
                "event_depth_stratified_event_pooled_support"
            ),
        }
    )
    ledger = equal_flags.merge(
        event_flags, on=keys, how="outer", indicator=True, validate="one_to_one"
    )
    if not ledger["_merge"].eq("both").all():
        raise ValueError(
            "Equal-junction and event-pooled 30-minute session keys do not match: "
            f"{ledger['_merge'].value_counts().to_dict()}"
        )
    ledger = ledger.drop(columns="_merge")
    gates = [
        "equal_primary_policy_support",
        "equal_primary_depth_support",
        "event_four_bias_event_pooled_support",
        "event_depth_stratified_event_pooled_support",
    ]
    ledger["window_minutes"] = WINDOW_MINUTES
    ledger["in_primary_groups"] = ledger["group"].isin(["naive", "LTM"])
    ledger["strict_joint_support_pass"] = ledger[gates].fillna(False).all(axis=1)
    ledger["included_primary_30min_strict_joint_cohort"] = (
        ledger["in_primary_groups"] & ledger["strict_joint_support_pass"]
    )
    # Short alias is the maintained validator-facing field name.
    ledger["included_primary_30min_strict_joint"] = ledger[
        "included_primary_30min_strict_joint_cohort"
    ]
    ledger["failed_gate_names"] = ledger.apply(_failed_gate_names, axis=1)
    ledger["exclusion_reason"] = np.select(
        [
            ~ledger["in_primary_groups"],
            ledger["in_primary_groups"] & ~ledger["strict_joint_support_pass"],
        ],
        [
            "not_Naive_or_Return_primary_group",
            "failed_30min_strict_joint_support:" + ledger["failed_gate_names"],
        ],
        default="included",
    )
    included = ledger.loc[
        ledger["included_primary_30min_strict_joint_cohort"], keys
    ].copy()
    counts = included.groupby("group")["nickname"].nunique().to_dict()
    excluded_primary = set(
        ledger.loc[
            ledger["in_primary_groups"]
            & ~ledger["included_primary_30min_strict_joint_cohort"],
            "nickname",
        ].astype(str)
    )
    naive_ids = set(included.loc[included["group"].eq("naive"), "animal_id"])
    return_ids = set(included.loc[included["group"].eq("LTM"), "animal_id"])
    overlap_values: dict[str, Any] = {
        "window_minutes": WINDOW_MINUTES,
        "n_naive_sessions": int(included["group"].eq("naive").sum()),
        "n_return_sessions": int(included["group"].eq("LTM").sum()),
        "n_naive_mice": len(naive_ids),
        "n_return_mice": len(return_ids),
        "n_overlapping_mice": len(naive_ids & return_ids),
        "n_union_mice": len(naive_ids | return_ids),
        "overlapping_animal_ids": ";".join(sorted(naive_ids & return_ids)),
    }
    if validate_expected:
        if counts != EXPECTED_PRIMARY_COUNTS:
            raise ValueError(f"Unexpected strict joint cohort counts: {counts}")
        if excluded_primary != EXPECTED_PRIMARY_EXCLUSIONS:
            raise ValueError(
                "Unexpected primary-group exclusions: "
                f"{sorted(excluded_primary)}"
            )
        for key, expected in EXPECTED_OVERLAP.items():
            if overlap_values[key] != expected:
                raise ValueError(
                    f"Unexpected cross-group mouse support for {key}: "
                    f"{overlap_values[key]} != {expected}"
                )
    ledger = ledger.sort_values(["group", "nickname"]).reset_index(drop=True)
    included = included.sort_values(["group", "nickname"]).reset_index(drop=True)
    return ledger, included, pd.DataFrame([overlap_values])


def filter_to_membership(
    policy: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    return policy.merge(
        membership[["nickname", "group"]].drop_duplicates(),
        on=["nickname", "group"],
        how="inner",
        validate="one_to_one",
    )


def policy_inference_tables(
    full_equal_policy: pd.DataFrame, strict_equal_policy: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Separate strict Naive/Return inference from natural Second-day control."""
    experienced = full_equal_policy.loc[
        full_equal_policy["group"].eq("experienced")
    ].copy()
    strict_primary_plus_natural_control = pd.concat(
        [strict_equal_policy, experienced], ignore_index=True
    )
    full_stats = base.equal.policy_cluster_stats(full_equal_policy)
    mixed_stats = base.equal.policy_cluster_stats(strict_primary_plus_natural_control)
    strict_policy = mixed_stats.loc[
        mixed_stats["contrast_id"].eq("return_vs_naive")
    ].reset_index(drop=True)
    natural_second = full_stats.loc[
        full_stats["contrast_id"].eq("second_vs_return")
    ].reset_index(drop=True)
    depth_all = base.equal.depth_cluster_stats(strict_primary_plus_natural_control)
    strict_depth = depth_all.loc[
        depth_all["contrast_id"].eq("return_vs_naive")
        & depth_all["support_policy"].eq("all_depth_strata_n_ge_10")
    ].reset_index(drop=True)
    group_summary = base.equal.group_summary(strict_primary_plus_natural_control)
    group_summary = group_summary.loc[
        group_summary["group"].isin(["naive", "LTM"])
    ].reset_index(drop=True)
    return {
        "full_policy_stats": full_stats,
        "strict_policy_stats": strict_policy,
        "natural_second_stats": natural_second,
        "strict_depth_stats": strict_depth,
        "strict_group_summary": group_summary,
    }


def run_equal_junction(
    strict_policy: pd.DataFrame,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    simulation = base.equal.run_simulations(
        strict_policy, n_walks=n_walks, seed=seed
    )
    counts = simulation.groupby("group")["nickname"].nunique().to_dict()
    if counts != EXPECTED_PRIMARY_COUNTS:
        raise ValueError(f"Unexpected equal-junction simulation rows: {counts}")
    calibration_rows: list[dict[str, Any]] = []
    for model, simulated_column in (
        ("four_bias", "four_own_half_cover_eff"),
        ("depth_stratified", "depth_own_half_cover_eff"),
    ):
        for group in ("naive", "LTM"):
            frame = simulation.loc[simulation["group"].eq(group)]
            metrics = base.equal.calibration_metrics(frame, simulated_column)
            calibration_rows.append(
                {
                    "window_minutes": WINDOW_MINUTES,
                    "model": model,
                    "group": group,
                    "group_label": base.GROUP_LABELS[group],
                    **metrics,
                    "bias_gate_abs_mean_error_le_0p05": bool(
                        abs(metrics["mean_error"]) <= 0.05
                    ),
                    "rank_gate_spearman_rho_ge_0p4": bool(
                        metrics["spearman_rho"] >= 0.4
                    ),
                    "combined_gate_pass": bool(
                        abs(metrics["mean_error"]) <= 0.05
                        and metrics["spearman_rho"] >= 0.4
                    ),
                }
            )
    calibration = pd.DataFrame(calibration_rows)
    gap, contributions, draws, overlap = (
        base.synchronized_equal_counterfactual_summary(
            simulation, n_bootstrap=n_bootstrap, seed=seed + 50000
        )
    )
    return {
        "simulation": simulation,
        "calibration": calibration,
        "gap": gap,
        "contributions": contributions,
        "draws": draws,
        "overlap": overlap,
    }


def run_event_pooled(
    strict_policy: pd.DataFrame,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    joined, alignment = base.pooled.align_targets(strict_policy)
    alignment = alignment.loc[~alignment["overlap_status"].eq("right_only")].copy()
    if len(alignment) != 80 or not alignment["overlap_status"].eq("both").all():
        raise ValueError(
            "Unexpected strict event-pooled target alignment: "
            f"n={len(alignment)}, statuses={alignment['overlap_status'].value_counts().to_dict()}"
        )
    results, exclusions = base.pooled.simulate_all(
        joined, n_walks=n_walks, seed=seed
    )
    results["window_minutes"] = pd.to_numeric(
        results["quota"], errors="raise"
    ).astype(int)
    if not results["simulation_supported"].all() or not exclusions.empty:
        raise ValueError(
            "The strict joint cohort unexpectedly failed an event-pooled simulation gate"
        )
    calibration, calibration_draws = base.shared.summarize(
        results, n_bootstrap=n_bootstrap, seed=seed + 50000
    )
    counterfactuals = base.pooled_ext.run_counterfactuals(
        strict_policy, results, n_walks=n_walks, seed=seed + 100000
    )
    gap, contributions, draws, overlap = (
        base.synchronized_event_counterfactual_summary(
            counterfactuals,
            results,
            n_bootstrap=n_bootstrap,
            seed=seed + 150000,
        )
    )
    return {
        "alignment": alignment.reset_index(drop=True),
        "results": results,
        "exclusions": exclusions,
        "calibration": calibration,
        "calibration_draws": calibration_draws,
        "counterfactuals": counterfactuals,
        "gap": gap,
        "contributions": contributions,
        "draws": draws,
        "overlap": overlap,
    }


def donor_refit_audit(
    full_equal: pd.DataFrame,
    strict_equal: pd.DataFrame,
    full_event: pd.DataFrame,
    strict_event: pd.DataFrame,
) -> pd.DataFrame:
    specs = (
        (
            "equal_junction",
            "four_bias",
            full_equal,
            strict_equal,
            "window_minutes",
            "primary_policy_support",
            POLICY_PARAMETERS,
            "unweighted mean of supported junction/side conditional probabilities",
        ),
        (
            "equal_junction",
            "depth_stratified",
            full_equal,
            strict_equal,
            "window_minutes",
            "primary_depth_support",
            (*POLICY_PARAMETERS, *DEPTH_PARAMETERS),
            "strict equal-junction depth gate: at least 10 eligible events in every depth stratum",
        ),
        (
            "event_pooled",
            "four_bias_event_pooled",
            full_event,
            strict_event,
            "quota",
            "four_bias_event_pooled_support",
            POLICY_PARAMETERS,
            "pooled successes divided by pooled eligible Explore decisions",
        ),
        (
            "event_pooled",
            "depth_stratified_event_pooled",
            full_event,
            strict_event,
            "quota",
            "depth_stratified_event_pooled_support",
            (*POLICY_PARAMETERS, *DEPTH_PARAMETERS),
            "event-pooled four-policy gate plus at least 10 eligible events in every depth stratum",
        ),
    )
    rows: list[dict[str, Any]] = []
    for (
        family,
        model,
        full,
        strict,
        window_column,
        support_column,
        parameters,
        support_definition,
    ) in specs:
        natural = full.loc[
            full[window_column].eq(WINDOW_MINUTES)
            & full["group"].eq("naive")
            & full[support_column]
        ]
        promoted = strict.loc[
            strict[window_column].eq(WINDOW_MINUTES)
            & strict["group"].eq("naive")
            & strict[support_column]
        ]
        for parameter in parameters:
            natural_mean = float(natural[parameter].mean())
            promoted_mean = float(promoted[parameter].mean())
            rows.append(
                {
                    "window_minutes": WINDOW_MINUTES,
                    "estimator_family": family,
                    "model": model,
                    "parameter": parameter,
                    "support_gate_column": support_column,
                    "support_gate_definition": support_definition,
                    "n_natural_support_naive_sessions": int(len(natural)),
                    "n_strict_joint_naive_sessions": int(len(promoted)),
                    "natural_support_naive_donor_mean": natural_mean,
                    "strict_joint_naive_donor_mean": promoted_mean,
                    "strict_minus_natural_donor_mean": promoted_mean - natural_mean,
                    "simulation_action": (
                        "own, donor, and hybrid walks rerun after strict-joint filtering; "
                        "no broader-cohort simulation rows reused"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_gap_robustness(
    equal_tables: dict[str, pd.DataFrame],
    event_tables: dict[str, pd.DataFrame],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equal_simulation = equal_tables["simulation"]
    event_results = event_tables["results"]
    event_counterfactuals = event_tables["counterfactuals"]
    model_frames: list[tuple[str, int, str, pd.DataFrame, pd.DataFrame]] = []
    equal_frame = equal_simulation.loc[
        equal_simulation["simulation_depth_support"]
    ]
    equal_naive = equal_frame.loc[equal_frame["group"].eq("naive")]
    equal_return = equal_frame.loc[equal_frame["group"].eq("LTM")]
    for model in ("four_bias", "depth_stratified"):
        model_frames.append(
            ("equal_junction", WINDOW_MINUTES, model, equal_naive, equal_return)
        )
    for model, _ in base.pooled.MODELS:
        event_naive = event_results.loc[
            event_results["estimator"].eq(model)
            & event_results["group"].eq("naive")
            & event_results["simulation_supported"]
        ]
        event_return = event_counterfactuals.loc[
            event_counterfactuals["estimator"].eq(model)
            & event_counterfactuals["counterfactual_supported"]
        ]
        model_frames.append(
            ("event_pooled", WINDOW_MINUTES, model, event_naive, event_return)
        )
    mouse_summary, mouse_draws = base.mouse_equal_gap_sensitivity(
        model_frames, n_bootstrap=n_bootstrap, seed=seed
    )

    rows: list[dict[str, Any]] = []
    for family, frame, model_column in (
        ("equal_junction", equal_tables["gap"], "model"),
        ("event_pooled", event_tables["gap"], "estimator"),
    ):
        for row in frame.itertuples(index=False):
            model = str(getattr(row, model_column))
            rows.append(
                {
                    "window_minutes": WINDOW_MINUTES,
                    "estimator_family": family,
                    "weighting": "session_weighted",
                    "model_family": (
                        "depth_stratified" if "depth" in model else "four_bias"
                    ),
                    "model": model,
                    "n_naive_rows": int(row.n_naive_rows),
                    "n_return_rows": int(row.n_return_rows),
                    "n_naive_mice": int(row.n_naive_mice),
                    "n_return_mice": int(row.n_return_mice),
                    "delta_real": float(row.delta_real),
                    "delta_simulated": float(row.delta_simulated),
                    "fraction_recapitulated": float(row.fraction_recapitulated),
                    "fraction_ci_low": float(row.fraction_ci_low),
                    "fraction_ci_high": float(row.fraction_ci_high),
                    "point_weighting": "session-weighted within group",
                    "uncertainty": (
                        "synchronized union-of-mice cluster bootstrap; fixed donor "
                        "means; Monte Carlo walk noise not propagated"
                    ),
                }
            )
    for row in mouse_summary.itertuples(index=False):
        model = str(row.model)
        rows.append(
            {
                "window_minutes": WINDOW_MINUTES,
                "estimator_family": str(row.estimator_family),
                "weighting": "mouse_equal",
                "model_family": (
                    "depth_stratified" if "depth" in model else "four_bias"
                ),
                "model": model,
                "n_naive_rows": int(row.n_naive_rows),
                "n_return_rows": int(row.n_return_rows),
                "n_naive_mice": int(row.n_naive_mice),
                "n_return_mice": int(row.n_return_mice),
                "delta_real": float(row.delta_real),
                "delta_simulated": float(row.delta_simulated),
                "fraction_recapitulated": float(row.fraction_recapitulated),
                "fraction_ci_low": float(row.fraction_ci_low),
                "fraction_ci_high": float(row.fraction_ci_high),
                "point_weighting": str(row.point_weighting),
                "uncertainty": str(row.uncertainty),
            }
        )
    robustness = pd.DataFrame(rows).sort_values(
        ["model_family", "estimator_family", "weighting"]
    ).reset_index(drop=True)
    expected_combinations = (
        robustness.groupby("model_family")[["estimator_family", "weighting"]]
        .apply(lambda frame: len(frame.drop_duplicates()))
        .to_dict()
    )
    if expected_combinations != {"depth_stratified": 4, "four_bias": 4}:
        raise ValueError(
            "Gap robustness does not contain four estimator/weighting combinations "
            f"per model: {expected_combinations}"
        )
    return robustness, mouse_draws


def build_exposure_audits(
    strict_equal: pd.DataFrame, strict_event: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    equal_columns = [
        "nickname",
        "group",
        "animal_id",
        "available_maze_minutes",
        "window_fully_observed",
        "explore_minutes_within_window",
        "n_explore_clips_intersecting_window",
        "n_events",
        "n_Bf",
        "n_Ba",
        "n_Lf",
        "n_Lo",
        "n_Lo_shallow",
        "n_Lo_mid",
        "n_Lo_deep",
    ]
    event_columns = [
        "nickname",
        "group",
        "completion_maze_minutes",
        "completion_decision_index",
        "n_raw_decisions_used",
        "n_unique_parameter_events",
    ]
    session = strict_equal[equal_columns].merge(
        strict_event[event_columns],
        on=["nickname", "group"],
        how="inner",
        validate="one_to_one",
    )
    session.insert(0, "window_minutes", WINDOW_MINUTES)
    summary = base.exposure_summary(strict_equal, strict_event)
    summary = summary.loc[summary["group"].isin(["naive", "LTM"])].reset_index(
        drop=True
    )
    return session, summary


def build_outcome_audits(
    input_dir: Path, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(input_dir / "predictor_outcome_overlap_session_audit.csv")
    session = source.merge(
        membership[["nickname", "group"]].drop_duplicates(),
        on=["nickname", "group"],
        how="inner",
        validate="one_to_one",
    )
    if len(session) != 80:
        raise ValueError(f"Expected 80 strict-cohort outcome rows, observed {len(session)}")
    summary = base.summarize_outcome_overlap_sessions(session)
    summary = summary.loc[
        summary["window_minutes"].eq(WINDOW_MINUTES)
        & summary["group"].isin(["naive", "LTM"])
    ].reset_index(drop=True)
    return session, summary


def write_tables(
    source_dir: Path,
    *,
    full_equal: pd.DataFrame,
    full_event: pd.DataFrame,
    strict_equal: pd.DataFrame,
    strict_event: pd.DataFrame,
    ledger: pd.DataFrame,
    membership: pd.DataFrame,
    overlap: pd.DataFrame,
    inference: dict[str, pd.DataFrame],
    equal_tables: dict[str, pd.DataFrame],
    event_tables: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    mouse_draws: pd.DataFrame,
    exposure_session: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    outcome_session: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    donor_audit: pd.DataFrame,
) -> dict[str, Path]:
    tables: dict[str, pd.DataFrame] = {
        "input_30min_equal_junction_policy_rows.csv": full_equal,
        "input_30min_event_pooled_policy_rows.csv": full_event,
        "primary_30min_natural_equal_junction_policy_rows.csv": full_equal,
        "primary_30min_strict_joint_membership_exclusion_ledger.csv": ledger,
        "primary_30min_strict_joint_included_sessions.csv": membership,
        "primary_30min_strict_joint_mouse_overlap_audit.csv": overlap,
        "primary_30min_strict_joint_equal_junction_policy_rows.csv": strict_equal,
        "primary_30min_strict_joint_event_pooled_policy_rows.csv": strict_event,
        "full_30min_equal_junction_policy_cluster_stats.csv": inference[
            "full_policy_stats"
        ],
        "primary_30min_strict_joint_policy_cluster_stats.csv": inference[
            "strict_policy_stats"
        ],
        "primary_30min_natural_second_return_policy_cluster_stats.csv": inference[
            "natural_second_stats"
        ],
        "primary_30min_strict_joint_depth_cluster_stats.csv": inference[
            "strict_depth_stats"
        ],
        "primary_30min_strict_joint_policy_group_summary.csv": inference[
            "strict_group_summary"
        ],
        "primary_30min_strict_joint_simulation_session_rows.csv": equal_tables[
            "simulation"
        ],
        "primary_30min_strict_joint_calibration_summary.csv": equal_tables[
            "calibration"
        ],
        "primary_30min_strict_joint_gap_recapitulation_summary.csv": equal_tables[
            "gap"
        ],
        "primary_30min_strict_joint_policy_contribution_summary.csv": equal_tables[
            "contributions"
        ],
        "primary_30min_strict_joint_synchronized_bootstrap_draws.csv": equal_tables[
            "draws"
        ],
        "primary_30min_strict_joint_equal_junction_mouse_overlap_audit.csv": equal_tables[
            "overlap"
        ],
        "sensitivity_30min_strict_joint_event_pooled_target_alignment.csv": event_tables[
            "alignment"
        ],
        "sensitivity_30min_strict_joint_event_pooled_simulation_session_rows.csv": event_tables[
            "results"
        ],
        "sensitivity_30min_strict_joint_event_pooled_exclusions.csv": event_tables[
            "exclusions"
        ],
        "sensitivity_30min_strict_joint_event_pooled_calibration_summary.csv": event_tables[
            "calibration"
        ],
        "sensitivity_30min_strict_joint_event_pooled_calibration_bootstrap_draws.csv": event_tables[
            "calibration_draws"
        ],
        "sensitivity_30min_strict_joint_event_pooled_counterfactual_return_rows.csv": event_tables[
            "counterfactuals"
        ],
        "sensitivity_30min_strict_joint_event_pooled_gap_recapitulation_summary.csv": event_tables[
            "gap"
        ],
        "sensitivity_30min_strict_joint_event_pooled_policy_contribution_summary.csv": event_tables[
            "contributions"
        ],
        "sensitivity_30min_strict_joint_event_pooled_synchronized_bootstrap_draws.csv": event_tables[
            "draws"
        ],
        "sensitivity_30min_strict_joint_event_pooled_mouse_overlap_audit.csv": event_tables[
            "overlap"
        ],
        "primary_30min_strict_joint_gap_robustness.csv": robustness,
        "primary_30min_strict_joint_mouse_equal_bootstrap_draws.csv": mouse_draws,
        "primary_30min_strict_joint_window_exposure_session_audit.csv": exposure_session,
        "primary_30min_strict_joint_window_exposure_audit.csv": exposure_summary,
        "primary_30min_strict_joint_predictor_outcome_overlap_session_audit.csv": outcome_session,
        "primary_30min_strict_joint_predictor_outcome_overlap_group_summary.csv": outcome_summary,
        "primary_30min_strict_joint_naive_donor_refit_audit.csv": donor_audit,
    }
    paths: dict[str, Path] = {}
    for filename, frame in tables.items():
        path = source_dir / filename
        frame.to_csv(path, index=False)
        paths[filename] = path
    manifest_rows = []
    for filename, frame in tables.items():
        manifest_rows.append(
            {
                "filename": filename,
                "n_rows": int(len(frame)),
                "n_columns": int(len(frame.columns)),
                "role": (
                    "input_snapshot" if filename.startswith("input_") else "derived_audit"
                ),
                "sha256": sha256(paths[filename]),
            }
        )
    table_manifest = pd.DataFrame(manifest_rows)
    table_manifest_path = source_dir / "source_table_manifest.csv"
    table_manifest.to_csv(table_manifest_path, index=False)
    paths[table_manifest_path.name] = table_manifest_path
    return paths


def markdown_table(frame: pd.DataFrame) -> str:
    return base.markdown_table(frame)


def write_report(
    output: Path,
    *,
    ledger: pd.DataFrame,
    overlap: pd.DataFrame,
    inference: dict[str, pd.DataFrame],
    equal_tables: dict[str, pd.DataFrame],
    event_tables: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    donor_audit: pd.DataFrame,
    n_walks: int,
    n_bootstrap: int,
) -> Path:
    counts = (
        ledger.loc[ledger["included_primary_30min_strict_joint_cohort"]]
        .groupby("group")
        .agg(n_sessions=("nickname", "nunique"), n_mice=("animal_id", "nunique"))
        .reset_index()
    )
    exclusions = ledger.loc[
        ledger["in_primary_groups"]
        & ~ledger["included_primary_30min_strict_joint_cohort"],
        ["nickname", "group", "failed_gate_names", "exclusion_reason"],
    ]
    policy = inference["strict_policy_stats"][
        [
            "policy_label",
            "n_target_rows",
            "n_baseline_rows",
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "p_cluster",
            "q_bh4_within_window_contrast",
        ]
    ]
    natural_second = inference["natural_second_stats"][
        [
            "policy_label",
            "n_target_rows",
            "n_baseline_rows",
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "p_cluster",
        ]
    ]
    depth = inference["strict_depth_stats"][
        [
            "depth_label",
            "n_target_rows",
            "n_baseline_rows",
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "p_cluster",
        ]
    ]
    calibration = equal_tables["calibration"][
        [
            "model",
            "group_label",
            "n_session_rows",
            "mean_error",
            "mae",
            "spearman_rho",
            "combined_gate_pass",
        ]
    ]
    event_calibration = event_tables["calibration"][
        [
            "estimator",
            "group_label",
            "n_session_rows",
            "mean_error",
            "mae",
            "spearman_rho",
            "combined_gate_pass",
        ]
    ]
    lines = [
        "# ED3 promoted fixed-30-minute turning-policy analysis",
        "",
        "## Conclusion and paper role",
        "",
        "The promoted estimand asks whether turning policies measured during the first 30 minutes of cumulative TimeInMaze distinguish Naive from Return sessions and how much of the whole-session exploration-efficiency gap a stationary biased walk can reconstruct. The four-policy walk is a diagnostic decomposition; the depth-varying stemward-exit walk is the model refinement. Neither is a causal mediation analysis or variance explained.",
        "",
        "## Frozen definitions",
        "",
        "- Window: cumulative all-mode TimeInMaze from session start, capped at 30 minutes. Non-Explore modes consume the clock.",
        "- Policy events: only decisions inside Explore-mode clips (`SplitModeClips` mode index 2).",
        "- `P_SF` (`Bf`): after entering from the stem/parent, continue into a child rather than reverse.",
        "- `P_SA` (`Ba`): conditional on continuing from the stem, choose the alternating child side under the maintained orientation convention.",
        "- `P_BF` (`Lf`): after entering from a child, leave that sampled child subtree rather than reverse into it.",
        "- `P_BS` (`Lo`): conditional on leaving the sampled child subtree, continue toward the stem/parent rather than enter the sibling child.",
        "- Primary estimator: compute each conditional probability separately at each supported junction/side, then take the unweighted mean across junction/side estimates.",
        "- Sensitivity estimator: pool eligible successes and eligible Explore decisions across junctions.",
        "- Strict joint gate: a primary session must pass equal-junction four-policy support, equal-junction depth support, event-pooled four-policy support, and event-pooled depth support at 30 minutes.",
        "- Outcome: whole-session half-coverage efficiency `32 / N32`, where `N32` is endpoint visits needed to encounter 32 unique endpoints.",
        "",
        "## Strict joint cohort",
        "",
        markdown_table(counts),
        "",
        markdown_table(exclusions),
        "",
        markdown_table(overlap),
        "",
        "The joint gate retains 22/23 Naive sessions and 58/58 Return sessions. `640day1` is the only primary-group exclusion because its equal-junction and event-pooled shallow-depth denominators are below 10. The cohort contains 11 mice represented in both groups; all confidence intervals for group-gap reconstruction sample the 29-mouse union once per draw.",
        "",
        "## Observed policy contrasts",
        "",
        markdown_table(policy),
        "",
        "Return-minus-Naive inference uses CR1 mouse-clustered covariance with Student t reference on the strict joint cohort.",
        "",
        "### Natural-support Second-day control",
        "",
        markdown_table(natural_second),
        "",
        "The Second-day-minus-Return control uses its natural 74 Second-day and 58 Return sessions; it is not artificially restricted by the Naive/Return strict cohort.",
        "",
        "## Depth-specific stemward-exit contrast",
        "",
        markdown_table(depth),
        "",
        "## Equal-junction biased-walk calibration",
        "",
        markdown_table(calibration),
        "",
        "## Event-pooled sensitivity calibration",
        "",
        markdown_table(event_calibration),
        "",
        "Own-policy calibration uses the predeclared joint gate `|mean error| <= 0.05` and Spearman rho `>= 0.40`. Passing this gate supports numerical calibration of the stationary walk; it does not establish biological sufficiency.",
        "",
        "## Gap recapitulation and robustness",
        "",
        markdown_table(robustness),
        "",
        f"Each own-policy, donor-policy, and hybrid condition was rerun with {n_walks} walks per session. Each interval uses {n_bootstrap} synchronized union-of-mice bootstrap draws. Donor means are held fixed at the full strict-cohort estimates and Monte Carlo walk noise is not propagated.",
        "",
        "## Donor refit audit",
        "",
        markdown_table(donor_audit),
        "",
        "## Exposure audit",
        "",
        markdown_table(exposure_summary),
        "",
        "Equal TimeInMaze does not equalize Explore duration or decision evidence; those quantities are disclosed rather than tuned.",
        "",
        "## Predictor-outcome temporality audit",
        "",
        markdown_table(outcome_summary),
        "",
        "Because many sessions have already reached 32 unique endpoints by 30 minutes, the policy window and whole-session efficiency are partly contemporaneous. Prospective, predictive, causal, mediation, R-squared, and variance-explained wording is not fair.",
        "",
        "## Interpretation boundary",
        "",
        "Thirty minutes is promoted for cross-paper consistency with other TimeInMaze analyses, not because it maximizes an observed effect, neural correlation, calibration statistic, or simulated fraction. The exact conclusion-bearing cohort, estimator, point weighting, and uncertainty unit are fixed and exported. Event pooling and mouse-equal weighting remain sensitivities.",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_qa(
    output: Path,
    *,
    ledger: pd.DataFrame,
    strict_equal: pd.DataFrame,
    equal_tables: dict[str, pd.DataFrame],
    event_tables: dict[str, pd.DataFrame],
    robustness: pd.DataFrame,
    source_paths: dict[str, Path],
) -> Path:
    included = ledger.loc[ledger["included_primary_30min_strict_joint_cohort"]]
    checks = {
        "strict_cohort_is_22_naive_58_return": (
            included.groupby("group")["nickname"].nunique().to_dict()
            == EXPECTED_PRIMARY_COUNTS
        ),
        "only_640day1_excluded_from_primary_groups": (
            set(
                ledger.loc[
                    ledger["in_primary_groups"]
                    & ~ledger["included_primary_30min_strict_joint_cohort"],
                    "nickname",
                ].astype(str)
            )
            == EXPECTED_PRIMARY_EXCLUSIONS
        ),
        "all_80_strict_primary_rows_fully_observe_30min_clock": (
            len(strict_equal) == 80
            and bool(strict_equal["window_fully_observed"].fillna(False).all())
        ),
        "all_equal_simulation_rows_depth_supported": bool(
            equal_tables["simulation"]["simulation_depth_support"].all()
        ),
        "all_event_simulation_rows_supported": bool(
            event_tables["results"]["simulation_supported"].all()
        ),
        "gap_robustness_has_8_rows": len(robustness) == 8,
        "all_source_tables_nonempty_except_expected_exclusion_table": all(
            path.stat().st_size > 0 for path in source_paths.values()
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Promoted 30-minute QA failed: {failed}")
    lines = [
        "# QA record",
        "",
        *[f"- PASS: {name}" for name in checks],
        "- PASS: input policies were filtered to exactly 30 cumulative TimeInMaze minutes.",
        "- PASS: policy values were not selected or tuned by effect size or neural correlation.",
        "- PASS: own, donor, and hybrid simulations were rerun after strict-joint filtering.",
        "- PASS: session-weighted and mouse-equal results use synchronized union-of-mice intervals.",
        "- LIMIT: fixed full-cohort donor means and Monte Carlo walk noise are not propagated in confidence intervals.",
        "- LIMIT: the whole-session half-coverage outcome is partly contemporaneous with the 30-minute predictor window.",
    ]
    path = output / "QA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-source-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_source_dir.resolve()
    output = args.output_dir.resolve()
    source_dir = output / "source_tables"
    output.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    configure_modules(output)

    full_equal, full_event = load_input_policies(input_dir)
    ledger, membership, overlap = build_strict_joint_ledger(full_equal, full_event)
    strict_equal = filter_to_membership(full_equal, membership)
    strict_event = filter_to_membership(full_event, membership)
    if len(strict_equal) != 80 or not strict_equal["window_fully_observed"].fillna(False).all():
        raise ValueError(
            "The promoted fixed-clock cohort must contain 80 sessions that each "
            "fully observe the first 30 TimeInMaze minutes"
        )
    inference = policy_inference_tables(full_equal, strict_equal)
    equal_tables = run_equal_junction(
        strict_equal,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    event_tables = run_event_pooled(
        strict_event,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 200000,
    )
    robustness, mouse_draws = build_gap_robustness(
        equal_tables,
        event_tables,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 700000,
    )
    exposure_session, exposure_summary = build_exposure_audits(
        strict_equal, strict_event
    )
    outcome_session, outcome_summary = build_outcome_audits(input_dir, membership)
    donor_audit = donor_refit_audit(
        full_equal, strict_equal, full_event, strict_event
    )
    source_paths = write_tables(
        source_dir,
        full_equal=full_equal,
        full_event=full_event,
        strict_equal=strict_equal,
        strict_event=strict_event,
        ledger=ledger,
        membership=membership,
        overlap=overlap,
        inference=inference,
        equal_tables=equal_tables,
        event_tables=event_tables,
        robustness=robustness,
        mouse_draws=mouse_draws,
        exposure_session=exposure_session,
        exposure_summary=exposure_summary,
        outcome_session=outcome_session,
        outcome_summary=outcome_summary,
        donor_audit=donor_audit,
    )
    report = write_report(
        output,
        ledger=ledger,
        overlap=overlap,
        inference=inference,
        equal_tables=equal_tables,
        event_tables=event_tables,
        robustness=robustness,
        exposure_summary=exposure_summary,
        outcome_summary=outcome_summary,
        donor_audit=donor_audit,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
    )
    qa = write_qa(
        output,
        ledger=ledger,
        strict_equal=strict_equal,
        equal_tables=equal_tables,
        event_tables=event_tables,
        robustness=robustness,
        source_paths=source_paths,
    )

    hash_targets = [*source_paths.values(), report, qa]
    hashes = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(output)),
                "sha256": sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in sorted(hash_targets)
        ]
    )
    hashes_path = source_dir / "artifact_sha256.csv"
    hashes.to_csv(hashes_path, index=False)

    manifest = {
        "created": "2026-08-04",
        "status": "promoted_fixed30min_analysis_authority",
        "script": Path(__file__),
        "window_minutes": WINDOW_MINUTES,
        "timebase": "cumulative all-mode TimeInMaze from session start",
        "behavior_mode": "Explore decisions only; non-Explore modes consume clock",
        "analysis_contract": {
            "window": "first 30 minutes of cumulative all-mode TimeInMaze from session start",
            "behavior_mode": "Explore decisions only; non-Explore modes consume clock",
            "primary_estimator": "equal-junction unweighted mean of supported node/side conditional probabilities",
            "sensitivity_estimator": "eligible-event-pooled conditional probabilities",
            "primary_support": "joint intersection of equal four, equal strict depth, event four, and event strict depth gates at 30 minutes",
            "outcome": "whole-session half_cover_eff = 32 / endpoint visits needed for 32 unique endpoints",
            "primary_point_weighting": "session-weighted within group",
            "sensitivity_point_weighting": "mouse-equal after averaging repeated sessions within mouse",
            "uncertainty": "synchronized union-of-mice cluster bootstrap",
            "claim_limit": "group-gap reconstruction diagnostic, not R-squared, variance explained, mediation, prediction, or causation",
        },
        "parameters": {
            "window_minutes": WINDOW_MINUTES,
            "equal_junction_min_total_events": int(base.equal.MIN_EVENTS),
            "equal_junction_min_supported_node_or_side_estimates_per_policy": int(
                base.equal.MIN_POLICY_SUPPORT
            ),
            "equal_junction_min_raw_events_per_depth_stratum": int(
                base.equal.MIN_POLICY_SUPPORT
            ),
            "event_pooled_min_raw_events_per_parameter": int(
                base.pooled.MIN_DENOMINATOR
            ),
            "n_walks_per_session_condition": int(args.n_walks),
            "n_synchronized_union_mouse_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "donor_uncertainty": "fixed strict-cohort donor means; donor estimation and Monte Carlo noise not propagated",
        },
        "cohort": {
            "session_counts": EXPECTED_PRIMARY_COUNTS,
            "excluded_primary_sessions": sorted(EXPECTED_PRIMARY_EXCLUSIONS),
            "all_80_strict_primary_rows_window_fully_observed": True,
            **EXPECTED_OVERLAP,
        },
        "inputs": {
            "equal_policy_rows": input_dir / "fixed_time_policy_rows.csv",
            "event_policy_rows": input_dir / "event_pooled_policy_rows.csv",
            "predictor_outcome_audit": input_dir
            / "predictor_outcome_overlap_session_audit.csv",
            "whole_session_scalar_target": base.equal.SCALAR_TABLE,
        },
        "input_sha256": {
            "equal_policy_rows": sha256(input_dir / "fixed_time_policy_rows.csv"),
            "event_policy_rows": sha256(input_dir / "event_pooled_policy_rows.csv"),
            "predictor_outcome_audit": sha256(
                input_dir / "predictor_outcome_overlap_session_audit.csv"
            ),
            "whole_session_scalar_target": sha256(base.equal.SCALAR_TABLE),
        },
        "source_tables": sorted(source_paths.values()),
        "report": report,
        "qa": qa,
        "artifact_hashes": hashes_path,
        "output_sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in [*hash_targets, hashes_path]
        },
        "live_script_sha256": sha256(Path(__file__)),
        "command": [
            sys.executable,
            str(Path(__file__).relative_to(REPO_ROOT)),
            *sys.argv[1:],
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8"
    )

    print("strict_joint_counts=" + str(EXPECTED_PRIMARY_COUNTS))
    print("equal_gap")
    print(equal_tables["gap"].to_string(index=False))
    print("event_gap")
    print(event_tables["gap"].to_string(index=False))
    print("manifest=" + str(manifest_path))


if __name__ == "__main__":
    main()
