"""Rerun ED2 policy reconstruction with each mouse's own Naive baseline.

The accepted ED2 analysis assigns every Return session one pooled Naive donor
vector.  This task-scoped rerun instead maps every displayed Return session to
the same animal's original Naive Day-1 session.  Multiple Return sessions from
one animal reuse that baseline.  The simulation target remains the Return
session's whole-session Explore endpoint count.

Primary point estimates give equal weight to Return sessions, matching the
accepted panel.  Confidence intervals resample animals and retain all Return
sessions from each sampled animal.  Animal-equal summaries are exported as a
sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_ed03_turning_bias_fixed_time as equal
from ca_analysis.exploration import (
    compute_scalar_efficiency_metrics,
    extract_explore_endnode_sequence,
)
from ca_analysis.maze import NewMaze
from ca_analysis.trajectory import SplitModeClips


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_RECORD = (
    REPO_ROOT
    / "docs"
    / "figure_version_candidates"
    / "ed02_1616_restored_depth_common_i_candidate_2026-08-20"
)
RETURN_POLICY_PATH = (
    ACCEPTED_RECORD
    / "analysis_inputs"
    / "strict"
    / "primary_30min_strict_joint_equal_junction_policy_rows.csv"
)
ACCEPTED_COUNTERFACTUAL_PATH = (
    ACCEPTED_RECORD / "cohort_audit" / "depth_common_counterfactual_session_rows.csv"
)
ACCEPTED_PANEL_D_PATH = ACCEPTED_RECORD / "plot_ready" / "panel_d_policy_contributions.csv"
ACCEPTED_PANEL_H_PATH = ACCEPTED_RECORD / "plot_ready" / "panel_h_four_vs_depth_gap.csv"
FULL_POLICY_PATH = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed_time_20_30min_explore_2026-08-04"
    / "source_tables"
    / "fixed_time_policy_rows.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "ed02_matched_naive_reconstruction_1616_2026-08-20"
)

WINDOW_MINUTES = 30
FOUR_PARAMETERS = ("Bf", "Ba", "Lf", "Lo")
DEPTH_PARAMETERS = (*FOUR_PARAMETERS, "Lo_shallow", "Lo_mid", "Lo_deep")
POLICY_LABELS = {"Bf": "P_SF", "Ba": "P_SA", "Lf": "P_BF", "Lo": "P_BS"}

# These are the original Day-1 sessions for the 18 mice represented by the
# full 58-session pooled Return roster.  The mapping is biological and is not
# inferred from membership in the pooled Naive comparison table.
MATCHED_NAIVE_SESSION = {
    "418": "418RRday1",
    "422": "422RRday1",
    "434": "434RRday1",
    "451": "451Rday1",
    "468": "468Lday1",
    "488": "488Rday1",
    "489": "489Lday1",
    "579": "579RLday1",
    "618": "618day1",
    "640": "640day1",
    "644": "644day1",
    "646": "646day1",
    "837": "837day1",
    "838": "838day1",
    "839": "839day1",
    "842": "842day1",
    "862": "862day1",
    "1616": "1616day1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else np.nan


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional tabulate dependency."""

    if frame.empty:
        return "(no rows)"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if not np.isfinite(value) else f"{value:.6g}"
            )
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_accepted_return_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = pd.read_csv(RETURN_POLICY_PATH)
    accepted = pd.read_csv(ACCEPTED_COUNTERFACTUAL_PATH)
    accepted = accepted.loc[
        accepted["group"].eq("LTM")
        & accepted["simulation_depth_support"].astype(bool)
    ].copy()
    accepted["animal_id"] = accepted["animal_id"].astype(str)
    policy["animal_id"] = policy["animal_id"].astype(str)
    return_names = set(accepted["nickname"].astype(str))
    policy = policy.loc[
        policy["group"].eq("LTM")
        & policy["window_minutes"].eq(WINDOW_MINUTES)
        & policy["nickname"].astype(str).isin(return_names)
    ].copy()
    if len(accepted) != 58 or accepted["nickname"].nunique() != 58:
        raise ValueError(f"Expected 58 accepted Return rows, found {len(accepted)}")
    if len(policy) != 58 or policy["nickname"].nunique() != 58:
        raise ValueError(f"Expected 58 matched Return policy rows, found {len(policy)}")
    missing_animals = set(policy["animal_id"]) - set(MATCHED_NAIVE_SESSION)
    if missing_animals:
        raise ValueError(f"No Naive-session mapping for animals: {sorted(missing_animals)}")
    own_columns = [
        "nickname",
        "animal_id",
        "real_half_cover_eff",
        "n_endnodes_target",
        "n_walks",
        "four_own_half_cover_eff",
        "depth_own_half_cover_eff",
    ]
    accepted = accepted[own_columns].rename(
        columns={
            "real_half_cover_eff": "return_observed_efficiency",
            "n_endnodes_target": "return_n_endnodes_target",
            "n_walks": "accepted_n_walks",
            "four_own_half_cover_eff": "return_four_own_simulated_efficiency",
            "depth_own_half_cover_eff": "return_depth_own_simulated_efficiency",
        }
    )
    merged = policy.merge(
        accepted, on=["nickname", "animal_id"], how="inner", validate="one_to_one"
    )
    return merged.reset_index(drop=True), accepted.reset_index(drop=True)


def extract_matched_naive_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute policy and scalar inputs for every biological baseline."""

    equal.legacy.bootstrap_environment(REPO_ROOT)
    pairs = [(session, "matched_naive") for session in MATCHED_NAIVE_SESSION.values()]
    trajectories, valid_pairs, skipped = equal.legacy._load_trajectories(pairs)
    loaded = {str(session) for session, _ in valid_pairs}
    expected = set(MATCHED_NAIVE_SESSION.values())
    if loaded != expected:
        raise ValueError(
            f"Matched Naive trajectory coverage differs: missing={sorted(expected-loaded)}, "
            f"unexpected={sorted(loaded-expected)}"
        )

    maze = NewMaze(6)
    policy_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    for session, _ in valid_pairs:
        animal = next(key for key, value in MATCHED_NAIVE_SESSION.items() if value == session)
        tf = trajectories[session]
        clips = np.asarray(SplitModeClips(tf, maze))
        explore = clips[np.where(clips[:, 3] == 2)] if clips.size else np.zeros((0, 4))
        clips_fn = lambda _tf, _ma, selected=explore: selected
        cutoff = float(WINDOW_MINUTES * 60)
        four = equal.compute_turning_bias_summary(
            tf,
            maze,
            split_mode_clips_fn=clips_fn,
            max_maze_seconds=cutoff,
            fps=equal.FPS,
        )
        depth = equal.compute_depth_stratified_lo(
            tf,
            maze,
            split_mode_clips_fn=clips_fn,
            max_maze_seconds=cutoff,
            fps=equal.FPS,
        )
        explore_seconds, n_clips = equal.explore_time_within_cutoff(
            tf, explore, cutoff
        )
        available = equal.session_maze_minutes(tf)
        policy_rows.append(
            {
                "window_minutes": WINDOW_MINUTES,
                "window_label": "First 30 min",
                "window_definition": (
                    "Explore-mode decisions at cumulative in-maze time <= cutoff "
                    "from session start"
                ),
                "nickname": session,
                "group": "matched_naive",
                "animal_id": animal,
                "available_maze_minutes": available,
                "window_fully_observed": bool(available >= WINDOW_MINUTES),
                "explore_minutes_within_window": explore_seconds / 60.0,
                "n_explore_clips_intersecting_window": int(n_clips),
                **four,
                **depth,
            }
        )

        sequence = extract_explore_endnode_sequence(tf, maze)
        scalar = compute_scalar_efficiency_metrics(
            np.asarray(sequence, dtype=int), num_states=32
        )
        scalar_rows.append(
            {
                "nickname": session,
                "group": "matched_naive",
                "animal_id": animal,
                **scalar,
            }
        )

    policy = equal.add_support_flags(pd.DataFrame(policy_rows))
    scalar = pd.DataFrame(scalar_rows)
    skipped_frame = pd.DataFrame(skipped)
    if skipped_frame.empty:
        skipped_frame = pd.DataFrame(columns=["nickname", "group", "error"])
    return policy.sort_values("animal_id").reset_index(drop=True), scalar.sort_values(
        "animal_id"
    ).reset_index(drop=True), skipped_frame


def build_recomputation_parity(
    baseline_policy: pd.DataFrame, baseline_scalar: pd.DataFrame
) -> pd.DataFrame:
    existing_policy = pd.read_csv(FULL_POLICY_PATH)
    existing_policy["animal_id"] = existing_policy["animal_id"].astype(str)
    existing_policy = existing_policy.loc[
        existing_policy["window_minutes"].eq(WINDOW_MINUTES)
        & existing_policy["nickname"].isin(MATCHED_NAIVE_SESSION.values())
    ].copy()
    existing_scalar = equal.load_scalar_table()
    existing_scalar["animal_id"] = existing_scalar["nickname"].map(
        lambda value: next(
            (key for key, session in MATCHED_NAIVE_SESSION.items() if session == value), ""
        )
    )
    existing_scalar = existing_scalar.loc[
        existing_scalar["nickname"].isin(MATCHED_NAIVE_SESSION.values())
    ].copy()

    rows: list[dict[str, Any]] = []
    policy_columns = [
        *FOUR_PARAMETERS,
        "Lo_shallow",
        "Lo_mid",
        "Lo_deep",
        "n_events",
        "n_Bf",
        "n_Ba",
        "n_Lf",
        "n_Lo",
        "n_Lo_shallow",
        "n_Lo_mid",
        "n_Lo_deep",
    ]
    for row in baseline_policy.itertuples(index=False):
        old = existing_policy.loc[existing_policy["nickname"].eq(row.nickname)]
        if old.empty:
            rows.append(
                {
                    "nickname": row.nickname,
                    "animal_id": row.animal_id,
                    "source": "policy",
                    "metric": "all",
                    "status": "newly_recovered_baseline_not_in_pooled_table",
                    "old_value": np.nan,
                    "recomputed_value": np.nan,
                    "absolute_difference": np.nan,
                }
            )
        else:
            old_row = old.iloc[0]
            for column in policy_columns:
                new_value = float(getattr(row, column))
                old_value = float(old_row[column])
                rows.append(
                    {
                        "nickname": row.nickname,
                        "animal_id": row.animal_id,
                        "source": "policy",
                        "metric": column,
                        "status": "parity_checked",
                        "old_value": old_value,
                        "recomputed_value": new_value,
                        "absolute_difference": abs(new_value - old_value),
                    }
                )

    for row in baseline_scalar.itertuples(index=False):
        old = existing_scalar.loc[existing_scalar["nickname"].eq(row.nickname)]
        if old.empty:
            rows.append(
                {
                    "nickname": row.nickname,
                    "animal_id": row.animal_id,
                    "source": "scalar",
                    "metric": "half_cover_eff",
                    "status": "newly_recovered_baseline_not_in_pooled_table",
                    "old_value": np.nan,
                    "recomputed_value": float(row.half_cover_eff),
                    "absolute_difference": np.nan,
                }
            )
        else:
            old_value = float(old.iloc[0]["half_cover_eff"])
            rows.append(
                {
                    "nickname": row.nickname,
                    "animal_id": row.animal_id,
                    "source": "scalar",
                    "metric": "half_cover_eff",
                    "status": "parity_checked",
                    "old_value": old_value,
                    "recomputed_value": float(row.half_cover_eff),
                    "absolute_difference": abs(float(row.half_cover_eff) - old_value),
                }
            )
    parity = pd.DataFrame(rows)
    checked = parity.loc[parity["status"].eq("parity_checked"), "absolute_difference"]
    if len(checked) and float(checked.max()) > 1e-12:
        raise ValueError(
            f"Recomputed baseline inputs failed parity; max delta={checked.max()}"
        )
    return parity


def build_pairing_ledger(
    return_input: pd.DataFrame,
    baseline_policy: pd.DataFrame,
    baseline_scalar: pd.DataFrame,
) -> pd.DataFrame:
    pairs = return_input.copy()
    pairs["matched_naive_session"] = pairs["animal_id"].map(MATCHED_NAIVE_SESSION)
    baseline_policy = baseline_policy.add_prefix("naive_").rename(
        columns={
            "naive_nickname": "matched_naive_session",
            "naive_animal_id": "animal_id",
        }
    )
    baseline_scalar = baseline_scalar[
        ["nickname", "animal_id", "n_explore_endnodes", "half_cover_eff"]
    ].add_prefix("naive_").rename(
        columns={
            "naive_nickname": "matched_naive_session",
            "naive_animal_id": "animal_id",
            "naive_half_cover_eff": "naive_observed_efficiency",
            "naive_n_explore_endnodes": "naive_n_endnodes_target",
        }
    )
    pairs = pairs.merge(
        baseline_policy,
        on=["animal_id", "matched_naive_session"],
        how="left",
        validate="many_to_one",
    ).merge(
        baseline_scalar,
        on=["animal_id", "matched_naive_session"],
        how="left",
        validate="many_to_one",
    )
    if pairs["naive_Bf"].isna().any() or pairs["naive_observed_efficiency"].isna().any():
        bad = pairs.loc[
            pairs["naive_Bf"].isna() | pairs["naive_observed_efficiency"].isna(),
            ["nickname", "animal_id", "matched_naive_session"],
        ]
        raise ValueError(f"Unresolved matched Naive inputs:\n{bad.to_string(index=False)}")

    pairs["pair_four_primary_eligible"] = (
        pairs["primary_policy_support"].astype(bool)
        & pairs["naive_primary_policy_support"].astype(bool)
    )
    pairs["pair_depth_primary_eligible"] = (
        pairs["primary_depth_support"].astype(bool)
        & pairs["naive_primary_depth_support"].astype(bool)
    )
    pairs["pair_common_primary_eligible"] = (
        pairs["pair_four_primary_eligible"]
        & pairs["pair_depth_primary_eligible"]
    )
    pairs["pair_four_simulation_eligible"] = (
        pairs["simulation_four_support"].astype(bool)
        & pairs["naive_simulation_four_support"].astype(bool)
    )
    pairs["pair_depth_simulation_eligible"] = (
        pairs["simulation_depth_support"].astype(bool)
        & pairs["naive_simulation_depth_support"].astype(bool)
    )
    pairs["observed_paired_difference"] = (
        pairs["return_observed_efficiency"] - pairs["naive_observed_efficiency"]
    )
    return pairs.sort_values(["animal_id", "nickname"]).reset_index(drop=True)


def run_matched_simulations(
    pairs: pd.DataFrame, *, n_walks: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        output = row._asdict()
        target = int(row.return_n_endnodes_target)
        own_four = tuple(float(getattr(row, parameter)) for parameter in FOUR_PARAMETERS)
        naive_four = tuple(
            float(getattr(row, f"naive_{parameter}")) for parameter in FOUR_PARAMETERS
        )
        naive_depth = tuple(
            float(getattr(row, f"naive_{parameter}")) for parameter in DEPTH_PARAMETERS
        )

        if bool(row.pair_four_simulation_eligible):
            value, short = equal.simulate_single(
                naive_four,
                target,
                n_walks,
                stable_seed(seed, row.nickname, row.matched_naive_session, "matched_four"),
            )
            output["matched_naive_four_simulated_efficiency"] = value
            output["matched_naive_four_short_sequences"] = short
            for index, parameter in enumerate(FOUR_PARAMETERS):
                hybrid = list(own_four)
                hybrid[index] = naive_four[index]
                value, short = equal.simulate_single(
                    tuple(hybrid),
                    target,
                    n_walks,
                    stable_seed(
                        seed,
                        row.nickname,
                        row.matched_naive_session,
                        f"matched_hybrid_{parameter}",
                    ),
                )
                output[f"matched_naive_{parameter}_hybrid_efficiency"] = value
                output[f"matched_naive_{parameter}_hybrid_short_sequences"] = short
        else:
            output["matched_naive_four_simulated_efficiency"] = np.nan
            output["matched_naive_four_short_sequences"] = np.nan
            for parameter in FOUR_PARAMETERS:
                output[f"matched_naive_{parameter}_hybrid_efficiency"] = np.nan
                output[f"matched_naive_{parameter}_hybrid_short_sequences"] = np.nan

        if bool(row.pair_depth_simulation_eligible):
            value, short = equal.simulate_depth(
                naive_depth,
                target,
                n_walks,
                stable_seed(seed, row.nickname, row.matched_naive_session, "matched_depth"),
            )
            output["matched_naive_depth_simulated_efficiency"] = value
            output["matched_naive_depth_short_sequences"] = short
        else:
            output["matched_naive_depth_simulated_efficiency"] = np.nan
            output["matched_naive_depth_short_sequences"] = np.nan

        output["n_walks_matched_condition"] = int(n_walks)
        rows.append(output)

    result = pd.DataFrame(rows)
    result["four_simulated_paired_difference"] = (
        result["return_four_own_simulated_efficiency"]
        - result["matched_naive_four_simulated_efficiency"]
    )
    result["depth_simulated_paired_difference"] = (
        result["return_depth_own_simulated_efficiency"]
        - result["matched_naive_depth_simulated_efficiency"]
    )
    for parameter in FOUR_PARAMETERS:
        result[f"contribution_{parameter}"] = (
            result["return_four_own_simulated_efficiency"]
            - result[f"matched_naive_{parameter}_hybrid_efficiency"]
        )
    return result


def _sample_animals(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    animals = np.asarray(sorted(frame["animal_id"].astype(str).unique()), dtype=object)
    sampled = rng.choice(animals, size=len(animals), replace=True)
    parts: list[pd.DataFrame] = []
    for replicate, animal in enumerate(sampled):
        part = frame.loc[frame["animal_id"].astype(str).eq(str(animal))].copy()
        part["bootstrap_cluster_replicate"] = int(replicate)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _gap_values(frame: pd.DataFrame) -> dict[str, float]:
    observed = float(frame["observed_paired_difference"].mean())
    four = float(frame["four_simulated_paired_difference"].mean())
    depth = float(frame["depth_simulated_paired_difference"].mean())
    return {
        "delta_real": observed,
        "delta_four": four,
        "delta_depth": depth,
        "fraction_four": four / observed if abs(observed) > 1e-12 else np.nan,
        "fraction_depth": depth / observed if abs(observed) > 1e-12 else np.nan,
        **{
            f"contribution_{parameter}": float(frame[f"contribution_{parameter}"].mean())
            for parameter in FOUR_PARAMETERS
        },
    }


def summarize_primary(
    simulation: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = simulation.loc[simulation["pair_common_primary_eligible"]].copy()
    four_support = simulation.loc[simulation["pair_four_primary_eligible"]].copy()
    if common.empty or four_support.empty:
        raise ValueError("No eligible matched pairs")
    point = _gap_values(common)
    rng = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    contribution_draws: list[dict[str, Any]] = []
    for draw in range(n_bootstrap):
        sampled_common = _sample_animals(common, rng)
        values = _gap_values(sampled_common)
        draws.append({"bootstrap_draw": draw, **values})
        sampled_four = _sample_animals(four_support, rng)
        contribution_draws.append(
            {
                "bootstrap_draw": draw,
                **{
                    f"contribution_{parameter}": float(
                        sampled_four[f"contribution_{parameter}"].mean()
                    )
                    for parameter in FOUR_PARAMETERS
                },
            }
        )
    draws_frame = pd.DataFrame(draws)
    contribution_draws_frame = pd.DataFrame(contribution_draws)

    gap_rows: list[dict[str, Any]] = []
    for model, delta_key, fraction_key in (
        ("four_bias", "delta_four", "fraction_four"),
        ("depth_stratified", "delta_depth", "fraction_depth"),
    ):
        gap_rows.append(
            {
                "window_minutes": WINDOW_MINUTES,
                "model": model,
                "n_naive_rows": int(common["matched_naive_session"].nunique()),
                "n_return_rows": int(len(common)),
                "n_naive_mice": int(common["animal_id"].nunique()),
                "n_return_mice": int(common["animal_id"].nunique()),
                "delta_real": point["delta_real"],
                "delta_simulated": point[delta_key],
                "fraction_recapitulated": point[fraction_key],
                "fraction_ci_low": _percentile(draws_frame[fraction_key], 2.5),
                "fraction_ci_high": _percentile(draws_frame[fraction_key], 97.5),
                "point_weighting": "session-weighted matched Return-Naive pairs",
                "uncertainty": (
                    "animal-cluster bootstrap retaining all matched Return-Naive pairs; "
                    "Monte Carlo simulation noise not propagated"
                ),
                "model_norm": model,
                "delta_simulated_ci_low": _percentile(draws_frame[delta_key], 2.5),
                "delta_simulated_ci_high": _percentile(draws_frame[delta_key], 97.5),
                "denominator_definition": (
                    "mean across Return sessions of observed Return minus same-animal "
                    "Naive efficiency"
                ),
                "numerator_definition": (
                    "mean across Return sessions of own-policy simulation minus simulation "
                    "using the same animal's Naive policy vector"
                ),
            }
        )
    gap = pd.DataFrame(gap_rows)

    contribution_rows: list[dict[str, Any]] = []
    contribution_point = {
        parameter: float(four_support[f"contribution_{parameter}"].mean())
        for parameter in FOUR_PARAMETERS
    }
    for parameter in FOUR_PARAMETERS:
        key = f"contribution_{parameter}"
        contribution_rows.append(
            {
                "window_minutes": WINDOW_MINUTES,
                "parameter": parameter,
                "n_return_rows": int(len(four_support)),
                "n_return_mice": int(four_support["animal_id"].nunique()),
                "delta_efficiency_mean": contribution_point[parameter],
                "delta_efficiency_ci_low": _percentile(
                    contribution_draws_frame[key], 2.5
                ),
                "delta_efficiency_ci_high": _percentile(
                    contribution_draws_frame[key], 97.5
                ),
                "contrast": (
                    "Return own policy minus Return with that parameter replaced by "
                    "the same animal's Naive-session value"
                ),
                "point_weighting": "session-weighted matched Return-Naive pairs",
                "uncertainty": (
                    "animal-cluster bootstrap retaining all matched Return-Naive pairs; "
                    "Monte Carlo simulation noise not propagated"
                ),
            }
        )
    return gap, pd.DataFrame(contribution_rows), draws_frame, contribution_draws_frame


def summarize_mouse_equal(
    simulation: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = simulation.loc[simulation["pair_common_primary_eligible"]].copy()
    value_columns = [
        "observed_paired_difference",
        "four_simulated_paired_difference",
        "depth_simulated_paired_difference",
    ]
    animal = common.groupby("animal_id", as_index=False)[value_columns].mean()

    def compute(frame: pd.DataFrame) -> dict[str, float]:
        observed = float(frame["observed_paired_difference"].mean())
        four = float(frame["four_simulated_paired_difference"].mean())
        depth = float(frame["depth_simulated_paired_difference"].mean())
        return {
            "delta_real": observed,
            "delta_four": four,
            "delta_depth": depth,
            "fraction_four": four / observed if abs(observed) > 1e-12 else np.nan,
            "fraction_depth": depth / observed if abs(observed) > 1e-12 else np.nan,
        }

    point = compute(animal)
    rng = np.random.default_rng(seed)
    draws: list[dict[str, Any]] = []
    for draw in range(n_bootstrap):
        indices = rng.integers(0, len(animal), size=len(animal))
        draws.append({"bootstrap_draw": draw, **compute(animal.iloc[indices])})
    draw_frame = pd.DataFrame(draws)
    rows = []
    for model, delta, fraction in (
        ("four_bias", "delta_four", "fraction_four"),
        ("depth_stratified", "delta_depth", "fraction_depth"),
    ):
        rows.append(
            {
                "model": model,
                "n_mice": int(len(animal)),
                "n_return_rows": int(len(common)),
                "delta_real": point["delta_real"],
                "delta_simulated": point[delta],
                "fraction_recapitulated": point[fraction],
                "fraction_ci_low": _percentile(draw_frame[fraction], 2.5),
                "fraction_ci_high": _percentile(draw_frame[fraction], 97.5),
                "point_weighting": (
                    "animal-equal after averaging matched Return-session differences "
                    "within mouse"
                ),
            }
        )
    return pd.DataFrame(rows), draw_frame


def build_legacy_comparison(
    gap: pd.DataFrame, contributions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_gap = pd.read_csv(ACCEPTED_PANEL_H_PATH)
    gap_compare = old_gap[
        [
            "model",
            "delta_real",
            "delta_simulated",
            "fraction_recapitulated",
            "fraction_ci_low",
            "fraction_ci_high",
            "n_return_rows",
            "n_return_mice",
        ]
    ].merge(
        gap[
            [
                "model",
                "delta_real",
                "delta_simulated",
                "fraction_recapitulated",
                "fraction_ci_low",
                "fraction_ci_high",
                "n_return_rows",
                "n_return_mice",
            ]
        ],
        on="model",
        suffixes=("_pooled_naive", "_matched_naive"),
        validate="one_to_one",
    )
    gap_compare["fraction_change_matched_minus_pooled"] = (
        gap_compare["fraction_recapitulated_matched_naive"]
        - gap_compare["fraction_recapitulated_pooled_naive"]
    )

    old_d = pd.read_csv(ACCEPTED_PANEL_D_PATH)
    contribution_compare = old_d[
        ["parameter", "delta_efficiency_mean", "delta_efficiency_ci_low", "delta_efficiency_ci_high"]
    ].merge(
        contributions[
            ["parameter", "delta_efficiency_mean", "delta_efficiency_ci_low", "delta_efficiency_ci_high"]
        ],
        on="parameter",
        suffixes=("_pooled_naive", "_matched_naive"),
        validate="one_to_one",
    )
    contribution_compare["mean_change_matched_minus_pooled"] = (
        contribution_compare["delta_efficiency_mean_matched_naive"]
        - contribution_compare["delta_efficiency_mean_pooled_naive"]
    )
    return gap_compare, contribution_compare


def render_audit_figures(
    simulation: pd.DataFrame,
    gap_compare: pd.DataFrame,
    output: Path,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    common = simulation.loc[simulation["pair_common_primary_eligible"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), constrained_layout=True)
    for axis, parameter in zip(axes.ravel(), FOUR_PARAMETERS):
        x = common[f"naive_{parameter}"].to_numpy(dtype=float)
        y = common[parameter].to_numpy(dtype=float)
        axis.scatter(x, y, s=12, color="#555555", alpha=0.65, linewidths=0)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        axis.plot([lo, hi], [lo, hi], color="#AAAAAA", linewidth=0.8, linestyle="--")
        axis.set_title(POLICY_LABELS[parameter], loc="left", fontsize=8)
        axis.set_xlabel("Same-animal Naive policy")
        axis.set_ylabel("Return policy")
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        path = output / f"matched_policy_inputs.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    ordered = common.sort_values("observed_paired_difference").reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    axis.axhline(0, color="#BBBBBB", linewidth=0.8)
    axis.plot(x, ordered["observed_paired_difference"], color="#222222", label="Observed")
    axis.plot(
        x,
        ordered["four_simulated_paired_difference"],
        color="#4C78A8",
        label="Four-policy simulation",
    )
    axis.plot(
        x,
        ordered["depth_simulated_paired_difference"],
        color="#E15759",
        label="Depth-varying simulation",
    )
    axis.set_xlabel("Matched Return-Naive pairs, ordered by observed difference")
    axis.set_ylabel("Return minus Naive efficiency")
    axis.legend(frameon=False, ncol=3, fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        path = output / f"paired_difference_inputs.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    positions = np.arange(len(gap_compare))
    width = 0.34
    axis.bar(
        positions - width / 2,
        100 * gap_compare["fraction_recapitulated_pooled_naive"],
        width,
        color="#BBBBBB",
        label="Pooled Naive mean (superseded)",
    )
    axis.bar(
        positions + width / 2,
        100 * gap_compare["fraction_recapitulated_matched_naive"],
        width,
        color=["#4C78A8", "#E15759"],
        label="Same-animal Naive",
    )
    axis.set_xticks(positions, ["Four-policy", "Depth-varying"])
    axis.set_ylabel("Observed paired gap reconstructed (%)")
    axis.legend(frameon=False, fontsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        path = output / f"pooled_vs_matched_reconstruction.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_report(
    output: Path,
    pairing: pd.DataFrame,
    gap: pd.DataFrame,
    contributions: pd.DataFrame,
    mouse_equal: pd.DataFrame,
    gap_compare: pd.DataFrame,
) -> Path:
    eligible_four = pairing.loc[pairing["pair_four_primary_eligible"]]
    eligible_common = pairing.loc[pairing["pair_common_primary_eligible"]]
    lines = [
        "# ED2 same-animal Naive reconstruction rerun",
        "",
        "## Estimand",
        "",
        "For every accepted Return session, the counterfactual simulation uses the 30-min Explore policy vector from the same mouse's original Naive Day-1 session. Multiple Return sessions from one mouse reuse that baseline. The observed denominator is the mean paired Return-minus-Naive efficiency difference over the same Return-session rows.",
        "",
        "This is a paired longitudinal reconstruction diagnostic. It is not variance explained, mediation, prospective prediction or a causal proportion.",
        "",
        "## Support",
        "",
        f"- Accepted Return roster: {len(pairing)} sessions from {pairing['animal_id'].nunique()} mice.",
        f"- Four-policy matched support: {len(eligible_four)} Return sessions from {eligible_four['animal_id'].nunique()} mice and {eligible_four['matched_naive_session'].nunique()} Naive baselines.",
        f"- Common four-policy plus depth matched support: {len(eligible_common)} Return sessions from {eligible_common['animal_id'].nunique()} mice and {eligible_common['matched_naive_session'].nunique()} Naive baselines.",
        "",
        "## Primary session-weighted paired reconstruction",
        "",
        markdown_table(gap),
        "",
        "## One-policy substitutions",
        "",
        markdown_table(contributions),
        "",
        "## Animal-equal sensitivity",
        "",
        markdown_table(mouse_equal),
        "",
        "## Superseded pooled-donor comparison",
        "",
        markdown_table(gap_compare),
        "",
        "## Interpretation",
        "",
        "The pooled Naive-mean result and its percentages are superseded for the longitudinal question. Figure panels d and h are affected because they replace Return-session policies. Panels b, c, e, f, g and i use observed or own-session policy estimates and are not numerically changed by the donor correction.",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(output: Path, args: argparse.Namespace, outputs: list[Path]) -> Path:
    input_paths = [
        RETURN_POLICY_PATH,
        ACCEPTED_COUNTERFACTUAL_PATH,
        ACCEPTED_PANEL_D_PATH,
        ACCEPTED_PANEL_H_PATH,
        FULL_POLICY_PATH,
    ]
    manifest = {
        "created": "2026-08-20",
        "status": "same_animal_naive_reconstruction_rerun",
        "script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "figure_contract": {
            "core_question": (
                "how much of each mouse's paired Naive-to-Return efficiency improvement "
                "is reconstructed by its paired policy change"
            ),
            "window": "first 30 minutes cumulative all-mode TimeInMaze",
            "policy_events": "Explore-mode decisions only",
            "outcome": "whole-session half-cover efficiency = 32/N32",
            "point_weighting": "Return-session weighted matched pairs",
            "uncertainty": "animal-cluster bootstrap retaining all pairs per sampled mouse",
        },
        "parameters": {
            "n_walks_per_matched_condition": int(args.n_walks),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
        },
        "matched_naive_session": MATCHED_NAIVE_SESSION,
        "inputs": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in input_paths},
        "outputs": {
            str(path.relative_to(output)): sha256(path)
            for path in outputs
            if path.exists() and path.is_file()
        },
    }
    path = output / "RUN_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    source = output / "source_tables"
    plot_ready = output / "plot_ready"
    source.mkdir(parents=True, exist_ok=True)
    plot_ready.mkdir(parents=True, exist_ok=True)

    return_input, _ = load_accepted_return_inputs()
    baseline_policy, baseline_scalar, skipped = extract_matched_naive_inputs()
    parity = build_recomputation_parity(baseline_policy, baseline_scalar)
    pairing = build_pairing_ledger(return_input, baseline_policy, baseline_scalar)
    simulation = run_matched_simulations(
        pairing, n_walks=args.n_walks, seed=args.seed
    )
    gap, contributions, draws, contribution_draws = summarize_primary(
        simulation, n_bootstrap=args.n_bootstrap, seed=args.seed + 100000
    )
    mouse_equal, mouse_equal_draws = summarize_mouse_equal(
        simulation, n_bootstrap=args.n_bootstrap, seed=args.seed + 200000
    )
    gap_compare, contribution_compare = build_legacy_comparison(gap, contributions)

    tables = {
        "matched_naive_policy_rows.csv": baseline_policy,
        "matched_naive_efficiency_rows.csv": baseline_scalar,
        "matched_naive_trajectory_load_skips.csv": skipped,
        "matched_naive_recomputation_parity.csv": parity,
        "return_to_naive_pairing_and_support.csv": pairing,
        "matched_counterfactual_session_rows.csv": simulation,
        "matched_gap_reconstruction_summary.csv": gap,
        "matched_policy_contribution_summary.csv": contributions,
        "matched_cluster_bootstrap_draws.csv": draws,
        "matched_policy_contribution_bootstrap_draws.csv": contribution_draws,
        "matched_mouse_equal_sensitivity.csv": mouse_equal,
        "matched_mouse_equal_bootstrap_draws.csv": mouse_equal_draws,
        "pooled_vs_matched_gap_comparison.csv": gap_compare,
        "pooled_vs_matched_policy_contribution_comparison.csv": contribution_compare,
    }
    outputs: list[Path] = []
    for name, frame in tables.items():
        path = source / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    panel_d = plot_ready / "panel_d_policy_contributions.csv"
    panel_h = plot_ready / "panel_h_four_vs_depth_gap.csv"
    contributions.to_csv(panel_d, index=False)
    gap.to_csv(panel_h, index=False)
    outputs.extend([panel_d, panel_h])

    audit_paths = render_audit_figures(
        simulation, gap_compare, output / "audit_figures"
    )
    outputs.extend(audit_paths)
    report = write_report(
        output, pairing, gap, contributions, mouse_equal, gap_compare
    )
    outputs.append(report)
    manifest = write_manifest(output, args, outputs)

    return {
        "output": str(output),
        "n_return_sessions": int(len(pairing)),
        "n_return_mice": int(pairing["animal_id"].nunique()),
        "n_four_primary_return_sessions": int(
            pairing["pair_four_primary_eligible"].sum()
        ),
        "n_common_primary_return_sessions": int(
            pairing["pair_common_primary_eligible"].sum()
        ),
        "gap": gap.to_dict(orient="records"),
        "manifest": str(manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
