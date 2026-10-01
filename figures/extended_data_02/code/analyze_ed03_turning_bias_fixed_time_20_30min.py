"""Task-scoped ED3 fixed-time comparison at 20 and 30 maze minutes.

The maintained ED3 first-1/32 bundle and the earlier 10/20-minute candidate
are left untouched.  The primary analysis freezes the maintained
equal-junction estimator for the four conditional branch policies.  A
separate event-pooled estimator is exported as a sensitivity analysis.

Both estimators use only decisions inside Explore-mode clips whose cumulative
in-maze time from session start is at or below the requested cutoff.  The
simulation target remains the maintained whole-session half-coverage
efficiency (32 / visits needed to reach 32 unique endpoints).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_ed03_decision_matched_windows as pooled
import analyze_ed03_decision_matched_windows_extensions as pooled_ext
import analyze_ed03_paper_faithful_four_bias as shared
import analyze_ed03_turning_bias_fixed_time as equal
from ca_analysis.exploration import extract_explore_endnode_sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "ed03_turning_bias_fixed_time_20_30min_explore_2026-08-04"
)
EVENT_CACHE = (
    REPO_ROOT
    / "docs"
    / "ed03_decision_matched_windows_2026-07-16"
    / "source_tables"
    / "decision_events.csv.gz"
)
OLD_EQUAL_POLICY = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed_time_10_20min_2026-07-16"
    / "source_tables"
    / "fixed_time_policy_rows.csv"
)
OLD_EVENT_POLICY = (
    REPO_ROOT
    / "docs"
    / "ed03_decision_matched_windows_2026-07-16"
    / "source_tables"
    / "window_policy_rows.csv"
)

WINDOWS_MINUTES = (20, 30)
WINDOW_SPECS: tuple[dict[str, Any], ...] = tuple(
    {
        "window_id": f"time_{minutes}m",
        "family": "clock",
        "quota": minutes,
        "label": f"{minutes} min",
    }
    for minutes in WINDOWS_MINUTES
)
GROUPS = equal.GROUP_ORDER
GROUP_LABELS = equal.GROUP_LABEL
GROUP_COLORS = equal.GROUP_COLOR
FORMATS = ("png", "svg", "pdf", "tiff")

POLICY_COLUMNS = (
    "Bf",
    "Ba",
    "Lf",
    "Lo",
    "Lo_shallow",
    "Lo_mid",
    "Lo_deep",
)
COUNT_COLUMNS = (
    "n_events",
    "n_Bf",
    "n_Ba",
    "n_Lf",
    "n_Lo",
    "n_Lo_shallow",
    "n_Lo_mid",
    "n_Lo_deep",
)
KNOWN_MISSING_SCALAR_TARGETS = {
    ("837-12day2", "experienced"),
    ("838-12day2", "experienced"),
    ("839-12day2", "experienced"),
}


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
            return str(value)
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
    """Freeze imported maintained helpers to this task's two clock windows."""
    equal.WINDOWS_MINUTES = WINDOWS_MINUTES
    pooled.WINDOWS = WINDOW_SPECS
    pooled.WINDOW_ORDER = [item["window_id"] for item in WINDOW_SPECS]
    pooled.WINDOW_LABEL = {item["window_id"]: item["label"] for item in WINDOW_SPECS}
    pooled.WINDOW_FAMILY = {item["window_id"]: item["family"] for item in WINDOW_SPECS}
    pooled.OUTPUT_DIR = output
    pooled.FIGURE_DIR = output / "figures"
    pooled.TABLE_DIR = output / "source_tables"
    pooled_ext.OUTPUT_DIR = output
    pooled_ext.TABLE_DIR = output / "source_tables"


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def extract_equal_junction_rows(source: Path, *, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_path = source / "fixed_time_policy_rows.csv"
    skipped_path = source / "trajectory_load_skips.csv"
    if policy_path.exists() and not force:
        policy = pd.read_csv(policy_path)
        try:
            skipped = pd.read_csv(skipped_path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            skipped = pd.DataFrame(columns=["nickname", "error"])
    else:
        policy, skipped = equal.extract_policy_rows()
        policy = equal.add_support_flags(policy)
        if skipped.empty and len(skipped.columns) == 0:
            skipped = pd.DataFrame(columns=["nickname", "error"])
        source.mkdir(parents=True, exist_ok=True)
        policy.to_csv(policy_path, index=False)
        skipped.to_csv(skipped_path, index=False)
    expected = set(WINDOWS_MINUTES)
    observed = set(pd.to_numeric(policy["window_minutes"]).astype(int).unique())
    if observed != expected:
        raise ValueError(f"Unexpected equal-junction windows: {sorted(observed)} != {sorted(expected)}")
    if "primary_policy_support" not in policy:
        policy = equal.add_support_flags(policy)
        policy.to_csv(policy_path, index=False)
    if not policy["window_definition"].str.contains("Explore-mode", regex=False).all():
        raise ValueError("Equal-junction policy rows are not explicitly Explore-mode restricted")
    return policy, skipped


def build_event_pooled_policies(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "nickname",
        "group",
        "animal_id",
        "decision_index",
        "maze_seconds",
        "depth_stratum",
        "Bf",
        "Ba",
        "Lf",
        "Lo",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Decision-event cache is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    event_columns = [column for column in events.columns if column not in {"nickname", "group", "animal_id"}]
    for (nickname, group, animal_id), frame in events.groupby(
        ["nickname", "group", "animal_id"], sort=True
    ):
        frame = frame.sort_values(["maze_seconds", "decision_index"])[event_columns]
        for window in WINDOW_SPECS:
            rows.append(
                {
                    "nickname": nickname,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "animal_id": str(animal_id),
                    **pooled.fit_window(frame, window),
                }
            )
    policies = pd.DataFrame(rows)
    for model, _ in pooled.MODELS:
        support = policies.apply(lambda row: pooled._support_reason(row, model), axis=1)
        policies[f"{model}_support"] = [item[0] for item in support]
        policies[f"{model}_support_reason"] = [item[1] for item in support]
    policies["window_definition"] = (
        "Explore-mode decisions at cumulative in-maze time <= cutoff from session start"
    )
    policies["estimator_definition"] = "eligible successes pooled over all decision events"
    return policies


def load_event_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        events = pd.read_csv(path)
    else:
        events, _, _, _ = pooled.load_and_fit()
    events["animal_id"] = events["animal_id"].astype(str)
    events["behavior_mode"] = "Explore"
    if not events["maze_seconds"].ge(0).all():
        raise ValueError("Decision event cache contains negative cumulative maze time")
    return events


def parity_audit(
    current: pd.DataFrame,
    reference_path: Path,
    *,
    reference_filter: pd.Series | None,
    metrics: Iterable[str],
    estimator: str,
) -> pd.DataFrame:
    columns = [column for column in metrics if column in current.columns]
    if not reference_path.exists():
        return pd.DataFrame(
            [{"estimator": estimator, "status": "reference_missing", "max_abs_delta": np.nan}]
        )
    reference = pd.read_csv(reference_path)
    if reference_filter is not None:
        reference = reference.loc[reference_filter(reference)].copy()
    current20 = current.loc[pd.to_numeric(current["window_minutes"], errors="coerce").eq(20)].copy()
    if "window_minutes" not in reference and "window_id" in reference:
        reference["window_minutes"] = reference["window_id"].map({"time_20m": 20})
    reference20 = reference.loc[pd.to_numeric(reference["window_minutes"], errors="coerce").eq(20)].copy()
    merged = current20.merge(
        reference20,
        on=["nickname", "group"],
        how="outer",
        suffixes=("_current", "_reference"),
        indicator=True,
    )
    rows: list[dict[str, Any]] = []
    for column in columns:
        current_column = f"{column}_current"
        reference_column = f"{column}_reference"
        if current_column not in merged or reference_column not in merged:
            continue
        delta = pd.to_numeric(merged[current_column], errors="coerce") - pd.to_numeric(
            merged[reference_column], errors="coerce"
        )
        finite = delta[np.isfinite(delta)]
        rows.append(
            {
                "estimator": estimator,
                "metric": column,
                "n_current_rows": int(len(current20)),
                "n_reference_rows": int(len(reference20)),
                "n_both_rows": int(merged["_merge"].eq("both").sum()),
                "n_current_only": int(merged["_merge"].eq("left_only").sum()),
                "n_reference_only": int(merged["_merge"].eq("right_only").sum()),
                "max_abs_delta": float(np.max(np.abs(finite))) if len(finite) else np.nan,
                "status": "compared",
            }
        )
    return pd.DataFrame(rows)


def target_overlap_audit(policy: pd.DataFrame) -> pd.DataFrame:
    policy_sessions = policy[["nickname", "group", "animal_id"]].drop_duplicates()
    scalar = equal.load_scalar_table()[
        ["nickname", "group", "n_explore_endnodes", "half_cover_eff"]
    ].drop_duplicates()
    audit = policy_sessions.merge(
        scalar,
        on=["nickname", "group"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    audit["included_in_simulation"] = audit["_merge"].eq("both")
    audit["exclusion_reason"] = np.select(
        [audit["_merge"].eq("left_only"), audit["_merge"].eq("right_only")],
        ["missing_whole_session_efficiency_target", "missing_loaded_policy_trajectory"],
        default="included",
    )
    audit = audit.rename(columns={"_merge": "overlap_status"})
    missing_targets = set(
        audit.loc[
            audit["overlap_status"].eq("left_only"), ["nickname", "group"]
        ].itertuples(index=False, name=None)
    )
    unexpected_right = audit.loc[audit["overlap_status"].eq("right_only")]
    if missing_targets != KNOWN_MISSING_SCALAR_TARGETS or not unexpected_right.empty:
        raise ValueError(
            "Unexpected policy/whole-session target alignment: "
            f"missing_targets={sorted(missing_targets)}; right_only={unexpected_right['nickname'].tolist()}"
        )
    audit["expected_known_gap"] = audit.apply(
        lambda row: (row["nickname"], row["group"]) in KNOWN_MISSING_SCALAR_TARGETS,
        axis=1,
    )
    return audit.sort_values(["group", "nickname"]).reset_index(drop=True)


def outcome_overlap_audit(
    policy: pd.DataFrame, decision_events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    membership = policy[["nickname", "group", "animal_id"]].drop_duplicates()
    pairs = list(membership[["nickname", "group"]].itertuples(index=False, name=None))
    equal.legacy.bootstrap_environment(REPO_ROOT)
    trajectories, valid_pairs, skipped = equal.legacy._load_trajectories(pairs)
    scalar = equal.load_scalar_table().rename(
        columns={
            "n_explore_endnodes": "scalar_n_explore_terminal_observations",
            "half_cover_eff": "scalar_half_cover_eff",
        }
    )
    event_groups = {
        (str(nickname), str(group)): frame
        for (nickname, group), frame in decision_events.groupby(["nickname", "group"], sort=False)
    }
    rows: list[dict[str, Any]] = []
    for nickname, group in valid_pairs:
        tf = trajectories[nickname]
        nodes, frames = extract_explore_endnode_sequence(
            tf, equal.NewMaze(6), return_frames=True
        )
        nodes = np.asarray(nodes, dtype=int)
        frames = np.asarray(frames, dtype=float)
        if len(frames):
            order = np.argsort(frames, kind="mergesort")
            nodes = nodes[order]
            frames = frames[order]
            maze_seconds = equal._frame_to_maze_seconds(
                frames, tf, fps=equal.FPS
            )
        else:
            maze_seconds = np.asarray([], dtype=float)
        seen: set[int] = set()
        n32_index: int | None = None
        for index, node in enumerate(nodes, start=1):
            seen.add(int(node))
            if len(seen) >= 32:
                n32_index = int(index)
                break
        half_seconds = (
            float(maze_seconds[n32_index - 1]) if n32_index is not None else np.nan
        )
        reconstructed = 32.0 / n32_index if n32_index is not None else 0.0
        event_frame = event_groups.get((str(nickname), str(group)), pd.DataFrame())
        junction_decisions_to32 = (
            int(event_frame["maze_seconds"].le(half_seconds).sum())
            if np.isfinite(half_seconds) and not event_frame.empty
            else np.nan
        )
        row: dict[str, Any] = {
            "nickname": nickname,
            "group": group,
            "group_label": GROUP_LABELS[group],
            "animal_id": equal.animal_id(nickname),
            "n_explore_terminal_observations": int(len(nodes)),
            "n_unique_terminal_nodes_whole_session": int(len(set(nodes.tolist()))),
            "reached_32_unique_terminal_nodes": bool(n32_index is not None),
            "n32_terminal_observation_index": n32_index,
            "n_valid_junction_decisions_by_n32": junction_decisions_to32,
            "half_cover_maze_seconds": half_seconds,
            "half_cover_maze_minutes": half_seconds / 60.0 if np.isfinite(half_seconds) else np.nan,
            "reconstructed_half_cover_eff": reconstructed,
        }
        for window in WINDOWS_MINUTES:
            within = nodes[maze_seconds <= window * 60.0]
            row[f"n_terminal_observations_by_{window}m"] = int(len(within))
            row[f"n_unique_terminal_nodes_by_{window}m"] = int(len(set(within.tolist())))
            row[f"already_at_32_unique_by_{window}m"] = bool(len(set(within.tolist())) >= 32)
        rows.append(row)
    session = pd.DataFrame(rows).merge(
        scalar,
        on=["nickname", "group"],
        how="left",
        validate="one_to_one",
    )
    session["absolute_efficiency_parity_delta"] = (
        session["reconstructed_half_cover_eff"] - session["scalar_half_cover_eff"]
    ).abs()
    finite = session["absolute_efficiency_parity_delta"].dropna()
    if len(finite) and float(finite.max()) > 1e-12:
        bad = session.loc[session["absolute_efficiency_parity_delta"].gt(1e-12)]
        raise ValueError(
            "Outcome-overlap reconstruction failed scalar parity: "
            f"{bad[['nickname', 'absolute_efficiency_parity_delta']].to_dict(orient='records')}"
        )
    summary_rows: list[dict[str, Any]] = []
    for window in WINDOWS_MINUTES:
        for group in GROUPS:
            sub = session.loc[session["group"].eq(group)]
            column = f"already_at_32_unique_by_{window}m"
            summary_rows.append(
                {
                    "window_minutes": int(window),
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "n_sessions": int(len(sub)),
                    "n_sessions_already_at_32_unique": int(sub[column].sum()),
                    "fraction_sessions_already_at_32_unique": float(sub[column].mean()),
                    "median_unique_terminal_nodes_by_window": float(
                        sub[f"n_unique_terminal_nodes_by_{window}m"].median()
                    ),
                    "median_n32_maze_minutes_among_reached": float(
                        sub["half_cover_maze_minutes"].dropna().median()
                    ),
                    "predictor_outcome_temporality": "partly_contemporaneous_if_count_above_zero",
                }
            )
    skipped_frame = pd.DataFrame(skipped)
    if skipped_frame.empty:
        skipped_frame = pd.DataFrame(columns=["nickname", "group", "reason"])
    return session, pd.DataFrame(summary_rows), skipped_frame


def summarize_outcome_overlap_sessions(session: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS_MINUTES:
        for group in [item for item in GROUPS if item in set(session["group"])]:
            sub = session.loc[session["group"].eq(group)]
            column = f"already_at_32_unique_by_{window}m"
            rows.append(
                {
                    "window_minutes": int(window),
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "n_sessions": int(len(sub)),
                    "n_sessions_already_at_32_unique": int(sub[column].sum()),
                    "fraction_sessions_already_at_32_unique": float(sub[column].mean()),
                    "median_unique_terminal_nodes_by_window": float(
                        sub[f"n_unique_terminal_nodes_by_{window}m"].median()
                    ),
                    "median_n32_maze_minutes_among_reached": float(
                        sub["half_cover_maze_minutes"].dropna().median()
                    ),
                    "predictor_outcome_temporality": "partly_contemporaneous_if_count_above_zero",
                }
            )
    return pd.DataFrame(rows)


def _resample_union_of_mice(
    naive: pd.DataFrame,
    returned: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    naive = naive.copy()
    returned = returned.copy()
    naive["animal_id"] = naive["animal_id"].astype(str)
    returned["animal_id"] = returned["animal_id"].astype(str)
    naive_ids = set(naive["animal_id"])
    return_ids = set(returned["animal_id"])
    union = np.asarray(sorted(naive_ids | return_ids), dtype=object)
    sampled = rng.choice(union, size=len(union), replace=True)

    def collect(frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for replicate, animal in enumerate(sampled):
            part = frame.loc[frame["animal_id"].eq(str(animal))].copy()
            if part.empty:
                continue
            part["bootstrap_cluster_replicate"] = int(replicate)
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()

    metadata = {
        "n_naive_mice": len(naive_ids),
        "n_return_mice": len(return_ids),
        "n_overlapping_mice": len(naive_ids & return_ids),
        "n_union_mice": len(naive_ids | return_ids),
        "overlapping_animal_ids": ";".join(sorted(naive_ids & return_ids)),
    }
    return collect(naive), collect(returned), metadata


def synchronized_equal_counterfactual_summary(
    simulation: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for window in WINDOWS_MINUTES:
        frame = simulation.loc[
            simulation["window_minutes"].eq(window) & simulation["simulation_depth_support"]
        ].copy()
        naive = frame.loc[frame["group"].eq("naive")].copy()
        returned = frame.loc[frame["group"].eq("LTM")].copy()

        def compute(n_frame: pd.DataFrame, r_frame: pd.DataFrame) -> dict[str, float]:
            if n_frame.empty or r_frame.empty:
                return {
                    "delta_real": np.nan,
                    "delta_four": np.nan,
                    "delta_depth": np.nan,
                    "fraction_four": np.nan,
                    "fraction_depth": np.nan,
                    **{f"contribution_{name}": np.nan for name in ("Bf", "Ba", "Lf", "Lo")},
                }
            delta_real = float(
                r_frame["real_half_cover_eff"].mean()
                - n_frame["real_half_cover_eff"].mean()
            )
            delta_four = float(
                (
                    r_frame["four_own_half_cover_eff"]
                    - r_frame["four_naive_donor_half_cover_eff"]
                ).mean()
            )
            delta_depth = float(
                (
                    r_frame["depth_own_half_cover_eff"]
                    - r_frame["depth_naive_donor_half_cover_eff"]
                ).mean()
            )
            values = {
                "delta_real": delta_real,
                "delta_four": delta_four,
                "delta_depth": delta_depth,
                "fraction_four": delta_four / delta_real if abs(delta_real) > 1e-12 else np.nan,
                "fraction_depth": delta_depth / delta_real if abs(delta_real) > 1e-12 else np.nan,
            }
            for parameter in ("Bf", "Ba", "Lf", "Lo"):
                values[f"contribution_{parameter}"] = float(
                    (
                        r_frame["four_own_half_cover_eff"]
                        - r_frame[f"hybrid_naive_{parameter}_half_cover_eff"]
                    ).mean()
                )
            return values

        point = compute(naive, returned)
        rng = np.random.default_rng(seed + 1000 * window)
        draws: list[dict[str, float]] = []
        metadata: dict[str, Any] | None = None
        for draw in range(n_bootstrap):
            sampled_naive, sampled_return, metadata = _resample_union_of_mice(
                naive, returned, rng
            )
            values = compute(sampled_naive, sampled_return)
            draws.append(values)
            draw_rows.append(
                {"window_minutes": int(window), "bootstrap_draw": int(draw), **values}
            )
        if metadata is None:
            _, _, metadata = _resample_union_of_mice(naive, returned, rng)
        for model, delta_key, fraction_key in (
            ("four_bias", "delta_four", "fraction_four"),
            ("depth_stratified", "delta_depth", "fraction_depth"),
        ):
            draw_frame = pd.DataFrame(draws)
            gap_rows.append(
                {
                    "window_minutes": int(window),
                    "model": model,
                    "n_naive_rows": int(len(naive)),
                    "n_return_rows": int(len(returned)),
                    **{key: metadata[key] for key in ("n_naive_mice", "n_return_mice")},
                    "delta_real": point["delta_real"],
                    "delta_simulated": point[delta_key],
                    "fraction_recapitulated": point[fraction_key],
                    "fraction_ci_low": float(np.nanpercentile(draw_frame[fraction_key], 2.5)),
                    "fraction_ci_high": float(np.nanpercentile(draw_frame[fraction_key], 97.5)),
                    "point_weighting": "session-weighted within group",
                    "uncertainty": "synchronized union-of-mice cluster bootstrap; fixed full-sample naive donor means; Monte Carlo simulation noise not propagated",
                }
            )
            overlap_rows.append(
                {
                    "window_minutes": int(window),
                    "estimator_family": "equal_junction",
                    "model": model,
                    **metadata,
                }
            )
        draw_frame = pd.DataFrame(draws)
        for parameter in ("Bf", "Ba", "Lf", "Lo"):
            key = f"contribution_{parameter}"
            contribution_rows.append(
                {
                    "window_minutes": int(window),
                    "parameter": parameter,
                    "n_return_rows": int(len(returned)),
                    "n_return_mice": int(returned["animal_id"].astype(str).nunique()),
                    "delta_efficiency_mean": point[key],
                    "delta_efficiency_ci_low": float(np.nanpercentile(draw_frame[key], 2.5)),
                    "delta_efficiency_ci_high": float(np.nanpercentile(draw_frame[key], 97.5)),
                    "contrast": "Return own policy minus Return with parameter replaced by fixed Naive common-support mean",
                    "point_weighting": "session-weighted Return rows",
                    "uncertainty": "synchronized union-of-mice cluster bootstrap; fixed donor means; Monte Carlo simulation noise not propagated",
                }
            )
    return (
        pd.DataFrame(gap_rows),
        pd.DataFrame(contribution_rows),
        pd.DataFrame(draw_rows),
        pd.DataFrame(overlap_rows),
    )


def synchronized_event_counterfactual_summary(
    counterfactuals: pd.DataFrame,
    results: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for window_index, window in enumerate(WINDOWS_MINUTES):
        window_id = f"time_{window}m"
        for model_index, (model, _) in enumerate(pooled.MODELS):
            naive = results.loc[
                results["window_id"].eq(window_id)
                & results["estimator"].eq(model)
                & results["group"].eq("naive")
                & results["simulation_supported"]
            ].copy()
            returned = counterfactuals.loc[
                counterfactuals["window_id"].eq(window_id)
                & counterfactuals["estimator"].eq(model)
                & counterfactuals["counterfactual_supported"]
            ].copy()

            def compute(n_frame: pd.DataFrame, r_frame: pd.DataFrame) -> dict[str, float]:
                if n_frame.empty or r_frame.empty:
                    return {
                        "delta_real": np.nan,
                        "delta_simulated": np.nan,
                        "fraction_recapitulated": np.nan,
                    }
                delta_real = float(
                    r_frame["observed_half_cover_eff"].mean()
                    - n_frame["observed_half_cover_eff"].mean()
                )
                delta_simulated = float(
                    (
                        r_frame["own_simulated_efficiency"]
                        - r_frame["naive_donor_simulated_efficiency"]
                    ).mean()
                )
                values = {
                    "delta_real": delta_real,
                    "delta_simulated": delta_simulated,
                    "fraction_recapitulated": (
                        delta_simulated / delta_real if abs(delta_real) > 1e-12 else np.nan
                    ),
                }
                if model == "four_bias_event_pooled":
                    for parameter in ("Bf", "Ba", "Lf", "Lo"):
                        values[f"contribution_{parameter}"] = float(
                            (
                                r_frame["own_simulated_efficiency"]
                                - r_frame[f"hybrid_naive_{parameter}_simulated_efficiency"]
                            ).mean()
                        )
                return values

            point = compute(naive, returned)
            rng = np.random.default_rng(seed + 1000 * window_index + 100 * model_index)
            draws: list[dict[str, float]] = []
            metadata: dict[str, Any] | None = None
            for draw in range(n_bootstrap):
                sampled_naive, sampled_return, metadata = _resample_union_of_mice(
                    naive, returned, rng
                )
                values = compute(sampled_naive, sampled_return)
                draws.append(values)
                draw_rows.append(
                    {
                        "window_id": window_id,
                        "window_minutes": int(window),
                        "estimator": model,
                        "bootstrap_draw": int(draw),
                        **values,
                    }
                )
            if metadata is None:
                _, _, metadata = _resample_union_of_mice(naive, returned, rng)
            draw_frame = pd.DataFrame(draws)
            gap_rows.append(
                {
                    "window_id": window_id,
                    "window_label": f"{window} min",
                    "window_family": "clock",
                    "window_minutes": int(window),
                    "estimator": model,
                    "n_naive_rows": int(len(naive)),
                    "n_return_rows": int(len(returned)),
                    **{key: metadata[key] for key in ("n_naive_mice", "n_return_mice")},
                    **point,
                    "fraction_ci_low": float(
                        np.nanpercentile(draw_frame["fraction_recapitulated"], 2.5)
                    ),
                    "fraction_ci_high": float(
                        np.nanpercentile(draw_frame["fraction_recapitulated"], 97.5)
                    ),
                    "point_weighting": "session-weighted within group",
                    "uncertainty": "synchronized union-of-mice cluster bootstrap; fixed full-sample naive donor means; Monte Carlo simulation noise not propagated",
                }
            )
            overlap_rows.append(
                {
                    "window_minutes": int(window),
                    "estimator_family": "event_pooled",
                    "model": model,
                    **metadata,
                }
            )
            if model == "four_bias_event_pooled":
                for parameter in ("Bf", "Ba", "Lf", "Lo"):
                    key = f"contribution_{parameter}"
                    contribution_rows.append(
                        {
                            "window_id": window_id,
                            "window_label": f"{window} min",
                            "window_family": "clock",
                            "window_minutes": int(window),
                            "parameter": parameter,
                            "n_return_rows": int(len(returned)),
                            "n_return_mice": int(returned["animal_id"].astype(str).nunique()),
                            "delta_efficiency_mean": point[key],
                            "delta_efficiency_ci_low": float(
                                np.nanpercentile(draw_frame[key], 2.5)
                            ),
                            "delta_efficiency_ci_high": float(
                                np.nanpercentile(draw_frame[key], 97.5)
                            ),
                            "contrast": "Return own policy minus Return with one parameter replaced by fixed Naive donor mean",
                            "point_weighting": "session-weighted Return rows",
                            "uncertainty": "synchronized union-of-mice cluster bootstrap; fixed donor means; Monte Carlo simulation noise not propagated",
                        }
                    )
    return (
        pd.DataFrame(gap_rows),
        pd.DataFrame(contribution_rows),
        pd.DataFrame(draw_rows),
        pd.DataFrame(overlap_rows),
    )


def build_common_support_ledger(
    equal_policy: pd.DataFrame, event_policy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["nickname", "group", "animal_id"]
    equal_flags = equal_policy[
        [*keys, "window_minutes", "primary_policy_support", "primary_depth_support"]
    ].copy()
    event_flags = event_policy[
        [
            *keys,
            "quota",
            "four_bias_event_pooled_support",
            "depth_stratified_event_pooled_support",
        ]
    ].copy()
    equal_flags["animal_id"] = equal_flags["animal_id"].astype(str)
    event_flags["animal_id"] = event_flags["animal_id"].astype(str)
    equal_wide = equal_flags.pivot(index=keys, columns="window_minutes")
    equal_wide.columns = [f"equal_{metric}_{int(window)}m" for metric, window in equal_wide.columns]
    event_wide = event_flags.pivot(index=keys, columns="quota")
    event_wide.columns = [f"event_{metric}_{int(window)}m" for metric, window in event_wide.columns]
    ledger = equal_wide.reset_index().merge(
        event_wide.reset_index(), on=keys, how="left", validate="one_to_one"
    )
    equal_required = [
        f"equal_{metric}_{window}m"
        for metric in ("primary_policy_support", "primary_depth_support")
        for window in WINDOWS_MINUTES
    ]
    event_required = [
        f"event_{metric}_{window}m"
        for metric in (
            "four_bias_event_pooled_support",
            "depth_stratified_event_pooled_support",
        )
        for window in WINDOWS_MINUTES
    ]
    ledger["in_primary_groups"] = ledger["group"].isin(["naive", "LTM"])
    ledger["equal_common_support_pass"] = ledger[equal_required].fillna(False).all(axis=1)
    ledger["event_common_support_pass"] = ledger[event_required].fillna(False).all(axis=1)
    ledger["included_fixed_common_cohort"] = (
        ledger["in_primary_groups"] & ledger["equal_common_support_pass"]
    )
    ledger["exclusion_reason"] = np.select(
        [
            ~ledger["in_primary_groups"],
            ledger["in_primary_groups"] & ~ledger["equal_common_support_pass"],
            ledger["included_fixed_common_cohort"] & ~ledger["event_common_support_pass"],
        ],
        [
            "not_Naive_or_Return_primary_group",
            "failed_equal_junction_four_or_strict_depth_gate_in_at_least_one_window",
            "equal_common_pass_but_event_pooled_common_gate_failed",
        ],
        default="included",
    )
    included = ledger.loc[ledger["included_fixed_common_cohort"], keys].copy()
    counts = included.groupby("group")["nickname"].nunique().to_dict()
    if counts != {"LTM": 58, "naive": 19}:
        raise ValueError(f"Unexpected fixed common cohort counts: {counts}")
    if not ledger.loc[
        ledger["included_fixed_common_cohort"], "event_common_support_pass"
    ].all():
        failed = ledger.loc[
            ledger["included_fixed_common_cohort"] & ~ledger["event_common_support_pass"],
            keys,
        ]
        raise ValueError(
            "Fixed equal-junction common cohort does not pass event-pooled gates: "
            f"{failed.to_dict(orient='records')}"
        )
    return ledger.sort_values(["group", "nickname"]).reset_index(drop=True), included


def filter_policy_to_membership(
    policy: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    keys = ["nickname", "group"]
    return policy.merge(
        membership[keys].drop_duplicates(),
        on=keys,
        how="inner",
        validate="many_to_one",
    )


def run_equal_common_cohort(
    policy: pd.DataFrame,
    source: Path,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    simulation = equal.run_simulations(policy, n_walks=n_walks, seed=seed)
    calibration = equal.calibration_summary(simulation)
    calibration = calibration.loc[calibration["group"].isin(["naive", "LTM"])].copy()
    policy_stats = equal.policy_cluster_stats(policy)
    depth_stats = equal.depth_cluster_stats(policy)
    policy_stats = policy_stats.loc[
        policy_stats["contrast_id"].eq("return_vs_naive")
    ].reset_index(drop=True)
    depth_stats = depth_stats.loc[
        depth_stats["contrast_id"].eq("return_vs_naive")
    ].reset_index(drop=True)
    group_stats = equal.group_summary(policy)
    group_stats = group_stats.loc[group_stats["group"].isin(["naive", "LTM"])].reset_index(
        drop=True
    )
    gap, contributions, draws, overlap = synchronized_equal_counterfactual_summary(
        simulation, n_bootstrap=n_bootstrap, seed=seed + 50000
    )
    outputs = {
        "policy": policy,
        "simulation": simulation,
        "calibration": calibration,
        "policy_stats": policy_stats,
        "depth_stats": depth_stats,
        "group_stats": group_stats,
        "gap": gap,
        "contributions": contributions,
        "draws": draws,
        "overlap": overlap,
    }
    filenames = {
        "policy": "primary_common_cohort_equal_junction_policy_rows.csv",
        "simulation": "primary_common_cohort_equal_junction_simulation_session_rows.csv",
        "calibration": "primary_common_cohort_equal_junction_calibration_summary.csv",
        "policy_stats": "primary_common_cohort_equal_junction_policy_cluster_stats.csv",
        "depth_stats": "primary_common_cohort_equal_junction_depth_cluster_stats.csv",
        "group_stats": "primary_common_cohort_equal_junction_policy_group_summary.csv",
        "gap": "primary_common_cohort_equal_junction_gap_recapitulation_summary.csv",
        "contributions": "primary_common_cohort_equal_junction_policy_contribution_summary.csv",
        "draws": "primary_common_cohort_equal_junction_synchronized_bootstrap_draws.csv",
        "overlap": "primary_common_cohort_equal_junction_mouse_overlap_audit.csv",
    }
    for key, filename in filenames.items():
        outputs[key].to_csv(source / filename, index=False)
    return outputs


def run_event_common_cohort(
    policies: pd.DataFrame,
    source: Path,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    joined, alignment = pooled.align_targets(policies)
    alignment = alignment.loc[~alignment["overlap_status"].eq("right_only")].reset_index(
        drop=True
    )
    if len(alignment) != 77 or not alignment["overlap_status"].eq("both").all():
        raise ValueError(
            "Unexpected fixed common-cohort event target alignment: "
            f"n={len(alignment)}, statuses={alignment['overlap_status'].value_counts().to_dict()}"
        )
    results, exclusions = pooled.simulate_all(joined, n_walks=n_walks, seed=seed)
    results["window_minutes"] = pd.to_numeric(results["quota"], errors="raise").astype(int)
    calibration, calibration_draws = shared.summarize(
        results, n_bootstrap=n_bootstrap, seed=seed
    )
    lookup = results[
        ["window_minutes", "window_id", "window_label", "window_family", "quota", "estimator"]
    ].drop_duplicates()
    calibration = calibration.merge(
        lookup,
        on=["window_minutes", "window_label", "estimator"],
        how="left",
        validate="many_to_one",
    )
    counterfactuals = pooled_ext.run_counterfactuals(
        policies, results, n_walks=n_walks, seed=seed + 100000
    )
    gap, contributions, draws, overlap = synchronized_event_counterfactual_summary(
        counterfactuals, results, n_bootstrap=n_bootstrap, seed=seed + 150000
    )
    outputs = {
        "policies": policies,
        "alignment": alignment,
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
    filenames = {
        "policies": "common_cohort_event_pooled_policy_rows.csv",
        "alignment": "common_cohort_event_pooled_target_overlap_audit.csv",
        "results": "common_cohort_event_pooled_simulation_session_rows.csv",
        "exclusions": "common_cohort_event_pooled_exclusions.csv",
        "calibration": "common_cohort_event_pooled_calibration_summary.csv",
        "calibration_draws": "common_cohort_event_pooled_calibration_bootstrap_draws.csv",
        "counterfactuals": "common_cohort_event_pooled_counterfactual_return_rows.csv",
        "gap": "common_cohort_event_pooled_gap_recapitulation_summary.csv",
        "contributions": "common_cohort_event_pooled_policy_contribution_summary.csv",
        "draws": "common_cohort_event_pooled_synchronized_bootstrap_draws.csv",
        "overlap": "common_cohort_event_pooled_mouse_overlap_audit.csv",
    }
    for key, filename in filenames.items():
        outputs[key].to_csv(source / filename, index=False)
    return outputs


def _animal_equal_mean(frame: pd.DataFrame, values: pd.Series) -> float:
    if frame.empty:
        return np.nan
    cluster_column = (
        "bootstrap_cluster_replicate"
        if "bootstrap_cluster_replicate" in frame.columns
        else "animal_id"
    )
    values = pd.Series(values.to_numpy(dtype=float), index=frame.index)
    return float(values.groupby(frame[cluster_column]).mean().mean())


def mouse_equal_gap_sensitivity(
    model_frames: Iterable[tuple[str, int, str, pd.DataFrame, pd.DataFrame]],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    for model_index, (family, window, model, naive, returned) in enumerate(model_frames):
        if family == "equal_junction":
            naive_observed = "real_half_cover_eff"
            return_observed = "real_half_cover_eff"
            if model == "four_bias":
                own = "four_own_half_cover_eff"
                donor = "four_naive_donor_half_cover_eff"
            else:
                own = "depth_own_half_cover_eff"
                donor = "depth_naive_donor_half_cover_eff"
        else:
            naive_observed = "observed_half_cover_eff"
            return_observed = "observed_half_cover_eff"
            own = "own_simulated_efficiency"
            donor = "naive_donor_simulated_efficiency"

        def compute(n_frame: pd.DataFrame, r_frame: pd.DataFrame) -> dict[str, float]:
            delta_real = _animal_equal_mean(r_frame, r_frame[return_observed]) - _animal_equal_mean(
                n_frame, n_frame[naive_observed]
            )
            delta_simulated = _animal_equal_mean(r_frame, r_frame[own] - r_frame[donor])
            return {
                "delta_real": delta_real,
                "delta_simulated": delta_simulated,
                "fraction_recapitulated": (
                    delta_simulated / delta_real if abs(delta_real) > 1e-12 else np.nan
                ),
            }

        point = compute(naive, returned)
        rng = np.random.default_rng(seed + 1000 * model_index)
        draws: list[dict[str, float]] = []
        for draw in range(n_bootstrap):
            sampled_naive, sampled_return, _ = _resample_union_of_mice(naive, returned, rng)
            values = compute(sampled_naive, sampled_return)
            draws.append(values)
            draw_rows.append(
                {
                    "estimator_family": family,
                    "window_minutes": int(window),
                    "model": model,
                    "bootstrap_draw": int(draw),
                    **values,
                }
            )
        draw_frame = pd.DataFrame(draws)
        rows.append(
            {
                "estimator_family": family,
                "window_minutes": int(window),
                "model": model,
                "n_naive_rows": int(len(naive)),
                "n_return_rows": int(len(returned)),
                "n_naive_mice": int(naive["animal_id"].astype(str).nunique()),
                "n_return_mice": int(returned["animal_id"].astype(str).nunique()),
                **point,
                "fraction_ci_low": float(
                    np.nanpercentile(draw_frame["fraction_recapitulated"], 2.5)
                ),
                "fraction_ci_high": float(
                    np.nanpercentile(draw_frame["fraction_recapitulated"], 97.5)
                ),
                "point_weighting": "animal-equal: session means within mouse, then group means across mice",
                "role": "sensitivity_not_manuscript_aligned_primary",
                "uncertainty": "synchronized union-of-mice cluster bootstrap; fixed donor means; Monte Carlo simulation noise not propagated",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def run_equal_junction(
    policy: pd.DataFrame,
    source: Path,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
    force_simulation: bool,
) -> dict[str, pd.DataFrame]:
    policy_stats = equal.policy_cluster_stats(policy)
    depth_stats = equal.depth_cluster_stats(policy)
    group_stats = equal.group_summary(policy)
    simulation_path = source / "equal_junction_simulation_session_rows.csv"
    if simulation_path.exists() and not force_simulation:
        simulation = pd.read_csv(simulation_path)
    else:
        simulation = equal.run_simulations(policy, n_walks=n_walks, seed=seed)
        simulation.to_csv(simulation_path, index=False)
    calibration = equal.calibration_summary(simulation)
    gap, contributions, draws = equal.summarize_counterfactual(
        simulation, n_bootstrap=n_bootstrap, seed=seed + 50000
    )
    tables = {
        "policy_stats": policy_stats,
        "depth_stats": depth_stats,
        "group_stats": group_stats,
        "simulation": simulation,
        "calibration": calibration,
        "gap": gap,
        "contributions": contributions,
        "draws": draws,
    }
    filenames = {
        "policy_stats": "equal_junction_policy_group_cluster_stats.csv",
        "depth_stats": "equal_junction_depth_group_cluster_stats.csv",
        "group_stats": "equal_junction_policy_group_summary.csv",
        "calibration": "equal_junction_calibration_summary.csv",
        "gap": "equal_junction_gap_recapitulation_summary_legacy_independent_cluster.csv",
        "contributions": "equal_junction_policy_contribution_summary_legacy_independent_cluster.csv",
        "draws": "equal_junction_counterfactual_bootstrap_draws_legacy_independent_cluster.csv",
    }
    for key, filename in filenames.items():
        tables[key].to_csv(source / filename, index=False)
    return tables


def run_event_pooled(
    events: pd.DataFrame,
    source: Path,
    *,
    n_walks: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    policies = build_event_pooled_policies(events)
    joined, alignment = pooled.align_targets(policies)
    results, exclusions = pooled.simulate_all(joined, n_walks=n_walks, seed=seed)
    results["window_minutes"] = pd.to_numeric(results["quota"], errors="raise").astype(int)
    summary, calibration_draws = shared.summarize(
        results, n_bootstrap=n_bootstrap, seed=seed
    )
    lookup = results[
        ["window_minutes", "window_id", "window_label", "window_family", "quota", "estimator"]
    ].drop_duplicates()
    summary = summary.merge(
        lookup,
        on=["window_minutes", "window_label", "estimator"],
        how="left",
        validate="many_to_one",
    )
    evidence = pooled.evidence_summary(results)
    group_summary, contrasts = pooled_ext.group_policy_tables(policies)
    counterfactuals = pooled_ext.run_counterfactuals(
        policies, results, n_walks=n_walks, seed=seed + 100000
    )
    gap, contributions, gap_draws = pooled_ext.summarize_counterfactuals(
        counterfactuals, results, n_bootstrap=n_bootstrap, seed=seed + 150000
    )
    gap["window_minutes"] = gap["window_id"].map(
        {item["window_id"]: item["quota"] for item in WINDOW_SPECS}
    )
    contributions["window_minutes"] = contributions["window_id"].map(
        {item["window_id"]: item["quota"] for item in WINDOW_SPECS}
    )
    selected_events = events.loc[events["maze_seconds"].le(max(WINDOWS_MINUTES) * 60)].copy()
    selected_events["eligible_20min"] = selected_events["maze_seconds"].le(20 * 60)
    selected_events["eligible_30min"] = selected_events["maze_seconds"].le(30 * 60)
    tables = {
        "policies": policies,
        "alignment": alignment,
        "results": results,
        "exclusions": exclusions,
        "calibration": summary,
        "calibration_draws": calibration_draws,
        "evidence": evidence,
        "group_summary": group_summary,
        "contrasts": contrasts,
        "counterfactuals": counterfactuals,
        "gap": gap,
        "contributions": contributions,
        "gap_draws": gap_draws,
        "events": selected_events,
    }
    filenames = {
        "policies": "event_pooled_policy_rows.csv",
        "alignment": "event_pooled_target_overlap_audit.csv",
        "results": "event_pooled_simulation_session_rows.csv",
        "exclusions": "event_pooled_window_model_exclusions.csv",
        "calibration": "event_pooled_calibration_summary.csv",
        "calibration_draws": "event_pooled_calibration_mouse_cluster_bootstrap_draws.csv",
        "evidence": "event_pooled_window_evidence_summary.csv",
        "group_summary": "event_pooled_policy_group_summary.csv",
        "contrasts": "event_pooled_policy_group_cluster_contrasts.csv",
        "counterfactuals": "event_pooled_counterfactual_return_session_rows.csv",
        "gap": "event_pooled_gap_recapitulation_summary_legacy_independent_cluster.csv",
        "contributions": "event_pooled_four_bias_parameter_contributions_legacy_independent_cluster.csv",
        "gap_draws": "event_pooled_gap_bootstrap_draws_legacy_independent_cluster.csv",
    }
    for key, filename in filenames.items():
        tables[key].to_csv(source / filename, index=False)
    selected_events.to_csv(
        source / "event_pooled_explore_decisions_through_30min.csv.gz",
        index=False,
        compression="gzip",
    )
    return tables


def support_summary(
    equal_policy: pd.DataFrame,
    pooled_policy: pd.DataFrame,
    equal_simulation: pd.DataFrame,
    pooled_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for estimator, frame, four_column, depth_column in (
        (
            "equal_junction",
            equal_policy,
            "primary_policy_support",
            "primary_depth_support",
        ),
        (
            "event_pooled",
            pooled_policy,
            "four_bias_event_pooled_support",
            "depth_stratified_event_pooled_support",
        ),
    ):
        window_column = "window_minutes" if estimator == "equal_junction" else "quota"
        for (window, group), sub in frame.groupby([window_column, "group"], sort=True):
            if estimator == "equal_junction":
                sim = equal_simulation.loc[
                    equal_simulation["window_minutes"].eq(int(window))
                    & equal_simulation["group"].eq(group)
                ]
                four_calibration = int(sim["four_own_half_cover_eff"].notna().sum())
                depth_calibration = int(sim["depth_own_half_cover_eff"].notna().sum())
            else:
                sim = pooled_results.loc[
                    pooled_results["window_minutes"].eq(int(window))
                    & pooled_results["group"].eq(group)
                ]
                four_calibration = int(
                    sim.loc[
                        sim["estimator"].eq("four_bias_event_pooled"),
                        "simulated_half_cover_eff",
                    ].notna().sum()
                )
                depth_calibration = int(
                    sim.loc[
                        sim["estimator"].eq("depth_stratified_event_pooled"),
                        "simulated_half_cover_eff",
                    ].notna().sum()
                )
            rows.append(
                {
                    "window_minutes": int(window),
                    "estimator": estimator,
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "n_all_sessions": int(len(sub)),
                    "n_four_policy_supported": int(sub[four_column].sum()),
                    "n_depth_supported": int(sub[depth_column].sum()),
                    "n_four_calibration_rows_with_target": four_calibration,
                    "n_depth_calibration_rows_with_target": depth_calibration,
                    "n_mice": int(sub["animal_id"].astype(str).nunique()),
                }
            )
    return pd.DataFrame(rows)


def exposure_summary(equal_policy: pd.DataFrame, pooled_policy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS_MINUTES:
        for group in GROUPS:
            equal_sub = equal_policy.loc[
                equal_policy["window_minutes"].eq(window) & equal_policy["group"].eq(group)
            ]
            pooled_sub = pooled_policy.loc[
                pooled_policy["quota"].eq(window) & pooled_policy["group"].eq(group)
            ]
            rows.append(
                {
                    "window_minutes": int(window),
                    "group": group,
                    "group_label": GROUP_LABELS[group],
                    "n_sessions": int(len(equal_sub)),
                    "median_explore_minutes_within_window": float(
                        equal_sub["explore_minutes_within_window"].median()
                    ),
                    "q10_explore_minutes_within_window": float(
                        equal_sub["explore_minutes_within_window"].quantile(0.10)
                    ),
                    "q90_explore_minutes_within_window": float(
                        equal_sub["explore_minutes_within_window"].quantile(0.90)
                    ),
                    "median_valid_explore_decisions": float(
                        pooled_sub["n_raw_decisions_used"].median()
                    ),
                    "median_min_four_parameter_denominator": float(
                        pooled_sub[["n_Bf", "n_Ba", "n_Lf", "n_Lo"]].min(axis=1).median()
                    ),
                    "median_min_depth_denominator": float(
                        pooled_sub[["n_Lo_shallow", "n_Lo_mid", "n_Lo_deep"]]
                        .min(axis=1)
                        .median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def donor_policy_refit_audit(
    full_equal: pd.DataFrame,
    common_equal: pd.DataFrame,
    full_event: pd.DataFrame,
    common_event: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = (
        (
            "equal_junction",
            "four_bias",
            full_equal,
            common_equal,
            "primary_policy_support",
            ("Bf", "Ba", "Lf", "Lo"),
        ),
        (
            "equal_junction",
            "depth_stratified",
            full_equal,
            common_equal,
            "simulation_depth_support",
            ("Bf", "Ba", "Lf", "Lo", "Lo_shallow", "Lo_mid", "Lo_deep"),
        ),
        (
            "event_pooled",
            "four_bias_event_pooled",
            full_event,
            common_event,
            "four_bias_event_pooled_support",
            ("Bf", "Ba", "Lf", "Lo"),
        ),
        (
            "event_pooled",
            "depth_stratified_event_pooled",
            full_event,
            common_event,
            "depth_stratified_event_pooled_support",
            ("Bf", "Ba", "Lf", "Lo", "Lo_shallow", "Lo_mid", "Lo_deep"),
        ),
    )
    support_definitions = {
        "primary_policy_support": "at least 30 Explore events and at least 10 supported node/side estimates for each four-policy term",
        "simulation_depth_support": "at least 30 Explore events plus finite depth probabilities; this is the maintained simulation gate, not the strict 10-events-per-depth-stratum gate",
        "four_bias_event_pooled_support": "at least 10 raw eligible events for each pooled four-policy term",
        "depth_stratified_event_pooled_support": "four-policy pooled support plus at least 10 raw eligible events in each depth stratum",
    }
    for family, model, full, common, support, parameters in specs:
        window_column = "window_minutes" if family == "equal_junction" else "quota"
        for window in WINDOWS_MINUTES:
            full_naive = full.loc[
                full[window_column].eq(window)
                & full["group"].eq("naive")
                & full[support]
            ]
            common_naive = common.loc[
                common[window_column].eq(window)
                & common["group"].eq("naive")
                & common[support]
            ]
            for parameter in parameters:
                full_mean = float(full_naive[parameter].mean())
                common_mean = float(common_naive[parameter].mean())
                rows.append(
                    {
                        "window_minutes": int(window),
                        "estimator_family": family,
                        "model": model,
                        "parameter": parameter,
                        "support_gate_column": support,
                        "support_gate_definition": support_definitions[support],
                        "n_full_support_naive_sessions": int(len(full_naive)),
                        "n_fixed_common_naive_sessions": int(len(common_naive)),
                        "full_support_naive_donor_mean": full_mean,
                        "fixed_common_naive_donor_mean": common_mean,
                        "common_minus_full_donor_mean": common_mean - full_mean,
                        "simulation_action": "donor and hybrid walks rerun from fixed-common donor mean",
                    }
                )
    return pd.DataFrame(rows)


def calibration_comparison(
    equal_calibration: pd.DataFrame, pooled_calibration: pd.DataFrame
) -> pd.DataFrame:
    equal_frame = equal_calibration.rename(
        columns={"model": "model_id", "real_mean": "observed_mean"}
    ).copy()
    equal_frame["estimator_family"] = "equal_junction"
    pooled_frame = pooled_calibration.rename(columns={"estimator": "model_id"}).copy()
    pooled_frame["estimator_family"] = "event_pooled"
    columns = [
        "window_minutes",
        "estimator_family",
        "model_id",
        "group",
        "group_label",
        "n_session_rows",
        "n_mice",
        "observed_mean",
        "simulated_mean",
        "mean_error",
        "mae",
        "spearman_rho",
        "slope",
        "combined_gate_pass",
    ]
    for frame in (equal_frame, pooled_frame):
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
    return pd.concat([equal_frame[columns], pooled_frame[columns]], ignore_index=True)


def gap_comparison(equal_gap: pd.DataFrame, pooled_gap: pd.DataFrame) -> pd.DataFrame:
    primary = equal_gap.copy()
    primary["estimator_family"] = "equal_junction_primary"
    primary = primary.rename(
        columns={"model": "model_id", "delta_simulated": "delta_model"}
    )
    sensitivity = pooled_gap.copy()
    sensitivity["estimator_family"] = "event_pooled_sensitivity"
    sensitivity = sensitivity.rename(
        columns={"estimator": "model_id", "delta_simulated": "delta_model"}
    )
    columns = [
        "window_minutes",
        "estimator_family",
        "model_id",
        "n_naive_rows",
        "n_return_rows",
        "n_naive_mice",
        "n_return_mice",
        "delta_real",
        "delta_model",
        "fraction_recapitulated",
        "fraction_ci_low",
        "fraction_ci_high",
    ]
    for frame in (primary, sensitivity):
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
    return pd.concat([primary[columns], sensitivity[columns]], ignore_index=True)


def uncertainty_method_audit(
    equal_legacy_gap: pd.DataFrame,
    equal_corrected_gap: pd.DataFrame,
    event_legacy_gap: pd.DataFrame,
    event_corrected_gap: pd.DataFrame,
    equal_legacy_contribution: pd.DataFrame,
    equal_corrected_contribution: pd.DataFrame,
    event_legacy_contribution: pd.DataFrame,
    event_corrected_contribution: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_gap(
        family: str,
        legacy: pd.DataFrame,
        corrected: pd.DataFrame,
        legacy_model: str,
        corrected_model: str,
    ) -> None:
        left = legacy.rename(columns={legacy_model: "model"})
        right = corrected.rename(columns={corrected_model: "model"})
        merged = left.merge(
            right,
            on=["window_minutes", "model"],
            how="outer",
            suffixes=("_legacy", "_corrected"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "component": "gap_recapitulation",
                    "estimator_family": family,
                    "window_minutes": int(row.window_minutes),
                    "model_or_parameter": row.model,
                    "point_legacy": row.fraction_recapitulated_legacy,
                    "point_corrected": row.fraction_recapitulated_corrected,
                    "ci_low_legacy_independent_cluster": row.fraction_ci_low_legacy,
                    "ci_high_legacy_independent_cluster": row.fraction_ci_high_legacy,
                    "ci_low_synchronized_union_mouse": row.fraction_ci_low_corrected,
                    "ci_high_synchronized_union_mouse": row.fraction_ci_high_corrected,
                }
            )

    def add_contribution(
        family: str, legacy: pd.DataFrame, corrected: pd.DataFrame
    ) -> None:
        merged = legacy.merge(
            corrected,
            on=["window_minutes", "parameter"],
            how="outer",
            suffixes=("_legacy", "_corrected"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "component": "four_bias_parameter_contribution",
                    "estimator_family": family,
                    "window_minutes": int(row.window_minutes),
                    "model_or_parameter": row.parameter,
                    "point_legacy": row.delta_efficiency_mean_legacy,
                    "point_corrected": row.delta_efficiency_mean_corrected,
                    "ci_low_legacy_independent_cluster": row.delta_efficiency_ci_low_legacy,
                    "ci_high_legacy_independent_cluster": row.delta_efficiency_ci_high_legacy,
                    "ci_low_synchronized_union_mouse": row.delta_efficiency_ci_low_corrected,
                    "ci_high_synchronized_union_mouse": row.delta_efficiency_ci_high_corrected,
                }
            )

    add_gap("equal_junction", equal_legacy_gap, equal_corrected_gap, "model", "model")
    add_gap(
        "event_pooled",
        event_legacy_gap,
        event_corrected_gap,
        "estimator",
        "estimator",
    )
    add_contribution(
        "equal_junction", equal_legacy_contribution, equal_corrected_contribution
    )
    add_contribution(
        "event_pooled", event_legacy_contribution, event_corrected_contribution
    )
    audit = pd.DataFrame(rows)
    audit["point_delta_corrected_minus_legacy"] = (
        audit["point_corrected"] - audit["point_legacy"]
    )
    audit["ci_width_legacy"] = (
        audit["ci_high_legacy_independent_cluster"]
        - audit["ci_low_legacy_independent_cluster"]
    )
    audit["ci_width_corrected"] = (
        audit["ci_high_synchronized_union_mouse"]
        - audit["ci_low_synchronized_union_mouse"]
    )
    audit["legacy_defect"] = (
        "Naive and Return mice sampled independently despite cross-group animal overlap"
    )
    audit["corrected_method"] = (
        "sample union animal IDs once per draw and retain every group/session row for sampled mouse"
    )
    return audit


def plot_primary_comparison(
    policy_stats: pd.DataFrame,
    calibration: pd.DataFrame,
    gap: pd.DataFrame,
    support: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    policy_labels = {item[0]: item[1] for item in equal.POLICIES}
    colors = {"Bf": "#4c78a8", "Ba": "#72b7b2", "Lf": "#f58518", "Lo": "#e45756"}
    offsets = np.linspace(-0.24, 0.24, 4)
    x = np.arange(len(WINDOWS_MINUTES), dtype=float)
    for offset, metric in zip(offsets, ("Bf", "Ba", "Lf", "Lo")):
        sub = (
            policy_stats.loc[
                policy_stats["contrast_id"].eq("return_vs_naive")
                & policy_stats["metric"].eq(metric)
            ]
            .set_index("window_minutes")
            .reindex(WINDOWS_MINUTES)
        )
        y = sub["beta_target_minus_baseline"].to_numpy(float)
        axes[0, 0].errorbar(
            x + offset,
            y,
            yerr=np.vstack(
                [y - sub["ci_low"].to_numpy(float), sub["ci_high"].to_numpy(float) - y]
            ),
            fmt="o",
            color=colors[metric],
            capsize=2,
            markersize=3.5,
            label=policy_labels[metric],
        )
    axes[0, 0].axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    axes[0, 0].set_xticks(x, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[0, 0].set_ylabel("Return - Naive probability")
    axes[0, 0].set_title("a  Observed policy contrasts", loc="left", fontweight="bold")
    axes[0, 0].legend(ncol=2, fontsize=5.2)

    primary_support = support.loc[support["estimator"].eq("equal_junction")]
    bar_width = 0.12
    positions = np.arange(len(WINDOWS_MINUTES), dtype=float)
    for group_index, group in enumerate(("naive", "LTM", "experienced")):
        sub = (
            primary_support.loc[primary_support["group"].eq(group)]
            .set_index("window_minutes")
            .reindex(WINDOWS_MINUTES)
        )
        axes[0, 1].bar(
            positions + (group_index - 1) * 2.2 * bar_width,
            sub["n_four_policy_supported"],
            width=bar_width,
            color=GROUP_COLORS[group],
            label=f"{GROUP_LABELS[group]} four",
        )
        axes[0, 1].bar(
            positions + (group_index - 1) * 2.2 * bar_width + bar_width,
            sub["n_four_calibration_rows_with_target"],
            width=bar_width,
            color=GROUP_COLORS[group],
            alpha=0.38,
        )
    axes[0, 1].set_xticks(positions, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[0, 1].set_ylabel("Supported sessions")
    axes[0, 1].set_title(
        "b  Full-cohort policy support (solid) vs target-aligned calibration (pale)",
        loc="left",
        fontweight="bold",
    )
    axes[0, 1].legend(fontsize=5.0, ncol=1)

    model_style = {
        "four_bias": ("#4c78a8", "o", "Four-bias"),
        "depth_stratified": ("#54a24b", "s", "Depth-varying $P_{BS}$"),
    }
    for model, (color, marker, label) in model_style.items():
        sub = gap.loc[gap["model"].eq(model)].set_index("window_minutes").reindex(WINDOWS_MINUTES)
        y = 100 * sub["fraction_recapitulated"].to_numpy(float)
        low = 100 * sub["fraction_ci_low"].to_numpy(float)
        high = 100 * sub["fraction_ci_high"].to_numpy(float)
        axes[1, 0].errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            color=color,
            marker=marker,
            capsize=2,
            label=label,
        )
    axes[1, 0].axhline(100, color="#777777", linestyle="--", linewidth=0.7)
    axes[1, 0].set_xticks(x, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[1, 0].set_ylabel("Gap recapitulated (%)")
    axes[1, 0].set_title("c  Policy-swap diagnostic", loc="left", fontweight="bold")
    axes[1, 0].legend(fontsize=5.2)

    marker = {"naive": "o", "LTM": "s"}
    linestyle = {"four_bias": "-", "depth_stratified": "--"}
    for model, (color, _, label) in model_style.items():
        for group in ("naive", "LTM"):
            sub = calibration.loc[
                calibration["model"].eq(model) & calibration["group"].eq(group)
            ].set_index("window_minutes").reindex(WINDOWS_MINUTES)
            axes[1, 1].plot(
                x,
                sub["spearman_rho"],
                color=color,
                linestyle=linestyle[model],
                marker=marker[group],
                label=f"{label}, {GROUP_LABELS[group]}",
            )
    axes[1, 1].axhline(0.4, color="#b24a4a", linestyle=":", linewidth=0.8)
    axes[1, 1].set_xticks(x, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[1, 1].set_ylabel("Spearman rho")
    axes[1, 1].set_title("d  Same-session calibration", loc="left", fontweight="bold")
    axes[1, 1].legend(fontsize=4.7, ncol=1)
    for axis in axes.ravel():
        axis.grid(axis="y", color="#e3e3e3", linewidth=0.5, zorder=0)
    fig.suptitle("ED3 fixed clock comparison: Explore-only decisions", fontsize=8.0, y=0.995)
    fig.text(
        0.5,
        0.005,
        "Primary estimator: unweighted mean across observed junctions. Intervals: CR1 (a) or mouse-cluster bootstrap (c).",
        ha="center",
        fontsize=5.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    return save_figure(fig, figure_dir / "ed03_equal_junction_20_vs_30min")


def plot_estimator_sensitivity(
    calibration: pd.DataFrame,
    gap: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    estimator_style = {
        "equal_junction": ("#4c78a8", "o", "Equal-junction primary"),
        "event_pooled": ("#f58518", "s", "Event-pooled sensitivity"),
    }
    x = np.arange(len(WINDOWS_MINUTES), dtype=float)
    for estimator, (color, marker_symbol, label) in estimator_style.items():
        for group, ls in (("naive", "-"), ("LTM", "--")):
            sub = calibration.loc[
                calibration["estimator_family"].eq(estimator)
                & calibration["group"].eq(group)
                & calibration["model_id"].astype(str).str.contains("four_bias")
            ].set_index("window_minutes").reindex(WINDOWS_MINUTES)
            axes[0].plot(
                x,
                sub["spearman_rho"],
                color=color,
                marker=marker_symbol,
                linestyle=ls,
                label=f"{label}, {GROUP_LABELS[group]}",
            )
    axes[0].axhline(0.4, color="#b24a4a", linestyle=":", linewidth=0.8)
    axes[0].set_xticks(x, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[0].set_ylabel("Four-bias calibration rho")
    axes[0].set_title("a  Estimator sensitivity", loc="left", fontweight="bold")
    axes[0].legend(fontsize=5.0)

    styles = {
        ("equal_junction_primary", "four_bias"): ("#4c78a8", "o", "Equal four"),
        ("equal_junction_primary", "depth_stratified"): ("#54a24b", "s", "Equal + depth"),
        ("event_pooled_sensitivity", "four_bias_event_pooled"): ("#f58518", "^", "Pooled four"),
        ("event_pooled_sensitivity", "depth_stratified_event_pooled"): ("#e45756", "D", "Pooled + depth"),
    }
    for (family, model), (color, marker_symbol, label) in styles.items():
        sub = gap.loc[
            gap["estimator_family"].eq(family) & gap["model_id"].eq(model)
        ].set_index("window_minutes").reindex(WINDOWS_MINUTES)
        axes[1].plot(
            x,
            100 * sub["fraction_recapitulated"],
            color=color,
            marker=marker_symbol,
            label=label,
        )
    axes[1].axhline(100, color="#777777", linestyle="--", linewidth=0.7)
    axes[1].set_xticks(x, [f"{item} min" for item in WINDOWS_MINUTES])
    axes[1].set_ylabel("Gap recapitulated (%)")
    axes[1].set_title("b  Descriptive policy swaps", loc="left", fontweight="bold")
    axes[1].legend(fontsize=5.0)
    for axis in axes:
        axis.grid(axis="y", color="#e3e3e3", linewidth=0.5)
    fig.suptitle("Estimator choice is a sensitivity analysis, not a selection rule", fontsize=8.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, figure_dir / "ed03_estimator_sensitivity_20_vs_30min")


def plot_manuscript_candidate(
    gap: pd.DataFrame,
    contributions: pd.DataFrame,
    depth_stats: pd.DataFrame,
    simulation: pd.DataFrame,
    calibration: pd.DataFrame,
    outcome_overlap: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    fig = plt.figure(figsize=(7.2, 5.4))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=(0.88, 1.12),
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.935,
        hspace=0.42,
        wspace=0.55,
    )
    ax_gap = fig.add_subplot(grid[0, 0:2])
    ax_contrib = fig.add_subplot(grid[0, 2])
    ax_depth = fig.add_subplot(grid[0, 3])
    ax_naive = fig.add_subplot(grid[1, 0:2])
    ax_return = fig.add_subplot(grid[1, 2:4], sharex=ax_naive, sharey=ax_naive)

    x = np.arange(len(WINDOWS_MINUTES), dtype=float)
    model_styles = {
        "four_bias": ("#4c78a8", "o", "Four policies"),
        "depth_stratified": ("#54a24b", "s", "Depth-varying $P_{BS}$"),
    }
    for model, (color, marker, label) in model_styles.items():
        sub = gap.loc[gap["model"].eq(model)].set_index("window_minutes").reindex(WINDOWS_MINUTES)
        y = 100.0 * sub["fraction_recapitulated"].to_numpy(float)
        lo = 100.0 * sub["fraction_ci_low"].to_numpy(float)
        hi = 100.0 * sub["fraction_ci_high"].to_numpy(float)
        ax_gap.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=color,
            marker=marker,
            capsize=2,
            linewidth=1.1,
            label=label,
        )
    ax_gap.axhline(100, color="#777777", linestyle="--", linewidth=0.7)
    ax_gap.set_xticks(x, ["20 min", "30 min"])
    ax_gap.set_ylabel("Naive-Return gap recapitulated (%)")
    ax_gap.set_title("a  Fixed common cohort", loc="left", fontweight="bold")
    ax_gap.legend(fontsize=5.3)

    parameter_colors = {
        "Bf": "#4c78a8",
        "Ba": "#72b7b2",
        "Lf": "#f58518",
        "Lo": "#e45756",
    }
    parameter_labels = {item[0]: item[1] for item in equal.POLICIES}
    for parameter, color in parameter_colors.items():
        sub = contributions.loc[contributions["parameter"].eq(parameter)].set_index(
            "window_minutes"
        ).reindex(WINDOWS_MINUTES)
        ax_contrib.plot(
            x,
            sub["delta_efficiency_mean"],
            marker="o",
            markersize=3,
            color=color,
            label=parameter_labels[parameter],
        )
    ax_contrib.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax_contrib.set_xticks(x, ["20", "30"])
    ax_contrib.set_xlabel("Maze min")
    ax_contrib.set_ylabel("Efficiency contribution")
    ax_contrib.set_title("b  Policy terms", loc="left", fontweight="bold")
    ax_contrib.legend(fontsize=4.7, ncol=2)

    depth_order = ["Lo_shallow", "Lo_mid", "Lo_deep"]
    depth_labels = ["Shallow", "Middle", "Deep"]
    depth_colors = {20: "#6a9f58", 30: "#2f6b3a"}
    for index, window in enumerate(WINDOWS_MINUTES):
        sub = (
            depth_stats.loc[
                depth_stats["support_policy"].eq("all_depth_strata_n_ge_10")
                & depth_stats["contrast_id"].eq("return_vs_naive")
                & depth_stats["window_minutes"].eq(window)
            ]
            .set_index("metric")
            .reindex(depth_order)
        )
        y = sub["beta_target_minus_baseline"].to_numpy(float)
        ax_depth.errorbar(
            np.arange(3) + (index - 0.5) * 0.12,
            y,
            yerr=np.vstack(
                [y - sub["ci_low"].to_numpy(float), sub["ci_high"].to_numpy(float) - y]
            ),
            fmt="o",
            color=depth_colors[window],
            markersize=3,
            capsize=2,
            label=f"{window} min",
        )
    ax_depth.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax_depth.set_xticks(np.arange(3), depth_labels, rotation=25, ha="right")
    ax_depth.set_ylabel(r"Return - Naive $P_{BS}$")
    ax_depth.set_title("   By tree depth", loc="left", fontweight="bold")
    ax_depth.legend(fontsize=4.8)

    window_style = {20: ("#6a9f58", "o"), 30: ("#2f6b3a", "s")}
    for axis, group in ((ax_naive, "naive"), (ax_return, "LTM")):
        for window in WINDOWS_MINUTES:
            color, marker = window_style[window]
            sub = simulation.loc[
                simulation["window_minutes"].eq(window)
                & simulation["group"].eq(group)
                & simulation["simulation_depth_support"]
            ].dropna(subset=["real_half_cover_eff", "depth_own_half_cover_eff"])
            axis.scatter(
                sub["real_half_cover_eff"],
                sub["depth_own_half_cover_eff"],
                s=17,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                linewidths=0.8,
                label=f"{window} min",
            )
            stat = calibration.loc[
                calibration["window_minutes"].eq(window)
                & calibration["group"].eq(group)
                & calibration["model"].eq("depth_stratified")
            ].iloc[0]
            axis.text(
                0.03,
                0.94 - (0.15 if window == 30 else 0.0),
                f"{window}m: bias={stat.mean_error:+.3f}, rho={stat.spearman_rho:.2f}",
                transform=axis.transAxes,
                va="top",
                fontsize=5.0,
                color=color,
            )
        axis.plot([0, 0.68], [0, 0.68], color="#888888", linestyle="--", linewidth=0.7)
        axis.set_xlim(0, 0.68)
        axis.set_ylim(0, 0.68)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#e5e5e5", linewidth=0.45)
        axis.set_title(
            f"c  {GROUP_LABELS[group]}" if group == "naive" else f"   {GROUP_LABELS[group]}",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Observed whole-session efficiency")
    ax_naive.set_ylabel("Depth-model simulated efficiency")
    ax_return.legend(loc="lower right", fontsize=5.2)

    overlap_text = "; ".join(
        f"{row.group_label} {int(row.window_minutes)}m: {int(row.n_sessions_already_at_32_unique)}/{int(row.n_sessions)}"
        for row in outcome_overlap.loc[outcome_overlap["group"].isin(["naive", "LTM"])].itertuples()
    )
    fig.suptitle(
        "ED3 candidate | Fixed-clock policy swaps reconstruct group-level exploration differences",
        fontsize=8.0,
        y=0.982,
    )
    fig.text(
        0.5,
        0.025,
        "Primary N=19 Naive sessions and 58 Return sessions; session-weighted points; synchronized union-of-mice bootstrap CIs. "
        f"Sessions already at 32 unique endpoints within policy window: {overlap_text}.",
        ha="center",
        va="bottom",
        fontsize=4.8,
        color="#555555",
    )
    return save_figure(fig, figure_dir / "ed03_manuscript_candidate_20_30_common_cohort")


def markdown_table(frame: pd.DataFrame) -> str:
    return equal.markdown_table(frame)


def write_report(
    output: Path,
    support: pd.DataFrame,
    exposure: pd.DataFrame,
    common_exposure: pd.DataFrame,
    calibration: pd.DataFrame,
    gap: pd.DataFrame,
    parity: pd.DataFrame,
    common_ledger: pd.DataFrame,
    target_audit: pd.DataFrame,
    outcome_overlap: pd.DataFrame,
    mouse_equal_gap: pd.DataFrame,
    full_support_gap: pd.DataFrame,
    uncertainty_audit: pd.DataFrame,
    donor_audit: pd.DataFrame,
) -> Path:
    calibration_display = calibration.loc[
        calibration["group"].isin(["naive", "LTM"]),
        [
            "window_minutes",
            "estimator_family",
            "model_id",
            "group_label",
            "n_session_rows",
            "mean_error",
            "mae",
            "spearman_rho",
            "combined_gate_pass",
        ],
    ]
    gap_display = gap[
        [
            "window_minutes",
            "estimator_family",
            "model_id",
            "n_naive_rows",
            "n_return_rows",
            "fraction_recapitulated",
            "fraction_ci_low",
            "fraction_ci_high",
        ]
    ]
    common_counts = (
        common_ledger.loc[common_ledger["included_fixed_common_cohort"]]
        .groupby("group")
        .agg(n_sessions=("nickname", "nunique"), n_mice=("animal_id", "nunique"))
        .reset_index()
    )
    target_summary = (
        target_audit.groupby(["group", "overlap_status"])
        .size()
        .rename("n_sessions")
        .reset_index()
    )
    uncertainty_summary = uncertainty_audit[
        [
            "component",
            "estimator_family",
            "window_minutes",
            "model_or_parameter",
            "point_delta_corrected_minus_legacy",
            "ci_width_legacy",
            "ci_width_corrected",
        ]
    ]
    lines = [
        "# ED3 fixed 20/30-minute Explore-only turning-policy rerun",
        "",
        "## Figure contract",
        "",
        "**Core conclusion.** On one fixed common-support Naive/Return cohort, compare 20- and 30-minute Explore-only policy estimates without selecting a window by effect size; treat policy swaps as group-level reconstruction diagnostics, not causal explanation.",
        "",
        "- Archetype: quantitative grid.",
        "- Primary evidence: equal-junction policy swaps, depth extension, strict fixed-cohort support, and own-policy calibration.",
        "- Sensitivity evidence: event-pooled versions from the same Explore decisions.",
        "- Export: Python/matplotlib SVG, PDF, 600-dpi PNG and TIFF; editable SVG/PDF text.",
        "- Reviewer risks: post hoc window choice, unequal Explore occupancy inside equal clock windows, estimator dependence, sparse depth support, cross-group repeated mice, and partial contemporaneity between the policy window and whole-session outcome.",
        "",
        "## Frozen definitions",
        "",
        "- Clock: cumulative **in-maze** time from session start, not wall-clock duration and not cumulative Explore time.",
        "- Behavior mode: policy events come only from clips classified as **Explore** (`SplitModeClips` mode index 2). Other modes may consume clock time but never contribute a policy event.",
        "- `Bf` (`P_SF`): after arriving from the parent/stem, continue into a child branch rather than reverse toward the parent.",
        "- `Ba` (`P_SA`): conditional on continuing from the parent/stem into a child, choose the alternating child side under the maintained maze-orientation convention.",
        "- `Lf` (`P_BF`): after arriving from a child, exit that sampled child subtree rather than reverse into the same child.",
        "- `Lo` (`P_BS`): conditional on exiting the sampled child subtree, continue toward the parent/stem rather than enter the sibling child.",
        "- Primary four-policy estimator: calculate each conditional probability separately at every observed non-root junction/side, then take the unweighted mean across those estimates. This preserves the maintained ED3 convention.",
        "- Equal-junction `n_Bf`, `n_Ba`, `n_Lf`, and `n_Lo` are numbers of supported node/side estimates, **not raw decision-event denominators**. The strict gate is at least 10 supported estimates for every policy.",
        "- Depth extension: pool eligible stemward-exit events within shallow (depths 1-2), middle (3-4), and deep (5) strata. `n_Lo_shallow/mid/deep` are raw eligible event counts; the strict gate is at least 10 in each stratum.",
        "- Sensitivity estimator: pool eligible successes and raw event denominators across all Explore decision events; all event-pooled `n_*` fields are raw eligible-event denominators.",
        "- Outcome: maintained whole-session half-coverage efficiency, `32 / N32`, where `N32` is the number of endpoint visits required to encounter 32 unique endpoints.",
        "",
        "## Primary fixed common-support cohort",
        "",
        "The conclusion-bearing cohort is the intersection across both 20- and 30-minute windows and both the four-policy and strict depth gates. It is frozen before comparing windows or models. Naive donor policies were re-estimated from these same 19 Naive sessions and every donor/hybrid simulation was rerun; filtering simulations generated from the broader donor pool was not used.",
        "",
        markdown_table(common_counts),
        "",
        "This retains 19/23 Naive sessions and 58/58 Return sessions. Four Naive sessions fail at least one cross-window four-policy or strict-depth gate; attrition is therefore asymmetric and the primary estimand applies to the retained common-support cohort.",
        "",
        "### Naive donor-policy refit audit",
        "",
        markdown_table(donor_audit),
        "",
        "The table contrasts the broader model/window-specific Naive support with the frozen common cohort. The maintained equal-junction simulation helper uses its finite-depth simulation gate (not the stricter 10-events-per-depth-stratum inferential gate), which is named explicitly in the table. Conclusion-bearing donor and hybrid simulations use only the fixed-common-cohort means; they were rerun rather than filtered from simulations generated with broader donor means.",
        "",
        "## Full-cohort support and target alignment audit",
        "",
        markdown_table(support),
        "",
        markdown_table(target_summary),
        "",
        "The three expected policy rows without maintained whole-session targets are `837-12day2`, `838-12day2`, and `839-12day2`. Policy-support counts and simulation/calibration denominators are therefore reported separately.",
        "",
        "## Full-cohort exposure audit",
        "",
        markdown_table(exposure),
        "",
        "## Conclusion-bearing common-cohort exposure audit",
        "",
        markdown_table(common_exposure),
        "",
        "## Calibration",
        "",
        markdown_table(calibration_display),
        "",
        "## Gap recapitulation",
        "",
        markdown_table(gap_display),
        "",
        "Point estimates above are session-weighted within each group, matching the maintained manuscript estimand. Confidence intervals sample the union of Naive and Return animal IDs once per draw and retain every group/session row belonging to each sampled animal. Donor means remain fixed at their full common-cohort estimates and Monte Carlo walk noise is not propagated.",
        "",
        "## Mouse-equal-weighted sensitivity",
        "",
        markdown_table(mouse_equal_gap),
        "",
        "This sensitivity first averages repeated sessions within mouse and then gives each mouse equal group weight. It does not replace the session-weighted primary result.",
        "",
        "## Predictor-outcome overlap",
        "",
        markdown_table(outcome_overlap),
        "",
        "The policy parameters use no endpoint identity or occupancy, but the comparison is not uniformly prospective: sessions already at 32 unique endpoints within the 20/30-minute clock have an outcome that is partly contemporaneous with the predictor window. Prospective or predictive wording is prohibited.",
        "",
        "## Full-support estimator/window sensitivity",
        "",
        markdown_table(full_support_gap),
        "",
        "These rows retain each model/window's available support and are provenance/sensitivity only; they are not used to choose 20 versus 30 minutes.",
        "",
        "## Bootstrap correction audit",
        "",
        markdown_table(uncertainty_summary),
        "",
        "The inherited bootstrap sampled Naive and Return mice independently even though animals overlap across groups. Those intervals remain exported with `legacy_independent_cluster` filenames. Primary intervals use the synchronized union-of-mice bootstrap.",
        "",
        "## Reproducibility of the 20-minute row",
        "",
        markdown_table(parity),
        "",
        "## Interpretation guardrails",
        "",
        "- The 20- and 30-minute windows were requested for consistency and comparison; neither is selected here by the largest effect, correlation, calibration statistic, neural correlation, or recapitulation ratio.",
        "- No paired inferential test of the 20-minute versus 30-minute estimates was specified or run. Numerical differences between windows therefore cannot establish that one is better.",
        "- Equal clock time is fair with respect to cumulative in-maze opportunity, but it does not equalize the amount of Explore behavior or the denominator of every conditional probability. Those quantities are exported explicitly.",
        "- Own-policy calibration asks whether a stationary biased walk reconstructs individual whole-session efficiency. It is implementation/model validation, not evidence that the model is biologically sufficient.",
        "- Policy-swap fractions are group-gap recapitulation diagnostics. They are not R-squared, variance explained, causal mediation, or a neural mechanism.",
        "- The maintained ED3 authority is not changed by this task-scoped bundle.",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_qa(
    output: Path,
    exports: list[Path],
    common_ledger: pd.DataFrame,
    parity: pd.DataFrame,
    uncertainty_audit: pd.DataFrame,
    target_audit: pd.DataFrame,
) -> Path:
    included = common_ledger.loc[common_ledger["included_fixed_common_cohort"]]
    counts = included.groupby("group")["nickname"].nunique().to_dict()
    compared = parity.loc[parity["status"].eq("compared")]
    max_parity_delta = float(compared["max_abs_delta"].fillna(0).max())
    max_point_delta = float(
        uncertainty_audit["point_delta_corrected_minus_legacy"].abs().max()
    )
    missing_targets = target_audit.loc[
        target_audit["overlap_status"].eq("left_only"), "nickname"
    ].drop_duplicates()
    expected_formats = {".png", ".svg", ".pdf", ".tiff"}
    export_formats = {path.suffix.lower() for path in exports}
    lines = [
        "# QA record",
        "",
        "- Backend: Python/matplotlib only.",
        "- Visual inspection: all three 2026-08-04 PNG previews passed for legibility, panel alignment, clipping, and label collisions; the supporting support panel is explicitly labelled `Full-cohort policy support`.",
        f"- Export formats: {', '.join(sorted(export_formats))}; expected manuscript formats present: {expected_formats.issubset(export_formats)}.",
        "- SVG and PDF retain editable vector text; PNG and TIFF were exported at 600 dpi.",
        f"- Frozen common cohort: {int(counts.get('naive', 0))} Naive and {int(counts.get('LTM', 0))} Return sessions.",
        f"- 20-minute policy parity maximum absolute delta: {max_parity_delta:.3g}.",
        f"- Corrected-versus-legacy bootstrap point-estimate maximum absolute delta: {max_point_delta:.3g}; only dependence-aware intervals change.",
        f"- Policy rows lacking scalar targets: {', '.join(sorted(missing_targets.astype(str)))}.",
        "- No paired 20-versus-30-minute inferential test was specified or run; the bundle does not select a window by the observed numerical result.",
        "- The task-scoped candidate was not promoted over the maintained ED3 authority.",
    ]
    path = output / "QA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-events-cache", type=Path, default=EVENT_CACHE)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-simulation", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    source = output / "source_tables"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    configure_modules(output)
    equal.configure_style()

    policy, skipped = extract_equal_junction_rows(source, force=args.force_extract)
    policy_path = source / "fixed_time_policy_rows.csv"
    if args.extract_only:
        print(f"fixed_time_policy_rows={policy_path}")
        print(
            policy.groupby(["window_minutes", "group"])[
                ["primary_policy_support", "primary_depth_support"]
            ].sum().to_string()
        )
        return

    equal_tables = run_equal_junction(
        policy,
        source,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        force_simulation=args.force_simulation,
    )
    events = load_event_cache(args.decision_events_cache.resolve())
    pooled_tables = run_event_pooled(
        events,
        source,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 200000,
    )

    target_audit = target_overlap_audit(policy)
    target_audit.to_csv(source / "equal_junction_target_overlap_audit.csv", index=False)
    outcome_session, outcome_summary, outcome_skips = outcome_overlap_audit(policy, events)
    outcome_session.to_csv(source / "predictor_outcome_overlap_session_audit.csv", index=False)
    outcome_summary.to_csv(source / "predictor_outcome_overlap_group_summary.csv", index=False)
    outcome_skips.to_csv(source / "predictor_outcome_overlap_trajectory_skips.csv", index=False)

    common_ledger, common_membership = build_common_support_ledger(
        policy, pooled_tables["policies"]
    )
    common_ledger.to_csv(source / "primary_common_cohort_membership_exclusion_ledger.csv", index=False)
    common_membership.to_csv(source / "primary_common_cohort_included_sessions.csv", index=False)
    outcome_common_session = outcome_session.merge(
        common_membership[["nickname", "group"]].drop_duplicates(),
        on=["nickname", "group"],
        how="inner",
        validate="one_to_one",
    )
    outcome_common_summary = summarize_outcome_overlap_sessions(outcome_common_session)
    outcome_common_session.to_csv(
        source / "primary_common_cohort_predictor_outcome_overlap_session_audit.csv",
        index=False,
    )
    outcome_common_summary.to_csv(
        source / "primary_common_cohort_predictor_outcome_overlap_group_summary.csv",
        index=False,
    )
    equal_common_policy = filter_policy_to_membership(policy, common_membership)
    event_common_policy = filter_policy_to_membership(
        pooled_tables["policies"], common_membership
    )
    equal_common = run_equal_common_cohort(
        equal_common_policy,
        source,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    event_common = run_event_common_cohort(
        event_common_policy,
        source,
        n_walks=args.n_walks,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 200000,
    )

    # Preserve inherited independent-group bootstrap files as legacy provenance,
    # then export corrected full-support sensitivity summaries separately.
    equal_full_gap, equal_full_contrib, equal_full_draws, equal_full_overlap = (
        synchronized_equal_counterfactual_summary(
            equal_tables["simulation"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 50000,
        )
    )
    event_full_gap, event_full_contrib, event_full_draws, event_full_overlap = (
        synchronized_event_counterfactual_summary(
            pooled_tables["counterfactuals"],
            pooled_tables["results"],
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 350000,
        )
    )
    equal_full_gap.to_csv(source / "equal_junction_gap_recapitulation_summary.csv", index=False)
    equal_full_contrib.to_csv(source / "equal_junction_policy_contribution_summary.csv", index=False)
    equal_full_draws.to_csv(
        source / "equal_junction_counterfactual_synchronized_bootstrap_draws.csv", index=False
    )
    event_full_gap.to_csv(source / "event_pooled_gap_recapitulation_summary.csv", index=False)
    event_full_contrib.to_csv(
        source / "event_pooled_four_bias_parameter_contributions.csv", index=False
    )
    event_full_draws.to_csv(
        source / "event_pooled_gap_synchronized_bootstrap_draws.csv", index=False
    )
    full_overlap = pd.concat(
        [equal_full_overlap, event_full_overlap], ignore_index=True
    )
    full_overlap.to_csv(source / "full_support_synchronized_mouse_overlap_audit.csv", index=False)
    uncertainty_audit = uncertainty_method_audit(
        equal_tables["gap"],
        equal_full_gap,
        pooled_tables["gap"],
        event_full_gap,
        equal_tables["contributions"],
        equal_full_contrib,
        pooled_tables["contributions"],
        event_full_contrib,
    )
    uncertainty_audit.to_csv(source / "legacy_vs_synchronized_bootstrap_audit.csv", index=False)

    equal_parity = parity_audit(
        policy,
        OLD_EQUAL_POLICY,
        reference_filter=lambda frame: frame["window_minutes"].eq(20),
        metrics=(*POLICY_COLUMNS, *COUNT_COLUMNS),
        estimator="equal_junction",
    )
    pooled_for_parity = pooled_tables["policies"].copy()
    pooled_for_parity["window_minutes"] = pooled_for_parity["quota"]
    pooled_parity = parity_audit(
        pooled_for_parity,
        OLD_EVENT_POLICY,
        reference_filter=lambda frame: frame["window_id"].eq("time_20m"),
        metrics=(*POLICY_COLUMNS, *COUNT_COLUMNS[1:]),
        estimator="event_pooled",
    )
    parity = pd.concat([equal_parity, pooled_parity], ignore_index=True)
    parity.to_csv(source / "twenty_min_policy_parity_audit.csv", index=False)
    compared = parity.loc[parity["status"].eq("compared")]
    if (
        compared[["n_current_only", "n_reference_only"]].fillna(0).to_numpy().max() > 0
        or compared["max_abs_delta"].fillna(0).max() > 1e-12
    ):
        raise ValueError("The 20-minute policy rerun failed parity against the prior audited bundle")

    model_frames: list[tuple[str, int, str, pd.DataFrame, pd.DataFrame]] = []
    for window in WINDOWS_MINUTES:
        equal_frame = equal_common["simulation"].loc[
            equal_common["simulation"]["window_minutes"].eq(window)
            & equal_common["simulation"]["simulation_depth_support"]
        ]
        equal_naive = equal_frame.loc[equal_frame["group"].eq("naive")]
        equal_return = equal_frame.loc[equal_frame["group"].eq("LTM")]
        for model in ("four_bias", "depth_stratified"):
            model_frames.append(
                ("equal_junction", window, model, equal_naive, equal_return)
            )
        for model, _ in pooled.MODELS:
            event_naive = event_common["results"].loc[
                event_common["results"]["window_minutes"].eq(window)
                & event_common["results"]["estimator"].eq(model)
                & event_common["results"]["group"].eq("naive")
                & event_common["results"]["simulation_supported"]
            ]
            event_return = event_common["counterfactuals"].loc[
                event_common["counterfactuals"]["window_id"].eq(f"time_{window}m")
                & event_common["counterfactuals"]["estimator"].eq(model)
                & event_common["counterfactuals"]["counterfactual_supported"]
            ]
            model_frames.append(
                ("event_pooled", window, model, event_naive, event_return)
            )
    mouse_equal_gap, mouse_equal_draws = mouse_equal_gap_sensitivity(
        model_frames,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 700000,
    )
    mouse_equal_gap.to_csv(source / "common_cohort_mouse_equal_gap_sensitivity.csv", index=False)
    mouse_equal_draws.to_csv(
        source / "common_cohort_mouse_equal_gap_synchronized_bootstrap_draws.csv",
        index=False,
    )

    support = support_summary(
        policy,
        pooled_tables["policies"],
        equal_tables["simulation"],
        pooled_tables["results"],
    )
    exposure = exposure_summary(policy, pooled_tables["policies"])
    common_exposure = exposure_summary(equal_common_policy, event_common_policy)
    common_exposure = common_exposure.loc[
        common_exposure["group"].isin(["naive", "LTM"])
    ].reset_index(drop=True)
    calibration = calibration_comparison(
        equal_common["calibration"], event_common["calibration"]
    )
    gap = gap_comparison(equal_common["gap"], event_common["gap"])
    full_support_gap = gap_comparison(equal_full_gap, event_full_gap)
    support.to_csv(source / "estimator_support_comparison.csv", index=False)
    exposure.to_csv(source / "window_exposure_audit.csv", index=False)
    common_exposure.to_csv(
        source / "primary_common_cohort_window_exposure_audit.csv", index=False
    )
    donor_audit = donor_policy_refit_audit(
        policy,
        equal_common_policy,
        pooled_tables["policies"],
        event_common_policy,
    )
    donor_audit.to_csv(
        source / "primary_common_cohort_naive_donor_policy_refit_audit.csv",
        index=False,
    )
    calibration.to_csv(source / "calibration_estimator_comparison.csv", index=False)
    gap.to_csv(source / "gap_estimator_comparison.csv", index=False)
    full_support_gap.to_csv(
        source / "full_support_gap_estimator_sensitivity.csv", index=False
    )

    exports = plot_manuscript_candidate(
        equal_common["gap"],
        equal_common["contributions"],
        equal_common["depth_stats"],
        equal_common["simulation"],
        equal_common["calibration"],
        outcome_common_summary,
        figures,
    )
    exports += plot_primary_comparison(
        equal_common["policy_stats"],
        equal_common["calibration"],
        equal_common["gap"],
        support,
        figures,
    )
    exports += plot_estimator_sensitivity(calibration, gap, figures)
    report = write_report(
        output,
        support,
        exposure,
        common_exposure,
        calibration,
        gap,
        parity,
        common_ledger,
        target_audit,
        outcome_common_summary,
        mouse_equal_gap,
        full_support_gap,
        uncertainty_audit,
        donor_audit,
    )
    qa = write_qa(
        output,
        exports,
        common_ledger,
        parity,
        uncertainty_audit,
        target_audit,
    )

    source_files = sorted(path for path in source.iterdir() if path.is_file())
    manifest = {
        "created": "2026-08-04",
        "script": Path(__file__),
        "status": "task_scoped_candidate_not_promoted",
        "figure_contract": {
            "core_conclusion": "Compare 20- and 30-minute fixed-clock estimates on one fixed common-support cohort without selecting by effect size.",
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "primary_estimator": "unweighted mean of node/side-specific conditional probabilities",
            "sensitivity_estimator": "eligible event-pooled conditional probabilities",
            "primary_common_cohort": "intersection across both clock windows and four-policy plus strict depth gates",
        },
        "timebase": "cumulative in-maze time from session start",
        "behavior_mode": "Explore only (SplitModeClips mode index 2)",
        "windows_minutes": list(WINDOWS_MINUTES),
        "representation": "non-root T-junction branch decisions on a depth-6 binary tree",
        "behavioral_target": "whole-session half_cover_eff = 32 / endpoint visits needed for 32 unique endpoints",
        "parameters": {
            "equal_junction_min_total_events": equal.MIN_EVENTS,
            "equal_junction_min_supported_node_or_side_estimates_per_policy": equal.MIN_POLICY_SUPPORT,
            "equal_junction_depth_min_raw_eligible_events_per_stratum": equal.MIN_POLICY_SUPPORT,
            "event_pooled_min_parameter_denominator": pooled.MIN_DENOMINATOR,
            "n_walks_per_session_condition": int(args.n_walks),
            "n_mouse_cluster_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "point_weighting": "session-weighted within group",
            "bootstrap": "synchronized union-of-mice cluster bootstrap",
            "donor_policy": "refit from fixed 19-session Naive common cohort separately at 20 and 30 minutes",
            "donor_uncertainty": "fixed donor means; donor estimation and Monte Carlo walk noise are not propagated",
        },
        "known_target_gaps": sorted(nickname for nickname, _ in KNOWN_MISSING_SCALAR_TARGETS),
        "primary_common_cohort_counts": common_membership.groupby("group")["nickname"].nunique().to_dict(),
        "primary_common_cohort_mice": common_membership.groupby("group")["animal_id"].nunique().to_dict(),
        "cross_group_mouse_overlap": equal_common["overlap"].to_dict(orient="records"),
        "inputs": {
            "cohort_membership": equal.REFERENCE_POLICY_ROWS,
            "whole_session_efficiency": equal.SCALAR_TABLE,
            "event_cache": args.decision_events_cache.resolve(),
        },
        "input_sha256": {
            "cohort_membership": sha256(equal.REFERENCE_POLICY_ROWS),
            "whole_session_efficiency": sha256(equal.SCALAR_TABLE),
            "event_cache": sha256(args.decision_events_cache.resolve()),
        },
        "source_tables": source_files,
        "figures": exports,
        "report": report,
        "qa": qa,
        "live_script_sha256": sha256(Path(__file__)),
        "output_sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in [*source_files, *exports, report, qa]
        },
        "command": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2) + "\n", encoding="utf-8")
    print(equal_common["calibration"].to_string(index=False))
    print(equal_common["gap"].to_string(index=False))
    print(event_common["calibration"].to_string(index=False))
    print(event_common["gap"].to_string(index=False))
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
