"""Audit ED6/Figure 3 curve support without a minimum curve-span exclusion.

ED6 uses all jointly observed Day i and Day i+1 30-min pivot positions within
each recording cycle. Figure 3 uses every finite fixed-reference Day i window
already summarized for each of the frozen current59 recording cycles. The
script writes caption-facing statistics and explicit cohort/support ledgers;
it does not render or modify figures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "docs"
    / "coverage5min_window_reanalysis_2026-08-05"
    / "rendered"
)
PIVOT_ROWS_PATH = SOURCE_ROOT / "ed07_pivot" / "ed07_pivot_maze_time_source_rows_current59.csv"
FIXED_ROOT = SOURCE_ROOT / "ed07_fixed_time_tile"
ROSTER_PATH = FIXED_ROOT / "ed07_current59_source_pair_provenance.csv"
FIXED_PAIR_PATH = FIXED_ROOT / "ed07_current59_pair_metric_inclusion_audit.csv"
FIXED_STATS_PATH = FIXED_ROOT / "ed07_current59_clustered_stats.csv"
BOUNDARY_PATH = (
    REPO_ROOT
    / "docs"
    / "sorted_data_full_figure_rerun_2026-06-11"
    / "figure2_main3_ed12_source_repair_842_2026-08-03"
    / "panels_hi_840_naive_cohort_boundary_audit.csv"
)
DEFAULT_OUT = REPO_ROOT / "docs" / "ed06_figure3_available_curve_support_2026-08-16"
CAPTION_PATH = (
    REPO_ROOT
    / "docs"
    / "manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)

METRICS = {
    "place_map_correlation": "Place map correlation",
    "pv_correlation": "Population vector correlation",
    "ensemble_rate_correlation": "Ensemble rate correlation",
}
EXPECTED_STAGE_COUNTS = {"naive": 22, "3mo": 14, "6mo": 12, "9mo": 7, "12mo": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _cluster_intercept_test(frame: pd.DataFrame, value_col: str) -> dict[str, float | int]:
    work = frame[[value_col, "animal"]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    groups = work["animal"].astype(str)
    n_clusters = int(groups.nunique())
    if len(work) < 3 or n_clusters < 3:
        raise ValueError(f"Insufficient support for clustered test of {value_col}")
    fit = sm.OLS(work[value_col].to_numpy(float), np.ones((len(work), 1))).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups, "use_correction": True},
    )
    estimate = float(fit.params[0])
    cluster_se = float(fit.bse[0])
    df = n_clusters - 1
    t_value = estimate / cluster_se
    critical = float(stats.t.ppf(0.975, df))
    return {
        "n_recording_cycles": int(len(work)),
        "n_animals": n_clusters,
        "estimate": estimate,
        "cluster_se": cluster_se,
        "t_value": t_value,
        "df": df,
        "p_two_sided": float(2.0 * stats.t.sf(abs(t_value), df)),
        "ci_low": estimate - critical * cluster_se,
        "ci_high": estimate + critical * cluster_se,
    }


def _load_roster() -> pd.DataFrame:
    roster = pd.read_csv(ROSTER_PATH, low_memory=False)
    if len(roster) != 59 or roster["pair_id"].nunique() != 59:
        raise ValueError("The frozen current59 roster must contain exactly 59 unique cycles")
    if roster["animal"].astype(str).nunique() != 22:
        raise ValueError("The frozen current59 roster must contain exactly 22 animals")
    stage_counts = roster.groupby("experience_group")["pair_id"].nunique().to_dict()
    if stage_counts != EXPECTED_STAGE_COUNTS:
        raise ValueError(f"Unexpected experience counts: {stage_counts}")
    return roster


def _build_ed06_pair_summaries(roster: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "pair_id",
        "animal",
        "pair_experience_group",
        "session_id",
        "day",
        "metric",
        "support_policy",
        "window_type",
        "window_size",
        "pivot_time_min",
        "sim_mean",
        "gate_pass",
        "included_value",
    ]
    rows = pd.read_csv(PIVOT_ROWS_PATH, usecols=usecols, low_memory=False)
    rows = rows[
        rows["pair_id"].isin(roster["pair_id"])
        & rows["metric"].isin(METRICS)
        & rows["window_type"].eq("maze_time")
        & pd.to_numeric(rows["window_size"], errors="coerce").eq(30.0)
        & rows["day"].isin(["day1", "day2"])
        & rows["gate_pass"].astype(str).str.lower().eq("true")
        & rows["included_value"].astype(str).str.lower().eq("true")
    ].copy()
    rows["pivot_time_min"] = pd.to_numeric(rows["pivot_time_min"], errors="coerce")
    rows["sim_mean"] = pd.to_numeric(rows["sim_mean"], errors="coerce")
    rows = rows[np.isfinite(rows["pivot_time_min"]) & np.isfinite(rows["sim_mean"])]

    grouped = (
        rows.groupby(
            [
                "pair_id",
                "animal",
                "pair_experience_group",
                "day",
                "metric",
                "pivot_time_min",
            ],
            as_index=False,
        )["sim_mean"]
        .mean()
    )
    summaries: list[dict[str, object]] = []
    roster_lookup = roster.set_index("pair_id")
    for pair_id in roster["pair_id"].astype(str):
        roster_row = roster_lookup.loc[pair_id]
        for metric in METRICS:
            panel = grouped[
                grouped["pair_id"].astype(str).eq(pair_id) & grouped["metric"].eq(metric)
            ]
            day1 = panel[panel["day"].eq("day1")][["pivot_time_min", "sim_mean"]].rename(
                columns={"sim_mean": "day_i_value"}
            )
            day2 = panel[panel["day"].eq("day2")][["pivot_time_min", "sim_mean"]].rename(
                columns={"sim_mean": "day_i1_value"}
            )
            common = day1.merge(day2, on="pivot_time_min", how="inner", validate="one_to_one")
            common = common.sort_values("pivot_time_min")
            n_points = int(len(common))
            if n_points < 3:
                raise ValueError(f"{pair_id} {metric} has only {n_points} joint pivots")
            x_hour = common["pivot_time_min"].to_numpy(float) / 60.0
            day_i = common["day_i_value"].to_numpy(float)
            day_i1 = common["day_i1_value"].to_numpy(float)
            summaries.append(
                {
                    "pair_id": pair_id,
                    "animal": str(roster_row["animal"]),
                    "experience_group": str(roster_row["experience_group"]),
                    "session_day_i": str(roster_row["session_a"]),
                    "session_day_i1": str(roster_row["session_b"]),
                    "metric": metric,
                    "metric_label": METRICS[metric],
                    "n_joint_pivots": n_points,
                    "first_joint_pivot_min": float(common["pivot_time_min"].min()),
                    "last_joint_pivot_min": float(common["pivot_time_min"].max()),
                    "joint_pivot_span_min": float(
                        common["pivot_time_min"].max() - common["pivot_time_min"].min()
                    ),
                    "day_i_within_curve_sd": float(np.std(day_i, ddof=1)),
                    "day_i1_within_curve_sd": float(np.std(day_i1, ddof=1)),
                    "day_i1_minus_day_i_sd": float(
                        np.std(day_i1, ddof=1) - np.std(day_i, ddof=1)
                    ),
                    "day_i_slope_r_per_hour": float(np.polyfit(x_hour, day_i, 1)[0]),
                    "day_i1_slope_r_per_hour": float(np.polyfit(x_hour, day_i1, 1)[0]),
                }
            )
    result = pd.DataFrame(summaries)
    expected = 59 * len(METRICS)
    if len(result) != expected:
        raise ValueError(f"Expected {expected} ED6 pair-metric rows; found {len(result)}")
    counts = result.groupby("metric")["pair_id"].nunique().to_dict()
    if counts != {metric: 59 for metric in METRICS}:
        raise ValueError(f"Not every ED6 metric retained all 59 cycles: {counts}")
    return result


def _build_ed06_stats(pair_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    families = {
        "day_i1_minus_day_i_within_curve_sd": "day_i1_minus_day_i_sd",
        "day_i_linear_slope": "day_i_slope_r_per_hour",
        "day_i1_linear_slope": "day_i1_slope_r_per_hour",
    }
    for estimand, value_col in families.items():
        family_rows: list[dict[str, object]] = []
        for metric, panel in pair_rows.groupby("metric", sort=False):
            record: dict[str, object] = {
                "estimand": estimand,
                "metric": metric,
                "metric_label": METRICS[metric],
                "value_column": value_col,
                **_cluster_intercept_test(panel, value_col),
            }
            if estimand == "day_i1_minus_day_i_within_curve_sd":
                record["mean_day_i_within_curve_sd"] = float(panel["day_i_within_curve_sd"].mean())
                record["mean_day_i1_within_curve_sd"] = float(panel["day_i1_within_curve_sd"].mean())
            family_rows.append(record)
        p_values = np.asarray([float(row["p_two_sided"]) for row in family_rows])
        adjusted = multipletests(p_values, method="fdr_bh")[1]
        for row, adjusted_p in zip(family_rows, adjusted):
            row["p_bh3_two_sided"] = float(adjusted_p)
            row["correction_family"] = estimand + "__BH3"
        output.extend(family_rows)
    return pd.DataFrame(output)


def _build_figure3_audit(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows = pd.read_csv(FIXED_PAIR_PATH, low_memory=False)
    pair_rows = pair_rows[
        pair_rows["metric"].isin(METRICS)
        & pair_rows["window_type"].eq("maze_time")
        & pd.to_numeric(pair_rows["window_size"], errors="coerce").eq(30.0)
    ].copy()
    pair_rows["included_in_slope_test"] = (
        pair_rows["included_in_slope_test"].astype(str).str.lower().eq("true")
    )
    for metric, panel in pair_rows.groupby("metric"):
        if len(panel) != 59 or panel["pair_id"].nunique() != 59:
            raise ValueError(f"Figure 3 {metric} does not contain 59 unique cycles")
        if not panel["included_in_slope_test"].all():
            failed = panel.loc[~panel["included_in_slope_test"], "pair_id"].tolist()
            raise ValueError(f"Figure 3 {metric} excluded cycles: {failed}")
        if set(panel["pair_id"].astype(str)) != set(roster["pair_id"].astype(str)):
            raise ValueError(f"Figure 3 {metric} pair IDs differ from current59")

    saved_stats = pd.read_csv(FIXED_STATS_PATH, low_memory=False)
    saved_stats = saved_stats[
        saved_stats["metric"].isin(METRICS)
        & saved_stats["window_type"].eq("maze_time")
        & pd.to_numeric(saved_stats["window_size"], errors="coerce").eq(30.0)
        & saved_stats["estimand"].eq("mean_pairwise_within_day1_slope")
    ].copy()
    if len(saved_stats) != len(METRICS):
        raise ValueError("Expected three Figure 3 fixed-reference slope rows")
    checks: list[dict[str, object]] = []
    for metric, panel in pair_rows.groupby("metric", sort=False):
        recomputed = _cluster_intercept_test(panel, "slope_per_hour_or_ktile")
        saved = saved_stats[saved_stats["metric"].eq(metric)].iloc[0]
        checks.append(
            {
                "metric": metric,
                "metric_label": METRICS[metric],
                **recomputed,
                "saved_estimate": float(saved["estimate"]),
                "saved_p_two_sided": float(saved["p_two_sided"]),
                "saved_p_bh18_two_sided": float(saved["q_bh18_two_sided"]),
                "abs_estimate_difference": abs(recomputed["estimate"] - float(saved["estimate"])),
                "all_59_cycles_retained": True,
                "curve_support_rule": "all finite available Day i windows within each recording cycle",
            }
        )
    check_frame = pd.DataFrame(checks)
    if check_frame["abs_estimate_difference"].max() > 1e-12:
        raise ValueError("Figure 3 saved and recomputed slopes do not match")
    return pair_rows, check_frame


def _write_report(
    out_dir: Path,
    roster: pd.DataFrame,
    ed06_pair_rows: pd.DataFrame,
    ed06_stats: pd.DataFrame,
    figure3_stats: pd.DataFrame,
) -> None:
    stage_rows = roster.groupby("experience_group")["pair_id"].nunique().to_dict()
    roster_animals = ", ".join(
        sorted(roster["animal"].astype(str).unique(), key=lambda value: int(value))
    )
    lines = [
        "# ED6 and Figure 3 available-curve support audit",
        "",
        "## Cohort",
        "",
        f"The frozen authority contains 59 recording cycles (118 Day i/Day i+1 sessions) from 22 animals: {roster_animals}.",
        "",
        f"Experience counts are {stage_rows}. Figure 3 retains all 59 cycles. ED6 now also retains all 59 cycles for every metric; no cycle is excluded solely because its eligible pivot span is shorter than 30 min.",
        "",
        "Mouse 840 and the later 840/841 longitudinal cycles remain outside the frozen authority. The accepted naive 841day1__841day2 cycle remains included. This upstream cohort boundary is separate from analysis support.",
        "",
        "## ED6 estimand",
        "",
        "For each recording cycle and metric, Day i and Day i+1 are restricted to their exactly intersecting valid 30-min pivot positions. Within-curve s.d. and a linear slope are calculated separately for each day on that pair-specific common range. Pair-level summaries are pooled across experience stages only after calculation, and inference uses CR1 standard errors clustered by animal with Student-t reference degrees of freedom. BH correction is applied separately across the three metrics for the s.d. contrast, Day i slopes and Day i+1 slopes.",
        "",
        "| metric | n cycles | minimum joint pivots | shortest span (min) | mean s.d. Day i | mean s.d. Day i+1 | Day i+1 - Day i s.d. | BH p | Day i slope (r/h) | BH p | Day i+1 slope (r/h) | BH p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        pairs = ed06_pair_rows[ed06_pair_rows["metric"].eq(metric)]
        sd_row = ed06_stats[
            ed06_stats["metric"].eq(metric)
            & ed06_stats["estimand"].eq("day_i1_minus_day_i_within_curve_sd")
        ].iloc[0]
        d1_row = ed06_stats[
            ed06_stats["metric"].eq(metric)
            & ed06_stats["estimand"].eq("day_i_linear_slope")
        ].iloc[0]
        d2_row = ed06_stats[
            ed06_stats["metric"].eq(metric)
            & ed06_stats["estimand"].eq("day_i1_linear_slope")
        ].iloc[0]
        lines.append(
            f"| {METRICS[metric]} | 59 | {int(pairs['n_joint_pivots'].min())} | "
            f"{pairs['joint_pivot_span_min'].min():.0f} | {sd_row['mean_day_i_within_curve_sd']:.6g} | "
            f"{sd_row['mean_day_i1_within_curve_sd']:.6g} | {sd_row['estimate']:.6g} | "
            f"{sd_row['p_bh3_two_sided']:.6g} | {d1_row['estimate']:.6g} | "
            f"{d1_row['p_bh3_two_sided']:.6g} | {d2_row['estimate']:.6g} | "
            f"{d2_row['p_bh3_two_sided']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Figure 3",
            "",
            "The 30-min fixed-reference curve moves only through Day i; the full Day i+1 representation is the reference. All 59 recording cycles contribute one pair-level slope calculated from every finite available Day i window. Shorter Day i+1 duration therefore does not remove a cycle from this analysis.",
            "",
            "| metric | cycles | animals | mean pair slope (r/h) | BH18 p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in figure3_stats.itertuples(index=False):
        lines.append(
            f"| {row.metric_label} | {row.n_recording_cycles} | {row.n_animals} | "
            f"{row.estimate:.6g} | {row.saved_p_bh18_two_sided:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Timebase and support",
            "",
            "The x axis is cumulative elapsed time in maze, not accumulated running time. Neural maps use speed-filtered samples (2 < speed < 60 cm/s), and each 30-min pivot side must contain at least 300 s of selected neural data.",
            "",
            "## Outputs",
            "",
            "- `current59_recording_cycle_roster.csv`: exact 59-cycle roster.",
            "- `cohort_boundary_840_841.csv`: maintained upstream boundary evidence.",
            "- `ed06_pair_metric_available_curve_summaries.csv`: per-cycle transformed values and support.",
            "- `ed06_available_curve_clustered_stats.csv`: caption-facing ED6 tests.",
            "- `figure3_30min_pair_slopes_current59.csv`: exact Figure 3 pair-level inputs.",
            "- `figure3_30min_slope_validation.csv`: saved-versus-recomputed Figure 3 statistics.",
            "- `validation.json`: machine-readable cohort and statistic parity checks.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(out_dir: Path = DEFAULT_OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    roster = _load_roster()
    ed06_pair_rows = _build_ed06_pair_summaries(roster)
    ed06_stats = _build_ed06_stats(ed06_pair_rows)
    figure3_pairs, figure3_stats = _build_figure3_audit(roster)
    boundary = pd.read_csv(BOUNDARY_PATH, low_memory=False)

    roster_out = roster[
        [
            "animal",
            "pair_id",
            "session_a",
            "session_b",
            "pair_family",
            "experience_group",
            "source_bundle",
            "effective_matching_mode",
            "provenance_basis",
        ]
    ].copy()
    roster_out["n_session_records"] = 2

    outputs = {
        "current59_recording_cycle_roster.csv": roster_out,
        "cohort_boundary_840_841.csv": boundary,
        "ed06_pair_metric_available_curve_summaries.csv": ed06_pair_rows,
        "ed06_available_curve_clustered_stats.csv": ed06_stats,
        "figure3_30min_pair_slopes_current59.csv": figure3_pairs,
        "figure3_30min_slope_validation.csv": figure3_stats,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, index=False)
    _write_report(out_dir, roster, ed06_pair_rows, ed06_stats, figure3_stats)

    caption = CAPTION_PATH.read_text(encoding="utf-8")
    caption_fragments = {
        "figure3_30min_slopes": "Mean slopes were 0.0867, 0.0907 and 0.0648",
        "figure3_all_cycles": "All 59 recording cycles from 22 animals contributed one slope per metric",
        "ed06_place_sd": "Within-curve s.d. was 0.0243 $r$ on Day $\\mathrm{i}+1$ and 0.0487 $r$ on Day $\\mathrm{i}$",
        "ed06_pv_sd": "Within-curve s.d. was 0.0274 $r$ on Day $\\mathrm{i}+1$ and 0.0432 $r$ on Day $\\mathrm{i}$",
        "ed06_rate_sd": "Within-curve s.d. was 0.0369 $r$ on Day $\\mathrm{i}+1$ and 0.0363 $r$ on Day $\\mathrm{i}$",
        "ed06_all_cycles": "Fixed reference and sliding pivot analyses both used 59 recording cycles from 22 animals",
        "ed06_shortest_curve": "the shortest included curves contained 24 pivots from 30 to 53 min",
    }
    caption_checks = {
        name: fragment in caption for name, fragment in caption_fragments.items()
    }
    if not all(caption_checks.values()):
        failed = [name for name, passed in caption_checks.items() if not passed]
        raise ValueError(f"Caption parity failed: {failed}")

    validation = {
        "status": "PASS",
        "current59_unique_cycles": int(roster["pair_id"].nunique()),
        "current59_unique_animals": int(roster["animal"].astype(str).nunique()),
        "ed06_cycles_by_metric": {
            metric: int(panel["pair_id"].nunique())
            for metric, panel in ed06_pair_rows.groupby("metric")
        },
        "ed06_minimum_joint_pivots": int(ed06_pair_rows["n_joint_pivots"].min()),
        "ed06_shortest_joint_span_min": float(ed06_pair_rows["joint_pivot_span_min"].min()),
        "figure3_cycles_by_metric": {
            metric: int(panel["pair_id"].nunique())
            for metric, panel in figure3_pairs.groupby("metric")
        },
        "figure3_max_abs_saved_estimate_difference": float(
            figure3_stats["abs_estimate_difference"].max()
        ),
        "caption_checks": caption_checks,
    }
    (out_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    input_paths = [
        PIVOT_ROWS_PATH,
        ROSTER_PATH,
        FIXED_PAIR_PATH,
        FIXED_STATS_PATH,
        BOUNDARY_PATH,
        Path(__file__).resolve(),
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "ed06_figure3_available_curve_support",
        "cohort": {
            "n_recording_cycles": 59,
            "n_session_records": 118,
            "n_animals": 22,
            "stage_counts": EXPECTED_STAGE_COUNTS,
            "additional_840_and_later_840_841": "remain outside frozen authority",
            "accepted_841_naive": "retained",
        },
        "ed06_support_rule": "all exactly joint valid Day i/Day i+1 30-min pivots per pair; minimum 3 points; no minimum span",
        "figure3_support_rule": "all finite available Day i windows per pair; 59/59 retained",
        "validated_caption": {
            "path": CAPTION_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(CAPTION_PATH),
        },
        "inputs": [
            {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(path)}
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
