"""Prepare Extended Data Figure 5e plot-ready tables from the approved context control.

The approved definition uses every calcium-valid outside-maze frame in each
session and samples the same session-specific number of calcium-valid in-maze
frames without replacement.  No position or speed filter is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs/equal_time_context_fairness_2026-08-20/combined_primary"
OUTPUT = (
    REPO
    / "docs/figure_refinement_2026-08-20"
    / "ed05_round17_context_only_all_outside"
    / "source_inputs"
)

PAIR_SOURCE = SOURCE / "equal_time_pair_summary_context_only_all_outside.csv"
FEASIBILITY_SOURCE = SOURCE / "equal_time_pair_feasibility_combined.csv"
STATS_SOURCE = SOURCE / "equal_time_clustered_stats_current_vs_primary.csv"
REPEATS_SOURCE = SOURCE / "equal_time_repeat_results_combined.csv"
SESSION_POOLS_SOURCE = SOURCE / "equal_time_session_frame_pool_audit.csv"
ANALYSIS_MANIFEST = SOURCE / "manifest.json"
ANALYSIS_REPORT = SOURCE / "REPORT.md"

PAIR_OUTPUT = OUTPUT / "ed05e_context_only_all_outside_pair_rows.csv"
MEANS_OUTPUT = OUTPUT / "ed05e_context_only_all_outside_condition_means.csv"
STATS_OUTPUT = OUTPUT / "ed05e_context_only_all_outside_clustered_stats.csv"
VALIDATION_OUTPUT = OUTPUT / "ED05E_SOURCE_PREPARATION_VALIDATION.json"

DEFINITION = "Context-only, all outside"
SOURCE_SET = "context_only_all_outside_2026-08-20"
VARIANTS = ("d1v1", "d1v2", "d2v1", "d2v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_analysis_manifest() -> dict[str, object]:
    manifest = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8"))
    _require(
        manifest.get("sampling_rule")
        == (
            "all calcium-valid outside-maze frames in each session; same "
            "session-specific number drawn without replacement from annotated "
            "in-maze frames"
        ),
        "Unexpected sampling rule in analysis manifest",
    )
    _require(
        manifest.get("maze_frame_rule")
        == "annotated in-maze context only; no position or speed filter",
        "Unexpected in-maze frame rule in analysis manifest",
    )
    _require(manifest.get("technical_repeats") == 20, "Expected 20 draws")
    _require(manifest.get("random_seed") == 0, "Expected deterministic seed 0")
    expected_hashes = manifest.get("output_sha256", {})
    _require(isinstance(expected_hashes, dict), "Manifest output hashes are missing")
    for path in (
        PAIR_SOURCE,
        FEASIBILITY_SOURCE,
        STATS_SOURCE,
        REPEATS_SOURCE,
        SESSION_POOLS_SOURCE,
    ):
        key = str(path.relative_to(REPO))
        expected = expected_hashes.get(key)
        _require(expected == sha256(path), f"Source hash mismatch: {key}")
    return manifest


def prepare() -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = _verify_analysis_manifest()

    keys = ["variant", "group", "animal_id", "src_session", "dst_session"]
    pairs = pd.read_csv(PAIR_SOURCE, dtype={"animal_id": str})
    feasibility = pd.read_csv(FEASIBILITY_SOURCE, dtype={"animal_id": str})
    _require(len(pairs) == 144, f"Expected 144 pair rows; found {len(pairs)}")
    _require(not pairs.duplicated(keys).any(), "Duplicate pair-variant rows")
    _require(
        pairs.groupby("variant", observed=True).size().to_dict()
        == {variant: 36 for variant in VARIANTS},
        "Expected 36 pair rows in each day combination",
    )
    _require(pairs["animal_id"].nunique() == 14, "Expected 14 animals")
    _require(
        set(pairs["equal_time_outside_r_count"].astype(int)) == {20}
        and set(pairs["equal_time_maze_r_count"].astype(int)) == {20},
        "Every pair summary must average 20 draws",
    )

    feasibility_columns = [
        *keys,
        "feasible",
        "n_matched",
        "maze_frame_policy",
        "sampling_policy",
        "outside_train_draw_frames",
        "outside_test_draw_frames",
        "maze_train_draw_frames",
        "maze_test_draw_frames",
        "outside_train_frames",
        "outside_test_frames",
        "annotated_maze_train_frames",
        "annotated_maze_test_frames",
    ]
    panel_pairs = pairs.merge(
        feasibility.loc[:, feasibility_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    _require(panel_pairs["feasible"].fillna(False).all(), "Every pair must be feasible")
    _require(
        set(panel_pairs["maze_frame_policy"]) == {"annotated_context"},
        "In-maze pool is not context-only",
    )
    _require(
        set(panel_pairs["sampling_policy"]) == {"match_all_outside"},
        "Outside pool is not the all-outside policy",
    )
    for side in ("train", "test"):
        _require(
            np.array_equal(
                panel_pairs[f"outside_{side}_draw_frames"].to_numpy(dtype=int),
                panel_pairs[f"maze_{side}_draw_frames"].to_numpy(dtype=int),
            ),
            f"{side} context draw sizes do not match",
        )
        _require(
            np.array_equal(
                panel_pairs[f"outside_{side}_frames"].to_numpy(dtype=int),
                panel_pairs[f"outside_{side}_draw_frames"].to_numpy(dtype=int),
            ),
            f"{side} does not use all outside frames",
        )
        _require(
            np.all(
                panel_pairs[f"annotated_maze_{side}_frames"].to_numpy(dtype=int)
                >= panel_pairs[f"maze_{side}_draw_frames"].to_numpy(dtype=int)
            ),
            f"{side} in-maze pool is too small",
        )
    panel_pairs["source_set"] = SOURCE_SET
    panel_pairs["feasibility_source"] = SOURCE_SET
    panel_pairs.to_csv(PAIR_OUTPUT, index=False)

    stats = pd.read_csv(STATS_SOURCE)
    primary = stats.loc[stats["definition"].eq(DEFINITION)].copy()
    _require(len(primary) == 12, f"Expected 12 primary statistic rows; found {len(primary)}")
    _require(
        primary.groupby("variant", observed=True).size().to_dict()
        == {variant: 3 for variant in VARIANTS},
        "Expected outside, maze and paired-difference statistics for every variant",
    )
    _require(set(primary["n_pair_rows"].astype(int)) == {36}, "Expected 36 pairs")
    _require(set(primary["n_animals"].astype(int)) == {14}, "Expected 14 animals")

    means = primary.loc[primary["estimand"].isin(["outside", "maze"])].copy()
    means = means.rename(columns={"estimand": "condition", "estimate": "beta"})
    means.to_csv(MEANS_OUTPUT, index=False)

    panel_stats = primary.rename(
        columns={
            "estimand": "analysis",
            "estimate": "beta",
            "pvalue_bh4": "q_bh4",
        }
    )
    panel_stats.to_csv(STATS_OUTPUT, index=False)

    d1 = panel_stats.loc[
        panel_stats["variant"].eq("d1v1")
        & panel_stats["analysis"].eq("maze_minus_outside")
    ].iloc[0]
    _require(abs(float(d1["beta"]) - 0.1350853279179621) < 1e-12, "D1-D1 estimate drift")
    _require(abs(float(d1["ci95_low"]) - 0.0949884197821155) < 1e-12, "D1-D1 CI drift")
    _require(abs(float(d1["ci95_high"]) - 0.1751822360538087) < 1e-12, "D1-D1 CI drift")
    _require(abs(float(d1["q_bh4"]) - 1.5754841589554758e-05) < 1e-16, "D1-D1 BH4 drift")

    repeat_rows = pd.read_csv(REPEATS_SOURCE)
    session_pools = pd.read_csv(SESSION_POOLS_SOURCE)
    _require(len(repeat_rows) == 2880, "Expected 2,880 repeat rows")
    _require(len(session_pools) == 100, "Expected 100 unique session-pool rows")
    _require(
        len(session_pools[["session_id"]].drop_duplicates()) == 100,
        "Session-pool audit contains duplicate sessions",
    )

    payload: dict[str, object] = {
        "schema": "ed05e-context-only-all-outside-source-preparation/v1",
        "status": "PASS",
        "definition": DEFINITION,
        "sampling_rule": manifest["sampling_rule"],
        "maze_frame_rule": manifest["maze_frame_rule"],
        "outside_frame_rule": manifest["outside_frame_rule"],
        "timebase": manifest["neural_frame_timebase"],
        "technical_repeats": 20,
        "random_seed": 0,
        "statistical_unit": manifest["statistical_unit"],
        "multiple_comparison_family": manifest["multiple_comparison_family"],
        "n_pair_rows": 144,
        "n_d1v1_pairs": 36,
        "n_animals": 14,
        "n_unique_sessions": 100,
        "d1v1": {
            "outside": float(
                means.loc[
                    means["variant"].eq("d1v1") & means["condition"].eq("outside"),
                    "beta",
                ].iloc[0]
            ),
            "maze": float(
                means.loc[
                    means["variant"].eq("d1v1") & means["condition"].eq("maze"),
                    "beta",
                ].iloc[0]
            ),
            "maze_minus_outside": float(d1["beta"]),
            "ci95": [float(d1["ci95_low"]), float(d1["ci95_high"])],
            "raw_p": float(d1["pvalue"]),
            "bh4_adjusted_p": float(d1["q_bh4"]),
        },
        "source_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                PAIR_SOURCE,
                FEASIBILITY_SOURCE,
                STATS_SOURCE,
                REPEATS_SOURCE,
                SESSION_POOLS_SOURCE,
                ANALYSIS_MANIFEST,
                ANALYSIS_REPORT,
            )
        },
        "output_sha256": {
            path.name: sha256(path) for path in (PAIR_OUTPUT, MEANS_OUTPUT, STATS_OUTPUT)
        },
    }
    VALIDATION_OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    result = prepare()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
