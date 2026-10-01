"""Validate the complete AF23 direct-entry reconstruction and propagated sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = (
    REPO_ROOT / "docs" / "turn_sequence_retention_2026-08-11_direct_entry_reconstruction"
)
HISTORICAL_ROOT = REPO_ROOT / "docs" / "turn_sequence_retention_2026-04-28"
DEFAULT_CENTRAL_ROOT = (
    REPO_ROOT / "docs" / "extended_data_figure_panel_refresh_2026-07-12" / "source_tables"
)
EXPECTED_TRAJECTORY_SHA256 = (
    "23F3053645DB1108F52B8F30C99D7CA90B564E57C1D7ADE466DFF60901E04513"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _close(left: Any, right: float, atol: float = 1e-12) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--central-source-root", type=Path, default=DEFAULT_CENTRAL_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "af23_first_route_resolution_2026-08-11" / "VALIDATION.json",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    analysis_root = args.analysis_root.resolve()
    central_root = args.central_source_root.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    trajectory_candidates = [
        REPO_ROOT / "mm-trajectories" / "489-3day1-tf",
        Path(r"D:\combineddata\mm-trajectories\489-3day1-tf"),
    ]
    trajectory_hashes = {
        str(path): _sha256(path) for path in trajectory_candidates if path.exists()
    }
    check(
        "trajectory_hash",
        bool(trajectory_hashes)
        and set(trajectory_hashes.values()) == {EXPECTED_TRAJECTORY_SHA256},
        trajectory_hashes,
    )

    current = pd.read_csv(analysis_root / "reward_path_first_turns.csv")
    historical = pd.read_csv(HISTORICAL_ROOT / "reward_path_first_turns.csv")
    keys = ["session_id", "paper_depth"]
    check(
        "event_key_closure",
        len(current) == len(historical) == 936
        and current[keys].drop_duplicates().shape[0] == 936
        and set(map(tuple, current[keys].to_numpy()))
        == set(map(tuple, historical[keys].to_numpy())),
        {"current_rows": len(current), "historical_rows": len(historical)},
    )
    check(
        "analysis_choice_provenance",
        current["choice_origin"].value_counts().to_dict()
        == {"observed": 934, "inferred": 2},
        current["choice_origin"].value_counts().to_dict(),
    )

    merged = historical.merge(current, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    changed_keys: set[tuple[str, int]] = set()
    core_columns = ["chosen_child", "choice_side", "abs_frame", "reward_correct", "source"]
    for _, row in merged.iterrows():
        changed = False
        for column in core_columns:
            old = row[f"{column}_old"]
            new = row[f"{column}_new"]
            if pd.isna(old) and pd.isna(new):
                continue
            if str(old) != str(new):
                changed = True
                break
        if changed:
            changed_keys.add((str(row["session_id"]), int(row["paper_depth"])))
    check(
        "only_chronologically_affected_rows_changed",
        changed_keys == {("489-3day1", 1), ("489-3day1", 2)},
        sorted(changed_keys),
    )

    session_rows = current.loc[current["session_id"].eq("489-3day1")].sort_values("paper_depth")
    expected_children = [1, 3, 9, 22, 46, 94]
    expected_correct = [True, False, False, True, True, True]
    expected_origins = ["inferred", "inferred", "observed", "observed", "observed", "observed"]
    check(
        "489_choice_sequence",
        session_rows["chosen_child"].astype(int).tolist() == expected_children
        and session_rows["reward_correct"].astype(bool).tolist() == expected_correct
        and session_rows["choice_origin"].tolist() == expected_origins
        and session_rows.iloc[:2]["abs_frame"].isna().all(),
        session_rows[
            ["paper_depth", "parent_node", "chosen_child", "reward_correct", "choice_origin", "abs_frame"]
        ].to_dict(orient="records"),
    )
    depth2 = session_rows.loc[session_rows["paper_depth"].eq(2)].iloc[0]
    check(
        "later_parent1_exit_retained_only_as_audit",
        int(depth2["later_recorded_chosen_child"]) == 4
        and _close(depth2["later_recorded_abs_frame"], 4102.0)
        and bool(depth2["later_recorded_parent_exit_available"]),
        {
            "later_child": int(depth2["later_recorded_chosen_child"]),
            "later_frame": float(depth2["later_recorded_abs_frame"]),
        },
    )

    route = pd.read_csv(analysis_root / "session_route_summary.csv")
    route489 = route.loc[route["session_id"].eq("489-3day1")].iloc[0]
    check(
        "489_session_denominator",
        int(route489["n_decisions_scored"]) == 6
        and int(route489["n_decisions_observed"]) == 4
        and int(route489["n_decisions_inferred"]) == 2
        and int(route489["n_correct"]) == 4
        and _close(route489["route_fraction_correct"], 4.0 / 6.0)
        and not bool(route489["all6_observed"])
        and bool(route489["all6_resolved"]),
        route489.to_dict(),
    )

    entry = pd.read_csv(analysis_root / "entry_path_reconstruction_audit.csv")
    entry489 = entry.loc[entry["session_id"].eq("489-3day1")].iloc[0]
    check(
        "entry_anchor_and_unique_paths",
        _close(entry489["first_valid_abs_frame"], 378.0)
        and int(entry489["first_valid_cell"]) == 51
        and int(entry489["first_valid_grid_x"]) == 5
        and int(entry489["first_valid_grid_y"]) == 1
        and int(entry489["first_valid_run_node_from_cell"]) == 16
        and int(entry489["first_valid_run_node_from_no"]) == 16
        and entry489["home_to_entry_run_path"] == "0;1;3;7;16"
        and entry489["home_to_entry_cell_path"]
        == "0;1;2;3;4;5;6;7;8;9;10;11;16;17;18;19;32;33;50;51",
        entry489.to_dict(),
    )

    combined = pd.read_csv(central_root / "ed04_route_events_current_plus_new12.csv")
    check(
        "combined_event_provenance",
        len(combined) == 972
        and combined["choice_origin"].value_counts().to_dict()
        == {"observed": 970, "inferred": 2}
        and not combined["reward_correct"].isna().any(),
        {
            "rows": len(combined),
            "origins": combined["choice_origin"].value_counts().to_dict(),
            "missing_correctness": int(combined["reward_correct"].isna().sum()),
        },
    )
    combined_route = pd.read_csv(central_root / "ed04_session_route_current_plus_new12.csv")
    combined489 = combined_route.loc[combined_route["session_id"].eq("489-3day1")].iloc[0]
    check(
        "combined_489_score",
        _close(combined489["route_fraction_correct"], 4.0 / 6.0)
        and int(combined489["n_decisions_observed"]) == 4
        and int(combined489["n_decisions_inferred"]) == 2,
        combined489.to_dict(),
    )

    group_stats = pd.read_csv(central_root / "ed04_route_group_stats_current_plus_new12.csv")
    return_mean = group_stats.loc[group_stats["condition"].eq("return"), "mean_route_fraction_correct"].iloc[0]
    check("return_group_mean", _close(return_mean, 0.6904761904761904), float(return_mean))

    contrasts = pd.read_csv(central_root / "ed04_route_stats_current_plus_new12.csv").set_index("comparison")
    expected_q = {
        "experienced_day2_vs_naive": 0.00044171178338260825,
        "experienced_day2_vs_return": 0.000157881,
        "experienced_day2_vs_latent_day2": 0.043543941295848064,
        "return_vs_naive": 0.164326986,
        "latent_day2_vs_naive": 0.5723311597921615,
        "return_vs_latent_day2": 0.739280295,
    }
    q_pass = all(
        name in contrasts.index and _close(contrasts.loc[name, "q_bh6"], value, atol=5e-10)
        for name, value in expected_q.items()
    )
    supported = contrasts.index[contrasts["supported_bh6"].astype(bool)].tolist()
    check(
        "bh6_statistics",
        q_pass
        and set(supported)
        == {
            "experienced_day2_vs_naive",
            "experienced_day2_vs_return",
            "experienced_day2_vs_latent_day2",
        },
        {
            "q": contrasts["q_bh6"].to_dict(),
            "supported": supported,
        },
    )

    central_manifest_path = central_root / "ed04_current_render_manifest.json"
    central_manifest = json.loads(central_manifest_path.read_text(encoding="utf-8"))
    hash_results: dict[str, bool] = {}
    refresh_root = central_root.parent
    for rel, expected in central_manifest.get("sha256", {}).items():
        path = refresh_root / rel
        hash_results[rel] = path.exists() and _sha256(path) == expected
    check(
        "central_manifest_hash_closure",
        central_manifest.get("status") == "af23_direct_entry_reconstruction_complete"
        and bool(hash_results)
        and all(hash_results.values()),
        {"files": len(hash_results), "failures": [key for key, value in hash_results.items() if not value]},
    )

    svg_path = analysis_root / "figures" / "entry_path_reconstruction_489-3day1.svg"
    svg = svg_path.read_text(encoding="utf-8") if svg_path.exists() else ""
    check(
        "entry_audit_svg_vector_editable",
        bool(svg) and "<image" not in svg and "489-3day1" in svg,
        {"path": str(svg_path), "embedded_images": svg.count("<image")},
    )

    passed = sum(item["passed"] for item in checks)
    payload = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "decision": {
            "session_id": "489-3day1",
            "assumption": "direct home to first valid location",
            "first_valid_location": {"cell": 51, "grid": [5, 1], "run_node": 16, "abs_frame": 378},
            "run_path": [0, 1, 3, 7, 16],
            "inferred_choices": [
                {"parent": 0, "child": 1, "reward_correct": True},
                {"parent": 1, "child": 3, "reward_correct": False},
            ],
            "directly_observed_choices_retained": 4,
            "resolved_denominator": 6,
            "n_correct": 4,
            "route_fraction_correct": 4.0 / 6.0,
        },
        "source_hashes": trajectory_hashes,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "passed": passed, "total": len(checks), "output": str(args.output)}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
