"""Build the full-roster ED2 complete-second-day Pearson candidate."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import analyze_ed03_turning_bias_fixed_time as equal
from ca_analysis.compat import bootstrap_environment
from ca_analysis.exploration import (
    compute_exploration_cycles,
    compute_scalar_efficiency_metrics,
    extract_explore_endnode_sequence,
)
from ca_analysis.figure_panels import DEFAULT_MAZE_ID
from ca_analysis.maze import NewMaze
from ca_analysis.trajectory import LoadTraj, SplitModeClips


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "ed02_1616_restored_base_candidate_2026-08-20"
PARENT = REPO / "docs" / "figure_version_candidates" / PARENT_NAME
CANDIDATE = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed02_1616_restored_pearson74_candidate_2026-08-20"
)
SOURCE_SCALARS = (
    REPO
    / "docs"
    / "behavior_panel_sensitivity_840_841_with_new12_baseline_2026-06-06"
    / "turning_bias_mechanism_current_plus_new12_exploration_scalars.csv"
)
MISSING_SESSIONS = ("837-12day2", "838-12day2", "839-12day2")
N_WALKS = 100
SEED = 20260804
COMPOSITE_STEM = "ed02_1616_restored_pearson74_candidate"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scalar_rows() -> pd.DataFrame:
    bootstrap_environment(repo_root=REPO)
    maze = NewMaze(DEFAULT_MAZE_ID)
    rows: list[dict[str, Any]] = []
    for nickname in MISSING_SESSIONS:
        trajectory = LoadTraj(f"{nickname}-tf")
        sequence = extract_explore_endnode_sequence(
            trajectory,
            maze,
            split_mode_clips_fn=SplitModeClips,
            explore_idx=2,
            end_level=6,
        )
        scalar = compute_scalar_efficiency_metrics(
            sequence, num_states=32, window_size=100, n_bins=20
        )
        cycles = compute_exploration_cycles(sequence, num_states=32, complete_only=True)
        complete_steps = (
            pd.to_numeric(cycles["steps_in_cycle"], errors="coerce").dropna()
            if not cycles.empty and "steps_in_cycle" in cycles.columns
            else pd.Series(dtype=float)
        )
        n_endnodes = int(scalar["n_explore_endnodes"])
        row = dict(scalar)
        row.update(
            {
                "nickname": nickname,
                "group": "Rewarded + LTM D2 (all)",
                "reward_context": "rewarded_ltm",
                "time_axis": "explore_node_bin",
                "median_steps_to_32": (
                    float(complete_steps.median()) if len(complete_steps) else np.nan
                ),
                "n_complete_cycles": int(len(complete_steps)),
                "cycles_per_1000_steps": (
                    float(len(complete_steps) / (n_endnodes / 1000.0))
                    if n_endnodes > 0
                    else np.nan
                ),
                "correction_note": (
                    "Round25: restored accepted 12-month Day 2 outcome target"
                ),
                "source_set": "new_12mo_day2_round25",
                "animal_id": nickname.split("-", 1)[0],
                "trajectory_id": f"{nickname}-tf",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def update_manifest_hashes(path: Path, root: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    fields = list(rows[0])
    for row in rows:
        artifact = root / row["path"]
        row["sha256"] = sha256(artifact)
        row["bytes"] = str(artifact.stat().st_size)
        if "rows" in row:
            row["rows"] = str(len(pd.read_csv(artifact)))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis_manifest() -> None:
    rows = []
    for path in sorted((CANDIDATE / "analysis_inputs").rglob("*.csv")):
        rows.append(
            {
                "path": path.relative_to(CANDIDATE).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "rows": len(pd.read_csv(path)),
            }
        )
    pd.DataFrame(rows).to_csv(CANDIDATE / "ANALYSIS_INPUT_MANIFEST.csv", index=False)


def build() -> dict[str, Any]:
    if CANDIDATE.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {CANDIDATE}")
    shutil.copytree(PARENT, CANDIDATE)
    for stale in (
        "RECORD_HASHES.csv",
        "RECORD_VALIDATION.json",
        "PROMOTION_COMPLETE.txt",
        "ED02_ROUND24_PROMOTION.json",
    ):
        (CANDIDATE / stale).unlink(missing_ok=True)

    adapter = REPO / "scripts" / "render_ed02_round25_pearson74_adapter.py"
    shutil.copy2(
        adapter,
        CANDIDATE / "source_code" / "render_ed03_fixed_time_30min_full_story.py",
    )
    shutil.copy2(Path(__file__), CANDIDATE / "source_code" / Path(__file__).name)

    added = scalar_rows()
    scalar = pd.read_csv(SOURCE_SCALARS, dtype={"animal_id": str})
    if set(scalar["nickname"]) & set(MISSING_SESSIONS):
        raise RuntimeError("Missing Day 2 targets unexpectedly exist in the source scalar table")
    scalar = pd.concat([scalar, added], ignore_index=True, sort=False)
    scalar_path = CANDIDATE / "analysis_inputs" / "natural" / "whole_session_efficiency_scalars.csv"
    scalar.to_csv(scalar_path, index=False)
    added.to_csv(
        CANDIDATE / "cohort_audit" / "restored_second_day_efficiency_scalars.csv",
        index=False,
    )

    policy_path = CANDIDATE / "analysis_inputs" / "natural" / "fixed_time_policy_rows.csv"
    policy = pd.read_csv(policy_path, dtype={"animal_id": str})
    policy = policy.loc[pd.to_numeric(policy["window_minutes"]).eq(30)].copy()
    policy_counts = policy.groupby("group").size().to_dict()
    if policy_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise RuntimeError(f"Natural policy cohort drifted: {policy_counts}")

    previous_path = (
        CANDIDATE
        / "analysis_inputs"
        / "natural"
        / "equal_junction_simulation_session_rows.csv"
    )
    previous = pd.read_csv(previous_path, dtype={"animal_id": str})
    equal.SCALAR_TABLE = scalar_path
    equal.WINDOWS_MINUTES = (30,)
    simulation = equal.run_simulations(policy, n_walks=N_WALKS, seed=SEED)
    simulation_counts = simulation.groupby("group").size().to_dict()
    if simulation_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise RuntimeError(f"Complete simulation cohort drifted: {simulation_counts}")

    common = previous.merge(
        simulation,
        on=["window_minutes", "nickname", "group", "animal_id"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    parity_columns = (
        "real_half_cover_eff",
        "n_endnodes_target",
        "four_own_half_cover_eff",
        "depth_own_half_cover_eff",
    )
    parity_rows = []
    for column in parity_columns:
        old = pd.to_numeric(common[f"{column}_old"], errors="coerce")
        new = pd.to_numeric(common[f"{column}_new"], errors="coerce")
        difference = (old - new).abs()
        maximum = float(difference.dropna().max()) if difference.notna().any() else 0.0
        parity_rows.append({"column": column, "max_abs_difference": maximum})
        if maximum > 1e-12 or not old.isna().equals(new.isna()):
            raise RuntimeError(f"Existing simulation rows drifted for {column}: {maximum}")
    pd.DataFrame(parity_rows).to_csv(
        CANDIDATE / "cohort_audit" / "round24_existing_simulation_parity.csv",
        index=False,
    )
    simulation.to_csv(previous_path, index=False)

    calibration = equal.calibration_summary(simulation)
    keep = [
        "window_minutes",
        "model",
        "group",
        "group_label",
        "n_session_rows",
        "n_mice",
        "mean_error",
        "mae",
        "pearson_r",
        "slope",
        "real_mean",
        "simulated_mean",
        "n_zero_simulated_rows",
    ]
    calibration = calibration[keep]
    calibration_path = (
        CANDIDATE
        / "analysis_inputs"
        / "natural"
        / "equal_junction_calibration_summary.csv"
    )
    calibration.to_csv(calibration_path, index=False)

    stage = CANDIDATE / ".round25_render_stage"
    round11 = load_module(
        REPO / "scripts" / "render_ed02_slide3_refinement_round21.py",
        "ed02_full_roster_round21_renderer",
    )
    try:
        round11.render(CANDIDATE, stage, 600)
        for panel in "abcdefghi":
            for source in (stage / "panels").glob(f"ed02{panel}*"):
                shutil.copy2(source, CANDIDATE / "panels" / source.name)
            for source in (stage / "plot_ready").glob(f"panel_{panel}_*"):
                shutil.copy2(source, CANDIDATE / "plot_ready" / source.name)
            for source in (stage / "ppt_vector").glob(f"ed02{panel}*"):
                shutil.copy2(source, CANDIDATE / "ppt_vector" / source.name)
        shutil.copy2(stage / "CAPTION_NOTES.md", CANDIDATE / "CAPTION_NOTES.md")
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    counts_path = CANDIDATE / "cohort_audit" / "drawn_panel_cohort_counts.csv"
    if counts_path.is_file():
        counts = pd.read_csv(counts_path)
    else:
        rows: list[dict[str, Any]] = []

        def add_points(panel: str, filename: str) -> None:
            frame = pd.read_csv(CANDIDATE / "plot_ready" / filename)
            for group, sub in frame.groupby("group", sort=False):
                rows.append(
                    {
                        "panel": panel,
                        "group": group,
                        "n_sessions": len(sub),
                        "n_mice": sub["animal_id"].astype(str).nunique(),
                        "contains_640day1": "640day1" in set(sub["nickname"]),
                    }
                )

        add_points("b", "panel_b_session_points.csv")
        add_points("c", "panel_c_session_points.csv")
        add_points("d", "panel_b_session_points.csv")
        e_groups = pd.read_csv(CANDIDATE / "plot_ready" / "panel_e_group_means.csv")
        for row in e_groups.itertuples(index=False):
            rows.append(
                {
                    "panel": "e",
                    "group": row.group,
                    "n_sessions": int(row.n_sessions),
                    "n_mice": int(row.n_mice),
                    "contains_640day1": row.group == "naive",
                }
            )
        add_points("f", "panel_f_session_points.csv")
        add_points("g", "panel_g_supported_session_rows.csv")
        h = pd.read_csv(CANDIDATE / "plot_ready" / "panel_h_four_vs_depth_gap.csv").iloc[0]
        rows.extend(
            [
                {
                    "panel": "h",
                    "group": "naive",
                    "n_sessions": int(h.n_naive_rows),
                    "n_mice": int(h.n_naive_mice),
                    "contains_640day1": False,
                },
                {
                    "panel": "h",
                    "group": "LTM",
                    "n_sessions": int(h.n_return_rows),
                    "n_mice": int(h.n_return_mice),
                    "contains_640day1": False,
                },
            ]
        )
        add_points("i", "panel_i_session_points.csv")
        counts = pd.DataFrame(rows)
    counts.to_csv(counts_path, index=False)

    for manifest_name in (
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
    ):
        update_manifest_hashes(CANDIDATE / manifest_name, CANDIDATE)

    source_manifest_path = CANDIDATE / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "status": "unpromoted_scientific_review_candidate",
            "source_record": f"docs/figure_version_records/{PARENT_NAME}",
            "review_round": 25,
            "frozen_renderer": {
                "path": "source_code/render_ed03_fixed_time_30min_full_story.py",
                "sha256": sha256(
                    CANDIDATE
                    / "source_code"
                    / "render_ed03_fixed_time_30min_full_story.py"
                ),
            },
            "round25_complete_second_day_pearson": {
                "restored_sessions": list(MISSING_SESSIONS),
                "calibration_counts": {"naive": 23, "LTM": 58, "experienced": 74},
                "manuscript_association": "Pearson r",
                "mean_bias_retained": True,
                "legacy_spearman_gate_removed_from_manuscript_facing_outputs": True,
                "changed_panels": ["f", "i"],
            },
        }
    )
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    exporter = load_module(
        REPO / "scripts" / "export_ed02_round11_vector_figure.py",
        "ed02_round25_exporter",
    )
    composite_root = CANDIDATE / "composite"
    composite = exporter.export(
        CANDIDATE,
        composite_root,
        composite_stem=COMPOSITE_STEM,
        review_round=25,
        manifest_name="COMPOSITE_MANIFEST.json",
    )

    f_points = pd.read_csv(CANDIDATE / "plot_ready" / "panel_f_session_points.csv")
    i_points = pd.read_csv(CANDIDATE / "plot_ready" / "panel_i_session_points.csv")
    f_summary = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_f_calibration_summary.csv"
    )
    i_summary = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_i_calibration_summary.csv"
    )
    caption = (CANDIDATE / "CAPTION_NOTES.md").read_text(encoding="utf-8")

    def direct_pearson(points: pd.DataFrame, column: str) -> dict[str, float]:
        return {
            group: float(
                stats.pearsonr(
                    sub["real_half_cover_eff"], sub[column]
                ).statistic
            )
            for group, sub in points.groupby("group")
        }

    direct = {
        "four_bias": direct_pearson(f_points, "four_own_half_cover_eff"),
        "depth_stratified": direct_pearson(i_points, "depth_own_half_cover_eff"),
    }
    saved = {
        "four_bias": f_summary.set_index("group")["pearson_r"].to_dict(),
        "depth_stratified": i_summary.set_index("group")["pearson_r"].to_dict(),
    }
    pearson_match = all(
        abs(direct[model][group] - saved[model][group]) < 1e-12
        for model in direct
        for group in direct[model]
    )
    b_points = pd.read_csv(CANDIDATE / "plot_ready" / "panel_b_session_points.csv")
    g_points = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_g_supported_session_rows.csv"
    )
    checks = {
        "three_scalar_rows_restored": set(added["nickname"]) == set(MISSING_SESSIONS),
        "complete_23_58_74_simulation": simulation_counts
        == {"LTM": 58, "experienced": 74, "naive": 23},
        "panel_f_complete_counts": f_points.groupby("group").size().to_dict()
        == {"LTM": 58, "experienced": 74, "naive": 23},
        "panel_i_complete_counts": i_points.groupby("group").size().to_dict()
        == {"LTM": 58, "experienced": 74, "naive": 23},
        "pearson_recomputed_from_points": pearson_match,
        "no_spearman_or_gate_columns": not any(
            "spearman" in column.lower() or "gate" in column.lower()
            for column in [*f_summary.columns, *i_summary.columns]
        ),
        "caption_uses_pearson_only": "Pearson" in caption
        and "Spearman" not in caption
        and "rho=" not in caption
        and "gate=" not in caption,
        "existing_simulation_rows_exact": all(
            row["max_abs_difference"] <= 1e-12 for row in parity_rows
        ),
        "panel_b_full_pooled_return_counts": b_points.groupby("group").size().to_dict()
        == {"LTM": 58, "naive": 23},
        "panel_g_full_depth_supported_counts": g_points.groupby("group").size().to_dict()
        == {"LTM": 58, "experienced": 74, "naive": 22},
        "nine_panels_four_formats": len(list((CANDIDATE / "panels").glob("*")))
        == 36,
        "composite_vector_only": composite["vector_audit"]["embedded_raster_images"]
        == 0,
        "composite_600dpi_square": composite["png_audit"]["size"] == [4500, 4500]
        and composite["tiff_audit"]["size"] == [4500, 4500],
    }
    status = "pass" if all(checks.values()) else "fail"
    qa = {
        "status": status,
        "checks": checks,
        "pearson": direct,
        "mean_bias": {
            "four_bias": f_summary.set_index("group")["mean_error"].to_dict(),
            "depth_stratified": i_summary.set_index("group")["mean_error"].to_dict(),
        },
        "composite": composite,
    }
    (CANDIDATE / "QA.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    (CANDIDATE / "QA.md").write_text(
        "# ED2 complete second-day Pearson candidate QA\n\n"
        + f"- Overall: `{status.upper()}`\n"
        + "\n".join(
            f"- {'PASS' if passed else 'FAIL'}: {name}"
            for name, passed in checks.items()
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "status": status,
        "parent": PARENT_NAME,
        "restored_sessions": list(MISSING_SESSIONS),
        "scalar_source_sha256": sha256(SOURCE_SCALARS),
        "n_walks": N_WALKS,
        "seed": SEED,
        "pearson": direct,
        "mean_bias": qa["mean_bias"],
    }
    (CANDIDATE / "DATA_BUILD_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (CANDIDATE / "REPORT.md").write_text(
        "# ED2 complete second-day Pearson candidate\n\n"
        f"Status: **{status.upper()}**.\n\n"
        "The accepted 12-month Day 2 sessions `837-12day2`, `838-12day2`, and "
        "`839-12day2` now have maintained whole-session half-coverage targets. "
        "Panels f/i therefore use 23 Naive, 58 Return, and 74 Second-day sessions. "
        "Pearson r and mean bias are reported descriptively; the historical Spearman "
        "threshold and binary pass/fail language are not used. Only panels f/i and "
        "their source tables change relative to Round 24.\n",
        encoding="utf-8",
    )
    write_analysis_manifest()
    if status != "pass":
        raise RuntimeError(f"Candidate QA failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
