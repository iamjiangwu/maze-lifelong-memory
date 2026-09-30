"""Build the full pooled-return ED2 candidate with panel i on panel g's cohort.

The preceding full-roster simulation rows are frozen.  No walks are rerun.  Panel
i is restricted to sessions with at least 10 P_BS events in every displayed
depth stratum, exactly matching panel g.  Only panel i, its two plot-ready
tables, the calibration-summary analysis input, and the assembled composite
are allowed to change.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "ed02_1616_restored_pearson74_candidate_2026-08-20"
PARENT = REPO / "docs" / "figure_version_candidates" / PARENT_NAME
CANDIDATE = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed02_1616_restored_depth_common_i_candidate_2026-08-20"
)
COMPOSITE_STEM = "ed02_1616_restored_depth_common_i_candidate"
GROUP_ORDER = ("naive", "LTM", "experienced")
GROUP_LABEL = {"naive": "Naive", "LTM": "Return", "experienced": "Second day"}
DEPTH_COLUMNS = ("n_Lo_shallow", "n_Lo_mid", "n_Lo_deep")
MIN_DEPTH_EVENTS = 10
EXPECTED_F = {"naive": 23, "LTM": 58, "experienced": 74}
EXPECTED_I = {"naive": 22, "LTM": 58, "experienced": 74}


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def update_manifest_hashes(path: Path, root: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
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


def depth_support(frame: pd.DataFrame) -> pd.Series:
    supported = pd.Series(True, index=frame.index)
    for column in DEPTH_COLUMNS:
        supported &= pd.to_numeric(frame[column], errors="coerce").ge(
            MIN_DEPTH_EVENTS
        )
    return supported


def calibration_row(frame: pd.DataFrame, group: str) -> dict[str, Any]:
    sub = frame.loc[frame["group"].eq(group)].copy()
    real = pd.to_numeric(sub["real_half_cover_eff"], errors="coerce").to_numpy()
    simulated = pd.to_numeric(
        sub["depth_own_half_cover_eff"], errors="coerce"
    ).to_numpy()
    finite = np.isfinite(real) & np.isfinite(simulated)
    real = real[finite]
    simulated = simulated[finite]
    error = simulated - real
    return {
        "window_minutes": 30,
        "model": "depth_stratified",
        "group": group,
        "group_label": GROUP_LABEL[group],
        "n_session_rows": len(real),
        "n_mice": sub.loc[finite, "animal_id"].astype(str).nunique(),
        "mean_error": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "pearson_r": float(stats.pearsonr(real, simulated).statistic),
        "slope": float(np.polyfit(real, simulated, 1)[0]),
        "real_mean": float(np.mean(real)),
        "simulated_mean": float(np.mean(simulated)),
        "n_zero_simulated_rows": int(np.count_nonzero(simulated == 0)),
    }


def _rewrite_caption_notes() -> None:
    text = (PARENT / "CAPTION_NOTES.md").read_text(encoding="utf-8")
    summary = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_i_calibration_summary.csv"
    ).set_index("group")
    stats_text = "; ".join(
        f"{GROUP_LABEL[group]}: r={float(summary.at[group, 'pearson_r']):.2f}, "
        f"bias={float(summary.at[group, 'mean_error']):+.3f}"
        for group in GROUP_ORDER
    )
    text = text.replace(
        "# Extended Data Figure 2 Round 25 caption notes",
        "# Extended Data Figure 2 Round 26 caption notes",
    )
    text = re.sub(
        r"\*\*i,\*\*.*?(?=\n\nThe behavioral clock)",
        "**i,** Observed versus depth-varying-P_BS-model half-coverage efficiency "
        "on the same all-depth-strata support as g (Naive day: 22 sessions, 22 mice; "
        "Return day: 58 sessions, 18 mice; Second day: 74 sessions, 24 mice). Pearson "
        f"r and mean bias are descriptive: {stats_text}.",
        text,
        flags=re.S,
    )
    text += (
        "\nRound-26 scientific note: panel i now uses the exact all-depth-strata "
        "support used by panel g. `640day1` is excluded from panel i because it has "
        "only 2 shallow-depth P_BS events, below the prespecified 10-event threshold. "
        "The accepted Round-25 simulation rows were filtered without rerunning walks; "
        "panels a-h are unchanged.\n"
    )
    (CANDIDATE / "CAPTION_NOTES.md").write_text(text, encoding="utf-8")


def build() -> dict[str, Any]:
    if CANDIDATE.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {CANDIDATE}")
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    shutil.copytree(PARENT, CANDIDATE)
    for stale in (
        "RECORD_HASHES.csv",
        "RECORD_VALIDATION.json",
        "PROMOTION_COMPLETE.txt",
    ):
        (CANDIDATE / stale).unlink(missing_ok=True)

    adapter = REPO / "scripts" / "render_ed02_round26_depth_common_i_adapter.py"
    shutil.copy2(
        adapter,
        CANDIDATE / "source_code" / "render_ed03_fixed_time_30min_full_story.py",
    )
    shutil.copy2(adapter, CANDIDATE / "source_code" / adapter.name)
    shutil.copy2(Path(__file__), CANDIDATE / "source_code" / Path(__file__).name)

    simulation_path = (
        CANDIDATE
        / "analysis_inputs"
        / "natural"
        / "equal_junction_simulation_session_rows.csv"
    )
    simulation = pd.read_csv(simulation_path, dtype={"animal_id": str})
    if simulation.groupby("group").size().to_dict() != EXPECTED_F:
        raise RuntimeError("Full-roster natural simulation cohort drifted")
    strict = simulation.loc[depth_support(simulation)].copy()
    strict_counts = strict.groupby("group").size().to_dict()
    if strict_counts != EXPECTED_I:
        raise RuntimeError(f"Unexpected depth-supported cohort: {strict_counts}")

    panel_g = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_g_supported_session_rows.csv",
        dtype={"animal_id": str},
    )
    sim_keys = set(zip(strict["group"], strict["nickname"]))
    g_keys = set(zip(panel_g["group"], panel_g["nickname"]))
    if sim_keys != g_keys:
        raise RuntimeError(
            f"Panel-i and panel-g support differ: i_only={sorted(sim_keys-g_keys)}, "
            f"g_only={sorted(g_keys-sim_keys)}"
        )
    excluded = simulation.loc[~depth_support(simulation)].copy()
    if excluded["nickname"].astype(str).tolist() != ["640day1"]:
        raise RuntimeError(f"Unexpected panel-i exclusions: {excluded['nickname'].tolist()}")

    audit_columns = [
        "window_minutes",
        "nickname",
        "group",
        "animal_id",
        *DEPTH_COLUMNS,
        "real_half_cover_eff",
        "depth_own_half_cover_eff",
    ]
    strict[audit_columns].to_csv(
        CANDIDATE / "cohort_audit" / "panel_i_depth_supported_sessions.csv",
        index=False,
    )
    excluded[audit_columns].assign(
        exclusion_reason="at least one depth stratum has fewer than 10 P_BS events"
    ).to_csv(
        CANDIDATE / "cohort_audit" / "panel_i_excluded_sessions.csv",
        index=False,
    )

    calibration_path = (
        CANDIDATE
        / "analysis_inputs"
        / "natural"
        / "equal_junction_calibration_summary.csv"
    )
    calibration = pd.read_csv(calibration_path)
    four = calibration.loc[calibration["model"].eq("four_bias")].copy()
    depth = pd.DataFrame([calibration_row(strict, group) for group in GROUP_ORDER])
    calibration = pd.concat([four, depth], ignore_index=True)
    calibration.to_csv(calibration_path, index=False)

    stage = CANDIDATE / ".round26_render_stage"
    round11 = load_module(
        REPO / "scripts" / "render_ed02_slide3_refinement_round21.py",
        "ed02_full_roster_round21_renderer",
    )
    try:
        round11.render(CANDIDATE, stage, 600)
        for source in (stage / "panels").glob("ed02i*"):
            shutil.copy2(source, CANDIDATE / "panels" / source.name)
        for source in (stage / "plot_ready").glob("panel_i_*"):
            shutil.copy2(source, CANDIDATE / "plot_ready" / source.name)
        for source in (stage / "ppt_vector").glob("ed02i*"):
            shutil.copy2(source, CANDIDATE / "ppt_vector" / source.name)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    _rewrite_caption_notes()

    counts_path = CANDIDATE / "cohort_audit" / "drawn_panel_cohort_counts.csv"
    counts = pd.read_csv(counts_path)
    for group in GROUP_ORDER:
        sub = strict.loc[strict["group"].eq(group)]
        mask = counts["panel"].eq("i") & counts["group"].eq(group)
        counts.loc[mask, "n_sessions"] = len(sub)
        counts.loc[mask, "n_mice"] = sub["animal_id"].astype(str).nunique()
        counts.loc[mask, "contains_640day1"] = "640day1" in set(sub["nickname"])
    counts.to_csv(counts_path, index=False)

    for manifest_name in (
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
    ):
        update_manifest_hashes(CANDIDATE / manifest_name, CANDIDATE)
    write_analysis_manifest()

    source_manifest_path = CANDIDATE / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "status": "unpromoted_scientific_review_candidate",
            "source_candidate": f"docs/figure_version_candidates/{PARENT_NAME}",
            "review_round": 29,
            "frozen_renderer": {
                "path": "source_code/render_ed03_fixed_time_30min_full_story.py",
                "sha256": sha256(
                    CANDIDATE
                    / "source_code"
                    / "render_ed03_fixed_time_30min_full_story.py"
                ),
            },
            "round26_depth_common_panel_i": {
                "support_rule": "n_Lo_shallow, n_Lo_mid and n_Lo_deep each >= 10",
                "panel_i_counts": EXPECTED_I,
                "excluded_session": "640day1",
                "simulation_policy": "filter preceding full-roster rows; no resimulation",
                "changed_panels": ["i"],
            },
        }
    )
    write_json(source_manifest_path, source_manifest)

    exporter = load_module(
        REPO / "scripts" / "export_ed02_round11_vector_figure.py",
        "ed02_round26_exporter",
    )
    composite = exporter.export(
        CANDIDATE,
        CANDIDATE / "composite",
        composite_stem=COMPOSITE_STEM,
        review_round=29,
        manifest_name="COMPOSITE_MANIFEST.json",
    )

    f_points = pd.read_csv(CANDIDATE / "plot_ready" / "panel_f_session_points.csv")
    i_points = pd.read_csv(CANDIDATE / "plot_ready" / "panel_i_session_points.csv")
    i_summary = pd.read_csv(
        CANDIDATE / "plot_ready" / "panel_i_calibration_summary.csv"
    )
    i_lookup = i_summary.set_index("group")
    direct_r = {
        group: float(
            stats.pearsonr(
                sub["real_half_cover_eff"], sub["depth_own_half_cover_eff"]
            ).statistic
        )
        for group, sub in i_points.groupby("group")
    }
    direct_bias = {
        group: float(
            (sub["depth_own_half_cover_eff"] - sub["real_half_cover_eff"]).mean()
        )
        for group, sub in i_points.groupby("group")
    }
    non_target_panel_equal = all(
        sha256(path) == sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob("*")
        if path.is_file() and not path.name.startswith("ed02i_")
    )
    panel_i_changed = all(
        sha256(path) != sha256(PARENT / "panels" / path.name)
        for path in (CANDIDATE / "panels").glob("ed02i_*")
    )
    non_target_tables_equal = all(
        sha256(path) == sha256(PARENT / "plot_ready" / path.name)
        for path in (CANDIDATE / "plot_ready").glob("panel_*.csv")
        if not path.name.startswith("panel_i_")
    )
    checks = {
        "accepted_simulation_rows_frozen": sha256(simulation_path)
        == sha256(
            PARENT
            / "analysis_inputs"
            / "natural"
            / "equal_junction_simulation_session_rows.csv"
        ),
        "panel_f_counts_unchanged": f_points.groupby("group").size().to_dict()
        == EXPECTED_F,
        "panel_i_counts_match_g": i_points.groupby("group").size().to_dict()
        == EXPECTED_I,
        "panel_i_exact_session_set_matches_g": set(
            zip(i_points["group"], i_points["nickname"])
        )
        == g_keys,
        "only_640day1_excluded": excluded["nickname"].tolist() == ["640day1"],
        "pearson_and_bias_recomputed": all(
            abs(float(i_lookup.loc[group, "pearson_r"]) - direct_r[group]) < 1e-12
            and abs(float(i_lookup.loc[group, "mean_error"]) - direct_bias[group]) < 1e-12
            for group in GROUP_ORDER
        ),
        "panel_i_has_no_spearman_or_gate_fields": not any(
            token in column.lower()
            for column in i_summary.columns
            for token in ("spearman", "rho", "gate", "pass")
        ),
        "panels_a_to_h_byte_identical": non_target_panel_equal,
        "panel_i_all_formats_changed": panel_i_changed,
        "non_target_plot_ready_tables_identical": non_target_tables_equal,
        "nine_panels_four_formats": all(
            len(list((CANDIDATE / "panels").glob(f"*.{suffix}"))) == 9
            for suffix in ("svg", "pdf", "png", "tiff")
        ),
        "sixteen_plot_ready_tables": len(
            list((CANDIDATE / "plot_ready").glob("panel_*.csv"))
        )
        == 16,
        "composite_vector_only": composite["vector_audit"]["embedded_raster_images"]
        == 0,
        "composite_600dpi_square": composite["png_audit"]["size"] == [4500, 4500]
        and composite["tiff_audit"]["size"] == [4500, 4500],
    }
    status = "pass" if all(checks.values()) else "fail"
    qa = {
        "status": status,
        "checks": checks,
        "panel_i_pearson": direct_r,
        "panel_i_mean_bias": direct_bias,
        "composite": composite,
    }
    write_json(CANDIDATE / "QA.json", qa)
    (CANDIDATE / "QA.md").write_text(
        "# ED2 Round 26 common-depth panel-i candidate QA\n\n"
        f"- Overall: `{status.upper()}`\n"
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
        "support_rule": "each displayed depth stratum has at least 10 P_BS events",
        "panel_i_counts": EXPECTED_I,
        "excluded_sessions": excluded["nickname"].tolist(),
        "simulation_rerun": False,
        "panel_i_pearson": direct_r,
        "panel_i_mean_bias": direct_bias,
    }
    write_json(CANDIDATE / "DATA_BUILD_RESULT.json", result)
    (CANDIDATE / "REPORT.md").write_text(
        "# ED2 Round 26 common-depth panel-i candidate\n\n"
        f"Status: **{status.upper()}**.\n\n"
        "Panel i now uses exactly the panel-g all-depth-strata cohort: 22 Naive, "
        "58 Return and 74 Second-day sessions. `640day1` is the sole removed row "
        "because its shallow P_BS support is 2 events, below 10. The accepted "
        "Round-25 simulation rows were filtered without rerunning walks. Panels "
        "a-h and all non-panel-i plot-ready tables are byte-identical to the preceding full-roster candidate.\n",
        encoding="utf-8",
    )
    if status != "pass":
        raise RuntimeError(f"Candidate QA failed: {checks}")
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
