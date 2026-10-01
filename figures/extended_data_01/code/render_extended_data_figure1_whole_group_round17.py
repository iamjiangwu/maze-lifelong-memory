"""Render the ED1 round-17 whole-group and first-30-min successor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO / "scripts/render_extended_data_figure1_round4.py"
AUDIT_DIR = REPO / "docs/ed01_panel_d_whole_group_2026-08-15"
AUDIT_CURVES = AUDIT_DIR / "plot_ready_group_curves.csv"
PANEL_H_DIR = REPO / "docs/ed01_panel_h_window_sensitivity_2026-08-15"
PANEL_H_ROWS = PANEL_H_DIR / "speed_by_session_window.csv"
PANEL_H_REFERENCE = PANEL_H_DIR / "group_summary.csv"
OUTPUT = REPO / "docs/ed01_whole_group_promotion_2026-08-15/round17_candidate"
SOURCE_DIR = OUTPUT / "source_data"
SOURCE_TABLE = SOURCE_DIR / "ed01_tile_accumulation_whole_group_common_support_current153.csv"
SPEED_TABLE = SOURCE_DIR / "ed01_panel_h_first30_session_rows.csv"
SPEED_REFERENCE = SOURCE_DIR / "ed01_panel_h_first30_group_summary.csv"
ROUND = 17

EXPECTED_COUNTS = {
    "naive_0mo": 23,
    "return_3mo": 17,
    "return_6mo": 17,
    "return_9mo": 12,
    "return_12mo": 10,
    "experienced_day2": 74,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_renderer():
    spec = importlib.util.spec_from_file_location("ed01_round17_base", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_source_table() -> pd.DataFrame:
    curves = pd.read_csv(AUDIT_CURVES)
    selected = curves.loc[
        curves["estimator"].eq("whole_group_observed_common_support")
    ].copy()
    if set(selected["event_type"]) != {"reward", "long_path_reward"}:
        raise RuntimeError("Whole-group source does not contain both ED1 event families")
    counts = (
        selected.groupby("month_group_key", observed=True)["n_sessions_averaged"]
        .first()
        .astype(int)
        .to_dict()
    )
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected whole-group session counts: {counts}")
    if selected["month_group_key"].nunique() != 6 or selected["axis_value_tiles"].max() != 5200:
        raise RuntimeError("Whole-group source must contain six groups through 5,200 transitions")
    if not selected["n_sessions_observed_at_x"].eq(selected["n_sessions_averaged"]).all():
        raise RuntimeError("The selected common-support range is not fully observed")

    selected["n_active_sessions"] = selected["n_sessions_observed_at_x"].astype(int)
    selected["initial_n_sessions"] = selected["n_sessions_averaged"].astype(int)
    selected["n_averaged_sessions"] = selected["n_sessions_averaged"].astype(int)
    selected["minimum_coverage_n_sessions"] = selected["n_sessions_averaged"].astype(int)
    selected["coverage_fraction"] = 1.0
    selected["global_coverage_window_tiles"] = 5200.0
    columns = [
        "event_type",
        "month_group_key",
        "month_group_label",
        "month_group_order",
        "color",
        "axis_value_tiles",
        "cumulative_mean",
        "cumulative_sem",
        "n_active_sessions",
        "initial_n_sessions",
        "n_averaged_sessions",
        "minimum_coverage_n_sessions",
        "coverage_fraction",
        "global_coverage_window_tiles",
    ]
    table = selected.loc[:, columns].sort_values(
        ["event_type", "month_group_order", "axis_value_tiles"],
        kind="stable",
    )
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SOURCE_TABLE, index=False)
    for name in (
        "source_session_inclusion_audit.csv",
        "dropped_sessions.csv",
        "group_support_summary.csv",
        "selection_diagnostic.csv",
        "estimator_endpoint_comparison.csv",
        "monotonicity_audit.csv",
        "QA_REPORT.json",
        "REPORT.md",
    ):
        shutil.copy2(AUDIT_DIR / name, SOURCE_DIR / name)
    return table


def build_speed_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(PANEL_H_ROWS, dtype={"animal_id": str}, low_memory=False)
    rows = rows.loc[
        rows["window_id"].eq("first_30_min")
        & rows["analysis_mode"].isin(["explore", "reward"])
        & rows["day"].isin(["day1", "day2"])
    ].copy()
    if len(rows) != 240 or rows["session_id"].nunique() != 120:
        raise RuntimeError(
            f"Panel-h first-30-min source must retain 240 mode rows from 120 sessions; "
            f"found {len(rows)} rows from {rows['session_id'].nunique()} sessions"
        )
    reference = pd.read_csv(PANEL_H_REFERENCE)
    reference = reference.loc[
        reference["summary_variant"].eq("exact_current_mode_rule")
        & reference["window_id"].eq("first_30_min")
        & reference["analysis_mode"].isin(["explore", "reward"])
        & reference["day"].isin(["day1", "day2"])
    ].copy()
    if len(reference) != 20:
        raise RuntimeError(f"Panel-h first-30-min reference must contain 20 rows; found {len(reference)}")
    valid_rows = rows.loc[
        pd.to_numeric(rows["mode_time_distance_rate_cm_s"], errors="coerce").notna()
    ]
    counts = (
        valid_rows.groupby(["analysis_mode", "day", "month"], observed=True)["session_id"]
        .nunique()
        .to_dict()
    )
    reference_counts = {
        (row.analysis_mode, row.day, int(row.month)): int(row.n_sessions)
        for row in reference.itertuples(index=False)
    }
    if counts != reference_counts:
        raise RuntimeError(f"Panel-h finite session counts do not match the reference: {counts}")
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SPEED_TABLE, index=False)
    reference.to_csv(SPEED_REFERENCE, index=False)
    panel_h_audit = SOURCE_DIR / "panel_h_first30_audit"
    panel_h_audit.mkdir(parents=True, exist_ok=True)
    for name in (
        "session_registry.csv",
        "session_source_qc.csv",
        "accepted_first60_parity.csv",
        "current_raw_first60_drift.csv",
        "VALIDATION.json",
        "MANIFEST.json",
        "REPORT.md",
    ):
        shutil.copy2(PANEL_H_DIR / name, panel_h_audit / name)
    return rows, reference


def main() -> int:
    table = build_source_table()
    speed_rows, speed_reference = build_speed_tables()
    renderer = load_renderer()
    renderer.OUTPUT = OUTPUT
    renderer.ROUND = ROUND
    renderer.GENERATED = "2026-08-15"
    renderer.STEM = OUTPUT / f"extended_data_figure1_source_rerender_round{ROUND}"
    renderer.SOURCE_AUDIT = OUTPUT / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv"
    renderer.LAYOUT_AUDIT = OUTPUT / f"LAYOUT_AUDIT_ROUND{ROUND}.csv"
    renderer.AXIS_SCALE_AUDIT = OUTPUT / f"AXIS_SCALE_AUDIT_ROUND{ROUND}.csv"
    renderer.CAPTION_NOTES = OUTPUT / f"CAPTION_NOTES_ROUND{ROUND}.md"
    renderer.RENDER_VALIDATION = OUTPUT / f"RENDER_VALIDATION_ROUND{ROUND}.json"
    renderer.MANIFEST = OUTPUT / f"MANIFEST_ROUND{ROUND}.json"
    renderer.TILE_CURVES = SOURCE_TABLE
    renderer.SPEED_ROWS = SPEED_TABLE
    renderer.SPEED_REFERENCE = SPEED_REFERENCE
    renderer.SPEED_WINDOW_ID = "first_30_min"
    renderer.SPEED_SUMMARY_VARIANT = "exact_current_mode_rule"
    renderer.SOURCE_FILES["d-e"] = (
        SOURCE_TABLE,
        "whole-group current153 maze-transition curves on fully observed common support",
    )
    renderer.SOURCE_FILES["h-rows"] = (
        SPEED_TABLE,
        "first-30-min session-level body-mass speeds from the accepted 120-session cohort",
    )
    renderer.SOURCE_FILES["h-reference"] = (
        SPEED_REFERENCE,
        "exact-current-rule first-30-min group summary parity target",
    )
    status = renderer.main()

    policy = {
        "schema": "ed01-round17-whole-group-first30-source-policy/v1",
        "status": "PASS",
        "estimator": "whole_group_observed_common_support",
        "representation_space": "corrected nose-derived transitions on the 15x15 maze grid",
        "event_types": sorted(table["event_type"].unique().tolist()),
        "session_counts": EXPECTED_COUNTS,
        "total_sessions": sum(EXPECTED_COUNTS.values()),
        "common_observed_range_tiles": [0, 5200],
        "terminal_carry_forward_used": False,
        "changing_session_composition_used": False,
        "source_table": SOURCE_TABLE.relative_to(REPO).as_posix(),
        "source_table_sha256": sha256(SOURCE_TABLE),
        "audit_group_curves_sha256": sha256(AUDIT_CURVES),
        "panel_h": {
            "window": "first 30 min of cumulative in-maze time",
            "statistical_unit": "session",
            "representation": "continuous body-mass position within trajectory modes",
            "estimator": "total retained adjacent-frame distance divided by eligible mode time",
            "summary_variant": "exact_current_mode_rule",
            "registry_sessions": int(speed_rows["session_id"].nunique()),
            "mode_rows": len(speed_rows),
            "reference_rows": len(speed_reference),
            "shared_y_limits_cm_s": [0, 14],
            "source_table": SPEED_TABLE.relative_to(REPO).as_posix(),
            "source_table_sha256": sha256(SPEED_TABLE),
            "reference_table": SPEED_REFERENCE.relative_to(REPO).as_posix(),
            "reference_table_sha256": sha256(SPEED_REFERENCE),
        },
        "style_authority": "accepted ED1 round-15 renderer and geometry",
        "display_text_changes": [
            "Body mass speed",
            "Shared 0-14 cm/s y scale for exploration and reward directed modes",
            "Within round coverage",
            "Coverage round efficiency",
        ],
    }
    (OUTPUT / "ROUND17_SOURCE_POLICY.json").write_text(
        json.dumps(policy, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
