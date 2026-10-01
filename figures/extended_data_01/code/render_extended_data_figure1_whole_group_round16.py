"""Render the ED1 round-16 whole-group successor with the accepted page style."""

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
OUTPUT = REPO / "docs/ed01_whole_group_promotion_2026-08-15/round16_candidate"
SOURCE_DIR = OUTPUT / "source_data"
SOURCE_TABLE = SOURCE_DIR / "ed01_tile_accumulation_whole_group_common_support_current153.csv"
ROUND = 16

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
    spec = importlib.util.spec_from_file_location("ed01_round16_base", BASE_RENDERER)
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


def main() -> int:
    table = build_source_table()
    renderer = load_renderer()
    renderer.OUTPUT = OUTPUT
    renderer.ROUND = ROUND
    renderer.GENERATED = "2026-08-15"
    renderer.STEM = OUTPUT / f"extended_data_figure1_source_rerender_round{ROUND}"
    renderer.SOURCE_AUDIT = OUTPUT / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv"
    renderer.LAYOUT_AUDIT = OUTPUT / f"LAYOUT_AUDIT_ROUND{ROUND}.csv"
    renderer.CAPTION_NOTES = OUTPUT / f"CAPTION_NOTES_ROUND{ROUND}.md"
    renderer.RENDER_VALIDATION = OUTPUT / f"RENDER_VALIDATION_ROUND{ROUND}.json"
    renderer.MANIFEST = OUTPUT / f"MANIFEST_ROUND{ROUND}.json"
    renderer.TILE_CURVES = SOURCE_TABLE
    renderer.SOURCE_FILES["d-e"] = (
        SOURCE_TABLE,
        "whole-group current153 maze-transition curves on fully observed common support",
    )
    status = renderer.main()

    policy = {
        "schema": "ed01-round16-whole-group-source-policy/v1",
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
        "style_authority": "accepted ED1 round-15 renderer and geometry",
        "display_text_changes": [
            "Body mass speed",
            "Within round coverage",
            "Coverage round efficiency",
        ],
    }
    (OUTPUT / "ROUND16_SOURCE_POLICY.json").write_text(
        json.dumps(policy, indent=2) + "\n",
        encoding="utf-8",
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
