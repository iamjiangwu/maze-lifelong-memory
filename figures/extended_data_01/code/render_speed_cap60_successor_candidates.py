"""Render ED1h and ED10a successors with a 60 cm/s tracking-artifact cap.

The script is intentionally narrow: it reuses the current version-record inputs for
every non-target role, changes only the validated speed tables, and writes candidates
outside ``docs/figure_version_records``.  Promotion is handled separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "docs/speed_cap_60_vs_80_figure_panel_audit_2026-08-19"
ED1_PARENT = (
    REPO
    / "docs/figure_version_records"
    / "extended_data_fig01_2026-08-17_round20_canonical_animal_release"
)
ED10_PARENT = (
    REPO
    / "docs/figure_version_records"
    / "extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order"
)
ED1_OUT = REPO / "docs/figure_version_candidates/ed01_speed_cap60_2026-08-19"
ED10_OUT = REPO / "docs/figure_version_candidates/ed10_speed_cap60_2026-08-19"
ED1_STEM = ED1_OUT / "ed01_speed_cap60_full_page_round21"
ED10_ROUND = "round12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {path}")
    path.mkdir(parents=True)


def image_difference(parent_path: Path, candidate_path: Path) -> dict[str, Any]:
    parent = np.asarray(Image.open(parent_path).convert("RGB"), dtype=np.int16)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.int16)
    if parent.shape != candidate.shape:
        raise RuntimeError(f"Raster shape mismatch: {parent.shape} vs {candidate.shape}")
    changed = np.any(parent != candidate, axis=2)
    ys, xs = np.where(changed)
    bbox = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return {
        "shape": list(parent.shape),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "difference_bbox_xyxy": bbox,
        "max_absolute_channel_delta": int(np.abs(parent - candidate).max()),
    }


def build_ed1_speed_sources() -> tuple[Path, Path]:
    source_dir = ED1_OUT / "source_data"
    source_dir.mkdir()
    rows = pd.read_csv(AUDIT / "ed01_h_session_rows_caps.csv", dtype={"animal_id": str})
    rows = rows.loc[
        np.isclose(pd.to_numeric(rows["max_segment_speed_cm_s"]), 60.0)
        & rows["window_id"].eq("first_30_min")
        & rows["analysis_mode"].isin(["explore", "reward"])
        & rows["day"].isin(["day1", "day2"])
    ].copy()
    if len(rows) != 306 or rows["session_id"].nunique() != 153:
        raise RuntimeError(
            f"ED1 cap-60 source must contain 306 rows / 153 sessions, found "
            f"{len(rows)} / {rows['session_id'].nunique()}"
        )
    summary = pd.read_csv(AUDIT / "ed01_h_group_summary_caps.csv")
    summary = summary.loc[
        np.isclose(pd.to_numeric(summary["max_segment_speed_cm_s"]), 60.0)
        & summary["window_id"].eq("first_30_min")
        & summary["analysis_mode"].isin(["explore", "reward"])
        & summary["day"].isin(["day1", "day2"])
    ].copy()
    if len(summary) != 20:
        raise RuntimeError(f"ED1 cap-60 group summary must contain 20 rows, found {len(summary)}")
    rows_path = source_dir / "ed01_panel_h_first30_session_rows_cap60.csv"
    summary_path = source_dir / "ed01_panel_h_first30_group_summary_cap60.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    return rows_path, summary_path


def render_ed1() -> dict[str, Any]:
    require_new_directory(ED1_OUT)
    rows_path, summary_path = build_ed1_speed_sources()
    renderer_path = REPO / "scripts/render_extended_data_figure1_round4.py"
    renderer = load_module(renderer_path, "ed01_speed_cap60_renderer")
    renderer.OUTPUT = ED1_OUT
    renderer.ROUND = 21
    renderer.GENERATED = "2026-08-19"
    renderer.STEM = ED1_STEM
    renderer.SOURCE_AUDIT = ED1_OUT / "SOURCE_DATA_AUDIT_ROUND21.csv"
    renderer.LAYOUT_AUDIT = ED1_OUT / "LAYOUT_AUDIT_ROUND21.csv"
    renderer.AXIS_SCALE_AUDIT = ED1_OUT / "AXIS_SCALE_AUDIT_ROUND21.csv"
    renderer.CAPTION_NOTES = ED1_OUT / "CAPTION_NOTES_ROUND21.md"
    renderer.RENDER_VALIDATION = ED1_OUT / "RENDER_VALIDATION_ROUND21.json"
    renderer.MANIFEST = ED1_OUT / "MANIFEST_ROUND21.json"

    plot_ready = ED1_PARENT / "plot_ready"
    renderer.ACCUMULATION = plot_ready / "ed01_bc_group_specific_80_session_curves.csv.gz"
    renderer.TILE_CURVES = plot_ready / "ed01_de_group_specific_80_group_curves.csv"
    renderer.ENDPOINT_ROWS = plot_ready / "ed01_endpoint_axis_session_rows_current153.csv"
    renderer.ENDPOINT_STATS = plot_ready / "ed01_endpoint_axis_cluster_stats_current153.csv"
    renderer.TILE_TIME_ROWS = plot_ready / "ed02_tile_time_loglog_rows_current153.csv"
    renderer.TILE_TIME_STATS = plot_ready / "ed02_tile_time_loglog_stats_current153.csv"
    renderer.SPEED_ROWS = rows_path
    renderer.SPEED_REFERENCE = summary_path
    renderer.SPEED_WINDOW_ID = "first_30_min"
    renderer.SPEED_SUMMARY_VARIANT = "exact_current_mode_rule"
    renderer.EFFICIENCY_TIME = plot_ready / "ed01_j_group_specific_80_time_session_bins.csv"
    renderer.EFFICIENCY_TILE = plot_ready / "ed01_k_group_specific_80_tile_session_bins.csv"
    renderer.TIME_CURVE_CAPTION_NOTE = (
        "Mean +/- SEM across all 153 sessions. Each group stops at its last "
        "10-s grid point with at least 80% recording support; cumulative terminal "
        "counts are carried forward after observation ends."
    )
    renderer.TILE_CURVE_CAPTION_NOTE = renderer.TIME_CURVE_CAPTION_NOTE
    renderer.EFFICIENCY_CAPTION_NOTE = (
        "Session-first mean +/- SEM among cycle-eligible sessions contributing to "
        "each bin. Each group stops at its last bin with at least 80% recording "
        "opportunity; efficiency values are not carried forward."
    )
    for key, path, role in (
        ("b-c", renderer.ACCUMULATION, "current ED1 group-specific 80% time curves"),
        ("d-e", renderer.TILE_CURVES, "current ED1 group-specific 80% transition curves"),
        ("f-rows", renderer.ENDPOINT_ROWS, "current ED1 endpoint rows"),
        ("f-stats", renderer.ENDPOINT_STATS, "current ED1 endpoint statistics"),
        ("g-rows", renderer.TILE_TIME_ROWS, "current ED1 tile-time rows"),
        ("g-stats", renderer.TILE_TIME_STATS, "current ED1 tile-time statistics"),
        ("h-rows", rows_path, "cap-60 first-30-min session-level body-center speeds"),
        ("h-reference", summary_path, "cap-60 first-30-min grouped speed summary"),
        ("j", renderer.EFFICIENCY_TIME, "current ED1 time-bin efficiency rows"),
        ("k", renderer.EFFICIENCY_TILE, "current ED1 transition-bin efficiency rows"),
    ):
        renderer.SOURCE_FILES[key] = (path, role)

    # Preserve the accepted ED1h 0--14 cm/s display contract.  The cap-60 data
    # peak just below the renderer's automatic 12-cm/s boundary, but changing
    # the axis would be an unrelated display change.
    draw_speed_axes = renderer.draw_speed_axes

    def draw_speed_axes_fixed_14(subgrid, summary):
        axes = draw_speed_axes(subgrid, summary)
        for axis in axes:
            axis.set_ylim(0.0, 14.0)
            axis.set_yticks(np.arange(0.0, 14.1, 2.0))
        return axes

    renderer.draw_speed_axes = draw_speed_axes_fixed_14

    status = renderer.main()
    if status != 0:
        raise RuntimeError(f"ED1 renderer returned {status}")

    rendered_summary = pd.read_csv(ED1_OUT / "SPEED_SUMMARY_ROUND21.csv")
    keys = ["analysis_mode", "day", "month"]
    expected = pd.read_csv(summary_path).set_index(keys).sort_index()
    observed = rendered_summary.set_index(keys).sort_index()
    deltas = {
        column: float(
            np.max(np.abs(observed[column].to_numpy(float) - expected[column].to_numpy(float)))
        )
        for column in ("speed_mean_cm_s", "speed_sem_cm_s")
    }
    parent_png = ED1_PARENT / "ed01_version_record_600dpi.png"
    candidate_png = ED1_STEM.with_suffix(".png")
    difference = image_difference(parent_png, candidate_png)
    layout = pd.read_csv(ED1_OUT / "LAYOUT_AUDIT_ROUND21.csv").set_index("panel")
    h = layout.loc["h"]
    width, height = Image.open(candidate_png).size
    allowed = [
        int(float(h.x0_fraction) * width) - 120,
        int((1.0 - float(h.y1_fraction)) * height) - 120,
        int(float(h.x1_fraction) * width) + 120,
        int((1.0 - float(h.y0_fraction)) * height) + 180,
    ]
    bbox = difference["difference_bbox_xyxy"]
    confined = bbox is not None and all(
        (bbox[0] >= allowed[0], bbox[1] >= allowed[1], bbox[2] <= allowed[2], bbox[3] <= allowed[3])
    )
    validation = {
        "schema": "ed01-speed-cap60-candidate-validation/v1",
        "status": "PASS"
        if max(deltas.values()) <= 1e-12 and confined and difference["changed_pixels"] > 0
        else "FAIL",
        "target_panel": "h",
        "cap_cm_s": 60.0,
        "speed_summary_max_abs_delta": deltas,
        "parent_to_candidate_raster_difference": difference,
        "allowed_panel_h_bbox_xyxy": allowed,
        "changed_pixels_confined_to_panel_h": confined,
        "inputs": {
            "session_rows": {"path": rows_path.relative_to(REPO).as_posix(), "sha256": sha256(rows_path)},
            "group_summary": {"path": summary_path.relative_to(REPO).as_posix(), "sha256": sha256(summary_path)},
            "parent_record": ED1_PARENT.name,
        },
    }
    write_json(ED1_OUT / "CANDIDATE_VALIDATION.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"ED1 candidate validation failed: {validation}")
    return validation


def build_ed10_sources() -> tuple[Path, Path]:
    source_dir = ED10_OUT / "source_data_cap60"
    source_dir.mkdir()
    cap = pd.read_csv(AUDIT / "ed10_a_matrix_stats_caps.csv")
    cap = cap.loc[np.isclose(pd.to_numeric(cap["max_segment_speed_cm_s"]), 60.0)].copy()
    current = pd.read_csv(ED10_PARENT / "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv")
    cap = cap.loc[:, current.columns]
    if len(cap) != 18 or list(cap["block"].drop_duplicates()) != list(current["block"].drop_duplicates()):
        raise RuntimeError("ED10 cap-60 matrix does not preserve the current 6x3 row order")
    matrix_path = source_dir / "POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX_CAP60.csv"
    cap.to_csv(matrix_path, index=False)
    frame = pd.read_csv(AUDIT / "ed10_fixed30_analysis_frame_cap60.csv", dtype={"animal_id": str})
    if len(frame) != 30 or frame["pair_key"].nunique() != 30 or frame["animal_id"].nunique() != 14:
        raise RuntimeError("ED10 cap-60 analysis frame must contain 30 pairs from 14 mice")
    # The audit retains the historical cap-80 alias for parity diagnostics.
    # The renderer's sign audit expects this display alias to point at the
    # promoted whole-session endpoint.
    frame["log_reward_over_pooled"] = frame[
        "whole_session_reward_approach_speed_advantage_raw"
    ]
    frame_path = source_dir / "CURRENT_FIXED_NEURON_PAIR_ROWS_CAP60.csv"
    frame.to_csv(frame_path, index=False)
    return matrix_path, frame_path


def render_ed10() -> dict[str, Any]:
    require_new_directory(ED10_OUT)
    matrix_path, frame_path = build_ed10_sources()
    renderer_path = REPO / "scripts/render_ed10_slide12_powerpoint_matched.py"
    renderer = load_module(renderer_path, "ed10_speed_cap60_renderer")
    renderer.TABLE_A = matrix_path
    renderer.PAIR_AUDIT = frame_path
    main5 = (
        REPO
        / "docs/figure_version_records_archive_2026-08-09"
        / "main_figure5_2026-08-12_round6_historical_hierarchical_order"
    )
    renderer.MAIN_FIGURE5_RECORD = main5
    renderer.MAIN_FIGURE5_MANIFEST = main5 / "figure5_source_manifest.json"
    renderer.MAIN_FIGURE5_SVG = main5 / "figure5_version_record.svg"
    renderer.MAIN_FIGURE5_RENDERER = main5 / "source_code/preview_figure3_assembly.py"
    renderer.main(output_root=ED10_OUT, output_round=ED10_ROUND)

    output_dir = ED10_OUT / "assembly" / ED10_ROUND
    parent_individual = ED10_PARENT / "individual_panels"
    unchanged = {}
    for panel, name in (("b", "b_variation_partition.png"), ("c", "c_level_policy_associations.png")):
        unchanged[panel] = {
            "parent_sha256": sha256(parent_individual / name),
            "candidate_sha256": sha256(output_dir / "panels" / name),
            "byte_identical": sha256(parent_individual / name)
            == sha256(output_dir / "panels" / name),
        }

    raw = pd.read_csv(ED10_OUT / "source_data/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv")
    expected_strength = pd.read_csv(matrix_path)
    expected = expected_strength.copy()
    expected.loc[expected["block"].isin(["reward_location", "whole_decoder"]), "pooled_pearson_r"] *= -1
    observed_values = raw["pooled_pearson_r"].to_numpy(float)
    expected_values = expected["pooled_pearson_r"].to_numpy(float)
    value_delta = float(np.max(np.abs(observed_values - expected_values)))

    parent_png = ED10_PARENT / "ed10_version_record_600dpi.png"
    candidate_png = ED10_OUT / f"extended_data_figure10_source_rerender_{ED10_ROUND}.png"
    difference = image_difference(parent_png, candidate_png)
    bbox = difference["difference_bbox_xyxy"]
    allowed = [0, 0, 4500, 1840]
    confined = bbox is not None and all(
        (bbox[0] >= allowed[0], bbox[1] >= allowed[1], bbox[2] <= allowed[2], bbox[3] <= allowed[3])
    )
    validation = {
        "schema": "ed10-speed-cap60-candidate-validation/v1",
        "status": "PASS"
        if value_delta <= 1e-12
        and all(row["byte_identical"] for row in unchanged.values())
        and confined
        and difference["changed_pixels"] > 0
        else "FAIL",
        "target_panel": "a",
        "cap_cm_s": 60.0,
        "display_matrix_max_abs_delta": value_delta,
        "unchanged_individual_panel_pngs": unchanged,
        "parent_to_candidate_raster_difference": difference,
        "allowed_panel_a_bbox_xyxy": allowed,
        "changed_pixels_confined_to_panel_a": confined,
        "inputs": {
            "strength_matrix": {"path": matrix_path.relative_to(REPO).as_posix(), "sha256": sha256(matrix_path)},
            "pair_rows": {"path": frame_path.relative_to(REPO).as_posix(), "sha256": sha256(frame_path)},
            "parent_record": ED10_PARENT.name,
        },
    }
    write_json(ED10_OUT / "CANDIDATE_VALIDATION.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"ED10 candidate validation failed: {validation}")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("both", "ed1", "ed10"), default="both")
    args = parser.parse_args()
    if not json.loads((AUDIT / "VALIDATION.json").read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("The source 60-vs-80 audit is not PASS")
    results = {}
    if args.target in {"both", "ed1"}:
        results["ed1"] = render_ed1()
    if args.target in {"both", "ed10"}:
        results["ed10"] = render_ed10()
    print(json.dumps({"status": "PASS", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
