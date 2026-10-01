"""Promote the approved no-filter/all-outside Extended Data Figure 5e update."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import promote_figure_resolution_geometry_2026_08_12 as base
from prepare_ed05_context_only_all_outside_sources import (
    ANALYSIS_MANIFEST,
    ANALYSIS_REPORT,
    FEASIBILITY_SOURCE,
    MEANS_OUTPUT,
    PAIR_OUTPUT,
    PAIR_SOURCE,
    REPEATS_SOURCE,
    SESSION_POOLS_SOURCE,
    STATS_OUTPUT,
    STATS_SOURCE,
    VALIDATION_OUTPUT,
)


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
PARENT_NAME = "extended_data_fig05_2026-08-16_round16_bh3"
TARGET_NAME = "extended_data_fig05_2026-08-20_round17_context_only_all_outside"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = REPO / "docs/figure_refinement_2026-08-20/ed05_round17_context_only_all_outside"
CAPTION_NOTES = REPO / "docs/figure_refinement_2026-08-20/CAPTION_NOTES_ROUND17.md"
STEM = "ed05_example_place_maps_assembly_round17"
ROOT_STEM = "ed05_version_record"

SPEC = base.Spec(
    "ed5",
    PARENT_NAME,
    TARGET_NAME,
    "docs/figure_refinement_2026-08-20/ed05_round17_context_only_all_outside",
    STEM,
    ROOT_STEM,
    (
        "use all calcium-valid outside-maze frames in each session",
        "sample the same session-specific number of calcium-valid in-maze frames without position or speed filtering",
        "supersede the fixed five-minute asymmetrically filtered panel-e estimator",
    ),
    f"{STEM}.png",
    f"{STEM}.tiff",
)


def _raster_difference(old_path: Path, new_path: Path) -> dict[str, object]:
    old = np.asarray(Image.open(old_path).convert("RGB"))
    new = np.asarray(Image.open(new_path).convert("RGB"))
    if old.shape != new.shape:
        raise AssertionError(f"Raster shape changed: {old.shape} -> {new.shape}")
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    if not len(xs):
        raise AssertionError("Context-control candidate is pixel-identical to its parent")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    # Panel e occupies the lower-right slot.  The conservative target box also
    # contains its panel letter, axes labels and significance annotation.
    target_box = [2200, 3300, old.shape[1], old.shape[0]]
    outside = changed.copy()
    outside[target_box[1] : target_box[3], target_box[0] : target_box[2]] = False
    if outside.any():
        raise AssertionError(f"Raster changes escaped panel e: {bbox}")
    return {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "changed_pixels": int(changed.sum()),
        "difference_bbox_exclusive": bbox,
        "allowed_panel_e_box_exclusive": target_box,
        "outside_panel_e_changed_pixels": int(outside.sum()),
        "changes_confined_to_panel_e": True,
    }


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_local_source_audit(stage: Path) -> None:
    roles = [
        ("gallery_shortlist_csv", "plot_ready/selected_32.csv"),
        ("gallery_display_fields", "plot_ready/selected_display_fields.csv"),
        ("gallery_shortlist_npz", "plot_ready/selected_maps_fields_45_shortlist32.npz"),
        ("day_combination_animal_values", "plot_ready/ed06_day_combination_animal_values_corrected36.csv"),
        ("day_combination_descriptive_stats", "plot_ready/ed06_day_combination_descriptive_stats_corrected36.csv"),
        ("day_combination_bh3_stats", "plot_ready/ed06_day_combination_paired_stats_corrected36.csv"),
        ("place_pair_authority_no_manual_mask", "plot_ready/place_pair_summary_corrected36_all_variants.csv"),
        ("pv_pair_authority_no_manual_mask", "plot_ready/pv_pair_summary_corrected36_all_variants.csv"),
        ("rate_pair_authority_no_manual_mask", "plot_ready/rate_pair_summary_corrected36_all_variants.csv"),
        ("place_pv_rate_exact_id_support_audit", "plot_ready/place_pv_rate_exact_support_audit.csv"),
        ("gallery_per_side_provenance", "plot_ready/ED5_PANEL_A_PER_SIDE_PROVENANCE.csv"),
        ("gallery_provenance_summary", "plot_ready/ED5_PANEL_A_PROVENANCE_SUMMARY.json"),
        ("panel_e_pair_rows", "plot_ready/ed05e_context_only_all_outside_pair_rows.csv"),
        ("panel_e_condition_means", "plot_ready/ed05e_context_only_all_outside_condition_means.csv"),
        ("panel_e_clustered_stats", "plot_ready/ed05e_context_only_all_outside_clustered_stats.csv"),
        ("panel_e_repeat_results", "plot_ready/ed05e_context_only_all_outside_repeat_results.csv"),
        ("panel_e_pair_feasibility", "plot_ready/ed05e_context_only_all_outside_pair_feasibility.csv"),
        ("panel_e_session_frame_pools", "plot_ready/ed05e_context_only_all_outside_session_frame_pools.csv"),
        ("panel_e_source_preparation_validation", "source_validation/context_only_all_outside_2026_08_20/ED05E_SOURCE_PREPARATION_VALIDATION.json"),
        ("panel_e_original_pair_summary", "source_validation/context_only_all_outside_2026_08_20/equal_time_pair_summary_context_only_all_outside.csv"),
        ("panel_e_original_clustered_stats", "source_validation/context_only_all_outside_2026_08_20/equal_time_clustered_stats_current_vs_primary.csv"),
        ("panel_e_original_analysis_manifest", "source_validation/context_only_all_outside_2026_08_20/manifest.json"),
        ("panel_e_original_analysis_report", "source_validation/context_only_all_outside_2026_08_20/REPORT.md"),
        ("panel_a_svg", "individual_panels/ed05_panel_a_place_map_gallery.svg"),
        ("panel_a_pdf", "individual_panels/ed05_panel_a_place_map_gallery.pdf"),
        ("panel_a_png", "individual_panels/ed05_panel_a_place_map_gallery.png"),
        ("panel_a_tiff", "individual_panels/ed05_panel_a_place_map_gallery.tiff"),
    ]
    rows = []
    for role, relative in roles:
        path = stage / relative
        if not path.is_file():
            raise RuntimeError(f"Missing record-local source: {relative}")
        rows.append(
            {
                "role": role,
                "path": relative,
                "sha256": base.sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(stage / "SOURCE_DATA_AUDIT.csv", index=False)


def _write_record_manifest(stage: Path) -> None:
    manifest = json.loads((CANDIDATE / "MANIFEST_ROUND17.json").read_text(encoding="utf-8"))
    record_prefix = f"docs/figure_version_records/{TARGET_NAME}"
    manifest["record"] = TARGET_NAME
    manifest["parent_record"] = PARENT_NAME
    manifest["outputs"] = [
        f"{record_prefix}/{ROOT_STEM}.svg",
        f"{record_prefix}/{ROOT_STEM}.pdf",
        f"{record_prefix}/{ROOT_STEM}_600dpi.png",
        f"{record_prefix}/{ROOT_STEM}_600dpi.tiff",
    ]
    manifest["viewable_without_powerpoint"] = {
        "png": f"{record_prefix}/extended_data_figure5_source_rerender.png",
        "tiff": f"{record_prefix}/extended_data_figure5_source_rerender.tiff",
    }
    manifest["gallery_outputs"] = {
        extension: f"{record_prefix}/individual_panels/ed05_panel_a_place_map_gallery.{extension}"
        for extension in ("svg", "png", "tiff", "pdf")
    }
    manifest["validation"] = f"{record_prefix}/ED05_VALIDATION.csv"
    manifest["source_audit"] = f"{record_prefix}/SOURCE_DATA_AUDIT.csv"
    manifest["caption_notes"] = f"{record_prefix}/CAPTION_NOTES.md"
    manifest["context_sampling_contract"] = {
        "outside_pool": "all calcium-valid frames outside annotated in-maze intervals",
        "inside_pool": "all calcium-valid frames within annotated in-maze intervals; no position or speed filter",
        "matching": "inside frames sampled without replacement to the outside-frame count separately within each session",
        "outside_sampling": "none; all eligible outside frames retained",
        "inside_draws": 20,
        "random_seed": 0,
        "neural_timebase": "original imaging-frame index",
        "statistical_unit": "long-gap session pair; CR1 standard errors clustered by animal",
        "multiplicity": "Benjamini-Hochberg across four day-combination paired tests",
    }
    manifest["panel_e_primary_result"] = {
        "variant": "d1v1",
        "n_pairs": 36,
        "n_animals": 14,
        "outside_r": 0.3625965148757885,
        "inside_r": 0.4976818427937508,
        "inside_minus_outside": 0.1350853279179621,
        "ci95": [0.0949884197821155, 0.1751822360538087],
        "raw_p": 6.197040371302975e-06,
        "bh4_adjusted_p": 1.5754841589554758e-05,
    }
    base.write_json(stage / "ED05_ASSEMBLY_MANIFEST.json", manifest)


def create() -> Path:
    if TARGET.is_dir():
        errors = base.verify_hashes(TARGET)
        if errors:
            raise RuntimeError(f"Existing target is not hash-closed: {errors[:10]}")
        return TARGET
    if not PARENT.is_dir():
        raise RuntimeError(f"Missing parent: {PARENT}")
    errors = base.verify_hashes(PARENT)
    if errors:
        raise RuntimeError(f"Parent is not hash-closed: {errors[:10]}")

    required = [
        *[CANDIDATE / f"{STEM}{suffix}" for suffix in (".svg", ".pdf", ".png", ".tiff")],
        CANDIDATE / "ED05_VALIDATION_ROUND17.csv",
        CANDIDATE / "SOURCE_DATA_AUDIT_ROUND17.csv",
        CANDIDATE / "LAYOUT_AUDIT_ROUND17.csv",
        CANDIDATE / "MANIFEST_ROUND17.json",
        CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND17.csv",
        CAPTION_NOTES,
        PAIR_OUTPUT,
        MEANS_OUTPUT,
        STATS_OUTPUT,
        VALIDATION_OUTPUT,
        PAIR_SOURCE,
        STATS_SOURCE,
        REPEATS_SOURCE,
        FEASIBILITY_SOURCE,
        SESSION_POOLS_SOURCE,
        ANALYSIS_MANIFEST,
        ANALYSIS_REPORT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing candidate inputs: {missing}")
    validation = pd.read_csv(CANDIDATE / "ED05_VALIDATION_ROUND17.csv")
    if not validation["status"].eq("PASS").all():
        raise RuntimeError("Candidate validation contains failures")

    stage = Path(tempfile.mkdtemp(prefix=".ed05-round17-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)

        for extension in (".svg", ".pdf"):
            _copy(CANDIDATE / f"{STEM}{extension}", stage / f"{ROOT_STEM}{extension}")
        _copy(CANDIDATE / f"{STEM}.png", stage / f"{ROOT_STEM}_600dpi.png")
        _copy(CANDIDATE / f"{STEM}.tiff", stage / f"{ROOT_STEM}_600dpi.tiff")
        base.render_pdf(stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png", 300)
        _copy(CANDIDATE / f"{STEM}.png", stage / "extended_data_figure5_source_rerender.png")
        _copy(CANDIDATE / f"{STEM}.tiff", stage / "extended_data_figure5_source_rerender.tiff")

        _copy(CANDIDATE / "ED05_VALIDATION_ROUND17.csv", stage / "ED05_VALIDATION.csv")
        _copy(CANDIDATE / "LAYOUT_AUDIT_ROUND17.csv", stage / "LAYOUT_AUDIT.csv")
        _copy(CAPTION_NOTES, stage / "CAPTION_NOTES.md")
        _copy(
            CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND17.csv",
            stage / "plot_ready/ed06_day_combination_paired_stats_corrected36.csv",
        )

        plot_ready = stage / "plot_ready"
        for source, name in (
            (PAIR_OUTPUT, "ed06_equal_time_pair_rows_corrected36.csv"),
            (MEANS_OUTPUT, "ed06_equal_time_condition_means_corrected36.csv"),
            (STATS_OUTPUT, "ed06_equal_time_clustered_stats_corrected36.csv"),
            (PAIR_OUTPUT, "ed05e_context_only_all_outside_pair_rows.csv"),
            (MEANS_OUTPUT, "ed05e_context_only_all_outside_condition_means.csv"),
            (STATS_OUTPUT, "ed05e_context_only_all_outside_clustered_stats.csv"),
            (REPEATS_SOURCE, "ed05e_context_only_all_outside_repeat_results.csv"),
            (FEASIBILITY_SOURCE, "ed05e_context_only_all_outside_pair_feasibility.csv"),
            (SESSION_POOLS_SOURCE, "ed05e_context_only_all_outside_session_frame_pools.csv"),
        ):
            _copy(source, plot_ready / name)

        evidence = stage / "source_validation/context_only_all_outside_2026_08_20"
        for source in required[4:]:
            _copy(source, evidence / source.name)
        for relative in (
            "scripts/equal_time_home_cage_rate_control.py",
            "scripts/summarize_equal_time_context_fairness.py",
            "scripts/compare_equal_time_context_sampling_policies.py",
            "scripts/prepare_ed05_context_only_all_outside_sources.py",
            "scripts/render_ed05_example_place_maps_assembly.py",
            "scripts/render_ed05_context_only_all_outside_round17.py",
            "scripts/promote_ed05_context_only_all_outside_round17.py",
            "scripts/ed_supplement_common.py",
            "scripts/build_ed06_example_place_map_shortlist_32.py",
            "scripts/build_ed06_example_place_map_candidates_v3.py",
            "tests/test_ed05_example_place_maps_assembly.py",
        ):
            source = REPO / relative
            _copy(source, stage / "source_code" / source.name)

        raster_audit = _raster_difference(
            PARENT / f"{ROOT_STEM}_600dpi.png",
            stage / f"{ROOT_STEM}_600dpi.png",
        )
        _write_local_source_audit(stage)
        _write_record_manifest(stage)
        timestamp = datetime.now(timezone.utc).isoformat()
        context_payload = {
            "schema": "ed05-context-only-all-outside-promotion/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "definition": "all calcium-valid outside frames; matched calcium-valid in-maze frames; no position or speed filter",
            "n_pair_rows": 144,
            "n_d1v1_pairs": 36,
            "n_animals": 14,
            "n_unique_sessions": 100,
            "technical_in_maze_draws": 20,
            "d1v1": {
                "outside_r": 0.3625965148757885,
                "inside_r": 0.4976818427937508,
                "inside_minus_outside": 0.1350853279179621,
                "ci95": [0.0949884197821155, 0.1751822360538087],
                "raw_p": 6.197040371302975e-06,
                "bh4_adjusted_p": 1.5754841589554758e-05,
            },
            "raster_audit": raster_audit,
            "created_utc": timestamp,
            "desktop_modified": False,
            "review_powerpoint_status": "inherited historical review artifact; publication exports are current authority",
        }
        base.write_json(stage / "CONTEXT_SAMPLING_CORRECTION_2026_08_20.json", context_payload)
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. Panel e uses all "
            "calcium-valid outside-maze frames in each session and samples the same "
            "session-specific number of calcium-valid frames from annotated in-maze "
            "intervals without position or speed filtering. The fixed five-minute, "
            "asymmetrically filtered panel-e estimator is superseded. Panels a--d, "
            "their values, statistics, source tables and visible artwork are unchanged.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python scripts/prepare_ed05_context_only_all_outside_sources.py`, "
            "then `python scripts/render_ed05_context_only_all_outside_round17.py`. "
            "Validate the candidate before promotion with "
            "`python scripts/promote_ed05_context_only_all_outside_round17.py --create`.\n",
            encoding="utf-8",
        )
        base.write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed05-context-only-all-outside-record-validation/v1",
                "status": "pass",
                "record": TARGET_NAME,
                "checks": {
                    "candidate_validation_pass": True,
                    "source_preparation_validation_pass": True,
                    "all_outside_frames_used": True,
                    "in_maze_draw_counts_match_outside_within_session": True,
                    "no_position_or_speed_filter": True,
                    "raster_changes_confined_to_panel_e": True,
                    "panels_a_to_d_pixel_identical": True,
                    "root_exports_complete": all(
                        (stage / f"{ROOT_STEM}{suffix}").is_file()
                        for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff")
                    ),
                    "desktop_modified": False,
                },
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n", encoding="utf-8"
        )
        base.write_hashes(stage)
        errors = base.verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Successor hash closure failed: {errors[:10]}")
        os.replace(stage, TARGET)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return TARGET


def archive() -> dict[str, str]:
    row = base.archive_parent(SPEC)
    base.update_archive_manifest([row])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result: object = str(create().relative_to(REPO)) if args.create else archive()
    print(json.dumps({"status": "PASS", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
