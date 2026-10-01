"""Promote the ED5 panel-a 15 x 15 correlation-label correction."""

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


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
PARENT_NAME = "extended_data_fig05_2026-08-20_round17_context_only_all_outside"
TARGET_NAME = "extended_data_fig05_2026-09-05_round18_15x15_labels"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = REPO / "docs/figure_refinement_2026-09-05/ed05_round18_15x15_labels"
CAPTION_NOTES = CANDIDATE.parent / "CAPTION_NOTES_ROUND18.md"
FIGURE2_RECORD = RECORDS / "main_figure2_2026-08-20_round6_canonical_ensemble_rate"
FIGURE2_PANEL_B = FIGURE2_RECORD / "panel_b_sources"
STEM = "ed05_example_place_maps_assembly_round18"
ROOT_STEM = "ed05_version_record"

SPEC = base.Spec(
    "ed5",
    PARENT_NAME,
    TARGET_NAME,
    "docs/figure_refinement_2026-09-05/ed05_round18_15x15_labels",
    STEM,
    ROOT_STEM,
    (
        "use the 15 x 15 analysis correlation for every panel-a r label",
        "retain the 45 x 45 heatmaps for display resolution",
        "leave panels b-e and their statistics unchanged",
    ),
    f"{STEM}.png",
    f"{STEM}.tiff",
)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _raster_difference(old_path: Path, new_path: Path) -> dict[str, object]:
    old = np.asarray(Image.open(old_path).convert("RGB"))
    new = np.asarray(Image.open(new_path).convert("RGB"))
    if old.shape != new.shape:
        raise AssertionError(f"Raster shape changed: {old.shape} -> {new.shape}")
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    if not len(xs):
        raise AssertionError("Panel-a label candidate is pixel-identical to its parent")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    # The gallery ends above the b-e quantitative row. Preserve the lower row
    # pixel-for-pixel so this remains a label-only panel-a correction.
    quantitative_row_top = 3290
    lower_changed = int(changed[quantitative_row_top:, :].sum())
    if lower_changed:
        raise AssertionError(
            f"Raster changes escaped panel a: {lower_changed} pixels below y={quantitative_row_top}"
        )
    return {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "changed_pixels": int(changed.sum()),
        "difference_bbox_exclusive": bbox,
        "panel_a_y_exclusive": [0, quantitative_row_top],
        "panels_b_to_e_changed_pixels": lower_changed,
        "changes_confined_to_panel_a": True,
    }


def _write_figure2_anchor_parity() -> Path:
    selected = pd.read_csv(
        PARENT / "plot_ready/selected_32.csv", dtype={"animal_id": str}
    )
    anchors = selected.loc[selected["fixed_main_fig2b"].astype(bool)].copy()
    anchors = anchors.sort_values("anchor_order", kind="stable")
    source_rows = []
    for path in sorted(FIGURE2_PANEL_B.glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            source_rows.append(
                {
                    "pair_key": str(data["pair_key"]),
                    "train_id": int(data["cell_a"]),
                    "test_id": int(data["cell_b"]),
                    "figure2_place_r_15x15": float(data["place_r"]),
                    "figure2_source_path": path.relative_to(REPO).as_posix(),
                    "figure2_source_sha256": base.sha256(path),
                }
            )
    source = pd.DataFrame(source_rows)
    parity = anchors[
        [
            "anchor_order",
            "visual_rank",
            "candidate_id",
            "animal_id",
            "pair_key",
            "train_id",
            "test_id",
            "place_r_15_analysis",
            "place_r_45_display",
        ]
    ].merge(
        source,
        on=["pair_key", "train_id", "test_id"],
        validate="one_to_one",
    )
    parity["absolute_15x15_delta"] = (
        parity["place_r_15_analysis"] - parity["figure2_place_r_15x15"]
    ).abs()
    parity["two_decimal_label_match"] = (
        parity["place_r_15_analysis"].round(2)
        == parity["figure2_place_r_15x15"].round(2)
    )
    if len(parity) != 4 or parity["absolute_15x15_delta"].max() > 1e-6:
        raise RuntimeError("ED5 fixed anchors do not match Figure 2 15 x 15 sources")
    if not parity["two_decimal_label_match"].all():
        raise RuntimeError("ED5 and Figure 2 printed anchor labels do not match")
    path = CANDIDATE / "FIGURE2_PANEL_B_LABEL_PARITY.csv"
    parity.to_csv(path, index=False)
    return path


def _write_local_source_audit(stage: Path) -> None:
    roles = [
        ("gallery_shortlist_csv", "plot_ready/selected_32.csv"),
        ("gallery_display_fields", "plot_ready/selected_display_fields.csv"),
        ("gallery_45x45_heatmap_arrays", "plot_ready/selected_maps_fields_45_shortlist32.npz"),
        ("gallery_15x15_correlation_label_audit", "plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv"),
        ("figure2_anchor_label_parity", "plot_ready/FIGURE2_PANEL_B_LABEL_PARITY.csv"),
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
    manifest = json.loads((CANDIDATE / "MANIFEST_ROUND18.json").read_text(encoding="utf-8"))
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
    manifest["panel_a_correlation_label_contract"] = {
        "heatmap_grid": "45x45",
        "printed_r_grid": "15x15",
        "printed_r_column": "place_r_15_analysis",
        "superseded_printed_r_column": "place_r_45_display",
        "figure2_anchor_count": 4,
        "figure2_anchor_parity_tolerance": 1e-6,
        "label_audit": f"{record_prefix}/plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv",
        "figure2_parity_audit": f"{record_prefix}/plot_ready/FIGURE2_PANEL_B_LABEL_PARITY.csv",
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
    parent_errors = base.verify_hashes(PARENT)
    allowed_parent_errors = {
        "unhashed:source_code/__pycache__/equal_time_home_cage_rate_control.cpython-310.pyc"
    }
    if set(parent_errors).difference(allowed_parent_errors):
        raise RuntimeError(f"Parent has unexpected hash errors: {parent_errors[:10]}")

    required = [
        *[CANDIDATE / f"{STEM}{suffix}" for suffix in (".svg", ".pdf", ".png", ".tiff")],
        CANDIDATE / "ED05_VALIDATION_ROUND18.csv",
        CANDIDATE / "SOURCE_DATA_AUDIT_ROUND18.csv",
        CANDIDATE / "LAYOUT_AUDIT_ROUND18.csv",
        CANDIDATE / "MANIFEST_ROUND18.json",
        CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND18.csv",
        CANDIDATE / "derived/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv",
        *[
            CANDIDATE / f"derived/ed05_compact_32_grid_8x8_round18.{extension}"
            for extension in ("svg", "pdf", "png", "tiff")
        ],
        CAPTION_NOTES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing candidate inputs: {missing}")
    validation = pd.read_csv(CANDIDATE / "ED05_VALIDATION_ROUND18.csv")
    if not validation["status"].eq("PASS").all():
        raise RuntimeError("Candidate validation contains failures")
    parity_path = _write_figure2_anchor_parity()

    stage = Path(tempfile.mkdtemp(prefix=".ed05-round18-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        pycache = stage / "source_code/__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)

        for extension in (".svg", ".pdf"):
            _copy(CANDIDATE / f"{STEM}{extension}", stage / f"{ROOT_STEM}{extension}")
        _copy(CANDIDATE / f"{STEM}.png", stage / f"{ROOT_STEM}_600dpi.png")
        _copy(CANDIDATE / f"{STEM}.tiff", stage / f"{ROOT_STEM}_600dpi.tiff")
        base.render_pdf(stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png", 300)
        _copy(CANDIDATE / f"{STEM}.png", stage / "extended_data_figure5_source_rerender.png")
        _copy(CANDIDATE / f"{STEM}.tiff", stage / "extended_data_figure5_source_rerender.tiff")

        _copy(CANDIDATE / "ED05_VALIDATION_ROUND18.csv", stage / "ED05_VALIDATION.csv")
        _copy(CANDIDATE / "LAYOUT_AUDIT_ROUND18.csv", stage / "LAYOUT_AUDIT.csv")
        _copy(CAPTION_NOTES, stage / "CAPTION_NOTES.md")
        _copy(
            CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND18.csv",
            stage / "plot_ready/ed06_day_combination_paired_stats_corrected36.csv",
        )
        _copy(
            CANDIDATE / "derived/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv",
            stage / "plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv",
        )
        _copy(parity_path, stage / "plot_ready/FIGURE2_PANEL_B_LABEL_PARITY.csv")
        for extension in ("svg", "pdf", "png", "tiff"):
            _copy(
                CANDIDATE / f"derived/ed05_compact_32_grid_8x8_round18.{extension}",
                stage / f"individual_panels/ed05_panel_a_place_map_gallery.{extension}",
            )

        evidence = stage / "source_validation/panel_a_15x15_labels_2026_09_05"
        for source in required[4:]:
            _copy(source, evidence / source.name)
        _copy(parity_path, evidence / parity_path.name)
        for source in FIGURE2_PANEL_B.glob("*.npz"):
            _copy(source, evidence / "figure2_panel_b_sources" / source.name)
        for relative in (
            "scripts/render_ed05_example_place_maps_assembly.py",
            "scripts/render_ed05_panel_a_15x15_labels_round18.py",
            "scripts/promote_ed05_panel_a_15x15_labels_round18.py",
            "scripts/prepare_ed05_context_only_all_outside_sources.py",
            "scripts/ed_supplement_common.py",
            "scripts/build_ed06_example_place_map_shortlist_32.py",
            "scripts/build_ed06_example_place_map_candidates_v3.py",
            "tests/test_ed05_example_place_maps_assembly.py",
            "tests/test_ed05_panel_a_15x15_labels.py",
        ):
            source = REPO / relative
            _copy(source, stage / "source_code" / source.name)

        raster_audit = _raster_difference(
            PARENT / f"{ROOT_STEM}_600dpi.png",
            stage / f"{ROOT_STEM}_600dpi.png",
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        label_audit = pd.read_csv(stage / "plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv")
        parity = pd.read_csv(stage / "plot_ready/FIGURE2_PANEL_B_LABEL_PARITY.csv")
        correction = {
            "schema": "ed05-panel-a-15x15-label-correction/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "scope": "panel-a printed r labels only",
            "heatmaps": "unchanged 45x45 display maps",
            "printed_r": "15x15 place-map analysis correlation",
            "label_count": int(len(label_audit)),
            "figure2_anchor_count": int(len(parity)),
            "figure2_anchor_max_absolute_delta": float(parity["absolute_15x15_delta"].max()),
            "previous_45x45_to_current_15x15_max_absolute_change": float(
                label_audit["displayed_minus_superseded"].abs().max()
            ),
            "raster_audit": raster_audit,
            "created_utc": timestamp,
            "desktop_modified": False,
        }
        base.write_json(stage / "PANEL_A_LABEL_CORRECTION_2026_09_05.json", correction)
        _write_local_source_audit(stage)
        _write_record_manifest(stage)
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. In panel a, the "
            "45 x 45 heatmaps are retained for visual detail, but every printed r "
            "value now uses the same 15 x 15 place-map correlation estimator as "
            "Main Figure 2. The four outlined Figure 2 examples therefore print "
            "0.63, 0.42, 0.18 and -0.20. Panels b--e, their data, statistics and "
            "visible artwork are unchanged.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python -B scripts/render_ed05_panel_a_15x15_labels_round18.py`. "
            "The renderer reads the frozen 45 x 45 heatmap arrays and the 15 x 15 "
            "analysis correlations in `plot_ready/selected_32.csv`; the current-record "
            "adapter can rebuild the accepted exports entirely from record-local "
            "sources with `python -B docs/figure_version_records/rebuild_current_extended_data_figures.py "
            "--artifact extended_data_figure_5 --check --output-dir <fresh-directory>`.\n",
            encoding="utf-8",
        )
        base.write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed05-panel-a-15x15-label-record-validation/v1",
                "status": "pass",
                "record": TARGET_NAME,
                "checks": {
                    "candidate_validation_pass": True,
                    "gallery_heatmaps_remain_45x45": True,
                    "all_32_labels_use_15x15_analysis_correlations": True,
                    "four_figure2_anchors_match_within_1e-6": True,
                    "panels_b_to_e_pixel_identical": True,
                    "panel_e_context_estimator_inherited_unchanged": True,
                    "all_outside_frames_used": True,
                    "in_maze_draw_counts_match_outside_within_session": True,
                    "no_position_or_speed_filter": True,
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
    # Round 17 acquired one interpreter cache after promotion. Hash the exact
    # existing tree before moving it so the archived parent remains intact and
    # independently verifiable.
    if PARENT.is_dir() and base.verify_hashes(PARENT):
        base.write_hashes(PARENT)
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
