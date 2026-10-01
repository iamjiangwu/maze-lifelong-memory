"""Freeze the accepted Extended Data Figure 1 build as an immutable version record.

Extended Data Figure 1 under the **merged numbering** of ``sup-figs-v2.pptx`` is slide 1
plus the empty slide 2, so it carries the old-ED1 acquisition panels (a-f) and the old-ED2
exploration panels (g-k) on one page.

Follows the Extended Data Figure 3 promotion pattern: copy the accepted exports, prior
review context, plot-ready inputs and rendering code into a dated folder, verify byte identity of
every copy, then re-assert the scientific contract from the **frozen** payload rather than
the live build.

The gate accepts the two divergences from the shared display contract that the bundle
records as ``DEVIATION`` (panel-f x tick rotation and 0.95x typography) but requires that
the deviation set is exactly those two, so a new one cannot slip in unnoticed.

Three inputs are deliberately referenced by hash instead of copied: the 14 MB panel-a
trajectory table and the two multi-megabyte raw trajectory files. Their compact drawn
derivatives - the masked panel-a coordinates and the panel-i end-node sequence - are frozen
instead, so every plotted value stays reproducible from the record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

BUNDLE = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed01_acquisition_and_exploration"
)
ASSEMBLY = BUNDLE / "assembly"
REVIEW = BUNDLE / "review_2026-08-05"

RECORD_ROOT = REPO_ROOT / "docs" / "figure_version_records" / (
    "extended_data_fig01_2026-08-09_round9_h_legend"
)
POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"

STEM = "ed01_acquisition_and_exploration_assembly"
ROUND = 9
PRIOR_REVIEW_ROUND = 7

EXPECTED_DEVIATIONS = {"xticklabels_not_rotated", "typography_matches_shared_contract"}

# Inputs too large to duplicate into an immutable record; referenced by sha256 in the
# assembly manifest, with their compact drawn derivatives frozen instead.
REFERENCE_ONLY_INPUTS = ("a", "i-naive", "i-return")

PAYLOAD: tuple[tuple[Path, str], ...] = (
    (ASSEMBLY / f"{STEM}.svg", "ed01_version_record.svg"),
    (ASSEMBLY / f"{STEM}.pdf", "ed01_version_record.pdf"),
    (ASSEMBLY / f"{STEM}.png", "ed01_version_record_600dpi.png"),
    (ASSEMBLY / f"{STEM}.tiff", "ed01_version_record_600dpi.tiff"),
    (ASSEMBLY / "ED01_ASSEMBLY_MANIFEST.json", "ED01_ASSEMBLY_MANIFEST.json"),
    (ASSEMBLY / "ED01_VALIDATION.csv", "ED01_VALIDATION.csv"),
    (ASSEMBLY / "ROUND9_RENDER_VALIDATION.json", "ROUND9_RENDER_VALIDATION.json"),
    (ASSEMBLY / "LAYOUT_AUDIT.csv", "LAYOUT_AUDIT.csv"),
    (ASSEMBLY / "SOURCE_DATA_AUDIT.csv", "SOURCE_DATA_AUDIT.csv"),
    (ASSEMBLY / "CAPTION_NOTES.md", "CAPTION_NOTES.md"),
    (
        REVIEW / f"ed01_revision_review_round{PRIOR_REVIEW_ROUND}.pptx",
        "ed01_prior_review_round7.pptx",
    ),
    (
        REVIEW / f"ed01_review_round{PRIOR_REVIEW_ROUND}_powerpoint.png",
        "ed01_prior_review_round7_render.png",
    ),
    (REVIEW / "ed01_round6_user_annotated.png", "ed01_round6_user_annotated.png"),
    (REVIEW / "REPORT.md", "PRIOR_REVIEW_REPORT.md"),
    (
        REPO_ROOT / "scripts" / "render_extended_data_figure1_round4.py",
        "source_code/render_extended_data_figure1_round4.py",
    ),
    (
        REPO_ROOT / "scripts" / "render_grs2026_first_maze_trial_trajectories.py",
        "source_code/render_grs2026_first_maze_trial_trajectories.py",
    ),
    (
        REPO_ROOT / "scripts" / "render_ed01_ed02_gap_closure.py",
        "source_code/render_ed01_ed02_gap_closure.py",
    ),
    (
        REPO_ROOT / "scripts" / "assemble_ed01_supplement_v2_bundle.py",
        "source_code/assemble_ed01_supplement_v2_bundle.py",
    ),
    (
        REPO_ROOT / "tests" / "test_extended_data_figure1_round4.py",
        "source_code/test_extended_data_figure1_round4.py",
    ),
    (Path(__file__), "source_code/promote_ed01_version_record.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    manifest_source = json.loads(
        (ASSEMBLY / "ED01_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )

    payload = list(PAYLOAD)
    # plot-ready inputs, except the three referenced by hash only
    for panel, entry in manifest_source["sources"].items():
        if panel in REFERENCE_ONLY_INPUTS:
            continue
        source = REPO_ROOT / entry["path"]
        payload.append((source, f"plot_ready/{Path(entry['path']).name}"))
    # compact drawn derivatives that stand in for the reference-only inputs
    for name in (
        f"PANEL_A_TRAJECTORY_PLOT_DATA_ROUND{ROUND}.csv.gz",
        f"PANEL_A_JUMP_MASK_AUDIT_ROUND{ROUND}.csv",
        f"PANEL_I_END_NODE_SEQUENCE_ROUND{ROUND}.csv",
        f"PANEL_I_CYCLE_SUMMARY_ROUND{ROUND}.csv",
        f"PANEL_I_SELECTION_AUDIT_ROUND{ROUND}.csv",
        f"ACCUMULATION_SUMMARY_ROUND{ROUND}.csv",
        f"SPEED_SUMMARY_ROUND{ROUND}.csv",
        f"TIME_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
        f"TILE_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
    ):
        payload.append((ASSEMBLY / "derived" / name, f"drawn_values/{name}"))

    missing = [str(src) for src, _ in payload if not src.exists()]
    if missing:
        raise SystemExit(f"Refusing to promote; missing payload: {missing}")

    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, str, int]] = []
    for source, relative in payload:
        destination = RECORD_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copy_hash = sha256(destination)
        record(
            f"copy_identical_{relative.replace('/', '_')}",
            source_hash == copy_hash,
            f"{relative} sha256 {copy_hash[:16]}",
        )
        copied.append((relative, copy_hash, destination.stat().st_size))

    # ------------------- re-assert the contract from the frozen payload, not the build
    validation = list(
        csv.DictReader((RECORD_ROOT / "ED01_VALIDATION.csv").open(encoding="utf-8"))
    )
    failed = [row for row in validation if row["status"] not in {"PASS", "DEVIATION"}]
    deviations = {row["check"] for row in validation if row["status"] == "DEVIATION"}
    record(
        "frozen_validation_no_failures",
        not failed and len(validation) >= 30,
        f"{len(validation)} checks, {len(failed)} failing",
    )
    record(
        "frozen_deviations_are_the_two_documented_ones",
        deviations == EXPECTED_DEVIATIONS,
        f"{sorted(deviations)}",
    )

    manifest = json.loads(
        (RECORD_ROOT / "ED01_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    audit = manifest["audit"]
    policy = manifest["source_policy"]

    record(
        "frozen_source_policy_no_panel_image_reuse",
        policy["reused_panel_svgs"] == 0
        and policy["reused_panel_rasters"] == 0
        and policy["rendered_directly_from_tables_and_raw_trajectories"] is True,
        "every panel drawn from source tables or raw trajectories; "
        f"{policy['in_canvas_raster_elements']} in-canvas raster elements only",
    )
    record(
        "frozen_svg_pdf_fully_vector",
        policy["in_canvas_raster_elements"] == 0
        and audit["svg_image_elements"] == 0
        and audit["pdf_image_objects"] == 0
        and audit["pdf_vector_drawing_objects"] > 1_000,
        f"SVG images={audit['svg_image_elements']}; PDF images="
        f"{audit['pdf_image_objects']}; PDF drawings="
        f"{audit['pdf_vector_drawing_objects']}",
    )
    record(
        "frozen_panel_g_h_square_and_centered",
        abs(audit["panel_g_box_aspect"] - 1.0) < 1e-6
        and all(abs(value - 1.0) < 1e-6 for value in audit["panel_h_box_aspects"])
        and audit["panel_fgh_center_alignment_delta_fraction"] < 1e-6,
        f"g aspect={audit['panel_g_box_aspect']}; h aspects="
        f"{audit['panel_h_box_aspects']}; center delta="
        f"{audit['panel_fgh_center_alignment_delta_fraction']}",
    )
    record(
        "frozen_panel_h_legend_lower_right_and_clear",
        audit["panel_h_legend_location"] == 4
        and audit["panel_h_legend_labels"] == ["Day 1", "Day 2"]
        and audit["panel_h_legend_data_overlap_count"] == 0,
        f"legend bbox={audit['panel_h_legend_bbox_axes_fraction']}; "
        f"overlap count={audit['panel_h_legend_data_overlap_count']}",
    )
    record(
        "frozen_current153_scope",
        audit["current153_sessions"] == 153 and audit["current153_mice"] == 36,
        f"{audit['current153_sessions']} sessions / {audit['current153_mice']} mice",
    )
    record(
        "frozen_tile_time_diagnostic",
        abs(audit["tile_time_slope_loglog"] - 0.9642052475245702) < 1e-12
        and audit["tile_time_cluster_p_two_sided"] < 1e-40,
        f"log-log slope {audit['tile_time_slope_loglog']:.6f}, "
        f"CR1 p={audit['tile_time_cluster_p_two_sided']:.3g}",
    )
    record(
        "frozen_panel_f_bh27_family",
        audit["panel_f_bh27_contrasts"] == 27
        and audit["panel_f_displayed_contrasts"] == 18,
        f"{audit['panel_f_displayed_contrasts']} of "
        f"{audit['panel_f_bh27_contrasts']} contrasts displayed",
    )
    pair = audit["panel_i_selected_pair"]
    record(
        "frozen_panel_i_example_pair",
        pair["animal_id"] == "451"
        and pair["naive_complete_cycles"] == 13
        and pair["return_complete_cycles"] == 11,
        f"mouse {pair['animal_id']}: {pair['naive_session']} vs {pair['return_session']}, "
        f"{pair['naive_complete_cycles']}/{pair['return_complete_cycles']} complete cycles",
    )
    limits = audit["panel_i_shared_x_limits"]
    record(
        "frozen_panel_i_column1_truncated_to_naive_range",
        limits["0"][0] == limits["0"][1]
        and abs(limits["0"][0][1] - 494.5) < 1e-9
        and abs(limits["1"][0][1] - 846.5) < 1e-9,
        "column 1 shares the naive two-cycle range on both rows (494 visits); "
        "column 2 spans 846 visits independently",
    )
    record(
        "frozen_panel_a_no_jump_mask",
        audit["panel_a_corrected_max_drawn_step_grid"] <= 0.63,
        f"max drawn step {audit['panel_a_corrected_max_drawn_step_grid']:.4f} grid units",
    )
    record(
        "frozen_no_background_grid",
        audit["grid_lines_visible"] == 0,
        f"{audit['grid_lines_visible']} gridlines",
    )
    record(
        "frozen_editable_svg_text",
        audit["svg_text_elements"] > 0,
        f"{audit['svg_text_elements']} live <text> elements",
    )
    record(
        "frozen_panel_letters_a_to_k",
        sorted(manifest["panels"]) == list("abcdefghijk"),
        f"{len(manifest['panels'])} panels a-k on one merged page",
    )
    record(
        "frozen_merged_numbering_declared",
        "merged numbering" in manifest["figure"] and "slide 1" in manifest["figure"],
        "record names the new ED number and the source slide",
    )
    record(
        "frozen_candidate_jk_flagged",
        any(
            "not promoted" in entry["promotion_status"]
            for panel, entry in manifest["sources"].items()
            if panel in {"j", "k"}
        ),
        "panels j-k recorded as candidate axis options, not promoted old-ED2 primaries",
    )

    # reference-only inputs must still be resolvable and unchanged
    for panel in REFERENCE_ONLY_INPUTS:
        entry = manifest["sources"][panel]
        path = REPO_ROOT / entry["path"]
        record(
            f"reference_only_input_{panel}",
            path.exists() and sha256(path) == entry["sha256"],
            f"{entry['path']} sha256 {entry['sha256'][:16]} unchanged; "
            "drawn values frozen under drawn_values/",
        )

    hashes_path = RECORD_ROOT / "RECORD_HASHES.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for relative, digest, size in sorted(copied):
            writer.writerow([relative, digest, size])

    result = {
        "figure": "Extended Data Figure 1 (merged numbering; slide 1 + empty slide 2 of "
        "sup-figs-v2.pptx)",
        "record": str(RECORD_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "accepted_round": ROUND,
        "payload_files": len(copied),
        "documented_deviations": sorted(deviations),
        "validation": checks,
        "validation_status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
        "status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
    }
    (RECORD_ROOT / "RECORD_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (RECORD_ROOT / "MANIFEST.md").write_text(
        "# Extended Data Figure 1 round-9 version record\n\n"
        "Accepted on 2026-08-09. This record supersedes "
        "`extended_data_fig01_2026-08-09_round8_vector_square_gh` without modifying it.\n\n"
        "Round 9 preserves every source row and plotted value, exports panel a's "
        "trajectory collections and colorbar as vector paths in SVG/PDF, and makes "
        "panel g plus both panel-h plotting boxes square and vertically centered in "
        "the f-g-h row. The shared Day 1/Day 2 legend is in the reward-directed "
        "panel's lower-right corner, clear of the plotted data. The prior round-7 "
        "review deck is retained only as layout history; the round-9 publication "
        "master is the authoritative visual.\n",
        encoding="utf-8",
    )
    if result["validation_status"] == "pass":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (RECORD_ROOT / "PROMOTION_COMPLETE.txt").write_text(
            f"PROMOTED {stamp} round {ROUND} -> Extended Data Figure 1 "
            "(merged numbering, slide 1 + empty slide 2)\n",
            encoding="utf-8",
        )
    # Hash last so the verdict, manifest, and completion stamp are immutable
    # payload rather than mutable files outside the record closure.
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for path in sorted(RECORD_ROOT.rglob("*")):
            if path.is_file() and path.name != "RECORD_HASHES.csv":
                writer.writerow(
                    [path.relative_to(RECORD_ROOT).as_posix(), sha256(path), path.stat().st_size]
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = promote()
    failed = [c for c in result["validation"] if c["status"] != "PASS"]
    print(
        json.dumps(
            {
                "record": result["record"],
                "payload_files": result["payload_files"],
                "checks": len(result["validation"]),
                "documented_deviations": result["documented_deviations"],
                "status": result["validation_status"],
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
