"""Freeze the accepted Extended Data Figure 6 build as an immutable version record.

Extended Data Figure 6 under the **merged numbering** of ``sup-figs-v2.pptx`` is slide 7
plus the empty slide 8 ("combine with 7"). Repository sources stay under old ED7
(``extended_data_fig_07``).

Follows the Extended Data Figure 1/3 promotion pattern: copy the accepted exports, review
deck, plot-ready inputs and rendering code into a dated folder, verify byte identity of
every copy, then re-assert the scientific and display contract from the **frozen** payload
rather than from the live build.

Two things this record deliberately freezes beyond the figure itself:

* every plot-ready table, with its sha256 checked against the value the renderer recorded,
  so the record proves which inputs produced these panels;
* the 2026-07-14 fifty-percent-gate curve table under ``gate_reference/``. It is not an
  input to the figure — it is the evidence for the record's central data-currency claim,
  that the five-minute gate moves the 20-min maze-time column and leaves the 10-min and
  tile-change columns untouched.

Round 5 supersedes round 4 with an axis-labelling and y-scaling contract, each asserted
here from the frozen payload:

* y titles drop the ``(r)`` qualifier and each metric row uses one title across both its
  panels, so c/d, e/f and g/h read identically;
* x titles adopt Extended Data Figure 1's vocabulary, ``Time in maze (min)`` and
  ``Maze-tile transitions``. The tile axis is consequently drawn in raw transitions, its
  source column being scaled x1,000 at the point of drawing;
* each metric row shares one y range across all five subpanels, superseding the exported
  canonical pivot range, which is retained in the manifest for provenance.
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

FIGURE_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed06_within_session_convergence"
)
ASSEMBLY = FIGURE_ROOT / "assembly"
RECORD_ROOT = (
    REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig06_2026-08-06_round5"
)
POINTER = REPO_ROOT / "docs" / "figure_version_records" / "CURRENT_EXTENDED_DATA_FIGURES.md"

STEM = "ed06_within_session_convergence_assembly"
ROUND = 5

# Frozen contract, re-asserted from the copied manifest.
COVERAGE_GATE_SEC = 300.0
COVERAGE_GATE_FRAMES = 7500
EXPECTED_PAIRS = 59
EXPECTED_ANIMALS = 22
PIVOT_PAIR_ROWS = 56
FIXED_BH_FAMILY = 18
PIVOT_BH_FAMILY = 9
TWENTY_MIN_NEW_ROWS = 243
PANEL_LETTERS = list("abcdefgh")
N_AXES = 17
N_DISPLAYED_STATISTICS = 15

# Round-5 axis contract, re-asserted from the frozen manifest.
X_LABEL_TIME = "Time in maze (min)"
X_LABEL_TILES = "Maze-tile transitions"
ROW_LETTER_PAIRS = (("c", "d"), ("e", "f"), ("g", "h"))
Y_LABELS = ("Place-map correlation", "PV correlation", "Ensemble-rate correlation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    manifest_source = ASSEMBLY / f"ED06_ASSEMBLY_MANIFEST_ROUND{ROUND}.json"
    live_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if live_manifest.get("status") != "pass":
        raise SystemExit(
            f"Refusing to promote; the round-{ROUND} build does not pass its own checks"
        )

    payload: list[tuple[Path, str]] = [
        (ASSEMBLY / f"{STEM}_round{ROUND}.svg", "ed06_version_record.svg"),
        (ASSEMBLY / f"{STEM}_round{ROUND}.pdf", "ed06_version_record.pdf"),
        (ASSEMBLY / f"{STEM}_round{ROUND}.png", "ed06_version_record_600dpi.png"),
        (ASSEMBLY / f"{STEM}_round{ROUND}.tiff", "ed06_version_record_600dpi.tiff"),
        (manifest_source, "ED06_ASSEMBLY_MANIFEST.json"),
        (ASSEMBLY / f"ED06_VALIDATION_ROUND{ROUND}.csv", "ED06_VALIDATION.csv"),
        (
            ASSEMBLY / f"ED06_DISPLAYED_STATISTICS_ROUND{ROUND}.csv",
            "ED06_DISPLAYED_STATISTICS.csv",
        ),
        (FIGURE_ROOT / f"CAPTION_NOTES_ROUND{ROUND}.md", "CAPTION_NOTES.md"),
        (FIGURE_ROOT / "README.md", "REBUILD_README.md"),
        (
            FIGURE_ROOT / f"extended_data_figure6_layout_review_round{ROUND}.pptx",
            "ed06_review_record.pptx",
        ),
        (
            REPO_ROOT
            / "tmp"
            / "extended_data_figure6_review_2026-08-06"
            / f"extended_data_figure6_round{ROUND}_review.png",
            "ed06_review_record_render.png",
        ),
        (
            FIGURE_ROOT / f"POWERPOINT_MANIFEST_ROUND{ROUND}.json",
            "POWERPOINT_MANIFEST.json",
        ),
        (
            REPO_ROOT / "scripts" / "render_ed06_within_session_convergence_assembly.py",
            "source_code/render_ed06_within_session_convergence_assembly.py",
        ),
        (
            REPO_ROOT / "scripts" / "ed_supplement_common.py",
            "source_code/ed_supplement_common.py",
        ),
        (
            REPO_ROOT
            / "tmp"
            / "extended_data_figure6_review_2026-08-06"
            / "build_review_deck.py",
            "source_code/build_review_deck.py",
        ),
        (
            REPO_ROOT / "tests" / "test_ed06_within_session_convergence_assembly.py",
            "source_code/test_ed06_within_session_convergence_assembly.py",
        ),
        (Path(__file__), "source_code/promote_ed06_version_record.py"),
    ]

    # Plot-ready inputs, keyed so the frozen copy can be checked against the recorded hash.
    source_expectations: list[tuple[str, str, str]] = []
    for name, entry in live_manifest["sources"].items():
        source = REPO_ROOT / entry["path"]
        relative = f"plot_ready/{Path(entry['path']).name}"
        payload.append((source, relative))
        source_expectations.append((name, relative, entry["sha256"]))

    reference = live_manifest["gate_comparison_reference"]
    reference_relative = (
        "gate_reference/ed07_current59_group_curve_summary_50pct_gate_2026-07-14.csv"
    )
    payload.append((REPO_ROOT / reference["path"], reference_relative))

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

    # ------------- re-assert from the frozen payload, not the live build ---------------
    validation = list(
        csv.DictReader((RECORD_ROOT / "ED06_VALIDATION.csv").open(encoding="utf-8"))
    )
    failed = [row for row in validation if row["status"] == "FAIL"]
    deviations = [row for row in validation if row["status"] == "DEVIATION"]
    record(
        "frozen_validation_all_pass",
        not failed and len(validation) >= 30,
        f"{len(validation)} checks, {len(failed)} failing",
    )
    record(
        "frozen_no_display_contract_deviations",
        not deviations,
        "this figure meets the shared Extended Data display contract with no recorded "
        "divergence",
    )

    manifest = json.loads(
        (RECORD_ROOT / "ED06_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    authority = manifest["data_authority"]
    record(
        "frozen_five_minute_coverage_gate",
        authority["coverage_gate_sec"] == COVERAGE_GATE_SEC
        and authority["coverage_gate_frames_at_dt_0p04"] == COVERAGE_GATE_FRAMES
        and "coverage5min_window_reanalysis_2026-08-05" in authority["bundle"],
        f"{COVERAGE_GATE_SEC:.0f} s / {COVERAGE_GATE_FRAMES:,} frames per window and per "
        "pivot side, superseding the 50%-of-window rule",
    )

    gate = manifest["gate_impact_on_displayed_columns"]
    record(
        "frozen_gate_impact_10min_unchanged",
        gate["10-min window"]["changed_under_five_minute_gate"] is False
        and gate["10-min window"]["newly_admitted_curve_rows"] == 0
        and gate["10-min window"]["max_abs_mean_change"] == 0.0,
        "10-min maze-time curves identical to the 2026-07-14 bundle",
    )
    record(
        "frozen_gate_impact_20min_changed",
        gate["20-min window"]["changed_under_five_minute_gate"] is True
        and gate["20-min window"]["newly_admitted_curve_rows"] == TWENTY_MIN_NEW_ROWS,
        f"{TWENTY_MIN_NEW_ROWS} newly admitted curve rows, max |dmean| "
        f"{gate['20-min window']['max_abs_mean_change']:.3f}",
    )
    record(
        "frozen_gate_impact_tile_unchanged",
        gate["3,000-tile window"]["changed_under_five_minute_gate"] is False,
        "tile-change curves identical; no fractional time-coverage gate applies",
    )

    audit_checks = {row["check"]: row for row in validation}
    for name in (
        "fixed_template_scope_59_pairs_22_mice",
        "pivot_scope_30min_56_pairs_22_mice",
        "fixed_template_bh18_family",
        "pivot_bh9_family_per_day",
        "pivot_timebase_is_maze_time",
        "no_statistics_on_panels",
        "no_background_grid_lines",
        "xticklabels_not_rotated",
        "no_panel_image_embedded",
        "x_axes_declare_their_drawn_unit",
        "typography_matches_main_figures_at_print",
        "schematic_row_added_as_requested",
        # Round-5 additions.
        "pivot_panels_share_the_metric_row_y_title",
        "y_titles_carry_no_r_qualifier",
        "x_titles_match_extended_data_figure1_vocabulary",
        "y_limits_shared_across_the_whole_metric_row",
        "shared_y_range_clips_no_curve_band",
    ):
        row = audit_checks.get(name)
        record(
            f"frozen_check_{name}",
            row is not None and row["status"] == "PASS",
            row["detail"] if row else "check absent from the frozen validation",
        )

    record(
        "frozen_panel_letters_a_to_h",
        sorted(manifest["panels"]) == PANEL_LETTERS,
        f"{len(manifest['panels'])} panels a-h",
    )

    panels = manifest["panels"]
    record(
        "frozen_row_y_titles_shared_and_unqualified",
        all(
            panels[fixed]["y_label"] == panels[pivot]["y_label"] == expected
            and "(r)" not in expected
            for (fixed, pivot), expected in zip(ROW_LETTER_PAIRS, Y_LABELS, strict=True)
        ),
        "; ".join(
            f"{fixed}/{pivot} '{panels[fixed]['y_label']}'"
            for fixed, pivot in ROW_LETTER_PAIRS
        ),
    )
    drawn_x_labels = {
        label for letter in "cdefgh" for label in panels[letter]["x_labels"]
    }
    record(
        "frozen_x_titles_follow_extended_data_figure1",
        drawn_x_labels == {X_LABEL_TIME, X_LABEL_TILES}
        and panels["c"]["x_labels"][-1] == X_LABEL_TILES
        and set(panels["d"]["x_labels"]) == {X_LABEL_TIME},
        "c-h use only " + " and ".join(sorted(drawn_x_labels)),
    )
    record(
        "frozen_tile_axis_drawn_in_raw_transitions",
        panels["c"]["x_display_units"][-1] == "maze-tile transition"
        and all(
            unit == "in-maze minute"
            for letter in "cdefgh"
            for unit in panels[letter]["x_display_units"][
                : -1 if letter in "ceg" else None
            ]
        ),
        "the tile column names the unit it draws; every other axis is in-maze minutes",
    )
    record(
        "frozen_one_y_range_per_metric_row",
        all(
            panels[fixed]["shared_y_limits"] == panels[pivot]["shared_y_limits"]
            and panels[pivot]["superseded_canonical_y_limits"]
            != panels[pivot]["shared_y_limits"]
            for fixed, pivot in ROW_LETTER_PAIRS
        ),
        "; ".join(
            f"{fixed}/{pivot} "
            f"[{panels[fixed]['shared_y_limits'][0]:.3f}, "
            f"{panels[fixed]['shared_y_limits'][1]:.3f}] "
            f"supersedes canonical "
            f"[{panels[pivot]['superseded_canonical_y_limits'][0]:.2f}, "
            f"{panels[pivot]['superseded_canonical_y_limits'][1]:.2f}]"
            for fixed, pivot in ROW_LETTER_PAIRS
        ),
    )
    statistics = list(
        csv.DictReader(
            (RECORD_ROOT / "ED06_DISPLAYED_STATISTICS.csv").open(encoding="utf-8")
        )
    )
    record(
        "frozen_caption_statistics_complete",
        len(statistics) == N_DISPLAYED_STATISTICS
        and len(manifest["displayed_statistics"]) == N_DISPLAYED_STATISTICS,
        f"{len(statistics)} slope estimands carried in the caption, one per quantitative "
        f"subpanel across {N_AXES} axes",
    )

    for name, relative, expected in source_expectations:
        record(
            f"frozen_input_hash_{name}",
            sha256(RECORD_ROOT / relative) == expected,
            f"{relative} matches the sha256 the renderer recorded",
        )
    record(
        "frozen_gate_reference_hash",
        sha256(RECORD_ROOT / reference_relative) == reference["sha256"],
        "the 50%-gate comparison table is frozen with the record as evidence, not as an input",
    )

    hashes_path = RECORD_ROOT / "RECORD_HASHES.csv"
    with hashes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        for relative, digest, size in sorted(copied):
            writer.writerow([relative, digest, size])

    result = {
        "figure": "Extended Data Figure 6 (merged numbering; slide 7 + empty slide 8 of "
        "sup-figs-v2.pptx)",
        "record": str(RECORD_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "accepted_round": ROUND,
        "canvas_inches": manifest["canvas_inches"],
        "payload_files": len(copied),
        "documented_deviations": [],
        "validation": checks,
        "validation_status": "pass" if all(c["status"] == "PASS" for c in checks) else "fail",
    }
    (RECORD_ROOT / "RECORD_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if result["validation_status"] == "pass":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (RECORD_ROOT / "PROMOTION_COMPLETE.txt").write_text(
            f"PROMOTED {stamp} round {ROUND} -> Extended Data Figure 6 "
            "(merged numbering, slide 7 + empty slide 8)\n",
            encoding="utf-8",
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
                "accepted_round": result["accepted_round"],
                "canvas_inches": result["canvas_inches"],
                "payload_files": result["payload_files"],
                "checks": len(result["validation"]),
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
