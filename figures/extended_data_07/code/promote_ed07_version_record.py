"""Promote ED7 display round 10 as an immutable merged-number version record."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed07_age_controls"
)
DISPLAY = FIGURE_ROOT / "assembly" / "display_round11"
RECORD_NAME = "extended_data_fig07_2026-08-09_round11_concise_row_titles"
PRIOR_RECORD_NAME = "extended_data_fig07_2026-08-09_round10_six_panel_age_controls"
RECORD_ROOT = REPO_ROOT / "docs" / "figure_version_records" / RECORD_NAME
DISPLAY_MANIFEST = DISPLAY / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json"
DISPLAY_VALIDATION = DISPLAY / "ED07_SIX_PANEL_VALIDATION.json"
ASSEMBLY_STEM = "ed07_age_controls_six_panel_round11"
TOP_STEM = "ed07_panels_abc_first30_to_day2_maintained58_round11"
BOTTOM_STEM = "ed07_panels_def_long_gap_clean36_round11"
PANEL_LETTERS = list("abcdef")
EXCLUDED_PAIRS = [
    "1636-6day1->1636-6day2",
    "1637-6day1->1637-6day2",
    "1639-6day1->1639-6day2",
    "325-6day1->325-6day2",
    "347-6day1->347-6day2",
    "579-6day1->579-6day2",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hash_inventory(record: Path) -> None:
    with (record / "RECORD_HASHES.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for path in sorted(record.rglob("*")):
            if path.is_file() and path.name != "RECORD_HASHES.csv":
                writer.writerow(
                    [
                        path.relative_to(record).as_posix(),
                        sha256(path),
                        path.stat().st_size,
                    ]
                )


def verify_hash_inventory(record: Path) -> list[str]:
    with (record / "RECORD_HASHES.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["path"]: row for row in rows}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors += [f"missing:{name}" for name in sorted(set(expected) - set(actual))]
    for relative in sorted(set(actual) & set(expected)):
        path = actual[relative]
        row = expected[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{relative}")
    return errors


def promote() -> dict[str, Any]:
    if RECORD_ROOT.exists():
        raise SystemExit(f"Refusing to overwrite immutable record: {RECORD_ROOT}")
    live_manifest = json.loads(DISPLAY_MANIFEST.read_text(encoding="utf-8"))
    live_validation = json.loads(DISPLAY_VALIDATION.read_text(encoding="utf-8"))
    if live_manifest.get("status") != "PASS" or live_validation.get("status") != "PASS":
        raise SystemExit("Refusing to promote until round-10 display and validation pass")

    input_destinations = {
        "first30_plot": "plot_ready/ED07_FIRST30_PLOT_READY_MAINTAINED58.csv",
        "first30_models": "plot_ready/ED07_FIRST30_AGE_MODELS_MAINTAINED58.csv",
        "maintained58_validation": "source_validation/ED07_MAINTAINED58_AUTHORITY_VALIDATION.json",
        "accepted_first30_source": "source_authority/combined_first30_points_all_metrics.csv",
        "long_gap_plot": "plot_ready/ed09_long_gap_age_joined_corrected36.csv",
        "long_gap_models": "plot_ready/ed09_long_gap_age_transition_fixed_models_corrected36.csv",
    }
    payload: list[tuple[Path, str]] = []
    for extension in ("svg", "pdf", "png", "tiff"):
        payload.extend(
            [
                (DISPLAY / f"{ASSEMBLY_STEM}.{extension}", f"ed07_version_record{('_600dpi' if extension in {'png', 'tiff'} else '')}.{extension}"),
                (DISPLAY / f"{TOP_STEM}.{extension}", f"component_exports/{TOP_STEM}.{extension}"),
                (DISPLAY / f"{BOTTOM_STEM}.{extension}", f"component_exports/{BOTTOM_STEM}.{extension}"),
            ]
        )
    payload.extend(
        [
            (DISPLAY_MANIFEST, "ED07_DISPLAY_MANIFEST.json"),
            (DISPLAY_VALIDATION, "ED07_DISPLAY_VALIDATION.json"),
            (DISPLAY / "ED07_SIX_PANEL_CAPTION_NOTES.md", "CAPTION_NOTES.md"),
            (DISPLAY / "ED07_SIX_PANEL_FIGURE_CONTRACT.md", "FIGURE_CONTRACT.md"),
            (DISPLAY / "ed07_slide9_geometry_mockup_round11.png", "ed07_slide_mockup.png"),
            (FIGURE_ROOT / "README.md", "REBUILD_README.md"),
            (
                REPO_ROOT / "scripts" / "render_ed07_slide9_powerpoint_matched.py",
                "source_code/render_ed07_slide9_powerpoint_matched.py",
            ),
            (
                REPO_ROOT / "scripts" / "render_ed07_slide9_age_controls.py",
                "source_code/render_ed07_slide9_age_controls.py",
            ),
            (
                REPO_ROOT / "scripts" / "render_ed07_slide9_age_controls_display.py",
                "source_code/render_ed07_slide9_age_controls_display.py",
            ),
            (
                REPO_ROOT / "scripts" / "validate_ed07_six_panel_round11.py",
                "source_code/validate_ed07_six_panel_round11.py",
            ),
            (Path(__file__), "source_code/promote_ed07_version_record.py"),
            (
                REPO_ROOT / "src" / "ca_analysis" / "figure_colors.py",
                "source_code/figure_colors.py",
            ),
        ]
    )
    frozen_inputs: list[dict[str, str]] = []
    for item in live_manifest["inputs"]:
        role = item["role"]
        source = REPO_ROOT / item["path"]
        relative = input_destinations[role]
        if sha256(source) != item["sha256"]:
            raise SystemExit(f"Live input hash drifted before promotion: {role}")
        payload.append((source, relative))
        frozen_inputs.append(
            {"role": role, "path": relative, "sha256": item["sha256"]}
        )
    for role, item in live_manifest["palette_authority"].items():
        source = REPO_ROOT / item["path"]
        if sha256(source) != item["sha256"]:
            raise SystemExit(f"Palette authority hash drifted before promotion: {role}")
        payload.append((source, f"palette_authority/{role}/{source.name}"))

    missing = [str(source) for source, _ in payload if not source.is_file()]
    if missing:
        raise SystemExit(f"Refusing to promote; missing payload: {missing}")

    checks: list[dict[str, str]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if condition else "FAIL",
                "detail": str(detail),
            }
        )
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    parent = RECORD_ROOT.parent
    with tempfile.TemporaryDirectory(prefix=".ed07_round11_record_", dir=parent) as temp:
        stage = Path(temp)
        for source, relative in payload:
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            check(
                f"copy_identical::{relative}",
                sha256(source) == sha256(destination),
                relative,
            )

        frozen_manifest = json.loads(
            (stage / "ED07_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
        )
        frozen_validation = json.loads(
            (stage / "ED07_DISPLAY_VALIDATION.json").read_text(encoding="utf-8")
        )
        check("frozen_display_pass", frozen_manifest["status"] == "PASS", frozen_manifest["status"])
        check("frozen_validation_pass", frozen_validation["status"] == "PASS", frozen_validation["status"])
        check("frozen_six_panels", frozen_manifest["panel_inventory"] == ["a/place_map", "b/population_vector", "c/ensemble_rate", "d/place_map", "e/population_vector", "f/ensemble_rate"], frozen_manifest["panel_inventory"])
        check("frozen_top_scope", frozen_manifest["top_primary_scope"] == "58 maintained observations / 21 mice; BH3", frozen_manifest["top_primary_scope"])
        check("frozen_bottom_scope", frozen_manifest["bottom_primary_scope"] == "36 pairs / 14 mice; BH4; current clean single-vintage primary", frozen_manifest["bottom_primary_scope"])
        check("frozen_exclusions", frozen_manifest["excluded_from_all_ed7_primary_analyses"] == EXCLUDED_PAIRS, frozen_manifest["excluded_from_all_ed7_primary_analyses"])
        check("frozen_no_manual_mask", frozen_manifest["manual_mask_profile"] == "none for every displayed source row", frozen_manifest["manual_mask_profile"])
        display = frozen_manifest["display_contract"]
        check("frozen_age_label", display["x_axis_label"] == "Age (days)", display["x_axis_label"])
        check("frozen_top_y_label_exact", display["top_y_axis_label"] == "Day 1 early vs Day 2 correlation", display["top_y_axis_label"])
        check("frozen_bottom_y_label_exact", display["bottom_y_axis_label"] == "Long-gap correlation", display["bottom_y_axis_label"])
        check("frozen_solid_circles", display["marker_policy"] == "uniform solid circle; source provenance is not visually encoded", display["marker_policy"])
        check("frozen_round10_layout_authority", display["derived_from"] == "display_round10", display["derived_from"])

        svg = stage / "ed07_version_record.svg"
        svg_text = svg.read_text(encoding="utf-8")
        check("frozen_svg_vector_pure", "<image" not in svg_text, svg_text.count("<image"))
        check("frozen_svg_editable_text", svg_text.count("<text") >= 60, svg_text.count("<text"))
        check("frozen_svg_no_r_qualifier", "(r)" not in svg_text, "(r) absent")
        check("frozen_panel_letters_a_to_f", all(f">{letter}<" in svg_text for letter in PANEL_LETTERS), PANEL_LETTERS)

        with fitz.open(stage / "ed07_version_record.pdf") as document:
            page = document[0]
            check("frozen_pdf_one_page", document.page_count == 1, document.page_count)
            check("frozen_pdf_vector_pure", len(page.get_images(full=True)) == 0, len(page.get_images(full=True)))
            check("frozen_pdf_selectable_text", len(page.get_text()) > 400, len(page.get_text()))
            check("frozen_pdf_no_r_qualifier", "(r)" not in page.get_text(), "(r) absent")
        with Image.open(stage / "ed07_version_record_600dpi.png") as image:
            check("frozen_png_dimensions", image.size == (4500, 3090), image.size)
            check("frozen_png_nonblank", max(ImageStat.Stat(image.convert("RGB")).stddev) > 1.0, ImageStat.Stat(image.convert("RGB")).stddev)
        with Image.open(stage / "ed07_version_record_600dpi.tiff") as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            check("frozen_tiff_600dpi", all(abs(value - 600.0) < 1.0 for value in dpi), dpi)
            check("frozen_tiff_lzw", str(image.info.get("compression", "")).lower() in {"tiff_lzw", "lzw"}, image.info.get("compression"))
        for item in frozen_inputs:
            check(
                f"frozen_input_hash::{item['role']}",
                sha256(stage / item["path"]) == item["sha256"],
                item["path"],
            )

        record_manifest = {
            "schema": "extended_data_figure_7_version_record/v2",
            "record": RECORD_NAME,
            "supersedes": PRIOR_RECORD_NAME,
            "figure": "Extended Data Figure 7 (merged numbering; slide 9 of sup-figs-v2.pptx)",
            "accepted_display_round": 11,
            "panel_letters": PANEL_LETTERS,
            "panel_map": {
                "a": "first-30-min-to-Day-2 place-map correlation",
                "b": "first-30-min-to-Day-2 population-vector correlation",
                "c": "first-30-min-to-Day-2 ensemble-rate correlation",
                "d": "long-gap place-map correlation",
                "e": "long-gap population-vector correlation",
                "f": "long-gap ensemble-rate correlation",
            },
            "source_numbering_note": "Working sources remain in the old ED7-named age-controls rebuild folder; this record uses merged new-ED7 numbering.",
            "scientific_inputs": frozen_inputs,
            "display_contract": display,
            "exclusions": EXCLUDED_PAIRS,
            "manual_mask_profile": frozen_manifest["manual_mask_profile"],
            "validation_checks": frozen_validation["checks_passed"],
            "prior_record_hash_inventory_sha256": sha256(
                REPO_ROOT
                / "docs"
                / "figure_version_records"
                / PRIOR_RECORD_NAME
                / "RECORD_HASHES.csv"
            ),
        }
        (stage / "ED07_VERSION_RECORD_MANIFEST.json").write_text(
            json.dumps(record_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "MANIFEST.md").write_text(
            "# Extended Data Figure 7 round-11 version record\n\n"
            "Accepted on 2026-08-09. This record supersedes "
            "`extended_data_fig07_2026-08-09_round10_six_panel_age_controls` without "
            "modifying its contents.\n\n"
            "The publication master is a six-panel a-f age-control figure. Round 11 "
            "retains the accepted round-10 geometry and every scientific input while "
            "changing the row y titles to `Day 1 early vs Day 2 correlation` and "
            "`Long-gap correlation`. Exact first-30-min/full-Day-2 and long-gap "
            "source/destination estimands remain explicit in the caption and frozen "
            "manifests.\n\n"
            "This directory is immutable; later revisions require a new record and "
            "current-pointer update. `RECORD_HASHES.csv` covers every file except itself.\n",
            encoding="utf-8",
        )
        result = {
            "status": "pass",
            "figure": "extended_data_7",
            "record": f"docs/figure_version_records/{RECORD_NAME}",
            "accepted_display_round": 11,
            "panel_count": 6,
            "validation": checks,
            "checks_passed": sum(row["status"] == "PASS" for row in checks),
            "documented_deviations": [],
        }
        (stage / "RECORD_VALIDATION.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PROMOTED {stamp} display round 11 -> Extended Data Figure 7 "
            "(merged numbering, slide 9)\n",
            encoding="utf-8",
        )
        write_hash_inventory(stage)
        check("stage_hash_closure", not verify_hash_inventory(stage), verify_hash_inventory(stage))
        # The closure check was appended after RECORD_VALIDATION.json was written; rewrite
        # the verdict and hashes once so the saved validation contains the complete result.
        result["validation"] = checks
        result["checks_passed"] = sum(row["status"] == "PASS" for row in checks)
        (stage / "RECORD_VALIDATION.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        write_hash_inventory(stage)
        errors = verify_hash_inventory(stage)
        if errors:
            raise AssertionError(f"Final staged hash closure failed: {errors}")
        shutil.move(str(stage), RECORD_ROOT)

    final_errors = verify_hash_inventory(RECORD_ROOT)
    if final_errors:
        raise AssertionError(f"Promoted record hash closure failed: {final_errors}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "record": str(RECORD_ROOT),
                "checks_passed": result["checks_passed"],
                "hashed_files": sum(1 for _ in csv.DictReader((RECORD_ROOT / "RECORD_HASHES.csv").open(encoding="utf-8"))),
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    promote()
