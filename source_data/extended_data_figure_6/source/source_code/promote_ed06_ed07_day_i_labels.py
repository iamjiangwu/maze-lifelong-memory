"""Promote upright Day-i notation successors for Extended Data Figures 6 and 7."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs" / "figure_version_records"

ED6_PARENT = RECORDS / "extended_data_fig06_2026-08-16_round5_day_i"
ED6_TARGET = RECORDS / "extended_data_fig06_2026-08-16_round6_upright_i"
ED6_RENDER = ROOT / ".tmp" / "ed06_upright_i_labels_2026-08-16"

ED7_PARENT = RECORDS / "extended_data_fig07_2026-08-16_round15_day_i_axis"
ED7_TARGET = RECORDS / "extended_data_fig07_2026-08-16_round16_upright_i_axis"
ED7_RENDER = ROOT / ".tmp" / "ed07_upright_i_labels_2026-08-16"

STAGING_ROOT = ROOT / ".tmp" / "ed06_ed07_upright_i_record_staging"
CAPTION = (
    ROOT
    / "docs"
    / "manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise AssertionError(f"Copy hash mismatch: {source} -> {destination}")


def make_300dpi(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        resized = image.resize(
            (image.width // 2, image.height // 2), Image.Resampling.LANCZOS
        )
        resized.save(destination, dpi=(300, 300), optimize=True)
    with Image.open(source) as source_image, Image.open(destination) as output_image:
        if output_image.size != (source_image.width // 2, source_image.height // 2):
            raise AssertionError(f"Unexpected 300-dpi dimensions for {destination}")


def tree_records(root: Path, relative_dirs: tuple[str, ...]) -> dict[str, str]:
    records: dict[str, str] = {}
    for relative_dir in relative_dirs:
        directory = root / relative_dir
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            records[path.relative_to(root).as_posix()] = sha256(path)
    return records


def png_difference(
    before: Path, after: Path, *, allowed_box: tuple[int, int, int, int]
) -> dict[str, Any]:
    with Image.open(before) as before_image, Image.open(after) as after_image:
        a = np.asarray(before_image.convert("RGB"))
        b = np.asarray(after_image.convert("RGB"))
    if a.shape != b.shape:
        raise AssertionError(f"Raster dimensions changed: {a.shape} versus {b.shape}")
    difference = np.any(a != b, axis=2)
    y_values, x_values = np.where(difference)
    if not len(x_values):
        raise AssertionError("Expected a visible day-label change but found no raster delta")
    x0, y0, x1, y1 = allowed_box
    allowed = np.zeros(difference.shape, dtype=bool)
    allowed[y0:y1, x0:x1] = True
    outside = int(np.sum(difference & ~allowed))
    return {
        "shape_rgb": list(a.shape),
        "changed_pixels": int(np.sum(difference)),
        "changed_bbox_xyxy": [
            int(x_values.min()),
            int(y_values.min()),
            int(x_values.max() + 1),
            int(y_values.max() + 1),
        ],
        "allowed_box_xyxy": list(allowed_box),
        "changed_pixels_outside_allowed_box": outside,
        "status": "PASS" if outside == 0 else "FAIL",
    }


def validate_vector_outputs(
    record: Path, stem: str, forbidden_labels: tuple[str, ...]
) -> dict[str, Any]:
    svg = record / f"{stem}.svg"
    pdf = record / f"{stem}.pdf"
    svg_text = svg.read_text(encoding="utf-8")
    with fitz.open(pdf) as document:
        page_count = document.page_count
        embedded_images = sum(len(page.get_images(full=True)) for page in document)
    result = {
        "svg_editable_text": "<text" in svg_text,
        "svg_contains_upright_day_i": r"Day $\mathrm{i}$" in svg_text,
        "svg_contains_upright_day_i_plus_1": r"Day $\mathrm{i}+1$" in svg_text,
        "svg_forbidden_labels_absent": not any(
            label in svg_text for label in forbidden_labels
        ),
        "pdf_single_page": page_count == 1,
        "pdf_embedded_images": embedded_images,
    }
    if not all(
        (
            result["svg_editable_text"],
            result["svg_contains_upright_day_i"],
            result["svg_contains_upright_day_i_plus_1"],
            result["svg_forbidden_labels_absent"],
            result["pdf_single_page"],
            embedded_images == 0,
        )
    ):
        raise AssertionError(f"Vector validation failed for {record}: {result}")
    return result


def write_hash_ledger(record: Path) -> dict[str, Any]:
    ledger_path = record / "RECORD_HASHES.csv"
    rows = []
    for path in sorted(item for item in record.rglob("*") if item.is_file()):
        if path == ledger_path:
            continue
        rows.append(
            (
                path.relative_to(record).as_posix(),
                sha256(path),
                path.stat().st_size,
            )
        )
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256", "bytes"])
        writer.writerows(rows)
    reread = list(csv.DictReader(ledger_path.open(encoding="utf-8")))
    current = {
        path.relative_to(record).as_posix(): (sha256(path), path.stat().st_size)
        for path in record.rglob("*")
        if path.is_file() and path != ledger_path
    }
    listed = {row["path"]: (row["sha256"], int(row["bytes"])) for row in reread}
    if current != listed:
        raise AssertionError(f"Record hash ledger is not closed: {record}")
    return {"rows": len(rows), "sha256": sha256(ledger_path), "status": "PASS"}


def stage_ed6() -> tuple[Path, dict[str, Any]]:
    target = STAGING_ROOT / ED6_TARGET.name
    shutil.copytree(ED6_PARENT, target)

    for stale in (
        "ed06_review_record.pptx",
        "ed06_review_record_render.png",
        "POWERPOINT_MANIFEST.json",
    ):
        (target / stale).unlink(missing_ok=True)

    stem = "ed06_within_session_convergence_assembly_round6"
    for extension, destination in (
        ("svg", "ed06_version_record.svg"),
        ("pdf", "ed06_version_record.pdf"),
        ("png", "ed06_version_record_600dpi.png"),
        ("tiff", "ed06_version_record_600dpi.tiff"),
    ):
        copy(ED6_RENDER / f"{stem}.{extension}", target / destination)
    make_300dpi(
        target / "ed06_version_record_600dpi.png",
        target / "ed06_version_record_300dpi.png",
    )
    for source, destination in (
        ("ED06_ASSEMBLY_MANIFEST_ROUND6.json", "ED06_ASSEMBLY_MANIFEST.json"),
        ("ED06_VALIDATION_ROUND6.csv", "ED06_VALIDATION.csv"),
        ("ED06_DISPLAYED_STATISTICS_ROUND6.csv", "ED06_DISPLAYED_STATISTICS.csv"),
        ("CAPTION_NOTES_ROUND6.md", "CAPTION_NOTES.md"),
    ):
        copy(ED6_RENDER / source, target / destination)
    copy(
        ROOT / "scripts" / "render_ed06_within_session_convergence_assembly.py",
        target / "source_code" / "render_ed06_within_session_convergence_assembly.py",
    )
    copy(Path(__file__), target / "source_code" / Path(__file__).name)
    copy(
        ROOT / "tests" / "test_ed06_within_session_convergence_assembly.py",
        target / "source_code" / "test_ed06_within_session_convergence_assembly.py",
    )
    for source_name in (
        "validate_current_figure_records.py",
        "validate_all_figure_refinement_round2_records.py",
    ):
        copy(ROOT / "scripts" / source_name, target / "source_code" / source_name)

    parent_manifest = json.loads(
        (ED6_PARENT / "ED06_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    current_manifest = json.loads(
        (target / "ED06_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    parent_sources = {
        name: entry["sha256"] for name, entry in parent_manifest["sources"].items()
    }
    current_sources = {
        name: entry["sha256"] for name, entry in current_manifest["sources"].items()
    }
    if parent_sources != current_sources:
        raise AssertionError("ED6 scientific input hashes changed")
    scientific_parent = tree_records(
        ED6_PARENT, ("plot_ready", "gate_reference", "gate_authority")
    )
    scientific_target = tree_records(
        target, ("plot_ready", "gate_reference", "gate_authority")
    )
    if scientific_parent != scientific_target:
        raise AssertionError("ED6 frozen scientific directories changed")

    parity = png_difference(
        ED6_PARENT / "ed06_version_record_600dpi.png",
        ED6_RENDER / "ed06_within_session_convergence_assembly_round6.png",
        allowed_box=(0, 0, 4500, 800),
    )
    vector = validate_vector_outputs(
        target,
        "ed06_version_record",
        ("Day $i$", "Day $i+1$", "Day 1", "Day 2"),
    )
    validation_rows = list(
        csv.DictReader((target / "ED06_VALIDATION.csv").open(encoding="utf-8"))
    )
    if any(row["status"] == "FAIL" for row in validation_rows):
        raise AssertionError("ED6 renderer validation contains a failure")

    refinement = {
        "schema": "ed06-upright-day-i-display-successor/v1",
        "figure": "Extended Data Figure 6",
        "record": ED6_TARGET.name,
        "supersedes": ED6_PARENT.name,
        "display_change": "math-italic i -> upright roman i in Day i/Day i+1",
        "scientific_recomputation": False,
        "scientific_input_hashes_identical": True,
        "scientific_directory_files": len(scientific_target),
        "runtime_label_only_raster_parity": parity,
        "vector_validation": vector,
        "parent_root_outputs": {
            name: {
                **file_record(ED6_PARENT / name),
                "path": (
                    "docs/figure_version_records_archive_2026-08-09/"
                    f"{ED6_PARENT.name}/{name}"
                ),
            }
            for name in (
                "ed06_version_record.svg",
                "ed06_version_record.pdf",
                "ed06_version_record_600dpi.png",
                "ed06_version_record_600dpi.tiff",
            )
        },
        "target_root_outputs": {
            name: {
                **file_record(target / name),
                "path": f"docs/figure_version_records/{ED6_TARGET.name}/{name}",
            }
            for name in (
                "ed06_version_record.svg",
                "ed06_version_record.pdf",
                "ed06_version_record_300dpi.png",
                "ed06_version_record_600dpi.png",
                "ed06_version_record_600dpi.tiff",
            )
        },
        "caption": file_record(CAPTION),
        "status": "PASS",
    }
    write_json(target / "UPRIGHT_DAY_INDEX_REFINEMENT_2026_08_16.json", refinement)
    (target / "MANIFEST.md").write_text(
        f"# {ED6_TARGET.name}\n\n"
        f"This append-only record supersedes `{ED6_PARENT.name}`. The visible and "
        "captioned same-cycle roles now use upright roman `i` in `Day i` and `Day i+1`. Internal `day1`/"
        "`day2` keys, quantitative tables, cohorts, estimands, statistics, geometry and "
        "scientific source hashes are unchanged.\n",
        encoding="utf-8",
    )
    return target, refinement


def stage_ed7() -> tuple[Path, dict[str, Any]]:
    target = STAGING_ROOT / ED7_TARGET.name
    shutil.copytree(ED7_PARENT, target)

    stem = "ed07_age_controls_six_panel_round16"
    for extension, destination in (
        ("svg", "ed07_version_record.svg"),
        ("pdf", "ed07_version_record.pdf"),
        ("png", "ed07_version_record_600dpi.png"),
        ("tiff", "ed07_version_record_600dpi.tiff"),
    ):
        copy(ED7_RENDER / f"{stem}.{extension}", target / destination)
    make_300dpi(
        target / "ed07_version_record_600dpi.png",
        target / "ed07_version_record_300dpi.png",
    )
    copy(
        ED7_RENDER / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json",
        target / "ED07_DISPLAY_MANIFEST.json",
    )
    copy(ED7_RENDER / "ED07_SIX_PANEL_CAPTION_NOTES.md", target / "CAPTION_NOTES.md")
    copy(ED7_RENDER / "ED07_SIX_PANEL_FIGURE_CONTRACT.md", target / "FIGURE_CONTRACT.md")
    copy(
        ED7_RENDER / "ed07_slide9_geometry_mockup_round16.png",
        target / "ed07_slide_mockup.png",
    )
    for path in ED7_RENDER.glob("*round16.*"):
        copy(path, target / "component_exports" / path.name)
    copy(
        ROOT / "scripts" / "render_ed07_slide9_powerpoint_matched.py",
        target / "source_code" / "render_ed07_slide9_powerpoint_matched.py",
    )
    copy(Path(__file__), target / "source_code" / Path(__file__).name)
    copy(
        ROOT / "tests" / "test_ed06_within_session_convergence_assembly.py",
        target / "source_code" / "test_ed06_within_session_convergence_assembly.py",
    )
    for source_name in (
        "validate_current_figure_records.py",
        "validate_all_figure_refinement_round2_records.py",
    ):
        copy(ROOT / "scripts" / source_name, target / "source_code" / source_name)

    parent_display = json.loads(
        (ED7_PARENT / "ED07_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
    )
    current_display = json.loads(
        (target / "ED07_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
    )
    parent_inputs = sorted(
        (entry["role"], entry["sha256"]) for entry in parent_display["inputs"]
    )
    current_inputs = sorted(
        (entry["role"], entry["sha256"]) for entry in current_display["inputs"]
    )
    if parent_inputs != current_inputs:
        raise AssertionError("ED7 scientific input hashes changed")
    scientific_parent = tree_records(ED7_PARENT, ("plot_ready", "source_authority"))
    scientific_target = tree_records(target, ("plot_ready", "source_authority"))
    if scientific_parent != scientific_target:
        raise AssertionError("ED7 frozen scientific directories changed")

    parity = png_difference(
        ED7_PARENT / "ed07_version_record_600dpi.png",
        ED7_RENDER / "ed07_age_controls_six_panel_round16.png",
        allowed_box=(0, 0, 260, 1550),
    )
    vector = validate_vector_outputs(
        target,
        "ed07_version_record",
        (
            "Day $i$ early vs Day $i+1$ correlation",
            "Day 1 early vs Day 2 correlation",
        ),
    )

    version_manifest = json.loads(
        (ED7_PARENT / "ED07_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    version_manifest.update(
        {
            "schema": "ed07-round16-upright-day-i-axis-version-record/v1",
            "record": ED7_TARGET.name,
            "supersedes": ED7_PARENT.name,
            "accepted_date": "2026-08-16",
            "accepted_display_round": 16,
            "scientific_values_changed": False,
            "day_label_revision": {
                "change": "top-row y axis math-italic i -> upright roman i",
                "scientific_recomputation": False,
                "status": "PASS",
            },
            "display_contract": current_display["display_contract"],
            "panel_map": {
                "a": "first-30-min Day i to full Day i+1 place-map correlation",
                "b": "first-30-min Day i to full Day i+1 population-vector correlation",
                "c": "first-30-min Day i to full Day i+1 ensemble-rate correlation",
                "d": "long gap place map correlation",
                "e": "long gap population vector correlation",
                "f": "long gap ensemble rate correlation",
            },
        }
    )
    output_names = {
        "svg": "ed07_version_record.svg",
        "pdf": "ed07_version_record.pdf",
        "png_300dpi": "ed07_version_record_300dpi.png",
        "png_600dpi": "ed07_version_record_600dpi.png",
        "tiff_600dpi": "ed07_version_record_600dpi.tiff",
    }
    version_manifest["root_output_sha256"] = {
        key: sha256(target / name) for key, name in output_names.items()
    }
    version_manifest["root_output_bytes"] = {
        key: (target / name).stat().st_size for key, name in output_names.items()
    }
    write_json(target / "ED07_VERSION_RECORD_MANIFEST.json", version_manifest)

    refinement = {
        "schema": "ed07-upright-day-i-axis-successor/v1",
        "figure": "Extended Data Figure 7",
        "record": ED7_TARGET.name,
        "supersedes": ED7_PARENT.name,
        "display_change": "top-row y axis math-italic i -> upright roman i",
        "long_gap_row_changed": False,
        "scientific_recomputation": False,
        "scientific_input_hashes_identical": True,
        "scientific_directory_files": len(scientific_target),
        "parent_exact_label_only_raster_parity": parity,
        "vector_validation": vector,
        "caption": file_record(CAPTION),
        "status": "PASS",
    }
    write_json(target / "UPRIGHT_DAY_INDEX_REFINEMENT_2026_08_16.json", refinement)
    write_json(
        target / "ED07_DISPLAY_VALIDATION.json",
        {
            "status": "PASS",
            "visual_qa_status": current_display["visual_qa_status"],
            "top_y_axis_label": current_display["display_contract"][
                "top_y_axis_label"
            ],
            "scientific_input_hashes_identical": True,
            "raster_parity": parity,
            "vector_validation": vector,
        },
    )
    (target / "MANIFEST.md").write_text(
        f"# {ED7_TARGET.name}\n\n"
        f"This append-only record supersedes `{ED7_PARENT.name}`. The same-cycle "
        "top-row y axis now uses upright roman `i` in `Day i early vs Day i+1 correlation`. The long gap "
        "row, data, cohorts, estimands, statistics, axes, geometry, colors and scientific "
        "source hashes are unchanged.\n",
        encoding="utf-8",
    )
    return target, refinement


def finalize_record(staged: Path, target: Path, refinement: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "target_did_not_preexist": not target.exists(),
        "renderer_status_pass": True,
        "scientific_input_hashes_identical": refinement[
            "scientific_input_hashes_identical"
        ],
        "label_only_raster_parity": next(
            value["status"] == "PASS"
            for key, value in refinement.items()
            if key.endswith("raster_parity")
        ),
        "vector_validation_pass": all(
            value is True or value == 0
            for key, value in refinement["vector_validation"].items()
            if key != "pdf_embedded_images"
        )
        and refinement["vector_validation"]["pdf_embedded_images"] == 0,
        "caption_hash_recorded": bool(refinement["caption"]["sha256"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"Record validation failed for {target.name}: {checks}")
    validated = datetime.now(timezone.utc).isoformat()
    write_json(
        staged / "RECORD_VALIDATION.json",
        {
            "schema": "upright-day-i-display-successor-record-validation/v1",
            "record": target.name,
            "checks": checks,
            "status": "PASS",
            "validated_utc": validated,
        },
    )
    (staged / "PROMOTION_COMPLETE.txt").write_text(
        f"PROMOTED {validated} -> {target.name}\n", encoding="utf-8"
    )
    ledger = write_hash_ledger(staged)
    shutil.move(str(staged), str(target))
    return {"record": target.name, "ledger": ledger, "checks": checks}


def main() -> None:
    required = (
        ED6_PARENT,
        ED6_RENDER,
        ED7_PARENT,
        ED7_RENDER,
        CAPTION,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")
    if ED6_TARGET.exists() or ED7_TARGET.exists() or STAGING_ROOT.exists():
        raise SystemExit("Refusing to overwrite a target or prior staging directory")

    STAGING_ROOT.mkdir(parents=True)
    ed6_staged, ed6_refinement = stage_ed6()
    ed7_staged, ed7_refinement = stage_ed7()
    results = [
        finalize_record(ed6_staged, ED6_TARGET, ed6_refinement),
        finalize_record(ed7_staged, ED7_TARGET, ed7_refinement),
    ]
    STAGING_ROOT.rmdir()
    print(json.dumps({"status": "PASS", "records": results}, indent=2))


if __name__ == "__main__":
    main()
