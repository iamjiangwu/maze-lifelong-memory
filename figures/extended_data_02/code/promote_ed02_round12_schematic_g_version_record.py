#!/usr/bin/env python3
"""Promote the ED2 Round-12 schematic/panel-g correction immutably."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
import zipfile

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "panels"
COMPOSITE_ROOT = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "composite"
DECK_ROOT = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "deck"
FIX_VALIDATION = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "ROUND12_FIX_VALIDATION.json"
PREVIOUS_RECORD = REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig02_2026-08-09_round12_vector_vor"
DEFAULT_RECORD_ROOT = REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor"
COMPOSITE_FILES = (
    "ed02_round12_schematic_g_fix_version_record.pdf",
    "ed02_round12_schematic_g_fix_version_record.svg",
    "ed02_round12_schematic_g_fix_version_record_600dpi.png",
    "ed02_round12_schematic_g_fix_version_record_600dpi.tiff",
    "ed02_round12_schematic_g_fix_version_record_preview_1440.png",
    "ED02_ROUND12_COMPOSITE_MANIFEST.json",
)
TOP_SOURCE_FILES = (
    "CAPTION_NOTES.md",
    "PANEL_MANIFEST.csv",
    "PPT_VECTOR_MANIFEST.csv",
    "SHARED_ASSET_MANIFEST.csv",
    "SOURCE_MANIFEST.json",
    "TABLE_MANIFEST.csv",
)
REVIEW_FILES = (
    "sup-figs-v2_ed02-slide3_round12_review.pptx",
    "ed02-slide3_round12_review.pptx",
    "cumulative-artifact-preview.png",
    "review-artifact-preview.png",
    "cumulative-after-inspect.ndjson",
    "review-after-inspect.ndjson",
    "cumulative.layout.json",
    "review.layout.json",
    "round12-build-report.json",
)
SOURCE_SCRIPTS = (
    "render_ed02_slide3_schematic_g_fix_round12.py",
    "render_ed02_slide3_aligned_round11.py",
    "export_ed02_round11_vector_figure.py",
    "export_ed02_round12_vector_figure.py",
    "build_ed02_slide3_round12_review.mjs",
    "validate_ed02_round12_fix.py",
    "validate_ed02_round12_schematic_g_version_record.py",
    "promote_ed02_round12_schematic_g_version_record.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hash_ledger(root: Path) -> int:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "RECORD_HASHES.csv")
    ]
    with (root / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def audit_stage(stage: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    check("panel_exports_36", len(list((stage / "panels").glob("*"))) == 36, len(list((stage / "panels").glob("*"))))
    check("plot_ready_16", len(list((stage / "plot_ready").glob("*.csv"))) == 16, len(list((stage / "plot_ready").glob("*.csv"))))
    check("ppt_vectors_10", len(list((stage / "ppt_vector").glob("*.svg"))) == 10, len(list((stage / "ppt_vector").glob("*.svg"))))
    check("shared_assets_4", len(list((stage / "shared_assets").glob("*"))) == 4, len(list((stage / "shared_assets").glob("*"))))
    inputs = list(csv.DictReader((stage / "INPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    input_ok = len(inputs) == 13 and all(
        (stage / row["record_path"]).is_file()
        and sha256(stage / row["record_path"]) == row["sha256"]
        and (stage / row["record_path"]).stat().st_size == int(row["bytes"])
        for row in inputs
    )
    check("analysis_inputs_13", input_ok, len(inputs))
    provenance = [path for path in (stage / "analysis_provenance").rglob("*") if path.is_file()]
    check("analysis_provenance_6", len(provenance) == 6, len(provenance))
    manifest = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    check("panel_a_contract", manifest["panel_a_display_contract_round12"]["style"]["arrow_mutation_scale"] == 6.0 and manifest["panel_a_display_contract_round12"]["point_artist_scale_vs_round11"] == 0.8, manifest["panel_a_display_contract_round12"])
    check("panel_g_contract", manifest["panel_g_display_contract_round12"]["group_positions"] == [0.28, 1.0, 1.72] and manifest["panel_g_display_contract_round12"]["source_values_changed"] is False, manifest["panel_g_display_contract_round12"])
    authority_text = "\n".join((stage / name).read_text(encoding="utf-8") for name in ("SOURCE_MANIFEST.json", "RENDER_RESULT.json", "ED02_VERSION_RECORD_MANIFEST.json"))
    check("authority_paths_portable", "\\tmp\\" not in authority_text.lower() and "/tmp/" not in authority_text.lower() and "D:\\" not in authority_text, "record-local authority")
    with fitz.open(stage / COMPOSITE_FILES[0]) as document:
        page = document[0]
        check("true_vector_pdf", document.page_count == 1 and page.rect.width == 540 and page.rect.height == 540 and len(page.get_images(full=True)) == 0 and len(page.get_drawings()) > 900, {"pages": document.page_count, "images": len(page.get_images(full=True)), "drawings": len(page.get_drawings())})
    svg = (stage / COMPOSITE_FILES[1]).read_text(encoding="utf-8")
    check("editable_composite_svg", svg.count("<text") > 100 and "<image" not in svg, {"text": svg.count("<text"), "image": svg.count("<image")})
    for name in COMPOSITE_FILES[2:4]:
        with Image.open(stage / name) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            check(f"high_resolution::{name}", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})
    for pptx in REVIEW_FILES[:2]:
        with zipfile.ZipFile(stage / "review_evidence" / pptx) as archive:
            vectors = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
        check(f"review_vector_inventory::{pptx}", len(vectors) == 10, len(vectors))
    bytecode = [path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.suffix == ".pyc" or "__pycache__" in path.parts]
    check("no_bytecode", not bytecode, bytecode)
    failures = [item for item in checks if item["status"] == "fail"]
    return {"status": "pass" if not failures else "fail", "check_count": len(checks), "failure_count": len(failures), "checks": checks}


def promote(record_root: Path) -> dict[str, object]:
    if record_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing record: {record_root}")
    stage = record_root.with_name(f".{record_root.name}.staging-{uuid.uuid4().hex[:10]}")
    stage.mkdir(parents=True)
    try:
        for directory in ("panels", "plot_ready", "ppt_vector", "shared_assets"):
            copy_tree(SOURCE_ROOT / directory, stage / directory)
        copy_tree(PREVIOUS_RECORD / "analysis_inputs", stage / "analysis_inputs")
        copy_tree(PREVIOUS_RECORD / "analysis_provenance", stage / "analysis_provenance")
        copy_tree(PREVIOUS_RECORD / "source_code", stage / "source_code")
        for name in TOP_SOURCE_FILES:
            copy_file(SOURCE_ROOT / name, stage / name)
        for name in ("INPUT_MANIFEST.csv", "UPSTREAM_INPUT_MANIFEST.csv"):
            copy_file(PREVIOUS_RECORD / name, stage / name)
        for name in COMPOSITE_FILES:
            copy_file(COMPOSITE_ROOT / name, stage / name)
        for name in REVIEW_FILES:
            copy_file(DECK_ROOT / name, stage / "review_evidence" / name)
        copy_file(FIX_VALIDATION, stage / "review_evidence" / "ROUND12_FIX_VALIDATION.json")
        for name in SOURCE_SCRIPTS:
            copy_file(REPO_ROOT / "scripts" / name, stage / "source_code" / name)

        source_manifest = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        source_manifest.update(
            {
                "status": "accepted_version_record",
                "record": record_root.name,
                "accepted_date": "2026-08-09",
                "source_record": PREVIOUS_RECORD.name,
                "primary_publication_output": COMPOSITE_FILES[0],
                "vector_companion": COMPOSITE_FILES[1],
                "raster_companions": list(COMPOSITE_FILES[2:4]),
                "frozen_renderer": {
                    "path": "source_code/render_ed02_slide3_schematic_g_fix_round12.py",
                    "sha256": sha256(stage / "source_code" / "render_ed02_slide3_schematic_g_fix_round12.py"),
                },
            }
        )
        write_json(stage / "SOURCE_MANIFEST.json", source_manifest)

        source_rows = [
            {"path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted((stage / "source_code").glob("*"))
            if path.is_file()
        ]
        with (stage / "SOURCE_CODE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_rows)

        composite_manifest = json.loads((stage / "ED02_ROUND12_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
        version_manifest = {
            "status": "accepted_version_record",
            "record": record_root.name,
            "accepted_date": "2026-08-09",
            "figure": "Extended Data Figure 2",
            "panel_inventory": "a-i exactly; no panel j",
            "panel_count": 9,
            "plot_ready_table_count": 16,
            "analysis_input_count": 13,
            "analysis_provenance_file_count": 6,
            "manual_mask_applicability": "not applicable; behavior-only analysis",
            "window_minutes": 30,
            "clock": "first 30 cumulative TimeInMaze minutes; all modes advance the clock",
            "policy_events": "Explore-mode junction decisions only",
            "source_record": PREVIOUS_RECORD.name,
            "review_round": 12,
            "primary_output": COMPOSITE_FILES[0],
            "vector_outputs": list(COMPOSITE_FILES[:2]),
            "high_resolution_outputs": list(COMPOSITE_FILES[2:4]),
            "vector_audit": composite_manifest["vector_audit"],
            "raster_audit": {"png": composite_manifest["png_audit"], "tiff": composite_manifest["tiff_audit"]},
            "display_changes": [
                "panel a point-sized arrow strokes, arrowheads, and text scaled by the same 0.8 factor as the grid",
                "panel a preceding-turn annotation cleared from the grid",
                "panel g shallow/middle/deep positions compressed to 72% and y-axis title/tick clearance restored",
            ],
            "scientific_values_changed": False,
        }
        write_json(stage / "ED02_VERSION_RECORD_MANIFEST.json", version_manifest)
        write_json(
            stage / "RENDER_RESULT.json",
            {
                "status": "pass",
                "renderer": "source_code/render_ed02_slide3_schematic_g_fix_round12.py",
                "vector_assembler": "source_code/export_ed02_round12_vector_figure.py",
                "review_deck_builder": "source_code/build_ed02_slide3_round12_review.mjs",
                "output_root": ".",
                "primary_output": COMPOSITE_FILES[0],
                "vector_outputs": list(COMPOSITE_FILES[:2]),
                "raster_outputs": list(COMPOSITE_FILES[2:4]),
            },
        )
        (stage / "MANIFEST.md").write_text(
            "# Extended Data Figure 2 | Round 12 corrected version record\n\n"
            "Accepted 2026-08-09. This immutable record freezes the exact a-i figure after proportional "
            "schematic-artist scaling in panel a and group-spacing/y-label correction in panel g. The "
            "primary PDF and SVG are vector; PNG and LZW-TIFF companions are 4500 x 4500 pixels at 600 dpi.\n",
            encoding="utf-8",
        )
        (stage / "FIGURE_LEGEND.md").write_text(
            "# Extended Data Figure 2\n\n"
            "Route-policy biases during the first 30 cumulative minutes in the maze. (a) Four policy "
            "definitions. (b,c) Session probabilities across Naive day, Return day, and Second day. "
            "(d) Signed exploration-efficiency differences after one-policy substitutions. (e) Policy-wise "
            "distances from Return-day means. (f) Four-policy model calibration. (g) Depth profile of P_BS. "
            "(h) Four-policy and depth-varying reconstructions. (i) Depth-varying model calibration.\n",
            encoding="utf-8",
        )
        (stage / "REPORT.md").write_text(
            "# Promotion report\n\n"
            "Panel a now scales strokes, arrowheads, and schematic text with its 0.8x grid, with measured "
            "annotation clearance. Panel g preserves all accepted values while compacting group positions "
            "and clearing both y-axis titles from their tick labels. All 16 plot-ready tables are byte-identical "
            "to the preceding accepted record.\n",
            encoding="utf-8",
        )
        audit = audit_stage(stage)
        if audit["status"] != "pass":
            raise RuntimeError(json.dumps(audit, indent=2))
        write_json(stage / "AUDIT_PROMOTION.json", audit)
        write_json(stage / "RECORD_VALIDATION.json", audit)
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={record_root.name}\nutc={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        entries = write_hash_ledger(stage)
        stage.rename(record_root)
        return {
            "status": "pass",
            "record": record_root.relative_to(REPO_ROOT).as_posix(),
            "record_files": len([path for path in record_root.rglob("*") if path.is_file()]),
            "hash_entries": entries,
        }
    except Exception:
        if stage.exists() and stage.resolve().parent == record_root.resolve().parent:
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    args = parser.parse_args()
    print(json.dumps(promote(args.record_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
