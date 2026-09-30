#!/usr/bin/env python3
"""Promote the accepted ED2 Round 11 figure as an immutable version record.

The promoter refuses to overwrite an existing record, assembles in a sibling
staging directory, writes validation/completion metadata before the hash
ledger, and atomically renames the completed staging directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed02_route_policies_and_biased_walk"
    / "assembly_round11_review"
)
REVIEW_ROOT = SOURCE_ROOT.parent / "review_2026-08-09_round11"
PREVIOUS_RECORD = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round11_axis_aligned_vor"
)
COMPOSITE_ROOT = REPO_ROOT / "tmp" / "ed02_round11_vor_2026-08-09" / "composite"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round12_vector_vor"
)
COMPOSITE_FILES = (
    "ed02_round11_axis_aligned_version_record.pdf",
    "ed02_round11_axis_aligned_version_record.svg",
    "ed02_round11_axis_aligned_version_record_600dpi.png",
    "ed02_round11_axis_aligned_version_record_600dpi.tiff",
    "ed02_round11_axis_aligned_version_record_preview_1440.png",
    "ED02_ROUND11_COMPOSITE_MANIFEST.json",
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
    "sup-figs-v2_ed02-slide3_round11_review.pptx",
    "ed02-slide3_round11_review.pptx",
    "ed02-slide3-round11_powerpoint_render.png",
    "ROUND11_REVIEW_VALIDATION.json",
    "ROUND11_REVIEW_VALIDATION.txt",
)
EXPECTED_TABLES = {
    "panel_a_schematic_rows.csv",
    "panel_b_contrasts.csv",
    "panel_b_session_points.csv",
    "panel_c_contrasts.csv",
    "panel_c_session_points.csv",
    "panel_d_policy_contributions.csv",
    "panel_e_distance_from_return.csv",
    "panel_e_group_means.csv",
    "panel_f_calibration_summary.csv",
    "panel_f_session_points.csv",
    "panel_g_depth_contrasts.csv",
    "panel_g_group_profiles.csv",
    "panel_g_supported_session_rows.csv",
    "panel_h_four_vs_depth_gap.csv",
    "panel_i_calibration_summary.csv",
    "panel_i_session_points.csv",
}


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hash_ledger(root: Path) -> None:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "RECORD_HASHES.csv"):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    with (root / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(entries)


def copy_tree_files(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source))


def audit_stage(stage: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    panel_files = [p for p in (stage / "panels").iterdir() if p.is_file()]
    check("panel_exports_36", len(panel_files) == 36, len(panel_files))
    check("panel_formats", {p.suffix.lower() for p in panel_files} == {".svg", ".pdf", ".png", ".tiff"}, sorted({p.suffix for p in panel_files}))
    tables = {p.name for p in (stage / "plot_ready").glob("*.csv")}
    check("plot_ready_16", tables == EXPECTED_TABLES, sorted(tables))
    check("outlined_vectors_10", len(list((stage / "ppt_vector").glob("*.svg"))) == 10, len(list((stage / "ppt_vector").glob("*.svg"))))
    check("shared_assets_4", len(list((stage / "shared_assets").iterdir())) == 4, len(list((stage / "shared_assets").iterdir())))
    input_rows = list(csv.DictReader((stage / "INPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    input_ok = len(input_rows) == 13 and all(
        (stage / row["record_path"]).is_file()
        and sha256(stage / row["record_path"]) == row["sha256"]
        and (stage / row["record_path"]).stat().st_size == int(row["bytes"])
        for row in input_rows
    )
    check("analysis_inputs_13", input_ok, len(input_rows))
    check("analysis_provenance_6", len(list((stage / "analysis_provenance").rglob("*.*"))) == 6, len(list((stage / "analysis_provenance").rglob("*.*"))))
    pdf_path = stage / COMPOSITE_FILES[0]
    with fitz.open(pdf_path) as document:
        page = document[0]
        pdf_detail = {
            "pages": document.page_count,
            "points": [page.rect.width, page.rect.height],
            "drawings": len(page.get_drawings()),
            "images": len(page.get_images(full=True)),
            "text_characters": len(page.get_text()),
        }
        pdf_ok = (
            document.page_count == 1
            and abs(page.rect.width - 540.0) < 1e-6
            and abs(page.rect.height - 540.0) < 1e-6
            and pdf_detail["drawings"] > 900
            and pdf_detail["images"] == 0
            and pdf_detail["text_characters"] > 500
        )
    check("composite_true_vector_pdf", pdf_ok, pdf_detail)
    svg_text = (stage / COMPOSITE_FILES[1]).read_text(encoding="utf-8")
    check("composite_editable_svg", svg_text.count("<text") > 100 and "<image" not in svg_text, {"text_tags": svg_text.count("<text"), "image_tags": svg_text.count("<image")})
    for name in COMPOSITE_FILES[2:4]:
        with Image.open(stage / name) as image:
            dpi = tuple(float(v) for v in image.info.get("dpi", (0, 0)))
            check(f"{Path(name).suffix[1:]}_600dpi", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})
    check("no_bytecode", not any(p.suffix == ".pyc" or "__pycache__" in p.parts for p in stage.rglob("*")), "record tree")
    failures = [item for item in checks if item["status"] == "fail"]
    return {
        "status": "pass" if not failures else "fail",
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def promote(record_root: Path) -> dict[str, object]:
    if record_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing record: {record_root}")
    stage = record_root.with_name(f".{record_root.name}.staging-{uuid.uuid4().hex[:10]}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        for directory in ("panels", "plot_ready", "ppt_vector", "shared_assets"):
            copy_tree_files(SOURCE_ROOT / directory, stage / directory)
        copy_tree_files(PREVIOUS_RECORD / "analysis_inputs", stage / "analysis_inputs")
        copy_tree_files(PREVIOUS_RECORD / "analysis_provenance", stage / "analysis_provenance")
        copy_tree_files(PREVIOUS_RECORD / "source_code", stage / "source_code")
        copy_tree_files(SOURCE_ROOT / "source_code", stage / "source_code")
        for name in TOP_SOURCE_FILES:
            copy_file(SOURCE_ROOT / name, stage / name)
        for name in ("INPUT_MANIFEST.csv", "UPSTREAM_INPUT_MANIFEST.csv"):
            copy_file(PREVIOUS_RECORD / name, stage / name)
        for name in COMPOSITE_FILES:
            copy_file(COMPOSITE_ROOT / name, stage / name)
        for name in REVIEW_FILES:
            copy_file(REVIEW_ROOT / name, stage / "review_evidence" / name)
        for script in (
            Path(__file__),
            REPO_ROOT / "scripts" / "export_ed02_round11_vector_figure.py",
            REPO_ROOT / "scripts" / "validate_ed02_round11_version_record.py",
        ):
            copy_file(script, stage / "source_code" / script.name)

        source_manifest = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        source_manifest.update(
            {
                "status": "accepted_version_record",
                "record": record_root.name,
                "accepted_date": "2026-08-09",
                "primary_publication_output": COMPOSITE_FILES[0],
                "vector_companion": COMPOSITE_FILES[1],
                "raster_companions": list(COMPOSITE_FILES[2:4]),
                "frozen_renderer": {
                    "path": "source_code/render_ed03_fixed_time_30min_full_story.py",
                    "sha256": sha256(stage / "source_code" / "render_ed03_fixed_time_30min_full_story.py"),
                },
            }
        )
        write_json(stage / "SOURCE_MANIFEST.json", source_manifest)

        source_rows = []
        for path in sorted((stage / "source_code").glob("*")):
            if path.is_file():
                source_rows.append({"path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
        with (stage / "SOURCE_CODE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_rows)

        composite_manifest = json.loads((stage / "ED02_ROUND11_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
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
            "review_round": 11,
            "primary_output": COMPOSITE_FILES[0],
            "vector_outputs": list(COMPOSITE_FILES[:2]),
            "high_resolution_outputs": list(COMPOSITE_FILES[2:4]),
            "vector_audit": composite_manifest["vector_audit"],
            "raster_audit": {"png": composite_manifest["png_audit"], "tiff": composite_manifest["tiff_audit"]},
            "display_changes": source_manifest["presentation_changes"],
            "panel_e_estimand": source_manifest["panel_e_estimand"],
        }
        write_json(stage / "ED02_VERSION_RECORD_MANIFEST.json", version_manifest)
        write_json(
            stage / "RENDER_RESULT.json",
            {
                "status": "pass",
                "renderer": "source_code/render_ed02_slide3_aligned_round11.py",
                "vector_assembler": "source_code/export_ed02_round11_vector_figure.py",
                "output_root": ".",
                "primary_output": COMPOSITE_FILES[0],
                "vector_outputs": list(COMPOSITE_FILES[:2]),
                "raster_outputs": list(COMPOSITE_FILES[2:4]),
            },
        )
        (stage / "MANIFEST.md").write_text(
            "# Extended Data Figure 2 | Round 11 version record\n\n"
            "Accepted 2026-08-09. This immutable record freezes the exact nine-panel a-i Round 11 figure, "
            "its 16 plot-ready tables, 13 analysis inputs, source renderers, and review decks. The primary "
            "publication master is a one-page vector PDF with zero embedded raster images; 600-dpi PNG and "
            "TIFF companions are included. ED2 is behavior-only, so neural manual masks are not applicable.\n",
            encoding="utf-8",
        )
        (stage / "FIGURE_LEGEND.md").write_text(
            "# Extended Data Figure 2\n\n"
            "Route-policy biases during the first 30 cumulative minutes in the maze. (a) Four policy "
            "definitions. (b,c) Session probabilities across Naive day, Return day, and Second day. "
            "(d) Signed exploration-efficiency differences after one-policy substitutions. (e) Absolute "
            "policy-wise distances from Return-day means. (f) Four-policy model calibration. (g) Depth "
            "profile of P_BS. (h) Four-policy and depth-varying reconstructions. (i) Depth-varying model "
            "calibration. Counts, uncertainty definitions, and tests remain in CAPTION_NOTES.md.\n",
            encoding="utf-8",
        )
        (stage / "REPORT.md").write_text(
            "# Promotion report\n\n"
            "Round 11 was promoted after vector assembly, 600-dpi raster export, scientific source/table "
            "parity checks, source-level axis alignment, and review-deck validation. The accepted panel "
            "inventory is a-i exactly; no panel j was added.\n",
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
        write_hash_ledger(stage)
        stage.rename(record_root)
        return {
            "status": "pass",
            "record": record_root.relative_to(REPO_ROOT).as_posix(),
            "record_files": len([p for p in record_root.rglob("*") if p.is_file()]),
            "hash_entries": sum(1 for _ in csv.DictReader((record_root / "RECORD_HASHES.csv").open(encoding="utf-8", newline=""))),
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    args = parser.parse_args()
    print(json.dumps(promote(args.record_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
