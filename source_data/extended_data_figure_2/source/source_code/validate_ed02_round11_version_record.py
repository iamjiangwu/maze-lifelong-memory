#!/usr/bin/env python3
"""Independent read-only validator for the ED2 Round 11 version record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round12_vector_vor"
)
EXPECTED_RECORD = "extended_data_fig02_2026-08-09_round12_vector_vor"
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
EXPECTED_PROVENANCE = {
    "natural/manifest.json",
    "natural/QA.md",
    "natural/REPORT.md",
    "strict/manifest.json",
    "strict/QA.md",
    "strict/REPORT.md",
}
PRIMARY_PDF = "ed02_round11_axis_aligned_version_record.pdf"
PRIMARY_SVG = "ed02_round11_axis_aligned_version_record.svg"
PRIMARY_PNG = "ed02_round11_axis_aligned_version_record_600dpi.png"
PRIMARY_TIFF = "ed02_round11_axis_aligned_version_record_600dpi.tiff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def check(self, name: str, passed: bool, detail: object) -> None:
        self.checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    def result(self) -> dict[str, object]:
        failures = [item for item in self.checks if item["status"] == "fail"]
        return {
            "status": "pass" if not failures else "fail",
            "check_count": len(self.checks),
            "failure_count": len(failures),
            "checks": self.checks,
        }


def close(actual: float, expected: float, tolerance: float = 1e-5) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def validate(record_root: Path) -> dict[str, object]:
    audit = Audit()
    audit.check("record_exists", record_root.is_dir(), str(record_root))
    if not record_root.is_dir():
        return audit.result()

    ledger_path = record_root / "RECORD_HASHES.csv"
    audit.check("hash_ledger_exists", ledger_path.is_file(), str(ledger_path))
    ledger_rows = list(csv.DictReader(ledger_path.open(encoding="utf-8-sig", newline=""))) if ledger_path.is_file() else []
    ledger_paths = [row["path"] for row in ledger_rows]
    actual_paths = sorted(
        p.relative_to(record_root).as_posix()
        for p in record_root.rglob("*")
        if p.is_file() and p.name != "RECORD_HASHES.csv"
    )
    audit.check("record_hash_closure_paths", sorted(ledger_paths) == actual_paths and len(set(ledger_paths)) == len(ledger_paths), {"ledger": len(ledger_paths), "actual": len(actual_paths)})
    for row in ledger_rows:
        path = record_root / row["path"]
        audit.check(
            f"record_hash::{row['path']}",
            path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]),
            row["sha256"],
        )

    version_manifest = json.loads((record_root / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("record_name", version_manifest.get("record") == EXPECTED_RECORD and record_root.name == EXPECTED_RECORD, version_manifest.get("record"))
    audit.check("accepted_status", version_manifest.get("status") == "accepted_version_record", version_manifest.get("status"))
    audit.check("panel_inventory_a_i_no_j", version_manifest.get("panel_inventory") == "a-i exactly; no panel j" and version_manifest.get("panel_count") == 9, version_manifest.get("panel_inventory"))
    audit.check("manual_mask_not_applicable", "not applicable" in version_manifest.get("manual_mask_applicability", "").lower(), version_manifest.get("manual_mask_applicability"))
    audit.check("fixed_30min_explore_contract", version_manifest.get("window_minutes") == 30 and "Explore-mode" in version_manifest.get("policy_events", ""), {"window": version_manifest.get("window_minutes"), "events": version_manifest.get("policy_events")})

    panel_files = sorted((record_root / "panels").glob("*"))
    audit.check("panel_export_count_36", len(panel_files) == 36, len(panel_files))
    stems_by_format: dict[str, set[str]] = {}
    for path in panel_files:
        stems_by_format.setdefault(path.suffix.lower(), set()).add(path.stem)
    audit.check("panel_formats_9_each", set(stems_by_format) == {".svg", ".pdf", ".png", ".tiff"} and all(len(stems) == 9 for stems in stems_by_format.values()), {key: len(value) for key, value in stems_by_format.items()})
    audit.check("panel_j_absent", not any("ed02j" in path.name.lower() for path in record_root.rglob("*")), "all paths")
    panel_manifest_rows = list(csv.DictReader((record_root / "PANEL_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("panel_manifest_36", len(panel_manifest_rows) == 36 and {row["panel"] for row in panel_manifest_rows} == set("abcdefghi"), len(panel_manifest_rows))
    for row in panel_manifest_rows:
        path = record_root / row["path"]
        audit.check(f"panel_manifest_hash::{row['path']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])

    tables = {path.name for path in (record_root / "plot_ready").glob("*.csv")}
    audit.check("plot_ready_exact_16", tables == EXPECTED_TABLES, sorted(tables))
    table_rows = list(csv.DictReader((record_root / "TABLE_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("table_manifest_16", len(table_rows) == 16 and {Path(row["path"]).name for row in table_rows} == EXPECTED_TABLES, len(table_rows))
    for row in table_rows:
        path = record_root / row["path"]
        audit.check(f"table_manifest_hash::{row['path']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])

    input_rows = list(csv.DictReader((record_root / "INPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    audit.check("analysis_input_inventory_13", len(input_rows) == 13 and len({row["source_key"] for row in input_rows}) == 13, len(input_rows))
    for row in input_rows:
        path = record_root / row["record_path"]
        audit.check(f"analysis_input_hash::{row['source_key']}", path.is_file() and sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"]), row["sha256"])
    provenance = {path.relative_to(record_root / "analysis_provenance").as_posix() for path in (record_root / "analysis_provenance").rglob("*") if path.is_file()}
    audit.check("analysis_provenance_exact_6", provenance == EXPECTED_PROVENANCE, sorted(provenance))

    source_manifest = json.loads((record_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("source_manifest_accepted", source_manifest.get("status") == "accepted_version_record" and source_manifest.get("record") == EXPECTED_RECORD, {"status": source_manifest.get("status"), "record": source_manifest.get("record")})
    audit.check("round11_contracts_present", all(key in source_manifest for key in ("panel_a_display_contract_round11", "axis_alignment_contract_round11", "category_spacing_contract_round11", "layout_contract_round11")), sorted(source_manifest))
    layout = source_manifest["layout_contract_round11"]
    labels = layout["panel_label_frames_720"]
    audit.check("round11_label_rows", all(close(labels[p]["top"], expected) for group, expected in (("abc", 27.5), ("def", 229.5), ("ghi", 473.5)) for p in group), labels)
    axis = source_manifest["axis_alignment_contract_round11"]
    audit.check("row2_axis_alignment", close(axis["row2"]["x_axis_baseline"], 411.225080) and close(axis["row2"]["matched_axes_top"], 232.345648), axis["row2"])
    audit.check("row3_axis_alignment", close(axis["row3"]["x_axis_baseline"], 649.986769) and close(axis["row3"]["matched_axes_top"], 501.645690), axis["row3"])
    spacing = source_manifest["category_spacing_contract_round11"]
    audit.check("category_spacing_reduced", spacing["d"]["reduction_fraction"] >= 0.20 and spacing["h"]["reduction_fraction"] >= 0.24, spacing)
    schematic = source_manifest["panel_a_display_contract_round11"]
    audit.check("schematic_0p8_clear", close(schematic["grid_scale_vs_round10"], 0.8) and schematic["preceding_turn"]["minimum_grid_clearance_source_pt"] >= 3.5, schematic)

    composite_manifest = json.loads((record_root / "ED02_ROUND11_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
    audit.check("composite_manifest_pass", composite_manifest.get("status") == "pass" and composite_manifest.get("panel_order") == list("abcdefghi"), composite_manifest.get("panel_order"))
    for key, name in (("pdf", PRIMARY_PDF), ("svg", PRIMARY_SVG), ("png", PRIMARY_PNG), ("tiff", PRIMARY_TIFF)):
        output = composite_manifest["outputs"][key]
        path = record_root / name
        audit.check(f"composite_manifest_hash::{key}", output["path"] == name and path.is_file() and sha256(path) == output["sha256"] and path.stat().st_size == int(output["bytes"]), output)
    with fitz.open(record_root / PRIMARY_PDF) as document:
        page = document[0]
        drawings = len(page.get_drawings())
        images = len(page.get_images(full=True))
        text = page.get_text()
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block for line in block["lines"] for span in line["spans"]]
        letter_spans = [span for span in spans if span["text"] in set("abcdefghi") and abs(float(span["size"]) - 10.5) < 0.15 and "Arial" in span["font"]]
        audit.check("primary_pdf_one_page_7p5in", document.page_count == 1 and close(page.rect.width, 540.0, 1e-6) and close(page.rect.height, 540.0, 1e-6), [document.page_count, page.rect.width, page.rect.height])
        audit.check("primary_pdf_true_vector", drawings > 900 and images == 0 and len(text) > 500, {"drawings": drawings, "images": images, "characters": len(text)})
        audit.check("primary_pdf_panel_letters", len(letter_spans) == 9, [{"text": span["text"], "font": span["font"], "size": span["size"]} for span in letter_spans])
    primary_svg = (record_root / PRIMARY_SVG).read_text(encoding="utf-8")
    audit.check("primary_svg_editable_vector", primary_svg.count("<text") > 100 and "<image" not in primary_svg, {"text_tags": primary_svg.count("<text"), "image_tags": primary_svg.count("<image")})
    audit.check("primary_svg_self_contained", not any(token in primary_svg for token in ('href="http', "href='http", 'xlink:href="http', "xlink:href='http")), "no external href")
    for name in (PRIMARY_PNG, PRIMARY_TIFF):
        with Image.open(record_root / name) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"high_resolution::{name}", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})

    for path in sorted((record_root / "panels").glob("*.png")) + sorted((record_root / "panels").glob("*.tiff")):
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"panel_raster_600dpi::{path.name}", min(dpi) >= 599.0, dpi)
    editable_svgs = sorted((record_root / "panels").glob("*.svg")) + sorted((record_root / "shared_assets").glob("*.svg"))
    audit.check("editable_svg_text", all("<text" in path.read_text(encoding="utf-8") for path in editable_svgs), [path.name for path in editable_svgs])
    outlined_svgs = sorted((record_root / "ppt_vector").glob("*.svg"))
    audit.check("outlined_svg_count_10", len(outlined_svgs) == 10 and all("<text" not in path.read_text(encoding="utf-8") for path in outlined_svgs), len(outlined_svgs))

    for pptx in ("sup-figs-v2_ed02-slide3_round11_review.pptx", "ed02-slide3_round11_review.pptx"):
        with zipfile.ZipFile(record_root / "review_evidence" / pptx) as archive:
            svg_media = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
            audit.check(f"powerpoint_vector_inventory::{pptx}", len(svg_media) == 10, len(svg_media))

    saved_validation = json.loads((record_root / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    audit.check("saved_validation_pass", saved_validation.get("status") == "pass" and saved_validation.get("failure_count") == 0, {"status": saved_validation.get("status"), "failures": saved_validation.get("failure_count")})
    audit.check("promotion_complete", (record_root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8").startswith("PASS\n"), (record_root / "PROMOTION_COMPLETE.txt").read_text(encoding="utf-8"))
    bytecode = [path.relative_to(record_root).as_posix() for path in record_root.rglob("*") if path.suffix == ".pyc" or "__pycache__" in path.parts]
    audit.check("no_bytecode_or_cache", not bytecode, bytecode)
    return audit.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args()
    result = validate(args.record_root.resolve())
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"ED02 ROUND11 VERSION RECORD: {result['status'].upper()}", f"checks={result['check_count']}", f"failures={result['failure_count']}"]
        lines.extend(f"{item['status'].upper()} {item['check']}: {item['detail']}" for item in result["checks"])
        args.text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
