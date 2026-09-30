#!/usr/bin/env python3
"""Fail-closed validation for the ED2 Round-12 schematic/panel-g correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "panels"
DEFAULT_COMPOSITE = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "composite"
DEFAULT_DECK = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "deck"
BASELINE = REPO_ROOT / "docs" / "figure_version_records" / "extended_data_fig02_2026-08-09_round12_vector_vor"
COMPOSITE_STEM = "ed02_round12_schematic_g_fix_version_record"


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


def pdf_spans(path: Path) -> list[dict[str, object]]:
    with fitz.open(path) as document:
        return [
            span
            for block in document[0].get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]


def validate(source: Path, composite: Path, deck: Path) -> dict[str, object]:
    audit = Audit()
    for label, root in (("source", source), ("composite", composite), ("deck", deck), ("baseline", BASELINE)):
        audit.check(f"{label}_root_exists", root.is_dir(), str(root))
    if not all(root.is_dir() for root in (source, composite, deck, BASELINE)):
        return audit.result()

    current_tables = sorted((source / "plot_ready").glob("*.csv"))
    baseline_tables = {path.name: path for path in (BASELINE / "plot_ready").glob("*.csv")}
    audit.check("plot_ready_exact_16", len(current_tables) == 16 and {path.name for path in current_tables} == set(baseline_tables), [path.name for path in current_tables])
    for path in current_tables:
        audit.check(f"table_byte_parity::{path.name}", sha256(path) == sha256(baseline_tables[path.name]), sha256(path))

    panel_files = sorted((source / "panels").glob("*"))
    audit.check("panel_exports_36", len(panel_files) == 36, len(panel_files))
    audit.check("panel_formats", {path.suffix.lower() for path in panel_files} == {".svg", ".pdf", ".png", ".tiff"}, sorted({path.suffix.lower() for path in panel_files}))
    audit.check("ppt_vectors_10", len(list((source / "ppt_vector").glob("*.svg"))) == 10, len(list((source / "ppt_vector").glob("*.svg"))))
    for path in sorted((source / "panels").glob("*.png")) + sorted((source / "panels").glob("*.tiff")):
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"panel_600dpi::{path.name}", min(dpi) >= 599.0, dpi)
    for path in sorted((source / "panels").glob("*.svg")):
        text = path.read_text(encoding="utf-8")
        audit.check(f"editable_svg::{path.name}", "<text" in text and "<image" not in text, {"text": text.count("<text"), "image": text.count("<image")})

    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    panel_a = manifest.get("panel_a_display_contract_round12", {})
    panel_g = manifest.get("panel_g_display_contract_round12", {})
    expected_style = {
        "cell_linewidth": 0.72,
        "arrow_linewidth": 1.0,
        "previous_arrow_linewidth": 1.0,
        "arrow_mutation_scale": 6.0,
        "title_fontsize": 5.52,
        "or_fontsize": 5.12,
        "annotation_fontsize": 4.96,
    }
    audit.check("panel_a_proportional_contract", panel_a.get("grid_scale_vs_round10") == 0.8 and panel_a.get("point_artist_scale_vs_round11") == 0.8 and panel_a.get("style") == expected_style, panel_a)
    audit.check("panel_g_compact_contract", panel_g.get("group_position_scale_vs_round11") == 0.72 and panel_g.get("group_positions") == [0.28, 1.0, 1.72] and panel_g.get("source_values_changed") is False, panel_g)
    renderer_source = (REPO_ROOT / "scripts" / "render_ed02_slide3_schematic_g_fix_round12.py").read_text(encoding="utf-8")
    for token in ("SCHEMATIC_ARTIST_SCALE = 0.8", "PANEL_G_GROUP_SCALE = 0.72", '"arrow_mutation_scale": 7.5 * SCHEMATIC_ARTIST_SCALE'):
        audit.check(f"renderer_contract::{token}", token in renderer_source, token)

    panel_a_pdf = source / "panels" / "ed02a_four_policy_schematic_round11.pdf"
    with fitz.open(panel_a_pdf) as document:
        page = document[0]
        annotation = next((fitz.Rect(span["bbox"]) for span in pdf_spans(panel_a_pdf) if span["text"] == "preceding turn"), None)
        candidates = [drawing["rect"] for drawing in page.get_drawings() if annotation and drawing["rect"].y0 >= annotation.y1 and drawing["rect"].x1 >= annotation.x0 and drawing["rect"].x0 <= annotation.x1]
        clearance = min((rect.y0 - annotation.y1 for rect in candidates), default=-1.0)
    audit.check("panel_a_preceding_turn_present", annotation is not None, annotation and list(annotation))
    audit.check("panel_a_preceding_turn_clearance", clearance >= 3.5, clearance)

    g_spans = pdf_spans(source / "panels" / "ed02g_depth_pbs_profile_round11.pdf")
    gaps = []
    for y_min, y_max in ((0.0, 70.0), (70.0, 150.0)):
        labels = [fitz.Rect(span["bbox"]) for span in g_spans if y_min <= float(span["bbox"][1]) < y_max and float(span["bbox"][2]) <= 14.0]
        ticks = [fitz.Rect(span["bbox"]) for span in g_spans if y_min <= float(span["bbox"][1]) < y_max and 14.0 < float(span["bbox"][0]) < 28.0 and any(ch.isdigit() for ch in str(span["text"]))]
        gaps.append(min((tick.x0 for tick in ticks), default=-1.0) - max((label.x1 for label in labels), default=999.0))
    audit.check("panel_g_ytitle_tick_nonoverlap", len(gaps) == 2 and min(gaps) >= 1.5, gaps)

    manifest_path = composite / "ED02_ROUND12_COMPOSITE_MANIFEST.json"
    composite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit.check("composite_manifest_pass", composite_manifest.get("status") == "pass" and composite_manifest.get("review_round") == 12 and composite_manifest.get("panel_order") == list("abcdefghi"), composite_manifest.get("panel_order"))
    pdf_path = composite / f"{COMPOSITE_STEM}.pdf"
    svg_path = composite / f"{COMPOSITE_STEM}.svg"
    with fitz.open(pdf_path) as document:
        page = document[0]
        audit.check("composite_true_vector_pdf", document.page_count == 1 and page.rect.width == 540 and page.rect.height == 540 and len(page.get_images(full=True)) == 0 and len(page.get_drawings()) > 900, {"pages": document.page_count, "images": len(page.get_images(full=True)), "drawings": len(page.get_drawings())})
    svg_text = svg_path.read_text(encoding="utf-8")
    audit.check("composite_editable_svg", svg_text.count("<text") > 100 and "<image" not in svg_text, {"text": svg_text.count("<text"), "image": svg_text.count("<image")})
    for suffix in ("png", "tiff"):
        path = composite / f"{COMPOSITE_STEM}_600dpi.{suffix}"
        with Image.open(path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            audit.check(f"composite_{suffix}_600dpi", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})

    build_report = json.loads((deck / "round12-build-report.json").read_text(encoding="utf-8"))
    audit.check("deck_build_pass", build_report.get("status") == "pass" and len(build_report.get("results", [])) == 2, build_report.get("status"))
    expected_vector_hashes = {sha256(source / "ppt_vector" / f"ed02{panel}_{stem}.svg") for panel, stem in (("a", "four_policy_schematic_round11"), ("g", "depth_pbs_profile_round11"))}
    for name in ("sup-figs-v2_ed02-slide3_round12_review.pptx", "ed02-slide3_round12_review.pptx"):
        with zipfile.ZipFile(deck / name) as archive:
            vector_media = [member for member in archive.namelist() if member.startswith("ppt/media/") and member.endswith(".svg")]
            media_hashes = {hashlib.sha256(archive.read(member)).hexdigest() for member in vector_media}
            notes_text = "\n".join(archive.read(member).decode("utf-8", errors="ignore") for member in archive.namelist() if member.startswith("ppt/notesSlides/") and member.endswith(".xml"))
        audit.check(f"deck_vector_inventory::{name}", len(vector_media) == 10 and expected_vector_hashes <= media_hashes, {"count": len(vector_media), "a_g_present": expected_vector_hashes <= media_hashes})
        audit.check(f"deck_notes::{name}", "arrowheads" in notes_text and "shallow, middle, and deep" in notes_text and "manual neural masks are not applicable" in notes_text, "round12 notes")
    audit.check("deck_preview_parity", sha256(deck / "cumulative-artifact-preview.png") == sha256(deck / "review-artifact-preview.png"), [sha256(deck / "cumulative-artifact-preview.png"), sha256(deck / "review-artifact-preview.png")])
    return audit.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--composite-root", type=Path, default=DEFAULT_COMPOSITE)
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = validate(args.source_root.resolve(), args.composite_root.resolve(), args.deck_root.resolve())
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    sys.exit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
