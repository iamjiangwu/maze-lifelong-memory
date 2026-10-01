"""Validate the ED1 round-15 append-only successor against its round-14 parent."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image


RECORD = Path(__file__).resolve().parents[1]
RECORDS = RECORD.parent
PARENT = RECORDS / "extended_data_fig01_2026-08-12_round14_coverage_round_labels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_spans(path: Path) -> list[dict]:
    spans: list[dict] = []
    with fitz.open(path) as document:
        for page in document:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans.extend(line.get("spans", []))
    return spans


def bracket_paths(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "stroke: #303030; stroke-width: 0.48; stroke-linecap: square" in line
    ]


def write_hashes() -> None:
    with (RECORD / "RECORD_HASHES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for path in sorted(RECORD.rglob("*")):
            if path.is_file() and path.name != "RECORD_HASHES.csv":
                writer.writerow(
                    [path.relative_to(RECORD).as_posix(), sha256(path), path.stat().st_size]
                )


def main() -> None:
    parent_svg = PARENT / "ed01_version_record.svg"
    successor_svg = RECORD / "ed01_version_record.svg"
    parent_spans = pdf_spans(PARENT / "ed01_version_record.pdf")
    successor_spans = pdf_spans(RECORD / "ed01_version_record.pdf")
    parent_stars = [span for span in parent_spans if span["text"] == "***"]
    successor_stars = [span for span in successor_spans if span["text"] == "***"]
    parent_brackets = bracket_paths(parent_svg)
    successor_brackets = bracket_paths(successor_svg)
    source_equal = pd.read_csv(PARENT / "SOURCE_DATA_AUDIT.csv").equals(
        pd.read_csv(RECORD / "SOURCE_DATA_AUDIT.csv")
    )
    layout_equal = pd.read_csv(PARENT / "LAYOUT_AUDIT.csv").equals(
        pd.read_csv(RECORD / "LAYOUT_AUDIT.csv")
    )
    render_validation = json.loads(
        (RECORD / "ROUND15_RENDER_VALIDATION.json").read_text(encoding="utf-8")
    )
    pdf_fonts = sorted({span["font"] for span in successor_spans})
    parent_source_hashes = pd.read_csv(PARENT / "SOURCE_DATA_AUDIT.csv")[
        ["panel", "source_path", "sha256"]
    ].to_dict("records")
    successor_source_hashes = pd.read_csv(RECORD / "SOURCE_DATA_AUDIT.csv")[
        ["panel", "source_path", "sha256"]
    ].to_dict("records")
    with Image.open(RECORD / "ed01_version_record_300dpi.png") as image:
        png300 = image.size == (2250, 2250) and tuple(round(v) for v in image.info["dpi"]) == (300, 300)
    with Image.open(RECORD / "ed01_version_record_600dpi.png") as image:
        png600 = image.size == (4500, 4500)
    with Image.open(RECORD / "ed01_version_record_600dpi.tiff") as image:
        tiff600 = image.size == (4500, 4500) and tuple(round(v) for v in image.info["dpi"]) == (600, 600)

    checks = {
        "parent_record_exists": PARENT.is_dir(),
        "renderer_validation_passes": render_validation["status"] == "pass",
        "scientific_source_audit_exact_parent_parity": source_equal,
        "scientific_source_hashes_exact_parent_parity": parent_source_hashes == successor_source_hashes,
        "layout_audit_exact_parent_parity": layout_equal,
        "all_18_bracket_svg_paths_exact_parent_parity": len(parent_brackets) == 18 and parent_brackets == successor_brackets,
        "all_18_star_texts_retained": len(parent_stars) == len(successor_stars) == 18,
        "star_centres_retained_with_expected_glyph_bbox_growth": all(
            abs((a["bbox"][0] + a["bbox"][2]) / 2 - (b["bbox"][0] + b["bbox"][2]) / 2) < 0.01
            and abs(a["bbox"][3] - b["bbox"][3]) < 0.01
            for a, b in zip(parent_stars, successor_stars)
        ),
        "stars_increased_from_3_9_to_4_5_pt": all(abs(s["size"] - 3.9) < 1e-5 for s in parent_stars) and all(abs(s["size"] - 4.5) < 1e-5 for s in successor_stars),
        "pdf_fonts_arial_only": set(pdf_fonts).issubset({"ArialMT", "Arial-BoldMT"}),
        "pdf_contains_no_dejavu": all("DejaVu" not in font for font in pdf_fonts),
        "svg_contains_no_dejavu": "DejaVu" not in successor_svg.read_text(encoding="utf-8"),
        "pdf_vector_only": render_validation["checks"]["pdf_contains_no_raster_images"],
        "svg_vector_only": render_validation["checks"]["svg_contains_no_raster_images"],
        "export_bundle_complete": all((RECORD / name).is_file() for name in (
            "ed01_version_record.svg", "ed01_version_record.pdf", "ed01_version_record_300dpi.png",
            "ed01_version_record_600dpi.png", "ed01_version_record_600dpi.tiff",
        )),
        "png_300dpi_valid": png300,
        "png_600dpi_valid": png600,
        "tiff_600dpi_valid": tiff600,
    }
    report = {
        "figure": "Extended Data Figure 1",
        "record": RECORD.name,
        "parent_record": PARENT.name,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "pdf_fonts": pdf_fonts,
        "panel_f": {
            "bracket_count": len(successor_brackets),
            "star_count": len(successor_stars),
            "parent_star_size_pt": sorted({round(s["size"], 3) for s in parent_stars}),
            "successor_star_size_pt": sorted({round(s["size"], 3) for s in successor_stars}),
        },
    }
    (RECORD / "ED01_ROUND15_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (RECORD / "RECORD_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_hashes()
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
