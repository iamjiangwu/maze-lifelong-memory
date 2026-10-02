"""Export the approved ED2 Round 11 page as vector PDF/SVG and 600-dpi rasters.

The nine scientific panels and shared b/c legend are placed from their
recorded vector PDFs using the exact 720-unit Round 11 PowerPoint frame map.
No screenshot or PowerPoint raster is used to build the publication figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import fitz
from PIL import Image
from matplotlib.font_manager import FontProperties, findfont


sys.dont_write_bytecode = True
# Resolve the installed Arial face on every OS; Matplotlib reports its fallback
# when Arial is unavailable. The font hash is recorded with the generated PDF.
ARIAL_BOLD = Path(findfont(FontProperties(family="Arial", weight="bold")))
PAGE_UNITS = 720.0
PAGE_INCHES = 7.5
POINTS_PER_UNIT = PAGE_INCHES * 72.0 / PAGE_UNITS
PANEL_ORDER = tuple("abcdefghi")
COMPOSITE_STEM = "ed02_round11_axis_aligned_version_record"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_rect(frame: dict[str, float]) -> fitz.Rect:
    left = float(frame["left"]) * POINTS_PER_UNIT
    top = float(frame["top"]) * POINTS_PER_UNIT
    right = (float(frame["left"]) + float(frame["width"])) * POINTS_PER_UNIT
    bottom = (float(frame["top"]) + float(frame["height"])) * POINTS_PER_UNIT
    return fitz.Rect(left, top, right, bottom)


def panel_pdf_map(source_root: Path) -> dict[str, Path]:
    rows = list(csv.DictReader((source_root / "PANEL_MANIFEST.csv").open(encoding="utf-8")))
    result = {
        row["panel"]: source_root / row["path"]
        for row in rows
        if row["format"] == "pdf"
    }
    if set(result) != set(PANEL_ORDER):
        raise RuntimeError(f"Expected panel PDFs a-i, found {sorted(result)}")
    return result


def save_rasters(pdf_path: Path, png_path: Path, tiff_path: Path, preview_path: Path) -> None:
    with fitz.open(pdf_path) as document:
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image.save(png_path, format="PNG", dpi=(600, 600), optimize=True)
        image.save(tiff_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
        preview = page.get_pixmap(matrix=fitz.Matrix(1440 / 540, 1440 / 540), alpha=False)
        Image.frombytes("RGB", (preview.width, preview.height), preview.samples).save(
            preview_path, format="PNG", dpi=(192, 192), optimize=True
        )


def export(
    source_root: Path,
    output_root: Path,
    *,
    composite_stem: str = COMPOSITE_STEM,
    review_round: int = 11,
    manifest_name: str = "ED02_ROUND11_COMPOSITE_MANIFEST.json",
) -> dict[str, Any]:
    if not ARIAL_BOLD.is_file():
        raise FileNotFoundError(f"Arial Bold font not found: {ARIAL_BOLD}")
    source_manifest = load_json(source_root / "SOURCE_MANIFEST.json")
    layout = source_manifest["layout_contract_round11"]
    frames = layout["panel_frames_720"]
    labels = layout["panel_label_frames_720"]
    legend = source_manifest["shared_bc_legend"]
    panel_pdfs = panel_pdf_map(source_root)
    legend_pdf = source_root / "shared_assets" / "ed02_shared_bc_experience_legend_round11.pdf"
    if not legend_pdf.is_file():
        raise FileNotFoundError(legend_pdf)

    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path = output_root / f"{composite_stem}.pdf"
    svg_path = output_root / f"{composite_stem}.svg"
    png_path = output_root / f"{composite_stem}_600dpi.png"
    tiff_path = output_root / f"{composite_stem}_600dpi.tiff"
    preview_path = output_root / f"{composite_stem}_preview_1440.png"
    manifest_path = output_root / manifest_name
    for path in (pdf_path, svg_path, png_path, tiff_path, preview_path, manifest_path):
        if path.exists():
            path.unlink()

    document = fitz.open()
    page = document.new_page(width=PAGE_INCHES * 72, height=PAGE_INCHES * 72)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1), overlay=False)
    page.insert_font(fontname="ArialBold", fontfile=str(ARIAL_BOLD))
    placements: list[dict[str, Any]] = []

    for panel in PANEL_ORDER:
        source_path = panel_pdfs[panel]
        with fitz.open(source_path) as panel_document:
            if panel_document.page_count != 1:
                raise RuntimeError(f"Panel PDF is not one page: {source_path}")
            source_rect = panel_document[0].rect
            destination = frame_rect(frames[panel])
            source_aspect = source_rect.width / source_rect.height
            destination_aspect = destination.width / destination.height
            if abs(source_aspect - destination_aspect) > 2.0e-6:
                raise RuntimeError(
                    f"Panel {panel} aspect mismatch: {source_aspect} vs {destination_aspect}"
                )
            page.show_pdf_page(destination, panel_document, 0, keep_proportion=True, overlay=True)
        label = labels[panel]
        label_x = float(label["left"]) * POINTS_PER_UNIT
        label_top = float(label["top"]) * POINTS_PER_UNIT
        page.insert_text(
            (label_x, label_top + 10.5),
            panel,
            fontname="ArialBold",
            fontsize=10.5,
            color=(0, 0, 0),
            overlay=True,
        )
        placements.append(
            {
                "role": panel,
                "source": source_path.relative_to(source_root).as_posix(),
                "source_sha256": sha256(source_path),
                "frame_720": frames[panel],
                "label_frame_720": labels[panel],
                "source_aspect": source_aspect,
                "destination_aspect": destination_aspect,
            }
        )

    with fitz.open(legend_pdf) as legend_document:
        if legend_document.page_count != 1:
            raise RuntimeError("Shared legend PDF is not one page")
        legend_destination = frame_rect(legend["frame_720"])
        legend_aspect = legend_document[0].rect.width / legend_document[0].rect.height
        if abs(legend_aspect - legend_destination.width / legend_destination.height) > 2.0e-6:
            raise RuntimeError("Shared legend aspect mismatch")
        page.show_pdf_page(legend_destination, legend_document, 0, keep_proportion=True, overlay=True)

    document.set_metadata(
        {
            "title": "Extended Data Figure 2 | experience-dependent route policies",
            "author": "Python source-vector assembly",
            "subject": f"Round {review_round} immutable version-record candidate",
            "keywords": "Extended Data Figure 2, vector, fixed 30 min, route policy",
        }
    )
    document.save(pdf_path, garbage=4, deflate=True, clean=True)
    document.close()
    with fitz.open(pdf_path) as vector_document:
        svg_path.write_text(
            vector_document[0].get_svg_image(text_as_path=False),
            encoding="utf-8",
        )
    save_rasters(pdf_path, png_path, tiff_path, preview_path)

    with fitz.open(pdf_path) as check:
        page = check[0]
        vector_audit = {
            "page_count": check.page_count,
            "page_points": [page.rect.width, page.rect.height],
            "drawing_objects": len(page.get_drawings()),
            "embedded_raster_images": len(page.get_images(full=True)),
            "font_count": len(page.get_fonts(full=True)),
            "selectable_text_characters": len(page.get_text()),
        }
    with Image.open(png_path) as image:
        png_audit = {
            "size": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
        }
    with Image.open(tiff_path) as image:
        tiff_audit = {
            "size": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
        }

    outputs = {}
    for key, path in (
        ("pdf", pdf_path),
        ("svg", svg_path),
        ("png", png_path),
        ("tiff", tiff_path),
        ("preview_png", preview_path),
    ):
        outputs[key] = {
            "path": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "status": "pass",
        "figure": "Extended Data Figure 2",
        "review_round": review_round,
        "backend": "Python with PyMuPDF vector placement from matplotlib panel PDFs",
        "page_inches": [PAGE_INCHES, PAGE_INCHES],
        "panel_order": list(PANEL_ORDER),
        "shared_legend": {
            "source": legend_pdf.relative_to(source_root).as_posix(),
            "source_sha256": sha256(legend_pdf),
            "frame_720": legend["frame_720"],
        },
        "panel_labels": {
            "font": "Arial Bold",
            "font_file": ARIAL_BOLD.name,
            "font_sha256": sha256(ARIAL_BOLD),
            "fontsize_pt": 10.5,
        },
        "placements": placements,
        "vector_audit": vector_audit,
        "png_audit": png_audit,
        "tiff_audit": tiff_audit,
        "outputs": outputs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(export(args.source_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
