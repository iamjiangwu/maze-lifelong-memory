"""Validate the native 600-dpi Main Figure 3 candidate without promotion."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure3_2026-08-11_round4_experience_labels"
)
CANDIDATE = (
    REPO_ROOT
    / "docs"
    / "figure_embedded_dpi_correction_2026-08-12"
    / "main3_600dpi_candidate"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _read_geometry(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("source_svg", None)
        row.pop("source_png", None)
    return rows


def _pdf_image_dpi(path: Path) -> list[dict[str, float | int]]:
    with fitz.open(path) as document:
        if len(document) != 1:
            raise RuntimeError(f"Expected one-page PDF: {path}")
        results = []
        for item in document[0].get_image_info(xrefs=True):
            x0, y0, x1, y1 = item["bbox"]
            results.append(
                {
                    "width_px": int(item["width"]),
                    "height_px": int(item["height"]),
                    "effective_dpi_x": float(item["width"] / (x1 - x0) * 72.0),
                    "effective_dpi_y": float(item["height"] / (y1 - y0) * 72.0),
                }
            )
        return results


def main() -> None:
    current_panels = CURRENT / "individual_panels"
    candidate_panels = CANDIDATE / "figure3_individual_panels"
    current_a_png = current_panels / "figure3_panel_a_top_example_row.png"
    candidate_a_png = candidate_panels / "figure3_panel_a_top_example_row.png"
    current_geometry = CURRENT / "figure3_axes_geometry.csv"
    candidate_geometry = CANDIDATE / "figure3_axes_geometry.csv"
    candidate_pdf = CANDIDATE / "figure3_source_publication.pdf"
    candidate_png = CANDIDATE / "figure3_source_publication.png"
    candidate_tiff = CANDIDATE / "figure3_source_publication_600dpi.tiff"

    with Image.open(candidate_png) as image:
        image.save(candidate_tiff, dpi=(600, 600), compression="tiff_lzw")

    panel_a_png_pixel_exact = _sha256(current_a_png) == _sha256(candidate_a_png)
    geometry_equal = _read_geometry(current_geometry) == _read_geometry(candidate_geometry)

    unrelated_checks = []
    for path in sorted(current_panels.glob("figure3_panel_*")):
        if "panel_a_" in path.name or path.suffix.lower() not in {".pdf", ".png"}:
            continue
        candidate_path = candidate_panels / path.name
        unrelated_checks.append(
            {
                "asset": path.name,
                "byte_exact": candidate_path.is_file() and _sha256(path) == _sha256(candidate_path),
            }
        )

    immutable_audits = []
    for name in (
        "figure3_day2_template_first30_alignment.csv",
        "figure3_day2_template_support_runs.csv",
    ):
        immutable_audits.append(
            {
                "asset": name,
                "byte_exact": _sha256(CURRENT / name) == _sha256(CANDIDATE / name),
            }
        )

    pdf_images = _pdf_image_dpi(candidate_pdf)
    minimum_pdf_dpi = min(
        min(item["effective_dpi_x"], item["effective_dpi_y"])
        for item in pdf_images
    )
    run_manifest = json.loads((CANDIDATE / "run_manifest.json").read_text(encoding="utf-8"))
    built_in_validation = run_manifest["validation"]["3"]
    status = "PASS" if all(
        (
            panel_a_png_pixel_exact,
            geometry_equal,
            all(item["byte_exact"] for item in unrelated_checks),
            all(item["byte_exact"] for item in immutable_audits),
            len(pdf_images) == 4,
            minimum_pdf_dpi >= 500.0,
            built_in_validation["expected_panel_count"] == 13,
            built_in_validation["svg_image_node_count"] == 4,
            built_in_validation["source_panel_files_exist"],
        )
    ) else "FAIL"

    validation = {
        "schema": "main-figure3-native-embedded-image-dpi-validation/v1",
        "status": status,
        "parent_record": _relative(CURRENT),
        "change": "increase the four panel-a scientific rasters from the implicit 100-dpi SVG/PDF default to the configured 600-dpi native render",
        "panel_a_600dpi_png_pixel_exact_to_parent": panel_a_png_pixel_exact,
        "axes_geometry_equal_ignoring_output_paths": geometry_equal,
        "unrelated_panel_pdf_png_byte_parity": unrelated_checks,
        "source_audit_table_parity": immutable_audits,
        "candidate_pdf_embedded_images": pdf_images,
        "candidate_pdf_minimum_effective_dpi": minimum_pdf_dpi,
        "built_in_validation": built_in_validation,
        "tiff": {
            "path": _relative(candidate_tiff),
            "sha256": _sha256(candidate_tiff),
        },
    }
    validation_path = CANDIDATE / "VALIDATION.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    panel_assets = sorted(
        _relative(path)
        for path in candidate_panels.iterdir()
        if path.is_file()
    )
    manifest = {
        "schema": "main-figure3-native-embedded-image-dpi-candidate/v1",
        "status": status,
        "parent_record": _relative(CURRENT),
        "current_record_unchanged": _relative(CURRENT),
        "root_assets": {
            suffix: _relative(CANDIDATE / filename)
            for suffix, filename in {
                "svg": "figure3_source_publication.svg",
                "pdf": "figure3_source_publication.pdf",
                "png": "figure3_source_publication.png",
                "tiff": "figure3_source_publication_600dpi.tiff",
                "html": "figure3_source_publication.html",
            }.items()
        },
        "individual_panel_directory": _relative(candidate_panels),
        "individual_panel_assets": panel_assets,
        "changed_renderer_files": ["scripts/render_main_figs_v3_publication.py"],
        "focused_validation": {
            "status": status,
            "report": _relative(validation_path),
            "validator": "scripts/validate_main_figure3_embedded_dpi_candidate.py",
            "dpi_audit": "docs/figure_embedded_dpi_correction_2026-08-12/EMBEDDED_IMAGE_DPI_AUDIT.json",
        },
    }
    (CANDIDATE / "CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
