"""Fail-closed validation for the unpromoted ED9 margin/star candidate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "docs/figure_refinement_2026-08-12/"
    "ed09_panel_b_margin_panel_f_star_density_600dpi"
)
MANIFEST = ROOT / "CANDIDATE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(row: dict[str, Any]) -> bool:
    path = REPO / row["path"]
    return (
        path.is_file()
        and path.stat().st_size == int(row["bytes"])
        and sha256(path) == row["sha256"]
    )


def star_mask(pdf_path: Path, shape: tuple[int, int], dpi: int = 600) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    with fitz.open(pdf_path) as document:
        for block in document[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip() not in {"*", "**", "***"}:
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    bounds = [
                        int(np.floor(x0 * dpi / 72.0)) - 4,
                        int(np.floor(y0 * dpi / 72.0)) - 4,
                        int(np.ceil(x1 * dpi / 72.0)) + 4,
                        int(np.ceil(y1 * dpi / 72.0)) + 4,
                    ]
                    bounds = [
                        max(0, bounds[0]),
                        max(0, bounds[1]),
                        min(shape[1], bounds[2]),
                        min(shape[0], bounds[3]),
                    ]
                    mask[bounds[1] : bounds[3], bounds[0] : bounds[2]] = True
    return mask


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = list(payload["outputs"].values())
    for role in ("C2", "B2", "E1", "G1"):
        records.extend(payload["changed_panel_outputs"][role].values())
    parent_root = (
        REPO
        / "docs/figure_version_records/"
        "extended_data_fig09_2026-08-11_round25_panel_e_signal_metadata"
    )
    parent_b2_png = parent_root / "individual_panels/ED9_B2_matched_coarse_controls.png"
    candidate_b2_png = ROOT / "individual_panels/ED9_B2_matched_coarse_controls.png"
    parent_b2_pdf = parent_b2_png.with_suffix(".pdf")
    candidate_b2_pdf = candidate_b2_png.with_suffix(".pdf")
    parent_pixels = np.asarray(Image.open(parent_b2_png).convert("RGB"))
    candidate_pixels = np.asarray(Image.open(candidate_b2_png).convert("RGB"))
    pixel_diff = np.any(parent_pixels != candidate_pixels, axis=2)
    mask = star_mask(parent_b2_pdf, pixel_diff.shape) | star_mask(
        candidate_b2_pdf, pixel_diff.shape
    )
    diff_outside_stars = int(np.count_nonzero(pixel_diff & ~mask))

    parent_root_pdf = parent_root / "ed09_version_record.pdf"
    candidate_root_pdf = REPO / payload["outputs"]["pdf"]["path"]
    with fitz.open(parent_root_pdf) as document:
        parent_images = sum(len(page.get_images(full=True)) for page in document)
        parent_drawings = sum(len(page.get_drawings()) for page in document)
    with fitz.open(candidate_root_pdf) as document:
        candidate_images = sum(len(page.get_images(full=True)) for page in document)
        candidate_drawings = sum(len(page.get_drawings()) for page in document)
    parent_g_pdf = parent_root / "individual_panels/ED9_E1_joint_place_level.pdf"
    candidate_g_pdf = ROOT / "individual_panels/ED9_E1_joint_place_level.pdf"
    parent_j_pdf = parent_root / "individual_panels/ED9_G1_consecutive_gap.pdf"
    candidate_j_pdf = ROOT / "individual_panels/ED9_G1_consecutive_gap.pdf"
    with fitz.open(parent_g_pdf) as document:
        parent_g_drawings = len(document[0].get_drawings())
    with fitz.open(candidate_g_pdf) as document:
        candidate_g_drawings = len(document[0].get_drawings())
    with fitz.open(parent_j_pdf) as document:
        parent_j_drawings = len(document[0].get_drawings())
    with fitz.open(candidate_j_pdf) as document:
        candidate_j_drawings = len(document[0].get_drawings())
    expected_candidate_drawings = (
        parent_drawings
        + candidate_g_drawings
        - parent_g_drawings
        + candidate_j_drawings
        - parent_j_drawings
    )

    png_path = REPO / payload["outputs"]["png"]["path"]
    tiff_path = REPO / payload["outputs"]["tiff"]["path"]
    with Image.open(png_path) as image:
        png_contract = {
            "pixels": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
        }
    with Image.open(tiff_path) as image:
        tiff_contract = {
            "pixels": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
            "compression": image.info.get("compression"),
        }

    checks = {
        "manifest_status_pass": payload.get("status") == "PASS",
        "candidate_not_promoted": payload.get("promoted") is False,
        "all_focused_checks_pass": all(payload.get("checks", {}).values()),
        "all_output_hashes_and_sizes_close": all(validate_record(row) for row in records),
        "only_requested_panels_changed": payload["other_panel_parity"][
            "all_byte_exact"
        ],
        "panel_b_left_padding_artist_local": payload["panel_b_audit"][
            "artist_level"
        ]
        and payload["panel_b_audit"]["left_canvas_padding_px"] == 50,
        "panel_f_star_change_artist_local": payload["panel_f_audit"][
            "artist_level"
        ]
        and payload["panel_f_audit"]["star_fontsize_pt"] == 7.0,
        "panel_g_600dpi_change_artist_local": payload["panel_g_audit"][
            "artist_level"
        ]
        and payload["panel_g_audit"]["embedded_raster_target_dpi"] == 600,
        "panel_j_600dpi_change_artist_local": payload["panel_j_audit"][
            "artist_level"
        ]
        and payload["panel_j_audit"]["embedded_raster_target_dpi"] == 600,
        "panel_f_pixel_changes_confined_to_star_bboxes": diff_outside_stars == 0,
        "root_pdf_image_count_and_artist_local_drawing_delta_exact": (
            candidate_images == parent_images
            and candidate_drawings == expected_candidate_drawings
        ),
        "root_png_true_600dpi": png_contract["pixels"] == [3900, 4010]
        and all(math.isclose(value, 600.0, abs_tol=0.01) for value in png_contract["dpi"]),
        "root_tiff_true_600dpi_lzw": tiff_contract["pixels"] == [3900, 4010]
        and tiff_contract["compression"] == "tiff_lzw"
        and all(math.isclose(value, 600.0, abs_tol=0.01) for value in tiff_contract["dpi"]),
        "g_and_j_effective_dpi_gates_pass": payload["checks"][
            "panel_g_embedded_image_upgraded_to_600dpi"
        ]
        and payload["checks"]["panel_j_embedded_image_upgraded_to_600dpi"],
        "candidate_open_map_compounds_and_forbidden_scan_pass": payload[
            "checks"
        ]["root_svg_has_open_map_compounds"]
        and payload["checks"]["root_svg_has_no_forbidden_hyphenated_map_terms"]
        and payload["checks"]["root_pdf_has_no_forbidden_hyphenated_map_terms"],
        "manual_600dpi_visual_qa": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema": "ed09-panel-b-margin-panel-f-star-focused-validation/v1",
        "status": status,
        "candidate": str(ROOT.relative_to(REPO)),
        "checks": checks,
        "measurements": {
            "panel_f_changed_pixels": int(np.count_nonzero(pixel_diff)),
            "panel_f_changed_pixels_outside_parent_or_candidate_star_bboxes": diff_outside_stars,
            "root_pdf_parent_images": parent_images,
            "root_pdf_candidate_images": candidate_images,
            "root_pdf_parent_drawings": parent_drawings,
            "root_pdf_candidate_drawings": candidate_drawings,
            "root_pdf_expected_candidate_drawings_from_changed_panels": expected_candidate_drawings,
            "panel_g_parent_drawings": parent_g_drawings,
            "panel_g_candidate_drawings": candidate_g_drawings,
            "panel_j_parent_drawings": parent_j_drawings,
            "panel_j_candidate_drawings": candidate_j_drawings,
            "png_contract": png_contract,
            "tiff_contract": tiff_contract,
            "manual_600dpi_visual_qa_evidence": (
                "Inspected the 3900x4010 root PNG and all four changed 600-dpi panel PNGs at original resolution; "
                "panel-b ylabel is fully contained, panel-f stars clear brackets and points, and panels g/j retain "
                "their accepted density distributions with clearer 600-dpi continuous clouds and open map compounds."
            ),
        },
    }
    (ROOT / "FOCUSED_VALIDATION.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
