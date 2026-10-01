from __future__ import annotations

import base64
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figure_resolution_repair_2026-08-12"
RECORDS = ROOT / "docs" / "figure_version_records"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_audit(path: Path) -> dict[str, Any]:
    with fitz.open(path) as document:
        page = document[0]
        images = []
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = [float(value) for value in info["bbox"]]
            width = int(info["width"])
            height = int(info["height"])
            images.append(
                {
                    "pixels": [width, height],
                    "bbox_pt": [x0, y0, x1, y1],
                    "effective_dpi": [
                        width * 72.0 / (x1 - x0),
                        height * 72.0 / (y1 - y0),
                    ],
                }
            )
        return {
            "page_count": len(document),
            "page_points": [float(page.rect.width), float(page.rect.height)],
            "images": images,
            "drawing_count": len(page.get_drawings()),
            "words": page.get_text("words"),
            "text": page.get_text(),
        }


def svg_audit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    images = []
    for image in root.iter("{http://www.w3.org/2000/svg}image"):
        href = image.attrib.get("{http://www.w3.org/1999/xlink}href", "")
        payload = base64.b64decode(href.split(",", 1)[1])
        pixmap = fitz.Pixmap(payload)
        width_pt = float(image.attrib["width"])
        height_pt = float(image.attrib["height"])
        images.append(
            {
                "pixels": [pixmap.width, pixmap.height],
                "bbox_pt": [
                    float(image.attrib["x"]),
                    float(image.attrib["y"]),
                    width_pt,
                    height_pt,
                ],
                "effective_dpi": [
                    pixmap.width * 72.0 / width_pt,
                    pixmap.height * 72.0 / height_pt,
                ],
            }
        )
    return {"viewBox": root.attrib.get("viewBox"), "images": images}


def raster_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "pixels": list(image.size),
            "dpi": [float(value) for value in image.info.get("dpi", (0.0, 0.0))],
            "mode": image.mode,
        }


def same_tree_by_name(left: Path, right: Path, pattern: str = "*") -> dict[str, Any]:
    left_files = sorted(path for path in left.glob(pattern) if path.is_file())
    right_by_name = {path.name: path for path in right.rglob(pattern) if path.is_file()}
    comparisons = []
    for source in left_files:
        candidate = right_by_name.get(source.name)
        comparisons.append(
            {
                "name": source.name,
                "left_sha256": sha256(source),
                "right_sha256": sha256(candidate) if candidate else None,
                "match": bool(candidate and sha256(source) == sha256(candidate)),
            }
        )
    return {
        "comparisons": comparisons,
        "all_match": bool(comparisons) and all(row["match"] for row in comparisons),
    }


def output_entry(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    ed01_dir = OUT / "ed01"
    ed02_dir = OUT / "ed02"
    ed04_dir = OUT / "ed04"
    ed05_dir = OUT / "ed05"

    ed04_parent = RECORDS / "extended_data_fig04_2026-08-11_round2_experience_layout"
    ed05_parent = RECORDS / "extended_data_fig05_2026-08-11_round12_normalized_rate"
    ed04_current_pdf = ed04_parent / "ed04_version_record.pdf"
    ed04_candidate_pdf = ed04_dir / "ed04_registration_quality_assembly.pdf"
    ed05_current_pdf = ed05_parent / "ed05_version_record.pdf"
    ed05_candidate_pdf = ed05_dir / "ed05_example_place_maps_assembly_round11.pdf"

    ed04_before = pdf_audit(ed04_current_pdf)
    ed04_after = pdf_audit(ed04_candidate_pdf)
    ed05_before = pdf_audit(ed05_current_pdf)
    ed05_after = pdf_audit(ed05_candidate_pdf)
    ed04_svg = svg_audit(ed04_dir / "ed04_registration_quality_assembly.svg")
    ed05_svg = svg_audit(ed05_dir / "ed05_example_place_maps_assembly_round11.svg")

    ed04_embedded_hashes = []
    ed04_manifest = json.loads(
        (ed04_dir / "ED04_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8")
    )
    for panel in ("a", "b"):
        upstream = ROOT / ed04_manifest["panels"][panel]["path"]
        frozen = ed04_parent / "embedded_sources" / upstream.name
        ed04_embedded_hashes.append(
            {
                "panel": panel,
                "upstream": rel(upstream),
                "frozen": rel(frozen),
                "upstream_sha256": sha256(upstream),
                "frozen_sha256": sha256(frozen),
                "match": sha256(upstream) == sha256(frozen),
            }
        )
    ed04_sources = same_tree_by_name(
        ed04_parent / "source_from_raw",
        ROOT
        / "docs"
        / "extended_data_supplement_v2_rebuild_2026-08-05"
        / "ed04_registration_quality"
        / "source_from_raw",
    )

    current_audit_rows = {
        row["role"]: row
        for row in csv.DictReader(
            (ed05_parent / "SOURCE_DATA_AUDIT.csv").open(encoding="utf-8-sig", newline="")
        )
    }
    candidate_audit_rows = {
        row["role"]: row
        for row in csv.DictReader(
            (ed05_dir / "SOURCE_DATA_AUDIT_ROUND11.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    }
    authoritative_roles = sorted(
        set(current_audit_rows)
        & set(candidate_audit_rows)
        - {
            "gallery_derived_svg",
            "gallery_derived_png",
            "gallery_derived_tiff",
            "gallery_derived_pdf",
        }
    )
    ed05_source_parity = [
        {
            "role": role,
            "current_sha256": current_audit_rows[role]["sha256"],
            "candidate_sha256": candidate_audit_rows[role]["sha256"],
            "match": current_audit_rows[role]["sha256"]
            == candidate_audit_rows[role]["sha256"],
        }
        for role in authoritative_roles
    ]
    old_gallery = (
        ROOT
        / "docs"
        / "extended_data_supplement_v2_rebuild_2026-08-05"
        / "ed05_example_place_maps"
        / "assembly"
        / "derived"
        / "ed05_compact_32_grid_8x8_round11.png"
    )
    new_gallery = ed05_dir / "derived" / "ed05_compact_32_grid_8x8_round11.png"
    with Image.open(old_gallery) as old_image, Image.open(new_gallery) as new_image:
        old_rgba = np.asarray(old_image.convert("RGBA"))
        new_rgba = np.asarray(new_image.convert("RGBA"))
    differences = np.any(old_rgba != new_rgba, axis=2)
    ys, xs = np.where(differences)
    gallery_pixel_parity = {
        "old": raster_audit(old_gallery),
        "new": raster_audit(new_gallery),
        "same_shape": old_rgba.shape == new_rgba.shape,
        "different_pixel_count": int(differences.sum()),
        "difference_bbox_px": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ],
        "scientific_map_region_below_title_exact": bool(
            np.array_equal(old_rgba[79:, :, :], new_rgba[79:, :, :])
        ),
        "note": "All changed pixels are confined to the requested top colorbar title change to 'Normalized rate'.",
    }

    ed01_validation = json.loads(
        (ed01_dir / "RENDER_VALIDATION_ROUND12.json").read_text(encoding="utf-8")
    )
    ed02_manifest = json.loads(
        (ed02_dir / "ED02_ROUND21_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8")
    )
    ed02_parent = RECORDS / "extended_data_fig02_2026-08-11_round21_layout_refinement"
    ed02_plot_ready = same_tree_by_name(ed02_parent / "plot_ready", ed02_dir / "plot_ready")
    ed02_render = json.loads((ed02_dir / "RENDER_RESULT.json").read_text(encoding="utf-8"))
    ed02_geometry = ed02_render.get("round21_display_only_changes", {})

    ed04_min_after_pdf = min(
        min(image["effective_dpi"]) for image in ed04_after["images"]
    )
    ed04_min_after_svg = min(
        min(image["effective_dpi"]) for image in ed04_svg["images"]
    )
    ed05_min_after_pdf = min(
        min(image["effective_dpi"]) for image in ed05_after["images"]
    )
    ed05_min_after_svg = min(
        min(image["effective_dpi"]) for image in ed05_svg["images"]
    )

    checks = {
        "ed01_renderer_validation_pass": ed01_validation["status"] == "pass",
        "ed01_h_axes_equal_g_height": ed01_validation["checks"][
            "panel_h_axes_match_panel_g_physical_height"
        ],
        "ed01_i_end_nodes_clear_of_every_tick": ed01_validation["checks"][
            "panel_i_pdf_end_nodes_clear_of_every_y_tick"
        ],
        "ed01_i_title_tick_clearance_at_least_1pt": ed01_validation[
            "panel_i_pdf_ylabel_tick_min_clearance_pt"
        ]
        >= 1.0,
        "ed01_relocated_letters_pass": ed01_validation["checks"][
            "panel_labels_g_h_j_k_outside_top_left_and_i_left"
        ],
        "ed02_renderer_status_pass": ed02_manifest["status"] == "pass",
        "ed02_vector_no_images": ed02_manifest["vector_audit"][
            "embedded_raster_images"
        ]
        == 0,
        "ed02_plot_ready_exact_parent_parity": ed02_plot_ready["all_match"],
        "ed02_preceding_turn_moved_left": ed02_geometry.get(
            "panel_a_preceding_turn_xy"
        )
        == [3.42, 3.51],
        "ed02_preceding_turn_arrow_clearance_at_least_1px": ed02_geometry.get(
            "panel_a_geometry_at_source_dpi", {}
        ).get("grey_arrow_clearance_px", -1)
        >= 1.0,
        "ed02_preceding_turn_box_clearance_at_least_1px": ed02_geometry.get(
            "panel_a_geometry_at_source_dpi", {}
        ).get("box_clearance_px", -1)
        >= 1.0,
        "ed02_e_spine_above_data_below_annotation_band": (
            ed02_geometry.get("panel_e_geometry", {}).get("data_to_spine_clearance", -1)
            > 0
            and ed02_geometry.get("panel_e_geometry", {}).get(
                "spine_to_annotation_band_clearance", -1
            )
            > 0
        ),
        "ed02_g_upper_lower_marker_x_exact": ed02_geometry.get(
            "panel_g_upper_lower_circle_x_aligned", False
        ),
        "ed04_pdf_all_images_at_least_599dpi": ed04_min_after_pdf >= 599.0,
        "ed04_svg_all_images_at_least_599dpi": ed04_min_after_svg >= 599.0,
        "ed04_source_embedded_images_exact": all(
            row["match"] for row in ed04_embedded_hashes
        ),
        "ed04_source_from_raw_exact": ed04_sources["all_match"],
        "ed04_vector_drawings_exact_parent_parity": ed04_before["drawing_count"]
        == ed04_after["drawing_count"]
        and ed04_before["words"] == ed04_after["words"]
        and ed04_before["text"] == ed04_after["text"],
        "ed05_pdf_panel_a_at_least_599dpi": ed05_min_after_pdf >= 599.0,
        "ed05_svg_panel_a_at_least_599dpi": ed05_min_after_svg >= 599.0,
        "ed05_authoritative_source_hashes_exact": all(
            row["match"] for row in ed05_source_parity
        ),
        "ed05_scientific_gallery_pixels_exact_below_title": gallery_pixel_parity[
            "scientific_map_region_below_title_exact"
        ],
        "ed05_panels_b_to_e_vector_content_exact_parent_parity": ed05_before[
            "drawing_count"
        ]
        == ed05_after["drawing_count"]
        and ed05_before["words"] == ed05_after["words"]
        and ed05_before["text"] == ed05_after["text"],
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "ed01_geometry": {
            "g_height_in": ed01_validation["panel_g_physical_height_in"],
            "h_heights_in": ed01_validation["panel_h_physical_heights_in"],
            "i_end_nodes_to_nearest_tick_clearance_pt": ed01_validation[
                "panel_i_pdf_ylabel_tick_min_clearance_pt"
            ],
            "i_end_nodes_page_clearance_pt": ed01_validation[
                "panel_i_pdf_ylabel_min_page_clearance_pt"
            ],
        },
        "ed02_geometry": ed02_geometry,
        "ed02_plot_ready_parity": ed02_plot_ready,
        "ed04_resolution": {
            "before_pdf": ed04_before["images"],
            "after_pdf": ed04_after["images"],
            "after_svg": ed04_svg["images"],
            "embedded_source_hashes": ed04_embedded_hashes,
            "source_from_raw": ed04_sources,
            "vector_drawings_before_after": [
                ed04_before["drawing_count"],
                ed04_after["drawing_count"],
            ],
            "note": "Sub-point image bbox changes are raster pixel-quantization only; all vector drawings and words are exactly unchanged.",
        },
        "ed05_resolution": {
            "before_pdf": ed05_before["images"],
            "after_pdf": ed05_after["images"],
            "after_svg": ed05_svg["images"],
            "source_parity": ed05_source_parity,
            "gallery_pixel_parity": gallery_pixel_parity,
            "vector_drawings_before_after": [
                ed05_before["drawing_count"],
                ed05_after["drawing_count"],
            ],
        },
        "visual_qa": {
            "status": "pass",
            "method": "600-dpi root rasters and 1440-px ED2 composite inspected",
            "findings": [
                "ED1 panel letters are consistent and panel i gutter lanes are clear.",
                "ED1 panel h axes match panel g height and retain one shared x label.",
                "ED2 preceding-turn label is close to the grid without overlap; panel e and g requested geometry is clear.",
                "ED4 scientific rasters are crisp at 600 dpi and vector panels are visually unchanged.",
                "ED5 place-map cells and labels are crisp; panels b-e are visually unchanged.",
            ],
        },
    }
    validation_path = OUT / "CANDIDATE_VALIDATION.json"
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    candidates = {
        "extended_data_figure_1": {
            "parent_record": rel(
                RECORDS / "extended_data_fig01_2026-08-11_round11_column_alignment"
            ),
            "root_outputs": {
                extension: output_entry(
                    ed01_dir
                    / f"extended_data_figure1_source_rerender_round12.{extension}"
                )
                for extension in ("svg", "pdf", "png", "tiff")
            },
            "individual_panel_assets": [],
            "changed_renderers": [
                "scripts/render_extended_data_figure1_round4.py",
                "scripts/render_extended_data_figure1_geometry_round12.py",
            ],
        },
        "extended_data_figure_2": {
            "parent_record": rel(ed02_parent),
            "root_outputs": {
                "svg": output_entry(ed02_dir / "ed02_round21_refinement_candidate.svg"),
                "pdf": output_entry(ed02_dir / "ed02_round21_refinement_candidate.pdf"),
                "png": output_entry(
                    ed02_dir / "ed02_round21_refinement_candidate_600dpi.png"
                ),
                "tiff": output_entry(
                    ed02_dir / "ed02_round21_refinement_candidate_600dpi.tiff"
                ),
            },
            "individual_panel_assets": [
                output_entry(path)
                for path in sorted((ed02_dir / "panels").glob("*.pdf"))
            ],
            "changed_renderers": [
                "scripts/render_ed02_slide3_refinement_round21.py",
            ],
        },
        "extended_data_figure_4": {
            "parent_record": rel(ed04_parent),
            "root_outputs": {
                extension: output_entry(
                    ed04_dir / f"ed04_registration_quality_assembly.{extension}"
                )
                for extension in ("svg", "pdf", "png", "tiff")
            },
            "individual_panel_assets": [],
            "changed_renderers": [
                "scripts/ed_supplement_common.py",
                "scripts/render_ed04_registration_quality_assembly.py",
            ],
        },
        "extended_data_figure_5": {
            "parent_record": rel(ed05_parent),
            "root_outputs": {
                extension: output_entry(
                    ed05_dir
                    / f"ed05_example_place_maps_assembly_round11.{extension}"
                )
                for extension in ("svg", "pdf", "png", "tiff")
            },
            "individual_panel_assets": [
                output_entry(path)
                for path in sorted((ed05_dir / "derived").glob("*"))
                if path.suffix.lower() in {".svg", ".pdf", ".png", ".tiff"}
            ],
            "changed_renderers": [
                "scripts/ed_supplement_common.py",
                "scripts/render_ed05_example_place_maps_assembly.py",
                "tests/test_ed05_example_place_maps_assembly.py",
            ],
        },
    }
    manifest = {
        "status": report["status"],
        "promotion_status": "candidate_only_not_promoted",
        "focused_validation": output_entry(validation_path),
        "figures": candidates,
    }
    (OUT / "CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "pass":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Candidate validation failed: {failed}")
    print(json.dumps({"status": "pass", "checks": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
