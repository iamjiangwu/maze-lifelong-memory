"""Fail-closed validation for the ED1/ED2/ED5 2026-08-12 follow-up."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz


REPO = Path(__file__).resolve().parents[1]
CANDIDATE = REPO / "docs" / "figure_refinement_2026-08-12"
RECORDS = REPO / "docs" / "figure_version_records"
ED1 = CANDIDATE / "ed01_round13_collision"
ED2 = CANDIDATE / "ed02_round23_preceding_turn"
ED5 = CANDIDATE / "ed05_round14_full_maps"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"


def _active_or_archived(name: str) -> Path:
    active = RECORDS / name
    return active if active.is_dir() else ARCHIVE / name


ED1_PARENT = _active_or_archived("extended_data_fig01_2026-08-12_round12_panel_geometry")
ED2_PARENT = _active_or_archived("extended_data_fig02_2026-08-12_round22_layout_clearance")
ED5_PARENT = _active_or_archived("extended_data_fig05_2026-08-12_round13_embedded_raster_resolution")
REPORT = CANDIDATE / "ED01_ED02_ED05_FOLLOWUP_VALIDATION.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_by_name(root: Path, pattern: str) -> dict[str, Path]:
    return {path.name: path for path in root.rglob(pattern) if path.is_file()}


def exact_named_tree(left: Path, right: Path, pattern: str) -> bool:
    a = files_by_name(left, pattern)
    b = files_by_name(right, pattern)
    return set(a) == set(b) and all(sha256(a[name]) == sha256(b[name]) for name in a)


def pdf_words(path: Path, wanted: set[str]) -> dict[str, list[fitz.Rect]]:
    found = {key: [] for key in wanted}
    with fitz.open(path) as document:
        page = document[0]
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).strip()
                    if text in found:
                        found[text].append(fitz.Rect(span["bbox"]))
    return found


def pdf_semantic_equal(left: Path, right: Path) -> bool:
    with fitz.open(left) as first, fitz.open(right) as second:
        if first.page_count != second.page_count:
            return False
        for index in range(first.page_count):
            a, b = first[index], second[index]
            if (
                a.rect != b.rect
                or a.get_text("words") != b.get_text("words")
                or a.get_drawings() != b.get_drawings()
                or len(a.get_images(full=True)) != len(b.get_images(full=True))
            ):
                return False
    return True


def main() -> int:
    ed1 = json.loads((ED1 / "RENDER_VALIDATION_ROUND13.json").read_text(encoding="utf-8"))
    ed2_render = json.loads((ED2 / "RENDER_RESULT.json").read_text(encoding="utf-8"))
    ed2_manifest = json.loads((ED2 / "ED02_ROUND23_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
    ed2_geometry = ed2_render["round23_display_only_changes"]["panel_a_geometry_at_source_dpi"]
    ed5 = json.loads((ED5 / "MANIFEST_ROUND14.json").read_text(encoding="utf-8"))

    parent_words = pdf_words(
        ED2_PARENT / "panels/ed02a_four_policy_schematic_round11.pdf",
        {"preceding", "turn"},
    )
    candidate_words = pdf_words(
        ED2 / "panels/ed02a_four_policy_schematic_round11.pdf",
        {"preceding", "turn"},
    )
    parent_preceding = parent_words["preceding"][0]
    candidate_preceding = candidate_words["preceding"][0]
    pdf_left_shift_pt = float(parent_preceding.x0 - candidate_preceding.x0)

    parent_ed2_panels = files_by_name(ED2_PARENT / "panels", "*.pdf")
    candidate_ed2_panels = files_by_name(ED2 / "panels", "*.pdf")
    unchanged_ed2 = all(
        pdf_semantic_equal(parent_ed2_panels[name], candidate_ed2_panels[name])
        for name in parent_ed2_panels
        if not name.startswith("ed02a_")
    )
    ed2_plot_ready_parity = exact_named_tree(ED2_PARENT / "plot_ready", ED2 / "plot_ready", "*.csv")

    parent_audit = {
        row["role"]: row
        for row in csv.DictReader((ED5_PARENT / "SOURCE_DATA_AUDIT.csv").open(encoding="utf-8-sig"))
    }
    candidate_audit = {
        row["role"]: row
        for row in csv.DictReader((ED5 / "SOURCE_DATA_AUDIT_ROUND14.csv").open(encoding="utf-8-sig"))
    }
    source_roles = sorted(
        set(parent_audit) & set(candidate_audit)
        - {"gallery_derived_svg", "gallery_derived_png", "gallery_derived_tiff", "gallery_derived_pdf"}
    )
    ed5_source_parity = all(
        parent_audit[role]["sha256"] == candidate_audit[role]["sha256"]
        for role in source_roles
    )
    full_maps = ed5["gallery_full_map_contract"]

    checks = {
        "ed1_renderer_pass": ed1["status"] == "pass",
        "ed1_i_three_columns_exactly_align_to_f": (
            ed1["panel_i_to_f_internal_column_center_max_delta_fraction"] == 0.0
            and ed1["panel_i_to_f_internal_column_width_max_delta_fraction"] == 0.0
        ),
        "ed1_i_clear_of_j": ed1["panel_i_to_j_ylabel_clearance_px"] >= 2.0,
        "ed1_second_row_legend_clear_of_d": (
            ed1["curve_legend_to_d_overlap_px2"] == 0.0
            and ed1["curve_legend_to_d_clearance_px"] >= 2.0
        ),
        "ed1_source_audit_exact_parent_parity": sha256(ED1_PARENT / "SOURCE_DATA_AUDIT.csv")
        == sha256(ED1 / "SOURCE_DATA_AUDIT_ROUND13.csv"),
        "ed2_composite_pass_and_vector_only": (
            ed2_manifest["status"] == "pass"
            and ed2_manifest["vector_audit"]["embedded_raster_images"] == 0
        ),
        "ed2_two_rendered_character_shift_exact": abs(
            ed2_geometry["actual_left_shift_px"] - ed2_geometry["two_character_shift_px"]
        ) < 1e-9,
        "ed2_pdf_moved_left": pdf_left_shift_pt > 0,
        "ed2_arrow_and_grid_clear": (
            ed2_geometry["grey_arrow_clearance_px"] >= 1.0
            and ed2_geometry["box_clearance_px"] >= 1.0
        ),
        "ed2_other_panel_pdfs_exact_parent_parity": unchanged_ed2,
        "ed2_plot_ready_exact_parent_parity": ed2_plot_ready_parity,
        "ed5_renderer_pass": ed5["status"] == "pass",
        "ed5_all_64_maps_fully_shown": (
            full_maps["map_axis_count"] == 64
            and full_maps["all_45x45_sources"]
            and full_maps["all_maps_fully_shown"]
        ),
        "ed5_authoritative_sources_exact_parent_parity": ed5_source_parity,
    }
    report: dict[str, Any] = {
        "schema": "ed01-ed02-ed05-followup-validation/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "ed1_i_to_j_clearance_px": ed1["panel_i_to_j_ylabel_clearance_px"],
            "ed1_legend_to_d_clearance_px": ed1["curve_legend_to_d_clearance_px"],
            "ed2_two_character_shift_px": ed2_geometry["two_character_shift_px"],
            "ed2_pdf_left_shift_pt": pdf_left_shift_pt,
            "ed2_grey_arrow_clearance_px": ed2_geometry["grey_arrow_clearance_px"],
            "ed2_grid_clearance_px": ed2_geometry["box_clearance_px"],
            "ed5_full_map_contract": full_maps,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise AssertionError(", ".join(key for key, value in checks.items() if not value))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
