"""Validate ED7 round 11 against the accepted round-10 record and source locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[1]
ED07_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed07_age_controls"
)
ASSEMBLY = ED07_ROOT / "assembly"
ROUND10 = ASSEMBLY / "display_round10"
ROUND11 = ASSEMBLY / "display_round11"
MANIFEST_NAME = "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json"
ROUND10_MANIFEST = ROUND10 / MANIFEST_NAME
ROUND11_MANIFEST = ROUND11 / MANIFEST_NAME
ROUND10_STEM = ROUND10 / "ed07_age_controls_six_panel_round10"
ROUND11_STEM = ROUND11 / "ed07_age_controls_six_panel_round11"
REPORT = ROUND11 / "ED07_SIX_PANEL_VALIDATION.json"
TOP_LABEL = "Day 1 early vs Day 2 correlation"
BOTTOM_LABEL = "Long-gap correlation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_signature(manifest: dict) -> list[tuple[str, str, str]]:
    return [
        (item["role"], item["path"], item["sha256"])
        for item in manifest["inputs"]
    ]


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"check": name, "pass": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    prior = json.loads(ROUND10_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(ROUND11_MANIFEST.read_text(encoding="utf-8"))
    check("round11_status", current["status"] == "PASS", current["status"])
    check(
        "round10_input_identity",
        input_signature(current) == input_signature(prior),
        input_signature(current),
    )
    for field in (
        "panel_inventory",
        "top_estimand",
        "top_primary_scope",
        "bottom_estimand",
        "bottom_primary_scope",
        "manual_mask_profile",
        "excluded_from_all_ed7_primary_analyses",
        "representation_space",
        "caption_only_statistics",
    ):
        check(
            f"round10_{field}_identity",
            current[field] == prior[field],
            current[field],
        )
    display = current["display_contract"]
    prior_display = prior["display_contract"]
    check("derived_from_round10", display["derived_from"] == "display_round10", display)
    check("top_y_label_exact", display["top_y_axis_label"] == TOP_LABEL, display)
    check("bottom_y_label_exact", display["bottom_y_axis_label"] == BOTTOM_LABEL, display)
    check("y_labels_one_line_without_r", all("\n" not in label and "(r)" not in label for label in (TOP_LABEL, BOTTOM_LABEL)), [TOP_LABEL, BOTTOM_LABEL])
    for field in (
        "row_size_inches",
        "assembly_size_inches",
        "x_axis_label",
        "marker_policy",
        "panel_letters",
    ):
        check(
            f"round10_display_{field}_identity",
            display[field] == prior_display[field],
            display[field],
        )

    output_records = {item["path"]: item for item in current["outputs"]}
    for relative, item in output_records.items():
        path = REPO_ROOT / relative
        check(f"output_exists::{path.name}", path.is_file(), relative)
        check(f"output_hash::{path.name}", sha256(path) == item["sha256"], item["sha256"])

    svg_text = ROUND11_STEM.with_suffix(".svg").read_text(encoding="utf-8")
    check("svg_vector_pure", "<image" not in svg_text, svg_text.count("<image"))
    check("svg_editable_text", svg_text.count("<text") >= 60, svg_text.count("<text"))
    for letter in "abcdef":
        check(f"panel_letter_{letter}", f">{letter}<" in svg_text, letter)
    for phrase in ("Age (days)", TOP_LABEL, BOTTOM_LABEL):
        check(f"svg_label::{phrase}", phrase in svg_text, phrase)
    for phrase in (
        "(r)",
        "First 30 min–Day 2 correlation",
        "Long-gap cross-session correlation",
        "Age on Day 1",
        "Destination-session age",
        "original-source",
        "sorted",
    ):
        check(f"svg_forbidden::{phrase}", phrase not in svg_text, phrase)

    with fitz.open(ROUND11_STEM.with_suffix(".pdf")) as document:
        page = document[0]
        pdf_text = page.get_text()
        check("pdf_one_page", document.page_count == 1, document.page_count)
        check("pdf_vector_pure", len(page.get_images(full=True)) == 0, len(page.get_images(full=True)))
        check("pdf_vector_drawings", len(page.get_drawings()) > 300, len(page.get_drawings()))
        check("pdf_selectable_text", len(pdf_text) > 350, len(pdf_text))
        check("pdf_labels_exact", TOP_LABEL in pdf_text and BOTTOM_LABEL in pdf_text, pdf_text)
        check("pdf_no_r_qualifier", "(r)" not in pdf_text, "(r) absent")

    prior_png = ROUND10_STEM.with_suffix(".png")
    current_png = ROUND11_STEM.with_suffix(".png")
    with Image.open(prior_png) as prior_image, Image.open(current_png) as current_image:
        check("round10_dimension_identity", current_image.size == prior_image.size == (4500, 3090), {"round10": list(prior_image.size), "round11": list(current_image.size)})
        difference = ImageChops.difference(prior_image.convert("RGB"), current_image.convert("RGB"))
        difference_bbox = difference.getbbox()
        check(
            "pixel_delta_confined_to_y_title_strip",
            difference_bbox is not None and difference_bbox[0] >= 100 and difference_bbox[2] <= 250,
            difference_bbox,
        )
        array_prior = np.asarray(prior_image.convert("RGB"), dtype=np.int16)
        array_current = np.asarray(current_image.convert("RGB"), dtype=np.int16)
        check(
            "pixels_right_of_y_title_strip_identical",
            np.array_equal(array_prior[:, 250:], array_current[:, 250:]),
            "x >= 250",
        )
    with Image.open(ROUND11_STEM.with_suffix(".tiff")) as tiff:
        dpi = tuple(float(value) for value in tiff.info["dpi"])
        check("tiff_600_dpi", tuple(round(value) for value in dpi) == (600, 600), list(dpi))
        check("tiff_lzw", tiff.tag_v2.get(259) == 5, tiff.tag_v2.get(259))

    root_identity = current["root_companion_identity"]
    check("root_companions_synchronized", root_identity["synchronized"] is True, root_identity)
    root_png = ED07_ROOT / "extended_data_figure7_source_rerender_round1.png"
    root_tiff = ED07_ROOT / "extended_data_figure7_source_rerender_round1.tiff"
    root_caption = ED07_ROOT / "ED07_CAPTION_NOTES.md"
    check("root_png_identity", sha256(root_png) == sha256(current_png), sha256(root_png))
    check("root_tiff_identity", sha256(root_tiff) == sha256(ROUND11_STEM.with_suffix(".tiff")), sha256(root_tiff))
    check("root_caption_identity", sha256(root_caption) == sha256(ROUND11 / "ED07_SIX_PANEL_CAPTION_NOTES.md"), sha256(root_caption))

    report = {
        "schema": "ed07_six_panel_round11_validation/v1",
        "status": "PASS",
        "checks_passed": len(checks),
        "checks": checks,
        "round11_manifest_sha256": sha256(ROUND11_MANIFEST),
        "pixel_difference_bbox_vs_round10": list(difference_bbox),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks_passed": len(checks), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
