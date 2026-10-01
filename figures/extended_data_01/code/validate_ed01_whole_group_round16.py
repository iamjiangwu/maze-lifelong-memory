"""Validate the ED1 round-16 whole-group candidate and promoted record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
ACTIVE = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
PARENT_NAME = "extended_data_fig01_2026-08-12_round15_arial_star_readability"
TARGET_NAME = "extended_data_fig01_2026-08-15_round16_whole_group"
CANDIDATE = REPO / "docs/ed01_whole_group_promotion_2026-08-15/round16_candidate"
STEM = "extended_data_figure1_source_rerender_round16"
ROOT_STEM = "ed01_version_record"

EXPECTED_COUNTS = {
    "naive_0mo": 23,
    "return_3mo": 17,
    "return_6mo": 17,
    "return_9mo": 12,
    "return_12mo": 10,
    "experienced_day2": 74,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_record() -> Path:
    active = ACTIVE / PARENT_NAME
    archived = ARCHIVE / PARENT_NAME
    if active.is_dir():
        return active
    if archived.is_dir():
        return archived
    raise FileNotFoundError(PARENT_NAME)


def csv_exact(left: Path, right: Path) -> bool:
    a = pd.read_csv(left, low_memory=False)
    b = pd.read_csv(right, low_memory=False)
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    except AssertionError:
        return False
    return True


def allowed_difference_mask(layout_path: Path, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    layout = pd.read_csv(layout_path).set_index("panel")
    expansions = {
        "d": (0.055, 0.025, 0.065, 0.035),
        "e": (0.055, 0.025, 0.065, 0.035),
        "h": (0.060, 0.025, 0.045, 0.035),
        "i": (0.040, 0.020, 0.045, 0.040),
    }
    for panel, (left, right, bottom, top) in expansions.items():
        row = layout.loc[panel]
        x0 = max(0, int(np.floor((float(row.x0_fraction) - left) * width)))
        x1 = min(width, int(np.ceil((float(row.x1_fraction) + right) * width)))
        y0 = max(0, int(np.floor((1.0 - float(row.y1_fraction) - top) * height)))
        y1 = min(height, int(np.ceil((1.0 - float(row.y0_fraction) + bottom) * height)))
        mask[y0:y1, x0:x1] = True
    return mask


def raster_difference_audit(parent_png: Path, candidate_png: Path) -> dict[str, object]:
    old = np.asarray(Image.open(parent_png).convert("RGB"))
    new = np.asarray(Image.open(candidate_png).convert("RGB"))
    if old.shape != new.shape:
        return {"same_shape": False, "outside_allowed_pixels": -1}
    changed = np.any(old != new, axis=2)
    allowed = allowed_difference_mask(CANDIDATE / "LAYOUT_AUDIT_ROUND16.csv", (old.shape[1], old.shape[0]))
    outside = changed & ~allowed
    audit_image = np.full_like(old, 255)
    audit_image[allowed] = [245, 245, 245]
    audit_image[changed & allowed] = [204, 55, 45]
    audit_image[outside] = [40, 90, 190]
    Image.fromarray(audit_image).save(CANDIDATE / "ROUND16_RASTER_DIFFERENCE_AUDIT.png")
    changed_y, changed_x = np.where(changed)
    bbox = None
    if changed_x.size:
        bbox = [
            int(changed_x.min()),
            int(changed_y.min()),
            int(changed_x.max()) + 1,
            int(changed_y.max()) + 1,
        ]
    return {
        "same_shape": True,
        "shape": list(old.shape),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "outside_allowed_pixels": int(outside.sum()),
        "changed_bbox_pixels": bbox,
        "allowed_panels": ["d", "e", "h", "i"],
    }


def verify_hash_ledger(record: Path) -> list[str]:
    ledger = record / "RECORD_HASHES.csv"
    if not ledger.is_file():
        return ["missing RECORD_HASHES.csv"]
    with ledger.open(newline="", encoding="utf-8-sig") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != ledger.name
    }
    errors = [f"unhashed:{path}" for path in sorted(set(actual) - set(expected))]
    errors.extend(f"missing:{path}" for path in sorted(set(expected) - set(actual)))
    for relative in sorted(set(actual) & set(expected)):
        row = expected[relative]
        path = actual[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{relative}")
    return errors


def validate(record_mode: bool) -> dict[str, object]:
    parent = parent_record()
    validation = json.loads((CANDIDATE / "RENDER_VALIDATION_ROUND16.json").read_text(encoding="utf-8"))
    source = pd.read_csv(
        CANDIDATE / "source_data/ed01_tile_accumulation_whole_group_common_support_current153.csv"
    )
    inclusion = pd.read_csv(CANDIDATE / "source_data/source_session_inclusion_audit.csv")
    group_counts = (
        source.groupby("month_group_key", observed=True)["n_averaged_sessions"]
        .first()
        .astype(int)
        .to_dict()
    )
    text_root = ET.parse(CANDIDATE / f"{STEM}.svg").getroot()
    svg_text = " ".join(
        element.text or "" for element in text_root.iter() if element.tag.endswith("text")
    )
    svg_images = sum(element.tag.endswith("image") for element in text_root.iter())
    with fitz.open(CANDIDATE / f"{STEM}.pdf") as document:
        pdf_pages = document.page_count
        pdf_images = sum(len(page.get_images(full=True)) for page in document)
        pdf_text = " ".join(page.get_text() for page in document)
        pdf_fonts = sorted(
            {
                span[3].split("+")[-1]
                for page in document
                for font in page.get_fonts(full=True)
                for span in [font]
            }
        )
    with Image.open(CANDIDATE / f"{STEM}.tiff") as image:
        tiff_size = list(image.size)
        tiff_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
        tiff_compression = str(image.info.get("compression", "")).lower()

    unchanged_pairs = {
        "accumulation": (
            parent / "drawn_values/ACCUMULATION_SUMMARY_ROUND15.csv",
            CANDIDATE / "ACCUMULATION_SUMMARY_ROUND16.csv",
        ),
        "speed": (
            parent / "drawn_values/SPEED_SUMMARY_ROUND15.csv",
            CANDIDATE / "SPEED_SUMMARY_ROUND16.csv",
        ),
        "time_efficiency": (
            parent / "drawn_values/TIME_EFFICIENCY_SUMMARY_ROUND15.csv",
            CANDIDATE / "TIME_EFFICIENCY_SUMMARY_ROUND16.csv",
        ),
        "tile_efficiency": (
            parent / "drawn_values/TILE_EFFICIENCY_SUMMARY_ROUND15.csv",
            CANDIDATE / "TILE_EFFICIENCY_SUMMARY_ROUND16.csv",
        ),
        "panel_i_cycles": (
            parent / "drawn_values/PANEL_I_CYCLE_SUMMARY_ROUND15.csv",
            CANDIDATE / "PANEL_I_CYCLE_SUMMARY_ROUND16.csv",
        ),
        "panel_i_sequence": (
            parent / "drawn_values/PANEL_I_END_NODE_SEQUENCE_ROUND15.csv",
            CANDIDATE / "PANEL_I_END_NODE_SEQUENCE_ROUND16.csv",
        ),
        "panel_i_selection": (
            parent / "drawn_values/PANEL_I_SELECTION_AUDIT_ROUND15.csv",
            CANDIDATE / "PANEL_I_SELECTION_AUDIT_ROUND16.csv",
        ),
        "panel_a_jump_mask": (
            parent / "drawn_values/PANEL_A_JUMP_MASK_AUDIT_ROUND15.csv",
            CANDIDATE / "PANEL_A_JUMP_MASK_AUDIT_ROUND16.csv",
        ),
        "layout": (
            parent / "LAYOUT_AUDIT.csv",
            CANDIDATE / "LAYOUT_AUDIT_ROUND16.csv",
        ),
    }
    unchanged = {name: csv_exact(left, right) for name, (left, right) in unchanged_pairs.items()}
    raster = raster_difference_audit(
        parent / "ed01_version_record_600dpi.png",
        CANDIDATE / f"{STEM}.png",
    )
    checks = {
        "renderer_validation_pass": validation.get("status") == "pass" and not validation.get("failed_checks"),
        "whole_group_counts_exact": group_counts == EXPECTED_COUNTS,
        "all_153_sessions_included": len(inclusion) == 153 and inclusion["whole_group_included"].astype(bool).all(),
        "no_load_failures": inclusion["load_ok"].astype(bool).all(),
        "fully_observed_0_to_5200": (
            float(source["axis_value_tiles"].min()) == 0.0
            and float(source["axis_value_tiles"].max()) == 5200.0
            and source["n_active_sessions"].eq(source["n_averaged_sessions"]).all()
        ),
        "both_event_families": set(source["event_type"]) == {"reward", "long_path_reward"},
        "open_body_mass_axis": "Body mass speed" in pdf_text and "Body-mass speed" not in pdf_text,
        "open_panel_i_titles": (
            "Within round coverage" in svg_text
            and "Coverage round efficiency" in svg_text
            and "Within-round" not in svg_text
            and "Coverage-round" not in svg_text
        ),
        "all_non_target_tables_exact": all(unchanged.values()),
        "raster_differences_confined_to_d_e_h_i": raster.get("outside_allowed_pixels") == 0,
        "svg_editable_no_images": svg_images == 0 and bool(svg_text.strip()),
        "pdf_vector_one_page": pdf_pages == 1 and pdf_images == 0,
        "pdf_arial_only": set(pdf_fonts).issubset({"ArialMT", "Arial-BoldMT"}),
        "tiff_600dpi_lzw": (
            tiff_size == [4500, 4500]
            and len(tiff_dpi) == 2
            and all(abs(value - 600) < 1 for value in tiff_dpi)
            and tiff_compression in {"tiff_lzw", "lzw"}
        ),
    }
    record_checks: dict[str, bool] = {}
    if record_mode:
        record = ACTIVE / TARGET_NAME
        record_checks = {
            "promoted_record_exists": record.is_dir(),
            "record_hash_closure": record.is_dir() and not verify_hash_ledger(record),
            "root_svg_matches_candidate": record.is_dir() and sha256(record / f"{ROOT_STEM}.svg") == sha256(CANDIDATE / f"{STEM}.svg"),
            "root_pdf_matches_candidate": record.is_dir() and sha256(record / f"{ROOT_STEM}.pdf") == sha256(CANDIDATE / f"{STEM}.pdf"),
            "root_600dpi_png_matches_candidate": record.is_dir() and sha256(record / f"{ROOT_STEM}_600dpi.png") == sha256(CANDIDATE / f"{STEM}.png"),
            "record_plot_ready_matches_candidate": record.is_dir() and sha256(record / "plot_ready/ed01_tile_accumulation_whole_group_common_support_current153.csv") == sha256(CANDIDATE / "source_data/ed01_tile_accumulation_whole_group_common_support_current153.csv"),
        }
        checks.update(record_checks)
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "schema": "ed01-round16-whole-group-validation/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "record_mode": record_mode,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "whole_group_counts": group_counts,
        "unchanged_non_target_tables": unchanged,
        "raster_difference_audit": raster,
        "pdf_fonts": pdf_fonts,
    }
    target = CANDIDATE / ("ROUND16_RECORD_VALIDATION.json" if record_mode else "ROUND16_CANDIDATE_VALIDATION.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    report = validate(args.record)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
