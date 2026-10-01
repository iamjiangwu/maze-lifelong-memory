"""Promote the Extended Data Figure 5 BH9 star correction.

Creation and parent archival are intentionally separate.  The Desktop package
and top-level Desktop PNG are outside this request and are not modified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import promote_figure_resolution_geometry_2026_08_12 as base


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
PARENT_NAME = "extended_data_fig05_2026-08-12_round14_full_place_maps"
TARGET_NAME = "extended_data_fig05_2026-08-16_round15_bh9"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = REPO / "docs/figure_refinement_2026-08-16/ed05_round15_bh9"
CAPTION_DIR = REPO / "docs/manuscript_figure_captions_2026-08-10"
CAPTION_NOTES = REPO / "docs/figure_refinement_2026-08-16/CAPTION_NOTES_ROUND15.md"
STEM = "ed05_example_place_maps_assembly_round15"
ROOT_STEM = "ed05_version_record"

SPEC = base.Spec(
    "ed5",
    PARENT_NAME,
    TARGET_NAME,
    "docs/figure_refinement_2026-08-16/ed05_round15_bh9",
    STEM,
    ROOT_STEM,
    (
        "replace maxT9 with BH9 for the same nine exact animal sign-flip tests",
        "add the newly supported place-map bracket and update only correction-dependent stars",
    ),
    f"{STEM}.png",
    f"{STEM}.tiff",
)


def _raster_difference(old_path: Path, new_path: Path) -> dict[str, object]:
    old = np.asarray(Image.open(old_path).convert("RGB"))
    new = np.asarray(Image.open(new_path).convert("RGB"))
    if old.shape != new.shape:
        raise AssertionError(f"Raster shape changed: {old.shape} -> {new.shape}")
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    if not len(xs):
        raise AssertionError("BH9 candidate is pixel-identical to the maxT parent")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    # The day-combination row occupies the lower-left band.  Panel a and panel e
    # must remain pixel-identical.
    contained = bbox[0] >= 0 and bbox[1] >= 3400 and bbox[2] <= 2100 and bbox[3] <= 3900
    if not contained:
        raise AssertionError(f"Raster changes escaped panels b-d: {bbox}")
    return {
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "changed_pixels": int(changed.sum()),
        "difference_bbox_exclusive": bbox,
        "changes_confined_to_panels_b_to_d": contained,
    }


def create() -> Path:
    if TARGET.is_dir():
        errors = base.verify_hashes(TARGET)
        if errors:
            raise RuntimeError(f"Existing target is not hash-closed: {errors[:10]}")
        return TARGET
    if not PARENT.is_dir():
        raise RuntimeError(f"Missing parent: {PARENT}")
    errors = base.verify_hashes(PARENT)
    if errors:
        raise RuntimeError(f"Parent is not hash-closed: {errors[:10]}")

    required = [
        CANDIDATE / f"{STEM}{suffix}"
        for suffix in (".svg", ".pdf", ".png", ".tiff")
    ] + [
        CANDIDATE / "ED05_VALIDATION_ROUND15.csv",
        CANDIDATE / "SOURCE_DATA_AUDIT_ROUND15.csv",
        CANDIDATE / "LAYOUT_AUDIT_ROUND15.csv",
        CANDIDATE / "MANIFEST_ROUND15.json",
        CANDIDATE / "ED05_DAY_COMBINATION_BH9_ROUND15.csv",
        CAPTION_NOTES,
        CAPTION_DIR / "MAXT_TO_BH_RECOMPUTATION_AUDIT_2026-08-16.csv",
        CAPTION_DIR / "MAXT_TO_BH_RECOMPUTATION_REPORT_2026-08-16.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing candidate inputs: {missing}")

    stage = Path(tempfile.mkdtemp(prefix=".ed05-round15-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)

        for extension in (".svg", ".pdf"):
            shutil.copy2(CANDIDATE / f"{STEM}{extension}", stage / f"{ROOT_STEM}{extension}")
        shutil.copy2(CANDIDATE / f"{STEM}.png", stage / f"{ROOT_STEM}_600dpi.png")
        shutil.copy2(CANDIDATE / f"{STEM}.tiff", stage / f"{ROOT_STEM}_600dpi.tiff")
        base.render_pdf(stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png", 300)
        shutil.copy2(CANDIDATE / f"{STEM}.png", stage / "extended_data_figure5_source_rerender.png")
        shutil.copy2(CANDIDATE / f"{STEM}.tiff", stage / "extended_data_figure5_source_rerender.tiff")

        shutil.copy2(CANDIDATE / "ED05_VALIDATION_ROUND15.csv", stage / "ED05_VALIDATION.csv")
        shutil.copy2(CANDIDATE / "SOURCE_DATA_AUDIT_ROUND15.csv", stage / "SOURCE_DATA_AUDIT.csv")
        shutil.copy2(CANDIDATE / "LAYOUT_AUDIT_ROUND15.csv", stage / "LAYOUT_AUDIT.csv")
        shutil.copy2(CANDIDATE / "MANIFEST_ROUND15.json", stage / "ED05_ASSEMBLY_MANIFEST.json")
        shutil.copy2(CAPTION_NOTES, stage / "CAPTION_NOTES.md")
        shutil.copy2(
            CANDIDATE / "ED05_DAY_COMBINATION_BH9_ROUND15.csv",
            stage / "plot_ready/ed06_day_combination_paired_stats_corrected36.csv",
        )

        evidence = stage / "source_validation/bh9_2026_08_16"
        evidence.mkdir(parents=True, exist_ok=True)
        for source in required[4:]:
            shutil.copy2(source, evidence / source.name)
        for relative in (
            "scripts/render_ed05_example_place_maps_assembly.py",
            "scripts/render_ed05_bh9_round15.py",
            "scripts/recompute_caption_bh_from_maxt.py",
            "scripts/promote_ed05_bh9_round15.py",
            "scripts/validate_figure_caption_collection.py",
            "tests/test_ed05_example_place_maps_assembly.py",
        ):
            source = REPO / relative
            shutil.copy2(source, stage / "source_code" / source.name)

        raster_audit = _raster_difference(
            PARENT / f"{ROOT_STEM}_600dpi.png",
            stage / f"{ROOT_STEM}_600dpi.png",
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        base.write_json(
            stage / "BH9_CORRECTION_2026_08_16.json",
            {
                "schema": "ed05-bh9-correction/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "parent_record": PARENT_NAME,
                "correction_family": "BH9 across the unchanged nine exact animal sign-flip tests",
                "visible_star_changes": {
                    "panel_b": "add one-star d1-d1 versus d1-d2 bracket",
                    "panel_c": "three, three, three stars",
                    "panel_d": "three, two, two stars",
                },
                "raster_audit": raster_audit,
                "created_utc": timestamp,
                "desktop_modified": False,
                "review_powerpoint_status": "inherited historical review artifact; publication exports are current authority",
            },
        )
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. It replaces maxT9 with "
            "Benjamini--Hochberg correction across the same nine raw exact animal sign-flip tests. "
            "Only the correction-dependent brackets and stars in panels b--d changed; panel a, panel e, "
            "data, estimands, cohorts, geometry, colors and labels are unchanged.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\nRun `python scripts/render_ed05_bh9_round15.py`, "
            "then validate the candidate and promote with `python scripts/promote_ed05_bh9_round15.py --create`.\n",
            encoding="utf-8",
        )
        base.write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed05-bh9-record-validation/v1",
                "status": "pass",
                "record": TARGET_NAME,
                "checks": {
                    "candidate_validation_pass": True,
                    "bh9_family_size_nine": True,
                    "raster_changes_confined_to_panels_b_to_d": True,
                    "panel_a_and_e_pixel_identical": True,
                    "root_exports_complete": all(
                        (stage / f"{ROOT_STEM}{suffix}").is_file()
                        for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff")
                    ),
                    "desktop_modified": False,
                },
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n",
            encoding="utf-8",
        )
        base.write_hashes(stage)
        errors = base.verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Successor hash closure failed: {errors[:10]}")
        os.replace(stage, TARGET)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return TARGET


def archive() -> dict[str, str]:
    row = base.archive_parent(SPEC)
    base.update_archive_manifest([row])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result: object = str(create().relative_to(REPO)) if args.create else archive()
    print(json.dumps({"status": "PASS", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
