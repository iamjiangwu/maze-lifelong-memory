"""Promote the Extended Data Figure 5 within-metric BH3 correction.

Creation and parent archival are intentionally separate. The Desktop package
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
PARENT_NAME = "extended_data_fig05_2026-08-16_round15_bh9"
TARGET_NAME = "extended_data_fig05_2026-08-16_round16_bh3"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = REPO / "docs/figure_refinement_2026-08-16/ed05_round16_bh3"
CAPTION_DIR = REPO / "docs/manuscript_figure_captions_2026-08-10"
CAPTION_NOTES = REPO / "docs/figure_refinement_2026-08-16/CAPTION_NOTES_ROUND16.md"
STEM = "ed05_example_place_maps_assembly_round16"
ROOT_STEM = "ed05_version_record"

SPEC = base.Spec(
    "ed5",
    PARENT_NAME,
    TARGET_NAME,
    "docs/figure_refinement_2026-08-16/ed05_round16_bh3",
    STEM,
    ROOT_STEM,
    (
        "use BH separately within each metric across its three exact sign-flip contrasts",
        "remove the unsupported place-map bracket while retaining the BH-dependent PV and rate stars",
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
        raise AssertionError("BH3 candidate is pixel-identical to the BH9 parent")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
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
        CANDIDATE / "ED05_VALIDATION_ROUND16.csv",
        CANDIDATE / "SOURCE_DATA_AUDIT_ROUND16.csv",
        CANDIDATE / "LAYOUT_AUDIT_ROUND16.csv",
        CANDIDATE / "MANIFEST_ROUND16.json",
        CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND16.csv",
        CAPTION_NOTES,
        CAPTION_DIR / "MAXT_TO_BH_RECOMPUTATION_AUDIT_2026-08-16.csv",
        CAPTION_DIR / "MAXT_TO_BH_RECOMPUTATION_REPORT_2026-08-16.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing candidate inputs: {missing}")

    stage = Path(tempfile.mkdtemp(prefix=".ed05-round16-", dir=RECORDS))
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

        shutil.copy2(CANDIDATE / "ED05_VALIDATION_ROUND16.csv", stage / "ED05_VALIDATION.csv")
        shutil.copy2(CANDIDATE / "SOURCE_DATA_AUDIT_ROUND16.csv", stage / "SOURCE_DATA_AUDIT.csv")
        shutil.copy2(CANDIDATE / "LAYOUT_AUDIT_ROUND16.csv", stage / "LAYOUT_AUDIT.csv")
        shutil.copy2(CANDIDATE / "MANIFEST_ROUND16.json", stage / "ED05_ASSEMBLY_MANIFEST.json")
        shutil.copy2(CAPTION_NOTES, stage / "CAPTION_NOTES.md")
        shutil.copy2(
            CANDIDATE / "ED05_DAY_COMBINATION_BH3_ROUND16.csv",
            stage / "plot_ready/ed06_day_combination_paired_stats_corrected36.csv",
        )

        evidence = stage / "source_validation/bh3_2026_08_16"
        evidence.mkdir(parents=True, exist_ok=True)
        for source in required[4:]:
            shutil.copy2(source, evidence / source.name)
        for relative in (
            "scripts/render_ed05_example_place_maps_assembly.py",
            "scripts/render_ed05_bh3_round16.py",
            "scripts/recompute_caption_bh_from_maxt.py",
            "scripts/promote_ed05_bh3_round16.py",
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
            stage / "BH3_CORRECTION_2026_08_16.json",
            {
                "schema": "ed05-bh3-correction/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "parent_record": PARENT_NAME,
                "correction_family": "BH separately within each metric across three exact animal sign-flip tests",
                "visible_star_state": {
                    "panel_b": "no significant brackets",
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
            f"This append-only record supersedes `{PARENT_NAME}`. It applies "
            "Benjamini--Hochberg correction separately within place map, PV and ensemble rate, "
            "across each metric's three planned raw exact animal sign-flip tests. Panel b has no "
            "significant bracket; the correction-dependent stars in panels c and d are retained. "
            "Panel a, panel e, data, estimands, cohorts, geometry, colors and labels are unchanged.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\nRun `python scripts/render_ed05_bh3_round16.py`, "
            "then validate the candidate and promote with `python scripts/promote_ed05_bh3_round16.py --create`.\n",
            encoding="utf-8",
        )
        base.write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed05-bh3-record-validation/v1",
                "status": "pass",
                "record": TARGET_NAME,
                "checks": {
                    "candidate_validation_pass": True,
                    "three_within_metric_bh3_families": True,
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
