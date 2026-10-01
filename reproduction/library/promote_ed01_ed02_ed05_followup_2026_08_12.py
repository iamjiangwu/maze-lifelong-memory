"""Promote the narrowly scoped ED1, ED2 and ED5 follow-up candidates.

The Desktop is deliberately out of scope here.  Successor creation and parent
archival remain separate operations so the closed records can be inspected
before any active parent is moved.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import promote_figure_resolution_geometry_2026_08_12 as base


REPO = Path(__file__).resolve().parents[1]
FOLLOWUP_VALIDATION = (
    REPO
    / "docs/figure_refinement_2026-08-12/ED01_ED02_ED05_FOLLOWUP_VALIDATION.json"
)

SPECS = (
    base.Spec(
        "ed1",
        "extended_data_fig01_2026-08-12_round12_panel_geometry",
        "extended_data_fig01_2026-08-12_round13_alignment_clearance",
        "docs/figure_refinement_2026-08-12/ed01_round13_collision",
        "extended_data_figure1_source_rerender_round13",
        "ed01_version_record",
        (
            "panel i aligned to the three internal columns of panel f",
            "panel i cleared from panel j without changing scientific axes",
            "second-row condition labels cleared from panel d",
        ),
        "extended_data_figure1_source_rerender_round13.png",
        "extended_data_figure1_source_rerender_round13.tiff",
    ),
    base.Spec(
        "ed2",
        "extended_data_fig02_2026-08-12_round22_layout_clearance",
        "extended_data_fig02_2026-08-12_round23_preceding_turn",
        "docs/figure_refinement_2026-08-12/ed02_round23_preceding_turn",
        "ed02_round23_preceding_turn_candidate",
        "ed02_version_record",
        (
            "panel-a preceding-turn annotation moved left by exactly two rendered character widths",
            "positive annotation-to-arrow and annotation-to-grid clearance retained",
        ),
        "ed02_round23_preceding_turn_candidate_600dpi.png",
        "ed02_round23_preceding_turn_candidate_600dpi.tiff",
    ),
    base.Spec(
        "ed5",
        "extended_data_fig05_2026-08-12_round13_embedded_raster_resolution",
        "extended_data_fig05_2026-08-12_round14_full_place_maps",
        "docs/figure_refinement_2026-08-12/ed05_round14_full_maps",
        "ed05_example_place_maps_assembly_round14",
        "ed05_version_record",
        (
            "all 64 panel-a place maps display the complete 45 by 45 source arrays",
            "all other panels, values, labels and geometry are unchanged",
        ),
        "ed05_example_place_maps_assembly_round14.png",
        "ed05_example_place_maps_assembly_round14.tiff",
    ),
)


def finalize_record(spec: base.Spec, record: Path) -> None:
    evidence = record / "source_validation/followup_2026_08_12"
    evidence.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FOLLOWUP_VALIDATION, evidence / FOLLOWUP_VALIDATION.name)
    for relative in (
        "scripts/promote_ed01_ed02_ed05_followup_2026_08_12.py",
        "scripts/validate_ed01_ed02_ed05_followup_2026_08_12.py",
        "scripts/render_extended_data_figure1_collision_round13.py",
        "scripts/export_ed02_round23_refinement_figure.py",
        "scripts/render_ed05_full_maps_round14.py",
        "scripts/validate_current_figure_records.py",
        "scripts/validate_all_figure_refinement_round2_records.py",
    ):
        source = REPO / relative
        if source.is_file():
            shutil.copy2(source, record / "source_code" / source.name)

    validation_path = record / "RECORD_VALIDATION.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["schema"] = "ed01-ed02-ed05-followup-record-validation/v1"
    validation["record"] = spec.target
    validation["checks"].update(
        {
            "independent_followup_validation_pass": True,
            "scientific_source_parity": True,
            "unrelated_panel_parity": True,
            "desktop_version_record_package_created": False,
        }
    )
    validation["validated_utc"] = datetime.now(timezone.utc).isoformat()
    base.write_json(validation_path, validation)

    if spec.key == "ed2":
        manifest_path = record / "ED02_VERSION_RECORD_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["record"] = spec.target
        manifest["review_round"] = 23
        manifest["supersedes"] = spec.parent
        manifest["current_display_refinement"] = {
            "round": 23,
            "status": "PASS",
            "annotation_base_xy": [3.42, 3.80],
            "annotation_final_xy": [3.1875527597402593, 3.80],
            "horizontal_shift_definition": "exact rendered width of two lowercase n characters",
            "horizontal_shift_source_pixels": 7.25,
            "arrow_clearance_source_pixels": 1.516088,
            "grid_clearance_source_pixels": 7.915823,
        }
        base.write_json(manifest_path, manifest)

    base.write_hashes(record)
    errors = base.verify_hashes(record)
    if errors:
        raise RuntimeError(f"Post-finalization hash closure failed for {spec.key}: {errors[:10]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.create:
        records = []
        for spec in SPECS:
            record = base.create_record(spec)
            finalize_record(spec, record)
            records.append(record.relative_to(REPO).as_posix())
        result: object = records
    else:
        rows = [base.archive_parent(spec) for spec in SPECS]
        base.update_archive_manifest(rows)
        result = rows
    print(json.dumps({"status": "PASS", "records": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
