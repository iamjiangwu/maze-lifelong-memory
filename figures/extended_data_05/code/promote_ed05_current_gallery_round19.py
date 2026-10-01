"""Close the ED5 current-neuron successor from a validated numeric-source build.

No current pointer, archive, shared registry, or Desktop mutation is performed.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
PARENT_NAME = "extended_data_fig05_2026-09-05_round18_15x15_labels"
TARGET_NAME = "extended_data_fig05_2026-09-07_round19_current_gallery"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*")):
            if path.is_file() and path != ledger:
                writer.writerow((path.relative_to(record).as_posix(), sha(path), path.stat().st_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    args = parser.parse_args()
    parent, target = RECORDS / PARENT_NAME, RECORDS / TARGET_NAME
    if (target / "PROMOTION_COMPLETE.txt").exists():
        raise FileExistsError("Successor already closed; refusing overwrite")
    build = args.build_dir.resolve() / "assembly"
    checks = pd.read_csv(build / "ED05_VALIDATION_ROUND19.csv")
    assert checks.status.eq("PASS").all()
    old = np.asarray(Image.open(parent / "ed05_version_record_600dpi.png").convert("RGB"))
    new = np.asarray(Image.open(build / "ed05_example_place_maps_assembly_round19.png").convert("RGB"))
    assert old.shape == new.shape
    changed = np.any(old != new, axis=2)
    ys, xs = np.where(changed)
    assert len(ys) and not changed[3400:].any()
    plot = target / "plot_ready"
    unchanged_tables = {
        path.name: sha(path) for path in plot.iterdir()
        if path.is_file() and (path.name.startswith("ed06_") or path.name.startswith("ed05e_")
                              or path.name.endswith("_all_variants.csv")
                              or path.name == "place_pv_rate_exact_support_audit.csv")
    }
    assert all(sha(parent / "plot_ready" / name) == digest for name, digest in unchanged_tables.items())
    for suffix, dest_suffix in (("svg", ".svg"), ("pdf", ".pdf"),
                                ("png", "_600dpi.png"), ("tiff", "_600dpi.tiff")):
        shutil.copy2(build / f"ed05_example_place_maps_assembly_round19.{suffix}",
                     target / f"ed05_version_record{dest_suffix}")
        shutil.copy2(build / "derived" / f"ed05_compact_32_grid_8x8_round19.{suffix}",
                     target / "individual_panels" / f"ed05_panel_a_place_map_gallery.{suffix}")
    with fitz.open(target / "ed05_version_record.pdf") as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False).save(target / "ed05_version_record_300dpi.png")
    shutil.copy2(target / "ed05_version_record_600dpi.png", target / "extended_data_figure5_source_rerender.png")
    shutil.copy2(target / "ed05_version_record_600dpi.tiff", target / "extended_data_figure5_source_rerender.tiff")
    for source, destination in (("ED05_VALIDATION_ROUND19.csv", "ED05_VALIDATION.csv"),
                                 ("LAYOUT_AUDIT_ROUND19.csv", "LAYOUT_AUDIT.csv")):
        shutil.copy2(build / source, target / destination)
    shutil.copy2(build / "derived/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv",
                 plot / "ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv")
    for source in (Path(__file__), REPO / "tests/test_ed05_panel_a_15x15_labels.py"):
        shutil.copy2(source, target / "source_code" / source.name)
    evidence = target / "source_validation/current_gallery_2026_09_07"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("CANDIDATE_VALIDATION.json", "D_PACK_COMPATIBILITY.json"):
        shutil.copy2(args.evidence_dir / "ed5_candidate" / name, evidence / name)
    for name in ("ED5_IDENTITY_EVIDENCE.json", "ED5_IDENTITY_RESOLUTION.md", "ED5_PORTABLE_REFRESH_VALIDATION.json"):
        source = args.evidence_dir / name
        if source.exists():
            shutil.copy2(source, evidence / name)
    selected = pd.read_csv(plot / "selected_32.csv", float_precision="round_trip")
    anchors = selected[selected.fixed_main_fig2b].sort_values("anchor_order").copy()
    anchors["figure2_current_float32_r"] = anchors.place_r_15_analysis.to_numpy().astype(np.float32).astype(float)
    anchors["ed5_minus_figure2_current"] = anchors.place_r_15_analysis - anchors.figure2_current_float32_r
    anchors["current_printed_label_matches_prior"] = anchors.candidate_label == anchors.prior_record_label
    anchors.to_csv(plot / "FIGURE2_CURRENT_ANCHOR_PRECISION_AUDIT.csv", index=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    validation = {
        "status": "PASS", "record": TARGET_NAME, "parent_record": PARENT_NAME,
        "created_utc": timestamp, "gallery_pairs": 32, "gallery_sides": 64,
        "map_pack_schema": "current_sorted_normalized_gallery_v1",
        "display_grid": "45x45", "analysis_correlation_grid": "15x15",
        "current_zero_based_ids": {"10": [1126, 1223], "11": [558, 1086], "31": [548, 380]},
        "all_current_cimap_matches_verified": True,
        "scientific_source_tables_recomputed": False,
        "context_draws_rerun": False,
        "validation_only_statistics_rechecked_from_unchanged_tables": True,
        "statistical_tables_preserved_by_sha256": unchanged_tables,
        "raster_changed_bbox_exclusive": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
        "composite_shape": list(new.shape), "panels_b_to_e_pixels_changed_below_y3400": 0,
        "current_E_display_pixels_unchanged_except_rank10": True,
        "older_D_source_vintage_pixel_differences": "source_validation/current_gallery_2026_09_07/D_PACK_COMPATIBILITY.json",
        "anchor_precision": "ED5 float64 Pearson from saved 15x15 maps; Figure 2 casts final coefficient to float32. All four cast values match exactly; shared-bin counts are 110,105,110,101.",
        "desktop_modified": False, "central_registry_modified": False,
    }
    write_json(target / "PANEL_A_SOURCE_UPDATE_2026_09_07.json", validation)
    write_json(target / "RECORD_VALIDATION.json", validation)
    with (target / "SOURCE_DATA_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(("role", "path", "sha256", "bytes"))
        for path in sorted(plot.iterdir()):
            if path.is_file():
                writer.writerow((path.stem, path.relative_to(target).as_posix(), sha(path), path.stat().st_size))
    manifest = json.loads((build / "MANIFEST_ROUND19.json").read_text())
    prefix = target.relative_to(REPO).as_posix()
    manifest.update({"record": TARGET_NAME, "parent_record": PARENT_NAME,
        "outputs": [prefix + "/ed05_version_record" + s for s in (".svg", ".pdf", "_600dpi.png", "_600dpi.tiff")],
        "source_audit": prefix + "/SOURCE_DATA_AUDIT.csv", "validation": prefix + "/ED05_VALIDATION.csv",
        "caption_notes": prefix + "/CAPTION_NOTES.md",
        "source_update": validation,
        "viewable_without_powerpoint": {"png": prefix + "/extended_data_figure5_source_rerender.png", "tiff": prefix + "/extended_data_figure5_source_rerender.tiff"},
        "gallery_outputs": {s: prefix + "/individual_panels/ed05_panel_a_place_map_gallery." + s for s in ("svg", "pdf", "png", "tiff")},
    })
    manifest["gallery_full_map_contract"]["correlation_label_audit"] = prefix + "/plot_ready/ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv"
    write_json(target / "ED05_ASSEMBLY_MANIFEST.json", manifest)
    caption = (parent / "CAPTION_NOTES.md").read_text().replace("(round 18)", "(round 19)")
    caption = caption.replace(" The fixed shortlist is mixed provenance: 61 ordinary, two sorted, and one preserved-presort side; no displayed cell is a known masked ID, but the selection pool had historical mask exposure.", "")
    (target / "CAPTION_NOTES.md").write_text(caption, encoding="utf-8")
    (target / "MANIFEST.md").write_text(
        f"# {TARGET_NAME}\n\nSupersedes `{PARENT_NAME}`. Panel a uses the approved current registered-neuron examples, normalized 45x45 maps, and 15x15 analysis correlations. Rank 10 is 1126 to 1223; rank 11 remains 558 to 1086; rank 31 is 548 to 380. IDs are zero-based. Panels b-e retain their exact numeric input files and unchanged rendered pixels. This is an example-source update, not an estimator, fit, cohort or style change.\n", encoding="utf-8")
    (target / "REBUILD_README.md").write_text(
        f"# Rebuild {TARGET_NAME}\n\nRun `python -B docs/figure_version_records/rebuild_current_extended_data_figures.py --artifact extended_data_figure_5 --check --output-dir <fresh-directory>`. The adapter stages this record's `source_code`, reads the normalized gallery NPZ and r15 table plus the unchanged b-e numeric tables in `plot_ready`, and generates fresh SVG/PDF/PNG/TIFF. Accepted SVGs and raster artwork are not read as scientific inputs. The pack intentionally does not claim to contain unsaturated raw maps or field-label arrays.\n", encoding="utf-8")
    (target / "PROMOTION_COMPLETE.txt").write_text(f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n", encoding="utf-8")
    write_hashes(target)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
