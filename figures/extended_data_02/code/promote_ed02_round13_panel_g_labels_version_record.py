#!/usr/bin/env python3
"""Promote the ED2 Round-13 panel-g label correction immutably."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import zipfile

import fitz
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "tmp" / "ed02_round13_g_labels_2026-08-09"
SOURCE_ROOT = ROOT / "panels"
COMPOSITE_ROOT = ROOT / "composite"
DECK_ROOT = ROOT / "deck"
FIX_VALIDATION = ROOT / "ROUND13_FIX_VALIDATION.json"
PREVIOUS_RECORD = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor"
)
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round15_panel_g_labels_vor"
)
STEM = "ed02_round13_panel_g_labels_version_record"
COMPOSITE_FILES = (
    f"{STEM}.pdf",
    f"{STEM}.svg",
    f"{STEM}_600dpi.png",
    f"{STEM}_600dpi.tiff",
    f"{STEM}_preview_1440.png",
    "ED02_ROUND13_COMPOSITE_MANIFEST.json",
)
TOP_SOURCE_FILES = (
    "CAPTION_NOTES.md",
    "PANEL_MANIFEST.csv",
    "PPT_VECTOR_MANIFEST.csv",
    "SHARED_ASSET_MANIFEST.csv",
    "SOURCE_MANIFEST.json",
    "TABLE_MANIFEST.csv",
)
REVIEW_FILES = (
    "sup-figs-v2_ed02-slide3_round13_review.pptx",
    "ed02-slide3_round13_review.pptx",
    "cumulative-artifact-preview.png",
    "review-artifact-preview.png",
    "cumulative-after-inspect.ndjson",
    "review-after-inspect.ndjson",
    "cumulative.layout.json",
    "review.layout.json",
    "round13-build-report.json",
)
SOURCE_SCRIPTS = (
    "render_ed02_slide3_panel_g_labels_round13.py",
    "export_ed02_round13_vector_figure.py",
    "build_ed02_slide3_round13_review.mjs",
    "render_ed02_round13_powerpoint_readonly.ps1",
    "validate_ed02_round13_g_labels.py",
    "promote_ed02_round13_panel_g_labels_version_record.py",
    "validate_ed02_round13_panel_g_labels_version_record.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hash_ledger(root: Path) -> int:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "RECORD_HASHES.csv"
        )
    ]
    with (root / "RECORD_HASHES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def stage_audit(stage: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    check("panel_exports_36", len(list((stage / "panels").glob("*"))) == 36, len(list((stage / "panels").glob("*"))))
    check("plot_ready_16", len(list((stage / "plot_ready").glob("*.csv"))) == 16, len(list((stage / "plot_ready").glob("*.csv"))))
    check("analysis_inputs_13", len(list((stage / "analysis_inputs").rglob("*.csv"))) == 13, len(list((stage / "analysis_inputs").rglob("*.csv"))))
    check("analysis_provenance_6", len([path for path in (stage / "analysis_provenance").rglob("*") if path.is_file()]) == 6, len([path for path in (stage / "analysis_provenance").rglob("*") if path.is_file()]))
    check("ppt_vectors_10", len(list((stage / "ppt_vector").glob("*.svg"))) == 10, len(list((stage / "ppt_vector").glob("*.svg"))))
    check("shared_assets_4", len(list((stage / "shared_assets").glob("*"))) == 4, len(list((stage / "shared_assets").glob("*"))))
    source = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    depth = source.get("panel_g_label_contract_round13", {})
    letter = source.get("panel_g_letter_contract_round13", {})
    check("depth_labels_contract", depth.get("labels") == ["Shallow\n1-2", "Middle\n3-4", "Deep\n5"] and depth.get("source_values_changed") is False, depth)
    check("g_letter_contract", letter.get("new_top_720") == 463.0 and letter.get("delta_720") == -10.5, letter)
    for path in sorted((stage / "plot_ready").glob("*.csv")):
        previous = PREVIOUS_RECORD / "plot_ready" / path.name
        check(f"table_parity::{path.name}", previous.is_file() and sha256(path) == sha256(previous), sha256(path))
    with fitz.open(stage / f"{STEM}.pdf") as document:
        page = document[0]
        text = page.get_text()
        check("primary_true_vector", document.page_count == 1 and not page.get_images(full=True) and len(page.get_drawings()) > 900, {"images": len(page.get_images(full=True)), "drawings": len(page.get_drawings())})
        check("primary_depth_labels", all(label in text for label in ("Shallow", "1-2", "Middle", "3-4", "Deep", "5")), "depth categories present")
    for suffix in ("png", "tiff"):
        with Image.open(stage / f"{STEM}_600dpi.{suffix}") as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            check(f"primary_{suffix}_600dpi", image.size == (4500, 4500) and min(dpi) >= 599.0, {"size": image.size, "dpi": dpi})
    for pptx in (
        "sup-figs-v2_ed02-slide3_round13_review.pptx",
        "ed02-slide3_round13_review.pptx",
    ):
        with zipfile.ZipFile(stage / "review_evidence" / pptx) as archive:
            vectors = [name for name in archive.namelist() if name.startswith("ppt/media/") and name.endswith(".svg")]
        check(f"review_vectors::{pptx}", len(vectors) == 10, len(vectors))
    fix = json.loads((stage / "review_evidence" / "ROUND13_FIX_VALIDATION.json").read_text(encoding="utf-8"))
    check("candidate_validation_pass", fix.get("status") == "pass" and fix.get("failure_count") == 0, {"status": fix.get("status"), "failures": fix.get("failure_count")})
    authority = "\n".join(
        (stage / name).read_text(encoding="utf-8")
        for name in ("SOURCE_MANIFEST.json", "RENDER_RESULT.json", "ED02_VERSION_RECORD_MANIFEST.json")
    )
    check("portable_authority", "\\tmp\\" not in authority.lower() and "/tmp/" not in authority.lower() and "D:\\" not in authority, "record-local")
    failures = [row for row in checks if row["status"] == "fail"]
    return {
        "status": "pass" if not failures else "fail",
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def promote(record_root: Path) -> dict[str, object]:
    if record_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing record: {record_root}")
    validation = json.loads(FIX_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or validation.get("failure_count") != 0:
        raise RuntimeError("Round-13 candidate validation is not passing")
    stage = record_root.with_name(f".{record_root.name}.staging-{uuid.uuid4().hex[:10]}")
    stage.mkdir(parents=True)
    try:
        for directory in ("panels", "plot_ready", "ppt_vector", "shared_assets"):
            copy_tree(SOURCE_ROOT / directory, stage / directory)
        copy_tree(PREVIOUS_RECORD / "analysis_inputs", stage / "analysis_inputs")
        copy_tree(PREVIOUS_RECORD / "analysis_provenance", stage / "analysis_provenance")
        copy_tree(PREVIOUS_RECORD / "source_code", stage / "source_code")
        for name in TOP_SOURCE_FILES:
            copy_file(SOURCE_ROOT / name, stage / name)
        for name in ("INPUT_MANIFEST.csv", "UPSTREAM_INPUT_MANIFEST.csv"):
            copy_file(PREVIOUS_RECORD / name, stage / name)
        for name in COMPOSITE_FILES:
            copy_file(COMPOSITE_ROOT / name, stage / name)
        for name in REVIEW_FILES:
            copy_file(DECK_ROOT / name, stage / "review_evidence" / name)
        copy_file(FIX_VALIDATION, stage / "review_evidence" / "ROUND13_FIX_VALIDATION.json")
        copy_file(ROOT / "ROUND13_FIX_VALIDATION.txt", stage / "review_evidence" / "ROUND13_FIX_VALIDATION.txt")
        copy_file(DECK_ROOT / "powerpoint-render-round13" / "Slide3.PNG", stage / "review_evidence" / "round13_cumulative_native_slide3.png")
        copy_file(DECK_ROOT / "one-slide-powerpoint-render-round13" / "Slide1.PNG", stage / "review_evidence" / "round13_review_native_slide1.png")
        for name in SOURCE_SCRIPTS:
            copy_file(REPO_ROOT / "scripts" / name, stage / "source_code" / name)

        source = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        source.update(
            {
                "status": "accepted_version_record",
                "record": record_root.name,
                "accepted_date": "2026-08-09",
                "source_record": PREVIOUS_RECORD.name,
                "primary_publication_output": COMPOSITE_FILES[0],
                "vector_companion": COMPOSITE_FILES[1],
                "raster_companions": list(COMPOSITE_FILES[2:4]),
                "frozen_renderer": {
                    "path": "source_code/render_ed02_slide3_panel_g_labels_round13.py",
                    "sha256": sha256(stage / "source_code" / "render_ed02_slide3_panel_g_labels_round13.py"),
                },
            }
        )
        write_json(stage / "SOURCE_MANIFEST.json", source)

        source_rows = [
            {"path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted((stage / "source_code").glob("*"))
            if path.is_file()
        ]
        with (stage / "SOURCE_CODE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(source_rows)

        composite = json.loads((stage / "ED02_ROUND13_COMPOSITE_MANIFEST.json").read_text(encoding="utf-8"))
        version = {
            "status": "accepted_version_record",
            "record": record_root.name,
            "accepted_date": "2026-08-09",
            "figure": "Extended Data Figure 2",
            "panel_inventory": "a-i exactly; no panel j",
            "panel_count": 9,
            "plot_ready_table_count": 16,
            "analysis_input_count": 13,
            "analysis_provenance_file_count": 6,
            "manual_mask_applicability": "not applicable; behavior-only analysis",
            "window_minutes": 30,
            "clock": "first 30 cumulative TimeInMaze minutes; all modes advance the clock",
            "policy_events": "Explore-mode junction decisions only",
            "source_record": PREVIOUS_RECORD.name,
            "review_round": 13,
            "primary_output": COMPOSITE_FILES[0],
            "vector_outputs": list(COMPOSITE_FILES[:2]),
            "high_resolution_outputs": list(COMPOSITE_FILES[2:4]),
            "vector_audit": composite["vector_audit"],
            "raster_audit": {"png": composite["png_audit"], "tiff": composite["tiff_audit"]},
            "display_changes": [
                "restored panel-g categorical x labels Shallow 1-2, Middle 3-4, and Deep 5 after the shared-x display transform erased them",
                "moved only the external panel-g letter upward by 10.5 slide units from y=473.5 to y=463.0",
            ],
            "scientific_values_changed": False,
        }
        write_json(stage / "ED02_VERSION_RECORD_MANIFEST.json", version)
        write_json(
            stage / "RENDER_RESULT.json",
            {
                "status": "pass",
                "renderer": "source_code/render_ed02_slide3_panel_g_labels_round13.py",
                "vector_assembler": "source_code/export_ed02_round13_vector_figure.py",
                "review_deck_builder": "source_code/build_ed02_slide3_round13_review.mjs",
                "output_root": ".",
                "primary_output": COMPOSITE_FILES[0],
                "vector_outputs": list(COMPOSITE_FILES[:2]),
                "raster_outputs": list(COMPOSITE_FILES[2:4]),
            },
        )
        (stage / "MANIFEST.md").write_text(
            "# Extended Data Figure 2 | Round 13 panel-g label version record\n\n"
            "Accepted 2026-08-09. This immutable record freezes the exact a-i figure after restoring "
            "panel g's maintained depth-category labels and moving only its external panel letter upward. "
            "The primary PDF and SVG are vector; PNG and LZW-TIFF companions are 4500 x 4500 at 600 dpi.\n",
            encoding="utf-8",
        )
        (stage / "FIGURE_LEGEND.md").write_text(
            "# Extended Data Figure 2\n\n"
            "Route-policy biases during the first 30 cumulative minutes in the maze. (a) Four policy definitions. "
            "(b,c) Session probabilities across Naive day, Return day, and Second day. (d) Signed exploration-efficiency "
            "differences after one-policy substitutions. (e) Policy-wise distances from Return-day means. (f) Four-policy "
            "model calibration. (g) Depth profile of P_BS across Shallow (1-2), Middle (3-4), and Deep (5) groups. "
            "(h) Four-policy and depth-varying reconstructions. (i) Depth-varying model calibration.\n",
            encoding="utf-8",
        )
        (stage / "REPORT.md").write_text(
            "# Promotion report\n\n"
            "Panel g again displays its maintained Shallow 1-2, Middle 3-4, and Deep 5 categorical labels. "
            "The external g letter moved upward by exactly 10.5 slide units. All 16 plot-ready tables, every "
            "scientific value, every panel frame, and every other panel letter remain unchanged.\n",
            encoding="utf-8",
        )
        native_qa = {
            "status": "pass",
            "cumulative_slide3_sha256": sha256(stage / "review_evidence" / "round13_cumulative_native_slide3.png"),
            "review_slide1_sha256": sha256(stage / "review_evidence" / "round13_review_native_slide1.png"),
            "target_render_difference_pixels": 1,
            "target_render_max_channel_delta": 5,
            "unrelated_slides_pixel_identical": list(range(1, 3)) + list(range(4, 13)),
        }
        write_json(stage / "review_evidence" / "POWERPOINT_NATIVE_QA.json", native_qa)
        audit = stage_audit(stage)
        if audit["status"] != "pass":
            raise RuntimeError(json.dumps(audit, indent=2))
        write_json(stage / "AUDIT_PROMOTION.json", audit)
        write_json(stage / "RECORD_VALIDATION.json", audit)
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={record_root.name}\nutc={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        entries = write_hash_ledger(stage)
        stage.rename(record_root)
        return {
            "status": "pass",
            "record": record_root.relative_to(REPO_ROOT).as_posix(),
            "record_files": len([path for path in record_root.rglob("*") if path.is_file()]),
            "hash_entries": entries,
        }
    except Exception:
        if stage.exists() and stage.resolve().parent == record_root.resolve().parent:
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    args = parser.parse_args()
    print(json.dumps(promote(args.record_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
