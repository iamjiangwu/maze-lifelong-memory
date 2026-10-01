"""Promote and archive the ED1 round-17 whole-group/first-30-min record."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import fitz
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_LEDGER = ARCHIVE / "ARCHIVE_MANIFEST.csv"
STYLE_PARENT_NAME = "extended_data_fig01_2026-08-12_round15_arial_star_readability"
PARENT_NAME = "extended_data_fig01_2026-08-15_round16_whole_group"
TARGET_NAME = "extended_data_fig01_2026-08-15_round17_whole_group_first30_shared_scale"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = REPO / "docs/ed01_whole_group_promotion_2026-08-15/round17_candidate"
CANDIDATE_STEM = "extended_data_figure1_source_rerender_round17"
ROOT_STEM = "ed01_version_record"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    ledger.unlink(missing_ok=True)
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*"), key=lambda item: item.as_posix().lower()):
            if path.is_file() and path.name != ledger.name:
                writer.writerow((path.relative_to(record).as_posix(), sha256(path), path.stat().st_size))


def verify_hashes(record: Path) -> list[str]:
    ledger = record / "RECORD_HASHES.csv"
    if not ledger.is_file():
        return ["missing:RECORD_HASHES.csv"]
    with ledger.open(newline="", encoding="utf-8-sig") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != ledger.name
    }
    errors = [f"unhashed:{key}" for key in sorted(set(actual) - set(expected))]
    errors.extend(f"missing:{key}" for key in sorted(set(expected) - set(actual)))
    for key in sorted(set(actual) & set(expected)):
        path = actual[key]
        row = expected[key]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{key}")
    return errors


def render_300dpi(pdf: Path, png: Path) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pix = document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    image.save(png, dpi=(300, 300), optimize=True)


def replace_text(payload: Any, old: str, new: str) -> Any:
    if isinstance(payload, dict):
        return {key: replace_text(value, old, new) for key, value in payload.items()}
    if isinstance(payload, list):
        return [replace_text(value, old, new) for value in payload]
    if isinstance(payload, str):
        return payload.replace(old, new)
    return payload


def localize_source_audit(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    record_table = (
        f"docs/figure_version_records/{TARGET_NAME}/plot_ready/"
        "ed01_tile_accumulation_whole_group_common_support_current153.csv"
    )
    record_speed_rows = (
        f"docs/figure_version_records/{TARGET_NAME}/plot_ready/"
        "ed01_panel_h_first30_session_rows.csv"
    )
    record_speed_reference = (
        f"docs/figure_version_records/{TARGET_NAME}/plot_ready/"
        "ed01_panel_h_first30_group_summary.csv"
    )
    for row in rows:
        if row["panel"] == "d-e":
            row["source_path"] = record_table
            row["source_stage"] = (
                "whole-group current153 maze-transition curves on fully observed common support"
            )
            row["sha256"] = sha256(
                destination.parent / "plot_ready/ed01_tile_accumulation_whole_group_common_support_current153.csv"
            )
        elif row["panel"] == "h-rows":
            row["source_path"] = record_speed_rows
            row["source_stage"] = "first-30-min session-level body-mass speed rows"
            row["sha256"] = sha256(
                destination.parent / "plot_ready/ed01_panel_h_first30_session_rows.csv"
            )
        elif row["panel"] == "h-reference":
            row["source_path"] = record_speed_reference
            row["source_stage"] = "first-30-min grouped-summary parity target"
            row["sha256"] = sha256(
                destination.parent / "plot_ready/ed01_panel_h_first30_group_summary.csv"
            )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_record() -> Path:
    if TARGET.exists():
        errors = verify_hashes(TARGET)
        if errors:
            raise RuntimeError(f"Existing target is not hash closed: {errors[:10]}")
        return TARGET
    if not PARENT.is_dir():
        raise FileNotFoundError(PARENT)
    parent_errors = verify_hashes(PARENT)
    if parent_errors:
        raise RuntimeError(f"Parent record hash closure failed: {parent_errors[:10]}")
    candidate_validation = json.loads(
        (CANDIDATE / "ROUND17_CANDIDATE_VALIDATION.json").read_text(encoding="utf-8")
    )
    if candidate_validation.get("status") != "PASS":
        raise RuntimeError("A passing round-17 candidate validation is required")
    shared_axis_audit = json.loads(
        (
            REPO
            / "docs/ed01_whole_group_promotion_2026-08-15/axis_scale_audit_round17/AUDIT.json"
        ).read_text(encoding="utf-8")
    )
    if shared_axis_audit.get("status") != "PASS":
        raise RuntimeError("A passing all-panel shared-axis audit is required")

    stage = Path(tempfile.mkdtemp(prefix=".ed01-round17-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)

        render_work = stage / "render_work"
        render_work.mkdir(exist_ok=True)
        for path in CANDIDATE.iterdir():
            if path.is_file() and path.name != "ROUND17_RASTER_DIFFERENCE_AUDIT.png":
                shutil.copy2(path, render_work / path.name)

        for extension in (".svg", ".pdf", ".png", ".tiff"):
            source = CANDIDATE / f"{CANDIDATE_STEM}{extension}"
            destination = stage / (
                f"{ROOT_STEM}{extension}"
                if extension in {".svg", ".pdf"}
                else f"{ROOT_STEM}_600dpi{extension}"
            )
            shutil.copy2(source, destination)
        render_300dpi(stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png")

        plot_ready = stage / "plot_ready"
        plot_ready.mkdir(exist_ok=True)
        source_data = CANDIDATE / "source_data"
        whole_group_name = "ed01_tile_accumulation_whole_group_common_support_current153.csv"
        shutil.copy2(source_data / whole_group_name, plot_ready / whole_group_name)
        shutil.copy2(
            source_data / whole_group_name,
            plot_ready / "ed01_tile_accumulation_by_return_month_group_curves_current153.csv",
        )
        for name in (
            "ed01_panel_h_first30_session_rows.csv",
            "ed01_panel_h_first30_group_summary.csv",
        ):
            shutil.copy2(source_data / name, plot_ready / name)
        panel_h_audit = source_data / "panel_h_first30_audit"
        for name in (
            "session_registry.csv",
            "session_source_qc.csv",
            "accepted_first60_parity.csv",
            "current_raw_first60_drift.csv",
            "VALIDATION.json",
            "MANIFEST.json",
            "REPORT.md",
        ):
            shutil.copy2(panel_h_audit / name, plot_ready / f"ed01_panel_h_{name}")
        for name in (
            "source_session_inclusion_audit.csv",
            "dropped_sessions.csv",
            "group_support_summary.csv",
            "selection_diagnostic.csv",
            "estimator_endpoint_comparison.csv",
            "monotonicity_audit.csv",
        ):
            shutil.copy2(source_data / name, plot_ready / f"ed01_round17_{name}")

        drawn_values = stage / "drawn_values"
        for name in (
            "ACCUMULATION_SUMMARY_ROUND17.csv",
            "SPEED_SUMMARY_ROUND17.csv",
            "TIME_EFFICIENCY_SUMMARY_ROUND17.csv",
            "TILE_EFFICIENCY_SUMMARY_ROUND17.csv",
            "PANEL_A_JUMP_MASK_AUDIT_ROUND17.csv",
            "PANEL_A_TRAJECTORY_PLOT_DATA_ROUND17.csv.gz",
            "PANEL_I_CYCLE_SUMMARY_ROUND17.csv",
            "PANEL_I_END_NODE_SEQUENCE_ROUND17.csv",
            "PANEL_I_SELECTION_AUDIT_ROUND17.csv",
        ):
            shutil.copy2(CANDIDATE / name, drawn_values / name)
        shutil.copy2(source_data / whole_group_name, drawn_values / "TILE_ACCUMULATION_WHOLE_GROUP_ROUND17.csv")
        shutil.copy2(
            CANDIDATE / "AXIS_SCALE_AUDIT_ROUND17.csv",
            drawn_values / "AXIS_SCALE_AUDIT_ROUND17.csv",
        )

        localize_source_audit(CANDIDATE / "SOURCE_DATA_AUDIT_ROUND17.csv", stage / "SOURCE_DATA_AUDIT.csv")
        shutil.copy2(stage / "SOURCE_DATA_AUDIT.csv", render_work / "SOURCE_DATA_AUDIT_ROUND17.csv")
        shutil.copy2(CANDIDATE / "LAYOUT_AUDIT_ROUND17.csv", stage / "LAYOUT_AUDIT.csv")
        shutil.copy2(CANDIDATE / "RENDER_VALIDATION_ROUND17.json", stage / "ROUND17_RENDER_VALIDATION.json")
        shutil.copy2(CANDIDATE / "ROUND17_CANDIDATE_VALIDATION.json", stage / "ED01_ROUND17_QA.json")
        shutil.copy2(CANDIDATE / "CAPTION_NOTES_ROUND17.md", stage / "CAPTION_NOTES.md")

        manifest = json.loads((CANDIDATE / "MANIFEST_ROUND17.json").read_text(encoding="utf-8"))
        candidate_rel = CANDIDATE.relative_to(REPO).as_posix()
        record_render_rel = f"docs/figure_version_records/{TARGET_NAME}/render_work"
        manifest = replace_text(manifest, candidate_rel, record_render_rel)
        manifest["record"] = TARGET_NAME
        manifest["supersedes"] = PARENT_NAME
        manifest["source_policy"]["transition_curve_policy"] = (
            "all 153 sessions on fully observed common support, 0-5,200 corrected nose tile transitions"
        )
        manifest["source_policy"]["panel_h_policy"] = (
            "first 30 min of cumulative in-maze time; session-level exact-current-rule "
            "body-mass speed; one shared 0-14 cm/s y scale"
        )
        write_json(stage / "ED01_ASSEMBLY_MANIFEST.json", manifest)
        write_json(render_work / "MANIFEST_ROUND17.json", manifest)

        policy = json.loads((CANDIDATE / "ROUND17_SOURCE_POLICY.json").read_text(encoding="utf-8"))
        policy = replace_text(
            policy,
            f"{candidate_rel}/source_data/{whole_group_name}",
            f"docs/figure_version_records/{TARGET_NAME}/plot_ready/{whole_group_name}",
        )
        policy = replace_text(
            policy,
            f"{candidate_rel}/source_data/ed01_panel_h_first30_session_rows.csv",
            f"docs/figure_version_records/{TARGET_NAME}/plot_ready/ed01_panel_h_first30_session_rows.csv",
        )
        policy = replace_text(
            policy,
            f"{candidate_rel}/source_data/ed01_panel_h_first30_group_summary.csv",
            f"docs/figure_version_records/{TARGET_NAME}/plot_ready/ed01_panel_h_first30_group_summary.csv",
        )
        write_json(stage / "ROUND17_SOURCE_POLICY.json", policy)

        evidence = stage / "source_validation/whole_group_first30_round17"
        evidence.mkdir(parents=True, exist_ok=True)
        for name in (
            "ROUND17_CANDIDATE_VALIDATION.json",
            "ROUND17_RASTER_DIFFERENCE_AUDIT.png",
            "ROUND17_SOURCE_POLICY.json",
        ):
            shutil.copy2(CANDIDATE / name, evidence / name)
        for name in ("REPORT.md", "QA_REPORT.json"):
            shutil.copy2(source_data / name, evidence / name)
        axis_evidence = stage / "source_validation/shared_axis_audit_round17"
        shutil.copytree(
            REPO / "docs/ed01_whole_group_promotion_2026-08-15/axis_scale_audit_round17",
            axis_evidence,
            dirs_exist_ok=True,
        )

        source_code = stage / "source_code"
        for relative in (
            "scripts/render_extended_data_figure1_round4.py",
            "scripts/render_extended_data_figure1_whole_group_round17.py",
            "scripts/audit_ed01_panel_d_whole_group.py",
            "scripts/analyze_ed01_panel_h_window_sensitivity.py",
            "scripts/audit_ed01_shared_axis_scales.py",
            "scripts/validate_ed01_whole_group_round17.py",
            "scripts/promote_ed01_whole_group_round17.py",
            "scripts/sync_ed01_round17_to_desktop.py",
            "scripts/validate_single_current_figure_record_layout.py",
            "scripts/validate_desktop_version_record_layout.py",
        ):
            source = REPO / relative
            if source.is_file():
                shutil.copy2(source, source_code / source.name)

        timestamp = datetime.now(timezone.utc).isoformat()
        promotion = {
            "schema": "ed01-round17-whole-group-first30-promotion/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "scientific_changes": [
                "panels d/e use all 153 sessions on fully observed 0-5,200-transition support",
                "panel f uses one common y scale within each measurement row without changing data or tests",
                "panel h uses first-30-min session-level body-mass speed from the accepted 120-session cohort",
            ],
            "display_changes": [
                "panel h y-axis uses Body mass speed",
                "panel h exploration and reward-directed plots share a 0-14 cm/s y scale",
                "panel i titles use Within round coverage and Coverage round efficiency",
            ],
            "root_exports": {
                suffix: sha256(stage / f"{ROOT_STEM}{suffix}")
                for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff")
            },
            "created_utc": timestamp,
        }
        write_json(stage / "ED01_ROUND17_PROMOTION.json", promotion)
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. Panels d and e now use "
            "all 153 sessions over the fully observed common 0--5,200 corrected nose tile "
            "transition range. Panel h uses the first 30 min and one shared 0--14 cm/s "
            "scale with the label `Body mass speed`; panel f uses one shared y scale within "
            "each measurement row; panel i uses `Within round coverage` "
            "and `Coverage round efficiency`. The accepted renderer, "
            "canvas, palette, typography and panel geometry are otherwise retained.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python scripts/audit_ed01_panel_d_whole_group.py`, then "
            "`python scripts/render_extended_data_figure1_whole_group_round17.py`, then "
            "`python scripts/audit_ed01_shared_axis_scales.py`, then "
            "`python scripts/validate_ed01_whole_group_round17.py`.\n",
            encoding="utf-8",
        )
        record_checks = {
            "candidate_validation_pass": True,
            "all_panel_shared_axis_audit_pass": True,
            "parent_hash_closure_before_copy": True,
            "root_svg_matches_candidate": sha256(stage / f"{ROOT_STEM}.svg") == sha256(CANDIDATE / f"{CANDIDATE_STEM}.svg"),
            "root_pdf_matches_candidate": sha256(stage / f"{ROOT_STEM}.pdf") == sha256(CANDIDATE / f"{CANDIDATE_STEM}.pdf"),
            "root_600dpi_png_matches_candidate": sha256(stage / f"{ROOT_STEM}_600dpi.png") == sha256(CANDIDATE / f"{CANDIDATE_STEM}.png"),
            "root_600dpi_tiff_matches_candidate": sha256(stage / f"{ROOT_STEM}_600dpi.tiff") == sha256(CANDIDATE / f"{CANDIDATE_STEM}.tiff"),
            "whole_group_plot_ready_present": (plot_ready / whole_group_name).is_file(),
            "panel_h_first30_rows_present": (plot_ready / "ed01_panel_h_first30_session_rows.csv").is_file(),
            "panel_h_first30_reference_present": (plot_ready / "ed01_panel_h_first30_group_summary.csv").is_file(),
        }
        if not all(record_checks.values()):
            raise RuntimeError(f"Record promotion checks failed: {record_checks}")
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed01-round17-record-validation/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "checks": record_checks,
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={TARGET_NAME}\ncreated_utc={timestamp}\n",
            encoding="utf-8",
        )
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"New record hash closure failed: {errors[:10]}")
        os.replace(stage, TARGET)
        return TARGET
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def tree_digest(root: Path) -> dict[str, Any]:
    walk_root = root
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        walk_root = Path("\\\\?\\" + str(root.resolve()))
    files = sorted(
        (path for path in walk_root.rglob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(walk_root).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode("utf-8"))
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def archive_parents() -> dict[str, Any]:
    names = (STYLE_PARENT_NAME, PARENT_NAME)
    sources = {name: RECORDS / name for name in names}
    destinations = {name: ARCHIVE / name for name in names}
    if not TARGET.is_dir() or verify_hashes(TARGET):
        raise RuntimeError("A closed round-17 successor is required before archiving")
    for name in names:
        if not sources[name].is_dir() or destinations[name].exists():
            raise RuntimeError(
                f"Archive precondition failed for {name}: "
                f"source={sources[name].exists()} destination={destinations[name].exists()}"
            )
    before = {name: tree_digest(sources[name]) for name in names}
    with ARCHIVE_LEDGER.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    declared = {row["record"] for row in rows}
    overlap = declared.intersection(names)
    if overlap:
        raise RuntimeError(f"Parents already declared in archive ledger: {sorted(overlap)}")
    original_ledger = ARCHIVE_LEDGER.read_bytes()
    temp_ledger = ARCHIVE_LEDGER.with_name(f".{ARCHIVE_LEDGER.name}.ed01-round17.tmp")
    moved: list[str] = []
    try:
        after: dict[str, dict[str, Any]] = {}
        for name in names:
            os.replace(sources[name], destinations[name])
            moved.append(name)
            after[name] = tree_digest(destinations[name])
            if before[name] != after[name]:
                raise RuntimeError(f"Archived parent tree changed during move: {name}")
            rows.append(
                {
                    "record": name,
                    "original_path": f"docs/figure_version_records/{name}",
                    "archive_path": f"docs/figure_version_records_archive_2026-08-09/{name}",
                    "file_count": str(after[name]["files"]),
                    "bytes": str(after[name]["bytes"]),
                    "tree_sha256": after[name]["sha256"],
                    "move_status": "verified",
                }
            )
        rows.sort(key=lambda row: row["record"].lower())
        with temp_ledger.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_ledger, ARCHIVE_LEDGER)
        return {"status": "PASS", "archived": list(names), "trees": after}
    except Exception:
        temp_ledger.unlink(missing_ok=True)
        ARCHIVE_LEDGER.write_bytes(original_ledger)
        for name in reversed(moved):
            if destinations[name].exists() and not sources[name].exists():
                os.replace(destinations[name], sources[name])
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result: object = create_record().relative_to(REPO).as_posix() if args.create else archive_parents()
    print(json.dumps({"status": "PASS", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
