"""Create and archive the approved ED1 Round 18 group-support record."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import fitz
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_LEDGER = ARCHIVE / "ARCHIVE_MANIFEST.csv"
PARENT_NAME = "extended_data_fig01_2026-08-15_round17_whole_group_first30_shared_scale"
TARGET_NAME = "extended_data_fig01_2026-08-15_round18_group80_matched153"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_group_specific_80_support_round18_2026-08-15"
)
PANEL_H_SOURCE = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_panel_h_matched153_window_sensitivity_2026-08-15"
)
CAPTION_AUTHORITY = (
    REPO
    / "docs/manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)
ROOT_STEM = "ed01_version_record"
CANDIDATE_STEM = "ed01_group_specific_80_full_page_round18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    ledger.unlink(missing_ok=True)
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*"), key=lambda item: item.as_posix().lower()):
            if path.is_file() and path.name != ledger.name:
                writer.writerow(
                    (path.relative_to(record).as_posix(), sha256(path), path.stat().st_size)
                )


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


def verify_candidate_manifest() -> list[str]:
    manifest = json.loads((CANDIDATE / "MANIFEST.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("status") != "PASS":
        errors.append("candidate_manifest_status")
    declared = {Path(row["path"]).name: row for row in manifest.get("files", [])}
    actual = {
        path.name: path
        for path in CANDIDATE.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if set(declared) != set(actual):
        errors.append("candidate_manifest_inventory")
    for name in sorted(set(declared) & set(actual)):
        row = declared[name]
        path = actual[name]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"candidate_hash:{name}")
    return errors


def render_300dpi(pdf: Path, png: Path) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pix = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
        png, dpi=(300, 300), optimize=True
    )


def unchanged_panel_audit() -> dict[str, Any]:
    parent = np.asarray(
        Image.open(PARENT / f"{ROOT_STEM}_600dpi.png").convert("RGB"), dtype=np.int16
    )
    candidate = np.asarray(
        Image.open(CANDIDATE / f"{CANDIDATE_STEM}.png").convert("RGB"), dtype=np.int16
    )
    if parent.shape != candidate.shape:
        raise RuntimeError("Parent and candidate raster shapes differ")
    layout = pd.read_csv(CANDIDATE / "LAYOUT_AUDIT_ROUND18.csv").set_index("panel")
    height, width = parent.shape[:2]
    panels: dict[str, Any] = {}
    for panel in ("a", "g", "i"):
        row = layout.loc[panel]
        x0 = max(0, int(math.floor(float(row.x0_fraction) * width)) - 40)
        x1 = min(width, int(math.ceil(float(row.x1_fraction) * width)) + 40)
        y0 = max(0, int(math.floor((1 - float(row.y1_fraction)) * height)) - 40)
        y1 = min(height, int(math.ceil((1 - float(row.y0_fraction)) * height)) + 40)
        delta = np.abs(parent[y0:y1, x0:x1] - candidate[y0:y1, x0:x1])
        panels[panel] = {
            "bbox_px": [x0, y0, x1, y1],
            "max_abs_channel_delta": int(delta.max()),
            "changed_channel_fraction": float((delta > 0).mean()),
            "exact_pixel_parity": bool(np.max(delta) == 0),
        }
    full_delta = np.abs(parent - candidate)
    result = {
        "schema": "ed01-round18-parent-parity/v1",
        "status": "PASS" if all(row["exact_pixel_parity"] for row in panels.values()) else "FAIL",
        "parent": PARENT_NAME,
        "candidate": CANDIDATE.relative_to(REPO).as_posix(),
        "unchanged_panels": panels,
        "full_changed_channel_fraction": float((full_delta > 0).mean()),
        "full_mean_abs_channel_delta": float(full_delta.mean()),
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"Unchanged-panel parity failed: {result}")
    return result


def localize_source_audit(source: Path, destination: Path) -> None:
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    mapping = {
        "b-c": "ed01_bc_group_specific_80_session_curves.csv.gz",
        "d-e": "ed01_de_group_specific_80_group_curves.csv",
        "h-rows": "ed01_panel_h_matched153_speed_by_session_window.csv",
        "h-reference": "ed01_panel_h_matched153_group_summary.csv",
        "j": "ed01_j_group_specific_80_time_session_bins.csv",
        "k": "ed01_k_group_specific_80_tile_session_bins.csv",
    }
    for row in rows:
        name = mapping.get(row["panel"])
        if name is None:
            continue
        local = destination.parent / "plot_ready" / name
        row["source_path"] = (
            f"docs/figure_version_records/{TARGET_NAME}/plot_ready/{name}"
        )
        row["sha256"] = sha256(local)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_round18_sources(stage: Path) -> None:
    plot_ready = stage / "plot_ready"
    drawn = stage / "drawn_values"
    render_work = stage / "render_work"
    validation = stage / "source_validation/round18_group80_matched153"
    for directory in (plot_ready, drawn, render_work, validation):
        directory.mkdir(parents=True, exist_ok=True)

    for path in CANDIDATE.iterdir():
        if path.is_file():
            shutil.copy2(path, render_work / path.name)

    candidate_sources = (
        "ed01_bc_group_specific_80_session_curves.csv.gz",
        "ed01_bc_group_specific_80_group_curves.csv",
        "ed01_de_group_specific_80_session_curves.csv.gz",
        "ed01_de_group_specific_80_group_curves.csv",
        "ed01_j_group_specific_80_time_session_bins.csv",
        "ed01_k_group_specific_80_tile_session_bins.csv",
        "ed01_curve_group_specific_80_support_endpoints.csv",
    )
    for name in candidate_sources:
        shutil.copy2(CANDIDATE / name, plot_ready / name)

    h_names = (
        "cohort_match_audit.csv",
        "failures.csv",
        "group_summary.csv",
        "registry_group_counts.csv",
        "REPORT.md",
        "session_registry.csv",
        "session_source_qc.csv",
        "speed_by_5min_bin.csv",
        "speed_by_session_window.csv",
        "summary_inclusion_audit.csv",
        "VALIDATION.json",
        "MANIFEST.json",
        "window_comparison.csv",
    )
    for name in h_names:
        destination_name = f"ed01_panel_h_matched153_{name}"
        shutil.copy2(PANEL_H_SOURCE / name, plot_ready / destination_name)

    drawn_names = (
        "ACCUMULATION_SUMMARY_ROUND18.csv",
        "AXIS_SCALE_AUDIT_ROUND18.csv",
        "PANEL_A_JUMP_MASK_AUDIT_ROUND18.csv",
        "PANEL_A_TRAJECTORY_PLOT_DATA_ROUND18.csv.gz",
        "PANEL_I_CYCLE_SUMMARY_ROUND18.csv",
        "PANEL_I_END_NODE_SEQUENCE_ROUND18.csv",
        "PANEL_I_SELECTION_AUDIT_ROUND18.csv",
        "SPEED_SUMMARY_ROUND18.csv",
        "TILE_EFFICIENCY_SUMMARY_ROUND18.csv",
        "TIME_EFFICIENCY_SUMMARY_ROUND18.csv",
    )
    for name in drawn_names:
        shutil.copy2(CANDIDATE / name, drawn / name)
    shutil.copy2(
        CANDIDATE / "ed01_curve_group_specific_80_support_endpoints.csv",
        drawn / "CURVE_SUPPORT_ENDPOINTS_ROUND18.csv",
    )

    for name in (
        "MANIFEST.json",
        "QA_REPORT.json",
        "REPORT.md",
        "RENDER_VALIDATION_ROUND18.json",
        "SOURCE_DATA_AUDIT_ROUND18.csv",
        "LAYOUT_AUDIT_ROUND18.csv",
        "CAPTION_NOTES_ROUND18.md",
    ):
        shutil.copy2(CANDIDATE / name, validation / name)
    for name in ("VALIDATION.json", "MANIFEST.json", "REPORT.md"):
        shutil.copy2(PANEL_H_SOURCE / name, validation / f"panel_h_{name}")
    shutil.copy2(CAPTION_AUTHORITY, validation / CAPTION_AUTHORITY.name)


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
        raise RuntimeError(f"Parent hash closure failed: {parent_errors[:10]}")
    candidate_errors = verify_candidate_manifest()
    if candidate_errors:
        raise RuntimeError(f"Candidate closure failed: {candidate_errors[:10]}")
    candidate_qa = json.loads((CANDIDATE / "QA_REPORT.json").read_text(encoding="utf-8"))
    render = json.loads(
        (CANDIDATE / "RENDER_VALIDATION_ROUND18.json").read_text(encoding="utf-8")
    )
    if candidate_qa.get("status") != "PASS" or render.get("status") != "pass":
        raise RuntimeError("Passing candidate and renderer QA are required")
    parity = unchanged_panel_audit()

    stage = Path(tempfile.mkdtemp(prefix=".ed01-round18-", dir=RECORDS))
    try:
        shutil.copytree(PARENT, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        copy_round18_sources(stage)

        for extension in (".svg", ".pdf", ".png", ".tiff"):
            source = CANDIDATE / f"{CANDIDATE_STEM}{extension}"
            suffix = extension if extension in {".svg", ".pdf"} else f"_600dpi{extension}"
            shutil.copy2(source, stage / f"{ROOT_STEM}{suffix}")
        render_300dpi(
            stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png"
        )

        localize_source_audit(
            CANDIDATE / "SOURCE_DATA_AUDIT_ROUND18.csv",
            stage / "SOURCE_DATA_AUDIT.csv",
        )
        shutil.copy2(CANDIDATE / "LAYOUT_AUDIT_ROUND18.csv", stage / "LAYOUT_AUDIT.csv")
        shutil.copy2(
            CANDIDATE / "RENDER_VALIDATION_ROUND18.json",
            stage / "ROUND18_RENDER_VALIDATION.json",
        )
        shutil.copy2(CANDIDATE / "QA_REPORT.json", stage / "ED01_ROUND18_QA.json")
        shutil.copy2(CANDIDATE / "CAPTION_NOTES_ROUND18.md", stage / "CAPTION_NOTES.md")
        write_json(stage / "ROUND18_PARENT_PARITY_AUDIT.json", parity)

        assembly = json.loads(
            (CANDIDATE / "MANIFEST_ROUND18.json").read_text(encoding="utf-8")
        )
        assembly["record"] = TARGET_NAME
        assembly["supersedes"] = PARENT_NAME
        assembly["outputs"] = {
            "svg": f"docs/figure_version_records/{TARGET_NAME}/{ROOT_STEM}.svg",
            "pdf": f"docs/figure_version_records/{TARGET_NAME}/{ROOT_STEM}.pdf",
            "png_600dpi": f"docs/figure_version_records/{TARGET_NAME}/{ROOT_STEM}_600dpi.png",
            "tiff_600dpi": f"docs/figure_version_records/{TARGET_NAME}/{ROOT_STEM}_600dpi.tiff",
        }
        assembly["source_policy"].update(
            {
                "source_audit": f"docs/figure_version_records/{TARGET_NAME}/SOURCE_DATA_AUDIT.csv",
                "panels_b_to_e_support": "group-specific 80% recording support; all 153 sessions retained; cumulative terminal counts carried forward",
                "panels_j_k_support": "group-specific 80% recording opportunity among 152 cycle-eligible sessions; no terminal efficiency carry-forward; minimum five metric contributors per displayed bin",
                "panel_h_policy": "matched 153-session registry; first 30 min of cumulative in-maze time; one shared 0-14 cm/s scale",
                "panel_f_policy": "endpoint-specific y scales because the three columns are distinct endpoint measures",
            }
        )
        write_json(stage / "ED01_ASSEMBLY_MANIFEST.json", assembly)

        support = pd.read_csv(
            stage / "plot_ready/ed01_curve_group_specific_80_support_endpoints.csv"
        )
        policy = {
            "schema": "ed01-round18-source-policy/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "cumulative_session_count": 153,
            "efficiency_cycle_eligible_session_count": 152,
            "efficiency_exclusion": {
                "session": "646-9day2",
                "reason": "no complete 32-endpoint coverage cycle; efficiency undefined",
            },
            "support_rows": support.astype(object).where(pd.notna(support), None).to_dict(
                orient="records"
            ),
            "terminal_carry": {
                "panels_b_to_e": True,
                "panels_j_k": False,
            },
        }
        write_json(stage / "ROUND18_SOURCE_POLICY.json", policy)

        source_code = stage / "source_code"
        source_code.mkdir(exist_ok=True)
        for relative in (
            "scripts/render_extended_data_figure1_round4.py",
            "scripts/analyze_ed01_group_specific_80_support.py",
            "scripts/analyze_ed01_panel_h_matched_behavior_cohort.py",
            "scripts/audit_ed01_panel_d_whole_group.py",
            "scripts/promote_ed01_group80_round18.py",
            "scripts/sync_ed01_round18_to_desktop.py",
            "scripts/validate_current_figure_records.py",
            "scripts/validate_all_figure_refinement_round2_records.py",
            "scripts/validate_single_current_figure_record_layout.py",
            "scripts/validate_desktop_version_record_layout.py",
        ):
            source = REPO / relative
            if source.is_file():
                shutil.copy2(source, source_code / source.name)

        timestamp = datetime.now(timezone.utc).isoformat()
        promotion = {
            "schema": "ed01-round18-group80-matched153-promotion/v1",
            "status": "PASS",
            "record": TARGET_NAME,
            "parent_record": PARENT_NAME,
            "scientific_changes": [
                "panels b-e extend to each group's maximal 80% recording-support endpoint with all 153 sessions retained in cumulative means",
                "panels j-k stop at each pooled group's 80% recording-opportunity endpoint without carrying undefined efficiency values",
                "panel h uses the matched 153-session first-30-min estimator on a shared 0-14 cm/s scale",
            ],
            "display_changes": [
                "panel f restores endpoint-specific y scales for its three distinct measures",
                "panels b-e and j-k now have unequal group curve lengths reflecting group-specific support",
            ],
            "unchanged_panel_pixel_parity": parity["unchanged_panels"],
            "root_exports": {
                suffix: sha256(stage / f"{ROOT_STEM}{suffix}")
                for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff")
            },
            "caption_authority": {
                "path": CAPTION_AUTHORITY.relative_to(REPO).as_posix(),
                "sha256": sha256(CAPTION_AUTHORITY),
                "ed1_updated": True,
            },
            "captions_changed": True,
            "created_utc": timestamp,
        }
        write_json(stage / "ED01_ROUND18_PROMOTION.json", promotion)
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This append-only record supersedes `{PARENT_NAME}`. Panels b--e and j/k "
            "use group-specific 80% recording-support endpoints. Cumulative panels b--e "
            "retain all 153 sessions with terminal-count carry-forward; j/k use 152 "
            "cycle-eligible sessions and never carry efficiency after session end. Panel f "
            "uses endpoint-specific y scales. Panel h uses the matched 153-session first-30-min "
            "source on one shared 0--14 cm/s scale. Panels a, g and i are pixel-identical to "
            "the parent. The ED1 caption was updated to the Round 18 support, scale and "
            "matched-cohort definitions.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python scripts/analyze_ed01_group_specific_80_support.py`, review the "
            "candidate, then run `python scripts/promote_ed01_group80_round18.py --create`.\n",
            encoding="utf-8",
        )

        checks = {
            "parent_hash_closure": not parent_errors,
            "candidate_manifest_closure": not candidate_errors,
            "candidate_qa_pass": candidate_qa.get("status") == "PASS",
            "renderer_validation_pass": render.get("status") == "pass",
            "panel_f_independent_scales": render.get("checks", {}).get(
                "panel_f_independent_endpoint_scales_restored"
            )
            is True,
            "panel_h_shared_scale": render.get("checks", {}).get(
                "panel_h_modes_share_identical_y_scale"
            )
            is True,
            "unchanged_panels_a_g_i_exact": parity.get("status") == "PASS",
            "all_support_endpoints_at_least_80pct": bool(
                support["support_fraction_at_endpoint"].ge(0.8 - 1e-12).all()
            ),
            "cumulative_axes_total_153": all(
                int(rows["n_sessions"].sum()) == 153
                for axis, rows in support.groupby("axis", observed=True)
                if axis in {"time_in_maze", "maze_tile_transitions"}
            ),
            "efficiency_axes_total_152": all(
                int(rows["n_sessions"].sum()) == 152
                for axis, rows in support.groupby("axis", observed=True)
                if axis.startswith("exploration_efficiency_")
            ),
            "root_svg_matches_candidate": sha256(stage / f"{ROOT_STEM}.svg")
            == sha256(CANDIDATE / f"{CANDIDATE_STEM}.svg"),
            "root_pdf_matches_candidate": sha256(stage / f"{ROOT_STEM}.pdf")
            == sha256(CANDIDATE / f"{CANDIDATE_STEM}.pdf"),
            "root_600dpi_png_matches_candidate": sha256(
                stage / f"{ROOT_STEM}_600dpi.png"
            )
            == sha256(CANDIDATE / f"{CANDIDATE_STEM}.png"),
            "root_600dpi_tiff_matches_candidate": sha256(
                stage / f"{ROOT_STEM}_600dpi.tiff"
            )
            == sha256(CANDIDATE / f"{CANDIDATE_STEM}.tiff"),
            "caption_authority_frozen": sha256(CAPTION_AUTHORITY)
            == sha256(
                stage
                / "source_validation/round18_group80_matched153"
                / CAPTION_AUTHORITY.name
            ),
        }
        checks = {key: bool(value) for key, value in checks.items()}
        if not all(checks.values()):
            raise RuntimeError(
                f"Record validation failed: {[key for key, value in checks.items() if not value]}"
            )
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed01-round18-record-validation/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "checks": checks,
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


def archive_parent() -> dict[str, Any]:
    source = PARENT
    destination = ARCHIVE / PARENT_NAME
    if not TARGET.is_dir() or verify_hashes(TARGET):
        raise RuntimeError("A closed Round 18 successor is required before archiving")
    if not source.is_dir() or destination.exists():
        raise RuntimeError(
            f"Archive precondition failed: source={source.exists()} destination={destination.exists()}"
        )
    before = tree_digest(source)
    with ARCHIVE_LEDGER.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if PARENT_NAME in {row["record"] for row in rows}:
        raise RuntimeError("Parent already declared in archive ledger")
    original_ledger = ARCHIVE_LEDGER.read_bytes()
    temporary = ARCHIVE_LEDGER.with_name(
        f".{ARCHIVE_LEDGER.name}.ed01-round18.tmp"
    )
    moved = False
    try:
        os.replace(source, destination)
        moved = True
        after = tree_digest(destination)
        if before != after:
            raise RuntimeError("Archived parent tree changed during move")
        rows.append(
            {
                "record": PARENT_NAME,
                "original_path": f"docs/figure_version_records/{PARENT_NAME}",
                "archive_path": f"docs/figure_version_records_archive_2026-08-09/{PARENT_NAME}",
                "file_count": str(after["files"]),
                "bytes": str(after["bytes"]),
                "tree_sha256": after["sha256"],
                "move_status": "verified",
            }
        )
        rows.sort(key=lambda row: row["record"].lower())
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, ARCHIVE_LEDGER)
        return {"status": "PASS", "archived": PARENT_NAME, "tree": after}
    except Exception:
        temporary.unlink(missing_ok=True)
        ARCHIVE_LEDGER.write_bytes(original_ledger)
        if moved and destination.exists() and not source.exists():
            os.replace(destination, source)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result: object
    if args.create:
        result = create_record().relative_to(REPO).as_posix()
    else:
        result = archive_parent()
    print(json.dumps({"status": "PASS", "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
