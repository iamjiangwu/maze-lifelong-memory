"""Promote the validated 60 cm/s ED1h and ED10a figure successors."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import fitz
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_LEDGER = ARCHIVE / "ARCHIVE_MANIFEST.csv"
POINTER = RECORDS / "CURRENT_EXTENDED_DATA_FIGURES.md"

ED1_PARENT_NAME = "extended_data_fig01_2026-08-17_round20_canonical_animal_release"
ED10_PARENT_NAME = "extended_data_fig10_2026-08-12_round11_black_correlations_ed2_policy_order"
ED1_TARGET_NAME = "extended_data_fig01_2026-08-19_round21_speed_cap60"
ED10_TARGET_NAME = "extended_data_fig10_2026-08-19_round12_speed_cap60"
ED1_PARENT = RECORDS / ED1_PARENT_NAME
ED10_PARENT = RECORDS / ED10_PARENT_NAME
ED1_TARGET = RECORDS / ED1_TARGET_NAME
ED10_TARGET = RECORDS / ED10_TARGET_NAME
ED1_CANDIDATE = REPO / "docs/figure_version_candidates/ed01_speed_cap60_2026-08-19"
ED10_CANDIDATE = REPO / "docs/figure_version_candidates/ed10_speed_cap60_2026-08-19"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extended(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def iter_relative_files(root: Path) -> list[tuple[str, Path]]:
    resolved = root.resolve()
    walk_root = extended(root)
    rows: list[tuple[str, Path]] = []
    for directory, _, names in os.walk(walk_root):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            normal_text = str(path)
            if normal_text.startswith("\\\\?\\"):
                normal_text = normal_text[4:]
            relative = Path(normal_text).relative_to(resolved).as_posix()
            rows.append((relative, path))
    return sorted(rows, key=lambda row: row[0].lower())


def copytree_long(source: Path, destination: Path) -> None:
    shutil.copytree(extended(source), extended(destination))


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(extended(path))


def write_hashes(record: Path) -> None:
    ledger = record / "RECORD_HASHES.csv"
    ledger.unlink(missing_ok=True)
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for relative, path in iter_relative_files(record):
            if relative != ledger.name:
                writer.writerow((relative, sha256(path), path.stat().st_size))


def verify_hashes(record: Path) -> list[str]:
    with (record / "RECORD_HASHES.csv").open(newline="", encoding="utf-8-sig") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {relative: path for relative, path in iter_relative_files(record) if relative != "RECORD_HASHES.csv"}
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors += [f"missing:{name}" for name in sorted(set(expected) - set(actual))]
    for name in sorted(set(actual) & set(expected)):
        row = expected[name]
        path = actual[name]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{name}")
    return errors


def render_300dpi(pdf: Path, output: Path) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
    Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).save(
        output, dpi=(300, 300), optimize=True
    )


def copy_script(stage: Path, name: str) -> None:
    shutil.copy2(REPO / "scripts" / name, stage / "source_code" / name)


def build_ed1_stage(stage: Path) -> None:
    copytree_long(ED1_PARENT, stage)
    stem = ED1_CANDIDATE / "ed01_speed_cap60_full_page_round21"
    outputs = {
        ".svg": "ed01_version_record.svg",
        ".pdf": "ed01_version_record.pdf",
        ".png": "ed01_version_record_600dpi.png",
        ".tiff": "ed01_version_record_600dpi.tiff",
    }
    for suffix, name in outputs.items():
        shutil.copy2(stem.with_suffix(suffix), stage / name)
    render_300dpi(stage / "ed01_version_record.pdf", stage / "ed01_version_record_300dpi.png")

    render_work = stage / "render_work"
    for path in ED1_CANDIDATE.iterdir():
        destination = render_work / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, destination)
    for name in (
        "ACCUMULATION_SUMMARY_ROUND21.csv",
        "AXIS_SCALE_AUDIT_ROUND21.csv",
        "LAYOUT_AUDIT_ROUND21.csv",
        "PANEL_A_JUMP_MASK_AUDIT_ROUND21.csv",
        "PANEL_A_TRAJECTORY_PLOT_DATA_ROUND21.csv.gz",
        "PANEL_I_CYCLE_SUMMARY_ROUND21.csv",
        "PANEL_I_END_NODE_SEQUENCE_ROUND21.csv",
        "PANEL_I_SELECTION_AUDIT_ROUND21.csv",
        "SPEED_SUMMARY_ROUND21.csv",
        "TILE_EFFICIENCY_SUMMARY_ROUND21.csv",
        "TIME_EFFICIENCY_SUMMARY_ROUND21.csv",
    ):
        shutil.copy2(ED1_CANDIDATE / name, stage / "drawn_values" / name)
    rows = ED1_CANDIDATE / "source_data/ed01_panel_h_first30_session_rows_cap60.csv"
    summary = ED1_CANDIDATE / "source_data/ed01_panel_h_first30_group_summary_cap60.csv"
    shutil.copy2(rows, stage / "plot_ready/ed01_panel_h_first30_session_rows_cap60.csv")
    shutil.copy2(summary, stage / "plot_ready/ed01_panel_h_first30_group_summary_cap60.csv")
    shutil.copy2(ED1_CANDIDATE / "SOURCE_DATA_AUDIT_ROUND21.csv", stage / "SOURCE_DATA_AUDIT.csv")
    shutil.copy2(ED1_CANDIDATE / "LAYOUT_AUDIT_ROUND21.csv", stage / "LAYOUT_AUDIT.csv")
    shutil.copy2(ED1_CANDIDATE / "RENDER_VALIDATION_ROUND21.json", stage / "ROUND21_RENDER_VALIDATION.json")
    shutil.copy2(ED1_CANDIDATE / "CANDIDATE_VALIDATION.json", stage / "ROUND21_CANDIDATE_VALIDATION.json")
    shutil.copy2(ED1_CANDIDATE / "CAPTION_NOTES_ROUND21.md", stage / "CAPTION_NOTES.md")
    with (stage / "CAPTION_NOTES.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Round 21 speed-cap update\n\n"
            "Panel h uses the same first-30-min session estimator and 153-session registry, "
            "with adjacent-frame displacement segments above 60 cm s^-1 rejected as tracking artifacts. "
            "The 0--14 cm s^-1 display scale and all support counts are unchanged.\n"
        )

    render_validation = json.loads(
        (ED1_CANDIDATE / "RENDER_VALIDATION_ROUND21.json").read_text(encoding="utf-8")
    )
    candidate_validation = json.loads(
        (ED1_CANDIDATE / "CANDIDATE_VALIDATION.json").read_text(encoding="utf-8")
    )
    source_policy = {
        "schema": "ed01-round21-speed-cap60-source-policy/v1",
        "status": "PASS",
        "target_panel": "h",
        "artifact_cap_cm_s": 60.0,
        "window": "first 30 min of cumulative in-maze time",
        "statistical_unit": "recording session",
        "estimator": "retained adjacent-frame distance divided by eligible mode time",
        "position_processing": "five-frame median smoothing within uninterrupted in-maze trajectories",
        "session_rows": "plot_ready/ed01_panel_h_first30_session_rows_cap60.csv",
        "group_summary": "plot_ready/ed01_panel_h_first30_group_summary_cap60.csv",
        "support_and_layout_change": False,
        "unchanged_panels": ["a", "b", "c", "d", "e", "f", "g", "i", "j", "k"],
    }
    write_json(stage / "ROUND21_SOURCE_POLICY.json", source_policy)
    manifest = json.loads((stage / "ED01_ASSEMBLY_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "record": ED1_TARGET_NAME,
            "round": 21,
            "generated": "2026-08-19",
            "supersedes": ED1_PARENT_NAME,
            "outputs": {
                "svg": f"docs/figure_version_records/{ED1_TARGET_NAME}/ed01_version_record.svg",
                "pdf": f"docs/figure_version_records/{ED1_TARGET_NAME}/ed01_version_record.pdf",
                "png_600dpi": f"docs/figure_version_records/{ED1_TARGET_NAME}/ed01_version_record_600dpi.png",
                "tiff_600dpi": f"docs/figure_version_records/{ED1_TARGET_NAME}/ed01_version_record_600dpi.tiff",
            },
            "source_policy": source_policy,
            "validation": {
                **render_validation,
                "speed_cap60_candidate": candidate_validation,
            },
        }
    )
    write_json(stage / "ED01_ASSEMBLY_MANIFEST.json", manifest)
    promotion = {
        "schema": "ed01-round21-speed-cap60-promotion/v1",
        "status": "PASS",
        "record": ED1_TARGET_NAME,
        "supersedes": ED1_PARENT_NAME,
        "target_panel": "h",
        "candidate_validation": candidate_validation,
        "root_png_sha256": sha256(stage / "ed01_version_record_600dpi.png"),
    }
    write_json(stage / "ED01_ROUND21_PROMOTION.json", promotion)
    write_json(
        stage / "RECORD_VALIDATION.json",
        {
            "schema": "ed01-round21-record-validation/v1",
            "status": "PASS",
            "record": ED1_TARGET_NAME,
            "validated_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "renderer_pass": render_validation.get("status") == "pass",
                "cap60_source_parity": candidate_validation.get("status") == "PASS",
                "changes_confined_to_panel_h": candidate_validation.get(
                    "changed_pixels_confined_to_panel_h"
                )
                is True,
                "shared_0_14_speed_axis_preserved": render_validation.get("checks", {}).get(
                    "panel_h_modes_share_identical_y_scale"
                )
                is True,
            },
        },
    )
    (stage / "MANIFEST.md").write_text(
        f"# {ED1_TARGET_NAME}\n\n"
        f"This append-only record supersedes `{ED1_PARENT_NAME}`. Only panel h changes: "
        "its first-30-min body-center speed estimates reject adjacent-frame displacement "
        "segments above 60 cm s^-1. The cohort, estimator, 0--14 cm s^-1 scale, layout and "
        "all other panels are unchanged.\n",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        "Run `python scripts/render_speed_cap60_successor_candidates.py --target ed1`, "
        "inspect `CANDIDATE_VALIDATION.json`, then run the promotion script.\n",
        encoding="utf-8",
    )
    (stage / "PROMOTION_COMPLETE.txt").write_text(
        f"Promoted {ED1_TARGET_NAME} on 2026-08-19; target panel h; artifact cap 60 cm/s.\n",
        encoding="utf-8",
    )
    for name in (
        "render_speed_cap60_successor_candidates.py",
        "promote_speed_cap60_ed01_ed10_2026_08_19.py",
        "render_extended_data_figure1_round4.py",
        "audit_speed_cap_60_vs_80_figures.py",
        "validate_current_figure_records.py",
    ):
        copy_script(stage, name)
    write_hashes(stage)
    errors = verify_hashes(stage)
    if errors:
        raise RuntimeError(f"ED1 stage hash closure failed: {errors[:10]}")


def build_ed10_stage(stage: Path) -> None:
    copytree_long(ED10_PARENT, stage)
    source_stem = ED10_CANDIDATE / "extended_data_figure10_source_rerender_round12"
    for suffix, name in (
        (".svg", "ed10_version_record.svg"),
        (".pdf", "ed10_version_record.pdf"),
        (".png", "ed10_version_record_600dpi.png"),
        (".tiff", "ed10_version_record_600dpi.tiff"),
    ):
        shutil.copy2(source_stem.with_suffix(suffix), stage / name)
        shutil.copy2(source_stem.with_suffix(suffix), stage / f"extended_data_figure10_source_rerender_round12{suffix}")
    render_300dpi(stage / "ed10_version_record.pdf", stage / "ed10_version_record_300dpi.png")
    shutil.copytree(
        ED10_CANDIDATE / "assembly/round12",
        stage / "assembly/round12",
        dirs_exist_ok=True,
    )
    for path in (ED10_CANDIDATE / "assembly/round12/panels").iterdir():
        if path.is_file():
            shutil.copy2(path, stage / "individual_panels" / path.name)

    strength = ED10_CANDIDATE / "source_data_cap60/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX_CAP60.csv"
    pair_rows = ED10_CANDIDATE / "source_data_cap60/CURRENT_FIXED_NEURON_PAIR_ROWS_CAP60.csv"
    raw = ED10_CANDIDATE / "source_data/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv"
    sign = ED10_CANDIDATE / "source_data/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json"
    membership = ED10_CANDIDATE / "source_data/ED10_COHORT_MEMBERSHIP.csv"
    shutil.copy2(strength, stage / "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv")
    shutil.copy2(pair_rows, stage / "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv")
    shutil.copy2(raw, stage / "plot_ready/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv")
    shutil.copy2(sign, stage / "plot_ready/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json")
    shutil.copy2(membership, stage / "plot_ready/ED10_COHORT_MEMBERSHIP.csv")
    shutil.copy2(raw, stage / "source_data/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv")
    shutil.copy2(sign, stage / "source_data/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json")
    shutil.copy2(membership, stage / "source_data/ED10_COHORT_MEMBERSHIP.csv")
    authority = stage / "source_authority/speed_cap60_audit"
    authority.mkdir()
    for path in (
        strength,
        pair_rows,
        REPO / "docs/speed_cap_60_vs_80_figure_panel_audit_2026-08-19/ed10_a_speed_column_comparison.csv",
        REPO / "docs/speed_cap_60_vs_80_figure_panel_audit_2026-08-19/VALIDATION.json",
        REPO / "docs/speed_cap_60_vs_80_figure_panel_audit_2026-08-19/REPORT.md",
    ):
        shutil.copy2(path, authority / path.name)

    shutil.copy2(ED10_CANDIDATE / "AUDIT_VALIDATION.json", stage / "AUDIT_VALIDATION.json")
    shutil.copy2(ED10_CANDIDATE / "CANDIDATE_VALIDATION.json", stage / "ED10_ROUND12_CANDIDATE_VALIDATION.json")
    shutil.copy2(ED10_CANDIDATE / "AUDIT_REPORT.md", stage / "AUDIT_REPORT.md")
    shutil.copy2(ED10_CANDIDATE / "ED10_CAPTION_NOTES.md", stage / "ED10_CAPTION_NOTES.md")
    shutil.copy2(ED10_CANDIDATE / "ED10_CAPTION_NOTES.md", stage / "CAPTION_NOTES.md")
    with (stage / "CAPTION_NOTES.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Round 12 speed-cap update\n\n"
            "The reward-approach speed endpoint rejects adjacent-frame displacement "
            "segments above 60 cm s^-1. Only the first column of panel a changes; "
            "the other 12 correlations and panels b/c are unchanged.\n"
        )
    shutil.copy2(sign, stage / "source_validation/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json")
    shutil.copy2(raw, stage / "source_validation/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv")
    shutil.copy2(ED10_CANDIDATE / "AUDIT_VALIDATION.json", stage / "source_validation/ED10_ROUND12_RENDER_VALIDATION.json")
    shutil.copy2(ED10_CANDIDATE / "CANDIDATE_VALIDATION.json", stage / "source_validation/ED10_ROUND12_SPEED_CAP60_VALIDATION.json")

    candidate_validation = json.loads(
        (ED10_CANDIDATE / "CANDIDATE_VALIDATION.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((stage / "ED10_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema": "ed10-round12-speed-cap60-version-record/v1",
            "record": ED10_TARGET_NAME,
            "record_revision": 12,
            "supersedes": ED10_PARENT_NAME,
            "accepted_utc": datetime.now(timezone.utc).isoformat(),
            "current_display_refinement": {
                "panel_a": "the reward-approach speed column uses a 60 cm/s displacement-artifact cap; all 18 values remain black normal-weight text",
                "panel_c": ["P_SF", "P_SA", "P_BF", "depth-weighted P_BS"],
            },
            "speed_policy": {
                "artifact_cap_cm_s": 60.0,
                "target_panel": "a",
                "changed_column": "whole_session_reward_approach_speed_advantage",
                "unchanged_cells": 12,
                "cellwise_inference": False,
            },
            "root_export_sha256": {
                "svg": sha256(stage / "ed10_version_record.svg"),
                "pdf": sha256(stage / "ed10_version_record.pdf"),
                "png_300dpi": sha256(stage / "ed10_version_record_300dpi.png"),
                "png_600dpi": sha256(stage / "ed10_version_record_600dpi.png"),
                "tiff_600dpi": sha256(stage / "ed10_version_record_600dpi.tiff"),
            },
        }
    )
    write_json(stage / "ED10_VERSION_RECORD_MANIFEST.json", manifest)
    write_json(
        stage / "ED10_ROUND12_PROMOTION.json",
        {
            "schema": "ed10-round12-speed-cap60-promotion/v1",
            "status": "PASS",
            "record": ED10_TARGET_NAME,
            "supersedes": ED10_PARENT_NAME,
            "target_panel": "a",
            "candidate_validation": candidate_validation,
            "root_png_sha256": sha256(stage / "ed10_version_record_600dpi.png"),
        },
    )
    write_json(
        stage / "RECORD_VALIDATION.json",
        {
            "schema": "ed10-round12-record-validation/v1",
            "status": "PASS",
            "record": ED10_TARGET_NAME,
            "validated_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "renderer_pass": json.loads(
                    (ED10_CANDIDATE / "AUDIT_VALIDATION.json").read_text(encoding="utf-8")
                ).get("status")
                == "PASS",
                "cap60_matrix_parity": candidate_validation.get("status") == "PASS",
                "changes_confined_to_panel_a": candidate_validation.get(
                    "changed_pixels_confined_to_panel_a"
                )
                is True,
                "panels_b_c_png_byte_identical": all(
                    row.get("byte_identical")
                    for row in candidate_validation.get("unchanged_individual_panel_pngs", {}).values()
                ),
            },
        },
    )
    (stage / "MANIFEST.md").write_text(
        f"# {ED10_TARGET_NAME}\n\n"
        f"This append-only record supersedes `{ED10_PARENT_NAME}`. Only panel a's reward "
        "approach speed column changes, using the validated 60 cm s^-1 displacement-artifact "
        "cap. The other 12 matrix cells, panels b/c, cohorts, layout and inference are unchanged.\n",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        "Run `python scripts/render_speed_cap60_successor_candidates.py --target ed10`, "
        "inspect `CANDIDATE_VALIDATION.json`, then run the promotion script.\n",
        encoding="utf-8",
    )
    (stage / "PROMOTION_COMPLETE.txt").write_text(
        f"Promoted {ED10_TARGET_NAME} on 2026-08-19; target panel a; artifact cap 60 cm/s.\n",
        encoding="utf-8",
    )
    for name in (
        "render_speed_cap60_successor_candidates.py",
        "promote_speed_cap60_ed01_ed10_2026_08_19.py",
        "render_ed10_slide12_powerpoint_matched.py",
        "audit_speed_cap_60_vs_80_figures.py",
        "validate_current_figure_records.py",
    ):
        copy_script(stage, name)
    write_hashes(stage)
    errors = verify_hashes(stage)
    if errors:
        raise RuntimeError(f"ED10 stage hash closure failed: {errors[:10]}")


def tree_digest(root: Path) -> dict[str, Any]:
    walk_root = Path("\\\\?\\" + str(root.resolve())) if os.name == "nt" else root
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


def main() -> int:
    for parent in (ED1_PARENT, ED10_PARENT):
        if not parent.is_dir():
            raise FileNotFoundError(parent)
    for target in (ED1_TARGET, ED10_TARGET):
        if target.exists() or (ARCHIVE / target.name).exists():
            raise FileExistsError(target)
    for candidate in (ED1_CANDIDATE, ED10_CANDIDATE):
        validation = json.loads((candidate / "CANDIDATE_VALIDATION.json").read_text(encoding="utf-8"))
        if validation.get("status") != "PASS":
            raise RuntimeError(f"Candidate is not PASS: {candidate}")
    pointer_original = POINTER.read_bytes()
    ledger_original = ARCHIVE_LEDGER.read_bytes()
    stage_root = RECORDS / ".s60"
    if stage_root.exists():
        raise FileExistsError(stage_root)
    stage_root.mkdir()
    ed1_stage = stage_root / ED1_TARGET_NAME
    ed10_stage = stage_root / ED10_TARGET_NAME
    installed: list[Path] = []
    archived: list[tuple[Path, Path]] = []
    try:
        build_ed1_stage(ed1_stage)
        build_ed10_stage(ed10_stage)
        os.replace(ed1_stage, ED1_TARGET)
        installed.append(ED1_TARGET)
        os.replace(ed10_stage, ED10_TARGET)
        installed.append(ED10_TARGET)

        with ARCHIVE_LEDGER.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        existing = {row["record"] for row in rows}
        for parent in (ED1_PARENT, ED10_PARENT):
            if parent.name in existing or (ARCHIVE / parent.name).exists():
                raise RuntimeError(f"Archive destination already exists: {parent.name}")
            before = tree_digest(parent)
            destination = ARCHIVE / parent.name
            os.replace(parent, destination)
            archived.append((destination, parent))
            after = tree_digest(destination)
            if before != after:
                raise RuntimeError(f"Archive tree changed during move: {parent.name}")
            rows.append(
                {
                    "record": parent.name,
                    "original_path": f"docs/figure_version_records/{parent.name}",
                    "archive_path": f"docs/figure_version_records_archive_2026-08-09/{parent.name}",
                    "file_count": str(after["files"]),
                    "bytes": str(after["bytes"]),
                    "tree_sha256": after["sha256"],
                    "move_status": "verified",
                }
            )
        rows.sort(key=lambda row: row["record"].lower())
        ledger_tmp = ARCHIVE_LEDGER.with_name(f".{ARCHIVE_LEDGER.name}.speed-cap60.tmp")
        with ledger_tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(ledger_tmp, ARCHIVE_LEDGER)

        pointer_text = pointer_original.decode("utf-8")
        pointer_text = pointer_text.replace(
            f"`{ED1_PARENT_NAME}/` | 2026-08-17 |",
            f"`{ED1_TARGET_NAME}/` | 2026-08-19 |",
        ).replace(
            f"`{ED10_PARENT_NAME}/` | 2026-08-12 |",
            f"`{ED10_TARGET_NAME}/` | 2026-08-19 |",
        )
        if ED1_TARGET_NAME not in pointer_text or ED10_TARGET_NAME not in pointer_text:
            raise RuntimeError("Current-record pointer replacement failed")
        pointer_tmp = POINTER.with_name(f".{POINTER.name}.speed-cap60.tmp")
        pointer_tmp.write_text(pointer_text, encoding="utf-8")
        os.replace(pointer_tmp, POINTER)
    except Exception:
        POINTER.write_bytes(pointer_original)
        ARCHIVE_LEDGER.write_bytes(ledger_original)
        for source, destination in reversed(archived):
            if source.exists() and not destination.exists():
                os.replace(source, destination)
        for target in reversed(installed):
            if target.exists():
                remove_tree(target)
        raise
    finally:
        remove_tree(stage_root)
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": [ED1_TARGET_NAME, ED10_TARGET_NAME],
                "archived": [ED1_PARENT_NAME, ED10_PARENT_NAME],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
