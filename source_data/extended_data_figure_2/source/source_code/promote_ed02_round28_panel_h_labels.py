"""Promote the ED2 Round 29 pooled-return mouse-1616 restoration."""

from __future__ import annotations

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
RECORDS = REPO / "docs" / "figure_version_records"
ARCHIVE = REPO / "docs" / "figure_version_records_archive_2026-08-09"
ARCHIVE_LEDGER = ARCHIVE / "ARCHIVE_MANIFEST.csv"
PARENT_NAME = "extended_data_fig02_2026-08-20_round28_panel_h_labels"
TARGET_NAME = "extended_data_fig02_2026-08-20_round29_1616_restored"
PARENT = RECORDS / PARENT_NAME
TARGET = RECORDS / TARGET_NAME
CANDIDATE = (
    REPO
    / "docs"
    / "figure_version_candidates"
    / "ed02_round29_1616_restored_candidate_2026-08-20"
)
CAPTION_AUTHORITY = (
    REPO
    / "docs"
    / "manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)
ROOT_STEM = "ed02_version_record"
CANDIDATE_STEM = "ed02_round29_1616_restored_candidate"
PROVENANCE_NAME = "round29_1616_restored"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
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


def render_300dpi(pdf: Path, png: Path) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False
        )
    Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).save(
        png, dpi=(300, 300), optimize=True
    )


def tree_digest(root: Path) -> dict[str, Any]:
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix().lower(),
    )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode("utf-8"))
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def write_source_code_manifest(stage: Path) -> None:
    with (stage / "SOURCE_CODE_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(
            (stage / "source_code").glob("*"), key=lambda item: item.name.lower()
        ):
            if path.is_file():
                writer.writerow(
                    (path.relative_to(stage).as_posix(), sha256(path), path.stat().st_size)
                )


def create_record() -> Path:
    if TARGET.exists():
        errors = verify_hashes(TARGET)
        if errors:
            raise RuntimeError(f"Existing target is not hash closed: {errors[:10]}")
        return TARGET
    parent_errors = verify_hashes(PARENT)
    if parent_errors:
        raise RuntimeError(f"Parent hash closure failed: {parent_errors[:10]}")
    qa = json.loads((CANDIDATE / "QA.json").read_text(encoding="utf-8"))
    candidate_checks = {
        "candidate_qa_pass": qa.get("status") == "pass"
        and all(bool(value) for value in qa.get("checks", {}).values()),
        "candidate_vector_only": qa["composite"]["vector_audit"][
            "embedded_raster_images"
        ]
        == 0,
        "candidate_600dpi_square": qa["composite"]["png_audit"]["size"]
        == [4500, 4500],
    }
    if not all(candidate_checks.values()):
        raise RuntimeError(f"Candidate checks failed: {candidate_checks}")

    timestamp = datetime.now(timezone.utc).isoformat()
    stage = Path(tempfile.mkdtemp(prefix=".ed02-round29-", dir=RECORDS))
    try:
        shutil.copytree(CANDIDATE, stage, dirs_exist_ok=True)
        for stale in (
            "RECORD_HASHES.csv",
            "RECORD_VALIDATION.json",
            "PROMOTION_COMPLETE.txt",
            "ED02_VERSION_RECORD_MANIFEST.json",
        ):
            (stage / stale).unlink(missing_ok=True)

        composite = stage / "composite"
        shutil.copy2(composite / f"{CANDIDATE_STEM}.pdf", stage / f"{ROOT_STEM}.pdf")
        shutil.copy2(composite / f"{CANDIDATE_STEM}.svg", stage / f"{ROOT_STEM}.svg")
        shutil.copy2(
            composite / f"{CANDIDATE_STEM}_600dpi.png",
            stage / f"{ROOT_STEM}_600dpi.png",
        )
        shutil.copy2(
            composite / f"{CANDIDATE_STEM}_600dpi.tiff",
            stage / f"{ROOT_STEM}_600dpi.tiff",
        )
        shutil.copy2(
            composite / f"{CANDIDATE_STEM}_preview_1440.png",
            stage / f"{ROOT_STEM}_preview_1440.png",
        )
        render_300dpi(stage / f"{ROOT_STEM}.pdf", stage / f"{ROOT_STEM}_300dpi.png")
        shutil.copy2(
            composite / "COMPOSITE_MANIFEST.json",
            stage / "ED02_ROUND29_COMPOSITE_MANIFEST.json",
        )

        validation = stage / "source_validation" / PROVENANCE_NAME
        validation.mkdir(parents=True, exist_ok=False)
        for name in (
            "CAPTION_NOTES.md",
            "DATA_BUILD_RESULT.json",
            "PANEL_MANIFEST.csv",
            "PPT_VECTOR_MANIFEST.csv",
            "QA.json",
            "QA.md",
            "REPORT.md",
        ):
            shutil.copy2(stage / name, validation / name)
        shutil.copy2(CAPTION_AUTHORITY, validation / CAPTION_AUTHORITY.name)
        shutil.copy2(
            composite / f"{CANDIDATE_STEM}_preview_1440.png",
            validation / f"{CANDIDATE_STEM}_preview_1440.png",
        )

        for source in (
            REPO / "scripts" / "build_ed02_round28_panel_h_labels_candidate.py",
            Path(__file__),
        ):
            shutil.copy2(source, stage / "source_code" / source.name)
        write_source_code_manifest(stage)

        source_manifest = json.loads(
            (stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        source_manifest.update(
            {
                "status": "accepted_version_record",
                "source_record": f"docs/figure_version_records/{TARGET_NAME}",
                "review_round": 29,
                "supersedes": PARENT_NAME,
            }
        )
        write_json(stage / "SOURCE_MANIFEST.json", source_manifest)

        version_manifest = json.loads(
            (PARENT / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        version_manifest.update(
            {
                "accepted_date": "2026-08-20",
                "record": TARGET_NAME,
                "review_round": 29,
                "source_record": PARENT_NAME,
                "supersedes": PARENT_NAME,
                "scientific_values_changed": True,
                "changed_panels": ["b", "c", "d", "e", "f", "g", "h", "i"],
                "scientific_correction": (
                    "Each Return session uses the same animal's original Naive session "
                    "as its policy donor, and the two eligible mouse-1616 Return sessions "
                    "are retained in the pooled ED2 cohort."
                ),
                "current_scientific_correction": {
                    "round": 29,
                    "status": "PASS",
                    "changes": [
                        "paired every Return session to the same animal's original Naive session",
                        "restored 1616bday1 and 1616b-3day1 to the pooled Return cohort",
                        "retained 58 Return sessions from 18 animals in panel d",
                        "used 54 Return sessions and 17 matched Naive baselines from 17 animals in panel h",
                        "excluded only mouse 640's four panel-h pairs because 640day1 had insufficient shallow-depth support",
                        "reran panels b-i and their cohort summaries",
                    ],
                },
                "panel_d_support": "58 Return sessions from 18 animals",
                "panel_h_support": (
                    "54 Return sessions paired with 17 Naive baselines from 17 animals"
                ),
                "panel_h_values_percent": {
                    "four_policy": 40.5,
                    "depth_varying_pbs": 86.7,
                },
                "current_display_refinement": {
                    "round": 29,
                    "status": "PASS",
                    "source": PARENT_NAME,
                    "policy": (
                        "restored the two eligible mouse-1616 Return sessions to the "
                        "pooled-return analyses while retaining panel-h label clearance"
                    ),
                },
                "publication_record_convention": {
                    "canonical_primary_stem": ROOT_STEM,
                    "record_change": "Round 29 pooled-return mouse-1616 restoration",
                },
                "vector_audit": qa["composite"]["vector_audit"],
                "raster_audit": {
                    "png": qa["composite"]["png_audit"],
                    "tiff": qa["composite"]["tiff_audit"],
                },
            }
        )
        write_json(stage / "ED02_VERSION_RECORD_MANIFEST.json", version_manifest)
        write_json(
            stage / "ED02_ROUND29_PROMOTION.json",
            {
                "schema": "ed02-round29-1616-restoration-promotion/v1",
                "status": "PASS",
                "record": TARGET_NAME,
                "parent_record": PARENT_NAME,
                "candidate": CANDIDATE.relative_to(REPO).as_posix(),
                "candidate_checks": candidate_checks,
                "label_audit": qa["label_audit"],
                "created_utc": timestamp,
            },
        )
        (stage / "MANIFEST.md").write_text(
            f"# {TARGET_NAME}\n\n"
            f"This scientific successor supersedes `{PARENT_NAME}`. It restores "
            "`1616bday1` and `1616b-3day1` to the pooled Return cohort, reruns all "
            "affected policy and reconstruction summaries, and retains the panel-h "
            "label clearance and upright P_BS typography.\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {TARGET_NAME}\n\n"
            "Run `python scripts/build_ed02_round28_panel_h_labels_candidate.py`, "
            "review its QA and composite, then run "
            "`python scripts/promote_ed02_round28_panel_h_labels.py`.\n",
            encoding="utf-8",
        )

        checks = {
            **candidate_checks,
            "parent_hash_closure": not parent_errors,
            "root_pdf_matches_candidate": sha256(stage / f"{ROOT_STEM}.pdf")
            == sha256(composite / f"{CANDIDATE_STEM}.pdf"),
            "root_svg_matches_candidate": sha256(stage / f"{ROOT_STEM}.svg")
            == sha256(composite / f"{CANDIDATE_STEM}.svg"),
            "root_600dpi_png_matches_candidate": sha256(
                stage / f"{ROOT_STEM}_600dpi.png"
            )
            == sha256(composite / f"{CANDIDATE_STEM}_600dpi.png"),
            "caption_authority_frozen": sha256(CAPTION_AUTHORITY)
            == sha256(validation / CAPTION_AUTHORITY.name),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Record checks failed: {checks}")
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "ed02-round29-record-validation/v1",
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


def archive_parent() -> dict[str, Any]:
    destination = ARCHIVE / PARENT_NAME
    if not TARGET.is_dir() or verify_hashes(TARGET):
        raise RuntimeError("A hash-closed successor is required before archiving")
    if not PARENT.is_dir() or destination.exists():
        raise RuntimeError("Round-28 archive precondition failed")
    before = tree_digest(PARENT)
    os.replace(PARENT, destination)
    after = tree_digest(destination)
    if before != after:
        raise RuntimeError("Archived parent tree changed during move")

    with ARCHIVE_LEDGER.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if PARENT_NAME in {row["record"] for row in rows}:
        raise RuntimeError("Round-28 parent already appears in archive ledger")
    rows.append(
        {
            "record": PARENT_NAME,
            "original_path": f"docs/figure_version_records/{PARENT_NAME}",
            "archive_path": (
                f"docs/figure_version_records_archive_2026-08-09/{PARENT_NAME}"
            ),
            "file_count": str(after["files"]),
            "bytes": str(after["bytes"]),
            "tree_sha256": after["sha256"],
            "move_status": "verified",
        }
    )
    temporary = ARCHIVE_LEDGER.with_name(f".{ARCHIVE_LEDGER.name}.round29.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, ARCHIVE_LEDGER)
    return after


def main() -> int:
    record = create_record()
    archived = archive_parent()
    print(
        json.dumps(
            {
                "status": "PASS",
                "record": str(record),
                "record_hash_errors": verify_hashes(record),
                "archived_parent": PARENT_NAME,
                "archived_parent_digest": archived,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
