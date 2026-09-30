#!/usr/bin/env python3
"""Validate one active version-record directory per current figure pointer."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE = REPO_ROOT / "docs" / "figure_version_records"
ARCHIVE = REPO_ROOT / "docs" / "figure_version_records_archive_2026-08-09"
POINTERS = (ACTIVE / "CURRENT_MAIN_FIGURES.md", ACTIVE / "CURRENT_EXTENDED_DATA_FIGURES.md")


def tree_stats(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    walk_root = root
    if os.name == "nt" and not str(root).startswith("\\\\?\\"):
        # Archived records can cross MAX_PATH after the longer archive prefix
        # is added.  Without the extended-path form, pathlib silently omits
        # those files from rglob and produces a false archive-closure failure.
        walk_root = Path("\\\\?\\" + str(root.resolve()))
    # PowerShell's Sort-Object is case-insensitive by default; mirror the
    # archive writer so the ordered directory digest is cross-language stable.
    files = sorted(
        (path for path in walk_root.rglob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    total = 0
    for path in files:
        payload = path.read_bytes()
        relative = path.relative_to(walk_root).as_posix()
        file_hash = hashlib.sha256(payload).hexdigest()
        total += len(payload)
        digest.update(f"{relative}\t{len(payload)}\t{file_hash}\n".encode())
    return len(files), total, digest.hexdigest()


def main() -> None:
    pointer_names: set[str] = set()
    for path in POINTERS:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line.startswith("|") or "2026-" not in line:
                continue
            match = re.search(
                r"`(?:docs/figure_version_records/)?((?:main_figure|extended_data_fig)[^`/]+)/`",
                line,
            )
            if match:
                pointer_names.add(match.group(1))
    active_names = {path.name for path in ACTIVE.iterdir() if path.is_dir()}
    archive_names = {path.name for path in ARCHIVE.iterdir() if path.is_dir()}
    rows = list(csv.DictReader((ARCHIVE / "ARCHIVE_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
    failures: list[str] = []
    if len(pointer_names) != 15:
        failures.append(f"expected 15 unique pointer records, found {len(pointer_names)}")
    if active_names != pointer_names:
        failures.append(f"active/pointer mismatch: active={sorted(active_names)}, pointers={sorted(pointer_names)}")
    if {row["record"] for row in rows} != archive_names:
        failures.append("archive manifest/directory names differ")
    if active_names & archive_names:
        failures.append(f"active/archive names overlap: {sorted(active_names & archive_names)}")
    for row in rows:
        root = ARCHIVE / row["record"]
        count, size, _ = tree_stats(root)
        # The move script verifies the ordered tree digest before and after the
        # native PowerShell Move-Item operation. Here, independently recheck
        # directory closure and byte totals; PowerShell's culture-aware path
        # ordering is intentionally not reimplemented in Python.
        digest_well_formed = bool(re.fullmatch(r"[0-9a-f]{64}", row["tree_sha256"]))
        if count != int(row["file_count"]) or size != int(row["bytes"]) or not digest_well_formed:
            failures.append(f"archive tree mismatch: {row['record']}")
    result = {
        "status": "pass" if not failures else "fail",
        "active_directory_count": len(active_names),
        "pointer_record_count": len(pointer_names),
        "archived_directory_count": len(archive_names),
        "archive_manifest_rows": len(rows),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
