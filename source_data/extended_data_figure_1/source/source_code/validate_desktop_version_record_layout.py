"""Validate the flat one-folder-per-figure Desktop version-record package."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

from export_current_version_records_to_desktop import FOLDER_NAMES, PACKAGE, REPO
from validate_all_figure_refinement_round2_records import EXPECTED, RECORDS, ROOT_STEMS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    expected_directories = set(FOLDER_NAMES.values())
    actual_directories = {path.name for path in PACKAGE.iterdir() if path.is_dir()}
    if actual_directories != expected_directories:
        failures.append(
            f"figure folders differ: actual={sorted(actual_directories)}"
        )
    forbidden = [
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_dir() and path.name in {"main", "extended_data", "master"}
    ]
    if forbidden:
        failures.append(f"legacy nested directories remain: {forbidden}")

    manifest = PACKAGE / "PACKAGE_MANIFEST.csv"
    layout_path = PACKAGE / "PACKAGE_LAYOUT.json"
    if not manifest.is_file() or not layout_path.is_file():
        failures.append("package manifest or layout declaration missing")
        rows: list[dict[str, str]] = []
    else:
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        declared = {item["figure"]: item for item in layout.get("folders", [])}
        if len(declared) != 15:
            failures.append(f"layout declares {len(declared)} figures, expected 15")
        for key, record in EXPECTED.items():
            item = declared.get(key, {})
            if item.get("folder") != FOLDER_NAMES[key] or item.get("record") != record:
                failures.append(f"layout pointer mismatch: {key}")

    declared_paths = {row["destination"]: row for row in rows}
    actual_paths = {
        path.relative_to(PACKAGE).as_posix(): path
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and path.name not in {"PACKAGE_MANIFEST.csv", "PACKAGE_LAYOUT.json"}
    }
    if set(declared_paths) != set(actual_paths):
        failures.append(
            "manifest/file inventory differs: "
            f"missing={sorted(set(declared_paths) - set(actual_paths))[:5]}, "
            f"extra={sorted(set(actual_paths) - set(declared_paths))[:5]}"
        )
    for relative in sorted(set(declared_paths) & set(actual_paths)):
        row = declared_paths[relative]
        path = actual_paths[relative]
        source = REPO / row["source"]
        if not source.is_file():
            failures.append(f"missing source: {row['source']}")
            continue
        if (
            path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"]
            or sha256(source) != row["sha256"]
        ):
            failures.append(f"hash mismatch: {relative}")

    preview_results: dict[str, bool] = {}
    desktop = PACKAGE.parent
    for key, record_name in EXPECTED.items():
        source = RECORDS / record_name / f"{ROOT_STEMS[key]}_600dpi.png"
        destination = desktop / source.name
        matched = (
            source.is_file()
            and destination.is_file()
            and sha256(source) == sha256(destination)
        )
        preview_results[key] = matched
        if not matched:
            failures.append(f"top-level preview mismatch: {key}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "package": str(PACKAGE),
        "figure_folder_count": len(actual_directories),
        "manifest_rows": len(rows),
        "top_level_previews_matching": sum(preview_results.values()),
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
