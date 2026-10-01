"""Synchronize only the promoted ED1 Round 18 record to Desktop ltm-figs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]
RECORD_NAME = "extended_data_fig01_2026-08-15_round18_group80_matched153"
RECORD = REPO / "docs/figure_version_records" / RECORD_NAME
CAPTION = (
    REPO
    / "docs/manuscript_figure_captions_2026-08-10"
    / "figure_captions_main1-5_extended1-10.tex"
)
DESKTOP = Path(r"C:\Users\jiangwu\Desktop\ltm-figs")
PACKAGE = DESKTOP / "version_record_panels_2026-08-11_round2"
ED1_FOLDER = PACKAGE / "extended_data_figure_01"
DOWNLOAD_CAPTION = Path(r"C:\Users\jiangwu\Downloads\figure_captions_tense_corrected.tex")
BACKUP = (
    DESKTOP.parent
    / "ltm-figs_archive_2026-08-15"
    / "ed01_before_round18_group80_matched153"
)
REPORT = (
    REPO
    / "docs/ed01_round18_group80_promotion_2026-08-15"
    / "DESKTOP_SYNC.json"
)
ROOT_STEM = "ed01_version_record"
NAMES = (
    f"{ROOT_STEM}.svg",
    f"{ROOT_STEM}.pdf",
    f"{ROOT_STEM}_300dpi.png",
    f"{ROOT_STEM}_600dpi.png",
    f"{ROOT_STEM}_600dpi.tiff",
    "MANIFEST.md",
    "RECORD_VALIDATION.json",
    "PROMOTION_COMPLETE.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.ed01-round18.tmp")
    shutil.copy2(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Staged copy hash mismatch: {destination}")
    os.replace(temporary, destination)


def make_backup() -> None:
    if BACKUP.exists():
        raise RuntimeError(f"Refusing to overwrite focused backup: {BACKUP}")
    BACKUP.mkdir(parents=True)
    shutil.copytree(ED1_FOLDER, BACKUP / "extended_data_figure_01")
    for source in (
        DESKTOP / f"{ROOT_STEM}_600dpi.png",
        PACKAGE / "PACKAGE_MANIFEST.csv",
        PACKAGE / "PACKAGE_LAYOUT.json",
        PACKAGE / CAPTION.name,
        DOWNLOAD_CAPTION,
    ):
        if source.is_file():
            shutil.copy2(source, BACKUP / source.name)


def update_package_metadata() -> None:
    manifest_path = PACKAGE / "PACKAGE_MANIFEST.csv"
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    sources = {
        f"extended_data_figure_01/{name}": RECORD / name for name in NAMES
    }
    sources[CAPTION.name] = CAPTION
    found: set[str] = set()
    for row in rows:
        destination = row["destination"]
        if destination not in sources:
            continue
        source = sources[destination]
        is_caption = destination == CAPTION.name
        row["figure"] = "all" if is_caption else "ed1"
        row["record"] = "caption_authority" if is_caption else RECORD_NAME
        row["source"] = source.relative_to(REPO).as_posix()
        row["sha256"] = sha256(source)
        row["bytes"] = str(source.stat().st_size)
        found.add(destination)
    if found != set(sources):
        raise RuntimeError(
            f"Desktop manifest lacks ED1 rows: {sorted(set(sources) - found)}"
        )
    temporary = manifest_path.with_name(f".{manifest_path.name}.ed01-round18.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest_path)

    layout_path = PACKAGE / "PACKAGE_LAYOUT.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    matches = [item for item in layout["folders"] if item["figure"] == "ed1"]
    if len(matches) != 1:
        raise RuntimeError("Desktop layout must declare ED1 exactly once")
    matches[0]["record"] = RECORD_NAME
    temporary_layout = layout_path.with_name(
        f".{layout_path.name}.ed01-round18.tmp"
    )
    temporary_layout.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_layout, layout_path)


def validate() -> dict[str, object]:
    pairs = {
        f"package/{name}": (RECORD / name, ED1_FOLDER / name) for name in NAMES
    }
    pairs["top_level_preview"] = (
        RECORD / f"{ROOT_STEM}_600dpi.png",
        DESKTOP / f"{ROOT_STEM}_600dpi.png",
    )
    pairs["package_caption"] = (CAPTION, PACKAGE / CAPTION.name)
    pairs["downloads_caption"] = (CAPTION, DOWNLOAD_CAPTION)
    matches = {
        key: left.is_file() and right.is_file() and sha256(left) == sha256(right)
        for key, (left, right) in pairs.items()
    }
    return {
        "status": "PASS" if all(matches.values()) else "FAIL",
        "record": RECORD_NAME,
        "matches": matches,
        "caption_authority_sha256": sha256(CAPTION),
        "backup": str(BACKUP),
        "desktop_ed1_files": len(
            [path for path in ED1_FOLDER.rglob("*") if path.is_file()]
        ),
    }


def main() -> int:
    if not RECORD.is_dir() or not PACKAGE.is_dir() or not ED1_FOLDER.is_dir():
        raise FileNotFoundError("Promoted record or Desktop ED1 package is missing")
    make_backup()
    for name in NAMES:
        atomic_copy(RECORD / name, ED1_FOLDER / name)
    atomic_copy(
        RECORD / f"{ROOT_STEM}_600dpi.png",
        DESKTOP / f"{ROOT_STEM}_600dpi.png",
    )
    atomic_copy(CAPTION, PACKAGE / CAPTION.name)
    atomic_copy(CAPTION, DOWNLOAD_CAPTION)
    update_package_metadata()
    result = validate()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
