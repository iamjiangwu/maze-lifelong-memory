"""Export the current figure masters and available panel bundles to ltm-figs."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from validate_all_figure_refinement_round2_records import EXPECTED, RECORDS, ROOT_STEMS, verify_hashes


REPO = Path(__file__).resolve().parents[1]
DESTINATION = Path(r"C:\Users\jiangwu\Desktop\ltm-figs")
PACKAGE = DESTINATION / "version_record_panels_2026-08-11_round2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    rows: list[dict[str, str | int]] = []
    for key, record_name in EXPECTED.items():
        record = RECORDS / record_name
        ok, errors = verify_hashes(record)
        if not ok:
            raise RuntimeError(f"Record hash closure failed: {record_name}; {errors[:5]}")
        stem = ROOT_STEMS[key]
        group = "main" if key.startswith("main") else "extended_data"
        for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff"):
            source = record / f"{stem}{suffix}"
            if not source.is_file():
                continue
            destination = PACKAGE / group / key / "master" / source.name
            copy_file(source, destination)
            if suffix == "_600dpi.png":
                copy_file(source, DESTINATION / source.name)
            rows.append(
                {
                    "figure": key,
                    "record": record_name,
                    "role": "master",
                    "source": source.relative_to(REPO).as_posix(),
                    "destination": destination.relative_to(DESTINATION).as_posix(),
                    "sha256": sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        panel_directories = [
            path
            for path in record.iterdir()
            if path.is_dir()
            and ("panel" in path.name.lower() or path.name in {"component_exports", "shared_assets"})
        ]
        for panel_directory in panel_directories:
            for source in sorted(panel_directory.rglob("*")):
                if not source.is_file() or source.suffix.lower() not in {".svg", ".pdf", ".png", ".tiff"}:
                    continue
                destination = (
                    PACKAGE
                    / group
                    / key
                    / "panels"
                    / panel_directory.name
                    / source.relative_to(panel_directory)
                )
                copy_file(source, destination)
                rows.append(
                    {
                        "figure": key,
                        "record": record_name,
                        "role": "panel",
                        "source": source.relative_to(REPO).as_posix(),
                        "destination": destination.relative_to(DESTINATION).as_posix(),
                        "sha256": sha256(destination),
                        "bytes": destination.stat().st_size,
                    }
                )
    caption = (
        REPO
        / "docs/manuscript_figure_captions_2026-08-10/figure_captions_main1-5_extended1-10.tex"
    )
    copy_file(caption, PACKAGE / caption.name)
    manifest = PACKAGE / "PACKAGE_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("figure", "record", "role", "source", "destination", "sha256", "bytes"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "status": "PASS",
            "package": str(PACKAGE),
            "files": len(rows),
            "manifest_sha256": sha256(manifest),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
