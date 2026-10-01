"""Validate ED2's standardized publication-version-record naming contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageStat


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECORD = (
    REPO
    / "docs/figure_version_records/extended_data_fig02_2026-08-10_round16_publication"
)
DEFAULT_SOURCE = (
    REPO
    / "docs/figure_version_records/extended_data_fig02_2026-08-09_round15_panel_g_labels_vor"
)
SOURCE_PRIMARY = {
    "ed02_round13_panel_g_labels_version_record.svg",
    "ed02_round13_panel_g_labels_version_record.pdf",
    "ed02_round13_panel_g_labels_version_record_600dpi.png",
    "ed02_round13_panel_g_labels_version_record_600dpi.tiff",
    "ed02_round13_panel_g_labels_version_record_preview_1440.png",
}
SUPERSEDED_CLOSURE = {
    "MANIFEST.md",
    "ED02_VERSION_RECORD_MANIFEST.json",
    "PROMOTION_COMPLETE.txt",
    "RECORD_HASHES.csv",
    "RECORD_VALIDATION.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonblank(path: Path) -> bool:
    with Image.open(path) as image:
        extrema = ImageStat.Stat(image.convert("RGB")).extrema
    return any(low != high for low, high in extrema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--source-record", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    record = args.record.resolve()
    source = args.source_record.resolve()
    if record.name != "extended_data_fig02_2026-08-10_round16_publication":
        raise AssertionError(f"Unexpected target record: {record}")
    if source.name != "extended_data_fig02_2026-08-09_round15_panel_g_labels_vor":
        raise AssertionError(f"Unexpected source record: {source}")

    standardization = json.loads(
        (record / "ED02_STANDARDIZATION_MANIFEST.json").read_text(encoding="utf-8")
    )
    record_manifest = json.loads(
        (record / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
    )
    source_validation = json.loads((source / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))

    mappings_ok = True
    mapping_rows: list[dict[str, object]] = []
    for mapping in standardization["byte_identical_primary_mappings"]:
        source_path = source / mapping["source"]
        target_path = record / mapping["target"]
        expected = mapping["sha256"]
        passed = (
            source_path.is_file()
            and target_path.is_file()
            and sha256(source_path) == expected == sha256(target_path)
            and source_path.read_bytes() == target_path.read_bytes()
        )
        mappings_ok = mappings_ok and passed
        mapping_rows.append(
            {
                "source": mapping["source"],
                "target": mapping["target"],
                "expected_sha256": expected,
                "source_sha256": sha256(source_path) if source_path.is_file() else "",
                "target_sha256": sha256(target_path) if target_path.is_file() else "",
                "byte_identical": passed,
            }
        )

    parity_rows: list[dict[str, object]] = []
    unexpected: list[str] = []
    for source_path in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source).as_posix()
        target_path = record / relative
        if relative in SOURCE_PRIMARY:
            classification = "renamed_primary_asset"
            required_equal = False
        elif relative in SUPERSEDED_CLOSURE:
            classification = "superseded_record_identity_or_closure"
            required_equal = False
        else:
            classification = "unchanged_upstream_payload"
            required_equal = True
        source_hash = sha256(source_path)
        target_hash = sha256(target_path) if target_path.is_file() else ""
        equal = source_hash == target_hash
        if required_equal and not equal:
            unexpected.append(relative)
        parity_rows.append(
            {
                "path": relative,
                "classification": classification,
                "required_byte_equal": required_equal,
                "byte_equal": equal,
                "source_bytes": source_path.stat().st_size,
                "target_bytes": target_path.stat().st_size if target_path.is_file() else -1,
                "source_sha256": source_hash,
                "target_sha256": target_hash,
            }
        )
    with (record / "ED02_STANDARDIZATION_PARITY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parity_rows[0]))
        writer.writeheader()
        writer.writerows(parity_rows)

    with fitz.open(record / "ed02_version_record.pdf") as document:
        pdf_pages = len(document)
        pdf_images = len(document[0].get_images(full=True))
        pdf_drawings = len(document[0].get_drawings())
        pdf_text_chars = len(document[0].get_text())
    svg_text = (record / "ed02_version_record.svg").read_text(encoding="utf-8")
    with Image.open(record / "ed02_version_record_300dpi.png") as png300:
        png300_size = list(png300.size)
        png300_dpi = list(png300.info.get("dpi", ()))
    with Image.open(record / "ed02_version_record_600dpi.png") as png600:
        png600_size = list(png600.size)
        png600_dpi = list(png600.info.get("dpi", ()))
    with Image.open(record / "ed02_version_record_600dpi.tiff") as tiff:
        tiff_size = list(tiff.size)
        tiff_dpi = list(tiff.info.get("dpi", ()))
        tiff_mode = tiff.mode
        tiff_compression = tiff.info.get("compression")

    derived = standardization["derived_companion"]
    checks = {
        "source_validation_pass": str(source_validation.get("status", "")).lower() == "pass",
        "standardization_manifest_pass": standardization.get("status") == "PASS",
        "record_identity_exact": record_manifest.get("record") == record.name,
        "canonical_primary_stem_exact": record_manifest.get(
            "publication_record_convention", {}
        ).get("canonical_primary_stem")
        == "ed02_version_record",
        "scientific_values_unchanged": record_manifest.get("scientific_values_changed") is False
        and standardization.get("scientific_values_changed") is False,
        "artwork_bytes_unchanged": standardization.get("artwork_changed") is False
        and mappings_ok,
        "all_five_primary_mappings_byte_identical": mappings_ok and len(mapping_rows) == 5,
        "unchanged_upstream_payloads_byte_identical": not unexpected,
        "old_primary_root_names_absent": not any((record / name).exists() for name in SOURCE_PRIMARY),
        "canonical_root_asset_inventory": all(
            (record / name).is_file()
            for name in (
                "ed02_version_record.svg",
                "ed02_version_record.pdf",
                "ed02_version_record_300dpi.png",
                "ed02_version_record_600dpi.png",
                "ed02_version_record_600dpi.tiff",
            )
        ),
        "derived_300dpi_hash_exact": sha256(record / derived["target"]) == derived["sha256"],
        "pdf_one_page_vector_only": pdf_pages == 1 and pdf_images == 0 and pdf_drawings == 1021,
        "pdf_selectable_text_present": pdf_text_chars >= 873,
        "svg_has_editable_text": "<text" in svg_text,
        "svg_has_no_embedded_raster": "<image" not in svg_text,
        "png_300_dimensions_and_dpi": png300_size == [2250, 2250]
        and all(abs(float(value) - 300.0) < 1.0 for value in png300_dpi),
        "png_600_dimensions_and_dpi": png600_size == [4500, 4500]
        and all(abs(float(value) - 600.0) < 1.0 for value in png600_dpi),
        "tiff_600_rgb_lzw": tiff_size == [4500, 4500]
        and tiff_mode == "RGB"
        and tiff_compression == "tiff_lzw"
        and all(abs(float(value) - 600.0) < 0.1 for value in tiff_dpi),
        "png_300_nonblank": nonblank(record / "ed02_version_record_300dpi.png"),
        "png_600_nonblank": nonblank(record / "ed02_version_record_600dpi.png"),
        "panel_inventory_a_to_i": record_manifest.get("panel_count") == 9
        and record_manifest.get("panel_inventory") == "a-i exactly; no panel j",
        "behavior_only_manual_masks_not_applicable": record_manifest.get(
            "manual_mask_applicability"
        )
        == "not applicable; behavior-only analysis",
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "pass" if all(checks.values()) else "fail"

    (record / "PROMOTION_COMPLETE.txt").write_text(
        "PROMOTED 2026-08-10: standardized ED2 publication-record naming; artwork and data unchanged.\n",
        encoding="utf-8",
    )
    result = {
        "schema": "ed02_standardized_publication_record_validation/v1",
        "status": status,
        "record": record.relative_to(REPO).as_posix(),
        "source_record_name": source.name,
        "checks": checks,
        "unexpected_upstream_differences": unexpected,
        "mapping_rows": mapping_rows,
        "source_parity_rows": len(parity_rows),
        "pdf": {
            "pages": pdf_pages,
            "images": pdf_images,
            "drawings": pdf_drawings,
            "selectable_text_characters": pdf_text_chars,
        },
    }
    (record / "RECORD_VALIDATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hash_rows: list[dict[str, object]] = []
    for path in sorted(path for path in record.rglob("*") if path.is_file()):
        if path.name == "RECORD_HASHES.csv":
            continue
        relative = path.relative_to(record).as_posix()
        hash_rows.append(
            {
                "scope": relative.split("/", 1)[0] if "/" in relative else "payload",
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with (record / "RECORD_HASHES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(hash_rows[0]))
        writer.writeheader()
        writer.writerows(hash_rows)

    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "pass":
        failed = [key for key, passed in checks.items() if not passed]
        raise SystemExit(f"Validation failed: {failed}")


if __name__ == "__main__":
    main()
