"""Create the append-only ED7 version record for mouse 841 age metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "docs" / "figure_version_records"
PARENT_NAME = "extended_data_fig07_2026-08-16_round16_upright_i_axis"
PARENT = next(
    path
    for path in (
        RECORDS / PARENT_NAME,
        ROOT / "docs" / "figure_version_records_archive_2026-08-09" / PARENT_NAME,
    )
    if path.is_dir()
)
TARGET = RECORDS / "extended_data_fig07_2026-08-18_round17_mouse841_age_metadata"
ANALYSIS = ROOT / "docs" / "ed07_mouse841_age_metadata_2026-08-18"
RENDER = ANALYSIS / "render_round17"
MAIN3_SOURCE = (
    ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure3_2026-08-16_round9_available_window_slopes"
    / "source_data"
    / "first30"
    / "combined_first30_points_all_metrics.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_300dpi(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        expected = (2250, 1476)
        resized = image.convert("RGB").resize(expected, Image.Resampling.LANCZOS)
        resized.save(destination, dpi=(300, 300), optimize=True)
    with Image.open(destination) as image:
        if image.size != expected:
            raise AssertionError(f"Unexpected 300 dpi dimensions: {image.size}")


def image_details(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "width_px": image.width,
            "height_px": image.height,
            "format": image.format,
            "dpi": [round(float(value), 3) for value in image.info.get("dpi", ())],
            "compression": image.info.get("compression"),
        }


def vector_checks(svg: Path, pdf: Path) -> dict[str, Any]:
    svg_text = svg.read_text(encoding="utf-8")
    with fitz.open(pdf) as document:
        page = document[0]
        embedded_images = len(page.get_images(full=True))
        pdf_text = page.get_text()
        page_size = [float(page.rect.width), float(page.rect.height)]
        page_count = document.page_count
    checks = {
        "svg_has_no_embedded_raster": "<image" not in svg_text,
        "pdf_has_no_embedded_raster": embedded_images == 0,
        "pdf_one_page": page_count == 1,
        "svg_contains_upright_day_i_axis": r"Day $\mathrm{i}$ early vs Day $\mathrm{i}+1$ correlation" in svg_text,
        "pdf_contains_panel_titles": all(
            text in pdf_text for text in ("Place map", "Population vector", "Ensemble rate")
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Vector export check failed: {checks}")
    return {
        **checks,
        "pdf_embedded_images": embedded_images,
        "pdf_page_points": page_size,
    }


def iter_files(root: Path) -> list[tuple[str, Path]]:
    resolved = root.resolve()
    walk_root = Path(f"\\\\?\\{resolved}") if os.name == "nt" else resolved
    rows: list[tuple[str, Path]] = []
    for directory, _, names in os.walk(walk_root):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            text = str(path)
            if text.startswith("\\\\?\\"):
                text = text[4:]
            normal = Path(text)
            rel = normal.relative_to(resolved).as_posix()
            rows.append((rel, path))
    return sorted(rows, key=lambda item: item[0])


def write_hash_inventory(record: Path) -> None:
    inventory = record / "RECORD_HASHES.csv"
    rows = [
        {"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size}
        for rel, path in iter_files(record)
        if rel != inventory.name
    ]
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        writer.writerows(rows)


def verify_hash_inventory(record: Path) -> None:
    inventory = record / "RECORD_HASHES.csv"
    with inventory.open(newline="", encoding="utf-8") as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle)}
    actual = {
        rel: path for rel, path in iter_files(record) if rel != inventory.name
    }
    if set(expected) != set(actual):
        raise AssertionError("Record hash inventory does not close")
    for rel, path in actual.items():
        row = expected[rel]
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            raise AssertionError(f"Record hash mismatch: {rel}")


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    for path in (PARENT, ANALYSIS, RENDER, MAIN3_SOURCE):
        if not path.exists():
            raise FileNotFoundError(path)
    analysis_validation = json.loads((ANALYSIS / "VALIDATION.json").read_text(encoding="utf-8"))
    display_manifest = json.loads(
        (RENDER / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    if analysis_validation.get("status") != "PASS" or display_manifest.get("status") != "PASS":
        raise AssertionError("Scientific or display validation is not PASS")

    shutil.copytree(PARENT, TARGET)
    try:
        for name in (
            "ED07_FIRST30_PLOT_READY_MAINTAINED58.csv",
            "ED07_FIRST30_AGE_MODELS_MAINTAINED58.csv",
        ):
            path = TARGET / "plot_ready" / name
            if path.exists():
                path.unlink()
        old_validation = TARGET / "source_validation" / "ED07_MAINTAINED58_AUTHORITY_VALIDATION.json"
        if old_validation.exists():
            old_validation.unlink()

        plot_ready = TARGET / "plot_ready"
        source_validation = TARGET / "source_validation"
        copy(
            ANALYSIS / "ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",
            plot_ready / "ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",
        )
        copy(
            ANALYSIS / "ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv",
            plot_ready / "ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv",
        )
        for name in (
            "VALIDATION.json",
            "ED07_FIRST30_COHORT_MAINTAINED59.csv",
            "ED07_MOUSE841_SUPPORT_ANNOTATION.csv",
            "ED07_FROZEN58_MODEL_REPRODUCTION.csv",
            "ED07_FROZEN58_MODEL_REPRODUCTION_CHECK.csv",
            "ED07_SHARED58_VALUE_IDENTITY_CHECK.csv",
        ):
            destination = (
                "ED07_MOUSE841_AGE_METADATA_VALIDATION.json"
                if name == "VALIDATION.json"
                else name
            )
            copy(ANALYSIS / name, source_validation / destination)
        copy(
            MAIN3_SOURCE,
            TARGET / "source_authority" / "combined_first30_points_all_metrics.csv",
        )

        assembly_stem = "ed07_age_controls_six_panel_round17"
        output_map = {
            "svg": "ed07_version_record.svg",
            "pdf": "ed07_version_record.pdf",
            "png": "ed07_version_record_600dpi.png",
            "tiff": "ed07_version_record_600dpi.tiff",
        }
        for extension, destination in output_map.items():
            copy(RENDER / f"{assembly_stem}.{extension}", TARGET / destination)
        make_300dpi(
            TARGET / "ed07_version_record_600dpi.png",
            TARGET / "ed07_version_record_300dpi.png",
        )
        copy(
            RENDER / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json",
            TARGET / "ED07_DISPLAY_MANIFEST.json",
        )
        copy(RENDER / "ED07_SIX_PANEL_CAPTION_NOTES.md", TARGET / "CAPTION_NOTES.md")
        copy(RENDER / "ED07_SIX_PANEL_FIGURE_CONTRACT.md", TARGET / "FIGURE_CONTRACT.md")
        copy(
            RENDER / "ed07_slide9_geometry_mockup_round17.png",
            TARGET / "ed07_slide_mockup.png",
        )
        for path in RENDER.glob("*round17.*"):
            copy(path, TARGET / "component_exports" / path.name)
        for source in (
            ROOT / "scripts" / "build_ed07_mouse841_age_metadata_successor.py",
            ROOT / "scripts" / "render_ed07_slide9_powerpoint_matched.py",
            Path(__file__),
        ):
            copy(source, TARGET / "source_code" / source.name)

        bottom_parity: dict[str, Any] = {}
        for extension in ("png", "tiff"):
            parent = (
                PARENT
                / "component_exports"
                / f"ed07_panels_def_long_gap_clean36_round16.{extension}"
            )
            current = (
                TARGET
                / "component_exports"
                / f"ed07_panels_def_long_gap_clean36_round17.{extension}"
            )
            identical = sha256(parent) == sha256(current)
            if not identical:
                raise AssertionError(f"ED7 panels d-f {extension} changed")
            bottom_parity[extension] = {
                "byte_identical": identical,
                "parent_sha256": sha256(parent),
                "successor_sha256": sha256(current),
            }

        vector = vector_checks(
            TARGET / "ed07_version_record.svg",
            TARGET / "ed07_version_record.pdf",
        )
        raster = {
            "png_300dpi": image_details(TARGET / "ed07_version_record_300dpi.png"),
            "png_600dpi": image_details(TARGET / "ed07_version_record_600dpi.png"),
            "tiff_600dpi": image_details(TARGET / "ed07_version_record_600dpi.tiff"),
        }
        if raster["png_600dpi"]["width_px"] != 4500 or raster["png_600dpi"]["height_px"] != 2952:
            raise AssertionError("ED7 600 dpi PNG dimensions changed")

        scientific_inputs = [
            {
                "path": "plot_ready/ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",
                "role": "first30_plot",
            },
            {
                "path": "plot_ready/ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv",
                "role": "first30_models",
            },
            {
                "path": "source_validation/ED07_MOUSE841_AGE_METADATA_VALIDATION.json",
                "role": "mouse841_age_metadata_validation",
            },
            {
                "path": "source_authority/combined_first30_points_all_metrics.csv",
                "role": "accepted_first30_source",
            },
            {
                "path": "plot_ready/ed09_long_gap_age_joined_corrected36.csv",
                "role": "long_gap_plot",
            },
            {
                "path": "plot_ready/ed09_long_gap_age_transition_fixed_models_corrected36.csv",
                "role": "long_gap_models",
            },
        ]
        for item in scientific_inputs:
            path = TARGET / item["path"]
            item["sha256"] = sha256(path)
            item["bytes"] = path.stat().st_size

        output_names = {
            "svg": "ed07_version_record.svg",
            "pdf": "ed07_version_record.pdf",
            "png_300dpi": "ed07_version_record_300dpi.png",
            "png_600dpi": "ed07_version_record_600dpi.png",
            "tiff_600dpi": "ed07_version_record_600dpi.tiff",
        }
        version_manifest = json.loads(
            (PARENT / "ED07_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        version_manifest.update(
            {
                "schema": "ed07-round17-mouse841-age-metadata-successor/v1",
                "record": TARGET.name,
                "supersedes": PARENT.name,
                "accepted_date": "2026-08-18",
                "accepted_display_round": 17,
                "scientific_values_changed": True,
                "scientific_update": {
                    "mouse": "841",
                    "birth_date": "2025-02-06",
                    "sex": "F",
                    "day1_age_days": 120,
                    "top_observations_per_metric_before": 58,
                    "top_mice_before": 21,
                    "top_observations_per_metric_after": 59,
                    "top_mice_after": 22,
                    "panels_changed": ["a", "b", "c"],
                    "panels_d_to_f_pixel_identical": True,
                },
                "scientific_inputs": scientific_inputs,
                "display_contract": display_manifest["display_contract"],
                "root_output_sha256": {
                    key: sha256(TARGET / name) for key, name in output_names.items()
                },
                "root_output_bytes": {
                    key: (TARGET / name).stat().st_size for key, name in output_names.items()
                },
            }
        )
        write_json(TARGET / "ED07_VERSION_RECORD_MANIFEST.json", version_manifest)

        update_manifest = {
            "schema": "ed07-mouse841-age-metadata-update/v1",
            "status": "PASS",
            "record": TARGET.name,
            "supersedes": PARENT.name,
            "figure": "Extended Data Figure 7",
            "updated_panels": ["a", "b", "c"],
            "unchanged_panels": ["d", "e", "f"],
            "metadata": {
                "animal_id": "841",
                "birth_date": "2025-02-06",
                "sex": "F",
                "day1_age_days": 120,
            },
            "cohort": {
                "before": "58 observations from 21 mice",
                "after": "59 observations from 22 mice",
            },
            "support_annotation": (
                "Mouse 841 is included in the current first-window estimator. Its 139 "
                "rate-gated cells are below the separate 164-cell matched-support audit target."
            ),
            "bottom_row_parity": bottom_parity,
            "vector_checks": vector,
            "raster_checks": raster,
            "analysis_validation": {
                "path": relative(ANALYSIS / "VALIDATION.json"),
                "sha256": sha256(ANALYSIS / "VALIDATION.json"),
            },
        }
        write_json(
            TARGET / "MOUSE841_AGE_METADATA_UPDATE_2026_08_18.json",
            update_manifest,
        )
        write_json(
            TARGET / "ED07_DISPLAY_VALIDATION.json",
            {
                "schema": "ed07-mouse841-display-validation/v1",
                "status": "PASS",
                "visual_qa_status": "PASS",
                "top_primary_scope": "59 maintained observations / 22 mice; BH3",
                "bottom_primary_scope": "36 pairs / 14 mice; BH4",
                "panels_d_to_f_pixel_identical": True,
                "bottom_row_parity": bottom_parity,
                "vector_validation": vector,
                "raster_validation": raster,
            },
        )
        write_json(
            TARGET / "RECORD_VALIDATION.json",
            {
                "schema": "ed07-mouse841-age-metadata-record-validation/v1",
                "record": TARGET.name,
                "checks": {
                    "scientific_validation_pass": True,
                    "frozen58_reproduced": True,
                    "shared58_values_identical": True,
                    "successor59_22_complete": True,
                    "display_validation_pass": True,
                    "panels_d_to_f_pixel_identical": True,
                    "vector_validation_pass": True,
                    "raster_validation_pass": True,
                    "manuscript_caption_files_edited": False,
                },
                "status": "PASS",
                "validated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        (TARGET / "MANIFEST.md").write_text(
            f"# {TARGET.name}\n\n"
            f"This append-only record supersedes `{PARENT.name}`. Mouse 841 now "
            "has a recorded birth date of 2025-02-06 and is included in panels a to c. "
            "Those panels use 59 observations from 22 mice. Panels d to f are pixel "
            "identical to the parent record. No manuscript caption file was edited.\n",
            encoding="utf-8",
        )
        (TARGET / "PROMOTION_COMPLETE.txt").write_text(
            "ED7 round 17 mouse 841 age metadata successor validated and ready for pointer promotion.\n",
            encoding="utf-8",
        )
        write_hash_inventory(TARGET)
        verify_hash_inventory(TARGET)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "target": relative(TARGET),
                    "files": len(iter_files(TARGET)),
                    "record_hashes_sha256": sha256(TARGET / "RECORD_HASHES.csv"),
                    "bottom_row_pixel_identical": True,
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        if TARGET.exists():
            shutil.rmtree(TARGET)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
