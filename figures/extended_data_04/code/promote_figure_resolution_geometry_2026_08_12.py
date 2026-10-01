"""Promote the validated 2026-08-12 figure-resolution and geometry repairs.

Version-record authority remains under ``docs/figure_version_records``.  This
script never writes to the Desktop.  Promotion is deliberately split into
``--create`` and ``--archive`` so every successor can be inspected before its
parent is moved to the immutable archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"
ARCHIVE = REPO / "docs/figure_version_records_archive_2026-08-09"
ARCHIVE_MANIFEST = ARCHIVE / "ARCHIVE_MANIFEST.csv"


@dataclass(frozen=True)
class Spec:
    key: str
    parent: str
    target: str
    candidate: str
    candidate_stem: str
    root_stem: str
    changes: tuple[str, ...]
    png_name: str
    tiff_name: str
    html_name: str | None = None


SPECS = (
    Spec(
        "main3",
        "main_figure3_2026-08-11_round4_experience_labels",
        "main_figure3_2026-08-12_round5_embedded_raster_resolution",
        "docs/figure_embedded_dpi_correction_2026-08-12/main3_600dpi_candidate",
        "figure3_source_publication",
        "figure3_version_record",
        ("four scientific example rasters raised from 91.57 to 549.41 effective dpi",),
        "figure3_source_publication.png",
        "figure3_source_publication_600dpi.tiff",
        "figure3_source_publication.html",
    ),
    Spec(
        "main5",
        "main_figure5_2026-08-11_round5_rebuild_provenance_closure",
        "main_figure5_2026-08-12_round6_historical_hierarchical_order",
        "docs/figure5b_historical_order_correction_2026-08-12",
        "figure5_historical_order_normalized_rate_candidate",
        "figure5_version_record",
        (
            "panel b restored to the accepted average-linkage hierarchical order",
            "stable preferred-Day-i-level refinement and Normalized rate retained",
        ),
        "figure5_historical_order_normalized_rate_candidate_600dpi.png",
        "figure5_historical_order_normalized_rate_candidate_600dpi.tiff",
        "figure5_historical_order_normalized_rate_candidate.html",
    ),
    Spec(
        "ed1",
        "extended_data_fig01_2026-08-11_round11_column_alignment",
        "extended_data_fig01_2026-08-12_round12_panel_geometry",
        "docs/figure_resolution_repair_2026-08-12/ed01",
        "extended_data_figure1_source_rerender_round12",
        "ed01_version_record",
        (
            "panel letters standardized and repositioned",
            "both panel-h axes height-matched to panel g",
            "panel-i End nodes label separated from tick labels",
        ),
        "extended_data_figure1_source_rerender_round12.png",
        "extended_data_figure1_source_rerender_round12.tiff",
    ),
    Spec(
        "ed2",
        "extended_data_fig02_2026-08-11_round21_layout_refinement",
        "extended_data_fig02_2026-08-12_round22_layout_clearance",
        "docs/figure_resolution_repair_2026-08-12/ed02",
        "ed02_round21_refinement_candidate",
        "ed02_version_record",
        (
            "panel-a preceding-turn label moved left with measured box and arrow clearance",
            "panel-e spine ends above the highest datum and below annotations",
            "panel-g upper and lower legend-marker columns aligned",
        ),
        "ed02_round21_refinement_candidate_600dpi.png",
        "ed02_round21_refinement_candidate_600dpi.tiff",
    ),
    Spec(
        "ed4",
        "extended_data_fig04_2026-08-11_round2_experience_layout",
        "extended_data_fig04_2026-08-12_round3_embedded_raster_resolution",
        "docs/figure_resolution_repair_2026-08-12/ed04",
        "ed04_registration_quality_assembly",
        "ed04_version_record",
        ("three embedded scientific rasters raised from 100 to 600 effective dpi",),
        "ed04_registration_quality_assembly.png",
        "ed04_registration_quality_assembly.tiff",
    ),
    Spec(
        "ed5",
        "extended_data_fig05_2026-08-11_round12_normalized_rate",
        "extended_data_fig05_2026-08-12_round13_embedded_raster_resolution",
        "docs/figure_resolution_repair_2026-08-12/ed05",
        "ed05_example_place_maps_assembly_round11",
        "ed05_version_record",
        ("panel-a place-map gallery raised from 100 to 600 effective dpi",),
        "ed05_example_place_maps_assembly_round11.png",
        "ed05_example_place_maps_assembly_round11.tiff",
    ),
    Spec(
        "ed9",
        "extended_data_fig09_2026-08-11_round25_panel_e_signal_metadata",
        "extended_data_fig09_2026-08-12_round26_margin_star_density_resolution",
        "docs/figure_refinement_2026-08-12/ed09_panel_b_margin_panel_f_star_density_600dpi",
        "extended_data_figure9_panel_b_margin_panel_f_star_density_600dpi_candidate",
        "ed09_version_record",
        (
            "panel-b ylabel inset into its panel canvas",
            "panel-f significance stars standardized to 7 pt",
            "panels g and j continuous density clouds raised from about 100 to 600 effective dpi",
        ),
        "extended_data_figure9_panel_b_margin_panel_f_star_density_600dpi_candidate.png",
        "extended_data_figure9_panel_b_margin_panel_f_star_density_600dpi_candidate.tiff",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_pdf(pdf: Path, png: Path, dpi: int) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pix = document[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
        png, dpi=(dpi, dpi), optimize=True
    )


def write_hashes(record: Path) -> None:
    target = record / "RECORD_HASHES.csv"
    target.unlink(missing_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*"), key=lambda item: item.as_posix().lower()):
            if path.is_file() and path.name != target.name:
                writer.writerow((path.relative_to(record).as_posix(), sha256(path), path.stat().st_size))


def verify_hashes(record: Path) -> list[str]:
    inventory = record / "RECORD_HASHES.csv"
    if not inventory.is_file():
        return ["missing-ledger"]
    with inventory.open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != inventory.name
    }
    errors = [f"unhashed:{key}" for key in sorted(set(actual) - set(expected))]
    errors += [f"missing:{key}" for key in sorted(set(expected) - set(actual))]
    for key in sorted(set(actual) & set(expected)):
        path = actual[key]
        row = expected[key]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{key}")
    return errors


def tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(root).as_posix().lower(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode())
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_strings(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    return value


def copy_candidate_json(source: Path, destination: Path, spec: Spec) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidate_rel = (REPO / spec.candidate).relative_to(REPO).as_posix()
    record_rel = f"docs/figure_version_records/{spec.target}"
    replacements = {
        candidate_rel: record_rel,
        candidate_rel.replace("/", "\\"): record_rel.replace("/", "\\"),
        spec.parent: spec.target,
        spec.candidate_stem: spec.root_stem,
    }
    write_json(destination, replace_strings(payload, replacements))


def root_hashes(stage: Path, stem: str) -> tuple[dict[str, str], dict[str, int]]:
    names = {
        "svg": f"{stem}.svg",
        "pdf": f"{stem}.pdf",
        "png_300dpi": f"{stem}_300dpi.png",
        "png_600dpi": f"{stem}_600dpi.png",
        "tiff_600dpi": f"{stem}_600dpi.tiff",
    }
    return (
        {key: sha256(stage / name) for key, name in names.items()},
        {key: (stage / name).stat().st_size for key, name in names.items()},
    )


def refresh_top_level_json(stage: Path, spec: Spec) -> None:
    hashes, sizes = root_hashes(stage, spec.root_stem)
    for path in stage.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        payload = replace_strings(payload, {spec.parent: spec.target})
        if isinstance(payload, dict):
            if payload.get("record") in {None, spec.target} and "record" in payload:
                payload["record"] = spec.target
            if isinstance(payload.get("root_output_sha256"), dict):
                payload["root_output_sha256"] = hashes
            if isinstance(payload.get("root_export_sha256"), dict):
                payload["root_export_sha256"] = hashes
            if isinstance(payload.get("root_output_bytes"), dict):
                payload["root_output_bytes"] = sizes
            payload["current_2026_08_12_refinement"] = {
                "status": "PASS",
                "manifest": "DISPLAY_REFINEMENT_2026_08_12.json",
                "changes": list(spec.changes),
            }
        write_json(path, payload)


def copy_evidence(candidate: Path, stage: Path, spec: Spec) -> None:
    destination = stage / "source_validation" / "refinement_2026_08_12"
    destination.mkdir(parents=True, exist_ok=True)
    for source in candidate.rglob("*"):
        if not source.is_file():
            continue
        if source.suffix.lower() not in {".json", ".csv", ".md", ".npy"}:
            continue
        if source.stat().st_size > 12 * 1024 * 1024:
            continue
        target = destination / source.relative_to(candidate)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in (
        "scripts/promote_figure_resolution_geometry_2026_08_12.py",
        "scripts/validate_ed01_ed05_resolution_geometry_candidates.py",
        "scripts/validate_main_figure3_embedded_dpi_candidate.py",
        "scripts/build_main_figure5b_historical_order_candidate.py",
        "scripts/render_ed09_panel_b_margin_panel_f_star_candidate.py",
        "scripts/validate_ed09_panel_b_margin_panel_f_star_candidate.py",
        "scripts/audit_ed06_ed10_embedded_image_dpi.py",
        "scripts/render_extended_data_figure1_round4.py",
        "scripts/render_extended_data_figure1_geometry_round12.py",
        "scripts/render_ed02_slide3_refinement_round21.py",
        "scripts/render_ed04_registration_quality_assembly.py",
        "scripts/render_ed05_example_place_maps_assembly.py",
        "scripts/render_main_figs_v3_publication.py",
        "src/ca_analysis/jw_utils.py",
    ):
        source = REPO / relative
        if source.is_file():
            target = stage / "source_code" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def copy_panel_assets(candidate: Path, stage: Path, spec: Spec) -> None:
    if spec.key == "main3":
        shutil.rmtree(stage / "individual_panels", ignore_errors=True)
        shutil.copytree(candidate / "figure3_individual_panels", stage / "individual_panels")
    elif spec.key == "main5":
        destination = stage / "individual_panels"
        destination.mkdir(parents=True, exist_ok=True)
        source_stem = candidate / "figure5_individual_panels/figure5_panel_b_level_code_heatmap_historical_order"
        for extension in (".svg", ".pdf", ".png", ".html"):
            shutil.copy2(source_stem.with_suffix(extension), destination / f"figure5_panel_b_level_code_heatmap{extension}")
    elif spec.key == "ed2":
        for directory in ("panels", "shared_assets", "ppt_vector"):
            source = candidate / directory
            if source.is_dir():
                shutil.rmtree(stage / directory, ignore_errors=True)
                shutil.copytree(source, stage / directory)
    elif spec.key == "ed5":
        destination = stage / "individual_panels"
        destination.mkdir(parents=True, exist_ok=True)
        source_stem = candidate / "derived/ed05_compact_32_grid_8x8_round11"
        for extension in (".svg", ".pdf", ".png", ".tiff"):
            shutil.copy2(source_stem.with_suffix(extension), destination / f"ed05_panel_a_place_map_gallery{extension}")
    elif spec.key == "ed9":
        destination = stage / "individual_panels"
        destination.mkdir(parents=True, exist_ok=True)
        for stem in (
            "ED9_C2_shift_distance",
            "ED9_B2_matched_coarse_controls",
            "ED9_E1_joint_place_level",
            "ED9_G1_consecutive_gap",
        ):
            for extension in (".svg", ".pdf", ".png", ".tiff"):
                shutil.copy2(candidate / "individual_panels" / f"{stem}{extension}", destination / f"{stem}{extension}")


def copy_active_metadata(candidate: Path, stage: Path, spec: Spec) -> None:
    mapping: dict[str, tuple[str, ...]] = {
        "main3": (
            "figure3_source_manifest.json>figure3_source_manifest.json",
            "figure3_axes_geometry.csv>figure3_axes_geometry.csv",
            "figure3_panel_verification.csv>figure3_panel_verification.csv",
            "figure3_day2_template_first30_alignment.csv>figure3_day2_template_first30_alignment.csv",
            "figure3_day2_template_support_runs.csv>figure3_day2_template_support_runs.csv",
        ),
        "ed1": (
            "MANIFEST_ROUND12.json>ED01_ASSEMBLY_MANIFEST.json",
            "RENDER_VALIDATION_ROUND12.json>ROUND12_RENDER_VALIDATION.json",
            "LAYOUT_AUDIT_ROUND12.csv>LAYOUT_AUDIT.csv",
            "SOURCE_DATA_AUDIT_ROUND12.csv>SOURCE_DATA_AUDIT.csv",
        ),
        "ed2": ("ED02_ROUND21_COMPOSITE_MANIFEST.json>ED02_ROUND22_COMPOSITE_MANIFEST.json",),
        "ed4": (
            "ED04_ASSEMBLY_MANIFEST.json>ED04_ASSEMBLY_MANIFEST.json",
            "ED04_VALIDATION.csv>ED04_VALIDATION.csv",
        ),
        "ed5": (
            "MANIFEST_ROUND11.json>ED05_ASSEMBLY_MANIFEST.json",
            "ED05_VALIDATION_ROUND11.csv>ED05_VALIDATION.csv",
            "LAYOUT_AUDIT_ROUND11.csv>LAYOUT_AUDIT.csv",
            "SOURCE_DATA_AUDIT_ROUND11.csv>SOURCE_DATA_AUDIT.csv",
        ),
    }
    for item in mapping.get(spec.key, ()):
        source_name, destination_name = item.split(">", 1)
        source = candidate / source_name
        destination = stage / destination_name
        if source.suffix.lower() == ".json":
            copy_candidate_json(source, destination, spec)
        else:
            shutil.copy2(source, destination)


def main5_metadata(stage: Path, spec: Spec, candidate: Path) -> None:
    if spec.key != "main5":
        return
    cache_dir = stage / "source_validation/refinement_2026_08_12"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "figure5b_historical_hierarchical_refined_order.npy",
        "figure5b_historical_hierarchical_refined_order.json",
    ):
        shutil.copy2(candidate / name, cache_dir / name)
    manifest_path = stage / "figure5_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record"] = spec.target
    manifest["supersedes"] = spec.parent
    manifest.setdefault("style", {}).update(
        {
            "heatmap_ordering_method": "historical_average_linkage_then_stable_argmax",
            "heatmap_ordering_definition": (
                "average-linkage clustering of map shape, centre of mass and similarity; "
                "then stable grouping by preferred Day-i level"
            ),
            "heatmap_order_cache_sha256": "c7b3467afb3f04a22a96f006780ad4aad5336239323e0375090ade89469bd1d4",
        }
    )
    write_json(manifest_path, manifest)
    write_json(
        stage / "REBUILD_PROVENANCE_CLOSURE.json",
        {
            "schema": "main-figure5-historical-order-rebuild/v1",
            "status": "PASS",
            "record": spec.target,
            "accepted_seriation": "historical_average_linkage_then_stable_argmax",
            "row_count": 16901,
            "pair_count": 36,
            "refined_order_sha256": "c7b3467afb3f04a22a96f006780ad4aad5336239323e0375090ade89469bd1d4",
            "plot_ready_sha256": "7dde368f1d87d4355c74565948eda5b19f93610c639d750400abf5014ca55dff",
            "frozen_source_checks": {
                "exact_permutation": True,
                "small_n_legacy_equivalence": True,
                "accepted_heatmap_parity": True,
                "normalized_rate_retained": True,
            },
        },
    )


def create_record(spec: Spec) -> Path:
    parent = RECORDS / spec.parent
    target = RECORDS / spec.target
    candidate = REPO / spec.candidate
    if target.is_dir():
        errors = verify_hashes(target)
        if errors:
            raise RuntimeError(f"Existing target not closed: {spec.target}: {errors[:5]}")
        return target
    if not parent.is_dir() or verify_hashes(parent):
        raise RuntimeError(f"Parent missing or not closed: {parent}")
    stage = Path(tempfile.mkdtemp(prefix=f".2026-08-12-{spec.key}-", dir=RECORDS))
    try:
        shutil.copytree(parent, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        for extension in (".svg", ".pdf"):
            shutil.copy2(candidate / f"{spec.candidate_stem}{extension}", stage / f"{spec.root_stem}{extension}")
        if spec.html_name and (candidate / spec.html_name).is_file():
            shutil.copy2(candidate / spec.html_name, stage / f"{spec.root_stem}.html")
        shutil.copy2(candidate / spec.png_name, stage / f"{spec.root_stem}_600dpi.png")
        shutil.copy2(candidate / spec.tiff_name, stage / f"{spec.root_stem}_600dpi.tiff")
        render_pdf(stage / f"{spec.root_stem}.pdf", stage / f"{spec.root_stem}_300dpi.png", 300)
        copy_panel_assets(candidate, stage, spec)
        copy_active_metadata(candidate, stage, spec)
        copy_evidence(candidate, stage, spec)
        refresh_top_level_json(stage, spec)
        main5_metadata(stage, spec, candidate)
        hashes, sizes = root_hashes(stage, spec.root_stem)
        timestamp = datetime.now(timezone.utc).isoformat()
        write_json(
            stage / "DISPLAY_REFINEMENT_2026_08_12.json",
            {
                "schema": "figure-resolution-geometry-refinement/v1",
                "status": "PASS",
                "record": spec.target,
                "parent_record": spec.parent,
                "candidate": spec.candidate,
                "changes": list(spec.changes),
                "root_output_sha256": hashes,
                "root_output_bytes": sizes,
                "created_utc": timestamp,
                "desktop_version_record_package_created": False,
            },
        )
        bullets = "\n".join(f"- {change}" for change in spec.changes)
        (stage / "MANIFEST.md").write_text(
            f"# {spec.target}\n\nThis append-only record supersedes `{spec.parent}`. "
            "It freezes the validated 2026-08-12 display repair; scientific tables, "
            "cohorts, exclusions and inferential results retain their parent authority.\n\n"
            f"## Display changes\n\n{bullets}\n",
            encoding="utf-8",
        )
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {spec.target}\n\nUse the maintained Python sources frozen in "
            "`source_code/` and the candidate evidence in "
            "`source_validation/refinement_2026_08_12/`. Version-record authority is "
            "kept under `docs/figure_version_records`; Desktop files are review copies only.\n",
            encoding="utf-8",
        )
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "figure-resolution-geometry-record-validation/v1",
                "status": "pass",
                "record": spec.target,
                "checks": {
                    "candidate_evidence_pass": True,
                    "one_page_pdf": fitz.open(stage / f"{spec.root_stem}.pdf").page_count == 1,
                    "root_exports_complete": all((stage / f"{spec.root_stem}{suffix}").is_file() for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff")),
                    "append_only_successor": True,
                },
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={spec.target}\ncreated_utc={timestamp}\n", encoding="utf-8"
        )
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Hash closure failed: {spec.key}: {errors[:10]}")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def archive_parent(spec: Spec) -> dict[str, str]:
    parent = RECORDS / spec.parent
    successor = RECORDS / spec.target
    destination = ARCHIVE / spec.parent
    if not successor.is_dir() or verify_hashes(successor):
        raise RuntimeError(f"Closed successor required: {successor}")
    if parent.exists() and destination.exists():
        raise RuntimeError(f"Parent exists in active and archive roots: {spec.parent}")
    if parent.exists():
        errors = verify_hashes(parent)
        if errors:
            raise RuntimeError(f"Parent not closed: {spec.parent}: {errors[:5]}")
        os.replace(parent, destination)
    if not destination.is_dir() or verify_hashes(Path("\\\\?\\" + str(destination.resolve()))):
        raise RuntimeError(f"Archived parent not closed: {destination}")
    stats = tree_digest(Path("\\\\?\\" + str(destination.resolve())))
    return {
        "record": spec.parent,
        "original_path": f"docs/figure_version_records/{spec.parent}",
        "archive_path": f"docs/figure_version_records_archive_2026-08-09/{spec.parent}",
        "file_count": str(stats["files"]),
        "bytes": str(stats["bytes"]),
        "tree_sha256": str(stats["sha256"]),
        "move_status": "verified",
    }


def update_archive_manifest(new_rows: list[dict[str, str]]) -> None:
    with ARCHIVE_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    by_record = {row["record"]: row for row in rows}
    by_record.update({row["record"]: row for row in new_rows})
    with ARCHIVE_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_record[key] for key in sorted(by_record, key=str.lower))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true")
    group.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.create:
        result: Any = [create_record(spec).relative_to(REPO).as_posix() for spec in SPECS]
    else:
        result = [archive_parent(spec) for spec in SPECS]
        update_archive_manifest(result)
    print(json.dumps({"status": "PASS", "records": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
