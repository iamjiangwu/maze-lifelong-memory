"""Promote the 2026-08-11 all-figure refinement candidates.

The workflow is deliberately two phase. ``--create`` clones each current
scientific authority into a new append-only record and replaces only the
validated display assets.  After the current-record pointer documents have
been updated, ``--archive`` atomically moves the superseded records into the
archive and updates its tree ledger.

No quantitative table, statistical result, exclusion contract, or source
authority is recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
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
CANDIDATE_ROOT = REPO / "docs/figure_refinement_round2_2026-08-11"


@dataclass(frozen=True)
class Spec:
    key: str
    parent: str
    target: str
    candidate_dir: str
    candidate_stem: str
    root_stem: str
    changes: tuple[str, ...]
    panel_trees: tuple[tuple[str, str], ...] = ()
    panel_files: tuple[tuple[str, str], ...] = ()


SPECS = (
    Spec(
        "main1",
        "main_figure1_2026-08-10_round3_session_count_label",
        "main_figure1_2026-08-11_round4_display_refinement",
        "main_figures",
        "figure1_source_publication",
        "figure1_version_record",
        (
            "panel-d x-axis title changed to Time",
            "panel-d/g/h significance stars physically standardized",
            "panel-h First visit route score uses open form",
        ),
        (("figure1_individual_panels", "figure1_individual_panels"),),
    ),
    Spec(
        "main2",
        "main_figure2_2026-08-11_round3_open_terms_square_schematics",
        "main_figure2_2026-08-11_round4_schematic_refinement",
        "main_figures",
        "figure2_source_publication",
        "figure2_version_record",
        (
            "c/h/m highlighted schematic examples made modestly less similar",
            "panel letters e/j/o moved left to clear adjacent 500 tick labels",
        ),
    ),
    Spec(
        "main3",
        "main_figure3_2026-08-11_round3_upright_day_i",
        "main_figure3_2026-08-11_round4_experience_labels",
        "main_figures",
        "figure3_source_publication",
        "figure3_version_record",
        (
            "panel-a Day labels raised for title-to-square clearance",
            "visible experience counts standardized to No. of experiences",
        ),
        (("figure3_individual_panels", "individual_panels"),),
    ),
    Spec(
        "main4",
        "main_figure4_2026-08-11_round4_complete_upright_i_density_ticks",
        "main_figure4_2026-08-11_round5_bottom_labels",
        "main_figures/native_figure4_5",
        "figure4_source_publication",
        "figure4_version_record",
        ("panel-a bottom Day i labels raised into the canvas",),
        panel_files=(("figure4_individual_panels/figure4_panel_a_decoding_method_schematic", "individual_panels/figure4_panel_a_d1"),),
    ),
    Spec(
        "main5",
        "main_figure5_2026-08-08_full_audit",
        "main_figure5_2026-08-11_round2_normalized_rate",
        "main_figures/native_figure4_5",
        "figure5_source_publication",
        "figure5_version_record",
        ("panel-b colour-bar title expanded to Normalized rate",),
    ),
    Spec(
        "ed1",
        "extended_data_fig01_2026-08-11_round10_open_terms",
        "extended_data_fig01_2026-08-11_round11_column_alignment",
        "extended_data_1_5/ed01",
        "extended_data_figure1_source_rerender_round11",
        "ed01_version_record",
        (
            "panel-a time scale uses integer 16 and Time in maze",
            "rows b-k use the trajectory-row four-column skeleton",
            "panel-i endpoint language uses End nodes and End node visit",
        ),
    ),
    Spec(
        "ed2",
        "extended_data_fig02_2026-08-11_round20_upright_policy_clearance",
        "extended_data_fig02_2026-08-11_round21_layout_refinement",
        "extended_data_1_5/ed02",
        "ed02_round21_refinement_candidate",
        "ed02_version_record",
        (
            "panel-a preceding-turn annotation moved lower-right with measured clearance",
            "panel-g comparison labels raised by 0.5 and 1.5 character heights",
            "panel-e left spine shortened and f/i model titles lowered",
        ),
        (("panels", "panels"), ("shared_assets", "shared_assets")),
    ),
    Spec(
        "ed4",
        "extended_data_fig04_2026-08-08_full_audit",
        "extended_data_fig04_2026-08-11_round2_experience_layout",
        "extended_data_1_5/ed04",
        "ed04_registration_quality_assembly",
        "ed04_version_record",
        (
            "panels c-i shifted upward as complete rows",
            "panels d/e use No. of experiences and panel g removes (r)",
        ),
    ),
    Spec(
        "ed5",
        "extended_data_fig05_2026-08-08_round11_full_audit",
        "extended_data_fig05_2026-08-11_round12_normalized_rate",
        "extended_data_1_5/ed05",
        "ed05_example_place_maps_assembly_round11",
        "ed05_version_record",
        ("panel-a colour-bar title capitalized as Normalized rate",),
    ),
    Spec(
        "ed6",
        "extended_data_fig06_2026-08-11_round3_maze_tile_label",
        "extended_data_fig06_2026-08-11_round4_arrow_population_vector",
        "extended_data_6_10/ed06",
        "ed06_within_session_convergence_assembly_round4",
        "ed06_version_record",
        (
            "straight sliding-window arrows made one-directional",
            "panels e/f use Population vector correlation",
        ),
    ),
    Spec(
        "ed7",
        "extended_data_fig07_2026-08-11_round12_open_long_gap",
        "extended_data_fig07_2026-08-11_round13_compact_legends",
        "extended_data_6_10/ed07",
        "ed07_age_controls_six_panel_round13",
        "ed07_version_record",
        ("redundant Experience and Transition legend titles removed and canvas compacted",),
        panel_files=(
            ("ed07_panels_abc_first30_to_day2_maintained58_round13", "component_exports/ed07_panels_abc_first30_to_day2_maintained58_round13"),
            ("ed07_panels_def_long_gap_clean36_round13", "component_exports/ed07_panels_def_long_gap_clean36_round13"),
        ),
    ),
    Spec(
        "ed8",
        "extended_data_fig08_2026-08-11_round12_source_authority_closure",
        "extended_data_fig08_2026-08-11_round13_population_vector_label",
        "extended_data_6_10/ed08",
        "extended_data_figure8_refinement_round2",
        "ed08_version_record",
        ("panel-c y-axis title expanded to Population vector correlation",),
        (("individual_panels", "individual_panels"),),
    ),
    Spec(
        "ed9",
        "extended_data_fig09_2026-08-11_round22_complete_metadata_closure",
        "extended_data_fig09_2026-08-11_round23_clearance_experience_label",
        "extended_data_6_10/ed09",
        "extended_data_figure9_refinement_round2",
        "ed09_version_record",
        (
            "panel a shifted 50 px left for a-b label clearance",
            "panel f Vertical tick moved inside its panel",
            "panel l uses No. of experiences",
        ),
        (("individual_panels", "individual_panels"),),
    ),
    Spec(
        "ed10",
        "extended_data_fig10_2026-08-11_round9_complete_metadata_closure",
        "extended_data_fig10_2026-08-11_round10_upright_policy_labels",
        "extended_data_6_10/ed10",
        "extended_data_figure10_source_rerender_refinement_round2",
        "ed10_version_record",
        ("panel-c policy P symbols and subscripts rendered upright",),
        (("assembly/refinement_round2/panels", "individual_panels"),),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_pdf(pdf: Path, png: Path, dpi: int, tiff: Path | None = None) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page PDF: {pdf}")
        pix = document[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png, dpi=(dpi, dpi), optimize=True)
    if tiff is not None:
        image.save(tiff, dpi=(dpi, dpi), compression="tiff_lzw")


def write_hashes(record: Path) -> None:
    target = record / "RECORD_HASHES.csv"
    target.unlink(missing_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "sha256", "bytes"))
        for path in sorted(record.rglob("*"), key=lambda item: item.as_posix().lower()):
            if path.is_file() and path.name != target.name:
                writer.writerow(
                    (path.relative_to(record).as_posix(), sha256(path), path.stat().st_size)
                )


def verify_hashes(record: Path) -> list[str]:
    inventory = record / "RECORD_HASHES.csv"
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
        digest.update(f"{relative}\t{size}\t{sha256(path)}\n".encode("utf-8"))
    return {"files": len(files), "bytes": total, "sha256": digest.hexdigest()}


def scientific_directory_digests(record: Path) -> dict[str, dict[str, Any]]:
    excluded = {
        "individual_panels",
        "panels",
        "component_exports",
        "shared_assets",
        "source_code",
        "source_validation",
        "review_evidence",
        "powerpoint",
        "powerpoint_review",
        "ppt_vector",
    }
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(record.iterdir(), key=lambda item: item.name.lower()):
        if (
            path.is_dir()
            and path.name not in excluded
            and not path.name.endswith("individual_panels")
        ):
            result[path.name] = tree_digest(path)
    return result


def copy_panel_tree(candidate: Path, source_rel: str, stage: Path, dest_rel: str) -> None:
    source = candidate / source_rel
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = stage / dest_rel
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def copy_panel_stem(candidate: Path, source_stem: str, stage: Path, dest_stem: str) -> None:
    matched = False
    for extension in (".svg", ".pdf", ".png", ".tiff"):
        source = candidate / f"{source_stem}{extension}"
        if source.is_file():
            destination = stage / f"{dest_stem}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            matched = True
    if not matched:
        raise FileNotFoundError(candidate / source_stem)


def replace_record_name(payload: Any, parent: str, target: str) -> Any:
    if isinstance(payload, dict):
        return {key: replace_record_name(value, parent, target) for key, value in payload.items()}
    if isinstance(payload, list):
        return [replace_record_name(value, parent, target) for value in payload]
    if isinstance(payload, str):
        return payload.replace(parent, target)
    return payload


def refresh_record_identity(stage: Path, parent: str, target: str) -> None:
    for path in stage.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        payload = replace_record_name(payload, parent, target)
        if isinstance(payload, dict) and payload.get("record") == parent:
            payload["record"] = target
        write_json(path, payload)


def refresh_current_manifest_fields(stage: Path, spec: Spec) -> None:
    """Keep current (non-history) record manifests consistent with new exports."""

    revision_match = re.search(r"round(\d+)", spec.target)
    revision = int(revision_match.group(1)) if revision_match else None
    root_hashes = {
        "svg": sha256(stage / f"{spec.root_stem}.svg"),
        "pdf": sha256(stage / f"{spec.root_stem}.pdf"),
        "png_300dpi": sha256(stage / f"{spec.root_stem}_300dpi.png"),
        "png_600dpi": sha256(stage / f"{spec.root_stem}_600dpi.png"),
        "tiff_600dpi": sha256(stage / f"{spec.root_stem}_600dpi.tiff"),
        "tiff_600dpi_lzw": sha256(stage / f"{spec.root_stem}_600dpi.tiff"),
    }
    for path in stage.glob("*VERSION_RECORD_MANIFEST.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = replace_record_name(payload, spec.parent, spec.target)
        if isinstance(payload, dict):
            payload["record"] = spec.target
            if revision is not None and "record_revision" in payload:
                payload["record_revision"] = revision
            for key in ("root_export_sha256", "root_output_sha256"):
                if isinstance(payload.get(key), dict):
                    payload[key] = {
                        name: root_hashes.get(name, value)
                        for name, value in payload[key].items()
                    }
            payload["current_display_refinement"] = {
                "manifest": "ROUND2_DISPLAY_REFINEMENT_MANIFEST.json",
                "changes": list(spec.changes),
            }
        write_json(path, payload)


def copy_current_candidate_metadata(candidate: Path, stage: Path, spec: Spec) -> None:
    mappings: dict[str, tuple[str, ...]] = {
        "ed1": (
            "MANIFEST_ROUND11.json>ED01_ASSEMBLY_MANIFEST.json",
            "RENDER_VALIDATION_ROUND11.json>ROUND11_RENDER_VALIDATION.json",
            "LAYOUT_AUDIT_ROUND11.csv>LAYOUT_AUDIT.csv",
            "SOURCE_DATA_AUDIT_ROUND11.csv>SOURCE_DATA_AUDIT.csv",
        ),
        "ed2": ("ED02_ROUND21_COMPOSITE_MANIFEST.json>ED02_ROUND21_COMPOSITE_MANIFEST.json",),
    }
    for mapping in mappings.get(spec.key, ()):
        source_name, destination_name = mapping.split(">", 1)
        source = candidate / source_name
        if source.is_file():
            shutil.copy2(source, stage / destination_name)
    # Record-localize any working-candidate paths in the promoted current metadata.
    for path in stage.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        text = json.dumps(payload)
        text = text.replace(
            (candidate / spec.candidate_stem).relative_to(REPO).as_posix(),
            spec.root_stem,
        )
        text = text.replace(spec.candidate_stem, spec.root_stem)
        write_json(path, json.loads(text))


def group_evidence(candidate: Path, stage: Path) -> None:
    evidence_dir = stage / "source_validation" / "refinement_round2"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for base in (candidate, candidate.parent):
        for name in ("VALIDATION.json", "CANDIDATE_MANIFEST.json"):
            source = base / name
            if source.is_file():
                shutil.copy2(source, evidence_dir / f"{base.name}_{name}")
    for source in candidate.iterdir():
        if (
            source.is_file()
            and source.suffix.lower() in {".json", ".csv", ".md"}
            and source.stat().st_size <= 10 * 1024 * 1024
            and source.name != "incremental_candidate_build.json"
        ):
            shutil.copy2(source, evidence_dir / source.name)


def copy_renderer_evidence(stage: Path) -> None:
    maintained = (
        "scripts/preview_figure1_assembly.py",
        "scripts/render_main_figs_v3_publication.py",
        "scripts/preview_figure3_assembly.py",
        "scripts/render_extended_data_figure1_round4.py",
        "scripts/render_extended_data_figure1_refinement_round11.py",
        "scripts/render_ed02_slide3_refinement_round21.py",
        "scripts/export_ed02_round21_refinement_figure.py",
        "scripts/render_ed02_slide3_upright_policy_round18.py",
        "scripts/render_ed04_registration_quality_assembly.py",
        "scripts/render_ed05_example_place_maps_assembly.py",
        "scripts/render_ed06_within_session_convergence_assembly.py",
        "scripts/render_ed07_slide9_powerpoint_matched.py",
        "scripts/render_ed10_slide12_powerpoint_matched.py",
        "docs/figure_refinement_round2_2026-08-11/extended_data_6_10/render_ed08_refinement_round2.py",
        "docs/figure_refinement_round2_2026-08-11/extended_data_6_10/render_ed09_refinement_round2.py",
        "scripts/build_main_figure_refinement_round2_native_panel_candidates.py",
        "scripts/promote_all_figure_refinement_round2_records.py",
        "scripts/validate_all_figure_refinement_round2_records.py",
        "scripts/export_current_version_records_to_desktop.py",
    )
    destination = stage / "source_code"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in maintained:
        source = REPO / relative
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def root_asset_payload(stage: Path, stem: str) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for suffix in (".svg", ".pdf", "_300dpi.png", "_600dpi.png", "_600dpi.tiff"):
        path = stage / f"{stem}{suffix}"
        payload[suffix] = {
            "path": path.relative_to(stage).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return payload


def write_extended_data_caption_notes(stage: Path, spec: Spec, inherited: str) -> None:
    inherited_body = re.sub(r"^#\s+Caption notes\s*", "", inherited, flags=re.IGNORECASE).strip()
    for old, new in (
        ("maze-tile", "maze tile"),
        ("long-bout", "long bout"),
        ("first-long bout", "first long bout"),
        ("long-gap", "long gap"),
    ):
        inherited_body = inherited_body.replace(old, new).replace(old.title(), new.title())
    bullets = "\n".join(f"- {change}" for change in spec.changes)
    text = (
        "# Caption notes\n\n"
        "The canonical manuscript caption was reviewed against this record. The current "
        "change is display-only; panel estimands, sampling units, statistics, nulls, "
        "multiplicity handling, eligibility and exclusions are unchanged from the parent "
        "scientific authority.\n\n"
        "## Current display refinements\n\n"
        f"{bullets}\n"
    )
    if inherited_body:
        text += (
            "\n## Inherited scientific caption notes\n\n"
            f"The following scientific notes are retained from `{spec.parent}`; any round "
            "references in this subsection identify that parent provenance.\n\n"
            f"{inherited_body}\n"
        )
    (stage / "CAPTION_NOTES.md").write_text(text, encoding="utf-8")


def create_record(spec: Spec) -> Path:
    parent = RECORDS / spec.parent
    target = RECORDS / spec.target
    candidate = CANDIDATE_ROOT / spec.candidate_dir
    if target.exists():
        errors = verify_hashes(target)
        if errors:
            raise RuntimeError(f"Existing target is not closed: {target}; {errors[:5]}")
        return target
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if not candidate.is_dir():
        raise FileNotFoundError(candidate)
    parent_hash_errors = verify_hashes(parent)
    if parent_hash_errors:
        raise RuntimeError(f"Parent hash closure failed: {parent}; {parent_hash_errors[:5]}")
    before_science = scientific_directory_digests(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".refine-{spec.key}-", dir=RECORDS))
    try:
        shutil.copytree(parent, stage, dirs_exist_ok=True)
        inherited_caption_notes = (
            (stage / "CAPTION_NOTES.md").read_text(encoding="utf-8")
            if (stage / "CAPTION_NOTES.md").is_file()
            else ""
        )
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"):
            (stage / stale).unlink(missing_ok=True)
        source_svg = candidate / f"{spec.candidate_stem}.svg"
        source_pdf = candidate / f"{spec.candidate_stem}.pdf"
        if not source_svg.is_file() or not source_pdf.is_file():
            raise FileNotFoundError(f"Candidate root missing for {spec.key}: {source_svg}, {source_pdf}")
        shutil.copy2(source_svg, stage / f"{spec.root_stem}.svg")
        shutil.copy2(source_pdf, stage / f"{spec.root_stem}.pdf")
        source_html = candidate / f"{spec.candidate_stem}.html"
        target_html = stage / f"{spec.root_stem}.html"
        if source_html.is_file():
            shutil.copy2(source_html, target_html)
        elif target_html.exists():
            target_html.unlink()
        render_pdf(
            stage / f"{spec.root_stem}.pdf",
            stage / f"{spec.root_stem}_600dpi.png",
            600,
            stage / f"{spec.root_stem}_600dpi.tiff",
        )
        render_pdf(
            stage / f"{spec.root_stem}.pdf",
            stage / f"{spec.root_stem}_300dpi.png",
            300,
        )
        for source_rel, dest_rel in spec.panel_trees:
            copy_panel_tree(candidate, source_rel, stage, dest_rel)
        for source_stem, dest_stem in spec.panel_files:
            copy_panel_stem(candidate, source_stem, stage, dest_stem)
        copy_current_candidate_metadata(candidate, stage, spec)
        group_evidence(candidate, stage)
        copy_renderer_evidence(stage)
        refresh_record_identity(stage, spec.parent, spec.target)
        after_science = scientific_directory_digests(stage)
        if before_science != after_science:
            different = {
                key: {"parent": before_science.get(key), "stage": after_science.get(key)}
                for key in sorted(set(before_science) | set(after_science))
                if before_science.get(key) != after_science.get(key)
            }
            raise RuntimeError(
                f"Scientific-directory parity failed for {spec.key}: "
                f"{json.dumps(different, sort_keys=True)}"
            )
        timestamp = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema": "all-figure-refinement-round2-version-record/v1",
            "status": "PASS",
            "record": spec.target,
            "parent_record": spec.parent,
            "candidate": candidate.relative_to(REPO).as_posix(),
            "changes": list(spec.changes),
            "scientific_directory_parity": after_science,
            "root_exports": root_asset_payload(stage, spec.root_stem),
            "created_utc": timestamp,
        }
        write_json(stage / "ROUND2_DISPLAY_REFINEMENT_MANIFEST.json", manifest)
        refresh_current_manifest_fields(stage, spec)
        (stage / "MANIFEST.md").write_text(
            f"# {spec.target}\n\n"
            f"This append-only record supersedes `{spec.parent}` and freezes the validated "
            "2026-08-11 display refinement. Quantitative tables, statistical inference, "
            "cohort eligibility, exclusions and source authorities are byte-identical to "
            "the parent record. See `ROUND2_DISPLAY_REFINEMENT_MANIFEST.json` for the exact "
            "display changes, root-export hashes and scientific-directory parity.\n",
            encoding="utf-8",
        )
        if spec.key.startswith("ed"):
            write_extended_data_caption_notes(stage, spec, inherited_caption_notes)
        (stage / "REBUILD_README.md").write_text(
            f"# Rebuild {spec.target}\n\n"
            "Render through the maintained Python sources frozen in `source_code`, validate "
            "the candidate evidence in `source_validation`, then run "
            "`python scripts/promote_all_figure_refinement_round2_records.py --create`. "
            "Promotion must preserve scientific-directory digests and close the complete "
            "record hash ledger.\n",
            encoding="utf-8",
        )
        checks = {
            "candidate_svg_and_pdf_exist": True,
            "single_page_pdf": fitz.open(stage / f"{spec.root_stem}.pdf").page_count == 1,
            "scientific_directories_byte_exact_parent": before_science == after_science,
            "root_svg_nonblank": (stage / f"{spec.root_stem}.svg").stat().st_size > 1000,
            "root_600dpi_png_exists": (stage / f"{spec.root_stem}_600dpi.png").is_file(),
            "root_600dpi_tiff_exists": (stage / f"{spec.root_stem}_600dpi.tiff").is_file(),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Promotion checks failed for {spec.key}: {checks}")
        write_json(
            stage / "RECORD_VALIDATION.json",
            {
                "schema": "all-figure-refinement-round2-record-validation/v1",
                "status": "PASS",
                "record": spec.target,
                "checks": checks,
                "validated_utc": timestamp,
            },
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"PASS\nrecord={spec.target}\ncreated_utc={timestamp}\n",
            encoding="utf-8",
        )
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Record hash closure failed for {spec.key}: {errors[:10]}")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def archive_parent(spec: Spec) -> dict[str, Any]:
    parent = RECORDS / spec.parent
    target = RECORDS / spec.target
    destination = ARCHIVE / spec.parent
    if not target.is_dir() or verify_hashes(target):
        raise RuntimeError(f"Closed successor required before archive: {target}")
    if parent.exists() and destination.exists():
        raise RuntimeError(f"Parent exists in both active and archive roots: {spec.parent}")
    if parent.exists():
        errors = verify_hashes(parent)
        if errors:
            raise RuntimeError(f"Parent hash closure failed before archive: {spec.parent}; {errors[:5]}")
        os.replace(parent, destination)
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    archived = Path("\\\\?\\" + str(destination.resolve()))
    errors = verify_hashes(archived)
    if errors:
        raise RuntimeError(f"Archived parent hash closure failed: {spec.parent}; {errors[:5]}")
    stats = tree_digest(archived)
    return {
        "record": spec.parent,
        "original_path": f"docs/figure_version_records/{spec.parent}",
        "archive_path": f"docs/figure_version_records_archive_2026-08-09/{spec.parent}",
        "file_count": str(stats["files"]),
        "bytes": str(stats["bytes"]),
        "tree_sha256": stats["sha256"],
        "move_status": "verified",
    }


def update_archive_manifest(new_rows: list[dict[str, Any]]) -> None:
    with ARCHIVE_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    by_record = {row["record"]: row for row in rows}
    for row in new_rows:
        by_record[row["record"]] = row
    ordered = [by_record[key] for key in sorted(by_record, key=str.lower)]
    with ARCHIVE_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def selected_specs(keys: list[str] | None) -> list[Spec]:
    if not keys:
        return list(SPECS)
    wanted = set(keys)
    return [spec for spec in SPECS if spec.key in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--archive", action="store_true")
    parser.add_argument("--key", action="append", choices=[spec.key for spec in SPECS])
    args = parser.parse_args()
    specs = selected_specs(args.key)
    if args.create:
        paths = [create_record(spec) for spec in specs]
        payload: Any = [path.relative_to(REPO).as_posix() for path in paths]
    else:
        rows = [archive_parent(spec) for spec in specs]
        update_archive_manifest(rows)
        payload = rows
    print(json.dumps({"status": "PASS", "records": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
