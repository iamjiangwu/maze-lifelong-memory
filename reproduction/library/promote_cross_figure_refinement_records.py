"""Promote the 2026-08-11 cross-figure display refinements.

The promotion is deliberately append-only.  Each new record is cloned from
the current immutable scientific authority, then only the requested display
exports, renderer/audit evidence, and record metadata are replaced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import fitz
from PIL import Image, ImageStat


REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / "docs/figure_version_records"

MAIN_CANDIDATE = (
    REPO
    / "docs/main_figs_v3_source_render_2026-06-24/"
    "cross_figure_refinement_round_2026-08-10"
)
MAIN4_CANDIDATE_ROOT = (
    REPO
    / "docs/main_fig4_d1_d2_source_comparison_2026-08-04/"
    "cross_figure_refinement_round_2026-08-10"
)
MAIN4_CANDIDATE = MAIN4_CANDIDATE_ROOT / "d1_to_later_d1"
ED2_CANDIDATE = REPO / "tmp/ed02_round17_contrast_labels_2026-08-10"
ED6_CANDIDATE = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed06_within_session_convergence/open_terms_round2_2026-08-10"
)
ED8_ROOT = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed08_reward_proximal_controls"
)
ED8_CANDIDATE = ED8_ROOT / "revision_round9_2026-08-10"
ED9_ROOT = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed09_tree_depth_level_controls"
)
ED9_CORRECTION = ED9_ROOT / "correction_2026-08-09"
ED9_CANDIDATE = ED9_CORRECTION / "open_terms_levels_round20_2026-08-10"
ED10_ROOT = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed10_reward_level_behavior_bridge"
)
ED10_CANDIDATE = ED10_ROOT / "assembly/round7"


RECORD_SPECS = {
    "main_figure_2": (
        "main_figure2_2026-08-10_round2_panel_q_no_zero_line",
        "main_figure2_2026-08-11_round3_open_terms_square_schematics",
    ),
    "main_figure_3": (
        "main_figure3_2026-08-08_full_audit",
        "main_figure3_2026-08-11_round2_timing_schematic_open_terms",
    ),
    "main_figure_4": (
        "main_figure4_2026-08-08_full_audit",
        "main_figure4_2026-08-11_round2_place_maps_label",
    ),
    "extended_data_2": (
        "extended_data_fig02_2026-08-10_round16_publication",
        "extended_data_fig02_2026-08-11_round17_contrast_labels",
    ),
    "extended_data_6": (
        "extended_data_fig06_2026-08-08_full_audit",
        "extended_data_fig06_2026-08-11_round2_open_terms",
    ),
    "extended_data_8": (
        "extended_data_fig08_2026-08-09_round8_cohort_map_fixed_label",
        "extended_data_fig08_2026-08-11_round9_open_terms",
    ),
    "extended_data_9": (
        "extended_data_fig09_2026-08-10_round19_c_corrected36",
        "extended_data_fig09_2026-08-11_round20_open_terms_levels",
    ),
    "extended_data_10": (
        "extended_data_fig10_2026-08-10_round6_publication",
        "extended_data_fig10_2026-08-11_round7_raw_decoder_error",
    ),
}


FORBIDDEN_VISIBLE_TERMS = (
    "Place-map",
    "Level-map",
    "Population-vector",
    "Ensemble-rate",
    "Whole-maze",
    "Reward-approach",
    "Reward-decoder",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def render_pdf(pdf: Path, png: Path, *, dpi: int, tiff: Path | None = None) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected a one-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    png.parent.mkdir(parents=True, exist_ok=True)
    image.save(png, dpi=(dpi, dpi), optimize=True)
    if tiff is not None:
        image.save(tiff, dpi=(dpi, dpi), compression="tiff_lzw")


def write_hashes(record: Path) -> None:
    target = record / "RECORD_HASHES.csv"
    target.unlink(missing_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "sha256", "bytes"])
        for path in sorted(record.rglob("*")):
            if path.is_file() and path.name != target.name:
                writer.writerow(
                    [path.relative_to(record).as_posix(), sha256(path), path.stat().st_size]
                )


def verify_hashes(record: Path) -> tuple[bool, list[str]]:
    with (record / "RECORD_HASHES.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors.extend(f"missing:{name}" for name in sorted(set(expected) - set(actual)))
    for relative in sorted(set(actual) & set(expected)):
        path = actual[relative]
        row = expected[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{relative}")
    return not errors, errors


def update_json(path: Path, updates: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_manifest(stage: Path, title: str, body: str) -> None:
    manifest = stage / "MANIFEST.md"
    prior = manifest.read_text(encoding="utf-8").rstrip()
    manifest.write_text(f"{prior}\n\n## {title}\n\n{body.strip()}\n", encoding="utf-8")


def copy_source_code(stage: Path, *sources: Path) -> None:
    for source in sources:
        copy(source, stage / "source_code" / source.name)


def promote_main2(stage: Path, target_name: str) -> dict[str, Any]:
    for suffix in ("svg", "pdf", "html"):
        copy(
            MAIN_CANDIDATE / f"figure2_source_publication.{suffix}",
            stage / f"figure2_version_record.{suffix}",
        )
    render_pdf(
        stage / "figure2_version_record.pdf",
        stage / "figure2_version_record_300dpi.png",
        dpi=300,
    )
    render_pdf(
        stage / "figure2_version_record.pdf",
        stage / "figure2_version_record_600dpi.png",
        dpi=600,
        tiff=stage / "figure2_version_record_600dpi.tiff",
    )
    for name in (
        "figure2_source_manifest.json",
        "figure2_panel_geometry.csv",
        "figure2_panel_verification.csv",
        "figure2_consecutive_gap_binned_curve_audit.csv",
        "figure2_consecutive_gap_displayed_estimand_bh4.csv",
        "run_manifest.json",
    ):
        destination = stage / name if name != "run_manifest.json" else stage / "source_validation/cross_figure_run_manifest.json"
        copy(MAIN_CANDIDATE / name, destination)
    update_json(
        stage / "figure2_source_manifest.json",
        {
            "record": target_name,
            "status": "accepted_immutable_version_record",
            "supersedes": RECORD_SPECS["main_figure_2"][0],
        },
    )
    copy_source_code(
        stage,
        REPO / "scripts/render_main_figs_v3_publication.py",
        REPO / "tests/test_figure2_schematic_layout.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": [
            "panels a and b translated down exactly 7 px at 300 dpi",
            "physical cells in schematics c, h and m made square",
            "visible representation names changed to open compounds",
        ],
        "preserved": ["panel q grey zero line remains absent", "all scientific tables"],
    }


def promote_main3(stage: Path, target_name: str) -> dict[str, Any]:
    for suffix in ("svg", "pdf", "html"):
        copy(
            MAIN_CANDIDATE / f"figure3_source_publication.{suffix}",
            stage / f"figure3_version_record.{suffix}",
        )
    render_pdf(stage / "figure3_version_record.pdf", stage / "figure3_version_record_300dpi.png", dpi=300)
    render_pdf(
        stage / "figure3_version_record.pdf",
        stage / "figure3_version_record_600dpi.png",
        dpi=600,
        tiff=stage / "figure3_version_record_600dpi.tiff",
    )
    for name in (
        "figure3_source_manifest.json",
        "figure3_axes_geometry.csv",
        "figure3_panel_verification.csv",
        "figure3_day2_template_first30_alignment.csv",
        "figure3_day2_template_support_runs.csv",
    ):
        copy(MAIN_CANDIDATE / name, stage / name)
    source_panels = MAIN_CANDIDATE / "figure3_individual_panels"
    shutil.copytree(source_panels, stage / "individual_panels", dirs_exist_ok=True)
    update_json(
        stage / "figure3_source_manifest.json",
        {
            "record": target_name,
            "status": "accepted_immutable_version_record",
            "supersedes": RECORD_SPECS["main_figure_3"][0],
        },
    )
    copy_source_code(
        stage,
        REPO / "scripts/render_main_figs_v3_publication.py",
        REPO / "tests/test_figure3_schematic_row_layout.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": [
            "Day i / Day i+1 timing schematic and elapsed-time arrow",
            "Experience labels changed to # of experiences",
            "visible representation names changed to open compounds",
        ],
        "preserved": ["scientific tables", "isotropic panel assembly"],
    }


def promote_main4(stage: Path, target_name: str) -> dict[str, Any]:
    masters = {
        "svg": "figure4_version_record.svg",
        "pdf": "figure4_version_record.pdf",
        "png": "figure4_version_record_600dpi.png",
        "tiff": "figure4_version_record_600dpi.tiff",
    }
    for suffix, destination in masters.items():
        copy(MAIN4_CANDIDATE / f"figure4_d1_source_to_later_d1.{suffix}", stage / destination)
    shutil.copytree(MAIN4_CANDIDATE / "individual_panels", stage / "individual_panels", dirs_exist_ok=True)
    copy(
        MAIN4_CANDIDATE / "figure4_panel_c_d1_source_rows.csv",
        stage / "plot_ready/figure4_panel_c_d1_source_rows.csv",
    )
    copy(MAIN4_CANDIDATE_ROOT / "RENDER_VALIDATION.json", stage / "source_validation/CROSS_FIGURE_RENDER_VALIDATION.json")
    shutil.copytree(
        MAIN_CANDIDATE / "figure4_panel_a_refinement",
        stage / "source_validation/figure4_panel_a_refinement",
        dirs_exist_ok=True,
    )
    source_manifest = stage / "figure4_source_manifest.json"
    update_json(
        source_manifest,
        {
            "record": target_name,
            "status": "accepted_immutable_version_record",
            "supersedes": RECORD_SPECS["main_figure_4"][0],
            "panel_a_display_label": "Day i place maps",
        },
    )
    copy_source_code(
        stage,
        REPO / "scripts/render_main_figs_v3_publication.py",
        REPO / "scripts/render_main_figure4_panel_a_refinement.py",
        REPO / "scripts/render_figure4_day_source_comparison.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": ["panel a label changed from template to place maps"],
        "preserved": ["panels b-f pixel-identical at 600 dpi", "all scientific tables"],
    }


def promote_ed2(stage: Path, target_name: str) -> dict[str, Any]:
    composite = ED2_CANDIDATE / "composite"
    base = "ed02_round13_panel_g_labels_version_record"
    for suffix, destination in {
        "svg": "ed02_version_record.svg",
        "pdf": "ed02_version_record.pdf",
        "png": "ed02_version_record_600dpi.png",
        "tiff": "ed02_version_record_600dpi.tiff",
    }.items():
        copy(composite / f"{base}{'_600dpi' if suffix in {'png', 'tiff'} else ''}.{suffix}", stage / destination)
    render_pdf(stage / "ed02_version_record.pdf", stage / "ed02_version_record_300dpi.png", dpi=300)
    copy(composite / f"{base}_preview_1440.png", stage / "ed02_version_record_preview_1440.png")
    for source in (ED2_CANDIDATE / "panels/panels").glob("ed02g_depth_pbs_profile_round11.*"):
        copy(source, stage / "panels" / source.name)
    copy(
        composite / "ED02_ROUND13_COMPOSITE_MANIFEST.json",
        stage / "source_validation/ED02_ROUND17_COMPOSITE_MANIFEST.json",
    )
    for name in ("RENDER_RESULT.json", "SOURCE_MANIFEST.json", "PANEL_MANIFEST.csv"):
        copy(ED2_CANDIDATE / "panels" / name, stage / "source_validation" / name)
    update_json(
        stage / "ED02_VERSION_RECORD_MANIFEST.json",
        {
            "record": target_name,
            "source_round": 17,
            "supersedes": RECORD_SPECS["extended_data_2"][0],
            "round17_display_change": "panel g contrast labels raised exactly 14 px at 600 dpi",
        },
    )
    copy_source_code(
        stage,
        REPO / "scripts/render_ed02_slide3_contrast_labels_round17.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": ["panel g lower contrast labels raised exactly 14 px"],
        "preserved": ["panels a-f, h and i", "panel h observed line/text color #27313A", "all tables"],
    }


def promote_ed6(stage: Path, target_name: str) -> dict[str, Any]:
    stem = "ed06_within_session_convergence_assembly_round2"
    for suffix, destination in {
        "svg": "ed06_version_record.svg",
        "pdf": "ed06_version_record.pdf",
        "png": "ed06_version_record_600dpi.png",
        "tiff": "ed06_version_record_600dpi.tiff",
    }.items():
        copy(ED6_CANDIDATE / f"{stem}.{suffix}", stage / destination)
    for source_name, destination in {
        "ED06_ASSEMBLY_MANIFEST_ROUND2.json": "ED06_ASSEMBLY_MANIFEST.json",
        "ED06_DISPLAYED_STATISTICS_ROUND2.csv": "ED06_DISPLAYED_STATISTICS.csv",
        "ED06_VALIDATION_ROUND2.csv": "ED06_VALIDATION.csv",
        "CAPTION_NOTES_ROUND2.md": "CAPTION_NOTES.md",
    }.items():
        copy(ED6_CANDIDATE / source_name, stage / destination)
    copy_source_code(
        stage,
        REPO / "scripts/render_ed06_within_session_convergence_assembly.py",
        Path(__file__).resolve(),
    )
    return {
        "record": target_name,
        "authorized_changes": ["visible place map and ensemble rate terms changed to open compounds"],
        "preserved": ["layout", "statistics", "source tables"],
    }


def promote_ed8(stage: Path, target_name: str) -> dict[str, Any]:
    for suffix, destination in {
        "svg": "ed08_version_record.svg",
        "pdf": "ed08_version_record.pdf",
        "png": "ed08_version_record_300dpi.png",
        "tiff": "ed08_version_record_600dpi.tiff",
    }.items():
        copy(ED8_ROOT / f"extended_data_figure8_source_rerender_round9.{suffix}", stage / destination)
    with Image.open(stage / "ed08_version_record_600dpi.tiff") as image:
        image.convert("RGB").save(stage / "ed08_version_record_600dpi.png", dpi=(600, 600), optimize=True)
    candidate_panels = ED8_CANDIDATE / "final_render/individual_panels"
    for letter in ("e", "f", "h", "i"):
        for source in candidate_panels.glob(f"ed08_panel_{letter}_*.*"):
            copy(source, stage / "individual_panels" / source.name)
    copy(ED8_CANDIDATE / "ED08_ROUND9_RENDER_MANIFEST.json", stage / "source_validation/ED08_ROUND9_RENDER_MANIFEST.json")
    update_json(
        stage / "ED08_VERSION_RECORD_MANIFEST.json",
        {
            "record": target_name,
            "source_round": 9,
            "supersedes": RECORD_SPECS["extended_data_8"][0],
            "round9_display_change": "open compound terms in panels e, f, h and i only",
        },
    )
    copy_source_code(stage, ED8_CANDIDATE / "render_ed08_round9.py", Path(__file__).resolve())
    return {
        "authorized_changes": ["open compound labels in panels e, f, h and i"],
        "preserved": ["panels a-d and g raster-exact", "geometry", "statistics", "source tables"],
    }


def promote_ed9(stage: Path, target_name: str) -> dict[str, Any]:
    source_stem = "extended_data_figure9_open_terms_levels_round20"
    for source_suffix, destination in {
        "svg": "ed09_version_record.svg",
        "pdf": "ed09_version_record.pdf",
        "_300dpi.png": "ed09_version_record_300dpi.png",
        "_600dpi.png": "ed09_version_record_600dpi.png",
        "_600dpi.tiff": "ed09_version_record_600dpi.tiff",
    }.items():
        copy(ED9_CANDIDATE / f"{source_stem}{source_suffix}", stage / destination)
    shutil.copytree(ED9_CANDIDATE / "rendered_panels", stage / "individual_panels", dirs_exist_ok=True)
    copy(ED9_CANDIDATE / "ROUND20_DISPLAY_AUDIT.json", stage / "source_validation/ROUND20_DISPLAY_AUDIT.json")
    manifest_path = stage / "ED09_VERSION_RECORD_MANIFEST.json"
    update_json(
        manifest_path,
        {
            "schema": "ed09-round20-version-record/v1",
            "record": target_name,
            "source_round": 20,
            "supersedes": RECORD_SPECS["extended_data_9"][0],
            "round20_display_changes": {
                "panel_a_axes": "raw tree-depth levels 0-6; matrix values and order unchanged",
                "visible_terms": "open compounds",
                "scientific_values_changed": False,
            },
            "round20_validation": "source_validation/ROUND20_DISPLAY_AUDIT.json",
        },
    )
    copy_source_code(
        stage,
        ED9_CORRECTION / "render_ed09_open_terms_levels_round20.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": ["panel a axes relabelled 0-6", "visible representation terms changed to open compounds"],
        "preserved": ["corrected36 panel c", "all plot-ready/statistics tables", "exclusion contract"],
    }


def promote_ed10(stage: Path, target_name: str) -> dict[str, Any]:
    stem = "extended_data_figure10_source_rerender_round7"
    for suffix, destination in {
        "svg": "ed10_version_record.svg",
        "pdf": "ed10_version_record.pdf",
        "png": "ed10_version_record_600dpi.png",
        "tiff": "ed10_version_record_600dpi.tiff",
    }.items():
        copy(ED10_CANDIDATE / f"{stem}.{suffix}", stage / destination)
    render_pdf(stage / "ed10_version_record.pdf", stage / "ed10_version_record_300dpi.png", dpi=300)
    shutil.copytree(ED10_CANDIDATE / "panels", stage / "individual_panels", dirs_exist_ok=True)
    for name in ("ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv", "ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json"):
        copy(ED10_ROOT / "source_data" / name, stage / "plot_ready" / name)
        copy(ED10_ROOT / "source_data" / name, stage / "source_validation" / name)
    manifest_path = stage / "ED10_VERSION_RECORD_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema": "ed10-round7-version-record/v1",
            "record": target_name,
            "source_round": 7,
            "supersedes": RECORD_SPECS["extended_data_10"][0],
            "decoder_display_convention": {
                "rows": ["Reward decoding error", "Whole maze decoding error"],
                "higher_means": "poorer decoding",
                "source": "plot_ready/ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv",
                "sign_audit": "source_validation/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json",
            },
        }
    )
    manifest["display_contract"].update(
        {
            "decoder_rows": "raw error; higher is poorer",
            "compound_terms": "open",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copy_source_code(
        stage,
        REPO / "scripts/render_ed10_slide12_powerpoint_matched.py",
        Path(__file__).resolve(),
    )
    return {
        "authorized_changes": [
            "panel a first two rows now display raw decoder error rather than negative-error strength",
            "visible compound terms changed to open forms",
        ],
        "preserved": ["other 12 matrix cells", "panel b commonality", "cohort 30 pairs / 14 mice / K=166", "panel c"],
    }


PROMOTERS: dict[str, Callable[[Path, str], dict[str, Any]]] = {
    "main_figure_2": promote_main2,
    "main_figure_3": promote_main3,
    "main_figure_4": promote_main4,
    "extended_data_2": promote_ed2,
    "extended_data_6": promote_ed6,
    "extended_data_8": promote_ed8,
    "extended_data_9": promote_ed9,
    "extended_data_10": promote_ed10,
}


def master_files(record: Path) -> tuple[Path, Path, Path, Path]:
    svg = next(record.glob("*_version_record.svg"))
    pdf = next(record.glob("*_version_record.pdf"))
    png = next(record.glob("*_version_record_600dpi.png"))
    tiff = next(record.glob("*_version_record_600dpi.tiff"))
    return svg, pdf, png, tiff


def basic_record_checks(record: Path, figure: str) -> dict[str, bool]:
    svg, pdf, png, tiff = master_files(record)
    svg_text = svg.read_text(encoding="utf-8")
    with fitz.open(pdf) as document:
        pdf_one_page = document.page_count == 1
    with Image.open(png) as image:
        png_nonblank = max(ImageStat.Stat(image.convert("RGB")).stddev) > 1.0
        png_width = image.width
    with Image.open(tiff) as image:
        tiff_nonblank = max(ImageStat.Stat(image.convert("RGB")).stddev) > 1.0
        tiff_dpi = image.info.get("dpi", (0.0, 0.0))
    checks = {
        "svg_nonblank": svg.stat().st_size > 1000,
        "pdf_one_page": pdf_one_page,
        "png_600dpi_nonblank": png_nonblank and png_width >= 3000,
        "tiff_600dpi_nonblank": tiff_nonblank and min(tiff_dpi) >= 590,
        "no_forbidden_visible_terms": not any(term in svg_text for term in FORBIDDEN_VISIBLE_TERMS),
    }
    if figure == "main_figure_2":
        # Matplotlib uses ``line2d_1`` for an ordinary tick in this export.
        # The removed reference artist is instead uniquely identified by its
        # accepted grey stroke.
        checks["panel_q_zero_line_absent"] = "#c7c7c7" not in svg_text.lower()
    if figure == "extended_data_9":
        audit = json.loads((record / "source_validation/ROUND20_DISPLAY_AUDIT.json").read_text(encoding="utf-8"))
        checks["round20_display_audit_pass"] = audit.get("status") == "PASS" and all(audit.get("checks", {}).values())
    if figure == "extended_data_10":
        audit = json.loads((record / "source_validation/ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json").read_text(encoding="utf-8"))
        checks["raw_decoder_sign_audit_pass"] = audit.get("status") == "PASS" and all(audit.get("checks", {}).values())
    return checks


def promote_one(figure: str) -> Path:
    parent_name, target_name = RECORD_SPECS[figure]
    parent = RECORDS / parent_name
    target = RECORDS / target_name
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite immutable record: {target}")
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".{target_name}.stage-", dir=RECORDS))
    try:
        shutil.copytree(parent, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json"):
            (stage / stale).unlink(missing_ok=True)
        refinement = PROMOTERS[figure](stage, target_name)
        refinement_manifest = {
            "schema": "cross-figure-refinement-record/v1",
            "status": "PASS",
            "record": target_name,
            "figure": figure,
            "supersedes": parent_name,
            "promoted_utc": datetime.now(timezone.utc).isoformat(),
            **refinement,
        }
        (stage / "CROSS_FIGURE_REFINEMENT_MANIFEST.json").write_text(
            json.dumps(refinement_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        append_manifest(
            stage,
            "2026-08-11 cross-figure refinement",
            "This immutable record supersedes `"
            + parent_name
            + "`. The scientific tables and exclusion authorities are inherited unchanged; "
            "the authorized display changes are enumerated in `CROSS_FIGURE_REFINEMENT_MANIFEST.json`.",
        )
        (stage / "PROMOTION_COMPLETE.txt").write_text(
            f"Promoted {target_name} at {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        checks = basic_record_checks(stage, figure)
        validation = {
            "schema": "cross-figure-record-validation/v1",
            "figure": figure,
            "record": target_name,
            "checks": checks,
            "status": "pass" if all(checks.values()) else "fail",
        }
        if validation["status"] != "pass":
            raise RuntimeError(f"Pre-hash validation failed for {figure}: {checks}")
        (stage / "RECORD_VALIDATION.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_hashes(stage)
        closed, errors = verify_hashes(stage)
        if not closed:
            raise RuntimeError(f"Hash closure failed for {figure}: {errors}")
        shutil.move(str(stage), target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def main() -> int:
    promoted = [promote_one(figure) for figure in RECORD_SPECS]
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": [path.relative_to(REPO).as_posix() for path in promoted],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
