"""Create append-only records for the 2026-08-11 language/typography pass.

Each target is cloned from its current scientific authority in a short staging
directory.  Only the validated display exports, current record documents, and
rebuild/audit sources are replaced; inherited scientific tables remain exact.
"""

from __future__ import annotations

import argparse
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
    "cross_figure_refinement_round2_2026-08-11"
)
MAIN4_CANDIDATE_ROOT = (
    REPO
    / "docs/main_fig4_d1_d2_source_comparison_2026-08-04/"
    "cross_figure_refinement_round2_2026-08-11"
)
MAIN4_CANDIDATE = MAIN4_CANDIDATE_ROOT / "d1_to_later_d1"
ED1_CANDIDATE = REPO / "docs/extended_data_figure1_source_rebuild_2026-08-05_round4_layout_clean"
ED2_CANDIDATE = REPO / "tmp/ed02_round18_upright_policy_2026-08-11"
ED6_CANDIDATE = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed06_within_session_convergence/maze_tile_round3_2026-08-11"
)
ED7_CANDIDATE = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed07_age_controls/assembly/display_round12"
)
ED8_ROOT = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed08_reward_proximal_controls"
)
ED8_CANDIDATE = ED8_ROOT / "revision_round10_2026-08-11"


SPECS = {
    "main3": (
        "main_figure3_2026-08-11_round2_timing_schematic_open_terms",
        "main_figure3_2026-08-11_round3_upright_day_i",
    ),
    "main4": (
        "main_figure4_2026-08-11_round2_place_maps_label",
        "main_figure4_2026-08-11_round3_upright_i_density_ticks",
    ),
    "ed1": (
        "extended_data_fig01_2026-08-09_round9_h_legend",
        "extended_data_fig01_2026-08-11_round10_open_terms",
    ),
    "ed2": (
        "extended_data_fig02_2026-08-11_round18_upright_policy_labels",
        "extended_data_fig02_2026-08-11_round19_upright_policy_complete",
    ),
    "ed6": (
        "extended_data_fig06_2026-08-11_round2_open_terms",
        "extended_data_fig06_2026-08-11_round3_maze_tile_label",
    ),
    "ed7": (
        "extended_data_fig07_2026-08-09_round11_concise_row_titles",
        "extended_data_fig07_2026-08-11_round12_open_long_gap",
    ),
    "ed8": (
        "extended_data_fig08_2026-08-11_round9_open_terms",
        "extended_data_fig08_2026-08-11_round10_horizontal_ticks",
    ),
}

FORBIDDEN_VISIBLE = (
    "maze-tile",
    "Maze-tile",
    "long-bout",
    "Long-bout",
    "long-gap",
    "Long-gap",
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


def replace_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_json(path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    write_json(path, payload)
    return payload


def copy_sources(stage: Path, *paths: Path) -> None:
    for source in paths:
        copy(source, stage / "source_code" / source.name)


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


def verify_hashes(record: Path) -> list[str]:
    with (record / "RECORD_HASHES.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    actual = {
        path.relative_to(record).as_posix(): path
        for path in record.rglob("*")
        if path.is_file() and path.name != "RECORD_HASHES.csv"
    }
    errors = [f"unhashed:{name}" for name in sorted(set(actual) - set(expected))]
    errors += [f"missing:{name}" for name in sorted(set(expected) - set(actual))]
    for relative in sorted(set(actual) & set(expected)):
        path = actual[relative]
        row = expected[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(f"hash:{relative}")
    return errors


def root_exports(stage: Path) -> tuple[Path, Path, Path, Path]:
    svg = next(stage.glob("*_version_record.svg"))
    pdf = next(stage.glob("*_version_record.pdf"))
    png = next(stage.glob("*_version_record_600dpi.png"))
    tiff = next(stage.glob("*_version_record_600dpi.tiff"))
    return svg, pdf, png, tiff


def basic_checks(stage: Path) -> dict[str, bool]:
    svg, pdf, png, tiff = root_exports(stage)
    svg_text = svg.read_text(encoding="utf-8")
    with fitz.open(pdf) as document:
        pdf_one_page = document.page_count == 1
        pdf_text = "".join(page.get_text() for page in document)
    with Image.open(png) as image:
        png_ok = min(image.size) >= 2000 and max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    with Image.open(tiff) as image:
        tiff_ok = min(image.info.get("dpi", (0, 0))) >= 590
    return {
        "svg_nonblank": svg.stat().st_size > 1000,
        "pdf_one_page": pdf_one_page,
        "png_nonblank_high_resolution": png_ok,
        "tiff_600dpi": tiff_ok,
        "forbidden_dashed_terms_absent_from_svg_pdf": not any(
            term in svg_text or term in pdf_text for term in FORBIDDEN_VISIBLE
        ),
    }


def completion(stage: Path, key: str, target: str, parent: str, changes: list[str]) -> None:
    payload = {
        "schema": "cross-figure-language-refinement-record/v1",
        "status": "PASS",
        "figure_key": key,
        "record": target,
        "supersedes": parent,
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_display_changes": changes,
        "scientific_tables_changed": False,
        "backend": "Python/matplotlib only",
    }
    write_json(stage / "CROSS_FIGURE_LANGUAGE_REFINEMENT_MANIFEST.json", payload)
    (stage / "PROMOTION_COMPLETE.txt").write_text(
        f"Promoted {target} at {payload['promoted_utc']}\n", encoding="utf-8"
    )
    checks = basic_checks(stage)
    validation = {
        "schema": "cross-figure-language-refinement-validation/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "record": target,
        "checks": checks,
    }
    write_json(stage / "RECORD_VALIDATION.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"Record validation failed for {target}: {checks}")


def promote_main3(stage: Path, target: str, parent: str) -> list[str]:
    for suffix in ("svg", "pdf", "html"):
        copy(MAIN_CANDIDATE / f"figure3_source_publication.{suffix}", stage / f"figure3_version_record.{suffix}")
    render_pdf(stage / "figure3_version_record.pdf", stage / "figure3_version_record_300dpi.png", 300)
    render_pdf(
        stage / "figure3_version_record.pdf",
        stage / "figure3_version_record_600dpi.png",
        600,
        stage / "figure3_version_record_600dpi.tiff",
    )
    replace_tree(MAIN_CANDIDATE / "figure3_individual_panels", stage / "individual_panels")
    for name in (
        "figure3_source_manifest.json",
        "figure3_axes_geometry.csv",
        "figure3_panel_verification.csv",
        "figure3_day2_template_first30_alignment.csv",
        "figure3_day2_template_support_runs.csv",
    ):
        copy(MAIN_CANDIDATE / name, stage / name)
    update_json(
        stage / "figure3_source_manifest.json",
        {"record": target, "status": "accepted_immutable_version_record", "supersedes": parent},
    )
    (stage / "MANIFEST.md").write_text(
        "# Main Figure 3 accepted record\n\n"
        "Round 3 preserves every scientific table and panel placement from round 2. "
        "In panel b, `Day i` and `Day i + 1` are literal upright Arial text, matching "
        "Main Figure 1 panel b; math-font substitution is not used.\n",
        encoding="utf-8",
    )
    copy_sources(
        stage,
        REPO / "scripts/render_main_figs_v3_publication.py",
        REPO / "tests/test_figure3_schematic_row_layout.py",
        Path(__file__).resolve(),
    )
    return ["panel-b Day i and Day i + 1 changed from mathtext to upright Arial"]


def promote_main4(stage: Path, target: str, parent: str) -> list[str]:
    for suffix, destination in {
        "svg": "figure4_version_record.svg",
        "pdf": "figure4_version_record.pdf",
        "png": "figure4_version_record_600dpi.png",
        "tiff": "figure4_version_record_600dpi.tiff",
    }.items():
        copy(MAIN4_CANDIDATE / f"figure4_d1_source_to_later_d1.{suffix}", stage / destination)
    render_pdf(stage / "figure4_version_record.pdf", stage / "figure4_version_record_300dpi.png", 300)
    replace_tree(MAIN4_CANDIDATE / "individual_panels", stage / "individual_panels")
    copy(MAIN4_CANDIDATE_ROOT / "RENDER_VALIDATION.json", stage / "source_validation/ROUND3_RENDER_VALIDATION.json")
    for manifest in (
        "FIGURE4_PANEL_A_REFINEMENT_MANIFEST.json",
        "FIGURE4_PANEL_E_TICK_REFINEMENT_MANIFEST.json",
    ):
        copy(MAIN_CANDIDATE / "figure4_individual_panels" / manifest, stage / "source_validation" / manifest)
    update_json(
        stage / "figure4_source_manifest.json",
        {
            "record": target,
            "status": "accepted_immutable_version_record",
            "supersedes": parent,
            "panel_a_display_labels": ["Day i place maps", "Day i + 90 activity"],
            "panel_e_density_origin_labels": ["0", "0"],
        },
    )
    (stage / "MANIFEST.md").write_text(
        "# Main Figure 4 accepted record\n\n"
        "Round 3 uses upright Arial `i` in both panel-a Day labels and compact `0` "
        "origins on panel e's two density axes. The panel-e heatmap payloads, vmax=0.6, "
        "density curves, axes, and all scientific tables are unchanged. Panels b, c, d "
        "and f are raster-exact to round 2.\n",
        encoding="utf-8",
    )
    copy_sources(
        stage,
        REPO / "scripts/render_main_figs_v3_publication.py",
        REPO / "scripts/render_main_figure4_panel_a_refinement.py",
        REPO / "scripts/render_main_figure4_panel_e_tick_refinement.py",
        REPO / "scripts/render_figure4_day_source_comparison.py",
        Path(__file__).resolve(),
    )
    return [
        "panel-a Day i labels changed from mathtext to upright Arial",
        "panel-e density origins changed from 0.0 to 0 without axis or data changes",
    ]


def promote_ed1(stage: Path, target: str, parent: str) -> list[str]:
    stem = "extended_data_figure1_source_rerender_round10"
    for suffix, destination in {
        "svg": "ed01_version_record.svg",
        "pdf": "ed01_version_record.pdf",
        "png": "ed01_version_record_600dpi.png",
        "tiff": "ed01_version_record_600dpi.tiff",
    }.items():
        copy(ED1_CANDIDATE / f"{stem}.{suffix}", stage / destination)
    render_pdf(stage / "ed01_version_record.pdf", stage / "ed01_version_record_300dpi.png", 300)
    copy(ED1_CANDIDATE / "MANIFEST_ROUND10.json", stage / "ED01_ASSEMBLY_MANIFEST.json")
    copy(ED1_CANDIDATE / "CAPTION_NOTES_ROUND10.md", stage / "CAPTION_NOTES.md")
    copy(ED1_CANDIDATE / "LAYOUT_AUDIT_ROUND10.csv", stage / "LAYOUT_AUDIT.csv")
    copy(ED1_CANDIDATE / "SOURCE_DATA_AUDIT_ROUND10.csv", stage / "SOURCE_DATA_AUDIT.csv")
    copy(ED1_CANDIDATE / "RENDER_VALIDATION_ROUND10.json", stage / "ROUND10_RENDER_VALIDATION.json")
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 1 accepted record\n\n"
        "Round 10 changes visible wording only: `maze tile transitions`, `time in maze`, "
        "`first long bout time`, and `reward directed`. All values, statistics, layouts, "
        "palettes, panel geometry, and exclusion logic are unchanged from round 9.\n",
        encoding="utf-8",
    )
    copy_sources(stage, REPO / "scripts/render_extended_data_figure1_round4.py", Path(__file__).resolve())
    return [
        "maze tile replaces maze-tile in d, e, f, g and k",
        "panel f uses time in maze",
        "panel g uses first long bout time",
        "panel h uses reward directed",
    ]


def promote_ed2(stage: Path, target: str, parent: str) -> list[str]:
    composite = ED2_CANDIDATE / "composite"
    stem = "ed02_round18_upright_policy_version_record"
    for suffix, destination in {
        "svg": "ed02_version_record.svg",
        "pdf": "ed02_version_record.pdf",
        "png": "ed02_version_record_600dpi.png",
        "tiff": "ed02_version_record_600dpi.tiff",
    }.items():
        source = composite / f"{stem}{'_600dpi' if suffix in {'png', 'tiff'} else ''}.{suffix}"
        copy(source, stage / destination)
    render_pdf(stage / "ed02_version_record.pdf", stage / "ed02_version_record_300dpi.png", 300)
    copy(composite / f"{stem}_preview_1440.png", stage / "ed02_version_record_preview_1440.png")
    for directory in ("panels", "plot_ready", "ppt_vector", "shared_assets"):
        replace_tree(ED2_CANDIDATE / "panels" / directory, stage / directory)
    for name in (
        "CAPTION_NOTES.md",
        "PANEL_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
        "RENDER_RESULT.json",
        "SHARED_ASSET_MANIFEST.csv",
        "SOURCE_MANIFEST.json",
        "TABLE_MANIFEST.csv",
    ):
        copy(ED2_CANDIDATE / "panels" / name, stage / name)
    copy(composite / "ED02_ROUND18_COMPOSITE_MANIFEST.json", stage / "ED02_ROUND18_COMPOSITE_MANIFEST.json")
    manifest = update_json(
        stage / "ED02_VERSION_RECORD_MANIFEST.json",
        {
            "accepted_date": "2026-08-11",
            "record": target,
            "source_round": 18,
            "review_round": 19,
            "supersedes": parent,
            "scientific_values_changed": False,
            "display_changes": [
                "all visible turning-bias P symbols and subscripts are upright",
                "panel-a preceding turn is split over two lines at (2.90, 3.82)",
                "panel-g lower contrast key is raised another exact 10 px at 600 dpi",
            ],
        },
    )
    del manifest
    (stage / "FIGURE_LEGEND.md").write_text(
        "# Extended Data Figure 2\n\n"
        "Route-policy biases during the first 30 cumulative minutes in the maze. "
        "(a) Four upright policy definitions. (b,c) Session probabilities across Naive "
        "day, Return day and Second day. (d) Signed exploration-efficiency differences "
        "after one-policy substitutions. (e) Policy-wise distances from Return-day means. "
        "(f) Four-policy model calibration. (g) Depth profile of upright "
        "$\\mathrm{P}_{\\mathrm{BS}}$. (h) Four-policy and depth-varying reconstructions. "
        "(i) Depth-varying model calibration.\n",
        encoding="utf-8",
    )
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 2 accepted record\n\n"
        "Round 19 preserves the nine-panel inventory, 16 plot-ready tables and exact "
        "scientific values. Upright policy notation follows the main-figure typography "
        "hierarchy. Panel a's two-line preceding-turn label clears both the schematic and "
        "grey arrow; panel g's lower labels are raised exactly 10 additional pixels.\n",
        encoding="utf-8",
    )
    copy_sources(
        stage,
        REPO / "scripts/render_ed02_slide3_upright_policy_round18.py",
        REPO / "scripts/export_ed02_round18_vector_figure.py",
        Path(__file__).resolve(),
    )
    return [
        "all visible turning-bias symbols changed to upright notation at role-specific sizes",
        "panel-a preceding turn split to two lines and moved lower-right",
        "panel-g lower contrast labels raised another exact 10 px at 600 dpi",
    ]


def promote_ed6(stage: Path, target: str, parent: str) -> list[str]:
    stem = "ed06_within_session_convergence_assembly_round3"
    for suffix, destination in {
        "svg": "ed06_version_record.svg",
        "pdf": "ed06_version_record.pdf",
        "png": "ed06_version_record_600dpi.png",
        "tiff": "ed06_version_record_600dpi.tiff",
    }.items():
        copy(ED6_CANDIDATE / f"{stem}.{suffix}", stage / destination)
    render_pdf(stage / "ed06_version_record.pdf", stage / "ed06_version_record_300dpi.png", 300)
    for source, destination in {
        "ED06_ASSEMBLY_MANIFEST_ROUND3.json": "ED06_ASSEMBLY_MANIFEST.json",
        "ED06_DISPLAYED_STATISTICS_ROUND3.csv": "ED06_DISPLAYED_STATISTICS.csv",
        "ED06_VALIDATION_ROUND3.csv": "ED06_VALIDATION.csv",
        "CAPTION_NOTES_ROUND3.md": "CAPTION_NOTES.md",
    }.items():
        copy(ED6_CANDIDATE / source, stage / destination)
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 6 accepted record\n\n"
        "Round 3 changes the visible tile-axis wording to `Maze tile transitions`. "
        "All curves, support gates, statistics, row limits, layout and source tables are "
        "unchanged from round 2.\n",
        encoding="utf-8",
    )
    copy_sources(stage, REPO / "scripts/render_ed06_within_session_convergence_assembly.py", Path(__file__).resolve())
    return ["visible maze-tile wording changed to maze tile"]


def promote_ed7(stage: Path, target: str, parent: str) -> list[str]:
    stem = "ed07_age_controls_six_panel_round12"
    for suffix, destination in {
        "svg": "ed07_version_record.svg",
        "pdf": "ed07_version_record.pdf",
        "png": "ed07_version_record_600dpi.png",
        "tiff": "ed07_version_record_600dpi.tiff",
    }.items():
        copy(ED7_CANDIDATE / f"{stem}.{suffix}", stage / destination)
    render_pdf(stage / "ed07_version_record.pdf", stage / "ed07_version_record_300dpi.png", 300)
    stage_components = stage / "component_exports"
    stage_components.mkdir(parents=True, exist_ok=True)
    for row_stem in (
        "ed07_panels_abc_first30_to_day2_maintained58_round12",
        "ed07_panels_def_long_gap_clean36_round12",
    ):
        for suffix in ("svg", "pdf", "png", "tiff"):
            copy(ED7_CANDIDATE / f"{row_stem}.{suffix}", stage_components / f"{row_stem}.{suffix}")
    candidate_manifest = json.loads(
        (ED7_CANDIDATE / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json").read_text(encoding="utf-8")
    )
    write_json(stage / "ED07_DISPLAY_MANIFEST.json", candidate_manifest)
    copy(
        ED7_CANDIDATE / "ED07_POWERPOINT_MATCHED_DISPLAY_MANIFEST.json",
        stage / "source_validation/ED07_ROUND12_SOURCE_MANIFEST.json",
    )
    display_checks = {
        "source_manifest_pass": candidate_manifest.get("status") == "PASS",
        "bottom_label_open": candidate_manifest.get("display_contract", {}).get("bottom_y_axis_label") == "Long gap correlation",
        "scientific_gate_pass": candidate_manifest.get("scientific_gate_status") == "PASS",
        "visual_qa_pass": candidate_manifest.get("visual_qa_status") == "PASS",
    }
    write_json(
        stage / "ED07_DISPLAY_VALIDATION.json",
        {"schema": "ed07-round12-display-validation/v1", "status": "PASS" if all(display_checks.values()) else "FAIL", "checks": display_checks},
    )
    copy(ED7_CANDIDATE / "ED07_SIX_PANEL_CAPTION_NOTES.md", stage / "CAPTION_NOTES.md")
    copy(ED7_CANDIDATE / "ED07_SIX_PANEL_FIGURE_CONTRACT.md", stage / "FIGURE_CONTRACT.md")
    manifest = json.loads((stage / "ED07_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "accepted_date": "2026-08-11",
            "record": target,
            "source_round": 12,
            "supersedes": parent,
            "bottom_y_axis_label": "Long gap correlation",
            "scientific_values_changed": False,
        }
    )
    write_json(stage / "ED07_VERSION_RECORD_MANIFEST.json", manifest)
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 7 accepted record\n\n"
        "Round 12 changes only the lower-row y-axis wording to `Long gap correlation`. "
        "The maintained58 and corrected36 cohorts, BH3/BH4 families, six biological "
        "exclusions, no-manual-mask profile, palette and geometry are unchanged.\n",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        "# Rebuild ED7 round 12\n\n"
        "Run `python scripts/render_ed07_slide9_powerpoint_matched.py --round 12 "
        "--visual-qa-status PASS`, validate the source manifest, then use the append-only "
        "cross-figure language promotion. Scientific input parity is locked by role, SHA-256 "
        "and byte count so administrative archival paths cannot change the cohort.\n",
        encoding="utf-8",
    )
    copy_sources(
        stage,
        REPO / "scripts/render_ed07_slide9_powerpoint_matched.py",
        REPO / "scripts/render_ed07_slide9_age_controls_display.py",
        Path(__file__).resolve(),
    )
    return ["lower-row Long-gap correlation changed to Long gap correlation"]


def promote_ed8(stage: Path, target: str, parent: str) -> list[str]:
    stem = "extended_data_figure8_source_rerender_round10"
    for suffix, destination in {
        "svg": "ed08_version_record.svg",
        "pdf": "ed08_version_record.pdf",
        "png": "ed08_version_record_600dpi.png",
        "tiff": "ed08_version_record_600dpi.tiff",
    }.items():
        copy(ED8_ROOT / f"{stem}.{suffix}", stage / destination)
    render_pdf(stage / "ed08_version_record.pdf", stage / "ed08_version_record_300dpi.png", 300)
    render_pdf(stage / "ed08_version_record.pdf", stage / "ed08_version_record_600dpi.png", 600)
    replace_tree(ED8_CANDIDATE / "final_render/individual_panels", stage / "individual_panels")
    copy(ED8_CANDIDATE / "ED08_ROUND10_RENDER_MANIFEST.json", stage / "source_validation/ED08_ROUND10_RENDER_MANIFEST.json")
    source_manifest = json.loads(
        (ED8_CANDIDATE / "ED08_ROUND10_RENDER_MANIFEST.json").read_text(encoding="utf-8")
    )
    svg, pdf, png, tiff = root_exports(stage)
    individual = [
        {
            "path": path.relative_to(stage).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted((stage / "individual_panels").rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "schema": "ed08-round10-version-record/v1",
        "status": "accepted_immutable_version_record",
        "accepted_date": "2026-08-11",
        "record": target,
        "source_round": 10,
        "supersedes": parent,
        "panel_letters": list("abcdefghi"),
        "display_to_source_role": source_manifest["display_to_source_role"],
        "display_change": "panel-i Rewarded/Timeout tick labels are horizontal and centred",
        "scientific_values_changed": False,
        "axes_geometry_changed": False,
        "placements_changed": False,
        "round10_source_validation": "source_validation/ED08_ROUND10_RENDER_MANIFEST.json",
        "axes_geometry": source_manifest.get("axes_geometry", []),
        "root_output_sha256": {
            "svg": sha256(svg),
            "pdf": sha256(pdf),
            "png_600dpi": sha256(png),
            "tiff_600dpi": sha256(tiff),
        },
        "individual_panel_files": individual,
        "inherited_authorities": [
            "plot_ready",
            "source_data",
            "statistics",
            "exclusion_audit",
            "source_authority",
            "powerpoint_review",
        ],
    }
    write_json(stage / "ED08_VERSION_RECORD_MANIFEST.json", manifest)
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 8 accepted record\n\n"
        "Round 10 inherits the complete round-9 display and makes only the four panel-i "
        "Rewarded/Timeout tick labels horizontal and centred. All source values, statistics, "
        "axes geometry, panel placements, exclusions and semantic colours are unchanged.\n",
        encoding="utf-8",
    )
    (stage / "CAPTION_NOTES.md").write_text(
        "# Caption notes\n\n"
        "Panel i compares rewarded and timeout outcomes in two maintained readouts. Tick "
        "orientation is a display choice and does not change the paired values, tests or "
        "sampling units. The accepted neural source retains exclusion_profile=none; the "
        "obsolete pre-sort manual neuron-ID profile is not applied. All other panel "
        "interpretations follow round 9.\n",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        "# Rebuild ED8 round 10\n\n"
        "Run `python docs/extended_data_supplement_v2_rebuild_2026-08-05/"
        "ed08_reward_proximal_controls/revision_round10_2026-08-11/render_ed08_round10.py`, "
        "then validate root and individual-panel hashes, horizontal tick rotations, exclusion "
        "parity and record closure before append-only promotion.\n",
        encoding="utf-8",
    )
    (stage / "SOURCE_NOTES.md").write_text(
        "# Source notes\n\n"
        "Round 10 imports the accepted round-9 renderer. The Main Figure 4 example authority "
        "was repointed administratively to its current immutable record; the referenced panel "
        "payload is unchanged. No quantitative analysis was recomputed.\n",
        encoding="utf-8",
    )
    copy_sources(stage, ED8_CANDIDATE / "render_ed08_round10.py", Path(__file__).resolve())
    return ["panel-i Rewarded/Timeout tick labels changed from -18 degrees to horizontal"]


PROMOTERS: dict[str, Callable[[Path, str, str], list[str]]] = {
    "main3": promote_main3,
    "main4": promote_main4,
    "ed1": promote_ed1,
    "ed2": promote_ed2,
    "ed6": promote_ed6,
    "ed7": promote_ed7,
    "ed8": promote_ed8,
}


def promote_one(key: str) -> Path:
    parent_name, target_name = SPECS[key]
    parent = RECORDS / parent_name
    target = RECORDS / target_name
    if target.exists():
        errors = verify_hashes(target)
        if not errors and all(basic_checks(target).values()):
            return target
        raise RuntimeError(f"Existing target is not a closed record: {target}; {errors}")
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    stage = Path(tempfile.mkdtemp(prefix=f".language-{key}-", dir=RECORDS))
    try:
        shutil.copytree(parent, stage, dirs_exist_ok=True)
        for stale in ("RECORD_HASHES.csv", "RECORD_VALIDATION.json"):
            (stage / stale).unlink(missing_ok=True)
        changes = PROMOTERS[key](stage, target_name, parent_name)
        completion(stage, key, target_name, parent_name, changes)
        write_hashes(stage)
        errors = verify_hashes(stage)
        if errors:
            raise RuntimeError(f"Hash closure failed for {target_name}: {errors}")
        shutil.move(str(stage), target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key",
        action="append",
        choices=tuple(SPECS),
        help="Promote only the selected figure key; may be supplied more than once.",
    )
    parser.add_argument(
        "--refresh-hashes",
        action="store_true",
        help="Refresh complete ledgers for already-promoted selected records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys = args.key or list(SPECS)
    if args.refresh_hashes:
        records = []
        for key in keys:
            record = RECORDS / SPECS[key][1]
            if not record.is_dir():
                raise FileNotFoundError(record)
            write_hashes(record)
            errors = verify_hashes(record)
            if errors:
                raise RuntimeError(f"Hash refresh failed for {record.name}: {errors}")
            records.append(record)
    else:
        records = [promote_one(key) for key in keys]
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": [path.relative_to(REPO).as_posix() for path in records],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
