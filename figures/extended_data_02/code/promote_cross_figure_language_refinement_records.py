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
import math
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
        "main_figure4_2026-08-11_round3_upright_i_density_ticks",
        "main_figure4_2026-08-11_round4_complete_upright_i_density_ticks",
    ),
    "ed1": (
        "extended_data_fig01_2026-08-09_round9_h_legend",
        "extended_data_fig01_2026-08-11_round10_open_terms",
    ),
    "ed2": (
        "extended_data_fig02_2026-08-11_round19_upright_policy_complete",
        "extended_data_fig02_2026-08-11_round20_upright_policy_clearance",
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
        "extended_data_fig08_2026-08-11_round10_horizontal_ticks",
        "extended_data_fig08_2026-08-11_round11_horizontal_ticks_metadata_closure",
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


def pdf_spans(path: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for block in document[0].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    item = dict(span)
                    item["direction"] = line.get("dir", (1.0, 0.0))
                    spans.append(item)
    return spans


def rect_distance(first: fitz.Rect, second: fitz.Rect) -> float:
    dx = max(first.x0 - second.x1, second.x0 - first.x1, 0.0)
    dy = max(first.y0 - second.y1, second.y0 - first.y1, 0.0)
    return math.hypot(dx, dy)


def ed2_annotation_clearances(path: Path) -> tuple[float, float]:
    with fitz.open(path) as document:
        page = document[0]
        annotation_spans = [
            span
            for span in pdf_spans(path)
            if span.get("text") in {"preceding", "turn"}
        ]
        if len(annotation_spans) != 2:
            return 0.0, 0.0
        annotation = fitz.Rect(annotation_spans[0]["bbox"])
        annotation.include_rect(fitz.Rect(annotation_spans[1]["bbox"]))
        drawings = page.get_drawings()
    grids = [
        fitz.Rect(item["rect"])
        for item in drawings
        if item.get("color") is not None
        and abs(float(item.get("width", 0.0)) - 0.72) < 0.02
    ]
    grey_arrows = [
        fitz.Rect(item["rect"])
        for item in drawings
        if item.get("color") is not None
        and abs(float(item.get("width", 0.0)) - 1.0) < 0.02
        and max(item["color"]) - min(item["color"]) < 0.02
    ]
    if not grids or not grey_arrows:
        return 0.0, 0.0
    return (
        min(rect_distance(annotation, rect) for rect in grids),
        min(rect_distance(annotation, rect) for rect in grey_arrows),
    )


def record_specific_checks(stage: Path, key: str) -> dict[str, bool]:
    if key == "main4":
        pdf_labels = [
            span
            for span in pdf_spans(stage / "figure4_version_record.pdf")
            if span.get("text") in {"Day i place maps", "Day i + 90 activity"}
        ]
        svg_text = (stage / "figure4_version_record.svg").read_text(encoding="utf-8")
        return {
            "root_day_i_labels_exact": len(pdf_labels) == 2,
            "root_day_i_labels_upright_arial": len(pdf_labels) == 2
            and all(
                "arial" in str(span.get("font", "")).lower()
                and "italic" not in str(span.get("font", "")).lower()
                and "oblique" not in str(span.get("font", "")).lower()
                for span in pdf_labels
            ),
            "root_svg_day_i_labels_literal": "Day i place maps" in svg_text
            and "Day i + 90 activity" in svg_text,
        }
    if key == "ed2":
        panel_a = next((stage / "panels").glob("ed02a_*.pdf"))
        grid_clearance, arrow_clearance = ed2_annotation_clearances(panel_a)
        source = json.loads((stage / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        result = json.loads((stage / "RENDER_RESULT.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (stage / "ED02_VERSION_RECORD_MANIFEST.json").read_text(encoding="utf-8")
        )
        report = (stage / "REPORT.md").read_text(encoding="utf-8")
        return {
            "annotation_grid_clearance_at_least_10px_600dpi": grid_clearance >= 1.2,
            "annotation_arrow_clearance_at_least_10px_600dpi": arrow_clearance >= 1.2,
            "source_manifest_accepted_record_local": source.get("status")
            == "accepted_current_version_record"
            and source.get("source_record")
            == f"docs/figure_version_records/{manifest.get('record')}"
            and str(source.get("frozen_renderer", {}).get("path", "")).startswith(
                "source_code/"
            ),
            "render_result_accepted_record_local": result.get("candidate_status")
            == "accepted_current_version_record"
            and result.get("output_root") == ".",
            "artwork_change_declared": manifest.get("artwork_bytes_changed") is True,
            "report_current": all(
                term in report
                for term in (
                    "upright policy notation",
                    "preceding turn",
                    "10 pixels",
                )
            ),
        }
    if key == "ed8":
        source_validation = json.loads(
            (
                stage
                / "source_validation/ED08_ROUND10_RENDER_MANIFEST.json"
            ).read_text(encoding="utf-8")
        )
        ticks = [
            span
            for span in pdf_spans(next((stage / "individual_panels").glob("ed08_panel_i_*.pdf")))
            if span.get("text") in {"Rewarded", "Timeout"}
        ]
        refinement = json.loads(
            (stage / "CROSS_FIGURE_REFINEMENT_MANIFEST.json").read_text(encoding="utf-8")
        )
        preserved = " ".join(refinement.get("preserved", []))
        return {
            "round10_source_validation_pass": source_validation.get("status") == "PASS"
            and source_validation.get("round10_validation", {}).get("status") == "PASS"
            and all(
                source_validation.get("round10_validation", {}).get("checks", {}).values()
            ),
            "panel_i_four_ticks_horizontal": [span["text"] for span in ticks]
            == ["Rewarded", "Timeout", "Rewarded", "Timeout"]
            and all(
                abs(float(span["direction"][0]) - 1.0) < 1e-6
                and abs(float(span["direction"][1])) < 1e-6
                for span in ticks
            ),
            "parity_claim_qualified": "raster-byte-exact" in preserved
            and "vector-metadata-only parity" in preserved
            and "panels a-h byte-exact to Round 9" not in preserved,
        }
    return {}


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
    checks.update(record_specific_checks(stage, key))
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
    copy(MAIN4_CANDIDATE_ROOT / "RENDER_VALIDATION.json", stage / "source_validation/ROUND4_RENDER_VALIDATION.json")
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
        "Round 4 closes the assembled-master typography defect: both panel-a Day labels "
        "now use literal upright Arial `i` in the individual panel and root composite. "
        "Panel e retains compact `0` density origins. Its heatmap payloads, vmax=0.6, "
        "density curves, axes, and all scientific tables are unchanged. Panels b, c, d "
        "and f are raster-exact to round 3.\n",
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
        "root and individual panel-a Day i labels use literal upright Arial",
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
            "review_round": 20,
            "supersedes": parent,
            "scientific_values_changed": False,
            "artwork_bytes_changed": True,
            "display_changes": [
                "all visible turning-bias P symbols and subscripts are upright",
                "panel-a preceding turn is split over two lines at (3.285, 3.82)",
                "panel-g lower contrast key is raised another exact 10 px at 600 dpi",
            ],
        },
    )
    manifest["publication_record_convention"] = {
        "canonical_primary_stem": "ed02_version_record",
        "record_change": (
            "round-20 display correction and metadata closure; scientific tables unchanged"
        ),
    }
    write_json(stage / "ED02_VERSION_RECORD_MANIFEST.json", manifest)

    source_manifest_path = stage / "SOURCE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest.update(
        {
            "status": "accepted_current_version_record",
            "source_record": f"docs/figure_version_records/{target}",
            "review_round": 20,
            "frozen_renderer": {
                "path": "source_code/render_ed02_slide3_upright_policy_round18.py",
                "sha256": sha256(REPO / "scripts/render_ed02_slide3_upright_policy_round18.py"),
            },
        }
    )
    source_manifest["panel_a_display_contract_round12"]["preceding_turn"]["xy"] = [
        3.285,
        3.82,
    ]
    source_manifest["panel_a_annotation_contract_round18"]["xy"] = [3.285, 3.82]
    write_json(source_manifest_path, source_manifest)

    render_result_path = stage / "RENDER_RESULT.json"
    render_result = json.loads(render_result_path.read_text(encoding="utf-8"))
    render_result.update(
        {
            "status": "pass",
            "candidate_status": "accepted_current_version_record",
            "record": target,
            "output_root": ".",
            "review_round": 20,
        }
    )
    render_result["panel_a_annotation_contract_round18"]["xy"] = [3.285, 3.82]
    render_result["panel_a_display_contract"]["preceding_turn"]["xy"] = [3.285, 3.82]
    write_json(render_result_path, render_result)

    caption_path = stage / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8").replace(
        "# Extended Data Fig. 2 round-18 caption notes",
        "# Extended Data Fig. 2 round-20 caption notes",
        1,
    )
    caption += (
        "\nRound-20 acceptance note: the two-line preceding turn annotation has measured "
        "10-pixel clearance from both the nearest grid and the grey arrow at 600 dpi. "
        "All policy symbols remain upright and all scientific values are unchanged.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")
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
        "Round 20 preserves the nine-panel inventory, 16 plot-ready tables and exact "
        "scientific values. Upright policy notation follows the main-figure typography "
        "hierarchy. Panel a's two-line preceding turn label has at least 10 px clearance "
        "from both the schematic grid and grey arrow at 600 dpi; panel g's lower labels "
        "are raised exactly 10 additional pixels.\n",
        encoding="utf-8",
    )
    (stage / "REPORT.md").write_text(
        "# Extended Data Figure 2 round-20 promotion report\n\n"
        "The nine-panel scientific content and all 16 plot-ready tables are unchanged. "
        "The accepted record uses upright policy notation in Arial at each role-specific size. The "
        "two-line preceding turn annotation is positioned at (3.285, 3.82), with measured "
        "clearance of at least 10 pixels from both the nearest grid and grey arrow at "
        "600 dpi. Panel g retains the exact additional 10-pixel label raise. Source and "
        "render metadata now identify this accepted record and its record-local renderer.\n",
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
        "panel-a preceding turn split to two lines with measured 10-pixel grid and arrow clearance",
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
    # Round 11 is a metadata/validation closure. Keep every accepted round-10
    # display asset byte-exact by cloning the parent and replacing only the
    # completed source-validation and record-facing metadata.
    copy(
        ED8_CANDIDATE / "ED08_ROUND10_RENDER_MANIFEST.json",
        stage / "source_validation/ED08_ROUND10_RENDER_MANIFEST.json",
    )
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
        "schema": "ed08-round11-metadata-closure-version-record/v1",
        "status": "accepted_immutable_version_record",
        "accepted_date": "2026-08-11",
        "record": target,
        "source_round": 10,
        "accepted_record_round": 11,
        "supersedes": parent,
        "panel_letters": list("abcdefghi"),
        "display_to_source_role": source_manifest["display_to_source_role"],
        "display_change": (
            "metadata-only closure; panel-i Rewarded/Timeout tick labels remain horizontal "
            "and centred exactly as accepted in round 10"
        ),
        "scientific_values_changed": False,
        "axes_geometry_changed": False,
        "placements_changed": False,
        "round10_source_validation": "source_validation/ED08_ROUND10_RENDER_MANIFEST.json",
        "round10_source_validation_status": source_manifest.get("status"),
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
    refinement_path = stage / "CROSS_FIGURE_REFINEMENT_MANIFEST.json"
    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
    refinement.update(
        {
            "record": target,
            "supersedes": parent,
            "authorized_changes": [
                "record metadata and validation closure only; no artwork change from round 10"
            ],
            "preserved": [
                "all round-10 root and individual display assets byte-exact",
                (
                    "relative to round 9, panels a-h are raster-byte-exact; SVG/PDF "
                    "have vector-metadata-only parity after timestamp normalization"
                ),
                "panel-i source values and non-text artwork",
                "geometry, statistics, source tables and exclusions",
            ],
        }
    )
    write_json(refinement_path, refinement)
    (stage / "MANIFEST.md").write_text(
        "# Extended Data Figure 8 accepted record\n\n"
        "Round 11 is byte-exact to the accepted round-10 artwork and closes its metadata "
        "and validation. The completed source validation gates all four panel-i labels, "
        "zero-degree rotation, centred alignment, axes geometry, source authority, colours "
        "and typography. All source values, statistics, placements and exclusions remain "
        "unchanged. Relative to round 9, panels a-h are raster-byte-exact; their vector "
        "exports differ only in metadata timestamps.\n",
        encoding="utf-8",
    )
    (stage / "CAPTION_NOTES.md").write_text(
        "# Caption notes\n\n"
        "Panel i compares rewarded and timeout outcomes in two maintained readouts. Tick "
        "orientation is a display choice and does not change the paired values, tests or "
        "sampling units. The accepted neural source retains exclusion_profile=none; the "
        "obsolete pre-sort manual neuron-ID profile is not applied. All other panel "
        "interpretations follow round 10.\n",
        encoding="utf-8",
    )
    (stage / "REBUILD_README.md").write_text(
        "# Rebuild ED8 round 11\n\n"
        "Run `python docs/extended_data_supplement_v2_rebuild_2026-08-05/"
        "ed08_reward_proximal_controls/revision_round10_2026-08-11/render_ed08_round10.py`, "
        "or use `--validate-existing` to close unchanged exports. Require source-validation "
        "status PASS, four exact horizontal centred tick labels, geometry/source/exclusion "
        "parity and complete record hashes before append-only promotion.\n",
        encoding="utf-8",
    )
    (stage / "SOURCE_NOTES.md").write_text(
        "# Source notes\n\n"
        "Round 10 imports the accepted round-9 renderer; round 11 closes validation and "
        "metadata without changing any display byte. The Main Figure 4 example authority is "
        "resolved from the immutable active/archive records. No quantitative analysis was "
        "recomputed.\n",
        encoding="utf-8",
    )
    copy_sources(stage, ED8_CANDIDATE / "render_ed08_round10.py", Path(__file__).resolve())
    return ["metadata and validation closure; artwork byte-exact to accepted round 10"]


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
