"""Promote the post-audit Main 1-5 and merged ED1-6 immutable records."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import fitz
import pandas as pd
from PIL import Image

SCRIPT_REPO = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO))

from scripts.validate_current_figure_records import (
    FIGURES,
    REPO,
    RECORD_ROOT,
    sha256,
    validate_record,
    verify_hash_closure,
)


TARGET_NAMES = {
    "main_figure_1": "main_figure1_2026-08-08_full_audit",
    "main_figure_2": "main_figure2_2026-08-08_full_audit",
    "main_figure_3": "main_figure3_2026-08-08_full_audit",
    "main_figure_4": "main_figure4_2026-08-08_full_audit",
    "main_figure_5": "main_figure5_2026-08-08_full_audit",
    "extended_data_1": "extended_data_fig01_2026-08-08_full_audit",
    "extended_data_2": "extended_data_fig02_2026-08-08_full_audit",
    "extended_data_3": "extended_data_fig03_2026-08-08_full_audit",
    "extended_data_4": "extended_data_fig04_2026-08-08_full_audit",
    "extended_data_5": "extended_data_fig05_2026-08-08_round11_full_audit",
    "extended_data_6": "extended_data_fig06_2026-08-08_full_audit",
}
BASE_NAMES = {
    "main_figure_1": "main_figures_2026-08-05",
    "main_figure_2": "main_figure2_2026-08-07_no_manual_mask_anchor_unmasked",
    "main_figure_3": "main_figure3_2026-08-07_no_manual_mask_final",
    "main_figure_4": "main_figure4_2026-08-05_round9",
    "main_figure_5": "main_figure5_2026-08-07_hierarchical_panelf_round6",
    "extended_data_1": "extended_data_fig01_2026-08-06_round7",
    "extended_data_2": "extended_data_fig02_2026-08-06_round2",
    "extended_data_3": "extended_data_fig03_2026-08-06_round7",
    "extended_data_4": "extended_data_fig04_2026-08-06_round7",
    "extended_data_5": "extended_data_fig05_2026-08-08_round10",
    "extended_data_6": "extended_data_fig06_2026-08-06_round5",
}
SKIP_NAMES = {"RECORD_HASHES.csv", "RECORD_VALIDATION.json", "PROMOTION_COMPLETE.txt"}


def copy_file(source: Path, target: Path, ledger: list[dict[str, object]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    ledger.append(
        {
            "source": str(source),
            "target": target.as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    )


def copy_tree(
    source: Path,
    target: Path,
    ledger: list[dict[str, object]],
    *,
    skip_prefixes: Iterable[str] = (),
) -> None:
    prefixes = tuple(prefix.replace("\\", "/").rstrip("/") for prefix in skip_prefixes)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        relative = path.relative_to(source).as_posix()
        if any(relative == prefix or relative.startswith(prefix + "/") for prefix in prefixes):
            continue
        copy_file(path, target / relative, ledger)


def copy_current_code(target: Path, ledger: list[dict[str, object]], paths: Iterable[str]) -> None:
    for relative in paths:
        source = REPO / relative
        copy_file(source, target / "source_code" / source.name, ledger)


def render_600(pdf: Path, png: Path, tiff: Path) -> None:
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected a one-page PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False)
        pixmap.save(png)
    with Image.open(png) as image:
        image.convert("RGB").save(tiff, format="TIFF", compression="tiff_lzw", dpi=(600, 600))


def build_ed2_composite(target: Path, ledger: list[dict[str, object]]) -> None:
    """Assemble the ten frozen vector panels into one editable merged-ED2 page."""
    width = 180 / 25.4 * 72
    height = 225 / 25.4 * 72
    margin = 5 / 25.4 * 72
    gutter_x = 4 / 25.4 * 72
    gutter_y = 3.2 / 25.4 * 72
    header = 8 / 25.4 * 72
    cell_width = (width - 2 * margin - gutter_x) / 2
    cell_height = (height - 2 * margin - header - 4 * gutter_y) / 5
    output = fitz.open()
    page = output.new_page(width=width, height=height)
    page.insert_text(
        (margin, margin + 7),
        "Extended Data Fig. 2 | first 30 min TimeInMaze; Explore decisions only",
        fontsize=6.6,
        color=(0.153, 0.192, 0.231),
    )
    inputs: list[dict[str, object]] = []
    for index, panel in enumerate("abcdefghij"):
        matches = sorted((target / "panels").glob(f"ed02{panel}_*.pdf"))
        if len(matches) != 1:
            raise ValueError(f"Expected one ED2 panel-{panel} PDF; found {matches}")
        source_path = matches[0]
        source = fitz.open(source_path)
        source_rect = source[0].rect
        row, column = divmod(index, 2)
        cell_x = margin + column * (cell_width + gutter_x)
        cell_y = margin + header + row * (cell_height + gutter_y)
        scale = min(cell_width / source_rect.width, cell_height / source_rect.height)
        drawn_width = source_rect.width * scale
        drawn_height = source_rect.height * scale
        rect = fitz.Rect(
            cell_x + (cell_width - drawn_width) / 2,
            cell_y + (cell_height - drawn_height) / 2,
            cell_x + (cell_width + drawn_width) / 2,
            cell_y + (cell_height + drawn_height) / 2,
        )
        page.show_pdf_page(rect, source, 0, keep_proportion=True)
        page.insert_text((cell_x, cell_y + 8), panel, fontsize=8.2, fontname="hebo")
        inputs.append(
            {
                "panel": panel,
                "path": source_path.relative_to(target).as_posix(),
                "sha256": sha256(source_path),
            }
        )
        source.close()
    pdf_path = target / "ed02_version_record.pdf"
    output.save(pdf_path, garbage=4, deflate=True)
    output.close()
    with fitz.open(pdf_path) as document:
        svg_path = target / "ed02_version_record.svg"
        svg_path.write_text(document[0].get_svg_image(text_as_path=False), encoding="utf-8")
    png_path = target / "ed02_version_record_600dpi.png"
    tiff_path = target / "ed02_version_record_600dpi.tiff"
    render_600(pdf_path, png_path, tiff_path)
    for path in (pdf_path, svg_path, png_path, tiff_path):
        ledger.append(
            {
                "source": "assembled from the ten frozen record-local panel PDFs",
                "target": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = target / "ED02_COMPOSITE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "displayed_figure_number": 2,
                "backend": "Python/PyMuPDF vector PDF assembly",
                "panel_order": list("abcdefghij"),
                "nominal_mm": [180, 225],
                "inputs": inputs,
                "outputs": {
                    path.suffix.lstrip("."): {
                        "path": path.name,
                        "sha256": sha256(path),
                    }
                    for path in (pdf_path, svg_path, png_path, tiff_path)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger.append(
        {
            "source": "generated composite provenance",
            "target": manifest.as_posix(),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        }
    )


def build_main1(target: Path, ledger: list[dict[str, object]]) -> None:
    base = RECORD_ROOT / BASE_NAMES["main_figure_1"]
    candidate = REPO / "tmp/full_vor_audit_render_2026-08-08"
    copy_tree(base, target, ledger, skip_prefixes=("figure1_", "MANIFEST.md"))
    mappings = {
        "figure1_source_publication.svg": "figure1_version_record.svg",
        "figure1_source_publication.pdf": "figure1_version_record.pdf",
        "figure1_source_manifest.json": "figure1_source_manifest.json",
        "figure1_panel_geometry.csv": "figure1_panel_geometry.csv",
        "figure1_panel_verification.csv": "figure1_panel_verification.csv",
        "figure1_alignment_audit.csv": "figure1_alignment_audit.csv",
        "figure1_individual_panels_manifest.json": "figure1_individual_panels_manifest.json",
    }
    for source, destination in mappings.items():
        copy_file(candidate / source, target / destination, ledger)
    copy_tree(candidate / "figure1_individual_panels", target / "figure1_individual_panels", ledger)
    render_600(
        target / "figure1_version_record.pdf",
        target / "figure1_version_record_600dpi.png",
        target / "figure1_version_record_600dpi.tiff",
    )
    for path in (target / "figure1_version_record_600dpi.png", target / "figure1_version_record_600dpi.tiff"):
        ledger.append({"source": "rendered from record PDF", "target": path.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = json.loads((candidate / "figure1_source_manifest.json").read_text(encoding="utf-8"))
    plot_sources: set[Path] = set()
    for row in manifest["sources"]:
        for token in str(row.get("source_path", "")).split(";"):
            source = REPO / token.strip().replace("\\", "/")
            if source.is_file() and source.suffix.lower() == ".csv":
                plot_sources.add(source)
    for token in manifest.get("audit_tables", {}).values():
        source = Path(str(token))
        if not source.is_absolute():
            source = REPO / source
        if source.is_file() and source.suffix.lower() == ".csv":
            plot_sources.add(source)
    for source in sorted(plot_sources):
        copy_file(source, target / "plot_ready" / source.name, ledger)

    sys.path.insert(0, str(REPO / "src"))
    from ca_analysis.figure_panels import extract_endnode_timeline

    timeline_rows: list[dict[str, object]] = []
    for nickname in ("434RRday1", "434-3day1", "434RRday2", "434-6day1", "1637Lday1", "1637Lday2"):
        endnodes, times = extract_endnode_timeline(nickname, lower_threshold=95, upper_threshold=126)
        timeline_rows.extend(
            {"nickname": nickname, "time_in_maze_s": float(time), "endnode": int(node)}
            for node, time in zip(endnodes, times, strict=True)
        )
    timeline_path = target / "plot_ready/figure1_endnode_timelines.csv"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(timeline_rows).to_csv(timeline_path, index=False)
    ledger.append({"source": "extract_endnode_timeline six displayed sessions", "target": timeline_path.as_posix(), "bytes": timeline_path.stat().st_size, "sha256": sha256(timeline_path)})
    copy_current_code(
        target,
        ledger,
        (
            "scripts/preview_figure1_assembly.py",
            "scripts/render_main_figs_v3_publication.py",
            "src/ca_analysis/figure_panels.py",
            "tests/test_figure1_eight_panel_layout.py",
            "tests/test_figure1_fixed30_turning_panel.py",
        ),
    )


def build_main2(target: Path, ledger: list[dict[str, object]]) -> None:
    base = RECORD_ROOT / BASE_NAMES["main_figure_2"]
    copy_tree(base, target, ledger, skip_prefixes=("source_data/gap_curve", "source_code", "SOURCE_COPIES.json", "MANIFEST.md"))
    gap = REPO / "docs/sorted_data_full_figure_rerun_2026-06-11/figure2_gap_anchor_no_manual_mask_2026-08-07"
    for name in ("gap_with_day1_day2_curve_data.csv", "same_stage_day1_day2_by_experience_curve_data.csv", "manifest.json", "REPORT.md"):
        copy_file(gap / name, target / "source_data/gap_curve" / name, ledger)
    audit = REPO / "docs/sorted_data_full_figure_rerun_2026-06-11/figure2_main3_ed12_source_repair_842_2026-08-03"
    for name in ("SOURCE_REPAIR_MANIFEST.json", "figure2_842_0to3_distribution_parity_audit.csv", "figure2_842_affected_pair_metric_before_after_support_audit.csv"):
        copy_file(audit / name, target / "source_data/gap_curve/upstream_corrected_842_audit" / name, ledger)
    copy_current_code(target, ledger, ("scripts/preview_figure2_assembly.py", "scripts/render_main_figs_v3_publication.py", "scripts/promote_main_figures_no_manual_mask.py", "scripts/validate_main_figure_no_manual_mask_record.py", "src/ca_analysis/figure_panels.py", "src/ca_analysis/exclusion_policy.py"))


def build_main3(target: Path, ledger: list[dict[str, object]]) -> None:
    base = RECORD_ROOT / BASE_NAMES["main_figure_3"]
    copy_tree(base, target, ledger, skip_prefixes=("source_data/long_gap", "source_code", "SOURCE_COPIES.json", "MANIFEST.md"))
    source = REPO / "docs/sorted_data_full_figure_rerun_2026-06-11/main3_long_gap_no_manual_mask_2026-08-07"
    copy_tree(source, target / "source_data/long_gap", ledger)
    copy_current_code(target, ledger, ("scripts/build_main3_long_gap_no_manual_mask_sources.py", "scripts/render_main_figs_v3_publication.py", "scripts/validate_main_figure_no_manual_mask_record.py", "scripts/export_fixed_reference_window_convergence.py", "src/ca_analysis/figure_panels.py"))


def build_main4(target: Path, ledger: list[dict[str, object]]) -> None:
    base = RECORD_ROOT / BASE_NAMES["main_figure_4"]
    candidate = REPO / "docs/main_fig4_d1_d2_source_comparison_2026-08-04/rendered_selected_d1_min5"
    page = candidate / "d1_to_later_d1"
    copy_tree(base, target, ledger, skip_prefixes=("figure4_", "individual_panels", "source_code", "plot_ready/figure4_panel_c_d1_source_rows.csv", "MANIFEST.md"))
    for suffix in ("svg", "pdf"):
        copy_file(page / f"figure4_d1_source_to_later_d1.{suffix}", target / f"figure4_version_record.{suffix}", ledger)
    copy_file(page / "figure4_d1_source_to_later_d1.png", target / "figure4_version_record_600dpi.png", ledger)
    copy_file(page / "figure4_d1_source_to_later_d1.tiff", target / "figure4_version_record_600dpi.tiff", ledger)
    copy_tree(page / "individual_panels", target / "individual_panels", ledger)
    copy_file(page / "figure4_panel_c_d1_source_rows.csv", target / "plot_ready/figure4_panel_c_d1_source_rows.csv", ledger)
    copy_file(candidate / "MANIFEST.json", target / "figure4_source_manifest.json", ledger)
    copy_file(candidate / "RENDER_VALIDATION.json", target / "source_validation/VALIDATION.json", ledger)
    copy_file(candidate / "REPORT.md", target / "source_validation/REPORT.md", ledger)
    old = base / "figure4_panel_verification.csv"
    copy_file(old, target / "figure4_panel_verification.csv", ledger)
    copy_file(base / "figure4_panel_geometry.csv", target / "figure4_panel_geometry.csv", ledger)
    copy_current_code(target, ledger, ("scripts/analyze_figure4_day_source_comparison.py", "scripts/render_figure4_day_source_comparison.py", "src/ca_analysis/figure_panels.py", "tests/test_figure4_day_source_current_layout.py"))


def build_main5(target: Path, ledger: list[dict[str, object]]) -> None:
    base = RECORD_ROOT / BASE_NAMES["main_figure_5"]
    copy_tree(base, target, ledger, skip_prefixes=("source_data/figure5_bundle", "source_code", "SOURCE_COPIES.json", "MANIFEST.md"))
    bundle = REPO / "docs/main_figs_v3_source_render_2026-06-24/figure5_no_manual_mask_2026-08-07"
    copy_tree(bundle, target / "source_data/figure5_bundle", ledger)
    candidate_manifest = REPO / "docs/main_figs_v3_no_manual_mask_render_2026-08-07/figure5_source_manifest.json"
    copy_file(candidate_manifest, target / "figure5_source_manifest.json", ledger)
    copy_current_code(target, ledger, ("scripts/preview_figure3_assembly.py", "scripts/render_main_figs_v3_publication.py", "scripts/build_figure5_corrected_842_sources.py", "scripts/validate_main_figure_no_manual_mask_record.py", "src/ca_analysis/figure_panels.py", "src/ca_analysis/level_heatmap_utils.py"))


def build_ed(target: Path, ledger: list[dict[str, object]], figure: str) -> None:
    base = RECORD_ROOT / BASE_NAMES[figure]
    if figure == "extended_data_5":
        copy_tree(base, target, ledger, skip_prefixes=("ed05_version_record", "ED05_ASSEMBLY_MANIFEST.json", "ED05_VALIDATION.csv", "SOURCE_DATA_AUDIT.csv", "LAYOUT_AUDIT.csv", "CAPTION_NOTES.md", "REBUILD_README.md", "source_code"))
        root = REPO / "docs/extended_data_supplement_v2_rebuild_2026-08-05/ed05_example_place_maps"
        assembly = root / "assembly"
        for suffix in ("svg", "pdf", "png", "tiff"):
            copy_file(assembly / f"ed05_example_place_maps_assembly_round11.{suffix}", target / f"ed05_version_record{('_600dpi' if suffix in {'png','tiff'} else '')}.{suffix}", ledger)
        copy_file(assembly / "MANIFEST_ROUND11.json", target / "ED05_ASSEMBLY_MANIFEST.json", ledger)
        copy_file(assembly / "ED05_VALIDATION_ROUND11.csv", target / "ED05_VALIDATION.csv", ledger)
        copy_file(assembly / "SOURCE_DATA_AUDIT_ROUND11.csv", target / "SOURCE_DATA_AUDIT.csv", ledger)
        copy_file(assembly / "LAYOUT_AUDIT_ROUND11.csv", target / "LAYOUT_AUDIT.csv", ledger)
        copy_file(root / "CAPTION_NOTES_ROUND11.md", target / "CAPTION_NOTES.md", ledger)
        copy_file(REPO / "docs/ed05_example_id_bridge_2026-08-07/README.md", target / "REBUILD_README.md", ledger)
        for name in ("ED5_PANEL_A_PER_SIDE_PROVENANCE.csv", "ED5_PANEL_A_PROVENANCE_SUMMARY.json"):
            copy_file(REPO / "docs/ed05_example_id_bridge_2026-08-07" / name, target / "plot_ready" / name, ledger)
        copy_current_code(target, ledger, ("scripts/render_ed05_example_place_maps_assembly.py", "scripts/build_ed05_gallery_provenance.py", "scripts/ed_supplement_common.py", "tests/test_ed05_example_place_maps_assembly.py"))
        return
    copy_tree(base, target, ledger, skip_prefixes=("MANIFEST.md",))
    if figure == "extended_data_2":
        build_ed2_composite(target, ledger)
    if figure == "extended_data_4":
        sources = (
            "docs/extended_data_figure_panel_refresh_2026-07-12/source_rendered_new/extended_data_fig_05/registration_flow_and_four_session/ed05_four_session_418_local_correlation_paper.png",
            "docs/extended_data_figure_panel_refresh_2026-07-12/extended_data_figures/extended_data_fig_05/03_illustrative_examples/ed05_four_session_418_coregistration_paper.png",
            "scripts/render_ed05_418_local_correlation.py",
            "scripts/render_ed05_registration_flow_and_four_session.py",
        )
        for relative in sources:
            source = REPO / relative
            copy_file(source, target / "embedded_sources" / source.name, ledger)
    if figure == "extended_data_6":
        copy_current_code(target, ledger, ("scripts/export_fixed_reference_window_convergence.py", "scripts/export_within_session_pivot_all_metrics.py", "scripts/render_ed06_within_session_convergence_assembly.py"))
        authority = REPO / "docs/coverage5min_window_reanalysis_2026-08-05"
        for path in sorted(authority.rglob("*MANIFEST*.json")):
            copy_file(path, target / "gate_authority" / path.relative_to(authority), ledger)


def write_manifest(target: Path, figure: str, base: Path) -> None:
    notes = {
        "main_figure_1": "Rerendered with current Python code; exact plot-ready behavior tables and six displayed timeline sessions are frozen.",
        "main_figure_2": "Panels f/k/p now freeze the exact no-manual-mask D1-D2 anchor table drawn by the accepted pixels.",
        "main_figure_3": "Displayed values are unchanged; row-wise long-gap provenance is consolidated and first-window content checks are record-local.",
        "main_figure_4": "Day-1 source remains authoritative; panel c now blanks six bins below five source frames.",
        "main_figure_5": "Hierarchical heatmap metadata is corrected; panel-f CR1 inference uses two-sided Student t13.",
        "extended_data_5": "Round 11 freezes qualified mixed provenance for panel a, exact b-d transformation closure, and the single-vintage panel-e rebuild.",
    }
    text = f"""# {figure.replace('_', ' ').title()} full-audit version record

Promoted on 2026-08-08 after the independent 99-panel audit. This directory is immutable;
later revisions require a new record and pointer update.

Supersedes `{base.name}`. {notes.get(figure, 'Displayed pixels and values are retained; the record is re-promoted with complete fail-closed hash closure and frozen reproducibility inputs.')}

`AUDIT_PROMOTION.json` records every copied/generated payload. `RECORD_VALIDATION.json`
contains the semantic/export result, and `RECORD_HASHES.csv` covers every record file except
itself, including the verdict and completion stamp.
"""
    (target / "MANIFEST.md").write_text(text, encoding="utf-8")


def write_hashes(target: Path) -> None:
    rows = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "RECORD_HASHES.csv":
            rows.append(
                {
                    "path": path.relative_to(target).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    with (target / "RECORD_HASHES.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    targets = {figure: RECORD_ROOT / TARGET_NAMES[figure] for figure in FIGURES}
    builders = {
        "main_figure_1": build_main1,
        "main_figure_2": build_main2,
        "main_figure_3": build_main3,
        "main_figure_4": build_main4,
        "main_figure_5": build_main5,
    }
    for figure in FIGURES:
        target = targets[figure]
        if target.exists():
            closure, _ = verify_hash_closure(target)
            validation_path = target / "RECORD_VALIDATION.json"
            if closure and validation_path.is_file():
                saved = json.loads(validation_path.read_text(encoding="utf-8"))
                if saved.get("status") == "pass":
                    print(f"SKIP complete immutable record: {target.name}")
                    continue
            if (target / "PROMOTION_COMPLETE.txt").exists() or (target / "RECORD_HASHES.csv").exists():
                raise RuntimeError(f"Refusing to modify incomplete-but-stamped record: {target}")
            print(f"RESUME incomplete unstamped record: {target.name}")
        else:
            target.mkdir(parents=True)
        ledger: list[dict[str, object]] = []
        if figure.startswith("main"):
            builders[figure](target, ledger)
        else:
            build_ed(target, ledger, figure)
        copy_current_code(target, ledger, ("scripts/promote_full_figure_audit_records.py", "scripts/validate_current_figure_records.py"))
        for row in ledger:
            row["target"] = Path(str(row["target"])).relative_to(target).as_posix()
        (target / "AUDIT_PROMOTION.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_manifest(target, figure, RECORD_ROOT / BASE_NAMES[figure])
        validation = validate_record(target, figure, require_hashes=False)
        (target / "RECORD_VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if validation["status"] != "pass":
            failed = [name for name, passed in validation["checks"].items() if not passed]
            raise RuntimeError(f"{figure} semantic promotion failed: {failed}")
        (target / "PROMOTION_COMPLETE.txt").write_text(
            f"PROMOTED {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} after full 99-panel audit.\n",
            encoding="utf-8",
        )
        write_hashes(target)
        closure, errors = verify_hash_closure(target)
        if not closure:
            raise RuntimeError(f"{figure} record closure failed: {errors}")
    print(json.dumps({figure: target.relative_to(REPO).as_posix() for figure, target in targets.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
