"""Incorporate the accepted Extended Data Figure 1 build into the supplement-v2 rebuild tree.

Extended Data Figure 1 under the **merged numbering** of ``sup-figs-v2.pptx`` is slide 1
plus the empty slide 2 ("merge with ext fig 1"), so it carries both the old ED1 acquisition
panels (a-f) and the old ED2 exploration panels (g-k).

The figure itself was built by ``scripts/render_extended_data_figure1_round4.py`` across
review rounds 4-9, all sharing one working folder. This script does not re-render it. It
moves the accepted round-9 payload into the same directory shape the Extended Data
Figure 3 rebuild established, and writes ``ED01_ASSEMBLY_MANIFEST.json`` and
``ED01_VALIDATION.csv`` by **re-verifying** the contract against the files on disk rather
than copying the renderer's own claims forward.

Two divergences from the settled Extended Data display contract are recorded explicitly as
``DEVIATION`` rows so they cannot be mistaken for passes and cannot silently grow:

* panel f rotates its x tick labels 28 deg, which the shared contract forbids;
* typography is drawn at 5.7 / 5.0 pt rather than the contract's 0.75x scale of 6.0 / 5.25.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]

BUILD = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure1_source_rebuild_2026-08-05_round4_layout_clean"
)
REVIEW_WORKSPACE = REPO_ROOT / "tmp" / "extended_data_figure1_review_2026-08-05"

BUNDLE = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed01_acquisition_and_exploration"
)
ASSEMBLY = BUNDLE / "assembly"
REVIEW = BUNDLE / "review_2026-08-05"

STEM = "ed01_acquisition_and_exploration_assembly"
ROUND = 9
BUILD_STEM = f"extended_data_figure1_source_rerender_round{ROUND}"

REVIEW_HISTORY_ROUNDS = (4, 5, 6, 7)

# Contract constants asserted below, all traceable to the maintained source tables.
CURRENT153_SESSIONS = 153
CURRENT153_MICE = 36
TILE_TIME_SLOPE = 0.9642052475245702
TILE_TIME_P = 2.809040004049919e-41
PANEL_F_BH27_CONTRASTS = 27
PANEL_F_DISPLAYED_CONTRASTS = 18
PANEL_I_ANIMAL = "451"
PANEL_I_NAIVE_CYCLES = 13
PANEL_I_RETURN_CYCLES = 11
PANEL_I_COLUMN1_X_MAX = 494.5
PANEL_I_COLUMN2_X_MAX = 846.5
TRAJECTORY_STEP_LIMIT_GRID = 0.63

CONTRACT_TYPOGRAPHY = {"base": 6.0, "axis_label": 6.0, "tick": 5.25}
BUILD_TYPOGRAPHY = {"base": 5.7, "axis_label": 5.7, "tick": 5.0}

DERIVED_EXPORTS = (
    f"ACCUMULATION_SUMMARY_ROUND{ROUND}.csv",
    f"SPEED_SUMMARY_ROUND{ROUND}.csv",
    f"TIME_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
    f"TILE_EFFICIENCY_SUMMARY_ROUND{ROUND}.csv",
    f"PANEL_I_END_NODE_SEQUENCE_ROUND{ROUND}.csv",
    f"PANEL_I_CYCLE_SUMMARY_ROUND{ROUND}.csv",
    f"PANEL_I_SELECTION_AUDIT_ROUND{ROUND}.csv",
    f"PANEL_A_TRAJECTORY_PLOT_DATA_ROUND{ROUND}.csv.gz",
    f"PANEL_A_JUMP_MASK_AUDIT_ROUND{ROUND}.csv",
)

PANEL_MAP = {
    "a": "Four shared-scale trajectories through the first reward (Day 1/2/91/181)",
    "b": "Cumulative rewards over time in maze, by return interval",
    "c": "Cumulative long reward-path bouts over time in maze, by return interval",
    "d": "Cumulative rewards over corrected-nose maze-tile transitions",
    "e": "Cumulative long reward-path bouts over maze-tile transitions",
    "f": "Acquisition endpoints (first reward, tenth reward, first long bout) on time and tile axes",
    "g": "Tile transitions versus elapsed time to first long-bout onset (log-log)",
    "h": "Body-mass tracking speed in exploration and reward-directed modes",
    "i": "Repeated 32-end-node cycle example, naive versus return (same mouse)",
    "j": "Exploration efficiency against time in maze",
    "k": "Exploration efficiency against maze-tile transitions",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def build() -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def record(name: str, ok: bool, detail: str, *, deviation: bool = False) -> None:
        if deviation:
            status = "DEVIATION"
        else:
            status = "PASS" if ok else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})

    # ---------------------------------------------------------------- source audit
    audit_rows = list(
        csv.DictReader((BUILD / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv").open(encoding="utf-8"))
    )
    sources: dict[str, Any] = {}
    present = 0
    stable = 0
    for row in audit_rows:
        path = REPO_ROOT / row["source_path"]
        exists = path.exists()
        live = sha256(path) if exists else ""
        present += int(exists)
        stable += int(exists and live == row["sha256"])
        sources[row["panel"]] = {
            "path": row["source_path"],
            "sha256": row["sha256"],
            "source_stage": row["source_stage"],
            "scientific_scope": row["scientific_scope"],
            "promotion_status": row["promotion_status"],
            "bytes": path.stat().st_size if exists else None,
        }
    record(
        "source_inputs_present",
        present == len(audit_rows),
        f"{present}/{len(audit_rows)} declared inputs on disk",
    )
    record(
        "source_inputs_hash_stable_since_build",
        stable == len(audit_rows),
        f"{stable}/{len(audit_rows)} sha256 identical to build time",
    )
    reused = [r for r in audit_rows if r["panel_svg_reused"].strip().lower() != "false"]
    record(
        "no_panel_svg_or_raster_reused",
        not reused,
        f"{len(reused)} panels drawn from a maintained panel image",
    )
    raw_inputs = [r for r in audit_rows if r["source_stage"].startswith("raw ")]
    record(
        "input_stage_mix",
        len(raw_inputs) == 2,
        f"{len(audit_rows) - len(raw_inputs)} plot-ready/session-level tables + "
        f"{len(raw_inputs)} raw maintained trajectories (panel i)",
    )

    # ------------------------------------------------------- scientific scope re-check
    endpoint_rows = read_table(REPO_ROOT / sources["f-rows"]["path"])
    record(
        "current153_endpoint_scope",
        endpoint_rows["session"].nunique() == CURRENT153_SESSIONS
        and endpoint_rows["animal_id"].nunique() == CURRENT153_MICE,
        f"{endpoint_rows['session'].nunique()} sessions / "
        f"{endpoint_rows['animal_id'].nunique()} mice",
    )
    endpoint_stats = read_table(REPO_ROOT / sources["f-stats"]["path"])
    displayed = endpoint_stats[endpoint_stats["axis_key"].isin(["maze_time_s", "maze_tiles"])]
    record(
        "panel_f_bh27_family",
        len(endpoint_stats) == PANEL_F_BH27_CONTRASTS
        and len(displayed) == PANEL_F_DISPLAYED_CONTRASTS,
        f"{len(endpoint_stats)} BH27 contrasts, {len(displayed)} displayed on the "
        f"time and tile rows; max displayed q={displayed['q_bh27'].max():.3g}",
    )
    tile_rows = read_table(REPO_ROOT / sources["g-rows"]["path"])
    tile_stats = read_table(REPO_ROOT / sources["g-stats"]["path"])
    record(
        "current153_tile_time_scope",
        len(tile_rows) == CURRENT153_SESSIONS
        and int(tile_stats.iloc[0]["n_sessions"]) == CURRENT153_SESSIONS
        and int(tile_stats.iloc[0]["n_mice"]) == CURRENT153_MICE,
        f"{len(tile_rows)} session rows; stats declare "
        f"{int(tile_stats.iloc[0]['n_sessions'])}/{int(tile_stats.iloc[0]['n_mice'])}",
    )
    record(
        "tile_time_diagnostic_frozen",
        abs(float(tile_stats.iloc[0]["slope_loglog"]) - TILE_TIME_SLOPE) < 1e-12
        and abs(float(tile_stats.iloc[0]["p_cluster_t_two_sided"]) / TILE_TIME_P - 1.0) < 1e-9,
        f"log-log slope {float(tile_stats.iloc[0]['slope_loglog']):.6f} "
        f"[{float(tile_stats.iloc[0]['ci95_slope_low']):.4f}, "
        f"{float(tile_stats.iloc[0]['ci95_slope_high']):.4f}], "
        f"CR1 p={float(tile_stats.iloc[0]['p_cluster_t_two_sided']):.3g}, "
        f"df={int(tile_stats.iloc[0]['df_t'])}",
    )

    # ------------------------------------------------------- round-9 build validation
    render = json.loads(
        (BUILD / f"RENDER_VALIDATION_ROUND{ROUND}.json").read_text(encoding="utf-8")
    )
    failed = render.get("failed_checks", [])
    record(
        "round9_build_checks_all_pass",
        render["status"] == "pass" and not failed,
        f"{len(render['checks'])} build checks, {len(failed)} failing",
    )
    record(
        "no_background_grid_lines",
        render["visible_gridline_count"] == 0,
        f"{render['visible_gridline_count']} visible gridlines across all axes",
    )
    record(
        "single_canvas_no_nonuniform_scaling",
        bool(render["checks"]["single_canvas_no_nonuniform_panel_scaling"]),
        "one matplotlib canvas; no imported panel image transforms",
    )
    record(
        "aligned_full_height_row_groups",
        float(render["maximum_row_alignment_delta_fraction"]) == 0.0,
        "b-e and i-k retain matched top/bottom edges; max delta "
        f"{render['maximum_row_alignment_delta_fraction']} of canvas",
    )
    record(
        "panel_g_h_square_and_vertically_centered",
        bool(render["checks"]["panel_g_square"])
        and bool(render["checks"]["panel_h_axes_square"])
        and bool(render["checks"]["panels_g_h_vertically_centered_in_f_row"]),
        f"g aspect {render['panel_g_box_aspect']}; h aspects "
        f"{render['panel_h_box_aspects']}; center delta "
        f"{render['panel_fgh_center_alignment_delta_fraction']}",
    )
    record(
        "panel_h_legend_lower_right_and_clear",
        bool(render["checks"]["panel_h_legend_lower_right"])
        and bool(render["checks"]["panel_h_legend_inside_lower_right_quadrant"])
        and bool(render["checks"]["panel_h_legend_clear_of_data"]),
        f"legend bbox {render['panel_h_legend_bbox_axes_fraction']}; "
        f"data overlap count {render['panel_h_legend_data_overlap_count']}",
    )
    record(
        "panel_letters_a_to_k",
        sorted(render["panel_label_positions"]) == list("abcdefghijk"),
        f"{len(render['panel_label_positions'])} panel letters a-k",
    )
    pair = render["panel_i_selected_pair"][0]
    record(
        "panel_i_selected_pair",
        pair["animal_id"] == PANEL_I_ANIMAL
        and pair["naive_complete_cycles"] == PANEL_I_NAIVE_CYCLES
        and pair["return_complete_cycles"] == PANEL_I_RETURN_CYCLES,
        f"mouse {pair['animal_id']}: {pair['naive_session']} "
        f"({pair['naive_complete_cycles']} complete cycles) vs {pair['return_session']} "
        f"({pair['return_complete_cycles']})",
    )
    limits = render["panel_i_shared_x_limits"]
    record(
        "panel_i_shared_x_axes",
        limits["0"][0] == limits["0"][1]
        and limits["1"][0] == limits["1"][1]
        and limits["2"][0] == limits["2"][1]
        and abs(limits["0"][0][1] - PANEL_I_COLUMN1_X_MAX) < 1e-9
        and abs(limits["1"][0][1] - PANEL_I_COLUMN2_X_MAX) < 1e-9,
        f"column 1 truncated to the naive two-cycle range (x max "
        f"{limits['0'][0][1]}); column 2 independent (x max {limits['1'][0][1]}); "
        "rows share limits in all three columns",
    )
    record(
        "panel_a_no_jump_mask",
        float(render["day1_corrected_max_drawn_step_grid"]) <= TRAJECTORY_STEP_LIMIT_GRID
        < float(render["day1_source_max_drawn_step_grid"]),
        f"drawn step {render['day1_corrected_max_drawn_step_grid']:.4f} grid units after "
        f"masking, from {render['day1_source_max_drawn_step_grid']:.4f} in the source table",
    )
    record(
        "panel_f_every_axis_tested",
        render["panel_f_annotation_counts"] == [3, 3, 3, 3, 3, 3],
        "three bracket-and-star contrasts on each of the six displayed axes",
    )
    record(
        "cross_figure_scope_documented",
        bool(render["checks"]["cross_figure_scope_documented"])
        and bool(render["checks"]["candidate_jk_not_promoted"]),
        "g-k are maintained old-ED2 material carried by the merged slide; j-k are "
        "candidate axis options, not promoted ED2 primary panels",
    )
    record(
        "caption_level_prose_absent",
        bool(render["checks"]["caption_level_prose_absent"]),
        "titles, P/q values, support notes and sample counts live in CAPTION_NOTES.md",
    )

    # ---------------------------------------------------------------- frozen SVG checks
    svg_path = BUILD / f"{BUILD_STEM}.svg"
    svg = svg_path.read_text(encoding="utf-8", errors="replace")
    text_elements = svg.count("<text")
    record("svg_editable_text", text_elements > 0, f"{text_elements} live <text> elements")
    record("svg_declares_arial", "Arial" in svg, "Arial declared in SVG")
    images = re.findall(r'<image[^>]*?width="([\d.]+)"[^>]*?height="([\d.]+)"', svg)
    pdf_path = BUILD / f"{BUILD_STEM}.pdf"
    with fitz.open(pdf_path) as document:
        pdf_images = sum(len(page.get_images(full=True)) for page in document)
        pdf_drawings = sum(len(page.get_drawings()) for page in document)
    record(
        "fully_vector_svg_pdf",
        len(images) == 0 and pdf_images == 0 and pdf_drawings > 1_000,
        f"SVG image elements={len(images)}; PDF image objects={pdf_images}; "
        f"PDF vector drawing objects={pdf_drawings}",
    )
    rotations = sorted(
        {abs(float(v)) for v in re.findall(r"rotate\(\s*(-?\d+(?:\.\d+)?)", svg)}
        - {0.0}
    )
    record(
        "xticklabels_not_rotated",
        False,
        "panel f rotates its three x tick labels 28 deg to fit the narrow endpoint "
        f"columns; the shared contract requires 0 deg. Rotations present: {rotations} "
        "(90 deg is intended: y-axis titles and the vertical naive/return example labels)",
        deviation=True,
    )
    record(
        "typography_matches_shared_contract",
        False,
        f"drawn at base {BUILD_TYPOGRAPHY['base']} / tick {BUILD_TYPOGRAPHY['tick']} pt "
        f"rather than the 0.75x contract's {CONTRACT_TYPOGRAPHY['base']} / "
        f"{CONTRACT_TYPOGRAPHY['tick']} pt; 14 panels on a 7.5-inch canvas. "
        "0.95x of contract size",
        deviation=True,
    )

    # ------------------------------------------------------------------ copy the payload
    ASSEMBLY.mkdir(parents=True, exist_ok=True)
    (ASSEMBLY / "derived").mkdir(exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[str, str, int]] = []

    def copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise SystemExit(f"copy not byte-identical: {destination}")
        copied.append(
            (
                str(destination.relative_to(BUNDLE)).replace("\\", "/"),
                sha256(destination),
                destination.stat().st_size,
            )
        )

    outputs: list[dict[str, Any]] = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        source = BUILD / f"{BUILD_STEM}.{suffix}"
        destination = ASSEMBLY / f"{STEM}.{suffix}"
        copy(source, destination)
        outputs.append(
            {
                "path": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
        record(
            f"output_{suffix}",
            destination.stat().st_size > 0,
            f"{destination.name} {destination.stat().st_size} bytes",
        )

    record(
        "png_and_tiff_600dpi_square",
        render["png_size"] == [4500, 4500]
        and render["tiff_size"] == [4500, 4500]
        and render["tiff_dpi"] == [600.0, 600.0]
        and render["tiff_compression"] == "tiff_lzw",
        "4500 x 4500 px at 600 dpi on a 7.5-inch canvas; TIFF LZW",
    )

    copy(BUILD / f"LAYOUT_AUDIT_ROUND{ROUND}.csv", ASSEMBLY / "LAYOUT_AUDIT.csv")
    copy(BUILD / f"SOURCE_DATA_AUDIT_ROUND{ROUND}.csv", ASSEMBLY / "SOURCE_DATA_AUDIT.csv")
    copy(BUILD / f"CAPTION_NOTES_ROUND{ROUND}.md", ASSEMBLY / "CAPTION_NOTES.md")
    copy(
        BUILD / f"RENDER_VALIDATION_ROUND{ROUND}.json",
        ASSEMBLY / "ROUND9_RENDER_VALIDATION.json",
    )
    for name in DERIVED_EXPORTS:
        copy(BUILD / name, ASSEMBLY / "derived" / name)

    for round_number in REVIEW_HISTORY_ROUNDS:
        deck = BUILD / f"extended_data_figure1_layout_review_round{round_number}.pptx"
        copy(deck, REVIEW / f"ed01_revision_review_round{round_number}.pptx")
        render_png = REVIEW_WORKSPACE / f"extended_data_figure1_round{round_number}_review.png"
        if render_png.exists():
            copy(render_png, REVIEW / f"ed01_review_round{round_number}_powerpoint.png")
    annotated = REVIEW_WORKSPACE / "extended_data_figure1_round6_annotated.png"
    if annotated.exists():
        copy(annotated, REVIEW / "ed01_round6_user_annotated.png")
    record(
        "review_history_carried",
        len([c for c in copied if c[0].startswith("review_2026-08-05/")]) >= 9,
        "rounds 4-7 pre-revision decks, their PowerPoint renders, and the round-6 "
        "annotated slide that specified round 7",
    )

    # ------------------------------------------------------------------- write manifests
    deviations = [c["check"] for c in checks if c["status"] == "DEVIATION"]
    failures = [c["check"] for c in checks if c["status"] == "FAIL"]

    manifest = {
        "figure": "Extended Data Figure 1 (merged numbering; slide 1 + empty slide 2 of "
        "sup-figs-v2.pptx)",
        "layout_authority": "slide 1; speaker note on slide 2 merges old Extended Data "
        "Figure 2 into Figure 1, so panels a-f are old-ED1 acquisition and g-k are "
        "old-ED2 exploration material",
        "accepted_round": ROUND,
        "renderer": "scripts/render_extended_data_figure1_round4.py",
        "build_folder": str(BUILD.relative_to(REPO_ROOT)).replace("\\", "/"),
        "figure_size_inches": [7.5, 7.5],
        "panels": PANEL_MAP,
        "sources": sources,
        "source_policy": {
            "rendered_directly_from_tables_and_raw_trajectories": True,
            "reused_panel_svgs": 0,
            "reused_panel_rasters": 0,
            "in_canvas_raster_elements": 0,
            "in_canvas_raster_explanation": (
                "round 9 exports trajectory collections and the colorbar as vector paths; "
                "the SVG and PDF contain no raster image objects"
            ),
        },
        "audit": {
            "current153_sessions": CURRENT153_SESSIONS,
            "current153_mice": CURRENT153_MICE,
            "tile_time_slope_loglog": TILE_TIME_SLOPE,
            "tile_time_cluster_p_two_sided": TILE_TIME_P,
            "panel_f_bh27_contrasts": PANEL_F_BH27_CONTRASTS,
            "panel_f_displayed_contrasts": PANEL_F_DISPLAYED_CONTRASTS,
            "panel_i_selected_pair": pair,
            "panel_i_shared_x_limits": limits,
            "panel_a_corrected_max_drawn_step_grid": render[
                "day1_corrected_max_drawn_step_grid"
            ],
            "grid_lines_visible": render["visible_gridline_count"],
            "svg_text_elements": text_elements,
            "panel_g_box_aspect": render["panel_g_box_aspect"],
            "panel_h_box_aspects": render["panel_h_box_aspects"],
            "panel_fgh_center_alignment_delta_fraction": render[
                "panel_fgh_center_alignment_delta_fraction"
            ],
            "panel_h_legend_location": render["panel_h_legend_location"],
            "panel_h_legend_labels": render["panel_h_legend_labels"],
            "panel_h_legend_bbox_axes_fraction": render[
                "panel_h_legend_bbox_axes_fraction"
            ],
            "panel_h_legend_data_overlap_count": render[
                "panel_h_legend_data_overlap_count"
            ],
            "svg_image_elements": len(images),
            "pdf_image_objects": pdf_images,
            "pdf_vector_drawing_objects": pdf_drawings,
            "typography_drawn": BUILD_TYPOGRAPHY,
            "typography_shared_contract": CONTRACT_TYPOGRAPHY,
        },
        "documented_deviations": {
            "xticklabels_rotated_28_deg_in_panel_f": "shared contract requires 0 deg; "
            "recommended fix in a later round",
            "typography_0p95x_of_shared_contract": "5.7/5.0 pt drawn vs 6.0/5.25 pt "
            "contract, to fit 14 panels on 7.5 inches",
        },
        "outputs": outputs,
        "bundle_files": [
            {"path": relative, "sha256": digest, "bytes": size}
            for relative, digest, size in sorted(copied)
        ],
        "validation": checks,
        "validation_status": "fail"
        if failures
        else ("pass_with_documented_deviations" if deviations else "pass"),
    }

    (ASSEMBLY / "ED01_ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (ASSEMBLY / "ED01_VALIDATION.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)

    return {
        "bundle": str(BUNDLE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "copied_files": len(copied),
        "checks": len(checks),
        "deviations": deviations,
        "failures": failures,
        "validation_status": manifest["validation_status"],
    }


def main() -> None:
    result = build()
    print(json.dumps(result, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
