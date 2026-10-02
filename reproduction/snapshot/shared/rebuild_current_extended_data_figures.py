#!/usr/bin/env python3
"""Shared rendering adapters for Extended Data Figures 1–7 and 10.

The public launcher authenticates and stages the included source data. These
adapters supply paths and current display settings to each numerical renderer;
no finished artwork is an input. ED8 and ED9 have separate figure-local builders.
Use the root reproduce.py command; see docs/CODE.md for the function map.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import fitz
import numpy as np
import pandas as pd


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]
FORMATS = ("svg", "pdf", "png", "tiff")


@contextlib.contextmanager
def frozen_stage(record: Path) -> Iterator[Path]:
    # Use a short temporary path so nested exports also work on Windows.
    with tempfile.TemporaryDirectory(prefix="edsrc_") as raw:
        stage = Path(raw)
        shutil.copytree(record / "source_code", stage / "scripts")
        runtime_source = VOR_ROOT / "runtime"
        runtime_root = stage / "shared_runtime"
        shutil.copytree(runtime_source, runtime_root)
        inserted = [stage, stage / "scripts", runtime_root]
        for path in reversed(inserted):
            sys.path.insert(0, str(path))
        try:
            yield stage
        finally:
            for path in inserted:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(path))


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render_ed1(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_extended_data_figure1_round4.py", "ed1_source_rebuild")
    plot = record / "plot_ready"
    drawn = record / "drawn_values"
    m.REPO = REPO_ROOT
    m.OUTPUT = output
    m.ROUND = 21
    m.GENERATED = "record-local source rebuild"
    m.STEM = output / "extended_data_figure1_source_rebuild_round21"
    m.SOURCE_AUDIT = output / "SOURCE_DATA_AUDIT_ROUND21.csv"
    m.LAYOUT_AUDIT = output / "LAYOUT_AUDIT_ROUND21.csv"
    m.AXIS_SCALE_AUDIT = output / "AXIS_SCALE_AUDIT_ROUND21.csv"
    m.CAPTION_NOTES = output / "CAPTION_NOTES_ROUND21.md"
    m.RENDER_VALIDATION = output / "RENDER_VALIDATION_ROUND21.json"
    m.MANIFEST = output / "MANIFEST_ROUND21.json"
    source_map = {
        "b-c": plot / "ed01_bc_group_specific_80_session_curves.csv.gz",
        "d-e": plot / "ed01_de_group_specific_80_group_curves.csv",
        "f-rows": plot / "ed01_endpoint_axis_session_rows_current153.csv",
        "f-stats": plot / "ed01_endpoint_axis_cluster_stats_current153.csv",
        "g-rows": plot / "ed02_tile_time_loglog_rows_current153.csv",
        "g-stats": plot / "ed02_tile_time_loglog_stats_current153.csv",
        "h-rows": plot / "ed01_panel_h_first30_session_rows_cap60.csv",
        "h-reference": plot / "ed01_panel_h_first30_group_summary_cap60.csv",
        "i-naive": drawn / "PANEL_I_END_NODE_SEQUENCE_ROUND21.csv",
        "i-return": drawn / "PANEL_I_CYCLE_SUMMARY_ROUND21.csv",
        "j": plot / "ed01_j_group_specific_80_time_session_bins.csv",
        "k": plot / "ed01_k_group_specific_80_tile_session_bins.csv",
    }
    m.SOURCE_FILES = {
        key: (path, "record-local maintained-derived source")
        for key, path in source_map.items()
    }
    m.PANEL_SCOPE = {
        key: ("accepted record", "record-local maintained-derived source")
        for key in source_map
    }
    m.PANEL_SCOPE["j"] = ("accepted record", "not promoted primary")
    m.PANEL_SCOPE["k"] = ("accepted record", "not promoted primary")
    m.ACCUMULATION = source_map["b-c"]
    m.TILE_CURVES = source_map["d-e"]
    m.ENDPOINT_ROWS = source_map["f-rows"]
    m.ENDPOINT_STATS = source_map["f-stats"]
    m.TILE_TIME_ROWS = source_map["g-rows"]
    m.TILE_TIME_STATS = source_map["g-stats"]
    m.SPEED_ROWS = source_map["h-rows"]
    m.SPEED_REFERENCE = source_map["h-reference"]
    m.EFFICIENCY_TIME = source_map["j"]
    m.EFFICIENCY_TILE = source_map["k"]
    m.SPEED_WINDOW_ID = "first_30_min"
    original_draw_speed_axes = m.draw_speed_axes

    def draw_speed_axes_current(subgrid: Any, summary: pd.DataFrame) -> list[Any]:
        axes = original_draw_speed_axes(subgrid, summary)
        for axis in axes:
            axis.set_ylim(0.0, 14.0)
            axis.set_yticks(np.arange(0.0, 14.1, 2.0))
        return axes

    # Panel h uses the same fixed 0-14 cm/s scale in both axes.
    m.draw_speed_axes = draw_speed_axes_current
    data = {
        "accumulation": pd.read_csv(source_map["b-c"], low_memory=False),
        "tile_curves": pd.read_csv(source_map["d-e"]),
        "endpoint_rows": pd.read_csv(source_map["f-rows"]),
        "endpoint_stats": pd.read_csv(source_map["f-stats"]),
        "tile_time_rows": pd.read_csv(source_map["g-rows"], low_memory=False),
        "tile_time_stats": pd.read_csv(source_map["g-stats"]),
        "speed_rows": pd.read_csv(source_map["h-rows"], dtype={"animal_id": str}),
        "speed_reference": pd.read_csv(source_map["h-reference"]),
        "example_sequence": pd.read_csv(drawn / "PANEL_I_END_NODE_SEQUENCE_ROUND21.csv"),
        "example_cycles": pd.read_csv(drawn / "PANEL_I_CYCLE_SUMMARY_ROUND21.csv"),
        "efficiency_time": pd.read_csv(source_map["j"]),
        "efficiency_tile": pd.read_csv(source_map["k"]),
    }
    m.configure_style()
    display_successor = stage / "scripts/render_ed01_no_trajectories_source.py"
    if not display_successor.is_file():
        raise FileNotFoundError("The current ten-panel ED1 display adapter is required")
    display = load_module(display_successor, "ed1_round23_display_source")
    return display.render(m, data, output)


def render_ed2(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    module = load_module(stage / "scripts/render_ed2.py", "ed2_current_renderer")
    return module.render(record, output, 600)


def render_ed3(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_ed03_reward_path_fluency_assembly.py", "ed3_source_rebuild")
    plot = record / "plot_ready"
    m.REPO_ROOT = REPO_ROOT
    m.load_fluency_module = lambda: load_module(
        stage / "scripts/render_ed04_fluency_story_current.py",
        "ed3_source_rebuild_fluency",
    )
    m.FLUENCY_ROWS = plot / "fluency_plot_rows.csv"
    m.FLUENCY_STATS = plot / "fluency_statistics.csv"
    m.DEPTH_TABLE = plot / "ed04_depth_small_multiples_current_plus_new12.csv"
    m.ROUTE_DEFINITION = plot / "ed04_route_definition_current_plus_new12.csv"
    result = m.run(output)
    if result["validation_status"] != "pass":
        raise RuntimeError("ED3 source renderer did not pass")
    return {"status": "PASS", "validation": result["validation"]}


def render_ed4(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_ed04_registration_quality_assembly.py", "ed4_source_rebuild")
    m.REPO_ROOT = REPO_ROOT
    m.SOURCE = record / "source_from_raw"
    m.ALIGNMENT = m.SOURCE
    m.LOCAL_CORRELATION_IMAGE = record / "embedded_sources/ed05_four_session_418_local_correlation_paper.png"
    m.COREGISTRATION_IMAGE = record / "embedded_sources/ed05_four_session_418_coregistration_paper.png"
    m.LEGACY_FOV_PANEL = m.LOCAL_CORRELATION_IMAGE
    original_canvas_save = m.Canvas.save

    def canvas_save_compatibility(
        canvas: Any, stem: Path, *, vector_dpi: float | None = None
    ) -> list[Path]:
        if vector_dpi is not None:
            canvas.fig.set_dpi(vector_dpi)
        return original_canvas_save(canvas, stem)

    m.Canvas.save = canvas_save_compatibility
    try:
        result = m.build(output)
    finally:
        m.Canvas.save = original_canvas_save
    if str(result["validation_status"]).upper() != "PASS":
        raise RuntimeError("ED4 source renderer did not pass")
    return {"status": "PASS", "validation": result["validation"]}


def render_ed5(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_ed05_example_place_maps_assembly.py", "ed5_source_rebuild")
    plot = record / "plot_ready"
    m.REPO_ROOT = REPO_ROOT
    m.DAY_COMB_STATS = plot / "ed05_day_combination_session_pair_contrasts_cr1_bh3.csv"
    m.DAY_COMB_DESC = plot / "ed05_day_combination_session_pair_descriptive_cr1.csv"
    m.EQUAL_PAIRS = plot / "ed05e_context_only_all_outside_pair_rows.csv"
    m.EQUAL_STATS = plot / "ed05e_context_only_all_outside_clustered_stats.csv"
    m.PLACE_PAIR_SOURCE = plot / "place_pair_summary_corrected36_all_variants.csv"
    m.PV_PAIR_SOURCE = plot / "pv_pair_summary_corrected36_all_variants.csv"
    m.RATE_PAIR_SOURCE = plot / "rate_pair_summary_corrected36_all_variants.csv"
    m.SELECTED_32 = plot / "selected_32.csv"
    m.SHORTLIST_NPZ = plot / "selected_maps_fields_45_shortlist32.npz"
    m.EQUAL_EXTRA_SOURCES = ()
    # The frozen renderer writes and rereads its publication-view companions in
    # ``output_dir.parent``. Nest its assembly directory so those fresh products
    # remain inside the isolated render root instead of beside it.
    result = m.assemble(output / "assembly")
    if str(result["status"]).upper() != "PASS":
        raise RuntimeError("ED5 source renderer did not pass")
    return {"status": "PASS", "checks": result["validation"]}


def render_ed6(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_ed06_within_session_convergence_assembly.py", "ed6_source_rebuild")
    plot = record / "plot_ready"
    m.REPO_ROOT = REPO_ROOT
    m.FIXED_CURVES = plot / "ed07_current59_group_curve_summary.csv"
    m.FIXED_STATS = plot / "ed07_current59_clustered_stats.csv"
    m.FIXED_SUPPORT = plot / "ed07_current59_curve_support_mask.csv"
    m.PIVOT_CURVES = plot / "ed07_pivot_experience_group_curve_summary_current59.csv"
    m.PIVOT_STATS = plot / "ed06_available_curve_clustered_stats.csv"
    m.PIVOT_LIMITS = plot / "ed07_pivot_experience_display_limits_current59.csv"
    m.FIGURE_ROOT = output
    m.DEFAULT_OUT = output
    result = m.render(output, 7)
    if str(result["validation_status"]).upper() != "PASS":
        raise RuntimeError("ED6 source renderer did not pass")
    return {"status": "PASS", "validation": result["validation"]}


def render_ed7(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    from ca_analysis.figure_colors import (
        CORRELATION_EXPERIENCE_COLORS,
        CORRELATION_MEASURE_COLORS,
    )
    from PIL import Image

    power = load_module(
        stage / "scripts/render_ed07_slide9_powerpoint_matched.py",
        "ed7_source_rebuild_powerpoint_matched",
    )
    plot = record / "plot_ready"
    first30 = power.normalize_first30_plot(
        pd.read_csv(
            plot / "ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",
            dtype=str,
        )
    )
    first30 = first30.loc[
        first30["source_scope"].eq(power.MAINTAINED_FIRST30_SCOPE)
    ].copy()
    first30_models = pd.read_csv(
        plot / "ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv"
    )
    long_gap = pd.read_csv(
        plot / "ed09_long_gap_age_joined_corrected36.csv",
        dtype={"animal_id": str},
    )
    long_models = pd.read_csv(
        plot / "ed09_long_gap_age_transition_fixed_models_corrected36.csv"
    )
    displayed_metrics = {spec["metric"] for spec in power.METRICS}
    long_gap = long_gap.loc[long_gap["metric"].isin(displayed_metrics)].copy()
    long_models = long_models.loc[
        long_models["metric"].isin(displayed_metrics)
    ].copy()

    stage_keys = {0: "naive", 1: "3mo", 2: "6mo", 3: "9mo", 4: "12mo"}
    palettes = {
        spec["key"]: {
            index: CORRELATION_EXPERIENCE_COLORS[spec["metric"]][stage_key]
            for index, stage_key in stage_keys.items()
        }
        for spec in power.METRICS
    }
    bases = {
        spec["key"]: CORRELATION_MEASURE_COLORS[spec["metric"]]
        for spec in power.METRICS
    }
    power.configure_style()
    top_stem = output / "ed07_panels_abc_first30_to_day2_maintained59_round17"
    bottom_stem = output / "ed07_panels_def_long_gap_clean36_round17"
    assembly_stem = output / "extended_data_figure7_source_rebuild"
    top = power.render_first30_row(
        first30,
        first30_models,
        palettes,
        bases,
        top_stem,
        model_specification="maintained59_primary",
    )
    bottom = power.render_long_gap_row(
        long_gap,
        long_models,
        palettes,
        bases,
        bottom_stem,
        bottom_scope="clean36",
    )
    assembly = power.render_six_panel_assembly(
        first30,
        first30_models,
        long_gap,
        long_models,
        palettes,
        bases,
        assembly_stem,
        bottom_scope="clean36",
        first30_model_specification="maintained59_primary",
    )

    expected_outputs = {
        stem.with_suffix(f".{extension}")
        for stem in (top_stem, bottom_stem, assembly_stem)
        for extension in FORMATS
    }
    missing_outputs = sorted(str(path) for path in expected_outputs if not path.is_file())
    if missing_outputs:
        raise RuntimeError(f"ED7 source renderer omitted outputs: {missing_outputs}")
    with Image.open(assembly_stem.with_suffix(".png")) as image:
        png_size = [image.width, image.height]
        png_dpi = [round(float(value)) for value in image.info.get("dpi", (0, 0))]
    with Image.open(assembly_stem.with_suffix(".tiff")) as image:
        tiff_size = [image.width, image.height]
        tiff_dpi = [round(float(value)) for value in image.info.get("dpi", (0, 0))]
    svg_text = assembly_stem.with_suffix(".svg").read_text(
        encoding="utf-8", errors="replace"
    )
    with fitz.open(assembly_stem.with_suffix(".pdf")) as document:
        pdf_pages = len(document)
        pdf_size_points = [document[0].rect.width, document[0].rect.height]
    validation = {
        "status": "PASS",
        "first30_rows": len(first30),
        "first30_mice": int(first30["animal_id"].nunique()),
        "long_gap_rows": len(long_gap),
        "long_gap_mice": int(long_gap["animal_id"].nunique()),
        "png_size": png_size,
        "png_dpi": png_dpi,
        "tiff_size": tiff_size,
        "tiff_dpi": tiff_dpi,
        "pdf_pages": pdf_pages,
        "pdf_size_points": pdf_size_points,
        "svg_text_nodes": svg_text.count("<text"),
        "svg_embedded_images": svg_text.count("<image"),
    }
    if not (
        validation["first30_rows"] == 177
        and validation["first30_mice"] == 22
        and validation["long_gap_rows"] == 108
        and validation["long_gap_mice"] == 14
        and png_size == [4500, 2952]
        and tiff_size == png_size
        and png_dpi == [600, 600]
        and tiff_dpi == [600, 600]
        and pdf_pages == 1
        and all(
            abs(observed - expected) < 0.02
            for observed, expected in zip(pdf_size_points, (540.0, 354.24))
        )
        and validation["svg_text_nodes"] > 0
        and validation["svg_embedded_images"] == 0
    ):
        validation["status"] = "FAIL"
        raise RuntimeError(f"ED7 source-render validation failed: {validation}")
    return {
        "status": "PASS",
        "_source_renderer": "source_code/render_ed07_slide9_powerpoint_matched.py",
        "validation": validation,
        "component_outputs": sorted(
            path.relative_to(output).as_posix()
            for path in (*top, *bottom, *assembly)
        ),
    }


def render_ed10(record: Path, output: Path, stage: Path) -> dict[str, Any]:
    m = load_module(stage / "scripts/render_ed10_slide12_powerpoint_matched.py", "ed10_source_rebuild")
    m.TABLE_A = record / "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv"
    m.TABLE_B = record / "plot_ready/REWARD_LEVEL_COMMONALITY.csv"
    m.PROFILE_TABLE = record / "plot_ready/REWARD_LEVEL_PROFILE_TESTS.csv"
    m.PAIR_AUDIT = record / "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
    m.TABLE_C = record / "plot_ready/level_policy_bridge_join_corrected36.csv"
    m.STATS_C = record / "plot_ready/level_policy_bridge_stats_corrected36.csv"
    round_match = re.search(r"_(round\d+)(?:_|$)", record.name)
    if round_match is None:
        raise RuntimeError(f"Cannot derive ED10 output round from record name: {record.name}")
    m.main(output_root=output, output_round=round_match.group(1))
    validation = json.loads((output / "AUDIT_VALIDATION.json").read_text(encoding="utf-8"))
    return {"status": validation["status"], "validation": validation}
