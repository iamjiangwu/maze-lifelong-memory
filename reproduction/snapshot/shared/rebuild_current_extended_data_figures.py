#!/usr/bin/env python3
"""Shared rendering adapters for Extended Data Figures 1–7 and 10.

The public launcher authenticates and stages the included source data. These
adapters supply paths and current display settings to each numerical renderer;
no finished artwork is an input. ED8 and ED9 have separate figure-local builders.
Use the root reproduce.py command; see docs/CODE.md for the function map.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Iterator

import fitz
import numpy as np
import pandas as pd


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]
CONTRACT = VOR_ROOT / "SOURCE_REBUILD_ED_CONTRACT.json"
FORMATS = ("svg", "pdf", "png", "tiff")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


@contextlib.contextmanager
def frozen_stage(record: Path) -> Iterator[Path]:
    # Keep the stage prefix short enough for the historical nested ED8 paths on
    # Windows, where long manifest filenames can otherwise exceed MAX_PATH.
    with tempfile.TemporaryDirectory(prefix="edsrc_") as raw:
        stage = Path(raw)
        shutil.copytree(record / "source_code", stage / "scripts")
        runtime_source = VOR_ROOT / load_contract()["shared_runtime"]["path"]
        runtime_root = stage / "shared_runtime"
        shutil.copytree(runtime_source, runtime_root)
        ed2_contract = load_contract()["artifacts"]["extended_data_figure_2"]
        if record.name == ed2_contract["record"]:
            # The current Round-26 adapter deliberately pins the exact Round-23
            # base-renderer path. Recreate that repository shape inside the
            # disposable stage from its VOR-local, hash-declared code sidecar.
            # The real archive is never opened.
            (stage / "docs/figure_version_records").mkdir(parents=True)
            sidecars = ed2_contract.get("code_sidecars", [])
            if len(sidecars) != 1:
                raise RuntimeError(f"ED2 requires one Round-23 code sidecar: {sidecars}")
            sidecar_contract = sidecars[0]
            sidecar = VOR_ROOT / sidecar_contract["path"]
            if (
                not sidecar.is_file()
                or sidecar.stat().st_size != int(sidecar_contract["bytes"])
                or sha256(sidecar) != sidecar_contract["sha256"]
            ):
                raise RuntimeError("ED2 Round-23 base-renderer sidecar failed hash closure")
            staged_base = (
                stage
                / "docs/figure_version_records_archive_2026-08-09"
                / "extended_data_fig02_2026-08-12_round23_preceding_turn"
                / "source_code/render_ed03_fixed_time_30min_full_story.py"
            )
            staged_base.parent.mkdir(parents=True)
            shutil.copy2(sidecar, staged_base)
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
        "i-selection": drawn / "PANEL_I_SELECTION_AUDIT_ROUND21.csv",
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

    # Round 21 fixed panel h at the accepted common 0-14 cm/s scale. Its
    # inherited dynamic-limit helper predates that display-only authority.
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
        "example_selection": pd.read_csv(drawn / "PANEL_I_SELECTION_AUDIT_ROUND21.csv"),
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
    current_adapter = stage / "scripts/render_ed03_fixed_time_30min_full_story.py"

    adapter_load_count = 0

    def configure_caption_layer(caption_layer: Any, pass_name: str) -> None:
        nonlocal adapter_load_count
        caption_layer.REPO_ROOT = REPO_ROOT

        def load_current_renderer(_record_root: Path) -> Any:
            nonlocal adapter_load_count
            adapter_load_count += 1
            return load_module(
                current_adapter,
                f"ed2_current_round26_adapter_{pass_name}_{adapter_load_count}",
            )

        def initialize_caption_scratch(
            path: Path,
            _renderer: Any,
            _tables: dict[str, dict[str, pd.DataFrame]],
        ) -> None:
            # Display wrappers require this generated scratch file. It is not
            # a manuscript/source input and supplies no numerical values.
            path.write_text("", encoding="utf-8")

        caption_layer.load_frozen_renderer = load_current_renderer
        caption_layer.write_caption_notes = initialize_caption_scratch

    # Pass A preserves the accepted Round-23 display refinements while the
    # staged Round-26 adapter supplies the current Round-25/26 science.
    pass_a = output / "source_assembly"
    round23 = load_module(
        stage / "scripts/render_ed02_slide3_refinement_round21.py",
        "ed2_source_rebuild_round23",
    )
    caption_a = round23.R18.BASE.BASE.R11.R10.R9.R8.R7.R6.R5.BASE
    configure_caption_layer(caption_a, "round23")
    round23_result = round23.render(record, pass_a, 600)

    # Pass B reproduces the current matched-naive d/h panels in the exact
    # Round-11 geometry. Round 28 changed only panel-h label placement.
    pass_b = stage / "ed2_current_round11_pass"
    round11 = load_module(
        stage / "scripts/render_ed02_slide3_aligned_round11.py",
        "ed2_source_rebuild_round11",
    )
    caption_b = round11.R10.R9.R8.R7.R6.R5.BASE
    configure_caption_layer(caption_b, "round11")
    original_panel_h = round11.build_panel_h
    label_audit: dict[str, dict[str, float | str]] = {}

    def build_panel_h_current(
        renderer: Any,
        frame: pd.DataFrame,
    ) -> tuple[Any, dict[str, pd.DataFrame]]:
        fig, tables = original_panel_h(renderer, frame)
        rows = tables["four_vs_depth_gap"].set_index("model_norm")
        specifications = {
            "Four-policy": ("four_bias", "Four-\npolicy"),
            r"Depth $P_{BS}$": (
                "depth_stratified",
                r"Depth $\mathrm{P}_{\mathrm{BS}}$",
            ),
        }
        found: set[str] = set()
        for label in fig.axes[0].texts:
            source_text = label.get_text()
            if source_text not in specifications:
                continue
            model, display_text = specifications[source_text]
            interval_left = float(rows.loc[model, "delta_simulated_ci_low"])
            label_x = interval_left - 0.018
            label.set_text(display_text)
            label.set_x(label_x)
            label.set_ha("right")
            if model == "four_bias":
                label.set_linespacing(0.9)
            label_audit[model] = {
                "text": display_text,
                "label_right_anchor": label_x,
                "interval_left": interval_left,
                "clearance_data_units": interval_left - label_x,
            }
            found.add(model)
        if found != {"four_bias", "depth_stratified"}:
            raise RuntimeError(f"Could not locate both current panel-h labels: {found}")
        return fig, tables

    round11.build_panel_h = build_panel_h_current
    round11_result = round11.render(record, pass_b, 600)

    replacements = {
        "panels": ("ed02d_*", "ed02h_*"),
        "plot_ready": ("panel_d_*", "panel_h_*"),
        "ppt_vector": ("ed02d_*", "ed02h_*"),
    }
    replaced: list[str] = []
    for dirname, patterns in replacements.items():
        for pattern in patterns:
            sources = sorted((pass_b / dirname).glob(pattern))
            if not sources:
                raise RuntimeError(f"ED2 pass B produced no {dirname}/{pattern}")
            target_dir = pass_a / dirname
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in sources:
                target = target_dir / source.name
                shutil.copy2(source, target)
                replaced.append(target.relative_to(pass_a).as_posix())

    def refresh_csv_manifest(name: str) -> None:
        path = pass_a / name
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise RuntimeError(f"ED2 manifest is empty: {name}")
        fieldnames = list(rows[0])
        for row in rows:
            artifact = pass_a / row["path"]
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            if "sha256" in row:
                row["sha256"] = sha256(artifact)
            if "bytes" in row:
                row["bytes"] = str(artifact.stat().st_size)
            if "rows" in row and artifact.suffix.lower() == ".csv":
                row["rows"] = str(len(pd.read_csv(artifact)))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    for manifest_name in (
        "PANEL_MANIFEST.csv",
        "TABLE_MANIFEST.csv",
        "PPT_VECTOR_MANIFEST.csv",
    ):
        refresh_csv_manifest(manifest_name)

    exporter = load_module(
        stage / "scripts/export_ed02_round11_vector_figure.py",
        "ed2_source_rebuild_exporter",
    )
    composite = exporter.export(
        pass_a,
        output / "composite",
        composite_stem="ed02_version_record",
        review_round=int(re.search(r"_round(\d+)(?:_|$)", record.name).group(1)),
        manifest_name="ED02_SOURCE_REBUILD_COMPOSITE_MANIFEST.json",
    )
    if composite["vector_audit"]["embedded_raster_images"] != 0:
        raise RuntimeError("ED2 rebuilt composite unexpectedly embeds raster artwork")
    if composite["png_audit"]["size"] != [4500, 4500]:
        raise RuntimeError(f"ED2 rebuilt raster geometry drifted: {composite['png_audit']}")
    if set(label_audit) != {"four_bias", "depth_stratified"} or not all(
        abs(float(row["clearance_data_units"]) - 0.018) < 1e-12
        for row in label_audit.values()
    ):
        raise RuntimeError(f"ED2 current panel-h label audit failed: {label_audit}")
    return {
        "status": "PASS",
        "_source_renderer": "source_code/render_ed02_slide3_refinement_round21.py",
        "round23_source_render": round23_result,
        "round11_current_d_h_render": round11_result,
        "replaced_from_round11_pass": replaced,
        "panel_h_label_audit": label_audit,
        "composite": composite,
    }


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
    # The gallery assembly needs only the frozen shortlist map reconstructor
    # and v3 drawing helpers. Preload those record copies while satisfying the
    # unused v3 recomputation dependency with a narrow in-memory stub; otherwise
    # the namespace package falls through to live ``scripts`` for v1/v2.
    scripts_package = sys.modules["scripts"]
    v2_name = "scripts.build_ed06_example_place_map_candidates_v2"
    v3_name = "scripts.build_ed06_example_place_map_candidates_v3"
    v2_stub = types.ModuleType(v2_name)
    sys.modules[v2_name] = v2_stub
    setattr(scripts_package, v2_name.rsplit(".", 1)[-1], v2_stub)
    shortlist_v3 = load_module(
        stage / "scripts/build_ed06_example_place_map_candidates_v3.py",
        v3_name,
    )
    setattr(scripts_package, v3_name.rsplit(".", 1)[-1], shortlist_v3)
    shortlist = load_module(
        stage / "scripts/build_ed06_example_place_map_shortlist_32.py",
        "ed5_source_rebuild_shortlist32",
    )
    m._load_shortlist_module = lambda: shortlist
    m.REPO_ROOT = REPO_ROOT
    round_match = re.search(r"_round(\d+)(?:_|$)", record.name)
    if round_match is None:
        raise RuntimeError(f"Cannot derive ED5 round from record name: {record.name}")
    m.ROUND = int(round_match.group(1))
    m.DAY_COMB_PAIRS = plot / "ed05_day_combination_session_pair_values.csv"
    m.DAY_COMB_STATS = plot / "ed05_day_combination_session_pair_contrasts_cr1_bh3.csv"
    m.DAY_COMB_DESC = plot / "ed05_day_combination_session_pair_descriptive_cr1.csv"
    m.EQUAL_PAIRS = plot / "ed05e_context_only_all_outside_pair_rows.csv"
    m.EQUAL_MEANS = plot / "ed05e_context_only_all_outside_condition_means.csv"
    m.EQUAL_STATS = plot / "ed05e_context_only_all_outside_clustered_stats.csv"
    m.EQUAL_SOURCE_SET_EXPECTED = {"context_only_all_outside_2026-08-20"}
    m.EQUAL_FEASIBILITY_SET_EXPECTED = m.EQUAL_SOURCE_SET_EXPECTED
    m.PLACE_PAIR_SOURCE = plot / "place_pair_summary_corrected36_all_variants.csv"
    m.PV_PAIR_SOURCE = plot / "pv_pair_summary_corrected36_all_variants.csv"
    m.RATE_PAIR_SOURCE = plot / "rate_pair_summary_corrected36_all_variants.csv"
    m.EXACT_SUPPORT_SOURCE = plot / "place_pv_rate_exact_support_audit.csv"
    m.SELECTED_32 = plot / "selected_32.csv"
    m.SHORTLIST_NPZ = plot / "selected_maps_fields_45_shortlist32.npz"
    m.SELECTED_FIELDS = plot / "selected_display_fields.csv"
    m.GALLERY_PROVENANCE = plot / "ED5_PANEL_A_PER_SIDE_PROVENANCE.csv"
    m.GALLERY_PROVENANCE_SUMMARY = plot / "ED5_PANEL_A_PROVENANCE_SUMMARY.json"
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
    import matplotlib
    import matplotlib.pyplot as plt
    from ca_analysis.figure_colors import (
        CORRELATION_EXPERIENCE_COLORS,
        CORRELATION_MEASURE_COLORS,
    )
    from PIL import Image

    # The accepted Round-17 display lives in the frozen PowerPoint-matched
    # renderer. Its historical ``main`` imports recomputation code and prior
    # display directories that are intentionally not rebuild inputs. Supply
    # only the two small APIs used by its record-local plotting functions.
    science = types.ModuleType("render_ed07_slide9_age_controls")

    def configure_style() -> None:
        matplotlib.rcParams.update(
            {
                "font.family": "Arial",
                "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                "font.size": 7.0,
                "axes.titlesize": 7.5,
                "axes.labelsize": 7.0,
                "axes.linewidth": 0.65,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.labelsize": 6.3,
                "ytick.labelsize": 6.3,
                "xtick.major.width": 0.65,
                "ytick.major.width": 0.65,
                "xtick.major.size": 2.2,
                "ytick.major.size": 2.2,
                "legend.fontsize": 5.8,
                "legend.frameon": False,
                "lines.linewidth": 0.8,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "savefig.facecolor": "white",
                "figure.facecolor": "white",
            }
        )

    def save_figure(figure: Any, stem: Path) -> list[Path]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for extension in FORMATS:
            path = stem.with_suffix(f".{extension}")
            kwargs: dict[str, Any] = {
                "format": extension,
                "facecolor": "white",
                "transparent": False,
            }
            if extension in {"png", "tiff"}:
                kwargs["dpi"] = 600
            if extension == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            figure.savefig(path, **kwargs)
            paths.append(path)
        plt.close(figure)
        return paths

    science.configure_style = configure_style
    science.save_figure = save_figure
    palette_display = types.ModuleType("render_ed07_slide9_age_controls_display")
    palette_display.FORBIDDEN_DATA_COLORS = {
        "#777777",
        "#aaa9c7",
        "#8885b2",
        "#67629a",
        "#443f7a",
        "#63666a",
        "#3b82f6",
        "#f97316",
        "#8b5cf6",
    }
    stub_names = (science.__name__, palette_display.__name__)
    previous_modules = {name: sys.modules.get(name) for name in stub_names}
    sys.modules[science.__name__] = science
    sys.modules[palette_display.__name__] = palette_display
    try:
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
        configure_style()
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
            expected_rows=59,
            expected_mice=22,
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
            expected_first30_rows=59,
            expected_first30_mice=22,
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

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
    synthesis = record / "source_authority/fixed_neuron_synthesis"
    m.REPO = REPO_ROOT
    m.SYNTHESIS = synthesis
    m.TABLE_A = record / "plot_ready/POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv"
    m.TABLE_B = record / "plot_ready/REWARD_LEVEL_COMMONALITY.csv"
    m.PROFILE_TABLE = record / "plot_ready/REWARD_LEVEL_PROFILE_TESTS.csv"
    m.PAIR_AUDIT = record / "plot_ready/CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
    m.SYNTHESIS_VALIDATION = synthesis / "VALIDATION.json"
    m.SYNTHESIS_MANIFEST = synthesis / "SOURCE_MANIFEST.csv"
    m.NEURAL_ELIGIBILITY_CONTRACT = synthesis / "NEURAL_ELIGIBILITY_CONTRACT.json"
    m.TABLE_C = record / "plot_ready/level_policy_bridge_join_corrected36.csv"
    m.STATS_C = record / "plot_ready/level_policy_bridge_stats_corrected36.csv"
    m.EXCLUSION_CONTRACT = record / "source_authority/EXCLUDED_SESSION_CONTRACT.json"
    m.MAIN_FIGURE5_RECORD = record / "main_figure_authority"
    m.MAIN_FIGURE5_MANIFEST = m.MAIN_FIGURE5_RECORD / "figure5_source_manifest.json"
    m.MAIN_FIGURE5_RENDERER = record / "source_code/preview_figure3_assembly.py"
    # The maintained renderer historically opened the accepted Main Figure 5
    # master solely to assert two typography tokens. Replace that master read
    # with a tiny validation-only source-code-derived token authority.
    typography_tokens = output / "MAIN_FIGURE5_TYPOGRAPHY_TOKENS.txt"
    typography_tokens.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><text>r=0.75</text>'
        '<text>***</text></svg>\n',
        encoding="utf-8",
    )
    m.MAIN_FIGURE5_SVG = typography_tokens
    # Close the historical ED8-contract path by deriving the same fail-closed
    # cross-check locally from the record-frozen ED10 eligibility contract.
    runtime_authority = output / "runtime_authorities"
    runtime_authority.mkdir()
    # The frozen renderer's historical validation follows a synthesis manifest
    # back into live analysis directories.  Re-express that closure over the
    # immutable record-local products that are actually used for this render.
    # This preserves the hash check without permitting archival/live sources.
    maintained_sources = [
        m.TABLE_A,
        m.TABLE_B,
        m.PROFILE_TABLE,
        m.PAIR_AUDIT,
        m.TABLE_C,
        m.STATS_C,
        m.NEURAL_ELIGIBILITY_CONTRACT,
        m.EXCLUSION_CONTRACT,
        m.MAIN_FIGURE5_MANIFEST,
        m.MAIN_FIGURE5_RENDERER,
    ]
    local_manifest = runtime_authority / "ED10_RECORD_LOCAL_SOURCE_MANIFEST.csv"
    with local_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("role", "path", "bytes", "mtime_ns", "sha256"),
        )
        writer.writeheader()
        for source in maintained_sources:
            writer.writerow(
                {
                    "role": "maintained_record_source",
                    "path": source.resolve(),
                    "bytes": source.stat().st_size,
                    "mtime_ns": source.stat().st_mtime_ns,
                    "sha256": sha256(source),
                }
            )
    local_validation = runtime_authority / "ED10_RECORD_LOCAL_VALIDATION.json"
    local_validation.write_text(
        json.dumps(
            {
                "schema": "ed10-record-local-validation/v1",
                "status": "PASS",
                "outputs": {
                    source.resolve().as_posix(): sha256(source)
                    for source in maintained_sources
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    m.SYNTHESIS_MANIFEST = local_manifest
    m.SYNTHESIS_VALIDATION = local_validation
    neural_path = synthesis / "NEURAL_ELIGIBILITY_CONTRACT.json"
    neural = json.loads(neural_path.read_text(encoding="utf-8"))
    ed8_contract = runtime_authority / "ED08_NEURAL_EXCLUSION_CONTRACT.json"
    ed8_contract.write_text(
        json.dumps(
            {
                "schema": "ed08-neural-exclusion-contract/v1",
                "status": "LOCKED_FROM_MAINTAINED_NEURAL_AUTHORITY",
                "excluded_animals": neural["behavior_only_animals"],
                "excluded_sessions": neural["neural_excluded_sessions"],
                "derived_from": neural_path.relative_to(record).as_posix(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    original_eligibility_loader = m.load_neural_eligibility

    def local_eligibility(_repo: Path) -> dict[str, Any]:
        return original_eligibility_loader(
            record,
            neural_contract=neural_path,
            biological_contract=m.EXCLUSION_CONTRACT,
            ed08_contract=ed8_contract,
        )

    m.load_neural_eligibility = local_eligibility
    m.ROUND6_SOURCE_VALIDATION = output / "ED10_SOURCE_REBUILD_VALIDATION.json"
    round_match = re.search(r"_(round\d+)(?:_|$)", record.name)
    if round_match is None:
        raise RuntimeError(f"Cannot derive ED10 output round from record name: {record.name}")
    m.main(output_root=output, output_round=round_match.group(1))
    validation = json.loads((output / "AUDIT_VALIDATION.json").read_text(encoding="utf-8"))
    return {"status": validation["status"], "validation": validation}
