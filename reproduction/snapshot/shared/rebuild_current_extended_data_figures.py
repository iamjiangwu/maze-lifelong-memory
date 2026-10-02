#!/usr/bin/env python3
"""Rebuild pointer-selected Extended Data figures from version-record sources.

The adapter never treats an accepted master composite as a quantitative input.
It stages the frozen ``source_code`` directory in a temporary repository-shaped
workspace, redirects every scientific input to the pointer-selected immutable
record, and writes a fresh SVG/PDF/PNG/TIFF build to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Iterator

import fitz
import numpy as np
import pandas as pd


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]
POINTER = VOR_ROOT / "CURRENT_EXTENDED_DATA_FIGURES.md"
MAIN_POINTER = VOR_ROOT / "CURRENT_MAIN_FIGURES.md"
CONTRACT = VOR_ROOT / "SOURCE_REBUILD_ED_CONTRACT.json"
FORMATS = ("svg", "pdf", "png", "tiff")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return len(files), digest.hexdigest()


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def pointer_records() -> dict[str, str]:
    text = POINTER.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for number in range(1, 11):
        prefix = f"extended_data_fig{number:02d}_"
        names = {
            token.strip("`/")
            for token in text.replace("|", " ").split()
            if token.strip("`/").startswith(prefix)
        }
        if len(names) != 1:
            raise RuntimeError(
                f"Pointer must select exactly one record for ED{number}: {sorted(names)}"
            )
        found[f"extended_data_figure_{number}"] = names.pop()
    return found


def record_root(artifact: str) -> Path:
    contract = load_contract()["artifacts"][artifact]
    selected = pointer_records()[artifact]
    if selected != contract["record"]:
        raise RuntimeError(
            f"Contract/pointer drift for {artifact}: {contract['record']} != {selected}"
        )
    root = VOR_ROOT / selected
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def current_main_figure4_name() -> str:
    """Resolve the sidecar provenance record through its live pointer."""

    text = MAIN_POINTER.read_text(encoding="utf-8")
    names = {
        Path(token.strip("`/")).name
        for token in text.replace("|", " ").split()
        if Path(token.strip("`/")).name.startswith("main_figure4_")
    }
    if len(names) != 1:
        raise RuntimeError(
            f"Main-figure pointer must select exactly one Figure 4 record: {sorted(names)}"
        )
    selected = names.pop()
    bundle = load_contract()["artifacts"]["extended_data_figure_8"][
        "source_bundle"
    ]
    if selected != bundle["source_record"]:
        raise RuntimeError(
            "ED8 Main Figure 4 source-bundle/pointer drift: "
            f"{bundle['source_record']} != {selected}"
        )
    return selected


def scientific_input_roots(artifact: str, record: Path) -> list[Path]:
    return [record]


def scientific_inputs(artifact: str, record: Path) -> list[Path]:
    number = int(artifact.rsplit("_", 1)[-1])
    dirs_by_number = {
        1: ("plot_ready", "drawn_values"),
        2: ("plot_ready", "analysis_inputs", "shared_assets"),
        3: ("plot_ready",),
        4: ("source_from_raw", "embedded_sources"),
        5: ("plot_ready",),
        6: ("plot_ready",),
        7: ("plot_ready", "source_authority", "palette_authority"),
        8: (
            "plot_ready",
            "source_data",
            "source_authority",
            "source_validation",
            "cohort_audit",
        ),
        9: (
            "plot_ready",
            "source_data",
            "source_history/round4_powerpoint_parity/source_data",
        ),
        10: ("plot_ready", "source_data", "source_authority"),
    }
    paths = [path for path in (record / "source_code").rglob("*") if path.is_file()]
    for dirname in dirs_by_number[number]:
        directory = record / dirname
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(path for path in directory.rglob("*") if path.is_file())
    root_extras = {
        8: (
            "ED08_ROUND20_RENDER_MANIFEST.json",
            "ED08_VERSION_RECORD_MANIFEST.json",
        ),
        9: ("PANEL_C_BOUNDARY_SMOOTHED_ROUND34_MANIFEST.json",),
        10: (
            "ED10_VERSION_RECORD_MANIFEST.json",
            "main_figure_authority/figure5_source_manifest.json",
        ),
    }
    paths.extend(record / name for name in root_extras.get(number, ()))
    if number == 9:
        if load_contract()["artifacts"][artifact].get("panel_h_null") == "neuron_id":
            paths.extend(record / name for name in (
                "source_data/panel_h_neuron_id_shuffle_round38/threshold_clustered_statistics.csv",
                "source_code/render_ed09h_neuron_id_shuffle.py",
                "PANEL_H_NEURON_ID_SHUFFLE_ROUND38_MANIFEST.json",
            ))
        # The accepted panel-b Round-14 renderer consumes this exact maximal42
        # display snapshot; declare only its two tabular inputs, not inherited
        # source-validation artwork.
        paths.extend(
            record / "source_validation/round14_panel_c_histogram/source_data" / name
            for name in (
                "C_MAXIMAL42_SHIFT_DISTANCE_10000PERM.csv",
                "ED12_MAXIMAL42_BH6_FAMILY.csv",
            )
        )
    if number == 8:
        # Superseded copies of accepted Main Figure 4 artwork remain in the ED8
        # record for history, but are deliberately not rebuild inputs.
        retired_artwork = {
            (record / "source_authority/main_figure4_panel_c_d1.svg").resolve(),
            (record / "source_authority/main_figure4_panel_c_d1.pdf").resolve(),
        }
        paths = [path for path in paths if path.resolve() not in retired_artwork]
        bundle = load_contract()["artifacts"][artifact]["source_bundle"]
        paths.append(VOR_ROOT / bundle["path"])
    return sorted(set(paths))


def check_artifact(artifact: str) -> dict[str, Any]:
    record = record_root(artifact)
    inputs = scientific_inputs(artifact, record)
    source_roots = scientific_input_roots(artifact, record)
    sidecar_contracts: dict[Path, dict[str, Any]] = {}
    if artifact == "extended_data_figure_8":
        bundle_contract = load_contract()["artifacts"][artifact]["source_bundle"]
        sidecar_contracts[(VOR_ROOT / bundle_contract["path"]).resolve()] = bundle_contract
    ledgers: dict[Path, dict[str, dict[str, str]]] = {}
    ledger_paths: list[Path] = []
    for root in source_roots:
        ledger_path = root / "RECORD_HASHES.csv"
        ledger_paths.append(ledger_path)
        ledgers[root] = {
            row["path"].replace("\\", "/"): row
            for row in csv.DictReader(
                ledger_path.open(encoding="utf-8-sig", newline="")
            )
        }
    missing_ledger: list[str] = []
    hash_failures: list[str] = []
    for path in inputs:
        owners = [root for root in source_roots if path.is_relative_to(root)]
        if not owners and path.resolve() in sidecar_contracts:
            sidecar = sidecar_contracts[path.resolve()]
            if (
                path.stat().st_size != int(sidecar["bytes"])
                or sha256(path) != sidecar["sha256"]
            ):
                hash_failures.append(path.name)
            continue
        if len(owners) != 1:
            missing_ledger.append(path.as_posix())
            continue
        owner = owners[0]
        relative = path.relative_to(owner).as_posix()
        label = f"{owner.name}/{relative}"
        row = ledgers[owner].get(relative)
        if row is None:
            missing_ledger.append(label)
            continue
        if int(row["bytes"]) != path.stat().st_size or row["sha256"] != sha256(path):
            hash_failures.append(label)

    forbidden = tuple(
        value.lower().replace("\\", "/")
        for value in load_contract()["policy"]["forbidden_reads"]
    )
    resolved = [path.resolve().as_posix().lower() for path in inputs]
    forbidden_hits = [
        value for value in resolved if any(token in value for token in forbidden)
    ]
    master_inputs = [
        f"{root.name}/{path.relative_to(root).as_posix()}"
        for path in inputs
        for root in source_roots
        if path.parent == root
        and path.suffix.lower() in {".svg", ".pdf", ".png", ".tif", ".tiff"}
    ]
    checks: dict[str, bool] = {
        "pointer_selects_contract_record": True,
        "record_hash_ledger_exists": all(path.is_file() for path in ledger_paths),
        "all_scientific_inputs_exist": all(path.is_file() for path in inputs),
        "all_scientific_inputs_in_hash_ledger": not missing_ledger,
        "all_scientific_input_hashes_match": not hash_failures,
        "all_scientific_inputs_record_local": all(
            any(path.is_relative_to(root) for root in source_roots)
            or path.resolve() in sidecar_contracts
            for path in inputs
        ),
        "no_forbidden_input_roots": not forbidden_hits,
        "accepted_master_not_an_input": not master_inputs,
    }
    if artifact == "extended_data_figure_8":
        bundle_path, bundle_contract = next(iter(sidecar_contracts.items()))
        checks["main_figure4_source_bundle_pointer_selected"] = (
            current_main_figure4_name() == bundle_contract["source_record"]
        )
        checks["main_figure4_source_bundle_hash_matches"] = (
            bundle_path.stat().st_size == int(bundle_contract["bytes"])
            and sha256(bundle_path) == bundle_contract["sha256"]
        )
        with zipfile.ZipFile(bundle_path) as archive:
            observed_members = {
                name: {
                    "bytes": archive.getinfo(name).file_size,
                    "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
                }
                for name in archive.namelist()
                if not name.endswith("/")
            }
        checks["main_figure4_source_bundle_members_match"] = (
            observed_members == bundle_contract["files"]
        )
    runtime_contract = load_contract()["shared_runtime"]
    runtime_root = VOR_ROOT / runtime_contract["path"]
    with zipfile.ZipFile(runtime_root) as archive:
        runtime_count = len([name for name in archive.namelist() if name.endswith(".py")])
    runtime_hash = sha256(runtime_root)
    checks["shared_runtime_is_vor_local"] = runtime_root.is_relative_to(VOR_ROOT)
    checks["shared_runtime_tree_hash_matches"] = (
        runtime_count == int(runtime_contract["python_files"])
        and runtime_hash == runtime_contract["sha256"]
    )
    code_sidecars = load_contract()["artifacts"][artifact].get("code_sidecars", [])
    code_sidecar_failures: list[str] = []
    for sidecar in code_sidecars:
        path = VOR_ROOT / sidecar["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(sidecar["bytes"])
            or sha256(path) != sidecar["sha256"]
        ):
            code_sidecar_failures.append(sidecar["path"])
    checks["code_sidecars_are_vor_local_and_hash_match"] = not code_sidecar_failures
    if artifact == "extended_data_figure_9":
        alpha = load_contract()["artifacts"][artifact]["panel_c_smoothing_alpha"]
        source = (
            record
            / "source_code/build_ed09_boundary_smoothed_panel_c_round34.py"
        ).read_text(encoding="utf-8")
        checks["panel_c_alpha_exactly_0_35"] = (
            np.isclose(alpha, 0.35) and '"alpha": 0.35' in source
        )
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": "extended-data-source-rebuild-check/v1",
        "status": status,
        "artifact": artifact,
        "record": record.name,
        "source_records": [root.name for root in source_roots],
        "source_sidecars": [path.name for path in sidecar_contracts],
        "scientific_input_files": len(inputs),
        "checks": checks,
        "missing_ledger_entries": missing_ledger,
        "hash_failures": hash_failures,
        "forbidden_hits": forbidden_hits,
        "master_inputs": master_inputs,
        "code_sidecars": [sidecar["path"] for sidecar in code_sidecars],
        "code_sidecar_failures": code_sidecar_failures,
        "shared_runtime": {
            "path": runtime_root.relative_to(VOR_ROOT).as_posix(),
            "files": runtime_count,
            "sha256": runtime_hash,
        },
    }


@contextlib.contextmanager
def frozen_stage(record: Path) -> Iterator[Path]:
    # Keep the stage prefix short enough for the historical nested ED8 paths on
    # Windows, where long manifest filenames can otherwise exceed MAX_PATH.
    with tempfile.TemporaryDirectory(prefix="edsrc_") as raw:
        stage = Path(raw)
        shutil.copytree(record / "source_code", stage / "scripts")
        runtime_archive = VOR_ROOT / load_contract()["shared_runtime"]["path"]
        runtime_root = stage / "shared_runtime"
        runtime_root.mkdir()
        with zipfile.ZipFile(runtime_archive) as archive:
            archive.extractall(runtime_root)
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


def write_build_manifest(
    artifact: str,
    record: Path,
    output: Path,
    renderer: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    outputs = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.suffix.lower().lstrip(".") in FORMATS:
            outputs.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    suffixes = {Path(row["path"]).suffix.lower().lstrip(".") for row in outputs}
    manifest = {
        "schema": "extended-data-record-local-source-rebuild/v1",
        "status": "PASS" if set(FORMATS).issubset(suffixes) else "FAIL",
        "artifact": artifact,
        "record": record.name,
        "record_path": record.relative_to(REPO_ROOT).as_posix(),
        "renderer": renderer.relative_to(record).as_posix(),
        "renderer_sha256": sha256(renderer),
        "quantitative_source_policy": (
            "record-local maintained-derived tables/arrays only; accepted master "
            "composite is not read"
        ),
        "fixed_assets": load_contract()["artifacts"][artifact]["fixed_assets"],
        "output_formats": sorted(suffixes),
        "outputs": outputs,
        "renderer_result": result,
    }
    path = output / "SOURCE_REBUILD_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    if manifest["status"] != "PASS":
        raise RuntimeError(f"Missing publication output format(s): {set(FORMATS) - suffixes}")
    return manifest


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


def stage_ed8_chain(record: Path, stage: Path) -> Path:
    ed8 = stage / "docs/extended_data_supplement_v2_rebuild_2026-08-05/ed08_reward_proximal_controls"
    ed8.mkdir(parents=True)
    shutil.copy2(stage / "scripts/render_ed08_reward_proximal_controls.py", ed8 / "render_ed08_reward_proximal_controls.py")
    rounds = {
        3: "2026-08-09",
        4: "2026-08-09",
        5: "2026-08-09",
        6: "2026-08-09",
        7: "2026-08-09",
        8: "2026-08-09",
        9: "2026-08-10",
        10: "2026-08-11",
    }
    for number, date in rounds.items():
        directory = ed8 / f"revision_round{number}_{date}"
        directory.mkdir()
        shutil.copy2(stage / f"scripts/render_ed08_round{number}.py", directory / f"render_ed08_round{number}.py")
    for number in (8, 9, 10):
        source = record / "source_validation" / f"ED08_ROUND{number}_RENDER_MANIFEST.json"
        date = rounds[number]
        shutil.copy2(source, ed8 / f"revision_round{number}_{date}" / source.name)
    correction = ed8 / "correction_2026-08-09/plot_ready"
    correction.mkdir(parents=True)
    for source in (record / "plot_ready").iterdir():
        if source.is_file():
            shutil.copy2(source, correction / source.name)
    contract = record / "source_authority/ED08_NEURAL_EXCLUSION_CONTRACT.json"
    round4 = ed8 / "revision_round4_2026-08-09"
    round4_plot = round4 / "neural_exclusion_rebuild/plot_ready"
    shutil.copytree(record / "plot_ready", round4_plot)
    shutil.copy2(contract, round4 / "ED08_NEURAL_EXCLUSION_CONTRACT.json")
    shutil.copy2(contract, stage / "docs/EXCLUDED_SESSION_CONTRACT.json")
    # Rebuild the Main Figure 4c D1 grammar authority from its current
    # pointer-selected numeric rows.  The ED8 record's copied SVG/PDF and the
    # accepted Main Figure 4 composite/panel artwork are intentionally unread.
    main4_bundle_contract = load_contract()["artifacts"]["extended_data_figure_8"][
        "source_bundle"
    ]
    main4_bundle = VOR_ROOT / main4_bundle_contract["path"]
    main4_bundle_stage = stage / "main4_source_bundle"
    main4_bundle_stage.mkdir()
    with zipfile.ZipFile(main4_bundle) as archive:
        archive.extractall(main4_bundle_stage)
    main4_source = main4_bundle_stage / "main4_panel_c"
    main4 = load_module(
        main4_source / "source_code/render_figure4_day_source_comparison.py",
        "ed8_main4_map_source_rebuild",
    )
    main4.configure_style()
    raw_rows = pd.read_csv(main4_source / "plot_ready/panel_c_example_map_rows.csv")
    raw_d1 = raw_rows.loc[raw_rows["variant"].eq("d1")].copy()
    generated_rows_path = stage / "main4_regenerated/figure4_panel_c_d1_source_rows.csv"
    generated_rows_path.parent.mkdir()
    figure = main4.map_panel(
        raw_d1,
        variant="d1",
        size=(main4.SLOTS["c"][2], main4.SLOTS["c"][3]),
        source_csv=generated_rows_path,
    )
    expected_rows = pd.read_csv(
        main4_source / "plot_ready/figure4_panel_c_d1_source_rows.csv"
    )
    generated_rows = pd.read_csv(generated_rows_path)
    pd.testing.assert_frame_equal(
        generated_rows.reset_index(drop=True),
        expected_rows.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )
    generated_pdf = generated_rows_path.with_suffix(".pdf")
    figure.savefig(
        generated_pdf,
        format="pdf",
        transparent=True,
        facecolor="none",
        bbox_inches=None,
    )
    generated_svg = generated_rows_path.with_suffix(".svg")
    generated_svg.write_bytes(main4.figure_svg(figure))
    validation = {
        "schema": "ed8-main4-map-grammar-source-rebuild/v1",
        "status": "PASS",
        "source_record": main4_bundle_contract["source_record"],
        "source_bundle": main4_bundle.name,
        "source_bundle_sha256": sha256(main4_bundle),
        "accepted_artwork_read": False,
        "numeric_rows_exact_current_record_parity": True,
        "inputs": {
            "raw_rows_sha256": sha256(
                main4_source / "plot_ready/panel_c_example_map_rows.csv"
            ),
            "expected_derived_rows_sha256": sha256(
                main4_source / "plot_ready/figure4_panel_c_d1_source_rows.csv"
            ),
            "renderer_sha256": sha256(
                main4_source / "source_code/render_figure4_day_source_comparison.py"
            ),
        },
    }
    # Historical wrapper names are administrative aliases only; every alias is
    # populated with this fresh numeric-row render.
    aliases = {
        "main_figure4_2026-08-08_full_audit": (
            "figure4_panel_a_d1",
            "figure4_panel_c_d1",
        ),
        "main_figure4_2026-08-11_round2_place_maps_label": (
            "figure4_panel_c_d1",
        ),
    }
    for record_name, stems in aliases.items():
        alias = stage / "docs/figure_version_records" / record_name
        panels = alias / "individual_panels"
        panels.mkdir(parents=True)
        for stem in stems:
            shutil.copy2(generated_svg, panels / f"{stem}.svg")
            shutil.copy2(generated_pdf, panels / f"{stem}.pdf")
        (alias / "RECORD_VALIDATION.json").write_text(
            json.dumps(validation, indent=2) + "\n",
            encoding="utf-8",
        )
    return ed8










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


RENDERERS = {
    1: render_ed1,
    2: render_ed2,
    3: render_ed3,
    4: render_ed4,
    5: render_ed5,
    6: render_ed6,
    7: render_ed7,
}


def render_artifact(artifact: str, output_dir: Path) -> dict[str, Any]:
    check = check_artifact(artifact)
    if check["status"] != "PASS":
        raise RuntimeError(f"Source-rebuild check failed: {check}")
    record = record_root(artifact)
    output = output_dir.resolve() / artifact
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    number = int(artifact.rsplit("_", 1)[-1])
    # Several frozen renderers record paths relative to the repository. Render
    # in a disposable repository-local directory, then copy the new products
    # to the caller's arbitrary destination (including another drive).
    scratch_parent = REPO_ROOT / ".tmp"
    scratch_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{artifact}_", dir=scratch_parent
    ) as raw_render:
        render_output = Path(raw_render) / "render"
        render_output.mkdir()
        modules_before = set(sys.modules)
        with frozen_stage(record) as stage:
            opened_for_read: set[Path] = set()
            audit_active = True

            def audit_hook(event: str, args: tuple[Any, ...]) -> None:
                nonlocal audit_active
                if not audit_active or event != "open" or not args:
                    return
                value = args[0]
                if not isinstance(value, (str, bytes, os.PathLike)):
                    return
                mode = args[1] if len(args) > 1 else "r"
                if isinstance(mode, str) and any(token in mode for token in "wax+"):
                    return
                try:
                    opened_for_read.add(Path(value).resolve())
                except (OSError, TypeError, ValueError):
                    return

            sys.addaudithook(audit_hook)
            result = RENDERERS[number](record, render_output, stage)
            audit_active = False
            new_local_modules: list[str] = []
            for name in sorted(set(sys.modules) - modules_before):
                origin = getattr(sys.modules.get(name), "__file__", None)
                if not origin:
                    continue
                path = Path(origin).resolve()
                # Some binary packages expose synthetic relative ``__file__``
                # labels (for example shiboken's embedded support modules).
                # Only an existing file can constitute a live-repository import.
                if (
                    path.is_file()
                    and path.is_relative_to(REPO_ROOT)
                    and not path.is_relative_to(VOR_ROOT)
                ):
                    new_local_modules.append(path.relative_to(REPO_ROOT).as_posix())
            if new_local_modules:
                raise RuntimeError(
                    "Frozen renderer imported live repository code outside the VOR root: "
                    + ", ".join(new_local_modules)
                )
            live_repo_reads = sorted(
                path.relative_to(REPO_ROOT).as_posix()
                for path in opened_for_read
                if path.is_file()
                and path.is_relative_to(REPO_ROOT)
                and not path.is_relative_to(VOR_ROOT)
                and not path.is_relative_to(render_output)
            )
            if live_repo_reads:
                raise RuntimeError(
                    "Frozen renderer read live repository files outside the VOR root: "
                    + ", ".join(live_repo_reads)
                )
            declared_inputs = {
                path.resolve() for path in scientific_inputs(artifact, record)
            }
            vor_record_reads: list[tuple[Path, Path]] = []
            for path in opened_for_read:
                if not path.is_file() or not path.is_relative_to(VOR_ROOT):
                    continue
                relative = path.relative_to(VOR_ROOT)
                if not relative.parts:
                    continue
                possible_record = VOR_ROOT / relative.parts[0]
                if (possible_record / "RECORD_HASHES.csv").is_file():
                    vor_record_reads.append((possible_record, path))
            undeclared_vor_record_reads = sorted(
                f"{owner.name}/{path.relative_to(owner).as_posix()}"
                for owner, path in vor_record_reads
                if path not in declared_inputs
            )
            if undeclared_vor_record_reads:
                raise RuntimeError(
                    "Frozen renderer read VOR-record files omitted from scientific_inputs: "
                    + ", ".join(undeclared_vor_record_reads)
                )
            record_reads = sorted(
                path.relative_to(record).as_posix()
                for path in opened_for_read
                if path.is_file() and path.is_relative_to(record)
            )
            contract_artifact = load_contract()["artifacts"][artifact]
            allowed_artwork = {
                str(value).replace("\\", "/")
                for key in ("fixed_assets", "validation_only_artwork")
                for value in contract_artifact.get(key, [])
                if "/" in str(value)
            }
            artwork_reads = {
                value
                for value in record_reads
                if Path(value).suffix.lower() in {".svg", ".pdf", ".png", ".tif", ".tiff"}
            }
            unknown_artwork = sorted(artwork_reads - allowed_artwork)
            if unknown_artwork:
                raise RuntimeError(
                    "Renderer read accepted artwork not declared fixed/validation-only: "
                    + ", ".join(unknown_artwork)
                )
            result["runtime_isolation"] = {
                "status": "PASS",
                "live_repository_modules_outside_vor": [],
                "live_repository_file_reads_outside_vor": [],
                "shared_runtime": load_contract()["shared_runtime"],
                "record_files_read": record_reads,
                "vor_record_files_read": sorted(
                    f"{owner.name}/{path.relative_to(owner).as_posix()}"
                    for owner, path in vor_record_reads
                ),
                "undeclared_vor_record_file_reads": [],
                "record_artwork_reads": sorted(artwork_reads),
                "validation_only_artwork": contract_artifact.get(
                    "validation_only_artwork", []
                ),
            }
        shutil.copytree(render_output, output, dirs_exist_ok=True)
    renderer = record / result.pop(
        "_source_renderer",
        load_contract()["artifacts"][artifact]["renderer"],
    )
    return write_build_manifest(artifact, record, output, renderer, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        required=True,
        choices=tuple(f"extended_data_figure_{number}" for number in range(1, 11)),
    )
    parser.add_argument("--check", action="store_true", help="Validate record-local source closure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Parent directory for a fresh <artifact>/ rebuild. Existing outputs are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.check and args.output_dir is None:
        raise SystemExit("Specify --check and/or --output-dir")
    payload: dict[str, Any] = {}
    if args.check:
        payload["check"] = check_artifact(args.artifact)
        if payload["check"]["status"] != "PASS":
            print(json.dumps(payload, indent=2))
            return 1
    if args.output_dir is not None:
        payload["build"] = render_artifact(args.artifact, args.output_dir)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
