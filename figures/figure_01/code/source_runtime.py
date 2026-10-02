"""Rebuild the pointer-selected Main Figures from their version-record sources.

This entry point is intentionally stored beside ``CURRENT_MAIN_FIGURES.md``.
It treats each dated record as immutable and writes only to ``--output-dir``.
Quantitative panels are rendered from record-frozen CSV/JSON/NPZ inputs,
record-frozen Python, and explicitly checksummed VOR-root source archives.
Main Figure 4 panel a is rebuilt from its compact numerical illustration;
panel e is rebuilt from linearized place-field arrays. No current master figure,
archived parent, candidate, Desktop export, or promotion tree is read as a
scientific source.

Examples
--------
Check one current record without rendering::

    python docs/figure_version_records/rebuild_current_main_figures.py \
        --artifact main_figure_2 --check

Rebuild one current figure into a new directory::

    python docs/figure_version_records/rebuild_current_main_figures.py \
        --artifact main_figure_2 --output-dir outputs/main2_source_rebuild
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Callable
import xml.etree.ElementTree as ET
import zipfile

# Dynamic imports must never write cache files into immutable selected records.
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.ticker import FormatStrFormatter


VOR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VOR_ROOT.parents[1]
POINTER = VOR_ROOT / "CURRENT_MAIN_FIGURES.md"
ARTIFACTS = tuple(f"main_figure_{index}" for index in range(1, 6))
FORBIDDEN_SOURCE_PARTS = (
    "figure_version_records_archive",
    "figure_version_candidates",
    "parent_record_history",
    "parent_record_snapshot",
    "round11_parent_snapshot",
    "desktop",
    "ltm-figs",
)
MASTER_EXPORT_RE = re.compile(
    r"figure[1-5]_version_record(?:_300dpi|_600dpi)?\.(?:svg|pdf|png|tiff|html)$",
    re.IGNORECASE,
)
RUNTIME_ARCHIVE = "SOURCE_REBUILD_MAIN_RUNTIME.zip"
RUNTIME_ARCHIVE_SHA256 = "ac0d1efc9ce078f3b42d16fdcaf48b6446fb42c9acc986b83d20fbd9d0cab4cc"
RUNTIME_ARCHIVE_BYTES = 472999
VOR_INPUT_SPECS = {
    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip": {
        "sha256": "af168bec8fe7cb23dc1289ffa796ef03bd3a104d48f8b6547d81a5182071c5ab",
        "bytes": 2084418,
        "role": "quantitative_input",
    },
    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF_MANIFEST.json": {
        "sha256": "4b334e3a51bca23e00c7146e8558ee166bf613a31741a594ddaf16910d432950",
        "bytes": 88000,
        "role": "quantitative_input_source_tree_manifest",
    },
}

_RUNTIME_TEMP: tempfile.TemporaryDirectory[str] | None = None
_RUNTIME_ROOT: Path | None = None
_LINEAR_PF_TEMP: tempfile.TemporaryDirectory[str] | None = None
_LINEAR_PF_ROOT: Path | None = None


@dataclass(frozen=True)
class ArtifactContract:
    artifact: str
    figure_number: int
    required_files: tuple[str, ...]
    required_globs: tuple[str, ...]
    scientific_source_kinds: tuple[str, ...]
    fixed_assets: tuple[dict[str, str], ...] = ()
    vor_input_files: tuple[str, ...] = ()


CONTRACTS = {
    "main_figure_1": ArtifactContract(
        artifact="main_figure_1",
        figure_number=1,
        required_files=(
            "source_code/preview_figure1_assembly.py",
            "source_code/figure_panels.py",
            "plot_ready/figure1_endnode_timelines.csv",
            "plot_ready/reward_acquisition_day1_month_points_current_plus_new12.csv",
            "plot_ready/reward_acquisition_day1_month_summary_current_plus_new12.csv",
            "plot_ready/reward_acquisition_day1_month_clustered_contrasts_current_plus_new12.csv",
            "plot_ready/exploration_first_chunk_current_plus_new_points.csv",
            "plot_ready/long_bout_group_points_current_plus_new12.csv",
            "plot_ready/long_bout_group_summary_current_plus_new12.csv",
            "plot_ready/long_bout_group_clustered_contrasts_current_plus_new12.csv",
            "plot_ready/ed04_session_route_current_plus_new12.csv",
            "plot_ready/ed04_route_group_stats_current_plus_new12.csv",
            "plot_ready/ed04_route_stats_current_plus_new12.csv",
        ),
        required_globs=("plot_ready/*.csv",),
        scientific_source_kinds=(
            "record-frozen plot-ready session rows and clustered summaries",
            "record-frozen end-node event rows",
            "deterministic maze/cycle schematics from frozen Python",
        ),
    ),
    "main_figure_2": ArtifactContract(
        artifact="main_figure_2",
        figure_number=2,
        required_files=(
            "source_code/render_main_figs_v3_publication.py",
            "source_code/preview_figure2_assembly.py",
            "source_code/figure_panels.py",
            "panel_a_sources/422_corr.npz",
            "panel_a_sources/422-3_corr.npz",
            "source_data/gap_curve/gap_with_day1_day2_curve_data.csv",
            "source_data/rate_chain/session_pair_rate_consecutive_gap_points_canonical.csv",
            "source_data/rate_chain/session_pair_rate_consecutive_gap_stats_canonical.csv",
            "source_data/rate_chain/SOURCE_MANIFEST.json",
            "source_validation/canonical_ensemble_rate_correction/audits/gap_curve_old_vs_canonical.csv",
            "source_validation/canonical_ensemble_rate_correction/audits/distribution_vs_gap_3month_parity.csv",
        ),
        required_globs=(
            "panel_b_sources/*.npz",
            "source_data/distribution/*",
            "source_data/consecutive_gap/*",
            "source_data/gap_curve/*",
            "source_data/rate_chain/*",
            "source_data/registration/*",
        ),
        scientific_source_kinds=(
            "record-local MATLAB local-correlation FIG source objects",
            "record-local example-neuron NPZ bundles",
            "record-frozen distribution/null NPZs and pair/support audits",
            "record-frozen gap-curve and consecutive-chain row tables",
        ),
    ),
    "main_figure_3": ArtifactContract(
        artifact="main_figure_3",
        figure_number=3,
        required_files=(
            "source_code/render_main_figs_v3_publication.py",
            "source_code/figure_panels.py",
            "source_data/example/451-3day1_to_451-6day1__row0161_A190_B267.npz",
            "source_data/first30/day2_template_group_curve_summary.csv",
            "source_data/first30/combined_first30_points_all_metrics.csv",
            "source_data/first30/combined_first30_stats_all_metrics.csv",
            "source_data/long_gap/main3_experience_trend_points_no_manual_mask.csv",
            "source_data/long_gap/main3_experience_trend_clustered_stats_no_manual_mask.csv",
            "source_data/support/ed07_current59_curve_support_mask.csv",
        ),
        required_globs=("source_data/first30/*", "source_data/long_gap/*"),
        scientific_source_kinds=(
            "record-frozen moving-window group curves and 59-cycle row authority",
            "record-frozen long-gap pair rows and clustered statistics",
            "record-local example-neuron NPZ bundle",
            "deterministic analysis schematics from frozen Python",
        ),
    ),
    # The historical fixed-SVG fallback is resolved to numerical illustration
    # inputs by _contract_paths when the selected record contains that pack.
    # check_artifact and _render_main4 apply the same source selection.
    "main_figure_4": ArtifactContract(
        artifact="main_figure_4",
        figure_number=4,
        required_files=(
            "source_code/render_figure4_day_source_comparison.py",
            "source_code/render_main_figs_v3_publication.py",
            "source_code/figure_panels.py",
            "plot_ready/ANALYSIS_VALIDATION.json",
            "plot_ready/panel_b_native_pair_rows.csv",
            "plot_ready/panel_c_example_map_rows.csv",
            "plot_ready/panel_d_native_reference_rows.csv",
            "plot_ready/panel_f_matched_pair_means.csv",
            "plot_ready/figure4_day_source_statistics.csv",
            "source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_pair_summary.csv",
            "source_data/neuron_id_shuffle/figure4b_neuron_id_shuffle_statistics.json",
            "source_validation/figure4_panel_a_refinement/figure4_panel_a_decoding_method_schematic.svg",
        ),
        required_globs=(
            "plot_ready/*.csv",
            "plot_ready/*.json",
            "source_data/neuron_id_shuffle/*.csv",
            "source_data/neuron_id_shuffle/*.json",
        ),
        scientific_source_kinds=(
            "record-frozen native decoder/map/reward-control/behavior row tables",
            "record-frozen within-pair neuron-ID shuffle rows and clustered test",
            "VOR-root frozen linearized place-field arrays for both route directions",
        ),
        fixed_assets=(
            {
                "panel": "a",
                "path": "source_validation/figure4_panel_a_refinement/figure4_panel_a_decoding_method_schematic.svg",
                "role": "fixed real-data method example; no displayed population estimate is read from the SVG",
            },
        ),
        vor_input_files=(
            "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip",
            "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF_MANIFEST.json",
        ),
    ),
    "main_figure_5": ArtifactContract(
        artifact="main_figure_5",
        figure_number=5,
        required_files=(
            "source_code/preview_figure3_assembly.py",
            "source_code/figure_panels.py",
            "source_code/level_heatmap_utils.py",
            "source_data/figure5_bundle/plot_data_cell228_level.npz",
            "source_data/figure5_bundle/plot_data_cell226_level.npz",
            "source_data/figure5_bundle/cross_session_decoding_accuracy_points.csv",
            "source_data/figure5_bundle/cross_session_decoding_accuracy_summary.csv",
            "source_data/figure5_bundle/cross_session_decoding_accuracy_stats.json",
            "source_data/figure5_bundle/level_map_vs_exploration_efficiency_primary_clean_rows.csv",
            "source_data/figure5_bundle/level_map_vs_exploration_efficiency_primary_clean_stats.json",
            "source_data/level_vectors_tree_depth7_corrected36_d1v1.csv",
            "source_data/neuron_id_shuffle/cross_session_decoding_accuracy_points.csv",
            "source_data/neuron_id_shuffle/cross_session_decoding_accuracy_summary.csv",
            "source_data/neuron_id_shuffle/cross_session_decoding_accuracy_stats.json",
            "source_validation/refinement_2026_08_12/figure5b_historical_hierarchical_refined_order.npy",
        ),
        required_globs=(
            "source_data/figure5_bundle/*.csv",
            "source_data/figure5_bundle/*.json",
            "source_data/figure5_bundle/*.npz",
            "source_data/neuron_id_shuffle/*.csv",
            "source_data/neuron_id_shuffle/*.json",
        ),
        scientific_source_kinds=(
            "record-frozen 16,901-neuron level-vector table and historical seriation",
            "record-frozen distribution/null NPZ inputs",
            "record-frozen within-pair neuron-ID shuffle decoder rows",
            "record-frozen behavior-association rows and clustered statistics",
            "deterministic level-assignment schematic from frozen Python",
        ),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pointer_records() -> dict[str, Path]:
    text = POINTER.read_text(encoding="utf-8")
    selected: dict[str, Path] = {}
    pattern = re.compile(
        r"^\|\s*Figure\s+([1-5])\s*\|\s*`([^`]+)`\s*\|",
        flags=re.MULTILINE,
    )
    for figure_number, relative in pattern.findall(text):
        artifact = f"main_figure_{figure_number}"
        path = (REPO_ROOT / Path(relative.rstrip("/\\"))).resolve()
        selected[artifact] = path
    if set(selected) != set(ARTIFACTS):
        raise RuntimeError(
            f"Could not resolve exactly five pointer-selected main records from {POINTER}: "
            f"{sorted(selected)}"
        )
    return selected


def _assert_record_local(record: Path, path: Path, *, allow_output: bool = False) -> None:
    resolved = path.resolve()
    if not allow_output:
        try:
            resolved.relative_to(record.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Scientific input escapes the selected record: {resolved}") from exc
    lowered = resolved.as_posix().lower()
    if any(part in lowered for part in FORBIDDEN_SOURCE_PARTS):
        raise RuntimeError(f"Forbidden source tree referenced: {resolved}")
    if not allow_output and MASTER_EXPORT_RE.search(resolved.name):
        raise RuntimeError(f"Current master export cannot be used as a source: {resolved}")


def _assert_vor_source(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != VOR_ROOT.resolve():
        raise RuntimeError(f"VOR-root sidecar must be an immediate child of {VOR_ROOT}: {resolved}")
    if not resolved.name.startswith("SOURCE_REBUILD_MAIN_"):
        raise RuntimeError(f"Unrecognized VOR-root sidecar name: {resolved.name}")
    lowered = resolved.as_posix().lower()
    if any(part in lowered for part in FORBIDDEN_SOURCE_PARTS):
        raise RuntimeError(f"Forbidden source tree referenced: {resolved}")
    if MASTER_EXPORT_RE.search(resolved.name):
        raise RuntimeError(f"Current master export cannot be used as a source: {resolved}")


def _verify_vor_source(
    filename: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    kind: str,
    role: str,
) -> dict[str, Any]:
    path = VOR_ROOT / filename
    _assert_vor_source(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    observed_hash = _sha256(path)
    if observed_bytes != int(expected_bytes) or observed_hash.lower() != expected_sha256.lower():
        raise RuntimeError(f"VOR-root source hash drift: {path}")
    if path.suffix.lower() == ".zip" and not zipfile.is_zipfile(path):
        raise RuntimeError(f"VOR-root source archive is not a valid ZIP: {path}")
    return {
        "path": filename,
        "path_base": "vor_root",
        "kind": kind,
        "role": role,
        "sha256": observed_hash,
        "bytes": observed_bytes,
    }


def _hash_index(record: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    with (record / "RECORD_HASHES.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = Path(str(row["path"])).as_posix()
            result[key] = (str(row["sha256"]).lower(), int(row["bytes"]))
    return result


def _contract_paths(record: Path, contract: ArtifactContract) -> list[Path]:
    paths = [record / relative for relative in contract.required_files]
    if contract.artifact == "main_figure_4":
        illustration = record / "illustration_data" / "spatial_decoder_example.npz"
        if illustration.is_file():
            # Numeric schematic inputs supersede the older fixed SVG source.
            fixed_source = record / contract.fixed_assets[0]["path"]
            paths.remove(fixed_source)
            paths.extend((illustration, illustration.with_suffix(".json")))
        aligned_panel_f = (
            record / "source_code" / "render_main_figure4_panel_f_ed10_k166.py"
        )
        if aligned_panel_f.is_file():
            paths.append(aligned_panel_f)
            aligned_sources = record / "source_data" / "panel_f_ed10_k166"
            matches = sorted(path for path in aligned_sources.glob("*") if path.is_file())
            if not matches:
                raise FileNotFoundError(
                    f"Aligned Main Figure 4f source directory is empty: {aligned_sources}"
                )
            paths.extend(matches)
    for pattern in contract.required_globs:
        matches = sorted(
            path for path in record.glob(pattern)
            if path.is_file() and path.suffix.lower() not in {".pyc", ".pyo"}
            and "__pycache__" not in path.parts
        )
        if not matches:
            raise FileNotFoundError(f"Required source pattern has no matches: {record / pattern}")
        paths.extend(matches)
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda path: path.relative_to(record).as_posix())


def _is_ten_panel_main1(record: Path) -> bool:
    manifest = record / "figure1_source_manifest.json"
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("schema") == "figure1_ten_panel_source_record/v1"


def contract_for_record(artifact: str, record: Path) -> ArtifactContract:
    """Select the accepted source contract without changing archived layouts."""
    if artifact != "main_figure_1" or not _is_ten_panel_main1(record):
        return CONTRACTS[artifact]
    return ArtifactContract(
        artifact="main_figure_1",
        figure_number=1,
        required_files=(
            "source_code/rebuild_figure1.py",
            "source_code/preview_figure1_assembly.py",
            "source_code/figure_panels.py",
            "source_data/SOURCE_REBUILD_MAIN_RUNTIME.zip",
            "source_data/maze_3d_original.png",
            "source_data/trajectory_examples.csv.gz",
            "plot_ready/figure1_endnode_timelines.csv",
        ),
        required_globs=("source_code/**/*", "plot_ready/*.csv"),
        scientific_source_kinds=(
            "record-frozen plot-ready session rows and clustered summaries",
            "complete six-session end-node event rows with archived-artwork parity",
            "record-frozen Day 1 and Day 91 first-reward trajectory samples",
            "deterministic maze and calendar timeline schematics from frozen Python",
            "author-supplied physical maze illustration",
        ),
        fixed_assets=({
            "path": "source_data/maze_3d_original.png",
            "role": "nonquantitative physical maze illustration in panel a",
        },),
    )


def check_artifact(artifact: str) -> dict[str, Any]:
    record = _pointer_records()[artifact]
    contract = contract_for_record(artifact, record)
    if not record.is_dir():
        raise FileNotFoundError(record)
    validation = json.loads((record / "RECORD_VALIDATION.json").read_text(encoding="utf-8"))
    if str(validation.get("status", "")).upper() not in {"PASS", "ACCEPTED", "COMPLETE"}:
        raise RuntimeError(f"Selected record validation is not passing: {record}")
    index = _hash_index(record)
    checked: list[dict[str, Any]] = []
    for path in _contract_paths(record, contract):
        _assert_record_local(record, path)
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(record).as_posix()
        if relative not in index:
            raise RuntimeError(f"Scientific source is not covered by RECORD_HASHES.csv: {path}")
        expected_hash, expected_bytes = index[relative]
        observed_bytes = path.stat().st_size
        observed_hash = _sha256(path)
        if observed_bytes != expected_bytes or observed_hash.lower() != expected_hash:
            raise RuntimeError(f"Record source hash drift: {path}")
        checked.append(
            {
                "path": relative,
                "path_base": "record",
                "kind": "code" if relative.startswith("source_code/")
                or relative == "source_data/SOURCE_REBUILD_MAIN_RUNTIME.zip" else "input",
                "sha256": observed_hash,
                "bytes": observed_bytes,
            }
        )
    if not (artifact == "main_figure_1" and _is_ten_panel_main1(record)):
        checked.append(
            _verify_vor_source(
                RUNTIME_ARCHIVE,
                expected_sha256=RUNTIME_ARCHIVE_SHA256,
                expected_bytes=RUNTIME_ARCHIVE_BYTES,
                kind="code",
                role="isolated source-rebuild runtime",
            )
        )
    for filename in contract.vor_input_files:
        spec = VOR_INPUT_SPECS[filename]
        checked.append(
            _verify_vor_source(
                filename,
                expected_sha256=str(spec["sha256"]),
                expected_bytes=int(spec["bytes"]),
                kind="input",
                role=str(spec["role"]),
            )
        )
    return {
        "status": "PASS",
        "artifact": artifact,
        "figure_number": contract.figure_number,
        "pointer": POINTER.relative_to(REPO_ROOT).as_posix(),
        "selected_record": record.relative_to(REPO_ROOT).as_posix(),
        "record_validation_status": validation.get("status"),
        "scientific_source_kinds": list(contract.scientific_source_kinds),
        "fixed_assets": (
            [] if artifact == "main_figure_4"
            and (record / "illustration_data/spatial_decoder_example.npz").is_file()
            else list(contract.fixed_assets)
        ),
        "forbidden_source_trees": list(FORBIDDEN_SOURCE_PARTS),
        "master_exports_as_sources": "forbidden",
        "checked_source_file_count": len(checked),
        "checked_sources": checked,
    }


def _load_module(name: str, path: Path, *, synthetic_file: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(synthetic_file or path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    # A few historically frozen helpers retain a UTF-8 BOM.  ``utf-8-sig``
    # consumes it while remaining identical to UTF-8 for every other source.
    source = path.read_text(encoding="utf-8-sig")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _extract_verified_archive(
    filename: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    prefix: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    _verify_vor_source(
        filename,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        kind="input",
        role="runtime extraction",
    )
    archive = VOR_ROOT / filename
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
    root = Path(temporary.name).resolve()
    try:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                member = info.filename
                parts = PurePosixPath(member).parts
                if (
                    not parts
                    or PurePosixPath(member).is_absolute()
                    or "\\" in member
                    or any(part in {"", ".", ".."} for part in parts)
                    or ":" in parts[0]
                ):
                    raise RuntimeError(f"Unsafe member in {archive.name}: {member!r}")
                destination = root.joinpath(*parts).resolve()
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Archive member escapes the extraction root: {member!r}"
                    ) from exc
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info, "r") as input_stream, destination.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
    except Exception:
        temporary.cleanup()
        raise
    return temporary, root


def _runtime_root() -> Path:
    global _RUNTIME_TEMP, _RUNTIME_ROOT
    if _RUNTIME_ROOT is None:
        _RUNTIME_TEMP, _RUNTIME_ROOT = _extract_verified_archive(
            RUNTIME_ARCHIVE,
            expected_sha256=RUNTIME_ARCHIVE_SHA256,
            expected_bytes=RUNTIME_ARCHIVE_BYTES,
            prefix="source_rebuild_main_runtime_",
        )
        if not (_RUNTIME_ROOT / "ca_analysis" / "__init__.py").is_file():
            raise RuntimeError("The source-rebuild runtime lacks ca_analysis/__init__.py")
    return _RUNTIME_ROOT


def _verify_linear_pf_tree(root: Path) -> dict[str, Any]:
    manifest_name = "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF_MANIFEST.json"
    archive_name = "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip"
    manifest_spec = VOR_INPUT_SPECS[manifest_name]
    archive_spec = VOR_INPUT_SPECS[archive_name]
    _verify_vor_source(
        manifest_name,
        expected_sha256=str(manifest_spec["sha256"]),
        expected_bytes=int(manifest_spec["bytes"]),
        kind="input",
        role=str(manifest_spec["role"]),
    )
    payload = json.loads((VOR_ROOT / manifest_name).read_text(encoding="utf-8"))
    if payload.get("schema") != "source-rebuild-main-figure4-panel-e-linear-pf/v1":
        raise RuntimeError("Unexpected Figure 4e source-tree manifest schema")
    if payload.get("status") != "PASS":
        raise RuntimeError("Figure 4e source-tree manifest is not passing")
    expected_source_path = (
        "docs/sorted_data_full_figure_rerun_2026-06-11/linearized_pf/sorted_export"
    )
    if payload.get("source_repo_relative_path") != expected_source_path:
        raise RuntimeError("Figure 4e source-tree provenance path drift")
    archive = payload.get("archive", {})
    if (
        archive.get("path")
        != f"docs/figure_version_records/{archive_name}"
        or archive.get("sha256") != archive_spec["sha256"]
        or int(archive.get("bytes", -1)) != int(archive_spec["bytes"])
    ):
        raise RuntimeError("Figure 4e source-tree manifest archive identity drift")

    rows = payload.get("members")
    if not isinstance(rows, list) or len(rows) != 465:
        raise RuntimeError("Figure 4e source-tree manifest must declare exactly 465 members")
    declared: dict[str, tuple[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("Malformed member in Figure 4e source-tree manifest")
        name = str(row.get("path", ""))
        member_path = PurePosixPath(name)
        if (
            not name
            or member_path.is_absolute()
            or len(member_path.parts) != 1
            or member_path.parts[0] in {".", ".."}
            or "\\" in name
            or ":" in name
        ):
            raise RuntimeError(f"Unsafe Figure 4e source-tree member: {name!r}")
        expected_hash = str(row.get("sha256", "")).lower()
        expected_bytes = int(row.get("bytes", -1))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or expected_bytes < 0:
            raise RuntimeError(f"Malformed hash metadata for Figure 4e source member: {name}")
        if name in declared:
            raise RuntimeError(f"Duplicate Figure 4e source-tree member: {name}")
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Figure 4e source-tree member is absent from archive: {name}")
        observed_bytes = path.stat().st_size
        observed_hash = _sha256(path)
        if observed_bytes != expected_bytes or observed_hash != expected_hash:
            raise RuntimeError(f"Figure 4e source-tree member hash drift: {name}")
        declared[name] = (expected_hash, expected_bytes)

    observed_names = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_names != set(declared):
        raise RuntimeError("Figure 4e source archive and source-tree manifest member sets differ")
    tree = payload.get("source_tree", {})
    canonical_text = "".join(
        f"{member_hash}  {member_bytes}  {name}\n"
        for name, (member_hash, member_bytes) in sorted(declared.items())
    )
    observed_tree_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    npy_count = sum(name.lower().endswith(".npy") for name in declared)
    json_count = sum(name.lower().endswith(".json") for name in declared)
    uncompressed_bytes = sum(size for _, size in declared.values())
    if (
        int(tree.get("member_count", -1)) != len(declared)
        or int(tree.get("npy_count", -1)) != npy_count
        or int(tree.get("json_count", -1)) != json_count
        or int(tree.get("uncompressed_bytes", -1)) != uncompressed_bytes
        or tree.get("tree_sha256") != observed_tree_hash
        or npy_count != 464
        or json_count != 1
        or "session_run_counts.json" not in declared
    ):
        raise RuntimeError("Figure 4e source-tree aggregate identity drift")

    expected_scope = {
        "representation_space": "22-bin linearized home-to-water and water-to-home routes",
        "pair_filter": "day1_min_runs",
        "minimum_runs_per_session_and_direction": 5,
        "stable_threshold": 0.3,
        "heatmap_neuron_set": "all valid matched neurons",
        "excluded_pair": ["1616day1", "1616bday1"],
        "home_to_water": {"pairs": 29, "animals": 13, "matched": 8600, "stable": 2188},
        "water_to_home": {"pairs": 28, "animals": 14, "matched": 8438, "stable": 2044},
    }
    if payload.get("scientific_scope") != expected_scope:
        raise RuntimeError("Figure 4e source-tree scientific scope drift")
    return payload


def _linear_pf_root() -> Path:
    global _LINEAR_PF_TEMP, _LINEAR_PF_ROOT
    if _LINEAR_PF_ROOT is None:
        filename = "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip"
        spec = VOR_INPUT_SPECS[filename]
        _LINEAR_PF_TEMP, _LINEAR_PF_ROOT = _extract_verified_archive(
            filename,
            expected_sha256=str(spec["sha256"]),
            expected_bytes=int(spec["bytes"]),
            prefix="source_rebuild_main_figure4_panel_e_",
        )
        _verify_linear_pf_tree(_LINEAR_PF_ROOT)
    return _LINEAR_PF_ROOT


def _remove_live_source_path() -> None:
    live_src = (REPO_ROOT / "src").resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == live_src:
                continue
        except (OSError, RuntimeError):
            pass
        retained.append(entry)
    sys.path[:] = retained


def _assert_isolated_ca_analysis(record: Path, runtime_root: Path) -> None:
    allowed_roots = (runtime_root.resolve(), (record / "source_code").resolve())
    live_src = (REPO_ROOT / "src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name != "ca_analysis" and not name.startswith("ca_analysis."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        resolved = Path(module_file).resolve()
        if resolved == live_src or live_src in resolved.parents:
            raise RuntimeError(f"Live repository dependency imported during source rebuild: {resolved}")
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise RuntimeError(f"Unfrozen ca_analysis dependency imported: {resolved}")


def _install_frozen_panel_modules(record: Path):
    runtime_root = _runtime_root()
    _remove_live_source_path()
    runtime_text = str(runtime_root)
    sys.path[:] = [entry for entry in sys.path if entry != runtime_text]
    sys.path.insert(0, runtime_text)
    for name in tuple(sys.modules):
        if name == "ca_analysis" or name.startswith("ca_analysis."):
            del sys.modules[name]
    os.environ["CA_COMPAT_LINK_ARTIFACTS"] = "0"
    ca_analysis = importlib.import_module("ca_analysis")
    compat = importlib.import_module("ca_analysis.compat")

    def frozen_bootstrap_environment(repo_root: str | os.PathLike | None = None):
        return types.SimpleNamespace(
            repo_root=str(Path(repo_root).resolve() if repo_root is not None else REPO_ROOT),
            added_paths=[],
            missing_paths=[],
        )

    compat.bootstrap_environment = frozen_bootstrap_environment

    # ``jw_utils.py`` is a large legacy utility module with costly optional
    # imports.  None of these five rebuilds calls it: Main Figure 5 freezes a
    # precomputed exact seriation, so the on-demand hierarchical fallback in
    # ``figure_panels.py`` is never reached.  Install only the lightweight
    # frozen helper that is actually executed.
    for stem in ("level_heatmap_utils",):
        frozen = record / "source_code" / f"{stem}.py"
        if frozen.is_file():
            loaded = _load_module(f"ca_analysis.{stem}", frozen)
            setattr(ca_analysis, stem, loaded)
    frozen_panels = _load_module(
        "ca_analysis.figure_panels",
        record / "source_code" / "figure_panels.py",
    )
    setattr(ca_analysis, "figure_panels", frozen_panels)
    _assert_isolated_ca_analysis(record, runtime_root)
    return frozen_panels


def _write_result_manifest(
    output_dir: Path,
    check: dict[str, Any],
    outputs: list[Path],
    *,
    notes: list[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "artifact": check["artifact"],
        "selected_record": check["selected_record"],
        "source_check": check,
        "outputs": [
            {
                "path": path.resolve().as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
        "notes": list(notes or []),
    }
    path = output_dir / f"{check['artifact']}_SOURCE_REBUILD_MANIFEST.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _svg_dimensions_mm(svg_path: Path) -> tuple[float, float]:
    root = ET.parse(svg_path).getroot()

    def to_mm(value: str) -> float:
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|in|pt|px)?\s*",
            str(value),
            flags=re.IGNORECASE,
        )
        if match is None:
            raise ValueError(f"Unsupported SVG length: {value!r}")
        number = float(match.group(1))
        unit = (match.group(2) or "px").lower()
        return number * {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0, "px": 25.4 / 96.0}[unit]

    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width and height:
        return to_mm(width), to_mm(height)
    viewbox = [float(value) for value in root.attrib["viewBox"].split()]
    return viewbox[2] * 25.4 / 72.0, viewbox[3] * 25.4 / 72.0


def _find_chrome() -> Path:
    candidates = (
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Chrome/Edge is required for vector SVG-to-PDF export")


def _wait_for_file(path: Path, *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable = 0
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.is_file() else -1
        if size > 0 and size == previous_size:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        previous_size = size
        time.sleep(0.15)
    raise RuntimeError(f"Export did not settle: {path}")


def _export_svg_bundle(source_svg: Path, output_stem: Path) -> list[Path]:
    """Materialize the accepted editable/raster output contract from fresh SVG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    target_svg = output_stem.with_suffix(".svg")
    if source_svg.resolve() != target_svg.resolve():
        shutil.copy2(source_svg, target_svg)
    width_mm, height_mm = _svg_dimensions_mm(target_svg)
    html = output_stem.with_suffix(".html")
    svg_text = target_svg.read_text(encoding="utf-8")
    html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"@page{{size:{width_mm:.6f}mm {height_mm:.6f}mm;margin:0}}"
        f"html,body{{margin:0;width:{width_mm:.6f}mm;height:{height_mm:.6f}mm;overflow:hidden}}"
        f"svg{{display:block;width:{width_mm:.6f}mm!important;height:{height_mm:.6f}mm!important}}"
        f"</style></head><body>{svg_text}</body></html>",
        encoding="utf-8",
    )
    pdf = output_stem.with_suffix(".pdf")
    command = [
        str(_find_chrome()),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-software-rasterizer",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf}",
        html.resolve().as_uri(),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(
            "Browser SVG-to-PDF export failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    _wait_for_file(pdf)
    png = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".png")
    tiff = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".tiff")
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError(f"Expected one-page figure PDF: {pdf}")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    image.save(png, dpi=(600, 600), optimize=True)
    image.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    return [target_svg, pdf, png, tiff, html]


def _export_main4_svg_bundle(
    source_svg: Path, output_stem: Path, module
) -> list[Path]:
    """Export Figure 4 with its frozen browser-screenshot raster contract."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    target_svg = output_stem.with_suffix(".svg")
    if source_svg.resolve() != target_svg.resolve():
        shutil.copy2(source_svg, target_svg)
    exported = module.export_svg(target_svg)
    pdf = Path(exported["pdf"])
    raw_png = Path(exported["png"])
    raw_tiff = Path(exported["tiff"])
    png = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".png")
    tiff = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".tiff")
    os.replace(raw_png, png)
    os.replace(raw_tiff, tiff)
    width_mm, height_mm = _svg_dimensions_mm(target_svg)
    html = output_stem.with_suffix(".html")
    svg_text = target_svg.read_text(encoding="utf-8")
    html.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        f"@page{{size:{width_mm:.6f}mm {height_mm:.6f}mm;margin:0}}"
        f"html,body{{width:{width_mm:.6f}mm;height:{height_mm:.6f}mm;margin:0;"
        "padding:0;background:white;overflow:hidden}}"
        f"svg{{display:block;width:{width_mm:.6f}mm!important;"
        f"height:{height_mm:.6f}mm!important}}"
        "*{print-color-adjust:exact;-webkit-print-color-adjust:exact}"
        f"</style></head><body>{svg_text}</body></html>",
        encoding="utf-8",
    )
    return [target_svg, pdf, png, tiff, html]


def _render_main1(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    if _is_ten_panel_main1(record):
        source = _load_module(
            "record_local_main1_ten_panel_rebuild",
            record / "source_code" / "rebuild_figure1.py",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return [Path(path) for path in source.rebuild(record, output_dir)]
    frozen_panels = _install_frozen_panel_modules(record)
    preview = _load_module(
        "record_local_main1_preview",
        record / "source_code" / "preview_figure1_assembly.py",
    )
    plot_ready = record / "plot_ready"
    preview.FIGURE1_REWARD_ACQUISITION_POINTS = (
        plot_ready / "reward_acquisition_day1_month_points_current_plus_new12.csv"
    )
    preview.FIGURE1_REWARD_ACQUISITION_SUMMARY = (
        plot_ready / "reward_acquisition_day1_month_summary_current_plus_new12.csv"
    )
    preview.FIGURE1_REWARD_ACQUISITION_CONTRASTS = (
        plot_ready / "reward_acquisition_day1_month_clustered_contrasts_current_plus_new12.csv"
    )
    preview.FIGURE1_EXPLORATION_POINTS = (
        plot_ready / "exploration_first_chunk_current_plus_new_points.csv"
    )
    preview.FIGURE1_LONG_BOUT_POINTS = plot_ready / "long_bout_group_points_current_plus_new12.csv"
    preview.FIGURE1_LONG_BOUT_SUMMARY = plot_ready / "long_bout_group_summary_current_plus_new12.csv"
    preview.FIGURE1_LONG_BOUT_CONTRASTS = (
        plot_ready / "long_bout_group_clustered_contrasts_current_plus_new12.csv"
    )
    preview.FIGURE1_ROUTE_SCORE_POINTS = plot_ready / "ed04_session_route_current_plus_new12.csv"
    preview.FIGURE1_ROUTE_SCORE_SUMMARY = plot_ready / "ed04_route_group_stats_current_plus_new12.csv"
    preview.FIGURE1_ROUTE_SCORE_CONTRASTS = plot_ready / "ed04_route_stats_current_plus_new12.csv"

    timeline = pd.read_csv(plot_ready / "figure1_endnode_timelines.csv")

    def extract_record_timeline(
        nickname: str,
        *,
        lower_threshold: int = frozen_panels.END_NODE_LOWER_THRESHOLD,
        upper_threshold: int = frozen_panels.END_NODE_UPPER_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = timeline.loc[timeline["nickname"].astype(str).eq(str(nickname))]
        if rows.empty:
            raise KeyError(f"Record timeline lacks session {nickname!r}")
        nodes = pd.to_numeric(rows["endnode"], errors="raise").to_numpy(dtype=int)
        times = pd.to_numeric(rows["time_in_maze_s"], errors="raise").to_numpy(dtype=float)
        keep = (nodes >= int(lower_threshold)) & (nodes <= int(upper_threshold))
        return nodes[keep], times[keep]

    frozen_panels.extract_endnode_timeline = extract_record_timeline
    output_dir.mkdir(parents=True, exist_ok=True)
    tree, rendered_sizes = preview.render_assembly_svg(dpi=300)
    svg = output_dir / "main_figure_1_source_rebuild.svg"
    tree.write(svg, encoding="utf-8", xml_declaration=True)
    layout = output_dir / "main_figure_1_source_rebuild_layout.json"
    layout.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selected_record": check["selected_record"],
                "assembly": "record-local source-native SVG composition",
                "rendered_sizes": rendered_sizes,
                "timeline_source": "plot_ready/figure1_endnode_timelines.csv",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [*_export_svg_bundle(svg, output_dir / "main_figure_1_source_rebuild"), layout]


def _load_publication_renderer(record: Path, name: str):
    _install_frozen_panel_modules(record)
    return _load_module(
        name,
        record / "source_code" / "render_main_figs_v3_publication.py",
        synthetic_file=REPO_ROOT / "scripts" / "render_main_figs_v3_publication.py",
    )


def _patch_figure2_helper_loader(publication, record: Path) -> None:
    helper_path = record / "source_code" / "preview_figure2_assembly.py"
    panel_a_top = record / "panel_a_sources" / "422_corr.npz"
    panel_a_bottom = record / "panel_a_sources" / "422-3_corr.npz"
    original_loader = publication._load_script_module

    def load(name: str, path: Path):
        module = original_loader(name, path)
        if Path(path).resolve() != helper_path.resolve():
            return module

        def render_panel_a(size: tuple[float, float]):
            module._set_common_rc()
            fig, axes, _data, _paths = module.render_stacked_local_correlation_panel(
                top_fig_path=panel_a_top,
                bottom_fig_path=panel_a_bottom,
                panel_size=size,
                layout={**module.PANEL_LAYOUT, "panel_size": size, "grid_hspace": 0.05},
                style={
                    "origin": "upper",
                    "shared_clim": True,
                    "shared_vmin": None,
                    "shared_vmax": None,
                    "show_axes": False,
                    "show_colorbars": False,
                    "show_panel_title": False,
                    "show_subplot_titles": False,
                    "side_label_transform": "axes",
                    "side_label_rotation": 90,
                    "side_label_ha": "center",
                    "side_label_va": "center",
                    "side_label_fontsize": 8.0,
                    "side_label_fontweight": "normal",
                    "top_side_label_xy": (-0.060, 0.50),
                    "bottom_side_label_xy": (-0.060, 0.50),
                    "show_scale_bar": True,
                    "scale_bar_panel": "bottom",
                    "scale_bar_xy": (295.0, 278.0),
                    "scale_bar_length_px": 60.0,
                    "scale_bar_linewidth": 2.4,
                    "scale_bar_color": "white",
                    "scale_bar_text_color": "white",
                    "scale_bar_text_offset_px": -10.0,
                    "scale_bar_fontsize": 6.4,
                    "scale_bar_fontweight": "normal",
                },
                labels={
                    "top_side_label": "Day 1",
                    "bottom_side_label": "Day 91",
                    "scale_bar_label": "100 µm",
                },
                save=False,
            )
            axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)
            if len(axes_flat) >= 2:
                axes_flat[0].set_position([0.055, 0.506, 0.940, 0.492])
                axes_flat[1].set_position([0.055, 0.006, 0.940, 0.492])
            module._standardize_axes_text(fig)
            return fig

        module._render_panel_a = render_panel_a
        return module

    publication._load_script_module = load


def _render_main2(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    publication = _load_publication_renderer(record, "record_local_main2_publication")
    distribution = record / "source_data" / "distribution"
    consecutive = record / "source_data" / "consecutive_gap"
    gap = record / "source_data" / "gap_curve"
    rate = record / "source_data" / "rate_chain"
    publication.FIGURE2_HELPER = record / "source_code" / "preview_figure2_assembly.py"
    publication.EXAMPLE_SAVED_DIR = record / "panel_b_sources"
    publication.FIGURE2_DISTRIBUTION_DIR = distribution
    publication.FIGURE2_DISTRIBUTION_PROVENANCE = distribution / "provenance_manifest.json"
    publication.FIGURE2_DISTRIBUTION_PAIR_INPUT = distribution / "pair_input_audit.csv"
    publication.FIGURE2_DISTRIBUTION_PAIR_SUPPORT = distribution / "pair_support_audit.csv"
    publication.FIGURE2_DISTRIBUTION_METRIC_SUMMARY = distribution / "metric_summary.csv"
    publication.FIGURE2_GAP_CURVE = gap / "gap_with_day1_day2_curve_data.csv"
    publication.FIGURE2_CONSECUTIVE_POINTS = (
        consecutive / "consecutive_gap_clean_long_gap_corrected_points.csv"
    )
    publication.FIGURE2_CONSECUTIVE_STATS = (
        consecutive / "consecutive_gap_clean_long_gap_corrected_stats.csv"
    )
    publication.FIGURE2_CONSECUTIVE_MANIFEST = (
        consecutive / "consecutive_gap_clean_long_gap_manifest.json"
    )
    publication.FIGURE2_CONSECUTIVE_UNRESTRICTED_SCOPE = (
        consecutive / "consecutive_gap_unrestricted_adjacent_23_chain_scope.csv"
    )
    publication.FIGURE2_CONSECUTIVE_CLEAN_SCOPE = (
        consecutive / "consecutive_gap_clean_long_gap_22_chain_scope.csv"
    )
    publication.FIGURE2_CONSECUTIVE_INPUT_AUDIT = (
        consecutive / "consecutive_gap_sorted_triplet_input_audit.csv"
    )
    publication.FIGURE2_OLD_NEW_REPLICATION_AUDIT = (
        consecutive / "figure2_ed12_consecutive_old_vs_new_replication_audit.csv"
    )
    publication.FIGURE2_RATE_CHAIN_POINTS = (
        rate / "session_pair_rate_consecutive_gap_points_canonical.csv"
    )
    publication.FIGURE2_RATE_CHAIN_STATS = (
        rate / "session_pair_rate_consecutive_gap_stats_canonical.csv"
    )
    publication.FIGURE2_RATE_CHAIN_MANIFEST = rate / "SOURCE_MANIFEST.json"
    publication.SOURCE_REPAIR_842_MANIFEST = distribution / "provenance_manifest.json"
    publication.FIGURE2_GAP_PAIR_AUDIT = (
        record
        / "source_validation"
        / "canonical_ensemble_rate_correction"
        / "audits"
        / "gap_curve_old_vs_canonical.csv"
    )
    publication.FIGURE2_GAP_DISTRIBUTION_PARITY = (
        record
        / "source_validation"
        / "canonical_ensemble_rate_correction"
        / "audits"
        / "distribution_vs_gap_3month_parity.csv"
    )

    def distribution_audit() -> dict[str, Any]:
        pair_input = pd.read_csv(publication.FIGURE2_DISTRIBUTION_PAIR_INPUT)
        support = pd.read_csv(publication.FIGURE2_DISTRIBUTION_PAIR_SUPPORT)
        if len(pair_input) != 36 or len(support) != 36:
            raise RuntimeError("Figure 2 distribution source is not the accepted 36-pair scope")
        return {
            "status": "PASS",
            "pair_count": 36,
            "source": "record-local frozen distribution bundle",
        }

    def consecutive_audit(points: pd.DataFrame, stats: pd.DataFrame) -> dict[str, Any]:
        clean = pd.read_csv(publication.FIGURE2_CONSECUTIVE_CLEAN_SCOPE)
        rate_rows = pd.read_csv(publication.FIGURE2_RATE_CHAIN_POINTS)
        if len(clean) != 22 or len(rate_rows.loc[rate_rows["scope"].eq("clean_long_gap_day1_triplets")]) != 22:
            raise RuntimeError("Figure 2 consecutive source is not the accepted 22-chain scope")
        return {
            "status": "PASS",
            "clean_chains": 22,
            "source": "record-local frozen consecutive and canonical-rate tables",
        }

    publication._validate_source_repair_842 = lambda: {
        "status": "PASS",
        "source": "record-local frozen correction audits",
    }
    publication._validate_figure2_distribution_sources = distribution_audit
    publication._validate_figure2_clean_consecutive_scope = consecutive_audit
    _patch_figure2_helper_loader(publication, record)
    output_dir.mkdir(parents=True, exist_ok=True)
    style = publication.PublicationStyle(dpi=300)
    manifest = publication.render_figure2(output_dir, style)
    manifest_path = output_dir / "figure2_source_manifest.json"
    return [
        *_export_svg_bundle(
            output_dir / "figure2_source_publication.svg",
            output_dir / "main_figure_2_source_rebuild",
        ),
        manifest_path,
    ]


def _render_main3(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    publication = _load_publication_renderer(record, "record_local_main3_publication")
    first30 = record / "source_data" / "first30"
    long_gap = record / "source_data" / "long_gap"
    publication.EXAMPLE_SAVED_DIR = record / "source_data" / "example"
    publication.DAY2_TEMPLATE_GROUP_CURVES = first30 / "day2_template_group_curve_summary.csv"
    publication.FIGURE3_FIRST30_POINTS_ALL_METRICS = (
        first30 / "combined_first30_points_all_metrics.csv"
    )
    publication.FIGURE3_FIRST30_STATS_ALL_METRICS = (
        first30 / "combined_first30_stats_all_metrics.csv"
    )
    publication.LONG_GAP_POINTS = (
        long_gap / "main3_experience_trend_points_no_manual_mask.csv"
    )
    publication.LONG_GAP_STATS = (
        long_gap / "main3_experience_trend_clustered_stats_no_manual_mask.csv"
    )
    publication.FIGURE3_CURVE_SUPPORT_MASK = (
        record / "source_data" / "support" / "ed07_current59_curve_support_mask.csv"
    )
    publication.DAY2_TEMPLATE_SOURCE_MANIFEST = first30 / "SOURCE_MANIFEST.json"
    publication.SOURCE_REPAIR_842_MANIFEST = long_gap / "SOURCE_MANIFEST.json"
    publication.FIGURE3_LEVEL_PAIR_AUDIT = long_gap / "MASK_ONLY_METRIC_DELTAS.csv"
    publication.FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT = (
        long_gap / "FINAL_VALUE_PROVENANCE.csv"
    )
    publication._ensure_sources = lambda: None

    def load_points() -> pd.DataFrame:
        points = pd.read_csv(publication.FIGURE3_FIRST30_POINTS_ALL_METRICS)
        points["metric_label"] = points["metric"].map(publication.METRIC_LABELS)
        return points.sort_values(
            ["metric", "experience_index", "animal"], kind="stable"
        ).reset_index(drop=True)

    publication._load_day2_template_30min_points = load_points
    output_dir.mkdir(parents=True, exist_ok=True)
    style = publication.PublicationStyle(dpi=600)
    publication.render_figure3(output_dir, style)
    # Preserve the source renderer's exact page dimensions and 600-dpi images.
    # Re-exporting the SVG's rounded millimetre attributes changes edge pixels.
    output_stem = output_dir / "main_figure_3_source_rebuild"
    outputs = []
    for suffix in (".svg", ".pdf", ".html"):
        target = output_stem.with_suffix(suffix)
        shutil.copy2(output_dir / f"figure3_source_publication{suffix}", target)
        outputs.append(target)
    with fitz.open(output_stem.with_suffix(".pdf")) as document:
        if document.page_count != 1:
            raise RuntimeError("Expected one-page Figure 3 source-rendered PDF")
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    png = output_stem.with_name(output_stem.name + "_600dpi").with_suffix(".png")
    tiff = png.with_suffix(".tiff")
    image.save(png, dpi=(600, 600), optimize=True)
    image.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    return [
        *outputs,
        png,
        tiff,
        output_dir / "figure3_source_manifest.json",
    ]


def _svg_payload(path: Path) -> bytes:
    return path.read_bytes()


def _embedded_image_hashes(payload: bytes) -> list[str]:
    root = ET.fromstring(payload)
    hashes: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "image":
            continue
        href = element.get("{http://www.w3.org/1999/xlink}href") or element.get("href")
        if not href or not href.startswith("data:") or "," not in href:
            continue
        encoded = href.split(",", 1)[1]
        image_payload = base64.b64decode(encoded)
        hashes.append(hashlib.sha256(image_payload).hexdigest())
    return hashes


def _optional_record_qa_reference(record: Path, relative: str) -> dict[str, Any] | None:
    """Hash-check a retained display only when it is available for post-build QA."""

    path = record / relative
    if not path.is_file():
        return None
    _assert_record_local(record, path)
    index = _hash_index(record)
    expected_hash, expected_bytes = index[Path(relative).as_posix()]
    observed_hash = _sha256(path)
    observed_bytes = path.stat().st_size
    if observed_hash != expected_hash or observed_bytes != expected_bytes:
        raise RuntimeError(f"Optional QA reference hash drift: {path}")
    return {
        "path": relative,
        "sha256": observed_hash,
        "bytes": observed_bytes,
        "payload": path.read_bytes(),
    }


def _compose_main4(publication, module, payloads: dict[str, bytes], output_svg: Path) -> None:
    width_pt = module.FIGSIZE[0] * 72.0
    height_pt = module.FIGSIZE[1] * 72.0
    root = ET.Element(
        f"{{{module.SVG_NS}}}svg",
        {
            "width": f"{module.WIDTH_MM:.3f}mm",
            "height": f"{module.HEIGHT_MM:.3f}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
            "version": "1.1",
        },
    )
    # Panel a uses the accepted PPT-style source crop rather than aspect-fit
    # placement.  Creating defs before the white background also preserves the
    # accepted editable SVG structure.
    ET.SubElement(root, f"{{{module.SVG_NS}}}defs")
    ET.SubElement(
        root,
        f"{{{module.SVG_NS}}}rect",
        {"width": "100%", "height": "100%", "fill": "white"},
    )
    for panel in "abcdef":
        if panel == "a":
            source = ET.fromstring(payloads[panel])
            _x0, _y0, source_w, source_h = module.svg_viewbox(source)
            publication._add_svg_panel_cropped_absolute(
                root,
                {
                    "children": [copy.deepcopy(child) for child in list(source)],
                    "viewbox_width": source_w,
                    "viewbox_height": source_h,
                },
                panel_id="figure4_a",
                slot_pt=tuple(value * 72.0 for value in module.SLOTS[panel]),
                src_rect=publication.FIGURE4_PANEL_A_PPT_CROP,
            )
        else:
            module.add_svg_panel(
                root,
                payloads[panel],
                panel_id=f"figure4_{panel}",
                slot=module.SLOTS[panel],
            )
    for panel, (x, y) in module.LABEL_POSITIONS.items():
        node = ET.SubElement(
            root,
            f"{{{module.SVG_NS}}}text",
            {
                "x": f"{x * 72.0:.4f}",
                "y": f"{y * 72.0:.4f}",
                "font-family": "Arial",
                "font-size": "10.5",
                "font-weight": "bold",
                "fill": "black",
            },
        )
        node.text = panel
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)


def _render_main4(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    publication = _load_publication_renderer(record, "record_local_main4_publication")
    module = _load_module(
        "record_local_main4_renderer",
        record / "source_code" / "render_figure4_day_source_comparison.py",
    )
    module.configure_style()
    aligned_panel_f_path = (
        record / "source_code" / "render_main_figure4_panel_f_ed10_k166.py"
    )
    aligned_panel_f = (
        _load_module("record_local_main4_panel_f", aligned_panel_f_path)
        if aligned_panel_f_path.is_file()
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = record / "plot_ready"
    native = pd.read_csv(analysis / "panel_b_native_pair_rows.csv", dtype={"animal_id": str})
    map_rows = pd.read_csv(analysis / "panel_c_example_map_rows.csv")
    panel_d = pd.read_csv(
        analysis / "panel_d_native_reference_rows.csv", dtype={"animal": str}
    )
    matched = pd.read_csv(analysis / "panel_f_matched_pair_means.csv", dtype={"animal_id": str})
    stat_table = pd.read_csv(analysis / "figure4_day_source_statistics.csv")
    shuffle_rows = pd.read_csv(
        record
        / "source_data"
        / "neuron_id_shuffle"
        / "figure4b_neuron_id_shuffle_pair_summary.csv",
        dtype={"animal_id": str},
    )
    shuffle_stats = json.loads(
        (
            record
            / "source_data"
            / "neuron_id_shuffle"
            / "figure4b_neuron_id_shuffle_statistics.json"
        ).read_text(encoding="utf-8")
    )
    if len(shuffle_rows) != 36 or shuffle_rows["animal_id"].nunique() != 14:
        raise RuntimeError("Main Figure 4 neuron-ID shuffle source has the wrong scope")
    panel_b_rows = shuffle_rows.rename(
        columns={"neuron_id_shuffle_error": "exact_label_permutation_error"}
    )
    panel_b_stat = pd.Series(
        {"cluster_p": float(shuffle_stats["cluster_regression"]["cluster_p"])}
    )
    d_rows = panel_d.loc[
        panel_d["variant"].eq("d1v1")
        & panel_d["included_reward_location_control"].astype(bool)
    ].copy()
    d_stat = stat_table.loc[
        stat_table["analysis"].eq("d1_native_reward_control_advantage")
    ].iloc[0]
    f_stat = stat_table.loc[
        stat_table["analysis"].eq("d1_matched_reward_behavior_association")
    ].iloc[0]
    # Current D1-only records may intentionally omit the historical D2
    # sensitivity column when panel f is aligned to the ED10 K=166 estimator.
    error_columns = ["d1_reward_proxy_error"]
    if "d2_reward_proxy_error" in matched.columns:
        error_columns.append("d2_reward_proxy_error")
    all_y = np.concatenate(
        [matched[column].to_numpy(dtype=float) for column in error_columns]
    )
    all_x = matched["first_long_bout_time_s"].to_numpy(dtype=float)
    shared_limits = (
        (0.0, float(all_x.max() * 1.08)),
        (0.0, float(all_y.max() * 1.10)),
    )
    helper = _load_module(
        "source_rebuild_main4_panel_e_helper",
        _runtime_root() / "source_rebuild" / "preview_figure2_assembly.py",
        synthetic_file=VOR_ROOT / "SOURCE_REBUILD_MAIN_RUNTIME.zip" / "preview_figure2_assembly.py",
    )
    helper.LINEAR_PF_DATA_ROOT = _linear_pf_root()
    linear_pf_manifest = _verify_linear_pf_tree(helper.LINEAR_PF_DATA_ROOT)
    linearized_utils = importlib.import_module("ca_analysis.linearized_pf_utils")
    expected_panel_e_scope = {
        "home_to_water": {"pairs": 29, "animals": 13, "matched": 8600, "stable": 2188},
        "water_to_home": {"pairs": 28, "animals": 14, "matched": 8438, "stable": 2044},
    }
    panel_e_scope: dict[str, dict[str, int]] = {}
    for run_label, expected in expected_panel_e_scope.items():
        pair_rows, _ = linearized_utils.load_pair_data(
            str(helper.LINEAR_PF_DATA_ROOT),
            run_label,
            stable_thr=0.3,
            include_place_fields=False,
            pair_filter="day1_min_runs",
        )
        pair_rows = [
            row
            for row in pair_rows
            if (str(row["id1"]), str(row["id2"]))
            != ("1616day1", "1616bday1")
        ]
        observed = {
            "pairs": len(pair_rows),
            "animals": len({linearized_utils.get_animal(str(row["id1"])) for row in pair_rows}),
            "matched": sum(int(row["n_matched"]) for row in pair_rows),
            "stable": sum(int(row["n_stable"]) for row in pair_rows),
        }
        if observed != expected:
            raise RuntimeError(
                f"Figure 4e {run_label} scope drift: observed={observed}, expected={expected}"
            )
        panel_e_scope[run_label] = observed
    panel_e_style = publication.PublicationStyle(dpi=600)
    with plt.rc_context(panel_e_style.rcparams()):
        panel_e_figure = helper._render_panel_l(
            (module.SLOTS["e"][2], module.SLOTS["e"][3]),
            heatmap_vmax=0.6,
        )
        for axis in panel_e_figure.axes[:3]:
            position = axis.get_position()
            axis.set_position(
                [position.x0 + 0.025, position.y0, position.width, position.height]
            )
        for axis in panel_e_figure.axes:
            axis.xaxis.label.set_size(5.0)
            axis.yaxis.label.set_size(5.2)
            axis.title.set_size(5.8)
            axis.tick_params(labelsize=4.8, pad=1.0)
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(4.8)
        for density_axis in (panel_e_figure.axes[2], panel_e_figure.axes[5]):
            density_axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        for text in panel_e_figure.texts:
            text.set_fontsize(5.2)
        panel_e_figure.canvas.draw()
        density_ticks = [
            [tick.get_text() for tick in panel_e_figure.axes[index].get_xticklabels()]
            for index in (2, 5)
        ]
        panel_e_payload_spec = publication._figure_to_svg_payload(
            "figure4_e",
            panel_e_figure,
            dpi=600,
            fixed_bbox=True,
        )
    plt.close(panel_e_figure)
    panel_e_exports_raw = publication._write_svg_payload_outputs(
        panel_e_payload_spec,
        output_dir,
        "main_figure_4_panel_e_source_rebuild",
        dpi=600,
        facecolor=panel_e_style.facecolor,
    )
    panel_e_exports = {
        kind: Path(path)
        for kind, path in panel_e_exports_raw.items()
        if kind in {"svg", "pdf", "png", "html"}
    }
    panel_e_svg = panel_e_exports["svg"]
    panel_e_payload = panel_e_svg.read_bytes()
    panel_c_rows = output_dir / "main_figure_4_panel_c_source_rows.csv"
    illustration = record / "illustration_data" / "spatial_decoder_example.npz"
    panel_a_exports: list[Path] = []
    if illustration.is_file():
        publication.FIGURE4_SCHEMATIC_DATA_ROOT = illustration.parent
        with plt.rc_context(panel_e_style.rcparams()):
            schematic_figure, schematic_metadata = publication._render_figure4_method_schematic(
                (module.SLOTS["a"][2], module.SLOTS["a"][3]), panel_e_style,
            )
            schematic_payload = module.figure_svg(schematic_figure)
        plt.close(schematic_figure)
        schematic_svg = output_dir / "main_figure_4_panel_a_source_rebuild.svg"
        schematic_svg.write_bytes(schematic_payload)
        panel_a_exports = _export_svg_bundle(
            schematic_svg, output_dir / "main_figure_4_panel_a_source_rebuild"
        )
        (output_dir / "main_figure_4_schematic_input_check.json").write_text(
            json.dumps(schematic_metadata, indent=2), encoding="utf-8",
        )
    else:
        schematic_payload = _svg_payload(
            record / "source_validation/figure4_panel_a_refinement/figure4_panel_a_decoding_method_schematic.svg"
        )
    payloads = {
        "a": schematic_payload,
        "b": module.figure_svg(
            module.paired_panel(
                panel_b_rows,
                panel_b_stat,
                variant="d1",
                panel="b",
                size=(module.SLOTS["b"][2], module.SLOTS["b"][3]),
            )
        ),
        "c": module.figure_svg(
            module.map_panel(
                map_rows.loc[map_rows["variant"].eq("d1")],
                variant="d1",
                size=(module.SLOTS["c"][2], module.SLOTS["c"][3]),
                source_csv=panel_c_rows,
            )
        ),
        "d": module.figure_svg(
            module.paired_panel(
                d_rows,
                d_stat,
                variant="d1",
                panel="d",
                size=(module.SLOTS["d"][2], module.SLOTS["d"][3]),
            )
        ),
        "e": panel_e_payload,
        "f": module.figure_svg(
            aligned_panel_f.render_panel(
                matched,
                shared_limits=shared_limits,
                size=(module.SLOTS["f"][2], module.SLOTS["f"][3]),
            )
            if aligned_panel_f is not None
            else module.behavior_panel(
                matched,
                f_stat,
                variant="d1",
                shared_limits=shared_limits,
                size=(module.SLOTS["f"][2], module.SLOTS["f"][3]),
            )
        ),
    }
    fresh_images = _embedded_image_hashes(panel_e_payload)
    accepted_reference = _optional_record_qa_reference(
        record, "panel_e_source/figure4_panel_e_source.svg"
    )
    accepted_images = (
        _embedded_image_hashes(accepted_reference["payload"])
        if accepted_reference is not None
        else []
    )
    image_parity = accepted_reference is not None and fresh_images == accepted_images
    if accepted_reference is not None and not image_parity:
        raise RuntimeError(
            "Fresh Figure 4e quantitative image payloads differ from the accepted QA reference"
        )
    if len(fresh_images) != 5:
        raise RuntimeError(f"Fresh Figure 4e must contain five embedded image payloads, found {len(fresh_images)}")
    panel_e_parity = output_dir / "main_figure_4_panel_e_source_parity.json"
    panel_e_parity.write_text(
        json.dumps(
            {
                "status": "PASS" if accepted_reference is not None else "NOT_RUN",
                "scientific_input": "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip",
                "scientific_input_sha256": VOR_INPUT_SPECS[
                    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip"
                ]["sha256"],
                "scientific_input_bytes": VOR_INPUT_SPECS[
                    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF.zip"
                ]["bytes"],
                "scientific_input_manifest": (
                    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF_MANIFEST.json"
                ),
                "scientific_input_manifest_sha256": VOR_INPUT_SPECS[
                    "SOURCE_REBUILD_MAIN_FIGURE4_PANEL_E_LINEAR_PF_MANIFEST.json"
                ]["sha256"],
                "source_repo_relative_path": linear_pf_manifest[
                    "source_repo_relative_path"
                ],
                "source_tree_sha256": linear_pf_manifest["source_tree"]["tree_sha256"],
                "source_tree_uncompressed_bytes": linear_pf_manifest["source_tree"][
                    "uncompressed_bytes"
                ],
                "npy_file_count": 464,
                "session_run_counts_present": True,
                "representation_space": "22-bin linearized home-to-water and water-to-home routes",
                "filter_provenance": {
                    "pair_filter": "day1_min_runs",
                    "minimum_runs_per_session_and_direction": 5,
                    "stable_threshold": 0.3,
                    "heatmap_neuron_set": "all valid matched neurons",
                    "excluded_pair": ["1616day1", "1616bday1"],
                    "session_exclusions": sorted(linearized_utils.EXCLUDED_SESSIONS),
                },
                "expected_scope": expected_panel_e_scope,
                "observed_scope": panel_e_scope,
                "heatmap_vmax": 0.6,
                "density_ticks": density_ticks,
                "fresh_embedded_image_sha256": fresh_images,
                "accepted_qa_reference": (
                    {
                        key: accepted_reference[key]
                        for key in ("path", "sha256", "bytes")
                    }
                    if accepted_reference is not None
                    else None
                ),
                "accepted_embedded_image_sha256": accepted_images,
                "exact_embedded_image_payload_parity": image_parity,
                "source_identity_scope": (
                    "The VOR archive freezes the maintained sorted_export directory named by "
                    "the accepted Figure 4 source manifest and exactly reproduces all five "
                    "accepted embedded raster payloads."
                    if image_parity
                    else "Accepted display reference unavailable; source-directory identity is unverified."
                ),
                "qa_reference_is_optional_and_not_a_scientific_input": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    svg = output_dir / "main_figure_4_source_rebuild.svg"
    _compose_main4(publication, module, payloads, svg)
    notes = output_dir / "main_figure_4_source_declarations.json"
    notes.write_text(
        json.dumps(
            {
                "status": "PASS",
                "fixed_assets": check["fixed_assets"],
                "quantitative_panels_rendered_from_sources": ["b", "c", "d", "e", "f"],
                "schematic_numeric_input": (
                    "illustration_data/spatial_decoder_example.npz"
                    if illustration.is_file() else None
                ),
                "panel_b_null": "within-pair neuron ID shuffle, 1000 repeats per pair",
                "panel_e_source": "VOR-root frozen linearized place-field NPY archive",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return [
        *_export_main4_svg_bundle(
            svg, output_dir / "main_figure_4_source_rebuild", module
        ),
        panel_c_rows,
        *panel_a_exports,
        *panel_e_exports.values(),
        panel_e_parity,
        notes,
    ]


def _render_main5(record: Path, output_dir: Path, check: dict[str, Any]) -> list[Path]:
    _install_frozen_panel_modules(record)
    publication = _load_publication_renderer(record, "record_local_main5_publication")
    module = _load_module(
        "record_local_main5_preview",
        record / "source_code" / "preview_figure3_assembly.py",
    )
    bundle = record / "source_data" / "figure5_bundle"
    shuffle = record / "source_data" / "neuron_id_shuffle"
    module.FIGURE5_CORRECTED_842_DIR = bundle
    module.FIGURE5_SOURCE_MANIFEST = bundle / "SOURCE_MANIFEST.json"
    module.FIGURE5_LEVEL_DISTRIBUTION_DIR = bundle
    module.FIGURE5_LEVEL_HEATMAP_PLOT_READY = (
        record / "source_data" / "level_vectors_tree_depth7_corrected36_d1v1.csv"
    )
    module.FIGURE5_LEVEL_HEATMAP_PLOT_READY_SHA256 = _sha256(
        module.FIGURE5_LEVEL_HEATMAP_PLOT_READY
    )
    module.FIGURE5_DECODER_DIR = shuffle
    module.SORTED_LEVEL_EXPLORATION_DIR = bundle

    def source_validation() -> dict[str, Any]:
        scope = pd.read_csv(bundle / "figure5_pair_scope_audit.csv")
        level_rows = pd.read_csv(module.FIGURE5_LEVEL_HEATMAP_PLOT_READY)
        if len(scope) != 36:
            raise RuntimeError("Main Figure 5 source scope is not 36 pairs")
        return {
            "status": "PASS",
            "included_pairs": 36,
            "level_neurons_total": int(len(level_rows)),
            "source": "record-local frozen Figure 5 bundle",
        }

    module.validate_figure5_corrected842_sources = source_validation
    historical_order = (
        record
        / "source_validation"
        / "refinement_2026_08_12"
        / "figure5b_historical_hierarchical_refined_order.npy"
    )
    module.PANEL_RENDERERS["e"] = lambda size: module._render_panel_e(
        size,
        refined_order_path=historical_order,
        refined_order_sha256=_sha256(historical_order),
    )
    original_decoder_renderer = module.PANEL_RENDERERS["d"]

    def render_decoder(size):
        # The accepted decoder export inherited these Figure 4 tick dimensions.
        with matplotlib.rc_context({
            "xtick.major.size": 1.8, "ytick.major.size": 1.8,
            "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        }):
            return original_decoder_renderer(size)

    module.PANEL_RENDERERS["d"] = render_decoder
    output_dir.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"savefig.dpi": 600}):
        tree = module.render_assembly_svg(dpi=600, row="all")
    # Reproduce the accepted publication top margin on newly rendered panels.
    top_margin_in = 0.12
    top_margin_pt = top_margin_in * 72.0
    root = tree.getroot()
    view_x, view_y, view_width, view_height = map(float, root.attrib["viewBox"].split())
    root.attrib["viewBox"] = (
        f"{view_x:.6f} {view_y:.6f} {view_width:.6f} {view_height + top_margin_pt:.6f}"
    )
    height_mm = round(module.EXPORT_WIDTH_MM * (module.CANVAS_SIZE[1] + top_margin_in) / module.CANVAS_SIZE[0], 3)
    root.attrib["height"] = f"{height_mm:.3f}mm"
    for child in root:
        if child.tag in {module._svg_tag("metadata"), module._svg_tag("defs")}:
            continue
        if child.tag == module._svg_tag("rect") and child.get("width") == "100%" and child.get("height") == "100%":
            continue
        child.attrib["transform"] = f"translate(0 {top_margin_pt:.6f}) {child.get('transform', '')}".strip()
    svg = output_dir / "main_figure_5_source_rebuild.svg"
    tree.write(svg, encoding="utf-8", xml_declaration=True)
    audit = output_dir / "main_figure_5_source_rebuild_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selected_record": check["selected_record"],
                "display_panel_b_order": "record-frozen historical average-linkage order",
                "display_panel_e_null": "record-frozen within-pair neuron ID shuffle",
                "source_validation": source_validation(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    png = output_dir / "main_figure_5_source_rebuild_600dpi.png"
    pdf = output_dir / "main_figure_5_source_rebuild.pdf"
    tiff = png.with_suffix(".tiff")
    # Browser screenshots preserve the accepted 4250 x 2938 raster canvas.
    html = publication._export_svg_with_chrome(
        svg, png, pdf, width_mm=module.EXPORT_WIDTH_MM, height_mm=height_mm, dpi=600,
    )
    with Image.open(png) as image:
        image.save(tiff, compression="tiff_lzw", dpi=(600, 600))
    return [svg, html, pdf, png, tiff, audit]


RENDERERS: dict[str, Callable[[Path, Path, dict[str, Any]], list[Path]]] = {
    "main_figure_1": _render_main1,
    "main_figure_2": _render_main2,
    "main_figure_3": _render_main3,
    "main_figure_4": _render_main4,
    "main_figure_5": _render_main5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=(*ARTIFACTS, "all"), required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify pointer selection, source closure, hashes, and forbidden-source policy without rendering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for rebuilt files. Required unless --check is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = list(ARTIFACTS) if args.artifact == "all" else [args.artifact]
    checks = {artifact: check_artifact(artifact) for artifact in artifacts}
    if args.check:
        print(json.dumps(checks if len(checks) > 1 else checks[artifacts[0]], indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --check is used")
    root_output = args.output_dir.resolve()
    pointer_records = _pointer_records()
    results: dict[str, Any] = {}
    for artifact in artifacts:
        artifact_output = root_output / artifact if len(artifacts) > 1 else root_output
        # Frozen panel helpers can set process-global rcParams (notably
        # ``savefig.bbox='tight'``).  Recreate fresh-process Matplotlib state
        # at every artifact boundary so ``--artifact all`` is identical to
        # five independent invocations and cannot distort later SVG panels.
        plt.close("all")
        matplotlib.rcdefaults()
        outputs = RENDERERS[artifact](pointer_records[artifact], artifact_output, checks[artifact])
        for output in outputs:
            if not output.is_file():
                raise RuntimeError(f"Renderer did not create expected output: {output}")
        manifest = _write_result_manifest(
            artifact_output,
            checks[artifact],
            outputs,
            notes=[
                "All scientific inputs resolved from the pointer-selected immutable record.",
                "No current master export, archived parent, candidate, Desktop, or ltm-figs file was used as a quantitative source.",
            ],
        )
        results[artifact] = {
            "status": "PASS",
            "output_dir": artifact_output.as_posix(),
            "manifest": manifest.as_posix(),
        }
    print(json.dumps(results if len(results) > 1 else results[artifacts[0]], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
