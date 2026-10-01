"""Fixed-reference Day i window convergence curves.

This exporter asks whether overlapping Day i maps approach a fixed full
Day i+1 reference map. It is deliberately separate from the existing
within-session pivot analyses, which compare pre/post windows inside one
session.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ca_analysis.compat import bootstrap_environment
from ca_analysis.config import DT, get_behav_wd, get_neural_wd
from ca_analysis.maze import NewMaze
from ca_analysis.similarity import _spatial_labels_from_inputs
from ca_analysis.spatial import (
    LEVEL_DEFINITION_GEOMETRIC8,
    normalize_level_definition,
)
from scripts import analysis_common as ac


DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "fixed_reference_window_convergence_2026-05-15"
DEFAULT_PAIR_FAMILIES = ("naive_day1_vs_naive_day2", "ltm_day1_vs_ltm_day2")
DEFAULT_TIME_WINDOWS_MIN = (10.0, 20.0, 30.0)
DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC = 5.0 * 60.0
DEFAULT_TILE_WINDOWS = (1000, 2000, 3000)
DEFAULT_TILE_AXIS_SOURCE = "selected_bodymass"
TILE_AXIS_SOURCES = ("selected_bodymass", "corrected_all_frame")
DEFAULT_WINDOW_TYPES = ("selected_time", "maze_time", "absolute_time", "tile_change")
DEFAULT_SUPPORT_POLICIES = ("scaled_nonoverlap", "figure4_fixed", "legacy_place_frame")
RATE_SUPPORT_POLICY = "not_applicable"
METRIC_CHOICES = (
    "place_map_correlation",
    "level_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)
DEFAULT_METRICS = ("place_map_correlation", "level_map_correlation")
METRIC_TO_MAP_TYPE = {
    "place_map_correlation": "place",
    "level_map_correlation": "level",
    "pv_correlation": "place",
}
METRIC_LABELS = {
    "place_map_correlation": "Place-map correlation",
    "level_map_correlation": "Level-map correlation",
    "pv_correlation": "PV correlation",
    "ensemble_rate_correlation": "Ensemble-rate correlation",
}
GROUP_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")
GROUP_COLORS = {
    "naive": "#4c78a8",
    "3mo": "#d62728",
    "6mo": "#b2182b",
    "9mo": "#7f0000",
    "12mo": "#4d0000",
}
MAP_COLORS = {"place": "#B95A58", "level": "#7976a2"}
EXPANDED_HARD_EXCLUSIONS = {
    "579-6day1",
    "579-6day2",
    "862-9day1",
    "862-9day2",
}


@dataclass(frozen=True)
class WindowDef:
    window_type: str
    window_size: float
    window_size_label: str
    stride: float
    stride_label: str
    bounds: np.ndarray
    rows: list[dict[str, Any]]
    n_nonoverlap: int
    span_value: float
    span_unit: str
    skip_reason: str = ""


def load_cimap_pairs(
    path_or_paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    *,
    split: bool = True,
    zero_based: bool = True,
    compose: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Load two-column CI maps without importing unrelated control modules."""

    def _load_single_matrix(path: str | Path) -> np.ndarray:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"CIMap not found: {path_obj}")
        mat = scipy.io.loadmat(str(path_obj))
        if "cell_to_index_map" in mat:
            raw = mat["cell_to_index_map"]
        else:
            keys = [key for key in mat if not key.startswith("__")]
            if not keys:
                raise KeyError(f"No non-private arrays found in {path_obj}")
            raw = mat[keys[0]]
        pairs = np.asarray(raw, dtype=int)
        if pairs.ndim == 1:
            pairs = pairs.reshape(-1, 1)
        if pairs.shape[1] < 2:
            raise ValueError(f"CIMap requires at least two columns, got shape {pairs.shape}")
        pairs = pairs[:, :2]
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        if zero_based:
            pairs = pairs - 1
        return pairs.astype(int, copy=False)

    def _compose_pair_chain(pair_list: list[np.ndarray]) -> np.ndarray:
        if not pair_list:
            return np.zeros((0, 2), dtype=int)
        current = np.asarray(pair_list[0], dtype=int)
        if current.ndim != 2 or current.shape[1] != 2:
            raise ValueError("CIMap chain items must be 2-column arrays.")
        for nxt in pair_list[1:]:
            nxt = np.asarray(nxt, dtype=int)
            if nxt.ndim != 2 or nxt.shape[1] != 2:
                raise ValueError("CIMap chain items must be 2-column arrays.")
            if current.size == 0 or nxt.size == 0:
                return np.zeros((0, 2), dtype=int)
            right_to_final = {int(mid): int(final) for mid, final in nxt.tolist()}
            chained = []
            for start, mid in current.tolist():
                final = right_to_final.get(int(mid))
                if final is not None:
                    chained.append((int(start), int(final)))
            current = np.asarray(chained, dtype=int)
            if current.size == 0:
                return np.zeros((0, 2), dtype=int)
        return current.reshape(-1, 2)

    if path_or_paths is None or str(path_or_paths).strip() == "":
        pairs = np.zeros((0, 2), dtype=int)
    elif isinstance(path_or_paths, (list, tuple)):
        paths = [path for path in path_or_paths if path is not None and str(path).strip()]
        matrices = [_load_single_matrix(path) for path in paths]
        if compose:
            pairs = _compose_pair_chain(matrices)
        else:
            if len(matrices) != 1:
                raise ValueError("load_cimap_pairs received multiple paths without compose=True.")
            pairs = matrices[0]
    else:
        if compose:
            raise ValueError("compose=True requires a list/tuple of CIMap paths.")
        pairs = _load_single_matrix(path_or_paths)
    if split:
        if pairs.size == 0:
            return np.asarray([], dtype=int), np.asarray([], dtype=int)
        return pairs[:, 0].astype(int), pairs[:, 1].astype(int)
    return pairs.astype(int, copy=False)


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_csv_float(text: str) -> list[float]:
    return [float(v) for v in _parse_csv(text)]


def _parse_csv_int(text: str) -> list[int]:
    return [int(float(v)) for v in _parse_csv(text)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _experience_group(row: pd.Series) -> str:
    if str(row.get("pair_family", "")) == "naive_day1_vs_naive_day2":
        return "naive"
    stage = str(row.get("memory_stage", "")).strip()
    return stage if stage in GROUP_ORDER else "return"


def _resolve_pair_ids(row: pd.Series, n_a: int, n_b: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return neuron IDs for session_a and session_b in row order."""
    mode = str(row.get("matching_mode", ""))
    if mode == "index_identity":
        n = int(min(n_a, n_b))
        ids = np.arange(n, dtype=int)
        return ids, ids.copy(), "index_identity"
    cimap_text = str(row.get("cimap_path", "")).strip()
    if not cimap_text:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), "missing_cimap"
    parts = [Path(part.strip()) for part in cimap_text.split(";") if part.strip()]
    compose = len(parts) > 1
    a_ids, b_ids = load_cimap_pairs(parts if compose else parts[0], split=True, compose=compose)
    a_ids = np.asarray(a_ids, dtype=int)
    b_ids = np.asarray(b_ids, dtype=int)
    keep = (a_ids >= 0) & (b_ids >= 0) & (a_ids < int(n_a)) & (b_ids < int(n_b))
    return a_ids[keep], b_ids[keep], "cimap_composed" if compose else "cimap_direct"


def _session_neural_path(root: str | Path, session_id: str) -> Path | None:
    from ca_analysis.io import _extract_info

    parsed = _extract_info(str(session_id))
    if parsed is None:
        return None
    mouse_id, exp_type, exp_date = parsed
    root = Path(root)
    neural_dir = root / mouse_id
    if neural_dir.exists() and exp_type != 0:
        neural_dir = root / f"{mouse_id}-{exp_type}"
    candidates = [
        neural_dir / f"final_day{exp_date}.mat",
        neural_dir / f"{exp_date}.mat",
        neural_dir / f"day{exp_date}.mat",
        neural_dir / f"final_{exp_date}.mat",
    ]
    return next((p for p in candidates if p.exists()), None)


def _resolve_neural_source_rows(use: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Attach per-session neural roots for single-root or sorted-first runs."""
    out = use.copy()
    default_root = str(args.neural_wd)
    sorted_root = str(getattr(args, "sorted_neural_wd", "") or "").strip()
    policy = str(getattr(args, "neural_source_policy", "single_root"))
    if policy != "sorted_first" or not sorted_root:
        out["neural_source_policy"] = "single_root"
        out["target_neural_source"] = "single_root"
        out["reference_neural_source"] = "single_root"
        out["target_neural_wd"] = default_root
        out["reference_neural_wd"] = default_root
        return out

    cache: dict[str, tuple[str, str]] = {}

    def source_for(session_id: str) -> tuple[str, str]:
        session = str(session_id)
        if session not in cache:
            if _session_neural_path(sorted_root, session) is not None:
                cache[session] = ("sorted", sorted_root)
            else:
                cache[session] = ("original_fallback", default_root)
        return cache[session]

    target_sources: list[str] = []
    reference_sources: list[str] = []
    target_roots: list[str] = []
    reference_roots: list[str] = []
    pair_policies: list[str] = []
    for _, row in out.iterrows():
        target_source, target_root = source_for(str(row["session_a"]))
        reference_source, reference_root = source_for(str(row["session_b"]))
        target_sources.append(target_source)
        reference_sources.append(reference_source)
        target_roots.append(target_root)
        reference_roots.append(reference_root)
        if target_source == "sorted" and reference_source == "sorted":
            pair_policies.append("all_sorted")
        elif target_source == "original_fallback" and reference_source == "original_fallback":
            pair_policies.append("all_original_fallback")
        else:
            pair_policies.append("mixed_sorted_original_fallback")
    out["neural_source_policy"] = pair_policies
    out["target_neural_source"] = target_sources
    out["reference_neural_source"] = reference_sources
    out["target_neural_wd"] = target_roots
    out["reference_neural_wd"] = reference_roots
    return out


def _dataset_axes_for_frames(shape: tuple[int, int], frames: np.ndarray) -> tuple[int, int, int]:
    max_frame = int(np.nanmax(frames)) if np.asarray(frames).size else -1
    if int(shape[0]) > max_frame:
        return 0, int(shape[0]), int(shape[1])
    if int(shape[1]) > max_frame:
        return 1, int(shape[1]), int(shape[0])
    raise ValueError(f"Neural matrix shape {shape} shorter than max selected frame {max_frame}")


def _nan_frames_for_dataset(path: str | Path, frames: np.ndarray, *, dataset: str = "C") -> tuple[np.ndarray, tuple[int, int, int]]:
    import h5py

    frames = np.asarray(frames, dtype=int)
    with h5py.File(str(path), "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset {dataset!r} not found in {path}")
        obj = handle[dataset]
        if not hasattr(obj, "shape") or len(obj.shape) != 2:
            raise ValueError(f"Unsupported nonmatrix {dataset!r} dataset in {path}")
        frame_axis, n_frames, n_neurons = _dataset_axes_for_frames(tuple(int(v) for v in obj.shape), frames)
        nan_chunks: list[np.ndarray] = []
        chunk = 10000
        for start in range(0, n_frames, chunk):
            stop = min(start + chunk, n_frames)
            if frame_axis == 0:
                block = np.asarray(obj[start:stop, :])
                bad = np.any(np.isnan(block), axis=1)
            else:
                block = np.asarray(obj[:, start:stop])
                bad = np.any(np.isnan(block), axis=0)
            if np.any(bad):
                nan_chunks.append(np.flatnonzero(bad).astype(int) + int(start))
    if nan_chunks:
        return np.concatenate(nan_chunks).astype(int), (frame_axis, n_frames, n_neurons)
    return np.asarray([], dtype=int), (frame_axis, n_frames, n_neurons)


def _load_bundle(session_id: str, *, behav_wd: str, neural_wd: str, level_definition: str, level_col: str) -> dict[str, Any]:
    selected_path = Path(behav_wd) / "selected_data" / f"{session_id}_selected_data.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected_data CSV: {selected_path}")
    selected_df = pd.read_csv(selected_path)
    neural_path = _session_neural_path(neural_wd, session_id)
    if neural_path is None:
        raise FileNotFoundError(f"Missing neural mat for {session_id} under {neural_wd}")
    frames = selected_df["frame"].to_numpy(dtype=int)
    nan_idx, axes = _nan_frames_for_dataset(neural_path, frames, dataset="C")
    if nan_idx.size:
        selected_df = selected_df.loc[~selected_df["frame"].isin(nan_idx)].reset_index(drop=True)
    else:
        selected_df = selected_df.reset_index(drop=True)
    if level_col not in selected_df.columns:
        _, maze_ce, level_list = ac._level_assignment_context(level_definition)
        selected_df = selected_df.copy()
        selected_df[level_col] = ac.assign_levels(selected_df, maze_ce, level_list)
        keep = selected_df[level_col].notna().to_numpy(dtype=bool)
        selected_df = selected_df.loc[keep].reset_index(drop=True)
    positions, positions_source = ac._maze_aware_positions(selected_df)
    return {
        "session_id": session_id,
        "selected_df": selected_df,
        "positions": positions,
        "positions_source": positions_source,
        "nan_frame_count": int(nan_idx.size),
        "neural_path": str(neural_path),
        "neural_frame_axis": int(axes[0]),
        "neural_n_frames": int(axes[1]),
        "neural_n_neurons": int(axes[2]),
    }


def _signal_matrix_from_path(bundle: dict[str, Any], signal_source: str, neuron_ids: np.ndarray) -> np.ndarray:
    source = str(signal_source).upper()
    dataset = "C" if source == "C" else "S" if source == "S" else ""
    if not dataset:
        raise ValueError("signal_source must be C or S")
    import h5py

    frames = bundle["selected_df"]["frame"].to_numpy(dtype=int)
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if frames.size == 0 or neuron_ids.size == 0:
        return np.zeros((frames.size, neuron_ids.size), dtype=np.float32)
    out = np.empty((frames.size, neuron_ids.size), dtype=np.float32)
    with h5py.File(str(bundle["neural_path"]), "r") as handle:
        obj = handle[dataset]
        frame_axis, _n_frames, n_neurons = _dataset_axes_for_frames(tuple(int(v) for v in obj.shape), frames)
        if np.any((neuron_ids < 0) | (neuron_ids >= int(n_neurons))):
            raise ValueError(f"Neuron id exceeds {dataset} matrix bounds for {bundle['session_id']}")
        chunk = 10000
        for r0 in range(0, frames.size, chunk):
            r1 = min(r0 + chunk, frames.size)
            frame_chunk = frames[r0:r1]
            if frame_axis == 0:
                block = np.asarray(obj[frame_chunk, :])[:, neuron_ids]
            else:
                block = np.asarray(obj[:, frame_chunk]).T[:, neuron_ids]
            out[r0:r1] = np.asarray(block, dtype=np.float32)
    return out


def _signal_matrix(bundle: dict[str, Any], signal_source: str) -> np.ndarray:
    source = str(signal_source).upper()
    if source == "C":
        frames = bundle["selected_df"]["frame"].to_numpy(dtype=int)
        raw = np.asarray(bundle["raw_counts"])
        if frames.size and int(np.nanmax(frames)) >= raw.shape[0]:
            raise ValueError(f"Frame index exceeds raw C matrix length for {bundle['session_id']}")
        if frames.size == 0:
            return np.asarray(raw[:0], dtype=np.float32)
        if np.array_equal(frames, np.arange(int(frames[0]), int(frames[0]) + frames.size, dtype=int)):
            return np.asarray(raw[int(frames[0]) : int(frames[0]) + frames.size], dtype=np.float32).copy()
        return np.asarray(raw[frames], dtype=np.float32).copy()
    if source == "S":
        return np.asarray(bundle["spike_counts"], dtype=np.float32).copy()
    raise ValueError("signal_source must be C or S")


def _select_signal_columns(signal: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=int)
    if ids.size == signal.shape[1] and np.array_equal(ids, np.arange(signal.shape[1], dtype=int)):
        return signal
    return np.asarray(signal[:, ids], dtype=np.float32)


def _drop_signal_payload(bundle: dict[str, Any]) -> None:
    bundle.pop("raw_counts", None)
    bundle.pop("spike_counts", None)


def _labels_for_map(bundle: dict[str, Any], map_type: str, *, level_col: str, n_levels: int) -> tuple[np.ndarray, int, str]:
    if map_type == "place":
        labels, _, _, source = _spatial_labels_from_inputs(
            bundle["selected_df"],
            np.asarray(bundle["positions"], dtype=float),
            n_bins_x=15,
            n_bins_y=15,
        )
        labels = np.asarray(labels, dtype=int).ravel()
        labels[(labels < 0) | (labels >= 225)] = -1
        return labels, 225, str(source)
    if map_type == "level":
        vals = pd.to_numeric(bundle["selected_df"][level_col], errors="coerce").to_numpy(dtype=float)
        labels = np.full(vals.shape, -1, dtype=int)
        finite = np.isfinite(vals)
        labels[finite] = vals[finite].astype(int)
        labels[(labels < 0) | (labels >= int(n_levels))] = -1
        return labels, int(n_levels), str(level_col)
    raise ValueError(f"Unknown map_type: {map_type}")


def _add_bodymass_tile_axis(selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = selected.copy()
    labels, _, _, label_source = _spatial_labels_from_inputs(
        out,
        out[["mass_x", "mass_y"]].to_numpy(dtype=float),
        n_bins_x=15,
        n_bins_y=15,
    )
    valid = np.isfinite(labels) & (labels >= 0) & (labels < 225)
    labels_i = np.full(labels.shape, -1, dtype=int)
    labels_i[valid] = labels[valid].astype(int)
    x = labels_i % 15
    y = labels_i // 15
    change = np.zeros(len(out), dtype=float)
    manhattan = np.zeros(len(out), dtype=float)
    if len(out) > 1:
        pair_valid = valid[1:] & valid[:-1]
        dist = (np.abs(x[1:] - x[:-1]) + np.abs(y[1:] - y[:-1])).astype(float)
        change[1:] = (pair_valid & (dist > 0)).astype(float)
        manhattan[1:] = np.where(pair_valid, dist, 0.0)
    out["place_bin_id_tile_axis"] = labels_i
    out["tile_change_step"] = change
    out["tile_manhattan_step"] = manhattan
    out["tile_manhattan_step_capped"] = np.minimum(manhattan, 6.0)
    out["cum_tile_change_count"] = np.cumsum(change)
    out["cum_tile_manhattan_steps"] = np.cumsum(manhattan)
    out["cum_tile_manhattan_steps_capped"] = np.cumsum(out["tile_manhattan_step_capped"].to_numpy(dtype=float))
    return out, {
        "tile_axis_source": "selected_bodymass",
        "tile_axis_position_source": str(label_source),
        "n_valid_tile_axis_labels": int(np.count_nonzero(valid)),
        "n_tile_change_events": int(np.sum(change > 0)),
        "total_tile_manhattan_steps": float(np.sum(manhattan)),
        "total_tile_manhattan_steps_capped": float(np.sum(out["tile_manhattan_step_capped"])),
        "max_tile_manhattan_step": float(np.nanmax(manhattan)) if manhattan.size else np.nan,
    }


def _legacy_binned_xy(x_raw: pd.Series, y_raw: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Bin normalized coordinates with the selected_df 15x15 convention."""
    source_x = pd.to_numeric(x_raw, errors="coerce").to_numpy(dtype=float)
    source_y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
    x = 1.0 - source_y
    y = 1.0 - source_x
    step = -1.0 / 15.0
    with np.errstate(invalid="ignore"):
        bx = np.floor((x - 1.0) / step)
        by = np.floor((y - 1.0) / step)
    bad = ~np.isfinite(x) | ~np.isfinite(y) | (bx < 0) | (bx >= 15) | (by < 0) | (by >= 15)
    bx = bx.astype(float, copy=False)
    by = by.astype(float, copy=False)
    bx[bad] = np.nan
    by[bad] = np.nan
    return bx, by


def _maze_cell_labels_from_bins(maze: Any, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    out = np.full(len(bx), -1, dtype=int)
    finite = np.isfinite(bx) & np.isfinite(by)
    for i in np.flatnonzero(finite):
        out[i] = int(maze.ce.get((int(by[i]), int(bx[i])), -1))
    return out


def _corrected_all_frame_tile_axis(nose_interp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build selected-style corrected tile changes on all trajectory frames."""
    raw = nose_interp.copy().reset_index(drop=True)
    required = {"frame", "x", "y", "mass_x", "mass_y"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"nose_interp_data missing required columns: {sorted(missing)}")

    maze = NewMaze(6)
    nose_bx, nose_by = _legacy_binned_xy(raw["x"], raw["y"])
    mass_bx, mass_by = _legacy_binned_xy(raw["mass_x"], raw["mass_y"])
    nose_cell = _maze_cell_labels_from_bins(maze, nose_bx, nose_by)
    mass_cell = _maze_cell_labels_from_bins(maze, mass_bx, mass_by)

    corr_bx = np.full(len(raw), np.nan, dtype=float)
    corr_by = np.full(len(raw), np.nan, dtype=float)
    rule = np.full(len(raw), "both_invalid", dtype=object)
    for i, (n_cell, m_cell) in enumerate(zip(nose_cell, mass_cell)):
        nose_valid = int(n_cell) >= 0
        mass_valid = int(m_cell) >= 0
        use_mass = False
        if nose_valid and mass_valid:
            use_mass = bool(float(maze.di[int(m_cell), int(n_cell)]) > 2.0)
            rule[i] = "mass_replaced_far_nose" if use_mass else "kept_nose"
        elif (not nose_valid) and mass_valid:
            use_mass = True
            rule[i] = "mass_replaced_invalid_nose"
        elif nose_valid and (not mass_valid):
            rule[i] = "kept_nose_mass_invalid"
        if use_mass:
            corr_bx[i] = mass_bx[i]
            corr_by[i] = mass_by[i]
        elif nose_valid:
            corr_bx[i] = nose_bx[i]
            corr_by[i] = nose_by[i]

    valid = np.isfinite(corr_bx) & np.isfinite(corr_by)
    labels = np.full(len(raw), -1, dtype=int)
    labels[valid] = corr_bx[valid].astype(int) + corr_by[valid].astype(int) * 15
    x = np.where(valid, labels % 15, -1)
    y = np.where(valid, labels // 15, -1)
    frames = pd.to_numeric(raw["frame"], errors="coerce").to_numpy(dtype=float)
    bouts = pd.to_numeric(raw.get("bout", pd.Series(np.nan, index=raw.index)), errors="coerce").to_numpy(dtype=float)

    change = np.zeros(len(raw), dtype=float)
    manhattan = np.zeros(len(raw), dtype=float)
    if len(raw) > 1:
        same_bout = (np.isfinite(bouts[1:]) & np.isfinite(bouts[:-1]) & (bouts[1:] == bouts[:-1])) | (~np.isfinite(bouts[1:]) & ~np.isfinite(bouts[:-1]))
        adjacent_frame = np.isfinite(frames[1:]) & np.isfinite(frames[:-1]) & ((frames[1:] - frames[:-1]) == 1.0)
        pair_valid = valid[1:] & valid[:-1] & same_bout & adjacent_frame
        dist = (np.abs(x[1:] - x[:-1]) + np.abs(y[1:] - y[:-1])).astype(float)
        change[1:] = (pair_valid & (dist > 0)).astype(float)
        manhattan[1:] = np.where(pair_valid, dist, 0.0)

    raw["corrected_tile_axis_bx"] = corr_bx
    raw["corrected_tile_axis_by"] = corr_by
    raw["corrected_tile_axis_label"] = labels
    raw["corrected_tile_axis_rule"] = rule
    raw["tile_change_step"] = change
    raw["tile_manhattan_step"] = manhattan
    raw["tile_manhattan_step_capped"] = np.minimum(manhattan, 6.0)
    raw["cum_tile_change_count"] = np.cumsum(change)
    raw["cum_tile_manhattan_steps"] = np.cumsum(manhattan)
    raw["cum_tile_manhattan_steps_capped"] = np.cumsum(raw["tile_manhattan_step_capped"].to_numpy(dtype=float))
    return raw, {
        "tile_axis_source": "corrected_all_frame",
        "tile_axis_position_source": "nose_interp_corrected_nose_mass_fallback",
        "n_raw_tile_axis_rows": int(len(raw)),
        "n_valid_raw_tile_axis_labels": int(np.count_nonzero(valid)),
        "n_raw_tile_change_events": int(np.sum(change > 0)),
        "total_raw_tile_manhattan_steps": float(np.sum(manhattan)),
        "total_raw_tile_manhattan_steps_capped": float(np.sum(raw["tile_manhattan_step_capped"])),
        "max_raw_tile_manhattan_step": float(np.nanmax(manhattan)) if manhattan.size else np.nan,
        "n_corrected_kept_nose": int(np.sum(rule == "kept_nose")),
        "n_corrected_mass_replaced_far_nose": int(np.sum(rule == "mass_replaced_far_nose")),
        "n_corrected_mass_replaced_invalid_nose": int(np.sum(rule == "mass_replaced_invalid_nose")),
        "n_corrected_kept_nose_mass_invalid": int(np.sum(rule == "kept_nose_mass_invalid")),
        "n_corrected_both_invalid": int(np.sum(rule == "both_invalid")),
    }


def _attach_corrected_all_frame_tile_axis(selected: pd.DataFrame, *, session_id: str, behav_wd: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = Path(behav_wd) / "selected_data" / f"{session_id}_nose_interp_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing nose_interp_data for corrected tile axis: {raw_path}")
    raw_axis, audit = _corrected_all_frame_tile_axis(pd.read_csv(raw_path))
    if "frame" not in selected.columns:
        raise ValueError("selected_df needs frame column for corrected all-frame tile axis")

    out = selected.copy()
    raw_unique = raw_axis.drop_duplicates("frame", keep="first").set_index("frame")
    frames = pd.to_numeric(out["frame"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(frames).all():
        raise ValueError("selected_df frame column contains non-finite values")
    frame_index = pd.Index(frames.astype(int), name="frame")
    cols = [
        "corrected_tile_axis_label",
        "corrected_tile_axis_bx",
        "corrected_tile_axis_by",
        "cum_tile_change_count",
        "cum_tile_manhattan_steps",
        "cum_tile_manhattan_steps_capped",
    ]
    aligned = raw_unique.reindex(frame_index)[cols].reset_index(drop=True)
    missing = aligned["cum_tile_change_count"].isna().to_numpy()
    for col in cols:
        out[col] = aligned[col].to_numpy()
    for col in ["cum_tile_change_count", "cum_tile_manhattan_steps", "cum_tile_manhattan_steps_capped"]:
        if out[col].isna().any():
            out[col] = out[col].ffill().bfill()
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            out[col] = vals - float(finite[0])
    cum = pd.to_numeric(out["cum_tile_change_count"], errors="coerce").to_numpy(dtype=float)
    delta = np.zeros(len(out), dtype=float)
    if len(out) > 1:
        pair_valid = np.isfinite(cum[1:]) & np.isfinite(cum[:-1])
        delta[1:] = np.where(pair_valid, np.maximum(cum[1:] - cum[:-1], 0.0), 0.0)
    out["tile_change_step"] = delta
    out["tile_manhattan_step"] = np.nan
    out["tile_manhattan_step_capped"] = np.nan
    out["place_bin_id_tile_axis"] = pd.to_numeric(out["corrected_tile_axis_label"], errors="coerce").fillna(-1).astype(int)
    audit.update(
        {
            "n_valid_tile_axis_labels": int(np.count_nonzero(~missing)),
            "n_missing_selected_tile_axis_frames": int(np.count_nonzero(missing)),
            "n_tile_change_events": int(np.nanmax(cum)) if np.isfinite(cum).any() else 0,
            "total_tile_manhattan_steps": np.nan,
            "total_tile_manhattan_steps_capped": np.nan,
            "max_tile_manhattan_step": np.nan,
        }
    )
    return out, audit


def _add_tile_axis(selected: pd.DataFrame, *, session_id: str, behav_wd: str, tile_axis_source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = str(tile_axis_source)
    if source == "selected_bodymass":
        return _add_bodymass_tile_axis(selected)
    if source == "corrected_all_frame":
        return _attach_corrected_all_frame_tile_axis(selected, session_id=session_id, behav_wd=behav_wd)
    raise ValueError(f"Unknown tile_axis_source: {tile_axis_source}")


def _positive_mean_diff(values: np.ndarray, fallback: float = DT) -> float:
    arr = np.asarray(values, dtype=float)
    d = np.diff(arr)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.mean(d)) if d.size else float(fallback)


def _mask_indices(mask: np.ndarray) -> str:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return ",".join(str(int(v)) for v in idx)


def _safe_atanh(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    ok = np.isfinite(arr)
    out[ok] = np.arctanh(np.clip(arr[ok], -0.999999, 0.999999))
    return out


def _time_window_def(
    selected: pd.DataFrame,
    *,
    axis: str,
    window_min: float,
    stride_sec: float,
    min_selected_duration_sec: float = DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC,
) -> WindowDef:
    n = int(len(selected))
    if n < 2:
        return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, 0.0, "sec")
    if axis == "selected_time":
        t = np.arange(n, dtype=float) * float(DT)
        span = float(n) * float(DT)
        maze_time = pd.to_numeric(selected.get("maze_time", pd.Series(np.arange(n) * DT)), errors="coerce").to_numpy(dtype=float)
    elif axis == "maze_time":
        if "maze_time" not in selected.columns:
            raise ValueError("maze_time windows require selected_df['maze_time']")
        raw = pd.to_numeric(selected["maze_time"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw)
        if np.count_nonzero(finite) < 2:
            raise ValueError("maze_time has fewer than two finite rows")
        t = raw - float(raw[finite][0])
        span = float(np.nanmax(t[finite]) - np.nanmin(t[finite]))
        if np.any(np.diff(t[finite]) < 0):
            return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, span, "sec", "maze_time_not_monotonic")
        max_maze_span_sec = 24.0 * 3600.0
        if span > max_maze_span_sec:
            return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, span, "sec", "maze_time_span_gt_24h")
        maze_time = raw
    elif axis == "absolute_time":
        if "frame" not in selected.columns:
            raise ValueError("absolute_time windows require selected_df['frame']")
        raw_frame = pd.to_numeric(selected["frame"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw_frame)
        if np.count_nonzero(finite) < 2:
            raise ValueError("frame has fewer than two finite rows")
        t = (raw_frame - float(raw_frame[finite][0])) * float(DT)
        span = float(np.nanmax(t[finite]) - np.nanmin(t[finite]))
        if np.any(np.diff(t[finite]) < 0):
            return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, span, "sec", "frame_time_not_monotonic")
        max_abs_span_sec = 24.0 * 3600.0
        if span > max_abs_span_sec:
            return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, span, "sec", "absolute_time_span_gt_24h")
        maze_time = pd.to_numeric(selected.get("maze_time", pd.Series(np.arange(n) * DT)), errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown time window axis: {axis}")
    win_sec = float(window_min) * 60.0
    if span <= win_sec:
        return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), [], 1, span, "sec")
    starts = np.arange(0.0, span - win_sec + 1e-9, float(stride_sec), dtype=float)
    bounds: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for idx, start in enumerate(starts):
        stop = float(start + win_sec)
        s = int(np.searchsorted(t, start, side="left"))
        e = int(np.searchsorted(t, stop, side="left"))
        if e <= s:
            continue
        e = min(e, n)
        bounds.append((s, e))
        mt_seg = maze_time[s:e]
        mt_finite = mt_seg[np.isfinite(mt_seg)]
        selected_duration_sec = float(e - s) * float(DT)
        rows.append(
            {
                "window_idx": int(len(bounds) - 1),
                "window_axis_idx": int(idx),
                "window_start_sec": float(start),
                "window_stop_sec": float(stop),
                "window_center_sec": float((start + stop) / 2.0),
                "window_start_tile": np.nan,
                "window_stop_tile": np.nan,
                "window_center_tile": np.nan,
                "window_start_idx": int(s),
                "window_stop_idx": int(e),
                "window_rows": int(e - s),
                "window_selected_duration_sec": selected_duration_sec,
                "window_elapsed_sec": float(stop - start),
                "window_coverage_fraction": float(selected_duration_sec / max(float(stop - start), 1e-12)),
                "window_coverage_min_selected_duration_sec": float(min_selected_duration_sec),
                "window_coverage_gate_rule": "selected_duration_sec >= fixed_minimum_sec",
                "window_coverage_gate_pass": bool(selected_duration_sec >= float(min_selected_duration_sec)),
                "window_stop_maze_sec": float(mt_finite[-1]) if mt_finite.size else np.nan,
                "window_elapsed_maze_sec": float(mt_finite[-1] - mt_finite[0]) if mt_finite.size > 1 else np.nan,
                "x_value": float(stop),
                "x_unit": "sec",
            }
        )
    n_nonoverlap = max(1, int(math.floor(float(span) / max(win_sec, 1e-12))))
    return WindowDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.asarray(bounds, dtype=int), rows, n_nonoverlap, span, "sec")


def _tile_window_def(
    selected: pd.DataFrame,
    *,
    session_id: str,
    behav_wd: str,
    window_tiles: int,
    stride_tiles: int,
    tile_axis_source: str,
) -> tuple[WindowDef, pd.DataFrame, dict[str, Any]]:
    tile_df, audit = _add_tile_axis(selected, session_id=session_id, behav_wd=behav_wd, tile_axis_source=tile_axis_source)
    cum = tile_df["cum_tile_change_count"].to_numpy(dtype=float)
    if cum.size and not np.isfinite(cum).all():
        cum = pd.Series(cum).ffill().bfill().to_numpy(dtype=float)
        tile_df["cum_tile_change_count"] = cum
    n = int(len(tile_df))
    max_count = float(np.nanmax(cum)) if cum.size else 0.0
    if n < 2 or max_count <= float(window_tiles):
        return (
            WindowDef("tile_change", float(window_tiles), f"{int(window_tiles)}tile", float(stride_tiles), f"{int(stride_tiles)}tile", np.zeros((0, 2), dtype=int), [], 1, max_count, "tile"),
            tile_df,
            audit,
        )
    starts = np.arange(0.0, max_count - float(window_tiles) + 1e-9, float(stride_tiles), dtype=float)
    maze_time = pd.to_numeric(tile_df.get("maze_time", pd.Series(np.arange(n) * DT)), errors="coerce").to_numpy(dtype=float)
    frames = pd.to_numeric(tile_df.get("frame", pd.Series(np.arange(n))), errors="coerce").to_numpy(dtype=float)
    bounds: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for idx, start in enumerate(starts):
        stop = float(start + float(window_tiles))
        s = int(np.searchsorted(cum, start, side="left"))
        e = int(np.searchsorted(cum, stop, side="right"))
        e = min(e, n)
        if e <= s:
            continue
        bounds.append((s, e))
        mt_seg = maze_time[s:e]
        mt_finite = mt_seg[np.isfinite(mt_seg)]
        frame_seg = frames[s:e]
        frame_finite = frame_seg[np.isfinite(frame_seg)]
        raw_frame_span = float(frame_finite[-1] - frame_finite[0] + 1.0) if frame_finite.size > 1 else np.nan
        selected_frame_fraction = float((e - s) / raw_frame_span) if np.isfinite(raw_frame_span) and raw_frame_span > 0 else np.nan
        cum_start = float(cum[s]) if np.isfinite(cum[s]) else np.nan
        cum_stop = float(cum[e - 1]) if e > s and np.isfinite(cum[e - 1]) else np.nan
        rows.append(
            {
                "window_idx": int(len(bounds) - 1),
                "window_axis_idx": int(idx),
                "window_start_sec": np.nan,
                "window_stop_sec": np.nan,
                "window_center_sec": np.nan,
                "window_start_tile": float(start),
                "window_stop_tile": float(stop),
                "window_center_tile": float((start + stop) / 2.0),
                "window_start_idx": int(s),
                "window_stop_idx": int(e),
                "window_rows": int(e - s),
                "window_start_frame": float(frame_finite[0]) if frame_finite.size else np.nan,
                "window_stop_frame": float(frame_finite[-1]) if frame_finite.size else np.nan,
                "window_raw_frame_span": raw_frame_span,
                "window_selected_frame_fraction": selected_frame_fraction,
                "window_selected_cum_tile_start": cum_start,
                "window_selected_cum_tile_stop": cum_stop,
                "window_selected_cum_tile_delta": float(cum_stop - cum_start) if np.isfinite(cum_start) and np.isfinite(cum_stop) else np.nan,
                "window_selected_duration_sec": float(e - s) * float(DT),
                "window_elapsed_sec": np.nan,
                "window_coverage_fraction": 1.0,
                "window_coverage_gate_pass": True,
                "window_stop_maze_sec": float(mt_finite[-1]) if mt_finite.size else np.nan,
                "window_elapsed_maze_sec": float(mt_finite[-1] - mt_finite[0]) if mt_finite.size > 1 else np.nan,
                "x_value": float(stop) / 1000.0,
                "x_unit": "k_tile_changes",
            }
        )
    n_nonoverlap = max(1, int(math.floor(max_count / max(float(window_tiles), 1e-12))))
    return (
        WindowDef("tile_change", float(window_tiles), f"{int(window_tiles)}tile", float(stride_tiles), f"{int(stride_tiles)}tile", np.asarray(bounds, dtype=int), rows, n_nonoverlap, max_count, "tile"),
        tile_df,
        audit,
    )


def _update_map(sum_map: np.ndarray, occ_counts: np.ndarray, labels: np.ndarray, signal: np.ndarray, s: int, e: int, sign: float) -> None:
    if e <= s:
        return
    y = labels[s:e]
    valid = (y >= 0) & (y < occ_counts.size)
    if not np.any(valid):
        return
    yy = y[valid].astype(int, copy=False)
    xx = np.nan_to_num(signal[s:e][valid], copy=False)
    np.add.at(sum_map, yy, float(sign) * xx)
    occ_counts += float(sign) * np.bincount(yy, minlength=occ_counts.size).astype(float)


def _iter_rate_maps(signal: np.ndarray, labels: np.ndarray, bounds: np.ndarray, *, n_bins: int, dt_eff: float) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    signal = np.asarray(signal, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    sum_map = np.zeros((int(n_bins), signal.shape[1]), dtype=float)
    occ_counts = np.zeros(int(n_bins), dtype=float)
    prev_s: int | None = None
    prev_e: int | None = None
    for s_raw, e_raw in np.asarray(bounds, dtype=int):
        s = int(max(0, min(signal.shape[0], s_raw)))
        e = int(max(0, min(signal.shape[0], e_raw)))
        if prev_s is None or s < prev_s or e < prev_e:
            sum_map.fill(0.0)
            occ_counts.fill(0.0)
            _update_map(sum_map, occ_counts, labels, signal, s, e, +1.0)
        else:
            _update_map(sum_map, occ_counts, labels, signal, prev_s, s, -1.0)
            _update_map(sum_map, occ_counts, labels, signal, prev_e, e, +1.0)
        prev_s, prev_e = s, e
        occ_sec = np.maximum(occ_counts, 0.0) * float(dt_eff)
        rates = np.divide(sum_map, np.maximum(occ_sec[:, None], 1e-12), out=np.zeros_like(sum_map), where=occ_sec[:, None] > 0)
        yield rates.copy(), occ_sec.copy()


def _full_rate_map(signal: np.ndarray, labels: np.ndarray, *, n_bins: int, dt_eff: float) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray([[0, int(signal.shape[0])]], dtype=int)
    return next(_iter_rate_maps(signal, labels, bounds, n_bins=n_bins, dt_eff=dt_eff))


def _entropy(occ: np.ndarray, support: np.ndarray) -> float:
    x = np.asarray(occ, dtype=float)[np.asarray(support, dtype=bool)]
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    p = x / float(np.sum(x))
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(float(max(len(occ), 2))))


def _policy_threshold(policy: str, map_type: str, *, n_nonoverlap: int, n_bins: int, dt: float = DT) -> tuple[float, float, int, int | None]:
    if policy == "scaled_nonoverlap":
        base = 5.0
        n = max(1, int(n_nonoverlap))
        return float(base), float(base / n), n, None
    if policy == "figure4_fixed":
        if map_type == "place":
            base = 13.0 * float(dt)
            return float(base), float(base), 1, 20
        base = 125.0 * float(dt)
        return float(base), float(base), 1, min(7, int(n_bins))
    if policy == "legacy_place_frame":
        if map_type != "place":
            raise ValueError("legacy_place_frame applies only to place maps")
        base = 13.0 * float(dt)
        return float(base), float(base), 1, 20
    raise ValueError(f"Unknown support policy: {policy}")


def _thresholded_map(rates: np.ndarray, occ: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(occ, dtype=float) >= float(threshold)
    out = np.asarray(rates, dtype=float).copy()
    out[~support, :] = np.nan
    return out, support


def _apply_policy_allowed_bins(map_arr: np.ndarray, support: np.ndarray, *, policy: str, map_type: str) -> tuple[np.ndarray, np.ndarray, str]:
    out = np.asarray(map_arr, dtype=float).copy()
    keep = np.ones(np.asarray(support).shape, dtype=bool)
    allowed = "all"
    if policy == "figure4_fixed" and map_type == "level" and keep.size >= 8:
        keep[7:] = False
        allowed = "0,1,2,3,4,5,6"
    support_out = np.asarray(support, dtype=bool) & keep
    out[~keep, :] = np.nan
    return out, support_out, allowed


def _per_cell_corr(ref_map: np.ndarray, target_map: np.ndarray, common_support: np.ndarray, *, min_bins: int) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(ref_map, dtype=float)[common_support]
    B = np.asarray(target_map, dtype=float)[common_support]
    if A.size == 0 or B.size == 0 or A.shape != B.shape:
        n = ref_map.shape[1] if ref_map.ndim == 2 else 0
        return np.full(n, np.nan, dtype=float), np.zeros(n, dtype=int)
    good = np.isfinite(A) & np.isfinite(B)
    valid_counts = np.sum(good, axis=0).astype(int)
    n_neurons = A.shape[1]
    out = np.full(n_neurons, np.nan, dtype=float)
    enough = valid_counts >= int(min_bins)
    if not np.any(enough):
        return out, valid_counts
    A0 = np.where(good, A, np.nan)
    B0 = np.where(good, B, np.nan)
    mean_a = np.nanmean(A0[:, enough], axis=0, keepdims=True)
    mean_b = np.nanmean(B0[:, enough], axis=0, keepdims=True)
    da = np.where(good[:, enough], A[:, enough] - mean_a, 0.0)
    db = np.where(good[:, enough], B[:, enough] - mean_b, 0.0)
    denom = np.sqrt(np.sum(da * da, axis=0) * np.sum(db * db, axis=0))
    num = np.sum(da * db, axis=0)
    vals = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 1e-12)
    out[np.flatnonzero(enough)] = vals
    return out, valid_counts


def _pv_corr_by_bin(ref_map: np.ndarray, target_map: np.ndarray, common_support: np.ndarray, *, min_neurons: int = 3) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(ref_map, dtype=float)
    B = np.asarray(target_map, dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        n_bins = A.shape[0] if A.ndim >= 1 else 0
        return np.full(n_bins, np.nan, dtype=float), np.zeros(n_bins, dtype=int)
    common = np.asarray(common_support, dtype=bool)
    good = np.isfinite(A) & np.isfinite(B)
    valid_counts = np.sum(good, axis=1).astype(int)
    out = np.full(A.shape[0], np.nan, dtype=float)
    eligible = common & (valid_counts >= int(min_neurons))
    if not np.any(eligible):
        return out, valid_counts
    A0 = np.where(good, A, np.nan)
    B0 = np.where(good, B, np.nan)
    mean_a = np.nanmean(A0[eligible], axis=1, keepdims=True)
    mean_b = np.nanmean(B0[eligible], axis=1, keepdims=True)
    da = np.where(good[eligible], A[eligible] - mean_a, 0.0)
    db = np.where(good[eligible], B[eligible] - mean_b, 0.0)
    denom = np.sqrt(np.sum(da * da, axis=1) * np.sum(db * db, axis=1))
    num = np.sum(da * db, axis=1)
    vals = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 1e-12)
    out[np.flatnonzero(eligible)] = vals
    return out, valid_counts


def _rate_vector(signal: np.ndarray, s: int, e: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    s_i = int(max(0, min(arr.shape[0], s)))
    e_i = int(max(0, min(arr.shape[0], e)))
    if e_i <= s_i:
        return np.full(arr.shape[1], np.nan, dtype=float)
    block = arr[s_i:e_i]
    valid_rows = np.isfinite(block).all(axis=1)
    block = block[valid_rows]
    if block.size == 0:
        return np.full(arr.shape[1], np.nan, dtype=float)
    duration_min = max(float(block.shape[0]) * float(DT) / 60.0, 1e-12)
    return np.sum(block, axis=0, dtype=float) / duration_min


def _rate_corr(ref_rate: np.ndarray, target_rate: np.ndarray) -> tuple[float, int, int, int]:
    ref = np.asarray(ref_rate, dtype=float)
    target = np.asarray(target_rate, dtype=float)
    finite = np.isfinite(ref) & np.isfinite(target)
    active_ref = ref > 0
    active_target = target > 0
    keep = finite & (active_ref | active_target)
    n_used = int(np.count_nonzero(keep))
    n_nonzero_both = int(np.count_nonzero(finite & active_ref & active_target))
    n_silent_both = int(np.count_nonzero(finite & ~active_ref & ~active_target))
    if n_used < 3:
        return np.nan, n_used, n_nonzero_both, n_silent_both
    return float(stats.pearsonr(target[keep], ref[keep]).statistic), n_used, n_nonzero_both, n_silent_both


def _safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else np.nan


def _safe_sem(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(stats.sem(arr)) if arr.size > 1 else np.nan


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    keep = np.isfinite(xv) & np.isfinite(yv)
    if np.count_nonzero(keep) < 3:
        return np.nan
    X = np.column_stack([np.ones(np.count_nonzero(keep)), xv[keep]])
    beta, _, _, _ = np.linalg.lstsq(X, yv[keep], rcond=None)
    return float(beta[1])


def _select_representative_indices(n: int) -> set[int]:
    if n <= 0:
        return set()
    return {0, int((n - 1) // 2), int(n - 1)}


def _reference_rows(
    pair_row: pd.Series,
    *,
    ref_signal: np.ndarray,
    ref_labels_by_map: dict[str, tuple[np.ndarray, int, str]],
    metrics: list[str],
    support_policies: list[str],
    level_col: str,
    n_levels: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    ref_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for map_type, (labels, n_bins, label_source) in ref_labels_by_map.items():
        dt_eff = float(DT)
        rates, occ = _full_rate_map(ref_signal, labels, n_bins=n_bins, dt_eff=dt_eff)
        for policy in support_policies:
            if policy == "legacy_place_frame" and map_type != "place":
                continue
            base_thr, eff_thr, n_nonoverlap, min_common = _policy_threshold(policy, map_type, n_nonoverlap=1, n_bins=n_bins)
            ref_map, ref_support = _thresholded_map(rates, occ, eff_thr)
            ref_map, ref_support, allowed_bins = _apply_policy_allowed_bins(ref_map, ref_support, policy=policy, map_type=map_type)
            key = (map_type, policy)
            ref_cache[key] = {
                "map": ref_map,
                "support": ref_support,
                "occ": occ,
                "base_threshold": base_thr,
                "effective_threshold": eff_thr,
                "n_nonoverlap": n_nonoverlap,
                "min_common_bins": min_common,
                "n_bins": n_bins,
                "allowed_bins": allowed_bins,
            }
            metric_names = [m for m in metrics if METRIC_TO_MAP_TYPE.get(m) == map_type]
            for metric_name in metric_names:
                rows.append(
                    {
                        "animal": str(pair_row.get("animal", "")),
                        "pair_id": str(pair_row.get("pair_id", "")),
                        "pair_family": str(pair_row.get("pair_family", "")),
                        "experience_group": _experience_group(pair_row),
                        "target_session": str(pair_row.get("session_a", "")),
                        "reference_session": str(pair_row.get("session_b", "")),
                        "metric": metric_name,
                        "map_type": map_type,
                        "support_policy": policy,
                        "reference_base_min_time_per_bin_sec": base_thr,
                        "reference_effective_min_time_per_bin_sec": eff_thr,
                        "reference_min_samples_per_bin_equivalent": int(math.ceil(float(eff_thr) / float(DT))),
                        "reference_n_nonoverlap_windows": n_nonoverlap,
                        "reference_min_common_bins": min_common,
                        "reference_valid_bins": int(np.count_nonzero(ref_support)),
                        "reference_valid_fraction": float(np.count_nonzero(ref_support) / max(n_bins, 1)),
                        "reference_valid_bin_indices": _mask_indices(ref_support),
                        "reference_occ_entropy": _entropy(occ, ref_support),
                        "n_reference_rows": int(ref_signal.shape[0]),
                        "level_col": level_col if map_type == "level" else "",
                        "n_levels": int(n_levels) if map_type == "level" else np.nan,
                        "allowed_bin_indices": allowed_bins,
                        "label_source": label_source,
                    }
                )
    return ref_cache, rows


def _pair_window_convergence(
    pair_row: pd.Series,
    *,
    behav_wd: str,
    neural_wd: str,
    signal_source: str,
    level_definition: str,
    level_col: str,
    n_levels: int,
    metrics: list[str],
    map_types: list[str],
    support_policies: list[str],
    window_types: list[str],
    time_windows_min: list[float],
    time_stride_sec: float,
    min_window_selected_duration_sec: float,
    tile_windows: list[int],
    tile_stride: int,
    tile_axis_source: str,
    primary_support_policy: str,
    per_cell_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    session_a = str(pair_row["session_a"])
    session_b = str(pair_row["session_b"])
    target_neural_wd = str(pair_row.get("target_neural_wd", "") or neural_wd)
    reference_neural_wd = str(pair_row.get("reference_neural_wd", "") or neural_wd)
    target_bundle = _load_bundle(session_a, behav_wd=behav_wd, neural_wd=target_neural_wd, level_definition=level_definition, level_col=level_col)
    ref_bundle = _load_bundle(session_b, behav_wd=behav_wd, neural_wd=reference_neural_wd, level_definition=level_definition, level_col=level_col)
    target_ids, ref_ids, matching_source = _resolve_pair_ids(
        pair_row, int(target_bundle["neural_n_neurons"]), int(ref_bundle["neural_n_neurons"])
    )
    if target_ids.size == 0 or ref_ids.size == 0:
        raise RuntimeError(f"No matched neurons for {session_a}->{session_b} ({matching_source})")
    target_signal = _signal_matrix_from_path(target_bundle, signal_source, target_ids)
    ref_signal = _signal_matrix_from_path(ref_bundle, signal_source, ref_ids)

    target_labels: dict[str, tuple[np.ndarray, int, str]] = {}
    ref_labels: dict[str, tuple[np.ndarray, int, str]] = {}
    map_types_needed = sorted({str(METRIC_TO_MAP_TYPE[m]) for m in metrics if m in METRIC_TO_MAP_TYPE})
    for map_type in map_types_needed:
        target_labels[map_type] = _labels_for_map(target_bundle, map_type, level_col=level_col, n_levels=n_levels)
        ref_labels[map_type] = _labels_for_map(ref_bundle, map_type, level_col=level_col, n_levels=n_levels)
    _drop_signal_payload(target_bundle)
    _drop_signal_payload(ref_bundle)
    gc.collect()
    ref_cache, ref_rows = _reference_rows(
        pair_row,
        ref_signal=ref_signal,
        ref_labels_by_map=ref_labels,
        metrics=metrics,
        support_policies=support_policies,
        level_col=level_col,
        n_levels=n_levels,
    )
    ref_rate = _rate_vector(ref_signal, 0, int(ref_signal.shape[0])) if "ensemble_rate_correlation" in metrics else None
    if ref_rate is not None:
        ref_rows.append(
            {
                "animal": str(pair_row.get("animal", "")),
                "pair_id": str(pair_row.get("pair_id", "")),
                "pair_family": str(pair_row.get("pair_family", "")),
                "experience_group": _experience_group(pair_row),
                "target_session": session_a,
                "reference_session": session_b,
                "metric": "ensemble_rate_correlation",
                "map_type": "ensemble_rate",
                "support_policy": RATE_SUPPORT_POLICY,
                "reference_valid_bins": np.nan,
                "reference_valid_fraction": np.nan,
                "n_reference_rows": int(ref_signal.shape[0]),
                "n_reference_finite_rates": int(np.count_nonzero(np.isfinite(ref_rate))),
                "reference_rate_duration_sec": float(ref_signal.shape[0]) * float(DT),
                "label_source": "matched_neuron_rate_vector",
            }
        )

    window_defs: list[tuple[WindowDef, pd.DataFrame | None, dict[str, Any]]] = []
    selected = target_bundle["selected_df"]
    if "selected_time" in window_types:
        for win_min in time_windows_min:
            window_defs.append((_time_window_def(selected, axis="selected_time", window_min=win_min, stride_sec=time_stride_sec, min_selected_duration_sec=min_window_selected_duration_sec), None, {}))
    if "maze_time" in window_types:
        for win_min in time_windows_min:
            window_defs.append((_time_window_def(selected, axis="maze_time", window_min=win_min, stride_sec=time_stride_sec, min_selected_duration_sec=min_window_selected_duration_sec), None, {}))
    if "absolute_time" in window_types:
        for win_min in time_windows_min:
            window_defs.append((_time_window_def(selected, axis="absolute_time", window_min=win_min, stride_sec=time_stride_sec, min_selected_duration_sec=min_window_selected_duration_sec), None, {}))
    if "tile_change" in window_types:
        for win_tiles in tile_windows:
            window_defs.append(
                _tile_window_def(
                    selected,
                    session_id=session_a,
                    behav_wd=behav_wd,
                    window_tiles=int(win_tiles),
                    stride_tiles=int(tile_stride),
                    tile_axis_source=tile_axis_source,
                )
            )

    curve_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    base_pair = {
        "animal": str(pair_row.get("animal", "")),
        "pair_id": str(pair_row.get("pair_id", "")),
        "pair_family": str(pair_row.get("pair_family", "")),
        "memory_stage": str(pair_row.get("memory_stage", "")),
        "experience_group": _experience_group(pair_row),
        "target_session": session_a,
        "reference_session": session_b,
        "neural_source_policy": str(pair_row.get("neural_source_policy", "single_root")),
        "target_neural_source": str(pair_row.get("target_neural_source", "single_root")),
        "reference_neural_source": str(pair_row.get("reference_neural_source", "single_root")),
        "target_neural_wd": target_neural_wd,
        "reference_neural_wd": reference_neural_wd,
        "matching_source": matching_source,
        "n_matched_neurons": int(target_ids.size),
        "signal_source": str(signal_source).upper(),
        "level_definition": str(level_definition),
    }

    for win_def, tile_df, tile_audit in window_defs:
        coverage_values = np.asarray([r.get("window_coverage_fraction", np.nan) for r in win_def.rows], dtype=float)
        status_rows.append(
            {
                **base_pair,
                "window_type": win_def.window_type,
                "window_size": float(win_def.window_size),
                "window_size_label": win_def.window_size_label,
                "stride": float(win_def.stride),
                "stride_label": win_def.stride_label,
                "n_windows": int(win_def.bounds.shape[0]),
                "n_nonoverlap_windows": int(win_def.n_nonoverlap),
                "span_value": float(win_def.span_value),
                "span_unit": win_def.span_unit,
                "skip_reason": win_def.skip_reason,
                "window_coverage_fraction_min": float(np.nanmin(coverage_values)) if coverage_values.size and np.isfinite(coverage_values).any() else np.nan,
                "window_coverage_fraction_median": float(np.nanmedian(coverage_values)) if coverage_values.size and np.isfinite(coverage_values).any() else np.nan,
                "n_windows_passing_coverage_gate": int(sum(bool(r.get("window_coverage_gate_pass", True)) for r in win_def.rows)),
                **tile_audit,
            }
        )
        if win_def.bounds.size == 0:
            continue
        selected_for_labels = tile_df if tile_df is not None else selected
        rep_indices = _select_representative_indices(int(win_def.bounds.shape[0]))
        for map_type, (labels0, n_bins, label_source) in target_labels.items():
            metric_names = [m for m in metrics if METRIC_TO_MAP_TYPE.get(m) == map_type]
            if not metric_names:
                continue
            if tile_df is not None and map_type == "place":
                labels = _labels_for_map({**target_bundle, "selected_df": selected_for_labels}, map_type, level_col=level_col, n_levels=n_levels)[0]
            else:
                labels = labels0
            dt_eff = float(DT)
            for w_idx, (rates, occ) in enumerate(_iter_rate_maps(target_signal, labels, win_def.bounds, n_bins=n_bins, dt_eff=dt_eff)):
                wrow = win_def.rows[w_idx]
                for policy in support_policies:
                    if policy == "legacy_place_frame" and map_type != "place":
                        continue
                    ref_key = (map_type, policy)
                    if ref_key not in ref_cache:
                        continue
                    base_thr, eff_thr, n_nonoverlap, min_common = _policy_threshold(policy, map_type, n_nonoverlap=win_def.n_nonoverlap, n_bins=n_bins)
                    target_map, target_support = _thresholded_map(rates, occ, eff_thr)
                    target_map, target_support, allowed_bins = _apply_policy_allowed_bins(target_map, target_support, policy=policy, map_type=map_type)
                    ref_info = ref_cache[ref_key]
                    ref_support = np.asarray(ref_info["support"], dtype=bool)
                    common = target_support & ref_support
                    min_bins = 3 if map_type == "place" else 4
                    gate_min_common = int(min_common) if min_common is not None else int(min_bins)
                    coverage_gate_pass = bool(wrow.get("window_coverage_gate_pass", True))
                    support_gate_pass = bool(int(np.count_nonzero(common)) >= gate_min_common)
                    gate_pass = bool(support_gate_pass and coverage_gate_pass)
                    for metric_name in metric_names:
                        if metric_name in {"place_map_correlation", "level_map_correlation"}:
                            if gate_pass:
                                values, valid_counts = _per_cell_corr(ref_info["map"], target_map, common, min_bins=min_bins)
                            else:
                                values = np.full(target_signal.shape[1], np.nan, dtype=float)
                                valid_counts = np.zeros(target_signal.shape[1], dtype=int)
                            unit_type = "neuron"
                            finite = np.isfinite(values)
                            n_finite_neurons = int(np.count_nonzero(finite))
                            n_finite_bins = np.nan
                            finite_neuron_fraction = float(n_finite_neurons / max(values.size, 1))
                            finite_bin_fraction = np.nan
                            min_neurons_for_pv = np.nan
                        elif metric_name == "pv_correlation":
                            if gate_pass:
                                values, valid_counts = _pv_corr_by_bin(ref_info["map"], target_map, common, min_neurons=3)
                            else:
                                values = np.full(int(n_bins), np.nan, dtype=float)
                                valid_counts = np.zeros(int(n_bins), dtype=int)
                            unit_type = "spatial_bin"
                            finite = np.isfinite(values)
                            n_finite_neurons = np.nan
                            n_finite_bins = int(np.count_nonzero(finite))
                            finite_neuron_fraction = np.nan
                            finite_bin_fraction = float(n_finite_bins / max(n_bins, 1))
                            min_neurons_for_pv = 3
                        else:
                            continue
                        z_values = _safe_atanh(values)
                        z_mean = _safe_mean(z_values)
                        rec = {
                            **base_pair,
                            "metric": metric_name,
                            "map_type": map_type,
                            "unit_type": unit_type,
                            "support_policy": policy,
                            "window_type": win_def.window_type,
                            "window_size": float(win_def.window_size),
                            "window_size_label": win_def.window_size_label,
                            "stride": float(win_def.stride),
                            "stride_label": win_def.stride_label,
                            **wrow,
                            "x_value": float(wrow["x_value"]),
                            "x_unit": str(wrow["x_unit"]),
                            "target_base_min_time_per_bin_sec": float(base_thr),
                            "target_effective_min_time_per_bin_sec": float(eff_thr),
                            "target_min_samples_per_bin_equivalent": int(math.ceil(float(eff_thr) / float(DT))),
                            "target_n_nonoverlap_windows": int(n_nonoverlap),
                            "reference_effective_min_time_per_bin_sec": float(ref_info["effective_threshold"]),
                            "reference_min_samples_per_bin_equivalent": int(math.ceil(float(ref_info["effective_threshold"]) / float(DT))),
                            "reference_n_nonoverlap_windows": int(ref_info["n_nonoverlap"]),
                            "min_bins": int(min_bins),
                            "min_common_bins": int(gate_min_common),
                            "min_neurons_for_pv": min_neurons_for_pv,
                            "target_valid_bins": int(np.count_nonzero(target_support)),
                            "reference_valid_bins": int(np.count_nonzero(ref_support)),
                            "common_bins": int(np.count_nonzero(common)),
                            "n_common_bins_passing": int(np.count_nonzero(common)),
                            "passes_bin_gate": gate_pass if map_type == "place" else np.nan,
                            "passes_level_gate": gate_pass if map_type == "level" else np.nan,
                            "support_gate_pass": support_gate_pass,
                            "coverage_gate_pass": coverage_gate_pass,
                            "target_valid_bin_indices": _mask_indices(target_support),
                            "reference_valid_bin_indices": _mask_indices(ref_support),
                            "common_bin_indices": _mask_indices(common),
                            "target_valid_fraction": float(np.count_nonzero(target_support) / max(n_bins, 1)),
                            "reference_valid_fraction": float(np.count_nonzero(ref_support) / max(n_bins, 1)),
                            "common_bin_fraction": float(np.count_nonzero(common) / max(n_bins, 1)),
                            "target_occ_entropy": _entropy(occ, target_support),
                            "reference_occ_entropy": _entropy(ref_info["occ"], ref_support),
                            "common_occ_entropy": _entropy(occ, common),
                            "gate_pass": gate_pass,
                            "n_finite_neurons": n_finite_neurons,
                            "n_finite_bins": n_finite_bins,
                            "n_finite_cell_corr": int(np.count_nonzero(finite)),
                            "finite_neuron_fraction": finite_neuron_fraction,
                            "finite_bin_fraction": finite_bin_fraction,
                            "cell_corr_mean": _safe_mean(values),
                            "cell_corr_z_mean": z_mean,
                            "cell_corr_zmean_backtransformed": float(np.tanh(z_mean)) if np.isfinite(z_mean) else np.nan,
                            "sim_mean": _safe_mean(values),
                            "sim_z_mean": z_mean,
                            "sim_median": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
                            "sim_std": float(np.nanstd(values)) if np.isfinite(values).any() else np.nan,
                            "comparison_kind": "fixed_reference_dayi_to_dayi_plus1",
                            "reference_window": "full_day_i_plus_1",
                            "target_window": "sliding_day_i",
                            "target_reference_same_session": False,
                            "target_windows_overlap_by_design": bool(float(win_def.stride) < float(win_def.window_size if win_def.window_type == "tile_change" else win_def.window_size * 60.0)),
                            "occupancy_timebase": "row_count_x_DT",
                            "allowed_bin_indices": allowed_bins,
                            "label_source": label_source,
                        }
                        mask_cols = {"target_valid_bin_indices", "reference_valid_bin_indices", "common_bin_indices"}
                        curve_rows.append({k: v for k, v in rec.items() if k not in mask_cols})
                        support_rows.append({k: rec[k] for k in rec if k not in {"sim_mean", "sim_median", "sim_std"}})
                        if metric_name in {"place_map_correlation", "level_map_correlation"} and (
                            per_cell_mode == "all" or (
                                per_cell_mode == "representative"
                                and policy == primary_support_policy
                                and int(wrow["window_idx"]) in rep_indices
                            )
                        ):
                            for local_idx, (tid, rid, val, vc) in enumerate(zip(target_ids, ref_ids, values, valid_counts)):
                                cell_rows.append(
                                    {
                                        **base_pair,
                                        "metric": metric_name,
                                        "map_type": map_type,
                                        "support_policy": policy,
                                        "window_type": win_def.window_type,
                                        "window_size": float(win_def.window_size),
                                        "window_size_label": win_def.window_size_label,
                                        "window_idx": int(wrow["window_idx"]),
                                        "window_role": "first" if int(wrow["window_idx"]) == 0 else ("last" if int(wrow["window_idx"]) == int(win_def.bounds.shape[0]) - 1 else "middle"),
                                        "x_value": float(wrow["x_value"]),
                                        "x_unit": str(wrow["x_unit"]),
                                        "target_neuron_id": int(tid),
                                        "reference_neuron_id": int(rid),
                                        "matched_neuron_local_idx": int(local_idx),
                                        "correlation": float(val) if np.isfinite(val) else np.nan,
                                        "fisher_z": float(np.arctanh(np.clip(val, -0.999999, 0.999999))) if np.isfinite(val) else np.nan,
                                        "valid_bins_for_cell": int(vc),
                                    }
                                )
        if "ensemble_rate_correlation" in metrics and ref_rate is not None:
            for w_idx, (s_raw, e_raw) in enumerate(np.asarray(win_def.bounds, dtype=int)):
                wrow = win_def.rows[w_idx]
                target_rate = _rate_vector(target_signal, int(s_raw), int(e_raw))
                rate_corr, n_used, n_nonzero_both, n_silent_both = _rate_corr(ref_rate, target_rate)
                z_mean = float(np.arctanh(np.clip(rate_corr, -0.999999, 0.999999))) if np.isfinite(rate_corr) else np.nan
                coverage_gate_pass = bool(wrow.get("window_coverage_gate_pass", True))
                finite_rate = np.isfinite(target_rate) & np.isfinite(ref_rate)
                rec = {
                    **base_pair,
                    "metric": "ensemble_rate_correlation",
                    "map_type": "ensemble_rate",
                    "unit_type": "matched_neuron_rate_vector",
                    "support_policy": RATE_SUPPORT_POLICY,
                    "window_type": win_def.window_type,
                    "window_size": float(win_def.window_size),
                    "window_size_label": win_def.window_size_label,
                    "stride": float(win_def.stride),
                    "stride_label": win_def.stride_label,
                    **wrow,
                    "x_value": float(wrow["x_value"]),
                    "x_unit": str(wrow["x_unit"]),
                    "target_base_min_time_per_bin_sec": np.nan,
                    "target_effective_min_time_per_bin_sec": np.nan,
                    "target_min_samples_per_bin_equivalent": np.nan,
                    "target_n_nonoverlap_windows": int(win_def.n_nonoverlap),
                    "reference_effective_min_time_per_bin_sec": np.nan,
                    "reference_min_samples_per_bin_equivalent": np.nan,
                    "reference_n_nonoverlap_windows": 1,
                    "min_bins": np.nan,
                    "min_common_bins": np.nan,
                    "min_neurons_for_pv": np.nan,
                    "target_valid_bins": np.nan,
                    "reference_valid_bins": np.nan,
                    "common_bins": np.nan,
                    "n_common_bins_passing": np.nan,
                    "passes_bin_gate": np.nan,
                    "passes_level_gate": np.nan,
                    "support_gate_pass": True,
                    "coverage_gate_pass": coverage_gate_pass,
                    "target_valid_fraction": np.nan,
                    "reference_valid_fraction": np.nan,
                    "common_bin_fraction": np.nan,
                    "target_occ_entropy": np.nan,
                    "reference_occ_entropy": np.nan,
                    "common_occ_entropy": np.nan,
                    "gate_pass": bool(coverage_gate_pass and np.isfinite(rate_corr)),
                    "n_finite_neurons": int(np.count_nonzero(finite_rate)),
                    "n_finite_bins": np.nan,
                    "n_finite_cell_corr": int(n_used),
                    "finite_neuron_fraction": float(n_used / max(target_signal.shape[1], 1)),
                    "finite_bin_fraction": np.nan,
                    "rate_n_used": int(n_used),
                    "rate_n_nonzero_both": int(n_nonzero_both),
                    "rate_n_silent_both": int(n_silent_both),
                    "cell_corr_mean": rate_corr,
                    "cell_corr_z_mean": z_mean,
                    "cell_corr_zmean_backtransformed": float(np.tanh(z_mean)) if np.isfinite(z_mean) else np.nan,
                    "sim_mean": rate_corr,
                    "sim_z_mean": z_mean,
                    "sim_median": rate_corr,
                    "sim_std": np.nan,
                    "comparison_kind": "fixed_reference_dayi_to_dayi_plus1",
                    "reference_window": "full_day_i_plus_1",
                    "target_window": "sliding_day_i",
                    "target_reference_same_session": False,
                    "target_windows_overlap_by_design": bool(float(win_def.stride) < float(win_def.window_size if win_def.window_type == "tile_change" else win_def.window_size * 60.0)),
                    "occupancy_timebase": "row_count_x_DT",
                    "allowed_bin_indices": "not_applicable",
                    "label_source": "matched_neuron_rate_vector",
                }
                curve_rows.append(rec)
                support_rows.append({k: rec[k] for k in rec if k not in {"sim_mean", "sim_median", "sim_std"}})
    return curve_rows, support_rows, cell_rows, ref_rows, status_rows


def _build_target_pairs(args: argparse.Namespace) -> pd.DataFrame:
    pair_manifest = str(getattr(args, "pair_manifest", "") or "").strip()
    if pair_manifest:
        manifest_path = Path(pair_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest = pd.read_csv(manifest_path)
    else:
        # Default cohort construction is intentionally lazy so `--help` and
        # explicit-manifest runs do not import the broad legacy MM stack.
        from ca_analysis.si_memory_controls import build_cross_session_pair_manifest

        manifest = build_cross_session_pair_manifest()
    use = manifest[manifest["pair_family"].isin(set(args.pair_families))].copy()
    extra_pairs_csv = getattr(args, "extra_pairs_csv", None)
    if extra_pairs_csv:
        extra_path = Path(extra_pairs_csv)
        if not extra_path.is_absolute():
            extra_path = REPO_ROOT / extra_path
        extra = pd.read_csv(extra_path)
        missing_cols = [col for col in manifest.columns if col not in extra.columns]
        for col in missing_cols:
            extra[col] = ""
        extra = extra[list(manifest.columns)]
        use = pd.concat([use, extra], ignore_index=True)
        use = use.drop_duplicates(subset=["pair_id"], keep="last")
    use = use[use["pair_available"].map(_as_bool)].copy()
    if not bool(args.include_hard_exclusions):
        excluded = set(EXPANDED_HARD_EXCLUSIONS)
        use = use[~use["session_a"].astype(str).isin(excluded) & ~use["session_b"].astype(str).isin(excluded)].copy()
    if args.only:
        only = set(args.only)
        use = use[use["pair_id"].isin(only) | use["session_a"].isin(only) | use["session_b"].isin(only)].copy()
    if args.max_pairs:
        use = use.head(int(args.max_pairs)).copy()
    use["experience_group"] = use.apply(_experience_group, axis=1)
    use = _resolve_neural_source_rows(use, args)
    return use.reset_index(drop=True)


def _compute_pair_job(row_dict: dict[str, Any], job_args: dict[str, Any]) -> dict[str, Any]:
    row = pd.Series(row_dict)
    t0 = time.perf_counter()
    try:
        curves, support, cells, refs, status = _pair_window_convergence(
            row,
            behav_wd=str(job_args["behav_wd"]),
            neural_wd=str(job_args["neural_wd"]),
            signal_source=str(job_args["signal_source"]),
            level_definition=str(job_args["level_definition"]),
            level_col=str(job_args["level_col"]),
            n_levels=int(job_args["n_levels"]),
            metrics=list(job_args["metrics"]),
            map_types=list(job_args["map_types"]),
            support_policies=list(job_args["support_policies"]),
            window_types=list(job_args["window_types"]),
            time_windows_min=[float(v) for v in job_args["time_windows_min"]],
            time_stride_sec=float(job_args["time_stride_sec"]),
            min_window_selected_duration_sec=float(job_args["min_window_selected_duration_sec"]),
            tile_windows=[int(v) for v in job_args["tile_windows"]],
            tile_stride=int(job_args["tile_stride"]),
            tile_axis_source=str(job_args["tile_axis_source"]),
            primary_support_policy=str(job_args["primary_support_policy"]),
            per_cell_mode=str(job_args["per_cell_mode"]),
        )
        return {
            "ok": True,
            "row": row_dict,
            "elapsed_sec": float(time.perf_counter() - t0),
            "curves": curves,
            "support": support,
            "cells": cells,
            "refs": refs,
            "status": status,
        }
    except Exception as exc:
        return {
            "ok": False,
            "row": row_dict,
            "elapsed_sec": float(time.perf_counter() - t0),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _summarize_pair_metrics(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if curves.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    curves = curves.copy()
    if "metric" not in curves.columns:
        curves["metric"] = curves["map_type"].astype(str).map(lambda x: f"{x}_map_correlation")
    rows: list[dict[str, Any]] = []
    group_cols = [
        "animal",
        "pair_id",
        "pair_family",
        "memory_stage",
        "experience_group",
        "target_session",
        "reference_session",
        "metric",
        "map_type",
        "support_policy",
        "window_type",
        "window_size",
        "window_size_label",
    ]
    for key, sub in curves.groupby(group_cols, dropna=False):
        d = sub.sort_values("x_value").copy()
        y = d["sim_mean"].to_numpy(dtype=float)
        yz = d["sim_z_mean"].to_numpy(dtype=float) if "sim_z_mean" in d.columns else np.full(len(d), np.nan, dtype=float)
        x = d["x_value"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        rec = dict(zip(group_cols, key))
        rec["n_windows"] = int(len(d))
        rec["n_finite_windows"] = int(np.count_nonzero(finite))
        rec["mean_common_bins"] = _safe_mean(d["common_bins"].to_numpy(dtype=float))
        rec["mean_finite_neuron_fraction"] = _safe_mean(d["finite_neuron_fraction"].to_numpy(dtype=float))
        rec["mean_finite_bin_fraction"] = _safe_mean(d["finite_bin_fraction"].to_numpy(dtype=float))
        if not np.any(finite):
            rec.update(first_value=np.nan, early_mean=np.nan, late_mean=np.nan, delta_late_minus_first=np.nan, slope_per_unit=np.nan, slope_per_hour_or_ktile=np.nan, first_z_value=np.nan, early_z_mean=np.nan, late_z_mean=np.nan, delta_late_z_minus_first=np.nan)
            rows.append(rec)
            continue
        x_f = x[finite]
        y_f = y[finite]
        yz_f = yz[finite]
        order = np.argsort(x_f)
        x_f = x_f[order]
        y_f = y_f[order]
        yz_f = yz_f[order]
        first_x = float(x_f[0])
        last_x = float(x_f[-1])
        span = max(last_x - first_x, 1e-12)
        seconds_axis = str(rec["window_type"]) in {"selected_time", "maze_time", "absolute_time"}
        early_span = 600.0 if seconds_axis else 1.0
        early = y_f[(x_f - first_x) <= early_span + 1e-9]
        late = y_f[(last_x - x_f) <= early_span + 1e-9]
        early_z = yz_f[(x_f - first_x) <= early_span + 1e-9]
        late_z = yz_f[(last_x - x_f) <= early_span + 1e-9]
        slope_unit = _slope(x_f - first_x, y_f)
        scale = 3600.0 if seconds_axis else 1.0
        rec.update(
            first_x=first_x,
            last_x=last_x,
            first_value=float(y_f[0]),
            early_mean=_safe_mean(early),
            late_mean=_safe_mean(late),
            delta_late_minus_first=float(_safe_mean(late) - y_f[0]) if np.isfinite(_safe_mean(late)) else np.nan,
            slope_per_unit=slope_unit,
            slope_per_hour_or_ktile=float(slope_unit * scale) if np.isfinite(slope_unit) else np.nan,
            first_z_value=float(yz_f[0]) if np.isfinite(yz_f[0]) else np.nan,
            early_z_mean=_safe_mean(early_z),
            late_z_mean=_safe_mean(late_z),
            delta_late_z_minus_first=float(_safe_mean(late_z) - yz_f[0]) if np.isfinite(_safe_mean(late_z)) and np.isfinite(yz_f[0]) else np.nan,
        )
        rows.append(rec)
    pair_metrics = pd.DataFrame(rows)

    animal_metrics = (
        pair_metrics.groupby(
            [
                "animal",
                "experience_group",
                "metric",
                "map_type",
                "support_policy",
                "window_type",
                "window_size",
                "window_size_label",
            ],
            dropna=False,
        )
        .agg(
            first_value=("first_value", "mean"),
            early_mean=("early_mean", "mean"),
            late_mean=("late_mean", "mean"),
            delta_late_minus_first=("delta_late_minus_first", "mean"),
            slope_per_hour_or_ktile=("slope_per_hour_or_ktile", "mean"),
            first_z_value=("first_z_value", "mean"),
            early_z_mean=("early_z_mean", "mean"),
            late_z_mean=("late_z_mean", "mean"),
            delta_late_z_minus_first=("delta_late_z_minus_first", "mean"),
            mean_common_bins=("mean_common_bins", "mean"),
            mean_finite_neuron_fraction=("mean_finite_neuron_fraction", "mean"),
            mean_finite_bin_fraction=("mean_finite_bin_fraction", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
    )

    summary_rows: list[dict[str, Any]] = []
    metric_cols = ["first_value", "early_mean", "late_mean", "delta_late_minus_first", "slope_per_hour_or_ktile", "first_z_value", "early_z_mean", "late_z_mean", "delta_late_z_minus_first", "mean_common_bins", "mean_finite_neuron_fraction", "mean_finite_bin_fraction"]
    for key, sub in animal_metrics.groupby(["experience_group", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label"], dropna=False):
        rec = dict(zip(["experience_group", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label"], key))
        rec["n_animals"] = int(sub["animal"].nunique())
        for col in metric_cols:
            vals = sub[col].to_numpy(dtype=float)
            rec[f"{col}_mean"] = _safe_mean(vals)
            rec[f"{col}_sem"] = _safe_sem(vals)
            rec[f"{col}_median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        summary_rows.append(rec)
    group_summary = pd.DataFrame(summary_rows)

    paired_rows: list[dict[str, Any]] = []
    ret = animal_metrics[animal_metrics["experience_group"].ne("naive")]
    naive = animal_metrics[animal_metrics["experience_group"].eq("naive")]
    merge_cols = ["animal", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label"]
    merged = ret.merge(naive, on=merge_cols, suffixes=("_return", "_naive"))
    for key, sub in merged.groupby(["experience_group_return", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label"], dropna=False):
        rec = dict(zip(["return_group", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label"], key))
        rec["n_animals"] = int(sub["animal"].nunique())
        for col in ["first_value", "early_mean", "late_mean", "delta_late_minus_first", "slope_per_hour_or_ktile", "first_z_value", "early_z_mean", "late_z_mean", "delta_late_z_minus_first"]:
            diff = sub[f"{col}_return"].to_numpy(dtype=float) - sub[f"{col}_naive"].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            rec[f"{col}_paired_diff_mean"] = _safe_mean(diff)
            rec[f"{col}_paired_diff_sem"] = _safe_sem(diff)
            if diff.size >= 3:
                rec[f"{col}_paired_t_p"] = float(stats.ttest_1samp(diff, 0.0, nan_policy="omit").pvalue)
            else:
                rec[f"{col}_paired_t_p"] = np.nan
            if diff.size >= 3 and np.any(diff != 0):
                try:
                    rec[f"{col}_wilcoxon_p"] = float(stats.wilcoxon(diff).pvalue)
                except Exception:
                    rec[f"{col}_wilcoxon_p"] = np.nan
            else:
                rec[f"{col}_wilcoxon_p"] = np.nan
        paired_rows.append(rec)
    paired_summary = pd.DataFrame(paired_rows)
    return pair_metrics, animal_metrics, group_summary, paired_summary


def _aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    curves = curves.copy()
    if "metric" not in curves.columns:
        curves["metric"] = curves["map_type"].astype(str).map(lambda x: f"{x}_map_correlation")
    animal = (
        curves.groupby(
            [
                "animal",
                "experience_group",
                "metric",
                "map_type",
                "support_policy",
                "window_type",
                "window_size",
                "window_size_label",
                "x_value",
                "x_unit",
            ],
            dropna=False,
        )
        .agg(sim_mean=("sim_mean", "mean"), common_bins=("common_bins", "mean"), finite_neuron_fraction=("finite_neuron_fraction", "mean"))
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for key, sub in animal.groupby(["experience_group", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label", "x_value", "x_unit"], dropna=False):
        vals = sub["sim_mean"].to_numpy(dtype=float)
        rec = dict(zip(["experience_group", "metric", "map_type", "support_policy", "window_type", "window_size", "window_size_label", "x_value", "x_unit"], key))
        rec.update(
            mean=_safe_mean(vals),
            sem=_safe_sem(vals),
            n_animals=int(sub["animal"].nunique()),
            mean_common_bins=_safe_mean(sub["common_bins"].to_numpy(dtype=float)),
            mean_finite_neuron_fraction=_safe_mean(sub["finite_neuron_fraction"].to_numpy(dtype=float)),
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_primary_curves(curve_summary: pd.DataFrame, out_dir: Path, *, support_policy: str) -> None:
    if curve_summary.empty:
        return
    configs = [
        ("selected_time", 20.0, "20 min selected-time windows", "Window stop in selected time (min)"),
        ("tile_change", 2000.0, "2000 tile-change windows", "Window stop (k tile changes)"),
    ]
    metric_order = [m for m in METRIC_CHOICES if m in set(curve_summary.get("metric", pd.Series(dtype=str)).astype(str))]
    if not metric_order:
        metric_order = [f"{m}_map_correlation" for m in ["place", "level"] if m in set(curve_summary["map_type"].astype(str))]
    fig, axes = plt.subplots(len(metric_order), 2, figsize=(10.0, max(3.0, 2.7 * len(metric_order))), sharey=False)
    axes = np.atleast_2d(axes)
    for r, metric_name in enumerate(metric_order):
        for c, (wtype, wsize, title, xlabel) in enumerate(configs):
            ax = axes[r, c]
            policy = RATE_SUPPORT_POLICY if metric_name == "ensemble_rate_correlation" else support_policy
            sub = curve_summary[
                curve_summary["support_policy"].eq(policy)
                & curve_summary["metric"].eq(metric_name)
                & curve_summary["window_type"].eq(wtype)
                & np.isclose(curve_summary["window_size"].astype(float), float(wsize))
            ]
            if sub.empty:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            for group in GROUP_ORDER:
                g = sub[sub["experience_group"].eq(group)].sort_values("x_value")
                if g.empty:
                    continue
                x = g["x_value"].to_numpy(dtype=float)
                if wtype != "tile_change":
                    x = x / 60.0
                y = g["mean"].to_numpy(dtype=float)
                sem = g["sem"].to_numpy(dtype=float)
                color = GROUP_COLORS.get(group, "0.3")
                ax.plot(x, y, lw=1.9, color=color, label=group)
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
            ax.set_title(f"{METRIC_LABELS.get(metric_name, metric_name)}: {title}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("corr. to full Day i+1", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.axhline(0, color="0.8", lw=0.7, zorder=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "primary_scaled_convergence_curves.png", dpi=220)
    fig.savefig(out_dir / "primary_scaled_convergence_curves.svg")
    plt.close(fig)


def _plot_support_policy_comparison(group_summary: pd.DataFrame, out_dir: Path) -> None:
    if group_summary.empty:
        return
    group_summary = group_summary.copy()
    if "metric" not in group_summary.columns:
        group_summary["metric"] = group_summary["map_type"].astype(str).map(lambda x: f"{x}_map_correlation")
    sub = group_summary[
        group_summary["window_type"].eq("selected_time")
        & np.isclose(group_summary["window_size"].astype(float), 20.0)
        & group_summary["experience_group"].isin(["naive", "3mo", "6mo", "9mo", "12mo"])
    ].copy()
    if sub.empty:
        return
    metric_order = [
        m
        for m in ("place_map_correlation", "level_map_correlation", "pv_correlation")
        if m in set(sub["metric"].astype(str))
    ]
    if not metric_order:
        return
    fig, axes = plt.subplots(1, len(metric_order), figsize=(4.2 * len(metric_order), 3.2), sharey=False, squeeze=False)
    for ax, metric_name in zip(axes.ravel(), metric_order):
        d = sub[sub["metric"].eq(metric_name)]
        policies = [p for p in DEFAULT_SUPPORT_POLICIES if p in set(d["support_policy"])]
        x0 = np.arange(len(GROUP_ORDER), dtype=float)
        width = 0.8 / max(len(policies), 1)
        for i, policy in enumerate(policies):
            vals = []
            errs = []
            for group in GROUP_ORDER:
                row = d[(d["experience_group"].eq(group)) & (d["support_policy"].eq(policy))]
                vals.append(float(row["first_value_mean"].iloc[0]) if len(row) else np.nan)
                errs.append(float(row["first_value_sem"].iloc[0]) if len(row) else np.nan)
            err_arr = np.asarray(errs, dtype=float)
            err_arr = np.where(np.isfinite(err_arr), err_arr, 0.0)
            ax.bar(x0 + (i - (len(policies) - 1) / 2) * width, vals, width=width, yerr=err_arr, label=policy, alpha=0.85)
        ax.set_xticks(x0)
        ax.set_xticklabels(GROUP_ORDER, rotation=30, ha="right")
        ax.set_title(METRIC_LABELS.get(metric_name, metric_name))
        ax.set_ylabel("first-window corr.")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "support_policy_first_window_comparison.png", dpi=220)
    fig.savefig(out_dir / "support_policy_first_window_comparison.svg")
    plt.close(fig)


def _format_num(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    return f"{val:.{digits}f}" if np.isfinite(val) else "NA"


def _write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    curves: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    group_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# Fixed-Reference Window Convergence Report",
        "",
        "## Question",
        "",
        "This analysis tests whether overlapping Day i representations gradually approach the full Day i+1 reference representation. It is not the same estimand as the existing pre/post within-session pivot curves.",
        "Place/level map metrics follow the maintained neuron-wise tuning-map Pearson convention across common bins/levels. PV correlation is bin-wise Pearson across matched neurons, then averaged across eligible spatial bins. Ensemble-rate correlation is Pearson correlation of matched-neuron full-window rate vectors after dropping neurons silent in both windows.",
        "",
        "## Primary Settings",
        "",
        f"- Metrics: `{', '.join(args.metrics)}`.",
        f"- Signal source: `{args.signal_source}`.",
        f"- Level definition: `{args.level_definition}`.",
        f"- Pair families: `{', '.join(args.pair_families)}`.",
        f"- Window axes: `{', '.join(args.window_types)}`.",
        f"- Time windows: `{', '.join(str(v) for v in args.time_windows_min)} min`, stride `{args.time_stride_sec} s`.",
        f"- Tile windows: `{', '.join(str(v) for v in args.tile_windows)}` tile changes, stride `{args.tile_stride} tile changes`.",
        f"- Tile axis source: `{args.tile_axis_source}`.",
        f"- Primary support policy: `{args.primary_support_policy}`.",
        "- Scaled support rule: `effective_min_time_per_bin = base_min_time_per_bin / floor(session_span / window_length)` with floor count clipped to at least 1.",
        "- Figure 4 fixed sensitivity rule: place maps use 13 samples/bin plus 20 common bins; level maps use 125 samples/level on levels 0-6 and require all 7 common levels.",
        "- Maze-time and tile windows define row bounds only; map occupancy is counted as selected rows times `DT` so discontinuities in `selected_df.maze_time` do not inflate occupancy.",
        f"- Maze-time and absolute-time windows require at least `{args.min_window_selected_duration_sec:g}` s (5 min at the maintained setting) of selected rows per window; lower-support windows remain auditable but do not contribute finite correlations.",
        "- Absolute-time windows use raw-frame elapsed time, `(selected_df.frame - first_selected_frame) * DT`, so they retain recording-time gaps and are best treated as a coverage/sensitivity axis.",
        "- Reference maps are full Day i+1 maps and therefore use one non-overlapping reference window.",
        f"- Ensemble-rate rows use support policy `{RATE_SUPPORT_POLICY}` because no occupancy-bin support mask is applied.",
        "",
        "## Cohort And Completeness",
        "",
        f"- Curve rows: `{len(curves)}`.",
        f"- Unique session pairs: `{curves['pair_id'].nunique() if not curves.empty else 0}`.",
        f"- Unique animals: `{curves['animal'].nunique() if not curves.empty else 0}`.",
        f"- Failures: `{len(failures)}`.",
    ]
    if not failures.empty:
        lines += ["", "Failure preview:", "", "```csv"]
        lines += failures.head(20).to_csv(index=False).strip().splitlines()
        lines += ["```"]

    def _report_support_mask(df: pd.DataFrame) -> pd.Series:
        metric = df["metric"].astype(str) if "metric" in df.columns else pd.Series("", index=df.index)
        return df["support_policy"].eq(args.primary_support_policy) | (
            metric.eq("ensemble_rate_correlation") & df["support_policy"].eq(RATE_SUPPORT_POLICY)
        )

    primary = group_summary[
        _report_support_mask(group_summary)
        & group_summary["window_type"].eq("selected_time")
        & np.isclose(group_summary["window_size"].astype(float), 20.0)
    ].copy() if not group_summary.empty else pd.DataFrame()
    if not primary.empty:
        show_cols = [
            "metric",
            "map_type",
            "experience_group",
            "n_animals",
            "first_value_mean",
            "first_value_sem",
            "early_mean_mean",
            "late_mean_mean",
            "delta_late_minus_first_mean",
            "first_z_value_mean",
            "delta_late_z_minus_first_mean",
            "mean_common_bins_mean",
            "mean_finite_neuron_fraction_mean",
            "mean_finite_bin_fraction_mean",
        ]
        lines += [
            "",
            "## Primary Selected-Time 20 Min Results",
            "",
            "The table below uses the primary map support policy for map-derived metrics and `not_applicable` for ensemble-rate rows.",
            "",
            "```csv",
        ]
        lines += primary[[c for c in show_cols if c in primary.columns]].sort_values(["metric", "experience_group"]).to_csv(index=False).strip().splitlines()
        lines += ["```"]

    tile_primary = group_summary[
        _report_support_mask(group_summary)
        & group_summary["window_type"].eq("tile_change")
        & np.isclose(group_summary["window_size"].astype(float), 2000.0)
    ].copy() if not group_summary.empty else pd.DataFrame()
    if not tile_primary.empty:
        lines += [
            "",
            "## Primary 2000-Tile Results",
            "",
            f"Tile axis source: `{args.tile_axis_source}`.",
            "",
            "```csv",
        ]
        lines += tile_primary[[c for c in show_cols if c in tile_primary.columns]].sort_values(["metric", "experience_group"]).to_csv(index=False).strip().splitlines()
        lines += ["```"]

    if not paired_summary.empty:
        pshow = paired_summary[
            _report_support_mask(paired_summary)
            & paired_summary["window_type"].isin(["selected_time", "tile_change"])
            & (
                (paired_summary["window_type"].eq("selected_time") & np.isclose(paired_summary["window_size"].astype(float), 20.0))
                | (paired_summary["window_type"].eq("tile_change") & np.isclose(paired_summary["window_size"].astype(float), 2000.0))
            )
        ].copy()
        if not pshow.empty:
            cols = [
                "return_group",
                "metric",
                "map_type",
                "window_type",
                "window_size_label",
                "n_animals",
                "first_value_paired_diff_mean",
                "first_value_paired_t_p",
                "early_mean_paired_diff_mean",
                "early_mean_paired_t_p",
                "slope_per_hour_or_ktile_paired_diff_mean",
                "slope_per_hour_or_ktile_paired_t_p",
            ]
            lines += [
                "",
                "## Paired Return-Minus-Naive Diagnostics",
                "",
                "These are animal-paired diagnostics for animals with both a naive pair and a return pair. They should not be treated as pointwise curve tests over overlapping windows.",
                "",
                "```csv",
            ]
            lines += pshow[[c for c in cols if c in pshow.columns]].sort_values(["metric", "window_type", "return_group"]).to_csv(index=False).strip().splitlines()
            lines += ["```"]

    lines += [
        "",
        "## Interpretation",
        "",
        "- The primary curves should be read as convergence trajectories toward a fixed future full-session map, not as independent observations at each overlapping window.",
        "- Adjacent curve points are overlapping-window summaries and should not be treated as independent statistical observations.",
        "- The scaled support policy directly addresses the unfairness of applying a whole-session occupancy threshold to a small target window.",
        "- If a return group starts above naive under scaled support and the same direction is visible under the tile-window sensitivity, that supports an initial representational advantage toward the Day i+1 map.",
        "- If a result is present only under the fixed Figure 4 sensitivity policy or only in one window axis, treat it as support-sensitive rather than a stable biological claim.",
        "",
        "## Saved Outputs",
        "",
        "- `window_convergence_curves.csv`",
        "- `window_support_audit.csv`",
        "- `reference_map_support_audit.csv`",
        "- Inclusion masks are embedded as `target_valid_bin_indices`, `reference_valid_bin_indices`, and `common_bin_indices` in `window_support_audit.csv`.",
        "- `per_cell_correlations.csv`",
        "- `pair_metric_summary.csv`",
        "- `animal_metric_summary.csv`",
        "- `group_metric_summary.csv`",
        "- `group_curve_summary.csv`",
        "- `paired_return_minus_naive_summary.csv`",
        "- `window_status.csv`",
        "- `failures.csv`",
        "- `run_manifest.json`",
        "- `primary_scaled_convergence_curves.png` / `.svg`",
        "- `support_policy_first_window_comparison.png` / `.svg`",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--behav-wd", default=get_behav_wd())
    p.add_argument("--neural-wd", default=get_neural_wd())
    p.add_argument("--sorted-neural-wd", default="", help="Optional sorted neural root for sorted-first source policy.")
    p.add_argument("--neural-source-policy", choices=["single_root", "sorted_first"], default="single_root")
    p.add_argument("--pair-families", nargs="+", default=list(DEFAULT_PAIR_FAMILIES))
    p.add_argument("--map-types", nargs="+", default=["place", "level"], choices=["place", "level"])
    p.add_argument("--metrics", nargs="+", default=None, choices=list(METRIC_CHOICES), help="Metrics to export. If omitted, derive place/level map metrics from --map-types for backward compatibility.")
    p.add_argument("--window-types", nargs="+", default=list(DEFAULT_WINDOW_TYPES), choices=list(DEFAULT_WINDOW_TYPES))
    p.add_argument("--time-windows-min", nargs="+", type=float, default=list(DEFAULT_TIME_WINDOWS_MIN))
    p.add_argument("--time-stride-sec", type=float, default=60.0)
    p.add_argument(
        "--min-window-selected-duration-sec",
        type=float,
        default=DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC,
        help="Fixed minimum selected-data duration required in each time window (default: 300 s / 5 min).",
    )
    p.add_argument("--tile-windows", nargs="+", type=int, default=list(DEFAULT_TILE_WINDOWS))
    p.add_argument("--tile-stride", type=int, default=100)
    p.add_argument("--tile-axis-source", default=DEFAULT_TILE_AXIS_SOURCE, choices=list(TILE_AXIS_SOURCES))
    p.add_argument("--support-policies", nargs="+", default=list(DEFAULT_SUPPORT_POLICIES), choices=list(DEFAULT_SUPPORT_POLICIES))
    p.add_argument("--primary-support-policy", default="scaled_nonoverlap", choices=list(DEFAULT_SUPPORT_POLICIES))
    p.add_argument("--signal-source", choices=["C", "S"], default="C")
    p.add_argument("--level-definition", default=LEVEL_DEFINITION_GEOMETRIC8)
    p.add_argument("--include-hard-exclusions", action="store_true")
    p.add_argument("--per-cell-mode", choices=["none", "representative", "all"], default="representative")
    p.add_argument("--only", nargs="*", default=None, help="Optional pair_id or session IDs to include.")
    p.add_argument("--pair-manifest", default="", help="Optional exact pair manifest; when omitted, construct the maintained default manifest.")
    p.add_argument("--extra-pairs-csv", default="", help="Optional CSV with additional pair-manifest rows to append before filtering.")
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1, help="Number of session-pair workers. Use 1 for deterministic sequential logs.")
    p.add_argument("--max-tasks-per-child", type=int, default=1, help="Recycle worker processes after this many pairs to release large neural arrays. Use 0 to disable.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> int:
    bootstrap_environment(repo_root=REPO_ROOT)
    args.level_definition = normalize_level_definition(args.level_definition)
    if args.metrics is None:
        metric_map = {"place": "place_map_correlation", "level": "level_map_correlation"}
        args.metrics = [metric_map[m] for m in args.map_types]
    else:
        args.metrics = list(dict.fromkeys(str(m) for m in args.metrics))
        args.map_types = sorted({METRIC_TO_MAP_TYPE[m] for m in args.metrics if m in METRIC_TO_MAP_TYPE})
    if not args.metrics:
        raise ValueError("At least one metric is required.")
    if args.primary_support_policy not in set(args.support_policies):
        raise ValueError("--primary-support-policy must also be included in --support-policies.")
    level_col = "level" if args.level_definition == LEVEL_DEFINITION_GEOMETRIC8 else "tree_depth"
    n_levels = 8 if args.level_definition == LEVEL_DEFINITION_GEOMETRIC8 else 7
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Pass --overwrite to replace files in place.")
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = _build_target_pairs(args)
    failure_columns = [*list(pair_df.columns), "failure_stage", "error", "elapsed_sec"]
    pair_df.to_csv(out_dir / "pair_manifest_used.csv", index=False)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "analysis": "fixed_reference_window_convergence",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "out_dir": str(out_dir),
                "behav_wd": str(args.behav_wd),
                "neural_wd": str(args.neural_wd),
                "sorted_neural_wd": str(args.sorted_neural_wd),
                "neural_source_policy": str(args.neural_source_policy),
                "pair_neural_source_counts": pair_df["neural_source_policy"].value_counts(dropna=False).to_dict() if "neural_source_policy" in pair_df else {},
                "pair_manifest": str(args.pair_manifest),
                "pair_manifest_mode": "explicit" if str(args.pair_manifest).strip() else "maintained_default",
                "pair_families": list(args.pair_families),
                "metrics": list(args.metrics),
                "map_types": list(args.map_types),
                "window_types": list(args.window_types),
                "time_windows_min": [float(v) for v in args.time_windows_min],
                "time_stride_sec": float(args.time_stride_sec),
                "min_window_selected_duration_sec": float(args.min_window_selected_duration_sec),
                "min_window_selected_frames_equivalent": int(math.ceil(float(args.min_window_selected_duration_sec) / float(DT))),
                "window_coverage_gate_rule": "selected_duration_sec >= fixed_minimum_sec",
                "tile_windows": [int(v) for v in args.tile_windows],
                "tile_stride": int(args.tile_stride),
                "tile_axis_source": str(args.tile_axis_source),
                "support_policies": list(args.support_policies),
                "primary_support_policy": str(args.primary_support_policy),
                "metric_definitions": {
                    "place_map_correlation": "neuron-wise spatial tuning-map Pearson across common place bins; window value is mean across finite neurons",
                    "level_map_correlation": "neuron-wise level tuning-map Pearson across common levels; window value is mean across finite neurons",
                    "pv_correlation": "bin-wise spatial population-vector Pearson across matched neurons; window value is mean across finite bins",
                    "ensemble_rate_correlation": "Pearson correlation of matched-neuron rate vectors, dropping neurons silent in both windows",
                },
                "fisher_z_sensitivity_columns": ["sim_z_mean", "first_z_value", "early_z_mean", "late_z_mean", "delta_late_z_minus_first"],
                "signal_source": str(args.signal_source),
                "level_definition": str(args.level_definition),
                "level_col": str(level_col),
                "n_levels": int(n_levels),
                "include_hard_exclusions": bool(args.include_hard_exclusions),
                "per_cell_mode": str(args.per_cell_mode),
                "n_jobs": int(args.n_jobs),
                "max_tasks_per_child": int(args.max_tasks_per_child),
                "n_pairs": int(len(pair_df)),
                "support_policy_definitions": {
                    "scaled_nonoverlap": "base 5s/bin divided by floor(session span / window length); reference full-session count is 1",
                    "figure4_fixed": "maintained Figure 4 sample-count gates: place 13 samples/bin plus 20 common bins; level 125 samples/level on levels 0-6 with all 7 common levels required",
                    "legacy_place_frame": "place-only 13 frames/bin plus 20 common bins",
                    RATE_SUPPORT_POLICY: "ensemble-rate vector metric; no spatial/level occupancy support mask",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_curves: list[dict[str, Any]] = []
    all_support: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_refs: list[dict[str, Any]] = []
    all_status: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    job_args = {
        "behav_wd": str(args.behav_wd),
        "neural_wd": str(args.neural_wd),
        "signal_source": str(args.signal_source),
        "level_definition": str(args.level_definition),
        "level_col": str(level_col),
        "n_levels": int(n_levels),
        "metrics": list(args.metrics),
        "map_types": list(args.map_types),
        "support_policies": list(args.support_policies),
        "window_types": list(args.window_types),
        "time_windows_min": [float(v) for v in args.time_windows_min],
        "time_stride_sec": float(args.time_stride_sec),
        "min_window_selected_duration_sec": float(args.min_window_selected_duration_sec),
        "tile_windows": [int(v) for v in args.tile_windows],
        "tile_stride": int(args.tile_stride),
        "tile_axis_source": str(args.tile_axis_source),
        "primary_support_policy": str(args.primary_support_policy),
        "per_cell_mode": str(args.per_cell_mode),
    }

    def consume_result(done_count: int, result: dict[str, Any]) -> None:
        row = result["row"]
        pair_label = f"{row.get('session_a', '')}->{row.get('session_b', '')}"
        if result.get("ok"):
            print(f"[{done_count}/{len(pair_df)}] done {pair_label} ({result['elapsed_sec']:.1f}s)", flush=True)
            all_curves.extend(result["curves"])
            all_support.extend(result["support"])
            all_cells.extend(result["cells"])
            all_refs.extend(result["refs"])
            all_status.extend(result["status"])
        else:
            print(f"[{done_count}/{len(pair_df)}] failed {pair_label}: {result.get('error')}", flush=True)
            failures.append(
                {
                    **row,
                    "failure_stage": "pair_compute",
                    "error": result.get("error", ""),
                    "elapsed_sec": float(result.get("elapsed_sec", np.nan)),
                }
            )
        pd.DataFrame(all_status).to_csv(out_dir / "window_status.partial.csv", index=False)
        pd.DataFrame(failures, columns=failure_columns).to_csv(out_dir / "failures.partial.csv", index=False)

    row_dicts = [row.to_dict() for _, row in pair_df.iterrows()]
    n_jobs = max(1, int(args.n_jobs))
    if n_jobs == 1:
        for idx, row_dict in enumerate(row_dicts, start=1):
            pair_label = f"{row_dict.get('session_a', '')}->{row_dict.get('session_b', '')}"
            print(f"[{idx}/{len(pair_df)}] start {pair_label}", flush=True)
            consume_result(idx, _compute_pair_job(row_dict, job_args))
    else:
        max_tasks = int(args.max_tasks_per_child)
        done_count = 0
        if max_tasks <= 0:
            with futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
                submitted = [pool.submit(_compute_pair_job, row_dict, job_args) for row_dict in row_dicts]
                for fut in futures.as_completed(submitted):
                    done_count += 1
                    consume_result(done_count, fut.result())
        else:
            batch_size = max(1, n_jobs * max_tasks)
            for batch_start in range(0, len(row_dicts), batch_size):
                batch = row_dicts[batch_start : batch_start + batch_size]
                with futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    submitted = [pool.submit(_compute_pair_job, row_dict, job_args) for row_dict in batch]
                    for fut in futures.as_completed(submitted):
                        done_count += 1
                        consume_result(done_count, fut.result())

    curves_df = pd.DataFrame(all_curves)
    support_df = pd.DataFrame(all_support)
    cells_df = pd.DataFrame(all_cells)
    refs_df = pd.DataFrame(all_refs)
    failures_df = pd.DataFrame(failures, columns=failure_columns)
    status_df = pd.DataFrame(all_status)
    curves_df.to_csv(out_dir / "window_convergence_curves.csv", index=False)
    support_df.to_csv(out_dir / "window_support_audit.csv", index=False)
    cells_df.to_csv(out_dir / "per_cell_correlations.csv", index=False)
    refs_df.to_csv(out_dir / "reference_map_support_audit.csv", index=False)
    failures_df.to_csv(out_dir / "failures.csv", index=False)
    status_df.to_csv(out_dir / "window_status.csv", index=False)

    pair_metrics, animal_metrics, group_summary, paired_summary = _summarize_pair_metrics(curves_df)
    curve_summary = _aggregate_curves(curves_df)
    pair_metrics.to_csv(out_dir / "pair_metric_summary.csv", index=False)
    animal_metrics.to_csv(out_dir / "animal_metric_summary.csv", index=False)
    group_summary.to_csv(out_dir / "group_metric_summary.csv", index=False)
    paired_summary.to_csv(out_dir / "paired_return_minus_naive_summary.csv", index=False)
    curve_summary.to_csv(out_dir / "group_curve_summary.csv", index=False)

    _plot_primary_curves(curve_summary, out_dir, support_policy=str(args.primary_support_policy))
    _plot_support_policy_comparison(group_summary, out_dir)
    _write_report(
        out_dir,
        args=args,
        curves=curves_df,
        pair_metrics=pair_metrics,
        group_summary=group_summary,
        paired_summary=paired_summary,
        failures=failures_df,
    )
    print(f"Wrote {out_dir}", flush=True)
    print(f"Curve rows: {len(curves_df)}", flush=True)
    print(f"Failures: {len(failures_df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))
