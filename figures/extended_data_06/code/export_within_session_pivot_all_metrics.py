"""Export sorted-first within-session pivot curves for all four metrics.

This is the pivot/pre-post counterpart to
``export_fixed_reference_window_convergence.py``.  For each session, it builds
matched pre-window and post-window maps around each pivot and exports:

- place-map correlation, averaged across neurons
- level-map correlation, averaged across neurons
- place-bin PV correlation, averaged across bins
- ensemble-rate vector correlation

The script writes one session CSV/status file at a time, so a long full-cohort
run can be resumed without losing completed sessions.
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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ca_analysis.config import DT, get_behav_wd, get_neural_wd
from ca_analysis.spatial import LEVEL_DEFINITION_GEOMETRIC8, discretize_positions
from scripts.export_fixed_reference_window_convergence import (
    DEFAULT_TILE_AXIS_SOURCE,
    DEFAULT_TILE_WINDOWS,
    DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC,
    DEFAULT_TIME_WINDOWS_MIN,
    DEFAULT_WINDOW_TYPES,
    GROUP_COLORS,
    GROUP_ORDER,
    METRIC_CHOICES,
    METRIC_LABELS,
    METRIC_TO_MAP_TYPE,
    RATE_SUPPORT_POLICY,
    _add_tile_axis,
    _apply_policy_allowed_bins,
    _drop_signal_payload,
    _entropy,
    _iter_rate_maps,
    _labels_for_map as _fixed_labels_for_map,
    _mask_indices,
    _per_cell_corr,
    _policy_threshold as _fixed_policy_threshold,
    _pv_corr_by_bin,
    _rate_corr,
    _rate_vector,
    _safe_atanh,
    _safe_mean,
    _safe_sem,
    _session_neural_path,
    _signal_matrix_from_path,
    _thresholded_map,
    _load_bundle,
)
from scripts.export_within_session_level_map_reconstruction import _build_manifest


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "docs"
    / "figure4b_within_session_sorted_first_exhaustive_2026-06-17"
    / "pivot_prepost_day1_day2"
    / "pivot_all_metrics_sorted_non1616_full"
)
DEFAULT_PIVOT_SUPPORT_POLICIES = ("legacy_pivot", "figure4_fixed")


def _policy_threshold(policy: str, map_type: str, *, n_nonoverlap: int, n_bins: int, dt: float = DT) -> tuple[float, float, int, int | None]:
    if policy == "legacy_pivot":
        if map_type == "place":
            return 0.5, 0.5, 1, None
        if map_type == "level":
            return 5.0, 5.0, 1, None
        raise ValueError(f"legacy_pivot does not apply to map type: {map_type}")
    return _fixed_policy_threshold(policy, map_type, n_nonoverlap=n_nonoverlap, n_bins=n_bins, dt=dt)


def _labels_for_map(bundle: dict[str, Any], map_type: str, *, level_col: str, n_levels: int) -> tuple[np.ndarray, int, str]:
    if map_type == "place" and {"mass_x", "mass_y"}.issubset(bundle["selected_df"].columns):
        positions = bundle["selected_df"][["mass_x", "mass_y"]].to_numpy(dtype=float)
        labels, _x_edges, _y_edges = discretize_positions(positions, n_bins_x=15, n_bins_y=15)
        labels = np.asarray(labels, dtype=int).ravel()
        labels[(labels < 0) | (labels >= 225)] = -1
        return labels, 225, "discretize_positions_mass_x_mass_y"
    return _fixed_labels_for_map(bundle, map_type, level_col=level_col, n_levels=n_levels)


def _dataset_axes_for_frames(shape: tuple[int, int], frames: np.ndarray) -> tuple[int, int, int]:
    max_frame = int(np.nanmax(frames)) if np.asarray(frames).size else -1
    if int(shape[0]) > max_frame:
        return 0, int(shape[0]), int(shape[1])
    if int(shape[1]) > max_frame:
        return 1, int(shape[1]), int(shape[0])
    raise ValueError(f"Neural matrix shape {shape} shorter than max selected frame {max_frame}")


def _signal_matrix_from_path(bundle: dict[str, Any], signal_source: str, neuron_ids: np.ndarray) -> np.ndarray:
    source = str(signal_source).upper()
    dataset = "C" if source == "C" else "S" if source == "S" else ""
    if not dataset:
        raise ValueError("signal_source must be C or S")
    import h5py

    frames = bundle["selected_df"]["frame"].to_numpy(dtype=int)
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if frames.size == 0 or neuron_ids.size == 0:
        return np.zeros((frames.size, neuron_ids.size), dtype=float)
    out = np.empty((frames.size, neuron_ids.size), dtype=float)
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
            out[r0:r1] = np.asarray(block, dtype=float)
    return out


def _iter_rate_maps(signal: np.ndarray, labels: np.ndarray, bounds: np.ndarray, *, n_bins: int, dt_eff: float):
    signal = np.asarray(signal, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_bins = int(n_bins)
    for s_raw, e_raw in np.asarray(bounds, dtype=int):
        s = int(max(0, min(signal.shape[0], s_raw)))
        e = int(max(0, min(signal.shape[0], e_raw)))
        sum_map = np.zeros((n_bins, signal.shape[1]), dtype=float)
        occ_counts = np.zeros(n_bins, dtype=float)
        if e > s:
            y = labels[s:e]
            valid = (y >= 0) & (y < n_bins)
            if np.any(valid):
                yy = y[valid].astype(int, copy=False)
                block = np.nan_to_num(signal[s:e][valid], copy=False)
                np.add.at(sum_map, yy, block)
                occ_counts += np.bincount(yy, minlength=n_bins).astype(float)
        occ_sec = occ_counts * float(dt_eff)
        rates = np.divide(sum_map, np.maximum(occ_sec[:, None], 1e-12), out=np.zeros_like(sum_map), where=occ_sec[:, None] > 0)
        yield rates, occ_sec


def _thresholded_map(rates: np.ndarray, occ: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(occ, dtype=float) >= float(threshold)
    out = np.asarray(rates, dtype=float).copy() + 1e-12
    out[~support, :] = np.nan
    return out, support


def _per_cell_corr(ref_map: np.ndarray, target_map: np.ndarray, common_support: np.ndarray, *, min_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the maintained pivot_window_tuning_similarity Pearson behavior."""
    A = np.asarray(ref_map, dtype=float)[np.asarray(common_support, dtype=bool)]
    B = np.asarray(target_map, dtype=float)[np.asarray(common_support, dtype=bool)]
    if A.size == 0 or B.size == 0 or A.shape != B.shape:
        n = ref_map.shape[1] if np.asarray(ref_map).ndim == 2 else 0
        return np.full(n, np.nan, dtype=float), np.zeros(n, dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (A - np.nanmean(A, axis=0, keepdims=True)) / (np.nanstd(A, axis=0, keepdims=True) + 1e-12)
        B = (B - np.nanmean(B, axis=0, keepdims=True)) / (np.nanstd(B, axis=0, keepdims=True) + 1e-12)
    good = np.isfinite(A) & np.isfinite(B)
    valid_counts = np.sum(good, axis=0).astype(int)
    out = np.full(A.shape[1], np.nan, dtype=float)
    enough = valid_counts >= int(min_bins)
    if not np.any(enough):
        return out, valid_counts
    A0 = np.where(good[:, enough], A[:, enough], np.nan)
    B0 = np.where(good[:, enough], B[:, enough], np.nan)
    mean_a = np.nanmean(A0, axis=0, keepdims=True)
    mean_b = np.nanmean(B0, axis=0, keepdims=True)
    da = np.where(good[:, enough], A[:, enough] - mean_a, 0.0)
    db = np.where(good[:, enough], B[:, enough] - mean_b, 0.0)
    denom = np.sqrt(np.sum(da * da, axis=0) * np.sum(db * db, axis=0))
    num = np.sum(da * db, axis=0)
    vals = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 0.0)
    out[np.flatnonzero(enough)] = vals
    return out, valid_counts


@dataclass(frozen=True)
class PivotDef:
    window_type: str
    window_size: float
    window_size_label: str
    stride: float
    stride_label: str
    pre_bounds: np.ndarray
    post_bounds: np.ndarray
    rows: list[dict[str, Any]]
    n_nonoverlap_windows: int
    span_value: float
    span_unit: str
    skip_reason: str = ""
    selected_for_labels: pd.DataFrame | None = None
    audit: dict[str, Any] | None = None


def _parse_csv_float(text: str) -> list[float]:
    return [float(v.strip()) for v in str(text).split(",") if v.strip()]


def _parse_csv_int(text: str) -> list[int]:
    return [int(float(v.strip())) for v in str(text).split(",") if v.strip()]


def _parse_csv_str(text: str) -> list[str]:
    return [v.strip() for v in str(text).split(",") if v.strip()]


def _session_group(month_label: str, cohort: str) -> str:
    if str(cohort) == "naive":
        return "naive"
    return str(month_label)


def _target_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = _build_manifest()
    manifest = manifest[~manifest["expanded_hard_exclusion"]].copy()
    manifest = manifest[manifest["day"].isin(set(args.days))].copy()
    if args.only:
        manifest = manifest[manifest["session_id"].astype(str).isin(set(args.only))].copy()
    if args.max_sessions is not None:
        manifest = manifest.head(int(args.max_sessions)).copy()
    default_root = str(args.neural_wd)
    sorted_root = str(args.sorted_neural_wd or "").strip()
    if str(args.neural_source_policy) == "sorted_first" and sorted_root:
        sources = []
        roots = []
        for session_id in manifest["session_id"].astype(str):
            if _session_neural_path(sorted_root, session_id) is not None:
                sources.append("sorted")
                roots.append(sorted_root)
            else:
                sources.append("original_fallback")
                roots.append(default_root)
        manifest["neural_source_policy"] = "sorted_first"
        manifest["neural_source"] = sources
        manifest["neural_wd_used"] = roots
    else:
        manifest["neural_source_policy"] = "single_root"
        manifest["neural_source"] = "single_root"
        manifest["neural_wd_used"] = default_root
    manifest["experience_group"] = [
        _session_group(month, cohort)
        for month, cohort in zip(manifest["month_label"], manifest["cohort"])
    ]
    return manifest.reset_index(drop=True)


def _time_axis(selected: pd.DataFrame, axis: str) -> tuple[np.ndarray, np.ndarray, float, str]:
    n = int(len(selected))
    if axis == "selected_time":
        t = np.arange(n, dtype=float) * float(DT)
        maze_time = pd.to_numeric(
            selected.get("maze_time", pd.Series(np.arange(n) * DT)),
            errors="coerce",
        ).to_numpy(dtype=float)
        return t, maze_time, float(n) * float(DT), ""
    if axis == "maze_time":
        if "maze_time" not in selected.columns:
            raise ValueError("maze_time pivots require selected_df['maze_time']")
        raw = pd.to_numeric(selected["maze_time"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw)
        if np.count_nonzero(finite) < 2:
            raise ValueError("maze_time has fewer than two finite rows")
        t = raw - float(raw[finite][0])
        if np.any(np.diff(t[finite]) < 0):
            return t, raw, float(np.nanmax(t[finite]) - np.nanmin(t[finite])), "maze_time_not_monotonic"
        span = float(np.nanmax(t[finite]) - np.nanmin(t[finite]))
        if span > 24.0 * 3600.0:
            return t, raw, span, "maze_time_span_gt_24h"
        return t, raw, span, ""
    if axis == "absolute_time":
        if "frame" not in selected.columns:
            raise ValueError("absolute_time pivots require selected_df['frame']")
        raw_frame = pd.to_numeric(selected["frame"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(raw_frame)
        if np.count_nonzero(finite) < 2:
            raise ValueError("frame has fewer than two finite rows")
        t = (raw_frame - float(raw_frame[finite][0])) * float(DT)
        if np.any(np.diff(t[finite]) < 0):
            return t, raw_frame, float(np.nanmax(t[finite]) - np.nanmin(t[finite])), "frame_time_not_monotonic"
        span = float(np.nanmax(t[finite]) - np.nanmin(t[finite]))
        if span > 24.0 * 3600.0:
            return t, raw_frame, span, "absolute_time_span_gt_24h"
        maze_time = pd.to_numeric(
            selected.get("maze_time", pd.Series(np.arange(n) * DT)),
            errors="coerce",
        ).to_numpy(dtype=float)
        return t, maze_time, span, ""
    raise ValueError(f"Unknown time axis: {axis}")


def _time_pivot_def(
    selected: pd.DataFrame,
    *,
    axis: str,
    window_min: float,
    stride_sec: float,
    min_selected_duration_sec: float = DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC,
) -> PivotDef:
    n = int(len(selected))
    if n < 2:
        return PivotDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int), [], 1, 0.0, "sec")
    t, maze_time, span, skip_reason = _time_axis(selected, axis)
    win_sec = float(window_min) * 60.0
    if skip_reason or span <= 2.0 * win_sec:
        return PivotDef(axis, float(window_min), f"{window_min:g}min", float(stride_sec), f"{stride_sec:g}s", np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int), [], 1, span, "sec", skip_reason)
    pivots = np.arange(win_sec, span - win_sec + 1e-9, float(stride_sec), dtype=float)
    pre_bounds: list[tuple[int, int]] = []
    post_bounds: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for idx, pivot in enumerate(pivots):
        pre_s = int(np.searchsorted(t, pivot - win_sec, side="left"))
        pre_e = int(np.searchsorted(t, pivot, side="left"))
        post_s = int(np.searchsorted(t, pivot, side="left"))
        post_e = int(np.searchsorted(t, pivot + win_sec, side="left"))
        pre_e = min(pre_e, n)
        post_e = min(post_e, n)
        if pre_e <= pre_s or post_e <= post_s:
            continue
        pre_bounds.append((pre_s, pre_e))
        post_bounds.append((post_s, post_e))
        mt_pre = maze_time[pre_s:pre_e]
        mt_post = maze_time[post_s:post_e]
        mt_join = maze_time[pre_s:post_e]
        mt_pre = mt_pre[np.isfinite(mt_pre)]
        mt_post = mt_post[np.isfinite(mt_post)]
        mt_join = mt_join[np.isfinite(mt_join)]
        pre_selected_duration_sec = float(pre_e - pre_s) * float(DT)
        post_selected_duration_sec = float(post_e - post_s) * float(DT)
        pre_coverage = float(pre_selected_duration_sec / max(win_sec, 1e-12))
        post_coverage = float(post_selected_duration_sec / max(win_sec, 1e-12))
        rows.append(
            {
                "pivot_idx": int(len(rows)),
                "pivot_axis_idx": int(idx),
                "pivot_time_sec": float(pivot),
                "pivot_time_min": float(pivot / 60.0),
                "pre_start_sec": float(pivot - win_sec),
                "pre_stop_sec": float(pivot),
                "post_start_sec": float(pivot),
                "post_stop_sec": float(pivot + win_sec),
                "pre_start_idx": int(pre_s),
                "pre_stop_idx": int(pre_e),
                "post_start_idx": int(post_s),
                "post_stop_idx": int(post_e),
                "pre_rows": int(pre_e - pre_s),
                "post_rows": int(post_e - post_s),
                "pre_selected_duration_sec": pre_selected_duration_sec,
                "post_selected_duration_sec": post_selected_duration_sec,
                "pre_window_coverage_fraction": pre_coverage,
                "post_window_coverage_fraction": post_coverage,
                "window_coverage_fraction": float(min(pre_coverage, post_coverage)),
                "window_coverage_min_selected_duration_sec": float(min_selected_duration_sec),
                "window_coverage_gate_rule": "each_side_selected_duration_sec >= fixed_minimum_sec",
                "window_coverage_gate_pass": bool(
                    pre_selected_duration_sec >= float(min_selected_duration_sec)
                    and post_selected_duration_sec >= float(min_selected_duration_sec)
                ),
                "pivot_time_maze_sec": float(mt_post[0]) if mt_post.size else np.nan,
                "pre_elapsed_maze_sec": float(mt_pre[-1] - mt_pre[0]) if mt_pre.size > 1 else np.nan,
                "post_elapsed_maze_sec": float(mt_post[-1] - mt_post[0]) if mt_post.size > 1 else np.nan,
                "window_elapsed_maze_sec": float(mt_join[-1] - mt_join[0]) if mt_join.size > 1 else np.nan,
                "x_value": float(pivot),
                "x_unit": "sec",
            }
        )
    n_nonoverlap = max(1, int(math.floor(float(span) / max(win_sec, 1e-12))))
    return PivotDef(
        axis,
        float(window_min),
        f"{window_min:g}min",
        float(stride_sec),
        f"{stride_sec:g}s",
        np.asarray(pre_bounds, dtype=int),
        np.asarray(post_bounds, dtype=int),
        rows,
        n_nonoverlap,
        span,
        "sec",
    )


def _tile_pivot_def(
    selected: pd.DataFrame,
    *,
    session_id: str,
    behav_wd: str,
    window_tiles: int,
    stride_tiles: int,
    tile_axis_source: str,
) -> PivotDef:
    tile_df, audit = _add_tile_axis(selected, session_id=session_id, behav_wd=behav_wd, tile_axis_source=tile_axis_source)
    cum = tile_df["cum_tile_change_count"].to_numpy(dtype=float)
    if cum.size and not np.isfinite(cum).all():
        cum = pd.Series(cum).ffill().bfill().to_numpy(dtype=float)
        tile_df["cum_tile_change_count"] = cum
    n = int(len(tile_df))
    max_count = float(np.nanmax(cum)) if cum.size else 0.0
    if n < 2 or max_count <= 2.0 * float(window_tiles):
        return PivotDef("tile_change", float(window_tiles), f"{int(window_tiles)}tile", float(stride_tiles), f"{int(stride_tiles)}tile", np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int), [], 1, max_count, "tile", selected_for_labels=tile_df, audit=audit)
    pivots = np.arange(float(window_tiles), max_count - float(window_tiles) + 1e-9, float(stride_tiles), dtype=float)
    maze_time = pd.to_numeric(tile_df.get("maze_time", pd.Series(np.arange(n) * DT)), errors="coerce").to_numpy(dtype=float)
    frames = pd.to_numeric(tile_df.get("frame", pd.Series(np.arange(n))), errors="coerce").to_numpy(dtype=float)
    pre_bounds: list[tuple[int, int]] = []
    post_bounds: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    for idx, pivot in enumerate(pivots):
        pre_s = int(np.searchsorted(cum, pivot - float(window_tiles), side="left"))
        pre_e = int(np.searchsorted(cum, pivot, side="right"))
        post_s = int(np.searchsorted(cum, pivot, side="left"))
        post_e = int(np.searchsorted(cum, pivot + float(window_tiles), side="right"))
        pre_e = min(pre_e, n)
        post_e = min(post_e, n)
        if pre_e <= pre_s or post_e <= post_s:
            continue
        pre_bounds.append((pre_s, pre_e))
        post_bounds.append((post_s, post_e))
        mt_join = maze_time[pre_s:post_e]
        mt_join = mt_join[np.isfinite(mt_join)]
        frame_join = frames[pre_s:post_e]
        frame_join = frame_join[np.isfinite(frame_join)]
        raw_frame_span = float(frame_join[-1] - frame_join[0] + 1.0) if frame_join.size > 1 else np.nan
        selected_frame_fraction = float((post_e - pre_s) / raw_frame_span) if np.isfinite(raw_frame_span) and raw_frame_span > 0 else np.nan
        rows.append(
            {
                "pivot_idx": int(len(rows)),
                "pivot_axis_idx": int(idx),
                "pivot_time_sec": np.nan,
                "pivot_time_min": np.nan,
                "pre_start_tile": float(pivot - float(window_tiles)),
                "pre_stop_tile": float(pivot),
                "post_start_tile": float(pivot),
                "post_stop_tile": float(pivot + float(window_tiles)),
                "pivot_tile": float(pivot),
                "pre_start_idx": int(pre_s),
                "pre_stop_idx": int(pre_e),
                "post_start_idx": int(post_s),
                "post_stop_idx": int(post_e),
                "pre_rows": int(pre_e - pre_s),
                "post_rows": int(post_e - post_s),
                "pre_selected_duration_sec": float(pre_e - pre_s) * float(DT),
                "post_selected_duration_sec": float(post_e - post_s) * float(DT),
                "window_coverage_fraction": 1.0,
                "window_coverage_gate_pass": True,
                "pivot_time_maze_sec": float(mt_join[0]) if mt_join.size else np.nan,
                "window_elapsed_maze_sec": float(mt_join[-1] - mt_join[0]) if mt_join.size > 1 else np.nan,
                "window_raw_frame_span": raw_frame_span,
                "window_selected_frame_fraction": selected_frame_fraction,
                "x_value": float(pivot / 1000.0),
                "x_unit": "k_tile_changes",
            }
        )
    n_nonoverlap = max(1, int(math.floor(max_count / max(float(window_tiles), 1e-12))))
    return PivotDef(
        "tile_change",
        float(window_tiles),
        f"{int(window_tiles)}tile",
        float(stride_tiles),
        f"{int(stride_tiles)}tile",
        np.asarray(pre_bounds, dtype=int),
        np.asarray(post_bounds, dtype=int),
        rows,
        n_nonoverlap,
        max_count,
        "tile",
        selected_for_labels=tile_df,
        audit=audit,
    )


def _session_out_path(out_dir: Path, session_id: str) -> Path:
    return out_dir / "session_results" / f"{session_id}.csv"


def _status_out_path(out_dir: Path, session_id: str) -> Path:
    return out_dir / "session_status" / f"{session_id}.csv"


def _base_session_fields(
    row: pd.Series,
    bundle: dict[str, Any],
    signal: np.ndarray,
    *,
    signal_source: str,
    active_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": str(row["session_id"]),
        "animal_id": str(row["animal_id"]),
        "day": str(row["day"]),
        "cohort": str(row["cohort"]),
        "group": str(row["group"]),
        "month_label": str(row["month_label"]),
        "experience_group": str(row["experience_group"]),
        "neural_source_policy": str(row.get("neural_source_policy", "single_root")),
        "neural_source": str(row.get("neural_source", "single_root")),
        "neural_wd_used": str(row.get("neural_wd_used", "")),
        "neural_path": str(bundle.get("neural_path", "")),
        "n_neurons_before_active_gate": int(active_gate.get("n_neurons_before_active_gate", signal.shape[1])),
        "n_neurons": int(signal.shape[1]),
        "n_neurons_after_active_gate": int(signal.shape[1]),
        "n_neurons_removed_by_active_gate": int(active_gate.get("n_neurons_removed_by_active_gate", 0)),
        "n_selected_rows": int(signal.shape[0]),
        "signal_source": str(signal_source).upper(),
        "map_value_source": f"{str(signal_source).upper()}_dataset",
        "active_gate_source": str(active_gate.get("active_gate_source", "none")),
        "active_gate_threshold_per_min": active_gate.get("active_gate_threshold_per_min", np.nan),
        "active_gate_rate_median": active_gate.get("active_gate_rate_median", np.nan),
        "level_definition": str(LEVEL_DEFINITION_GEOMETRIC8),
        "comparison_kind": "within_session_pivot_pre_post",
        "reference_window": "pre_pivot",
        "target_window": "post_pivot",
        "target_reference_same_session": True,
    }


def _active_gate_neuron_ids(bundle: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    n_neurons = int(bundle["neural_n_neurons"])
    neuron_ids = np.arange(n_neurons, dtype=int)
    gate_source = str(args.active_gate_source)
    threshold = float(args.min_active_signal_per_min)
    if gate_source == "none" or not np.isfinite(threshold):
        return neuron_ids, {
            "active_gate_source": "none",
            "active_gate_threshold_per_min": np.nan,
            "active_gate_rate_median": np.nan,
            "n_neurons_before_active_gate": n_neurons,
            "n_neurons_removed_by_active_gate": 0,
        }
    gate_signal = _signal_matrix_from_path(bundle, gate_source, neuron_ids)
    total_time_min = max(float(gate_signal.shape[0]) * float(DT) / 60.0, 1e-12)
    rates = np.nansum(np.asarray(gate_signal, dtype=float), axis=0) / total_time_min
    keep = np.asarray(rates >= threshold, dtype=bool)
    kept_ids = neuron_ids[keep]
    return kept_ids, {
        "active_gate_source": f"{gate_source}_dataset",
        "active_gate_threshold_per_min": threshold,
        "active_gate_rate_median": float(np.nanmedian(rates)) if rates.size else np.nan,
        "n_neurons_before_active_gate": n_neurons,
        "n_neurons_removed_by_active_gate": int(n_neurons - kept_ids.size),
    }


def _metric_record(
    base: dict[str, Any],
    pivot: PivotDef,
    row_fields: dict[str, Any],
    *,
    metric_name: str,
    map_type: str,
    unit_type: str,
    support_policy: str,
    pre_support: np.ndarray | None,
    post_support: np.ndarray | None,
    common_support: np.ndarray | None,
    values: np.ndarray | None,
    valid_counts: np.ndarray | None,
    gate_pass: bool,
    support_gate_pass: bool,
    coverage_gate_pass: bool,
    n_bins: int | None,
    allowed_bins: str,
    label_source: str,
    rate_info: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    vals = np.asarray(values if values is not None else np.asarray([], dtype=float), dtype=float)
    finite = np.isfinite(vals)
    z_values = _safe_atanh(vals)
    z_mean = _safe_mean(z_values)
    rec = {
        **base,
        "metric": metric_name,
        "map_type": map_type,
        "unit_type": unit_type,
        "support_policy": support_policy,
        "window_type": pivot.window_type,
        "window_size": float(pivot.window_size),
        "window_size_label": pivot.window_size_label,
        "stride": float(pivot.stride),
        "stride_label": pivot.stride_label,
        "target_n_nonoverlap_windows": int(pivot.n_nonoverlap_windows),
        **row_fields,
        "min_bins": 3 if map_type == "place" else (4 if map_type == "level" else np.nan),
        "min_neurons_for_pv": 3 if metric_name == "pv_correlation" else np.nan,
        "pre_valid_bins": int(np.count_nonzero(pre_support)) if pre_support is not None else np.nan,
        "post_valid_bins": int(np.count_nonzero(post_support)) if post_support is not None else np.nan,
        "common_bins": int(np.count_nonzero(common_support)) if common_support is not None else np.nan,
        "pre_valid_fraction": float(np.count_nonzero(pre_support) / max(int(n_bins or 1), 1)) if pre_support is not None else np.nan,
        "post_valid_fraction": float(np.count_nonzero(post_support) / max(int(n_bins or 1), 1)) if post_support is not None else np.nan,
        "common_bin_fraction": float(np.count_nonzero(common_support) / max(int(n_bins or 1), 1)) if common_support is not None else np.nan,
        "support_gate_pass": bool(support_gate_pass),
        "coverage_gate_pass": bool(coverage_gate_pass),
        "gate_pass": bool(gate_pass),
        "n_finite_neurons": int(np.count_nonzero(finite)) if unit_type == "neuron" else np.nan,
        "n_finite_bins": int(np.count_nonzero(finite)) if unit_type == "spatial_bin" else np.nan,
        "n_finite_cell_corr": int(np.count_nonzero(finite)),
        "finite_neuron_fraction": float(np.count_nonzero(finite) / max(vals.size, 1)) if unit_type == "neuron" and vals.size else np.nan,
        "finite_bin_fraction": float(np.count_nonzero(finite) / max(int(n_bins or 1), 1)) if unit_type == "spatial_bin" else np.nan,
        "cell_corr_mean": _safe_mean(vals),
        "cell_corr_z_mean": z_mean,
        "cell_corr_zmean_backtransformed": float(np.tanh(z_mean)) if np.isfinite(z_mean) else np.nan,
        "sim_mean": _safe_mean(vals),
        "sim_z_mean": z_mean,
        "sim_median": float(np.nanmedian(vals)) if vals.size and np.isfinite(vals).any() else np.nan,
        "sim_std": float(np.nanstd(vals)) if vals.size and np.isfinite(vals).any() else np.nan,
        "allowed_bin_indices": allowed_bins,
        "label_source": label_source,
    }
    if valid_counts is not None and np.asarray(valid_counts).size:
        rec["valid_count_min"] = int(np.nanmin(valid_counts))
        rec["valid_count_median"] = float(np.nanmedian(valid_counts))
    if common_support is not None:
        rec["common_bin_indices"] = _mask_indices(common_support)
    if rate_info is not None:
        n_used, n_nonzero_both, n_silent_both = rate_info
        rec["rate_n_used"] = int(n_used)
        rec["rate_n_nonzero_both"] = int(n_nonzero_both)
        rec["rate_n_silent_both"] = int(n_silent_both)
    return rec


def _compute_session(args: argparse.Namespace, row: pd.Series) -> None:
    session_id = str(row["session_id"])
    result_path = _session_out_path(args.out_dir, session_id)
    status_path = _status_out_path(args.out_dir, session_id)
    if result_path.exists() and status_path.exists() and not args.overwrite:
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    neural_wd_used = str(row.get("neural_wd_used", args.neural_wd) or args.neural_wd)
    source_fields = {
        "neural_source_policy": str(row.get("neural_source_policy", "single_root")),
        "neural_source": str(row.get("neural_source", "single_root")),
        "neural_wd_used": neural_wd_used,
    }
    curve_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    try:
        bundle = _load_bundle(
            session_id,
            behav_wd=str(args.behav_wd),
            neural_wd=neural_wd_used,
            level_definition=LEVEL_DEFINITION_GEOMETRIC8,
            level_col="level",
        )
        neuron_ids, active_gate = _active_gate_neuron_ids(bundle, args)
        signal = _signal_matrix_from_path(bundle, str(args.signal_source), neuron_ids)
        base = _base_session_fields(row, bundle, signal, signal_source=str(args.signal_source), active_gate=active_gate)
        selected = bundle["selected_df"]

        pivot_defs: list[PivotDef] = []
        if "selected_time" in args.window_types:
            for win in args.time_windows_min:
                pivot_defs.append(_time_pivot_def(selected, axis="selected_time", window_min=float(win), stride_sec=float(args.time_stride_sec), min_selected_duration_sec=float(args.min_window_selected_duration_sec)))
        if "maze_time" in args.window_types:
            for win in args.time_windows_min:
                pivot_defs.append(_time_pivot_def(selected, axis="maze_time", window_min=float(win), stride_sec=float(args.time_stride_sec), min_selected_duration_sec=float(args.min_window_selected_duration_sec)))
        if "absolute_time" in args.window_types:
            for win in args.time_windows_min:
                pivot_defs.append(_time_pivot_def(selected, axis="absolute_time", window_min=float(win), stride_sec=float(args.time_stride_sec), min_selected_duration_sec=float(args.min_window_selected_duration_sec)))
        if "tile_change" in args.window_types:
            for win in args.tile_windows:
                pivot_defs.append(
                    _tile_pivot_def(
                        selected,
                        session_id=session_id,
                        behav_wd=str(args.behav_wd),
                        window_tiles=int(win),
                        stride_tiles=int(args.tile_stride),
                        tile_axis_source=str(args.tile_axis_source),
                    )
                )

        map_types_needed = sorted({METRIC_TO_MAP_TYPE[m] for m in args.metrics if m in METRIC_TO_MAP_TYPE})
        _drop_signal_payload(bundle)
        gc.collect()

        for pivot in pivot_defs:
            coverage_values = np.asarray([r.get("window_coverage_fraction", np.nan) for r in pivot.rows], dtype=float)
            status_rows.append(
                {
                    **base,
                    "window_type": pivot.window_type,
                    "window_size": float(pivot.window_size),
                    "window_size_label": pivot.window_size_label,
                    "stride": float(pivot.stride),
                    "stride_label": pivot.stride_label,
                    "n_pivots": int(pivot.pre_bounds.shape[0]),
                    "n_nonoverlap_windows": int(pivot.n_nonoverlap_windows),
                    "span_value": float(pivot.span_value),
                    "span_unit": pivot.span_unit,
                    "skip_reason": pivot.skip_reason,
                    "window_coverage_fraction_min": float(np.nanmin(coverage_values)) if coverage_values.size and np.isfinite(coverage_values).any() else np.nan,
                    "window_coverage_fraction_median": float(np.nanmedian(coverage_values)) if coverage_values.size and np.isfinite(coverage_values).any() else np.nan,
                    "n_pivots_passing_coverage_gate": int(sum(bool(r.get("window_coverage_gate_pass", True)) for r in pivot.rows)),
                    **(pivot.audit or {}),
                }
            )
            if pivot.pre_bounds.size == 0 or pivot.post_bounds.size == 0:
                continue

            selected_for_labels = pivot.selected_for_labels if pivot.selected_for_labels is not None else selected
            labels_by_map: dict[str, tuple[np.ndarray, int, str]] = {}
            for map_type in map_types_needed:
                labels_bundle = {**bundle, "selected_df": selected_for_labels}
                labels_by_map[map_type] = _labels_for_map(labels_bundle, map_type, level_col="level", n_levels=8)

            for map_type, (labels, n_bins, label_source) in labels_by_map.items():
                metric_names = [m for m in args.metrics if METRIC_TO_MAP_TYPE.get(m) == map_type]
                if not metric_names:
                    continue
                pre_iter = _iter_rate_maps(signal, labels, pivot.pre_bounds, n_bins=n_bins, dt_eff=float(DT))
                post_iter = _iter_rate_maps(signal, labels, pivot.post_bounds, n_bins=n_bins, dt_eff=float(DT))
                for w_idx, (pre_pair, post_pair) in enumerate(zip(pre_iter, post_iter)):
                    pre_rates, pre_occ = pre_pair
                    post_rates, post_occ = post_pair
                    row_fields = pivot.rows[w_idx]
                    coverage_gate_pass = bool(row_fields.get("window_coverage_gate_pass", True))
                    for policy in args.support_policies:
                        if policy == "legacy_place_frame" and map_type != "place":
                            continue
                        base_thr, eff_thr, n_nonoverlap, min_common = _policy_threshold(policy, map_type, n_nonoverlap=pivot.n_nonoverlap_windows, n_bins=n_bins)
                        pre_map, pre_support = _thresholded_map(pre_rates, pre_occ, eff_thr)
                        post_map, post_support = _thresholded_map(post_rates, post_occ, eff_thr)
                        pre_map, pre_support, allowed_bins = _apply_policy_allowed_bins(pre_map, pre_support, policy=policy, map_type=map_type)
                        post_map, post_support, _ = _apply_policy_allowed_bins(post_map, post_support, policy=policy, map_type=map_type)
                        common = pre_support & post_support
                        min_bins = 3 if map_type == "place" else 4
                        gate_min_common = int(min_common) if min_common is not None else int(min_bins)
                        support_gate_pass = bool(int(np.count_nonzero(common)) >= gate_min_common)
                        gate_pass = bool(support_gate_pass and coverage_gate_pass)
                        for metric_name in metric_names:
                            if metric_name in {"place_map_correlation", "level_map_correlation"}:
                                if gate_pass:
                                    values, valid_counts = _per_cell_corr(pre_map, post_map, common, min_bins=min_bins)
                                else:
                                    values = np.full(signal.shape[1], np.nan, dtype=float)
                                    valid_counts = np.zeros(signal.shape[1], dtype=int)
                                curve_rows.append(
                                    _metric_record(
                                        base,
                                        pivot,
                                        row_fields,
                                        metric_name=metric_name,
                                        map_type=map_type,
                                        unit_type="neuron",
                                        support_policy=policy,
                                        pre_support=pre_support,
                                        post_support=post_support,
                                        common_support=common,
                                        values=values,
                                        valid_counts=valid_counts,
                                        gate_pass=gate_pass,
                                        support_gate_pass=support_gate_pass,
                                        coverage_gate_pass=coverage_gate_pass,
                                        n_bins=n_bins,
                                        allowed_bins=allowed_bins,
                                        label_source=label_source,
                                    )
                                )
                            elif metric_name == "pv_correlation":
                                if gate_pass:
                                    values, valid_counts = _pv_corr_by_bin(pre_map, post_map, common, min_neurons=3)
                                else:
                                    values = np.full(int(n_bins), np.nan, dtype=float)
                                    valid_counts = np.zeros(int(n_bins), dtype=int)
                                curve_rows.append(
                                    _metric_record(
                                        base,
                                        pivot,
                                        row_fields,
                                        metric_name=metric_name,
                                        map_type=map_type,
                                        unit_type="spatial_bin",
                                        support_policy=policy,
                                        pre_support=pre_support,
                                        post_support=post_support,
                                        common_support=common,
                                        values=values,
                                        valid_counts=valid_counts,
                                        gate_pass=gate_pass,
                                        support_gate_pass=support_gate_pass,
                                        coverage_gate_pass=coverage_gate_pass,
                                        n_bins=n_bins,
                                        allowed_bins=allowed_bins,
                                        label_source=label_source,
                                    )
                                )

            if "ensemble_rate_correlation" in args.metrics:
                for w_idx, (pre_bound, post_bound) in enumerate(zip(pivot.pre_bounds, pivot.post_bounds)):
                    row_fields = pivot.rows[w_idx]
                    pre_rate = _rate_vector(signal, int(pre_bound[0]), int(pre_bound[1]))
                    post_rate = _rate_vector(signal, int(post_bound[0]), int(post_bound[1]))
                    rate_corr, n_used, n_nonzero_both, n_silent_both = _rate_corr(pre_rate, post_rate)
                    values = np.asarray([rate_corr], dtype=float)
                    curve_rows.append(
                        _metric_record(
                            base,
                            pivot,
                            row_fields,
                            metric_name="ensemble_rate_correlation",
                            map_type="ensemble_rate",
                            unit_type="matched_neuron_rate_vector",
                            support_policy=RATE_SUPPORT_POLICY,
                            pre_support=None,
                            post_support=None,
                            common_support=None,
                            values=values,
                            valid_counts=None,
                            gate_pass=bool(row_fields.get("window_coverage_gate_pass", True) and np.isfinite(rate_corr)),
                            support_gate_pass=True,
                            coverage_gate_pass=bool(row_fields.get("window_coverage_gate_pass", True)),
                            n_bins=None,
                            allowed_bins="not_applicable",
                            label_source="matched_neuron_rate_vector",
                            rate_info=(n_used, n_nonzero_both, n_silent_both),
                        )
                    )
    except Exception as exc:
        status_rows.append(
            {
                "session_id": session_id,
                **source_fields,
                "status": "error",
                "error": repr(exc),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
        )
    finally:
        try:
            del bundle
        except UnboundLocalError:
            pass
        gc.collect()

    pd.DataFrame(curve_rows).to_csv(result_path, index=False)
    status = pd.DataFrame(status_rows)
    if not status.empty and "elapsed_sec" not in status.columns:
        status["elapsed_sec"] = float(time.perf_counter() - t0)
    status.to_csv(status_path, index=False)


def _compute_session_job(args_dict: dict[str, Any], row_dict: dict[str, Any]) -> str:
    args = argparse.Namespace(**args_dict)
    row = pd.Series(row_dict)
    _compute_session(args, row)
    return str(row_dict.get("session_id", ""))


def _read_session_files(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curves = []
    statuses = []
    for path in sorted((out_dir / "session_results").glob("*.csv")) if (out_dir / "session_results").exists() else []:
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            curves.append(df)
    for path in sorted((out_dir / "session_status").glob("*.csv")) if (out_dir / "session_status").exists() else []:
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            statuses.append(df)
    return (
        pd.concat(curves, ignore_index=True, sort=False) if curves else pd.DataFrame(),
        pd.concat(statuses, ignore_index=True, sort=False) if statuses else pd.DataFrame(),
    )


def _summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    work = curves.copy()
    work = work[np.isfinite(pd.to_numeric(work["sim_mean"], errors="coerce"))].copy()
    work["sim_mean"] = pd.to_numeric(work["sim_mean"], errors="coerce")
    animal_cols = [
        "day",
        "experience_group",
        "animal_id",
        "metric",
        "map_type",
        "support_policy",
        "window_type",
        "window_size",
        "window_size_label",
        "x_value",
        "x_unit",
    ]
    animal = (
        work.groupby(animal_cols, dropna=False, as_index=False)
        .agg(mean=("sim_mean", "mean"), n_sessions=("session_id", "nunique"), n_rows=("sim_mean", "size"))
    )
    rows: list[dict[str, Any]] = []
    group_cols = [c for c in animal_cols if c != "animal_id"]
    for key, sub in animal.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, key))
        vals = sub["mean"].to_numpy(dtype=float)
        rows.append(
            {
                **rec,
                "mean": _safe_mean(vals),
                "sem": _safe_sem(vals),
                "std": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan,
                "n_animals": int(sub["animal_id"].nunique()),
                "n_session_means": int(sub["n_sessions"].sum()),
                "n_rows": int(sub["n_rows"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["day", "metric", "support_policy", "window_type", "window_size", "experience_group", "x_value"])


def _plot_metric_day_window(group: pd.DataFrame, out_dir: Path, *, metric: str, day: str, window_type: str, support_policy: str) -> None:
    sub = group[
        group["metric"].eq(metric)
        & group["day"].eq(day)
        & group["window_type"].eq(window_type)
        & group["support_policy"].eq(support_policy)
    ].copy()
    if sub.empty:
        return
    sizes = sorted(float(v) for v in sub["window_size"].dropna().unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.8 * len(sizes), 3.6), squeeze=False, sharey=True)
    y_vals: list[float] = []
    for ax, size in zip(axes[0], sizes):
        panel = sub[np.isclose(sub["window_size"].astype(float), float(size))]
        for group_name in GROUP_ORDER:
            d = panel[panel["experience_group"].eq(group_name)].sort_values("x_value")
            if d.empty:
                continue
            x = pd.to_numeric(d["x_value"], errors="coerce").to_numpy(dtype=float)
            if str(d["x_unit"].dropna().iloc[0]) == "sec":
                x = x / 60.0
            y = pd.to_numeric(d["mean"], errors="coerce").to_numpy(dtype=float)
            sem = pd.to_numeric(d["sem"], errors="coerce").fillna(0).to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if not np.any(ok):
                continue
            x = x[ok]
            y = y[ok]
            sem = sem[ok]
            y_vals.extend(y)
            y_vals.extend(y - sem)
            y_vals.extend(y + sem)
            color = GROUP_COLORS.get(group_name, "0.35")
            ax.plot(x, y, color=color, lw=2.0, label=group_name)
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
        title = f"{size:g} min" if window_type != "tile_change" else f"{size:g} tiles"
        ax.set_title(title)
        ax.set_xlabel("Pivot time (min)" if window_type != "tile_change" else "Pivot position (k tile changes)")
        ax.grid(True, axis="y", color="0.88", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    finite_y = np.asarray([v for v in y_vals if np.isfinite(v)], dtype=float)
    if finite_y.size:
        lo, hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        if metric != "ensemble_rate_correlation":
            lo = min(lo, -0.05)
            hi = max(hi, 1.0)
        pad = max(0.05, 0.08 * max(hi - lo, 0.1))
        for ax in axes[0]:
            ax.set_ylim(lo - pad, hi + pad)
    axes[0, 0].set_ylabel(METRIC_LABELS.get(metric, metric))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    safe_metric = metric.replace("_correlation", "")
    path = out_dir / f"{day}_pivot_{safe_metric}_{window_type}_{support_policy}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_all(group: pd.DataFrame, out_dir: Path, *, support_policies: list[str]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
        }
    )
    available_policies = set(group["support_policy"].astype(str))
    for metric in METRIC_CHOICES:
        metric_policies = [RATE_SUPPORT_POLICY] if metric == "ensemble_rate_correlation" else list(support_policies)
        for policy in metric_policies:
            if policy not in available_policies:
                continue
            for day in ["day1", "day2"]:
                for window_type in DEFAULT_WINDOW_TYPES:
                    _plot_metric_day_window(group, plot_dir, metric=metric, day=day, window_type=window_type, support_policy=policy)


def _write_report(out_dir: Path, manifest: pd.DataFrame, curves: pd.DataFrame, status: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Full Within-Session Pivot All-Metrics Export",
        "",
        "This folder contains the sorted-first pivot/pre-post within-session analysis.",
        "",
        "## Scope",
        "",
        f"- Manifest sessions: `{len(manifest)}`.",
        f"- Curve rows: `{len(curves)}`.",
        f"- Status rows: `{len(status)}`.",
        f"- Metrics: `{', '.join(args.metrics)}`.",
        f"- Window types: `{', '.join(args.window_types)}`.",
        f"- Time windows min: `{', '.join(str(v) for v in args.time_windows_min)}`.",
        f"- Each side of a time pivot requires at least `{args.min_window_selected_duration_sec:g}` s of selected data (`{int(math.ceil(float(args.min_window_selected_duration_sec) / float(DT)))}` frames at DT={DT:g} s).",
        f"- Tile windows: `{', '.join(str(v) for v in args.tile_windows)}`.",
        f"- Support policies: `{', '.join(args.support_policies)}`.",
        f"- Primary plot support policy: `{args.primary_support_policy}`.",
        f"- Signal source: `{str(args.signal_source).upper()}`.",
        f"- Active-neuron gate source: `{args.active_gate_source}` at `{args.min_active_signal_per_min:g}` signal/min.",
        "",
        "## Neural Source Counts",
        "",
        "```csv",
        manifest["neural_source"].value_counts(dropna=False).rename_axis("neural_source").reset_index(name="n_sessions").to_csv(index=False).strip(),
        "```",
        "",
        "## Saved Outputs",
        "",
        "- `per_pivot_curves.csv`",
        "- `session_status.csv`",
        "- `group_curve_summary.csv`",
        "- `plots/`",
        "- `manifest.json`",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _target_manifest(args)
    manifest.to_csv(args.out_dir / "session_manifest.csv", index=False)
    if int(args.n_jobs) > 1:
        args_dict = vars(args).copy()
        rows = [row.to_dict() for _, row in manifest.iterrows()]
        completed = 0
        with futures.ProcessPoolExecutor(max_workers=int(args.n_jobs)) as pool:
            future_to_session = {
                pool.submit(_compute_session_job, args_dict, row_dict): str(row_dict["session_id"])
                for row_dict in rows
            }
            for fut in futures.as_completed(future_to_session):
                session_id = future_to_session[fut]
                completed += 1
                try:
                    fut.result()
                    print(f"[{completed}/{len(rows)}] done {session_id}", flush=True)
                except Exception as exc:
                    print(f"[{completed}/{len(rows)}] failed {session_id}: {exc!r}", flush=True)
    else:
        for idx, row in manifest.iterrows():
            print(f"[{idx + 1}/{len(manifest)}] {row['session_id']}", flush=True)
            _compute_session(args, row)
    curves, status = _read_session_files(args.out_dir)
    curves.to_csv(args.out_dir / "per_pivot_curves.csv", index=False)
    status.to_csv(args.out_dir / "session_status.csv", index=False)
    failures = status[status.get("status", pd.Series(dtype=str)).astype(str).eq("error")] if not status.empty and "status" in status.columns else pd.DataFrame()
    failures.to_csv(args.out_dir / "failures.csv", index=False)
    group = _summarize_curves(curves)
    group.to_csv(args.out_dir / "group_curve_summary.csv", index=False)
    if not group.empty:
        _plot_all(group, args.out_dir, support_policies=list(args.support_policies))
    payload = {
        "analysis": "within_session_pivot_all_metrics",
        "out_dir": str(args.out_dir),
        "behav_wd": str(args.behav_wd),
        "neural_wd": str(args.neural_wd),
        "sorted_neural_wd": str(args.sorted_neural_wd),
        "neural_source_policy": str(args.neural_source_policy),
        "neural_source_counts": manifest["neural_source"].value_counts(dropna=False).to_dict(),
        "days": list(args.days),
        "metrics": list(args.metrics),
        "window_types": list(args.window_types),
        "time_windows_min": [float(v) for v in args.time_windows_min],
        "min_window_selected_duration_sec": float(args.min_window_selected_duration_sec),
        "min_window_selected_frames_equivalent": int(math.ceil(float(args.min_window_selected_duration_sec) / float(DT))),
        "window_coverage_gate_rule": "each_side_selected_duration_sec >= fixed_minimum_sec",
        "tile_windows": [int(v) for v in args.tile_windows],
        "support_policies": list(args.support_policies),
        "primary_support_policy": str(args.primary_support_policy),
        "signal_source": str(args.signal_source).upper(),
        "active_gate_source": str(args.active_gate_source),
        "min_active_signal_per_min": float(args.min_active_signal_per_min),
        "n_manifest_sessions": int(len(manifest)),
        "n_curve_rows": int(len(curves)),
        "n_status_rows": int(len(status)),
        "n_failures": int(len(failures)),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.out_dir, manifest, curves, status, args)
    print(f"Wrote {args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--behav-wd", default=get_behav_wd())
    p.add_argument("--neural-wd", default=get_neural_wd())
    p.add_argument("--sorted-neural-wd", default="")
    p.add_argument("--neural-source-policy", choices=["single_root", "sorted_first"], default="single_root")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--days", nargs="+", default=["day1", "day2"], choices=["day1", "day2"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--metrics", default=",".join(METRIC_CHOICES), type=_parse_csv_str)
    p.add_argument("--window-types", default=",".join(DEFAULT_WINDOW_TYPES), type=_parse_csv_str)
    p.add_argument("--time-windows-min", default=",".join(str(v) for v in DEFAULT_TIME_WINDOWS_MIN), type=_parse_csv_float)
    p.add_argument("--time-stride-sec", type=float, default=60.0)
    p.add_argument(
        "--min-window-selected-duration-sec",
        type=float,
        default=DEFAULT_MIN_WINDOW_SELECTED_DURATION_SEC,
        help="Fixed minimum selected-data duration required on each side of a time pivot (default: 300 s / 5 min).",
    )
    p.add_argument("--tile-windows", default=",".join(str(v) for v in DEFAULT_TILE_WINDOWS), type=_parse_csv_int)
    p.add_argument("--tile-stride", type=int, default=100)
    p.add_argument("--tile-axis-source", default=DEFAULT_TILE_AXIS_SOURCE)
    p.add_argument("--support-policies", default=",".join(DEFAULT_PIVOT_SUPPORT_POLICIES), type=_parse_csv_str)
    p.add_argument("--primary-support-policy", default="legacy_pivot")
    p.add_argument("--signal-source", choices=["C", "S"], default="C")
    p.add_argument("--active-gate-source", choices=["S", "C", "none"], default="S")
    p.add_argument("--min-active-signal-per-min", type=float, default=0.1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
