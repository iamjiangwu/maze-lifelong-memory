"""Draw the Figure 4 method schematic and compose its SVG crop.

build_figure4_standardized_pv supplies source paths and calls these
helpers; the remaining panels have their own source renderers.
Historical path defaults below are replaced by that launcher.
"""

from __future__ import annotations

import json

import math

import os

import pickle

from dataclasses import asdict, dataclass, field, replace as dataclass_replace

from pathlib import Path

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.colors import Normalize

from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    PANEL_LAYOUT,
    apply_panel_rcparams,
    compute_example_cell_arrow_data,
    get_panel_maze,
    load_neuron_data_bundle,
    render_spatial_decoding_error_map,
    render_example_cell_day12_placemap_panel,
    render_example_cell_grid_panel,
    resolve_neuron_data_path,
)

from ca_analysis.figure_colors import get_correlation_experience_colors

REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURE4_PANEL_A_PPT_CROP = {"l": 12400, "t": 15055, "r": 9866, "b": 9331}

FIGURE4_SCHEMATIC_PAIR = ("837day1", "837-3day1")

FIGURE4_SCHEMATIC_DECODER_PKL = (
    REPO_ROOT
    / "docs"
    / "cross_session_decoding"
    / f"cross_session_{FIGURE4_SCHEMATIC_PAIR[0]}_to_{FIGURE4_SCHEMATIC_PAIR[1]}.pkl"
)

FIGURE4_SCHEMATIC_MAP_BINS = 15

FIGURE4_SCHEMATIC_N_CELLS = 5

FIGURE4_SCHEMATIC_SOURCE_CELL_IDS = (117, 12, 10, 24, 70)

FIGURE4_SCHEMATIC_TRACE_POINTS = 5000

SVG_NS = "http://www.w3.org/2000/svg"

XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)

ET.register_namespace("xlink", XLINK_NS)

EXPERIENCE_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")

FIGURE2_TOP_ROW_SHIFT_PX = 7.0

FIGURE2_TOP_ROW_SHIFT_REFERENCE_DPI = 300.0

FIGURE2_TOP_ROW_SHIFT_IN = FIGURE2_TOP_ROW_SHIFT_PX / FIGURE2_TOP_ROW_SHIFT_REFERENCE_DPI

FIGURE2_PPT_TOP_SLOTS = {
    "a": (0.048486439195100615, 0.04500218722659668 + FIGURE2_TOP_ROW_SHIFT_IN, 1.4323129921259843, 2.1070395888014),
    "b_labels": (1.4199792213473317, FIGURE2_TOP_ROW_SHIFT_IN, 0.125, 2.3166218285214346),
    "b1": (1.5450, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b2": (2.7072189413823273, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b3": (3.8293285214348205, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b4": (4.946116579177603, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
}

FIGURE2_PANEL_B_MIN_COLUMN_GAP_IN = min(
    FIGURE2_PPT_TOP_SLOTS[f"b{index + 1}"][0]
    - (FIGURE2_PPT_TOP_SLOTS[f"b{index}"][0] + FIGURE2_PPT_TOP_SLOTS[f"b{index}"][2])
    for index in range(1, 4)
)

FIGURE3_FIRST30_EXPERIENCE_COLOR_GROUPS = {
    index: group for index, group in enumerate(EXPERIENCE_ORDER)
}

FIGURE3_LONG_GAP_EXPERIENCE_COLOR_GROUPS = {
    index: EXPERIENCE_ORDER[index] for index in range(1, len(EXPERIENCE_ORDER))
}

EXPERIENCE_INDEX = {label: index for index, label in enumerate(EXPERIENCE_ORDER)}

@dataclass(frozen=True)
class PublicationStyle:
    """Single formatting contract for source-rendered manuscript panels."""

    font_family: str = "Arial"
    base_fontsize: float = 6.0
    axis_label_fontsize: float = 6.0
    tick_fontsize: float = 5.4
    title_fontsize: float = 6.2
    legend_fontsize: float = 4.9
    annotation_fontsize: float = 5.4
    panel_letter_fontsize: float = 10.5
    panel_letter_weight: str = "bold"
    axis_linewidth: float = 0.65
    tick_width: float = 0.55
    tick_length: float = 1.8
    grid_linewidth: float = 0.35
    grid_alpha: float = 0.26
    show_grid: bool = True
    data_linewidth: float = 1.25
    fit_linewidth: float = 1.10
    sem_alpha: float = 0.07
    point_size: float = 15.0
    figure2_marker_scale: float = 1.0
    figure2_pv_distribution_xlabel_x: float | None = None
    point_alpha: float = 0.72
    low_support_alpha: float = 0.24
    low_support_linewidth: float = 0.65
    figure_size: tuple[float, float] = (5.35, 7.25)
    figure2_canvas_size: tuple[float, float] = (7.10, 6.18)
    figure2_panel_a_slot: tuple[float, float, float, float] = (0.06, 0.04, 1.70, 2.24)
    figure2_panel_b_slot: tuple[float, float, float, float] = (1.72, 0.03, 5.34, 2.26)
    figure2_lower_slot: tuple[float, float, float, float] = (0.02, 2.30, 7.05, 3.84)
    figure2_export_width_mm: float = 180.0
    figure4_canvas_size: tuple[float, float] = (7.50, 4.00)
    figure4_export_width_mm: float = 180.0
    figure4_panel_a_slot: tuple[float, float, float, float] = (0.056, 0.127, 3.822, 1.646)
    figure4_panel_b_slot: tuple[float, float, float, float] = (3.878, 0.000, 0.749, 2.089)
    figure4_panel_c_slot: tuple[float, float, float, float] = (4.760, 0.205, 2.005, 1.490)
    figure4_panel_d_slot: tuple[float, float, float, float] = (6.550, 0.000, 0.749, 2.089)
    figure4_panel_e_slot: tuple[float, float, float, float] = (-0.030, 1.724, 4.899, 2.126)
    figure4_panel_f_slot: tuple[float, float, float, float] = (4.720, 1.773, 2.650, 2.226)
    figure3_top_bounds: tuple[float, float, float, float] = (0.145, 0.895, 0.785, 0.965)
    figure3_lower_bounds: tuple[float, float, float, float] = (0.145, 0.895, 0.070, 0.724)
    figure3_lower_wspace: float = 0.32
    figure3_lower_hspace: float = 0.44
    figure3_export_width_mm: float = 136.0
    day2_template_xlim_min: float = 25.0
    day2_template_xlim_max: float = 185.0
    max_lower_panel_aspect: float = 0.95
    dpi: int = 300
    facecolor: str = "white"
    max_blank_right_fraction: float = 0.12
    max_blank_bottom_fraction: float = 0.12
    figure3_metric_y_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "place_map_correlation": (0.0, 0.65),
            "pv_correlation": (0.0, 0.75),
            "ensemble_rate_correlation": (0.0, 0.90),
            "level_map_correlation": (0.0, 0.90),
        }
    )
    figure3_metric_y_ticks: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "place_map_correlation": (0.0, 0.2, 0.4, 0.6),
            "pv_correlation": (0.0, 0.2, 0.4, 0.6),
            "ensemble_rate_correlation": (0.2, 0.4, 0.6, 0.8),
            "level_map_correlation": (0.2, 0.4, 0.6, 0.8),
        }
    )
    metric_colors: dict[str, str] = field(
        default_factory=lambda: {
            "place_map_correlation": CORRELATION_MEASURE_COLORS["place_map_correlation"],
            "pv_correlation": CORRELATION_MEASURE_COLORS["pv_correlation"],
            "ensemble_rate_correlation": CORRELATION_MEASURE_COLORS["ensemble_rate_correlation"],
            "level_map_correlation": CORRELATION_MEASURE_COLORS["level_map_correlation"],
        }
    )
    day2_group_colors: dict[str, dict[str, str]] = field(
        default_factory=get_correlation_experience_colors
    )

    def rcparams(self) -> dict[str, object]:
        return {
            "font.family": self.font_family,
            "font.size": self.base_fontsize,
            "axes.labelsize": self.axis_label_fontsize,
            "axes.titlesize": self.title_fontsize,
            "xtick.labelsize": self.tick_fontsize,
            "ytick.labelsize": self.tick_fontsize,
            "legend.fontsize": self.legend_fontsize,
            "axes.linewidth": self.axis_linewidth,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": self.facecolor,
        }

def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())

def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"

def _matrix_frames_by_neuron(matrix: np.ndarray, n_neurons: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {arr.shape}")
    if arr.shape[1] == n_neurons:
        return arr
    if arr.shape[0] == n_neurons:
        return arr.T
    raise ValueError(f"matrix shape {arr.shape} does not match n_neurons={n_neurons}")

def _load_figure4_schematic_session(session_id: str) -> dict[str, object]:
    from ca_analysis.config import get_behav_wd, get_neural_wd
    from ca_analysis.io import load_behav_neural

    _, selected_df, _, _raw_counts, calcium, spike_counts, _ = load_behav_neural(
        session_id,
        behav_wd=get_behav_wd(),
        neural_wd=get_neural_wd(),
    )
    if spike_counts is None:
        raise RuntimeError(f"{session_id}: missing spike/activity matrix")
    n_neurons = int(np.asarray(spike_counts).shape[0])
    spike_tn_all = _matrix_frames_by_neuron(spike_counts, n_neurons)
    calcium_tn_all = _matrix_frames_by_neuron(calcium, n_neurons) if calcium is not None else spike_tn_all

    selected_df = selected_df.copy().reset_index(drop=True)
    frames = selected_df["frame"].to_numpy(dtype=int)
    valid = (frames >= 0) & (frames < spike_tn_all.shape[0]) & (frames < calcium_tn_all.shape[0])
    if calcium is not None:
        valid &= ~np.any(np.isnan(calcium_tn_all[frames.clip(0, calcium_tn_all.shape[0] - 1), :]), axis=1)
    selected_df = selected_df.loc[valid].reset_index(drop=True)
    frames = selected_df["frame"].to_numpy(dtype=int)

    return {
        "selected_df": selected_df,
        "positions": selected_df[["mass_x", "mass_y"]].to_numpy(dtype=float),
        "spike_tn": spike_tn_all[frames, :],
        "calcium_tn": calcium_tn_all[frames, :],
        "n_neurons": n_neurons,
    }

def _choose_figure4_schematic_cells(lam: np.ndarray, traces_tn: np.ndarray, n_cells: int = 5) -> np.ndarray:
    peak = np.nanmax(lam, axis=0)
    trace_span = np.nanpercentile(traces_tn, 98, axis=0) - np.nanpercentile(traces_tn, 2, axis=0)
    finite = np.isfinite(peak) & (peak > 0) & np.isfinite(trace_span) & (trace_span > 0)
    candidates = np.flatnonzero(finite)
    if candidates.size < n_cells:
        raise RuntimeError(f"not enough source-backed Figure 4A cells: {candidates.size}")
    top_n = min(max(n_cells * 12, 25), int(candidates.size))
    top = candidates[np.argsort(peak[candidates])[-top_n:]]
    peak_bins = np.nanargmax(np.where(np.isfinite(lam), lam, -np.inf), axis=0)
    top = top[np.argsort(peak_bins[top])]
    quantiles = np.linspace(0, top.size - 1, n_cells).round().astype(int)
    chosen = top[quantiles]
    _, unique_idx = np.unique(chosen, return_index=True)
    if unique_idx.size != n_cells:
        remaining = [idx for idx in top if idx not in set(chosen)]
        chosen = list(chosen[np.sort(unique_idx)])
        chosen.extend(remaining[: n_cells - len(chosen)])
        chosen = np.asarray(chosen, dtype=int)
    return chosen.astype(int)

def _normalize_trace_segment(segment_tn: np.ndarray) -> np.ndarray:
    lo = np.nanpercentile(segment_tn, 5, axis=0)
    hi = np.nanpercentile(segment_tn, 99, axis=0)
    denom = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    norm = (segment_tn - lo) / denom
    return np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.15)

def _figure4_mean_activity_maps(
    activity_tn: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    activity = np.asarray(activity_tn, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if activity.ndim != 2:
        raise ValueError(f"expected 2D activity matrix, got {activity.shape}")
    valid = (labels >= 0) & (labels < int(n_bins)) & np.all(np.isfinite(activity), axis=1)
    if not np.any(valid):
        raise RuntimeError("Figure 4A schematic has no valid activity samples for map construction")
    labels = labels[valid]
    activity = np.clip(np.nan_to_num(activity[valid], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    occupancy = np.bincount(labels, minlength=int(n_bins)).astype(float)
    maps = np.full((int(n_bins), activity.shape[1]), np.nan, dtype=float)
    keep = occupancy >= int(min_samples)
    for neuron in range(activity.shape[1]):
        sums = np.bincount(labels, weights=activity[:, neuron], minlength=int(n_bins)).astype(float)
        maps[keep, neuron] = sums[keep] / np.maximum(occupancy[keep], 1.0)
    return maps, occupancy

def _figure4_softmax(log_values: np.ndarray) -> np.ndarray:
    log_values = np.asarray(log_values, dtype=float)
    finite = np.isfinite(log_values)
    out = np.zeros_like(log_values, dtype=float)
    if not np.any(finite):
        return out
    shifted = log_values[finite] - float(np.nanmax(log_values[finite]))
    out[finite] = np.exp(np.clip(shifted, -80.0, 0.0))
    total = float(out.sum())
    if total > 0:
        out /= total
    return out

def _figure4_label_xy(label: int, n_bins_x: int) -> tuple[int, int]:
    label = int(label)
    n_bins_x = int(n_bins_x)
    return label % n_bins_x, label // n_bins_x

def _choose_diverse_figure4_schematic_cells(
    ranked_cells: np.ndarray,
    map_matrix: np.ndarray,
    peak_bins: np.ndarray,
    n_cells: int,
    n_bins_x: int,
) -> np.ndarray:
    ranked_cells = np.asarray(ranked_cells, dtype=int)
    map_matrix = np.asarray(map_matrix, dtype=float)
    peak_bins = np.asarray(peak_bins, dtype=int)
    if ranked_cells.size <= int(n_cells):
        return ranked_cells[: int(n_cells)]

    def _corr(a_idx: int, b_idx: int) -> float:
        a = np.nan_to_num(map_matrix[:, int(a_idx)], nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(map_matrix[:, int(b_idx)], nan=0.0, posinf=0.0, neginf=0.0)
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _peak_dist(a_idx: int, b_idx: int) -> float:
        ax, ay = _figure4_label_xy(int(peak_bins[int(a_idx)]), int(n_bins_x))
        bx, by = _figure4_label_xy(int(peak_bins[int(b_idx)]), int(n_bins_x))
        return float(math.hypot(ax - bx, ay - by))

    for min_dist, max_corr in ((13.0, 0.55), (10.0, 0.65), (7.0, 0.75), (4.0, 0.85)):
        chosen: list[int] = []
        for cell in ranked_cells.tolist():
            if all(_peak_dist(cell, prev) >= min_dist and _corr(cell, prev) <= max_corr for prev in chosen):
                chosen.append(int(cell))
            if len(chosen) == int(n_cells):
                return np.asarray(chosen, dtype=int)

    chosen = list(ranked_cells[: int(n_cells)])
    return np.asarray(chosen, dtype=int)

FIGURE4_SCHEMATIC_DATA_ROOT = REPO_ROOT / "docs/figure_illustration_sources"

def _build_figure4_schematic_source() -> dict[str, object]:
    # The method diagram needs only its displayed values, not a second set of
    # full neural sessions. The quantitative panels use their own decoder data.
    illustration_root = FIGURE4_SCHEMATIC_DATA_ROOT
    illustration_path = illustration_root / "spatial_decoder_example.npz"
    if illustration_path.is_file():
        metadata = json.loads((illustration_root / "spatial_decoder_example.json").read_text())
        with np.load(illustration_path, allow_pickle=False) as arrays:
            source = dict(metadata)
            source["template_maps"] = np.asarray(arrays["place_maps"], dtype=float)
            source["trace_segment"] = np.asarray(arrays["activity"], dtype=float)
            source["posterior"] = np.asarray(arrays["posterior"], dtype=float)
        source["decoder_pkl"] = REPO_ROOT / str(source["decoder_pkl"])
        return source
    from ca_analysis.config import DT
    from ca_analysis.similarity import compute_tuning_and_prior
    from ca_analysis.spatial import discretize_positions

    if not FIGURE4_SCHEMATIC_DECODER_PKL.exists():
        raise FileNotFoundError(FIGURE4_SCHEMATIC_DECODER_PKL)
    with open(FIGURE4_SCHEMATIC_DECODER_PKL, "rb") as f:
        decoder_payload = pickle.load(f)

    src_session, dst_session = FIGURE4_SCHEMATIC_PAIR
    src = _load_figure4_schematic_session(src_session)
    dst = _load_figure4_schematic_session(dst_session)
    train_ids = np.asarray(decoder_payload["train_ids"], dtype=int)
    test_ids = np.asarray(decoder_payload["test_ids"], dtype=int)
    valid_pair = (
        (train_ids >= 0)
        & (test_ids >= 0)
        & (train_ids < int(src["n_neurons"]))
        & (test_ids < int(dst["n_neurons"]))
    )
    train_ids = train_ids[valid_pair]
    test_ids = test_ids[valid_pair]
    if train_ids.size == 0:
        raise RuntimeError("Figure 4A decoder pair has no valid matched neurons")

    n_map_bins = int(FIGURE4_SCHEMATIC_MAP_BINS)
    n_map_total = n_map_bins * n_map_bins
    labels_train_45, _x_edges_45, _y_edges_45 = discretize_positions(
        np.asarray(src["positions"]),
        n_map_bins,
        n_map_bins,
    )
    source_calcium = np.asarray(src["calcium_tn"], dtype=float)[:, train_ids]
    target_traces = np.asarray(dst["calcium_tn"], dtype=float)[:, test_ids]
    calcium_maps, _map_occupancy = _figure4_mean_activity_maps(
        source_calcium,
        labels_train_45,
        n_bins=n_map_total,
        min_samples=3,
    )

    peak = np.nanpercentile(calcium_maps, 99.5, axis=0)
    trace_span = np.nanpercentile(target_traces, 99.5, axis=0) - np.nanpercentile(target_traces, 20, axis=0)
    finite = np.isfinite(peak) & (peak > 0) & np.isfinite(trace_span) & (trace_span > 0)
    candidates = np.flatnonzero(finite)
    n_display_cells = int(FIGURE4_SCHEMATIC_N_CELLS)
    if candidates.size < n_display_cells:
        chosen = _choose_figure4_schematic_cells(
            np.nan_to_num(calcium_maps, nan=0.0),
            target_traces,
            n_cells=n_display_cells,
        )
        candidates = chosen
    top_n = min(max(80, n_display_cells * 20), int(candidates.size))
    combined = peak[candidates] * np.sqrt(np.maximum(trace_span[candidates], 1e-12))
    top = candidates[np.argsort(combined)[-top_n:]]
    score_basis = _normalize_trace_segment(target_traces[:, top])
    score = np.nanpercentile(score_basis, 90, axis=1)
    segment_len = min(int(FIGURE4_SCHEMATIC_TRACE_POINTS), int(score.size))
    if segment_len < 20:
        raise RuntimeError("Figure 4A target activity segment is too short")
    saved_decoded_pos = np.asarray(decoder_payload.get("decoded_pos", []), dtype=int)
    saved_true_pos = np.asarray(decoder_payload.get("labels_pos", []), dtype=int)
    if score.size > segment_len:
        max_start = int(score.size - segment_len)
        later_min = int(round(score.size * 0.55))
        if saved_decoded_pos.size == score.size and saved_true_pos.size == score.size:
            saved_decoded_xy = np.asarray([_figure4_label_xy(label, 15) for label in saved_decoded_pos], dtype=float)
            saved_true_xy = np.asarray([_figure4_label_xy(label, 15) for label in saved_true_pos], dtype=float)
            saved_errors = np.sqrt(np.sum((saved_decoded_xy - saved_true_xy) ** 2, axis=1))
            candidate_centers = np.flatnonzero(
                (np.arange(score.size) >= later_min)
                & (np.arange(score.size) >= segment_len // 2)
                & (np.arange(score.size) <= score.size - max(1, segment_len // 2) - 1)
                & np.isfinite(saved_errors)
                & (saved_errors <= 2.5)
                & np.isfinite(score)
            )
            if candidate_centers.size:
                posterior_center = int(candidate_centers[np.argmax(score[candidate_centers])])
                start = int(np.clip(posterior_center - segment_len // 2, 0, max_start))
            else:
                csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(score, nan=0.0))])
                window_score = csum[segment_len:] - csum[:-segment_len]
                start_min = min(max_start, later_min)
                start = int(start_min + np.argmax(window_score[start_min:])) if start_min < window_score.size else max_start
        else:
            csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(score, nan=0.0))])
            window_score = csum[segment_len:] - csum[:-segment_len]
            start_min = min(max_start, later_min)
            start = int(start_min + np.argmax(window_score[start_min:])) if start_min < window_score.size else max_start
    else:
        start = 0
    stop = start + segment_len
    top_segment = _normalize_trace_segment(target_traces[start:stop, top])
    segment_amp = np.nanpercentile(top_segment, 99, axis=0) - np.nanpercentile(top_segment, 25, axis=0)
    selected_rank = segment_amp * np.sqrt(np.maximum(peak[top], 1e-12))
    peak_bins = np.nanargmax(np.where(np.isfinite(calcium_maps), calcium_maps, -np.inf), axis=0)
    forced_source_cells = tuple(int(v) for v in FIGURE4_SCHEMATIC_SOURCE_CELL_IDS)
    if forced_source_cells:
        if len(forced_source_cells) != n_display_cells:
            raise RuntimeError(
                "Figure 4A forced source-cell list must match "
                f"FIGURE4_SCHEMATIC_N_CELLS={n_display_cells}"
            )
        source_cell_lookup = {int(cell_id): idx for idx, cell_id in enumerate(train_ids.tolist())}
        missing = [cell_id for cell_id in forced_source_cells if cell_id not in source_cell_lookup]
        if missing:
            raise RuntimeError(f"Figure 4A forced source cells are not matched in the decoder CIMap: {missing}")
        chosen = np.asarray([source_cell_lookup[cell_id] for cell_id in forced_source_cells], dtype=int)
    else:
        ranked_top = top[np.argsort(selected_rank)[::-1]]
        chosen = _choose_diverse_figure4_schematic_cells(
            ranked_top,
            calcium_maps,
            peak_bins,
            n_display_cells,
            n_map_bins,
        )
        chosen = chosen[np.argsort(peak_bins[chosen])]
    trace_segment = _normalize_trace_segment(target_traces[start:stop, chosen])

    maps = []
    for idx in chosen:
        neuron_map = np.asarray(calcium_maps[:, idx], dtype=float).reshape(n_map_bins, n_map_bins)
        maps.append(neuron_map)

    decode_bins = 15
    labels_train_decode, _x_edges_decode, _y_edges_decode = discretize_positions(
        np.asarray(src["positions"]),
        decode_bins,
        decode_bins,
    )
    labels_dst_decode, _x_edges_dst_decode, _y_edges_dst_decode = discretize_positions(
        np.asarray(dst["positions"]),
        decode_bins,
        decode_bins,
    )
    spike_train_decode = np.asarray(src["spike_tn"], dtype=float)[:, train_ids]
    valid_decode_train = np.isfinite(labels_train_decode) & (np.asarray(labels_train_decode) >= 0)
    lam_decode, prior_decode = compute_tuning_and_prior(
        np.nan_to_num(spike_train_decode[valid_decode_train], nan=0.0, posinf=0.0, neginf=0.0),
        np.asarray(labels_train_decode, dtype=int)[valid_decode_train],
        n_bins=decode_bins * decode_bins,
        dt=float(DT),
        filter_ratemap=False,
        sigma=1.0,
        min_time=0.0,
        eps=1e-12,
        decode_head="none",
    )
    segment_activity = np.nan_to_num(
        np.asarray(dst["spike_tn"], dtype=float)[start:stop, :][:, test_ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    log_rate = np.log(lam_decode + 1e-12)
    log_offset = -float(DT) * np.sum(lam_decode, axis=1) + np.log(prior_decode + 1e-12)
    log_post_segment = (log_rate @ segment_activity.T) + log_offset[:, None]
    decoded_segment = np.nanargmax(log_post_segment, axis=0).astype(int)
    labels_segment = np.asarray(labels_dst_decode, dtype=int)[start:stop]
    decoded_xy = np.asarray([_figure4_label_xy(label, decode_bins) for label in decoded_segment], dtype=float)
    true_xy = np.asarray([_figure4_label_xy(label, decode_bins) for label in labels_segment], dtype=float)
    distances = np.sqrt(np.sum((decoded_xy - true_xy) ** 2, axis=1))
    log_max = np.nanmax(log_post_segment, axis=0)
    exp_sum = np.sum(np.exp(np.clip(log_post_segment - log_max[None, :], -80.0, 0.0)), axis=0)
    top_prob = 1.0 / np.maximum(exp_sum, 1e-12)
    valid_posterior_frames = np.flatnonzero(np.isfinite(top_prob) & np.isfinite(distances))
    close_frames = valid_posterior_frames[distances[valid_posterior_frames] <= 2.5]
    posterior_candidates = close_frames if close_frames.size else valid_posterior_frames
    if posterior_candidates.size:
        vline_idx = int(posterior_candidates[np.argmax(top_prob[posterior_candidates])])
    else:
        vline_idx = int(np.argmax(np.nanmean(trace_segment, axis=1)))
    posterior_index = int(np.clip(start + vline_idx, 0, target_traces.shape[0] - 1))
    posterior = _figure4_softmax(log_post_segment[:, vline_idx]).reshape(decode_bins, decode_bins)
    decoded_label_45 = int(np.nanargmax(posterior))
    true_label_45 = int(np.asarray(labels_dst_decode, dtype=int)[posterior_index])

    decoder_frames = np.asarray(decoder_payload.get("frames", []), dtype=int)
    dst_frame = int(dst["selected_df"].iloc[posterior_index]["frame"])
    decoder_frame_matches = np.flatnonzero(decoder_frames == dst_frame) if decoder_frames.size else np.array([], dtype=int)
    decoder_payload_index = int(decoder_frame_matches[0]) if decoder_frame_matches.size else int(
        min(posterior_index, max(len(decoder_frames) - 1, 0))
    )

    return {
        "src_session": src_session,
        "dst_session": dst_session,
        "decoder_pkl": FIGURE4_SCHEMATIC_DECODER_PKL,
        "n_matched_neurons": int(train_ids.size),
        "selected_train_ids": [int(train_ids[idx]) for idx in chosen],
        "selected_test_ids": [int(test_ids[idx]) for idx in chosen],
        "selected_match_indices": [int(idx) for idx in chosen],
        "template_maps": maps,
        "trace_segment": trace_segment,
        "trace_start_index": int(start),
        "trace_stop_index": int(stop),
        "vline_idx": int(vline_idx),
        "posterior": posterior,
        "posterior_index": int(posterior_index),
        "posterior_frame": int(dst_frame),
        "posterior_decoded_label": int(decoded_label_45),
        "posterior_true_label": int(true_label_45),
        "posterior_decoded_xy": [int(v) for v in _figure4_label_xy(decoded_label_45, decode_bins)],
        "posterior_true_xy": [int(v) for v in _figure4_label_xy(true_label_45, decode_bins)],
        "saved_decoder_payload_index": int(decoder_payload_index),
        "saved_decoder_decoded_pos_15": int(saved_decoded_pos[decoder_payload_index])
        if saved_decoded_pos.size
        else None,
        "saved_decoder_true_pos_15": int(saved_true_pos[decoder_payload_index]) if saved_true_pos.size else None,
        "n_bins_x": int(n_map_bins),
        "n_bins_y": int(n_map_bins),
        "posterior_n_bins_x": int(decode_bins),
        "posterior_n_bins_y": int(decode_bins),
        "dt": float(DT),
        "template_source": f"source-session C matrix averaged in {n_map_bins}x{n_map_bins} position bins",
        "activity_source": "target-session C activity traces at matched-neuron ids",
        "posterior_source": "real 15x15 Poisson decoder posterior recomputed from source-session S tuning and target-session S activity",
    }

def _render_figure4_method_schematic(size: tuple[float, float], style: PublicationStyle) -> tuple[plt.Figure, dict[str, object]]:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    source = _build_figure4_schematic_source()
    title_font = style.axis_label_fontsize * 0.72
    arrow_font = style.annotation_fontsize * 0.70
    tiny_font = style.tick_fontsize * 0.62
    maze = get_panel_maze(6)
    map_cmap = plt.get_cmap("magma").copy()
    map_cmap.set_bad("#f4f4f4")
    posterior_cmap = plt.get_cmap("magma").copy()
    posterior_cmap.set_bad("#f7f7f7")
    trace_y_centers = [0.825, 0.690, 0.500, 0.365, 0.230]

    def _add_real_map(
        bounds: tuple[float, float, float, float],
        image: np.ndarray,
        *,
        is_posterior: bool = False,
        zorder: int = 1,
    ) -> plt.Axes:
        from scipy.ndimage import gaussian_filter
        from matplotlib.patches import Rectangle

        map_ax = ax.inset_axes(bounds, transform=ax.transAxes)
        map_ax.set_zorder(zorder)
        arr = np.asarray(image, dtype=float)
        grid_bins_y, grid_bins_x = int(arr.shape[0]), int(arr.shape[1])
        wall_scale_x = float(grid_bins_x) / 15.0
        wall_scale_y = float(grid_bins_y) / 15.0
        if is_posterior:
            plot_arr = gaussian_filter(np.nan_to_num(arr, nan=0.0), sigma=0.55)
            vmax = float(np.nanmax(plot_arr)) if np.any(np.isfinite(plot_arr)) else 1.0
        else:
            finite = np.isfinite(arr)
            filled = np.where(finite, arr, 0.0)
            plot_arr = gaussian_filter(filled, sigma=1.0)
            positive = plot_arr[np.isfinite(plot_arr) & (plot_arr > 0)]
            vmax = float(np.nanpercentile(positive, 99.7)) if positive.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        map_ax.set_facecolor("#fcfcfc")
        map_ax.imshow(
            np.ma.masked_invalid(plot_arr),
            origin="upper",
            interpolation="nearest",
            cmap=posterior_cmap if is_posterior else map_cmap,
            vmin=0.0,
            vmax=vmax,
        )
        map_ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                grid_bins_x,
                grid_bins_y,
                fill=False,
                edgecolor="0.15",
                linewidth=0.45 if not is_posterior else 0.60,
                zorder=4,
            )
        )
        map_ax.plot(
            maze.wa[:, 0] * wall_scale_x,
            maze.wa[:, 1] * wall_scale_y,
            color="0.10" if not is_posterior else "0.15",
            linewidth=0.26 if not is_posterior else 0.42,
            alpha=0.95,
            zorder=3,
        )
        map_ax.set_xlim(-0.5, grid_bins_x - 0.5)
        map_ax.set_ylim(grid_bins_y - 0.5, -0.5)
        map_ax.set_aspect("equal")
        map_ax.axis("off")
        return map_ax

    template_maps = [np.asarray(m, dtype=float) for m in source["template_maps"]]
    map_geometry_options = {
        "low": (0.045, 0.230, 0.086, 0.052, [0.765, 0.620, 0.410, 0.265, 0.140]),
        "medium": (0.035, 0.230, 0.092, 0.075, [0.760, 0.615, 0.405, 0.260, 0.130]),
        "high": (0.012, 0.235, 0.128, 0.105, [0.760, 0.600, 0.420, 0.260, 0.100]),
        "wide": (0.005, 0.230, 0.106, 0.125, [0.748, 0.602, 0.392, 0.246, 0.112]),
    }
    map_geometry_key = os.environ.get("FIGURE4_SCHEMATIC_TILT", "high").strip().lower()
    if map_geometry_key not in map_geometry_options:
        raise ValueError(
            f"Unknown FIGURE4_SCHEMATIC_TILT={map_geometry_key!r}; "
            f"expected one of {sorted(map_geometry_options)}"
        )
    map_x0, map_w, map_h, map_shear, map_y0s = map_geometry_options[map_geometry_key]
    if map_geometry_key == "high":
        map_y0s = [float(y_center) - map_h / 2.0 for y_center in trace_y_centers]

    def _draw_slanted_template_map(y0: float, neuron_map: np.ndarray, *, zorder: int) -> None:
        from matplotlib.patches import Polygon

        arr = np.asarray(neuron_map, dtype=float)
        finite = np.isfinite(arr)
        if np.any(finite):
            baseline = float(np.nanpercentile(arr[finite], 55.0))
            image = np.where(finite, np.clip(arr - baseline, 0.0, None), 0.0)
            vmax = float(np.nanpercentile(image[image > 0], 99.5)) if np.any(image > 0) else 1.0
        else:
            image = np.zeros_like(arr, dtype=float)
            vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
        pixels_per_bin = 8
        tile = np.repeat(np.repeat(image, pixels_per_bin, axis=0), pixels_per_bin, axis=1)
        sheet = Polygon(
            [
                (map_x0, y0),
                (map_x0 + map_w, y0),
                (map_x0 + map_w + map_shear, y0 + map_h),
                (map_x0 + map_shear, y0 + map_h),
            ],
            closed=True,
            facecolor="#020204",
            edgecolor="none",
            antialiased=True,
            transform=ax.transAxes,
            zorder=zorder - 0.05,
        )
        ax.add_patch(sheet)
        slant = math.degrees(math.atan2(map_shear, map_h))
        transform = matplotlib.transforms.Affine2D().skew_deg(slant, 0).translate(map_x0, y0) + ax.transAxes
        ax.imshow(
            tile,
            extent=(0.0, map_w, 0.0, map_h),
            origin="upper",
            cmap=map_cmap,
            norm=norm,
            interpolation="nearest",
            resample=False,
            aspect="auto",
            transform=transform,
            zorder=zorder,
            clip_on=False,
        )
    for layer, (y0, neuron_map) in enumerate(zip(map_y0s, template_maps, strict=True)):
        _draw_slanted_template_map(y0, neuron_map, zorder=1 + layer * 0.1)
    stack_center_x = map_x0 + map_w / 2.0 + map_shear / 2.0
    map_gap_mid_y = (map_y0s[1] + map_y0s[2] + map_h) / 2.0
    gap_dot_y = map_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, stack_center_x),
        gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    bottom_title_y = 0.025
    ax.text(stack_center_x, bottom_title_y, "Day i place maps", ha="center", va="bottom", fontsize=title_font)

    trace_left, trace_right = 0.405, 0.635
    trace_x = np.linspace(trace_left, trace_right, int(FIGURE4_SCHEMATIC_TRACE_POINTS))
    trace_segment = np.asarray(source["trace_segment"], dtype=float)
    if trace_segment.shape[0] != trace_x.size:
        trace_x = np.linspace(trace_left, trace_right, trace_segment.shape[0])
    vline_x = float(np.interp(int(source["vline_idx"]), [0, max(trace_segment.shape[0] - 1, 1)], [trace_x[0], trace_x[-1]]))
    frame_dx = (trace_right - trace_left) / max(trace_segment.shape[0] - 1, 1)
    visible_half_width = max(frame_dx / 2.0, 0.0014)
    trace_band_ymin = min(y - 0.050 + 0.010 for y in trace_y_centers) - 0.006
    trace_band_ymax = max(y - 0.050 + 0.010 + 0.073 for y in trace_y_centers) + 0.006
    ax.axvspan(
        max(trace_left, vline_x - visible_half_width),
        min(trace_right, vline_x + visible_half_width),
        ymin=max(0.0, trace_band_ymin),
        ymax=min(1.0, trace_band_ymax),
        color="#e5e5e5",
        alpha=0.85,
        linewidth=0,
        zorder=0,
    )
    for index, y_center in enumerate(trace_y_centers):
        baseline = y_center - 0.050
        trace = trace_segment[:, index]
        trace = 0.010 + 0.073 * np.clip(trace, 0.0, 1.0)
        ax.plot(trace_x, baseline + trace, color="#202020", linewidth=0.70)
    trace_gap_mid_y = (trace_y_centers[1] + trace_y_centers[2]) / 2.0
    trace_gap_dot_y = trace_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, (trace_left + trace_right) / 2.0),
        trace_gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    scale_bar_frames = 10.0 / max(float(source["dt"]), 1e-12)
    scale_bar_len = (scale_bar_frames / max(trace_segment.shape[0] - 1, 1)) * (trace_right - trace_left)
    scale_bar_x0 = trace_left
    scale_bar_y = 0.145
    ax.plot(
        [scale_bar_x0, scale_bar_x0 + scale_bar_len],
        [scale_bar_y, scale_bar_y],
        color="#202020",
        linewidth=0.90,
        solid_capstyle="butt",
    )
    ax.text(
        scale_bar_x0 + scale_bar_len / 2.0,
        scale_bar_y - 0.030,
        "10 s",
        ha="center",
        va="top",
        fontsize=tiny_font,
        color="#202020",
    )
    ax.text(
        (trace_left + trace_right) / 2.0,
        bottom_title_y,
        "Day i + 90 activity",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )

    ax.text(0.377, 0.500, "x", ha="center", va="center", fontsize=style.axis_label_fontsize * 1.00)
    posterior = np.asarray(source["posterior"], dtype=float)
    posterior_h = 0.710
    posterior_w = posterior_h * float(size[1]) / float(size[0])
    posterior_x = 0.675
    posterior_y = 0.150
    arrow_start_x = trace_right + 0.010
    arrow_end_x = posterior_x - 0.010
    ax.plot(
        [arrow_start_x, arrow_end_x],
        [0.500, 0.500],
        color="0.25",
        linewidth=0.65,
        marker=">",
        markevery=[-1],
        markersize=1.25,
        markeredgewidth=0,
        markerfacecolor="0.25",
        solid_capstyle="butt",
        zorder=3,
    )

    posterior_ax = _add_real_map((posterior_x, posterior_y, posterior_w, posterior_h), posterior, is_posterior=True)
    decoded_x, decoded_y = [int(v) for v in source["posterior_decoded_xy"]]
    true_x, true_y = [int(v) for v in source["posterior_true_xy"]]
    posterior_ax.scatter(
        [decoded_x],
        [decoded_y],
        s=2.8,
        marker="x",
        linewidths=0.32,
        color="#00A6D6",
        zorder=5,
    )
    posterior_ax.scatter(
        [true_x],
        [true_y],
        s=2.8,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.32,
        zorder=5,
    )
    cbar_x = posterior_x + posterior_w * 0.18
    cbar_y = posterior_y + posterior_h + 0.025
    cbar_w = posterior_w * 0.64
    cbar_h = 0.014
    posterior_cbar_ax = ax.inset_axes((cbar_x, cbar_y, cbar_w, cbar_h), transform=ax.transAxes)
    posterior_cbar_ax.set_zorder(6)
    posterior_cbar_ax.imshow(
        np.linspace(0.0, 1.0, 128)[None, :],
        aspect="auto",
        cmap=posterior_cmap,
        interpolation="nearest",
    )
    posterior_cbar_ax.set_axis_off()
    ax.text(
        cbar_x + cbar_w / 2.0,
        cbar_y + cbar_h + 0.006,
        "P(position)",
        fontsize=tiny_font * 0.92,
        color="#202020",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        zorder=6,
    )
    legend_y = posterior_y - 0.035
    decoded_legend_x = posterior_x + 0.030
    true_legend_x = posterior_x + 0.150
    ax.scatter(
        [decoded_legend_x],
        [legend_y],
        s=4.0,
        marker="x",
        linewidths=0.42,
        color="#00A6D6",
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        decoded_legend_x + 0.012,
        legend_y,
        "Decoded",
        fontsize=tiny_font,
        color="#00A6D6",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.scatter(
        [true_legend_x],
        [legend_y],
        s=4.0,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.42,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        true_legend_x + 0.012,
        legend_y,
        "Real",
        fontsize=tiny_font,
        color="#F04E37",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.text(
        posterior_x + posterior_w / 2.0,
        bottom_title_y,
        "Decoded position",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )
    meta = {
        "src_session": str(source["src_session"]),
        "dst_session": str(source["dst_session"]),
        "decoder_pkl": _rel(Path(source["decoder_pkl"])),
        "n_matched_neurons": int(source["n_matched_neurons"]),
        "selected_train_ids": list(source["selected_train_ids"]),
        "selected_test_ids": list(source["selected_test_ids"]),
        "selected_match_indices": list(source["selected_match_indices"]),
        "trace_start_index": int(source["trace_start_index"]),
        "trace_stop_index": int(source["trace_stop_index"]),
        "vline_idx": int(source["vline_idx"]),
        "posterior_index": int(source["posterior_index"]),
        "posterior_frame": int(source["posterior_frame"]),
        "posterior_decoded_label": int(source["posterior_decoded_label"]),
        "posterior_true_label": int(source["posterior_true_label"]),
        "posterior_decoded_xy": list(source["posterior_decoded_xy"]),
        "posterior_true_xy": list(source["posterior_true_xy"]),
        "n_bins_x": int(source["n_bins_x"]),
        "n_bins_y": int(source["n_bins_y"]),
        "posterior_n_bins_x": int(source["posterior_n_bins_x"]),
        "posterior_n_bins_y": int(source["posterior_n_bins_y"]),
        "dt": float(source["dt"]),
        "template_source": str(source["template_source"]),
        "activity_source": str(source["activity_source"]),
        "posterior_source": str(source["posterior_source"]),
    }
    return fig, meta

def _svg_defs(root: ET.Element) -> ET.Element:
    for child in root:
        if child.tag == _svg_tag("defs"):
            return child
    return ET.SubElement(root, _svg_tag("defs"))

def _add_svg_panel_cropped_absolute(
    root: ET.Element,
    panel_svg: dict[str, object],
    *,
    panel_id: str,
    slot_pt: tuple[float, float, float, float],
    src_rect: dict[str, int],
) -> dict[str, float]:
    """Place a source-rendered panel in absolute figure coordinates with PPT-style clipping."""

    slot_x, slot_y, slot_w, slot_h = (float(value) for value in slot_pt)
    view_w = float(panel_svg["viewbox_width"])
    view_h = float(panel_svg["viewbox_height"])
    left = float(src_rect.get("l", 0)) / 100000.0
    top = float(src_rect.get("t", 0)) / 100000.0
    right = float(src_rect.get("r", 0)) / 100000.0
    bottom = float(src_rect.get("b", 0)) / 100000.0
    crop_x = view_w * left
    crop_y = view_h * top
    crop_w = view_w * max(1.0 - left - right, 1e-9)
    crop_h = view_h * max(1.0 - top - bottom, 1e-9)
    scale_x = slot_w / crop_w
    scale_y = slot_h / crop_h
    clip_id = f"clip_{panel_id}"
    clip = ET.SubElement(
        _svg_defs(root),
        _svg_tag("clipPath"),
        {"id": clip_id, "clipPathUnits": "userSpaceOnUse"},
    )
    ET.SubElement(
        clip,
        _svg_tag("rect"),
        {
            "x": f"{slot_x:.6f}",
            "y": f"{slot_y:.6f}",
            "width": f"{slot_w:.6f}",
            "height": f"{slot_h:.6f}",
        },
    )
    outer = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": f"panel_{panel_id}",
            "clip-path": f"url(#{clip_id})",
        },
    )
    group = ET.SubElement(
        outer,
        _svg_tag("g"),
        {
            "transform": (
                f"translate({slot_x - crop_x * scale_x:.6f} {slot_y - crop_y * scale_y:.6f}) "
                f"scale({scale_x:.8f} {scale_y:.8f})"
            ),
        },
    )
    for child in panel_svg["children"]:
        group.append(child)
    return {
        "slot_x_pt": float(slot_x),
        "slot_y_pt": float(slot_y),
        "slot_width_pt": float(slot_w),
        "slot_height_pt": float(slot_h),
        "source_viewbox_width_pt": float(view_w),
        "source_viewbox_height_pt": float(view_h),
        "crop_x_pt": float(crop_x),
        "crop_y_pt": float(crop_y),
        "crop_width_pt": float(crop_w),
        "crop_height_pt": float(crop_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "rendered_width": float(slot_w),
        "rendered_height": float(slot_h),
        "scale": float(scale_x),
    }
