from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, label as ndi_label
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sps

from .config import FPS, N_BINS_X, N_BINS_Y
from .maze import NewMaze


def discretize_positions(positions, n_bins_x=N_BINS_X, n_bins_y=N_BINS_Y):
    """
    Clip positions to [0, 1] and discretize to linear bin labels.

    Supports:
    - 2D positions with shape (T, 2)
    - 1D positions with shape (T,) or (T, 1), using a degenerate y-bin
    """
    positions = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values")
    if positions.ndim == 1:
        x = np.clip(positions, 0.0, 1.0)
        y = np.zeros_like(x)
    elif positions.ndim == 2:
        clipped = np.clip(positions, 0.0, 1.0)
        x = clipped[:, 0]
        y = clipped[:, 1] if clipped.shape[1] > 1 else np.zeros_like(x)
    else:
        raise ValueError('positions must be 1D or 2D')

    x_edges = np.linspace(0.0, 1.0, n_bins_x + 1)
    y_edges = np.linspace(0.0, 1.0, n_bins_y + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins_x - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins_y - 1)
    labels = ix + iy * n_bins_x
    return labels, x_edges, y_edges


def build_count_matrix(spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids=None):
    """
    Build N x T windowed count matrix by summing raw counts over each frame window.
    """
    if neuron_ids is None:
        neuron_ids = np.arange(spike_counts.shape[1])
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    frames = np.asarray(frames, dtype=int).ravel()
    if frames.size == 0:
        return np.zeros((len(neuron_ids), 0), dtype=float)
    raw_counts = np.asarray(raw_counts, dtype=float)
    if raw_counts.ndim != 2:
        raise ValueError("raw_counts must be 2D.")
    frame_col = selected_df["frame"].to_numpy(dtype=int)
    max_frame = int(np.nanmax(frame_col[frames])) if frames.size else -1
    n_neurons_total = int(spike_counts.shape[1])
    if raw_counts.shape[1] == n_neurons_total and raw_counts.shape[0] > max_frame:
        raw_ft = raw_counts
    elif raw_counts.shape[0] == n_neurons_total and raw_counts.shape[1] > max_frame:
        raw_ft = raw_counts.T
    elif raw_counts.shape[1] == n_neurons_total:
        raw_ft = raw_counts
    elif raw_counts.shape[0] == n_neurons_total:
        raw_ft = raw_counts.T
    else:
        raise ValueError(
            f"raw_counts shape {raw_counts.shape} is incompatible with spike_counts shape {spike_counts.shape}."
        )
    if max_frame >= raw_ft.shape[0]:
        raise ValueError(
            f"selected_df frame index {max_frame} exceeds raw_counts length {raw_ft.shape[0]} after orientation check."
        )
    tw = int(time_window)
    if tw <= 0:
        rf = frame_col[frames]
        return np.nan_to_num(raw_ft[rf][:, neuron_ids], nan=0.0).T

    raw_sel = np.nan_to_num(raw_ft[:, neuron_ids], nan=0.0)
    csum = np.vstack([np.zeros((1, raw_sel.shape[1]), dtype=float), np.cumsum(raw_sel, axis=0)])
    rf = frame_col[frames]
    starts = np.clip(rf - tw, 0, raw_sel.shape[0])
    ends = np.clip(rf + 1, 0, raw_sel.shape[0])
    out = csum[ends] - csum[starts]
    return out.T


def generate_level_list(ma_ce, grid_size=N_BINS_X):
    """
    Generate eight hierarchical level masks over the maze occupancy map.
    """
    n_cells = len(ma_ce)
    levels = [np.zeros(n_cells, dtype=int) for _ in range(8)]

    for i in range(0, 7):
        levels[0][ma_ce[(i, 7)]] = 1

    for j in range(4, 11):
        levels[1][ma_ce[(7, j)]] = 1

    for i in range(4, 11):
        for j in [3, 11]:
            levels[2][ma_ce[(i, j)]] = 1

    for i in [3, 11]:
        for j in [3, 11]:
            for dj in [0, -1, 1]:
                idx = ma_ce.get((i, j + dj))
                if idx is not None:
                    levels[3][idx] = 1

    for i in [3, 11]:
        for j in range(1, 14, 4):
            for di in [0, -1, 1]:
                idx = ma_ce.get((i + di, j))
                if idx is not None:
                    levels[4][idx] = 1

    for i in range(1, grid_size, 4):
        for j in range(1, grid_size, 4):
            levels[5][ma_ce[(i, j)]] = 1

    for i in range(1, grid_size, 4):
        for j in range(0, grid_size, 2):
            levels[6][ma_ce[(i, j)]] = 1

    for i in range(0, grid_size, 2):
        for j in range(0, grid_size, 2):
            levels[7][ma_ce[(i, j)]] = 1

    return levels


LEVEL_DEFINITION_GEOMETRIC8 = "geometric8"
LEVEL_DEFINITION_TREE_DEPTH7 = "tree_depth7"


def normalize_level_definition(level_definition: str | None = None) -> str:
    """Normalize accepted names for maze level definitions."""
    text = str(level_definition or LEVEL_DEFINITION_GEOMETRIC8).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    aliases = {
        "geometric": LEVEL_DEFINITION_GEOMETRIC8,
        "geometric8": LEVEL_DEFINITION_GEOMETRIC8,
        "old_geometric": LEVEL_DEFINITION_GEOMETRIC8,
        "old_geometric8": LEVEL_DEFINITION_GEOMETRIC8,
        "8": LEVEL_DEFINITION_GEOMETRIC8,
        "8_level": LEVEL_DEFINITION_GEOMETRIC8,
        "level8": LEVEL_DEFINITION_GEOMETRIC8,
        "tree_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "tree_depth7": LEVEL_DEFINITION_TREE_DEPTH7,
        "true_tree_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "true_depth": LEVEL_DEFINITION_TREE_DEPTH7,
        "7": LEVEL_DEFINITION_TREE_DEPTH7,
        "7_level": LEVEL_DEFINITION_TREE_DEPTH7,
        "level7": LEVEL_DEFINITION_TREE_DEPTH7,
    }
    if text not in aliases:
        raise ValueError(
            "level_definition must be one of "
            f"{LEVEL_DEFINITION_GEOMETRIC8!r} or {LEVEL_DEFINITION_TREE_DEPTH7!r}; "
            f"got {level_definition!r}."
        )
    return aliases[text]


def generate_tree_depth_level_list(maze=None):
    """
    Generate seven true tree-depth masks over the maze occupancy map.

    The depth is the binary-tree run depth for each maze cell's owning run.
    For ``NewMaze(6)`` this covers all 176 maze cells with counts
    ``8, 8, 16, 16, 32, 32, 64`` for depths ``0..6``.
    """
    ma = NewMaze(6) if maze is None else maze
    run_parent = np.asarray(ma.pa, dtype=int).ravel()
    run_depth = np.full(run_parent.shape[0], -1, dtype=int)
    for run_idx in range(run_parent.shape[0]):
        depth = 0
        cur = int(run_idx)
        guard = 0
        while cur >= 0 and guard <= run_parent.shape[0] + 2:
            cur = int(run_parent[cur])
            if cur >= 0:
                depth += 1
            guard += 1
        run_depth[run_idx] = int(depth)

    n_levels = int(np.nanmax(run_depth)) + 1
    levels = [np.zeros(len(ma.xc), dtype=int) for _ in range(n_levels)]
    for cell_id, run_idx in enumerate(np.asarray(ma.rc, dtype=int).ravel()):
        if 0 <= int(run_idx) < run_depth.shape[0]:
            levels[int(run_depth[int(run_idx)])][int(cell_id)] = 1
    return levels


def get_level_list(level_definition: str | None = None, *, maze=None, ma_ce=None, grid_size=N_BINS_X):
    """Return level masks for the requested maze level definition."""
    definition = normalize_level_definition(level_definition)
    if definition == LEVEL_DEFINITION_GEOMETRIC8:
        if ma_ce is None:
            ma_ce = (NewMaze(6) if maze is None else maze).ce
        return generate_level_list(ma_ce, grid_size=grid_size)
    return generate_tree_depth_level_list(NewMaze(6) if maze is None else maze)


def assign_levels_for_definition(
    selected_df,
    level_definition: str | None = None,
    *,
    maze=None,
    frame_col="frame",
    bx_col="mass_bx",
    by_col="mass_by",
    grid_size=N_BINS_X,
):
    """Assign levels using either the geometric-8 or true-depth-7 definition."""
    ma = NewMaze(6) if maze is None else maze
    level_list = get_level_list(level_definition, maze=ma, ma_ce=ma.ce, grid_size=grid_size)
    return assign_levels(
        selected_df,
        ma.ce,
        level_list,
        frame_col=frame_col,
        bx_col=bx_col,
        by_col=by_col,
    )


def assign_levels(selected_df, ma_ce, level_list, frame_col='frame', bx_col='mass_bx', by_col='mass_by'):
    """
    Assign a level index to each row from occupied maze cell coordinates.
    """
    frames = selected_df[frame_col].to_numpy(dtype=int)
    bx = selected_df[bx_col].to_numpy(dtype=int)
    by = selected_df[by_col].to_numpy(dtype=int)
    if frames.size == 0:
        return []

    # Match legacy dict semantics (last duplicate frame wins).
    frame_positions = {}
    for f, bxx, byy in zip(frames.tolist(), bx.tolist(), by.tolist()):
        frame_positions[int(f)] = (int(bxx), int(byy))
    key_frames = np.array(sorted(frame_positions.keys()), dtype=int)
    key_cid = np.array(
        [ma_ce.get((int(frame_positions[f][1]), int(frame_positions[f][0])), -1) for f in key_frames],
        dtype=int,
    )

    valid_idx = np.where(key_cid >= 0, np.arange(key_cid.size, dtype=int), -1)
    last_valid_idx = np.maximum.accumulate(valid_idx)
    pos = np.searchsorted(key_frames, frames, side="right") - 1
    cid_rows = np.full(frames.size, -1, dtype=int)
    ok = pos >= 0
    if np.any(ok):
        lv = last_valid_idx[pos[ok]]
        valid = lv >= 0
        cid_rows_ok = np.full(lv.shape, -1, dtype=int)
        cid_rows_ok[valid] = key_cid[lv[valid]]
        cid_rows[ok] = cid_rows_ok

    level_by_cell = np.full(len(level_list[0]), -1, dtype=int)
    for idx, mask in enumerate(level_list):
        level_by_cell[np.asarray(mask) == 1] = idx
    level_ids = np.where(cid_rows >= 0, level_by_cell[cid_rows], -1)
    return [None if lv < 0 else int(lv) for lv in level_ids.tolist()]


def arena_binned(x, y, xdim, ydim, pix_cm, pf_cm):
    """
    Bin x/y trajectories into spatial bins and return linearized indices + occupancy.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    d_x = xdim[0] - xdim[1]
    d_y = ydim[0] - ydim[1]
    nBnx = int(d_x / (pix_cm * pf_cm))
    nBny = int(d_y / (pix_cm * pf_cm))
    bn_x = np.linspace(xdim[0], xdim[1], nBnx + 1)
    bn_y = np.linspace(ydim[0], ydim[1], nBny + 1)
    bx = ((x - bn_x[0]) / (bn_x[1] - bn_x[0])).astype(int)
    by = ((y - bn_y[0]) / (bn_y[1] - bn_y[0])).astype(int)
    s = np.vstack((bx, by))
    lin_s = np.ravel_multi_index(s, (nBnx, nBny))
    occ_map = sps.csr_matrix(
        (np.ones(len(bx)), (bx, by)),
        shape=(nBnx, nBny),
        dtype=float,
    ).todense()
    return lin_s.astype(float), occ_map, nBnx, nBny, bx, by


def _select_interval_rows(selected_df, start_frame, end_frame, frame_col="frame"):
    df = selected_df.copy()
    if frame_col in df.columns:
        start = float(start_frame) if np.isfinite(start_frame) else -np.inf
        end = float(end_frame) if np.isfinite(end_frame) else np.inf
        df = df[(df[frame_col] > start) & (df[frame_col] < end)].copy()
    return df


def _coerce_signal_matrix(signal, frame_idx):
    arr = np.asarray(signal, dtype=float)
    frame_idx = np.asarray(frame_idx, dtype=int)
    if arr.ndim == 1:
        idx = np.clip(frame_idx, 0, max(0, arr.shape[0] - 1))
        return arr[idx][:, None]
    if arr.ndim != 2:
        raise ValueError("signal must be 1D or 2D")
    max_frame = int(np.nanmax(frame_idx)) if frame_idx.size else -1
    if arr.shape[0] > max_frame:
        idx = np.clip(frame_idx, 0, max(0, arr.shape[0] - 1))
        return arr[idx]
    if arr.shape[1] > max_frame:
        idx = np.clip(frame_idx, 0, max(0, arr.shape[1] - 1))
        return arr.T[idx]
    raise ValueError("signal does not align to frame indices")


def compute_place_maps_interval(
    selected_df,
    signal,
    start_frame=0,
    end_frame=np.inf,
    *,
    frame_col="frame",
    x_col="x",
    y_col="y",
    n_bins_x=45,
    n_bins_y=45,
    invert_xy=True,
):
    """
    Vectorized place-map computation for one or many neurons over a frame interval.

    Returns:
    - rate_sums: (n_bins_x, n_bins_y, n_neurons)
    - place_maps: (n_bins_x, n_bins_y, n_neurons)
    - occupancy: (n_bins_x, n_bins_y)
    """
    df = _select_interval_rows(selected_df, start_frame, end_frame, frame_col=frame_col)
    n_bins_x = int(n_bins_x)
    n_bins_y = int(n_bins_y)
    if df.empty:
        z3 = np.zeros((n_bins_x, n_bins_y, 1), dtype=float)
        z2 = np.zeros((n_bins_x, n_bins_y), dtype=float)
        return z3.copy(), z3, z2

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"selected_df must include '{x_col}' and '{y_col}'")
    x_raw = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    if invert_xy:
        # Legacy semantics: x <- 1-y, y <- 1-x
        x_use = 1.0 - y_raw
        y_use = 1.0 - x_raw
    else:
        x_use = x_raw
        y_use = y_raw
    x_use = np.clip(x_use, 0.0, 1.0)
    y_use = np.clip(y_use, 0.0, 1.0)

    bx = np.floor((1.0 - x_use) * n_bins_x).astype(int)
    by = np.floor((1.0 - y_use) * n_bins_y).astype(int)
    bx = np.clip(bx, 0, n_bins_x - 1)
    by = np.clip(by, 0, n_bins_y - 1)
    bins = bx * n_bins_y + by
    K = n_bins_x * n_bins_y

    frame_idx = (
        np.asarray(df[frame_col], dtype=int)
        if frame_col in df.columns
        else np.arange(len(df), dtype=int)
    )
    vals = _coerce_signal_matrix(signal, frame_idx)
    vals = np.nan_to_num(vals, nan=0.0)
    N = vals.shape[1]

    occupancy = np.bincount(bins, minlength=K).astype(float)
    rate_sums_flat = np.zeros((K, N), dtype=float)
    np.add.at(rate_sums_flat, bins, vals)
    place_flat = np.divide(
        rate_sums_flat,
        occupancy[:, None],
        out=np.full((K, N), np.nan, dtype=float),
        where=occupancy[:, None] > 0,
    )
    rate_sums = rate_sums_flat.reshape(n_bins_x, n_bins_y, N)
    place_maps = place_flat.reshape(n_bins_x, n_bins_y, N)
    occ2d = occupancy.reshape(n_bins_x, n_bins_y)
    return rate_sums, place_maps, occ2d


def compute_directional_place_maps_interval(
    selected_df,
    signal,
    start_frame=0,
    end_frame=np.inf,
    *,
    frame_col="frame",
    head_col="head_cat",
    n_head_bins=8,
    n_bins_x=N_BINS_X,
    n_bins_y=N_BINS_Y,
):
    """
    Vectorized directional place maps over a frame interval.

    Returns:
    - rate_sums: (n_head_bins, n_bins_y, n_bins_x)
    - place_maps: (n_head_bins, n_bins_y, n_bins_x)
    - occupancy: (n_head_bins, n_bins_y, n_bins_x)
    """
    df = _select_interval_rows(selected_df, start_frame, end_frame, frame_col=frame_col)
    n_head_bins = int(n_head_bins)
    n_bins_x = int(n_bins_x)
    n_bins_y = int(n_bins_y)
    if df.empty:
        z = np.zeros((n_head_bins, n_bins_y, n_bins_x), dtype=float)
        return z.copy(), z.copy(), z

    frame_idx = (
        np.asarray(df[frame_col], dtype=int)
        if frame_col in df.columns
        else np.arange(len(df), dtype=int)
    )
    vals = np.asarray(_coerce_signal_matrix(signal, frame_idx), dtype=float).ravel()
    vals = np.nan_to_num(vals, nan=0.0)

    if {"mass_bx", "mass_by"}.issubset(df.columns):
        bx = np.floor(pd.to_numeric(df["mass_bx"], errors="coerce").to_numpy(dtype=float)).astype(int)
        by = np.floor(pd.to_numeric(df["mass_by"], errors="coerce").to_numpy(dtype=float)).astype(int)
    elif {"x", "y"}.issubset(df.columns):
        bx = np.floor(pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float) * n_bins_x).astype(int)
        by = np.floor((1.0 - pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)) * n_bins_y).astype(int)
    else:
        raise ValueError("selected_df needs mass_bx/mass_by or x/y for directional place maps")
    bx = np.clip(bx, 0, n_bins_x - 1)
    by = np.clip(by, 0, n_bins_y - 1)

    if head_col in df.columns:
        hk = pd.to_numeric(df[head_col], errors="coerce").to_numpy(dtype=float)
        hk = np.nan_to_num(hk, nan=0.0).astype(int)
    else:
        hk = np.zeros(len(df), dtype=int)
    hk = np.clip(hk, 0, n_head_bins - 1)

    occ = np.zeros((n_head_bins, n_bins_y, n_bins_x), dtype=float)
    rate = np.zeros((n_head_bins, n_bins_y, n_bins_x), dtype=float)
    np.add.at(occ, (hk, by, bx), 1.0)
    np.add.at(rate, (hk, by, bx), vals)
    place = np.divide(rate, occ, out=np.zeros_like(rate), where=occ > 0)
    return rate, place, occ


def generate_placeMap_with_start_and_end_time_single_cell(selected_df, S_trans, start_frame, end_frame):
    """
    Backward-compatible single-neuron API, routed to vectorized map builder.

    The original implementation is archived under:
    `ca_analysis.archive.place_map_legacy.generate_placeMap_with_start_and_end_time_single_cell_legacy`.
    """
    _, place_maps, _ = compute_place_maps_interval(
        selected_df,
        S_trans,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_col="frame",
        x_col="x",
        y_col="y",
        n_bins_x=45,
        n_bins_y=45,
        invert_xy=True,
    )
    if place_maps.shape[2] == 0:
        return np.zeros((45, 45), dtype=float)
    return np.asarray(place_maps[:, :, 0], dtype=float)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    keep = np.isfinite(pvals)
    if not np.any(keep):
        return qvals
    p = pvals[keep]
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(p)
    q[order] = q_ranked
    qvals[keep] = q
    return qvals


def _safe_corr(a, b, mask=None, min_n=3, eps=1e-12):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if mask is None:
        good = np.isfinite(a) & np.isfinite(b)
    else:
        good = np.asarray(mask, dtype=bool).ravel() & np.isfinite(a) & np.isfinite(b)
    if int(np.sum(good)) < int(min_n):
        return np.nan
    aa = a[good]
    bb = b[good]
    if np.nanstd(aa) < eps or np.nanstd(bb) < eps:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def _first_sustained_hit(y, threshold, min_consecutive=2):
    y = np.asarray(y, dtype=float)
    c = int(max(1, min_consecutive))
    if y.size < c:
        return np.nan
    for i in range(0, y.size - c + 1):
        seg = y[i:i + c]
        if np.all(np.isfinite(seg)) and np.all(seg >= float(threshold)):
            return float(i + 1)
    return np.nan


def _compute_bin_level_lookup(levels, bins, K):
    out = np.full(int(K), np.nan, dtype=float)
    levels = np.asarray(levels, dtype=float)
    bins = np.asarray(bins, dtype=int)
    finite = np.isfinite(levels)
    for b in range(int(K)):
        vals = levels[(bins == b) & finite]
        if vals.size:
            u, c = np.unique(vals.astype(int), return_counts=True)
            out[b] = int(u[np.argmax(c)])
    return out


def _to_tn_matrix(S, *, n_rows=None, frames=None):
    arr = np.asarray(S, dtype=float)
    if arr.ndim != 2:
        raise ValueError('S must be 2D')
    if n_rows is not None:
        n_rows = int(n_rows)
        if arr.shape[0] == n_rows:
            return arr
        if arr.shape[1] == n_rows:
            return arr.T
    if frames is not None and len(frames):
        fr = np.asarray(frames, dtype=int)
        mf = int(np.nanmax(fr))
        if mf < arr.shape[0]:
            return arr[fr]
        if mf < arr.shape[1]:
            return arr.T[fr]
    return arr if arr.shape[0] >= arr.shape[1] else arr.T


def _normalize_xy_for_bins(df):
    if {'mass_x', 'mass_y'}.issubset(df.columns):
        xy = df[['mass_x', 'mass_y']].to_numpy(dtype=float)
    elif {'x', 'y'}.issubset(df.columns):
        xy = df[['x', 'y']].to_numpy(dtype=float)
    else:
        raise ValueError('selected_df requires mass_x/mass_y or x/y columns')
    mins = np.nanmin(xy, axis=0, keepdims=True)
    maxs = np.nanmax(xy, axis=0, keepdims=True)
    den = maxs - mins
    den[den == 0] = 1.0
    xy01 = (xy - mins) / den
    return np.clip(xy01, 0.0, 1.0)


def build_session_pack(df, S, n_bins_x, n_bins_y, min_frames_per_visit=5, fps=FPS,
                       bin_col_x=None, bin_col_y=None):
    out = df.copy()
    sort_col = 'maze_time' if 'maze_time' in out.columns else ('frame' if 'frame' in out.columns else None)
    if sort_col is not None:
        out = out.sort_values(sort_col).reset_index(drop=True)
    frames = out['frame'].to_numpy(dtype=int) if 'frame' in out.columns else np.arange(len(out), dtype=int)
    if bin_col_x is not None and bin_col_y is not None:
        bx = out[bin_col_x].to_numpy(dtype=int)
        by = out[bin_col_y].to_numpy(dtype=int)
        bins = bx + by * int(n_bins_x)
    else:
        pos01 = _normalize_xy_for_bins(out)
        bins, _, _ = discretize_positions(pos01, n_bins_x=int(n_bins_x), n_bins_y=int(n_bins_y))
    bins = bins.astype(int)
    K = int(n_bins_x) * int(n_bins_y)
    levels = out['level'].to_numpy(dtype=float) if 'level' in out.columns else np.full(len(out), np.nan, dtype=float)
    bin_level = _compute_bin_level_lookup(levels, bins, K)
    occ_counts = np.bincount(bins, minlength=K).astype(float)
    S_tn = _to_tn_matrix(S, n_rows=len(out), frames=frames)
    if S_tn.shape[0] != len(out):
        n = min(S_tn.shape[0], len(out))
        S_tn = S_tn[:n]
        out = out.iloc[:n].reset_index(drop=True)
        frames = frames[:n]
        bins = bins[:n]
        levels = levels[:n]
    maze_time = out['maze_time'].to_numpy(dtype=float) if 'maze_time' in out.columns else (frames.astype(float) / float(fps))

    visits_by_level = {}
    finite_levels = np.unique(levels[np.isfinite(levels)].astype(int))
    for lv in finite_levels:
        idx = np.where(levels == lv)[0]
        if idx.size == 0:
            continue
        runs = []
        s = int(idx[0])
        p = int(idx[0])
        for t in idx[1:]:
            t = int(t)
            if t == p + 1:
                p = t
                continue
            if (p - s + 1) >= int(min_frames_per_visit):
                runs.append(np.arange(s, p + 1, dtype=int))
            s = t
            p = t
        if (p - s + 1) >= int(min_frames_per_visit):
            runs.append(np.arange(s, p + 1, dtype=int))
        if runs:
            visits_by_level[int(lv)] = runs
    return {
        'df': out,
        'S': S_tn,
        'frames': frames,
        'bins': bins,
        'levels': levels,
        'bin_level': bin_level,
        'occ_counts': occ_counts,
        'K': K,
        'n_bins_x': int(n_bins_x),
        'n_bins_y': int(n_bins_y),
        'maze_time': maze_time,
        'visits_by_level': visits_by_level,
    }


def compute_rate_maps_for_neurons(session_pack, neuron_ids=None):
    """
    Vectorized per-bin rate maps for one or many neurons.

    Returns array with shape (K, n_neurons_selected).
    """
    K = int(session_pack["K"])
    bins = np.asarray(session_pack["bins"], dtype=int)
    S = np.asarray(session_pack["S"], dtype=float)
    occ = np.asarray(session_pack["occ_counts"], dtype=float)
    if neuron_ids is None:
        neuron_ids = np.arange(S.shape[1], dtype=int)
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if neuron_ids.size == 0:
        return np.zeros((K, 0), dtype=float)
    vals = np.nan_to_num(S[:, neuron_ids], nan=0.0)
    spk = np.zeros((K, vals.shape[1]), dtype=float)
    np.add.at(spk, bins, vals)
    rm = np.divide(
        spk,
        occ[:, None],
        out=np.full((K, vals.shape[1]), np.nan, dtype=float),
        where=occ[:, None] > 0,
    )
    return rm


def compute_rate_map_for_neuron(session_pack, neuron_id):
    rm = compute_rate_maps_for_neurons(session_pack, neuron_ids=[int(neuron_id)])
    return np.asarray(rm[:, 0], dtype=float)


def compute_half_maps_for_neurons(session_pack, neuron_ids=None):
    """
    Vectorized first-half and second-half rate maps.

    Returns tuple:
    - rm1: (K, n_neurons_selected)
    - rm2: (K, n_neurons_selected)
    """
    K = int(session_pack["K"])
    bins = np.asarray(session_pack["bins"], dtype=int)
    S = np.asarray(session_pack["S"], dtype=float)
    if neuron_ids is None:
        neuron_ids = np.arange(S.shape[1], dtype=int)
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    vals = np.nan_to_num(S[:, neuron_ids], nan=0.0)
    T = len(bins)
    mid = T // 2
    idx1 = np.arange(0, mid, dtype=int)
    idx2 = np.arange(mid, T, dtype=int)
    occ1 = np.bincount(bins[idx1], minlength=K).astype(float)
    occ2 = np.bincount(bins[idx2], minlength=K).astype(float)
    spk1 = np.zeros((K, vals.shape[1]), dtype=float)
    spk2 = np.zeros((K, vals.shape[1]), dtype=float)
    if idx1.size:
        np.add.at(spk1, bins[idx1], vals[idx1])
    if idx2.size:
        np.add.at(spk2, bins[idx2], vals[idx2])
    rm1 = np.divide(
        spk1,
        occ1[:, None],
        out=np.full((K, vals.shape[1]), np.nan, dtype=float),
        where=occ1[:, None] > 0,
    )
    rm2 = np.divide(
        spk2,
        occ2[:, None],
        out=np.full((K, vals.shape[1]), np.nan, dtype=float),
        where=occ2[:, None] > 0,
    )
    return rm1, rm2


def compute_half_maps_for_neuron(session_pack, neuron_id):
    rm1, rm2 = compute_half_maps_for_neurons(session_pack, neuron_ids=[int(neuron_id)])
    return np.asarray(rm1[:, 0], dtype=float), np.asarray(rm2[:, 0], dtype=float)


def compute_binwise_shuffle_pmap(session_pack, neuron_id, n_shuffles=200, fdr_q=0.05, rng=None, min_occ=3.0):
    """
    Compute per-bin shuffle p/q maps for one neuron.

    `n_shuffles` controls p-value resolution: minimum non-zero p is
    `1 / (n_shuffles + 1)`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs_map = compute_rate_map_for_neuron(session_pack, neuron_id)
    K = int(session_pack['K'])
    occ = np.asarray(session_pack['occ_counts'], dtype=float)
    valid = np.isfinite(occ) & (occ >= float(min_occ))
    bins = np.asarray(session_pack['bins'], dtype=int)
    S = np.asarray(session_pack['S'], dtype=float)
    s = np.asarray(S[:, int(neuron_id)], dtype=float)
    s = np.nan_to_num(s, nan=0.0)

    ge_count = np.ones(K, dtype=float)
    null_means = []
    null_stds = []
    T = len(s)
    for _ in range(int(max(0, n_shuffles))):
        shift = 0 if T <= 1 else int(rng.integers(1, T))
        s_sh = np.roll(s, shift)
        spk_sh = np.bincount(bins, weights=s_sh, minlength=K).astype(float)
        rm_sh = np.divide(spk_sh, occ, out=np.full(K, np.nan, dtype=float), where=occ > 0)
        cmp = np.zeros(K, dtype=bool)
        cmp[valid] = np.asarray(rm_sh[valid] >= obs_map[valid], dtype=bool)
        ge_count += cmp.astype(float)
        if np.any(valid):
            null_means.append(float(np.nanmean(rm_sh[valid])))
            null_stds.append(float(np.nanstd(rm_sh[valid])))

    denom = float(int(max(0, n_shuffles)) + 1)
    p_map = np.full(K, np.nan, dtype=float)
    p_map[valid] = ge_count[valid] / denom
    q_map = np.full(K, np.nan, dtype=float)
    q_valid = bh_fdr(p_map[valid])
    q_map[valid] = q_valid
    sig_mask = np.zeros(K, dtype=bool)
    sig_mask[valid] = np.isfinite(q_valid) & (q_valid <= float(fdr_q))
    return {
        'obs_map': obs_map,
        'p_map': p_map,
        'q_map': q_map,
        'sig_mask': sig_mask,
        'null_summary': {
            'n_shuffles': int(max(0, n_shuffles)),
            'null_mean': float(np.nanmean(null_means)) if len(null_means) else np.nan,
            'null_std': float(np.nanmean(null_stds)) if len(null_stds) else np.nan,
            'n_sig_bins': int(np.sum(sig_mask)),
            'min_occ_for_sig': float(min_occ),
        },
    }


def _make_connectivity_structure(conn=4):
    if int(conn) == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    return np.ones((3, 3), dtype=bool)


def _build_bin_to_ma_cell(n_bins_x, n_bins_y, ma_obj):
    out = np.full((int(n_bins_x), int(n_bins_y)), -1, dtype=int)
    if ma_obj is None or not hasattr(ma_obj, 'ce') or ma_obj.ce is None:
        return out
    for r in range(int(n_bins_x)):
        for c in range(int(n_bins_y)):
            cid = ma_obj.ce.get((int(c), int(r)), None)
            if cid is None:
                cid = ma_obj.ce.get((int(r), int(c)), None)
            if cid is not None:
                out[r, c] = int(cid)
    return out


def _grow_component_with_support(
    seed_comp2d,
    obs2d,
    occ2d,
    p2d=None,
    forbidden_seed2d=None,
    support_connectivity=8,
    support_min_occ=3,
    support_rate_frac=0.05,
    support_pmax=0.20,
    bin_to_ma_cell2d=None,
    ma_di=None,
    support_max_map_dist=np.inf,
):
    grown = np.asarray(seed_comp2d, dtype=bool).copy()
    if not np.any(grown):
        return grown
    if forbidden_seed2d is None:
        forbidden_seed2d = np.zeros_like(grown, dtype=bool)
    peak_val = float(np.nanmax(np.asarray(obs2d, dtype=float)[grown]))
    if (not np.isfinite(peak_val)) or peak_val <= 0:
        return grown
    rate_thr = float(peak_val) * float(max(0.0, support_rate_frac))
    support_ok = np.isfinite(obs2d) & (np.asarray(obs2d, dtype=float) >= rate_thr)
    support_ok &= np.isfinite(occ2d) & (np.asarray(occ2d, dtype=float) >= float(support_min_occ))
    if p2d is not None:
        support_ok &= np.isfinite(p2d) & (np.asarray(p2d, dtype=float) <= float(support_pmax))
    support_ok &= (~np.asarray(forbidden_seed2d, dtype=bool))
    # maze-topology distance constraint (legacy: support_max_map_dist=6.0)
    if (np.isfinite(support_max_map_dist) and bin_to_ma_cell2d is not None
            and ma_di is not None):
        b2ma = np.asarray(bin_to_ma_cell2d, dtype=int)
        seed_ma_ids = np.unique(b2ma[grown])
        seed_ma_ids = seed_ma_ids[seed_ma_ids >= 0]
        dist_ok = np.zeros_like(grown, dtype=bool)
        if seed_ma_ids.size > 0:
            valid_mask = b2ma >= 0
            valid_ids = b2ma[valid_mask]
            dsub = np.asarray(ma_di, dtype=float)[np.ix_(valid_ids, seed_ma_ids)]
            min_d = np.nanmin(dsub, axis=1) if dsub.size else np.full(
                valid_ids.shape, np.inf)
            dist_ok[valid_mask] = np.isfinite(min_d) & (
                min_d <= float(support_max_map_dist))
        support_ok &= dist_ok
    structure = _make_connectivity_structure(support_connectivity)
    while True:
        ring = binary_dilation(grown, structure=structure) & (~grown)
        add = ring & support_ok
        if not np.any(add):
            break
        grown |= add
    return grown


def _peak_from_mask(mask2d, obs2d):
    vals = np.asarray(obs2d, dtype=float)
    m = np.asarray(mask2d, dtype=bool)
    if not np.any(m & np.isfinite(vals)):
        return np.nan, (-1, -1)
    idx = np.nanargmax(np.where(m, vals, np.nan))
    r, c = np.unravel_index(idx, vals.shape)
    return float(vals[r, c]), (int(r), int(c))


def _line_coords(r0, c0, r1, c1):
    n = int(max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)))) + 1
    rr = np.rint(np.linspace(r0, r1, n)).astype(int)
    cc = np.rint(np.linspace(c0, c1, n)).astype(int)
    return rr, cc


def _majority_level(mask2d, bin_level2d):
    lv = np.asarray(bin_level2d, dtype=float)[np.asarray(mask2d, dtype=bool)]
    lv = lv[np.isfinite(lv)]
    if lv.size == 0:
        return np.nan
    u, c = np.unique(lv.astype(int), return_counts=True)
    return int(u[np.argmax(c)])


def _grid_shortest_path_len(start_rc, goal_rc, traversable2d, connectivity=4):
    r0, c0 = int(start_rc[0]), int(start_rc[1])
    r1, c1 = int(goal_rc[0]), int(goal_rc[1])
    trav = np.asarray(traversable2d, dtype=bool)
    nr, nc = trav.shape
    if r0 < 0 or r0 >= nr or c0 < 0 or c0 >= nc:
        return np.inf
    if r1 < 0 or r1 >= nr or c1 < 0 or c1 >= nc:
        return np.inf
    if (not trav[r0, c0]) or (not trav[r1, c1]):
        return np.inf
    if int(connectivity) == 8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dist = -np.ones((nr, nc), dtype=int)
    dist[r0, c0] = 0
    q = [(r0, c0)]
    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        if r == r1 and c == c1:
            return float(dist[r, c])
        d_next = int(dist[r, c]) + 1
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            if (not trav[rr, cc]) or (dist[rr, cc] >= 0):
                continue
            dist[rr, cc] = d_next
            q.append((rr, cc))
    return np.inf


def _should_merge_components(
    comp_a,
    comp_b,
    obs2d,
    occ2d,
    bin_level2d,
    merge_gap_bins=1,
    merge_max_peak_dist=6.0,
    merge_saddle_ratio=0.25,
    merge_min_line_occ=3,
    merge_require_same_level=True,
    merge_max_map_dist=np.inf,
    merge_path_connectivity=4,
):
    ma = np.asarray(comp_a['mask2d'], dtype=bool)
    mb = np.asarray(comp_b['mask2d'], dtype=bool)
    close = np.any(binary_dilation(ma, iterations=max(0, int(merge_gap_bins))) & mb) if int(merge_gap_bins) > 0 else np.any(ma & mb)
    if not close:
        return False
    peak_a, (ra, ca) = _peak_from_mask(ma, obs2d)
    peak_b, (rb, cb) = _peak_from_mask(mb, obs2d)
    if (ra < 0) or (rb < 0) or (not np.isfinite(peak_a)) or (not np.isfinite(peak_b)):
        return False
    dist = float(np.sqrt((ra - rb) ** 2 + (ca - cb) ** 2))
    if dist > float(merge_max_peak_dist):
        return False
    if bool(merge_require_same_level):
        la = _majority_level(ma, bin_level2d)
        lb = _majority_level(mb, bin_level2d)
        if np.isfinite(la) and np.isfinite(lb) and int(la) != int(lb):
            return False
    traversable = np.isfinite(occ2d) & (np.asarray(occ2d, dtype=float) > 0)
    dist_map = _grid_shortest_path_len((ra, ca), (rb, cb), traversable, connectivity=int(merge_path_connectivity))
    if (not np.isfinite(dist_map)) or (dist_map > float(merge_max_map_dist)):
        return False
    rr, cc = _line_coords(ra, ca, rb, cb)
    rr = np.clip(rr, 0, obs2d.shape[0] - 1)
    cc = np.clip(cc, 0, obs2d.shape[1] - 1)
    line_obs = np.asarray(obs2d[rr, cc], dtype=float)
    line_occ = np.asarray(occ2d[rr, cc], dtype=float)
    finite_obs = line_obs[np.isfinite(line_obs)]
    if finite_obs.size == 0:
        return False
    saddle = float(np.nanmin(finite_obs))
    if saddle < float(merge_saddle_ratio) * float(min(peak_a, peak_b)):
        return False
    finite_occ = line_occ[np.isfinite(line_occ)]
    if finite_occ.size == 0:
        return False
    if float(np.nanmedian(finite_occ)) < float(merge_min_line_occ):
        return False
    return True


def _centroid_from_mask(mask2d, obs2d=None):
    mask2d = np.asarray(mask2d, dtype=bool)
    rr, cc = np.indices(mask2d.shape)
    if obs2d is None:
        w = np.ones(mask2d.shape, dtype=float)
    else:
        w = np.nan_to_num(np.asarray(obs2d, dtype=float), nan=0.0)
        w = np.clip(w, a_min=0.0, a_max=None)
    w = w * mask2d.astype(float)
    if np.sum(w) > 0:
        cx = float(np.sum(rr * w) / np.sum(w))
        cy = float(np.sum(cc * w) / np.sum(w))
    else:
        r = rr[mask2d]
        c = cc[mask2d]
        cx = float(np.mean(r)) if r.size else np.nan
        cy = float(np.mean(c)) if c.size else np.nan
    return cx, cy


def extract_fields_from_sigmask(
    obs_map,
    sig_mask,
    occ_counts,
    bin_level_lookup,
    n_bins_x,
    n_bins_y,
    min_component_size=1,
    component_connectivity=4,
    half_maps=None,
    p_map=None,
    enable_support_growth=False,
    support_connectivity=8,
    support_min_occ=3,
    support_rate_frac=0.05,
    support_pmax=0.20,
    support_max_map_dist=np.inf,
    support_grow_beyond_sig=False,
    bin_to_ma_cell2d=None,
    ma_di=None,
    enable_field_merge=True,
    merge_gap_bins=1,
    merge_max_peak_dist=6.0,
    merge_saddle_ratio=0.25,
    merge_min_line_occ=3,
    merge_require_same_level=True,
    merge_max_map_dist=np.inf,
    merge_path_connectivity=4,
):
    n_bins_x = int(n_bins_x)
    n_bins_y = int(n_bins_y)
    obs2d = np.asarray(obs_map, dtype=float).reshape(n_bins_x, n_bins_y)
    sig2d = np.asarray(sig_mask, dtype=bool).reshape(n_bins_x, n_bins_y)
    occ2d = np.asarray(occ_counts, dtype=float).reshape(n_bins_x, n_bins_y)
    bin_level2d = np.asarray(bin_level_lookup, dtype=float).reshape(n_bins_x, n_bins_y)
    p2d = None if p_map is None else np.asarray(p_map, dtype=float).reshape(n_bins_x, n_bins_y)

    structure = _make_connectivity_structure(component_connectivity)
    lab, n_comp = ndi_label(sig2d, structure=structure)
    masks = []
    merge_parts = []
    for cid in range(1, int(n_comp) + 1):
        comp2d = lab == cid
        if int(np.sum(comp2d)) < int(min_component_size):
            continue
        if enable_support_growth:
            if support_grow_beyond_sig:
                # Legacy: only block other seeds' sig bins; allow growth
                # into non-significant bins that meet rate/occ/p thresholds
                forbidden = sig2d & (~comp2d)
            else:
                # Current: confine growth within the significance mask
                forbidden = ~sig2d
            comp2d = _grow_component_with_support(
                comp2d,
                obs2d=obs2d,
                occ2d=occ2d,
                p2d=p2d,
                forbidden_seed2d=forbidden,
                support_connectivity=support_connectivity,
                support_min_occ=support_min_occ,
                support_rate_frac=support_rate_frac,
                support_pmax=support_pmax,
                bin_to_ma_cell2d=bin_to_ma_cell2d,
                ma_di=ma_di,
                support_max_map_dist=support_max_map_dist,
            )
        masks.append(comp2d)
        merge_parts.append(1)

    if enable_field_merge and len(masks) > 1:
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(masks):
                j = i + 1
                while j < len(masks):
                    if _should_merge_components(
                        {'mask2d': masks[i]},
                        {'mask2d': masks[j]},
                        obs2d=obs2d,
                        occ2d=occ2d,
                        bin_level2d=bin_level2d,
                        merge_gap_bins=merge_gap_bins,
                        merge_max_peak_dist=merge_max_peak_dist,
                        merge_saddle_ratio=merge_saddle_ratio,
                        merge_min_line_occ=merge_min_line_occ,
                        merge_require_same_level=merge_require_same_level,
                        merge_max_map_dist=merge_max_map_dist,
                        merge_path_connectivity=merge_path_connectivity,
                    ):
                        masks[i] = masks[i] | masks[j]
                        merge_parts[i] = int(merge_parts[i]) + int(merge_parts[j])
                        del masks[j]
                        del merge_parts[j]
                        changed = True
                    else:
                        j += 1
                i += 1

    fields = []
    hm1 = hm2 = None
    if half_maps is not None:
        hm1 = np.asarray(half_maps[0], dtype=float).ravel()
        hm2 = np.asarray(half_maps[1], dtype=float).ravel()
    for fid, comp2d in enumerate(masks):
        comp = comp2d.ravel()
        occ_support = float(np.nansum(occ2d.ravel()[comp]))
        finite_idx = np.where(comp & np.isfinite(obs2d.ravel()))[0]
        if finite_idx.size:
            peak_idx = int(finite_idx[np.argmax(obs2d.ravel()[finite_idx])])
            peak_val = float(obs2d.ravel()[peak_idx])
        else:
            peak_idx = -1
            peak_val = np.nan
        cx, cy = _centroid_from_mask(comp2d, obs2d)
        level = _majority_level(comp2d, bin_level2d)
        half_corr = _safe_corr(hm1, hm2, mask=comp, min_n=2) if (hm1 is not None and hm2 is not None) else np.nan
        fields.append(
            {
                'field_idx': int(fid),
                'mask': comp.copy(),
                'mask2d': comp2d.copy(),
                'area': int(np.sum(comp2d)),
                'peak_bin': int(peak_idx) if peak_idx >= 0 else -1,
                'peak_value': float(peak_val),
                'centroid_x': float(cx),
                'centroid_y': float(cy),
                'level': float(level) if np.isfinite(level) else np.nan,
                'occ_support': float(occ_support),
                'half_corr': float(half_corr) if np.isfinite(half_corr) else np.nan,
                'n_merged_parts': int(merge_parts[fid]),
            }
        )
    return fields


def _shift_mask_2d(mask2d, dr=0, dc=0):
    m = np.asarray(mask2d, dtype=bool)
    out = np.zeros_like(m, dtype=bool)
    nr, nc = m.shape
    r0_src = max(0, -int(dr))
    r1_src = min(nr, nr - int(dr))
    c0_src = max(0, -int(dc))
    c1_src = min(nc, nc - int(dc))
    r0_dst = max(0, int(dr))
    r1_dst = min(nr, nr + int(dr))
    c0_dst = max(0, int(dc))
    c1_dst = min(nc, nc + int(dc))
    if r1_src > r0_src and c1_src > c0_src and r1_dst > r0_dst and c1_dst > c0_dst:
        out[r0_dst:r1_dst, c0_dst:c1_dst] = m[r0_src:r1_src, c0_src:c1_src]
    return out


def score_field_pair(field_a, field_b, obs_map_a=None, obs_map_b=None, n_bins_x=None, n_bins_y=None):
    ma = np.asarray(field_a.get('mask2d', field_a.get('mask')), dtype=bool)
    mb = np.asarray(field_b.get('mask2d', field_b.get('mask')), dtype=bool)
    if ma.ndim == 1:
        if n_bins_x is None or n_bins_y is None:
            raise ValueError('n_bins_x and n_bins_y required for flattened masks')
        ma = ma.reshape(int(n_bins_x), int(n_bins_y))
    if mb.ndim == 1:
        if n_bins_x is None or n_bins_y is None:
            raise ValueError('n_bins_x and n_bins_y required for flattened masks')
        mb = mb.reshape(int(n_bins_x), int(n_bins_y))
    inter = int(np.sum(ma & mb))
    uni = int(np.sum(ma | mb))
    jaccard = (float(inter) / float(uni)) if uni > 0 else 0.0
    cax = float(field_a.get('centroid_x', np.nan))
    cay = float(field_a.get('centroid_y', np.nan))
    cbx = float(field_b.get('centroid_x', np.nan))
    cby = float(field_b.get('centroid_y', np.nan))
    if np.isfinite(cax) and np.isfinite(cay) and np.isfinite(cbx) and np.isfinite(cby):
        cd = float(np.sqrt((cax - cbx) ** 2 + (cay - cby) ** 2))
    else:
        cd = np.nan
    level_a = field_a.get('level', np.nan)
    level_b = field_b.get('level', np.nan)
    level_match = bool(np.isfinite(level_a) and np.isfinite(level_b) and int(level_a) == int(level_b))
    level_factor = 1.0 if level_match else 0.7
    dist_term = 1.0 / (1.0 + cd) if np.isfinite(cd) else 0.0
    score = level_factor * (0.7 * jaccard + 0.3 * dist_term)
    return {
        'score': float(score),
        'jaccard': float(jaccard),
        'centroid_dist': float(cd) if np.isfinite(cd) else np.nan,
        'level_match': bool(level_match),
        'overlap_bins': int(inter),
    }


def match_fields_hungarian(fields_a, fields_b, obs_map_a=None, obs_map_b=None, n_bins_x=None, n_bins_y=None, min_score=0.0):
    m = len(fields_a)
    n = len(fields_b)
    if m == 0 or n == 0:
        return pd.DataFrame(columns=['field_idx_s1', 'field_idx_s2', 'score', 'jaccard', 'centroid_dist', 'level_match', 'overlap_bins', 'retained'])
    score_mat = np.full((m, n), -1.0, dtype=float)
    detail = {}
    for i, fa in enumerate(fields_a):
        for j, fb in enumerate(fields_b):
            d = score_field_pair(fa, fb, obs_map_a=obs_map_a, obs_map_b=obs_map_b, n_bins_x=n_bins_x, n_bins_y=n_bins_y)
            score_mat[i, j] = float(d['score'])
            detail[(i, j)] = d
    ridx, cidx = linear_sum_assignment(-score_mat)
    rows = []
    for i, j in zip(ridx, cidx):
        d = detail[(int(i), int(j))]
        keep = bool(float(d['score']) >= float(min_score))
        rows.append(
            {
                'field_idx_s1': int(fields_a[int(i)].get('field_idx', i)),
                'field_idx_s2': int(fields_b[int(j)].get('field_idx', j)),
                'score': float(d['score']),
                'jaccard': float(d['jaccard']),
                'centroid_dist': float(d['centroid_dist']) if np.isfinite(d['centroid_dist']) else np.nan,
                'level_match': bool(d['level_match']),
                'overlap_bins': int(d['overlap_bins']),
                'retained': bool(keep),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('score', ascending=False).reset_index(drop=True)
    return out


def compute_match_null(fields_a, fields_b, n_shuffles=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if len(fields_a) == 0 or len(fields_b) == 0:
        return {'observed_mean_score': np.nan, 'null_mean': np.nan, 'null_std': np.nan, 'p_ge': np.nan}
    m, n = len(fields_a), len(fields_b)
    score_mat = np.zeros((m, n), dtype=float)
    for i, fa in enumerate(fields_a):
        for j, fb in enumerate(fields_b):
            score_mat[i, j] = score_field_pair(fa, fb)['score']
    ridx, cidx = linear_sum_assignment(-score_mat)
    obs = float(np.nanmean(score_mat[ridx, cidx])) if len(ridx) else np.nan
    k = min(m, n)
    null_vals = []
    for _ in range(int(max(0, n_shuffles))):
        rsel = rng.choice(m, size=k, replace=False) if m > k else np.arange(m)
        csel = rng.choice(n, size=k, replace=False) if n > k else np.arange(n)
        rng.shuffle(csel)
        null_vals.append(float(np.nanmean(score_mat[rsel, csel])))
    null_vals = np.asarray(null_vals, dtype=float)
    p_ge = float((1.0 + np.sum(null_vals >= obs)) / (1.0 + len(null_vals))) if len(null_vals) else np.nan
    return {
        'observed_mean_score': float(obs),
        'null_mean': float(np.nanmean(null_vals)) if len(null_vals) else np.nan,
        'null_std': float(np.nanstd(null_vals)) if len(null_vals) else np.nan,
        'p_ge': p_ge,
        'n_shuffles': int(max(0, n_shuffles)),
    }


# ---------------------------------------------------------------------------
# Phase-D: place-field pair pipeline functions
# ---------------------------------------------------------------------------

def score_field_pair_full(f1, f2, map1, map2, n_bins_x, n_bins_y,
                          weights=None, dist_norm=None):
    """Score a field pair using legacy weighted formula.

    Default weights: {iou: 0.70, dice: 0.30, corr: 0.00, dist: 0.00}.
    No level penalty (unlike ``score_field_pair``).

    Returns dict with iou, dice, local_corr, centroid_dist, score,
    area_s1, area_s2, overlap_bins.
    """
    if weights is None:
        weights = {'iou': 0.70, 'dice': 0.30, 'corr': 0.00, 'dist': 0.00}
    w_iou = float(weights.get('iou', 0.70))
    w_dice = float(weights.get('dice', 0.30))
    w_corr = float(weights.get('corr', 0.00))
    w_dist = float(weights.get('dist', 0.00))

    ma = np.asarray(f1.get('mask2d', f1.get('mask')), dtype=bool)
    mb = np.asarray(f2.get('mask2d', f2.get('mask')), dtype=bool)
    if ma.ndim == 1:
        ma = ma.reshape(int(n_bins_x), int(n_bins_y))
    if mb.ndim == 1:
        mb = mb.reshape(int(n_bins_x), int(n_bins_y))

    inter = int(np.sum(ma & mb))
    uni = int(np.sum(ma | mb))
    area_s1 = int(np.sum(ma))
    area_s2 = int(np.sum(mb))

    iou = float(inter) / float(uni) if uni > 0 else 0.0
    dice = 2.0 * float(inter) / float(area_s1 + area_s2) if (area_s1 + area_s2) > 0 else 0.0

    # local correlation on union mask
    m1_flat = np.asarray(map1, dtype=float).ravel()
    m2_flat = np.asarray(map2, dtype=float).ravel()
    union_flat = (ma | mb).ravel()
    local_corr = _safe_corr(m1_flat, m2_flat, mask=union_flat, min_n=2)

    # centroid distance
    cax = float(f1.get('centroid_x', np.nan))
    cay = float(f1.get('centroid_y', np.nan))
    cbx = float(f2.get('centroid_x', np.nan))
    cby = float(f2.get('centroid_y', np.nan))
    if np.isfinite(cax) and np.isfinite(cay) and np.isfinite(cbx) and np.isfinite(cby):
        centroid_dist = float(np.sqrt((cax - cbx) ** 2 + (cay - cby) ** 2))
    else:
        centroid_dist = np.nan

    # score components
    corr_term = float(local_corr) if np.isfinite(local_corr) else 0.0
    if dist_norm is None:
        dist_norm = float(np.sqrt(float(n_bins_x) ** 2 + float(n_bins_y) ** 2))
    dist_term = (1.0 - centroid_dist / dist_norm) if np.isfinite(centroid_dist) else 0.0
    dist_term = max(0.0, dist_term)

    score = w_iou * iou + w_dice * dice + w_corr * corr_term - w_dist * dist_term

    return {
        'iou': float(iou),
        'dice': float(dice),
        'local_corr': float(local_corr) if np.isfinite(local_corr) else np.nan,
        'centroid_dist': float(centroid_dist) if np.isfinite(centroid_dist) else np.nan,
        'score': float(score),
        'area_s1': int(area_s1),
        'area_s2': int(area_s2),
        'overlap_bins': int(inter),
    }


def match_fields_hungarian_full(fields_a, fields_b, map_a, map_b,
                                n_bins_x, n_bins_y, *,
                                n_null_repeats=200, rng=None, weights=None):
    """Hungarian matching with shared-pool null model (legacy-compatible).

    Returns DataFrame with 26 per-pair columns (caller adds pair_idx,
    train_id, test_id to make 29).  Only matched pairs (one-to-one).
    ``retained`` defaults to False; caller applies retention policy.
    """
    _COLS = [
        'field_idx_s1', 'field_idx_s2', 'level',
        'area_s1', 'area_s2', 'peak_s1', 'peak_s2',
        'centroid_x_s1', 'centroid_y_s1', 'centroid_x_s2', 'centroid_y_s2',
        'occupancy_support_s1', 'occupancy_support_s2',
        'half_corr_s1', 'half_corr_s2',
        'iou', 'dice', 'centroid_dist', 'local_corr', 'overlap_bins',
        'match_score', 'p_match', 'q_match', 'retained',
        'null_n', 'null_mean', 'null_std',
    ]
    m = len(fields_a)
    n = len(fields_b)
    if m == 0 or n == 0:
        return pd.DataFrame(columns=_COLS)
    if rng is None:
        rng = np.random.default_rng(0)
    if weights is None:
        weights = {'iou': 0.70, 'dice': 0.30, 'corr': 0.00, 'dist': 0.00}

    nbx, nby = int(n_bins_x), int(n_bins_y)
    dist_norm = float(np.sqrt(float(nbx) ** 2 + float(nby) ** 2))

    # --- build score matrix ---
    score_mat = np.full((m, n), -1.0, dtype=float)
    detail = {}
    for i, fa in enumerate(fields_a):
        for j, fb in enumerate(fields_b):
            d = score_field_pair_full(fa, fb, map_a, map_b, nbx, nby,
                                      weights=weights, dist_norm=dist_norm)
            score_mat[i, j] = d['score']
            detail[(i, j)] = d

    # --- Hungarian assignment ---
    ridx, cidx = linear_sum_assignment(-score_mat)
    k = len(ridx)

    # --- Shared-pool null model ---
    null_scores = []
    if n_null_repeats > 0 and k > 0:
        # precompute masks for fields_b
        masks_b = []
        for fb in fields_b:
            mb = np.asarray(fb.get('mask2d', fb.get('mask')), dtype=bool)
            if mb.ndim == 1:
                mb = mb.reshape(nbx, nby)
            masks_b.append(mb)

        for _rep in range(int(n_null_repeats)):
            dx = int(rng.integers(0, nbx))
            dy = int(rng.integers(0, nby))
            # shift all field_b masks
            shifted_fields_b = []
            for jj, fb in enumerate(fields_b):
                shifted_mask = np.roll(np.roll(masks_b[jj], dx, axis=0), dy, axis=1)
                fb_copy = dict(fb)
                fb_copy['mask2d'] = shifted_mask
                # shift centroids
                cx = fb.get('centroid_x', np.nan)
                cy = fb.get('centroid_y', np.nan)
                if np.isfinite(cx):
                    fb_copy['centroid_x'] = (float(cx) + dx) % nbx
                if np.isfinite(cy):
                    fb_copy['centroid_y'] = (float(cy) + dy) % nby
                shifted_fields_b.append(fb_copy)

            # recompute score matrix with shifted fields
            null_mat = np.full((m, n), -1.0, dtype=float)
            for i, fa in enumerate(fields_a):
                for j, sfb in enumerate(shifted_fields_b):
                    null_mat[i, j] = score_field_pair_full(
                        fa, sfb, map_a, map_b, nbx, nby,
                        weights=weights, dist_norm=dist_norm)['score']
            nr, nc = linear_sum_assignment(-null_mat)
            for ri, ci in zip(nr, nc):
                null_scores.append(float(null_mat[ri, ci]))

    null_scores = np.asarray(null_scores, dtype=float)
    null_n = len(null_scores)
    null_mean = float(np.nanmean(null_scores)) if null_n > 0 else np.nan
    null_std = float(np.nanstd(null_scores)) if null_n > 0 else np.nan

    # --- build output rows ---
    rows = []
    obs_scores = []
    for ri, ci in zip(ridx, cidx):
        d = detail[(int(ri), int(ci))]
        fa = fields_a[int(ri)]
        fb = fields_b[int(ci)]
        obs_s = float(d['score'])
        obs_scores.append(obs_s)

        p_match = float((1.0 + np.sum(null_scores >= obs_s)) / (1.0 + null_n)) if null_n > 0 else np.nan

        lev = fa.get('level', np.nan)
        if not np.isfinite(lev):
            lev = fb.get('level', np.nan)

        rows.append({
            'field_idx_s1': int(fa.get('field_idx', ri)),
            'field_idx_s2': int(fb.get('field_idx', ci)),
            'level': float(lev) if np.isfinite(lev) else np.nan,
            'area_s1': int(d['area_s1']),
            'area_s2': int(d['area_s2']),
            'peak_s1': float(fa.get('peak_value', np.nan)),
            'peak_s2': float(fb.get('peak_value', np.nan)),
            'centroid_x_s1': float(fa.get('centroid_x', np.nan)),
            'centroid_y_s1': float(fa.get('centroid_y', np.nan)),
            'centroid_x_s2': float(fb.get('centroid_x', np.nan)),
            'centroid_y_s2': float(fb.get('centroid_y', np.nan)),
            'occupancy_support_s1': float(fa.get('occ_support', np.nan)),
            'occupancy_support_s2': float(fb.get('occ_support', np.nan)),
            'half_corr_s1': float(fa.get('half_corr', np.nan)),
            'half_corr_s2': float(fb.get('half_corr', np.nan)),
            'iou': float(d['iou']),
            'dice': float(d['dice']),
            'centroid_dist': float(d['centroid_dist']) if np.isfinite(d['centroid_dist']) else np.nan,
            'local_corr': float(d['local_corr']) if np.isfinite(d['local_corr']) else np.nan,
            'overlap_bins': int(d['overlap_bins']),
            'match_score': obs_s,
            'p_match': p_match,
            'q_match': np.nan,  # filled below
            'retained': False,
            'null_n': int(null_n),
            'null_mean': null_mean,
            'null_std': null_std,
        })

    # BH-FDR across all matches
    if rows:
        p_vals = np.array([r['p_match'] for r in rows], dtype=float)
        q_vals = bh_fdr(p_vals)
        for i, r in enumerate(rows):
            r['q_match'] = float(q_vals[i]) if np.isfinite(q_vals[i]) else np.nan

    out = pd.DataFrame(rows, columns=_COLS)
    if len(out):
        out = out.sort_values('match_score', ascending=False).reset_index(drop=True)
    return out


def cumulative_visit_rate_maps(session_pack, neuron_id, visits):
    """Progressive cumulative rate maps from visit list.

    Parameters
    ----------
    session_pack : dict
        From ``build_session_pack``.
    neuron_id : int
        Neuron column index in ``session_pack['S']``.
    visits : list of np.ndarray
        Each element is an array of row indices for one visit.

    Returns
    -------
    np.ndarray, shape (n_visits, K)
        Row *i* is the rate map after accumulating visits 0..i.
    """
    K = int(session_pack['K'])
    bins = np.asarray(session_pack['bins'], dtype=int)
    S = np.asarray(session_pack['S'], dtype=float)
    vals = S[:, int(neuron_id)]
    n = len(visits)
    if n == 0:
        return np.empty((0, K), dtype=float)
    cum_spk = np.zeros(K, dtype=float)
    cum_occ = np.zeros(K, dtype=float)
    out = np.full((n, K), np.nan, dtype=float)
    for vi, idx in enumerate(visits):
        idx = np.asarray(idx, dtype=int)
        b = bins[idx]
        v = np.nan_to_num(vals[idx], nan=0.0)
        np.add.at(cum_spk, b, v)
        np.add.at(cum_occ, b, np.ones(len(b), dtype=float))
        mask = cum_occ > 0
        out[vi, mask] = cum_spk[mask] / cum_occ[mask]
    return out


# --- Pluggable establishment strategies ---

def event_establishment_visit(vis_df, threshold=0.5, min_consecutive=2):
    """Public wrapper around ``_establishment_visit``.

    Returns (visit_idx: float, curve: np.ndarray).
    """
    idx = _establishment_visit(vis_df, threshold=threshold,
                               min_consecutive=min_consecutive)
    if vis_df is None or vis_df.empty:
        return (np.nan, np.array([], dtype=float))
    y = np.asarray(vis_df['has_event'].astype(float).to_numpy(), dtype=float)
    curve = np.cumsum(y) / np.arange(1, len(y) + 1, dtype=float)
    return (float(idx) if np.isfinite(idx) else np.nan, curve)


def corr_establishment_visit(cum_maps, ref_map, local_mask, *,
                              threshold=0.3, min_consecutive=2, min_pts=5):
    """Correlation-based establishment: first sustained r >= threshold.

    Parameters
    ----------
    cum_maps : np.ndarray, shape (n_visits, K)
        From ``cumulative_visit_rate_maps``.
    ref_map : np.ndarray, shape (K,)
        Reference (full-session) rate map.
    local_mask : np.ndarray, shape (K,), bool
        Bins to include in correlation.

    Returns
    -------
    (visit_idx: float, curve: np.ndarray)
    """
    ref = np.asarray(ref_map, dtype=float).ravel()
    mask = np.asarray(local_mask, dtype=bool).ravel()
    n = cum_maps.shape[0] if cum_maps.ndim == 2 else 0
    if n == 0:
        return (np.nan, np.array([], dtype=float))
    curve = np.full(n, np.nan, dtype=float)
    for i in range(n):
        curve[i] = _safe_corr(cum_maps[i], ref, mask=mask, min_n=int(min_pts))
    idx = _first_sustained_hit(curve, threshold=threshold,
                                min_consecutive=min_consecutive)
    return (float(idx) if np.isfinite(idx) else np.nan, curve)


def changepoint_establishment_visit(vis_df, *, activity_col='mean_activity',
                                     pen=1.0):
    """Change-point establishment using ``ruptures`` package.

    Returns (visit_idx: float, curve: np.ndarray).
    If ``ruptures`` is not available, returns (np.nan, []) with a warning.
    """
    if vis_df is None or vis_df.empty:
        return (np.nan, np.array([], dtype=float))
    try:
        import ruptures as rpt
    except ImportError:
        warnings.warn(
            "ruptures package not available; changepoint establishment "
            "will return NaN. Install with: pip install ruptures",
            stacklevel=2)
        return (np.nan, np.array([], dtype=float))
    y = np.asarray(vis_df[activity_col].to_numpy(), dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    if y.size < 3:
        return (np.nan, y)
    algo = rpt.Pelt(model="rbf", min_size=2).fit(y.reshape(-1, 1))
    bkpts = algo.predict(pen=float(pen))
    # first changepoint (in visit-count space, 1-based)
    if bkpts and bkpts[0] < len(y):
        return (float(bkpts[0]), y)
    return (np.nan, y)


def compute_establishment(method, **kwargs):
    """Dispatcher for establishment strategies.

    Parameters
    ----------
    method : str
        One of ``"event"``, ``"corr"``, ``"changepoint"``.
    **kwargs :
        Forwarded to the selected strategy function.
    """
    if method == 'event':
        return event_establishment_visit(**kwargs)
    elif method == 'corr':
        return corr_establishment_visit(**kwargs)
    elif method == 'changepoint':
        return changepoint_establishment_visit(**kwargs)
    else:
        raise ValueError(f"Unknown establishment method: {method!r}")


def _rebuild_display_mask(mask2d, occ2d, bridge_iters=1, min_fragment_size=2):
    m = np.asarray(mask2d, dtype=bool).copy()
    occ_ok = np.asarray(occ2d, dtype=float) > 0
    m &= occ_ok
    if int(bridge_iters) > 0:
        m = binary_closing(m, structure=np.ones((3, 3), dtype=bool), iterations=int(bridge_iters))
    m = binary_opening(m, structure=_make_connectivity_structure(8))
    lab, n = ndi_label(m, structure=_make_connectivity_structure(8))
    out = np.zeros_like(m, dtype=bool)
    for cid in range(1, int(n) + 1):
        comp = lab == cid
        if int(np.sum(comp)) >= int(min_fragment_size):
            out |= comp
    return out


def _draw_mask_outline(ax, mask2d, color='white', linewidth=1.8):
    m = np.asarray(mask2d, dtype=bool)
    nrows, ncols = m.shape
    for r in range(nrows):
        for c in range(ncols):
            if not m[r, c]:
                continue
            if (r == 0) or (not m[r - 1, c]):
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5], color=color, lw=linewidth)
            if (r == nrows - 1) or (not m[r + 1, c]):
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=color, lw=linewidth)
            if (c == 0) or (not m[r, c - 1]):
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5], color=color, lw=linewidth)
            if (c == ncols - 1) or (not m[r, c + 1]):
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=color, lw=linewidth)


def find(parent, i):
    i = int(i)
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return int(parent[i])


def union(parent, rank, x, y):
    rx = find(parent, x)
    ry = find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1


def _extract_field_visits(session_pack, neuron_id, field_mask_flat, event_threshold=0.0, min_visit_frames=1, frame_rate_hz=25.0):
    field_mask_flat = np.asarray(field_mask_flat, dtype=bool)
    frames = np.asarray(session_pack['frames'], dtype=int)
    bins = np.asarray(session_pack['bins'], dtype=int)
    S = np.asarray(session_pack['S'], dtype=float)
    vals = np.asarray(S[:, int(neuron_id)], dtype=float)
    maze_time = np.asarray(session_pack.get('maze_time', frames.astype(float) / float(frame_rate_hz)), dtype=float)
    if maze_time.shape[0] != frames.shape[0]:
        maze_time = frames.astype(float) / float(frame_rate_hz)
    in_field = field_mask_flat[bins]
    T = len(in_field)
    visits = []
    traces = []
    pos = 0
    visit_idx = 0
    while pos < T:
        if not in_field[pos]:
            pos += 1
            continue
        start = pos
        while pos < T and in_field[pos]:
            pos += 1
        end = pos - 1
        n_frames = int(end - start + 1)
        if n_frames < int(min_visit_frames):
            continue
        visit_idx += 1
        tr = vals[start:end + 1]
        event_mask = np.asarray(tr > float(event_threshold), dtype=bool)
        has_event = bool(np.any(event_mask))
        first_event_local = int(np.where(event_mask)[0][0]) if has_event else -1
        visits.append(
            {
                'visit_idx': int(visit_idx),
                'start_row': int(start),
                'end_row': int(end),
                'n_frames': int(n_frames),
                'start_maze_time': float(maze_time[start]) if np.isfinite(maze_time[start]) else np.nan,
                'end_maze_time': float(maze_time[end]) if np.isfinite(maze_time[end]) else np.nan,
                'mean_activity': float(np.nanmean(tr)) if tr.size else np.nan,
                'peak_activity': float(np.nanmax(tr)) if tr.size else np.nan,
                'event_count': int(np.sum(event_mask)),
                'has_event': bool(has_event),
                'first_event_offset': (int(first_event_local) if has_event else np.nan),
                'first_event_abs_frame': (int(frames[start + first_event_local]) if has_event else np.nan),
            }
        )
        traces.append(np.asarray(tr, dtype=float))
    return pd.DataFrame(visits), traces


def _traces_to_matrix(traces):
    if len(traces) == 0:
        return np.empty((0, 0), dtype=float)
    max_len = int(max(len(t) for t in traces))
    mat = np.full((len(traces), max_len), np.nan, dtype=float)
    for i, tr in enumerate(traces):
        mat[i, :len(tr)] = tr
    return mat


def _first_event_visit(vis_df):
    if vis_df is None or vis_df.empty:
        return np.nan
    k = vis_df.loc[vis_df['has_event'], 'visit_idx']
    return float(k.iloc[0]) if len(k) else np.nan


def _cumulative_visit_maps(vis_df):
    if vis_df is None or vis_df.empty:
        return pd.DataFrame(columns=['visit_idx', 'cumulative_events'])
    out = vis_df[['visit_idx', 'has_event']].copy()
    out['cumulative_events'] = np.cumsum(out['has_event'].astype(int))
    return out[['visit_idx', 'cumulative_events']]


def _build_local_mask(mask2d, radius=1, connectivity=8):
    st = _make_connectivity_structure(connectivity)
    return binary_dilation(np.asarray(mask2d, dtype=bool), structure=st, iterations=max(0, int(radius)))


def _establishment_visit(vis_df, threshold=0.5, min_consecutive=2):
    if vis_df is None or vis_df.empty:
        return np.nan
    y = np.asarray(vis_df['has_event'].astype(float).to_numpy(), dtype=float)
    if y.size == 0:
        return np.nan
    sm = np.cumsum(y) / np.arange(1, len(y) + 1, dtype=float)
    return _first_sustained_hit(sm, threshold=threshold, min_consecutive=min_consecutive)


def _plot_accumulation_curve(ax, vis_df, title='', color='tab:blue', time_scale=1.0):
    if vis_df is None or vis_df.empty:
        ax.set_title(f'{title}: no visits')
        ax.set_xlabel('maze_time (s)')
        ax.set_ylabel('# visits')
        ax.grid(True, alpha=0.25)
        return
    t = np.asarray(vis_df['start_maze_time'], dtype=float)
    ok = np.isfinite(t)
    if not np.any(ok):
        t = np.asarray(vis_df['visit_idx'], dtype=float)
        ok = np.isfinite(t)
    t = t[ok] * float(time_scale)
    has_ev = np.asarray(vis_df.loc[ok, 'has_event'], dtype=bool)
    order = np.argsort(t)
    t = t[order]
    has_ev = has_ev[order]
    y = np.arange(1, len(t) + 1, dtype=float)
    ax.step(t, y, where='post', color=color, lw=2)
    ax.scatter(t, y, s=14, c=color, alpha=0.85)
    if np.any(has_ev):
        ax.scatter(t[has_ev], y[has_ev], s=30, c='red', marker='o', label='visit has event')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title(title)
    ax.set_xlabel('maze_time (s)')
    ax.set_ylabel('# visits')
    ax.grid(True, alpha=0.25)


def _plot_visit_heatmap(ax, vis_df, traces, title='', cmap='magma'):
    """Visit-stacked calcium heatmap: rows=visits, columns=frames, color=S amplitude."""
    if traces is None or len(traces) == 0:
        ax.set_title(f'{title}: no visits')
        ax.set_xlabel('frame within visit')
        ax.set_ylabel('visit')
        ax.grid(True, alpha=0.25)
        return
    mat = _traces_to_matrix(traces)
    if mat.size == 0:
        ax.set_title(f'{title}: no visits')
        return
    im = ax.imshow(mat, aspect='auto', cmap=cmap, interpolation='nearest',
                   origin='upper')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='S amplitude')
    # cyan dots at first event offset per visit
    if vis_df is not None and not vis_df.empty and 'first_event_offset' in vis_df.columns:
        for i, (_, row) in enumerate(vis_df.iterrows()):
            offset = row.get('first_event_offset', np.nan)
            if np.isfinite(offset):
                ax.plot(int(offset), i, 'o', color='cyan', ms=4,
                        markeredgecolor='white', markeredgewidth=0.3)
    ax.set_xlabel('frame within visit')
    ax.set_ylabel('visit')
    ax.set_title(title)


def render_pf_pair_page(
    obs_s1,
    obs_s2,
    field_s1,
    field_s2,
    *,
    visits_s1=None,
    visits_s2=None,
    traces_s1=None,
    traces_s2=None,
    title='Single-neuron place-field diagnostic',
):
    obs_s1 = np.asarray(obs_s1, dtype=float)
    obs_s2 = np.asarray(obs_s2, dtype=float)
    m1 = np.asarray(field_s1.get('mask2d', field_s1.get('mask')), dtype=bool)
    m2 = np.asarray(field_s2.get('mask2d', field_s2.get('mask')), dtype=bool)
    if m1.ndim == 1:
        m1 = m1.reshape(obs_s1.shape)
    if m2.ndim == 1:
        m2 = m2.reshape(obs_s2.shape)
    # Transpose for correct x/y orientation
    obs_s1 = obs_s1.T
    obs_s2 = obs_s2.T
    m1 = m1.T
    m2 = m2.T
    vmax = np.nanmax([np.nanmax(obs_s1), np.nanmax(obs_s2)])
    if (not np.isfinite(vmax)) or vmax <= 0:
        vmax = None
    # 3×2 layout: rate maps, visit heatmaps, accumulation curves
    has_heatmap = (traces_s1 is not None and len(traces_s1) > 0) or \
                  (traces_s2 is not None and len(traces_s2) > 0)
    n_rows = 3 if has_heatmap else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(11.5, 4.0 * n_rows))
    # Row 0: rate maps
    im0 = axes[0, 0].imshow(obs_s1, origin='upper', cmap='viridis', vmin=0, vmax=vmax)
    _draw_mask_outline(axes[0, 0], m1, color='tomato', linewidth=1.8)
    axes[0, 0].set_title(f"S1 field {field_s1.get('field_idx', '?')}")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im1 = axes[0, 1].imshow(obs_s2, origin='upper', cmap='viridis', vmin=0, vmax=vmax)
    _draw_mask_outline(axes[0, 1], m2, color='cyan', linewidth=1.8)
    axes[0, 1].set_title(f"S2 field {field_s2.get('field_idx', '?')}")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    if has_heatmap:
        # Row 1: visit-stacked calcium heatmaps
        _plot_visit_heatmap(axes[1, 0], visits_s1, traces_s1, title='S1 visit heatmap')
        _plot_visit_heatmap(axes[1, 1], visits_s2, traces_s2, title='S2 visit heatmap')
        # Row 2: accumulation curves
        _plot_accumulation_curve(axes[2, 0], visits_s1, title='S1 accumulation', color='tomato', time_scale=1.0)
        _plot_accumulation_curve(axes[2, 1], visits_s2, title='S2 accumulation', color='cyan', time_scale=1.0)
    else:
        # Row 1: accumulation curves (no heatmap data)
        _plot_accumulation_curve(axes[1, 0], visits_s1, title='S1 accumulation', color='tomato', time_scale=1.0)
        _plot_accumulation_curve(axes[1, 1], visits_s2, title='S2 accumulation', color='cyan', time_scale=1.0)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig, axes


def _save_session_field_maps_pdf(pdf_path, session_label, neuron_ids, pf_dict, n_bins_x, n_bins_y, progress_every=25):
    rows = []
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.02, 0.98, f"{session_label} place maps with detected fields\nTotal neurons: {len(neuron_ids)}", va='top', ha='left', family='monospace', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        t0 = time.time()
        for idx, nid in enumerate(neuron_ids, start=1):
            rec = pf_dict[int(nid)]
            obs_map = np.asarray(rec['det']['obs_map'], dtype=float)
            fields = rec['fields']
            rows.append({'session': session_label, 'neuron_id': int(nid), 'n_fields': int(len(fields)), 'has_field': bool(len(fields) > 0)})
            if len(fields) == 0:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.0))
            im = ax.imshow(obs_map.reshape(n_bins_x, n_bins_y).T, origin='upper', cmap='viridis')
            for f in fields:
                m = np.asarray(f['mask'], dtype=bool).reshape(n_bins_x, n_bins_y).T
                _draw_mask_outline(ax, m, color='white', linewidth=1.0)
                cx = f.get('centroid_x', np.nan)
                cy = f.get('centroid_y', np.nan)
                if np.isfinite(cx) and np.isfinite(cy):
                    ax.text(float(cx), float(cy), str(int(f.get('field_idx', 0))), color='tomato', fontsize=8, ha='center', va='center')
            ax.set_title(f"{session_label} neuron={int(nid)} | n_fields={len(fields)}")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            if idx % int(progress_every) == 0:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-9)
                eta = (len(neuron_ids) - idx) / max(rate, 1e-9)
                print(f"{session_label} map export {idx}/{len(neuron_ids)} | elapsed={elapsed:.1f}s | ETA={eta:.1f}s")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase-D: allpairs PDF rendering + full report export
# ---------------------------------------------------------------------------

def render_pf_allpairs_page(pdf, pair_idx, train_id, test_id,
                            fields_s1, fields_s2, map_s1, map_s2,
                            match_df, session_pack_s1, session_pack_s2,
                            n_bins_x, n_bins_y, *, max_match_rows=4,
                            visits_data=None):
    """Render 2+ page sequence for one neuron pair in the allpairs PDF.

    Page 1: Field detection overview (S1 + S2 maps with field outlines).
    Page 2: Top matched fields table (up to *max_match_rows* rows).
    Pages 3+: One page per retained match with visit heatmaps + curves.
    """
    nbx, nby = int(n_bins_x), int(n_bins_y)
    map1 = np.asarray(map_s1, dtype=float).reshape(nbx, nby)
    map2 = np.asarray(map_s2, dtype=float).reshape(nbx, nby)
    vmax = np.nanmax([np.nanmax(map1), np.nanmax(map2)])
    if (not np.isfinite(vmax)) or vmax <= 0:
        vmax = None

    # --- Page 1: overview ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for ax, m, fields, label, cmap_color in [
        (axes[0], map1, fields_s1, f'S1 neuron={train_id}', 'tomato'),
        (axes[1], map2, fields_s2, f'S2 neuron={test_id}', 'cyan'),
    ]:
        im = ax.imshow(m.T, origin='upper', cmap='viridis', vmin=0, vmax=vmax)
        for f in fields:
            fm = np.asarray(f.get('mask2d', f.get('mask')), dtype=bool)
            if fm.ndim == 1:
                fm = fm.reshape(nbx, nby)
            _draw_mask_outline(ax, fm.T, color=cmap_color, linewidth=1.2)
            cx = f.get('centroid_x', np.nan)
            cy = f.get('centroid_y', np.nan)
            if np.isfinite(cx) and np.isfinite(cy):
                ax.text(float(cx), float(cy), str(int(f.get('field_idx', 0))),
                        color='white', fontsize=7, ha='center', va='center')
        ax.set_title(f'{label} | {len(fields)} fields')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Pair {pair_idx}: train={train_id}, test={test_id}', y=1.02)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # --- Page 2: match table ---
    if match_df is not None and len(match_df) > 0:
        show_df = match_df.head(int(max_match_rows))
        display_cols = ['field_idx_s1', 'field_idx_s2', 'match_score', 'iou',
                        'dice', 'overlap_bins', 'p_match', 'q_match', 'retained']
        show_cols = [c for c in display_cols if c in show_df.columns]
        fig, ax = plt.subplots(figsize=(11.5, 2.0 + 0.4 * len(show_df)))
        ax.axis('off')
        cell_text = []
        for _, row in show_df.iterrows():
            cell_text.append([
                f'{row[c]:.4f}' if isinstance(row[c], float) else str(row[c])
                for c in show_cols
            ])
        if cell_text:
            tbl = ax.table(cellText=cell_text, colLabels=show_cols,
                           loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.3)
        ax.set_title(f'Pair {pair_idx}: top {len(show_df)} matches '
                      f'({int(match_df["retained"].sum())} retained)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # --- Pages 3+: per retained match detail ---
    if match_df is not None:
        retained = match_df[match_df['retained']]
        for _, mrow in retained.iterrows():
            fi_s1 = int(mrow['field_idx_s1'])
            fi_s2 = int(mrow['field_idx_s2'])
            f1 = next((f for f in fields_s1 if f.get('field_idx', -1) == fi_s1), None)
            f2 = next((f for f in fields_s2 if f.get('field_idx', -1) == fi_s2), None)
            if f1 is None or f2 is None:
                continue
            vis_s1 = None
            vis_s2 = None
            tr_s1 = None
            tr_s2 = None
            if visits_data is not None:
                key = (pair_idx, fi_s1, fi_s2)
                vd = visits_data.get(key, {})
                vis_s1 = vd.get('vis_s1')
                vis_s2 = vd.get('vis_s2')
                tr_s1 = vd.get('traces_s1')
                tr_s2 = vd.get('traces_s2')
            title = (f'Pair {pair_idx} | match f{fi_s1}-f{fi_s2} | '
                     f'score={mrow["match_score"]:.3f} iou={mrow["iou"]:.3f}')
            fig, _ = render_pf_pair_page(
                map1, map2, f1, f2,
                visits_s1=vis_s1, visits_s2=vis_s2,
                traces_s1=tr_s1, traces_s2=tr_s2,
                title=title)
            pdf.savefig(fig)
            plt.close(fig)


def save_pf_pairs_report(out_dir, tag, cand_df, run_meta,
                         pf_s1, pf_s2, session_pack_s1, session_pack_s2,
                         n_bins_x, n_bins_y, *,
                         visits_data=None, progress_every=25,
                         max_pairs=None, establishment_df=None):
    """Export full artifact bundle.

    Writes:
      pf_pairs_{tag}.csv
      pf_pairs_{tag}_config.json
      pf_pairs_cell15_cell16_allpairs_{tag}.pdf
      pf_maps_with_fields_s1_{tag}.pdf
      pf_maps_with_fields_s2_{tag}.pdf
      pf_maps_field_summary_{tag}.csv
      pf_pair_visits_{tag}.pkl / .parquet
      pf_pairs_{tag}_establishment.csv  (if establishment_df provided)
    """
    import json
    import pickle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nbx, nby = int(n_bins_x), int(n_bins_y)

    # --- CSV --- reorder columns to match legacy format
    csv_path = out_dir / f'pf_pairs_{tag}.csv'
    _legacy_col_order = [
        'pair_idx', 'train_id', 'test_id',
        'field_idx_s1', 'field_idx_s2', 'level',
        'area_s1', 'area_s2', 'peak_s1', 'peak_s2',
        'centroid_x_s1', 'centroid_y_s1', 'centroid_x_s2', 'centroid_y_s2',
        'occupancy_support_s1', 'occupancy_support_s2',
        'half_corr_s1', 'half_corr_s2',
        'iou', 'dice', 'centroid_dist', 'local_corr',
        'overlap_bins', 'match_score', 'p_match', 'q_match',
        'retained', 'null_n', 'null_mean', 'null_std',
    ]
    _out_cols = [c for c in _legacy_col_order if c in cand_df.columns]
    _extra = [c for c in cand_df.columns if c not in _out_cols]
    _out_cols += _extra
    cand_df[_out_cols].to_csv(csv_path, index=False)
    print(f'  CSV: {csv_path} ({len(cand_df)} rows)')

    # --- Config JSON ---
    json_path = out_dir / f'pf_pairs_{tag}_config.json'
    with open(json_path, 'w') as f:
        json.dump(run_meta, f, indent=2, default=str)
    print(f'  Config: {json_path}')

    # --- Visits artifact ---
    if visits_data is not None:
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet as _pq  # noqa: F401
            _HAS_PARQUET = True
        except ImportError:
            _HAS_PARQUET = False
        if _HAS_PARQUET:
            vis_path = out_dir / f'pf_pair_visits_{tag}.parquet'
            # flatten visits_data to DataFrame
            vis_rows = []
            for key, vd in visits_data.items():
                pair_idx, fi_s1, fi_s2 = key
                for side, vdf in [('s1', vd.get('vis_s1')), ('s2', vd.get('vis_s2'))]:
                    if vdf is not None and len(vdf) > 0:
                        tmp = vdf.copy()
                        tmp['pair_idx'] = pair_idx
                        tmp['field_idx_s1'] = fi_s1
                        tmp['field_idx_s2'] = fi_s2
                        tmp['side'] = side
                        vis_rows.append(tmp)
            if vis_rows:
                vis_all = pd.concat(vis_rows, ignore_index=True)
                vis_all.to_parquet(vis_path, index=False)
            else:
                pd.DataFrame().to_parquet(vis_path, index=False)
            print(f'  Visits: {vis_path}')
        else:
            vis_path = out_dir / f'pf_pair_visits_{tag}.pkl'
            with open(vis_path, 'wb') as f:
                pickle.dump(visits_data, f, protocol=4)
            print(f'  Visits: {vis_path}')

    # --- Establishment sidecar ---
    if establishment_df is not None and len(establishment_df) > 0:
        est_path = out_dir / f'pf_pairs_{tag}_establishment.csv'
        establishment_df.to_csv(est_path, index=False)
        print(f'  Establishment: {est_path} ({len(establishment_df)} rows)')

    # --- Per-session field map PDFs ---
    s1_label = run_meta.get('session1_id', 's1')
    s2_label = run_meta.get('session2_id', 's2')

    s1_nids = sorted(pf_s1.keys())
    s2_nids = sorted(pf_s2.keys())
    s1_pdf_path = out_dir / f'pf_maps_with_fields_s1_{tag}.pdf'
    s2_pdf_path = out_dir / f'pf_maps_with_fields_s2_{tag}.pdf'
    summary1 = _save_session_field_maps_pdf(s1_pdf_path, s1_label, s1_nids,
                                            pf_s1, nbx, nby,
                                            progress_every=progress_every)
    summary2 = _save_session_field_maps_pdf(s2_pdf_path, s2_label, s2_nids,
                                            pf_s2, nbx, nby,
                                            progress_every=progress_every)
    summary = pd.concat([summary1, summary2], ignore_index=True)
    summary_path = out_dir / f'pf_maps_field_summary_{tag}.csv'
    summary.to_csv(summary_path, index=False)
    print(f'  Field summary: {summary_path}')

    # --- Allpairs PDF ---
    allpairs_pdf_path = out_dir / f'pf_pairs_cell15_cell16_allpairs_{tag}.pdf'
    print(f'  Allpairs PDF: {allpairs_pdf_path}')

    # group candidate rows by pair_idx
    if 'pair_idx' not in cand_df.columns:
        cand_df = cand_df.copy()
        cand_df['pair_idx'] = 0
    pair_groups = cand_df.groupby('pair_idx')
    pair_indices = sorted(pair_groups.groups.keys())
    if max_pairs is not None and int(max_pairs) > 0:
        pair_indices = pair_indices[:int(max_pairs)]

    with PdfPages(allpairs_pdf_path) as pdf:
        # title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        total_retained = int(cand_df['retained'].sum()) if 'retained' in cand_df.columns else 0
        ax.text(0.02, 0.98,
                f"Place Field Pair Analysis\n"
                f"Session 1: {s1_label}\n"
                f"Session 2: {s2_label}\n"
                f"Total pairs: {len(pair_indices)}\n"
                f"Total candidates: {len(cand_df)}\n"
                f"Retained: {total_retained}",
                va='top', ha='left', family='monospace', fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        t0 = time.time()
        for count, pidx in enumerate(pair_indices, 1):
            grp = pair_groups.get_group(pidx)
            train_id = int(grp['train_id'].iloc[0]) if 'train_id' in grp.columns else 0
            test_id = int(grp['test_id'].iloc[0]) if 'test_id' in grp.columns else 0
            f_s1 = pf_s1.get(train_id, {}).get('fields', [])
            f_s2 = pf_s2.get(test_id, {}).get('fields', [])
            m_s1 = pf_s1.get(train_id, {}).get('det', {}).get('obs_map', np.zeros(nbx * nby))
            m_s2 = pf_s2.get(test_id, {}).get('det', {}).get('obs_map', np.zeros(nbx * nby))

            render_pf_allpairs_page(
                pdf, pidx, train_id, test_id,
                f_s1, f_s2, m_s1, m_s2,
                grp, session_pack_s1, session_pack_s2,
                nbx, nby, visits_data=visits_data)

            if count % int(progress_every) == 0:
                elapsed = time.time() - t0
                rate = count / max(elapsed, 1e-9)
                eta = (len(pair_indices) - count) / max(rate, 1e-9)
                print(f'  Allpairs PDF {count}/{len(pair_indices)} '
                      f'| elapsed={elapsed:.1f}s | ETA={eta:.1f}s')

    print(f'  Done: {len(pair_indices)} pairs rendered')
