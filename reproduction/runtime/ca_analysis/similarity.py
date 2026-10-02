from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, zscore, rankdata
from .maze import NewMaze
from .spatial import build_count_matrix, discretize_positions
import json
import os
import pickle
import time
from collections import deque
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.ndimage import gaussian_filter1d


def _shuffle_rows_independent(X, rng):
    """Shuffle row order independently for each column."""
    X = np.asarray(X)
    try:
        return rng.permuted(X, axis=0)
    except AttributeError:
        (K, M) = X.shape
        perm_idx = rng.random((K, M)).argsort(axis=0)
        return np.take_along_axis(X, perm_idx, axis=0)


def _spatial_labels_from_inputs(selected_df, positions, *, n_bins_x, n_bins_y):
    """Prefer maze-aware binned columns when available, else discretize positions."""
    if (
        n_bins_y > 1
        and {"mass_bx", "mass_by"}.issubset(selected_df.columns)
        and len(selected_df) > 0
    ):
        bx = pd.to_numeric(selected_df["mass_bx"], errors="coerce").to_numpy(dtype=float)
        by = pd.to_numeric(selected_df["mass_by"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(bx).all() and np.isfinite(by).all():
            bx = np.clip(np.floor(bx).astype(int), 0, int(n_bins_x) - 1)
            by = np.clip(np.floor(by).astype(int), 0, int(n_bins_y) - 1)
            labels = bx + by * int(n_bins_x)
            x_edges = np.arange(int(n_bins_x) + 1, dtype=float)
            y_edges = np.arange(int(n_bins_y) + 1, dtype=float)
            return labels, x_edges, y_edges, "mass_bx_mass_by"
    labels, x_edges, y_edges = discretize_positions(positions, n_bins_x=n_bins_x, n_bins_y=n_bins_y)
    return labels, x_edges, y_edges, "discretize_positions"


def _maze_reachable_mask(n_bins_x, n_bins_y, maze_id):
    if maze_id is None:
        return None
    maze = NewMaze(int(maze_id))
    mask = np.zeros(int(n_bins_x) * int(n_bins_y), dtype=bool)
    for by, bx in maze.ce.keys():
        if 0 <= int(bx) < int(n_bins_x) and 0 <= int(by) < int(n_bins_y):
            mask[int(bx) + int(by) * int(n_bins_x)] = True
    return mask


def _occupancy_entropy(prior, eps=1e-12):
    p = np.asarray(prior, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(np.maximum(p, eps))).sum())


def _mean_jaccard_by_lag(masks, lags=(1, 5, 10, 20)):
    masks = [np.asarray(m, dtype=bool) for m in masks]
    out = []
    n = len(masks)
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            continue
        vals = []
        for i in range(0, n - lag):
            a = masks[i]
            b = masks[i + lag]
            union = np.count_nonzero(a | b)
            vals.append(np.nan if union == 0 else float(np.count_nonzero(a & b) / union))
        out.append(
            {
                "lag": lag,
                "n_pairs": int(len(vals)),
                "mean_jaccard": (
                    float(np.nanmean(vals))
                    if len(vals) and np.any(np.isfinite(vals))
                    else np.nan
                ),
            }
        )
    return out


def _linear_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(keep) < 2:
        return np.nan
    xv = x[keep]
    yv = y[keep]
    X = np.column_stack([np.ones_like(xv), xv])
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    return float(beta[1])


def _resolve_min_time_threshold(
    *,
    min_time_per_bin_sec,
    min_time_policy='fixed',
    time_axis=None,
    win_len_sec=None,
    dt_eff=None,
    n_samples=None,
):
    base = float(min_time_per_bin_sec)
    policy = str(min_time_policy)
    if policy == 'fixed':
        return base, 1
    if policy != 'scaled_nonoverlap':
        raise ValueError("min_time_policy must be 'fixed' or 'scaled_nonoverlap'.")

    n_nonoverlap = 1
    W = float(win_len_sec) if win_len_sec is not None else np.nan
    if time_axis is not None:
        t = np.asarray(time_axis, dtype=float)
        finite = np.isfinite(t)
        if np.count_nonzero(finite) >= 2 and np.isfinite(W) and W > 0:
            t_use = t[finite]
            span = float(t_use[-1] - t_use[0])
            if span > 0:
                n_nonoverlap = max(1, int(np.floor(span / W)))
    elif dt_eff is not None and n_samples is not None:
        dt_eff = float(dt_eff)
        n_samples = int(max(int(n_samples), 0))
        if np.isfinite(dt_eff) and dt_eff > 0 and np.isfinite(W) and W > 0:
            approx_span = float(n_samples) * dt_eff
            n_nonoverlap = max(1, int(np.floor(approx_span / W)))

    return float(base / float(n_nonoverlap)), int(n_nonoverlap)


def _normalize_support_prior(prior, support_mask, *, eps=1e-12):
    prior = np.asarray(prior, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    out = np.where(support_mask, prior, 0.0)
    denom = float(np.nansum(out))
    if denom > eps:
        out = out / denom
    else:
        out = np.zeros_like(prior, dtype=float)
    return out


def _uniform_support_prior(support_mask):
    support_mask = np.asarray(support_mask, dtype=bool)
    out = np.zeros(support_mask.shape, dtype=float)
    count = int(np.count_nonzero(support_mask))
    if count > 0:
        out[support_mask] = 1.0 / float(count)
    return out


def _safe_total_variation(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    return float(0.5 * np.nansum(np.abs(p - q)))


def _safe_js_divergence(p, q, *, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    if not np.any(p) and not np.any(q):
        return np.nan
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p_sum = float(np.nansum(p))
    q_sum = float(np.nansum(q))
    if p_sum <= eps or q_sum <= eps:
        return np.nan
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    keep_p = (p > eps) & (m > eps)
    keep_q = (q > eps) & (m > eps)
    kl_pm = float(np.nansum(p[keep_p] * np.log2(p[keep_p] / m[keep_p])))
    kl_qm = float(np.nansum(q[keep_q] * np.log2(q[keep_q] / m[keep_q])))
    return float(0.5 * (kl_pm + kl_qm))


def _circular_shift_columns(X, shifts):
    X = np.asarray(X, dtype=float)
    shifts = np.asarray(shifts, dtype=int).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if shifts.size != X.shape[1]:
        raise ValueError("shifts length must match X columns")
    out = np.empty_like(X)
    for j, shift in enumerate(shifts.tolist()):
        out[:, j] = np.roll(X[:, j], int(shift))
    return out


def compute_tuning_and_prior(spike_counts, labels, head_labels=None, n_bins=15 * 15, dt=0.04, filter_ratemap=False, sigma=1.0, min_time=5.0, eps=1e-12, decode_head='none', labels_lvl=None, n_lvls=None, spatial_shape=None):
    if decode_head == 'joint':
        decode_head = 'joint_pos'
    (T, N) = spike_counts.shape

    def _get_spatial_layout(n_bins, spatial_shape):
        """
        Returns (mode, shape) where:
          mode='1d', shape=(K,)
          mode='2d', shape=(nx, ny)
        """
        if spatial_shape is not None:
            spatial_shape = tuple((int(s) for s in spatial_shape))
            prod = 1
            for s in spatial_shape:
                prod *= s
            if prod != n_bins:
                raise ValueError(f'spatial_shape {spatial_shape} has product {prod}, but n_bins={n_bins}.')
            if len(spatial_shape) == 1:
                return ('1d', spatial_shape)
            elif len(spatial_shape) == 2:
                return ('2d', spatial_shape)
            else:
                raise ValueError(f'spatial_shape must be 1D (K,) or 2D (nx, ny), got {spatial_shape}.')
        side = int(np.sqrt(n_bins))
        if side * side == n_bins:
            return ('2d', (side, side))
        else:
            return ('1d', (n_bins,))
    (spatial_mode, spatial_shape) = _get_spatial_layout(n_bins, spatial_shape)
    occupancy = np.zeros(n_bins, dtype=float)
    count_map = np.zeros((n_bins, N), dtype=float)
    for b in range(n_bins):
        mb = labels == b
        if mb.any():
            occupancy[b] = mb.sum() * dt
            count_map[b] = spike_counts[mb].sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lambdas_spatial = count_map / np.maximum(occupancy[:, None], eps)
    low_occ = occupancy < min_time
    if np.any(low_occ):
        lambdas_spatial[low_occ] = 0.0
        occupancy[low_occ] = 0.0
    if filter_ratemap:
        sm = np.zeros_like(lambdas_spatial)
        if spatial_mode == '2d':
            (nx, ny) = spatial_shape
            for i in range(N):
                grid = lambdas_spatial[:, i].reshape(nx, ny)
                sm[:, i] = gaussian_filter(grid, sigma=sigma, mode='constant').ravel()
        else:
            for i in range(N):
                sm[:, i] = gaussian_filter(lambdas_spatial[:, i], sigma=sigma, mode='constant')
        lambdas_spatial = sm
    lambdas_spatial = lambdas_spatial + eps
    prior_spatial = np.zeros(n_bins, dtype=float)
    if occupancy.sum() > 0:
        prior_spatial = occupancy / occupancy.sum()
    if decode_head == 'none':
        return (lambdas_spatial, prior_spatial)
    if head_labels is None:
        raise ValueError("head_labels is required when decode_head!='none'.")
    n_head = int(np.max(head_labels)) + 1
    if decode_head == 'independent':
        occ_hd = np.zeros(n_head, dtype=float)
        cnt_hd = np.zeros((n_head, N), dtype=float)
        for h in range(n_head):
            mh = head_labels == h
            if mh.any():
                occ_hd[h] = mh.sum() * dt
                cnt_hd[h] = spike_counts[mh].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambdas_hd = cnt_hd / np.maximum(occ_hd[:, None], eps)
        low_hd = occ_hd < min_time
        if np.any(low_hd):
            lambdas_hd[low_hd] = 0.0
            occ_hd[low_hd] = 0.0
        lambdas_hd = lambdas_hd + eps
        prior_hd = np.zeros(n_head, dtype=float)
        if occ_hd.sum() > 0:
            prior_hd = occ_hd / occ_hd.sum()
        return (lambdas_spatial, prior_spatial, lambdas_hd, prior_hd)
    if decode_head == 'joint_pos':
        occ_j = np.zeros((n_bins, n_head), dtype=float)
        cnt_j = np.zeros((n_bins, n_head, N), dtype=float)
        for b in range(n_bins):
            mb = labels == b
            if not mb.any():
                continue
            hb = head_labels[mb]
            Xb = spike_counts[mb]
            for h in range(n_head):
                mh = hb == h
                if mh.any():
                    occ_j[b, h] = mh.sum() * dt
                    cnt_j[b, h] = Xb[mh].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambdas_j = cnt_j / np.maximum(occ_j[..., None], eps)
        low_j = occ_j < max(min_time / 8.0, dt)
        if np.any(low_j):
            lambdas_j[low_j] = 0.0
            occ_j[low_j] = 0.0
        if filter_ratemap:
            for h in range(n_head):
                for i in range(N):
                    vec = lambdas_j[:, h, i]
                    if spatial_mode == '2d':
                        (nx, ny) = spatial_shape
                        grid = vec.reshape(nx, ny)
                        lambdas_j[:, h, i] = gaussian_filter(grid, sigma=sigma, mode='constant').ravel()
                    else:
                        lambdas_j[:, h, i] = gaussian_filter(vec, sigma=sigma, mode='constant')
        lambdas_j = lambdas_j + eps
        prior_j = np.zeros_like(occ_j)
        tot = occ_j.sum()
        if tot > 0:
            prior_j = occ_j / tot
        return (lambdas_j, prior_j)
    if decode_head == 'joint_lvl':
        if labels_lvl is None:
            raise ValueError("labels_lvl is required for decode_head='joint_lvl'.")
        if n_lvls is None:
            n_lvls = int(np.max(labels_lvl)) + 1
        occ_lh = np.zeros((n_lvls, n_head), dtype=float)
        cnt_lh = np.zeros((n_lvls, n_head, N), dtype=float)
        for l in range(n_lvls):
            ml = labels_lvl == l
            if not ml.any():
                continue
            hl = head_labels[ml]
            Xl = spike_counts[ml]
            for h in range(n_head):
                mh = hl == h
                if mh.any():
                    occ_lh[l, h] = mh.sum() * dt
                    cnt_lh[l, h] = Xl[mh].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            lambdas_lh = cnt_lh / np.maximum(occ_lh[..., None], eps)
        low_lh = occ_lh < max(min_time / 8.0, dt)
        if np.any(low_lh):
            lambdas_lh[low_lh] = 0.0
            occ_lh[low_lh] = 0.0
        lambdas_lh = lambdas_lh + eps
        prior_lh = np.zeros_like(occ_lh)
        tot = occ_lh.sum()
        if tot > 0:
            prior_lh = occ_lh / tot
        return (lambdas_spatial, prior_spatial, lambdas_lh, prior_lh)
    raise ValueError(f'Unknown decode_head mode: {decode_head}')

def compute_tuning_and_prior_peak(spike_counts, labels, head_labels=None, n_bins=15 * 15, dt=0.04, filter_ratemap=False, sigma=1.0, min_visits=10, eps=1e-12, decode_head='none', labels_lvl=None, n_lvls=None, spatial_shape=None, min_time=None):
    if decode_head == 'joint':
        decode_head = 'joint_pos'
    (T, N) = spike_counts.shape

    def _get_spatial_layout(n_bins, spatial_shape):
        if spatial_shape is not None:
            spatial_shape = tuple((int(s) for s in spatial_shape))
            prod = 1
            for s in spatial_shape:
                prod *= s
            if prod != n_bins:
                raise ValueError(f'spatial_shape {spatial_shape} has product {prod}, but n_bins={n_bins}.')
            if len(spatial_shape) == 1:
                return ('1d', spatial_shape)
            elif len(spatial_shape) == 2:
                return ('2d', spatial_shape)
            else:
                raise ValueError(f'spatial_shape must be 1D (K,) or 2D (nx, ny), got {spatial_shape}.')
        side = int(np.sqrt(n_bins))
        if side * side == n_bins:
            return ('2d', (side, side))
        else:
            return ('1d', (n_bins,))
    (spatial_mode, spatial_shape) = _get_spatial_layout(n_bins, spatial_shape)

    def build_runs(key_fn):
        runs = []
        s = 0
        prev = key_fn(0)
        for t in range(1, T):
            cur = key_fn(t)
            if cur != prev:
                runs.append((prev, s, t))
                (prev, s) = (cur, t)
        runs.append((prev, s, T))
        return runs
    runs_pos = build_runs(lambda t: (int(labels[t]),))
    lam_sp = np.zeros((n_bins, N), dtype=float)
    visits_sp = np.zeros(n_bins, dtype=int)
    for (key, s, e) in runs_pos:
        b = key[0]
        seg = spike_counts[s:e]
        if seg.size == 0:
            continue
        lam_sp[b] += np.nanmax(seg, axis=0)
        visits_sp[b] += 1
    occupancy_sp = np.bincount(np.asarray(labels, dtype=int), minlength=n_bins).astype(float) * float(dt)
    ok_sp = visits_sp >= min_visits
    if min_time is not None:
        ok_sp &= occupancy_sp >= float(min_time)
    lam_sp[ok_sp] = lam_sp[ok_sp] / np.maximum(visits_sp[ok_sp, None], 1)
    lam_sp[~ok_sp] = 0.0
    lam_sp = lam_sp / dt
    if filter_ratemap:
        sm = np.zeros_like(lam_sp)
        if spatial_mode == '2d':
            (nx, ny) = spatial_shape
            for i in range(N):
                sm[:, i] = gaussian_filter(lam_sp[:, i].reshape(nx, ny), sigma, mode='reflect').ravel()
        else:
            for i in range(N):
                sm[:, i] = gaussian_filter(lam_sp[:, i], sigma, mode='reflect')
        lam_sp = sm
    lam_sp += eps
    prior_sp = np.zeros(n_bins, dtype=float)
    if np.any(ok_sp):
        v = visits_sp.astype(float).copy()
        v[~ok_sp] = 0.0
        s = v.sum()
        if s > 0:
            prior_sp = v / s
    if decode_head == 'none':
        return (lam_sp, prior_sp)
    if head_labels is None:
        raise ValueError("head_labels is required when decode_head!='none'.")
    n_head = int(np.max(head_labels)) + 1
    if decode_head in ('independent', 'joint_pos', 'joint_lvl'):
        runs_hd = build_runs(lambda t: (int(head_labels[t]),))
        lam_hd = np.zeros((n_head, N), dtype=float)
        visits_hd = np.zeros(n_head, dtype=int)
        for (key, s, e) in runs_hd:
            h = key[0]
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_hd[h] += np.nanmax(seg, axis=0)
            visits_hd[h] += 1
        th_hd = max(3, min_visits // 4)
        ok_hd = visits_hd >= th_hd
        if min_time is not None:
            occupancy_hd = np.bincount(np.asarray(head_labels, dtype=int), minlength=n_head).astype(float) * float(dt)
            ok_hd &= occupancy_hd >= max(float(min_time) / 8.0, float(dt))
        lam_hd[ok_hd] = lam_hd[ok_hd] / np.maximum(visits_hd[ok_hd, None], 1)
        lam_hd[~ok_hd] = 0.0
        lam_hd = lam_hd / dt + eps
        prior_hd = np.zeros(n_head, dtype=float)
        v = visits_hd.astype(float).copy()
        v[~ok_hd] = 0.0
        s = v.sum()
        if s > 0:
            prior_hd = v / s
        if decode_head == 'independent':
            return (lam_sp, prior_sp, lam_hd, prior_hd)
    if decode_head == 'joint_pos':
        runs_pxh = build_runs(lambda t: (int(labels[t]), int(head_labels[t])))
        lam_pxh = np.zeros((n_bins, n_head, N), dtype=float)
        visits_pxh = np.zeros((n_bins, n_head), dtype=int)
        for (key, s, e) in runs_pxh:
            (b, h) = key
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_pxh[b, h] += np.nanmax(seg, axis=0)
            visits_pxh[b, h] += 1
        th_j = max(2, min_visits // 5)
        ok_j = visits_pxh >= th_j
        if min_time is not None:
            occupancy_pxh = np.zeros((n_bins, n_head), dtype=float)
            np.add.at(
                occupancy_pxh,
                (np.asarray(labels, dtype=int), np.asarray(head_labels, dtype=int)),
                float(dt),
            )
            ok_j &= occupancy_pxh >= max(float(min_time) / 8.0, float(dt))
        lam_pxh[ok_j] = lam_pxh[ok_j] / np.maximum(visits_pxh[ok_j, None], 1)
        lam_pxh[~ok_j] = 0.0
        lam_pxh = lam_pxh / dt
        if filter_ratemap:
            for h in range(n_head):
                for i in range(N):
                    vec = lam_pxh[:, h, i]
                    if spatial_mode == '2d':
                        (nx, ny) = spatial_shape
                        lam_pxh[:, h, i] = gaussian_filter(vec.reshape(nx, ny), sigma, mode='reflect').ravel()
                    else:
                        lam_pxh[:, h, i] = gaussian_filter(vec, sigma, mode='reflect')
        lam_pxh += eps
        prior_pxh = np.zeros_like(visits_pxh, dtype=float)
        vv = visits_pxh.astype(float).copy()
        vv[~ok_j] = 0.0
        s = vv.sum()
        if s > 0:
            prior_pxh = vv / s
        return (lam_pxh, prior_pxh)
    if decode_head == 'joint_lvl':
        if labels_lvl is None:
            raise ValueError("labels_lvl is required for decode_head='joint_lvl'.")
        if n_lvls is None:
            n_lvls = int(np.max(labels_lvl)) + 1
        runs_lxh = build_runs(lambda t: (int(labels_lvl[t]), int(head_labels[t])))
        lam_lxh = np.zeros((n_lvls, n_head, N), dtype=float)
        visits_lxh = np.zeros((n_lvls, n_head), dtype=int)
        for (key, s, e) in runs_lxh:
            (l, h) = key
            seg = spike_counts[s:e]
            if seg.size == 0:
                continue
            lam_lxh[l, h] += np.nanmax(seg, axis=0)
            visits_lxh[l, h] += 1
        th_lh = max(2, min_visits // 5)
        ok_lh = visits_lxh >= th_lh
        if min_time is not None:
            occupancy_lxh = np.zeros((n_lvls, n_head), dtype=float)
            np.add.at(
                occupancy_lxh,
                (np.asarray(labels_lvl, dtype=int), np.asarray(head_labels, dtype=int)),
                float(dt),
            )
            ok_lh &= occupancy_lxh >= max(float(min_time) / 8.0, float(dt))
        lam_lxh[ok_lh] = lam_lxh[ok_lh] / np.maximum(visits_lxh[ok_lh, None], 1)
        lam_lxh[~ok_lh] = 0.0
        lam_lxh = lam_lxh / dt + eps
        prior_lxh = np.zeros_like(visits_lxh, dtype=float)
        vv = visits_lxh.astype(float).copy()
        vv[~ok_lh] = 0.0
        s = vv.sum()
        if s > 0:
            prior_lxh = vv / s
        return (lam_sp, prior_sp, lam_lxh, prior_lxh)
    raise ValueError(f'Unknown decode_head mode: {decode_head}')

def _to_int_array(x):
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.ravel()
    return x.astype(int, copy=False)

def _pick_matched_neurons(cimap, n_neurons=None, random_state=0, n_train=None, n_test=None):
    rng = np.random.RandomState(random_state)
    if cimap is None:
        if n_train is None or n_test is None:
            raise ValueError('When cimap=None, provide n_train and n_test.')
        K = min(int(n_train), int(n_test))
        train_all = np.arange(K, dtype=int)
        test_all = np.arange(K, dtype=int)
    else:
        pairs = np.asarray(cimap, dtype=int)
        if pairs.min() >= 0:
            pairs = pairs - 1
        valid = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
        # Enforce session-specific bounds when provided.
        if n_test is not None:
            valid &= pairs[:, 0] < int(n_test)
        if n_train is not None:
            valid &= pairs[:, 1] < int(n_train)
        (test_all, train_all) = (pairs[valid, 0], pairs[valid, 1])
    if n_neurons is not None and len(train_all) > n_neurons:
        sel = rng.choice(len(train_all), int(n_neurons), replace=False)
        (train_all, test_all) = (train_all[sel], test_all[sel])
    return (train_all, test_all)

def _level_values_to_labels(values, level_categories=None):
    raw = np.asarray(values)
    labels = np.full(raw.shape[0], -1, dtype=int)
    if level_categories is None:
        series = pd.Series(raw)
        valid = series.notna().to_numpy(dtype=bool)
        uniq = np.unique(series[valid].to_numpy()) if np.any(valid) else np.array([], dtype=int)
        lvl_map = {v: i for (i, v) in enumerate(uniq)}
        for i, value in enumerate(raw):
            if pd.notna(value):
                labels[i] = lvl_map.get(value, -1)
        return labels, uniq, int(np.sum(labels < 0))

    uniq = np.asarray(level_categories)
    cats_numeric = pd.to_numeric(pd.Series(uniq), errors='coerce').to_numpy(dtype=float)
    values_numeric = pd.to_numeric(pd.Series(raw), errors='coerce').to_numpy(dtype=float)
    if np.all(np.isfinite(cats_numeric)):
        lvl_map = {float(v): i for (i, v) in enumerate(cats_numeric)}
        for i, value in enumerate(values_numeric):
            if np.isfinite(value):
                labels[i] = lvl_map.get(float(value), -1)
    else:
        lvl_map = {v: i for (i, v) in enumerate(uniq)}
        for i, value in enumerate(raw):
            if pd.notna(value):
                labels[i] = lvl_map.get(value, -1)
    return labels, uniq, int(np.sum(labels < 0))


def _fit_maps_for_session(spike_counts, raw_counts, selected_df, positions, neuron_ids, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, eps=1e-12, spatial_labels=None, level_categories=None):
    positions_arr = np.asarray(positions)
    is_1d_pos = positions_arr.ndim == 1 or (positions_arr.ndim == 2 and positions_arr.shape[1] == 1)
    if is_1d_pos and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 so that n_bins = n_bins_x. (Current n_bins_y != 1.)')
    n_bins = n_bins_x * n_bins_y
    dropped_invalid_spatial_labels = 0
    if spatial_labels is None:
        (labels_pos, x_edges, y_edges) = discretize_positions(positions_arr, n_bins_x, n_bins_y)
    else:
        labels_raw = np.asarray(spatial_labels).ravel()
        if labels_raw.shape[0] != selected_df.shape[0]:
            raise ValueError('spatial_labels length must match selected_df rows.')
        labels_float = pd.to_numeric(pd.Series(labels_raw), errors='coerce').to_numpy(dtype=float)
        valid_spatial = np.isfinite(labels_float) & (labels_float >= 0) & (labels_float < n_bins)
        dropped_invalid_spatial_labels = int(valid_spatial.size - np.sum(valid_spatial))
        labels_pos = np.full(labels_float.shape, -1, dtype=int)
        labels_pos[valid_spatial] = labels_float[valid_spatial].astype(int, copy=False)
        if dropped_invalid_spatial_labels:
            labels_pos = labels_float[valid_spatial].astype(int, copy=False)
            selected_df = selected_df.loc[valid_spatial].reset_index(drop=True)
            positions_arr = positions_arr[valid_spatial]
            spike_counts = np.asarray(spike_counts)[valid_spatial, :]
        x_edges = None
        y_edges = None
    if is_1d_pos:
        spatial_shape_pos = (n_bins,)
    else:
        spatial_shape_pos = (n_bins_x, n_bins_y)
    labels_lvl = None
    uniq_lvls = None
    n_lvls = 0
    if mode in ('level', 'joint_lvl'):
        if level_col not in selected_df.columns:
            raise ValueError(f'{mode} requires level_col in selected_df.')
        lv = selected_df[level_col].values
        labels_lvl, uniq_lvls, dropped_invalid_level_labels = _level_values_to_labels(lv, level_categories)
        n_lvls = len(uniq_lvls)
    else:
        dropped_invalid_level_labels = 0
    labels_hd = None
    n_hd = 0
    if mode in ('hd', 'joint_pos', 'joint_lvl'):
        if head_col not in selected_df.columns:
            raise ValueError(f'{mode} requires head_col in selected_df.')
        hv = selected_df[head_col].values
        uniq_hd = np.unique(hv)
        hd_map = {v: i for (i, v) in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[v] for v in hv], dtype=int)
        n_hd = len(uniq_hd)
    frames = np.arange(len(positions_arr))
    Q = build_count_matrix(spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids).T
    dt_eff = dt * (time_window + 1)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    if mode == 'spatial':
        (lam_sp, pr_sp) = tuner(Q, labels_pos, None, n_bins, dt_eff, filter_ratemap, sigma, eps=eps, decode_head='none', spatial_shape=spatial_shape_pos)
        maps = lam_sp
        valid_mask = pr_sp > 0
        meta = dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges, dropped_invalid_spatial_labels=dropped_invalid_spatial_labels)
    elif mode == 'level':
        (lam_lvl, pr_lvl) = compute_tuning_and_prior(Q, labels_lvl, None, n_lvls, dt_eff, False, 0.0, eps=eps, decode_head='none')
        maps = lam_lvl
        valid_mask = pr_lvl > 0
        meta = dict(n_lvls=n_lvls, uniq_lvls=uniq_lvls, dropped_invalid_level_labels=dropped_invalid_level_labels)
    elif mode == 'hd':
        (_, _, lam_hd, pr_hd) = tuner(Q, labels_pos, labels_hd, n_bins, dt_eff, filter_ratemap, sigma, eps=eps, decode_head='independent', spatial_shape=spatial_shape_pos)
        maps = lam_hd
        valid_mask = pr_hd > 0
        meta = dict(n_hd=n_hd)
    elif mode == 'joint_pos':
        (lam_jp, pr_jp) = tuner(Q, labels_pos, labels_hd, n_bins, dt_eff, filter_ratemap, sigma, eps=eps, decode_head='joint_pos', spatial_shape=spatial_shape_pos)
        maps = lam_jp.reshape(n_bins * n_hd, -1)
        valid_mask = pr_jp.ravel() > 0
        meta = dict(n_bins=n_bins, n_hd=n_hd, x_edges=x_edges, y_edges=y_edges)
    elif mode == 'joint_lvl':
        (_, _, lam_lh, pr_lh) = tuner(Q, labels_pos, labels_hd, n_bins, dt_eff, filter_ratemap, sigma, eps=eps, decode_head='joint_lvl', labels_lvl=labels_lvl, n_lvls=n_lvls, spatial_shape=spatial_shape_pos)
        maps = lam_lh.reshape(n_lvls * n_hd, -1)
        valid_mask = pr_lh.ravel() > 0
        meta = dict(n_lvls=n_lvls, n_hd=n_hd, uniq_lvls=uniq_lvls, dropped_invalid_level_labels=dropped_invalid_level_labels)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    maps = np.asarray(maps, float)
    maps[maps < eps] = eps
    meta['dropped_invalid_spatial_labels'] = int(dropped_invalid_spatial_labels)
    return (maps, valid_mask, meta)

def _vector_similarity(a, b, metric='pearson'):
    if metric == 'pearson':
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    elif metric == 'spearman':
        (r, _) = spearmanr(a, b)
        return float(r)
    elif metric == 'cosine':
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(np.dot(a, b) / (na * nb))
    else:
        raise ValueError(f'Unknown metric: {metric}')

def _normalize_columns(X, mode='l2', eps=1e-12):
    X = np.asarray(X, float).copy()
    if mode in (None, 'none'):
        return X
    if mode in ('l2', 'length'):
        norms = np.linalg.norm(X, axis=0)
        return X / (norms + eps)
    if mode == 'l1':
        norms = np.sum(np.abs(X), axis=0)
        return X / (norms + eps)
    if mode == 'zscore':
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        return (X - mu) / (sd + eps)
    if mode == 'max':
        max_vals = np.nanmax(np.abs(X), axis=0, keepdims=True)
        return X / (max_vals + eps)
    raise ValueError(f'Unknown normalization mode: {mode}')

def _pick_matched_neurons_for_level(cimap, n_train, n_test, n_neurons=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    if cimap is not None:
        pairs = np.array(cimap, int) - 1
        ok = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] < n_test) & (pairs[:, 1] < n_train)
        pairs = pairs[ok]
        if n_neurons is not None and len(pairs) > n_neurons:
            sel = rng.choice(len(pairs), n_neurons, replace=False)
            pairs = pairs[sel]
        train_ids = pairs[:, 1].astype(int)
        test_ids = pairs[:, 0].astype(int)
    else:
        m = n_train if n_neurons is None else min(n_neurons, n_train, n_test)
        ids = rng.choice(min(n_train, n_test), m, replace=False)
        train_ids = ids.astype(int)
        test_ids = ids.astype(int)
    return (train_ids, test_ids)

def _build_level_labels(selected_df_train, selected_df_test, level_col, level_categories=None):
    lv_tr_raw = selected_df_train[level_col].values
    lv_te_raw = selected_df_test[level_col].values
    if level_categories is None:
        labels_all, uniq_lvls, _ = _level_values_to_labels(np.concatenate([lv_tr_raw, lv_te_raw]))
        labels_tr = labels_all[: len(lv_tr_raw)]
        labels_te = labels_all[len(lv_tr_raw) :]
    else:
        labels_tr, uniq_lvls, _ = _level_values_to_labels(lv_tr_raw, level_categories)
        labels_te, _, _ = _level_values_to_labels(lv_te_raw, level_categories)
    return (labels_tr, labels_te, uniq_lvls)

def _windowed_counts(spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids):
    Q = build_count_matrix(spike_counts, raw_counts, selected_df, frames, time_window, neuron_ids).T
    return Q

def _fit_level_map(Q, level_labels, K, dt_eff, tuner, eps=1e-12):
    (lam, pr) = tuner(Q, level_labels, None, K, dt_eff, False, 0.0, eps=eps, decode_head='none')
    valid = pr > 0
    return (lam, pr, valid)

def _normalize_cols_level(X, mode='zscore', eps=1e-12):
    if mode is None:
        return X
    if mode == 'zscore':
        return zscore(X, axis=0, ddof=1, nan_policy='omit')
    if mode == 'l2':
        n = np.linalg.norm(X, axis=0, keepdims=True)
        return X / (n + eps)
    if mode == 'auto':
        return zscore(X, axis=0, ddof=1, nan_policy='omit')
    raise ValueError("normalization must be 'zscore' | 'l2' | 'auto' | None")

def _best_circ_corr_and_lag(A, B, min_levels=3, eps=1e-12):
    (Kp, M) = A.shape
    best_corr = np.full(M, np.nan, float)
    best_lag = np.full(M, np.nan, float)
    for i in range(M):
        a = A[:, i]
        b = B[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        a = a[good]
        b = b[good]
        L = len(a)
        if L < min_levels:
            continue
        az = (a - a.mean()) / (a.std(ddof=1) + eps)
        bz = (b - b.mean()) / (b.std(ddof=1) + eps)
        dots = np.empty(L)
        for s in range(L):
            r = np.roll(bz, s)
            dots[s] = np.dot(az, r) / (L - 1 + eps)
        s_best = int(np.argmax(dots))
        val = float(np.clip(dots[s_best], -1.0, 1.0))
        signed = s_best if s_best <= L // 2 else s_best - L
        best_corr[i] = val
        best_lag[i] = float(signed)
    return (best_corr, best_lag)

def cross_session_tuning_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, metric='pearson', normalization='auto', intersect_mask=True, eps=1e-12, min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', spatial_labels_train=None, spatial_labels_test=None, min_bins=None, level_categories=None):
    rng = np.random.RandomState(random_state)
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    if rate_source == 'spike':
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == 'raw':
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")

    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    if train_ids is not None and test_ids is not None:
        tr_ids = _to_int_array(train_ids)
        te_ids = _to_int_array(test_ids)
        if tr_ids.shape[0] != te_ids.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr_ids.size == 0:
            raise ValueError('Empty train_ids/test_ids after parsing.')
        if tr_ids.min() < 0 or tr_ids.max() >= n_tr:
            raise ValueError('train_ids out of range for TRAIN session (expected zero-based).')
        if te_ids.min() < 0 or te_ids.max() >= n_te:
            raise ValueError('test_ids out of range for TEST session (expected zero-based).')
        if n_neurons is not None and tr_ids.size > n_neurons:
            sel = rng.choice(tr_ids.size, int(n_neurons), replace=False)
            tr_ids = tr_ids[sel]
            te_ids = te_ids[sel]
        train_ids = tr_ids
        test_ids = te_ids
    else:
        (train_ids, test_ids) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    train_ids = np.asarray(train_ids)
    test_ids = np.asarray(test_ids)
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[train_ids] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[test_ids] >= float(thr_te)
    if drop_if_below == 'either':
        keep_mask = pass_tr & pass_te
    elif drop_if_below == 'both':
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    train_ids = train_ids[keep_mask]
    test_ids = test_ids[keep_mask]
    M = int(len(train_ids))
    if M == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=0)
        return dict(train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), similarity=np.array([], dtype=float), metric=metric, mode=mode, valid_counts=np.array([], dtype=int), summary=summary, meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), dropped=dict(n_pairs_initial=int(len(keep_mask)), n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()), policy=drop_if_below))
    (maps_tr, mask_tr, meta_tr) = _fit_maps_for_session(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, train_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_train, level_categories=level_categories)
    (maps_te, mask_te, meta_te) = _fit_maps_for_session(spike_counts_test, raw_counts_test, selected_df_test, pos_te, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_test, level_categories=level_categories)
    mask = mask_tr & mask_te if intersect_mask else mask_tr | mask_te
    idx = np.where(mask)[0]
    min_bins_failed = min_bins is not None and idx.size < int(min_bins)
    if idx.size == 0 or min_bins_failed:
        sim = np.full(M, np.nan, float)
        valid_counts = np.full(M, int(idx.size), int)
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=M)
        meta_tr['rate_per_min'] = rate_tr
        meta_tr['threshold'] = thr_tr
        meta_tr['rate_source'] = rate_source
        meta_tr['min_bins'] = min_bins
        meta_tr['min_bins_failed'] = bool(min_bins_failed)
        meta_te['rate_per_min'] = rate_te
        meta_te['threshold'] = thr_te
        meta_te['rate_source'] = rate_source
        meta_te['min_bins'] = min_bins
        meta_te['min_bins_failed'] = bool(min_bins_failed)
        return dict(train_map=maps_tr, test_map=maps_te, train_mask=mask_tr, test_mask=mask_te, train_ids=train_ids, test_ids=test_ids, similarity=sim, metric=metric, mode=mode, valid_counts=valid_counts, summary=summary, meta_train=meta_tr, meta_test=meta_te, dropped=dict(n_pairs_initial=int(len(keep_mask)), n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()), policy=drop_if_below))
    A = maps_tr[idx, :]
    B = maps_te[idx, :]
    if normalization == 'auto':
        norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        norm_mode = normalization
    A = _normalize_columns(A, mode=norm_mode, eps=eps)
    B = _normalize_columns(B, mode=norm_mode, eps=eps)
    sim = np.empty(M, float)
    valid_counts = np.empty(M, int)
    for i in range(M):
        a = A[:, i]
        b = B[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        valid_counts[i] = int(good.sum())
        sim[i] = np.nan if valid_counts[i] == 0 else _vector_similarity(a[good], b[good], metric=metric)
    summary = dict(mean=float(np.nanmean(sim)), std=float(np.nanstd(sim)), median=float(np.nanmedian(sim)), n=int(M))
    meta_tr['rate_per_min'] = rate_tr
    meta_tr['threshold'] = thr_tr
    meta_tr['rate_source'] = rate_source
    meta_tr['min_bins'] = min_bins
    meta_tr['min_bins_failed'] = False
    meta_te['rate_per_min'] = rate_te
    meta_te['threshold'] = thr_te
    meta_te['rate_source'] = rate_source
    meta_te['min_bins'] = min_bins
    meta_te['min_bins_failed'] = False
    return dict(train_map=maps_tr, test_map=maps_te, train_mask=mask_tr, test_mask=mask_te, train_ids=train_ids, test_ids=test_ids, similarity=sim, metric=metric, mode=mode, valid_counts=valid_counts, summary=summary, meta_train=meta_tr, meta_test=meta_te, dropped=dict(n_pairs_initial=int(len(keep_mask)), n_pairs_dropped=int(np.size(keep_mask) - keep_mask.sum()), policy=drop_if_below))

def cross_session_level_shift_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap=None, level_col='level', dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, normalization='auto', intersect_mask=True, min_levels=3, eps=1e-12, level_categories=None):
    rng = np.random.RandomState(random_state)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    n_train = spike_counts_train.shape[1]
    n_test = spike_counts_test.shape[1]
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr.min() < 0 or tr.max() >= n_train:
            raise ValueError('train_ids out of range for TRAIN session (expected zero-based).')
        if te.min() < 0 or te.max() >= n_test:
            raise ValueError('test_ids out of range for TEST session (expected zero-based).')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (train_ids, test_ids) = _pick_matched_neurons_for_level(cimap, n_train, n_test, n_neurons=n_neurons, rng=rng)
    M = len(train_ids)
    if M == 0:
        raise ValueError('No matched neurons after selection.')
    (labels_tr, labels_te, uniq_lvls) = _build_level_labels(selected_df_train, selected_df_test, level_col, level_categories=level_categories)
    K = len(uniq_lvls)
    frames_tr = np.arange(len(positions_train))
    frames_te = np.arange(len(positions_test))
    Q_tr = _windowed_counts(spike_counts_train, raw_counts_train, selected_df_train, frames_tr, time_window, train_ids)
    Q_te = _windowed_counts(spike_counts_test, raw_counts_test, selected_df_test, frames_te, time_window, test_ids)
    dt_eff = dt * (int(time_window) + 1)
    (lam_tr, pr_tr, valid_tr) = _fit_level_map(Q_tr, labels_tr, K, dt_eff, tuner, eps)
    (lam_te, pr_te, valid_te) = _fit_level_map(Q_te, labels_te, K, dt_eff, tuner, eps)
    mask = valid_tr & valid_te if intersect_mask else valid_tr | valid_te
    idx = np.where(mask)[0]
    if idx.size < min_levels:
        sim = np.full(M, np.nan, float)
        best_lag = np.full(M, np.nan, float)
        valid_counts = np.zeros(M, int)
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n=M)
        meta_tr = dict(n_levels=K, dt_eff=dt_eff)
        meta_te = dict(n_levels=K, dt_eff=dt_eff)
        return dict(train_ids=train_ids, test_ids=test_ids, best_corr=sim, best_lag=best_lag, levels=uniq_lvls, valid_counts=valid_counts, summary=summary, meta_train=meta_tr, meta_test=meta_te)
    A = lam_tr[idx, :]
    B = lam_te[idx, :]
    A = _normalize_cols_level(A, normalization, eps)
    B = _normalize_cols_level(B, normalization, eps)
    (best_corr, best_lag) = _best_circ_corr_and_lag(A, B, min_levels=min_levels, eps=eps)
    valid_counts = np.full(M, idx.size, int)
    summary = dict(mean=float(np.nanmean(np.abs(best_lag))), std=float(np.nanstd(np.abs(best_lag))), median=float(np.nanmedian(np.abs(best_lag))), n=int(M))
    meta_tr = dict(n_levels=K, dt_eff=dt_eff)
    meta_te = dict(n_levels=K, dt_eff=dt_eff)
    return dict(train_ids=train_ids, test_ids=test_ids, best_corr=best_corr, best_lag=best_lag, levels=uniq_lvls, valid_counts=valid_counts, summary=summary, meta_train=meta_tr, meta_test=meta_te)

def cross_session_pv_correlation(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, exclude_train_ids=None, exclude_test_ids=None, exclude_pairs=None, metric='pearson', normalization='auto', intersect_mask=True, min_neurons=3, eps=1e-12, min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', spatial_labels_train=None, spatial_labels_test=None, level_categories=None):
    rng = np.random.RandomState(random_state)
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    if rate_source == 'spike':
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == 'raw':
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")

    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError('train_ids out of range for TRAIN session (expected zero-based).')
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError('test_ids out of range for TEST session (expected zero-based).')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (train_ids, test_ids) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
    if len(train_ids) == 0:
        raise ValueError('No matched neurons selected.')
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    n_pairs_initial = int(train_ids.size)
    keep_user = np.ones(train_ids.size, dtype=bool)
    if exclude_train_ids is not None:
        ex_tr = _to_int_array(exclude_train_ids)
        if ex_tr.size:
            keep_user &= ~np.isin(train_ids, ex_tr)
    if exclude_test_ids is not None:
        ex_te = _to_int_array(exclude_test_ids)
        if ex_te.size:
            keep_user &= ~np.isin(test_ids, ex_te)
    if exclude_pairs is not None:
        ex_pairs = np.asarray(exclude_pairs, dtype=int)
        if not (ex_pairs.ndim == 1 and ex_pairs.size == 0):
            if ex_pairs.ndim != 2 or ex_pairs.shape[1] != 2:
                raise ValueError('exclude_pairs must be an array-like of (train_id, test_id) pairs.')
            ex_set = set(map(tuple, ex_pairs.tolist()))
            pair_mask = np.array([(tr, te) not in ex_set for (tr, te) in zip(train_ids, test_ids)], dtype=bool)
            keep_user &= pair_mask
    train_ids = train_ids[keep_user]
    test_ids = test_ids[keep_user]
    n_dropped_by_user = int((~keep_user).sum())
    if train_ids.size == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n_bins=0)
        return dict(pvc=np.array([], dtype=float), mask=None, maps_tr=None, maps_te=None, metric=metric, mode=mode, normalization=normalization, train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), valid_counts=None, summary=summary, meta_train=dict(rate_per_min=rate_tr, threshold=min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min, rate_source=rate_source), dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=0, n_pairs_final=0, policy=drop_if_below))
    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == 'either':
        keep_rate = pass_tr & pass_te
    elif drop_if_below == 'both':
        keep_rate = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    n_dropped_by_rate = int((~keep_rate).sum())
    train_ids = np.asarray(train_ids)[keep_rate]
    test_ids = np.asarray(test_ids)[keep_rate]
    if train_ids.size == 0:
        summary = dict(mean=np.nan, std=np.nan, median=np.nan, n_bins=0)
        return dict(pvc=np.array([], dtype=float), mask=None, maps_tr=None, maps_te=None, metric=metric, mode=mode, normalization=normalization, train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), valid_counts=None, summary=summary, meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=n_dropped_by_rate, n_pairs_final=0, policy=drop_if_below))
    (maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te) = _prefit_train_and_test_maps(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, spike_counts_test, raw_counts_test, selected_df_test, pos_te, train_ids, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels_train=spatial_labels_train, spatial_labels_test=spatial_labels_test, level_categories=level_categories)
    if normalization == 'auto':
        norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        norm_mode = normalization
    (pvc, mask) = _pv_from_maps(maps_tr, maps_te, mask_tr, mask_te, metric=metric, normalization=norm_mode, intersect_mask=intersect_mask, min_neurons=min_neurons, eps=eps)
    summary = dict(mean=float(np.nanmean(pvc)) if pvc.size else np.nan, std=float(np.nanstd(pvc)) if pvc.size else np.nan, median=float(np.nanmedian(pvc)) if pvc.size else np.nan, n_bins=int(np.isfinite(pvc).sum()) if pvc.size else 0)
    meta_tr = dict(meta_tr)
    meta_te = dict(meta_te)
    meta_tr['rate_per_min'] = rate_tr
    meta_tr['threshold'] = thr_tr
    meta_tr['rate_source'] = rate_source
    meta_te['rate_per_min'] = rate_te
    meta_te['threshold'] = thr_te
    meta_te['rate_source'] = rate_source
    return dict(pvc=pvc, mask=mask, maps_tr=maps_tr, maps_te=maps_te, metric=metric, mode=mode, normalization=norm_mode, train_ids=train_ids, test_ids=test_ids, valid_counts=None, summary=summary, meta_train=meta_tr, meta_test=meta_te, dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=n_dropped_by_rate, n_pairs_final=int(train_ids.size), policy=drop_if_below))

def cross_session_pv_cross_similarity_matrix(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, metric='pearson', normalization='auto', intersect_mask=True, eps=1e-12, min_neurons=3, drop_rows_with_too_few_neurons=True, level_categories=None):
    try:
        from scipy.stats import rankdata as _rankdata
    except Exception:

        def _rankdata(x, method='average'):
            order = np.argsort(x)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(order), dtype=float)
            return ranks
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    rng = np.random.RandomState(random_state)
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError('train_ids out of range for TRAIN session (zero-based expected).')
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError('test_ids out of range for TEST session (zero-based expected).')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (train_ids, test_ids) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
    M = len(train_ids)
    if M == 0:
        raise ValueError('No matched neurons after selection.')
    (maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te) = _prefit_train_and_test_maps(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, spike_counts_test, raw_counts_test, selected_df_test, pos_te, train_ids, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, level_categories=level_categories)
    (K_tr, M_tr) = maps_tr.shape
    (K_te, M_te) = maps_te.shape
    if M_tr != M_te:
        raise ValueError(f'Train/Test neuron counts disagree after selection: {M_tr} vs {M_te}')
    if intersect_mask and K_tr == K_te:
        common_idx = np.where(mask_tr.astype(bool) & mask_te.astype(bool))[0]
        rows_tr = common_idx
        rows_te = common_idx
    else:
        rows_tr = np.where(mask_tr)[0]
        rows_te = np.where(mask_te)[0]
    A = maps_tr[rows_tr, :]
    B = maps_te[rows_te, :]
    good_tr = np.isfinite(A)
    good_te = np.isfinite(B)
    min_good = max(1, int(min_neurons))
    keep_tr = good_tr.sum(axis=1) >= min_good
    keep_te = good_te.sum(axis=1) >= min_good
    if drop_rows_with_too_few_neurons:
        A = A[keep_tr, :]
        B = B[keep_te, :]
        rows_tr = rows_tr[keep_tr]
        rows_te = rows_te[keep_te]
        good_tr = good_tr[keep_tr, :]
        good_te = good_te[keep_te, :]
    A = _impute_neuronwise(A, good_tr)
    B = _impute_neuronwise(B, good_te)
    if normalization == 'auto':
        row_norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        row_norm_mode = normalization
    if metric == 'spearman':
        A = np.apply_along_axis(_rankdata, 1, A)
        B = np.apply_along_axis(_rankdata, 1, B)
        A = np.apply_along_axis(_norm_row, 1, A, mode='zscore', eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode='zscore', eps=eps)
        denom = max(M, 1)
        sim = np.clip(A @ B.T / denom, -1.0, 1.0)
    elif metric == 'pearson':
        A = np.apply_along_axis(_norm_row, 1, A, mode='zscore', eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode='zscore', eps=eps)
        denom = max(M, 1)
        sim = np.clip(A @ B.T / denom, -1.0, 1.0)
    else:
        A = np.apply_along_axis(_norm_row, 1, A, mode='l2', eps=eps)
        B = np.apply_along_axis(_norm_row, 1, B, mode='l2', eps=eps)
        sim = A @ B.T
    return dict(matrix=sim, rows_train_idx=rows_tr, rows_test_idx=rows_te, metric=metric, mode=mode, normalization=row_norm_mode, train_ids=train_ids, test_ids=test_ids, meta_train=meta_tr, meta_test=meta_te)

def run_similarity_pipeline(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=None, train_ids=None, test_ids=None, exclude_train_ids=None, exclude_test_ids=None, exclude_pairs=None, session1_name='session1', session2_name='session2', modes=('spatial', 'joint_pos', 'level', 'joint_lvl'), level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=0, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, eps=1e-12, metric='pearson', normalization='auto', intersect_mask=True, min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', run_pf_shuffle=True, pf_shuffle_repeats=1000, pf_shuffle_session='test', pf_shuffle_types=('map', 'neuron_id'), run_pv_cross_similarity=True, pv_metric='pearson', pv_normalization='auto', min_neurons=3, drop_rows_with_too_few_neurons=True, run_pv_correlation=True, run_pv_pf_shuffle=True, pv_pf_shuffle_repeats=1000, pv_pf_shuffle_session='test', pv_pf_shuffle_types=('neuron_id', 'map'), run_level_shift=True, min_levels=3, level_norm='zscore', level_shift_pf_shuffle_types=('neuron_id', 'map'), level_shift_pf_shuffle_session='test', run_pivot_tuning=False, run_within_session_pv=False, pivot_win_len_sec=20 * 60, pivot_step_sec=60, pivot_time_axis='selected', frame_dt_sec=None, min_time_per_bin_sec=5.0, run_spatial_info=False, si_min_time_per_bin_sec=5.0, si_control='intersection_native', output_dir='similarity_results', save_json_summaries=True, do_plots=True, overwrite=False):
    import json
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    def _ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def _exists(path):
        try:
            return not overwrite and os.path.exists(path) and (os.path.getsize(path) > 0)
        except OSError:
            return False

    def _save_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def _load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_json(obj, path):
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, np.generic) else str(x))

    def _save_hist(values, out_png, title, bins=40):
        try:
            v = np.asarray(values).ravel()
            v = v[np.isfinite(v)]
            if v.size == 0:
                return
            plt.figure(figsize=(6, 4))
            plt.hist(v, bins=bins)
            plt.title(title)
            plt.xlabel('value')
            plt.ylabel('count')
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as e:
            print(f'[pipeline] (plot skipped: {title}) {e}')

    def _save_heatmap(mat, out_png, title):
        try:
            M = np.asarray(mat)
            if M.ndim != 2 or M.size == 0:
                return
            plt.figure(figsize=(6, 5))
            plt.imshow(M, aspect='auto')
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception as e:
            print(f'[pipeline] (heatmap skipped: {title}) {e}')

    def _save_one_shuffle_hist(matrix2d, out_png_prefix, title_prefix, rng):
        try:
            X = np.asarray(matrix2d)
            if X.ndim != 2 or X.shape[0] == 0:
                return
            r_idx = int(rng.randint(X.shape[0]))
            row = X[r_idx]
            row = row[np.isfinite(row)]
            if row.size == 0:
                return
            out_png = f'{out_png_prefix}_one_shuffle_hist_r{r_idx}.png'
            title = f'{title_prefix} — one shuffle (r={r_idx})'
            _save_hist(row, out_png, title)
        except Exception as e:
            print(f'[pipeline] (one-shuffle plot skipped: {title_prefix}) {e}')
    _ensure_dir(output_dir)
    name1_ok = bool(session1_name)
    name2_ok = bool(session2_name)
    both_named = name1_ok and name2_ok
    single_session_mode = not both_named
    if single_session_mode:
        print('\n=== Single-session mode: at least one session name is empty → only within-session analyses will run. Cross-session steps are skipped. ===')
    available_sessions = []
    if name1_ok:
        available_sessions.append((session1_name, spike_counts1, raw_counts1, selected_df1, positions1))
    if name2_ok:
        available_sessions.append((session2_name, spike_counts2, raw_counts2, selected_df2, positions2))

    def _is_1d_pos(pos):
        arr = np.asarray(pos)
        return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)
    if available_sessions:
        dims_1d = [_is_1d_pos(POS) for (_, _, _, _, POS) in available_sessions]
        if len(set(dims_1d)) > 1:
            raise ValueError('run_similarity_pipeline: mixed 1D and 2D positions across sessions are not supported. Please ensure all sessions use either 1D positions (shape (T,) or (T,1)) or 2D positions (T,2).')
        if dims_1d[0] and n_bins_y != 1:
            raise ValueError('run_similarity_pipeline: detected 1D positions (shape (T,) or (T,1)), but n_bins_y != 1. For 1D, set n_bins_y=1 and use n_bins_x as the number of spatial bins.')
    pair_tag = f'{session1_name}_to_{session2_name}' if both_named else session1_name or session2_name or 'single_session'
    print(f'\n=== Saving similarity pipeline results to: {output_dir} ===')
    rng = np.random.RandomState(random_state)
    cimap_swapped = cimap[:, [1, 0]] if cimap is not None else None

    def _has(df, col):
        return col is not None and col in df.columns
    have_level = _has(selected_df1, level_col) and _has(selected_df2, level_col) if both_named else available_sessions and _has(available_sessions[0][3], level_col)
    have_head = _has(selected_df1, head_col) and _has(selected_df2, head_col) if both_named else available_sessions and _has(available_sessions[0][3], head_col)
    expanded_modes = []
    for m in modes:
        mm = 'joint_pos' if m == 'joint' else m
        if mm in ('level', 'joint_lvl') and (not have_level):
            print(f'[pipeline] Skipping mode={mm} (missing level_col).')
            continue
        if mm in ('hd', 'joint_pos', 'joint_lvl') and (not have_head):
            print(f'[pipeline] Skipping mode={mm} (missing head_col).')
            continue
        expanded_modes.append(mm)
    results = {'pivot_tuning': {}, 'within_session_pv': {}}
    if both_named:
        results.update({'tuning_similarity': {}, 'pf_shuffle': {}, 'pv_cross_similarity': {}, 'pv_correlation': {}, 'pv_pf_shuffle': {}, 'level_shift': {}})
    if run_pivot_tuning or run_within_session_pv:
        for mode in expanded_modes:
            if run_pivot_tuning:
                piv_mode_dir = os.path.join(output_dir, 'pivot_tuning', f'mode_{mode}')
                _ensure_dir(piv_mode_dir)
            if run_within_session_pv:
                wsv_mode_dir = os.path.join(output_dir, 'within_session_pv', f'mode_{mode}')
                _ensure_dir(wsv_mode_dir)
            for (sess_name, SC, RC, DF, POS) in available_sessions:
                if run_pivot_tuning:
                    piv_base = os.path.join(piv_mode_dir, f'pivot_tuning_{sess_name}')
                    piv_pkl = f'{piv_base}.pkl'
                    piv_json = f'{piv_base}.json'
                    piv_hist = f'{piv_base}_sim_mean_hist.png'
                    if _exists(piv_pkl):
                        print(f'[pipeline] (single) pivot_tuning mode={mode}, session={sess_name} … exists → skip')
                        res_pivot = _load_pickle(piv_pkl)
                    else:
                        print(f'[pipeline] (single) pivot_tuning mode={mode}, session={sess_name} …', end='', flush=True)
                        try:
                            res_pivot = pivot_window_tuning_similarity(SC, RC, DF, POS, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, neuron_ids=None, random_state=random_state, win_len_sec=pivot_win_len_sec, pivot_step_sec=pivot_step_sec, metric=metric, normalization=normalization, intersect_mask=intersect_mask, min_bins=3, eps=eps, min_time_per_bin_sec=min_time_per_bin_sec, min_spikes_per_min=min_spikes_per_min, rate_source=rate_source, pivot_time_axis=pivot_time_axis, frame_dt_sec=frame_dt_sec)
                            _save_pickle(res_pivot, piv_pkl)
                            if save_json_summaries:
                                _save_json({'n_pivots': int(res_pivot['pivot_times'].size), 'norm': res_pivot.get('normalization', '')}, piv_json)
                            if do_plots and 'sim_mean' in res_pivot:
                                _save_hist(res_pivot['sim_mean'], piv_hist, title=f'Pivot tuning mean similarity ({mode}) • {sess_name}')
                            print(' done.')
                        except Exception as e:
                            print(f' ERROR: {e}')
                            res_pivot = {'error': str(e)}
                            _save_pickle(res_pivot, piv_pkl)
                    results['pivot_tuning'].setdefault(sess_name, {})[mode] = res_pivot
                if run_within_session_pv:
                    wsv_base = os.path.join(wsv_mode_dir, f'within_session_pv_{sess_name}')
                    wsv_pkl = f'{wsv_base}.pkl'
                    wsv_json = f'{wsv_base}.json'
                    wsv_hist = f'{wsv_base}_sim_mean_hist.png'
                    if _exists(wsv_pkl):
                        print(f'[pipeline] (single) within_session_pv mode={mode}, session={sess_name} … exists → skip')
                        res_wsv = _load_pickle(wsv_pkl)
                    else:
                        print(f'[pipeline] (single) within_session_pv mode={mode}, session={sess_name} …', end='', flush=True)
                        try:
                            res_wsv = within_session_pv_temporal_similarity(SC, RC, DF, POS, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, neuron_ids=None, random_state=random_state, win_len_sec=pivot_win_len_sec, pivot_step_sec=pivot_step_sec, pivot_time_axis=pivot_time_axis, frame_dt_sec=frame_dt_sec, metric=pv_metric, normalization=pv_normalization, intersect_mask=intersect_mask, eps=eps, min_time_per_bin_sec=min_time_per_bin_sec, min_spikes_per_min=min_spikes_per_min, rate_source=rate_source)
                            _save_pickle(res_wsv, wsv_pkl)
                            if save_json_summaries:
                                _save_json({'n_pivots': int(res_wsv['pivot_times'].size), 'norm': res_wsv.get('normalization', '')}, wsv_json)
                            if do_plots and 'sim_mean' in res_wsv:
                                _save_hist(res_wsv['sim_mean'], wsv_hist, title=f'Within-session PV mean ({mode}) • {sess_name}')
                            print(' done.')
                        except Exception as e:
                            print(f' ERROR: {e}')
                            res_wsv = {'error': str(e)}
                            _save_pickle(res_wsv, wsv_pkl)
                    results['within_session_pv'].setdefault(sess_name, {})[mode] = res_wsv
                if run_spatial_info:
                    si_root = os.path.join(output_dir, 'spatial_info', 'mode_spatial')
                    _ensure_dir(si_root)
                    for (sess_name, SC, RC, DF, POS) in available_sessions:
                        si_base = os.path.join(si_root, f'spatial_info_{sess_name}')
                        si_pkl = f'{si_base}.pkl'
                        si_json = f'{si_base}.json'
                        si_hist = f'{si_base}_SI_hist.png'
                        if _exists(si_pkl):
                            print(f'[pipeline] (single) spatial_info • {sess_name} … exists → skip')
                            res_si = _load_pickle(si_pkl)
                        else:
                            print(f'[pipeline] (single) spatial_info • {sess_name} …', end='', flush=True)
                            try:
                                res_si = compute_spatial_info_session_rategated(SC, RC, DF, POS, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, min_time_per_bin_sec=si_min_time_per_bin_sec, eps=eps, neuron_ids=None, min_spikes_per_min=min_spikes_per_min, rate_source=rate_source)
                                _save_pickle(res_si, si_pkl)
                                if save_json_summaries:
                                    _save_json({'n_neurons': int(res_si['SI_native'].size), 'median_bits_per_spike': float(np.nanmedian(res_si['SI_native'])) if res_si['SI_native'].size else np.nan}, si_json)
                                if do_plots and res_si['SI_native'].size:
                                    _save_hist(res_si['SI_native'], si_hist, title=f'Spatial information (native, rate-gated) • {sess_name}')
                                print(' done.')
                            except Exception as e:
                                print(f' ERROR: {e}')
                                res_si = {'error': str(e)}
                                _save_pickle(res_si, si_pkl)
                        results.setdefault('spatial_info', {}).setdefault(sess_name, {})['single'] = res_si
    if both_named:
        for mode in expanded_modes:
            sub = os.path.join(output_dir, f'mode_{mode}')
            _ensure_dir(sub)
            print(f'\n--- MODE: {mode} ---')
            tuning_pkl = os.path.join(sub, f'tuning_{pair_tag}.pkl')
            tuning_json = os.path.join(sub, f'tuning_summary_{pair_tag}.json')
            tuning_hist = os.path.join(sub, f'tuning_hist_{pair_tag}.png')
            if run_spatial_info and mode == 'spatial':
                si_root = os.path.join(output_dir, 'spatial_info', 'mode_spatial')
                _ensure_dir(si_root)
                si_pair_pkl = os.path.join(si_root, f'spatial_info_pair_{pair_tag}.pkl')
                si_pair_json = os.path.join(si_root, f'spatial_info_pair_{pair_tag}.json')
                if _exists(si_pair_pkl):
                    print(f'[pipeline] 1b) spatial_info (matched) … exists → skip')
                    res_sim = _load_pickle(si_pair_pkl)
                else:
                    print(f'[pipeline] 1b) spatial_info (matched) …', end='', flush=True)
                    try:
                        res_sim = cross_session_spatial_information_rategated(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=cimap_swapped, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, min_time_per_bin_sec=si_min_time_per_bin_sec, eps=eps, n_neurons=n_neurons, random_state=random_state, train_ids=train_ids, test_ids=test_ids, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below, si_control=si_control)
                        _save_pickle(res_sim, si_pair_pkl)
                        if save_json_summaries:
                            _save_json({'control': si_control, 'n_pairs': int(res_sim['train_ids'].size), 'median_native_s1': float(np.nanmedian(res_sim['SI_native_s1'])) if res_sim['SI_native_s1'].size else np.nan, 'median_native_s2': float(np.nanmedian(res_sim['SI_native_s2'])) if res_sim['SI_native_s2'].size else np.nan, 'median_matched_s1': float(np.nanmedian(res_sim['SI_matched_s1'])) if res_sim['SI_matched_s1'].size else np.nan, 'median_matched_s2': float(np.nanmedian(res_sim['SI_matched_s2'])) if res_sim['SI_matched_s2'].size else np.nan}, si_pair_json)
                        print(' done.')
                    except Exception as e:
                        print(f' ERROR: {e}')
                        res_sim = {'error': str(e)}
                        _save_pickle(res_sim, si_pair_pkl)
                results.setdefault('spatial_info', {})['pair'] = res_sim
            if _exists(tuning_pkl):
                print(f'[pipeline] 1) Tuning similarity … exists → skip')
                res = _load_pickle(tuning_pkl)
                results['tuning_similarity'][mode] = res
                if save_json_summaries and (not _exists(tuning_json)):
                    _save_json({'summary': res.get('summary', '')}, tuning_json)
                if do_plots and (not _exists(tuning_hist)) and ('similarity' in res):
                    _save_hist(res['similarity'], tuning_hist, title=f'Tuning similarity ({mode}) • {pair_tag}')
            else:
                print(f'[pipeline] 1) Tuning similarity …', end='', flush=True)
                try:
                    res = cross_session_tuning_similarity(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=cimap_swapped, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, train_ids=train_ids, test_ids=test_ids, metric=metric, normalization=normalization, intersect_mask=intersect_mask, eps=eps, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below)
                    results['tuning_similarity'][mode] = res
                    _save_pickle(res, tuning_pkl)
                    if save_json_summaries:
                        _save_json({'summary': res.get('summary', '')}, tuning_json)
                    if do_plots and 'similarity' in res:
                        _save_hist(res['similarity'], tuning_hist, title=f'Tuning similarity ({mode}) • {pair_tag}')
                    print(' done.')
                except Exception as e:
                    print(f' ERROR: {e}')
                    results['tuning_similarity'][mode] = {'error': str(e)}
            if run_pf_shuffle:
                results['pf_shuffle'].setdefault(mode, {})
                for stype in pf_shuffle_types:
                    tag = f'pf_shuffle_{stype}_{pair_tag}'
                    pf_pkl = os.path.join(sub, f'{tag}.pkl')
                    pf_hist = os.path.join(sub, f'{tag}_hist.png')
                    if _exists(pf_pkl):
                        print(f'[pipeline] 2) PF-shuffle (tuning) type={stype} … exists → skip')
                        res_pf = _load_pickle(pf_pkl)
                        results['pf_shuffle'][mode][stype] = res_pf
                        if do_plots:
                            if 'sim_per_repeat' in res_pf and (not _exists(pf_hist)):
                                _save_hist(res_pf['sim_per_repeat'], pf_hist, title=f'PF-shuffle (tuning {stype}) per-repeat means ({mode}) • {pair_tag}')
                            if 'sim_matrix' in res_pf and isinstance(res_pf['sim_matrix'], np.ndarray):
                                if not any((f.startswith(f'{tag}_one_shuffle_hist_r') for f in os.listdir(sub))):
                                    _save_one_shuffle_hist(res_pf['sim_matrix'], os.path.join(sub, f'{tag}'), f'PF-shuffle (tuning {stype}) per-neuron corr ({mode}) • {pair_tag}', rng)
                    else:
                        print(f'[pipeline] 2) PF-shuffle (tuning) type={stype} (repeats={pf_shuffle_repeats}) …', end='', flush=True)
                        try:
                            res_pf = pf_shuffle_tuning_similarity(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=cimap_swapped, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, metric=metric or 'pearson', normalization='auto' if normalization == 'none' else normalization, intersect_mask=intersect_mask, eps=eps, n_repeats=pf_shuffle_repeats, shuffle_session=pf_shuffle_session, shuffle_type=stype, train_ids=train_ids, test_ids=test_ids, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below)
                            results['pf_shuffle'][mode][stype] = res_pf
                            _save_pickle(res_pf, pf_pkl)
                            if do_plots:
                                if 'sim_per_repeat' in res_pf:
                                    _save_hist(res_pf['sim_per_repeat'], pf_hist, title=f'PF-shuffle (tuning {stype}) per-repeat means ({mode}) • {pair_tag}')
                                if 'sim_matrix' in res_pf and isinstance(res_pf['sim_matrix'], np.ndarray):
                                    _save_one_shuffle_hist(res_pf['sim_matrix'], os.path.join(sub, f'{tag}'), f'PF-shuffle (tuning {stype}) per-neuron corr ({mode}) • {pair_tag}', rng)
                            print(' done.')
                        except Exception as e:
                            print(f' ERROR: {e}')
                            results['pf_shuffle'][mode][stype] = {'error': str(e)}
            if run_pv_cross_similarity:
                pv_pkl = os.path.join(sub, f'pv_cross_similarity_{pair_tag}.pkl')
                pv_heatmap = os.path.join(sub, f'pv_cross_similarity_heatmap_{pair_tag}.png')
                if _exists(pv_pkl):
                    print(f'[pipeline] 3) PV cross-similarity matrix … exists → skip')
                    res_pv = _load_pickle(pv_pkl)
                    results['pv_cross_similarity'][mode] = res_pv
                    if do_plots and (not _exists(pv_heatmap)):
                        mat = None
                        for k in ('matrix', 'pv_matrix', 'cross_similarity', 'similarity_matrix', 'M'):
                            if isinstance(res_pv.get(k, None), np.ndarray) and res_pv[k].ndim == 2:
                                mat = res_pv[k]
                                break
                        if mat is None:
                            for v in res_pv.values():
                                if isinstance(v, np.ndarray) and v.ndim == 2:
                                    mat = v
                                    break
                        if mat is not None:
                            _save_heatmap(mat, pv_heatmap, title=f'PV cross-similarity ({mode}) • {pair_tag}')
                else:
                    print(f'[pipeline] 3) PV cross-similarity matrix …', end='', flush=True)
                    try:
                        res_pv = cross_session_pv_cross_similarity_matrix(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=cimap_swapped, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, metric=pv_metric, normalization=pv_normalization, intersect_mask=intersect_mask, eps=eps, min_neurons=min_neurons, drop_rows_with_too_few_neurons=drop_rows_with_too_few_neurons, train_ids=train_ids, test_ids=test_ids)
                        results['pv_cross_similarity'][mode] = res_pv
                        _save_pickle(res_pv, pv_pkl)
                        mat = None
                        for k in ('matrix', 'pv_matrix', 'cross_similarity', 'similarity_matrix', 'M'):
                            if isinstance(res_pv.get(k, None), np.ndarray) and res_pv[k].ndim == 2:
                                mat = res_pv[k]
                                break
                        if mat is None:
                            for v in res_pv.values():
                                if isinstance(v, np.ndarray) and v.ndim == 2:
                                    mat = v
                                    break
                        if do_plots and mat is not None:
                            _save_heatmap(mat, pv_heatmap, title=f'PV cross-similarity ({mode}) • {pair_tag}')
                        print(' done.')
                    except Exception as e:
                        print(f' ERROR: {e}')
                        results['pv_cross_similarity'][mode] = {'error': str(e)}
            if run_pv_correlation:
                pvc_pkl = os.path.join(sub, f'pv_correlation_{pair_tag}.pkl')
                pvc_hist = os.path.join(sub, f'pv_correlation_hist_{pair_tag}.png')
                if _exists(pvc_pkl):
                    print(f'[pipeline] 4) PV correlation … exists → skip')
                    res_pvc = _load_pickle(pvc_pkl)
                    results['pv_correlation'][mode] = res_pvc
                    if do_plots and (not _exists(pvc_hist)) and ('pvc' in res_pvc):
                        _save_hist(res_pvc['pvc'], pvc_hist, title=f'PV correlation ({mode}) • {pair_tag}')
                else:
                    print(f'[pipeline] 4) PV correlation …', end='', flush=True)
                    try:
                        res_pvc = cross_session_pv_correlation(spike_counts_train=spike_counts1, raw_counts_train=raw_counts1, selected_df_train=selected_df1, positions_train=positions1, spike_counts_test=spike_counts2, raw_counts_test=raw_counts2, selected_df_test=selected_df2, positions_test=positions2, cimap=cimap_swapped, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, exclude_train_ids=exclude_train_ids, exclude_test_ids=exclude_test_ids, exclude_pairs=exclude_pairs, metric=metric, normalization='auto' if normalization == 'none' else normalization, intersect_mask=intersect_mask, min_neurons=min_neurons, eps=eps, train_ids=train_ids, test_ids=test_ids, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below)
                        results['pv_correlation'][mode] = res_pvc
                        _save_pickle(res_pvc, pvc_pkl)
                        if do_plots and 'pvc' in res_pvc:
                            _save_hist(res_pvc['pvc'], pvc_hist, title=f'PV correlation ({mode}) • {pair_tag}')
                        print(' done.')
                    except Exception as e:
                        print(f' ERROR: {e}')
                        results['pv_correlation'][mode] = {'error': str(e)}
            if run_pv_pf_shuffle:
                results['pv_pf_shuffle'].setdefault(mode, {})
                for stype in pv_pf_shuffle_types:
                    tag = f'pv_pf_shuffle_{stype}_{pair_tag}'
                    pvpf_pkl = os.path.join(sub, f'{tag}.pkl')
                    pvpf_hist = os.path.join(sub, f'{tag}_hist.png')
                    if _exists(pvpf_pkl):
                        print(f'[pipeline] 5) PF–PV shuffle type={stype} … exists → skip')
                        res_pvpf = _load_pickle(pvpf_pkl)
                        results['pv_pf_shuffle'][mode][stype] = res_pvpf
                        if do_plots and 'pv_shuffles' in res_pvpf and (not _exists(pvpf_hist)):
                            means = np.nanmean(res_pvpf['pv_shuffles'], axis=1)
                            _save_hist(means, pvpf_hist, title=f'PF–PV shuffle ({stype}) per-repeat means ({mode}) • {pair_tag}')
                            if not any((f.startswith(f'{tag}_one_shuffle_hist_r') for f in os.listdir(sub))):
                                _save_one_shuffle_hist(res_pvpf['pv_shuffles'], os.path.join(sub, f'{tag}'), f'PF–PV shuffle ({stype}) per-bin PV corr ({mode}) • {pair_tag}', rng)
                    else:
                        print(f'[pipeline] 5) PF–PV shuffle type={stype} (repeats={pv_pf_shuffle_repeats}) …', end='', flush=True)
                        try:
                            res_pvpf = pf_shuffle_pv_similarity(spike_counts_train=spike_counts1, raw_counts_train=raw_counts1, selected_df_train=selected_df1, positions_train=positions1, spike_counts_test=spike_counts2, raw_counts_test=raw_counts2, selected_df_test=selected_df2, positions_test=positions2, cimap=cimap_swapped, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, n_neurons=n_neurons, random_state=random_state, exclude_train_ids=exclude_train_ids, exclude_test_ids=exclude_test_ids, exclude_pairs=exclude_pairs, metric=metric, normalization='auto' if normalization == 'none' else normalization, intersect_mask=intersect_mask, min_neurons=min_neurons, repeats=pv_pf_shuffle_repeats, rng=None, return_observed=True, shuffle_type=stype, shuffle_session=pv_pf_shuffle_session, train_ids=train_ids, test_ids=test_ids, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below)
                            results['pv_pf_shuffle'][mode][stype] = res_pvpf
                            _save_pickle(res_pvpf, pvpf_pkl)
                            if do_plots and 'pv_shuffles' in res_pvpf:
                                means = np.nanmean(res_pvpf['pv_shuffles'], axis=1)
                                _save_hist(means, pvpf_hist, title=f'PF–PV shuffle ({stype}) per-repeat means ({mode}) • {pair_tag}')
                                _save_one_shuffle_hist(res_pvpf['pv_shuffles'], os.path.join(sub, f'{tag}'), f'PF–PV shuffle ({stype}) per-bin PV corr ({mode}) • {pair_tag}', rng)
                            print(' done.')
                        except Exception as e:
                            print(f' ERROR: {e}')
                            results['pv_pf_shuffle'][mode][stype] = {'error': str(e)}
        if run_level_shift and have_level:
            sub = os.path.join(output_dir, 'level_suite')
            _ensure_dir(sub)
            lvl_pkl = os.path.join(sub, f'level_shift_{pair_tag}.pkl')
            lvl_json = os.path.join(sub, f'level_shift_summary_{pair_tag}.json')
            lvl_hist = os.path.join(sub, f'level_shift_hist_{pair_tag}.png')
            print(f'\n--- LEVEL SUITE ---')
            if _exists(lvl_pkl):
                print(f'[pipeline] 6) Level-shift similarity … exists → skip')
                res_lvl = _load_pickle(lvl_pkl)
                results['level_shift']['main'] = res_lvl
                if save_json_summaries and (not _exists(lvl_json)):
                    _save_json({'summary': res_lvl.get('summary', '')}, lvl_json)
                if do_plots:
                    vec = res_lvl.get('best_lag', None)
                    if vec is None:
                        for k in ('best_lags', 'lags', 'values'):
                            if k in res_lvl:
                                vec = res_lvl[k]
                                break
                    if vec is not None and (not _exists(lvl_hist)):
                        _save_hist(vec, lvl_hist, title=f'Level-shift tuning similarity • {pair_tag}')
            else:
                print(f'[pipeline] 6) Level-shift similarity …', end='', flush=True)
                try:
                    res_lvl = cross_session_level_shift_similarity(spike_counts1, raw_counts1, selected_df1, positions1, spike_counts2, raw_counts2, selected_df2, positions2, cimap=cimap_swapped, level_col=level_col, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, train_ids=train_ids, test_ids=test_ids, intersect_mask=False, min_levels=min_levels, normalization=level_norm, eps=eps)
                    results['level_shift']['main'] = res_lvl
                    _save_pickle(res_lvl, lvl_pkl)
                    if save_json_summaries:
                        _save_json({'summary': res_lvl.get('summary', '')}, lvl_json)
                    if do_plots:
                        vec = res_lvl.get('best_lag', None)
                        if vec is None:
                            for k in ('best_lags', 'lags', 'values'):
                                if k in res_lvl:
                                    vec = res_lvl[k]
                                    break
                        if vec is not None:
                            _save_hist(vec, lvl_hist, title=f'Level-shift tuning similarity • {pair_tag}')
                    print(' done.')
                except Exception as e:
                    print(f' ERROR: {e}')
                    results['level_shift']['main'] = {'error': str(e)}
            for stype in level_shift_pf_shuffle_types:
                tag = f'level_shift_pf_shuffle_{stype}_{pair_tag}'
                shuffle_pkl = os.path.join(sub, f'{tag}.pkl')
                shuffle_hist = os.path.join(sub, f'{tag}_hist.png')
                result_key = f'pf_shuffle_{stype}'
                if _exists(shuffle_pkl):
                    print(f'[pipeline] 7) Level-shift PF-shuffle type={stype} ... exists -> skip')
                    res_lvl_pf = _load_pickle(shuffle_pkl)
                    results['level_shift'][result_key] = res_lvl_pf
                    if do_plots and not _exists(shuffle_hist):
                        matrix = res_lvl_pf.get('best_lag_matrix')
                        if matrix is not None:
                            _save_hist(np.asarray(matrix).ravel(), shuffle_hist, title=f'Level-shift PF-shuffle ({stype}) best lags - {pair_tag}')
                    continue
                print(f'[pipeline] 7) Level-shift PF-shuffle type={stype} ...', end='', flush=True)
                try:
                    res_lvl_pf = pf_shuffle_level_shift_similarity(
                        spike_counts1,
                        raw_counts1,
                        selected_df1,
                        positions1,
                        spike_counts2,
                        raw_counts2,
                        selected_df2,
                        positions2,
                        cimap=cimap_swapped,
                        level_col=level_col,
                        dt=dt,
                        time_window=time_window,
                        use_peak=use_peak,
                        filter_ratemap=filter_ratemap,
                        sigma=sigma,
                        n_neurons=n_neurons,
                        random_state=random_state,
                        train_ids=train_ids,
                        test_ids=test_ids,
                        normalization='auto',
                        restrict_to_common=False,
                        min_levels=min_levels,
                        n_repeats=max(100, pf_shuffle_repeats),
                        shuffle_session=level_shift_pf_shuffle_session,
                        shuffle_type=stype,
                        eps=eps,
                    )
                    results['level_shift'][result_key] = res_lvl_pf
                    _save_pickle(res_lvl_pf, shuffle_pkl)
                    if do_plots:
                        matrix = res_lvl_pf.get('best_lag_matrix')
                        if matrix is not None:
                            _save_hist(np.asarray(matrix).ravel(), shuffle_hist, title=f'Level-shift PF-shuffle ({stype}) best lags - {pair_tag}')
                    print(' done.')
                except Exception as e:
                    print(f' ERROR: {e}')
                    results['level_shift'][result_key] = {'error': str(e)}
        elif run_level_shift:
            print('[pipeline] Skipping level-shift suite (level_col missing or single-session mode).')
    else:
        print('\n[pipeline] Cross-session analyses skipped (single-session mode).')
    print('\n[pipeline] Pipeline complete.')
    return results

def pf_shuffle_level_shift_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap=None, level_col='level', dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, normalization='auto', restrict_to_common=True, min_levels=3, n_repeats=100, shuffle_session='test', shuffle_type='neuron_id', eps=1e-12, level_categories=None):
    if shuffle_session not in ('test', 'train'):
        raise ValueError("shuffle_session must be 'test' or 'train'.")
    if shuffle_type not in ('neuron_id', 'map'):
        raise ValueError("shuffle_type must be 'neuron_id' or 'map'.")
    rng = np.random.default_rng(random_state)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    Ntr = spike_counts_train.shape[1]
    Nte = spike_counts_test.shape[1]
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr.min() < 0 or tr.max() >= Ntr:
            raise ValueError('train_ids out of range for TRAIN session (expected zero-based).')
        if te.min() < 0 or te.max() >= Nte:
            raise ValueError('test_ids out of range for TEST session (expected zero-based).')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (train_ids, test_ids) = _pick_matched_neurons_for_level(cimap, Ntr, Nte, n_neurons, rng)
    M = len(train_ids)
    if M == 0:
        raise ValueError('No matched neurons for PF-shuffle.')
    (labels_tr, labels_te, uniq_lvls) = _build_level_labels(selected_df_train, selected_df_test, level_col, level_categories=level_categories)
    K = len(uniq_lvls)
    frames_tr = np.arange(len(positions_train))
    frames_te = np.arange(len(positions_test))
    dt_eff = dt * (int(time_window) + 1)
    Q_tr = _windowed_counts(spike_counts_train, raw_counts_train, selected_df_train, frames_tr, time_window, train_ids)
    Q_te = _windowed_counts(spike_counts_test, raw_counts_test, selected_df_test, frames_te, time_window, test_ids)
    (lam_tr, pr_tr, valid_tr) = _fit_level_map(Q_tr, labels_tr, K, dt_eff, tuner, eps)
    (lam_te, pr_te, valid_te) = _fit_level_map(Q_te, labels_te, K, dt_eff, tuner, eps)
    mask_levels = valid_tr & valid_te if restrict_to_common else valid_tr | valid_te
    idx = np.where(mask_levels)[0]
    Kp = idx.size
    if Kp < min_levels:
        return dict(best_corr_matrix=np.full((n_repeats, M), np.nan), best_lag_matrix=np.full((n_repeats, M), np.nan), mean=np.nan, std=np.nan, levels=uniq_lvls, meta=dict(n_levels=K, n_pairs=M, dt_eff=dt_eff, shuffle_session=shuffle_session, shuffle_type=shuffle_type, normalization=normalization), train_ids=train_ids, test_ids=test_ids)
    A_full = lam_tr
    B_full = lam_te
    A_base = A_full[idx, :]
    B_base = B_full[idx, :]
    best_corr_mat = np.full((n_repeats, M), np.nan, float)
    best_lag_mat = np.full((n_repeats, M), np.nan, float)
    for r in range(n_repeats):
        if shuffle_type == 'neuron_id':
            if shuffle_session == 'test':
                perm = rng.permutation(M)
                A_use = A_base
                B_use = B_base[:, perm]
            else:
                perm = rng.permutation(M)
                A_use = A_base[:, perm]
                B_use = B_base
        elif shuffle_session == 'test':
            B_shuf_full = _shuffle_rows_independent(B_full, rng)
            A_use = A_base
            B_use = B_shuf_full[idx, :]
        else:
            A_shuf_full = _shuffle_rows_independent(A_full, rng)
            A_use = A_shuf_full[idx, :]
            B_use = B_base
        A_n = _normalize_cols_level(A_use, normalization, eps)
        B_n = _normalize_cols_level(B_use, normalization, eps)
        (bc, bl) = _best_circ_corr_and_lag(A_n, B_n, min_levels=min_levels, eps=eps)
        best_corr_mat[r, :] = bc
        best_lag_mat[r, :] = bl
    return dict(best_corr_matrix=best_corr_mat, best_lag_matrix=best_lag_mat, mean=float(np.nanmean(np.abs(best_lag_mat))), std=float(np.nanstd(np.abs(best_lag_mat))), levels=uniq_lvls, meta=dict(n_levels=K, n_pairs=M, dt_eff=dt_eff, shuffle_session=shuffle_session, shuffle_type=shuffle_type, normalization=normalization), train_ids=train_ids, test_ids=test_ids)

def pf_shuffle_tuning_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, metric='pearson', normalization='auto', intersect_mask=True, eps=1e-12, n_repeats=100, shuffle_session='test', shuffle_type='map', min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', spatial_labels_train=None, spatial_labels_test=None, min_bins=None, level_categories=None):
    try:
        from scipy.stats import spearmanr as _scipy_spearmanr
    except Exception:
        _scipy_spearmanr = None
    if shuffle_session not in ('test', 'train'):
        raise ValueError("shuffle_session must be 'test' or 'train'")
    if shuffle_type not in ('map', 'neuron_id'):
        raise ValueError("shuffle_type must be 'map' or 'neuron_id'")
    rng = np.random.default_rng(random_state)
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    if rate_source == 'spike':
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == 'raw':
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")

    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    if train_ids is not None and test_ids is not None:
        tr_ids = _to_int_array(train_ids)
        te_ids = _to_int_array(test_ids)
        if tr_ids.shape[0] != te_ids.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr_ids.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr_ids.min() < 0 or tr_ids.max() >= n_tr:
            raise ValueError('train_ids out of range for TRAIN (zero-based).')
        if te_ids.min() < 0 or te_ids.max() >= n_te:
            raise ValueError('test_ids out of range for TEST (zero-based).')
        if n_neurons is not None and tr_ids.size > n_neurons:
            sel = rng.choice(tr_ids.size, int(n_neurons), replace=False)
            tr_ids = tr_ids[sel]
            te_ids = te_ids[sel]
        train_ids = tr_ids
        test_ids = te_ids
    else:
        (train_ids, test_ids) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == 'either':
        keep_mask = pass_tr & pass_te
    elif drop_if_below == 'both':
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    train_ids = np.asarray(train_ids)[keep_mask]
    test_ids = np.asarray(test_ids)[keep_mask]
    M = int(len(train_ids))
    if M == 0:
        return dict(sim_matrix=np.zeros((0, 0), float), sim_per_repeat=np.zeros((0,), float), mean=np.nan, std=np.nan, mode=mode, metric=metric, normalization=normalization, n_repeats=int(n_repeats), train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), shuffle_type=shuffle_type, shuffle_session=shuffle_session, meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), dropped=dict(n_pairs_initial=int(keep_mask.size), n_pairs_dropped=int(keep_mask.size - keep_mask.sum()), policy=drop_if_below))
    (maps_tr, mask_tr, meta_tr) = _fit_maps_for_session(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, train_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_train, level_categories=level_categories)
    (maps_te, mask_te, meta_te) = _fit_maps_for_session(spike_counts_test, raw_counts_test, selected_df_test, pos_te, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_test, level_categories=level_categories)
    if maps_tr.shape != maps_te.shape:
        raise ValueError(f'Train/Test maps shape mismatch after selection: {maps_tr.shape} vs {maps_te.shape}')
    mask = mask_tr & mask_te if intersect_mask else mask_tr | mask_te
    idx = np.where(mask)[0]
    min_bins_failed = min_bins is not None and idx.size < int(min_bins)
    if idx.size == 0 or min_bins_failed:
        return dict(sim_matrix=np.zeros((int(n_repeats), M), float), sim_per_repeat=np.zeros((int(n_repeats),), float), mean=np.nan, std=np.nan, mode=mode, metric=metric, normalization=normalization, n_repeats=int(n_repeats), train_ids=train_ids, test_ids=test_ids, shuffle_type=shuffle_type, shuffle_session=shuffle_session, meta_train=meta_tr | dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source, min_bins=min_bins, min_bins_failed=bool(min_bins_failed)), meta_test=meta_te | dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source, min_bins=min_bins, min_bins_failed=bool(min_bins_failed)), dropped=dict(n_pairs_initial=int(keep_mask.size), n_pairs_dropped=int(keep_mask.size - keep_mask.sum()), policy=drop_if_below))
    A_full = maps_tr
    B_full = maps_te
    A_base = A_full[idx, :]
    B_base = B_full[idx, :]
    K_full = A_full.shape[0]
    if normalization == 'auto':
        norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        norm_mode = normalization

    def _normalize_columns(X, mode='l2', eps=1e-12):
        X = X.astype(float, copy=True)
        if mode == 'zscore':
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True) + eps
            return (X - mu) / sd
        elif mode in ('l2', 'length'):
            nrm = np.linalg.norm(X, axis=0, keepdims=True) + eps
            return X / nrm
        elif mode == 'l1':
            nrm = np.abs(X).sum(axis=0, keepdims=True) + eps
            return X / nrm
        elif mode == 'none':
            return X
        else:
            return X

    def _spearman(u, v):
        if _scipy_spearmanr is not None:
            return float(_scipy_spearmanr(u, v)[0])
        ru = np.argsort(np.argsort(u))
        rv = np.argsort(np.argsort(v))
        u0 = (ru - ru.mean()) / (ru.std() + eps)
        v0 = (rv - rv.mean()) / (rv.std() + eps)
        return float(np.dot(u0, v0) / (np.linalg.norm(u0) * np.linalg.norm(v0) + eps))
    sim_rows = []
    sim_means = []
    for _ in range(n_repeats):
        if shuffle_type == 'map':
            if shuffle_session == 'test':
                B_shuf = _shuffle_rows_independent(B_full, rng)
                A_use = A_base
                B_use = B_shuf[idx, :]
            else:
                A_shuf = _shuffle_rows_independent(A_full, rng)
                A_use = A_shuf[idx, :]
                B_use = B_base
        elif shuffle_session == 'test':
            perm_cols = rng.permutation(M)
            A_use = A_base
            B_use = B_base[:, perm_cols]
        else:
            perm_cols = rng.permutation(M)
            A_use = A_base[:, perm_cols]
            B_use = B_base
        A_n = _normalize_columns(A_use, mode=norm_mode, eps=eps)
        B_n = _normalize_columns(B_use, mode=norm_mode, eps=eps)
        vals = np.empty(M, float)
        for i in range(M):
            a = A_n[:, i]
            b = B_n[:, i]
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() == 0:
                vals[i] = np.nan
            else:
                (aa, bb) = (a[good], b[good])
                if metric == 'pearson':
                    aa0 = aa - aa.mean()
                    bb0 = bb - bb.mean()
                    denom = np.linalg.norm(aa0) * np.linalg.norm(bb0) + eps
                    vals[i] = float(np.dot(aa0, bb0) / denom)
                elif metric == 'spearman':
                    vals[i] = _spearman(aa, bb)
                else:
                    denom = np.linalg.norm(aa) * np.linalg.norm(bb) + eps
                    vals[i] = float(np.dot(aa, bb) / denom)
        sim_rows.append(vals)
        sim_means.append(np.nanmean(vals))
    sim_matrix = np.vstack(sim_rows) if sim_rows else np.zeros((0, M), float)
    sim_per_repeat = np.asarray(sim_means, float)
    meta_tr |= dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source, min_bins=min_bins, min_bins_failed=False)
    meta_te |= dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source, min_bins=min_bins, min_bins_failed=False)
    return dict(sim_matrix=sim_matrix, sim_per_repeat=sim_per_repeat, mean=float(np.nanmean(sim_per_repeat)) if sim_per_repeat.size else np.nan, std=float(np.nanstd(sim_per_repeat)) if sim_per_repeat.size else np.nan, mode=mode, metric=metric, normalization=normalization, n_repeats=int(n_repeats), train_ids=train_ids, test_ids=test_ids, shuffle_type=shuffle_type, shuffle_session=shuffle_session, meta_train=meta_tr, meta_test=meta_te, dropped=dict(n_pairs_initial=int(keep_mask.size), n_pairs_dropped=int(keep_mask.size - keep_mask.sum()), policy=drop_if_below))

def pf_shuffle_pv_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, eps=1e-12, n_neurons=None, random_state=0, train_ids=None, test_ids=None, exclude_train_ids=None, exclude_test_ids=None, exclude_pairs=None, metric='pearson', normalization='auto', intersect_mask=True, min_neurons=3, repeats=1000, rng=None, return_observed=True, shuffle_type='neuron_id', shuffle_session='test', min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', level_categories=None):
    if rng is None:
        rng = np.random.default_rng(random_state)
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    if shuffle_type not in ('neuron_id', 'map'):
        raise ValueError("shuffle_type must be 'neuron_id' or 'map'")
    if shuffle_session not in ('test', 'train'):
        raise ValueError("shuffle_session must be 'test' or 'train'")
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    if rate_source == 'spike':
        Ctr_for_rate = spike_counts_train
        Cte_for_rate = spike_counts_test
    elif rate_source == 'raw':
        Ctr_for_rate = raw_counts_train
        Cte_for_rate = raw_counts_test
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")

    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    if train_ids is not None and test_ids is not None:
        tr = _to_int_array(train_ids)
        te = _to_int_array(test_ids)
        if tr.shape[0] != te.shape[0]:
            raise ValueError('train_ids and test_ids must have the same length.')
        if tr.size == 0:
            raise ValueError('Empty train_ids/test_ids.')
        if tr.min() < 0 or tr.max() >= n_tr:
            raise ValueError('train_ids out of range for TRAIN session (expected zero-based).')
        if te.min() < 0 or te.max() >= n_te:
            raise ValueError('test_ids out of range for TEST session (expected zero-based).')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (train_ids, test_ids) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
    if len(train_ids) == 0:
        raise ValueError('No matched neurons selected.')
    train_ids = np.asarray(train_ids, dtype=int)
    test_ids = np.asarray(test_ids, dtype=int)
    n_pairs_initial = int(train_ids.size)
    keep_user = np.ones(train_ids.size, dtype=bool)
    if exclude_train_ids is not None:
        ex_tr = _to_int_array(exclude_train_ids)
        if ex_tr.size:
            keep_user &= ~np.isin(train_ids, ex_tr)
    if exclude_test_ids is not None:
        ex_te = _to_int_array(exclude_test_ids)
        if ex_te.size:
            keep_user &= ~np.isin(test_ids, ex_te)
    if exclude_pairs is not None:
        ex_pairs = np.asarray(exclude_pairs, dtype=int)
        if not (ex_pairs.ndim == 1 and ex_pairs.size == 0):
            if ex_pairs.ndim != 2 or ex_pairs.shape[1] != 2:
                raise ValueError('exclude_pairs must be array-like of (train_id, test_id) pairs.')
            ex_set = set(map(tuple, ex_pairs.tolist()))
            pair_mask = np.array([(tr, te) not in ex_set for (tr, te) in zip(train_ids, test_ids)], dtype=bool)
            keep_user &= pair_mask
    train_ids = train_ids[keep_user]
    test_ids = test_ids[keep_user]
    n_dropped_by_user = int((~keep_user).sum())
    if train_ids.size == 0:
        out = dict(pv_shuffles=np.empty((0, 0), float), pv_summary=dict(mean=np.nan, std=np.nan, median=np.nan), metric=metric, mode=mode, normalization='zscore' if metric in ('pearson', 'spearman') and normalization == 'auto' else 'l2' if metric == 'cosine' and normalization == 'auto' else normalization, train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), meta_train=dict(rate_per_min=rate_tr, threshold=min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min, rate_source=rate_source), shuffle_type=shuffle_type, shuffle_session=shuffle_session, repeats=int(repeats), dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=0, n_pairs_final=0, policy=drop_if_below))
        if return_observed:
            out.update(dict(pv_obs=np.array([], float), mask=None, maps_tr=None, maps_te=None))
        return out
    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    pass_tr = np.ones_like(train_ids, dtype=bool)
    pass_te = np.ones_like(test_ids, dtype=bool)
    if thr_tr is not None:
        pass_tr = rate_tr[np.asarray(train_ids)] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[np.asarray(test_ids)] >= float(thr_te)
    if drop_if_below == 'either':
        keep_mask = pass_tr & pass_te
    elif drop_if_below == 'both':
        keep_mask = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    n_dropped_by_rate = int((~keep_mask).sum())
    train_ids = np.asarray(train_ids)[keep_mask]
    test_ids = np.asarray(test_ids)[keep_mask]
    if train_ids.size == 0:
        out = dict(pv_shuffles=np.empty((0, 0), float), pv_summary=dict(mean=np.nan, std=np.nan, median=np.nan), metric=metric, mode=mode, normalization='zscore' if metric in ('pearson', 'spearman') and normalization == 'auto' else 'l2' if metric == 'cosine' and normalization == 'auto' else normalization, train_ids=np.array([], dtype=int), test_ids=np.array([], dtype=int), meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), shuffle_type=shuffle_type, shuffle_session=shuffle_session, repeats=int(repeats), dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=n_dropped_by_rate, n_pairs_final=0, policy=drop_if_below))
        if return_observed:
            out.update(dict(pv_obs=np.array([], float), mask=None, maps_tr=None, maps_te=None))
        return out
    (maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te) = _prefit_train_and_test_maps(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, spike_counts_test, raw_counts_test, selected_df_test, pos_te, train_ids, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, level_categories=level_categories)
    if normalization == 'auto':
        norm_mode = 'zscore' if metric in ('pearson', 'spearman') else 'l2'
    else:
        norm_mode = normalization
    (pv_obs, mask) = _pv_from_maps(maps_tr, maps_te, mask_tr, mask_te, metric=metric, normalization=norm_mode, intersect_mask=intersect_mask, min_neurons=min_neurons, eps=eps)
    pv_shuffles = []
    for _ in range(repeats):
        if shuffle_type == 'neuron_id':
            if shuffle_session == 'test':
                perm = rng.permutation(maps_te.shape[1])
                maps_te_shuf = maps_te[:, perm]
                maps_tr_use = maps_tr
                maps_te_use = maps_te_shuf
            else:
                perm = rng.permutation(maps_tr.shape[1])
                maps_tr_shuf = maps_tr[:, perm]
                maps_tr_use = maps_tr_shuf
                maps_te_use = maps_te
        elif shuffle_session == 'test':
            maps_te_shuf = _shuffle_rows_independent(maps_te, rng)
            maps_tr_use = maps_tr
            maps_te_use = maps_te_shuf
        else:
            maps_tr_shuf = _shuffle_rows_independent(maps_tr, rng)
            maps_tr_use = maps_tr_shuf
            maps_te_use = maps_te
        (pv_shuf, _) = _pv_from_maps(maps_tr_use, maps_te_use, mask_tr, mask_te, metric=metric, normalization=norm_mode, intersect_mask=intersect_mask, min_neurons=min_neurons, eps=eps)
        pv_shuffles.append(pv_shuf)
    pv_shuffles = np.vstack(pv_shuffles) if pv_shuffles else np.empty((0, 0), float)
    pv_summary = dict(mean=float(np.nanmean(pv_shuffles)) if pv_shuffles.size else np.nan, std=float(np.nanstd(pv_shuffles)) if pv_shuffles.size else np.nan, median=float(np.nanmedian(pv_shuffles)) if pv_shuffles.size else np.nan)
    out = dict(pv_shuffles=pv_shuffles, pv_summary=pv_summary, metric=metric, mode=mode, normalization=norm_mode, train_ids=train_ids, test_ids=test_ids, meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), shuffle_type=shuffle_type, shuffle_session=shuffle_session, repeats=int(repeats), dropped=dict(n_pairs_initial=n_pairs_initial, n_pairs_dropped_by_user=n_dropped_by_user, n_pairs_dropped_by_rate=n_dropped_by_rate, n_pairs_final=int(train_ids.size), policy=drop_if_below))
    if return_observed:
        out.update(dict(pv_obs=pv_obs, mask=mask, maps_tr=maps_tr, maps_te=maps_te))
    return out

def pivot_window_tuning_similarity(spike_counts, raw_counts, selected_df, positions, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=0, use_peak=False, filter_ratemap=False, sigma=1.0, neuron_ids=None, random_state=0, win_len_sec=20 * 60, pivot_step_sec=60, metric='pearson', normalization='auto', intersect_mask=True, min_bins=3, eps=1e-12, min_time_per_bin_sec=5.0, min_spikes_per_min=None, rate_source='spike', pivot_time_axis='selected', frame_dt_sec=None, level_categories=None, *, permute_post_window_level=False):
    (T_sel, N_total) = spike_counts.shape
    assert 'frame' in selected_df.columns, "selected_df must contain a 'frame' column"
    assert T_sel == len(selected_df), 'spike_counts rows must match selected_df rows'
    if permute_post_window_level and mode != 'level':
        raise ValueError("permute_post_window_level is only supported for mode='level'.")
    rng = np.random.default_rng(random_state) if permute_post_window_level else None
    pos = np.asarray(positions)
    is_1d = pos.ndim == 1 or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError('pivot_window_tuning_similarity: for 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and use n_bins_x for the number of spatial bins.')
    if neuron_ids is None:
        neuron_ids = np.arange(N_total, dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    if pivot_time_axis == 'selected':
        t_sel = np.arange(T_sel, dtype=float) * float(dt)
        frame_dt_used = None
        frame_dt_inferred = False
    elif pivot_time_axis == 'frame':
        frame_idx = selected_df['frame'].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)
            frame_dt_inferred = True
        else:
            frame_dt_inferred = False
        t_sel = frame_idx * float(frame_dt_sec)
        frame_dt_used = float(frame_dt_sec)
    elif pivot_time_axis == 'maze':
        if 'maze_time' not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time'] in seconds.")
        t_sel = selected_df['maze_time'].to_numpy(dtype=float)
        frame_dt_used = None
        frame_dt_inferred = False
    else:
        raise ValueError("pivot_time_axis must be one of {'selected','frame','maze'}.")
    # The selected/frame/maze clock chooses window bounds only. Each retained
    # row still contributes its sampled exposure, even when selected_df has gaps.
    dt_eff = float(dt) * (int(time_window) + 1)

    if rate_source == 'spike':
        C_for_rate = spike_counts
        dt_for_rate = dt
    elif rate_source == 'raw':
        frames_cont = selected_df['frame'].to_numpy().astype(int)
        C_for_rate = raw_counts[frames_cont, :]
        dt_for_rate = float(dt)
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    rate_per_min = _spikes_per_min(C_for_rate, dt_for_rate)
    if min_spikes_per_min is not None:
        thr = float(min_spikes_per_min)
        keep_mask = rate_per_min[neuron_ids] >= thr
        neuron_ids = neuron_ids[keep_mask]
        if neuron_ids.size == 0:
            W = float(win_len_sec)
            if t_sel[-1] - t_sel[0] < 2 * W:
                piv_times = np.array([], dtype=float)
            else:
                t0 = t_sel[0] + W
                t1 = t_sel[-1] - W
                piv_times = np.arange(t0, t1 + 1e-09, float(pivot_step_sec), dtype=float)
            return dict(pivot_bounds=np.zeros((piv_times.size, 2, 2), dtype=int), pivot_times=piv_times, pivot_times_maze=None, sim_matrix=np.zeros((piv_times.size, 0), dtype=float), sim_mean=np.full(piv_times.size, np.nan, float), sim_std=np.full(piv_times.size, np.nan, float), mode=mode, metric=metric, normalization=normalization, meta=dict(x_edges=None, y_edges=None, n_bins_x=n_bins_x, n_bins_y=n_bins_y, kept_neuron_ids=np.array([], dtype=int), rate_per_min=rate_per_min, threshold=thr, rate_source=rate_source, pivot_time_axis=pivot_time_axis, frame_dt_used=frame_dt_used, frame_dt_inferred=frame_dt_inferred, dt=dt, time_window=time_window, sample_exposure_sec=dt_eff, win_len_sec=win_len_sec, pivot_step_sec=pivot_step_sec, intersect_mask=bool(intersect_mask), min_time_per_bin_sec=min_time_per_bin_sec, min_bins=min_bins, post_window_level_permutation_enabled=bool(permute_post_window_level)))
    M = int(neuron_ids.size)
    n_bins = n_bins_x * n_bins_y
    (labels_pos, x_edges, y_edges) = discretize_positions(pos, n_bins_x, n_bins_y)
    labels_lvl = None
    uniq_lvls = None
    if mode in ('level', 'joint_lvl'):
        v = selected_df[level_col].values
        if level_categories is None:
            v_num = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy(dtype=float)
            finite_v = v_num[np.isfinite(v_num)]
            if finite_v.size == v_num.size and np.allclose(finite_v, np.round(finite_v)) and np.nanmin(finite_v) >= 0 and np.nanmax(finite_v) <= 7:
                uniq_lvls = np.arange(8, dtype=int)
                labels_lvl = v_num.astype(int)
            else:
                labels_lvl, uniq_lvls, _ = _level_values_to_labels(v)
        else:
            labels_lvl, uniq_lvls, _ = _level_values_to_labels(v, level_categories)
        n_lvls = len(uniq_lvls)
    labels_hd = None
    uniq_hd = None
    if mode in ('hd', 'joint_pos'):
        h = selected_df[head_col].values
        uniq_hd = np.unique(h)
        hd_map = {val: i for (i, val) in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[val] for val in h])
        n_hd = len(uniq_hd) if uniq_hd is not None else None
    W = float(win_len_sec)
    step_sec = float(pivot_step_sec)
    if t_sel[-1] - t_sel[0] <= 2 * W:
        raise ValueError('Session too short for chosen win_len_sec on the selected time axis.')
    t0 = t_sel[0] + W
    t1 = t_sel[-1] - W
    pivot_times = np.arange(t0, t1 + 1e-09, step_sec, dtype=float)
    n_pivots = pivot_times.size
    bounds = np.zeros((n_pivots, 2, 2), dtype=int)
    for (j, tp) in enumerate(pivot_times):
        s_b = int(np.searchsorted(t_sel, tp - W, side='left'))
        e_b = int(np.searchsorted(t_sel, tp, side='left'))
        s_a = int(np.searchsorted(t_sel, tp, side='left'))
        e_a = int(np.searchsorted(t_sel, tp + W, side='left'))
        bounds[j, 0, :] = [s_b, e_b]
        bounds[j, 1, :] = [s_a, e_a]
    if 'maze_time' in selected_df.columns and n_pivots > 0:
        maze_t = selected_df['maze_time'].to_numpy(dtype=float)
        pivot_center_idx = bounds[:, 1, 0]
        pivot_times_maze = maze_t[pivot_center_idx]
    else:
        pivot_times_maze = None
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    if normalization == 'auto':
        col_norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        col_norm_mode = normalization

    def _normalize_columns(X, mode_):
        if mode_ == 'none':
            return X
        if mode_ == 'zscore':
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True)
            return (X - mu) / (sd + eps)
        if mode_ == 'l2':
            nrm = np.linalg.norm(X, axis=0, keepdims=True)
            return X / (nrm + eps)
        return X

    def _per_neuron_similarity(A, B):
        A = _normalize_columns(A, col_norm_mode)
        B = _normalize_columns(B, col_norm_mode)
        out = np.full(M, np.nan, float)
        for i in range(M):
            (a, b) = (A[:, i], B[:, i])
            good = np.isfinite(a) & np.isfinite(b)
            if good.sum() < min_bins:
                continue
            if metric == 'pearson':
                out[i] = np.corrcoef(a[good], b[good])[0, 1]
            elif metric == 'spearman':
                out[i] = spearmanr(a[good], b[good])[0]
            else:
                na = np.linalg.norm(a[good])
                nb = np.linalg.norm(b[good])
                out[i] = float(a[good] @ b[good] / (na * nb + eps))
        return out
    Q_full = build_count_matrix(spike_counts, raw_counts, selected_df, frames=np.arange(T_sel), time_window=time_window, neuron_ids=neuron_ids).T

    def _occ_secs_spatial(labels, n_bins_local):
        return np.bincount(labels, minlength=n_bins_local).astype(float) * dt_eff

    def _occ_secs_level(labels_level, nL):
        return np.bincount(labels_level, minlength=nL).astype(float) * dt_eff

    def _occ_secs_hd(labels_hd_local, nH):
        return np.bincount(labels_hd_local, minlength=nH).astype(float) * dt_eff

    def _occ_secs_joint_pos(pos_labels, hd_labels, nB, nH):
        idx = pos_labels.astype(int) * nH + hd_labels.astype(int)
        occ = np.bincount(idx, minlength=nB * nH).astype(float) * dt_eff
        return occ.reshape(nB, nH)

    def _occ_secs_joint_lvl(pos_labels, lvl_labels, nB, nL):
        occL = np.zeros((nL, nB), dtype=float)
        for L in range(nL):
            m = lvl_labels == L
            if m.any():
                occL[L] = np.bincount(pos_labels[m], minlength=nB).astype(float) * dt_eff
        return occL

    def _norm_entropy(occ, denom):
        occ = np.asarray(occ, dtype=float)
        occ = occ[np.isfinite(occ) & (occ > 0)]
        if occ.size == 0:
            return np.nan
        p = occ / np.sum(occ)
        return float(-np.sum(p * np.log(np.maximum(p, eps))) / denom)

    def _nonidentity_permutation(n):
        n = int(n)
        if n <= 1:
            return np.arange(n, dtype=int)
        perm = rng.permutation(n)
        if np.array_equal(perm, np.arange(n)):
            perm = np.roll(perm, 1)
        return perm

    def _selected_speed_series():
        if 'speed' in selected_df.columns:
            return pd.to_numeric(selected_df['speed'], errors='coerce').to_numpy(dtype=float), 'selected_df.speed', 2.0
        arr = np.asarray(positions, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != T_sel or arr.shape[1] < 2:
            return np.full(T_sel, np.nan, dtype=float), 'unavailable', np.nan
        speed = np.full(T_sel, np.nan, dtype=float)
        if T_sel > 1:
            dxy = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
            if 'frame' in selected_df.columns:
                frames = selected_df['frame'].to_numpy(dtype=float)
                dframe = np.diff(frames)
                adjacent = dframe == 1
                denom = np.where(adjacent, np.maximum(dframe * float(dt), eps), np.nan)
            else:
                denom = np.full(T_sel - 1, max(float(dt), eps), dtype=float)
            speed[1:] = dxy / denom
        return speed, 'position_diff_adjacent_selected_frames', np.nan

    speed_series, speed_source, low_speed_threshold = _selected_speed_series()
    sim_rows = []
    n_valid_levels = np.full(n_pivots, np.nan, dtype=float)
    occ_entropy_pre = np.full(n_pivots, np.nan, dtype=float)
    occ_entropy_post = np.full(n_pivots, np.nan, dtype=float)
    frac_valid_bins_eq2 = np.full(n_pivots, np.nan, dtype=float)
    frac_valid_bins_eq3 = np.full(n_pivots, np.nan, dtype=float)
    frac_valid_bins_ge4 = np.full(n_pivots, np.nan, dtype=float)
    selected_speed_mean = np.full(n_pivots, np.nan, dtype=float)
    selected_low_speed_fraction = np.full(n_pivots, np.nan, dtype=float)
    for j in range(n_pivots):
        (s_b, e_b) = bounds[j, 0]
        (s_a, e_a) = bounds[j, 1]
        speed_win = speed_series[s_b:e_a]
        if np.isfinite(speed_win).any():
            selected_speed_mean[j] = float(np.nanmean(speed_win))
            if np.isfinite(low_speed_threshold):
                selected_low_speed_fraction[j] = float(np.nanmean(speed_win <= low_speed_threshold))
        Qb = Q_full[s_b:e_b, :]
        Qa = Q_full[s_a:e_a, :]
        yb_pos = labels_pos[s_b:e_b]
        ya_pos = labels_pos[s_a:e_a]
        if mode == 'spatial':
            occ_b = _occ_secs_spatial(yb_pos, n_bins)
            occ_a = _occ_secs_spatial(ya_pos, n_bins)
            (lamb_b, _pr_b) = tuner(Qb, yb_pos, None, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            (lamb_a, _pr_a) = tuner(Qa, ya_pos, None, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            (A, B) = (lamb_b[mask, :], lamb_a[mask, :])
        elif mode == 'level':
            yb_lvl = labels_lvl[s_b:e_b]
            nL = int(n_lvls)
            ya_lvl = labels_lvl[s_a:e_a]
            occ_b = _occ_secs_level(yb_lvl, nL)
            occ_a = _occ_secs_level(ya_lvl, nL)
            (lamb_b, _) = tuner(Qb, yb_lvl, None, nL, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            (lamb_a, _) = tuner(Qa, ya_lvl, None, nL, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            (A, B) = (lamb_b[mask, :], lamb_a[mask, :])
            n_valid_levels[j] = float(np.sum(mask))
            entropy_denom = np.log(float(max(nL, 2)))
            occ_entropy_pre[j] = _norm_entropy(occ_b[mask], entropy_denom)
            occ_entropy_post[j] = _norm_entropy(occ_a[mask], entropy_denom)
            if permute_post_window_level and B.shape[0] > 1:
                B = B[_nonidentity_permutation(B.shape[0]), :]
        elif mode == 'hd':
            yb_hd = labels_hd[s_b:e_b]
            nH = int(np.max(labels_hd)) + 1
            ya_hd = labels_hd[s_a:e_a]
            occ_b = _occ_secs_hd(yb_hd, nH)
            occ_a = _occ_secs_hd(ya_hd, nH)
            (_, _, lamb_b, _) = tuner(Qb, yb_pos, yb_hd, n_bins, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='independent')
            (_, _, lamb_a, _) = tuner(Qa, ya_pos, ya_hd, n_bins, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='independent')
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            (A, B) = (lamb_b[mask, :], lamb_a[mask, :])
        elif mode == 'joint_pos':
            yb_hd = labels_hd[s_b:e_b]
            nH = int(np.max(labels_hd)) + 1
            ya_hd = labels_hd[s_a:e_a]
            occ_b = _occ_secs_joint_pos(yb_pos, yb_hd, n_bins, nH)
            occ_a = _occ_secs_joint_pos(ya_pos, ya_hd, n_bins, nH)
            (lamb_b, _) = tuner(Qb, yb_pos, yb_hd, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='joint')
            (lamb_a, _) = tuner(Qa, ya_pos, ya_hd, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='joint')
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = lamb_b[mask].reshape(-1, lamb_b.shape[-1])
            B = lamb_a[mask].reshape(-1, lamb_a.shape[-1])
        elif mode == 'joint_lvl':
            yb_lvl = labels_lvl[s_b:e_b]
            nL = int(n_lvls)
            ya_lvl = labels_lvl[s_a:e_a]
            occ_b = _occ_secs_joint_lvl(yb_pos, yb_lvl, n_bins, nL)
            occ_a = _occ_secs_joint_lvl(ya_pos, ya_lvl, n_bins, nL)
            (lamL_b, lamL_a) = ([], [])
            for Lidx in range(nL):
                mb = yb_lvl == Lidx
                ma = ya_lvl == Lidx
                if mb.any():
                    (lb, _) = tuner(Qb[mb], yb_pos[mb], None, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
                else:
                    lb = np.zeros((n_bins, M))
                if ma.any():
                    (la, _) = tuner(Qa[ma], ya_pos[ma], None, n_bins, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
                else:
                    la = np.zeros((n_bins, M))
                lamL_b.append(lb)
                lamL_a.append(la)
            lamb_b = np.stack(lamL_b, axis=0)
            lamb_a = np.stack(lamL_a, axis=0)
            if intersect_mask:
                mask = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec)
            else:
                mask = (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = lamb_b[mask].reshape(-1, M)
            B = lamb_a[mask].reshape(-1, M)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
        if A.size == 0 or B.size == 0:
            sim_rows.append(np.full(M, np.nan, float))
            continue
        valid_counts = np.sum(np.isfinite(A) & np.isfinite(B), axis=0)
        if valid_counts.size:
            frac_valid_bins_eq2[j] = float(np.mean(valid_counts == 2))
            frac_valid_bins_eq3[j] = float(np.mean(valid_counts == 3))
            frac_valid_bins_ge4[j] = float(np.mean(valid_counts >= 4))
        sim_rows.append(_per_neuron_similarity(A, B))
    sim_matrix = np.vstack(sim_rows) if len(sim_rows) else np.zeros((0, M))
    sim_mean = np.nanmean(sim_matrix, axis=1) if sim_matrix.size else np.array([])
    sim_std = np.nanstd(sim_matrix, axis=1) if sim_matrix.size else np.array([])
    retained_neuron_count = np.sum(np.isfinite(sim_matrix), axis=1).astype(int) if sim_matrix.size else np.zeros(n_pivots, dtype=int)
    occ_entropy_mean = np.full(n_pivots, np.nan, dtype=float)
    occ_entropy_stack = np.vstack([occ_entropy_pre, occ_entropy_post])
    occ_entropy_count = np.sum(np.isfinite(occ_entropy_stack), axis=0)
    if np.any(occ_entropy_count > 0):
        occ_entropy_mean[occ_entropy_count > 0] = (
            np.nansum(occ_entropy_stack[:, occ_entropy_count > 0], axis=0)
            / occ_entropy_count[occ_entropy_count > 0]
        )
    meta = dict(x_edges=x_edges, y_edges=y_edges, n_bins_x=n_bins_x, n_bins_y=n_bins_y, kept_neuron_ids=neuron_ids, rate_per_min=rate_per_min, threshold=min_spikes_per_min, rate_source=rate_source, pivot_time_axis=pivot_time_axis, frame_dt_used=frame_dt_used, frame_dt_inferred=frame_dt_inferred, pivot_times_maze=pivot_times_maze, dt=dt, time_window=time_window, sample_exposure_sec=dt_eff, win_len_sec=win_len_sec, pivot_step_sec=pivot_step_sec, intersect_mask=bool(intersect_mask), min_time_per_bin_sec=min_time_per_bin_sec, min_bins=min_bins, n_valid_levels=n_valid_levels, occ_entropy_pre=occ_entropy_pre, occ_entropy_post=occ_entropy_post, occ_entropy_mean=occ_entropy_mean, level_entropy_denominator_n=int(n_lvls) if mode == 'level' else None, retained_neuron_count=retained_neuron_count, frac_neurons_valid_bins_eq2=frac_valid_bins_eq2, frac_neurons_valid_bins_eq3=frac_valid_bins_eq3, frac_neurons_valid_bins_ge4=frac_valid_bins_ge4, selected_speed_mean=selected_speed_mean, selected_low_speed_fraction=selected_low_speed_fraction, selected_speed_source=speed_source, selected_low_speed_threshold=low_speed_threshold, post_window_level_permutation_enabled=bool(permute_post_window_level), post_window_level_permutation_kind='post_valid_level_rows_nonidentity' if permute_post_window_level else 'none', random_state=random_state if permute_post_window_level else None)
    if uniq_lvls is not None:
        meta['levels'] = uniq_lvls
    if uniq_hd is not None:
        meta['head_bins'] = uniq_hd
    return dict(pivot_bounds=bounds, pivot_times=pivot_times, pivot_times_maze=pivot_times_maze, sim_matrix=sim_matrix, sim_mean=sim_mean, sim_std=sim_std, mode=mode, metric=metric, normalization=col_norm_mode, meta=meta)

def within_session_pv_temporal_similarity(spike_counts, raw_counts, selected_df, positions, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=0, use_peak=False, filter_ratemap=False, sigma=1.0, neuron_ids=None, random_state=0, win_len_sec=20 * 60, pivot_step_sec=60, pivot_time_axis='selected', frame_dt_sec=None, metric='pearson', normalization='auto', intersect_mask=True, eps=1e-12, min_time_per_bin_sec=5.0, min_spikes_per_min=None, rate_source='spike', *, spatial_labels=None, level_categories=None, return_window_maps=False, reference_maps=None, fisher_z=False, min_neurons=None, return_vector_stats=False):
    (T_sel, N_total) = spike_counts.shape
    assert 'frame' in selected_df.columns
    assert T_sel == len(selected_df)
    pos = np.asarray(positions)
    is_1d = pos.ndim == 1 or (pos.ndim == 2 and pos.shape[1] == 1)
    if spatial_labels is None and is_1d and n_bins_y != 1:
        raise ValueError('within_session_pv_temporal_similarity: for 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and use n_bins_x for the number of spatial bins.')
    if neuron_ids is None:
        neuron_ids = np.arange(N_total, dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    M = int(neuron_ids.size)
    if spatial_labels is None:
        (labels_pos, x_edges, y_edges) = discretize_positions(pos, n_bins_x, n_bins_y)
    else:
        labels_pos = np.asarray(spatial_labels, dtype=int).ravel()
        if labels_pos.shape[0] != T_sel:
            raise ValueError('spatial_labels length must match selected_df rows.')
        x_edges = None
        y_edges = None
    n_bins_pos = n_bins_x * n_bins_y
    labels_lvl = None
    uniq_lvls = None
    if mode in ('level', 'joint_lvl'):
        v = selected_df[level_col].values
        labels_lvl, uniq_lvls, _ = _level_values_to_labels(v, level_categories)
        nL = len(uniq_lvls)
    labels_hd = None
    uniq_hd = None
    if mode in ('hd', 'joint_pos'):
        h = selected_df[head_col].values
        uniq_hd = np.unique(h)
        hd_map = {val: i for (i, val) in enumerate(uniq_hd)}
        labels_hd = np.array([hd_map[val] for val in h])
        nH = len(uniq_hd)
    if pivot_time_axis == 'selected':
        t_sel = np.arange(T_sel, dtype=float) * float(dt)
        frame_dt_used = None
        frame_dt_inferred = False
    elif pivot_time_axis == 'frame':
        frame_idx = selected_df['frame'].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)
            frame_dt_inferred = True
        else:
            frame_dt_inferred = False
        t_sel = frame_idx * float(frame_dt_sec)
        frame_dt_used = float(frame_dt_sec)
    elif pivot_time_axis == 'maze':
        if 'maze_time' not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time']")
        t_sel = selected_df['maze_time'].to_numpy(dtype=float)
        frame_dt_used = None
        frame_dt_inferred = False
    else:
        raise ValueError("pivot_time_axis must be 'selected'|'frame'|'maze'")
    # The pivot clock selects rows; exposure is determined by sampled rows.
    dt_eff = float(dt) * (int(time_window) + 1)

    if rate_source == 'spike':
        C_for_rate = spike_counts
        dt_for_rate = dt
    elif rate_source == 'raw':
        frames_cont = selected_df['frame'].to_numpy().astype(int)
        C_for_rate = raw_counts[frames_cont, :]
        dt_for_rate = float(dt)
    else:
        raise ValueError("rate_source must be 'spike' or 'raw'.")
    rate_per_min = _spikes_per_min(C_for_rate, dt_for_rate)
    if min_spikes_per_min is not None:
        thr = float(min_spikes_per_min)
        neuron_ids = neuron_ids[rate_per_min[neuron_ids] >= thr]
        M = int(neuron_ids.size)
        if M == 0:
            return dict(pivot_bounds=np.zeros((0, 2, 2), int), pivot_times=np.zeros((0,), float), pivot_times_maze=None, sim_per_bin=np.zeros((0, 0), float), sim_mean=np.zeros((0,), float), mode=mode, metric=metric, normalization=normalization, meta=dict(x_edges=x_edges, y_edges=y_edges, n_bins_x=n_bins_x, n_bins_y=n_bins_y, kept_neuron_ids=np.array([], int), rate_per_min=rate_per_min, threshold=thr, rate_source=rate_source, pivot_time_axis=pivot_time_axis, frame_dt_used=frame_dt_used if pivot_time_axis == 'frame' else None, frame_dt_inferred=frame_dt_inferred if pivot_time_axis == 'frame' else False, sample_exposure_sec=dt_eff, min_time_per_bin_sec=min_time_per_bin_sec))
    W = float(win_len_sec)
    step_sec = float(pivot_step_sec)
    if reference_maps is None:
        if t_sel[-1] - t_sel[0] <= 2 * W:
            raise ValueError('Session too short for chosen win_len_sec on this axis.')
        t0 = t_sel[0] + W
        t1 = t_sel[-1] - W
        pivot_times = np.arange(t0, t1 + 1e-09, step_sec, dtype=float)
    else:
        if mode != 'spatial':
            raise ValueError("reference_maps is currently supported only for mode='spatial'.")
        half_w = W / 2.0
        if t_sel[-1] - t_sel[0] <= W:
            raise ValueError('Session too short for chosen win_len_sec on this axis.')
        t0 = t_sel[0] + half_w
        t1 = t_sel[-1] - half_w
        pivot_times = np.arange(t0, t1 + 1e-09, step_sec, dtype=float)
    n_pivots = pivot_times.size
    bounds = np.zeros((n_pivots, 2, 2), dtype=int)
    for (j, tp) in enumerate(pivot_times):
        if reference_maps is None:
            s_b = int(np.searchsorted(t_sel, tp - W, side='left'))
            e_b = int(np.searchsorted(t_sel, tp, side='left'))
            s_a = int(np.searchsorted(t_sel, tp, side='left'))
            e_a = int(np.searchsorted(t_sel, tp + W, side='left'))
        else:
            s_b = int(np.searchsorted(t_sel, tp - half_w, side='left'))
            e_b = int(np.searchsorted(t_sel, tp + half_w, side='left'))
            s_a = s_b
            e_a = e_b
        bounds[j, 0] = (s_b, e_b)
        bounds[j, 1] = (s_a, e_a)
    if 'maze_time' in selected_df.columns and n_pivots > 0:
        maze_t = selected_df['maze_time'].to_numpy(dtype=float)
        pivot_times_maze = maze_t[bounds[:, 1, 0]]
    else:
        pivot_times_maze = None
    if normalization == 'auto':
        row_norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        row_norm_mode = normalization

    def _rowwise_z(X):
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, keepdims=True)
        out = np.full_like(X, np.nan, dtype=float)
        valid = np.isfinite(sd[:, 0]) & (sd[:, 0] > eps)
        out[valid] = (X[valid] - mu[valid]) / sd[valid]
        return out

    def _rowwise_l2(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (n + eps)

    def _safe_row_stats(X):
        X = np.asarray(X, dtype=float)
        mean = np.full(X.shape[0], np.nan, dtype=float)
        std = np.full(X.shape[0], np.nan, dtype=float)
        nfinite = np.sum(np.isfinite(X), axis=1).astype(int)
        for idx in range(X.shape[0]):
            keep = np.isfinite(X[idx])
            if not np.any(keep):
                continue
            mean[idx] = float(np.mean(X[idx, keep]))
            std[idx] = float(np.std(X[idx, keep]))
        return (mean, std, nfinite)
    Q_full = build_count_matrix(spike_counts, raw_counts, selected_df, frames=np.arange(T_sel), time_window=time_window, neuron_ids=neuron_ids).T
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior

    def occ_secs_spatial(lbl, nB):
        return np.bincount(lbl, minlength=nB).astype(float) * dt_eff

    def occ_secs_level(lbl, nL):
        return np.bincount(lbl, minlength=nL).astype(float) * dt_eff

    def occ_secs_hd(lbl, nH):
        return np.bincount(lbl, minlength=nH).astype(float) * dt_eff

    def occ_secs_joint_pos(pos_lbl, hd_lbl, nB, nH):
        idx = pos_lbl.astype(int) * nH + hd_lbl.astype(int)
        occ = np.bincount(idx, minlength=nB * nH).astype(float) * dt_eff
        return occ.reshape(nB, nH)

    def occ_secs_joint_lvl(pos_lbl, lvl_lbl, nB, nL):
        out = np.zeros((nL, nB), float)
        for L in range(nL):
            m = lvl_lbl == L
            if m.any():
                out[L] = np.bincount(pos_lbl[m], minlength=nB).astype(float) * dt_eff
        return out
    if mode == 'spatial':
        K_full = n_bins_pos
    elif mode == 'level':
        K_full = nL
    elif mode == 'hd':
        K_full = nH
    elif mode == 'joint_pos':
        K_full = n_bins_pos * nH
    elif mode == 'joint_lvl':
        K_full = nL * n_bins_pos
    else:
        raise ValueError(f'Unknown mode: {mode}')
    sim_per_bin = np.full((n_pivots, K_full), np.nan, float)
    sim_mean = np.full(n_pivots, np.nan, float)
    window_maps = [] if return_window_maps else None
    window_vector_mean = np.full((n_pivots, K_full), np.nan, float) if return_vector_stats else None
    window_vector_std = np.full((n_pivots, K_full), np.nan, float) if return_vector_stats else None
    window_vector_nfinite = np.zeros((n_pivots, K_full), dtype=int) if return_vector_stats else None
    ref_maps = None
    ref_mask = None
    reference_vector_mean = None
    reference_vector_std = None
    reference_vector_nfinite = None
    if reference_maps is not None:
        ref_maps = np.asarray(reference_maps, dtype=float)
        if ref_maps.shape == (M, K_full):
            ref_maps = ref_maps.T
        if ref_maps.shape != (K_full, M):
            raise ValueError(
                f'reference_maps must have shape ({K_full}, {M}) or ({M}, {K_full}); got {ref_maps.shape}.'
            )
        ref_mask = np.any(np.isfinite(ref_maps), axis=1)
        if return_vector_stats:
            (reference_vector_mean, reference_vector_std, reference_vector_nfinite) = _safe_row_stats(ref_maps)
    for j in range(n_pivots):
        (s_b, e_b) = bounds[j, 0]
        (s_a, e_a) = bounds[j, 1]
        Qb = Q_full[s_b:e_b]
        Qa = Q_full[s_a:e_a]
        yb_pos = labels_pos[s_b:e_b]
        ya_pos = labels_pos[s_a:e_a]
        yb_lvl = labels_lvl[s_b:e_b] if labels_lvl is not None else None
        ya_lvl = labels_lvl[s_a:e_a] if labels_lvl is not None else None
        yb_hd = labels_hd[s_b:e_b] if labels_hd is not None else None
        ya_hd = labels_hd[s_a:e_a] if labels_hd is not None else None
        row_idx = None
        window_map_full = None
        if mode == 'spatial':
            occ_b = occ_secs_spatial(yb_pos, n_bins_pos)
            occ_a = occ_secs_spatial(ya_pos, n_bins_pos)
            (Ab, _) = tuner(Qb, yb_pos, None, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            (Aa, _) = tuner(Qa, ya_pos, None, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            window_map_full = Aa
            if return_vector_stats and window_map_full is not None:
                cur_mean, cur_std, cur_nfinite = _safe_row_stats(window_map_full)
                window_vector_mean[j, :] = cur_mean
                window_vector_std[j, :] = cur_std
                window_vector_nfinite[j, :] = cur_nfinite
            if reference_maps is not None:
                mask_cur = occ_a >= min_time_per_bin_sec
                pv, _ = _pv_from_maps(
                    ref_maps,
                    Aa,
                    ref_mask,
                    mask_cur,
                    metric=metric,
                    normalization=row_norm_mode,
                    intersect_mask=intersect_mask,
                    min_neurons=1 if min_neurons is None else int(min_neurons),
                    eps=eps,
                )
                sim_per_bin[j, :] = pv
                sim_mean[j] = float(np.nanmean(pv)) if np.isfinite(pv).any() else np.nan
                if return_window_maps:
                    window_maps.append(np.asarray(window_map_full, dtype=float))
                continue
            keep = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec) if intersect_mask else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = Ab[keep]
            B = Aa[keep]
            row_idx = np.where(keep)[0]
        elif mode == 'level':
            occ_b = occ_secs_level(yb_lvl, nL)
            occ_a = occ_secs_level(ya_lvl, nL)
            (Ab, _) = tuner(Qb, yb_lvl, None, nL, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            (Aa, _) = tuner(Qa, ya_lvl, None, nL, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
            window_map_full = Aa
            keep = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec) if intersect_mask else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = Ab[keep]
            B = Aa[keep]
            row_idx = np.where(keep)[0]
        elif mode == 'hd':
            occ_b = occ_secs_hd(yb_hd, nH)
            occ_a = occ_secs_hd(ya_hd, nH)
            (_, _, Ab, _) = tuner(Qb, yb_pos, yb_hd, n_bins_pos, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='independent')
            (_, _, Aa, _) = tuner(Qa, ya_pos, ya_hd, n_bins_pos, dt_eff, False, 0.0, min_time=min_time_per_bin_sec, eps=eps, decode_head='independent')
            window_map_full = Aa
            keep = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec) if intersect_mask else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = Ab[keep]
            B = Aa[keep]
            row_idx = np.where(keep)[0]
        elif mode == 'joint_pos':
            occ_b = occ_secs_joint_pos(yb_pos, yb_hd, n_bins_pos, nH)
            occ_a = occ_secs_joint_pos(ya_pos, ya_hd, n_bins_pos, nH)
            (AJ_b, _) = tuner(Qb, yb_pos, yb_hd, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='joint')
            (AJ_a, _) = tuner(Qa, ya_pos, ya_hd, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='joint')
            window_map_full = AJ_a.reshape(-1, AJ_a.shape[-1])
            keep = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec) if intersect_mask else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = AJ_b[keep].reshape(-1, AJ_b.shape[-1])
            B = AJ_a[keep].reshape(-1, AJ_a.shape[-1])
            row_idx = np.where(keep.reshape(-1))[0]
        elif mode == 'joint_lvl':
            occ_b = occ_secs_joint_lvl(yb_pos, yb_lvl, n_bins_pos, nL)
            occ_a = occ_secs_joint_lvl(ya_pos, ya_lvl, n_bins_pos, nL)
            lamL_b = []
            lamL_a = []
            for L in range(nL):
                mb = yb_lvl == L
                ma = ya_lvl == L
                if mb.any():
                    (lb, _) = tuner(Qb[mb], yb_pos[mb], None, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
                else:
                    lb = np.zeros((n_bins_pos, M))
                if ma.any():
                    (la, _) = tuner(Qa[ma], ya_pos[ma], None, n_bins_pos, dt_eff, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
                else:
                    la = np.zeros((n_bins_pos, M))
                lamL_b.append(lb)
                lamL_a.append(la)
            Ab = np.stack(lamL_b, 0)
            Aa = np.stack(lamL_a, 0)
            window_map_full = Aa.reshape(-1, M)
            keep = (occ_b >= min_time_per_bin_sec) & (occ_a >= min_time_per_bin_sec) if intersect_mask else (occ_b >= min_time_per_bin_sec) | (occ_a >= min_time_per_bin_sec)
            A = Ab[keep].reshape(-1, M)
            B = Aa[keep].reshape(-1, M)
            row_idx = np.where(keep.reshape(-1))[0]
        else:
            raise ValueError(f'Unknown mode: {mode}')
        if return_vector_stats and window_map_full is not None:
            window_map_arr = np.asarray(window_map_full, dtype=float)
            cur_mean, cur_std, cur_nfinite = _safe_row_stats(window_map_arr)
            window_vector_mean[j, :] = cur_mean
            window_vector_std[j, :] = cur_std
            window_vector_nfinite[j, :] = cur_nfinite
        if A.size == 0 or B.size == 0:
            if return_window_maps:
                window_maps.append(np.asarray(window_map_full, dtype=float) if window_map_full is not None else np.zeros((K_full, M), dtype=float))
            continue
        if min_neurons is not None:
            enough = (np.isfinite(A) & np.isfinite(B)).sum(axis=1) >= int(min_neurons)
            A = A[enough]
            B = B[enough]
            row_idx = row_idx[enough]
            if A.size == 0 or B.size == 0:
                if return_window_maps:
                    window_maps.append(np.asarray(window_map_full, dtype=float) if window_map_full is not None else np.zeros((K_full, M), dtype=float))
                continue
        if metric == 'spearman':
            A = np.apply_along_axis(rankdata, 1, A)
            B = np.apply_along_axis(rankdata, 1, B)
            if normalization == 'auto' or row_norm_mode == 'zscore':
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == 'l2':
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            denom = max(A.shape[1], 1)
            s = np.sum(A * B, axis=1) / denom
            s = np.clip(s, -1.0, 1.0)
        elif metric == 'pearson':
            if normalization == 'auto' or row_norm_mode == 'zscore':
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == 'l2':
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            denom = max(A.shape[1], 1)
            s = np.sum(A * B, axis=1) / denom
            s = np.clip(s, -1.0, 1.0)
        else:
            if row_norm_mode == 'zscore':
                A = _rowwise_z(A)
                B = _rowwise_z(B)
            elif row_norm_mode == 'l2':
                A = _rowwise_l2(A)
                B = _rowwise_l2(B)
            na = np.linalg.norm(A, axis=1)
            nb = np.linalg.norm(B, axis=1)
            s = np.sum(A * B, axis=1) / (na * nb + eps)
        sim_per_bin[j, row_idx] = s
        sim_mean[j] = float(np.nanmean(s)) if np.isfinite(s).any() else np.nan
        if return_window_maps:
            window_maps.append(np.asarray(window_map_full, dtype=float) if window_map_full is not None else np.zeros((K_full, M), dtype=float))
    if fisher_z:
        sim_per_bin = np.where(
            np.isfinite(sim_per_bin),
            np.arctanh(np.clip(sim_per_bin, -0.9999, 0.9999)),
            np.nan,
        )
        sim_mean = np.asarray(
            [float(np.nanmean(row)) if np.isfinite(row).any() else np.nan for row in sim_per_bin],
            dtype=float,
        ) if sim_per_bin.size else np.array([])
    meta = dict(x_edges=x_edges, y_edges=y_edges, n_bins_x=n_bins_x, n_bins_y=n_bins_y, kept_neuron_ids=neuron_ids, rate_per_min=rate_per_min, threshold=min_spikes_per_min, rate_source=rate_source, pivot_time_axis=pivot_time_axis, frame_dt_used=frame_dt_used if pivot_time_axis == 'frame' else None, frame_dt_inferred=frame_dt_inferred if pivot_time_axis == 'frame' else False, sample_exposure_sec=dt_eff, min_time_per_bin_sec=min_time_per_bin_sec, levels=uniq_lvls if uniq_lvls is not None else None, head_bins=uniq_hd if uniq_hd is not None else None, fisher_z=bool(fisher_z), reference_mode=reference_maps is not None, min_neurons=min_neurons)
    out = dict(pivot_bounds=bounds, pivot_times=pivot_times, pivot_times_maze=pivot_times_maze, sim_per_bin=sim_per_bin, sim_mean=sim_mean, mode=mode, metric=metric, normalization=row_norm_mode, meta=meta)
    if return_window_maps:
        out['window_maps'] = window_maps
    if return_vector_stats:
        out['window_vector_mean'] = window_vector_mean
        out['window_vector_std'] = window_vector_std
        out['window_vector_nfinite'] = window_vector_nfinite
        out['reference_vector_mean'] = reference_vector_mean
        out['reference_vector_std'] = reference_vector_std
        out['reference_vector_nfinite'] = reference_vector_nfinite
    return out

def within_level_component_shuffle_spatial_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, level_col='level', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, n_neurons=None, random_state=0, train_ids=None, test_ids=None, metric='pearson', normalization='auto', intersect_mask=True, eps=1e-12, min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', n_repeats=1000, shuffle_session='test', transform_scope='global', shuffle_domain='connected_component', majority_min_frac=0.5, drop_unknown_level_bins=True, reverse_prob=0.5, smoothness_match=False, smoothness_target=1.0, smoothness_tol=0.2, max_attempts_multiplier=20, smoothness_filter=False, smoothness_filter_mode='adaptive', smoothness_edge_domain='within_component', smoothness_filter_alpha=0.35, smoothness_filter_alpha_grid=None, smoothness_filter_n_iters=1, smoothness_filter_max_iters=4, verbose=False, progress_every=25, return_null_matrix=True):
    if shuffle_session not in ('test', 'train'):
        raise ValueError("shuffle_session must be 'test' or 'train'.")
    if transform_scope not in ('global', 'per_neuron'):
        raise ValueError("transform_scope must be 'global' or 'per_neuron'.")
    if shuffle_domain not in ('connected_component', 'depth', 'component_blocks'):
        raise ValueError("shuffle_domain must be 'connected_component', 'depth', or 'component_blocks'.")
    if smoothness_filter_mode not in ('adaptive', 'fixed'):
        raise ValueError("smoothness_filter_mode must be 'adaptive' or 'fixed'.")
    if smoothness_edge_domain not in ('within_component', 'component_boundary', 'all_kept'):
        raise ValueError("smoothness_edge_domain must be 'within_component', 'component_boundary', or 'all_kept'.")
    if float(smoothness_tol) < 0:
        raise ValueError('smoothness_tol must be >= 0.')
    if int(max_attempts_multiplier) < 1:
        raise ValueError('max_attempts_multiplier must be >= 1.')
    if float(smoothness_filter_alpha) < 0 or float(smoothness_filter_alpha) > 1:
        raise ValueError('smoothness_filter_alpha must be in [0, 1].')
    if smoothness_filter_alpha_grid is None:
        smoothness_filter_alpha_grid = np.array([], dtype=float)
    else:
        smoothness_filter_alpha_grid = np.unique(
            np.asarray(smoothness_filter_alpha_grid, dtype=float).ravel()
        )
    if np.any(~np.isfinite(smoothness_filter_alpha_grid)) or np.any(
        (smoothness_filter_alpha_grid < 0) | (smoothness_filter_alpha_grid > 1)
    ):
        raise ValueError('smoothness_filter_alpha_grid values must be finite and in [0, 1].')
    if int(smoothness_filter_n_iters) < 0:
        raise ValueError('smoothness_filter_n_iters must be >= 0.')
    if int(smoothness_filter_max_iters) < 0:
        raise ValueError('smoothness_filter_max_iters must be >= 0.')
    if int(progress_every) < 1:
        raise ValueError('progress_every must be >= 1.')
    if bool(verbose):
        from time import perf_counter as _wlc_perf_counter
        _wlc_t0 = _wlc_perf_counter()

        def _wlc_log(msg):
            elapsed = _wlc_perf_counter() - _wlc_t0
            print(f'[WLC {elapsed:7.1f}s] {msg}')
    else:
        _wlc_log = None
    rng = np.random.RandomState(random_state)
    obs = cross_session_tuning_similarity(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap=cimap, mode='spatial', level_col=level_col, head_col='head_cat', n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, n_neurons=n_neurons, random_state=random_state, train_ids=train_ids, test_ids=test_ids, metric=metric, normalization=normalization, intersect_mask=intersect_mask, eps=eps, min_spikes_per_min=min_spikes_per_min, min_spikes_per_min_train=min_spikes_per_min_train, min_spikes_per_min_test=min_spikes_per_min_test, rate_source=rate_source, drop_if_below=drop_if_below)
    train_ids_used = np.asarray(obs.get('train_ids', []), dtype=int)
    test_ids_used = np.asarray(obs.get('test_ids', []), dtype=int)
    if train_ids_used.size == 0:
        return dict(observed=obs, observed_similarity=np.array([], dtype=float), observed_similarity_resid=np.array([], dtype=float), observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), null_mean_raw=np.array([], dtype=float), null_mean_resid=np.array([], dtype=float), p_mean_raw=np.nan, p_mean_resid=np.nan, p_neuron_raw=np.array([], dtype=float), p_neuron_resid=np.array([], dtype=float), null_raw_matrix=np.zeros((0, 0), dtype=float), null_resid_matrix=np.zeros((0, 0), dtype=float), components=[], component_space=dict(log10_n_states=0.0, n_nontrivial=0), smoothness=dict(observed_energy=np.array([], dtype=float), null_ratio=np.array([], dtype=float), ratio_mean=np.nan, ratio_std=np.nan), dropped=obs.get('dropped', {}), ids=dict(train_ids=train_ids_used, test_ids=test_ids_used), kept_bins_global=np.array([], dtype=int), kept_levels=np.array([], dtype=int), majority_meta=None)
    maps_tr = np.asarray(obs['train_map'], dtype=float)
    maps_te = np.asarray(obs['test_map'], dtype=float)
    mask_tr = np.asarray(obs['train_mask'], dtype=bool)
    mask_te = np.asarray(obs['test_mask'], dtype=bool)
    mask = mask_tr & mask_te if intersect_mask else mask_tr | mask_te
    idx_states = np.where(mask)[0].astype(int)
    if idx_states.size == 0:
        return dict(observed=obs, observed_similarity=np.array([], dtype=float), observed_similarity_resid=np.array([], dtype=float), observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), null_mean_raw=np.array([], dtype=float), null_mean_resid=np.array([], dtype=float), p_mean_raw=np.nan, p_mean_resid=np.nan, p_neuron_raw=np.array([], dtype=float), p_neuron_resid=np.array([], dtype=float), null_raw_matrix=np.zeros((0, train_ids_used.size), dtype=float), null_resid_matrix=np.zeros((0, train_ids_used.size), dtype=float), components=[], component_space=dict(log10_n_states=0.0, n_nontrivial=0), smoothness=dict(observed_energy=np.array([], dtype=float), null_ratio=np.array([], dtype=float), ratio_mean=np.nan, ratio_std=np.nan), dropped=obs.get('dropped', {}), ids=dict(train_ids=train_ids_used, test_ids=test_ids_used), kept_bins_global=np.array([], dtype=int), kept_levels=np.array([], dtype=int), majority_meta=None)
    (bin_level_global, majority_meta) = _wlc_build_majority_bin_levels(positions_train, selected_df_train, positions_test, selected_df_test, level_col=level_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, majority_min_frac=majority_min_frac)
    if drop_unknown_level_bins:
        keep_lvl = bin_level_global[idx_states] >= 0
        idx_states = idx_states[keep_lvl]
    if idx_states.size == 0:
        return dict(observed=obs, observed_similarity=np.array([], dtype=float), observed_similarity_resid=np.array([], dtype=float), observed_summary=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), observed_summary_resid=dict(mean=np.nan, std=np.nan, median=np.nan, n=0), null_mean_raw=np.array([], dtype=float), null_mean_resid=np.array([], dtype=float), p_mean_raw=np.nan, p_mean_resid=np.nan, p_neuron_raw=np.array([], dtype=float), p_neuron_resid=np.array([], dtype=float), null_raw_matrix=np.zeros((0, train_ids_used.size), dtype=float), null_resid_matrix=np.zeros((0, train_ids_used.size), dtype=float), components=[], component_space=dict(log10_n_states=0.0, n_nontrivial=0), smoothness=dict(observed_energy=np.array([], dtype=float), null_ratio=np.array([], dtype=float), ratio_mean=np.nan, ratio_std=np.nan), dropped=obs.get('dropped', {}), ids=dict(train_ids=train_ids_used, test_ids=test_ids_used), kept_bins_global=np.array([], dtype=int), kept_levels=np.array([], dtype=int), majority_meta=majority_meta)
    kept_bins_global = idx_states.astype(int)
    kept_levels = bin_level_global[kept_bins_global].astype(int)
    A_base = maps_tr[kept_bins_global, :]
    B_base = maps_te[kept_bins_global, :]
    (Kk, M) = A_base.shape
    shuffle_bins_global = kept_bins_global[kept_levels >= 0]
    comps_global = _wlc_build_components(bin_level_global, shuffle_bins_global, n_bins_x, n_bins_y)
    global_to_local = {g: i for (i, g) in enumerate(kept_bins_global.tolist())}
    component_orders_local = []
    components = []
    component_ids_local = np.full(Kk, -1, dtype=int)
    log10_n_states = 0.0
    n_nontrivial = 0
    for component_id, c in enumerate(comps_global):
        ord_global = _wlc_order_component_bins(c['bins_global'], n_bins_x, n_bins_y)
        ord_local = np.array([global_to_local[g] for g in ord_global], dtype=int)
        component_orders_local.append(ord_local)
        component_ids_local[ord_local] = int(component_id)
        sz = int(ord_local.size)
        component_n_states = 1 if sz <= 1 else 2 if sz == 2 else 2 * sz
        topology_signature = _wlc_order_adjacency_signature(ord_global, n_bins_x, n_bins_y)
        components.append(dict(level=int(c['level']), size=sz, bins_global=c['bins_global'], order_global=ord_global, topology_signature=topology_signature, n_states=int(component_n_states)))
    component_block_families = _wlc_build_component_block_families(components, component_orders_local)
    if shuffle_domain == 'connected_component':
        shuffle_groups_local = component_orders_local
    elif shuffle_domain == 'depth':
        shuffle_groups_local = _wlc_build_depth_groups(kept_levels)
    else:
        shuffle_groups_local = [
            np.concatenate([component_orders_local[int(component_id)] for component_id in family['component_ids']])
            for family in component_block_families
        ]
    shuffle_groups = []
    log10_n_states = 0.0
    n_nontrivial = 0
    for group_local in shuffle_groups_local:
        sz = int(group_local.size)
        level = int(kept_levels[group_local[0]]) if sz else -1
        component_ids = np.unique(component_ids_local[group_local]) if sz else np.array([], dtype=int)
        if shuffle_domain == 'connected_component':
            n_states = 1 if sz <= 1 else 2 if sz == 2 else 2 * sz
            log10_states = np.log10(float(n_states)) if n_states > 0 else 0.0
            component_size = sz
        elif shuffle_domain == 'depth':
            n_states = None
            log10_states = float(np.sum(np.log10(np.arange(2, sz + 1, dtype=float)))) if sz > 1 else 0.0
            component_size = None
        else:
            family = component_block_families[len(shuffle_groups)]
            n_components = int(len(family['component_ids']))
            n_states = None
            log10_states = float(np.sum(np.log10(np.arange(2, n_components + 1, dtype=float)))) if n_components > 1 else 0.0
            component_size = int(family['component_size'])
        if (shuffle_domain == 'component_blocks' and component_ids.size > 1) or (shuffle_domain != 'component_blocks' and sz > 1):
            n_nontrivial += 1
        log10_n_states += log10_states
        shuffle_groups.append(
            dict(
                level=level,
                size=sz,
                component_size=component_size,
                topology_signature=(
                    component_block_families[len(shuffle_groups)]['topology_signature']
                    if shuffle_domain == 'component_blocks'
                    else None
                ),
                n_components=int(component_ids.size),
                component_ids=component_ids,
                bins_global=kept_bins_global[group_local],
                n_states=n_states,
                log10_n_states=float(log10_states),
            )
        )
    (obs_raw, obs_valid) = _wlc_similarity_per_neuron(A_base, B_base, metric=metric, normalization=normalization, eps=eps)
    A_res_base = _wlc_residualize_by_level(A_base, kept_levels)
    B_res_base = _wlc_residualize_by_level(B_base, kept_levels)
    (obs_resid, obs_resid_valid) = _wlc_similarity_per_neuron(A_res_base, B_res_base, metric=metric, normalization=normalization, eps=eps)
    obs_summary = dict(mean=float(np.nanmean(obs_raw)), std=float(np.nanstd(obs_raw)), median=float(np.nanmedian(obs_raw)), n=int(M))
    obs_summary_resid = dict(mean=float(np.nanmean(obs_resid)), std=float(np.nanstd(obs_resid)), median=float(np.nanmedian(obs_resid)), n=int(M))
    edges_within_component = _wlc_edges_local_from_components(comps_global, global_to_local, n_bins_x, n_bins_y)
    edges_all_kept = _wlc_edges_local_all_kept(kept_bins_global, global_to_local, n_bins_x, n_bins_y)
    if edges_all_kept.size == 0:
        edges_component_boundary = np.zeros((0, 2), dtype=int)
    else:
        is_boundary = component_ids_local[edges_all_kept[:, 0]] != component_ids_local[edges_all_kept[:, 1]]
        edges_component_boundary = edges_all_kept[is_boundary]
    edges_local = {
        'within_component': edges_within_component,
        'component_boundary': edges_component_boundary,
        'all_kept': edges_all_kept,
    }[smoothness_edge_domain]
    target_obs = B_base if shuffle_session == 'test' else A_base
    obs_energy = _wlc_laplacian_energy_cols(target_obs, edges_local)
    target_repeats = int(n_repeats)
    max_attempts = target_repeats * int(max_attempts_multiplier)
    if _wlc_log is not None:
        _wlc_log(f'start: target_repeats={target_repeats}, max_attempts={max_attempts}, scope={transform_scope}, shuffle_session={shuffle_session}, shuffle_domain={shuffle_domain}, smoothness_edge_domain={smoothness_edge_domain}, smooth_match={bool(smoothness_match)}, smooth_filter={bool(smoothness_filter)}')
    null_raw_rows = []
    null_resid_rows = []
    null_mean_raw_list = []
    null_mean_resid_list = []
    null_ratio_list = []
    null_ratio_pre_filter_list = []
    null_filter_iters_list = []
    null_cross_component_fraction_list = []
    null_depth_preserved_list = []
    null_within_component_values_preserved_list = []
    null_block_assignment_values_preserved_list = []
    alpha_grid_raw_lists = [[] for _ in smoothness_filter_alpha_grid]
    alpha_grid_resid_lists = [[] for _ in smoothness_filter_alpha_grid]
    alpha_grid_ratio_lists = [[] for _ in smoothness_filter_alpha_grid]
    n_attempts = 0
    while len(null_mean_raw_list) < target_repeats and n_attempts < max_attempts:
        n_attempts += 1
        attempt_cross_component_fractions = []
        attempt_depth_preserved = True
        attempt_within_component_values_preserved = shuffle_domain == 'component_blocks'
        attempt_block_assignment_values_preserved = shuffle_domain == 'component_blocks'
        if transform_scope == 'global':
            if shuffle_domain == 'component_blocks':
                perm = _wlc_sample_component_block_mapping(
                    Kk,
                    component_orders_local,
                    component_block_families,
                    rng=rng,
                )
            else:
                perm = np.arange(Kk, dtype=int)
                for group_local in shuffle_groups_local:
                    if shuffle_domain == 'connected_component':
                        mapped = _wlc_sample_component_mapping(group_local, rng=rng, reverse_prob=reverse_prob)
                    else:
                        mapped = _wlc_sample_depth_mapping(group_local, rng=rng)
                    perm[group_local] = mapped
            attempt_depth_preserved = bool(np.array_equal(kept_levels[perm], kept_levels))
            attempt_cross_component_fractions.append(float(np.mean(component_ids_local[perm] != component_ids_local)))
            if shuffle_session == 'test':
                A_use = A_base
                B_use = B_base[perm, :]
            else:
                A_use = A_base[perm, :]
                B_use = B_base
        elif shuffle_session == 'test':
            A_use = A_base
            B_use = B_base.copy()
            for i in range(M):
                if shuffle_domain == 'component_blocks':
                    perm_i = _wlc_sample_component_block_mapping(
                        Kk,
                        component_orders_local,
                        component_block_families,
                        rng=rng,
                    )
                else:
                    perm_i = np.arange(Kk, dtype=int)
                    for group_local in shuffle_groups_local:
                        if shuffle_domain == 'connected_component':
                            mapped = _wlc_sample_component_mapping(group_local, rng=rng, reverse_prob=reverse_prob)
                        else:
                            mapped = _wlc_sample_depth_mapping(group_local, rng=rng)
                        perm_i[group_local] = mapped
                attempt_depth_preserved = attempt_depth_preserved and bool(np.array_equal(kept_levels[perm_i], kept_levels))
                attempt_cross_component_fractions.append(float(np.mean(component_ids_local[perm_i] != component_ids_local)))
                B_use[:, i] = B_base[perm_i, i]
        else:
            A_use = A_base.copy()
            for i in range(M):
                if shuffle_domain == 'component_blocks':
                    perm_i = _wlc_sample_component_block_mapping(
                        Kk,
                        component_orders_local,
                        component_block_families,
                        rng=rng,
                    )
                else:
                    perm_i = np.arange(Kk, dtype=int)
                    for group_local in shuffle_groups_local:
                        if shuffle_domain == 'connected_component':
                            mapped = _wlc_sample_component_mapping(group_local, rng=rng, reverse_prob=reverse_prob)
                        else:
                            mapped = _wlc_sample_depth_mapping(group_local, rng=rng)
                        perm_i[group_local] = mapped
                attempt_depth_preserved = attempt_depth_preserved and bool(np.array_equal(kept_levels[perm_i], kept_levels))
                attempt_cross_component_fractions.append(float(np.mean(component_ids_local[perm_i] != component_ids_local)))
                A_use[:, i] = A_base[perm_i, i]
            B_use = B_base
        target_pre_filter = B_use if shuffle_session == 'test' else A_use
        pre_filter_energy = _wlc_laplacian_energy_cols(target_pre_filter, edges_local)
        with np.errstate(invalid='ignore', divide='ignore'):
            pre_filter_ratio = pre_filter_energy / obs_energy
        pre_filter_ratio_med = float(np.nanmedian(pre_filter_ratio))
        alpha_grid_pending = []
        for alpha_value in smoothness_filter_alpha_grid:
            alpha_grid_pending.append(
                _wlc_evaluate_fixed_alpha(
                    A_use,
                    B_use,
                    A_res_base=A_res_base,
                    B_res_base=B_res_base,
                    kept_levels=kept_levels,
                    edges=edges_local,
                    obs_energy=obs_energy,
                    shuffle_session=shuffle_session,
                    alpha=float(alpha_value),
                    n_iters=int(smoothness_filter_n_iters),
                    metric=metric,
                    normalization=normalization,
                    eps=eps,
                )
            )
        n_filter_iters = 0
        if smoothness_filter:
            if shuffle_domain == 'component_blocks':
                attempt_within_component_values_preserved = False
            if shuffle_session == 'test':
                if smoothness_filter_mode == 'adaptive':
                    (B_use, _, n_filter_iters) = _wlc_smooth_to_target_ratio(B_use, edges_local, obs_energy, target=smoothness_target, max_iters=int(smoothness_filter_max_iters), alpha=float(smoothness_filter_alpha))
                else:
                    n_filter_iters = int(smoothness_filter_n_iters)
                    B_use = _wlc_component_graph_smooth_cols(B_use, edges_local, alpha=float(smoothness_filter_alpha), n_iters=n_filter_iters)
            elif smoothness_filter_mode == 'adaptive':
                (A_use, _, n_filter_iters) = _wlc_smooth_to_target_ratio(A_use, edges_local, obs_energy, target=smoothness_target, max_iters=int(smoothness_filter_max_iters), alpha=float(smoothness_filter_alpha))
            else:
                n_filter_iters = int(smoothness_filter_n_iters)
                A_use = _wlc_component_graph_smooth_cols(A_use, edges_local, alpha=float(smoothness_filter_alpha), n_iters=n_filter_iters)
        if shuffle_session == 'test':
            A_res_use = A_res_base
            B_res_use = _wlc_residualize_by_level(B_use, kept_levels)
            shuf_energy = _wlc_laplacian_energy_cols(B_use, edges_local)
        else:
            A_res_use = _wlc_residualize_by_level(A_use, kept_levels)
            B_res_use = B_res_base
            shuf_energy = _wlc_laplacian_energy_cols(A_use, edges_local)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = shuf_energy / obs_energy
        ratio_med = float(np.nanmedian(ratio))
        if smoothness_match:
            if not np.isfinite(ratio_med) or abs(ratio_med - float(smoothness_target)) > float(smoothness_tol):
                if _wlc_log is not None and n_attempts % int(progress_every) == 0:
                    _wlc_log(f'attempt={n_attempts}, accepted={len(null_mean_raw_list)} (rejected by smoothness_match)')
                continue
        (s_raw, _) = _wlc_similarity_per_neuron(A_use, B_use, metric=metric, normalization=normalization, eps=eps)
        (s_res, _) = _wlc_similarity_per_neuron(A_res_use, B_res_use, metric=metric, normalization=normalization, eps=eps)
        null_raw_rows.append(s_raw)
        null_resid_rows.append(s_res)
        null_mean_raw_list.append(float(np.nanmean(s_raw)))
        null_mean_resid_list.append(float(np.nanmean(s_res)))
        null_ratio_list.append(ratio_med)
        null_ratio_pre_filter_list.append(pre_filter_ratio_med)
        null_filter_iters_list.append(int(n_filter_iters))
        null_cross_component_fraction_list.append(float(np.mean(attempt_cross_component_fractions)))
        null_depth_preserved_list.append(bool(attempt_depth_preserved))
        null_within_component_values_preserved_list.append(bool(attempt_within_component_values_preserved))
        null_block_assignment_values_preserved_list.append(bool(attempt_block_assignment_values_preserved))
        for alpha_index, alpha_result in enumerate(alpha_grid_pending):
            alpha_grid_raw_lists[alpha_index].append(alpha_result['null_raw_mean'])
            alpha_grid_resid_lists[alpha_index].append(alpha_result['null_resid_mean'])
            alpha_grid_ratio_lists[alpha_index].append(alpha_result['smooth_ratio'])
        if _wlc_log is not None and (len(null_mean_raw_list) == 1 or len(null_mean_raw_list) % int(progress_every) == 0 or len(null_mean_raw_list) == target_repeats):
            _wlc_log(f'attempt={n_attempts}, accepted={len(null_mean_raw_list)}/{target_repeats}, median_ratio={ratio_med:.3f}')
    if len(null_mean_raw_list) == 0:
        null_raw = np.zeros((0, M), dtype=float)
        null_resid = np.zeros((0, M), dtype=float)
        null_mean_raw = np.array([], dtype=float)
        null_mean_resid = np.array([], dtype=float)
        null_ratio = np.array([], dtype=float)
        null_ratio_pre_filter = np.array([], dtype=float)
        null_filter_iters = np.array([], dtype=int)
        null_cross_component_fraction = np.array([], dtype=float)
        null_depth_preserved = np.array([], dtype=bool)
        null_within_component_values_preserved = np.array([], dtype=bool)
        null_block_assignment_values_preserved = np.array([], dtype=bool)
    else:
        null_raw = np.vstack(null_raw_rows)
        null_resid = np.vstack(null_resid_rows)
        null_mean_raw = np.asarray(null_mean_raw_list, dtype=float)
        null_mean_resid = np.asarray(null_mean_resid_list, dtype=float)
        null_ratio = np.asarray(null_ratio_list, dtype=float)
        null_ratio_pre_filter = np.asarray(null_ratio_pre_filter_list, dtype=float)
        null_filter_iters = np.asarray(null_filter_iters_list, dtype=int)
        null_cross_component_fraction = np.asarray(null_cross_component_fraction_list, dtype=float)
        null_depth_preserved = np.asarray(null_depth_preserved_list, dtype=bool)
        null_within_component_values_preserved = np.asarray(null_within_component_values_preserved_list, dtype=bool)
        null_block_assignment_values_preserved = np.asarray(null_block_assignment_values_preserved_list, dtype=bool)
    p_neuron_raw = _wlc_empirical_p_ge(obs_raw, null_raw)
    p_neuron_resid = _wlc_empirical_p_ge(obs_resid, null_resid)
    p_mean_raw = _wlc_empirical_p_ge(np.array([obs_summary['mean']], dtype=float), null_mean_raw)[0]
    p_mean_resid = _wlc_empirical_p_ge(np.array([obs_summary_resid['mean']], dtype=float), null_mean_resid)[0]
    out = dict(observed=obs, observed_similarity=obs_raw, observed_similarity_resid=obs_resid, observed_valid_counts=obs_valid, observed_valid_counts_resid=obs_resid_valid, observed_summary=obs_summary, observed_summary_resid=obs_resid, null_mean_raw=null_mean_raw, null_mean_resid=null_mean_resid, p_mean_raw=float(p_mean_raw), p_mean_resid=float(p_mean_resid), p_neuron_raw=p_neuron_raw, p_neuron_resid=p_neuron_resid, components=components, component_space=dict(log10_n_states=float(log10_n_states), n_nontrivial=int(n_nontrivial)), transformation=dict(shuffle_domain=str(shuffle_domain), groups=shuffle_groups, cross_component_fraction=null_cross_component_fraction, cross_component_fraction_mean=float(np.nanmean(null_cross_component_fraction)) if null_cross_component_fraction.size > 0 else np.nan, cross_component_fraction_std=float(np.nanstd(null_cross_component_fraction)) if null_cross_component_fraction.size > 0 else np.nan, cross_component_fraction_min=float(np.nanmin(null_cross_component_fraction)) if null_cross_component_fraction.size > 0 else np.nan, cross_component_fraction_max=float(np.nanmax(null_cross_component_fraction)) if null_cross_component_fraction.size > 0 else np.nan, depth_preserved=null_depth_preserved, depth_preserved_all=bool(np.all(null_depth_preserved)) if null_depth_preserved.size > 0 else False, block_assignment_values_preserved=null_block_assignment_values_preserved, block_assignment_values_preserved_all=bool(np.all(null_block_assignment_values_preserved)) if null_block_assignment_values_preserved.size > 0 else False, within_component_values_preserved=null_within_component_values_preserved, within_component_values_preserved_all=bool(np.all(null_within_component_values_preserved)) if null_within_component_values_preserved.size > 0 else False), smoothness=dict(observed_energy=obs_energy, null_ratio=null_ratio, ratio_mean=float(np.nanmean(null_ratio)), ratio_std=float(np.nanstd(null_ratio)), match_enabled=bool(smoothness_match), target=float(smoothness_target), tol=float(smoothness_tol), filter_enabled=bool(smoothness_filter), filter_mode=str(smoothness_filter_mode), filter_alpha=float(smoothness_filter_alpha), filter_n_iters_fixed=int(smoothness_filter_n_iters), filter_max_iters=int(smoothness_filter_max_iters), filter_iters=null_filter_iters, filter_iters_mean=float(np.nanmean(null_filter_iters)) if null_filter_iters.size > 0 else np.nan, filter_iters_std=float(np.nanstd(null_filter_iters)) if null_filter_iters.size > 0 else np.nan, n_attempts=int(n_attempts), n_accepted=int(null_ratio.size), acceptance_rate=float(null_ratio.size) / float(n_attempts) if n_attempts > 0 else np.nan), dropped=obs.get('dropped', {}), ids=dict(train_ids=train_ids_used, test_ids=test_ids_used), kept_bins_global=kept_bins_global, kept_levels=kept_levels, majority_meta=majority_meta, shuffle_session=shuffle_session, transform_scope=transform_scope, shuffle_domain=shuffle_domain, n_repeats=int(n_repeats), metric=metric, normalization=normalization)
    out['observed_summary_resid'] = obs_summary_resid
    out['smoothness']['edge_domain'] = str(smoothness_edge_domain)
    out['smoothness']['null_ratio_pre_filter'] = null_ratio_pre_filter
    out['smoothness']['ratio_pre_filter_mean'] = float(np.nanmean(null_ratio_pre_filter)) if null_ratio_pre_filter.size > 0 else np.nan
    out['smoothness']['ratio_pre_filter_std'] = float(np.nanstd(null_ratio_pre_filter)) if null_ratio_pre_filter.size > 0 else np.nan
    out['smoothness']['n_edges'] = int(edges_local.shape[0])
    out['smoothness']['n_edges_within_component'] = int(edges_within_component.shape[0])
    out['smoothness']['n_edges_component_boundary'] = int(edges_component_boundary.shape[0])
    out['smoothness']['n_edges_all_kept'] = int(edges_all_kept.shape[0])
    out['smoothness_alpha_sensitivity'] = dict(
        alphas=smoothness_filter_alpha_grid.copy(),
        n_iters=int(smoothness_filter_n_iters),
        null_mean_raw=np.asarray(alpha_grid_raw_lists, dtype=float),
        null_mean_resid=np.asarray(alpha_grid_resid_lists, dtype=float),
        smooth_ratio=np.asarray(alpha_grid_ratio_lists, dtype=float),
        smooth_ratio_pre_filter=null_ratio_pre_filter.copy(),
    )
    if return_null_matrix:
        out['null_raw_matrix'] = null_raw
        out['null_resid_matrix'] = null_resid
    if _wlc_log is not None:
        _wlc_log(f"done: accepted={null_ratio.size}/{target_repeats}, attempts={n_attempts}, acceptance_rate={out['smoothness']['acceptance_rate']:.3f}")
    return out

def _norm_row(v, mode, eps=1e-12):
    if mode == 'none':
        return v
    elif mode == 'zscore':
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd <= eps:
            return np.full_like(v, np.nan, dtype=float)
        return (v - mu) / sd
    elif mode == 'l2':
        return v / (np.linalg.norm(v) + eps)
    else:
        return v

def _pv_from_maps(maps_tr, maps_te, mask_tr, mask_te, metric='pearson', normalization='auto', intersect_mask=True, min_neurons=3, eps=1e-12):
    assert maps_tr.shape == maps_te.shape
    (K, M) = maps_tr.shape
    if intersect_mask:
        mask = mask_tr.astype(bool) & mask_te.astype(bool)
    else:
        mask = mask_tr.astype(bool) | mask_te.astype(bool)
    if normalization == 'auto':
        row_norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        row_norm_mode = normalization
    pv = np.full(K, np.nan, float)
    for b in range(K):
        if not mask[b]:
            continue
        a_raw = maps_tr[b, :].astype(float, copy=False)
        c_raw = maps_te[b, :].astype(float, copy=False)
        good = np.isfinite(a_raw) & np.isfinite(c_raw)
        if good.sum() < min_neurons:
            continue
        a0 = a_raw[good]
        c0 = c_raw[good]
        if metric == 'pearson' and (np.nanstd(a0) <= eps or np.nanstd(c0) <= eps):
            continue
        a = _norm_row(a0, row_norm_mode, eps=eps)
        c = _norm_row(c0, row_norm_mode, eps=eps)
        if metric == 'pearson':
            aa = a - np.mean(a)
            cc = c - np.mean(c)
            denom = np.linalg.norm(aa) * np.linalg.norm(cc)
            if denom <= eps:
                continue
            pv[b] = float(np.dot(aa, cc) / denom)
        elif metric == 'spearman':
            pv[b] = spearmanr(a, c, nan_policy='omit')[0]
        else:
            pv[b] = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + eps))
    return (pv, mask)

def _prefit_train_and_test_maps(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, train_ids, test_ids, mode='spatial', level_col='level', head_col='head_cat', n_bins_x=15, n_bins_y=15, dt=0.04, time_window=1, use_peak=False, filter_ratemap=False, sigma=1.0, eps=1e-12, spatial_labels_train=None, spatial_labels_test=None, level_categories=None):
    (maps_tr, mask_tr, meta_tr) = _fit_maps_for_session(spike_counts_train, raw_counts_train, selected_df_train, positions_train, train_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_train, level_categories=level_categories)
    (maps_te, mask_te, meta_te) = _fit_maps_for_session(spike_counts_test, raw_counts_test, selected_df_test, positions_test, test_ids, mode=mode, level_col=level_col, head_col=head_col, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, eps=eps, spatial_labels=spatial_labels_test, level_categories=level_categories)
    return (maps_tr, mask_tr, meta_tr, maps_te, mask_te, meta_te)

def _impute_neuronwise(X, good_mask):
    has_any = np.any(good_mask, axis=0)
    col_means = np.zeros(X.shape[1], dtype=float)
    if np.any(has_any):
        col_means[has_any] = np.nanmean(np.where(good_mask[:, has_any], X[:, has_any], np.nan), axis=0)
    Xf = X.copy()
    bad = ~np.isfinite(Xf)
    if np.any(bad):
        Xf[bad] = col_means[np.newaxis, :][bad]
    Xf[~np.isfinite(Xf)] = 0.0
    return Xf

def _spikes_per_min(C, dt_sec):
    C = np.asarray(C, float)
    total_events = C.sum(axis=0)
    total_minutes = C.shape[0] * float(dt_sec) / 60.0
    if total_minutes <= 0:
        return np.zeros(C.shape[1], float)
    return total_events / total_minutes

def _skaggs_info(lambdas_spatial: np.ndarray, prior_spatial: np.ndarray, eps: float=1e-12) -> np.ndarray:
    L = np.asarray(lambdas_spatial, float) + eps
    p = np.asarray(prior_spatial, float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)
    p_sum = float(p.sum())
    if p_sum <= 0.0:
        return np.zeros(L.shape[1], dtype=float)
    p = p / p_sum
    # Compute mean rate without additive offset; then floor for numerical safety.
    r_bar = (p[:, None] * L).sum(axis=0)
    r_bar = np.maximum(r_bar, eps)
    ratio = np.maximum(L / r_bar[None, :], eps)
    I = (p[:, None] * ratio * np.log2(ratio)).sum(axis=0)
    I[~np.isfinite(I)] = 0.0
    I[I < 0.0] = 0.0
    return I

def _wlc_build_components(bin_level_global, kept_bins_global, n_bins_x, n_bins_y):
    kept_bins_global = np.asarray(kept_bins_global, dtype=int)
    kept_set = set(kept_bins_global.tolist())
    visited = set()
    comps = []
    for b0 in kept_bins_global.tolist():
        if b0 in visited:
            continue
        lvl0 = int(bin_level_global[b0])
        stack = [b0]
        visited.add(b0)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in _wlc_neighbors_4(u, n_bins_x, n_bins_y):
                if v in kept_set and v not in visited and (int(bin_level_global[v]) == lvl0):
                    visited.add(v)
                    stack.append(v)
        comp = np.asarray(sorted(comp), dtype=int)
        comps.append(dict(level=lvl0, bins_global=comp))
    return comps

def _wlc_build_majority_bin_levels(positions_train, selected_df_train, positions_test, selected_df_test, level_col='level', n_bins_x=15, n_bins_y=15, majority_min_frac=0.5):
    (lab_tr, _, _) = discretize_positions(positions_train, n_bins_x, n_bins_y)
    (lab_te, _, _) = discretize_positions(positions_test, n_bins_x, n_bins_y)
    lv_tr_raw = np.asarray(selected_df_train[level_col].values)
    lv_te_raw = np.asarray(selected_df_test[level_col].values)
    if lv_tr_raw.shape[0] != lab_tr.shape[0]:
        raise ValueError('selected_df_train level_col length must match positions_train length.')
    if lv_te_raw.shape[0] != lab_te.shape[0]:
        raise ValueError('selected_df_test level_col length must match positions_test length.')
    valid_tr = ~pd.isna(lv_tr_raw)
    valid_te = ~pd.isna(lv_te_raw)
    uniq_vals = np.unique(np.concatenate([lv_tr_raw[valid_tr], lv_te_raw[valid_te]]))
    K = int(n_bins_x * n_bins_y)
    if uniq_vals.size == 0:
        return (np.full(K, -1, dtype=int), dict(uniq_level_values=np.array([], dtype=float), votes=np.zeros((K, 0), dtype=int), majority_frac=np.zeros(K, dtype=float), visited_counts=np.zeros(K, dtype=int), majority_min_frac=float(majority_min_frac)))
    level_to_idx = {v: i for (i, v) in enumerate(uniq_vals.tolist())}
    lv_tr = np.array([level_to_idx[v] for v in lv_tr_raw[valid_tr]], dtype=int)
    lv_te = np.array([level_to_idx[v] for v in lv_te_raw[valid_te]], dtype=int)
    votes = np.zeros((K, uniq_vals.size), dtype=int)
    np.add.at(votes, (lab_tr[valid_tr], lv_tr), 1)
    np.add.at(votes, (lab_te[valid_te], lv_te), 1)
    visited_counts = votes.sum(axis=1)
    best = np.argmax(votes, axis=1)
    best_count = votes[np.arange(K), best]
    majority_frac = np.zeros(K, dtype=float)
    mask_vis = visited_counts > 0
    majority_frac[mask_vis] = best_count[mask_vis] / visited_counts[mask_vis]
    bin_level = np.full(K, -1, dtype=int)
    keep = mask_vis & (majority_frac >= float(majority_min_frac))
    bin_level[keep] = best[keep]
    return (bin_level, dict(uniq_level_values=uniq_vals, votes=votes, majority_frac=majority_frac, visited_counts=visited_counts, majority_min_frac=float(majority_min_frac)))

def _wlc_component_graph_smooth_cols(X, edges, alpha=0.35, n_iters=1):
    X = np.asarray(X, dtype=float)
    edges = np.asarray(edges, dtype=int)
    alpha = float(alpha)
    n_iters = int(n_iters)
    if edges.size == 0 or n_iters <= 0 or alpha <= 0.0:
        return X.copy()
    alpha = max(0.0, min(1.0, alpha))
    (K, M) = X.shape
    u = edges[:, 0]
    v = edges[:, 1]
    deg = np.zeros(K, dtype=float)
    np.add.at(deg, u, 1.0)
    np.add.at(deg, v, 1.0)
    has_nb = deg > 0
    out = X.copy()
    for _ in range(n_iters):
        nb_sum = np.zeros((K, M), dtype=float)
        np.add.at(nb_sum, u, out[v, :])
        np.add.at(nb_sum, v, out[u, :])
        nb_avg = out.copy()
        nb_avg[has_nb, :] = nb_sum[has_nb, :] / deg[has_nb, None]
        out = (1.0 - alpha) * out + alpha * nb_avg
    return out

def _wlc_evaluate_fixed_alpha(A, B, *, A_res_base, B_res_base, kept_levels, edges, obs_energy, shuffle_session, alpha, n_iters, metric='pearson', normalization='auto', eps=1e-12):
    """Evaluate one fixed smoothing strength without changing the shuffled draw."""
    A_use = np.asarray(A, dtype=float)
    B_use = np.asarray(B, dtype=float)
    if shuffle_session == 'test':
        B_use = _wlc_component_graph_smooth_cols(
            B_use, edges, alpha=float(alpha), n_iters=int(n_iters)
        )
        A_res_use = A_res_base
        B_res_use = _wlc_residualize_by_level(B_use, kept_levels)
        shuffled_map = B_use
    elif shuffle_session == 'train':
        A_use = _wlc_component_graph_smooth_cols(
            A_use, edges, alpha=float(alpha), n_iters=int(n_iters)
        )
        A_res_use = _wlc_residualize_by_level(A_use, kept_levels)
        B_res_use = B_res_base
        shuffled_map = A_use
    else:
        raise ValueError("shuffle_session must be 'test' or 'train'.")
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = _wlc_laplacian_energy_cols(shuffled_map, edges) / obs_energy
    (s_raw, _) = _wlc_similarity_per_neuron(
        A_use, B_use, metric=metric, normalization=normalization, eps=eps
    )
    (s_res, _) = _wlc_similarity_per_neuron(
        A_res_use, B_res_use, metric=metric, normalization=normalization, eps=eps
    )
    return dict(
        null_raw_mean=float(np.nanmean(s_raw)),
        null_resid_mean=float(np.nanmean(s_res)),
        smooth_ratio=float(np.nanmedian(ratio)),
    )

def _wlc_edges_local_from_components(components, global_to_local, n_bins_x, n_bins_y):
    edges = []
    for comp in components:
        bins = comp['bins_global']
        bset = set(bins.tolist())
        for u in bins.tolist():
            for v in _wlc_neighbors_4(u, n_bins_x, n_bins_y):
                if v in bset and u < v:
                    edges.append((global_to_local[u], global_to_local[v]))
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(edges, dtype=int)

def _wlc_edges_local_all_kept(kept_bins_global, global_to_local, n_bins_x, n_bins_y):
    kept_bins_global = np.asarray(kept_bins_global, dtype=int)
    kept_set = set(kept_bins_global.tolist())
    edges = []
    for u in kept_bins_global.tolist():
        for v in _wlc_neighbors_4(u, n_bins_x, n_bins_y):
            if v in kept_set and u < v:
                edges.append((global_to_local[u], global_to_local[v]))
    if not edges:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(edges, dtype=int)

def _wlc_empirical_p_ge(obs, null_vals_2d):
    obs = np.asarray(obs, dtype=float)
    null_vals_2d = np.asarray(null_vals_2d, dtype=float)
    if null_vals_2d.ndim == 1:
        null_vals_2d = null_vals_2d[:, None]
    M = obs.shape[0]
    p = np.full(M, np.nan, dtype=float)
    for i in range(M):
        oi = obs[i]
        col = null_vals_2d[:, i]
        good = np.isfinite(col)
        if not np.isfinite(oi) or good.sum() == 0:
            continue
        c = int(np.sum(col[good] >= oi))
        p[i] = (c + 1.0) / (good.sum() + 1.0)
    return p

def _wlc_laplacian_energy_cols(X, edges):
    X = np.asarray(X, dtype=float)
    edges = np.asarray(edges, dtype=int)
    M = X.shape[1]
    if edges.size == 0:
        return np.full(M, np.nan, dtype=float)
    u = edges[:, 0]
    v = edges[:, 1]
    D = X[u, :] - X[v, :]
    with np.errstate(invalid='ignore'):
        E = np.nanmean(D * D, axis=0)
    return E

def _wlc_neighbors_4(bin_idx, n_bins_x, n_bins_y):
    x = int(bin_idx % n_bins_x)
    y = int(bin_idx // n_bins_x)
    out = []
    if x > 0:
        out.append(bin_idx - 1)
    if x < n_bins_x - 1:
        out.append(bin_idx + 1)
    if y > 0:
        out.append(bin_idx - n_bins_x)
    if y < n_bins_y - 1:
        out.append(bin_idx + n_bins_x)
    return out

def _wlc_order_component_bins(component_bins_global, n_bins_x, n_bins_y):
    bins = np.asarray(component_bins_global, dtype=int)
    if bins.size <= 1:
        return bins.copy()
    bin_set = set(bins.tolist())
    neigh = {b: [] for b in bins.tolist()}
    for b in bins.tolist():
        for nb in _wlc_neighbors_4(b, n_bins_x, n_bins_y):
            if nb in bin_set:
                neigh[b].append(nb)
        neigh[b].sort()
    deg = {b: len(neigh[b]) for b in bins.tolist()}
    n = bins.size
    deg1 = [b for b in bins.tolist() if deg[b] == 1]
    maxdeg = max(deg.values()) if n > 0 else 0
    if maxdeg <= 2 and len(deg1) == 2:
        start = min(deg1)
        order = [start]
        prev = None
        cur = start
        for _ in range(n - 1):
            nxts = [x for x in neigh[cur] if x != prev]
            if not nxts:
                break
            nxt = min(nxts)
            order.append(nxt)
            (prev, cur) = (cur, nxt)
        if len(order) == n:
            return np.asarray(order, dtype=int)
    if maxdeg == 2 and all((d == 2 for d in deg.values())):
        start = int(np.min(bins))
        nbs = neigh[start]
        if len(nbs) >= 1:
            order = [start]
            prev = None
            cur = start
            for _ in range(n - 1):
                nxts = [x for x in neigh[cur] if x != prev]
                if not nxts:
                    break
                nxt = min(nxts) if prev is None else nxts[0]
                if nxt in order:
                    break
                order.append(nxt)
                (prev, cur) = (cur, nxt)
            if len(order) == n:
                return np.asarray(order, dtype=int)
    seed = int(np.min(bins))
    q = deque([seed])
    seen = {seed}
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in neigh[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return np.asarray(order, dtype=int)

def _wlc_residualize_by_level(X, level_idx_rows):
    Xr = np.asarray(X, dtype=float).copy()
    lv = np.asarray(level_idx_rows, dtype=int)
    for l in np.unique(lv):
        if l < 0:
            continue
        ridx = np.where(lv == l)[0]
        if ridx.size == 0:
            continue
        with np.errstate(invalid='ignore'):
            mu = np.nanmean(Xr[ridx, :], axis=0, keepdims=True)
        Xr[ridx, :] = Xr[ridx, :] - mu
    return Xr

def _wlc_sample_component_mapping(order_local, rng, reverse_prob=0.5):
    order_local = np.asarray(order_local, dtype=int)
    n = order_local.size
    if n <= 1:
        return order_local.copy()
    if n == 2:
        if rng.rand() < 0.5:
            return order_local[::-1].copy()
        return order_local.copy()
    k = int(rng.randint(0, n))
    mapped = np.roll(order_local, k)
    if rng.rand() < float(reverse_prob):
        mapped = mapped[::-1]
    return mapped.copy()

def _wlc_build_depth_groups(level_idx_rows):
    level_idx_rows = np.asarray(level_idx_rows, dtype=int)
    groups = []
    for level in np.unique(level_idx_rows):
        if level < 0:
            continue
        groups.append(np.where(level_idx_rows == level)[0].astype(int))
    return groups

def _wlc_sample_depth_mapping(group_local, rng):
    group_local = np.asarray(group_local, dtype=int)
    if group_local.size <= 1:
        return group_local.copy()
    return np.asarray(rng.permutation(group_local), dtype=int)

def _wlc_order_adjacency_signature(order_global, n_bins_x, n_bins_y):
    order_global = np.asarray(order_global, dtype=int)
    rank = {int(bin_idx): int(index) for index, bin_idx in enumerate(order_global)}
    edges = []
    for bin_idx in order_global:
        for neighbor in _wlc_neighbors_4(int(bin_idx), n_bins_x, n_bins_y):
            if neighbor in rank and rank[int(bin_idx)] < rank[neighbor]:
                edges.append((rank[int(bin_idx)], rank[neighbor]))
    return tuple(sorted(edges))

def _wlc_build_component_block_families(components, component_orders_local):
    if len(components) != len(component_orders_local):
        raise ValueError('components and component_orders_local must have the same length.')
    grouped = {}
    for component_id, (component, order_local) in enumerate(zip(components, component_orders_local)):
        order_local = np.asarray(order_local, dtype=int)
        topology_signature = tuple(
            tuple(int(value) for value in edge)
            for edge in component.get('topology_signature', ())
        )
        key = (int(component['level']), int(order_local.size), topology_signature)
        grouped.setdefault(key, []).append(int(component_id))
    families = []
    for (level, component_size, topology_signature), component_ids in sorted(grouped.items()):
        families.append(
            dict(
                level=int(level),
                component_size=int(component_size),
                topology_signature=topology_signature,
                component_ids=np.asarray(component_ids, dtype=int),
            )
        )
    return families

def _wlc_sample_component_block_mapping(n_rows, component_orders_local, families, rng):
    perm = np.arange(int(n_rows), dtype=int)
    for family in families:
        component_ids = np.asarray(family['component_ids'], dtype=int)
        source_component_ids = np.asarray(rng.permutation(component_ids), dtype=int)
        for destination_id, source_id in zip(component_ids, source_component_ids):
            destination_order = np.asarray(component_orders_local[int(destination_id)], dtype=int)
            source_order = np.asarray(component_orders_local[int(source_id)], dtype=int)
            if destination_order.size != source_order.size:
                raise ValueError('Component-block permutation requires equal-size source and destination components.')
            perm[destination_order] = source_order
    return perm

def _wlc_similarity_per_neuron(A, B, metric='pearson', normalization='auto', eps=1e-12):
    if normalization == 'auto':
        norm_mode = {'cosine': 'l2', 'pearson': 'zscore', 'spearman': 'zscore'}.get(metric, 'l2')
    else:
        norm_mode = normalization
    A_n = _normalize_columns(A, mode=norm_mode, eps=eps)
    B_n = _normalize_columns(B, mode=norm_mode, eps=eps)
    M = A_n.shape[1]
    sim = np.full(M, np.nan, dtype=float)
    valid_counts = np.zeros(M, dtype=int)
    for i in range(M):
        a = A_n[:, i]
        b = B_n[:, i]
        good = np.isfinite(a) & np.isfinite(b)
        valid_counts[i] = int(good.sum())
        if valid_counts[i] == 0:
            continue
        sim[i] = _vector_similarity(a[good], b[good], metric=metric)
    return (sim, valid_counts)

def _wlc_smooth_to_target_ratio(X, edges, obs_energy, target=1.0, max_iters=4, alpha=0.35):
    X = np.asarray(X, dtype=float)
    obs_energy = np.asarray(obs_energy, dtype=float)
    target = float(target)
    max_iters = int(max_iters)
    alpha = float(alpha)
    best_X = X.copy()
    best_iter = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio0 = _wlc_laplacian_energy_cols(best_X, edges) / obs_energy
    best_ratio = float(np.nanmedian(ratio0))
    if np.isfinite(best_ratio):
        best_score = abs(best_ratio - target)
    else:
        best_score = np.inf
    cur = X.copy()
    for it in range(1, max_iters + 1):
        cur = _wlc_component_graph_smooth_cols(cur, edges, alpha=alpha, n_iters=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = _wlc_laplacian_energy_cols(cur, edges) / obs_energy
        ratio_med = float(np.nanmedian(ratio))
        score = abs(ratio_med - target) if np.isfinite(ratio_med) else np.inf
        if score < best_score:
            best_score = score
            best_ratio = ratio_med
            best_X = cur.copy()
            best_iter = int(it)
    return (best_X, best_ratio, best_iter)

def compute_spatial_info_matched(sp1, sp2, *, control: str='intersection_avg', eps: float=1e-12):
    p1 = sp1['prior'].astype(float)
    p2 = sp2['prior'].astype(float)
    L1 = sp1['rates'].astype(float)
    L2 = sp2['rates'].astype(float)
    inter = (p1 > 0) & (p2 > 0)
    if not np.any(inter):
        K = p1.shape[0]
        z1 = np.zeros(L1.shape[1], float) if L1.ndim == 2 else np.array([], float)
        z2 = np.zeros(L2.shape[1], float) if L2.ndim == 2 else np.array([], float)
        return dict(mask_intersection=np.zeros(K, bool), SI_matched_s1=z1, SI_matched_s2=z2)
    p1i = np.where(inter, p1, 0.0)
    p2i = np.where(inter, p2, 0.0)
    if control == 'intersection_avg':
        p_common = (p1i + p2i) / max(p1i.sum() + p2i.sum(), eps)
        p_common = p_common / max(p_common.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(mask_intersection=inter, prior_common=p_common, SI_matched_s1=SI1, SI_matched_s2=SI2)
    if control == 'intersection_uniform':
        k = inter.sum()
        p_common = np.where(inter, 1.0 / k, 0.0)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(mask_intersection=inter, prior_common=p_common, SI_matched_s1=SI1, SI_matched_s2=SI2)
    if control == 'intersection_s1':
        p_common = p1i / max(p1i.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(mask_intersection=inter, prior_common=p_common, SI_matched_s1=SI1, SI_matched_s2=SI2)
    if control == 'intersection_s2':
        p_common = p2i / max(p2i.sum(), eps)
        SI1 = _skaggs_info(L1, p_common, eps=eps)
        SI2 = _skaggs_info(L2, p_common, eps=eps)
        return dict(mask_intersection=inter, prior_common=p_common, SI_matched_s1=SI1, SI_matched_s2=SI2)
    if control in ('intersection_native', 'intersection_own', 'intersection_per_session'):
        p1n = p1i / max(p1i.sum(), eps)
        p2n = p2i / max(p2i.sum(), eps)
        SI1 = _skaggs_info(L1, p1n, eps=eps)
        SI2 = _skaggs_info(L2, p2n, eps=eps)
        return dict(mask_intersection=inter, prior_s1_intersection=p1n, prior_s2_intersection=p2n, SI_matched_s1=SI1, SI_matched_s2=SI2)
    raise ValueError(f"Unknown control='{control}'")

def compute_spatial_info_session_rategated(spike_counts, raw_counts, selected_df, positions, *, n_bins_x=15, n_bins_y=15, dt=0.04, time_window=0, use_peak=False, filter_ratemap=False, sigma=1.0, min_time_per_bin_sec=5.0, eps=1e-12, neuron_ids=None, min_spikes_per_min=None, rate_source='spike'):
    pos = np.asarray(positions)
    is_1d = pos.ndim == 1 or (pos.ndim == 2 and pos.shape[1] == 1)
    if is_1d and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    (labels_pos, x_edges, y_edges) = discretize_positions(pos, n_bins_x, n_bins_y)
    n_bins = int(n_bins_x * n_bins_y)
    if neuron_ids is None:
        neuron_ids = np.arange(spike_counts.shape[1], dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)
    C_for_rate = spike_counts if rate_source == 'spike' else raw_counts
    dt_eff_rate = float(dt)
    rates_pm_all = _spikes_per_min(C_for_rate, dt_eff_rate)
    if min_spikes_per_min is not None:
        keep = rates_pm_all[neuron_ids] >= float(min_spikes_per_min)
        neuron_ids = neuron_ids[keep]
    if neuron_ids.size == 0:
        return dict(SI_native=np.array([], float), prior=np.zeros(n_bins, float), rates=np.zeros((n_bins, 0), float), mask=np.zeros(n_bins, bool), kept_neuron_ids=neuron_ids, meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges))
    frames = np.arange(len(pos))
    Q = build_count_matrix(spike_counts, raw_counts, selected_df, frames=frames, time_window=time_window, neuron_ids=neuron_ids).T
    dt_eff_tuning = dt * (time_window + 1)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    (lam_sp, prior_sp) = tuner(Q, labels_pos, None, n_bins, dt_eff_tuning, filter_ratemap, sigma, min_time=min_time_per_bin_sec, eps=eps, decode_head='none')
    SI_native = _skaggs_info(lam_sp, prior_sp, eps=eps)
    mask = prior_sp > 0
    return dict(SI_native=SI_native, prior=prior_sp, rates=lam_sp, mask=mask, kept_neuron_ids=neuron_ids, meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges))

def cross_session_spatial_information_rategated(spike_counts_train, raw_counts_train, selected_df_train, positions_train, spike_counts_test, raw_counts_test, selected_df_test, positions_test, cimap, n_bins_x=15, n_bins_y=15, dt=0.04, time_window=0, use_peak=False, filter_ratemap=False, sigma=1.0, min_time_per_bin_sec=5.0, eps=1e-12, n_neurons=None, random_state=0, train_ids=None, test_ids=None, min_spikes_per_min=None, min_spikes_per_min_train=None, min_spikes_per_min_test=None, rate_source='spike', drop_if_below='either', si_control='intersection_avg'):
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).')
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')
    rng = np.random.RandomState(random_state)
    n_tr = spike_counts_train.shape[1]
    n_te = spike_counts_test.shape[1]
    Ctr_for_rate = spike_counts_train if rate_source == 'spike' else raw_counts_train
    Cte_for_rate = spike_counts_test if rate_source == 'spike' else raw_counts_test
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)
    if train_ids is not None and test_ids is not None:
        tr = np.asarray(train_ids, int)
        te = np.asarray(test_ids, int)
        if tr.size == 0 or te.size == 0 or tr.size != te.size:
            raise ValueError('train_ids and test_ids must be same-length, non-empty.')
        if tr.min() < 0 or tr.max() >= n_tr or te.min() < 0 or (te.max() >= n_te):
            raise ValueError('train_ids/test_ids out of range.')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            (tr, te) = (tr[sel], te[sel])
        (train_ids, test_ids) = (tr, te)
    else:
        (tr, te) = _pick_matched_neurons(cimap, n_neurons=n_neurons, random_state=random_state, n_train=n_tr, n_test=n_te)
        (train_ids, test_ids) = (np.asarray(tr, int), np.asarray(te, int))
    if train_ids.size == 0:
        return dict(train_ids=np.array([], int), test_ids=np.array([], int), SI_native_s1=np.array([], float), SI_native_s2=np.array([], float), SI_matched_s1=np.array([], float), SI_matched_s2=np.array([], float), mask_intersection=None, prior_common=None, meta_train=dict(rate_per_min=rate_tr, threshold=min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min, rate_source=rate_source), summary=dict(median_native_s1=np.nan, median_native_s2=np.nan, median_matched_s1=np.nan, median_matched_s2=np.nan))
    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    pass_tr = np.ones_like(train_ids, bool)
    pass_te = np.ones_like(test_ids, bool)
    if thr_tr is not None:
        pass_tr = rate_tr[train_ids] >= float(thr_tr)
    if thr_te is not None:
        pass_te = rate_te[test_ids] >= float(thr_te)
    if drop_if_below == 'either':
        keep_rate = pass_tr & pass_te
    elif drop_if_below == 'both':
        keep_rate = pass_tr | pass_te
    else:
        raise ValueError("drop_if_below must be 'either' or 'both'.")
    train_ids = train_ids[keep_rate]
    test_ids = test_ids[keep_rate]
    if train_ids.size == 0:
        return dict(train_ids=np.array([], int), test_ids=np.array([], int), SI_native_s1=np.array([], float), SI_native_s2=np.array([], float), SI_matched_s1=np.array([], float), SI_matched_s2=np.array([], float), mask_intersection=None, prior_common=None, meta_train=dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source), meta_test=dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source), summary=dict(median_native_s1=np.nan, median_native_s2=np.nan, median_matched_s1=np.nan, median_matched_s2=np.nan))
    s1 = compute_spatial_info_session_rategated(spike_counts_train, raw_counts_train, selected_df_train, pos_tr, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, min_time_per_bin_sec=min_time_per_bin_sec, eps=eps, neuron_ids=train_ids, min_spikes_per_min=None, rate_source=rate_source)
    s2 = compute_spatial_info_session_rategated(spike_counts_test, raw_counts_test, selected_df_test, pos_te, n_bins_x=n_bins_x, n_bins_y=n_bins_y, dt=dt, time_window=time_window, use_peak=use_peak, filter_ratemap=filter_ratemap, sigma=sigma, min_time_per_bin_sec=min_time_per_bin_sec, eps=eps, neuron_ids=test_ids, min_spikes_per_min=None, rate_source=rate_source)
    sim = compute_spatial_info_matched(s1, s2, control=si_control, eps=eps)
    out = dict(train_ids=train_ids, test_ids=test_ids, SI_native_s1=s1['SI_native'], SI_native_s2=s2['SI_native'], SI_matched_s1=sim['SI_matched_s1'], SI_matched_s2=sim['SI_matched_s2'], mask_intersection=sim.get('mask_intersection', None))
    if 'prior_common' in sim:
        out['prior_common'] = sim['prior_common']
    if 'prior_s1_intersection' in sim:
        out['prior_s1_intersection'] = sim['prior_s1_intersection']
    if 'prior_s2_intersection' in sim:
        out['prior_s2_intersection'] = sim['prior_s2_intersection']
    out['meta_train'] = dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source)
    out['meta_test'] = dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source)
    out['summary'] = dict(median_native_s1=float(np.nanmedian(out['SI_native_s1'])) if out['SI_native_s1'].size else np.nan, median_native_s2=float(np.nanmedian(out['SI_native_s2'])) if out['SI_native_s2'].size else np.nan, median_matched_s1=float(np.nanmedian(out['SI_matched_s1'])) if out['SI_matched_s1'].size else np.nan, median_matched_s2=float(np.nanmedian(out['SI_matched_s2'])) if out['SI_matched_s2'].size else np.nan)
    return out


def pivot_window_spatial_information_fast(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    *,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    win_len_sec=10 * 60,
    pivot_step_sec=60,
    pivot_time_axis='selected',
    frame_dt_sec=None,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    neuron_ids=None,
    min_spikes_per_min=None,
    rate_source='spike',
    prior_mode='empirical',
    min_time_policy='fixed',
    support_mode='window',
    common_support_frac=1.0,
    common_support_min_occ_sec=None,
    common_support_reference='all_windows',
    maze_id=None,
    overlap_lags=(1, 5, 10, 20),
    return_window_payloads=False,
):
    """
    Sliding-window spatial-information trajectory for one session.

    This is a simplified migration of the 251110 correlation SI-window branch:
    - computes SI per neuron in each window,
    - supports empirical or uniform occupancy prior over visited bins,
    - emits window-level coverage diagnostics and optional common-support masking.
    """
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    is_1d = pos.shape[1] == 1
    if is_1d and n_bins_y != 1:
        raise ValueError('For 1D positions set n_bins_y=1.')

    Xs = np.asarray(spike_counts, dtype=float)
    Xr = np.asarray(raw_counts, dtype=float)
    T = int(Xs.shape[0])
    if len(selected_df) != T:
        raise ValueError('spike_counts and selected_df must have matching row counts.')

    labels, x_edges, y_edges, label_source = _spatial_labels_from_inputs(
        selected_df,
        pos,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
    )
    n_bins = int(n_bins_x * n_bins_y)
    reachable_mask = _maze_reachable_mask(n_bins_x=n_bins_x, n_bins_y=n_bins_y, maze_id=maze_id)
    reachable_bin_count = int(np.count_nonzero(reachable_mask)) if reachable_mask is not None else None
    if support_mode not in {'window', 'common_fraction', 'equal_bin_count'}:
        raise ValueError("support_mode must be 'window', 'common_fraction', or 'equal_bin_count'")
    if common_support_reference != 'all_windows':
        raise NotImplementedError("Only common_support_reference='all_windows' is implemented in this pass.")
    common_support_frac = float(common_support_frac)
    if not (0.0 < common_support_frac <= 1.0):
        raise ValueError('common_support_frac must be in (0, 1].')

    if neuron_ids is None:
        neuron_ids = np.arange(Xs.shape[1], dtype=int)
    else:
        neuron_ids = np.asarray(neuron_ids, dtype=int)

    if min_spikes_per_min is not None:
        src = Xs if rate_source == 'spike' else Xr
        rates = _spikes_per_min(src, dt)
        keep = rates[neuron_ids] >= float(min_spikes_per_min)
        neuron_ids = neuron_ids[keep]

    if neuron_ids.size == 0:
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([], dtype=float),
            si_matrix=np.zeros((0, 0), dtype=float),
            si_mean=np.array([], dtype=float),
            si_std=np.array([], dtype=float),
            kept_neuron_ids=neuron_ids,
            meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
        )

    if pivot_time_axis == 'selected':
        t = np.arange(T, dtype=float) * float(dt)
    elif pivot_time_axis == 'maze':
        if 'maze_time' not in selected_df.columns:
            raise ValueError("pivot_time_axis='maze' requires selected_df['maze_time']")
        t = selected_df['maze_time'].to_numpy(dtype=float)
    elif pivot_time_axis == 'frame':
        f = selected_df['frame'].to_numpy(dtype=float)
        if frame_dt_sec is None:
            frame_dt_sec = float(dt)
        t = f * float(frame_dt_sec)
    else:
        raise ValueError("pivot_time_axis must be 'selected'|'maze'|'frame'")

    half = float(win_len_sec) / 2.0
    centers = []
    cur = float(t[0]) + half
    while cur + half <= float(t[-1]):
        centers.append(cur)
        cur += float(pivot_step_sec)

    if len(centers) == 0:
        return dict(
            pivot_bounds=[],
            pivot_times=np.array([], dtype=float),
            si_matrix=np.zeros((0, len(neuron_ids)), dtype=float),
            si_mean=np.array([], dtype=float),
            si_std=np.array([], dtype=float),
            kept_neuron_ids=neuron_ids,
            meta=dict(n_bins=n_bins, x_edges=x_edges, y_edges=y_edges),
        )

    bounds = []
    for c in centers:
        m = (t >= (c - half)) & (t < (c + half))
        idx = np.where(m)[0]
        if idx.size == 0:
            bounds.append((0, 0))
        else:
            bounds.append((int(idx[0]), int(idx[-1]) + 1))

    si_rows = []
    dt_eff = float(dt) * float(time_window + 1)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    effective_min_time_per_bin_sec, n_nonoverlap_windows = _resolve_min_time_threshold(
        min_time_per_bin_sec=min_time_per_bin_sec,
        min_time_policy=min_time_policy,
        time_axis=t,
        win_len_sec=win_len_sec,
        dt_eff=dt_eff,
        n_samples=T,
    )
    support_min_time = (
        float(common_support_min_occ_sec)
        if common_support_min_occ_sec is not None
        else float(effective_min_time_per_bin_sec)
    )
    window_payloads = []
    support_masks = []
    analysis_support_masks = []
    metrics_rows = []
    for s, e in bounds:
        if e <= s:
            window_payloads.append(None)
            support_masks.append(np.zeros(n_bins, dtype=bool))
            metrics_rows.append(
                {
                    "window_start": int(s),
                    "window_end": int(e),
                    "window_len": int(max(e - s, 0)),
                    "visited_bin_count": 0,
                    "valid_bin_count_after_min_occ": 0,
                    "reachable_bin_fraction": np.nan,
                    "occupancy_seconds_total": 0.0,
                    "occupancy_seconds_per_valid_bin": np.nan,
                    "occupancy_entropy": np.nan,
                    "max_single_bin_fraction": np.nan,
                    "per_window_neuron_count": int(len(neuron_ids)),
                }
            )
            continue
        frames = np.arange(s, e, dtype=int)
        if time_window == 0:
            Q = Xs[s:e, :][:, neuron_ids]
            labs = labels[s:e]
        else:
            Q = build_count_matrix(Xs, Xr, selected_df, frames=frames, time_window=time_window, neuron_ids=neuron_ids).T
            labs = labels[s:e]

        lam, prior = tuner(
            Q,
            labs,
            None,
            n_bins=n_bins,
            dt=dt_eff,
            filter_ratemap=filter_ratemap,
            sigma=sigma,
            min_time=effective_min_time_per_bin_sec,
            eps=eps,
            decode_head='none',
        )
        occ_raw = np.bincount(labs.astype(int), minlength=n_bins).astype(float) * dt_eff
        visited_mask = occ_raw > 0
        valid_mask = occ_raw >= float(effective_min_time_per_bin_sec)
        support_mask = occ_raw >= float(support_min_time)
        support_masks.append(support_mask)
        valid_bin_count = int(np.count_nonzero(valid_mask))
        occ_valid = occ_raw[valid_mask]
        reachable_bin_fraction = (
            float(valid_bin_count / reachable_bin_count) if reachable_bin_count not in (None, 0) else np.nan
        )
        metrics_rows.append(
            {
                "window_start": int(s),
                "window_end": int(e),
                "window_len": int(e - s),
                "visited_bin_count": int(np.count_nonzero(visited_mask)),
                "valid_bin_count_after_min_occ": valid_bin_count,
                "reachable_bin_fraction": reachable_bin_fraction,
                "occupancy_seconds_total": float(occ_raw.sum()),
                "occupancy_seconds_per_valid_bin": (
                    float(occ_valid.sum() / valid_bin_count) if valid_bin_count > 0 else np.nan
                ),
                "occupancy_entropy": _occupancy_entropy(prior, eps=eps),
                "max_single_bin_fraction": float(np.nanmax(prior)) if np.any(prior > 0) else np.nan,
                "per_window_neuron_count": int(len(neuron_ids)),
            }
        )
        window_payloads.append(
            {
                "lam": lam,
                "prior": prior,
                "occ_raw": occ_raw,
                "valid_mask": valid_mask,
                "support_mask": support_mask,
            }
        )

    support_masks_arr = np.asarray(support_masks, dtype=bool) if support_masks else np.zeros((0, n_bins), dtype=bool)
    common_support_mask = None
    equalized_bin_count = None
    if support_mode == 'common_fraction' and support_masks_arr.size:
        support_freq = np.mean(support_masks_arr, axis=0)
        common_support_mask = support_freq >= common_support_frac
    elif support_mode == 'equal_bin_count':
        valid_counts = [
            int(np.count_nonzero(payload["valid_mask"]))
            for payload in window_payloads
            if payload is not None
        ]
        equalized_bin_count = int(min(valid_counts)) if valid_counts else 0

    for payload in window_payloads:
        if payload is None:
            analysis_support_masks.append(np.zeros(n_bins, dtype=bool))
            si_rows.append(np.full(len(neuron_ids), np.nan, dtype=float))
            continue
        lam = payload["lam"]
        prior = payload["prior"]
        if common_support_mask is None and support_mode != 'equal_bin_count':
            support_use = payload["valid_mask"]
        elif support_mode == 'equal_bin_count':
            support_use = np.zeros_like(payload["valid_mask"], dtype=bool)
            valid_idx = np.flatnonzero(payload["valid_mask"])
            if equalized_bin_count is not None and equalized_bin_count > 0 and valid_idx.size >= int(equalized_bin_count):
                occ_valid = np.asarray(payload["occ_raw"], dtype=float)[valid_idx]
                order = np.argsort(-occ_valid, kind='stable')
                chosen = valid_idx[order[: int(equalized_bin_count)]]
                support_use[chosen] = True
        else:
            support_use = np.asarray(common_support_mask, dtype=bool)
        analysis_support_masks.append(np.asarray(support_use, dtype=bool))
        if not np.any(support_use):
            si_rows.append(np.full(len(neuron_ids), np.nan, dtype=float))
            continue
        if prior_mode == 'uniform':
            prior_use = np.zeros_like(prior)
            valid = (prior > 0) & support_use
            if np.any(valid):
                prior_use[valid] = 1.0 / float(np.sum(valid))
        else:
            prior_use = np.where(support_use, prior, 0.0)
            psum = float(prior_use.sum())
            if psum > 0:
                prior_use = prior_use / psum
        if float(np.nansum(prior_use)) <= eps:
            si_rows.append(np.full(len(neuron_ids), np.nan, dtype=float))
            continue
        si_rows.append(_skaggs_info(lam, prior_use, eps=eps))

    common_support_count = int(np.count_nonzero(common_support_mask)) if common_support_mask is not None else None
    for row, payload in zip(metrics_rows, window_payloads):
        if payload is None:
            row["fraction_of_common_support"] = np.nan
            continue
        if common_support_mask is None or common_support_count in (None, 0):
            row["fraction_of_common_support"] = np.nan
            continue
        row["fraction_of_common_support"] = float(
            np.count_nonzero(payload["support_mask"] & common_support_mask) / common_support_count
        )

    si_mat = np.asarray(si_rows, dtype=float)
    si_mean = np.nanmean(si_mat, axis=1) if si_mat.size else np.array([], dtype=float)
    si_std = np.nanstd(si_mat, axis=1) if si_mat.size else np.array([], dtype=float)
    window_metrics = pd.DataFrame(metrics_rows)
    overlap_summary = _mean_jaccard_by_lag(support_masks, lags=overlap_lags)
    window_prior_matrix = np.vstack(
        [payload["prior"] if payload is not None else np.zeros(n_bins, dtype=float) for payload in window_payloads]
    ) if window_payloads else np.zeros((0, n_bins), dtype=float)
    window_valid_mask_matrix = np.vstack(
        [payload["valid_mask"] if payload is not None else np.zeros(n_bins, dtype=bool) for payload in window_payloads]
    ) if window_payloads else np.zeros((0, n_bins), dtype=bool)
    window_support_mask_matrix = np.vstack(
        [payload["support_mask"] if payload is not None else np.zeros(n_bins, dtype=bool) for payload in window_payloads]
    ) if window_payloads else np.zeros((0, n_bins), dtype=bool)
    window_analysis_support_mask_matrix = np.vstack(analysis_support_masks) if analysis_support_masks else np.zeros((0, n_bins), dtype=bool)

    out = dict(
        pivot_bounds=bounds,
        pivot_times=np.asarray(centers, dtype=float),
        si_matrix=si_mat,
        si_mean=si_mean,
        si_std=si_std,
        window_metrics=window_metrics,
        window_prior_matrix=window_prior_matrix,
        window_valid_mask_matrix=window_valid_mask_matrix,
        window_support_mask_matrix=window_support_mask_matrix,
        window_analysis_support_mask_matrix=window_analysis_support_mask_matrix,
        kept_neuron_ids=neuron_ids,
        meta=dict(
            n_bins=n_bins,
            x_edges=x_edges,
            y_edges=y_edges,
            prior_mode=prior_mode,
            min_time_policy=min_time_policy,
            base_min_time_per_bin_sec=float(min_time_per_bin_sec),
            effective_min_time_per_bin_sec=float(effective_min_time_per_bin_sec),
            scaled_nonoverlap_window_count=int(n_nonoverlap_windows),
            timebase=pivot_time_axis,
            label_source=label_source,
            maze_id=maze_id,
            reachable_bin_count=reachable_bin_count,
            support_mode=support_mode,
            common_support_frac=(common_support_frac if support_mode == 'common_fraction' else None),
            common_support_reference=common_support_reference,
            common_support_min_occ_sec=support_min_time,
            common_support_mask=common_support_mask,
            common_support_bin_count=common_support_count,
            equalized_bin_count=equalized_bin_count,
            overlap_summary=overlap_summary,
        ),
    )
    if bool(return_window_payloads):
        out["window_payloads"] = window_payloads
    return out


def paired_pivot_window_spatial_information_fast(
    spike_counts_train,
    raw_counts_train,
    selected_df_train,
    positions_train,
    spike_counts_test,
    raw_counts_test,
    selected_df_test,
    positions_test,
    *,
    cimap=None,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    win_len_sec=10 * 60,
    pivot_step_sec=60,
    pivot_time_axis='selected',
    frame_dt_sec=None,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    n_neurons=None,
    random_state=0,
    train_ids=None,
    test_ids=None,
    min_spikes_per_min=None,
    min_spikes_per_min_train=None,
    min_spikes_per_min_test=None,
    rate_source='spike',
    drop_if_below='either',
    prior_mode='empirical',
    min_time_policy='fixed',
    common_support_min_occ_sec=None,
    maze_id=None,
    overlap_lags=(1, 5, 10, 20),
    pair_support_mode='equal_bin_count',
):
    """Windowed within-session SI for a matched session pair.

    The two sessions are first restricted to the same matched-neuron set, then
    each paired window is evaluated on the same spatial support mask. Support
    can be the full per-window intersection or an equalized bin-count subset.
    In addition to the default empirical-prior SI, this helper also exposes
    shared-prior variants so downstream analyses can distinguish occupancy
    reweighting from tuning changes.
    """
    pos_tr = np.asarray(positions_train)
    pos_te = np.asarray(positions_test)
    is_1d_tr = pos_tr.ndim == 1 or (pos_tr.ndim == 2 and pos_tr.shape[1] == 1)
    is_1d_te = pos_te.ndim == 1 or (pos_te.ndim == 2 and pos_te.shape[1] == 1)
    if is_1d_tr != is_1d_te:
        raise ValueError(
            f'TRAIN and TEST positions must have the same dimensionality (got TRAIN 1D={is_1d_tr}, TEST 1D={is_1d_te}).'
        )
    if is_1d_tr and n_bins_y != 1:
        raise ValueError('For 1D positions (shape (T,) or (T,1)), please set n_bins_y=1 and n_bins_x = desired number of spatial bins.')

    rng = np.random.RandomState(random_state)
    n_tr = int(np.asarray(spike_counts_train).shape[1])
    n_te = int(np.asarray(spike_counts_test).shape[1])
    Ctr_for_rate = spike_counts_train if rate_source == 'spike' else raw_counts_train
    Cte_for_rate = spike_counts_test if rate_source == 'spike' else raw_counts_test
    rate_tr = _spikes_per_min(Ctr_for_rate, dt)
    rate_te = _spikes_per_min(Cte_for_rate, dt)

    if train_ids is not None and test_ids is not None:
        tr = np.asarray(train_ids, int)
        te = np.asarray(test_ids, int)
        if tr.size == 0 or te.size == 0 or tr.size != te.size:
            raise ValueError('train_ids and test_ids must be same-length, non-empty.')
        if tr.min() < 0 or tr.max() >= n_tr or te.min() < 0 or te.max() >= n_te:
            raise ValueError('train_ids/test_ids out of range.')
        if n_neurons is not None and tr.size > n_neurons:
            sel = rng.choice(tr.size, int(n_neurons), replace=False)
            tr = tr[sel]
            te = te[sel]
        train_ids = tr
        test_ids = te
    else:
        tr, te = _pick_matched_neurons(
            cimap,
            n_neurons=n_neurons,
            random_state=random_state,
            n_train=n_tr,
            n_test=n_te,
        )
        train_ids = np.asarray(tr, int)
        test_ids = np.asarray(te, int)

    thr_tr = min_spikes_per_min_train if min_spikes_per_min_train is not None else min_spikes_per_min
    thr_te = min_spikes_per_min_test if min_spikes_per_min_test is not None else min_spikes_per_min
    if train_ids.size:
        pass_tr = np.ones_like(train_ids, bool) if thr_tr is None else rate_tr[train_ids] >= float(thr_tr)
        pass_te = np.ones_like(test_ids, bool) if thr_te is None else rate_te[test_ids] >= float(thr_te)
        if drop_if_below == 'either':
            keep_rate = pass_tr & pass_te
        elif drop_if_below == 'both':
            keep_rate = pass_tr | pass_te
        else:
            raise ValueError("drop_if_below must be 'either' or 'both'.")
        train_ids = train_ids[keep_rate]
        test_ids = test_ids[keep_rate]

    if pair_support_mode not in {'intersection', 'equal_bin_count'}:
        raise ValueError("pair_support_mode must be 'intersection' or 'equal_bin_count'.")

    base_kwargs = dict(
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        dt=dt,
        time_window=time_window,
        win_len_sec=win_len_sec,
        pivot_step_sec=pivot_step_sec,
        pivot_time_axis=pivot_time_axis,
        frame_dt_sec=frame_dt_sec,
        use_peak=use_peak,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        min_time_per_bin_sec=min_time_per_bin_sec,
        eps=eps,
        min_spikes_per_min=None,
        rate_source=rate_source,
        prior_mode=prior_mode,
        min_time_policy=min_time_policy,
        support_mode='window',
        common_support_frac=1.0,
        common_support_min_occ_sec=common_support_min_occ_sec,
        common_support_reference='all_windows',
        maze_id=maze_id,
        overlap_lags=overlap_lags,
        return_window_payloads=True,
    )

    train_base = pivot_window_spatial_information_fast(
        spike_counts_train,
        raw_counts_train,
        selected_df_train,
        positions_train,
        neuron_ids=train_ids,
        **base_kwargs,
    )
    test_base = pivot_window_spatial_information_fast(
        spike_counts_test,
        raw_counts_test,
        selected_df_test,
        positions_test,
        neuron_ids=test_ids,
        **base_kwargs,
    )

    payloads_tr = list(train_base.get('window_payloads', []))
    payloads_te = list(test_base.get('window_payloads', []))
    n_pair_windows = min(len(payloads_tr), len(payloads_te))
    payloads_tr = payloads_tr[:n_pair_windows]
    payloads_te = payloads_te[:n_pair_windows]

    n_bins = int(train_base.get('meta', {}).get('n_bins', n_bins_x * n_bins_y))
    common_valid_masks = []
    common_valid_counts = []
    pair_support_masks = []
    for payload_tr, payload_te in zip(payloads_tr, payloads_te):
        if payload_tr is None or payload_te is None:
            common_valid = np.zeros(n_bins, dtype=bool)
        else:
            common_valid = np.asarray(payload_tr['valid_mask'], dtype=bool) & np.asarray(payload_te['valid_mask'], dtype=bool)
        common_valid_masks.append(common_valid)
        common_valid_counts.append(int(np.count_nonzero(common_valid)))

    pair_equalized_bin_count = None
    if pair_support_mode == 'equal_bin_count':
        positive_counts = [int(c) for c in common_valid_counts if int(c) > 0]
        pair_equalized_bin_count = int(min(positive_counts)) if positive_counts else 0

    for payload_tr, payload_te, common_valid in zip(payloads_tr, payloads_te, common_valid_masks):
        support_use = np.zeros(n_bins, dtype=bool)
        if payload_tr is None or payload_te is None:
            pair_support_masks.append(support_use)
            continue
        valid_idx = np.flatnonzero(common_valid)
        if pair_support_mode == 'intersection':
            support_use = np.asarray(common_valid, dtype=bool)
        elif pair_equalized_bin_count is not None and pair_equalized_bin_count > 0 and valid_idx.size >= int(pair_equalized_bin_count):
            occ_tr = np.asarray(payload_tr['occ_raw'], dtype=float)[valid_idx]
            occ_te = np.asarray(payload_te['occ_raw'], dtype=float)[valid_idx]
            score = np.minimum(occ_tr, occ_te)
            order = np.argsort(-score, kind='stable')
            chosen = valid_idx[order[: int(pair_equalized_bin_count)]]
            support_use[chosen] = True
        pair_support_masks.append(np.asarray(support_use, dtype=bool))

    common_avg_priors = []
    uniform_support_priors = []
    prior_js_divergence = []
    prior_total_variation = []
    pair_fraction_of_session_a_valid = []
    pair_fraction_of_session_b_valid = []
    for payload_tr, payload_te, support_use in zip(payloads_tr, payloads_te, pair_support_masks):
        if payload_tr is None or payload_te is None:
            common_avg_priors.append(np.zeros(n_bins, dtype=float))
            uniform_support_priors.append(np.zeros(n_bins, dtype=float))
            prior_js_divergence.append(np.nan)
            prior_total_variation.append(np.nan)
            pair_fraction_of_session_a_valid.append(np.nan)
            pair_fraction_of_session_b_valid.append(np.nan)
            continue
        prior_tr = _normalize_support_prior(payload_tr['prior'], support_use, eps=eps)
        prior_te = _normalize_support_prior(payload_te['prior'], support_use, eps=eps)
        avg_prior = 0.5 * (prior_tr + prior_te)
        avg_sum = float(np.nansum(avg_prior))
        if avg_sum > eps:
            avg_prior = avg_prior / avg_sum
        else:
            avg_prior = np.zeros(n_bins, dtype=float)
        common_avg_priors.append(avg_prior)
        uniform_support_priors.append(_uniform_support_prior(support_use))
        prior_js_divergence.append(_safe_js_divergence(prior_tr, prior_te, eps=eps))
        prior_total_variation.append(_safe_total_variation(prior_tr, prior_te))
        valid_tr = int(np.count_nonzero(np.asarray(payload_tr['valid_mask'], dtype=bool)))
        valid_te = int(np.count_nonzero(np.asarray(payload_te['valid_mask'], dtype=bool)))
        kept = int(np.count_nonzero(support_use))
        pair_fraction_of_session_a_valid.append(float(kept / valid_tr) if valid_tr > 0 else np.nan)
        pair_fraction_of_session_b_valid.append(float(kept / valid_te) if valid_te > 0 else np.nan)

    def _finalize_pair_side(
        base_result,
        payloads,
        kept_ids,
        *,
        prior_variant='empirical',
        shared_prior_matrix=None,
        pair_support_fraction_of_session_valid=None,
    ):
        si_rows = []
        metrics = base_result.get('window_metrics', pd.DataFrame()).copy()
        if isinstance(metrics, pd.DataFrame):
            metrics = metrics.iloc[:n_pair_windows].reset_index(drop=True)
        else:
            metrics = pd.DataFrame(metrics).iloc[:n_pair_windows].reset_index(drop=True)
        prior_rows = []
        for idx, (payload, support_use) in enumerate(zip(payloads, pair_support_masks)):
            if payload is None:
                si_rows.append(np.full(len(kept_ids), np.nan, dtype=float))
                prior_rows.append(np.zeros(n_bins, dtype=float))
                continue
            lam = np.asarray(payload['lam'], dtype=float)
            prior = np.asarray(payload['prior'], dtype=float)
            if prior_variant == 'uniform':
                prior_use = np.asarray(
                    shared_prior_matrix[idx] if shared_prior_matrix is not None else _uniform_support_prior(support_use),
                    dtype=float,
                )
            elif prior_variant == 'common_avg':
                prior_use = np.asarray(
                    shared_prior_matrix[idx] if shared_prior_matrix is not None else np.zeros_like(prior, dtype=float),
                    dtype=float,
                )
            elif prior_mode == 'uniform':
                prior_use = _uniform_support_prior((prior > 0) & support_use)
            else:
                prior_use = np.where(support_use, prior, 0.0)
                psum = float(prior_use.sum())
                if psum > 0:
                    prior_use = prior_use / psum
            if not np.any(support_use) or float(np.nansum(prior_use)) <= eps:
                si_rows.append(np.full(len(kept_ids), np.nan, dtype=float))
            else:
                si_rows.append(_skaggs_info(lam, prior_use, eps=eps))
            prior_rows.append(np.asarray(prior_use, dtype=float))

        si_mat = np.asarray(si_rows, dtype=float)
        si_mean = np.nanmean(si_mat, axis=1) if si_mat.size else np.array([], dtype=float)
        si_std = np.nanstd(si_mat, axis=1) if si_mat.size else np.array([], dtype=float)
        if len(metrics) == n_pair_windows and n_pair_windows:
            metrics['pair_common_valid_bin_count'] = np.asarray(common_valid_counts, dtype=int)
            metrics['pair_support_bin_count'] = np.asarray(
                [int(np.count_nonzero(mask)) for mask in pair_support_masks],
                dtype=int,
            )
            frac = np.full(n_pair_windows, np.nan, dtype=float)
            for i, count in enumerate(common_valid_counts):
                if int(count) > 0:
                    frac[i] = float(np.count_nonzero(pair_support_masks[i]) / float(count))
            metrics['pair_fraction_of_common_valid'] = frac
            metrics['pair_support_fraction_of_session_valid'] = np.asarray(
                pair_support_fraction_of_session_valid if pair_support_fraction_of_session_valid is not None else np.full(n_pair_windows, np.nan, dtype=float),
                dtype=float,
            )
            metrics['pair_prior_js_divergence'] = np.asarray(prior_js_divergence, dtype=float)
            metrics['pair_prior_total_variation'] = np.asarray(prior_total_variation, dtype=float)

        out = {
            key: value
            for key, value in base_result.items()
            if key != 'window_payloads'
        }
        out['pivot_bounds'] = list(base_result.get('pivot_bounds', []))[:n_pair_windows]
        out['pivot_times'] = np.asarray(base_result.get('pivot_times', np.array([])), dtype=float)[:n_pair_windows]
        out['si_matrix'] = si_mat
        out['si_mean'] = si_mean
        out['si_std'] = si_std
        out['window_applied_prior_matrix'] = (
            np.vstack(prior_rows) if prior_rows else np.zeros((0, n_bins), dtype=float)
        )
        out['window_metrics'] = metrics
        out['window_common_valid_mask_matrix'] = (
            np.vstack(common_valid_masks) if common_valid_masks else np.zeros((0, n_bins), dtype=bool)
        )
        out['window_pair_support_mask_matrix'] = (
            np.vstack(pair_support_masks) if pair_support_masks else np.zeros((0, n_bins), dtype=bool)
        )
        meta = dict(base_result.get('meta', {}))
        meta['pair_support_mode'] = str(pair_support_mode)
        meta['pair_prior_variant'] = str(prior_variant)
        meta['pair_equalized_bin_count'] = (
            int(pair_equalized_bin_count) if pair_equalized_bin_count is not None else None
        )
        meta['paired_window_count'] = int(n_pair_windows)
        meta['pair_overlap_summary'] = _mean_jaccard_by_lag(pair_support_masks, lags=overlap_lags)
        out['meta'] = meta
        return out

    train_out = _finalize_pair_side(
        train_base,
        payloads_tr,
        train_ids,
        prior_variant='empirical',
        pair_support_fraction_of_session_valid=pair_fraction_of_session_a_valid,
    )
    test_out = _finalize_pair_side(
        test_base,
        payloads_te,
        test_ids,
        prior_variant='empirical',
        pair_support_fraction_of_session_valid=pair_fraction_of_session_b_valid,
    )
    train_out_common_avg = _finalize_pair_side(
        train_base,
        payloads_tr,
        train_ids,
        prior_variant='common_avg',
        shared_prior_matrix=common_avg_priors,
        pair_support_fraction_of_session_valid=pair_fraction_of_session_a_valid,
    )
    test_out_common_avg = _finalize_pair_side(
        test_base,
        payloads_te,
        test_ids,
        prior_variant='common_avg',
        shared_prior_matrix=common_avg_priors,
        pair_support_fraction_of_session_valid=pair_fraction_of_session_b_valid,
    )
    train_out_uniform = _finalize_pair_side(
        train_base,
        payloads_tr,
        train_ids,
        prior_variant='uniform',
        shared_prior_matrix=uniform_support_priors,
        pair_support_fraction_of_session_valid=pair_fraction_of_session_a_valid,
    )
    test_out_uniform = _finalize_pair_side(
        test_base,
        payloads_te,
        test_ids,
        prior_variant='uniform',
        shared_prior_matrix=uniform_support_priors,
        pair_support_fraction_of_session_valid=pair_fraction_of_session_b_valid,
    )

    return {
        'train_ids': np.asarray(train_ids, dtype=int),
        'test_ids': np.asarray(test_ids, dtype=int),
        'meta_train': dict(rate_per_min=rate_tr, threshold=thr_tr, rate_source=rate_source),
        'meta_test': dict(rate_per_min=rate_te, threshold=thr_te, rate_source=rate_source),
        'paired_window_count': int(n_pair_windows),
        'pair_support_mode': str(pair_support_mode),
        'pair_equalized_bin_count': (
            int(pair_equalized_bin_count) if pair_equalized_bin_count is not None else None
        ),
        'train_window_result': train_out,
        'test_window_result': test_out,
        'train_window_result_common_avg_prior': train_out_common_avg,
        'test_window_result_common_avg_prior': test_out_common_avg,
        'train_window_result_uniform_prior': train_out_uniform,
        'test_window_result_uniform_prior': test_out_uniform,
        'pair_common_avg_prior_matrix': (
            np.vstack(common_avg_priors) if common_avg_priors else np.zeros((0, n_bins), dtype=float)
        ),
        'pair_uniform_prior_matrix': (
            np.vstack(uniform_support_priors) if uniform_support_priors else np.zeros((0, n_bins), dtype=float)
        ),
        'pair_prior_js_divergence': np.asarray(prior_js_divergence, dtype=float),
        'pair_prior_total_variation': np.asarray(prior_total_variation, dtype=float),
        'pair_support_fraction_of_session_a_valid': np.asarray(pair_fraction_of_session_a_valid, dtype=float),
        'pair_support_fraction_of_session_b_valid': np.asarray(pair_fraction_of_session_b_valid, dtype=float),
    }


def coverage_regress_windowed_si(
    window_result,
    *,
    covariate_cols=('reachable_bin_fraction', 'occupancy_entropy', 'max_single_bin_fraction'),
    min_points=5,
):
    pivot_times = np.asarray(window_result.get('pivot_times', np.array([])), dtype=float)
    si_mat = np.asarray(window_result.get('si_matrix', np.zeros((0, 0))), dtype=float)
    metrics = window_result.get('window_metrics', pd.DataFrame())
    if not isinstance(metrics, pd.DataFrame):
        metrics = pd.DataFrame(metrics)
    if si_mat.ndim != 2:
        raise ValueError('window_result["si_matrix"] must be 2D')
    if si_mat.shape[0] != len(pivot_times):
        raise ValueError('pivot_times length must match si_matrix rows')
    if len(metrics) != si_mat.shape[0]:
        raise ValueError('window_metrics length must match si_matrix rows')
    covariate_cols = tuple(str(c) for c in covariate_cols)
    X_cov = np.column_stack(
        [pd.to_numeric(metrics[c], errors='coerce').to_numpy(dtype=float) for c in covariate_cols]
    ) if covariate_cols else np.zeros((si_mat.shape[0], 0), dtype=float)
    residual_mat = np.full_like(si_mat, np.nan, dtype=float)
    rows = []
    for j in range(si_mat.shape[1]):
        y = si_mat[:, j]
        keep = np.isfinite(y) & np.isfinite(pivot_times)
        if X_cov.size:
            keep &= np.all(np.isfinite(X_cov), axis=1)
        n_keep = int(np.count_nonzero(keep))
        raw_slope = _linear_slope(pivot_times, y)
        if n_keep < max(int(min_points), X_cov.shape[1] + 2):
            rows.append(
                {
                    'neuron_index': int(j),
                    'n_points': n_keep,
                    'raw_slope': raw_slope,
                    'residual_slope': np.nan,
                    'r2': np.nan,
                }
            )
            continue
        X = np.column_stack([np.ones(n_keep, dtype=float), X_cov[keep]])
        yk = y[keep]
        beta, _, _, _ = np.linalg.lstsq(X, yk, rcond=None)
        fitted = X @ beta
        resid = yk - fitted
        residual_mat[keep, j] = resid
        ss_res = float(np.sum((yk - fitted) ** 2))
        ss_tot = float(np.sum((yk - np.mean(yk)) ** 2))
        r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
        rows.append(
            {
                'neuron_index': int(j),
                'n_points': n_keep,
                'raw_slope': raw_slope,
                'residual_slope': _linear_slope(pivot_times[keep], resid),
                'r2': r2,
            }
        )
    regression_df = pd.DataFrame(rows)
    if residual_mat.size:
        finite_counts = np.sum(np.isfinite(residual_mat), axis=1)
        residual_mean = np.full(residual_mat.shape[0], np.nan, dtype=float)
        if np.any(finite_counts > 0):
            residual_mean[finite_counts > 0] = (
                np.nansum(residual_mat[finite_counts > 0], axis=1) / finite_counts[finite_counts > 0]
            )
    else:
        residual_mean = np.array([], dtype=float)
    observed_mean = np.nanmean(si_mat, axis=1) if si_mat.size else np.array([], dtype=float)
    summary = {
        'covariates': list(covariate_cols),
        'n_windows': int(si_mat.shape[0]),
        'n_neurons': int(si_mat.shape[1]),
        'n_neurons_regressed': int(regression_df['residual_slope'].notna().sum()) if not regression_df.empty else 0,
        'raw_mean_slope': _linear_slope(pivot_times, observed_mean),
        'residual_mean_slope': _linear_slope(pivot_times, residual_mean),
        'mean_r2': float(pd.to_numeric(regression_df['r2'], errors='coerce').mean()) if not regression_df.empty else np.nan,
    }
    return {
        'residual_matrix': residual_mat,
        'residual_mean': residual_mean,
        'observed_mean': observed_mean,
        'regression_df': regression_df,
        'summary': summary,
    }


def stationary_tuning_windowed_si_reference(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    *,
    observed_window_result=None,
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    min_spikes_per_min=None,
    rate_source='spike',
    prior_mode='empirical',
    min_time_policy='fixed',
    support_mode='window',
    common_support_frac=1.0,
    common_support_min_occ_sec=None,
    common_support_reference='all_windows',
    maze_id=None,
    **pivot_kwargs,
):
    if observed_window_result is None:
        observed_window_result = pivot_window_spatial_information_fast(
            spike_counts,
            raw_counts,
            selected_df,
            positions,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            dt=dt,
            time_window=time_window,
            use_peak=use_peak,
            filter_ratemap=filter_ratemap,
            sigma=sigma,
            min_time_per_bin_sec=min_time_per_bin_sec,
            eps=eps,
            min_spikes_per_min=min_spikes_per_min,
            rate_source=rate_source,
            prior_mode=prior_mode,
            min_time_policy=min_time_policy,
            support_mode=support_mode,
            common_support_frac=common_support_frac,
            common_support_min_occ_sec=common_support_min_occ_sec,
            common_support_reference=common_support_reference,
            maze_id=maze_id,
            **pivot_kwargs,
        )
    kept_neuron_ids = np.asarray(observed_window_result.get('kept_neuron_ids', np.array([], dtype=int)), dtype=int)
    if kept_neuron_ids.size == 0:
        return {
            'stationary_si_matrix': np.zeros((0, 0), dtype=float),
            'stationary_si_mean': np.array([], dtype=float),
            'stationary_slope': np.nan,
            'excess_si_mean': np.array([], dtype=float),
            'excess_slope': np.nan,
            'summary': {'stationary_slope': np.nan, 'excess_slope': np.nan},
        }

    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    labels_all, _, _, _ = _spatial_labels_from_inputs(
        selected_df,
        pos,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
    )
    Xs = np.asarray(spike_counts, dtype=float)
    Xr = np.asarray(raw_counts, dtype=float)
    if time_window == 0:
        Q_full = Xs[:, kept_neuron_ids]
    else:
        frames = np.arange(Xs.shape[0], dtype=int)
        Q_full = build_count_matrix(Xs, Xr, selected_df, frames=frames, time_window=time_window, neuron_ids=kept_neuron_ids).T
    dt_eff = float(dt) * float(time_window + 1)
    effective_min_time_per_bin_sec = float(
        observed_window_result.get('meta', {}).get('effective_min_time_per_bin_sec', min_time_per_bin_sec)
    )
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    lam_global, _ = tuner(
        Q_full,
        labels_all,
        None,
        n_bins=int(n_bins_x * n_bins_y),
        dt=dt_eff,
        filter_ratemap=filter_ratemap,
        sigma=sigma,
        min_time=effective_min_time_per_bin_sec,
        eps=eps,
        decode_head='none',
    )
    prior_matrix = np.asarray(observed_window_result.get('window_prior_matrix', np.zeros((0, 0))), dtype=float)
    valid_matrix = np.asarray(observed_window_result.get('window_valid_mask_matrix', np.zeros((0, 0), dtype=bool)), dtype=bool)
    analysis_support_matrix = np.asarray(
        observed_window_result.get('window_analysis_support_mask_matrix', np.zeros((0, 0), dtype=bool)),
        dtype=bool,
    )
    common_support_mask = observed_window_result.get('meta', {}).get('common_support_mask', None)
    stationary_rows = []
    for i in range(prior_matrix.shape[0]):
        prior = prior_matrix[i]
        valid_mask = valid_matrix[i] if i < valid_matrix.shape[0] else np.zeros_like(prior, dtype=bool)
        if i < analysis_support_matrix.shape[0] and analysis_support_matrix.shape[1] == prior.shape[0]:
            support_use = analysis_support_matrix[i]
        elif common_support_mask is None:
            support_use = valid_mask
        else:
            support_use = np.asarray(common_support_mask, dtype=bool)
        if prior_mode == 'uniform':
            prior_use = np.zeros_like(prior)
            keep = (prior > 0) & support_use
            if np.any(keep):
                prior_use[keep] = 1.0 / float(np.sum(keep))
        else:
            prior_use = np.where(support_use, prior, 0.0)
            psum = float(prior_use.sum())
            if psum > 0:
                prior_use = prior_use / psum
        stationary_rows.append(_skaggs_info(lam_global, prior_use, eps=eps))
    stationary_si_matrix = np.asarray(stationary_rows, dtype=float)
    stationary_si_mean = np.nanmean(stationary_si_matrix, axis=1) if stationary_si_matrix.size else np.array([], dtype=float)
    observed_mean = np.asarray(observed_window_result.get('si_mean', np.array([])), dtype=float)
    excess_si_mean = observed_mean - stationary_si_mean if observed_mean.shape == stationary_si_mean.shape else np.array([], dtype=float)
    pivot_times = np.asarray(observed_window_result.get('pivot_times', np.array([])), dtype=float)
    stationary_slope = _linear_slope(pivot_times, stationary_si_mean)
    excess_slope = _linear_slope(pivot_times, excess_si_mean) if excess_si_mean.size else np.nan
    return {
        'stationary_si_matrix': stationary_si_matrix,
        'stationary_si_mean': stationary_si_mean,
        'stationary_slope': stationary_slope,
        'excess_si_mean': excess_si_mean,
        'excess_slope': excess_slope,
        'summary': {
            'stationary_slope': stationary_slope,
            'excess_slope': excess_slope,
            'n_windows': int(stationary_si_matrix.shape[0]),
            'n_neurons': int(stationary_si_matrix.shape[1]) if stationary_si_matrix.ndim == 2 else 0,
        },
    }


def occupancy_matched_windowed_si(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    *,
    observed_window_result=None,
    n_repeats=100,
    random_state=0,
    match_mode='common_min',
    n_bins_x=15,
    n_bins_y=15,
    dt=0.04,
    time_window=0,
    use_peak=False,
    filter_ratemap=False,
    sigma=1.0,
    min_time_per_bin_sec=1.0,
    eps=1e-12,
    min_spikes_per_min=None,
    rate_source='spike',
    prior_mode='empirical',
    min_time_policy='fixed',
    support_mode='window',
    common_support_frac=1.0,
    common_support_min_occ_sec=None,
    common_support_reference='all_windows',
    maze_id=None,
    **pivot_kwargs,
):
    if match_mode != 'common_min':
        raise NotImplementedError("Only match_mode='common_min' is implemented in this pass.")
    if observed_window_result is None:
        observed_window_result = pivot_window_spatial_information_fast(
            spike_counts,
            raw_counts,
            selected_df,
            positions,
            n_bins_x=n_bins_x,
            n_bins_y=n_bins_y,
            dt=dt,
            time_window=time_window,
            use_peak=use_peak,
            filter_ratemap=filter_ratemap,
            sigma=sigma,
            min_time_per_bin_sec=min_time_per_bin_sec,
            eps=eps,
            min_spikes_per_min=min_spikes_per_min,
            rate_source=rate_source,
            prior_mode=prior_mode,
            min_time_policy=min_time_policy,
            support_mode=support_mode,
            common_support_frac=common_support_frac,
            common_support_min_occ_sec=common_support_min_occ_sec,
            common_support_reference=common_support_reference,
            maze_id=maze_id,
            **pivot_kwargs,
        )
    kept_neuron_ids = np.asarray(observed_window_result.get('kept_neuron_ids', np.array([], dtype=int)), dtype=int)
    bounds = list(observed_window_result.get('pivot_bounds', []))
    pivot_times = np.asarray(observed_window_result.get('pivot_times', np.array([])), dtype=float)
    observed_mean = np.asarray(observed_window_result.get('si_mean', np.array([])), dtype=float)
    observed_slope = _linear_slope(pivot_times, observed_mean)
    n_windows = int(len(bounds))
    if kept_neuron_ids.size == 0 or n_windows == 0 or int(n_repeats) <= 0:
        return {
            'matched_si_mean_matrix': np.zeros((max(int(n_repeats), 0), n_windows), dtype=float),
            'matched_mean_trajectory': np.array([], dtype=float),
            'matched_slopes': np.array([], dtype=float),
            'matched_bins': np.array([], dtype=int),
            'target_frame_counts_per_bin': np.array([], dtype=int),
            'summary': {
                'match_mode': str(match_mode),
                'n_repeats': int(max(n_repeats, 0)),
                'n_windows': n_windows,
                'n_neurons': int(kept_neuron_ids.size),
                'common_bin_count': 0,
                'matched_bin_count': 0,
                'matched_total_frames_per_window': 0,
                'matched_total_seconds_per_window': 0.0,
                'observed_slope': observed_slope,
                'matched_mean_slope': np.nan,
                'matched_std_slope': np.nan,
                'matched_slope_q05': np.nan,
                'matched_slope_q50': np.nan,
                'matched_slope_q95': np.nan,
                'matched_minus_observed_slope': np.nan,
                'same_sign_frac': np.nan,
            },
        }

    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[:, None]
    labels_all, _, _, _ = _spatial_labels_from_inputs(
        selected_df,
        pos,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
    )
    valid_matrix = np.asarray(
        observed_window_result.get('window_valid_mask_matrix', np.zeros((n_windows, int(n_bins_x * n_bins_y)), dtype=bool)),
        dtype=bool,
    )
    if valid_matrix.shape[0] != n_windows:
        raise ValueError('window_valid_mask_matrix rows must match pivot_bounds length.')
    common_mask = np.all(valid_matrix, axis=0) if valid_matrix.size else np.zeros(int(n_bins_x * n_bins_y), dtype=bool)
    common_support_mask = observed_window_result.get('meta', {}).get('common_support_mask', None)
    if common_support_mask is not None:
        common_mask &= np.asarray(common_support_mask, dtype=bool)
    common_bins = np.flatnonzero(common_mask).astype(int)
    dt_eff = float(dt) * float(time_window + 1)
    effective_min_time_per_bin_sec = float(
        observed_window_result.get('meta', {}).get('effective_min_time_per_bin_sec', min_time_per_bin_sec)
    )
    min_frames_per_bin = max(1, int(np.ceil(float(effective_min_time_per_bin_sec) / max(dt_eff, eps))))
    if common_bins.size == 0:
        return {
            'matched_si_mean_matrix': np.full((int(n_repeats), n_windows), np.nan, dtype=float),
            'matched_mean_trajectory': np.full(n_windows, np.nan, dtype=float),
            'matched_slopes': np.full(int(n_repeats), np.nan, dtype=float),
            'matched_bins': np.array([], dtype=int),
            'target_frame_counts_per_bin': np.array([], dtype=int),
            'summary': {
                'match_mode': str(match_mode),
                'n_repeats': int(n_repeats),
                'n_windows': n_windows,
                'n_neurons': int(kept_neuron_ids.size),
                'common_bin_count': 0,
                'matched_bin_count': 0,
                'matched_total_frames_per_window': 0,
                'matched_total_seconds_per_window': 0.0,
                'observed_slope': observed_slope,
                'matched_mean_slope': np.nan,
                'matched_std_slope': np.nan,
                'matched_slope_q05': np.nan,
                'matched_slope_q50': np.nan,
                'matched_slope_q95': np.nan,
                'matched_minus_observed_slope': np.nan,
                'same_sign_frac': np.nan,
            },
        }

    window_bin_frames = []
    counts = np.zeros((n_windows, common_bins.size), dtype=int)
    for i, (s, e) in enumerate(bounds):
        frames = np.arange(int(s), int(e), dtype=int)
        labs = labels_all[s:e].astype(int)
        frame_sets = []
        for j, b in enumerate(common_bins.tolist()):
            idx = frames[labs == int(b)]
            frame_sets.append(idx)
            counts[i, j] = int(idx.size)
        window_bin_frames.append(frame_sets)
    target_frame_counts = counts.min(axis=0) if counts.size else np.array([], dtype=int)
    matched_keep = target_frame_counts >= int(min_frames_per_bin)
    matched_bins = common_bins[matched_keep]
    target_frame_counts = target_frame_counts[matched_keep].astype(int)
    matched_support_mask = np.zeros(int(n_bins_x * n_bins_y), dtype=bool)
    matched_support_mask[matched_bins] = True
    if matched_bins.size == 0:
        return {
            'matched_si_mean_matrix': np.full((int(n_repeats), n_windows), np.nan, dtype=float),
            'matched_mean_trajectory': np.full(n_windows, np.nan, dtype=float),
            'matched_slopes': np.full(int(n_repeats), np.nan, dtype=float),
            'matched_bins': np.array([], dtype=int),
            'target_frame_counts_per_bin': np.array([], dtype=int),
            'summary': {
                'match_mode': str(match_mode),
                'n_repeats': int(n_repeats),
                'n_windows': n_windows,
                'n_neurons': int(kept_neuron_ids.size),
                'common_bin_count': int(common_bins.size),
                'matched_bin_count': 0,
                'matched_total_frames_per_window': 0,
                'matched_total_seconds_per_window': 0.0,
                'observed_slope': observed_slope,
                'matched_mean_slope': np.nan,
                'matched_std_slope': np.nan,
                'matched_slope_q05': np.nan,
                'matched_slope_q50': np.nan,
                'matched_slope_q95': np.nan,
                'matched_minus_observed_slope': np.nan,
                'same_sign_frac': np.nan,
            },
        }
    matched_window_bin_frames = [
        [frame_sets[j] for j in np.flatnonzero(matched_keep).tolist()]
        for frame_sets in window_bin_frames
    ]

    Xs = np.asarray(spike_counts, dtype=float)
    Xr = np.asarray(raw_counts, dtype=float)
    tuner = compute_tuning_and_prior_peak if use_peak else compute_tuning_and_prior
    rng = np.random.default_rng(random_state)
    matched_si_mean_matrix = np.full((int(n_repeats), n_windows), np.nan, dtype=float)
    matched_slopes = np.full(int(n_repeats), np.nan, dtype=float)
    for rep in range(int(n_repeats)):
        si_rows = []
        for frame_sets in matched_window_bin_frames:
            sampled = []
            for idx, n_take in zip(frame_sets, target_frame_counts.tolist()):
                if idx.size == 0 or int(n_take) <= 0:
                    continue
                if idx.size == int(n_take):
                    choose = idx
                else:
                    choose = rng.choice(idx, size=int(n_take), replace=False)
                sampled.append(np.asarray(choose, dtype=int))
            if not sampled:
                si_rows.append(np.full(kept_neuron_ids.size, np.nan, dtype=float))
                continue
            sampled_frames = np.sort(np.concatenate(sampled).astype(int))
            labs_sub = labels_all[sampled_frames].astype(int)
            if time_window == 0:
                Q = Xs[sampled_frames][:, kept_neuron_ids]
            else:
                Q = build_count_matrix(
                    Xs,
                    Xr,
                    selected_df,
                    frames=sampled_frames,
                    time_window=time_window,
                    neuron_ids=kept_neuron_ids,
                ).T
            lam, prior = tuner(
                Q,
                labs_sub,
                None,
                n_bins=int(n_bins_x * n_bins_y),
                dt=dt_eff,
                filter_ratemap=filter_ratemap,
                sigma=sigma,
                min_time=effective_min_time_per_bin_sec,
                eps=eps,
                decode_head='none',
            )
            if prior_mode == 'uniform':
                prior_use = np.zeros_like(prior)
                if matched_bins.size:
                    prior_use[matched_bins] = 1.0 / float(matched_bins.size)
            else:
                prior_use = np.where(matched_support_mask, prior, 0.0)
                psum = float(prior_use.sum())
                if psum > 0:
                    prior_use = prior_use / psum
            si_rows.append(_skaggs_info(lam, prior_use, eps=eps))
        if si_rows:
            si_mat = np.asarray(si_rows, dtype=float)
            finite_counts = np.sum(np.isfinite(si_mat), axis=1)
            si_mean = np.full(si_mat.shape[0], np.nan, dtype=float)
            good = finite_counts > 0
            if np.any(good):
                si_mean[good] = np.nansum(si_mat[good], axis=1) / finite_counts[good]
            matched_si_mean_matrix[rep, :si_mean.shape[0]] = si_mean
            matched_slopes[rep] = _linear_slope(pivot_times, si_mean)

    finite_repeat_counts = np.sum(np.isfinite(matched_si_mean_matrix), axis=0)
    matched_mean_trajectory = np.full(n_windows, np.nan, dtype=float)
    good_repeat_cols = finite_repeat_counts > 0
    if np.any(good_repeat_cols):
        matched_mean_trajectory[good_repeat_cols] = (
            np.nansum(matched_si_mean_matrix[:, good_repeat_cols], axis=0) / finite_repeat_counts[good_repeat_cols]
        )
    finite_slopes = np.isfinite(matched_slopes)
    same_sign_frac = np.nan
    if np.isfinite(observed_slope) and np.any(finite_slopes):
        tol = 1e-12
        same_sign_frac = float(
            np.mean(
                [
                    (abs(float(observed_slope)) <= tol)
                    or (abs(float(v)) <= tol)
                    or (np.sign(float(observed_slope)) == np.sign(float(v)))
                    for v in matched_slopes[finite_slopes]
                ]
            )
        )
    matched_mean_slope = float(np.nanmean(matched_slopes)) if np.any(finite_slopes) else np.nan
    matched_std_slope = float(np.nanstd(matched_slopes)) if np.any(finite_slopes) else np.nan
    matched_q = (
        np.nanpercentile(matched_slopes[finite_slopes], [5.0, 50.0, 95.0])
        if np.any(finite_slopes)
        else np.array([np.nan, np.nan, np.nan], dtype=float)
    )
    return {
        'matched_si_mean_matrix': matched_si_mean_matrix,
        'matched_mean_trajectory': matched_mean_trajectory,
        'matched_slopes': matched_slopes,
        'matched_bins': matched_bins,
        'target_frame_counts_per_bin': target_frame_counts,
        'summary': {
            'match_mode': str(match_mode),
            'n_repeats': int(n_repeats),
            'n_windows': n_windows,
            'n_neurons': int(kept_neuron_ids.size),
            'common_bin_count': int(common_bins.size),
            'matched_bin_count': int(matched_bins.size),
            'matched_total_frames_per_window': int(target_frame_counts.sum()),
            'matched_total_seconds_per_window': float(target_frame_counts.sum() * dt_eff),
            'min_target_frames_per_bin': int(target_frame_counts.min()) if target_frame_counts.size else 0,
            'median_target_frames_per_bin': float(np.median(target_frame_counts)) if target_frame_counts.size else np.nan,
            'observed_slope': observed_slope,
            'matched_mean_slope': matched_mean_slope,
            'matched_std_slope': matched_std_slope,
            'matched_slope_q05': float(matched_q[0]),
            'matched_slope_q50': float(matched_q[1]),
            'matched_slope_q95': float(matched_q[2]),
            'matched_minus_observed_slope': (
                float(matched_mean_slope - observed_slope)
                if np.isfinite(matched_mean_slope) and np.isfinite(observed_slope)
                else np.nan
            ),
            'same_sign_frac': same_sign_frac,
        },
    }


def circular_shift_windowed_si_null(
    spike_counts,
    raw_counts,
    selected_df,
    positions,
    *,
    n_repeats=200,
    random_state=0,
    min_shift_frac=0.1,
    **pivot_kwargs,
):
    Xs = np.asarray(spike_counts, dtype=float)
    Xr = np.asarray(raw_counts, dtype=float)
    if Xs.ndim != 2 or Xr.ndim != 2:
        raise ValueError('spike_counts and raw_counts must be 2D')
    if Xs.shape != Xr.shape:
        raise ValueError('spike_counts and raw_counts must have the same shape')
    obs = pivot_window_spatial_information_fast(
        Xs,
        Xr,
        selected_df,
        positions,
        **pivot_kwargs,
    )
    obs_slope = _linear_slope(obs.get('pivot_times', np.array([])), obs.get('si_mean', np.array([])))
    T, N = Xs.shape
    if T <= 1 or N == 0 or int(n_repeats) <= 0:
        return {
            'observed': obs,
            'observed_slope': obs_slope,
            'null_slopes': np.array([], dtype=float),
            'summary': {'p_ge_obs': np.nan, 'null_mean': np.nan, 'null_std': np.nan, 'n_repeats': int(max(n_repeats, 0))},
        }
    rng = np.random.default_rng(random_state)
    min_shift = max(1, int(np.floor(float(min_shift_frac) * float(T))))
    if min_shift >= T:
        min_shift = 1
    null_slopes = np.full(int(n_repeats), np.nan, dtype=float)
    for i in range(int(n_repeats)):
        shifts = rng.integers(min_shift, T, size=N, endpoint=False)
        sp_shift = _circular_shift_columns(Xs, shifts)
        raw_shift = _circular_shift_columns(Xr, shifts)
        shuf = pivot_window_spatial_information_fast(
            sp_shift,
            raw_shift,
            selected_df,
            positions,
            **pivot_kwargs,
        )
        null_slopes[i] = _linear_slope(shuf.get('pivot_times', np.array([])), shuf.get('si_mean', np.array([])))
    finite = np.isfinite(null_slopes)
    p_ge_obs = np.nan
    if np.isfinite(obs_slope) and np.any(finite):
        p_ge_obs = float((1.0 + np.sum(null_slopes[finite] >= obs_slope)) / (1.0 + np.sum(finite)))
    return {
        'observed': obs,
        'observed_slope': obs_slope,
        'null_slopes': null_slopes,
        'summary': {
            'p_ge_obs': p_ge_obs,
            'null_mean': float(np.nanmean(null_slopes)) if np.any(finite) else np.nan,
            'null_std': float(np.nanstd(null_slopes)) if np.any(finite) else np.nan,
            'n_repeats': int(n_repeats),
        },
    }
