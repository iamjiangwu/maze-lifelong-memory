"""Shared utilities for linearized place-field stability analysis.

Functions and constants used across the linearized-pf-stability scripts
and notebook.
"""

import os
import re
import glob
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, binomtest
from collections import defaultdict

# ---- Constants ----

EXCLUDED_SESSIONS = frozenset({
    "434-6day1", "434-6day2", "488-9day1", "488-9day2",
    "579-6day1", "579-6day2", "579-9day1", "579-9day2",
    "418-12day1", "418-12day2", "451-12day1", "451-12day2",
    "862-9day1", "862-9day2",
})

LEAF_NODES = frozenset({
    47, 48, 63, 64, 73, 74, 79, 80, 93, 94, 95, 96,
    109, 110, 125, 126,
})

N_BINS = 22
WP_BINS = 2
DT = 0.04
MIN_TIME_PER_BIN = 0.2
SMOOTH_SIGMA = 1.0
STABLE_THR = 0.3
MIN_RUNS = 5
STOP_GAP = 10
RNG_SEED = 42
N_PERM = 10_000
GRID_SIZE = 15  # 15x15 maze grid


def get_data_root():
    return os.path.join(os.environ.get("BEHAV_WD", r"D:\combineddata"),
                        "linear_pf_extended")


def get_fig_dir(script_file):
    return os.path.join(os.path.dirname(script_file), "..", "docs")


# ---- Session metadata helpers ----

def get_animal(sid):
    """Extract numeric animal ID from session name."""
    m = re.match(r"^(\d+)", sid)
    return m.group(1) if m else sid


def get_timepoint_key(sid):
    """Sortable (month, sub_order) key for temporal ordering."""
    base = re.sub(r"day\d+$", "", sid)
    m = re.match(r"^(\d+)([a-zA-Z]*)(?:-(\d+))?$", base)
    if not m:
        return (0, 0)
    letter = m.group(2) or ""
    month_str = m.group(3)
    month = int(month_str) if month_str else 0
    sub_order = 1 if "b" in letter.lower() else 0
    return (month, sub_order)


def build_experience_map(run_counts_path, include_day2=False):
    """Build ordinal experience ranks for each session."""
    if os.path.exists(run_counts_path):
        with open(run_counts_path) as fh:
            all_sessions = list(json.load(fh).keys())
    else:
        return {}

    if not include_day2:
        all_sessions = [s for s in all_sessions if "day1" in s.lower()]

    animal_sessions = defaultdict(set)
    for s in all_sessions:
        animal_sessions[get_animal(s)].add(s)

    exp_map = {}
    for animal, sessions in animal_sessions.items():
        sorted_s = sorted(sessions, key=get_timepoint_key)
        tps = list(dict.fromkeys(get_timepoint_key(s) for s in sorted_s))
        tp_rank = {tp: i for i, tp in enumerate(tps)}
        for s in sorted_s:
            exp_map[s] = tp_rank[get_timepoint_key(s)]
    return exp_map


# ---- Place-field computation ----

def compute_1d_place_fields(spikes, positions_1d, n_bins=N_BINS, dt=DT,
                             min_time_per_bin=MIN_TIME_PER_BIN,
                             smooth_sigma=SMOOTH_SIGMA):
    """Compute 1D linearized place fields.

    Returns (rate_maps, occupancy_sec, bin_centers) or (None, None, None).
    """
    if spikes is None or positions_1d is None or len(positions_1d) == 0:
        return None, None, None
    positions_1d = np.asarray(positions_1d)
    spikes = np.asarray(spikes)
    n_neurons, n_time = spikes.shape
    if positions_1d.shape[0] != n_time:
        raise ValueError("positions_1d length must match spikes time axis")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(positions_1d, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    occupancy = np.bincount(bin_idx, minlength=n_bins).astype(float) * dt
    occ_safe = occupancy.copy()
    occ_safe[occ_safe < min_time_per_bin] = np.nan

    # Vectorized: build bin indicator matrix, compute all neurons at once
    from scipy.sparse import csr_matrix
    indicator = csr_matrix(
        (np.ones(len(bin_idx)), (bin_idx, np.arange(len(bin_idx)))),
        shape=(n_bins, len(bin_idx)))
    rate_maps = np.asarray(spikes @ indicator.T) / occ_safe[None, :]

    if smooth_sigma and smooth_sigma > 0:
        nan_any = np.isnan(rate_maps).any(axis=1)
        nan_all = np.isnan(rate_maps).all(axis=1)
        clean = ~nan_any
        smoothed = np.empty_like(rate_maps)
        if clean.any():
            smoothed[clean] = gaussian_filter1d(
                rate_maps[clean], sigma=smooth_sigma, axis=1, mode="nearest")
        partial = nan_any & ~nan_all
        for i in np.where(partial)[0]:
            row = rate_maps[i]
            nm = np.isnan(row)
            vi = np.where(~nm)[0]
            interp = np.interp(np.arange(n_bins), vi, row[vi])
            interp = gaussian_filter1d(interp, sigma=smooth_sigma,
                                       mode="nearest")
            interp[nm] = np.nan
            smoothed[i] = interp
        smoothed[nan_all] = rate_maps[nan_all]
        rate_maps = smoothed

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return rate_maps, occupancy, bin_centers


def peak_normalize(maps):
    """Row-wise peak normalization (peak = 1, NaN preserved)."""
    maps = np.asarray(maps, dtype=float)
    mx = np.nanmax(maps, axis=1)
    mx[mx <= 0] = np.nan
    return maps / mx[:, None]


def rowwise_corr(a, b):
    """Row-wise Pearson correlation with NaN masking (vectorized)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = np.full(a.shape[0], np.nan, dtype=float)

    # Fast path: rows with no NaN
    has_nan = np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1)
    clean = ~has_nan
    if clean.any():
        ac = a[clean] - np.nanmean(a[clean], axis=1, keepdims=True)
        bc = b[clean] - np.nanmean(b[clean], axis=1, keepdims=True)
        num = (ac * bc).sum(axis=1)
        denom = np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
        denom[denom == 0] = np.nan
        out[clean] = num / denom

    # Slow path: rows with NaN
    for i in np.where(has_nan)[0]:
        x, y = a[i], b[i]
        m = ~np.isnan(x) & ~np.isnan(y)
        if np.count_nonzero(m) < 5:
            continue
        out[i] = np.corrcoef(x[m], y[m])[0, 1]
    return out


# ---- Spatial analysis helpers ----

def near_wp(peak, wp_bins=WP_BINS, n_bins=N_BINS):
    """True if peak is in the last ``wp_bins`` bins. Accepts arrays."""
    return np.asarray(peak) >= (n_bins - wp_bins)


def wp_fraction(peaks, wp_bins=WP_BINS, n_bins=N_BINS, wp_end="last"):
    """Fraction of peaks near the water port."""
    if len(peaks) == 0:
        return np.nan
    peaks = np.asarray(peaks)
    if wp_end == "last":
        return np.sum(peaks >= (n_bins - wp_bins)) / len(peaks)
    else:
        return np.sum(peaks < wp_bins) / len(peaks)


def field_width_halfmax(rate_map):
    """Number of bins above half-max."""
    mx = np.nanmax(rate_map)
    if mx <= 0 or np.isnan(mx):
        return np.nan
    return np.sum(rate_map >= mx / 2)


def wilcoxon_one_sided(diffs):
    """One-sided Wilcoxon signed-rank test (H1: diffs > 0).

    Returns (statistic, p_one_sided) or (nan, nan) if too few observations.
    """
    nonzero = np.asarray(diffs)
    nonzero = nonzero[nonzero != 0]
    if len(nonzero) < 6:
        return np.nan, np.nan
    stat, p_one = wilcoxon(nonzero, alternative="greater")
    return stat, p_one


# ---- Shared data loaders ----

def load_run_counts(data_root=None):
    """Load session_run_counts.json (cached-friendly)."""
    if data_root is None:
        data_root = get_data_root()
    rc_path = os.path.join(data_root, "session_run_counts.json")
    if os.path.exists(rc_path):
        with open(rc_path) as fh:
            return json.load(fh)
    return {}


def load_session_peaks(data_root, run_label, run_counts=None, min_runs=MIN_RUNS):
    """Load all day1 session-level peak arrays for one run type.

    Returns (session_peaks_dict, detected_n_bins).
    """
    if run_counts is None:
        run_counts = load_run_counts(data_root)

    pf_files = sorted(glob.glob(
        os.path.join(data_root, f"*__{run_label}__pf_all.npy")))

    session_peaks = {}
    detected_n_bins = None

    for filepath in pf_files:
        bn = os.path.basename(filepath)
        sid = bn.split(f"__{run_label}__")[0]
        if "day1" not in sid.lower():
            continue
        if sid in EXCLUDED_SESSIONS:
            continue
        if run_counts.get(sid, {}).get(run_label, 0) < min_runs:
            continue
        pf = np.load(filepath).astype(float)
        if detected_n_bins is None:
            detected_n_bins = pf.shape[1]
        n_valid = np.sum(~np.isnan(pf), axis=1)
        valid = (n_valid >= pf.shape[1] // 2) & (np.nanmax(pf, axis=1) > 0)
        pf = pf[valid]
        if pf.size == 0:
            continue
        session_peaks[sid] = np.nanargmax(pf, axis=1)

    if detected_n_bins is None:
        detected_n_bins = N_BINS
    return session_peaks, detected_n_bins


def load_pair_data(data_root, run_label, run_counts=None, min_runs=MIN_RUNS,
                   stable_thr=STABLE_THR, include_place_fields=False,
                   pair_filter="day1_min_runs"):
    """Load pair-level matched neuron data for one run type.

    Returns list of dicts with keys: id1, id2, peaks, stable, all_peaks,
    stable_peaks, unstable_peaks, corrs, n_stable, n_matched.
    When ``include_place_fields`` is True, each row also includes peak-
    normalized matched matrices ``pf1`` and ``pf2`` for the two sessions.
    ``pair_filter`` controls which precomputed pair files are included:
    ``all_saved``, ``day1_day1``, ``day1_day1_not_excluded``, or
    ``day1_min_runs`` (the maintained stricter subset).
    Also returns detected_n_bins.
    """
    if run_counts is None:
        run_counts = load_run_counts(data_root)

    valid_pair_filters = {
        "all_saved",
        "day1_day1",
        "day1_day1_not_excluded",
        "day1_min_runs",
    }
    if pair_filter not in valid_pair_filters:
        raise ValueError(f"pair_filter must be one of {sorted(valid_pair_filters)}")

    corr_files = sorted(glob.glob(
        os.path.join(data_root, f"*__{run_label}__corrs.npy")))

    pair_info = []
    detected_n_bins = None

    for cf in corr_files:
        bn = os.path.basename(cf)
        prefix = bn.split(f"__{run_label}__")[0]
        m = re.match(r"^(.+?)_to_(.+?)$", prefix)
        if not m:
            continue
        id1, id2 = m.groups()
        is_day1_day1 = "day1" in id1.lower() and "day1" in id2.lower()
        has_excluded_session = id1 in EXCLUDED_SESSIONS or id2 in EXCLUDED_SESSIONS
        id1_runs = run_counts.get(id1, {}).get(run_label, 0) if run_counts else 0
        id2_runs = run_counts.get(id2, {}).get(run_label, 0) if run_counts else 0
        has_min_runs = id1_runs >= min_runs and id2_runs >= min_runs

        if pair_filter == "day1_day1" and not is_day1_day1:
            continue
        if pair_filter == "day1_day1_not_excluded" and (not is_day1_day1 or has_excluded_session):
            continue
        if pair_filter == "day1_min_runs" and (not is_day1_day1 or has_excluded_session or not has_min_runs):
            continue

        corrs = np.load(cf).astype(float)
        pf1_path = cf.replace("__corrs.npy", "__pf1_matched.npy")
        if not os.path.exists(pf1_path):
            continue
        pf1 = np.load(pf1_path).astype(float)
        pf2 = None
        if include_place_fields:
            pf2_path = cf.replace("__corrs.npy", "__pf2_matched.npy")
            if not os.path.exists(pf2_path):
                continue
            pf2 = np.load(pf2_path).astype(float)
        if detected_n_bins is None:
            detected_n_bins = pf1.shape[1]

        n_valid = np.sum(~np.isnan(pf1), axis=1)
        valid = ((n_valid >= pf1.shape[1] // 2) &
                 (np.nanmax(pf1, axis=1) > 0) &
                 ~np.isnan(corrs))
        pf1 = pf1[valid]
        if include_place_fields and pf2 is not None:
            pf2 = pf2[valid]
        corrs = corrs[valid]
        if len(corrs) == 0:
            continue

        peaks = np.nanargmax(pf1, axis=1)
        stable = corrs >= stable_thr

        pair_row = dict(
            id1=id1, id2=id2,
            peaks=peaks, stable=stable,
            all_peaks=peaks,
            all_corrs=corrs,
            stable_peaks=peaks[stable],
            unstable_peaks=peaks[~stable],
            n_stable=int(stable.sum()),
            n_unstable=int((~stable).sum()),
            n_matched=len(peaks),
            pair_filter=pair_filter,
        )
        if include_place_fields:
            pair_row.update(
                pf1=pf1,
                pf2=pf2,
            )
        pair_info.append(pair_row)

    if detected_n_bins is None:
        detected_n_bins = N_BINS
    return pair_info, detected_n_bins
