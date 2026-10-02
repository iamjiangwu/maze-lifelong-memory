import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

def sem(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return np.std(arr, ddof=1) / np.sqrt(arr.size)

def compute_lifetime_sparsity(rates, *, normalized=False):
    rates = np.asarray(rates, dtype=float)
    mean_rate = np.nanmean(rates, axis=0) if rates.ndim == 2 else np.nanmean(rates)
    mean_sq = np.nanmean(rates ** 2, axis=0) if rates.ndim == 2 else np.nanmean(rates ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = (mean_rate ** 2) / mean_sq
    if not normalized:
        return raw
    n = rates.shape[0] if rates.ndim > 0 else 1
    if n <= 1:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        return (1.0 - raw) / (1.0 - 1.0 / float(n))

def compute_gini_coefficient(rates):
    x = np.asarray(rates, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return np.nan
    x = np.sort(x)
    n = x.size
    denom = x.sum()
    if denom == 0:
        return 0.0
    return (2 * np.sum((np.arange(1, n + 1)) * x)) / (n * denom) - (n + 1) / n

def compute_effective_dimensionality(rate_matrix):
    x = np.asarray(rate_matrix, dtype=float)
    if x.ndim != 2:
        return np.nan
    if x.shape[0] == 0 or x.shape[1] < 2:
        return np.nan
    x = x - np.nanmean(x, axis=1, keepdims=True)
    x = np.nan_to_num(x, nan=0.0)
    cov = np.atleast_2d(np.cov(x))
    if not np.all(np.isfinite(cov)):
        return np.nan
    vals = np.linalg.eigvalsh(cov)
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.nan
    return (vals.sum() ** 2) / np.sum(vals ** 2)


def holm_correction(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    flat = pvals.ravel()
    reject = np.zeros(flat.shape, dtype=bool)
    p_adj = np.full(flat.shape, np.nan, dtype=float)
    keep = np.isfinite(flat)
    m = int(np.count_nonzero(keep))
    if m == 0:
        return reject.reshape(pvals.shape), p_adj.reshape(pvals.shape)
    p = flat[keep]
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m, dtype=float)
    for i, p in enumerate(p_sorted):
        adj_sorted[i] = (m - i) * p
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    p_adj_finite = np.empty(m, dtype=float)
    p_adj_finite[order] = adj_sorted
    p_adj[keep] = p_adj_finite
    reject[keep] = p_adj_finite <= float(alpha)
    return reject.reshape(pvals.shape), p_adj.reshape(pvals.shape)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(pvals.shape, np.nan, dtype=float)
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


def cliffs_delta(x, y):
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    cmp = a[:, None] - b[None, :]
    return float((np.sum(cmp > 0) - np.sum(cmp < 0)) / cmp.size)


def split_pre_post(values, pivot_index: int, *, pre_n: int | None = None, post_n: int | None = None):
    x = np.asarray(values, dtype=float).ravel()
    p = int(np.clip(pivot_index, 0, max(0, len(x))))
    pre = x[:p]
    post = x[p:]
    if pre_n is not None:
        n = int(max(0, pre_n))
        pre = pre[-n:] if n else pre[:0]
    if post_n is not None:
        post = post[: int(max(0, post_n))]
    return pre, post


def summarize_pre_post(values, pivot_index: int, *, pre_n: int | None = None, post_n: int | None = None):
    pre, post = split_pre_post(values, pivot_index, pre_n=pre_n, post_n=post_n)

    def _safe_mean(a):
        return float(np.nanmean(a)) if len(a) else np.nan

    def _safe_med(a):
        return float(np.nanmedian(a)) if len(a) else np.nan

    m0, m1 = _safe_mean(pre), _safe_mean(post)
    med0, med1 = _safe_med(pre), _safe_med(post)
    return {
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "mean_pre": m0,
        "mean_post": m1,
        "mean_delta": float(m1 - m0) if np.isfinite(m0) and np.isfinite(m1) else np.nan,
        "median_pre": med0,
        "median_post": med1,
        "median_delta": float(med1 - med0) if np.isfinite(med0) and np.isfinite(med1) else np.nan,
    }


def _sigmoid(x, L, x0, k, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return {"params": [np.nan, np.nan, np.nan, np.nan], "r2": np.nan, "yhat": np.full_like(y, np.nan)}
    p0 = [float(np.nanmax(y) - np.nanmin(y)), float(np.nanmedian(x)), 0.2, float(np.nanmin(y))]
    try:
        popt, _ = curve_fit(_sigmoid, x, y, p0=p0, maxfev=20000)
        yhat = _sigmoid(x, *popt)
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        return {"params": [float(v) for v in popt], "r2": float(r2), "yhat": yhat}
    except Exception:
        return {"params": [np.nan, np.nan, np.nan, np.nan], "r2": np.nan, "yhat": np.full_like(y, np.nan)}


def bin_event_rate(event_frames, *, start_frame: int, end_frame: int, bin_size_frames: int = 250):
    ef = np.asarray(event_frames, dtype=float).ravel()
    ef = ef[np.isfinite(ef)]
    s = int(start_frame)
    e = int(end_frame)
    b = max(1, int(bin_size_frames))
    if e <= s:
        return {"bin_start": np.zeros(0, dtype=int), "rate": np.zeros(0, dtype=float)}
    bins = np.arange(s, e, b, dtype=int)
    if bins.size == 0 or bins[0] != s:
        bins = np.insert(bins, 0, s)
    if bins[-1] != e:
        bins = np.append(bins, e)
    cnt, _ = np.histogram(ef, bins=bins)
    rates = cnt.astype(float) / np.diff(bins).astype(float)
    return {"bin_start": bins[:-1], "rate": rates}


def nansem(values):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def bootstrap_median_ci(values, *, n_boot=1000, alpha=0.05, seed=0):
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(max(1, n_boot)), dtype=float)
    for i in range(boots.size):
        boots[i] = np.nanmedian(rng.choice(x, size=x.size, replace=True))
    lo = float(np.nanquantile(boots, alpha / 2.0))
    hi = float(np.nanquantile(boots, 1.0 - alpha / 2.0))
    return {"median": float(np.nanmedian(x)), "ci_low": lo, "ci_high": hi}


def permutation_median_diff(x, y, *, n_perm=2000, seed=0):
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"obs": np.nan, "p": np.nan}
    obs = float(np.nanmedian(a) - np.nanmedian(b))
    pool = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(max(1, n_perm)), dtype=float)
    for i in range(diffs.size):
        perm = rng.permutation(pool.size)
        diffs[i] = float(np.nanmedian(pool[perm[:n_a]]) - np.nanmedian(pool[perm[n_a:]]))
    p = float((np.sum(np.abs(diffs) >= abs(obs)) + 1.0) / (diffs.size + 1.0))
    return {"obs": obs, "p": p}


def kendall_tau_with_perm(x, y, *, n_perm=2000, seed=0):
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2:
        return {"tau": np.nan, "p": np.nan, "perm_p": np.nan}
    tau, p = scipy_stats.kendalltau(a, b)
    if not np.isfinite(tau):
        return {"tau": float(tau), "p": float(p), "perm_p": np.nan}
    rng = np.random.default_rng(int(seed))
    null_abs = np.empty(int(max(1, n_perm)), dtype=float)
    for i in range(null_abs.size):
        t0, _ = scipy_stats.kendalltau(a, b[rng.permutation(b.size)])
        null_abs[i] = abs(float(t0)) if np.isfinite(t0) else np.nan
    valid_null = np.isfinite(null_abs)
    if not np.any(valid_null):
        perm_p = np.nan
    else:
        perm_p = float((np.sum(null_abs[valid_null] >= abs(float(tau))) + 1.0) / (np.sum(valid_null) + 1.0))
    return {"tau": float(tau), "p": float(p), "perm_p": perm_p}


def sig_label_from_p(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def kendall_tau(x, y):
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return np.nan, np.nan
    tau, p = scipy_stats.kendalltau(a, b)
    return float(tau), float(p)


def perm_p(stat_fn, x, y, *, n_perm=2000, seed=0):
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return np.nan
    obs = float(stat_fn(a, b))
    if not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(int(seed))
    ge = 0
    n_valid = 0
    for _ in range(int(max(1, n_perm))):
        stat = float(stat_fn(a, b[rng.permutation(b.size)]))
        if not np.isfinite(stat):
            continue
        n_valid += 1
        if abs(stat) >= abs(obs):
            ge += 1
    if n_valid == 0:
        return np.nan
    return float((ge + 1.0) / (n_valid + 1.0))
