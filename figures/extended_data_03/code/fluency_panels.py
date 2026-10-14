"""Condition definitions and mouse-clustered confidence intervals for ED3.

The assembly calls one_group_estimates for displayed dwell means and confidence
intervals. Included statistics tables supply significance annotations; the
supported statistics launcher recomputes those tests.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np

import pandas as pd

from scipy import stats

CONDITION_ORDER = ("latent", "return")

CONDITION_LABEL = {"latent": "Latent day2", "return": "Return"}

COLOR = {
    "latent": "#91877A",
    "return": "#C53B38",
    "axis": "#222222",
    "grid": "#D9D9D9",
    "neutral": "#6E7478",
}

CLAIMS = (
    {
        "claim_id": "ed4_broad_pre_reward_explore",
        "stratum": "pre_explore_all_junctions",
        "title": "Broad pre-reward fluency",
        "ylabel": "Mean junction dwell (s)",
        "stem": "ed04a_broad_pre_reward_fluency_current",
        "footer": "exploration mode; all internal junctions before first reward",
    },
    {
        "claim_id": "ed4_broad_post_reward_explore",
        "stratum": "all_post_explore_all_junctions",
        "title": "Broad post-reward fluency",
        "ylabel": "Mean junction dwell (s)",
        "stem": "ed04b_broad_post_reward_fluency_current",
        "footer": "exploration mode; all internal junctions after first reward",
    },
    {
        "claim_id": "ed4_pre_reward_direct_chain_specificity_check",
        "stratum": "pre_explore_direct_minus_other",
        "title": "Pre-reward path specificity",
        "ylabel": "Reward-path minus other dwell (s)",
        "stem": "ed04c_pre_reward_path_specificity_current",
        "footer": "exploration mode; reward-path minus other-junction dwell before first reward",
    },
    {
        "claim_id": "ed4_reward_approach_after_reward",
        "stratum": "all_post_reward_approach_direct_chain",
        "title": "Reward-path approaches after reward",
        "ylabel": "Mean reward-path junction dwell (s)",
        "stem": "ed04d_post_port_direct_approach_current",
        "footer": "reward-approach mode; reward-path junctions after first reward",
    },
)

def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def cluster_fit(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """CR1 sandwich fit with a t(G-1) reference distribution."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    if x.ndim == 1:
        x = x[:, None]
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    finite = (
        np.isfinite(y)
        & np.all(np.isfinite(x), axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    y, x, clusters, weights = y[finite], x[finite], clusters[finite], weights[finite]
    n, k = x.shape
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if n <= k or g < 3:
        raise ValueError(f"Cluster fit needs n>k and >=3 mice; got n={n}, k={k}, G={g}")
    xtwx_inv = np.linalg.pinv(x.T @ (weights[:, None] * x))
    beta = xtwx_inv @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = scale * xtwx_inv @ meat @ xtwx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = beta / se
    p_two = 2.0 * stats.t.sf(np.abs(t_value), df=g - 1)
    return {
        "n_rows": int(n),
        "n_mice": int(g),
        "df": int(g - 1),
        "beta": beta,
        "se": se,
        "t": t_value,
        "p_two_sided": p_two,
        "covariance": covariance,
    }

def one_group_estimates(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        subset = rows.loc[rows["condition"].eq(condition)].copy()
        fit = cluster_fit(
            subset["mean_decision_time_s"].to_numpy(dtype=float),
            np.ones((len(subset), 1), dtype=float),
            subset["animal"].astype(str).to_numpy(),
        )
        mean = float(fit["beta"][0])
        se = float(fit["se"][0])
        critical = float(stats.t.ppf(0.975, fit["df"]))
        records.append(
            {
                "condition": condition,
                "mean": mean,
                "ci95_low": mean - critical * se,
                "ci95_high": mean + critical * se,
            }
        )
    return pd.DataFrame(records)
