#!/usr/bin/env python3
"""Render Main Figure 4f from the exact ED10a K=166 pair-level estimator."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster


FIGURE4_ORANGE = "#e29957"
BOOTSTRAP_SEED = 202608041


def significance_star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def fit_statistics(rows: pd.DataFrame) -> dict[str, Any]:
    """Return pooled Pearson r and nominal OLS inference with standard errors clustered by animal."""

    required = {
        "animal_id",
        "first_long_bout_time_s",
        "d1_reward_proxy_error",
        "neuron_target_n",
        "n_target_frames",
        "n_technical_draws",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise AssertionError(f"Main Figure 4f source is missing columns: {missing}")
    if len(rows) != 30 or rows["animal_id"].astype(str).nunique() != 14:
        raise AssertionError("Main Figure 4f requires 30 session pairs from 14 mice")
    if not rows["neuron_target_n"].eq(166).all():
        raise AssertionError("Main Figure 4f is not using the ED10 K=166 support")
    if not rows["n_target_frames"].eq(2_000).all():
        raise AssertionError("Main Figure 4f requires 2,000 target frames per pair")
    if not rows["n_technical_draws"].eq(50).all():
        raise AssertionError("Main Figure 4f requires 50 technical draws per pair")

    behavior = pd.to_numeric(
        rows["first_long_bout_time_s"], errors="raise"
    ).to_numpy(float)
    neural = pd.to_numeric(
        rows["d1_reward_proxy_error"], errors="raise"
    ).to_numpy(float)
    behavior_z = (behavior - behavior.mean()) / behavior.std(ddof=1)
    neural_z = (neural - neural.mean()) / neural.std(ddof=1)
    # Match ED10a's inferential direction exactly: behavior ~ neural measure.
    # The display keeps behavior on x and the neural measure on y, while Pearson
    # r and the standardized slope are symmetric point estimates.
    model = sm.OLS(
        behavior_z, sm.add_constant(neural_z, has_constant="add")
    ).fit()
    groups = rows["animal_id"].astype(str).to_numpy()
    covariance = cov_cluster(model, groups, use_correction=True)
    slope = float(model.params[1])
    se = float(np.sqrt(covariance[1, 1]))
    df = int(np.unique(groups).size - 1)
    t_value = slope / se
    p_value = float(2.0 * stats.t.sf(abs(t_value), df))
    critical = float(stats.t.ppf(0.975, df))
    return {
        "pearson_r": float(stats.pearsonr(behavior, neural).statistic),
        "standardized_beta": slope,
        "cluster_se": se,
        "cluster_t": t_value,
        "cluster_df": df,
        "cluster_p": p_value,
        "ci95_low": slope - critical * se,
        "ci95_high": slope + critical * se,
        "mean_reward_decoding_error": float(neural.mean()),
        "n_pairs": int(len(rows)),
        "n_animals": int(np.unique(groups).size),
    }


def cluster_bootstrap_band(
    rows: pd.DataFrame, xfit: np.ndarray, *, seed: int = BOOTSTRAP_SEED
) -> tuple[np.ndarray, np.ndarray]:
    x = rows["first_long_bout_time_s"].to_numpy(dtype=float)
    y = rows["d1_reward_proxy_error"].to_numpy(dtype=float)
    animal = rows["animal_id"].astype(str).to_numpy()
    mice = np.asarray(sorted(set(animal)))
    rng = np.random.default_rng(seed)
    fitted_lines: list[np.ndarray] = []
    for _ in range(5_000):
        sampled = rng.choice(mice, size=len(mice), replace=True)
        indices = np.concatenate([np.flatnonzero(animal == mouse) for mouse in sampled])
        if np.unique(x[indices]).size < 2:
            continue
        fit = stats.linregress(x[indices], y[indices])
        fitted_lines.append(fit.intercept + fit.slope * xfit)
    array = np.asarray(fitted_lines)
    return np.quantile(array, 0.025, axis=0), np.quantile(array, 0.975, axis=0)


def render_panel(
    rows: pd.DataFrame,
    *,
    size: tuple[float, float],
    shared_limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> plt.Figure:
    """Draw the accepted Main Figure 4f grammar with the ED10 K=166 fit."""

    result = fit_statistics(rows)
    x = rows["first_long_bout_time_s"].to_numpy(dtype=float)
    y = rows["d1_reward_proxy_error"].to_numpy(dtype=float)
    if shared_limits is None:
        shared_limits = ((0.0, float(x.max() * 1.08)), (0.0, float(y.max() * 1.10)))
    xfit = np.linspace(shared_limits[0][0], shared_limits[0][1], 200)
    fit = stats.linregress(x, y)
    low, high = cluster_bootstrap_band(rows, xfit)

    fig, ax = plt.subplots(figsize=size)
    ax.fill_between(xfit, low, high, color=FIGURE4_ORANGE, alpha=0.16, lw=0)
    ax.plot(xfit, fit.intercept + fit.slope * xfit, color=FIGURE4_ORANGE, lw=1.1)
    ax.scatter(x, y, s=15, color=FIGURE4_ORANGE, alpha=0.75, edgecolors="none", zorder=3)
    ax.text(
        0.04,
        0.95,
        significance_star(float(result["cluster_p"])),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.75,
        color=FIGURE4_ORANGE,
        fontweight="bold",
    )
    ax.text(
        0.96,
        0.08,
        f"r={float(result['pearson_r']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
    )
    ax.set_xlabel("First long bout time (s)")
    ax.set_ylabel("Reward decoding error (bins)", labelpad=1.0)
    ax.set_xlim(*shared_limits[0])
    ax.set_ylim(*shared_limits[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.2)
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.20, top=0.975)
    return fig
