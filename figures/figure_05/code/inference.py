"""Inference and display summaries calculated from the supplied observations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numpy.typing import NDArray
from release_rendering.statistics import benjamini_hochberg, cr1_ols, linear_contrast
FloatArray = NDArray[np.float64]

def summarize_level_stability_vs_efficiency(
    rows: pd.DataFrame,
) -> dict[str, float | int]:
    """Regress later exploration on neural stability; report Pearson r and CR1 inference."""

    if len(rows) < 3 or rows["animal_id"].nunique() < 2:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
            "cluster_p": float("nan"),
        }
    x = rows["level_map_correlation"].to_numpy(dtype=float)
    y = rows["exploration_efficiency"].to_numpy(dtype=float)
    correlation = stats.pearsonr(x, y)
    ordinary = stats.linregress(x, y)
    design = np.column_stack((np.ones(len(rows), dtype=float), x))
    clustered = cr1_ols(y, design, rows["animal_id"].astype(str).tolist())
    slope = linear_contrast(clustered, [0.0, 1.0])
    return {
        "pearson_r": float(correlation.statistic),
        "pearson_p": float(correlation.pvalue),
        "slope": float(ordinary.slope),
        "intercept": float(ordinary.intercept),
        "cluster_p": slope.p_two_sided,
        "cluster_se": slope.standard_error,
        "cluster_t": slope.t_value,
        "cluster_df": slope.degrees_of_freedom,
        "cluster_ci95_low": slope.confidence_low,
        "cluster_ci95_high": slope.confidence_high,
        "r_squared": float(correlation.statistic) ** 2,
    }


def _animal_bootstrap_association_band(x, y, animals, xfit):
    """Figure4 convention: 5,000 whole-animal draws; percentile fitted-mean band."""
    groups = np.asarray(animals).astype(str)
    mice = np.asarray(sorted(set(groups)))
    rng = np.random.default_rng(202608041)
    fitted_lines = []
    for _ in range(5_000):
        sampled = rng.choice(mice, size=len(mice), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == mouse) for mouse in sampled])
        if np.unique(x[indices]).size < 2:
            continue
        fit = stats.linregress(x[indices], y[indices])
        fitted_lines.append(fit.intercept + fit.slope * xfit)
    return np.quantile(np.asarray(fitted_lines), (0.025, 0.975), axis=0)


def prepare_level_stability_vs_efficiency_display(
    values: pd.DataFrame, inference: dict[str, float | int]
) -> pd.DataFrame:
    """Calculate behavior-response OLS and a 5,000-animal-bootstrap 95% band."""

    x = values["level_map_correlation"].to_numpy(float)
    y = values["exploration_efficiency"].to_numpy(float)
    if len(values) < 3:
        return pd.DataFrame()
    x_fit = np.linspace(float(x.min()), float(x.max()), 100)
    y_fit = float(inference["intercept"]) + float(inference["slope"]) * x_fit
    low, high = _animal_bootstrap_association_band(
        x, y, values["animal_id"].astype(str).to_numpy(), x_fit
    )
    return pd.DataFrame(
        {
            "level_map_correlation": x_fit,
            "fitted_exploration_efficiency": y_fit,
            "ci95_low": low,
            "ci95_high": high,
        }
    )

