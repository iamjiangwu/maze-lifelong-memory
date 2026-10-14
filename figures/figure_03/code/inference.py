"""Animal-clustered tests from cycle and recording-pair observations."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Sequence
from types import SimpleNamespace
import math
import itertools
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from release_rendering.statistics import cr1_ols, linear_contrast, benjamini_hochberg, sem
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SIMILARITY_METRICS = (
    "place_map_correlation",
    "level_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)

def clustered_slope_test(
    frame: pd.DataFrame,
    *,
    x_column: str,
    value_column: str,
) -> dict[str, object]:
    """Fit value on experience with animal-clustered CR1 covariance."""

    use = frame[[x_column, value_column, "animal_id"]].dropna()
    if len(use) <= 2 or use["animal_id"].nunique() < 2:
        return {
            "estimate": np.nan,
            "cluster_se": np.nan,
            "cluster_t": np.nan,
            "cluster_df": np.nan,
            "p_two_sided": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "n_rows": int(len(use)),
            "n_animals": int(use["animal_id"].nunique()),
        }
    x = use[x_column].to_numpy(dtype=float)
    y = use[value_column].to_numpy(dtype=float)
    fit = cr1_ols(y, np.column_stack((np.ones(len(use)), x)), use["animal_id"].tolist())
    contrast = linear_contrast(fit, [0.0, 1.0])
    pearson = stats.pearsonr(x, y)
    return {
        "estimate": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "p_two_sided": contrast.p_two_sided,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n_rows": fit.n_observations,
        "n_animals": fit.n_clusters,
    }

def clustered_intercept_test(frame: pd.DataFrame) -> dict[str, object]:
    """Test the mean pair slope with animal-clustered CR1 covariance."""

    use = frame[["slope_per_hour_or_ktile", "animal_id"]].dropna()
    if len(use) <= 1 or use["animal_id"].nunique() < 2:
        return {
            "mean_slope": np.nan,
            "cluster_se": np.nan,
            "cluster_t": np.nan,
            "cluster_df": np.nan,
            "p_two_sided": np.nan,
            "n_rows": int(len(use)),
            "n_animals": int(use["animal_id"].nunique()),
        }
    fit = cr1_ols(
        use["slope_per_hour_or_ktile"].to_numpy(dtype=float),
        np.ones((len(use), 1), dtype=float),
        use["animal_id"].tolist(),
    )
    contrast = linear_contrast(fit, [1.0])
    return {
        "mean_slope": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "p_two_sided": contrast.p_two_sided,
        "n_rows": fit.n_observations,
        "n_animals": fit.n_clusters,
    }

def experience_statistics(points, x_column):
    result=pd.DataFrame([dict(metric=metric, **clustered_slope_test(points.loc[points.metric.eq(metric)],x_column=x_column,value_column='value')) for metric in SIMILARITY_METRICS])
    result['p_bh4']=benjamini_hochberg(result.p_two_sided.to_numpy(float))
    return result

def convergence_statistics(slopes):
    records=[]
    for (metric,axis,size),group in slopes.groupby(['metric','window_axis','window_size'],sort=False):
        records.append(dict(metric=metric,window_axis=axis,window_size=size,**clustered_intercept_test(group)))
    result=pd.DataFrame(records)
    if len(result)!=12:raise ValueError('Convergence inference requires the complete 12-test family')
    result['p_bh12']=benjamini_hochberg(result.p_two_sided.to_numpy(float))
    return result
