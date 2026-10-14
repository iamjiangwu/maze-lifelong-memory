"""Inference and display summaries calculated from the supplied observations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numpy.typing import NDArray
from release_rendering.statistics import benjamini_hochberg, cr1_ols, linear_contrast
FloatArray = NDArray[np.float64]

FloatArray = NDArray[np.float64]


MINIMUM_CORRELATION_SUPPORT = 3


def pearson_between_vectors(left: FloatArray, right: FloatArray) -> float:
    """Pearson correlation between two finite vectors with support >= 3."""

    x = np.asarray(left, dtype=float).ravel()
    y = np.asarray(right, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < MINIMUM_CORRELATION_SUPPORT or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


_scalar_pearson = pearson_between_vectors

FloatArray = NDArray[np.float64]


def _binned_curve(
    x: FloatArray, y: FloatArray, limits: tuple[float, float]
) -> pd.DataFrame:
    """Return the 13-bin descriptive line used over dense recurrence points."""

    edges = np.linspace(limits[0], limits[1], 14)
    rows: list[dict[str, float | int]] = []
    for index in range(13):
        use = (
            (x >= edges[index]) & (x <= edges[index + 1])
            if index == 12
            else (x >= edges[index]) & (x < edges[index + 1])
        )
        if np.count_nonzero(use) >= 3:
            rows.append(
                {
                    "bin_index": index,
                    "previous_correlation": float(
                        (edges[index] + edges[index + 1]) / 2
                    ),
                    "next_correlation": float(np.mean(y[use])),
                    "n_points": int(np.count_nonzero(use)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _cluster_confidence_band(
    x: FloatArray,
    covariance: FloatArray,
    *,
    degrees_of_freedom: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a clustered 95% confidence band for an intercept+slope fit."""

    design = np.column_stack((np.ones(x.size), x))
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    half_width = critical * np.sqrt(np.maximum(variance, 0.0))
    return x, -half_width, half_width


def summarize_recurrence_points(points: pd.DataFrame) -> pd.DataFrame:
    """Fit animal-clustered recurrence slopes and apply the four-test BH correction."""

    rows = []
    for metric in ("place", "pv", "level", "rate"):
        group = points.loc[points["metric"].eq(metric)].copy()
        finite = np.isfinite(group["previous_correlation"]) & np.isfinite(
            group["next_correlation"]
        )
        group = group.loc[finite]
        x = group["previous_correlation"].to_numpy(float)
        y = group["next_correlation"].to_numpy(float)
        row: dict[str, object] = {
            "record_type": "statistic",
            "metric": metric,
            "n_rows": int(len(group)),
            "n_animals": int(group["animal_id"].nunique()),
            "pearson_r": _scalar_pearson(x, y),
            "intercept": np.nan,
            "slope": np.nan,
            "slope_se": np.nan,
            "p_two_sided": np.nan,
            "df": np.nan,
        }
        if int(group["animal_id"].nunique()) >= 2 and len(group) > 2:
            fit = cr1_ols(
                y,
                np.column_stack((np.ones(len(group)), x)),
                group["animal_id"].astype(str).tolist(),
            )
            slope = linear_contrast(fit, [0.0, 1.0])
            row.update(
                {
                    "intercept": float(fit.beta[0]),
                    "slope": slope.estimate,
                    "slope_se": slope.standard_error,
                    "p_two_sided": slope.p_two_sided,
                    "df": slope.degrees_of_freedom,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["p_bh4"] = benjamini_hochberg(frame["p_two_sided"].to_numpy(float))
    return frame


def prepare_recurrence_display(
    points: pd.DataFrame, statistic: pd.Series, metric: str
) -> pd.DataFrame:
    """Calculate the descriptive binned curve or clustered fit band shown."""

    x = points["previous_correlation"].to_numpy(float)
    y = points["next_correlation"].to_numpy(float)
    limits = {"place": (-0.50, 1.0), "pv": (-0.05, 0.85), "rate": (-0.10, 0.80)}[metric]
    if metric in {"place", "pv"}:
        binned = _binned_curve(x, y, limits)
        return binned.assign(
            record_type="binned_mean",
            metric=metric,
            animal_id="",
            chain_key="",
            point_id="",
        )
    if not np.isfinite(float(statistic["slope"])):
        return pd.DataFrame()
    fit = cr1_ols(
        y,
        np.column_stack((np.ones(x.size), x)),
        points["animal_id"].astype(str).tolist(),
    )
    x_fit = np.linspace(limits[0], limits[1], 120)
    center = fit.beta[0] + fit.beta[1] * x_fit
    _, lower, upper = _cluster_confidence_band(
        x_fit, fit.covariance, degrees_of_freedom=fit.degrees_of_freedom
    )
    return pd.DataFrame(
        {
            "record_type": "fit_line",
            "metric": metric,
            "previous_correlation": x_fit,
            "next_correlation": center,
            "ci95_low": center + lower,
            "ci95_high": center + upper,
        }
    )

