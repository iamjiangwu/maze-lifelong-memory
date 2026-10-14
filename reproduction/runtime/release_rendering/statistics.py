"""Small, explicit statistical helpers used by the figure notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._validation import contains_boolean, finite_positive_float


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ClusterOLSResult:
    """OLS coefficients and animal-clustered CR1 covariance."""

    beta: FloatArray
    covariance: FloatArray
    residuals: FloatArray
    finite_mask: BoolArray
    n_observations: int
    n_coefficients: int
    n_clusters: int
    degrees_of_freedom: int


@dataclass(frozen=True)
class ContrastResult:
    """One linear contrast evaluated with Student t(G-1) inference."""

    estimate: float
    standard_error: float
    t_value: float
    degrees_of_freedom: int
    p_two_sided: float
    confidence_low: float
    confidence_high: float


def sem(values: ArrayLike) -> float:
    """Standard error of the mean over finite observations."""

    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1) / np.sqrt(array.size))


def mean_of_finite_values(values: Iterable[float] | FloatArray) -> float:
    """Return the mean after discarding NaN and infinite values."""

    array = np.asarray(
        values if isinstance(values, np.ndarray) else tuple(values),
        dtype=float,
    )
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def benjamini_hochberg(p_values: ArrayLike) -> FloatArray:
    """Benjamini-Hochberg adjusted p values over the supplied finite family."""

    if contains_boolean(p_values):
        raise ValueError("p_values must be numeric, not boolean")
    values = np.asarray(p_values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("Finite p values must lie in [0, 1]")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values.ravel()))
    if finite_indices.size == 0:
        return adjusted
    flat = values.ravel()
    order_local = np.argsort(flat[finite_indices])
    ranked_indices = finite_indices[order_local]
    ranked = flat[ranked_indices]
    correction = ranked * ranked.size / np.arange(1, ranked.size + 1)
    correction = np.minimum.accumulate(correction[::-1])[::-1]
    adjusted.ravel()[ranked_indices] = np.clip(correction, 0.0, 1.0)
    return adjusted


def _valid_cluster(value: object) -> bool:
    """Return whether one cluster label can identify a biological unit."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Real):
        numeric = float(value)
        if np.isnan(numeric):
            return False
        if not np.isfinite(numeric):
            raise ValueError("numeric cluster labels must be finite or missing")
        return True
    try:
        return not bool(value != value)
    except (TypeError, ValueError):
        return False


def _ordered_unique(values: Sequence[Hashable]) -> list[Hashable]:
    """Return unique cluster IDs without sorting or type coercion."""

    output: list[Hashable] = []
    seen: set[Hashable] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def cr1_ols(
    y: ArrayLike,
    design: ArrayLike,
    clusters: Sequence[Hashable],
) -> ClusterOLSResult:
    """Fit OLS with CR1 cluster covariance and G-1 reference degrees of freedom.

    Rows with non-finite outcomes/design values or missing cluster IDs are
    removed.  ``clusters`` must already contain the canonical biological unit;
    this function does not merge aliases or treat sessions as animals.
    """

    outcome = np.asarray(y, dtype=float).ravel()
    matrix = np.asarray(design, dtype=float)
    group = np.asarray(clusters, dtype=object).ravel()
    if contains_boolean(clusters):
        raise ValueError("clusters must not contain boolean labels")
    if matrix.ndim != 2:
        raise ValueError("design must be a two-dimensional matrix")
    if matrix.shape[0] != outcome.size or group.size != outcome.size:
        raise ValueError("y, design, and clusters must contain the same number of rows")
    valid_groups = np.asarray([_valid_cluster(value) for value in group], dtype=bool)
    finite = np.isfinite(outcome) & np.isfinite(matrix).all(axis=1) & valid_groups
    outcome_kept = outcome[finite]
    matrix_kept = matrix[finite]
    group_kept = group[finite]
    n_observations, n_coefficients = matrix_kept.shape
    unique_groups = _ordered_unique(group_kept.tolist())
    n_clusters = len(unique_groups)
    if n_observations <= n_coefficients:
        raise ValueError("CR1 OLS requires more observations than coefficients")
    if n_clusters < 2:
        raise ValueError("CR1 OLS requires at least two biological clusters")
    design_rank = int(np.linalg.matrix_rank(matrix_kept))
    if design_rank < n_coefficients:
        raise ValueError(
            "CR1 OLS design matrix must have full column rank; "
            f"found rank {design_rank} for {n_coefficients} coefficients"
        )

    bread = np.linalg.pinv(matrix_kept.T @ matrix_kept)
    beta = bread @ (matrix_kept.T @ outcome_kept)
    residuals = outcome_kept - matrix_kept @ beta
    meat = np.zeros((n_coefficients, n_coefficients), dtype=float)
    for cluster in unique_groups:
        use = group_kept == cluster
        score = matrix_kept[use].T @ residuals[use]
        meat += np.outer(score, score)
    correction = (n_clusters / (n_clusters - 1.0)) * (
        (n_observations - 1.0) / (n_observations - n_coefficients)
    )
    covariance = correction * bread @ meat @ bread
    return ClusterOLSResult(
        beta=np.asarray(beta, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        residuals=np.asarray(residuals, dtype=float),
        finite_mask=finite,
        n_observations=int(n_observations),
        n_coefficients=int(n_coefficients),
        n_clusters=int(n_clusters),
        degrees_of_freedom=int(n_clusters - 1),
    )


def linear_contrast(
    fit: ClusterOLSResult,
    contrast: ArrayLike,
    *,
    confidence_level: float = 0.95,
) -> ContrastResult:
    """Evaluate one coefficient contrast using the fit's CR1 covariance."""

    vector = np.asarray(contrast, dtype=float).ravel()
    if vector.shape != fit.beta.shape:
        raise ValueError("contrast must have one value per fitted coefficient")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    estimate = float(vector @ fit.beta)
    variance = float(vector @ fit.covariance @ vector)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    t_value = estimate / standard_error if standard_error > 0 else float("nan")
    p_value = (
        float(2.0 * stats.t.sf(abs(t_value), df=fit.degrees_of_freedom))
        if np.isfinite(t_value)
        else float("nan")
    )
    critical = float(
        stats.t.ppf(
            1.0 - (1.0 - confidence_level) / 2.0,
            df=fit.degrees_of_freedom,
        )
    )
    return ContrastResult(
        estimate=estimate,
        standard_error=standard_error,
        t_value=float(t_value),
        degrees_of_freedom=fit.degrees_of_freedom,
        p_two_sided=p_value,
        confidence_low=float(estimate - critical * standard_error),
        confidence_high=float(estimate + critical * standard_error),
    )


def significance_stars(
    p_value: float,
    *,
    thresholds: Sequence[tuple[float, str]],
    nonsignificant: str = "n.s.",
    missing: str = "",
) -> str:
    """Format significance using caller-supplied, visible thresholds.

    Thresholds must be strictly increasing, for example
    ``((0.001, '***'), (0.01, '**'), (0.05, '*'))``.  No correction family is
    selected here; pass the exact adjusted or unadjusted p value intended for
    the panel.
    """

    if isinstance(p_value, (bool, np.bool_)):
        raise ValueError("p_value must be numeric, not boolean")
    if not np.isfinite(p_value):
        return missing
    if p_value < 0 or p_value > 1:
        raise ValueError("p_value must lie in [0, 1]")
    limits = [
        finite_positive_float(limit, name="star threshold") for limit, _ in thresholds
    ]
    if any(limit > 1 for limit in limits):
        raise ValueError("Star thresholds must be in (0, 1]")
    if any(right <= left for left, right in zip(limits, limits[1:])):
        raise ValueError("Star thresholds must be strictly increasing")
    for (limit, label) in thresholds:
        if p_value < float(limit):
            return str(label)
    return nonsignificant


def exact_cluster_sign_flip_p(
    differences: FloatArray, clusters: Sequence[str]
) -> float:
    """Flip each animal's summed pair differences for an equal-pair mean test."""

    differences = np.asarray(differences, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    if not np.isfinite(differences).all():
        raise ValueError("Day-combination differences must be finite")
    values = np.asarray(
        [differences[clusters == animal].sum() for animal in np.unique(clusters)],
        dtype=float,
    )
    pattern_ids = np.arange(2**values.size, dtype=np.uint64)[:, None]
    bit_positions = np.arange(values.size, dtype=np.uint64)[None, :]
    signs = 2.0 * ((pattern_ids >> bit_positions) & 1).astype(float) - 1.0
    null_values = signs @ values
    observed = abs(float(values.sum()))
    return float(np.mean(np.abs(null_values) >= observed - 1e-15))


def mean_t_interval(values: Iterable[float]) -> tuple[float, float, float]:
    """Return mean and two-sided 95% Student-t limits."""

    array = np.asarray(tuple(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, np.nan, np.nan
    error = sem(array)
    critical = float(stats.t.ppf(0.975, len(array) - 1))
    return mean, mean - critical * error, mean + critical * error


__all__ = [
    "ClusterOLSResult",
    "ContrastResult",
    "benjamini_hochberg",
    "cr1_ols",
    "exact_cluster_sign_flip_p",
    "linear_contrast",
    "mean_of_finite_values",
    "mean_t_interval",
    "sem",
    "significance_stars",
]
