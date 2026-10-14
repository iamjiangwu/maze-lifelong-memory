"""Decoder contrasts, behavior regression and route mouse-level tests."""
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
FloatArray = NDArray[np.float64]

def _decoder_stats_difference(
    difference: FloatArray, animals: Iterable[str]
) -> dict[str, float | int]:
    """Return the animal-clustered intercept test for one pair-level contrast."""

    values = np.asarray(difference, dtype=float)
    fit = cr1_ols(values, np.ones((len(values), 1)), list(animals))
    contrast = linear_contrast(fit, [1.0])
    return {
        "estimate": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "cluster_p": contrast.p_two_sided,
        "ci95_low": contrast.confidence_low,
        "ci95_high": contrast.confidence_high,
        "n_pairs": fit.n_observations,
        "n_animals": fit.n_clusters,
    }

def _association_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    """Return Pearson correlation and standardized animal-clustered slope."""

    x = frame["first_long_bout_time_s"].to_numpy(float)
    y = frame["reward_decoding_error"].to_numpy(float)
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    # Test behavior as the outcome, as in the neural and behavior matrix.
    # The display fits this same direction: decoding error x, behavior y.
    fit = cr1_ols(zx, np.column_stack((np.ones(len(zy)), zy)), frame["animal_id"])
    contrast = linear_contrast(fit, [0.0, 1.0])
    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "standardized_slope": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "cluster_p": contrast.p_two_sided,
        "ci95_low": contrast.confidence_low,
        "ci95_high": contrast.confidence_high,
        "n_pairs": fit.n_observations,
        "n_animals": fit.n_clusters,
    }

def combine_whole_maze_shuffle_values(pair_values: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine whole maze pair rows and attach the animal clustered comparison."""

    whole_maze_values = pd.concat(pair_values, ignore_index=True)
    for key, value in _decoder_stats_difference(
        whole_maze_values["shuffle_minus_data"].to_numpy(float),
        whole_maze_values["animal_id"],
    ).items():
        whole_maze_values[f"stat_{key}"] = value
    return whole_maze_values

def combine_reward_control_values(pair_values: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Combine eligible reward control rows and attach the clustered comparison."""

    reward_control_values = pd.concat(pair_values, ignore_index=True)
    reward_control_values = reward_control_values.loc[
        reward_control_values["included"]
    ].reset_index(drop=True)
    for key, value in _decoder_stats_difference(
        reward_control_values["control_minus_reward"].to_numpy(float),
        reward_control_values["animal_id"],
    ).items():
        reward_control_values[f"stat_{key}"] = value
    return reward_control_values

def calculate_reward_decoding_association(
    pair_values: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the nominal animal clustered test to the saved decoder pairs."""

    values = pair_values.sort_values("pair_key", kind="stable").reset_index(drop=True)
    for key, value in _association_stats(values).items():
        values[f"stat_{key}"] = value
    return values

def prepare_reward_decoding_vs_first_bout_display(
    values: pd.DataFrame,
) -> pd.DataFrame:
    """Fit behavior on neural error with a whole-animal bootstrap band."""

    x = values["reward_decoding_error"].to_numpy(float)
    y = values["first_long_bout_time_s"].to_numpy(float)
    animals = values["animal_id"].astype(str).to_numpy()
    unique_animals = np.asarray(sorted(set(animals)))
    x_limit = float(x.max() * 1.08)
    grid = np.linspace(0.0, x_limit, 200)
    fit = stats.linregress(x, y)
    fitted = fit.intercept + fit.slope * grid
    rng = np.random.default_rng(202608041)
    bootstrap_lines: list[FloatArray] = []
    for _ in range(5_000):
        sampled = rng.choice(unique_animals, size=len(unique_animals), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(animals == animal) for animal in sampled]
        )
        if np.unique(x[indices]).size < 2:
            continue
        sampled_fit = stats.linregress(x[indices], y[indices])
        bootstrap_lines.append(
            sampled_fit.intercept + sampled_fit.slope * grid
        )
    lines = np.asarray(bootstrap_lines, dtype=float)
    low, high = np.quantile(lines, (0.025, 0.975), axis=0)
    return pd.DataFrame(
        {
            "reward_decoding_error": grid,
            "fitted_first_long_bout_time_s": fitted,
            "ci95_low": low,
            "ci95_high": high,
        }
    )
@dataclass(frozen=True)
class RouteDisplay:
    """Display arrays, animal averaged curves and reward contrasts."""

    heatmaps: dict[str, tuple[FloatArray, FloatArray]]
    curves: pd.DataFrame
    statistics: pd.DataFrame

def reward_contrast(profile: FloatArray) -> dict[str, float]:
    """Compare the single reward bin with the mean of all other route bins."""

    return dict(port=float(profile[-1]), away=float(np.nanmean(profile[:-1])))

def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    """Equal-observation mean and 95% Student-t interval over finite values."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    half = float(stats.sem(values) * stats.t.ppf(0.975, len(values) - 1))
    return mean, mean - half, mean + half

def exact_sign_flip(values: np.ndarray) -> float:
    """Two-sided exact test of mouse paired differences, without Monte Carlo."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
    return float(np.mean(np.abs(signs @ values) >= abs(values.sum()) - 1e-12))

def summarize_reward_contrasts(animal_values: pd.DataFrame) -> pd.DataFrame:
    """Paired reward-minus-elsewhere tests with BH across the supplied groups."""

    rows = []
    for group, frame in animal_values.groupby("group", sort=False):
        paired = frame.loc[np.isfinite(frame[["port", "away"]]).all(axis=1)]
        delta = (paired.port - paired.away).to_numpy()
        mean, lo, hi = mean_interval(delta)
        rows.append(
            dict(
                group=group,
                mean=mean,
                lo=lo,
                hi=hi,
                n_mice=len(delta),
                p_exact=exact_sign_flip(delta),
            )
        )
    result = pd.DataFrame(rows)
    result["adjusted_p"] = benjamini_hochberg(result.p_exact.to_numpy())
    return result

def route_summary(heatmaps, pair_profiles):
    animals=pair_profiles.groupby(['group','animal','bin'],as_index=False)[['pv','pv_sh']].mean()
    curves=[]
    for (group,spatial_bin),frame in animals.groupby(['group','bin']):
        mean,lo,hi=mean_interval(frame.pv)
        curves.append(dict(group=group,bin=spatial_bin,mean=mean,lo=lo,hi=hi,shuffle=frame.pv_sh.mean(),n_mice=len(frame)))
    contrasts=[]
    for (group,animal),frame in animals.groupby(['group','animal']):
        contrasts.append(dict(group=group,animal=animal,**reward_contrast(frame.sort_values('bin').pv.to_numpy())))
    return RouteDisplay(heatmaps,pd.DataFrame(curves),summarize_reward_contrasts(pd.DataFrame(contrasts)))
