"""Display-derived level-profile metrics for ED6 example-map curation.

These metrics describe the independently normalized 45 x 45 maps used in the
review gallery.  They are selection aids, not inferential level-map stability
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_ACTIVITY_FLOOR = 0.20
DEFAULT_RELIABLE_FINE_BINS = 9
N_DISPLAY_LEVELS = 7


@dataclass(frozen=True)
class LevelActivityProfile:
    """Seven-level activity-density profile and its support audit."""

    support_bins: np.ndarray
    support_weight: np.ndarray
    density_above_floor: np.ndarray
    weighted_density: np.ndarray
    probability: np.ndarray


def compute_level_activity_profile(
    normalized_map: np.ndarray,
    common_mask: np.ndarray,
    display_levels: np.ndarray,
    *,
    activity_floor: float = DEFAULT_ACTIVITY_FLOOR,
    reliable_fine_bins: int = DEFAULT_RELIABLE_FINE_BINS,
) -> LevelActivityProfile:
    """Compute a geometry-balanced activity-density profile over levels 1-7.

    Per-level means prevent deeper levels from winning merely because they
    contain more branches.  A support weight downweights levels represented by
    fewer than one 15 x 15 maze tile (nine 45 x 45 fine bins).
    """

    values = np.asarray(normalized_map, dtype=float)
    support = np.asarray(common_mask, dtype=bool)
    levels = np.asarray(display_levels)
    if values.shape != support.shape or values.shape != levels.shape:
        raise ValueError("map, common_mask, and display_levels must share a shape")
    if not 0.0 <= float(activity_floor) < 1.0:
        raise ValueError("activity_floor must be in [0, 1)")
    if int(reliable_fine_bins) <= 0:
        raise ValueError("reliable_fine_bins must be positive")

    valid = support & np.isfinite(values)
    signal = np.clip(values - float(activity_floor), 0.0, None)
    support_bins = np.zeros(N_DISPLAY_LEVELS, dtype=int)
    density = np.zeros(N_DISPLAY_LEVELS, dtype=float)
    for level in range(1, N_DISPLAY_LEVELS + 1):
        mask = valid & (levels == level)
        support_bins[level - 1] = int(np.sum(mask))
        if support_bins[level - 1]:
            density[level - 1] = float(np.mean(signal[mask]))

    support_weight = np.minimum(
        1.0, support_bins.astype(float) / float(reliable_fine_bins)
    )
    weighted_density = support_weight * density
    total = float(np.sum(weighted_density))
    probability = (
        weighted_density / total if total > 0.0 else np.zeros_like(weighted_density)
    )
    return LevelActivityProfile(
        support_bins=support_bins,
        support_weight=support_weight,
        density_above_floor=density,
        weighted_density=weighted_density,
        probability=probability,
    )


def level_profile_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Return shared profile mass, equivalently one minus total variation."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != (N_DISPLAY_LEVELS,) or b.shape != (N_DISPLAY_LEVELS,):
        raise ValueError("level profiles must each have length 7")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise ValueError("level profiles must be nonnegative")
    if not np.isclose(np.sum(a), 1.0) or not np.isclose(np.sum(b), 1.0):
        return float("nan")
    return float(np.minimum(a, b).sum())


def encoded_level_jaccard(first: Iterable[int], second: Iterable[int]) -> float:
    """Return Jaccard overlap of two display-field-derived level sets."""

    a = {int(value) for value in first}
    b = {int(value) for value in second}
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def ordered_level_earth_mover(first: np.ndarray, second: np.ndarray) -> float:
    """Return 1D Earth-mover distance in display-level units (range 0-6)."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != (N_DISPLAY_LEVELS,) or b.shape != (N_DISPLAY_LEVELS,):
        raise ValueError("level profiles must each have length 7")
    return float(np.abs(np.cumsum(a - b)[:-1]).sum())


def preservation_stratum(value: float) -> str:
    """Return a descriptive, non-inferential preservation stratum."""

    numeric = float(value)
    if not np.isfinite(numeric):
        return "undefined"
    if numeric < 0.30:
        return "low_<0.30"
    if numeric < 0.75:
        return "intermediate_0.30-<0.75"
    return "high_>=0.75"


__all__ = [
    "DEFAULT_ACTIVITY_FLOOR",
    "DEFAULT_RELIABLE_FINE_BINS",
    "LevelActivityProfile",
    "compute_level_activity_profile",
    "encoded_level_jaccard",
    "level_profile_overlap",
    "ordered_level_earth_mover",
    "preservation_stratum",
]
