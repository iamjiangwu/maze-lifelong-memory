"""Selection helpers for the ED6 real place-map example gallery.

The public scoring contract is intentionally narrow: cross-session place-map
correlation and spatial information are never inputs to the visual-quality
score.  Correlation is used only by :func:`select_balanced_candidates` as a
coverage constraint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import label as ndi_label
from scipy.optimize import Bounds, LinearConstraint, milp


CORRELATION_BIN_EDGES = np.asarray(
    [-1.000001, -0.20, -0.10, 0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.000001],
    dtype=float,
)
CORRELATION_BIN_LABELS = (
    "[-1,-.20)",
    "[-.20,-.10)",
    "[-.10,0)",
    "[0,.10)",
    "[.10,.20)",
    "[.20,.35)",
    "[.35,.50)",
    "[.50,.70)",
    "[.70,.85)",
    "[.85,1]",
)


def safe_corr(a: np.ndarray, b: np.ndarray, *, min_points: int = 3) -> float:
    """Finite Pearson correlation with explicit support and variance checks."""

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    if int(keep.sum()) < int(min_points):
        return float("nan")
    a = a[keep]
    b = b[keep]
    if float(np.nanstd(a)) <= 0.0 or float(np.nanstd(b)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def assign_correlation_bins(values: Iterable[float]) -> pd.Categorical:
    """Assign the fixed negative-to-near-one coverage bins used by ED6."""

    values = np.asarray(list(values), dtype=float)
    return pd.cut(
        values,
        bins=CORRELATION_BIN_EDGES,
        labels=CORRELATION_BIN_LABELS,
        include_lowest=True,
        right=False,
        ordered=True,
    )


def _bounded_log_scale(values: pd.Series, cap: float) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=cap, neginf=0.0)
    x = np.clip(x, 0.0, cap)
    return np.log1p(x) / np.log1p(float(cap))


def _reliability_scale(values: pd.Series) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=-0.10, posinf=1.0, neginf=-1.0)
    return np.clip((x + 0.10) / 0.90, 0.0, 1.0)


def _geometric_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(a, 0.0, 1.0) * np.clip(b, 0.0, 1.0))


def compute_pre_visual_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Return table-based visual-quality components and a prescore.

    Only activity, within-session split-half reliability, field coverage,
    peak sharpness, component compactness, and common occupancy support enter
    the score.  In particular, ``place_map_corr``, entropy, symmetry, and any
    spatial-information column are ignored even if present in ``frame``.
    """

    required = {
        "rate_per_min_train",
        "rate_per_min_test",
        "splithalf_corr_train",
        "splithalf_corr_test",
        "halfmax_coverage_train",
        "halfmax_coverage_test",
        "peak_sharpness_train",
        "peak_sharpness_test",
        "halfmax_components_train",
        "halfmax_components_test",
        "common_valid_bin_fraction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"visual prescore is missing columns: {missing}")

    rate_a = _bounded_log_scale(frame["rate_per_min_train"], cap=12.0)
    rate_b = _bounded_log_scale(frame["rate_per_min_test"], cap=12.0)
    activity = _geometric_pair(rate_a, rate_b)

    rel_a = _reliability_scale(frame["splithalf_corr_train"])
    rel_b = _reliability_scale(frame["splithalf_corr_test"])
    reliability = _geometric_pair(rel_a, rel_b)

    cov_a = np.sqrt(
        np.clip(
            pd.to_numeric(frame["halfmax_coverage_train"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            / 0.08,
            0.0,
            1.0,
        )
    )
    cov_b = np.sqrt(
        np.clip(
            pd.to_numeric(frame["halfmax_coverage_test"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            / 0.08,
            0.0,
            1.0,
        )
    )
    coverage = _geometric_pair(cov_a, cov_b)

    sharp_a = _bounded_log_scale(frame["peak_sharpness_train"], cap=35.0)
    sharp_b = _bounded_log_scale(frame["peak_sharpness_test"], cap=35.0)
    sharpness = _geometric_pair(sharp_a, sharp_b)

    comp_a = pd.to_numeric(frame["halfmax_components_train"], errors="coerce")
    comp_b = pd.to_numeric(frame["halfmax_components_test"], errors="coerce")
    comp_a = np.maximum(comp_a.fillna(9.0).to_numpy(dtype=float), 1.0)
    comp_b = np.maximum(comp_b.fillna(9.0).to_numpy(dtype=float), 1.0)

    def component_score(counts: np.ndarray) -> np.ndarray:
        # One or two coherent fields both score maximally; this deliberately
        # avoids penalizing symmetric two-lobe maps. Three fields remain good,
        # while progressively fragmented maps are downweighted.
        return np.where(
            counts <= 2,
            1.0,
            np.where(
                counts <= 3,
                0.90,
                np.clip(0.90 / np.sqrt(np.maximum(counts - 2.0, 1.0)), 0.0, 1.0),
            ),
        )

    compactness = _geometric_pair(component_score(comp_a), component_score(comp_b))

    support = pd.to_numeric(frame["common_valid_bin_fraction"], errors="coerce")
    support = np.clip(support.fillna(0.0).to_numpy(dtype=float) / 0.45, 0.0, 1.0)

    score = (
        0.30 * activity
        + 0.32 * reliability
        + 0.10 * coverage
        + 0.10 * sharpness
        + 0.12 * compactness
        + 0.06 * support
    )
    return pd.DataFrame(
        {
            "activity_score": activity,
            "within_session_reliability_score": reliability,
            "field_coverage_score": coverage,
            "peak_sharpness_score": sharpness,
            "field_compactness_score": compactness,
            "common_support_score": support,
            "pre_visual_score": np.clip(score, 0.0, 1.0),
        },
        index=frame.index,
    )


def normalize_pair_shared_max(
    map_a: np.ndarray,
    map_b: np.ndarray,
    common_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mask and normalize a pair by one shared nonnegative maximum."""

    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    mask = np.asarray(common_mask, dtype=bool)
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("map_a, map_b, and common_mask must have identical shapes")
    a = np.where(mask, np.maximum(a, 0.0), np.nan)
    b = np.where(mask, np.maximum(b, 0.0), np.nan)
    maxima = [np.nanmax(x) for x in (a, b) if np.any(np.isfinite(x))]
    shared_max = float(max(maxima)) if maxima else 0.0
    if not np.isfinite(shared_max) or shared_max <= 0.0:
        return np.full_like(a, np.nan), np.full_like(b, np.nan), 0.0
    return a / shared_max, b / shared_max, shared_max


def normalize_pair_independent_quantile(
    map_a: np.ndarray,
    map_b: np.ndarray,
    common_mask: np.ndarray,
    *,
    quantile: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Normalize each map independently on common support using a robust quantile.

    Nonnegative values are divided by their own within-mask quantile and
    clipped to ``[0, 1]``.  Pixels outside ``common_mask`` remain ``NaN`` so
    the rendered white region continues to encode missing common occupancy.
    A positive maximum is used only when the requested quantile is zero (a
    degenerate map with fewer positive pixels than the quantile can resolve).
    """

    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    mask = np.asarray(common_mask, dtype=bool)
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError("map_a, map_b, and common_mask must have identical shapes")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile must be in (0, 1]")

    def normalize_one(values: np.ndarray) -> tuple[np.ndarray, float]:
        valid = mask & np.isfinite(values)
        normalized = np.full(values.shape, np.nan, dtype=float)
        if not np.any(valid):
            return normalized, 0.0
        nonnegative = np.maximum(values[valid], 0.0)
        scale = float(np.quantile(nonnegative, float(quantile)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.max(nonnegative)) if nonnegative.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            normalized[valid] = 0.0
            return normalized, 0.0
        normalized[valid] = np.clip(nonnegative / scale, 0.0, 1.0)
        return normalized, scale

    normalized_a, scale_a = normalize_one(a)
    normalized_b, scale_b = normalize_one(b)
    return normalized_a, normalized_b, scale_a, scale_b


def _neighbor_coherence(values: np.ndarray, mask: np.ndarray) -> float:
    pairs_a: list[np.ndarray] = []
    pairs_b: list[np.ndarray] = []
    for dy, dx in ((0, 1), (1, 0)):
        if dy == 0:
            m = mask[:, :-1] & mask[:, 1:]
            pairs_a.append(values[:, :-1][m])
            pairs_b.append(values[:, 1:][m])
        else:
            m = mask[:-1, :] & mask[1:, :]
            pairs_a.append(values[:-1, :][m])
            pairs_b.append(values[1:, :][m])
    if not pairs_a:
        return 0.0
    corr = safe_corr(np.concatenate(pairs_a), np.concatenate(pairs_b), min_points=12)
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip((corr + 0.15) / 1.15, 0.0, 1.0))


def _active_region_smoothness(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    support_threshold: float = 0.10,
) -> float:
    """Normalized adjacent-pixel smoothness around the visible signal.

    Restricting edges to those where at least one endpoint has appreciable
    signal prevents the shared zero-valued background from making every map
    appear similarly coherent.
    """

    numerator = 0.0
    denominator = 0.0
    for first, second, valid in (
        (values[:, :-1], values[:, 1:], mask[:, :-1] & mask[:, 1:]),
        (values[:-1, :], values[1:, :], mask[:-1, :] & mask[1:, :]),
    ):
        active_edge = valid & (
            (first >= float(support_threshold)) | (second >= float(support_threshold))
        )
        if not np.any(active_edge):
            continue
        a = np.clip(first[active_edge], 0.0, 1.0)
        b = np.clip(second[active_edge], 0.0, 1.0)
        numerator += float(np.sum(np.abs(a - b)))
        denominator += float(np.sum(np.maximum(a, b)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))


def display_map_features(
    normalized_map: np.ndarray, common_mask: np.ndarray
) -> dict[str, float]:
    """Measure salience and coherence of one normalized display map."""

    arr = np.asarray(normalized_map, dtype=float)
    mask = np.asarray(common_mask, dtype=bool) & np.isfinite(arr)
    finite = arr[mask]
    if finite.size < 12 or float(np.nanmax(finite)) <= 0.0:
        return {
            "salience": 0.0,
            "neighbor_coherence": 0.0,
            "largest_component_fraction": 0.0,
            "active_area_score": 0.0,
            "peak_y": -1.0,
            "peak_x": -1.0,
            "active_components": 0.0,
        }

    own_max = float(np.nanmax(finite))
    salience = float(np.clip(np.nanquantile(finite, 0.95), 0.0, 1.0))
    coherence = _neighbor_coherence(arr, mask)
    active = mask & (arr >= 0.30 * own_max)
    labels, n_components = ndi_label(active)
    sizes = np.bincount(labels[active].ravel()) if np.any(active) else np.asarray([])
    if sizes.size:
        ordered_sizes = np.sort(sizes)[::-1]
        coherent_fraction = float(ordered_sizes[:3].sum() / ordered_sizes.sum())
        count_score = (
            1.0
            if n_components <= 2
            else (0.90 if n_components == 3 else 0.90 / np.sqrt(n_components - 2.0))
        )
        largest_fraction = float(
            np.clip(0.70 * coherent_fraction + 0.30 * count_score, 0.0, 1.0)
        )
    else:
        largest_fraction = 0.0
    active_fraction = float(active.sum() / max(int(mask.sum()), 1))
    active_area_score = float(np.sqrt(np.clip(active_fraction / 0.06, 0.0, 1.0)))
    peak_flat = int(np.nanargmax(np.where(mask, arr, np.nan)))
    peak_y, peak_x = np.unravel_index(peak_flat, arr.shape)
    return {
        "salience": salience,
        "neighbor_coherence": coherence,
        "largest_component_fraction": largest_fraction,
        "active_area_score": active_area_score,
        "peak_y": float(peak_y),
        "peak_x": float(peak_x),
        "active_components": float(n_components),
    }


def _active_area_band_score(active_fraction: float) -> float:
    """Reward a visible but localized high-activity footprint.

    The plateau from 2% to 12% is intentionally broad.  Maps below 1% or
    above 20% receive at most a small score, which prevents isolated pixels
    and nearly uniform maps from ranking well after per-map normalization.
    """

    knots = np.asarray([0.0, 0.01, 0.02, 0.12, 0.20, 0.30, 1.0], dtype=float)
    scores = np.asarray([0.0, 0.15, 1.0, 1.0, 0.15, 0.0, 0.0], dtype=float)
    return float(np.interp(float(active_fraction), knots, scores))


def _component_count_score(n_components: int) -> float:
    """Reward one to three fields and progressively penalize fragmentation."""

    count = int(n_components)
    if count <= 0:
        return 0.0
    if count <= 3:
        return 1.0
    if count == 4:
        return 0.65
    if count == 5:
        return 0.45
    return float(np.clip(0.45 / np.sqrt(count - 4.0), 0.05, 0.45))


def _v2_map_quality(
    contrast: np.ndarray,
    coherence: np.ndarray,
    active_area: np.ndarray,
    component: np.ndarray,
) -> np.ndarray:
    """Image-only map score used by the v2 candidate ranker."""

    contrast = np.clip(np.asarray(contrast, dtype=float), 0.0, 1.0)
    coherence = np.clip(np.asarray(coherence, dtype=float), 0.0, 1.0)
    active_area = np.clip(np.asarray(active_area, dtype=float), 0.0, 1.0)
    component = np.clip(np.asarray(component, dtype=float), 0.0, 1.0)
    base = 0.25 * contrast + 0.32 * coherence + 0.23 * active_area + 0.20 * component
    localization_gate = np.sqrt(contrast * active_area)
    return np.clip(base * (0.50 + 0.50 * localization_gate), 0.0, 1.0)


def display_map_features_v2(
    normalized_map: np.ndarray,
    common_mask: np.ndarray,
    *,
    active_threshold: float = 0.50,
) -> dict[str, float]:
    """Measure localized, coherent 45x45 map appearance for v2 ranking.

    Inputs are expected to be independently normalized display maps.  The
    score is based only on pixels: local smoothness, robust contrast, a broad
    2--12% high-activity-area target, and whether the active region forms one
    to three components.  It deliberately does not use cross-session map
    correlation, spatial information, entropy, or symmetry.
    """

    arr = np.asarray(normalized_map, dtype=float)
    mask = np.asarray(common_mask, dtype=bool)
    if arr.ndim != 2 or arr.shape != mask.shape:
        raise ValueError(
            "normalized_map and common_mask must be identically shaped 2D arrays"
        )
    if not 0.0 < float(active_threshold) <= 1.0:
        raise ValueError("active_threshold must be in (0, 1]")
    mask = mask & np.isfinite(arr)
    finite = np.clip(arr[mask], 0.0, 1.0)

    empty = {
        "contrast_score": 0.0,
        "neighbor_coherence": 0.0,
        "active_fraction": 0.0,
        "active_area_score": 0.0,
        "largest_component_fraction": 0.0,
        "top3_component_fraction": 0.0,
        "component_count_score": 0.0,
        "component_score": 0.0,
        "image_quality_score": 0.0,
        "peak_y": -1.0,
        "peak_x": -1.0,
        "active_components": 0.0,
        "active_pixel_count": 0.0,
    }
    if finite.size < 12 or float(np.max(finite, initial=0.0)) <= 0.0:
        return empty

    ordered = np.sort(finite)
    top_count = max(1, int(np.ceil(0.05 * ordered.size)))
    top_mean = float(np.mean(ordered[-top_count:]))
    baseline = float(np.median(ordered))
    contrast = float(np.clip((top_mean - baseline) / 0.55, 0.0, 1.0))
    coherence = _active_region_smoothness(arr, mask)

    active = mask & (arr >= float(active_threshold))
    labels, n_components = ndi_label(active)
    sizes = (
        np.bincount(labels.ravel())[1:] if n_components else np.asarray([], dtype=int)
    )
    active_count = int(sizes.sum()) if sizes.size else 0
    active_fraction = float(active_count / max(int(mask.sum()), 1))
    area_score = _active_area_band_score(active_fraction)

    if sizes.size:
        ordered_sizes = np.sort(sizes)[::-1]
        largest_fraction = float(ordered_sizes[0] / active_count)
        top3_fraction = float(ordered_sizes[:3].sum() / active_count)
        count_score = _component_count_score(int(n_components))
        component_score = float(
            np.clip(0.55 * count_score + 0.45 * top3_fraction, 0.0, 1.0)
        )
    else:
        largest_fraction = 0.0
        top3_fraction = 0.0
        count_score = 0.0
        component_score = 0.0

    quality = float(_v2_map_quality(contrast, coherence, area_score, component_score))
    peak_flat = int(np.nanargmax(np.where(mask, arr, np.nan)))
    peak_y, peak_x = np.unravel_index(peak_flat, arr.shape)
    return {
        "contrast_score": contrast,
        "neighbor_coherence": coherence,
        "active_fraction": active_fraction,
        "active_area_score": area_score,
        "largest_component_fraction": largest_fraction,
        "top3_component_fraction": top3_fraction,
        "component_count_score": count_score,
        "component_score": component_score,
        "image_quality_score": quality,
        "peak_y": float(peak_y),
        "peak_x": float(peak_x),
        "active_components": float(n_components),
        "active_pixel_count": float(active_count),
    }


def combine_pair_quality_scores(score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
    """Combine two map scores while making the weaker map carry most weight."""

    a = np.clip(np.asarray(score_a, dtype=float), 0.0, 1.0)
    b = np.clip(np.asarray(score_b, dtype=float), 0.0, 1.0)
    return 0.60 * np.minimum(a, b) + 0.40 * (a + b) / 2.0


def compute_visual_score_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the image-driven v2 rank score for paired display maps.

    Each map score is 80% pixel-derived appearance, 15% within-session
    reliability, and 5% activity.  The two map scores are then combined as
    ``0.6 * min + 0.4 * mean`` so a single poor-looking side cannot be hidden
    by its partner.  Cross-session correlation, spatial information, entropy,
    symmetry, and table-based aggregate prescores never enter the ranking.
    """

    feature_names = (
        "contrast_score",
        "neighbor_coherence",
        "active_area_score",
        "component_score",
    )
    required = {f"{name}_{side}" for name in feature_names for side in ("a", "b")}
    required.update(
        {
            "rate_per_min_train",
            "rate_per_min_test",
            "splithalf_corr_train",
            "splithalf_corr_test",
        }
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v2 visual score is missing columns: {missing}")

    def values(name: str, side: str) -> np.ndarray:
        return (
            pd.to_numeric(frame[f"{name}_{side}"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    contrast_a, contrast_b = values("contrast_score", "a"), values(
        "contrast_score", "b"
    )
    coherence_a = values("neighbor_coherence", "a")
    coherence_b = values("neighbor_coherence", "b")
    area_a, area_b = values("active_area_score", "a"), values("active_area_score", "b")
    component_a, component_b = values("component_score", "a"), values(
        "component_score", "b"
    )
    image_a = _v2_map_quality(contrast_a, coherence_a, area_a, component_a)
    image_b = _v2_map_quality(contrast_b, coherence_b, area_b, component_b)
    reliability_a = _reliability_scale(frame["splithalf_corr_train"])
    reliability_b = _reliability_scale(frame["splithalf_corr_test"])
    activity_a = _bounded_log_scale(frame["rate_per_min_train"], cap=12.0)
    activity_b = _bounded_log_scale(frame["rate_per_min_test"], cap=12.0)
    map_score_a = 0.80 * image_a + 0.15 * reliability_a + 0.05 * activity_a
    map_score_b = 0.80 * image_b + 0.15 * reliability_b + 0.05 * activity_b
    pair_score = combine_pair_quality_scores(map_score_a, map_score_b)

    return pd.DataFrame(
        {
            "display_contrast_score": combine_pair_quality_scores(
                contrast_a, contrast_b
            ),
            "display_coherence_score": combine_pair_quality_scores(
                coherence_a, coherence_b
            ),
            "display_active_area_score": combine_pair_quality_scores(area_a, area_b),
            "display_component_score": combine_pair_quality_scores(
                component_a, component_b
            ),
            "map_image_score_a": image_a,
            "map_image_score_b": image_b,
            "display_image_score": combine_pair_quality_scores(image_a, image_b),
            "display_reliability_score": combine_pair_quality_scores(
                reliability_a, reliability_b
            ),
            "display_activity_score": combine_pair_quality_scores(
                activity_a, activity_b
            ),
            "map_visual_score_a": map_score_a,
            "map_visual_score_b": map_score_b,
            "visual_score": pair_score,
        },
        index=frame.index,
    )


def compute_final_visual_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine prescore and 45x45 image appearance without cross-session r."""

    required = {
        "pre_visual_score",
        "salience_a",
        "salience_b",
        "neighbor_coherence_a",
        "neighbor_coherence_b",
        "largest_component_fraction_a",
        "largest_component_fraction_b",
        "active_area_score_a",
        "active_area_score_b",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"final visual score is missing columns: {missing}")

    def pair(name_a: str, name_b: str) -> np.ndarray:
        a = (
            pd.to_numeric(frame[name_a], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        b = (
            pd.to_numeric(frame[name_b], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return _geometric_pair(a, b)

    salience = pair("salience_a", "salience_b")
    coherence = pair("neighbor_coherence_a", "neighbor_coherence_b")
    component = pair("largest_component_fraction_a", "largest_component_fraction_b")
    active_area = pair("active_area_score_a", "active_area_score_b")
    image_a = (
        0.35
        * pd.to_numeric(frame["salience_a"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.35
        * pd.to_numeric(frame["neighbor_coherence_a"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.20
        * pd.to_numeric(frame["largest_component_fraction_a"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.10
        * pd.to_numeric(frame["active_area_score_a"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    image_b = (
        0.35
        * pd.to_numeric(frame["salience_b"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.35
        * pd.to_numeric(frame["neighbor_coherence_b"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.20
        * pd.to_numeric(frame["largest_component_fraction_b"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 0.10
        * pd.to_numeric(frame["active_area_score_b"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    image_score = 0.60 * np.minimum(image_a, image_b) + 0.40 * (image_a + image_b) / 2.0
    pre = (
        pd.to_numeric(frame["pre_visual_score"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    final = 0.65 * np.clip(pre, 0.0, 1.0) + 0.35 * np.clip(image_score, 0.0, 1.0)
    return pd.DataFrame(
        {
            "display_salience_score": salience,
            "display_coherence_score": coherence,
            "display_component_score": component,
            "display_active_area_score": active_area,
            "display_image_score": image_score,
            "visual_score": np.clip(final, 0.0, 1.0),
        },
        index=frame.index,
    )


def morphology_signature(row: pd.Series, *, n_peak_zones: int = 3) -> str:
    """Coarse display morphology used only as a nonredundancy constraint."""

    def zone(value: object) -> int:
        v = float(value)
        if not np.isfinite(v) or v < 0:
            return -1
        return min(int(v * n_peak_zones / 45.0), n_peak_zones - 1)

    comp_a = min(int(max(float(row.get("active_components_a", 0.0)), 0.0)), 3)
    comp_b = min(int(max(float(row.get("active_components_b", 0.0)), 0.0)), 3)
    return (
        f"A{zone(row.get('peak_y_a', -1))}{zone(row.get('peak_x_a', -1))}"
        f"_B{zone(row.get('peak_y_b', -1))}{zone(row.get('peak_x_b', -1))}"
        f"_C{comp_a}{comp_b}"
    )


def select_balanced_candidates(
    frame: pd.DataFrame,
    *,
    target_n: int,
    required_pairs: Sequence[str],
    min_per_pair: int = 2,
    max_per_pair: int = 4,
    min_per_corr_bin: int = 6,
    max_per_corr_bin: int = 14,
    max_per_morphology: int | None = 4,
    min_low_extreme: int = 0,
    low_extreme_threshold: float = -0.25,
    min_high_extreme: int = 0,
    high_extreme_threshold: float = 0.90,
    correlation_col: str = "place_r_15_analysis",
    max_per_node: int | None = 1,
    conflict_pairs: Sequence[tuple[str, str]] | None = None,
    score_col: str = "visual_score",
    fixed_col: str = "fixed_main_fig2b",
) -> pd.DataFrame:
    """Maximize visual score subject to pair, r-spectrum, and diversity rules."""

    required_columns = {"candidate_id", "pair_key", "correlation_bin", score_col}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"selection table is missing columns: {missing}")
    work = frame.reset_index(drop=True).copy()
    if work["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")
    n = len(work)
    if n < int(target_n):
        raise ValueError(f"only {n} candidates for target_n={target_n}")

    rows: list[sparse.csr_matrix] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint_row(indices: np.ndarray | Sequence[int]) -> sparse.csr_matrix:
        indices = np.asarray(indices, dtype=int).ravel()
        if indices.size == 0:
            return sparse.csr_matrix((1, n), dtype=float)
        return sparse.csr_matrix(
            (
                np.ones(indices.size, dtype=float),
                (np.zeros(indices.size, dtype=int), indices),
            ),
            shape=(1, n),
        )

    rows.append(constraint_row(np.arange(n, dtype=int)))
    lower.append(float(target_n))
    upper.append(float(target_n))

    fixed = (
        work.get(fixed_col, pd.Series(False, index=work.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    for idx in np.flatnonzero(fixed):
        rows.append(constraint_row([int(idx)]))
        lower.append(1.0)
        upper.append(1.0)

    for pair_key in required_pairs:
        indices = np.flatnonzero(
            work["pair_key"].astype(str).eq(str(pair_key)).to_numpy()
        )
        if int(indices.size) < int(min_per_pair):
            raise ValueError(
                f"pair {pair_key!r} has only {int(indices.size)} candidates"
            )
        rows.append(constraint_row(indices))
        lower.append(float(min_per_pair))
        upper.append(float(max_per_pair))

    for label in CORRELATION_BIN_LABELS:
        indices = np.flatnonzero(
            work["correlation_bin"].astype(str).eq(str(label)).to_numpy()
        )
        if int(indices.size) < int(min_per_corr_bin):
            raise ValueError(
                f"correlation bin {label!r} has only {int(indices.size)} candidates"
            )
        rows.append(constraint_row(indices))
        lower.append(float(min_per_corr_bin))
        upper.append(float(max_per_corr_bin))

    if int(min_low_extreme) > 0 or int(min_high_extreme) > 0:
        if correlation_col not in work.columns:
            raise ValueError(f"selection table is missing {correlation_col!r}")
        corr = pd.to_numeric(work[correlation_col], errors="coerce").to_numpy(
            dtype=float
        )
        if int(min_low_extreme) > 0:
            indices = np.flatnonzero(
                np.isfinite(corr) & (corr <= float(low_extreme_threshold))
            )
            if int(indices.size) < int(min_low_extreme):
                raise ValueError(f"only {int(indices.size)} low-extreme candidates")
            rows.append(constraint_row(indices))
            lower.append(float(min_low_extreme))
            upper.append(float(target_n))
        if int(min_high_extreme) > 0:
            indices = np.flatnonzero(
                np.isfinite(corr) & (corr >= float(high_extreme_threshold))
            )
            if int(indices.size) < int(min_high_extreme):
                raise ValueError(f"only {int(indices.size)} high-extreme candidates")
            rows.append(constraint_row(indices))
            lower.append(float(min_high_extreme))
            upper.append(float(target_n))

    if max_per_morphology is not None and "morphology_signature" in work.columns:
        for signature, group in work.groupby(
            "morphology_signature", sort=True, dropna=False
        ):
            indices = group.index.to_numpy(dtype=int)
            fixed_count = int(np.sum(fixed[indices]))
            rows.append(constraint_row(indices))
            lower.append(0.0)
            upper.append(float(max(int(max_per_morphology), fixed_count)))

    if max_per_node is not None and {"node_a", "node_b"}.issubset(work.columns):
        node_to_indices: dict[str, set[int]] = {}
        for idx, row in work[["node_a", "node_b"]].iterrows():
            for node in (str(row["node_a"]), str(row["node_b"])):
                if node and node.lower() != "nan":
                    node_to_indices.setdefault(node, set()).add(int(idx))
        for indices in node_to_indices.values():
            if len(indices) <= int(max_per_node):
                continue
            use = np.asarray(sorted(indices), dtype=int)
            fixed_count = int(np.sum(fixed[use]))
            rows.append(constraint_row(use))
            lower.append(0.0)
            upper.append(float(max(int(max_per_node), fixed_count)))

    if conflict_pairs:
        id_to_index = {
            str(value): int(idx) for idx, value in enumerate(work["candidate_id"])
        }
        for first, second in conflict_pairs:
            if str(first) not in id_to_index or str(second) not in id_to_index:
                continue
            indices = np.asarray(
                [id_to_index[str(first)], id_to_index[str(second)]], dtype=int
            )
            fixed_count = int(np.sum(fixed[indices]))
            rows.append(constraint_row(indices))
            lower.append(0.0)
            upper.append(float(max(1, fixed_count)))

    matrix = sparse.vstack(rows, format="csr")
    score = (
        pd.to_numeric(work[score_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    # Stable, tiny tie-break in candidate order; it cannot alter substantive scores.
    tie = np.arange(n, dtype=float) / max(n, 1)
    objective = -score + 1e-9 * tie
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0},
    )
    if not bool(result.success) or result.x is None:
        raise RuntimeError(
            f"balanced selection failed: status={result.status}, {result.message}"
        )
    chosen = np.flatnonzero(result.x > 0.5)
    if chosen.size != int(target_n):
        raise RuntimeError(f"solver selected {chosen.size}, expected {target_n}")
    selected = work.iloc[chosen].copy()
    selected = selected.sort_values(
        [score_col, "candidate_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    selected.insert(0, "visual_rank", np.arange(1, len(selected) + 1, dtype=int))
    return selected


__all__ = [
    "CORRELATION_BIN_EDGES",
    "CORRELATION_BIN_LABELS",
    "assign_correlation_bins",
    "combine_pair_quality_scores",
    "compute_final_visual_score",
    "compute_pre_visual_score",
    "compute_visual_score_v2",
    "display_map_features",
    "display_map_features_v2",
    "morphology_signature",
    "normalize_pair_independent_quantile",
    "normalize_pair_shared_max",
    "safe_corr",
    "select_balanced_candidates",
]
