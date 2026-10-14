from __future__ import annotations
from collections.abc import Mapping, Sequence, Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Final
from types import SimpleNamespace
from io import BytesIO
import itertools
import hashlib
import copy
import re
import xml.etree.ElementTree as ET
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MatplotlibPath
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from PIL import Image
from release_rendering.plotting import *
from release_rendering.statistics import *
from release_rendering.maze import build_maze, run_path_between
from release_rendering.types import FigureAssemblyResult, PanelResult, RunMode
from release_rendering.common import validate_output_request as _validate_output_request
from release_rendering.common import significance_text
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GRID_SIZE = 15
CANONICAL_MAZE_DEPTH = 6
REWARD_NODE = 94
TARGET_NEURONS = 181
COORDINATED_DRAWS = 50
DECODER_CONTROL_BINS = (91, 121, 133)
DECODER_TARGET_BIN = 103


NEURAL_ORDER: Final = (
    "reward_decoding_error",
    "whole_maze_decoding_error",
    "place_map_correlation",
    "level_map_correlation",
    "population_vector_correlation",
    "ensemble_rate_correlation",
)


NEURAL_LABEL: Final = {
    "reward_decoding_error": "Reward decoding error",
    "whole_maze_decoding_error": "Whole maze decoding error",
    "place_map_correlation": "Place map correlation",
    "level_map_correlation": "Level map correlation",
    "population_vector_correlation": "PV correlation",
    "ensemble_rate_correlation": "Ensemble rate correlation",
}


BEHAVIOR_ORDER: Final = (
    "reward_approach_speed_advantage",
    "first_long_bout_time_s",
    "exploration_efficiency",
)


BEHAVIOR_LABEL: Final = {
    "reward_approach_speed_advantage": "Reward approach\nspeed advantage",
    "first_long_bout_time_s": "First long\nbout time",
    "exploration_efficiency": "Exploration\nefficiency",
}


POLICY_ORDER: Final = (
    "P_SF",
    "P_SA",
    "P_BF",
    "P_BS_depth_weighted",
)


POLICY_LABEL: Final = {
    "P_SF": r"$\mathrm{P}_{\mathrm{SF}}$",
    "P_SA": r"$\mathrm{P}_{\mathrm{SA}}$",
    "P_BF": r"$\mathrm{P}_{\mathrm{BF}}$",
    "P_BS_depth_weighted": r"Depth-weighted $\mathrm{P}_{\mathrm{BS}}$",
}


def calculate_neural_behavior_correlations(rows: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson effects and animal clustered tests for the 6 x 3 matrix."""

    records: list[dict[str, object]] = []
    for neural_index, neural in enumerate(NEURAL_ORDER):
        for behavior_index, behavior in enumerate(BEHAVIOR_ORDER):
            x = rows[neural].to_numpy(dtype=float)
            y = rows[behavior].to_numpy(dtype=float)
            result = stats.pearsonr(x, y)
            fit = cr1_ols(y, np.column_stack((np.ones(len(x)), x)), rows["animal_id"])
            contrast = linear_contrast(fit, (0.0, 1.0))
            records.append(
                {
                    "record_type": "correlation",
                    "neural_readout": neural,
                    "neural_label": NEURAL_LABEL[neural],
                    "neural_index": neural_index,
                    "behavior_measure": behavior,
                    "behavior_label": BEHAVIOR_LABEL[behavior].replace("\n", " "),
                    "behavior_index": behavior_index,
                    "pearson_r": float(result.statistic),
                    "nominal_p_descriptive": float(result.pvalue),
                    "cr1_slope": contrast.estimate,
                    "cr1_se": contrast.standard_error,
                    "cr1_t": contrast.t_value,
                    "cr1_df": contrast.degrees_of_freedom,
                    "cr1_p_two_sided": contrast.p_two_sided,
                    "n_pairs": int(len(rows)),
                    "n_animals": int(rows["animal_id"].nunique()),
                }
            )
    summary = pd.DataFrame.from_records(records)
    # Each row tests three behavior outcomes for one neural measure.
    summary["cr1_bh3_adjusted_p"] = summary.groupby(
        "neural_readout", sort=False
    )["cr1_p_two_sided"].transform(benjamini_hochberg)
    summary["cr1_bh3_star"] = summary["cr1_bh3_adjusted_p"].map(
        lambda p_value: significance_stars(
            p_value,
            thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")),
            nonsignificant="",
        )
    )
    pair_records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        for neural in NEURAL_ORDER:
            for behavior in BEHAVIOR_ORDER:
                pair_records.append(
                    {
                        "record_type": "pair_value",
                        "pair_key": row.pair_key,
                        "animal_id": row.animal_id,
                        "neural_readout": neural,
                        "behavior_measure": behavior,
                        "neural_value": float(getattr(row, neural)),
                        "behavior_value": float(getattr(row, behavior)),
                        "matched_neurons_per_draw": TARGET_NEURONS,
                        "coordinated_draws": COORDINATED_DRAWS,
                    }
                )
    return pd.concat(
        [summary, pd.DataFrame.from_records(pair_records)],
        ignore_index=True,
        sort=False,
    )


def _r_squared(outcome: FloatArray, predictors: Sequence[FloatArray]) -> float:
    """Return ordinary least-squares R-squared with an intercept."""

    y = np.asarray(outcome, dtype=float)
    columns = [
        np.ones(len(y)),
        *[np.asarray(value, dtype=float) for value in predictors],
    ]
    design = np.column_stack(columns)
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    total = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - fitted) ** 2)) / total


def calculate_commonality_components(rows: pd.DataFrame) -> pd.DataFrame:
    """Partition pooled two predictor R squared for both behavior outcomes."""

    reward = rows["reward_decoding_error"].to_numpy(dtype=float)
    level = rows["level_map_correlation"].to_numpy(dtype=float)
    records: list[dict[str, object]] = []
    for outcome, label in (
        ("first_long_bout_time_s", "First long bout time"),
        ("exploration_efficiency", "Exploration efficiency"),
    ):
        y = rows[outcome].to_numpy(dtype=float)
        r2_reward = _r_squared(y, (reward,))
        r2_level = _r_squared(y, (level,))
        r2_full = _r_squared(y, (reward, level))
        components = (
            ("reward_decoder_unique", r2_full - r2_level),
            ("shared", r2_reward + r2_level - r2_full),
            ("level_map_unique", r2_full - r2_reward),
        )
        for component_index, (component, value) in enumerate(components):
            records.append(
                {
                    "record_type": "commonality_component",
                    "outcome": outcome,
                    "outcome_label": label,
                    "component": component,
                    "component_index": component_index,
                    "explained_variance_fraction": value,
                    "explained_variance_percent": 100.0 * value,
                    "full_model_r_squared": r2_full,
                    "n_pairs": len(rows),
                    "n_animals": rows["animal_id"].nunique(),
                }
            )
    pair_values = rows[
        [
            "pair_key",
            "animal_id",
            "reward_decoding_error",
            "level_map_correlation",
            "first_long_bout_time_s",
            "exploration_efficiency",
        ]
    ].copy()
    pair_values.insert(0, "record_type", "pair_value")
    return pd.concat(
        [pd.DataFrame.from_records(records), pair_values],
        ignore_index=True,
        sort=False,
    )


def calculate_neural_behavior_profile_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """Test the three captioned reward-versus-level profiles by animal sign flips.

    Reward error and long-bout time are oriented toward better performance.
    The exact null flips the aggregated residual score of each animal; the
    three resulting two-sided p values receive one BH3 correction.
    """
    animals = rows["animal_id"].astype(str).to_numpy()
    animal_order = sorted(set(animals))
    n_animals = len(animal_order)
    if not 2 <= n_animals <= 20:
        raise ValueError("Exact profile tests require 2 to 20 animals")
    reward = _zscore(_zscore(-rows["reward_decoding_error"].to_numpy(dtype=float)))
    level = _zscore(_zscore(rows["level_map_correlation"].to_numpy(dtype=float)))
    early = _zscore(_zscore(-rows["first_long_bout_time_s"].to_numpy(dtype=float)))
    exploration = _zscore(_zscore(rows["exploration_efficiency"].to_numpy(dtype=float)))

    def animal_scores(
        outcome: FloatArray, target: FloatArray, nuisance: FloatArray,
        group: NDArray[np.str_],
    ) -> FloatArray:
        y_residual = outcome - nuisance @ np.linalg.lstsq(nuisance, outcome, rcond=None)[0]
        x_residual = target - nuisance @ np.linalg.lstsq(nuisance, target, rcond=None)[0]
        products = y_residual * x_residual
        return np.asarray(
            [float(products[group == animal].sum()) for animal in animal_order],
            dtype=float,
        )

    x = np.concatenate((reward, level))
    code_level = np.concatenate((np.zeros(len(rows)), np.ones(len(rows))))
    direct_design = np.column_stack((np.ones(len(x)), x, code_level))
    scores = []
    for outcome in (early, exploration):
        scores.append(animal_scores(
            np.concatenate((outcome, outcome)), -x * code_level,
            direct_design, np.tile(animals, 2),
        ))
    y = np.concatenate((early, early, exploration, exploration))
    x4 = np.tile(x, 2)
    code4 = np.tile(code_level, 2)
    behavior = np.concatenate((np.zeros(2 * len(rows)), np.ones(2 * len(rows))))
    target = x4 * code4 * behavior
    nuisance = np.column_stack((
        np.ones(len(y)), x4, code4, behavior, x4 * code4,
        x4 * behavior, code4 * behavior,
    ))
    scores.append(animal_scores(y, target, nuisance, np.tile(animals, 4)))
    score = np.vstack(scores)
    norm = np.sqrt(np.sum(score * score, axis=1, keepdims=True))
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Degenerate animal-score profile contrast")
    standardized = score / norm
    values = np.arange(2 ** n_animals, dtype=np.uint32)[:, None]
    signs = 2.0 * ((values >> np.arange(n_animals, dtype=np.uint32)) & 1) - 1.0
    permuted = signs @ standardized.T
    observed = standardized.sum(axis=1)
    raw_p = np.mean(np.abs(permuted) >= np.abs(observed)[None, :] - 1e-15, axis=0)
    early_difference = float(stats.pearsonr(reward, early).statistic - stats.pearsonr(level, early).statistic)
    exploration_difference = float(stats.pearsonr(reward, exploration).statistic - stats.pearsonr(level, exploration).statistic)
    return pd.DataFrame({
        "record_type": "profile_inference",
        "claim": [
            "first_long_reward_minus_level", "exploration_reward_minus_level",
            "reward_level_profile_crossover",
        ],
        "estimate": [early_difference, exploration_difference, early_difference - exploration_difference],
        "score_t": observed,
        "exact_p_two_sided": raw_p,
        "bh3_adjusted_p": benjamini_hochberg(raw_p),
        "correction_family": "BH across three captioned pooled profile contrasts",
        "n_patterns": len(signs),
        "n_pairs": len(rows),
        "n_animals": n_animals,
    })


def _zscore(values: FloatArray) -> FloatArray:
    """Standardize finite values using the pooled population standard deviation."""

    values = np.asarray(values, dtype=float)
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot standardize a constant or non-finite vector")
    return (values - float(np.mean(values))) / scale


def _cluster_slope(
    frame: pd.DataFrame, predictor: str, outcome: str
) -> dict[str, float | int]:
    """Fit the pooled standardized slope with animal-clustered CR1 inference."""

    use = (
        frame[["animal_id", predictor, outcome]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    x = _zscore(use[predictor].to_numpy(dtype=float))
    y = _zscore(use[outcome].to_numpy(dtype=float))
    fit = cr1_ols(
        y,
        np.column_stack((np.ones(len(use)), x)),
        use["animal_id"].astype(str).tolist(),
    )
    contrast = linear_contrast(fit, np.asarray([0.0, 1.0]))
    return {
        "cluster_beta_standardized": contrast.estimate,
        "cluster_se": contrast.standard_error,
        "cluster_t": contrast.t_value,
        "cluster_df": contrast.degrees_of_freedom,
        "cluster_p": contrast.p_two_sided,
        "n_pairs": fit.n_observations,
        "n_animals": fit.n_clusters,
    }


def calculate_policy_inference(
    points: pd.DataFrame,
    run_mode: RunMode,
) -> pd.DataFrame:
    """Calculate policy Pearson r and animal clustered inference."""

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    rows: list[dict[str, object]] = []
    policy_columns = {
        "P_SF": "P_SF",
        "P_SA": "P_SA",
        "P_BF": "P_BF",
        "P_BS_depth_weighted": "P_BS_depth_weighted",
    }
    for key, column in policy_columns.items():
        result = stats.pearsonr(
            points["level_map_correlation"].to_numpy(dtype=float),
            points[column].to_numpy(dtype=float),
        )
        cluster = _cluster_slope(points, "level_map_correlation", column)
        rows.append(
            {
                "policy_key": key,
                "policy_label": POLICY_LABEL[key],
                "pearson_r": float(result.statistic),
                "pearson_p_descriptive": float(result.pvalue),
                **cluster,
                "adjusted_p": np.nan,
                "correction_family": "",
            }
        )
    inference = pd.DataFrame.from_records(rows)

    # The family is exactly the four displayed first-30-minute associations.
    inference["adjusted_p"] = benjamini_hochberg(
        inference["cluster_p"].to_numpy(dtype=float)
    )
    inference["correction_family"] = (
        "BH across four displayed first-30-minute turning-policy associations"
    )
    return inference


FloatArray = NDArray[np.float64]


_ARTIFACT_KEY: Final = "extended_data_figure_10"


_POLICY_COLOR: Final = "#7976A2"


_ROLE_HEIGHTS_720: Final = {"a": 261.62, "b": 167.97, "c": 202.21}


_ROLE_GAPS_720: Final = {"a_b": 17.52 / 4.0, "b_c": 20.83 / 2.0}


_CONTENT_HEIGHT_720: Final = sum(_ROLE_HEIGHTS_720.values()) + sum(
    _ROLE_GAPS_720.values()
)


_CONTENT_MARGIN_720: Final = (720.0 - _CONTENT_HEIGHT_720) / 2.0


_ROLE_TOPS_720: Final = {
    "a": _CONTENT_MARGIN_720,
    "b": (
        _CONTENT_MARGIN_720
        + _ROLE_HEIGHTS_720["a"]
        + _ROLE_GAPS_720["a_b"]
    ),
    "c": (
        _CONTENT_MARGIN_720
        + _ROLE_HEIGHTS_720["a"]
        + _ROLE_GAPS_720["a_b"]
        + _ROLE_HEIGHTS_720["b"]
        + _ROLE_GAPS_720["b_c"]
    ),
}


_ASSEMBLY_RECTS: Final = {
    role: (
        17.28 / 720.0,
        (720.0 - _ROLE_TOPS_720[role] - _ROLE_HEIGHTS_720[role]) / 720.0,
        685.44 / 720.0,
        _ROLE_HEIGHTS_720[role] / 720.0,
    )
    for role in ("a", "b", "c")
}


_DISPLAY_STYLE: Final = {
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
}


def _neural_behavior_matrix(values: pd.DataFrame) -> FloatArray:
    """Arrange the 18 calculated correlations in their displayed order."""

    summary = values.loc[values["record_type"].eq("correlation")]
    matrix = np.full((len(NEURAL_ORDER), len(BEHAVIOR_ORDER)), np.nan)
    for row in summary.itertuples(index=False):
        matrix[int(row.neural_index), int(row.behavior_index)] = float(row.pearson_r)
    if not np.isfinite(matrix).all():
        raise ValueError("The neural and behavior correlation matrix is incomplete")
    return matrix


def _inside(role: Sequence[float], local: Sequence[float]) -> list[float]:
    """Convert coordinates within one figure role to canvas coordinates."""

    return [
        role[0] + local[0] * role[2],
        role[1] + local[1] * role[3],
        local[2] * role[2],
        local[3] * role[3],
    ]


def _write_role_label(figure: Figure, role: Sequence[float], label: str) -> None:
    """Place a publication panel label at the accepted role origin."""

    figure.text(
        role[0] + 0.006 * role[2],
        role[1] + 0.972 * role[3],
        label,
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _draw_neural_behavior_correlations(
    figure: Figure,
    role: Sequence[float],
    values: pd.DataFrame,
    *,
    label: str,
) -> list[plt.Axes]:
    """Draw the accepted matrix and continuous correlation key."""

    # Display magnitudes; saved coefficients and two-sided inference stay signed.
    matrix = np.abs(_neural_behavior_matrix(values))
    axis = figure.add_axes(_inside(role, (0.205, 0.20, 0.690, 0.69)))
    color_axis = figure.add_axes(_inside(role, (0.910, 0.20, 0.018, 0.69)))
    normalization = mpl.colors.Normalize(vmin=0.0, vmax=0.8)
    magnitude_cmap = mpl.colors.ListedColormap(
        mpl.colormaps["RdBu_r"](np.linspace(0.5, 1.0, 256)),
        name="absolute_pearson",
    )
    axis.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap=magnitude_cmap,
        norm=normalization,
        shading="flat",
        rasterized=False,
    )
    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    axis.set_xticks(
        range(len(BEHAVIOR_ORDER)),
        [BEHAVIOR_LABEL[key] for key in BEHAVIOR_ORDER],
    )
    axis.set_yticks(
        range(len(NEURAL_ORDER)),
        [NEURAL_LABEL[key] for key in NEURAL_ORDER],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="black",
                fontweight="normal",
            )
            cell = values.loc[
                values["record_type"].eq("correlation")
                & values["neural_index"].eq(row)
                & values["behavior_index"].eq(column)
            ]
            if len(cell) != 1:
                raise ValueError("Each neural and behavior combination needs one test")
            star = str(cell["cr1_bh3_star"].iloc[0])
            if star:
                axis.annotate(
                    star,
                    xy=(column, row),
                    xytext=(12.0, 3.0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold",
                    annotation_clip=True,
                )
    axis.tick_params(length=0, pad=4)
    for spine in axis.spines.values():
        spine.set_visible(False)

    edges = np.linspace(0.0, 0.8, 129)
    centers = ((edges[:-1] + edges[1:]) / 2.0)[:, None]
    color_axis.pcolormesh(
        (0, 1),
        edges,
        centers,
        cmap=magnitude_cmap,
        norm=normalization,
        shading="flat",
        rasterized=False,
    )
    color_axis.set_xlim(0, 1)
    color_axis.set_ylim(0.0, 0.8)
    color_axis.set_xticks([])
    color_axis.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
    color_axis.yaxis.tick_right()
    color_axis.yaxis.set_label_position("right")
    color_axis.set_ylabel("Absolute Pearson correlation")
    _write_role_label(figure, role, label)
    return [axis, color_axis]


def _draw_commonality_components(
    figure: Figure,
    role: Sequence[float],
    values: pd.DataFrame,
    *,
    label: str,
) -> list[plt.Axes]:
    """Draw the two accepted horizontal commonality bars."""

    components = values.loc[values["record_type"].eq("commonality_component")]
    colors = {
        "reward_decoder_unique": "#E29957",
        "shared": "#7F7F7F",
        "level_map_unique": "#7976A2",
    }
    labels = {
        "reward_decoder_unique": "Reward decoder unique",
        "shared": "Shared",
        "level_map_unique": "Level map unique",
    }
    outcomes = ("first_long_bout_time_s", "exploration_efficiency")
    axes = [
        figure.add_axes(_inside(role, (0.070, 0.23, 0.405, 0.54))),
        figure.add_axes(_inside(role, (0.555, 0.23, 0.405, 0.54))),
    ]
    for axis, outcome in zip(axes, outcomes, strict=True):
        selected = components.loc[components["outcome"].eq(outcome)].sort_values(
            "component_index"
        )
        if len(selected) != 3:
            raise ValueError(f"Expected three commonality components for {outcome}")
        left = 0.0
        for row in selected.itertuples(index=False):
            width = float(row.explained_variance_percent)
            color = colors[str(row.component)]
            axis.barh(
                [0],
                [width],
                left=[left],
                color=color,
                height=0.52,
                edgecolor="white",
                linewidth=0.5,
            )
            if width >= 6.0:
                axis.text(
                    left + width / 2.0,
                    0,
                    f"{width:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="black" if row.component == "shared" else "white",
                    fontweight="bold",
                )
            left += width
        axis.set_title(
            str(selected["outcome_label"].iloc[0]), fontweight="bold", pad=2
        )
        axis.set_xlim(0, 60)
        axis.set_yticks([])
        axis.set_xlabel("Explained variance (%)")
        despine(axis)
        axis.tick_params(pad=3.5)
    handles = [
        mpl.patches.Patch(facecolor=colors[key], edgecolor="none", label=labels[key])
        for key in ("reward_decoder_unique", "shared", "level_map_unique")
    ]
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.34),
        frameon=False,
        ncol=3,
        handlelength=1.0,
        columnspacing=0.8,
    )
    _write_role_label(figure, role, label)
    return axes


def _point_label_clearance(
    points: FloatArray,
    box: mpl.transforms.Bbox,
    marker_radius: float,
) -> float:
    """Return the shortest display-space gap from points to a text box."""

    horizontal = np.maximum.reduce(
        (box.x0 - points[:, 0], np.zeros(len(points)), points[:, 0] - box.x1)
    )
    vertical = np.maximum.reduce(
        (box.y0 - points[:, 1], np.zeros(len(points)), points[:, 1] - box.y1)
    )
    return float(np.min(np.hypot(horizontal, vertical)) - marker_radius)


def _add_correlation_label(
    axis: plt.Axes,
    x: FloatArray,
    y: FloatArray,
    correlation: float,
) -> None:
    """Place the descriptive correlation clear of the plotted points."""

    candidates = (
        (0.94, 0.08),
        (0.94, 0.16),
        (0.82, 0.08),
        (0.82, 0.16),
        (0.94, 0.24),
    )
    radius = (np.sqrt(10.0) / 2.0) * axis.figure.dpi / 72.0
    best: tuple[float, mpl.text.Text] | None = None
    for floor_extensions in range(7):
        if floor_extensions:
            best[1].remove()
            best = None
            bottom, top = axis.get_ylim()
            axis.set_ylim(bottom - 0.05 * (top - bottom), top)
        points = axis.transData.transform(np.column_stack((x, y)))
        for x_position, y_position in candidates:
            artist = axis.text(
                x_position,
                y_position,
                f"r={correlation:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.2,
                color="black",
                zorder=5,
            )
            axis.figure.canvas.draw()
            clearance = _point_label_clearance(
                points,
                artist.get_window_extent(axis.figure.canvas.get_renderer()),
                radius,
            )
            if best is None or clearance > best[0]:
                if best is not None:
                    best[1].remove()
                best = (clearance, artist)
            else:
                artist.remove()
            if clearance >= 2.0:
                break
        if best is not None and best[0] >= 2.0:
            break


def _draw_policy_associations(
    figure: Figure,
    role: Sequence[float],
    values: pd.DataFrame,
    run_mode: RunMode,
    *,
    label: str,
) -> list[plt.Axes]:
    """Draw the accepted four policy association axes."""

    points = values.loc[values["record_type"].eq("point")]
    inference = values.loc[values["record_type"].eq("inference")]
    fits = values.loc[values["record_type"].eq("fit")]
    left, right, gap = 0.090, 0.985, 0.080
    width = (right - left - 3.0 * gap) / 4.0
    axes = [
        figure.add_axes(
            _inside(role, (left + index * (width + gap), 0.245, width, 0.640))
        )
        for index in range(4)
    ]
    for axis, policy in zip(axes, POLICY_ORDER, strict=True):
        selected = points.loc[points["policy_key"].eq(policy)]
        x = selected["level_map_correlation"].to_numpy(dtype=float)
        y = selected["policy_value"].to_numpy(dtype=float)
        axis.scatter(
            x,
            y,
            s=10,
            color=_POLICY_COLOR,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        selected_fit = fits.loc[fits["policy_key"].eq(policy)]
        grid = selected_fit["level_map_correlation"].to_numpy(dtype=float)
        fitted = selected_fit["fitted_policy_value"].to_numpy(dtype=float)
        lower = selected_fit["lower_95"].to_numpy(dtype=float)
        upper = selected_fit["upper_95"].to_numpy(dtype=float)
        if np.isfinite(lower).any() and np.isfinite(upper).any():
            axis.fill_between(
                grid,
                lower,
                upper,
                color=_POLICY_COLOR,
                alpha=0.16,
                linewidth=0,
            )
        axis.plot(grid, fitted, color=_POLICY_COLOR, linewidth=0.9)
        axis.set_xlim(-0.05, 0.55)
        axis.set_xticks((0.0, 0.2, 0.4))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        result = inference.loc[inference["policy_key"].eq(policy)].iloc[0]
        if run_mode != "full":
            raise ValueError("run_mode must be 'full'")
        adjusted = float(result["adjusted_p"])
        if not np.isfinite(adjusted):
            raise ValueError("The four displayed policies require finite BH4-adjusted p values")
        p_value = adjusted
        call = significance_stars(
            p_value,
            thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")),
            nonsignificant="n.s.",
        )
        if call != "n.s.":
            bottom, top = axis.get_ylim()
            axis.set_ylim(bottom, max(top, float(np.max(y)) + 0.20 * (top - bottom)))
        annotation = axis.text(
            0.08,
            0.92,
            call,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
            fontweight="normal" if call == "n.s." else "bold",
            color="black",
            zorder=5,
        )
        marker_radius = (np.sqrt(10.0) / 2.0) * axis.figure.dpi / 72.0
        for _ in range(6):
            axis.figure.canvas.draw()
            clearance = _point_label_clearance(
                axis.transData.transform(np.column_stack((x, y))),
                annotation.get_window_extent(
                    axis.figure.canvas.get_renderer()
                ),
                marker_radius,
            )
            if clearance >= 2.0:
                break
            bottom, top = axis.get_ylim()
            axis.set_ylim(bottom, top + 0.05 * (top - bottom))
        if call == "n.s.":
            annotation.set_fontsize(8.0)
        axis.set_ylabel(POLICY_LABEL[policy])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        _add_correlation_label(axis, x, y, float(result["pearson_r"]))
        despine(axis)
        axis.tick_params(pad=3.5)
    figure.text(
        role[0] + 0.53 * role[2],
        role[1] + 0.060 * role[3],
        "Level map correlation",
        fontsize=8.0,
        ha="center",
        va="bottom",
    )
    _write_role_label(figure, role, label)
    return axes


def _save_rgb_raster_exports(output_paths: Sequence[Path]) -> None:
    """Store final PNG and TIFF pages in the version of record RGB mode."""

    for path in output_paths:
        suffix = path.suffix.casefold()
        if suffix not in {".png", ".tiff"}:
            continue
        # Decode from memory so Pillow cannot retain a Windows file handle on
        # the export that is replaced below.
        with Image.open(BytesIO(path.read_bytes())) as source:
            image = source.convert("RGB")
        candidate = path.with_name(f"{path.name}.rgb")
        candidate.unlink(missing_ok=True)
        try:
            if suffix == ".tiff":
                image.save(
                    candidate,
                    format="TIFF",
                    compression="tiff_lzw",
                    dpi=(600, 600),
                )
            else:
                image.save(candidate, format="PNG", dpi=(600, 600))
            candidate.replace(path)
        finally:
            candidate.unlink(missing_ok=True)


def assemble_neural_behavior_controls(
    neural_behavior_values: pd.DataFrame,
    commonality_values: pd.DataFrame,
    policy_values: pd.DataFrame,
    *,
    run_mode: RunMode,
    output_dir: Path,
    random_seed: int,
) -> FigureAssemblyResult:
    """Render the three calculated results in the accepted page geometry."""

    destination = _validate_output_request(
        run_mode=run_mode,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    with version_record_style(_DISPLAY_STYLE):
        figure = plt.figure(figsize=(7.5, 7.5))
        _draw_neural_behavior_correlations(
            figure,
            _ASSEMBLY_RECTS["a"],
            neural_behavior_values,
            label="a",
        )
        _draw_commonality_components(
            figure,
            _ASSEMBLY_RECTS["b"],
            commonality_values,
            label="b",
        )
        _draw_policy_associations(
            figure,
            _ASSEMBLY_RECTS["c"],
            policy_values,
            run_mode,
            label="c",
        )
        output_paths = save_figure(
            figure,
            destination / "Extended_Data_Figure_10",
            formats=("svg", "pdf", "png", "tiff"),
            raster_dpi=600,
            preserve_canvas=True,
        )
        _save_rgb_raster_exports(output_paths)
        plt.close(figure)
    root = REPOSITORY_ROOT.resolve()
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT_KEY,
        run_mode=run_mode,
        output_paths=tuple(path.resolve() for path in output_paths),
    )
