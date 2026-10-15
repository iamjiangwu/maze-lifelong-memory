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
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
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


EXPERIENCE_GROUP_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")


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


LEVEL_COUNT = CANONICAL_MAZE_DEPTH + 1


def summarize_preferred_level_inference(
    pair_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Test the two planned preferred-level effects with one BH2 family.

    Each effect is calculated once per session pair. The intercept-only model
    uses animal-clustered CR1 uncertainty and a one-sided greater-than-zero
    test, matching the accepted figure record.
    """

    required = {"animal_id", "delta_adjacent", "shorter_shift_distance"}
    missing = sorted(required.difference(pair_summaries.columns))
    if missing:
        raise ValueError(f"Preferred-level pair summaries are missing columns: {missing}")
    definitions = (
        (
            "adjacent_shift_enrichment",
            "delta_adjacent",
            "observed adjacent-shift fraction minus within-pair destination-ID null",
        ),
        (
            "shorter_shift_distance",
            "shorter_shift_distance",
            "null mean shift distance minus observed mean shift distance",
        ),
    )
    rows: list[dict[str, object]] = []
    for metric, value_column, effect_definition in definitions:
        values = pair_summaries[value_column].to_numpy(dtype=float)
        fit = cr1_ols(
            values,
            np.ones((len(pair_summaries), 1), dtype=float),
            pair_summaries["animal_id"].astype(str).tolist(),
        )
        contrast = linear_contrast(fit, [1.0])
        p_directional = float(
            stats.t.sf(contrast.t_value, df=contrast.degrees_of_freedom)
        )
        rows.append(
            {
                "record_type": "statistic",
                "metric": metric,
                "effect_definition": effect_definition,
                "estimate": contrast.estimate,
                "se_cr1": contrast.standard_error,
                "ci95_low": contrast.confidence_low,
                "ci95_high": contrast.confidence_high,
                "t": contrast.t_value,
                "cluster_df": contrast.degrees_of_freedom,
                "p_directional": p_directional,
                "p_two_sided": contrast.p_two_sided,
                "n_pairs": fit.n_observations,
                "n_mice": fit.n_clusters,
                "covariance": "CR1 intercept-only; Student t(G-1)",
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["p_bh2"] = benjamini_hochberg(
        statistics["p_directional"].to_numpy(dtype=float)
    )
    return statistics


@lru_cache(maxsize=2)
def dumbbell_grid_labels(representation: str) -> IntArray:
    """Return one direct 8-bin horizontal or vertical control partition."""

    mappings = {
        "horizontal_8": {
            0: (0, 1),
            1: (4, 5),
            2: (2, 3),
            3: (6, 7),
            4: (8, 9),
            5: (12, 13),
            6: (10, 11),
            7: (14, 15),
        },
        "vertical_8": {
            0: (0, 4),
            1: (1, 5),
            2: (2, 6),
            3: (3, 7),
            4: (8, 12),
            5: (9, 13),
            6: (10, 14),
            7: (11, 15),
        },
    }
    if representation not in mappings:
        raise KeyError(representation)
    dumbbell = np.full(GRID_SIZE * GRID_SIZE, -1, dtype=np.int64)
    connectors = {3, 7, 11}
    for by in range(GRID_SIZE):
        for bx in range(GRID_SIZE):
            if bx in connectors or by in connectors:
                continue
            gx = 0 if bx <= 2 else 1 if bx <= 6 else 2 if bx <= 10 else 3
            gy = 0 if by <= 2 else 1 if by <= 6 else 2 if by <= 10 else 3
            dumbbell[bx + GRID_SIZE * by] = gx + 4 * gy
    output = np.full_like(dumbbell, -1)
    for category, members in mappings[representation].items():
        output[np.isin(dumbbell, np.asarray(members, dtype=np.int64))] = category
    return output


def spatial_null_statistic(data: SpatialNullData) -> pd.DataFrame:
    """Test the observed-minus-spatial-null pair difference with animal clusters."""

    observed = data.pairs["observed_pair_mean"].to_numpy(dtype=float)
    shuffled = data.pairs["shuffle_pair_mean"].to_numpy(dtype=float)
    statistic: dict[str, object] = {
        "record_type": "statistic",
        "estimate_observed_minus_shuffle": mean_of_finite_values(
            observed - shuffled
        ),
        "se_cr1": np.nan,
        "t_value": np.nan,
        "degrees_of_freedom": 0,
        "p_greater": np.nan,
        "n_pairs": len(data.pairs),
        "n_animals": data.pairs["animal_id"].nunique(),
    }
    if len(data.pairs) >= 2 and data.pairs["animal_id"].nunique() >= 2:
        fit = cr1_ols(
            observed - shuffled,
            np.ones((len(data.pairs), 1), dtype=float),
            data.pairs["animal_id"].astype(str).tolist(),
        )
        contrast = linear_contrast(fit, [1.0])
        statistic.update(
            {
                "estimate_observed_minus_shuffle": contrast.estimate,
                "se_cr1": contrast.standard_error,
                "t_value": contrast.t_value,
                "degrees_of_freedom": contrast.degrees_of_freedom,
                "p_greater": float(
                    stats.t.sf(contrast.t_value, df=contrast.degrees_of_freedom)
                ),
            }
        )
    return pd.DataFrame([statistic])


def summarize_eight_bin_inference(wide: pd.DataFrame) -> pd.DataFrame:
    """Test the two planned ordinary-Level-minus-axis contrasts with one BH2 family."""

    required = {
        "animal_id",
        "pair_key",
        "horizontal_8",
        "level_code",
        "vertical_8",
    }
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise ValueError(f"Eight-bin pair values are missing columns: {missing}")
    definitions = (
        (
            "level_minus_horizontal8",
            wide["level_code"].to_numpy(dtype=float)
            - wide["horizontal_8"].to_numpy(dtype=float),
        ),
        (
            "level_minus_vertical8",
            wide["level_code"].to_numpy(dtype=float)
            - wide["vertical_8"].to_numpy(dtype=float),
        ),
    )
    rows: list[dict[str, object]] = []
    for contrast_name, values in definitions:
        fit = cr1_ols(
            values,
            np.ones((len(wide), 1), dtype=float),
            wide["animal_id"].astype(str).tolist(),
        )
        contrast = linear_contrast(fit, [1.0])
        p_directional = float(
            stats.t.sf(contrast.t_value, df=contrast.degrees_of_freedom)
        )
        rows.append(
            {
                "record_type": "statistic",
                "contrast": contrast_name,
                "estimate": contrast.estimate,
                "se_cr1": contrast.standard_error,
                "ci95_low": contrast.confidence_low,
                "ci95_high": contrast.confidence_high,
                "t": contrast.t_value,
                "cluster_df": contrast.degrees_of_freedom,
                "p_directional_greater": p_directional,
                "p_two_sided": contrast.p_two_sided,
                "p_primary": p_directional,
                "alternative": "greater",
                "n_pairs": fit.n_observations,
                "n_animals": fit.n_clusters,
                "covariance": "CR1 intercept-only; Student t(G-1)",
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["p_bh2"] = benjamini_hochberg(
        statistics["p_primary"].to_numpy(dtype=float)
    )
    return statistics


def clustered_cycle_display(points: pd.DataFrame) -> pd.DataFrame:
    """Prepare the source-defined CR1 confidence band for a cycle association."""
    x=points['experience'].to_numpy(float)
    y=points['value'].to_numpy(float)
    fit=cr1_ols(y,np.column_stack((np.ones(len(x)),x)),points['animal_id'].astype(str).tolist())
    grid=np.linspace(float(x.min()),float(x.max()),100)
    design=np.column_stack((np.ones(len(grid)),grid))
    estimate=design@fit.beta
    se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',design,fit.covariance,design),0))
    critical=stats.t.ppf(.975,fit.degrees_of_freedom)
    return pd.DataFrame({'record_type':'display_fit','x_value':grid,'estimate':estimate,
        'ci_low':estimate-critical*se,'ci_high':estimate+critical*se})


def summarize_decoder_accuracy(rows: pd.DataFrame) -> dict[str, float | int]:
    """Test observed-minus-shuffle accuracy with animal-clustered CR1 SEs."""

    if len(rows) < 2 or rows["animal_id"].nunique() < 2:
        return {
            "estimate": float("nan"),
            "standard_error": float("nan"),
            "t_value": float("nan"),
            "degrees_of_freedom": 0,
            "p_greater": float("nan"),
        }
    delta = rows["observed_accuracy"].to_numpy(dtype=float) - rows[
        "shuffle_accuracy"
    ].to_numpy(dtype=float)
    fit = cr1_ols(
        delta,
        np.ones((len(rows), 1), dtype=float),
        rows["animal_id"].astype(str).tolist(),
    )
    contrast = linear_contrast(fit, [1.0])
    return {
        "estimate": contrast.estimate,
        "standard_error": contrast.standard_error,
        "t_value": contrast.t_value,
        "degrees_of_freedom": contrast.degrees_of_freedom,
        "p_greater": float(
            stats.t.sf(
                contrast.t_value,
                df=contrast.degrees_of_freedom,
            )
        ),
    }


_LEVEL_COLOR = "#7976A2"


_PLACE_COLOR = "#B95A58"


_RECORD_DPI = 600


def _recurrence_adjusted_probability(statistic: pd.DataFrame) -> float:
    """Return the required BH4 probability for the single recurrence row."""

    if len(statistic) != 1 or "p_bh4" not in statistic.columns:
        raise ValueError("Recurrence requires exactly one statistic row with p_bh4")
    probability = float(statistic.iloc[0]["p_bh4"])
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise ValueError("Recurrence p_bh4 must be a finite probability")
    return probability


def _style_record_axis(axis: Axes) -> None:
    """Apply the restrained axis style used by the accepted figure record."""

    axis.grid(False, which="both")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(axis="both", labelsize=6.0, width=0.65, length=2.0, pad=1.5)
    axis.xaxis.label.set_fontsize(6.5)
    axis.yaxis.label.set_fontsize(6.5)


def _record_jitter(keys: pd.Series, width: float) -> FloatArray:
    """Return deterministic horizontal jitter without using a saved layout table."""

    offsets = []
    for key in keys.astype(str):
        raw = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        offsets.append(((raw % 10_001) / 10_000.0 - 0.5) * 2.0 * width)
    return np.asarray(offsets, dtype=float)


def _significance_marks(p_value: float) -> str:
    """Map a displayed P value to the manuscript's star notation."""

    if not np.isfinite(p_value) or p_value >= 0.05:
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def _draw_record_bracket(
    axis: Axes,
    left: float,
    right: float,
    height: float,
    text: str,
) -> None:
    """Center the unchanged comparison token in a compact physical-size gap."""
    bounds=axis.get_window_extent()
    width_mm,height_mm=np.array([bounds.width,bounds.height])/axis.figure.dpi*25.4
    low,high=axis.get_ylim()
    y=(height-low)/(high-low)
    weight="normal" if text=="n.s." else "bold"
    fontsize=6.5 if text=="n.s." else 10.0
    glyph=TextPath((0,0),text,size=fontsize,prop=FontProperties(family="Arial",weight=weight)).get_extents()
    mm_per_pt=25.4/72.0
    half_gap=(glyph.width*mm_per_pt/2+.25)/width_mm*np.ptp(axis.get_xlim())
    center=(left+right)/2
    cap=.20/height_mm
    key=f"paired_compact_{left:g}_{right:g}"
    axis.plot([left,left,center-half_gap],[y-cap,y,y],transform=axis.get_xaxis_transform(),
              color="black",lw=.6,clip_on=False,zorder=10,gid=key+"_left")
    axis.plot([center+half_gap,right,right],[y,y,y-cap],transform=axis.get_xaxis_transform(),
              color="black",lw=.6,clip_on=False,zorder=10,gid=key+"_right")
    baseline=y-(glyph.y0+glyph.height/2)*mm_per_pt/height_mm
    axis.text(center,baseline,text,transform=axis.get_xaxis_transform(),ha="center",va="baseline",
              fontsize=fontsize,fontweight=weight,color="black",clip_on=False,zorder=11,gid=key+"_text")


def _draw_record_partition(axis: Axes, representation: str, title: str) -> None:
    """Draw one eight-bin map using the accepted Figure 5 schematic grammar."""

    maze = build_maze(CANONICAL_MAZE_DEPTH)
    labels = dumbbell_grid_labels(representation)
    palette = plt.get_cmap("magma")(np.linspace(0.0, 1.0, 8))
    for (maze_x, maze_y), _cell in maze.coordinate_to_cell.items():
        # Both controls use an x-fast display grid.
        stored_bin = int(maze_x) + GRID_SIZE * int(maze_y)
        label = int(labels[stored_bin])
        if label < 0:
            continue
        axis.add_patch(
            Rectangle(
                (maze_x - 0.5, maze_y - 0.5),
                1.0,
                1.0,
                facecolor=palette[label],
                edgecolor="none",
                linewidth=0.0,
            )
        )
    row_scale = (930.0 / _RECORD_DPI) / 4.4
    axis.plot(
        maze.wall_xy[:, 0],
        maze.wall_xy[:, 1],
        color="black",
        linewidth=1.55 * row_scale,
        solid_capstyle="round",
    )
    axis.set_xlim(-0.65, 14.65)
    axis.set_ylim(-0.65, 14.65)
    axis.set_aspect("equal")
    axis.set_facecolor("white")
    axis.axis("off")
    axis.set_title(title, fontsize=6.2, pad=2.0)


def _add_support_encoded_line(
    axis: Axes,
    x_values: FloatArray,
    y_values: FloatArray,
    high_support: BoolArray,
    *,
    color: str,
    label: str,
) -> None:
    """Draw only segments whose two endpoints meet the display support cutoff."""

    high_segments: list[FloatArray] = []
    for index in range(max(len(x_values) - 1, 0)):
        segment = np.asarray(
            [
                [x_values[index], y_values[index]],
                [x_values[index + 1], y_values[index + 1]],
            ],
            dtype=float,
        )
        if not np.isfinite(segment).all():
            continue
        if high_support[index] and high_support[index + 1]:
            high_segments.append(segment)
    if high_segments:
        lines = LineCollection(
            high_segments,
            colors=[color],
            linewidths=1.15,
            alpha=1.0,
            zorder=3,
            label=label,
        )
        lines.set_capstyle("round")
        axis.add_collection(lines)


def _draw_preferred_level_transition_axis(
    axis: Axes,
    values: PreferenceData,
) -> AxesImage:
    """Draw the preferred level transition heat map used in the summary."""

    matrix = values.transition.pivot(
        index="target_preferred_level",
        columns="source_preferred_level",
        values="observed_minus_shuffle",
    ).to_numpy(dtype=float)
    np.fill_diagonal(matrix, np.nan)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("#F2F2F2")
    image = axis.imshow(
        matrix,
        origin="lower",
        cmap=colormap,
        norm=TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.1),
        interpolation="nearest",
        aspect="equal",
    )
    axis.set_xticks(range(LEVEL_COUNT), range(LEVEL_COUNT))
    axis.set_yticks(range(LEVEL_COUNT), range(LEVEL_COUNT))
    axis.set_xlabel("Day i level")
    axis.set_ylabel("Day i+90 level")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def _draw_preferred_level_distance_axis(
    axis: Axes,
    values: PreferenceData,
) -> None:
    """Draw the observed and neuron ID null shift curves used in the summary."""

    shown = values.distance.loc[
        values.distance["shift_distance"].between(1, 6)
    ].sort_values("shift_distance")
    x_values = shown["shift_distance"].to_numpy(dtype=float)
    observed_line, = axis.plot(
        x_values,
        shown["observed_fraction"],
        color=_LEVEL_COLOR,
        marker="o",
        markersize=3.0,
        linewidth=1.1,
        label="Observed",
        zorder=3,
    )
    shuffle_error = (
        shown["shuffle_ci_high"].to_numpy(dtype=float)
        - shown["shuffle_ci_low"].to_numpy(dtype=float)
    ) / (2.0 * 1.96)
    null_errors = axis.errorbar(
        x_values,
        shown["shuffle_fraction"],
        yerr=shuffle_error,
        color="#7F7F7F",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#7F7F7F",
        markersize=2.7,
        linewidth=0.9,
        linestyle="--",
        capsize=1.4,
        label="ID null",
        zorder=2,
    )
    axis.set_xticks(range(1, 7))
    axis.set_ylim(0.04, 0.365)
    axis.set_xlabel("Level shift distance")
    axis.xaxis.label.set_x(0.45625)
    axis.set_ylabel("Fraction of shifted cells")
    axis.legend(
        # Keep the plotted error bars; show only the series line in the key.
        handles=[observed_line, null_errors.lines[0]],
        labels=["Observed", "ID null"],
        frameon=False,
        fontsize=5.8,
        loc="upper right",
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.1,
    )
    _style_record_axis(axis)


def _draw_spatial_reassignment_axes(
    pair_axis: Axes,
    histogram_axis: Axes,
    values: SpatialNullData,
    display_summary: pd.DataFrame,
    statistic: pd.DataFrame,
) -> None:
    """Draw pair values and the 1,000 cohort null means used in the summary."""

    pairs = values.pairs
    for row in pairs.itertuples(index=False):
        pair_axis.plot(
            [0, 1],
            [row.shuffle_pair_mean, row.observed_pair_mean],
            color="#D9D9D9",
            linewidth=0.42,
            alpha=0.65,
            zorder=1,
        )
    jitter = _record_jitter(pairs["pair_key"], 0.04)
    pair_axis.scatter(
        jitter,
        pairs["shuffle_pair_mean"],
        s=7.5,
        facecolor="white",
        edgecolor="#7F7F7F",
        linewidth=0.55,
        alpha=0.76,
        zorder=2,
    )
    pair_axis.scatter(
        1 + jitter,
        pairs["observed_pair_mean"],
        s=7.5,
        color=_PLACE_COLOR,
        edgecolor="none",
        alpha=0.70,
        zorder=2,
    )
    summary_by_group = display_summary.set_index("display_group")
    required_groups = {"spatial_reassignment", "observed"}
    missing_groups = sorted(required_groups.difference(summary_by_group.index))
    if missing_groups:
        raise ValueError(f"Spatial display summary is missing groups: {missing_groups}")
    pair_means = [
        float(summary_by_group.loc["spatial_reassignment", "estimate"]),
        float(summary_by_group.loc["observed", "estimate"]),
    ]
    pair_axis.plot([0, 1], pair_means, color="#202020", linewidth=1.0, zorder=3)
    pair_axis.scatter(
        [0],
        [pair_means[0]],
        s=24,
        facecolor="white",
        edgecolor="#7F7F7F",
        linewidth=1.0,
        zorder=4,
    )
    pair_axis.scatter(
        [1],
        [pair_means[1]],
        s=24,
        color=_PLACE_COLOR,
        edgecolor="#202020",
        linewidth=0.35,
        zorder=4,
    )
    pair_axis.set_xticks([0, 1], ["Within-level\nshuffle", "Data"])
    pair_axis.set_xlim(-0.18, 1.18)
    pair_axis.set_ylim(-0.03, 0.60)
    pair_axis.set_ylabel("Place-map correlation")
    if len(statistic) != 1 or "p_greater" not in statistic.columns:
        raise ValueError("Spatial reassignment requires one saved p_greater value")
    spatial_test = statistic.iloc[0]
    _draw_record_bracket(
        pair_axis,
        0,
        1,
        0.49,
        _significance_marks(float(spatial_test["p_greater"])),
    )
    _style_record_axis(pair_axis)

    cohort_null = values.draws["cohort_shuffle_mean"].to_numpy(dtype=float)
    counts, _, _ = histogram_axis.hist(
        cohort_null,
        bins=45,
        color="#6F6F6F",
        alpha=0.78,
        edgecolor="none",
    )
    observed_mean = pair_means[1]
    histogram_axis.axvline(
        observed_mean,
        color=_PLACE_COLOR,
        linestyle=(0, (4, 2)),
        linewidth=1.35,
    )
    null_low = float(np.nanmin(cohort_null))
    null_high = float(np.nanmax(cohort_null))
    null_span = max(observed_mean - null_low, null_high - null_low)
    histogram_axis.set_xlim(
        null_low - 0.08 * null_span,
        observed_mean + 0.10 * null_span,
    )
    histogram_axis.set_ylim(0, max(1.0, 1.2 * float(np.max(counts))))
    histogram_axis.set_xlabel("Place-map correlation")
    histogram_axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    histogram_axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    histogram_axis.text(
        0.08,
        0.95,
        "Shuffle",
        transform=histogram_axis.transAxes,
        ha="left",
        va="top",
        color="#6F6F6F",
        fontweight="bold",
    )
    histogram_axis.text(
        0.86,
        0.78,
        "Data",
        transform=histogram_axis.transAxes,
        ha="right",
        va="center",
        color=_PLACE_COLOR,
        fontweight="bold",
    )
    _style_record_axis(histogram_axis)


def _draw_eight_bin_controls_axis(
    axis: Axes,
    wide: pd.DataFrame,
    display_summary: pd.DataFrame,
    adjusted: Mapping[str, float],
) -> None:
    """Draw the shared ordinary Level and axis controls."""

    order = ("horizontal_8", "level_code", "vertical_8")
    summary_by_representation = display_summary.set_index("representation")
    missing = sorted(set(order).difference(summary_by_representation.index))
    if missing:
        raise ValueError(f"Eight-bin display summary is missing rows: {missing}")
    for row in wide.itertuples(index=False):
        axis.plot(
            range(3),
            [float(getattr(row, name)) for name in order],
            color="#D9D9D9",
            linewidth=0.4,
            alpha=0.7,
            zorder=1,
        )
    colors = ("#7F7F7F", _LEVEL_COLOR, "#7F7F7F")
    offsets = _record_jitter(pd.Series(wide.index if "pair_key" not in wide else wide["pair_key"]), 0.09)
    for index, (name, color) in enumerate(zip(order, colors, strict=True)):
        values = wide[name].to_numpy(dtype=float)
        axis.scatter(
            np.full(values.size, index) + offsets,
            values,
            s=9,
            color=color,
            alpha=0.65,
            edgecolor="none",
            zorder=2,
        )
        axis.scatter(
            index,
            float(summary_by_representation.loc[name, "estimate"]),
            s=32,
            facecolor=color if name == "level_code" else "white",
            edgecolor=color,
            linewidth=1.0,
            zorder=4,
        )
    axis.set_xticks(
        range(3),
        ["Horizontal\n8-bin", "Level", "Vertical\n8-bin"],
    )
    axis.set_ylabel("Pair-mean r")
    axis.set_ylim(-0.23, 0.83)
    axis.set_xlim(-0.35, 2.35)
    axis.yaxis.set_major_locator(MaxNLocator(5))
    _draw_record_bracket(
        axis,
        0,
        1,
        0.62,
        _significance_marks(adjusted["level_minus_horizontal8"]),
    )
    _draw_record_bracket(
        axis,
        1,
        2,
        0.72,
        _significance_marks(adjusted["level_minus_vertical8"]),
    )
    _style_record_axis(axis)


def _draw_joint_neuron_axis(axis: Axes, values: pd.DataFrame) -> None:
    """Draw the shared raw joint neuron cloud."""

    shown = values.loc[
        values[["place_map_correlation", "level_map_correlation"]]
        .notna()
        .all(axis=1)
    ]
    axis.scatter(
        shown["place_map_correlation"],
        shown["level_map_correlation"],
        s=1.4,
        color=_LEVEL_COLOR,
        edgecolor="none",
        alpha=0.045,
        rasterized=True,
    )
    axis.plot([-1, 1], [-1, 1], "--", color="#A6A6A6", linewidth=0.7)
    axis.set_xlim(-0.65, 1.02)
    axis.set_ylim(-0.65, 1.02)
    axis.set_xticks([-0.5, 0.0, 0.5, 1.0])
    axis.set_yticks([-0.5, 0.0, 0.5, 1.0])
    axis.set_xlabel("Place map correlation")
    axis.set_ylabel("Level map correlation")
    _style_record_axis(axis)


def _draw_gap_axis(axis: Axes, values: pd.DataFrame) -> None:
    """Draw the shared recording gap summary on a linear day axis."""

    summary = values.loc[values["record_type"].eq("summary")].sort_values(
        "time_days"
    )
    x_values = summary["time_days"].to_numpy(dtype=float)
    axis.fill_between(
        x_values,
        summary["ci95_low"].to_numpy(dtype=float),
        summary["ci95_high"].to_numpy(dtype=float),
        color=_LEVEL_COLOR,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        x_values,
        summary["level_map_correlation"].to_numpy(dtype=float),
        color=_LEVEL_COLOR,
        marker="o",
        markersize=3.0,
        linewidth=1.2,
        label="Data",
    )
    axis.plot(
        x_values,
        summary["neuron_id_shuffle_mean"].to_numpy(dtype=float),
        color="#7F7F7F",
        marker="s",
        markerfacecolor="#7F7F7F",
        markersize=2.4,
        linewidth=0.75,
        linestyle=(0, (2, 1.4)),
        label="Shuffle",
    )
    axis.set_xlim(-8, 370)
    axis.set_ylim(-0.08, 0.84)
    axis.set_xticks([1, 90, 180, 270, 360], ["1", "90", "180", "270", "360"])
    axis.set_xlabel("Days apart")
    axis.set_ylabel("Correlation")
    axis.legend(
        frameon=False,
        fontsize=5.8,
        loc="upper right",
        handlelength=1.1,
        handletextpad=0.3,
        borderaxespad=0.1,
        labelspacing=0.2,
    )
    _style_record_axis(axis)


def _draw_recurrence_axis(
    axis: Axes,
    points: pd.DataFrame,
    display_trend: pd.DataFrame,
    adjusted_probability: float,
) -> None:
    """Draw the shared raw recurrence cloud and 13 bin mean trend."""

    x_values = points["previous_level_map_correlation"].to_numpy(dtype=float)
    y_values = points["next_level_map_correlation"].to_numpy(dtype=float)
    axis.scatter(
        x_values,
        y_values,
        s=4.0,
        color=_LEVEL_COLOR,
        edgecolor="none",
        alpha=0.035,
        rasterized=True,
    )
    axis.plot(
        display_trend["x_value"].to_numpy(dtype=float),
        display_trend["mean"].to_numpy(dtype=float),
        color=_LEVEL_COLOR,
        linewidth=1.3,
        zorder=3,
    )
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    axis.set_xticks([-1, 0, 1])
    axis.set_yticks([-1, 0, 1])
    axis.set_xlabel("Previous gap correlation")
    axis.set_ylabel("Next gap correlation")
    axis.text(
        0.08,
        0.91,
        _significance_marks(adjusted_probability),
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=13.0,
        fontweight="bold",
    )
    _style_record_axis(axis)


def _draw_within_cycle_depth_axis(
    axis: Axes,
    display_values: pd.DataFrame,
) -> None:
    """Draw the prepared smoothed and support encoded tree depth curves."""

    group_colors = dict(
        zip(
            EXPERIENCE_GROUP_ORDER,
            ("#BDB9D6", "#A19CC3", "#8A85B2", "#6F6B99", "#4E4A75"),
            strict=True,
        )
    )
    for group in EXPERIENCE_GROUP_ORDER:
        group_rows = display_values.loc[
            display_values["experience_group"].eq(group)
        ].sort_values("x_minutes", kind="stable")
        if group_rows.empty:
            continue
        x_values = group_rows["x_minutes"].to_numpy(dtype=float)
        mean = group_rows["display_mean"].to_numpy(dtype=float)
        error = group_rows["display_sem"].to_numpy(dtype=float)
        color = group_colors[group]
        axis.fill_between(
            x_values,
            mean - error,
            mean + error,
            where=group_rows["high_support"].to_numpy(dtype=bool),
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        _add_support_encoded_line(
            axis,
            x_values,
            mean,
            group_rows["high_support"].to_numpy(dtype=bool),
            color=color,
            label=group,
        )
    axis.set_xlim(25, 185)
    axis.set_ylim(0.10, 0.86)
    axis.set_xticks([50, 100, 150])
    axis.set_yticks([0.2, 0.4, 0.6, 0.8])
    axis.set_xlabel("Time in maze (min)")
    axis.set_ylabel("Level map correlation")
    axis.legend(
        frameon=False,
        fontsize=5.5,
        ncol=2,
        loc="upper left",
        handlelength=0.8,
        columnspacing=0.6,
        handletextpad=0.2,
        borderaxespad=0.05,
        labelspacing=0.2,
    )
    _style_record_axis(axis)


def _draw_experience_axis(
    axis: Axes,
    points: pd.DataFrame,
    display_fit: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw the shared experience trend, Pearson r, and BH4 mark."""

    x_values = points["experience"].to_numpy(dtype=float)
    y_values = points["value"].to_numpy(dtype=float)
    # Match the source's cycle colors and deterministic point separation.
    colors=('#BDB9D6','#A19CC3','#8A85B2','#6F6B99','#4E4A75')
    if 'pair_key' in points:
        keys=points['pair_key'].astype(str)
    else:
        # CSV columns with missing values load integral animal IDs as floats.
        # Canonicalize only the display-jitter key, so 418 and 418.0 separate
        # points identically without changing any observation or statistic.
        animal_keys=points['animal_id'].map(
            lambda value: str(int(value)) if float(value).is_integer() else str(value)
        )
        keys=animal_keys+'|'+points['experience'].astype(int).astype(str)
    offsets=_record_jitter(keys,.07)
    for experience in sorted(set(x_values.astype(int))):
        mask=x_values==experience
        axis.scatter(x_values[mask]+offsets[mask],y_values[mask],s=12,
            color=colors[experience],alpha=.72,edgecolor='none',zorder=3)
    if not display_fit.empty:
        line_x = display_fit["x_value"].to_numpy(dtype=float)
        estimate = display_fit["estimate"].to_numpy(dtype=float)
        axis.fill_between(
            line_x,
            display_fit["ci_low"].to_numpy(dtype=float),
            display_fit["ci_high"].to_numpy(dtype=float),
            color=_LEVEL_COLOR,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(line_x, estimate, color=_LEVEL_COLOR, linewidth=1.25)
    axis.axhline(0, color="#BEBEBE", linewidth=0.5)
    axis.set_xticks([1, 2, 3, 4])
    axis.set_xlabel("No. of cycles")
    axis.set_ylabel("Level map correlation")
    level_test = statistics.loc[
        statistics["metric"].eq("level_map_correlation")
    ]
    if len(level_test) != 1:
        raise ValueError("experience_statistics must contain one level-map test")
    correlation = float(level_test.iloc[0]["pearson_r"])
    axis.text(
        0.05,
        0.82,
        f"r={correlation:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    axis.text(
        0.05,
        0.96,
        _significance_marks(float(level_test.iloc[0]["p_bh4"])),
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="black",
        fontsize=13.0,
        fontweight="bold",
    )
    _style_record_axis(axis)


def _draw_level_pv_histogram(axis: Axes, values: pd.DataFrame) -> None:
    """Draw pooled pair-level densities; levels are not biological replicates."""
    centers=(values.bin_left.to_numpy()+values.bin_right.to_numpy())/2
    widths=values.bin_right.to_numpy()-values.bin_left.to_numpy()
    axis.bar(centers,values.shuffle_density,width=widths,color='#8a8a8a',alpha=.62,edgecolor='none')
    axis.bar(centers,values.observed_density,width=widths,color=_LEVEL_COLOR,alpha=.72,edgecolor='none')
    axis.set(xlim=(-.45,1),xlabel='Level PV correlation',ylabel='Density',xticks=[0,.5,1])
    axis.set_ylim(0,max(values.shuffle_density.max(),values.observed_density.max())*1.4)
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.text(.05,.97,'Shuffle',transform=axis.transAxes,va='top',color='#6f6f6f',fontweight='bold')
    axis.text(.62,.37,'Data',transform=axis.transAxes,color=_LEVEL_COLOR,fontweight='bold')


def _draw_mean_null(axis: Axes, values: pd.DataFrame, *, population: bool) -> None:
    """Draw the permutation distribution of a cohort mean without new inference."""
    axis.hist(values.shuffle_mean_correlation,bins=80,density=True,color='#7F7F7F',alpha=.75,edgecolor='#555555',linewidth=.5)
    observed=float(values.observed_mean_correlation.iloc[0])
    axis.axvline(observed,color=_LEVEL_COLOR,linestyle='--',linewidth=1.2)
    axis.set(xlabel='Mean level PV\ncorrelation' if population else 'Mean level map\ncorrelation',ylabel='Density')
    axis.set_xlim(-.035 if population else 0,.44 if population else .30)
    axis.set_xticks([0,.2,.4] if population else [0,.1,.2,.3])
    axis.text(.18 if population else .20,.77,'Shuffle',transform=axis.transAxes,color='#7F7F7F')
    if population:
        axis.text(.55,.52,'Data',transform=axis.transAxes,color=_LEVEL_COLOR)
    else:
        axis.annotate('Data',xy=(observed,.52),xycoords=('data','axes fraction'),
            xytext=(-4,0),textcoords='offset points',ha='right',color=_LEVEL_COLOR)


def _draw_level_decoder(axis: Axes, values: pd.DataFrame, inference: Mapping[str, float]) -> None:
    """Draw transferred level-decoder pair values and their existing CR1 test."""
    for row in values.itertuples():
        axis.plot([0,1],[row.observed_accuracy,row.shuffle_accuracy],color='#8a8a8a',alpha=.28,lw=.65,zorder=1)
    for i,column in enumerate(('observed_accuracy','shuffle_accuracy')):
        y=values[column].to_numpy(float)
        color=_LEVEL_COLOR if i==0 else '#8a8a8a'
        axis.bar(i,y.mean(),width=.66,facecolor='none',edgecolor=color,linewidth=.8,alpha=1.,zorder=2)
        axis.scatter(np.full(len(y),i)+_record_jitter(values.pair_key,.045),y,s=11,color=color,edgecolors='none',alpha=.72,zorder=3)
    p=float(inference['p_greater'])
    axis.set(xticks=[0,1],xticklabels=['Data','Shuffle'],ylabel='Decoding accuracy',ylim=(0,.62),yticks=[0,.2,.4,.6],xlim=(-.5,1.5))
    # Keep the same comparison and stars; only their physical arrangement changes.
    stars = _significance_marks(p)
    fontsize = 6.5 if stars == "n.s." else 10.0
    bounds = axis.get_window_extent()
    width_mm, height_mm = np.array([bounds.width,bounds.height]) / axis.figure.dpi * 25.4
    glyph = TextPath((0,0),stars,size=fontsize,
                     prop=FontProperties(family='Arial',weight='bold')).get_extents()
    mm_per_pt = 25.4/72.
    half_gap = (glyph.width*mm_per_pt/2+.25)/width_mm*np.ptp(axis.get_xlim())
    y, cap = .585/.62, .20/height_mm
    axis.plot([0,0,.5-half_gap],[y-cap,y,y],transform=axis.get_xaxis_transform(),
              color='black',lw=.6,clip_on=False,zorder=10,gid='decoder_compact_left')
    axis.plot([.5+half_gap,1,1],[y,y,y-cap],transform=axis.get_xaxis_transform(),
              color='black',lw=.6,clip_on=False,zorder=10,gid='decoder_compact_right')
    baseline = y-(glyph.y0+glyph.height/2)*mm_per_pt/height_mm
    axis.text(.5,baseline,stars,transform=axis.get_xaxis_transform(),ha='center',
              va='baseline',fontsize=fontsize,fontweight='bold',color='black',clip_on=False,
              zorder=11,gid='decoder_compact_stars')


def render_tree_depth_control_summary(
    *, preferred_level_values: PreferenceData, preferred_level_statistics: pd.DataFrame,
    spatial_reassignment_values: SpatialNullData, spatial_reassignment_display: pd.DataFrame,
    spatial_reassignment_statistic: pd.DataFrame, partition_correlations: pd.DataFrame,
    partition_display_summary: pd.DataFrame, partition_statistics: pd.DataFrame,
    joint_neuron_values: pd.DataFrame, gap_values: pd.DataFrame, recurrence_values: pd.DataFrame,
    recurrence_display_trend: pd.DataFrame, recurrence_statistic: pd.DataFrame,
    within_cycle_display: pd.DataFrame, experience_values: pd.DataFrame,
    experience_display_fit: pd.DataFrame, experience_statistics: pd.DataFrame,
    population_histogram: pd.DataFrame, population_null: pd.DataFrame,
    mean_level_null: pd.DataFrame, decoder_values: pd.DataFrame,
    decoder_inference: Mapping[str,float], first30_values: pd.DataFrame,
    first30_display_fit: pd.DataFrame, first30_statistics: pd.DataFrame,
) -> Figure:
    """Draw the current fifteen-panel page from explicit computed inputs.

    Positions are in millimetres on the source's 183 by 132 mm page. Every
    array comes from a notebook producer; this renderer reads no files.
    """
    width,height=183.,132.
    boxes={'a':(0,0),'b':(35,0),'c':(63,0),'d':(89,0),'e':(114,0),'f':(155,0),
           'g':(0,45),'h':(56,45),'i':(91,45),'j':(127,45),
           'k':(0,86),'l':(37,86),'m':(73,86),'n':(110,86),'o':(146.5,86)}
    with plt.rc_context(with_figure_font({'font.family':'Arial','font.size':6.5,'axes.labelsize':6.5,
        'xtick.labelsize':6.,'ytick.labelsize':6.,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})):
        figure=plt.figure(figsize=(width/25.4,height/25.4))
        def axis(name: str,x: float,top: float,w: float,h: float) -> Axes:
            ax=figure.add_axes([x/width,1-(top+h)/height,w/width,h/height],label=name)
            ax.set_gid(name); ax.tick_params(length=2,pad=1.5,width=.65,labelsize=6.)
            ax.xaxis.labelpad=2.;ax.yaxis.labelpad=2.
            return ax
        for label,(x,top) in boxes.items():
            figure.text((x+.6)/width,1-(top+1)/height,label,fontsize=10.5,fontweight='bold',va='top')
        _draw_level_pv_histogram(axis('a',8,10,25,26),population_histogram)
        _draw_mean_null(axis('b',43,10,17,26),population_null,population=True)
        _draw_mean_null(axis('c',70,10,16,26),mean_level_null,population=False)
        _draw_level_decoder(axis('d',96,10,16,26),decoder_values,decoder_inference)
        transition=axis('e',121,10,26,26)
        image=_draw_preferred_level_transition_axis(transition,preferred_level_values)
        colorbar=figure.colorbar(image,cax=axis('e_colorbar',148.5,12,1.3,22),ticks=[-.1,0,.1])
        colorbar.ax.tick_params(length=1.5,pad=1)
        colorbar.ax.set_yticklabels(['−0.1','0','0.1'])
        transition.set_title('Δ conditional probability',fontsize=6.5,pad=3)
        _draw_preferred_level_distance_axis(axis('f',165,10,16,26),preferred_level_values)
        for x,rep,title in [(0.5,'horizontal_8','Horizontal 8-bin'),(28.5,'vertical_8','Vertical 8-bin')]:
            _draw_record_partition(axis('g_'+rep,x,51,26,26),rep,title)
        adjusted=dict(zip(partition_statistics.contrast,partition_statistics.p_bh2))
        _draw_eight_bin_controls_axis(axis('h',64,51,25,26),partition_correlations,partition_display_summary,adjusted)
        _draw_joint_neuron_axis(axis('i',101,51,24,26),joint_neuron_values)
        pair_axis=axis('j_pair',136,51,18,26)
        null_axis=axis('j_null',163.3,51,17,26)
        _draw_spatial_reassignment_axes(pair_axis,null_axis,spatial_reassignment_values,spatial_reassignment_display,spatial_reassignment_statistic)
        pair_axis.set_ylabel('Place map correlation')
        null_axis.set(xlabel='Place map\ncorrelation',ylabel='Count')
        null_axis.xaxis.set_major_locator(MaxNLocator(2))
        null_axis.yaxis.set_major_locator(MaxNLocator(3,integer=True))
        for text in null_axis.texts:text.set_fontsize(6);text.set_fontweight('normal')
        _draw_gap_axis(axis('k',8,91,27,31),gap_values)
        _draw_recurrence_axis(axis('l',44,91,27,31),recurrence_values,recurrence_display_trend,_recurrence_adjusted_probability(recurrence_statistic))
        _draw_within_cycle_depth_axis(axis('m',81,91,27,31),within_cycle_display)
        first=axis('n',118,91,26.5,31)
        _draw_experience_axis(first,first30_values,first30_display_fit,first30_statistics)
        first.set(xticks=[0,1,2,3,4],xlabel='No. of cycles',xlim=(-.45,4.45),ylim=(0,.75),yticks=[0,.2,.4,.6])
        first.set_title('First 30 min',fontsize=6.5,pad=3)
        last=axis('o',154.5,91,26.5,31)
        _draw_experience_axis(last,experience_values,experience_display_fit,experience_statistics)
        last.set(xlabel='Cycle',xlim=(.55,4.45),ylim=(-.06,.58))
        last.set_title('Long gap',fontsize=6.5,pad=3)
        for ax in (first,last):
            for line in list(ax.lines):
                if np.all(np.asarray(line.get_ydata())==0):line.remove()
            for text in ax.texts:text.set_color('black')
        for ax in figure.axes:
            ax.xaxis.label.set_fontsize(6.5);ax.yaxis.label.set_fontsize(6.5)
    return figure


def save_tree_depth_control_page(figure: Figure, output_dir: Path) -> None:
    """Export the current page and native-size individual panels a through o."""
    from matplotlib.transforms import Bbox
    from release_rendering.plotting import save_figure
    output_dir.mkdir(parents=True,exist_ok=True)
    save_figure(figure,output_dir/'Extended_Data_Figure_9',
        formats=('svg','pdf','png','tiff'),raster_dpi=600,preserve_canvas=True)
    boxes={'a':(0,0,35,45),'b':(35,0,28,45),'c':(63,0,26,45),
        'd':(89,0,25,45),'e':(114,0,41,45),'f':(155,0,28,45),
        'g':(0,45,56,41),'h':(56,45,35,41),'i':(91,45,36,41),'j':(127,45,56,41),
        'k':(0,86,37,46),'l':(37,86,36,46),'m':(73,86,37,46),
        'n':(110,86,36.5,46),'o':(146.5,86,36.5,46)}
    for letter,(x,top,w,h) in boxes.items():
        box=Bbox.from_bounds(x/25.4,(132-top-h)/25.4,w/25.4,h/25.4)
        for suffix in ('svg','pdf','png'):
            savefig_reproducibly(figure,output_dir/f'extended_data_figure_9_panel_{letter}.{suffix}',
                bbox_inches=box,pad_inches=0,dpi=600,facecolor='white')


FloatArray = NDArray[np.float64]


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

def _scalar_pearson(left, right):
    x, y = np.asarray(left,float), np.asarray(right,float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else float("nan")
