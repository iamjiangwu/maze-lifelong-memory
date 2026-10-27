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


CONTEXT_REPEATS = 20


VARIANT_ORDER = ("d1v1", "d1v2", "d2v1", "d2v2")


METRIC_COLUMNS = {
    "place_map_correlation": "place_map_correlation",
    "pv_correlation": "pv_correlation",
    "ensemble_rate_correlation": "ensemble_rate_correlation",
}


def calculate_day_combination_statistics(
    pair_values: pd.DataFrame,
) -> pd.DataFrame:
    """Test every pair of day combinations with clustered uncertainty and sign flips.

    The six paired differences of each metric form one family: exact sign-flip
    p values over animals (shown in the figure) and CR1 p values are each
    Benjamini-Hochberg adjusted across the six.
    """

    rows: list[dict[str, object]] = []
    for metric in METRIC_COLUMNS:
        selected = pair_values.loc[pair_values["metric"].eq(metric)]
        pivot = selected.pivot(
            index=["animal_id", "group"], columns="variant", values="value"
        )
        pivot = pivot.reindex(columns=VARIANT_ORDER)
        if not np.isfinite(pivot.to_numpy(float)).all():
            raise ValueError(f"Incomplete or nonfinite paired day combinations: {metric}")
        clusters = pivot.index.get_level_values("animal_id").to_numpy(str)
        contrast_indices: list[int] = []
        for left, right in itertools.combinations(VARIANT_ORDER, 2):
            first = pivot[left].to_numpy(float)
            second = pivot[right].to_numpy(float)
            difference = first - second
            fit = cr1_ols(difference, np.ones((len(pivot), 1)), clusters)
            result = linear_contrast(fit, [1.0])
            rows.append(
                {
                    "metric": metric,
                    "family": "all_pairwise",
                    "contrast": f"{left}_minus_{right}",
                    "test": "session-pair contrast; CR1 clustered by animal",
                    "n_session_pairs": len(pivot),
                    "n_animals": fit.n_clusters,
                    "statistic": result.t_value,
                    "p_value": result.p_two_sided,
                    "mean_difference": float(difference.mean()),
                    "se_cr1": result.standard_error,
                    "ci95_low": result.confidence_low,
                    "ci95_high": result.confidence_high,
                    "cluster_df": result.degrees_of_freedom,
                    "p_exact_cluster_signflip": exact_cluster_sign_flip_p(difference, clusters),
                }
            )
            contrast_indices.append(len(rows) - 1)
        adjusted = benjamini_hochberg(
            [rows[index]["p_exact_cluster_signflip"] for index in contrast_indices]
        )
        for index, adjusted_value in zip(
            contrast_indices,
            adjusted,
            strict=True,
        ):
            rows[index]["p_bh6_within_metric_exact_cluster_signflip"] = float(adjusted_value)
        adjusted_cr1 = benjamini_hochberg(
            [rows[index]["p_value"] for index in contrast_indices]
        )
        for index, adjusted_value in zip(contrast_indices, adjusted_cr1, strict=True):
            rows[index]["p_bh6_within_metric_cr1"] = float(adjusted_value)
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "metric",
            "family",
            "contrast",
            "test",
            "n_session_pairs",
            "n_animals",
            "statistic",
            "p_value",
            "mean_difference",
            "se_cr1",
            "ci95_low",
            "ci95_high",
            "cluster_df",
            "p_exact_cluster_signflip",
            "p_bh6_within_metric_exact_cluster_signflip",
            "p_bh6_within_metric_cr1",
        ),
    )


def summarize_day_combination_values(pair_values: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-pair means and animal-clustered CR1 95% intervals."""

    rows: list[dict[str, object]] = []
    for (metric, variant), selected in pair_values.groupby(
        ["metric", "variant"],
        observed=True,
        sort=False,
    ):
        values = selected["value"].to_numpy(float)
        mean = float(np.mean(values))
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite day-combination pair values: {metric}/{variant}")
        fit = cr1_ols(
            values, np.ones((len(values), 1)), selected["animal_id"].astype(str).tolist()
        )
        result = linear_contrast(fit, [1.0])
        rows.append(
            {
                "metric": metric,
                "variant": variant,
                "n_animals": int(selected["animal_id"].nunique()),
                "n_session_pairs": len(selected),
                "mean": mean,
                "standard_error": result.standard_error,
                "ci95_low": result.confidence_low,
                "ci95_high": result.confidence_high,
                "cluster_df": result.degrees_of_freedom,
                "aggregation": "equal-weighted session-pair mean",
                "interval": "CR1 by animal; Student t(G-1)",
            }
        )
    return pd.DataFrame.from_records(rows)


def calculate_context_statistics(pair_values: pd.DataFrame) -> pd.DataFrame:
    """Compute displayed d1v1 context means, CIs and its single difference test."""

    if not pair_values["variant"].eq("d1v1").all():
        raise ValueError("Context publication values must contain only d1v1")

    rows: list[dict[str, object]] = []
    output_columns = [
        "variant",
        "estimand",
        "n_pair_rows",
        "n_animals",
        "estimate",
        "cluster_standard_error",
        "ci95_low",
        "ci95_high",
        "t_value",
        "degrees_of_freedom",
        "p_value",
    ]
    columns = {
        "outside": "outside_correlation",
        "maze": "maze_correlation",
        "maze_minus_outside": "maze_minus_outside",
    }
    for variant in ("d1v1",):
        selected = pair_values.loc[pair_values["variant"].eq(variant)]
        if selected["animal_id"].nunique() < 2:
            continue
        for estimand, column in columns.items():
            outcome = selected[column].to_numpy(float)
            fit = cr1_ols(
                outcome,
                np.ones((len(selected), 1), dtype=float),
                selected["animal_id"].astype(str).tolist(),
            )
            contrast = linear_contrast(fit, [1.0])
            rows.append(
                {
                    "variant": variant,
                    "estimand": estimand,
                    "n_pair_rows": fit.n_observations,
                    "n_animals": fit.n_clusters,
                    "estimate": contrast.estimate,
                    "cluster_standard_error": contrast.standard_error,
                    "ci95_low": contrast.confidence_low,
                    "ci95_high": contrast.confidence_high,
                    "t_value": contrast.t_value,
                    "degrees_of_freedom": contrast.degrees_of_freedom,
                    "p_value": contrast.p_two_sided if estimand == "maze_minus_outside" else np.nan,
                }
            )
    output = pd.DataFrame.from_records(rows, columns=output_columns)
    return output


def average_context_draws_by_pair(repeats: pd.DataFrame) -> pd.DataFrame:
    """Average technical draws to one context value per source-target pair."""

    group_columns = [
        "pair_key",
        "variant",
        "group",
        "animal_id",
        "source_session",
        "target_session",
    ]
    values = (
        repeats.groupby(group_columns, observed=True, sort=False)
        .agg(
            outside_correlation=("outside_correlation", "mean"),
            maze_correlation=("maze_correlation", "mean"),
            maze_minus_outside=("maze_minus_outside", "mean"),
            repeat_count=("repeat", "nunique"),
            outside_source_frames=("outside_source_frames", "first"),
            outside_target_frames=("outside_target_frames", "first"),
            maze_source_draw_frames=("maze_source_draw_frames", "first"),
            maze_target_draw_frames=("maze_target_draw_frames", "first"),
            outside_neuron_count=("outside_neuron_count", "first"),
            mean_maze_neuron_count=("maze_neuron_count", "mean"),
        )
        .reset_index()
    )
    if not values["repeat_count"].eq(CONTEXT_REPEATS).all():
        raise AssertionError("Every context pair must contain exactly 20 maze draws")
    return values


_ARTIFACT = "extended_data_figure_5"


_METRIC_COLORS = {
    "place_map_correlation": "#B95A58",
    "pv_correlation": "#E29957",
    "ensemble_rate_correlation": "#4A5E65",
}


_STAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


_ACCEPTED_PAGE_SIZE_IN = (5.021503718285214, 6.943229440069992)


_ACCEPTED_GALLERY_REGION_IN = (
    0.02,
    1.2929046369203858,
    4.981503718285214,
    5.635324803149606,
)


_ACCEPTED_DAY_COMBINATION_REGION_IN = (
    0.020001093613298337,
    0.08000000000000007,
    3.95,
    1.2129024496937884,
)


_ACCEPTED_CONTEXT_REGION_IN = (
    4.053333333333334,
    0.08000000000000007,
    0.9309820647419073,
    1.2129024496937884,
)


_GALLERY_RENDER_SIZE_IN = (8.0, 9.05)


_ACCEPTED_FONT_SCALE = 0.75


_ACCEPTED_PANEL_ID_SIZE = 13.0 * _ACCEPTED_FONT_SCALE


_ACCEPTED_DISPLAY_STYLE = {
    "font.size": 8.0 * _ACCEPTED_FONT_SCALE,
    "axes.labelsize": 8.0 * _ACCEPTED_FONT_SCALE,
    "axes.titlesize": 8.0 * _ACCEPTED_FONT_SCALE,
    "xtick.labelsize": 7.0 * _ACCEPTED_FONT_SCALE,
    "ytick.labelsize": 7.0 * _ACCEPTED_FONT_SCALE,
    "legend.fontsize": 7.0 * _ACCEPTED_FONT_SCALE,
    "axes.linewidth": 0.8 * _ACCEPTED_FONT_SCALE,
    "xtick.major.width": 0.8 * _ACCEPTED_FONT_SCALE,
    "ytick.major.width": 0.8 * _ACCEPTED_FONT_SCALE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
}


_ACCEPTED_VARIANT_LABELS = {
    "d1v1": "d1-d1",
    "d1v2": "d1-d2",
    "d2v1": "d2-d1",
    "d2v2": "d2-d2",
}


def _draw_map(axis: plt.Axes, values: FloatArray) -> None:
    """Draw one normalized 45 x 45 place map with the maze outline."""

    maze = build_maze(6)
    axis.imshow(
        values,
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.plot(
        maze.wall_xy[:, 0] * 3.0,
        maze.wall_xy[:, 1] * 3.0,
        color="white",
        linewidth=0.45,
        alpha=0.96,
    )
    axis.set_xlim(-0.5, 44.5)
    axis.set_ylim(44.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _significant_pairs(statistics: pd.DataFrame) -> list[tuple[int, int, float]]:
    """Return (left, right, adjusted p) for significant pairs, shortest first."""

    tests = statistics.loc[
        statistics["family"].eq("all_pairwise")
        & statistics["test"].eq("session-pair contrast; CR1 clustered by animal")
    ]
    significant: list[tuple[int, int, float]] = []
    for row in tests.itertuples(index=False):
        left, _, right = str(row.contrast).partition("_minus_")
        if left not in VARIANT_ORDER or right not in VARIANT_ORDER:
            continue
        adjusted_p = float(row.p_bh6_within_metric_exact_cluster_signflip)
        if significance_stars(adjusted_p, thresholds=_STAR_THRESHOLDS) != "n.s.":
            significant.append((VARIANT_ORDER.index(left), VARIANT_ORDER.index(right), adjusted_p))
    significant.sort(key=lambda item: (item[1] - item[0], item[0]))
    return significant


def _context_display_summary(
    pair_values: pd.DataFrame,
    statistics: pd.DataFrame,
    condition: str,
) -> tuple[float, float, float]:
    """Return the full CR1 summary for one condition."""

    estimand = "outside" if condition == "outside" else "maze"
    selected = statistics.loc[
        statistics["variant"].eq("d1v1")
        & statistics["estimand"].eq(estimand)
    ]
    if not selected.empty:
        row = selected.iloc[0]
        return (
            float(row["estimate"]),
            float(row["ci95_low"]),
            float(row["ci95_high"]),
        )
    column = f"{condition}_correlation"
    value = float(
        pair_values.loc[
            pair_values["variant"].eq("d1v1"),
            column,
        ].mean()
    )
    return value, value, value


def _render_matched_map_gallery_image(payload: GalleryPayload) -> np.ndarray:
    """Rasterize the raw-derived map gallery at its accepted source size."""

    pair_count = len(payload.pairs)
    if pair_count == 0 or pair_count > 32:
        raise ValueError("The matched-map gallery requires between 1 and 32 pairs")
    if not (
        len(payload.normalized_maps)
        == len(payload.correlations)
        == pair_count
    ):
        raise ValueError("Gallery maps, correlations and pair declarations disagree")

    gallery_figure = plt.figure(
        figsize=_GALLERY_RENDER_SIZE_IN,
        facecolor="white",
        dpi=400,
    )
    gallery_grid = gallery_figure.add_gridspec(
        8,
        4,
        left=0.002,
        right=0.998,
        top=0.955,
        bottom=0.008,
        wspace=0.028,
        hspace=0.040,
    )
    for position, (pair, maps, correlation) in enumerate(
        zip(
            payload.pairs,
            payload.normalized_maps,
            payload.correlations,
            strict=True,
        )
    ):
        row, column = divmod(position, 4)
        card_region = gallery_grid[row, column].get_position(gallery_figure)
        header_fraction = 0.12
        inset = 0.022 * min(card_region.width, card_region.height)
        map_bottom = card_region.y0 + inset
        map_top = card_region.y1 - header_fraction * card_region.height
        map_height = max(map_top - map_bottom, 1e-6)
        inner_width = max(card_region.width - 2.0 * inset, 1e-6)
        map_gap = 0.012 * card_region.width
        map_width = (inner_width - map_gap) / 2.0
        map_left = card_region.x0 + inset
        for map_index, map_values in enumerate(maps):
            axis = gallery_figure.add_axes(
                [
                    map_left + map_index * (map_width + map_gap),
                    map_bottom,
                    map_width,
                    map_height,
                ]
            )
            _draw_map(axis, map_values)
            axis.set_aspect("auto")
        header_y = map_top + 0.55 * (card_region.y1 - map_top)
        gallery_figure.text(
            0.5 * (card_region.x0 + card_region.x1),
            header_y,
            f"r = {float(correlation):.2f}",
            ha="center",
            va="center",
            fontsize=5.0,
            color="#303030",
            clip_on=False,
            zorder=21,
        )
        gallery_figure.add_artist(
            Rectangle(
                (card_region.x0, card_region.y0),
                card_region.width,
                card_region.height,
                transform=gallery_figure.transFigure,
                facecolor="none",
                edgecolor=(
                    "#B64342"
                    if pair.highlighted_reference_example
                    else "#B8B8B8"
                ),
                linewidth=1.10 if pair.highlighted_reference_example else 0.45,
                joinstyle="miter",
                clip_on=False,
                zorder=20,
            )
        )

    colorbar_axis = gallery_figure.add_axes([0.34, 0.968, 0.32, 0.008])
    colorbar = gallery_figure.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(0.0, 1.0),
            cmap=plt.get_cmap("viridis").copy(),
        ),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(
        labelsize=5.95,
        length=1.5,
        pad=1.0,
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
    )
    colorbar.set_label("Normalized activity", fontsize=7.0, labelpad=1.5)

    canvas = FigureCanvasAgg(gallery_figure)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(gallery_figure)
    return image


def _page_axis(
    figure: Figure,
    region_inches: tuple[float, float, float, float],
) -> plt.Axes:
    """Create an axis from a lower-left inch rectangle on the accepted page."""

    page_width, page_height = _ACCEPTED_PAGE_SIZE_IN
    left, bottom, width, height = region_inches
    return figure.add_axes(
        [
            left / page_width,
            bottom / page_height,
            width / page_width,
            height / page_height,
        ]
    )


def _write_accepted_label(
    figure: Figure,
    x_inches: float,
    y_inches: float,
    label: str,
    *,
    vertical_alignment: str,
) -> None:
    """Write one accepted lowercase role label at an inch coordinate."""

    page_width, page_height = _ACCEPTED_PAGE_SIZE_IN
    figure.text(
        x_inches / page_width,
        y_inches / page_height,
        label,
        ha="left",
        va=vertical_alignment,
        fontsize=_ACCEPTED_PANEL_ID_SIZE,
        fontweight="bold",
    )


def _draw_accepted_significance_bracket(
    axis: plt.Axes,
    x_left: float,
    x_right: float,
    adjusted_p: float,
    *,
    y_fraction: float,
) -> bool:
    """Draw the accepted tick-and-star bracket when the result is significant."""

    stars = significance_stars(adjusted_p, thresholds=_STAR_THRESHOLDS)
    if stars == "n.s.":
        return False
    low, high = axis.get_ylim()
    span = high - low
    y_base = low + y_fraction * span
    tick = 0.022 * span
    axis.plot(
        [x_left, x_left, x_right, x_right],
        [y_base, y_base + tick, y_base + tick, y_base],
        color="#202020",
        linewidth=0.7,
        clip_on=False,
        zorder=5,
    )
    axis.text(
        (x_left + x_right) / 2.0,
        y_base + tick - 0.024 * span,
        stars,
        ha="center",
        va="bottom",
        fontsize=8.0 * _ACCEPTED_FONT_SCALE,
        fontweight="bold",
        color="#202020",
        clip_on=False,
        zorder=6,
    )
    return True


def _draw_day_combination_comparisons(
    figure: Figure,
    pair_metric_values: pd.DataFrame,
    summaries: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw the three session-pair day-combination comparisons in one band."""

    region_left, region_bottom, region_width, region_height = (
        _ACCEPTED_DAY_COMBINATION_REGION_IN
    )
    shifted_left = region_left + 8.0 * _ACCEPTED_FONT_SCALE / 72.0
    ylabel_inset = 0.18
    # Separate numeric y ticks need additional space between the axes. Keep
    # each b-d plotting width unchanged by using the space released by e.
    comparison_gap = 0.16
    usable_width = region_width - ylabel_inset - 0.02
    comparison_width = (usable_width - 2.0 * comparison_gap) / 3.0
    plot_bottom = region_bottom + 0.11
    plot_height = region_height - 0.11 - 0.18
    random_generator = np.random.default_rng(20260806)
    metric_roles = (
        ("place_map_correlation", "c", 1),
        ("pv_correlation", "b", 0),
        ("ensemble_rate_correlation", "d", 2),
    )

    # Preserve metric draw order so the seeded jitter stays attached to each metric.
    for metric, accepted_label, metric_index in metric_roles:
        plot_left = (
            shifted_left
            + ylabel_inset
            + metric_index * (comparison_width + comparison_gap)
        )
        axis = _page_axis(
            figure,
            (plot_left, plot_bottom, comparison_width, plot_height),
        )
        selected_values = pair_metric_values.loc[pair_metric_values["metric"].eq(metric)]
        selected_summaries = summaries.loc[summaries["metric"].eq(metric)]
        data_top = 0.0
        for variant_index, variant in enumerate(VARIANT_ORDER):
            values = selected_values.loc[
                selected_values["variant"].eq(variant), "value"
            ].to_numpy(float)
            summary = selected_summaries.loc[
                selected_summaries["variant"].eq(variant)
            ]
            if values.size == 0 or len(summary) != 1:
                raise ValueError(f"Missing calculated {metric}/{variant} values")
            summary_row = summary.iloc[0]
            mean = float(summary_row["mean"])
            ci_low = float(summary_row["ci95_low"])
            ci_high = float(summary_row["ci95_high"])
            data_top = max(data_top, float(np.max(values)), ci_high)
            color = _METRIC_COLORS[metric]
            axis.bar(
                variant_index,
                mean,
                width=0.60,
                facecolor="none",
                edgecolor=color,
                linewidth=0.8,
                alpha=1.0,
                zorder=1,
            )
            axis.scatter(
                np.full(values.size, variant_index, dtype=float)
                + random_generator.normal(0.0, 0.06, values.size),
                values,
                s=11.0,
                facecolor=color,
                edgecolor="none",
                alpha=0.72,
                zorder=2,
            )
            axis.errorbar(
                variant_index,
                mean,
                yerr=np.array([[mean - ci_low], [ci_high - mean]]),
                color="black",
                fmt="none",
                capsize=4.0 * _ACCEPTED_FONT_SCALE,
                elinewidth=1.1 * _ACCEPTED_FONT_SCALE,
                capthick=1.1 * _ACCEPTED_FONT_SCALE,
                zorder=4,
            )
        axis.set_xticks(
            range(len(VARIANT_ORDER)),
            [_ACCEPTED_VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
        )
        # Separate scales make each distribution readable. The author-selected
        # zero lower bounds intentionally leave small negative values off scale.
        y_top, y_ticks = {
            "pv_correlation": (0.65, [0.0, 0.2, 0.4, 0.6]),
            "place_map_correlation": (0.50, [0.0, 0.2, 0.4]),
            "ensemble_rate_correlation": (0.80, [0.0, 0.2, 0.4, 0.6, 0.8]),
        }[metric]
        axis.set_ylim(0.0, y_top)
        axis.set_yticks(y_ticks)
        axis.tick_params(axis="both", length=2.0, pad=0.8)
        for tick_label in axis.get_xticklabels():
            tick_label.set_clip_on(False)
        for spine in axis.spines.values():
            spine.set_color("#202020")
        if metric_index == 0:
            axis.set_ylabel("correlation", labelpad=1.0)

        significant = _significant_pairs(statistics.loc[statistics["metric"].eq(metric)])
        base_height = 0.756 if metric == "ensemble_rate_correlation" else 0.535
        tier_step = 0.050 if metric == "ensemble_rate_correlation" else 0.040
        for tier, (start, stop, adjusted_p) in enumerate(significant):
            _draw_compact_comparison_bracket(
                axis,
                float(start),
                float(stop),
                adjusted_p,
                height=base_height + tier_step * tier,
            )
        _write_accepted_label(
            figure,
            plot_left - 0.05,
            plot_bottom + plot_height + 0.045,
            accepted_label,
            vertical_alignment="bottom",
        )


def _draw_compact_comparison_bracket(
    axis: plt.Axes, left: float, right: float, adjusted_p: float, *, height: float,
) -> None:
    """Center unchanged stars in a gap, without masking observations."""
    stars = significance_stars(adjusted_p, thresholds=_STAR_THRESHOLDS)
    if stars == "n.s.":
        return
    bounds = axis.get_window_extent()
    width_mm, height_mm = np.array([bounds.width, bounds.height]) / axis.figure.dpi * 25.4
    low, high = axis.get_ylim()
    y = (height - low) / (high - low)
    glyph = TextPath((0, 0), stars, size=10.0,
                     prop=FontProperties(family="Arial", weight="bold")).get_extents()
    mm_per_pt = 25.4 / 72.0
    half_gap = (glyph.width * mm_per_pt / 2.0 + 0.25) / width_mm * np.ptp(axis.get_xlim())
    center = (left + right) / 2.0
    cap = 0.20 / height_mm
    key = f"compact_{left:g}_{right:g}"
    for side, x, yy in (
        ("left", [left, left, center-half_gap], [y-cap, y, y]),
        ("right", [center+half_gap, right, right], [y, y, y-cap]),
    ):
        axis.plot(x, yy, transform=axis.get_xaxis_transform(), color="black",
                  linewidth=0.6, clip_on=False, zorder=5, gid=f"{key}_{side}")
    baseline = y - (glyph.y0 + glyph.height/2.0) * mm_per_pt / height_mm
    axis.text(center, baseline, stars, transform=axis.get_xaxis_transform(),
              ha="center", va="baseline", fontsize=10.0, fontweight="bold",
              color="black", clip_on=False, zorder=6, gid=f"{key}_stars")


def _draw_maze_context_comparison(
    figure: Figure,
    pair_values: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    """Draw the saved outside-versus-maze comparison on its unchanged scale."""

    region_left, region_bottom, region_width, region_height = (
        _ACCEPTED_CONTEXT_REGION_IN
    )
    plot_left = region_left + 0.12
    plot_bottom = region_bottom + 0.11
    plot_width = region_width - 0.14
    plot_height = region_height - 0.11 - 0.18
    axis = _page_axis(
        figure,
        (plot_left, plot_bottom, plot_width, plot_height),
    )
    selected = pair_values.loc[pair_values["variant"].eq("d1v1")]
    if selected.empty:
        raise ValueError("No Day 1-Day 1 maze-context values were calculated")
    outside = selected["outside_correlation"].to_numpy(float)
    maze = selected["maze_correlation"].to_numpy(float)
    for outside_value, maze_value in zip(outside, maze, strict=True):
        axis.plot(
            [0.0, 1.0],
            [outside_value, maze_value],
            color="#B0B0B0",
            alpha=0.35,
            linewidth=0.55,
            zorder=1,
        )
    axis.scatter(
        np.zeros(outside.size),
        outside,
        s=10.0 * _ACCEPTED_FONT_SCALE**2,
        color="#9A9A9A",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )
    axis.scatter(
        np.ones(maze.size),
        maze,
        s=10.0 * _ACCEPTED_FONT_SCALE**2,
        color="#4A5E65",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )
    for x_value, condition, color in (
        (0.0, "outside", "#9A9A9A"),
        (1.0, "maze", "#4A5E65"),
    ):
        mean, low, high = _context_display_summary(
            pair_values,
            statistics,
            condition,
        )
        axis.errorbar(
            x_value,
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            fmt="o",
            color="black",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.2 * _ACCEPTED_FONT_SCALE,
            capsize=2.2 * _ACCEPTED_FONT_SCALE,
            elinewidth=0.9 * _ACCEPTED_FONT_SCALE,
            zorder=4,
        )
    axis.set_xticks([0.0, 1.0], ["Outside", "Maze"])
    axis.set_ylim(0.0, 0.8)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axis.tick_params(axis="both", length=2.0, pad=0.8)
    for tick_label in axis.get_xticklabels():
        tick_label.set_clip_on(False)
    for spine in axis.spines.values():
        spine.set_color("#202020")

    test = statistics.loc[
        statistics["variant"].eq("d1v1")
        & statistics["estimand"].eq("maze_minus_outside")
    ]
    if not test.empty:
        _draw_compact_comparison_bracket(
            axis,
            0.0,
            1.0,
            float(test.iloc[0]["p_value"]),
            height=0.80,
        )
    _write_accepted_label(
        figure,
        plot_left - 0.05,
        plot_bottom + plot_height + 0.045,
        "e",
        vertical_alignment="bottom",
    )


def _validate_accepted_page_support(
    payload: GalleryPayload,
    pair_metric_values: pd.DataFrame,
    context_pair_values: pd.DataFrame,
    run_mode: RunMode,
) -> None:
    """Check the accepted full-run support."""

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    if len(payload.pairs) != 32:
        raise ValueError("The accepted map gallery requires 32 matched pairs")
    for metric in METRIC_COLUMNS:
        selected = pair_metric_values.loc[pair_metric_values["metric"].eq(metric)]
        counts = selected.groupby("variant", observed=True)["animal_id"].nunique()
        pair_counts = selected.groupby("variant", observed=True).size()
        duplicates = selected.duplicated(["animal_id", "group", "variant"]).any()
        if (
            counts.to_dict() != {variant: 14 for variant in VARIANT_ORDER}
            or pair_counts.to_dict() != {variant: 36 for variant in VARIANT_ORDER}
            or duplicates
        ):
            raise ValueError(f"{metric}: accepted support is 36 pairs / 14 animals per variant")
    displayed_context = context_pair_values.loc[
        context_pair_values["variant"].eq("d1v1")
    ]
    if (
        len(displayed_context) != 36
        or displayed_context["animal_id"].nunique() != 14
    ):
        raise ValueError("The accepted context comparison requires 36 pairs / 14 animals")


def assemble_long_gap_stability_controls(
    gallery: GalleryPayload,
    day_combination_pair_metric_values: pd.DataFrame,
    day_combination_summaries: pd.DataFrame,
    day_combination_statistics: pd.DataFrame,
    maze_context_pair_values: pd.DataFrame,
    maze_context_statistics: pd.DataFrame,
    *,
    run_mode: RunMode,
    output_dir: Path,
) -> FigureAssemblyResult:
    """Present the explicit calculations in the accepted page geometry."""

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    destination = Path(output_dir).resolve()
    _validate_accepted_page_support(
        gallery,
        day_combination_pair_metric_values,
        maze_context_pair_values,
        run_mode,
    )

    with version_record_style(_ACCEPTED_DISPLAY_STYLE):
        gallery_image = _render_matched_map_gallery_image(gallery)
        figure = plt.figure(figsize=_ACCEPTED_PAGE_SIZE_IN, facecolor="white")
        gallery_axis = _page_axis(figure, _ACCEPTED_GALLERY_REGION_IN)
        gallery_axis.imshow(
            gallery_image,
            interpolation="antialiased",
            aspect="auto",
        )
        gallery_axis.set_axis_off()
        gallery_left, gallery_bottom, _, gallery_height = (
            _ACCEPTED_GALLERY_REGION_IN
        )
        _write_accepted_label(
            figure,
            gallery_left + 0.01,
            gallery_bottom + (0.968 + 0.5 * 0.008) * gallery_height,
            "a",
            vertical_alignment="center",
        )
        _draw_day_combination_comparisons(
            figure,
            day_combination_pair_metric_values,
            day_combination_summaries,
            day_combination_statistics,
        )
        _draw_maze_context_comparison(
            figure,
            maze_context_pair_values,
            maze_context_statistics,
        )
        figure.set_dpi(600)
        output_paths = save_figure(
            figure,
            destination / "Extended_Data_Figure_5",
            formats=("svg", "pdf", "png", "tiff"),
            raster_dpi=600,
            preserve_canvas=True,
        )
        plt.close(figure)

    root = REPOSITORY_ROOT.resolve()
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT,
        run_mode=run_mode,
        output_paths=tuple(
            path.resolve() for path in output_paths
        ),
    )
