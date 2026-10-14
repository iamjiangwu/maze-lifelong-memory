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


EXPERIENCE_GROUP_COLORS = {
    "place_map_correlation": {
        "naive": "#D89C98",
        "3mo": "#CB817D",
        "6mo": "#BF6965",
        "9mo": "#A84E4B",
        "12mo": "#78312F",
    },
    "pv_correlation": {
        "naive": "#EDBD8F",
        "3mo": "#E8A76F",
        "6mo": "#E29957",
        "9mo": "#BD7137",
        "12mo": "#8A4A1F",
    },
    "ensemble_rate_correlation": {
        "naive": "#9FB0B5",
        "3mo": "#7D9198",
        "6mo": "#637980",
        "9mo": "#4A5E65",
        "12mo": "#2C3A3F",
    },
}


DISPLAY_METRICS = (
    "place_map_correlation",
    "pv_correlation",
    "ensemble_rate_correlation",
)


def _empty_model_row(
    *,
    metric: str,
    n_rows: int,
    n_animals: int,
    n_parameters: int,
    design_rank: int,
    model_columns: list[str],
) -> dict[str, object]:
    """Create a complete model record when support is too small for inference."""

    return {
        "record_type": "age_model",
        "metric": metric,
        "n_rows": n_rows,
        "n_animals": n_animals,
        "n_parameters": n_parameters,
        "design_rank": design_rank,
        "model_columns": ";".join(model_columns),
        "beta_per_30_days": np.nan,
        "cluster_se": np.nan,
        "cluster_t": np.nan,
        "cluster_df": np.nan,
        "p_two_sided": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "r_squared": np.nan,
    }


def _fit_age_model(
    points: pd.DataFrame,
    *,
    metric: str,
    age_column: str,
    experience_column: str,
) -> dict[str, object]:
    """Fit correlation on age per 30 days and categorical experience.

    Pair rows are retained. CR1 covariance is clustered by biological animal,
    and inference uses the Student t reference with G-1 degrees of freedom.
    """

    columns = ["animal_id", experience_column, age_column, "value"]
    selected = points.loc[points["metric"].eq(metric), columns].copy()
    for column in (experience_column, age_column, "value"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna().reset_index(drop=True)
    experiences = sorted(selected[experience_column].astype(int).unique())
    age = selected[age_column].to_numpy(dtype=float)
    centered_age = (age - float(np.mean(age))) / 30.0 if len(age) else age
    design_columns = [np.ones(len(selected), dtype=float), centered_age]
    model_columns = ["intercept", "age_per_30_days"]
    experience_values = selected[experience_column].to_numpy(dtype=int)
    for experience in experiences[1:]:
        design_columns.append((experience_values == experience).astype(float))
        model_columns.append(f"experience_{experience}")
    design = np.column_stack(design_columns)
    n_rows, n_parameters = design.shape
    n_animals = int(selected["animal_id"].nunique())
    rank = int(np.linalg.matrix_rank(design))
    empty = _empty_model_row(
        metric=metric,
        n_rows=n_rows,
        n_animals=n_animals,
        n_parameters=n_parameters,
        design_rank=rank,
        model_columns=model_columns,
    )
    if n_rows <= n_parameters or n_animals < 3 or rank != n_parameters:
        return empty
    fit = cr1_ols(
        selected["value"].to_numpy(dtype=float),
        design,
        selected["animal_id"].astype(str).tolist(),
    )
    age_contrast = np.zeros(n_parameters, dtype=float)
    age_contrast[1] = 1.0
    result = linear_contrast(fit, age_contrast)
    outcome = selected["value"].to_numpy(dtype=float)
    total = float(np.sum((outcome - float(np.mean(outcome))) ** 2))
    return {
        **empty,
        "beta_per_30_days": result.estimate,
        "cluster_se": result.standard_error,
        "cluster_t": result.t_value,
        "cluster_df": result.degrees_of_freedom,
        "p_two_sided": result.p_two_sided,
        "ci95_low": result.confidence_low,
        "ci95_high": result.confidence_high,
        "r_squared": (
            1.0 - float(np.sum(fit.residuals**2)) / total if total > 0 else np.nan
        ),
    }


def fit_age_model_family(
    points: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    age_column: str,
    experience_column: str,
    family_name: str,
) -> pd.DataFrame:
    """Fit one age model per metric and apply the declared BH family."""

    rows = [
        _fit_age_model(
            points,
            metric=metric,
            age_column=age_column,
            experience_column=experience_column,
        )
        for metric in metrics
    ]
    models = pd.DataFrame(rows)
    models[family_name] = benjamini_hochberg(
        models["p_two_sided"].to_numpy(dtype=float)
    )
    models[
        "multiplicity_family"
    ] = f"{family_name.upper()} across {len(metrics)} neural readouts"
    return models


def prepare_age_display_lines(
    points: pd.DataFrame,
    model: pd.Series,
    *,
    metric: str,
    long_gap: bool,
) -> pd.DataFrame:
    """Calculate the experience-specific line coordinates drawn over pair values."""

    selected = points.loc[points["metric"].eq(metric)].copy()
    beta = float(model["beta_per_30_days"])
    age_column = "destination_age_days" if long_gap else "age_day_i_days"
    experience_column = "experience" if long_gap else "experience_index"
    rows: list[dict[str, object]] = []
    for experience, group in selected.groupby(experience_column, sort=True):
        x = group[age_column].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if not np.isfinite(beta) or len(x) < 2 or np.ptp(x) <= 0:
            continue
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
        line_y = float(np.mean(y)) + beta * (line_x - float(np.mean(x))) / 30.0
        rows.extend(
            {
                "record_type": "fit_line",
                "metric": metric,
                "experience": int(experience),
                "display_age_days": float(age),
                "display_value": float(value),
            }
            for age, value in zip(line_x, line_y, strict=True)
        )
    return pd.DataFrame.from_records(
        rows,
        columns=(
            "record_type",
            "metric",
            "experience",
            "display_age_days",
            "display_value",
        ),
    )


_ARTIFACT_KEY = "extended_data_figure_7"


_METRIC_LABELS = {
    "place_map_correlation": "Place map",
    "level_map_correlation": "Maze-depth map",
    "pv_correlation": "PV",
    "ensemble_rate_correlation": "Ensemble rate",
}


_METRIC_COLORS = {
    "place_map_correlation": "#B95A58",
    "pv_correlation": "#E29957",
    "ensemble_rate_correlation": "#4A5E65",
}


_GROUP_KEYS = {0: "naive", 1: "3mo", 2: "6mo", 3: "9mo", 4: "12mo"}


_GROUP_LABELS = {
    0: "Cycle 0 (naive)",
    1: "Cycle 1",
    2: "Cycle 2",
    3: "Cycle 3",
    4: "Cycle 4",
}


_TRANSITION_LABELS = {
    1: "Cycle 1 (0-3 mo)",
    2: "Cycle 2 (3-6 mo)",
    3: "Cycle 3 (6-9 mo)",
    4: "Cycle 4 (9-12 mo)",
}


_ASSEMBLY_SIZE_IN = (7.5, 4.92)


_PRIOR_ASSEMBLY_SIZE_IN = (7.5, 5.15)


_TITLE_SLOT_IN = (_PRIOR_ASSEMBLY_SIZE_IN[1] - _ASSEMBLY_SIZE_IN[1]) / 2.0


_PRIOR_AXES_HEIGHT_IN = (
    (0.86 - 0.095) * _PRIOR_ASSEMBLY_SIZE_IN[1] / (2.0 + 0.72)
)


_ASSEMBLY_BOTTOM = (
    0.095 * _PRIOR_ASSEMBLY_SIZE_IN[1] / _ASSEMBLY_SIZE_IN[1]
)


_ASSEMBLY_HSPACE = (
    0.72 * _PRIOR_AXES_HEIGHT_IN - _TITLE_SLOT_IN
) / _PRIOR_AXES_HEIGHT_IN


_ASSEMBLY_TOP = (
    0.095 * _PRIOR_ASSEMBLY_SIZE_IN[1]
    + 2.0 * _PRIOR_AXES_HEIGHT_IN
    + _ASSEMBLY_HSPACE * _PRIOR_AXES_HEIGHT_IN
) / _ASSEMBLY_SIZE_IN[1]


_DISPLAY_STYLE = {
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "legend.fontsize": 5.8,
    "legend.frameon": False,
    "lines.linewidth": 0.8,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}


def _group_color(metric: str, experience: int) -> str:
    """Return the shared metric and experience color."""

    try:
        return EXPERIENCE_GROUP_COLORS[metric][_GROUP_KEYS[int(experience)]]
    except KeyError as error:
        raise ValueError(f"Unsupported experience index: {experience}") from error


def _draw_age_axis(
    axis: plt.Axes,
    points: pd.DataFrame,
    display_lines: pd.DataFrame,
    *,
    metric: str,
    long_gap: bool,
    show_ylabel: bool,
) -> None:
    """Draw one age association in the accepted panel grammar."""

    age_column = "destination_age_days" if long_gap else "age_day_i_days"
    experience_column = "experience" if long_gap else "experience_index"
    selected = points.loc[points["metric"].eq(metric)].copy()
    for experience, group in selected.groupby(experience_column, sort=True):
        index = int(experience)
        color = _group_color(metric, index)
        axis.scatter(
            group[age_column],
            group["value"],
            s=12,
            marker="o",
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            alpha=0.90,
            zorder=3,
        )
        line = display_lines.loc[
            display_lines["metric"].eq(metric)
            & display_lines["experience"].eq(index)
        ].sort_values("display_age_days")
        if not line.empty:
            axis.plot(
                line["display_age_days"],
                line["display_value"],
                color=color,
                linewidth=1.05,
                zorder=2,
            )

    axis.set_title(
        _METRIC_LABELS[metric],
        loc="left",
        color=_METRIC_COLORS[metric],
        fontweight="bold",
        pad=1.6,
    )
    axis.set_xlabel("Age (days)")
    if long_gap:
        if show_ylabel:
            axis.set_ylabel("Long gap correlation")
        axis.set_xlim(175, 680)
        axis.set_xticks([200, 350, 500, 650])
        axis.set_ylim(-0.08, 0.75)
    else:
        if show_ylabel:
            axis.set_ylabel(
                r"Day $\mathrm{i}$ early vs Day $\mathrm{i}+1$ correlation"
            )
        axis.set_xlim(80, 680)
        axis.set_xticks([120, 300, 480, 660])
        axis.set_ylim(-0.04, 1.0)
    despine(axis)


def _legend_handle(colors: list[str]) -> tuple[Line2D, ...]:
    """Return the three metric-colored dots used for one legend entry."""

    return tuple(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=3.3,
        )
        for color in colors
    )


def _add_experience_legend(
    figure: Figure,
    *,
    experiences: range,
    labels: dict[int, str],
    anchor_y: float,
    columns: int,
) -> None:
    """Add the accepted metric-aware experience legend."""

    handles = [
        _legend_handle(
            [_group_color(metric, experience) for metric in DISPLAY_METRICS]
        )
        for experience in experiences
    ]
    figure.legend(
        handles,
        [labels[experience] for experience in experiences],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        loc="upper center",
        bbox_to_anchor=(0.55, anchor_y),
        ncol=columns,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.78 if columns == 5 else 0.80,
    )


def _add_panel_letters(axes: list[plt.Axes], letters: tuple[str, ...]) -> None:
    """Place one lowercase letter at the accepted location on each axis."""

    for axis, letter in zip(axes, letters, strict=True):
        axis.text(
            -0.18,
            1.08,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="black",
            clip_on=False,
        )


def _validate_page_support(
    within_points: pd.DataFrame,
    long_points: pd.DataFrame,
    run_mode: RunMode,
) -> None:
    """Check the accepted full cohort."""

    if run_mode != "full":
        raise ValueError("run_mode must be 'full'")
    for metric in DISPLAY_METRICS:
        within = within_points.loc[within_points["metric"].eq(metric)]
        long = long_points.loc[long_points["metric"].eq(metric)]
        if len(within) != 59 or within["animal_id"].nunique() != 22:
            raise RuntimeError(
                f"{metric}: expected 59 recording cycles from 22 animals"
            )
        if len(long) != 36 or long["animal_id"].nunique() != 14:
            raise RuntimeError(f"{metric}: expected 36 long gap pairs from 14 animals")


def assemble_age_association_page(
    within_points: pd.DataFrame,
    within_display_lines: pd.DataFrame,
    long_points: pd.DataFrame,
    long_display_lines: pd.DataFrame,
    *,
    run_mode: RunMode,
    output_dir: Path,
    random_seed: int,
) -> FigureAssemblyResult:
    """Arrange the six calculated age associations as the accepted page."""

    destination = _validate_output_request(
        run_mode=run_mode,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    _validate_page_support(within_points, long_points, run_mode)
    with version_record_style(_DISPLAY_STYLE):
        with mpl.rc_context({"mathtext.fontset": "dejavusans"}):
            figure, axes = plt.subplots(
                2,
                3,
                figsize=_ASSEMBLY_SIZE_IN,
                sharex="row",
                sharey="row",
            )
            for index, metric in enumerate(DISPLAY_METRICS):
                _draw_age_axis(
                    axes[0, index],
                    within_points,
                    within_display_lines,
                    metric=metric,
                    long_gap=False,
                    show_ylabel=index == 0,
                )
                _draw_age_axis(
                    axes[1, index],
                    long_points,
                    long_display_lines,
                    metric=metric,
                    long_gap=True,
                    show_ylabel=index == 0,
                )
            _add_panel_letters(list(axes[0]), ("a", "b", "c"))
            _add_panel_letters(list(axes[1]), ("d", "e", "f"))
            _add_experience_legend(
                figure,
                experiences=range(5),
                labels=_GROUP_LABELS,
                anchor_y=0.995,
                columns=5,
            )
            _add_experience_legend(
                figure,
                experiences=range(1, 5),
                labels=_TRANSITION_LABELS,
                anchor_y=0.505,
                columns=4,
            )
            figure.subplots_adjust(
                left=0.078,
                right=0.995,
                bottom=_ASSEMBLY_BOTTOM,
                top=_ASSEMBLY_TOP,
                wspace=0.23,
                hspace=_ASSEMBLY_HSPACE,
            )
            output_paths = save_figure(
                figure,
                destination / "Extended_Data_Figure_7",
                formats=("svg", "pdf", "png", "tiff"),
                raster_dpi=600,
                preserve_canvas=True,
            )
            plt.close(figure)
    root = REPOSITORY_ROOT.resolve()
    return FigureAssemblyResult(
        artifact_key=_ARTIFACT_KEY,
        run_mode=run_mode,
        output_paths=tuple(
            path.resolve() for path in output_paths
        ),
    )
