"""Draw ED7 age-association panels from included observations and model coefficients.

The launcher supplies the data and semantic palettes. The renderer plots
within-cycle lines using saved age slopes, without fitting upstream models.
"""

from __future__ import annotations

from pathlib import Path

from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.legend_handler import HandlerTuple

from matplotlib.lines import Line2D

METRICS = (
    {
        "key": "place",
        "label": "Place map",
        "metric": "place_map_correlation",
    },
    {
        "key": "pv",
        "label": "PV",
        "metric": "pv_correlation",
    },
    {
        "key": "rate",
        "label": "Ensemble rate",
        "metric": "ensemble_rate_correlation",
    },
)

STAGE_LABELS = {
    0: "Cycle 0 (naive)",
    1: "Cycle 1",
    2: "Cycle 2",
    3: "Cycle 3",
    4: "Cycle 4",
}

TRANSITION_LABELS = {
    1: "Cycle 1 (0-3 mo)",
    2: "Cycle 2 (3-6 mo)",
    3: "Cycle 3 (6-9 mo)",
    4: "Cycle 4 (9-12 mo)",
}

MAINTAINED_FIRST30_SCOPE = "accepted_main_figure3_first30_fixed_reference"

ROW_SIZE_IN = (7.15, 2.55)

PRIOR_ASSEMBLY_SIZE_IN = (7.5, 5.15)

ASSEMBLY_SIZE_IN = (7.5, 4.92)

_TITLE_SLOT_IN = (PRIOR_ASSEMBLY_SIZE_IN[1] - ASSEMBLY_SIZE_IN[1]) / 2.0

_PRIOR_AXES_HEIGHT_IN = (
    (0.86 - 0.095) * PRIOR_ASSEMBLY_SIZE_IN[1] / (2.0 + 0.72)
)

_COMPACT_BOTTOM = 0.095 * PRIOR_ASSEMBLY_SIZE_IN[1] / ASSEMBLY_SIZE_IN[1]

_COMPACT_HSPACE = (
    0.72 * _PRIOR_AXES_HEIGHT_IN - _TITLE_SLOT_IN
) / _PRIOR_AXES_HEIGHT_IN

_COMPACT_TOP = (
    0.095 * PRIOR_ASSEMBLY_SIZE_IN[1]
    + 2.0 * _PRIOR_AXES_HEIGHT_IN
    + _COMPACT_HSPACE * _PRIOR_AXES_HEIGHT_IN
) / ASSEMBLY_SIZE_IN[1]

PANEL_LETTERS_TOP = ("a", "b", "c")

PANEL_LETTERS_BOTTOM = ("d", "e", "f")

TOP_YLABEL = r"Day $\mathrm{i}$ early vs Day $\mathrm{i}+1$ correlation"

BOTTOM_YLABEL = "Long gap correlation"

X_LABEL = "Age (days)"

def pick(row: pd.Series, names: Iterable[str]) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    raise KeyError(f"None of {tuple(names)} is present and finite")

def normalize_first30_plot(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename = {}
    if "animal" in result.columns and "animal_id" not in result.columns:
        rename["animal"] = "animal_id"
    if "first_value" in result.columns and "value" not in result.columns:
        rename["first_value"] = "value"
    result = result.rename(columns=rename)
    required = {"animal_id", "experience_index", "age_day1_days", "metric", "value"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise AssertionError(f"First30 plot table is missing {missing}")
    for column in ("experience_index", "age_day1_days", "value"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[list(required)].isna().any().any():
        raise AssertionError("First30 plot table has missing primary fields")
    result["animal_id"] = result["animal_id"].astype(str)
    if "source_scope" not in result.columns:
        result["source_scope"] = "maintained_first30_authority"
    result["source_scope"] = result["source_scope"].astype(str)
    return result

def model_lookup(models: pd.DataFrame, specification: str) -> pd.DataFrame:
    subset = models.copy()
    if "specification" in subset.columns:
        subset = subset.loc[subset["specification"].astype(str).eq(specification)]
    if len(subset) != 3:
        raise AssertionError(
            f"Expected three {specification} displayed models, found {len(subset)}"
        )
    return subset.set_index("metric", drop=False)

def model_values(row: pd.Series, *, q_family: str) -> dict[str, float | int]:
    beta = float(pick(row, ("beta_per_30d", "beta_age_per_30d")))
    p_value = float(
        pick(
            row,
            (
                "p_cluster",
                "p_cluster_two_sided",
                "p_CR1_student_t_G_minus_1",
            ),
        )
    )
    q_candidates = (
        "q_bh3",
        "q_bh3_displayed",
        "q_BH3_age",
        "p_cluster_q_bh3",
    ) if q_family == "BH3" else (
        "p_cluster_q_bh4",
        "q_bh4",
        "q_BH4_age",
    )
    q_value = float(pick(row, q_candidates))
    return {
        "beta": beta,
        "p": p_value,
        "q": q_value,
        "n": int(pick(row, ("n_rows", "n_observations"))),
        "mice": int(pick(row, ("n_animals", "n_mice"))),
    }

def tuple_handle(colors: list[str]) -> tuple[Line2D, ...]:
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

def add_panel_letters(axes: Iterable[plt.Axes], letters: Iterable[str]) -> None:
    for ax, letter in zip(axes, letters):
        ax.text(
            -0.18,
            1.08,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

def draw_first30_axes(
    axes: Iterable[plt.Axes],
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    *,
    model_specification: str = "maintained59_primary",
) -> None:
    axes = list(axes)
    lookup = model_lookup(models, model_specification)
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        subset = frame.loc[frame["metric"].eq(spec["metric"])].copy()
        model = model_values(lookup.loc[spec["metric"]], q_family="BH3")
        for stage, group in subset.groupby("experience_index", sort=True):
            stage = int(stage)
            color = palettes[spec["key"]][stage]
            ax.scatter(
                group["age_day1_days"],
                group["value"],
                s=12,
                marker="o",
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
                zorder=3,
            )
            x = group["age_day1_days"].to_numpy(dtype=float)
            y = group["value"].to_numpy(dtype=float)
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
            line_y = float(np.mean(y)) + float(model["beta"]) * (
                line_x - float(np.mean(x))
            ) / 30.0
            ax.plot(line_x, line_y, color=color, linewidth=1.05, zorder=2)
        ax.set_title(
            spec["label"],
            loc="left",
            fontweight="bold",
            color=bases[spec["key"]],
            pad=1.6,
        )
        ax.set_xlabel(X_LABEL)
        if index == 0:
            ax.set_ylabel(TOP_YLABEL)
        ax.set_xlim(80, 680)
        ax.set_xticks([120, 300, 480, 660])
        ax.set_ylim(-0.04, 1.0)
    add_panel_letters(axes, PANEL_LETTERS_TOP)

def draw_long_gap_axes(
    axes: Iterable[plt.Axes],
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    *,
    bottom_scope: str,
) -> None:
    axes = list(axes)
    expected_specification = "clean36" if bottom_scope == "clean36" else "maximal38"
    lookup = model_lookup(models, expected_specification)
    for index, (ax, spec) in enumerate(zip(axes, METRICS)):
        subset = frame.loc[frame["metric"].eq(spec["metric"])].copy()
        model = model_values(lookup.loc[spec["metric"]], q_family="BH4")
        for experience, group in subset.groupby("experience", sort=True):
            experience = int(experience)
            color = palettes[spec["key"]][experience]
            ax.scatter(
                group["age_dst_days"],
                group["value"],
                s=12,
                marker="o",
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=0.90,
                zorder=3,
            )
            x = group["age_dst_days"].to_numpy(dtype=float)
            y = group["value"].to_numpy(dtype=float)
            if len(x) >= 3 and np.std(x) > 0:
                line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
                line_y = float(np.mean(y)) + float(model["beta"]) * (
                    line_x - float(np.mean(x))
                ) / 30.0
                ax.plot(line_x, line_y, color=color, linewidth=1.05, zorder=2)
        ax.set_title(
            spec["label"],
            loc="left",
            fontweight="bold",
            color=bases[spec["key"]],
            pad=1.6,
        )
        ax.set_xlabel(X_LABEL)
        if index == 0:
            ax.set_ylabel(BOTTOM_YLABEL)
        ax.set_xlim(175, 680)
        ax.set_xticks([200, 350, 500, 650])
        ax.set_ylim(-0.08, 0.75)
    add_panel_letters(axes, PANEL_LETTERS_BOTTOM)

def add_stage_legend(
    fig: plt.Figure,
    palettes: dict[str, dict[int, str]],
    *,
    anchor_y: float,
) -> None:
    stage_handles = [
        tuple_handle([palettes[spec["key"]][stage] for spec in METRICS])
        for stage in range(5)
    ]
    fig.legend(
        stage_handles,
        [STAGE_LABELS[stage] for stage in range(5)],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        loc="upper center",
        bbox_to_anchor=(0.55, anchor_y),
        ncol=5,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.78,
    )

def add_transition_legend(
    fig: plt.Figure,
    palettes: dict[str, dict[int, str]],
    *,
    anchor_y: float,
) -> None:
    transition_handles = [
        tuple_handle([palettes[spec["key"]][experience] for spec in METRICS])
        for experience in range(1, 5)
    ]
    fig.legend(
        transition_handles,
        [TRANSITION_LABELS[experience] for experience in range(1, 5)],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.18)},
        loc="upper center",
        bbox_to_anchor=(0.55, anchor_y),
        ncol=4,
        handlelength=1.55,
        handletextpad=0.25,
        columnspacing=0.80,
    )

def render_first30_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
    *,
    model_specification: str = "maintained59_primary",
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=ROW_SIZE_IN, sharex=True, sharey=True)
    draw_first30_axes(
        axes,
        frame,
        models,
        palettes,
        bases,
        model_specification=model_specification,
    )
    add_stage_legend(fig, palettes, anchor_y=0.995)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.205, top=0.72, wspace=0.23)
    return save_figure(fig, stem)

def render_long_gap_row(
    frame: pd.DataFrame,
    models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
    *,
    bottom_scope: str,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=ROW_SIZE_IN, sharex=True, sharey=True)
    draw_long_gap_axes(
        axes,
        frame,
        models,
        palettes,
        bases,
        bottom_scope=bottom_scope,
    )
    add_transition_legend(fig, palettes, anchor_y=0.995)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.205, top=0.72, wspace=0.23)
    return save_figure(fig, stem)

def render_six_panel_assembly(
    first30: pd.DataFrame,
    first30_models: pd.DataFrame,
    long_gap: pd.DataFrame,
    long_models: pd.DataFrame,
    palettes: dict[str, dict[int, str]],
    bases: dict[str, str],
    stem: Path,
    *,
    bottom_scope: str,
    first30_model_specification: str = "maintained59_primary",
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=ASSEMBLY_SIZE_IN,
        sharex="row",
        sharey="row",
    )
    draw_first30_axes(
        axes[0],
        first30,
        first30_models,
        palettes,
        bases,
        model_specification=first30_model_specification,
    )
    draw_long_gap_axes(
        axes[1],
        long_gap,
        long_models,
        palettes,
        bases,
        bottom_scope=bottom_scope,
    )
    add_stage_legend(fig, palettes, anchor_y=0.995)
    add_transition_legend(fig, palettes, anchor_y=0.505)
    fig.subplots_adjust(
        left=0.078,
        right=0.995,
        bottom=_COMPACT_BOTTOM,
        top=_COMPACT_TOP,
        wspace=0.23,
        hspace=_COMPACT_HSPACE,
    )
    return save_figure(fig, stem)

FORMATS = ("svg", "pdf", "png", "tiff")

def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

def save_figure(figure: Any, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in FORMATS:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "format": extension,
            "facecolor": "white",
            "transparent": False,
        }
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths
