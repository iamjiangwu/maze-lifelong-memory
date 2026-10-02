"""Shared plotting primitives for first-30-minute Extended Data Figure 2.

This is a plot-only renderer.  It reads the frozen 30-minute source tables,
never refits a policy or reruns the stochastic walk. The current caption-layer
renderer assembles the displayed panels from these primitives.

Clock definition: the first 30 cumulative minutes of TimeInMaze from session
start.  All behavioral modes advance the clock; turning policies are estimated
from Explore-mode junction decisions only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
from numbers import Number
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ca_analysis.behavior_figure_colors import (
    DIAGNOSTIC_COLORS,
    FIGURE_NEUTRAL_COLORS,
    TURNING_MODEL_COLORS,
    TURNING_POLICY_COLORS,
    get_behavior_group_colors,
)
from ca_analysis.figure_panels import (
    GRID_ROUTE_CHOICE_SCHEMATIC_COLORS,
    GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY,
    GRID_ROUTE_CHOICE_SCHEMATIC_TEXT,
    render_grid_route_choice_schematic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_ROOT = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed30min_promoted_2026-08-04"
    / "source_tables"
)
DEFAULT_NATURAL_ROOT = (
    REPO_ROOT
    / "docs"
    / "ed03_turning_bias_fixed_time_20_30min_explore_2026-08-04"
    / "source_tables"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_03"
    / "fixed_time_30min_full_story_2026-08-04"
)

WINDOW_MINUTES = 30
FORMATS = ("svg", "pdf", "png", "tiff")
POLICY_ORDER = ("Bf", "Ba", "Lf", "Lo")
POLICY_LABEL = {
    "Bf": r"$P_{SF}$",
    "Ba": r"$P_{SA}$",
    "Lf": r"$P_{BF}$",
    "Lo": r"$P_{BS}$",
}
POLICY_DEFINITION = {
    "Bf": "Parent arrival: enter a child rather than reverse to the parent/stem.",
    "Ba": (
        "Conditional on entering a child: choose the orientation-defined "
        "alternating rather than non-alternating child."
    ),
    "Lf": (
        "Child arrival: leave the arrival-child branch rather than reverse "
        "along it."
    ),
    "Lo": (
        "Conditional on leaving the arrival-child branch: take the parent/stem "
        "rather than the sibling child."
    ),
}
POLICY_SHORT = {
    "Bf": "enter child vs reverse",
    "Ba": "alternating vs non-alternating child",
    "Lf": "leave arrival-child branch vs reverse",
    "Lo": "parent/stem vs sibling",
}
RANDOM_REFERENCE = {"Bf": 2.0 / 3.0, "Ba": 0.5, "Lf": 2.0 / 3.0, "Lo": 0.5}
GROUP_ORDER = ("naive", "LTM", "experienced")
GROUP_LABEL = {"naive": "Naive", "LTM": "Return", "experienced": "Second day"}
_BEHAVIOR_COLORS = get_behavior_group_colors()
GROUP_COLOR = {
    "naive": _BEHAVIOR_COLORS["naive"],
    "LTM": _BEHAVIOR_COLORS["LTM"],
    "experienced": _BEHAVIOR_COLORS["experienced"],
}
GROUP_MARKER = {"naive": "o", "LTM": "s", "experienced": "^"}
POLICY_MARKER = {"Bf": "o", "Ba": "s", "Lf": "^", "Lo": "D"}
DEPTH_ORDER = ("Lo_shallow", "Lo_mid", "Lo_deep")
DEPTH_LABEL = {
    "Lo_shallow": "Shallow\nlevels 1-2",
    "Lo_mid": "Middle\nlevels 3-4",
    "Lo_deep": "Deep\nlevel 5",
}

INK = FIGURE_NEUTRAL_COLORS["ink"]
BACKGROUND = FIGURE_NEUTRAL_COLORS["background"]
MUTED = FIGURE_NEUTRAL_COLORS["muted"]
REFERENCE = FIGURE_NEUTRAL_COLORS["reference"]
LIGHT = FIGURE_NEUTRAL_COLORS["light"]
PALE = FIGURE_NEUTRAL_COLORS["pale"]
FOUR = TURNING_MODEL_COLORS["four_bias"]
DEPTH = TURNING_MODEL_COLORS["depth_stratified"]
PASS = DIAGNOSTIC_COLORS["pass"]
FAIL = DIAGNOSTIC_COLORS["fail"]

PANEL_META: dict[str, dict[str, str]] = {
    "a": {
        "stem": "ed03a_four_policy_schematic_fixed_30min_explore",
        "role": "definition",
        "claim": "Four local route-choice probabilities define the fixed-clock walk.",
    },
    "b": {
        "stem": "ed03b_naive_vs_return_policy_scatter_fixed_30min_explore",
        "role": "primary",
        "claim": "Return sessions shift the four-policy profile relative to Naive sessions.",
    },
    "c": {
        "stem": "ed03c_return_vs_second_day_policy_scatter_fixed_30min_explore",
        "role": "primary",
        "claim": "Second-day policies remain related to, but are not equal to, Return policies.",
    },
    "d": {
        "stem": "ed03d_four_policy_contributions_fixed_30min_explore",
        "role": "mechanism",
        "claim": "One-at-a-time policy substitutions make unequal contributions to simulated efficiency.",
    },
    "e": {
        "stem": "ed03e_four_policy_gap_reconstruction_fixed_30min_explore",
        "role": "model_test",
        "claim": "The baseline four-policy walk reconstructs only part of the observed efficiency gap.",
    },
    "f": {
        "stem": "ed03f_four_policy_calibration_fixed_30min_explore",
        "role": "model_test",
        "claim": "Four-policy calibration is evaluated across Naive, Return, and Second-day sessions.",
    },
    "g": {
        "stem": "ed03g_depth_pbs_profile_fixed_30min_explore",
        "role": "refinement",
        "claim": "The P_BS profile and its experience contrasts vary with maze depth.",
    },
    "h": {
        "stem": "ed03h_four_vs_depth_reconstruction_fixed_30min_explore",
        "role": "primary",
        "claim": "Depth-varying P_BS improves gap reconstruction relative to four fixed policies.",
    },
    "i": {
        "stem": "ed03i_depth_calibration_fixed_30min_explore",
        "role": "model_test",
        "claim": "Depth-model calibration passes declared gates for Naive and Return, but not Second day.",
    },
    "j": {
        "stem": "ed03j_estimator_exposure_support_fixed_30min_explore",
        "role": "audit",
        "claim": "Model ordering is estimator-robust, with group-specific exposure and support made explicit.",
    },
}

SOURCE_FILES = {
    "strict_policy_rows": "primary_30min_strict_joint_equal_junction_policy_rows.csv",
    "strict_policy_contrasts": "primary_30min_strict_joint_policy_cluster_stats.csv",
    "strict_contributions": "primary_30min_strict_joint_policy_contribution_summary.csv",
    "strict_gap": "primary_30min_strict_joint_gap_recapitulation_summary.csv",
    "natural_policy_rows": "fixed_time_policy_rows.csv",
    "natural_policy_contrasts": "equal_junction_policy_group_cluster_stats.csv",
    "natural_depth_contrasts": "equal_junction_depth_group_cluster_stats.csv",
    "natural_calibration": "equal_junction_calibration_summary.csv",
    "natural_simulation_rows": "equal_junction_simulation_session_rows.csv",
}


def configure_style() -> None:
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.titlesize": 6.7,
            "axes.labelsize": 6.2,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 5.0,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Number):
        converted = float(value)
        return None if not np.isfinite(converted) else converted
    if value is None or isinstance(value, str):
        return value
    return str(value)


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _read_30min(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"animal_id": str})
    if "window_minutes" not in frame.columns:
        raise ValueError(f"Quantitative source lacks window_minutes: {path}")
    minutes = pd.to_numeric(frame["window_minutes"], errors="coerce")
    frame = frame.loc[minutes.eq(WINDOW_MINUTES)].copy()
    if frame.empty:
        raise ValueError(f"No 30-minute rows in {path}")
    frame["window_minutes"] = WINDOW_MINUTES
    return frame.reset_index(drop=True)


def load_sources(
    strict_root: Path = DEFAULT_STRICT_ROOT,
    natural_root: Path = DEFAULT_NATURAL_ROOT,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    strict_root = Path(strict_root).resolve()
    natural_root = Path(natural_root).resolve()
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for key, filename in SOURCE_FILES.items():
        root = strict_root if key.startswith("strict_") else natural_root
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[key] = _read_30min(path)
        paths[key] = path
    validate_source_cohorts(frames)
    return frames, paths


def validate_source_cohorts(frames: dict[str, pd.DataFrame]) -> None:
    strict_counts = frames["strict_policy_rows"].groupby("group").size().to_dict()
    if strict_counts != {"LTM": 58, "naive": 22}:
        raise ValueError(f"Strict cohort drifted: {strict_counts}")
    natural_counts = frames["natural_policy_rows"].groupby("group").size().to_dict()
    if natural_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural cohort drifted: {natural_counts}")
    simulation_counts = frames["natural_simulation_rows"].groupby("group").size().to_dict()
    if simulation_counts != {"LTM": 58, "experienced": 71, "naive": 23}:
        raise ValueError(f"Natural calibration cohort drifted: {simulation_counts}")
    for key, frame in frames.items():
        if not pd.to_numeric(frame["window_minutes"], errors="coerce").eq(30).all():
            raise ValueError(f"Non-30-minute row survived in {key}")


def save_panel(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _correct_policy_descriptions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "metric" in out.columns:
        out["display_description"] = out["metric"].map(POLICY_SHORT)
        if "description" in out.columns:
            out["description"] = out["display_description"]
    return out


def panel_a() -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    text = deepcopy(GRID_ROUTE_CHOICE_SCHEMATIC_TEXT)
    title_map = {
        "stem_to_branch": r"$P_{SA}$",
        "stem_forward": r"$P_{SF}$",
        "branch_to_stem": r"$P_{BS}$",
        "branch_forward": r"$P_{BF}$",
    }
    for row in text["rows"]:
        row["title"] = title_map[row["key"]]
        if row["key"] == "stem_to_branch":
            row["annotations"] = [
                {
                    "text": "preceding\nturn",
                    "xy": (2.92, 3.52),
                    "ha": "right",
                    "va": "center",
                }
            ]
    fig, ax, schematic, _ = render_grid_route_choice_schematic(
        layout={"panel_size": (2.8, 3.95), "tight_layout": False},
        geometry=deepcopy(GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 1.45,
            "arrow_linewidth": 2.05,
            "previous_arrow_linewidth": 2.05,
            "arrow_mutation_scale": 10,
            "title_fontsize": 8.0,
            "title_fontweight": "normal",
            "or_fontsize": 8.0,
            "annotation_fontsize": 8.4,
        },
        text=text,
        save=False,
    )
    fig.set_size_inches(5.0, 2.62, forward=True)
    ax.set_position([0.015, 0.05, 0.43, 0.91])
    y0 = 0.91
    definitions = []
    for index, metric in enumerate(POLICY_ORDER):
        y = y0 - 0.145 * index
        fig.text(0.48, y, POLICY_LABEL[metric], ha="left", va="top", fontsize=7.3, color=INK)
        fig.text(
            0.55,
            y,
            POLICY_DEFINITION[metric],
            ha="left",
            va="top",
            fontsize=5.8,
            color=INK,
            wrap=True,
        )
        definitions.append(
            {
                "window_minutes": 30,
                "metric": metric,
                "policy_label": POLICY_LABEL[metric],
                "plain_language_definition": POLICY_DEFINITION[metric],
                "equal_next_edge_reference": RANDOM_REFERENCE[metric],
            }
        )
    fig.text(
        0.48,
        0.27,
        "Analysis window",
        ha="left",
        va="top",
        fontsize=6.0,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.48,
        0.225,
        "First 30 cumulative min in maze. All modes advance the clock;\n"
        "only Explore-mode junction decisions estimate the policies.",
        ha="left",
        va="top",
        fontsize=5.7,
        color=MUTED,
        linespacing=1.25,
    )
    fig.text(
        0.48,
        0.09,
        "Walk inputs: four session-level probabilities; fixed within depth.",
        ha="left",
        va="top",
        fontsize=5.7,
        color=FOUR,
    )
    schematic_rows = pd.DataFrame(
        [
            {
                "window_minutes": 30,
                "helper": "render_grid_route_choice_schematic",
                "row_key": row["key"],
                "policy_label": row["title"],
            }
            for row in schematic["text"]["rows"]
        ]
    )
    return fig, {"policy_definitions": pd.DataFrame(definitions), "schematic_rows": schematic_rows}


def prepare_two_axis_scatter(
    policy_rows: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    contrast_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = policy_rows.loc[policy_rows["group"].isin(groups)].copy()
    points = numeric(points, (*POLICY_ORDER,))
    stats = contrasts.loc[
        contrasts["contrast_id"].eq(contrast_id) & contrasts["metric"].isin(POLICY_ORDER)
    ].copy()
    stats = _correct_policy_descriptions(stats)
    stats = numeric(
        stats,
        (
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "q_bh4_within_window_contrast",
        ),
    )
    stats["metric"] = pd.Categorical(stats["metric"], POLICY_ORDER, ordered=True)
    stats = stats.sort_values("metric").reset_index(drop=True)
    if len(stats) != 4:
        raise ValueError(f"Expected four {contrast_id} policy contrasts")
    return points.reset_index(drop=True), stats


def _q_text(value: float) -> str:
    if value < 0.001:
        return "q<0.001"
    return f"q={value:.3f}"


def policy_scatter_panel(
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    *,
    footer: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    metric_pairs = (("Bf", "Lf"), ("Ba", "Lo"))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.28), sharex=False, sharey=False)
    for ax, pair in zip(axes, metric_pairs):
        x_metric, y_metric = pair
        ax.axvline(
            RANDOM_REFERENCE[x_metric], color=REFERENCE, linestyle="--", linewidth=0.8
        )
        ax.axhline(
            RANDOM_REFERENCE[y_metric], color=REFERENCE, linestyle="--", linewidth=0.8
        )
        for group in groups:
            sub = points.loc[points["group"].eq(group)]
            color = GROUP_COLOR[group]
            ax.scatter(
                sub[x_metric],
                sub[y_metric],
                s=11,
                marker=GROUP_MARKER[group],
                facecolors=BACKGROUND,
                edgecolors=color,
                linewidths=0.55,
                alpha=0.62,
                label=(
                    f"{GROUP_LABEL[group]}  {len(sub)} sessions / "
                    f"{sub['animal_id'].astype(str).nunique()} mice"
                ),
            )
            ax.scatter(
                [sub[x_metric].mean()],
                [sub[y_metric].mean()],
                s=29,
                marker=GROUP_MARKER[group],
                facecolors=color,
                edgecolors=BACKGROUND,
                linewidths=0.7,
                zorder=4,
            )
        x_stat = contrasts.loc[contrasts["metric"].astype(str).eq(x_metric)].iloc[0]
        y_stat = contrasts.loc[contrasts["metric"].astype(str).eq(y_metric)].iloc[0]
        ax.text(
            0.03,
            0.97,
            (
                f"CR1 / BH4\n"
                f"{POLICY_LABEL[x_metric]}  {float(x_stat['beta_target_minus_baseline']):+.3f}, "
                f"{_q_text(float(x_stat['q_bh4_within_window_contrast']))}\n"
                f"{POLICY_LABEL[y_metric]}  {float(y_stat['beta_target_minus_baseline']):+.3f}, "
                f"{_q_text(float(y_stat['q_bh4_within_window_contrast']))}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.7,
            color=INK,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(POLICY_LABEL[x_metric])
        ax.set_ylabel(POLICY_LABEL[y_metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        handletextpad=0.25,
        columnspacing=1.0,
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=5.0, color=MUTED)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.17, top=0.85, wspace=0.34)
    return fig, {"session_points": points, "contrasts": contrasts}


def panel_d(frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["parameter"] = pd.Categorical(rows["parameter"], POLICY_ORDER, ordered=True)
    rows = rows.sort_values("parameter").reset_index(drop=True)
    rows = numeric(
        rows,
        ("delta_efficiency_mean", "delta_efficiency_ci_low", "delta_efficiency_ci_high"),
    )
    fig, ax = plt.subplots(figsize=(3.9, 2.15))
    y = np.arange(len(rows))[::-1]
    mean = 100 * rows["delta_efficiency_mean"].to_numpy(dtype=float)
    low = 100 * rows["delta_efficiency_ci_low"].to_numpy(dtype=float)
    high = 100 * rows["delta_efficiency_ci_high"].to_numpy(dtype=float)
    ax.axvline(0, color=INK, linewidth=0.7)
    for yi, value, lo, hi, parameter in zip(
        y, mean, low, high, rows["parameter"]
    ):
        parameter = str(parameter)
        color = TURNING_POLICY_COLORS[parameter]
        ax.hlines(yi, 0, value, color=color, linewidth=1.2)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt=POLICY_MARKER[parameter],
            color=color,
            markersize=4.0,
            capsize=1.8,
            elinewidth=0.9,
        )
        ax.text(
            value,
            yi + 0.17,
            f"{value:+.1f}",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=color,
        )
    ax.set_yticks(y, [POLICY_LABEL[str(item)] for item in rows["parameter"]])
    ax.set_xlabel(
        "Return-own minus same Return session with one policy\n"
        "fixed to the Naive donor mean (percentage points)"
    )
    ax.set_xlim(min(-4.8, low.min() - 0.8), max(10.5, high.max() + 0.8))
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Baseline four-policy walk | one substitution at a time", loc="left")
    return fig, {"policy_contributions": rows}




def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "pass"))


def normalize_model(value: Any) -> str:
    text = str(value).lower()
    if "depth" in text:
        return "depth_stratified"
    if "four" in text:
        return "four_bias"
    raise ValueError(f"Unrecognized model: {value!r}")


def prepare_calibration(
    session_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if model not in {"four_bias", "depth_stratified"}:
        raise ValueError(model)
    simulated_column = (
        "four_own_half_cover_eff" if model == "four_bias" else "depth_own_half_cover_eff"
    )
    points = session_frame.loc[session_frame["group"].isin(GROUP_ORDER)].copy()
    points = numeric(points, ("real_half_cover_eff", simulated_column))
    points = points.dropna(subset=["real_half_cover_eff", simulated_column]).reset_index(drop=True)
    summary = summary_frame.copy()
    summary["model_norm"] = summary["model"].map(normalize_model)
    summary = summary.loc[
        summary["model_norm"].eq(model) & summary["group"].isin(GROUP_ORDER)
    ].copy()
    summary = numeric(summary, ("mean_error", "spearman_rho", "mae"))
    if "combined_gate_pass" in summary.columns:
        summary["combined_gate_pass"] = as_bool(summary["combined_gate_pass"])
    expected = {"naive": 23, "LTM": 58, "experienced": 71}
    observed = points.groupby("group").size().to_dict()
    if observed != expected:
        raise ValueError(f"{model} calibration rows drifted: {observed}")
    if len(summary) != 3:
        raise ValueError(f"{model} calibration requires three summary rows")
    return points, summary.reset_index(drop=True), simulated_column




def prepare_depth_profile(
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = policy_rows.loc[policy_rows["group"].isin(GROUP_ORDER)].copy()
    if "primary_depth_support" in rows.columns:
        rows = rows.loc[as_bool(rows["primary_depth_support"])].copy()
    rows = numeric(rows, DEPTH_ORDER)
    melted = rows[["nickname", "animal_id", "group", *DEPTH_ORDER]].melt(
        id_vars=["nickname", "animal_id", "group"],
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"]).reset_index(drop=True)
    summary = (
        melted.groupby(["group", "metric"], as_index=False)["value"]
        .agg(
            mean="mean",
            q25=lambda values: np.quantile(values, 0.25),
            q75=lambda values: np.quantile(values, 0.75),
            n_session_rows="size",
        )
    )
    contrasts = contrast_frame.copy()
    if "support_policy" in contrasts.columns:
        contrasts = contrasts.loc[contrasts["support_policy"].eq("all_depth_strata_n_ge_10")]
    contrasts = contrasts.loc[
        contrasts["contrast_id"].isin(("return_vs_naive", "second_vs_return"))
        & contrasts["metric"].isin(DEPTH_ORDER)
    ].copy()
    contrasts = numeric(
        contrasts,
        (
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "q_bh3_within_window_contrast",
        ),
    )
    if len(summary) != 9 or len(contrasts) != 6:
        raise ValueError(
            f"Depth profile requires 9 group summaries and 6 contrasts; got {len(summary)}, {len(contrasts)}"
        )
    return rows.reset_index(drop=True), summary.reset_index(drop=True), contrasts.reset_index(drop=True)












def make_contact_sheet(
    panel_pngs: dict[str, Path],
    output_root: Path,
    dpi: int = 300,
) -> list[Path]:
    width_mm, height_mm = 180, 225
    width_px = round(width_mm / 25.4 * dpi)
    height_px = round(height_mm / 25.4 * dpi)
    margin = round(5 / 25.4 * dpi)
    gutter_x = round(4 / 25.4 * dpi)
    gutter_y = round(3.2 / 25.4 * dpi)
    header = round(8 / 25.4 * dpi)
    cell_w = (width_px - 2 * margin - gutter_x) // 2
    cell_h = (height_px - 2 * margin - header - 4 * gutter_y) // 5
    canvas = Image.new("RGB", (width_px, height_px), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    regular_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    title_font = ImageFont.truetype(regular_path, round(6.6 / 72 * dpi))
    letter_font = ImageFont.truetype(bold_path, round(8.2 / 72 * dpi))
    draw.text(
        (margin, margin),
        "Extended Data Fig. 3 | first 30 min TimeInMaze; Explore decisions only",
        fill=INK,
        font=title_font,
    )
    for index, panel in enumerate("abcdefghij"):
        image = Image.open(panel_pngs[panel]).convert("RGB")
        scale = min(cell_w / image.width, cell_h / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        row, column = divmod(index, 2)
        x0 = margin + column * (cell_w + gutter_x) + (cell_w - resized.width) // 2
        y0 = margin + header + row * (cell_h + gutter_y) + (cell_h - resized.height) // 2
        canvas.paste(resized, (x0, y0))
        cell_x = margin + column * (cell_w + gutter_x)
        cell_y = margin + header + row * (cell_h + gutter_y)
        draw.text((cell_x, cell_y), panel, fill=INK, font=letter_font)
    png_path = output_root / "ED03_fixed_30min_full_story_review_composite_180x225mm.png"
    pdf_path = output_root / "ED03_fixed_30min_full_story_review_composite_180x225mm.pdf"
    canvas.save(png_path, dpi=(dpi, dpi))
    canvas.save(pdf_path, resolution=dpi)
    return [png_path, pdf_path]


def validate_outputs(
    output_root: Path,
    panel_paths: dict[str, list[Path]],
    table_paths: list[Path],
    composite_paths: list[Path],
    dpi: int,
) -> dict[str, Any]:
    failures: list[str] = []
    palette_failures: list[str] = []
    expected_palette_by_panel = {
        "a": {
            GRID_ROUTE_CHOICE_SCHEMATIC_COLORS["active_arrow"],
            GRID_ROUTE_CHOICE_SCHEMATIC_COLORS["previous_arrow"],
        },
        "b": {GROUP_COLOR["naive"], GROUP_COLOR["LTM"], REFERENCE},
        "c": {GROUP_COLOR["LTM"], GROUP_COLOR["experienced"], REFERENCE},
        "d": set(TURNING_POLICY_COLORS.values()),
        "e": {FOUR},
        "f": set(GROUP_COLOR.values()),
        "g": set(GROUP_COLOR.values()),
        "h": {FOUR, DEPTH},
        "i": set(GROUP_COLOR.values()),
        "j": {*GROUP_COLOR.values(), FOUR, DEPTH},
    }
    forbidden_palette = {
        "#2474B5",
        "#C7473E",
        "#76589A",
        "#2489BC",
        "#69747D",
        "#D6DCE1",
        "#F0F2F4",
        "#65727D",
        "#D27842",
        "#9DA6AE",
    }
    allowed_palette_by_panel = {
        "a": {
            BACKGROUND,
            INK,
            MUTED,
            FOUR,
            GRID_ROUTE_CHOICE_SCHEMATIC_COLORS["cell_edge"],
            GRID_ROUTE_CHOICE_SCHEMATIC_COLORS["active_arrow"],
            GRID_ROUTE_CHOICE_SCHEMATIC_COLORS["previous_arrow"],
        },
        "b": {BACKGROUND, INK, MUTED, REFERENCE, GROUP_COLOR["naive"], GROUP_COLOR["LTM"]},
        "c": {BACKGROUND, INK, MUTED, REFERENCE, GROUP_COLOR["LTM"], GROUP_COLOR["experienced"]},
        "d": {BACKGROUND, INK, *TURNING_POLICY_COLORS.values()},
        "e": {BACKGROUND, INK, FOUR, PALE},
        "f": {BACKGROUND, INK, MUTED, LIGHT, *GROUP_COLOR.values()},
        "g": {BACKGROUND, INK, MUTED, LIGHT, *GROUP_COLOR.values()},
        "h": {BACKGROUND, INK, FOUR, DEPTH, PALE},
        "i": {BACKGROUND, INK, MUTED, LIGHT, PASS, *GROUP_COLOR.values()},
        "j": {
            BACKGROUND,
            INK,
            MUTED,
            LIGHT,
            PALE,
            FOUR,
            DEPTH,
            *GROUP_COLOR.values(),
        },
    }
    expected_panels = set(PANEL_META)
    if set(panel_paths) != expected_panels:
        failures.append(f"panel letters mismatch: {sorted(panel_paths)}")
    for panel, paths in panel_paths.items():
        suffixes = {path.suffix.lstrip(".") for path in paths}
        if suffixes != set(FORMATS):
            failures.append(f"panel {panel} formats: {sorted(suffixes)}")
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty: {path}")
        svg = next(path for path in paths if path.suffix == ".svg")
        svg_text = svg.read_text(encoding="utf-8")
        svg_lower = svg_text.lower()
        if "<text" not in svg_text:
            failures.append(f"SVG text is not editable: {svg}")
        if "1/32" in svg_text or "first_1_32" in svg_text:
            failures.append(f"deprecated decision-fraction label in {svg}")
        missing_colors = sorted(
            color
            for color in expected_palette_by_panel[panel]
            if color.lower() not in svg_lower
        )
        if missing_colors:
            palette_failures.append(
                f"panel {panel} omits canonical semantic colors: {missing_colors}"
            )
        stale_colors = sorted(
            color for color in forbidden_palette if color.lower() in svg_lower
        )
        if stale_colors:
            palette_failures.append(
                f"panel {panel} retains superseded colors: {stale_colors}"
            )
        observed_hex = {
            value.lower()
            for value in re.findall(r"#[0-9A-Fa-f]{6}", svg_text)
        }
        unexpected_colors = sorted(
            observed_hex.difference(
                color.lower() for color in allowed_palette_by_panel[panel]
            )
        )
        if unexpected_colors:
            palette_failures.append(
                f"panel {panel} contains undeclared semantic colors: {unexpected_colors}"
            )
        for suffix in (".png", ".tiff"):
            raster = next(path for path in paths if path.suffix == suffix)
            with Image.open(raster) as image:
                if suffix == ".tiff" and image.info.get("compression") != "tiff_lzw":
                    failures.append(f"TIFF is not LZW-compressed: {raster}")
                raster_dpi = image.info.get("dpi", (None, None))
                if raster_dpi[0] is not None and abs(float(raster_dpi[0]) - dpi) > 1.0:
                    failures.append(f"Raster dpi mismatch in {raster}: {raster_dpi}")
    for path in table_paths:
        frame = pd.read_csv(path)
        if "window_minutes" not in frame.columns:
            failures.append(f"Source Data lacks window_minutes: {path}")
        elif not pd.to_numeric(frame["window_minutes"], errors="coerce").eq(30).all():
            failures.append(f"Non-30-minute Source Data: {path}")
    composite_png = next(path for path in composite_paths if path.suffix == ".png")
    with Image.open(composite_png) as image:
        composite_size = list(image.size)
    failures.extend(palette_failures)
    qa = {
        "pass": not failures,
        "failures": failures,
        "panel_count": len(panel_paths),
        "panel_artifact_count": sum(len(paths) for paths in panel_paths.values()),
        "source_table_count": len(table_paths),
        "all_quantitative_source_rows_window_minutes": 30,
        "active_svg_deprecated_fraction_label_count": 0 if not any("fraction label" in item for item in failures) else None,
        "editable_svg_text_required": True,
        "semantic_palette_verified": not palette_failures,
        "expected_raster_dpi": dpi,
        "tiff_compression": "tiff_lzw",
        "composite_nominal_mm": [180, 225],
        "composite_pixel_size": composite_size,
        "plot_only_no_simulation_rerun": True,
    }
    return qa
