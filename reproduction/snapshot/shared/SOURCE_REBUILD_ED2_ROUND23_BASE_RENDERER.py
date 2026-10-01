"""Render the complete fixed-30-minute Extended Data Figure 3 story.

This is a plot-only renderer.  It reads the frozen 30-minute source tables,
never refits a policy or reruns the stochastic walk, and exports a ten-panel
evidence sequence that establishes the four-policy model before the
depth-varying refinement.

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
    "strict_robustness": "primary_30min_strict_joint_gap_robustness.csv",
    "natural_policy_rows": "fixed_time_policy_rows.csv",
    "natural_policy_contrasts": "equal_junction_policy_group_cluster_stats.csv",
    "natural_depth_contrasts": "equal_junction_depth_group_cluster_stats.csv",
    "natural_calibration": "equal_junction_calibration_summary.csv",
    "natural_simulation_rows": "equal_junction_simulation_session_rows.csv",
    "natural_exposure": "window_exposure_audit.csv",
    "natural_support": "estimator_support_comparison.csv",
    "natural_outcome_overlap": "predictor_outcome_overlap_group_summary.csv",
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
    overlap_counts = frames["natural_outcome_overlap"].set_index("group")["n_sessions"].to_dict()
    if overlap_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural outcome-overlap cohort drifted: {overlap_counts}")
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


def panel_e(frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    row = rows.loc[rows["model"].astype(str).str.contains("four")].copy()
    if len(row) != 1:
        raise ValueError("Panel e requires one strict four-policy gap row")
    row = numeric(
        row,
        ("delta_real", "delta_simulated", "fraction_recapitulated", "fraction_ci_low", "fraction_ci_high"),
    ).reset_index(drop=True)
    observed = float(row.loc[0, "delta_real"])
    simulated = float(row.loc[0, "delta_simulated"])
    low = observed * float(row.loc[0, "fraction_ci_low"])
    high = observed * float(row.loc[0, "fraction_ci_high"])
    fraction = 100 * float(row.loc[0, "fraction_recapitulated"])
    fraction_low = 100 * float(row.loc[0, "fraction_ci_low"])
    fraction_high = 100 * float(row.loc[0, "fraction_ci_high"])
    fig, ax = plt.subplots(figsize=(3.9, 2.0))
    ax.axvline(observed, color=INK, linewidth=1.1)
    ax.axvspan(observed - 0.0025, observed + 0.0025, color=PALE, zorder=0)
    ax.errorbar(
        simulated,
        0,
        xerr=[[simulated - low], [high - simulated]],
        fmt="o",
        color=FOUR,
        markersize=5.2,
        capsize=2.1,
        elinewidth=1.2,
    )
    ax.text(
        simulated,
        0.19,
        f"{fraction:.1f}% [{fraction_low:.1f}, {fraction_high:.1f}]",
        ha="center",
        va="bottom",
        color=FOUR,
        fontsize=6.0,
        fontweight="bold",
    )
    ax.text(
        observed,
        0.52,
        f"Observed gap = {observed:.3f}",
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=INK,
    )
    ax.set_yticks([0], ["Four-policy walk"])
    ax.set_ylim(-0.45, 0.72)
    ax.set_xlim(0, max(0.30, high + 0.03))
    ax.set_xlabel("Return - Naive half-coverage efficiency gap (absolute)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Baseline model reconstructs only part of the observed gap", loc="left")
    return fig, {"four_policy_gap": row}


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


def calibration_panel(
    session_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    model: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    points, summary, simulated_column = prepare_calibration(session_frame, summary_frame, model)
    title = "Four-policy walk" if model == "four_bias" else r"Depth-varying $P_{BS}$ walk"
    maximum = float(np.nanmax(points[["real_half_cover_eff", simulated_column]].to_numpy()))
    limit = max(0.62, np.ceil((maximum + 0.03) / 0.1) * 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.12), sharex=True, sharey=True)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = points.loc[points["group"].eq(group)]
        stat = summary.loc[summary["group"].eq(group)].iloc[0]
        color = GROUP_COLOR[group]
        ax.plot((0, limit), (0, limit), color=LIGHT, linestyle="--", linewidth=0.8)
        ax.scatter(
            sub["real_half_cover_eff"],
            sub[simulated_column],
            s=12,
            facecolors=BACKGROUND,
            edgecolors=color,
            linewidths=0.6,
            alpha=0.78,
        )
        passed = bool(stat["combined_gate_pass"])
        ax.text(
            0.04,
            0.96,
            (
                f"rho = {float(stat['spearman_rho']):.2f}\n"
                f"bias = {float(stat['mean_error']):+.3f}\n"
                f"{'PASS' if passed else 'FAIL'}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.9,
            color=PASS if passed else MUTED,
            fontweight="bold" if passed else "normal",
        )
        ax.set_title(
            f"{GROUP_LABEL[group]}\n{len(sub)} sessions / {sub['animal_id'].astype(str).nunique()} mice",
            loc="left",
            color=color,
            pad=2.5,
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed efficiency")
    axes[0].set_ylabel("Simulated efficiency")
    fig.suptitle(
        f"{title} calibration | descriptive gates: |bias| <= 0.05 and rho >= 0.40",
        x=0.06,
        y=1.015,
        ha="left",
        fontsize=6.7,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.79, wspace=0.25)
    return fig, {"session_points": points, "calibration_summary": summary}


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


def panel_g(
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows, summary, contrasts = prepare_depth_profile(policy_rows, contrast_frame)
    fig, (ax, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(5.0, 3.25),
        sharex=True,
        gridspec_kw={"height_ratios": (1.55, 1.0)},
    )
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.16, top=0.80, hspace=0.43)
    x = np.arange(3, dtype=float)
    for group in GROUP_ORDER:
        sub = summary.loc[summary["group"].eq(group)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        means = sub["mean"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        color = GROUP_COLOR[group]
        ax.fill_between(x, q25, q75, color=color, alpha=0.09, linewidth=0)
        ax.plot(
            x,
            means,
            marker=GROUP_MARKER[group],
            color=color,
            markersize=3.6,
            label=(
                f"{GROUP_LABEL[group]}  n="
                f"{int(sub['n_session_rows'].iloc[0])} sessions"
            ),
        )
    ax.set_ylabel(r"$P_{BS}$")
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x, [])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.985),
        ncol=3,
        handlelength=1.1,
        columnspacing=0.9,
    )
    ax.text(
        0.995,
        0.03,
        "Lines = session means; bands = session IQR",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.7,
        color=MUTED,
    )
    ax.spines["bottom"].set_visible(False)

    contrast_specs = (
        ("return_vs_naive", "Return - Naive", GROUP_COLOR["LTM"], "o", 0.055),
        ("second_vs_return", "Second day - Return", GROUP_COLOR["experienced"], "s", -0.055),
    )
    ax_delta.axhline(0, color=LIGHT, linewidth=0.75)
    for contrast_id, label, color, marker, offset in contrast_specs:
        sub = contrasts.loc[contrasts["contrast_id"].eq(contrast_id)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        mean = sub["beta_target_minus_baseline"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        ax_delta.errorbar(
            x + offset,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            fmt=marker,
            color=color,
            markersize=3.4,
            capsize=1.7,
            elinewidth=0.85,
            label=label,
        )
        for xi, hi, q in zip(x + offset, high, sub["q_bh3_within_window_contrast"]):
            ax_delta.text(
                xi,
                hi + 0.010,
                _q_text(float(q)),
                ha="center",
                va="bottom",
                fontsize=4.0,
                color=color,
                rotation=14,
            )
    ax_delta.set_ylabel("Policy contrast\n(mouse CR1)")
    ax_delta.set_ylim(-0.025, 0.34)
    ax_delta.set_xticks(x, [DEPTH_LABEL[key] for key in DEPTH_ORDER])
    ax_delta.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.31),
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
    )
    ax_delta.text(
        0.0,
        1.13,
        "95% CI; BH3 q values; root level 0 excluded",
        transform=ax_delta.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.6,
        color=MUTED,
    )
    return fig, {"supported_session_rows": rows, "group_profiles": summary, "depth_contrasts": contrasts}


def panel_h(frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["model_norm"] = rows["model"].map(normalize_model)
    rows["model_norm"] = pd.Categorical(
        rows["model_norm"], ("four_bias", "depth_stratified"), ordered=True
    )
    rows = rows.sort_values("model_norm").reset_index(drop=True)
    rows = numeric(
        rows,
        ("delta_real", "delta_simulated", "fraction_recapitulated", "fraction_ci_low", "fraction_ci_high"),
    )
    if len(rows) != 2:
        raise ValueError("Panel h requires strict four-policy and depth gap rows")
    observed = float(rows["delta_real"].iloc[0])
    rows["delta_simulated_ci_low"] = rows["delta_real"] * rows["fraction_ci_low"]
    rows["delta_simulated_ci_high"] = rows["delta_real"] * rows["fraction_ci_high"]
    fig, ax = plt.subplots(figsize=(4.15, 2.15))
    ax.axvline(observed, color=INK, linewidth=1.05)
    ax.axvspan(observed - 0.0025, observed + 0.0025, color=PALE, zorder=0)
    y = (1, 0)
    labels = ("Four-policy walk", r"Depth-varying $P_{BS}$ walk")
    colors = (FOUR, DEPTH)
    markers = ("o", "s")
    for yi, (_, row), color, marker in zip(y, rows.iterrows(), colors, markers):
        value = float(row["delta_simulated"])
        low = float(row["delta_simulated_ci_low"])
        high = float(row["delta_simulated_ci_high"])
        ax.errorbar(
            value,
            yi,
            xerr=[[value - low], [high - value]],
            fmt=marker,
            color=color,
            markersize=4.8,
            capsize=2,
            elinewidth=1.1,
        )
        ax.text(
            value,
            yi + 0.19,
            (
                f"{100 * float(row['fraction_recapitulated']):.1f}% "
                f"[{100 * float(row['fraction_ci_low']):.1f}, "
                f"{100 * float(row['fraction_ci_high']):.1f}]"
            ),
            ha="center",
            va="bottom",
            fontsize=5.3,
            color=color,
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.48, 1.56)
    ax.set_xlim(0, max(0.32, float(rows["delta_simulated_ci_high"].max()) + 0.025))
    ax.set_xlabel("Return - Naive half-coverage efficiency gap (absolute)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Depth refinement follows the baseline model test", loc="left")
    return fig, {"four_vs_depth_gap": rows}


def _normalize_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["model_norm"] = rows["model"].map(normalize_model)
    rows["estimator_norm"] = np.where(
        rows["estimator_family"].astype(str).str.contains("event"),
        "event_pooled",
        "equal_junction",
    )
    rows["weighting_norm"] = np.where(
        rows["weighting"].astype(str).str.contains("mouse"), "mouse", "session"
    )
    return numeric(rows, ("fraction_recapitulated", "fraction_ci_low", "fraction_ci_high"))


def panel_j(
    robustness_frame: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    support_frame: pd.DataFrame,
    outcome_overlap_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    robustness = _normalize_robustness(robustness_frame)
    category_order = (
        ("equal_junction", "session"),
        ("event_pooled", "session"),
        ("equal_junction", "mouse"),
        ("event_pooled", "mouse"),
    )
    expected = {
        (model, estimator, weighting)
        for model in ("four_bias", "depth_stratified")
        for estimator, weighting in category_order
    }
    observed = set(
        zip(
            robustness["model_norm"],
            robustness["estimator_norm"],
            robustness["weighting_norm"],
        )
    )
    if observed != expected:
        raise ValueError(f"Robustness cells drifted: missing={sorted(expected - observed)}")
    exposure = exposure_frame.loc[exposure_frame["group"].isin(GROUP_ORDER)].copy()
    exposure = numeric(
        exposure,
        (
            "n_sessions",
            "median_explore_minutes_within_window",
            "median_valid_explore_decisions",
        ),
    )
    support = support_frame.loc[
        support_frame["estimator"].eq("equal_junction")
        & support_frame["group"].isin(GROUP_ORDER)
    ].copy()
    overlap = outcome_overlap_frame.loc[
        outcome_overlap_frame["group"].isin(GROUP_ORDER)
    ].copy()
    overlap = numeric(
        overlap,
        (
            "n_sessions",
            "n_sessions_already_at_32_unique",
            "fraction_sessions_already_at_32_unique",
        ),
    )
    audit = exposure.merge(
        support[
            [
                "group",
                "n_all_sessions",
                "n_four_policy_supported",
                "n_depth_supported",
                "n_depth_calibration_rows_with_target",
            ]
        ],
        on="group",
        how="inner",
    ).merge(
        overlap[
            [
                "group",
                "n_sessions_already_at_32_unique",
                "fraction_sessions_already_at_32_unique",
            ]
        ],
        on="group",
        how="inner",
    )
    audit["group_order"] = pd.Categorical(audit["group"], GROUP_ORDER, ordered=True)
    audit = audit.sort_values("group_order").reset_index(drop=True)
    if len(audit) != 3:
        raise ValueError("Panel j exposure/support audit requires three groups")

    fig = plt.figure(figsize=(5.8, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.04, 1.12), wspace=0.29)
    ax = fig.add_subplot(gs[0, 0])
    y_base = np.arange(4)[::-1]
    labels = (
        "Equal junction | session",
        "Event pooled | session",
        "Equal junction | mouse equal",
        "Event pooled | mouse equal",
    )
    ax.axvline(100, color=INK, linewidth=0.8)
    for model, color, marker, offset in (
        ("four_bias", FOUR, "o", 0.09),
        ("depth_stratified", DEPTH, "s", -0.09),
    ):
        values, lows, highs = [], [], []
        for estimator, weighting in category_order:
            row = robustness.loc[
                robustness["model_norm"].eq(model)
                & robustness["estimator_norm"].eq(estimator)
                & robustness["weighting_norm"].eq(weighting)
            ].iloc[0]
            values.append(100 * float(row["fraction_recapitulated"]))
            lows.append(100 * float(row["fraction_ci_low"]))
            highs.append(100 * float(row["fraction_ci_high"]))
        values_array = np.asarray(values)
        low_array = np.asarray(lows)
        high_array = np.asarray(highs)
        ax.errorbar(
            values_array,
            y_base + offset,
            xerr=np.vstack([values_array - low_array, high_array - values_array]),
            fmt=marker,
            color=color,
            markersize=3.7,
            capsize=1.6,
            elinewidth=0.85,
            label=("Four policy" if model == "four_bias" else r"Depth $P_{BS}$"),
        )
    ax.set_yticks(y_base, labels)
    ax.set_xlabel("Gap reconstructed (%)")
    ax.set_xlim(30, 135)
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.53, 1.14), ncol=2, handlelength=1.0)
    ax.set_title("Estimator and weighting robustness", loc="left")

    table_ax = fig.add_subplot(gs[0, 1])
    table_ax.set_xlim(0, 1)
    table_ax.set_ylim(-0.10, 1.05)
    table_ax.axis("off")
    table_ax.set_title("30-min exposure and analysis support", loc="left", pad=3)
    headers = ("Group", "Explore", "Decisions", "Four/Depth", "Target", ">=32")
    x_positions = (0.00, 0.29, 0.46, 0.63, 0.82, 0.95)
    for x, header in zip(x_positions, headers):
        table_ax.text(x, 0.91, header, ha="left", va="bottom", fontsize=4.8, color=MUTED)
    table_ax.plot((0, 1), (0.88, 0.88), color=LIGHT, linewidth=0.8)
    for index, row in audit.iterrows():
        y = 0.71 - 0.24 * index
        group = str(row["group"])
        total = int(row["n_all_sessions"])
        values = (
            GROUP_LABEL[group],
            f"{float(row['median_explore_minutes_within_window']):.1f} min",
            f"{float(row['median_valid_explore_decisions']):,.0f}",
            (
                f"{int(row['n_four_policy_supported'])}/"
                f"{int(row['n_depth_supported'])}"
            ),
            f"{int(row['n_depth_calibration_rows_with_target'])}/{total}",
            f"{int(row['n_sessions_already_at_32_unique'])}/{total}",
        )
        for x, value in zip(x_positions, values):
            table_ax.text(
                x,
                y,
                value,
                ha="left",
                va="center",
                fontsize=5.0,
                color=GROUP_COLOR[group] if x == 0 else INK,
                fontweight="bold" if x == 0 else "normal",
            )
        if index < 2:
            table_ax.plot((0, 1), (y - 0.12, y - 0.12), color=PALE, linewidth=0.7)
    table_ax.text(
        0,
        0.02,
        "Four/Depth = supported rows. Target = observed efficiency. >=32 = sessions already at 32 unique terminal endpoints.",
        ha="left",
        va="bottom",
        fontsize=4.4,
        color=MUTED,
        wrap=True,
    )
    return fig, {
        "robustness": robustness,
        "exposure_support_outcome_overlap": audit.drop(columns="group_order"),
    }


def write_panel_tables(
    table_root: Path,
    panel: str,
    tables: dict[str, pd.DataFrame],
) -> list[Path]:
    table_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for label, frame in tables.items():
        output = frame.copy()
        if "window_minutes" not in output.columns:
            output.insert(0, "window_minutes", WINDOW_MINUTES)
        else:
            output["window_minutes"] = WINDOW_MINUTES
        path = table_root / f"panel_{panel}_{label}.csv"
        output.to_csv(path, index=False)
        outputs.append(path)
    return outputs


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


def _format_policy_results(frame: pd.DataFrame) -> str:
    rows = frame.copy()
    rows["metric"] = pd.Categorical(rows["metric"], POLICY_ORDER, ordered=True)
    rows = rows.sort_values("metric")
    return "; ".join(
        (
            f"{POLICY_LABEL[str(row.metric)]} {float(row.beta_target_minus_baseline):+.3f} "
            f"[{float(row.ci_low):+.3f}, {float(row.ci_high):+.3f}], "
            f"{_q_text(float(row.q_bh4_within_window_contrast))}"
        )
        for row in rows.itertuples(index=False)
    )


def write_legend(output_root: Path, tables: dict[str, dict[str, pd.DataFrame]]) -> Path:
    b_stats = tables["b"]["contrasts"]
    c_stats = tables["c"]["contrasts"]
    g_stats = tables["g"]["depth_contrasts"]
    h_gap = tables["h"]["four_vs_depth_gap"]
    legend = f"""# Extended Data Fig. 3 | Experience-dependent route policies and biased-walk reconstruction

All panels use the first 30 cumulative minutes of TimeInMaze from session start. Non-Explore epochs consume the clock, whereas only Explore-mode junction decisions contribute policy estimates. The primary policy estimator gives each supported junction/side equal weight (the equal-junction estimator). Half-coverage efficiency is the session-level exploration-efficiency outcome used by the maintained analysis.

**a,** Source-native route-choice schematic defining the four local probabilities used by the fixed-policy walk. {POLICY_LABEL['Bf']} is entry into a child rather than reversal after parent arrival; {POLICY_LABEL['Ba']} is the orientation-defined alternating child conditional on entering a child; {POLICY_LABEL['Lf']} is leaving the arrival-child branch rather than reversing along it after child arrival; and {POLICY_LABEL['Lo']} is parent/stem rather than sibling conditional on leaving that branch.

**b,** Session-level strict common-support policy profiles for Naive (22 sessions, 22 mice) and Return (58 sessions, 18 mice). Crosshairs show equal-next-edge references. Filled symbols show session-weighted group means. Return-minus-Naive estimates (mouse-clustered CR1 95% confidence intervals; BH correction across four policies): {_format_policy_results(b_stats)}.

**c,** Natural-support Return (58 sessions, 18 mice) and Second-day (74 sessions, 24 mice) policy profiles. Second-day-minus-Return estimates use the same CR1/BH4 family: {_format_policy_results(c_stats)}. Visual resemblance is not an equality or equivalence result; the four inferential contrasts remain the authority.

**d,** Return-own simulated efficiency minus the same Return session after replacing one policy with the fixed strict-cohort Naive donor mean. Points are session-weighted means and intervals are synchronized union-of-mice cluster-bootstrap 95% intervals. Simulation Monte Carlo noise is not propagated.

**e,** Baseline four-policy reconstruction of the strict Return-minus-Naive efficiency gap: 53.3% [43.5%, 67.7%]. The vertical line is the observed absolute gap.

**f,** Observed versus four-policy simulated half-coverage efficiency for natural-support Naive, Return and Second-day calibration rows (23, 58 and 71 sessions). rho and mean bias are descriptive. The declared descriptive calibration gates, absolute mean bias <=0.05 and Spearman rho >=0.40, are model diagnostics rather than hypothesis tests.

**g,** Natural-support depth profiles of {POLICY_LABEL['Lo']} for Naive, Return and Second-day sessions passing all-three-depth support (22, 58 and 74 sessions). Lines show session means and bands show session interquartile ranges. Lower estimates are mouse-clustered CR1 contrasts with 95% confidence intervals and BH correction across the three depth strata separately for Return-minus-Naive and Second-day-minus-Return. The six plotted rows are available in Source Data ({len(g_stats)} rows).

**h,** Strict-cohort comparison of four-policy and depth-varying {POLICY_LABEL['Lo']} gap reconstruction: {100 * float(h_gap.loc[h_gap['model_norm'].astype(str).eq('four_bias'), 'fraction_recapitulated'].iloc[0]):.1f}% versus {100 * float(h_gap.loc[h_gap['model_norm'].astype(str).eq('depth_stratified'), 'fraction_recapitulated'].iloc[0]):.1f}%. Intervals use synchronized union-of-mice cluster bootstrap draws with fixed donor means; simulation Monte Carlo noise is not propagated.

**i,** Observed versus depth-model simulated efficiency for the same three natural-support calibration groups (23, 58 and 71 sessions). The declared descriptive gates pass for Naive and Return but not Second day; this limits the refined walk's diagnostic scope to the Naive-to-Return comparison.

**j,** Left, strict-cohort sensitivity to equal-junction versus event-pooled policy estimators and session-weighted versus mouse-equal aggregation. Right, natural-cohort median Explore exposure, valid decisions, policy support, calibration-target support, and the number of sessions already reaching 32 unique terminal endpoints during the predictor window. A common 30-minute clock does not imply identical Explore exposure, and the endpoint overlap makes this a contemporaneous reconstruction rather than prospective prediction.

Source Data are provided as one or more plot-ready CSV files for each panel. Session rows are the plotted unit; repeated sessions from the same mouse are handled by the stated clustered uncertainty procedures where inferential intervals are shown.

Reconstruction percentages are not variance explained, prospective prediction, causal mediation, biological sufficiency, or causal shares. Natural-support panels preserve all eligible Second-day evidence; strict-support panels preserve the common Naive/Return model-comparison cohort. These cohort roles are intentionally different and are labeled panel by panel.
"""
    path = output_root / "FIGURE_LEGEND.md"
    path.write_text(legend, encoding="utf-8")
    return path


def write_report(
    output_root: Path,
    input_paths: dict[str, Path],
    tables: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    second = tables["c"]["contrasts"]
    calibration_four = tables["f"]["calibration_summary"]
    calibration_depth = tables["i"]["calibration_summary"]
    overlap = tables["j"]["exposure_support_outcome_overlap"]
    inputs = "\n".join(
        f"- `{key}`: `{path}` (SHA-256 `{sha256(path)}`)"
        for key, path in sorted(input_paths.items())
    )
    panel_lines = "\n".join(
        f"- **{panel}.** {meta['claim']} Role: `{meta['role']}`; stem: `{meta['stem']}`."
        for panel, meta in PANEL_META.items()
    )
    four_gates = {
        GROUP_LABEL[str(row.group)]: bool(row.combined_gate_pass)
        for row in calibration_four.itertuples(index=False)
    }
    depth_gates = {
        GROUP_LABEL[str(row.group)]: bool(row.combined_gate_pass)
        for row in calibration_depth.itertuples(index=False)
    }
    overlap_text = ", ".join(
        (
            f"{GROUP_LABEL[str(row.group)]} "
            f"{int(row.n_sessions_already_at_32_unique)}/{int(row.n_all_sessions)}"
        )
        for row in overlap.itertuples(index=False)
    )
    report = f"""# ED3 full-story fixed-30-minute renderer report

## Figure contract

**Core conclusion.** Experience changes four local route-choice policies during the first 30 minutes in the maze, including further Second-day changes; a four-fixed-policy walk reconstructs part of the Naive-to-Return exploration-efficiency gap, while a depth-varying {POLICY_LABEL['Lo']} refinement reconstructs that gap more closely and passes the declared descriptive gates for Naive and Return sessions.

**Evidence logic.** The page first defines the four policies (a), shows Naive-to-Return and Return-to-Second-day observations (b-c), then evaluates the complete baseline four-policy simulator through substitutions, gap reconstruction, and calibration (d-f). Only after that baseline evidence does it introduce the depth refinement and test its reconstruction/calibration (g-i). Panel j exposes estimator, weighting, exposure, and support sensitivity.

**Archetype.** Schematic-led quantitative grid with the policy definition as the entry point and the model comparison as the central evidence chain.

## Estimands and fairness

- Clock: first 30 cumulative minutes of TimeInMaze from session start. All modes consume clock time; only Explore-mode junction decisions estimate policies.
- Primary policy estimator: equal weight to each eligible next edge within a junction/session before session-level aggregation. Event-pooled estimates appear only in panel j sensitivity.
- Strict Naive/Return cohort: 22 Naive sessions from 22 mice and 58 Return sessions from 18 mice; 29 mice in the union, 11 represented in both groups. It supports panels b, d, e, h and the robustness half of j.
- Natural cohort: 23 Naive, 58 Return and 74 Second-day policy sessions. Calibration has observed outcome targets for 23, 58 and 71 sessions, respectively. This supports panels c, f, g, i and the exposure/support half of j.
- The cohort change is explicit rather than hidden: strict common support preserves an internally matched Naive-to-Return model comparison; natural support retains the manuscript-bearing Second-day evidence.
- Policy scatter proximity is descriptive. In particular, panel c does not claim equality or equivalence. Second-day-minus-Return estimates are {_format_policy_results(second)}.
- The four-policy substitution estimand changes one fitted policy at a time while holding the other three at the Return session's values; it is not a causal decomposition and the components need not sum to the total gap.
- Gap intervals use synchronized mouse-cluster bootstrap draws. Donor means are fixed and Monte Carlo walk noise is not propagated.
- Calibration gates are descriptive diagnostics, not statistical tests and not a model-selection rule. Four-policy gates: `{four_gates}`. Depth-model gates: `{depth_gates}`.
- The depth refinement is supported for the Naive-to-Return comparison. Its Second-day calibration fails the declared combined gate, so it should not be presented as a complete Second-day generative account.
- Sessions already reaching 32 unique terminal endpoints during the 30-minute predictor window: {overlap_text}. The model is therefore a contemporaneous reconstruction diagnostic, not prospective prediction.

## Panel map

{panel_lines}

## Frozen input tables

{inputs}

## Export contract

Every panel is exported as editable-text SVG, PDF, 600-dpi PNG and 600-dpi LZW TIFF by default. Each panel has plot-ready CSV Source Data. The review composite is 180 x 225 mm, within the requested 230-mm height ceiling. `PANEL_MANIFEST.csv`, `INPUT_MANIFEST.csv`, `QA.md`, the JSON render manifest, and `artifact_sha256.csv` provide traceability.

No simulator was rerun by this renderer.
"""
    path = output_root / "REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


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


def write_qa(output_root: Path, qa: dict[str, Any]) -> Path:
    lines = [
        "# ED3 full-story render QA",
        "",
        f"- Overall: `{'PASS' if qa['pass'] else 'FAIL'}`",
        f"- Panels: {qa['panel_count']} ({qa['panel_artifact_count']} SVG/PDF/PNG/TIFF artifacts)",
        f"- Plot-ready Source Data files: {qa['source_table_count']}",
        "- Quantitative Source Data: all rows explicitly marked `window_minutes=30`",
        "- Active SVGs: editable text required; no deprecated decision-fraction labels",
        "- Semantic palette: exact Main Figure group colors, established policy colors, and distinct model-family colors required",
        f"- Raster export: {qa['expected_raster_dpi']} dpi; TIFF compression `tiff_lzw`",
        "- Review composite: 180 x 225 mm (within 230-mm ceiling)",
        "- Analysis execution: plot only; no simulation or model fit rerun",
    ]
    if qa["failures"]:
        lines.extend(["", "## Failures", "", *[f"- {item}" for item in qa["failures"]]])
    path = output_root / "QA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render(
    strict_root: Path = DEFAULT_STRICT_ROOT,
    natural_root: Path = DEFAULT_NATURAL_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dpi: int = 600,
) -> dict[str, Any]:
    configure_style()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table_root = output_root / "source_tables"
    frames, input_paths = load_sources(strict_root, natural_root)

    b_points, b_contrasts = prepare_two_axis_scatter(
        frames["strict_policy_rows"],
        frames["strict_policy_contrasts"],
        ("naive", "LTM"),
        "return_vs_naive",
    )
    c_points, c_contrasts = prepare_two_axis_scatter(
        frames["natural_policy_rows"],
        frames["natural_policy_contrasts"],
        ("LTM", "experienced"),
        "second_vs_return",
    )
    panel_builders = {
        "a": panel_a,
        "b": lambda: policy_scatter_panel(
            b_points,
            b_contrasts,
            ("naive", "LTM"),
            footer=(
                "Equal-next-edge references = dashed lines; strict common support; "
                "filled symbols = session-weighted means."
            ),
        ),
        "c": lambda: policy_scatter_panel(
            c_points,
            c_contrasts,
            ("LTM", "experienced"),
            footer=(
                "Equal-next-edge references = dashed lines; natural support; "
                "visual resemblance is not equality or equivalence."
            ),
        ),
        "d": lambda: panel_d(frames["strict_contributions"]),
        "e": lambda: panel_e(frames["strict_gap"]),
        "f": lambda: calibration_panel(
            frames["natural_simulation_rows"], frames["natural_calibration"], "four_bias"
        ),
        "g": lambda: panel_g(
            frames["natural_policy_rows"], frames["natural_depth_contrasts"]
        ),
        "h": lambda: panel_h(frames["strict_gap"]),
        "i": lambda: calibration_panel(
            frames["natural_simulation_rows"],
            frames["natural_calibration"],
            "depth_stratified",
        ),
        "j": lambda: panel_j(
            frames["strict_robustness"],
            frames["natural_exposure"],
            frames["natural_support"],
            frames["natural_outcome_overlap"],
        ),
    }
    panel_paths: dict[str, list[Path]] = {}
    panel_tables: dict[str, dict[str, pd.DataFrame]] = {}
    table_paths: list[Path] = []
    for panel in PANEL_META:
        fig, tables = panel_builders[panel]()
        paths = save_panel(fig, output_root / PANEL_META[panel]["stem"], dpi)
        panel_paths[panel] = paths
        panel_tables[panel] = tables
        table_paths.extend(write_panel_tables(table_root, panel, tables))

    panel_pngs = {
        panel: next(path for path in paths if path.suffix == ".png")
        for panel, paths in panel_paths.items()
    }
    composite_paths = make_contact_sheet(panel_pngs, output_root)

    input_manifest = pd.DataFrame(
        [
            {
                "source_key": key,
                "path": str(path),
                "sha256": sha256(path),
                "row_count_after_30min_filter": len(frames[key]),
                "window_minutes": 30,
            }
            for key, path in sorted(input_paths.items())
        ]
    )
    input_manifest_path = output_root / "INPUT_MANIFEST.csv"
    input_manifest.to_csv(input_manifest_path, index=False)

    panel_manifest_rows = []
    for panel, paths in panel_paths.items():
        for path in paths:
            panel_manifest_rows.append(
                {
                    "panel": panel,
                    "stem": PANEL_META[panel]["stem"],
                    "role": PANEL_META[panel]["role"],
                    "claim": PANEL_META[panel]["claim"],
                    "window_minutes": 30,
                    "decision_mode": "Explore only",
                    "format": path.suffix.lstrip("."),
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    panel_manifest_path = output_root / "PANEL_MANIFEST.csv"
    pd.DataFrame(panel_manifest_rows).to_csv(panel_manifest_path, index=False)

    legend_path = write_legend(output_root, panel_tables)
    report_path = write_report(output_root, input_paths, panel_tables)
    qa = validate_outputs(output_root, panel_paths, table_paths, composite_paths, dpi)
    qa_path = write_qa(output_root, qa)

    pre_manifest_files = sorted(
        [path for paths in panel_paths.values() for path in paths]
        + table_paths
        + composite_paths
        + [input_manifest_path, panel_manifest_path, legend_path, report_path, qa_path]
    )
    render_manifest = {
        "analysis": "ED3 full-story fixed-30-minute Explore-only plot-only render",
        "renderer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "window_minutes": 30,
        "clock": "cumulative TimeInMaze from session start; all modes consume clock time",
        "decision_filter": "Explore-mode junction decisions only",
        "no_simulation_rerun": True,
        "panel_stems": {panel: meta["stem"] for panel, meta in PANEL_META.items()},
        "panel_roles": {panel: meta["role"] for panel, meta in PANEL_META.items()},
        "displayed_groups": ["Naive", "Return", "Second day"],
        "displayed_second_day": True,
        "second_day_displayed": True,
        "second_day_panels": ["c", "f", "g", "i", "j"],
        "semantic_palette": {
            "behavior_groups": {
                "Naive": GROUP_COLOR["naive"],
                "Return": GROUP_COLOR["LTM"],
                "Second day": GROUP_COLOR["experienced"],
            },
            "turning_policies": dict(TURNING_POLICY_COLORS),
            "models": {
                "four_policy": FOUR,
                "depth_varying": DEPTH,
            },
            "diagnostics": {"pass_text": PASS, "fail_text": MUTED},
            "references": {
                "observed": INK,
                "random_choice": REFERENCE,
                "identity": LIGHT,
            },
            "neutrals": dict(FIGURE_NEUTRAL_COLORS),
            "schematic": dict(GRID_ROUTE_CHOICE_SCHEMATIC_COLORS),
        },
        "policy_definitions": {
            metric: {
                "label": POLICY_LABEL[metric],
                "definition": POLICY_DEFINITION[metric],
                "equal_next_edge_reference": RANDOM_REFERENCE[metric],
            }
            for metric in POLICY_ORDER
        },
        "cohorts": {
            "strict_policy": {
                "Naive": {"sessions": 22, "mice": 22},
                "Return": {"sessions": 58, "mice": 18},
            },
            "natural_policy": {
                "Naive": {"sessions": 23, "mice": 23},
                "Return": {"sessions": 58, "mice": 18},
                "Second day": {"sessions": 74, "mice": 24},
            },
            "natural_calibration": {
                "Naive": {"sessions": 23, "mice": 23},
                "Return": {"sessions": 58, "mice": 18},
                "Second day": {"sessions": 71, "mice": 24},
            },
        },
        "calibration_gates": {
            "absolute_mean_bias_le": 0.05,
            "spearman_rho_ge": 0.40,
            "role": "descriptive diagnostic; not a hypothesis test",
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path), "rows": len(frames[key])}
            for key, path in sorted(input_paths.items())
        },
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in pre_manifest_files
        ],
        "qa": qa,
    }
    render_manifest_path = output_root / "ed03_fixed_time_30min_full_story_render_manifest.json"
    render_manifest_path.write_text(
        json.dumps(json_ready(render_manifest), indent=2) + "\n", encoding="utf-8"
    )

    hash_targets = sorted(pre_manifest_files + [render_manifest_path])
    hash_path = output_root / "artifact_sha256.csv"
    pd.DataFrame(
        [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in hash_targets
        ]
    ).to_csv(hash_path, index=False)

    if not qa["pass"]:
        raise RuntimeError("ED3 full-story render QA failed: " + "; ".join(qa["failures"]))
    return {
        "output_root": output_root,
        "panel_manifest": panel_manifest_path,
        "input_manifest": input_manifest_path,
        "render_manifest": render_manifest_path,
        "hash_ledger": hash_path,
        "legend": legend_path,
        "report": report_path,
        "qa": qa_path,
        "composite": composite_paths[0],
        "panel_count": len(panel_paths),
        "artifact_count": len(hash_targets) + 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--natural-root", type=Path, default=DEFAULT_NATURAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render(args.strict_root, args.natural_root, args.output_root, args.dpi)
    print(json.dumps(json_ready(result), indent=2))


if __name__ == "__main__":
    main()
