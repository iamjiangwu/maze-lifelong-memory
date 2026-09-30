"""Render a caption-minimal ED2 slide-3 review candidate from current data.

The supplied PowerPoint remains the panel-inventory authority: exactly a-i are
retained in the original three-row order.  The accepted round-3 record is the
scientific input authority.  This review renderer changes presentation only,
except panel e, which is replaced by a direct current-data comparison of each
policy's absolute distance from the Return mean.

Statistics, sample sizes, test names, correction families, calibration gates,
and numerical point annotations are exported to caption notes rather than
drawn inside the panels.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp" / "ed02_round4_polish_2026-08-08" / "panels"
)

PANEL_ORDER = tuple("abcdefghi")
FORMATS = ("svg", "pdf", "png", "tiff")

# Exact 96-px/in destination frames from the round-4 slide plan.  Rendering at
# these physical sizes avoids PowerPoint-side distortion and keeps typography
# at the intended printed size.
PANEL_SIZE_IN = {
    "a": (112 / 96, 172 / 96),
    "b": (276 / 96, 172 / 96),
    "c": (276 / 96, 172 / 96),
    "d": (160 / 96, 184 / 96),
    "e": (200 / 96, 184 / 96),
    "f": (304 / 96, 164 / 96),
    "g": (208 / 96, 216 / 96),
    "h": (152 / 96, 164 / 96),
    "i": (304 / 96, 164 / 96),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_caption_minimal_round4",
    "b": "ed02b_naive_vs_return_policy_scatter_caption_minimal_round4",
    "c": "ed02c_return_vs_second_day_policy_scatter_caption_minimal_round4",
    "d": "ed02d_four_policy_contributions_caption_minimal_round4",
    "e": "ed02e_distance_from_return_caption_minimal_round4",
    "f": "ed02f_four_policy_calibration_caption_minimal_round4",
    "g": "ed02g_depth_pbs_profile_caption_minimal_round4",
    "h": "ed02h_four_vs_depth_reconstruction_caption_minimal_round4",
    "i": "ed02i_depth_calibration_caption_minimal_round4",
}

PANEL_ROLE = {
    "a": "four branch-policy definitions",
    "b": "Naive versus Return policy probabilities",
    "c": "Return versus Second-day policy probabilities",
    "d": "one-policy-at-a-time contribution test",
    "e": "policy-wise absolute distance from Return for Naive and Second day",
    "f": "four-policy walk calibration",
    "g": "depth-dependent P_BS profile and contrasts",
    "h": "four-policy versus depth-varying reconstruction",
    "i": "depth-varying walk calibration",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_renderer(record_root: Path) -> Any:
    path = record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ed02_round3_frozen_renderer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen renderer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_style(renderer: Any) -> None:
    renderer.configure_style()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    text = deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_TEXT)
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
    fig, _, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={"panel_size": PANEL_SIZE_IN["a"], "tight_layout": False},
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 0.9,
            "arrow_linewidth": 1.25,
            "previous_arrow_linewidth": 1.25,
            "arrow_mutation_scale": 7.5,
            "title_fontsize": 6.0,
            "title_fontweight": "normal",
            "or_fontsize": 5.6,
            "annotation_fontsize": 5.4,
        },
        text=text,
        save=False,
    )
    fig.set_size_inches(*PANEL_SIZE_IN["a"], forward=True)
    rows = pd.DataFrame(
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
    return fig, {"schematic_rows": rows}


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    pairs = (("Bf", "Lf"), ("Ba", "Lo"))
    fig, axes = plt.subplots(1, 2, figsize=PANEL_SIZE_IN[panel])
    for ax, (x_metric, y_metric) in zip(axes, pairs):
        ax.axvline(
            renderer.RANDOM_REFERENCE[x_metric],
            color=renderer.REFERENCE,
            linestyle="--",
            linewidth=0.75,
        )
        ax.axhline(
            renderer.RANDOM_REFERENCE[y_metric],
            color=renderer.REFERENCE,
            linestyle="--",
            linewidth=0.75,
        )
        for group in groups:
            sub = points.loc[points["group"].eq(group)]
            color = renderer.GROUP_COLOR[group]
            ax.scatter(
                sub[x_metric],
                sub[y_metric],
                s=11,
                marker=renderer.GROUP_MARKER[group],
                facecolors=renderer.BACKGROUND,
                edgecolors=color,
                linewidths=0.55,
                alpha=0.64,
                label=renderer.GROUP_LABEL[group],
            )
            ax.scatter(
                [sub[x_metric].mean()],
                [sub[y_metric].mean()],
                s=28,
                marker=renderer.GROUP_MARKER[group],
                facecolors=color,
                edgecolors=renderer.BACKGROUND,
                linewidths=0.7,
                zorder=4,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks((0, 0.5, 1.0))
        ax.set_yticks((0, 0.5, 1.0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(renderer.POLICY_LABEL[x_metric], labelpad=1.0)
        ax.set_ylabel(renderer.POLICY_LABEL[y_metric], labelpad=1.0)
        clean_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.85,
    )
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.17, top=0.80, wspace=0.34)
    return fig, {"session_points": points, "contrasts": contrasts}


def build_panel_d(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["parameter"] = pd.Categorical(
        rows["parameter"], renderer.POLICY_ORDER, ordered=True
    )
    rows = rows.sort_values("parameter").reset_index(drop=True)
    for column in ("delta_efficiency_mean", "delta_efficiency_ci_low", "delta_efficiency_ci_high"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    y = np.arange(len(rows))[::-1]
    mean = 100 * rows["delta_efficiency_mean"].to_numpy(dtype=float)
    low = 100 * rows["delta_efficiency_ci_low"].to_numpy(dtype=float)
    high = 100 * rows["delta_efficiency_ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["d"])
    ax.axvline(0, color=renderer.INK, linewidth=0.7)
    for yi, value, lo, hi, parameter in zip(y, mean, low, high, rows["parameter"]):
        key = str(parameter)
        color = renderer.TURNING_POLICY_COLORS[key]
        ax.hlines(yi, 0, value, color=color, linewidth=1.15)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt=renderer.POLICY_MARKER[key],
            color=color,
            markersize=4.0,
            capsize=1.8,
            elinewidth=0.9,
        )
    ax.set_yticks(y, [renderer.POLICY_LABEL[str(item)] for item in rows["parameter"]])
    ax.set_xlabel("Efficiency change (pp)", labelpad=2.0)
    ax.set_xlim(min(-4.8, low.min() - 0.8), max(10.5, high.max() + 0.8))
    ax.set_xticks((-4, 0, 5, 10))
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.21, top=0.96)
    return fig, {"policy_contributions": rows}


def build_panel_e(renderer: Any, rows: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    metrics = list(renderer.POLICY_ORDER)
    natural = rows.copy()
    natural[metrics] = natural[metrics].apply(pd.to_numeric, errors="raise")
    means = natural.groupby("group")[metrics].mean()
    table = pd.DataFrame(
        {
            "window_minutes": 30,
            "metric": metrics,
            "policy_label": [renderer.POLICY_LABEL[key] for key in metrics],
            "return_mean": means.loc["LTM", metrics].to_numpy(dtype=float),
            "naive_mean": means.loc["naive", metrics].to_numpy(dtype=float),
            "second_day_mean": means.loc["experienced", metrics].to_numpy(dtype=float),
        }
    )
    table["abs_naive_minus_return"] = abs(table["naive_mean"] - table["return_mean"])
    table["abs_second_day_minus_return"] = abs(
        table["second_day_mean"] - table["return_mean"]
    )
    table["support"] = "natural primary-policy support"
    table["group_mean_weighting"] = "session weighted"
    counts = (
        natural.groupby("group", as_index=False)
        .agg(n_sessions=("group", "size"), n_mice=("animal_id", "nunique"))
    )
    group_means = (
        means.rename_axis("group")
        .reset_index()
        .merge(counts, on="group", validate="one_to_one")
        .assign(window_minutes=30, support="natural primary-policy support")
    )

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["e"])
    x = np.arange(len(metrics), dtype=float)
    naive = table["abs_naive_minus_return"].to_numpy(dtype=float)
    second = table["abs_second_day_minus_return"].to_numpy(dtype=float)
    for xi, first, last in zip(x, naive, second):
        ax.plot(
            (xi - 0.10, xi + 0.10),
            (first, last),
            color=renderer.LIGHT,
            linewidth=0.9,
            zorder=1,
        )
    ax.scatter(
        x - 0.10,
        naive,
        s=19,
        color=renderer.GROUP_COLOR["naive"],
        edgecolor="white",
        linewidth=0.45,
        label="Naive",
        zorder=2,
    )
    ax.scatter(
        x + 0.10,
        second,
        s=19,
        color=renderer.GROUP_COLOR["experienced"],
        edgecolor="white",
        linewidth=0.45,
        label="Second day",
        zorder=2,
    )
    ax.set_xticks(x, table["policy_label"])
    ax.set_ylabel("Absolute difference\nfrom Return", labelpad=1.5)
    ax.set_ylim(-0.006, 0.125)
    ax.set_yticks((0, 0.05, 0.10))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.53, 1.02),
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
    )
    clean_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.18, top=0.82)
    return fig, {"distance_from_return": table, "group_means": group_means}


def build_calibration(
    renderer: Any,
    session_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    model: str,
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    points, summary, simulated_column = renderer.prepare_calibration(
        session_frame, summary_frame, model
    )
    maximum = float(
        np.nanmax(points[["real_half_cover_eff", simulated_column]].to_numpy())
    )
    limit = max(0.62, np.ceil((maximum + 0.03) / 0.1) * 0.1)
    fig, axes = plt.subplots(1, 3, figsize=PANEL_SIZE_IN[panel], sharex=True, sharey=True)
    for ax, group in zip(axes, renderer.GROUP_ORDER):
        sub = points.loc[points["group"].eq(group)]
        color = renderer.GROUP_COLOR[group]
        ax.plot((0, limit), (0, limit), color=renderer.LIGHT, linestyle="--", linewidth=0.75)
        ax.scatter(
            sub["real_half_cover_eff"],
            sub[simulated_column],
            s=11,
            facecolors=renderer.BACKGROUND,
            edgecolors=color,
            linewidths=0.55,
            alpha=0.78,
        )
        ax.set_title(renderer.GROUP_LABEL[group], color=color, pad=2.0)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks((0, 0.4, 0.8))
        ax.set_yticks((0, 0.4, 0.8))
        clean_axes(ax)
    fig.supxlabel("Observed efficiency", y=0.025)
    fig.supylabel("Simulated efficiency", x=0.018)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.22, top=0.82, wspace=0.23)
    return fig, {"session_points": points, "calibration_summary": summary}


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows, summary, contrasts = renderer.prepare_depth_profile(policy_rows, contrast_frame)
    fig, (ax, ax_delta) = plt.subplots(
        2,
        1,
        figsize=PANEL_SIZE_IN["g"],
        sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0)},
    )
    x = np.arange(3, dtype=float)
    for group in renderer.GROUP_ORDER:
        sub = summary.loc[summary["group"].eq(group)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], renderer.DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        means = sub["mean"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        color = renderer.GROUP_COLOR[group]
        ax.fill_between(x, q25, q75, color=color, alpha=0.09, linewidth=0)
        ax.plot(
            x,
            means,
            marker=renderer.GROUP_MARKER[group],
            color=color,
            markersize=3.6,
            linewidth=1.0,
            label=renderer.GROUP_LABEL[group],
        )
    ax.set_ylabel(r"$P_{BS}$", labelpad=1.2)
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x, [])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.25),
        ncol=3,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    ax.spines["bottom"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)

    specs = (
        ("return_vs_naive", "Return - Naive", renderer.GROUP_COLOR["LTM"], "o", 0.06),
        (
            "second_vs_return",
            "Second day - Return",
            renderer.GROUP_COLOR["experienced"],
            "s",
            -0.06,
        ),
    )
    ax_delta.axhline(0, color=renderer.LIGHT, linewidth=0.75)
    for contrast_id, label, color, marker, offset in specs:
        sub = contrasts.loc[contrasts["contrast_id"].eq(contrast_id)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], renderer.DEPTH_ORDER, ordered=True)
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
    ax_delta.set_ylabel(r"$\Delta P_{BS}$", labelpad=1.2)
    ax_delta.set_ylim(-0.025, 0.34)
    ax_delta.set_xticks(x, ("Shallow\n1-2", "Middle\n3-4", "Deep\n5"))
    ax_delta.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.26),
        ncol=2,
        handlelength=0.9,
        handletextpad=0.25,
        columnspacing=0.55,
    )
    clean_axes(ax_delta)
    if "window_minutes" not in summary:
        summary = summary.copy()
        summary.insert(0, "window_minutes", 30)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.14, top=0.87, hspace=0.47)
    return fig, {
        "supported_session_rows": rows,
        "group_profiles": summary,
        "depth_contrasts": contrasts,
    }


def build_panel_h(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["model_norm"] = rows["model"].map(renderer.normalize_model)
    rows["model_norm"] = pd.Categorical(
        rows["model_norm"], ("four_bias", "depth_stratified"), ordered=True
    )
    rows = rows.sort_values("model_norm").reset_index(drop=True)
    for column in (
        "delta_real",
        "delta_simulated",
        "fraction_recapitulated",
        "fraction_ci_low",
        "fraction_ci_high",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    observed = float(rows["delta_real"].iloc[0])
    rows["delta_simulated_ci_low"] = rows["delta_real"] * rows["fraction_ci_low"]
    rows["delta_simulated_ci_high"] = rows["delta_real"] * rows["fraction_ci_high"]

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["h"])
    ax.axvline(observed, color=renderer.INK, linewidth=1.0)
    ax.text(
        observed,
        1.38,
        "Observed",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=renderer.INK,
    )
    labels = ("Four policy", r"Depth $P_{BS}$")
    colors = (renderer.FOUR, renderer.DEPTH)
    markers = ("o", "s")
    for yi, (_, row), color, marker in zip((1, 0), rows.iterrows(), colors, markers):
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
            capsize=2.0,
            elinewidth=1.1,
        )
    ax.set_yticks((1, 0), labels)
    ax.set_ylim(-0.48, 1.58)
    ax.set_xlim(0, max(0.34, float(rows["delta_simulated_ci_high"].max()) + 0.04))
    ax.set_xlabel("Return - Naive\nefficiency gap", labelpad=2.0)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.24, top=0.92)
    return fig, {"four_vs_depth_gap": rows}


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": None, "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def format_q(value: float) -> str:
    return "<0.001" if value < 0.001 else f"={value:.3f}"


def write_caption_notes(
    path: Path,
    renderer: Any,
    tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    b_points = tables["b"]["session_points"]
    b_stats = tables["b"]["contrasts"]
    c_points = tables["c"]["session_points"]
    c_stats = tables["c"]["contrasts"]
    e_rows = tables["e"]["distance_from_return"]
    e_group_means = tables["e"]["group_means"]
    f_summary = tables["f"]["calibration_summary"]
    g_rows = tables["g"]["supported_session_rows"]
    g_stats = tables["g"]["depth_contrasts"]
    h_rows = tables["h"]["four_vs_depth_gap"]
    i_summary = tables["i"]["calibration_summary"]

    def cohort(points: pd.DataFrame, groups: tuple[str, ...]) -> str:
        return "; ".join(
            f"{renderer.GROUP_LABEL[group]}: {int(points['group'].eq(group).sum())} sessions, "
            f"{int(points.loc[points['group'].eq(group), 'animal_id'].astype(str).nunique())} mice"
            for group in groups
        )

    def summarized_cohort(frame: pd.DataFrame, groups: tuple[str, ...]) -> str:
        return "; ".join(
            f"{renderer.GROUP_LABEL[group]}: "
            f"{int(frame.loc[frame['group'].eq(group), 'n_sessions'].iloc[0])} sessions, "
            f"{int(frame.loc[frame['group'].eq(group), 'n_mice'].iloc[0])} mice"
            for group in groups
        )

    def policy_stats(frame: pd.DataFrame) -> str:
        rows = []
        for metric in renderer.POLICY_ORDER:
            row = frame.loc[frame["metric"].astype(str).eq(metric)].iloc[0]
            rows.append(
                f"{renderer.POLICY_LABEL[metric]} {float(row['beta_target_minus_baseline']):+.3f}, "
                f"BH4 q{format_q(float(row['q_bh4_within_window_contrast']))}"
            )
        return "; ".join(rows)

    e_distances = "; ".join(
        f"{row.policy_label}: {row.abs_naive_minus_return:.5f} versus "
        f"{row.abs_second_day_minus_return:.5f}"
        for row in e_rows.itertuples(index=False)
    )
    e_centroid_naive = float(
        np.linalg.norm(e_rows["return_mean"].to_numpy() - e_rows["naive_mean"].to_numpy())
    )
    e_centroid_second = float(
        np.linalg.norm(
            e_rows["return_mean"].to_numpy() - e_rows["second_day_mean"].to_numpy()
        )
    )

    def calibration_text(summary: pd.DataFrame) -> str:
        parts = []
        for group in renderer.GROUP_ORDER:
            row = summary.loc[summary["group"].eq(group)].iloc[0]
            gate = bool(renderer.as_bool(pd.Series([row["combined_gate_pass"]])).iloc[0])
            parts.append(
                f"{renderer.GROUP_LABEL[group]}: rho={float(row['spearman_rho']):.2f}, "
                f"bias={float(row['mean_error']):+.3f}, gate={'pass' if gate else 'fail'}"
            )
        return "; ".join(parts)

    depth_stats = []
    for contrast_id, label in (
        ("return_vs_naive", "Return - Naive"),
        ("second_vs_return", "Second day - Return"),
    ):
        sub = g_stats.loc[g_stats["contrast_id"].eq(contrast_id)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], renderer.DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        values = ", ".join(
            f"{renderer.DEPTH_LABEL[str(row.metric)].replace(chr(10), ' ')}: "
            f"{float(row.beta_target_minus_baseline):+.3f} "
            f"[{float(row.ci_low):+.3f}, {float(row.ci_high):+.3f}], "
            f"BH3 q{format_q(float(row.q_bh3_within_window_contrast))}"
            for row in sub.itertuples(index=False)
        )
        depth_stats.append(f"{label}: {values}")

    gap_values = []
    for row in h_rows.itertuples(index=False):
        model = "four-policy" if row.model_norm == "four_bias" else "depth-varying P_BS"
        gap_values.append(
            f"{model}: {100 * float(row.fraction_recapitulated):.1f}% "
            f"[{100 * float(row.fraction_ci_low):.1f}, {100 * float(row.fraction_ci_high):.1f}]"
        )

    d_rows = tables["d"]["policy_contributions"]
    d_values = "; ".join(
        f"{renderer.POLICY_LABEL[str(row.parameter)]} "
        f"{100 * float(row.delta_efficiency_mean):+.1f} pp "
        f"[{100 * float(row.delta_efficiency_ci_low):+.1f}, "
        f"{100 * float(row.delta_efficiency_ci_high):+.1f}]"
        for row in d_rows.itertuples(index=False)
    )

    text = f"""# Extended Data Fig. 2 round-4 caption notes

**a,** Four branch-policy definitions at non-root binary-tree junctions.

**b,** Naive and Return session-level policy probabilities under strict common support. {cohort(b_points, ('naive', 'LTM'))}. Mouse-cluster CR1 contrasts with BH correction across four policies: {policy_stats(b_stats)}. Filled symbols denote session-weighted group means; dashed lines denote equal-next-edge references.

**c,** Return and Second-day session-level policy probabilities under natural primary-policy support. {cohort(c_points, ('LTM', 'experienced'))}. Mouse-cluster CR1 contrasts with BH correction across four policies: {policy_stats(c_stats)}. Visual proximity is descriptive and is not an equivalence test.

**d,** Change in simulated Return efficiency after fixing one policy at a time to the Naive donor mean ({cohort(b_points, ('naive', 'LTM'))}). Estimates and synchronized union-of-mice cluster-bootstrap 95% confidence intervals: {d_values}. Donor means were fixed and Monte Carlo simulation noise was not propagated.

**e,** Absolute policy-wise difference from the Return session-weighted group mean under natural primary-policy support ({summarized_cohort(e_group_means, tuple(renderer.GROUP_ORDER))}). Naive versus Second-day distances, respectively: {e_distances}. The four-dimensional Euclidean centroid distance was {e_centroid_naive:.5f} from Return to Naive and {e_centroid_second:.5f} from Return to Second day. This panel is descriptive; it does not establish equivalence.

**f,** Observed versus four-policy-model half-coverage efficiency. Naive: 23 sessions, 23 mice; Return: 58 sessions, 18 mice; Second day: 71 sessions, 24 mice. Descriptive calibration gates were absolute mean bias no greater than 0.05 and Spearman rho at least 0.40: {calibration_text(f_summary)}.

**g,** Depth-dependent P_BS profiles (session means with interquartile bands) and mouse-cluster CR1 contrasts with 95% confidence intervals ({cohort(g_rows, tuple(renderer.GROUP_ORDER))}); root level 0 was excluded and BH correction was applied within each three-depth contrast family. {'; '.join(depth_stats)}.

**h,** Reconstruction of the observed Return-minus-Naive half-coverage efficiency gap ({cohort(b_points, ('naive', 'LTM'))}). Fractions reconstructed with synchronized union-of-mice cluster-bootstrap 95% confidence intervals: {'; '.join(gap_values)}. Donor means were fixed and Monte Carlo simulation noise was not propagated. Reconstruction is contemporaneous and is neither prospective prediction nor variance explained.

**i,** Observed versus depth-varying-P_BS-model half-coverage efficiency, with the same cohorts and descriptive calibration gates as f: {calibration_text(i_summary)}.

The behavioral clock is the first 30 cumulative minutes of all-mode TimeInMaze from session start; all modes advance the clock, whereas policy events are Explore-mode junction decisions only. This is a behavior-only analysis; neural manual masks do not apply. All exact values and statistical units are retained in the plot-ready tables and this caption note rather than printed inside the artwork.
"""
    path.write_text(text, encoding="utf-8")


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    record_root = record_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    renderer = load_frozen_renderer(record_root)
    configure_style(renderer)
    frames, input_paths = renderer.load_sources(
        record_root / "analysis_inputs" / "strict",
        record_root / "analysis_inputs" / "natural",
    )

    b_points, b_contrasts = renderer.prepare_two_axis_scatter(
        frames["strict_policy_rows"],
        frames["strict_policy_contrasts"],
        ("naive", "LTM"),
        "return_vs_naive",
    )
    c_points, c_contrasts = renderer.prepare_two_axis_scatter(
        frames["natural_policy_rows"],
        frames["natural_policy_contrasts"],
        ("LTM", "experienced"),
        "second_vs_return",
    )

    builders: dict[str, Callable[[], tuple[plt.Figure, dict[str, pd.DataFrame]]]] = {
        "a": lambda: build_panel_a(renderer),
        "b": lambda: build_policy_scatter(renderer, b_points, b_contrasts, ("naive", "LTM"), "b"),
        "c": lambda: build_policy_scatter(renderer, c_points, c_contrasts, ("LTM", "experienced"), "c"),
        "d": lambda: build_panel_d(renderer, frames["strict_contributions"]),
        "e": lambda: build_panel_e(renderer, frames["natural_policy_rows"]),
        "f": lambda: build_calibration(
            renderer,
            frames["natural_simulation_rows"],
            frames["natural_calibration"],
            "four_bias",
            "f",
        ),
        "g": lambda: build_panel_g(
            renderer, frames["natural_policy_rows"], frames["natural_depth_contrasts"]
        ),
        "h": lambda: build_panel_h(renderer, frames["strict_gap"]),
        "i": lambda: build_calibration(
            renderer,
            frames["natural_simulation_rows"],
            frames["natural_calibration"],
            "depth_stratified",
            "i",
        ),
    }

    panel_dir = output_root / "panels"
    table_dir = output_root / "plot_ready"
    panel_manifest: list[dict[str, Any]] = []
    table_manifest: list[dict[str, Any]] = []
    panel_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for panel in PANEL_ORDER:
        fig, tables = builders[panel]()
        panel_tables[panel] = tables
        paths = save_figure(fig, panel_dir / PANEL_STEM[panel], dpi)
        for path in paths:
            panel_manifest.append(
                {
                    "panel": panel,
                    "role": PANEL_ROLE[panel],
                    "format": path.suffix.lstrip("."),
                    "path": str(path.relative_to(output_root)).replace("\\", "/"),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "width_in": PANEL_SIZE_IN[panel][0],
                    "height_in": PANEL_SIZE_IN[panel][1],
                }
            )
        for name, frame in tables.items():
            table_path = write_table(frame, table_dir / f"panel_{panel}_{name}.csv")
            table_manifest.append(
                {
                    "panel": panel,
                    "table": name,
                    "path": str(table_path.relative_to(output_root)).replace("\\", "/"),
                    "rows": len(frame),
                    "sha256": sha256(table_path),
                    "bytes": table_path.stat().st_size,
                }
            )

    for name, rows in (("PANEL_MANIFEST.csv", panel_manifest), ("TABLE_MANIFEST.csv", table_manifest)):
        with (output_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_caption_notes(output_root / "CAPTION_NOTES.md", renderer, panel_tables)

    source_manifest = {
        "status": "review_candidate_not_current_version_record",
        "panel_inventory_authority": "supplied PowerPoint slide 3; exactly nine roles a-i",
        "source_record": str(record_root.relative_to(REPO_ROOT)).replace("\\", "/"),
        "frozen_renderer": {
            "path": str(
                (record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py").relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "sha256": sha256(
                record_root / "source_code" / "render_ed03_fixed_time_30min_full_story.py"
            ),
        },
        "inputs": [
            {
                "key": key,
                "path": str(path.relative_to(record_root)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for key, path in sorted(input_paths.items())
        ],
        "window_minutes": 30,
        "clock": "first 30 cumulative TimeInMaze minutes; all modes advance the clock",
        "policy_events": "Explore-mode junction decisions only",
        "manual_mask_applicability": "not applicable; behavior-only analysis",
        "visible_text_policy": (
            "axes, units, group/policy identities, uncertainty, and essential references only; "
            "statistics, counts, gates, and verbose titles moved to CAPTION_NOTES.md"
        ),
        "panel_e_estimand": (
            "absolute difference between each Naive or Second-day session-weighted group mean "
            "and the Return session-weighted group mean, separately for each of four policies"
        ),
    }
    (output_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "status": "pass",
        "candidate_status": "review_only_not_promoted",
        "panel_count": len(PANEL_ORDER),
        "panel_order": list(PANEL_ORDER),
        "panel_artifacts": len(panel_manifest),
        "plot_ready_tables": len(table_manifest),
        "dpi": dpi,
        "output_root": str(output_root),
    }
    (output_root / "RENDER_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(render(args.record_root, args.output_root, args.dpi), indent=2))


if __name__ == "__main__":
    main()
