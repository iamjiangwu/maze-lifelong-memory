"""Render the ED2 slide-3 round-5 layout-optimized review candidate.

The accepted round-3 record remains the scientific authority and the supplied
PowerPoint remains the inventory authority (exactly panels a-i).  This pass
changes only final physical panel sizes, compact margins, and panel d's sign
key.  Statistics and cohort counts remain in the caption notes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_caption_minimal_round4.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round5_layout_2026-08-08" / "panels"


PANEL_SIZE_IN = {
    "a": (148 / 96, 192 / 96),
    "b": (258 / 96, 192 / 96),
    "c": (258 / 96, 192 / 96),
    "d": (160 / 96, 200 / 96),
    "e": (168 / 96, 200 / 96),
    "f": (336 / 96, 200 / 96),
    "g": (176 / 96, 200 / 96),
    "h": (152 / 96, 200 / 96),
    "i": (336 / 96, 200 / 96),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_layout_optimized_round5",
    "b": "ed02b_naive_vs_return_policy_scatter_layout_optimized_round5",
    "c": "ed02c_return_vs_second_day_policy_scatter_layout_optimized_round5",
    "d": "ed02d_four_policy_contributions_layout_optimized_round5",
    "e": "ed02e_distance_from_return_layout_optimized_round5",
    "f": "ed02f_four_policy_calibration_layout_optimized_round5",
    "g": "ed02g_depth_pbs_profile_layout_optimized_round5",
    "h": "ed02h_four_vs_depth_reconstruction_layout_optimized_round5",
    "i": "ed02i_depth_calibration_layout_optimized_round5",
}


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round4_renderer", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def build_panel_d(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Plot the unchanged Return-own minus Naive-fixed contrast with a sign key."""
    rows = frame.copy()
    rows["parameter"] = pd.Categorical(
        rows["parameter"], renderer.POLICY_ORDER, ordered=True
    )
    rows = rows.sort_values("parameter").reset_index(drop=True)
    numeric = (
        "delta_efficiency_mean",
        "delta_efficiency_ci_low",
        "delta_efficiency_ci_high",
    )
    rows[list(numeric)] = rows[list(numeric)].apply(pd.to_numeric, errors="raise")
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

    # The saved contrast is unmodified Return minus the same Return simulation
    # with one policy replaced by the Naive donor mean.  The compact sign key
    # makes both the operation and direction explicit without printing values.
    zero_fraction = 5.0 / 15.5
    ax.text(
        0.14,
        0.985,
        "Naive-substituted\nhigher",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.6,
        color=renderer.MUTED,
        linespacing=0.92,
    )
    ax.text(
        0.68,
        0.985,
        "Unmodified Return\nhigher",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=4.6,
        color=renderer.MUTED,
        linespacing=0.92,
    )
    arrow_style = {
        "arrowstyle": "->",
        "color": renderer.MUTED,
        "linewidth": 0.65,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    ax.annotate(
        "",
        xy=(0.02, 0.895),
        xytext=(zero_fraction - 0.02, 0.895),
        xycoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=(0.98, 0.895),
        xytext=(zero_fraction + 0.02, 0.895),
        xycoords=ax.transAxes,
        arrowprops=arrow_style,
    )
    ax.set_yticks(y, [renderer.POLICY_LABEL[str(item)] for item in rows["parameter"]])
    ax.set_xlabel("Efficiency difference (pp)", labelpad=2.0)
    ax.set_xlim(-5.0, 10.5)
    ax.set_xticks((-5, 0, 5, 10))
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.20, top=0.96)
    return fig, {"policy_contributions": rows}


def build_panel_e(renderer: Any, rows: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Keep round-4's estimand while using the narrower final slot cleanly."""
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
        .assign(window_minutes=30, support="natural primary-policy support")
    )
    group_means = (
        means.reset_index()
        .merge(counts, on="group", how="left", validate="one_to_one")
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
    ax.set_ylabel(r"$|\Delta P|$ from Return", labelpad=1.2)
    ax.set_ylim(-0.006, 0.125)
    ax.set_yticks((0, 0.05, 0.10))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, 1.005),
        ncol=2,
        handletextpad=0.18,
        columnspacing=0.50,
        borderaxespad=0,
    )
    BASE.clean_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.16, top=0.84)
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
        ax.set_title(renderer.GROUP_LABEL[group], color=color, pad=1.2)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("N")
        ax.set_xticks((0, 0.4, 0.8))
        ax.set_yticks((0, 0.4, 0.8))
        BASE.clean_axes(ax)
    fig.supxlabel("Observed efficiency", y=0.025)
    fig.supylabel("Simulated efficiency", x=0.014)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.20, top=0.84, wspace=0.17)
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
            label="Day 2" if group == "experienced" else renderer.GROUP_LABEL[group],
        )
    ax.set_ylabel(r"$P_{BS}$", labelpad=1.0)
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x, [])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.24),
        ncol=3,
        handlelength=0.75,
        handletextpad=0.15,
        columnspacing=0.55,
    )
    ax.spines["bottom"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)

    specs = (
        ("return_vs_naive", "Return - Naive", renderer.GROUP_COLOR["LTM"], "o", 0.06),
        (
            "second_vs_return",
            "Day 2 - Return",
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
    ax_delta.set_ylabel(r"$\Delta P_{BS}$", labelpad=1.0)
    ax_delta.set_ylim(-0.025, 0.34)
    ax_delta.set_xticks(x, ("Shallow\n1-2", "Middle\n3-4", "Deep\n5"))
    ax_delta.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.25),
        ncol=2,
        handlelength=0.75,
        handletextpad=0.15,
        columnspacing=0.55,
    )
    BASE.clean_axes(ax_delta)
    if "window_minutes" not in summary:
        summary = summary.copy()
        summary.insert(0, "window_minutes", 30)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.14, top=0.87, hspace=0.47)
    return fig, {
        "supported_session_rows": rows,
        "group_profiles": summary,
        "depth_contrasts": contrasts,
    }


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    BASE.PANEL_SIZE_IN = PANEL_SIZE_IN
    BASE.PANEL_STEM = PANEL_STEM
    BASE.build_panel_d = build_panel_d
    BASE.build_panel_e = build_panel_e
    BASE.build_calibration = build_calibration
    BASE.build_panel_g = build_panel_g
    result = BASE.render(record_root, output_root, dpi)

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8").replace(
        "# Extended Data Fig. 2 round-4 caption notes",
        "# Extended Data Fig. 2 round-5 caption notes",
        1,
    )
    caption_path.write_text(caption, encoding="utf-8")

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 5
    manifest["presentation_changes"] = [
        "larger schematic and calibration panels",
        "narrower panels e and g",
        "equal third-row heights",
        "panel d sign defined as unmodified Return minus Naive-substituted Return",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
