"""Render the ED2 slide-3 round-7 typography and label-correction review.

Round 7 preserves the current nine-slot a-i review inventory and all frozen
inputs.  Relative to the accepted round-3 record, panel e retains the deliberate
round-4 redesign from signed shifts-from-Naive to absolute policy-wise distance
from Return; round 7 additionally displays panel d as the absolute value of the
same frozen signed policy-contribution contrast.  Both derived displays keep
their source values and definitions auditable in plot-ready tables and caption
notes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# The frozen record is imported as executable source authority.  Never let the
# import machinery write __pycache__ into that immutable directory.
sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND6_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_visible_alignment_round6.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round7_typography_2026-08-08" / "panels"

BASE_FONT_SIZE = 6.5
AXIS_LABEL_SIZE = 6.5
TICK_LABEL_SIZE = 5.8
TITLE_SIZE = 6.5
LEGEND_SIZE = 5.4
ANNOTATION_SIZE = 5.4
MODEL_TITLE_SIZE = 6.7
PANEL_D_NEUTRAL = "#68737D"

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_typography_round7",
    "b": "ed02b_naive_vs_return_policy_scatter_typography_round7",
    "c": "ed02c_return_vs_second_day_policy_scatter_typography_round7",
    "d": "ed02d_absolute_policy_contributions_typography_round7",
    "e": "ed02e_distance_from_return_typography_round7",
    "f": "ed02f_four_policy_calibration_typography_round7",
    "g": "ed02g_depth_pbs_profile_typography_round7",
    "h": "ed02h_four_vs_depth_reconstruction_typography_round7",
    "i": "ed02i_depth_calibration_typography_round7",
}


def load_round6() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round6_renderer", ROUND6_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND6_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R6 = load_round6()
ROUND6_CONFIGURE_STYLE = R6.R5.BASE.configure_style
ROUND6_BUILD_PANEL_A = R6.build_panel_a
ROUND6_BUILD_POLICY_SCATTER = R6.build_policy_scatter
ROUND6_BUILD_PANEL_E = R6.build_panel_e
ROUND6_BUILD_PANEL_G = R6.build_panel_g
ROUND4_BUILD_PANEL_H = R6.ROUND4_BUILD_PANEL_H


def configure_style(renderer: Any) -> None:
    """Use a legible on-slide contract while retaining Main Figure hierarchy."""

    ROUND6_CONFIGURE_STYLE(renderer)
    renderer.GROUP_LABEL["naive"] = "Naive day"
    renderer.GROUP_LABEL["LTM"] = "Return day"
    renderer.GROUP_LABEL["experienced"] = "Second day"
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 0.70,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    """Save editable masters plus a PowerPoint-safe all-vector SVG derivative."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in R6.R5.BASE.FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": None, "facecolor": "white"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, **kwargs)
        outputs.append(path)

    ppt_path = stem.parent.parent / "ppt_vector" / f"{stem.name}.svg"
    ppt_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(ppt_path, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return outputs


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND6_BUILD_PANEL_A(renderer)
    size_map = {6.0: 6.9, 5.6: 6.4, 5.4: 6.2}
    for ax in fig.axes:
        for text in ax.texts:
            current = float(text.get_fontsize())
            replacement = size_map.get(round(current, 1))
            if replacement is not None:
                text.set_fontsize(replacement)
    return fig, tables


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND6_BUILD_POLICY_SCATTER(
        renderer, points, contrasts, groups, panel
    )
    # b/c use one shared three-experience legend in b; c has no legend.
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if panel == "b":
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=renderer.GROUP_MARKER[group],
                markerfacecolor="white",
                markeredgecolor=renderer.GROUP_COLOR[group],
                markeredgewidth=0.7,
                markersize=4.0,
            )
            for group in renderer.GROUP_ORDER
        ]
        fig.legend(
            handles,
            [renderer.GROUP_LABEL[group] for group in renderer.GROUP_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.56, 0.985),
            ncol=3,
            fontsize=LEGEND_SIZE,
            handlelength=0.60,
            handletextpad=0.12,
            columnspacing=0.42,
            borderaxespad=0,
        )
    # Retain a real white safety margin around the shared b/c axes.  The larger
    # round-7 tick labels otherwise reached the left SVG/PNG boundary even
    # though they looked acceptable in the composite.
    fig.subplots_adjust(left=0.125, right=0.955, bottom=0.13, top=0.84, wspace=0.34)
    return fig, tables


def _absolute_interval(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    crosses = (low <= 0) & (high >= 0)
    absolute_low = np.where(crosses, 0.0, np.minimum(np.abs(low), np.abs(high)))
    absolute_high = np.maximum(np.abs(low), np.abs(high))
    return absolute_low, absolute_high


def build_panel_d(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["parameter"] = pd.Categorical(
        rows["parameter"], renderer.POLICY_ORDER, ordered=True
    )
    rows = rows.sort_values("parameter").reset_index(drop=True)
    signed_columns = (
        "delta_efficiency_mean",
        "delta_efficiency_ci_low",
        "delta_efficiency_ci_high",
    )
    rows[list(signed_columns)] = rows[list(signed_columns)].apply(
        pd.to_numeric, errors="raise"
    )
    signed_mean = rows["delta_efficiency_mean"].to_numpy(dtype=float)
    signed_low = rows["delta_efficiency_ci_low"].to_numpy(dtype=float)
    signed_high = rows["delta_efficiency_ci_high"].to_numpy(dtype=float)
    absolute_low, absolute_high = _absolute_interval(signed_low, signed_high)
    rows["absolute_delta_efficiency_mean"] = np.abs(signed_mean)
    rows["absolute_delta_efficiency_ci_low"] = absolute_low
    rows["absolute_delta_efficiency_ci_high"] = absolute_high
    rows["direction"] = np.where(
        signed_mean < 0, "naive_substituted_higher", "unmodified_return_higher"
    )
    rows["interval_transform"] = (
        "absolute image of signed 95% CI; lower bound is zero when signed CI crosses zero"
    )

    y = np.arange(len(rows))[::-1]
    mean = 100 * rows["absolute_delta_efficiency_mean"].to_numpy(dtype=float)
    low = 100 * rows["absolute_delta_efficiency_ci_low"].to_numpy(dtype=float)
    high = 100 * rows["absolute_delta_efficiency_ci_high"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=R6.PANEL_SIZE_IN["d"])
    for yi, value, lo, hi, signed, parameter in zip(
        y, mean, low, high, signed_mean, rows["parameter"]
    ):
        marker = "o"
        face = "white" if signed < 0 else PANEL_D_NEUTRAL
        ax.hlines(yi, 0, value, color=PANEL_D_NEUTRAL, linewidth=1.25)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt=marker,
            color=PANEL_D_NEUTRAL,
            markerfacecolor=face,
            markeredgecolor=PANEL_D_NEUTRAL,
            markeredgewidth=0.85,
            markersize=5.0,
            capsize=2.0,
            elinewidth=1.0,
        )
    ax.set_yticks(
        y, [renderer.POLICY_LABEL[str(item)] for item in rows["parameter"]]
    )
    ax.set_xlabel("|Efficiency difference| (pp)", labelpad=3.0)
    ax.set_xlim(0, max(9.6, float(high.max()) + 0.45))
    ax.set_xticks((0, 4, 8))
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.16, top=0.96)
    return fig, {"policy_contributions": rows}


def build_panel_e(
    renderer: Any, rows: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND6_BUILD_PANEL_E(renderer, rows)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    old = ax.get_legend()
    if old is not None:
        old.remove()
    label_map = {"Naive": "Naive day", "Day 2": "Second day", "Second day": "Second day"}
    ax.legend(
        handles,
        [label_map.get(label, label) for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.46, 1.015),
        ncol=2,
        fontsize=5.3,
        handletextpad=0.06,
        columnspacing=0.12,
        markerscale=0.86,
        borderaxespad=0,
    )
    ax.set_ylabel(r"$|\Delta P|$ from Return day", labelpad=4.0)
    # The narrow panel needs both a larger left canvas reserve and an explicit
    # label coordinate: this leaves the ylabel inside the vector canvas while
    # separating it from the 0.00/0.05/0.10 tick labels.
    ax.yaxis.set_label_coords(-0.333, 0.5)
    fig.subplots_adjust(left=0.302, right=0.96, bottom=0.13, top=0.92)
    return fig, tables


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
    fig, axes = plt.subplots(
        1,
        3,
        figsize=R6.PANEL_SIZE_IN[panel],
        sharex=True,
        sharey=True,
    )
    for ax, group in zip(axes, renderer.GROUP_ORDER):
        sub = points.loc[points["group"].eq(group)]
        color = renderer.GROUP_COLOR[group]
        ax.plot(
            (0, limit),
            (0, limit),
            color=renderer.LIGHT,
            linestyle="--",
            linewidth=0.8,
        )
        ax.scatter(
            sub["real_half_cover_eff"],
            sub[simulated_column],
            s=13,
            facecolors=renderer.BACKGROUND,
            edgecolors=color,
            linewidths=0.60,
            alpha=0.78,
        )
        ax.set_title(
            renderer.GROUP_LABEL[group],
            color=color,
            pad=0.0,
            fontsize=6.5,
            y=1.13,
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks((0, 0.4, 0.8))
        ax.set_yticks((0, 0.4, 0.8))
        R6.R5.BASE.clean_axes(ax)
    model_title = (
        "Four-policy model"
        if model == "four_bias"
        else r"Depth-varying $P_{BS}$ model"
    )
    fig.suptitle(model_title, y=0.975, fontsize=MODEL_TITLE_SIZE, fontweight="bold")
    fig.supxlabel("Observed efficiency", y=0.075, fontsize=AXIS_LABEL_SIZE)
    fig.supylabel("Simulated efficiency", x=0.025, fontsize=AXIS_LABEL_SIZE)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.84, wspace=0.12)
    return fig, {"session_points": points, "calibration_summary": summary}


def _replace_legend(
    ax: plt.Axes,
    labels: list[str],
    *,
    ncol: int,
    bbox: tuple[float, float],
    fontsize: float,
) -> None:
    handles, _ = ax.get_legend_handles_labels()
    old = ax.get_legend()
    if old is not None:
        old.remove()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=bbox,
        ncol=ncol,
        fontsize=fontsize,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND6_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    ax, ax_delta = fig.axes[:2]
    _replace_legend(
        ax,
        ["Naive day", "Return day", "Second day"],
        ncol=2,
        bbox=(0.58, 1.02),
        fontsize=5.4,
    )
    _replace_legend(
        ax_delta,
        ["Return day − Naive day", "Second day − Return day"],
        ncol=1,
        bbox=(0.58, 1.02),
        fontsize=5.4,
    )
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.15, top=0.89, hspace=0.44)
    return fig, tables


def build_panel_h(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND4_BUILD_PANEL_H(renderer, frame)
    ax = fig.axes[0]
    ax.set_yticks([])
    rows = tables["four_vs_depth_gap"].copy()
    rows["model_norm"] = pd.Categorical(
        rows["model_norm"], ("four_bias", "depth_stratified"), ordered=True
    )
    rows = rows.sort_values("model_norm").reset_index(drop=True)
    labels = ("Four-policy", r"Depth $P_{BS}$")
    for yi, row, label in zip((1, 0), rows.itertuples(index=False), labels):
        value = float(row.delta_simulated)
        ax.text(
            value - 0.045,
            yi,
            label,
            ha="right",
            va="center",
            fontsize=5.4,
            color=renderer.INK,
        )
    observed = float(rows["delta_real"].iloc[0])
    for text in list(ax.texts):
        if text.get_text() == "Observed":
            text.set_x(observed - 0.020)
            text.set_ha("right")
            text.set_fontsize(5.4)
    ax.set_xlabel("Return day − Naive day\nefficiency gap", labelpad=2.5)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.95)
    return fig, tables


def _write_ppt_vector_manifest(output_root: Path) -> None:
    rows = []
    for path in sorted((output_root / "ppt_vector").glob("*.svg")):
        rows.append(
            {
                "role": path.name[4],
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "sha256": R6.R5.BASE.sha256(path),
                "bytes": path.stat().st_size,
                "text_mode": "outlined paths for PowerPoint rendering only",
            }
        )
    path = output_root / "PPT_VECTOR_MANIFEST.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "path", "sha256", "bytes", "text_mode"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _update_caption(output_root: Path) -> None:
    path = output_root / "CAPTION_NOTES.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = pd.read_csv(output_root / "plot_ready" / "panel_d_policy_contributions.csv")
    values = "; ".join(
        f"{row.parameter} {100 * row.absolute_delta_efficiency_mean:.1f} pp "
        f"[{100 * row.absolute_delta_efficiency_ci_low:.1f}, "
        f"{100 * row.absolute_delta_efficiency_ci_high:.1f}]"
        for row in rows.itertuples(index=False)
    )
    replacement = (
        "**d,** Absolute magnitude of the same signed Return-day-own-policy minus "
        "Return-day-with-one-policy-fixed-to-the-Naive-day-mean efficiency contrast. "
        "Open symbols denote negative signed contrasts (Naive-substituted simulation "
        "higher); filled symbols denote positive signed contrasts (unmodified Return "
        f"simulation higher). Magnitudes and absolute images of the saved signed 95% CIs: {values}. "
        "For an interval crossing zero, the displayed absolute lower bound is zero. "
        "One-at-a-time substitutions are nonlinear, non-additive, and not causal components; "
        "donor means were fixed and Monte Carlo simulation noise was not propagated."
    )
    for index, line in enumerate(lines):
        if line.startswith("# Extended Data Fig. 2 round-6"):
            lines[index] = "# Extended Data Fig. 2 round-7 caption notes"
        elif line.startswith("**d,**"):
            lines[index] = replacement
    lines.extend(
        [
            "",
            "Round-7 display note: visible experience names are standardized to Naive day, "
            "Return day, and Second day. Scientific panels are delivered as editable SVG/PDF "
            "and 600-dpi PNG/TIFF; separate outlined-text SVG derivatives are restricted to "
            "PowerPoint placement because the editable SVG text renderer is not reliable there.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    R6.PANEL_STEM = PANEL_STEM
    R6.R5.BASE.configure_style = configure_style
    R6.R5.BASE.save_figure = save_figure
    R6.build_panel_a = build_panel_a
    R6.build_policy_scatter = build_policy_scatter
    R6.build_panel_d = build_panel_d
    R6.build_panel_e = build_panel_e
    R6.build_calibration = build_calibration
    R6.build_panel_g = build_panel_g
    R6.build_panel_h = build_panel_h

    result = R6.render(record_root, output_root, dpi)
    _write_ppt_vector_manifest(output_root)
    _update_caption(output_root)

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 7
    manifest["presentation_changes"] = [
        "one shared Naive day / Return day / Second day legend for panels b and c",
        "panel d displays absolute magnitudes using one neutral color and marker fill for signed direction",
        "larger Arial typography throughout",
        "corrected y-label clearance in e, f, and i",
        "vertically centered f/i calibration axes with closer shared x labels",
        "collision-free direct labels in h",
        "distinct Four-policy and Depth-varying P_BS model titles",
        "editable vector masters plus PowerPoint-safe vector derivatives",
    ]
    manifest["accepted_round3_display_derivations"] = {
        "d": (
            "absolute magnitude of the accepted frozen signed contribution "
            "contrast; signed values and direction are retained"
        ),
        "e": (
            "deliberate replacement of the accepted signed shift-from-Naive "
            "summary with absolute Naive-day and Second-day distances from "
            "Return-day group means to display policy-profile proximity"
        ),
    }
    manifest["typography_contract"] = {
        "font_family": "Arial",
        "base_pt": BASE_FONT_SIZE,
        "axis_label_pt": AXIS_LABEL_SIZE,
        "tick_pt": TICK_LABEL_SIZE,
        "title_pt": TITLE_SIZE,
        "legend_pt": LEGEND_SIZE,
        "panel_letter_pt": 10.5,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result["review_round"] = 7
    result["typography_contract"] = manifest["typography_contract"]
    result["shared_bc_legend"] = "embedded once in panel b; panel c has no legend"
    result["ppt_vector_manifest"] = "PPT_VECTOR_MANIFEST.csv"
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
