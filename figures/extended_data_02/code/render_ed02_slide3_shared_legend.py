"""Render the ED2 slide-3 Round 8 review from the accepted Round 3 authority.

Round 8 keeps the nine scientific slots a-i and the frozen fixed-30-minute
inputs.  It restores panel d to the accepted signed exploration-efficiency
difference, moves the panel-e experience legend slightly upward, replaces the
panel-b-only legend with one source-built legend shared by b and c, and uses
hollow circles consistently for experience groups.
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
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND7_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_typography_round7.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp" / "ed02_round8_shared_legend_2026-08-08" / "panels"
)

PANEL_D_NEUTRAL = "#68737D"
SHARED_LEGEND_STEM = "ed02_shared_bc_experience_legend_round8"
GROUP_ORDER = ("naive", "LTM", "experienced")
GROUP_LABEL = {
    "naive": "Naive day",
    "LTM": "Return day",
    "experienced": "Second day",
}
GROUP_COLOR = {"naive": "#1F90D5", "LTM": "#D82929", "experienced": "#5F8C62"}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_round8",
    "b": "ed02b_naive_vs_return_policy_scatter_round8",
    "c": "ed02c_return_vs_second_day_policy_scatter_round8",
    "d": "ed02d_signed_policy_contributions_round8",
    "e": "ed02e_distance_from_return_round8",
    "f": "ed02f_four_policy_calibration_round8",
    "g": "ed02g_depth_pbs_profile_round8",
    "h": "ed02h_four_vs_depth_reconstruction_round8",
    "i": "ed02i_depth_calibration_round8",
}


def _load_round7() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round7_renderer", ROUND7_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND7_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R7 = _load_round7()
ROUND7_CONFIGURE_STYLE = R7.configure_style
ROUND7_BUILD_POLICY_SCATTER = R7.build_policy_scatter
ROUND7_BUILD_PANEL_E = R7.build_panel_e
ROUND7_BUILD_PANEL_G = R7.build_panel_g


def configure_style(renderer: Any) -> None:
    """Retain Round 7 typography and make color the experience identifier."""

    ROUND7_CONFIGURE_STYLE(renderer)
    renderer.GROUP_MARKER.update({group: "o" for group in renderer.GROUP_ORDER})


def _hollow_path_collections(fig: plt.Figure) -> None:
    for ax in fig.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            edge = collection.get_edgecolors()
            if edge is None or len(edge) == 0:
                continue
            collection.set_facecolor("white")
            collection.set_linewidth(max(0.75, float(np.max(collection.get_linewidths()))))


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND7_BUILD_POLICY_SCATTER(
        renderer, points, contrasts, groups, panel
    )
    # The experience key is a true figure-level asset shared by b and c.
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    _hollow_path_collections(fig)
    return fig, tables


def build_panel_d(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Show the accepted signed contrast on the native efficiency scale."""

    rows = frame.copy()
    rows["parameter"] = pd.Categorical(
        rows["parameter"], renderer.POLICY_ORDER, ordered=True
    )
    rows = rows.sort_values("parameter").reset_index(drop=True)
    value_columns = (
        "delta_efficiency_mean",
        "delta_efficiency_ci_low",
        "delta_efficiency_ci_high",
    )
    rows[list(value_columns)] = rows[list(value_columns)].apply(
        pd.to_numeric, errors="raise"
    )

    mean = rows["delta_efficiency_mean"].to_numpy(dtype=float)
    low = rows["delta_efficiency_ci_low"].to_numpy(dtype=float)
    high = rows["delta_efficiency_ci_high"].to_numpy(dtype=float)
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=R7.R6.PANEL_SIZE_IN["d"])
    ax.axvline(0, color=renderer.REFERENCE, linewidth=0.85, zorder=0)
    for yi, value, lo, hi in zip(y, mean, low, high):
        ax.hlines(yi, 0, value, color=PANEL_D_NEUTRAL, linewidth=1.25, zorder=1)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            color=PANEL_D_NEUTRAL,
            markerfacecolor="white",
            markeredgecolor=PANEL_D_NEUTRAL,
            markeredgewidth=0.90,
            markersize=5.0,
            capsize=2.0,
            elinewidth=1.0,
            zorder=2,
        )
    ax.set_yticks(
        y, [renderer.POLICY_LABEL[str(item)] for item in rows["parameter"]]
    )
    ax.set_xlabel("Exploration efficiency\ndifference", labelpad=1.5)
    ax.set_xlim(-0.04, 0.096)
    ax.set_xticks((-0.04, 0, 0.04, 0.08))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
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
    fig, tables = ROUND7_BUILD_PANEL_E(renderer, rows)
    ax = fig.axes[0]
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    # Raise the key only slightly; the axes and estimand remain unchanged.
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=renderer.GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in ("naive", "experienced")
    ]
    ax.legend(
        handles,
        ["Naive day", "Second day"],
        loc="upper center",
        bbox_to_anchor=(0.44, 1.035),
        ncol=2,
        fontsize=5.4,
        handletextpad=0.06,
        columnspacing=0.12,
        markerscale=0.86,
        borderaxespad=0,
    )
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        label = collection.get_label()
        if label in {"Naive", "Naive day"}:
            color = renderer.GROUP_COLOR["naive"]
        elif label == "Second day":
            color = renderer.GROUP_COLOR["experienced"]
        else:
            continue
        collection.set_facecolor("white")
        collection.set_edgecolor(color)
        collection.set_linewidth(0.85)
    return fig, tables


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Retain Round 7 data/layout but use hollow circles for every experience key."""

    fig, tables = ROUND7_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    for ax in fig.axes[:2]:
        for line in ax.lines:
            label = line.get_label()
            if label.startswith("_"):
                continue
            if line.get_marker() not in {"None", "", "none"}:
                line.set_marker("o")
                line.set_markerfacecolor("white")
                line.set_markeredgecolor(line.get_color())
                line.set_markeredgewidth(0.80)
        for container in ax.containers:
            lines = getattr(container, "lines", None)
            if not lines:
                continue
            marker_line = lines[0]
            if isinstance(marker_line, Line2D):
                marker_line.set_marker("o")
                marker_line.set_markerfacecolor("white")
                marker_line.set_markeredgecolor(marker_line.get_color())
                marker_line.set_markeredgewidth(0.80)

    ax, ax_delta = fig.axes[:2]
    for target in (ax, ax_delta):
        legend = target.get_legend()
        if legend is not None:
            legend.remove()
    top_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor=renderer.GROUP_COLOR[group], markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in renderer.GROUP_ORDER
    ]
    ax.legend(
        top_handles,
        [renderer.GROUP_LABEL[group] for group in renderer.GROUP_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.58, 1.02),
        ncol=2,
        fontsize=5.4,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )
    contrast_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=0.80, markersize=4.0,
        )
        for color in (renderer.GROUP_COLOR["LTM"], renderer.GROUP_COLOR["experienced"])
    ]
    ax_delta.legend(
        contrast_handles,
        ["Return day − Naive day", "Second day − Return day"],
        loc="upper center",
        bbox_to_anchor=(0.58, 1.02),
        ncol=1,
        fontsize=5.4,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )
    return fig, tables


def _save_shared_legend(fig: plt.Figure, output_root: Path, dpi: int) -> list[Path]:
    editable_root = output_root / "shared_assets"
    ppt_root = output_root / "ppt_vector"
    editable_root.mkdir(parents=True, exist_ok=True)
    ppt_root.mkdir(parents=True, exist_ok=True)
    stem = editable_root / SHARED_LEGEND_STEM
    outputs: list[Path] = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": None, "transparent": True}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, **kwargs)
        outputs.append(path)
    ppt_path = ppt_root / f"{SHARED_LEGEND_STEM}.svg"
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(ppt_path, bbox_inches=None, transparent=True)
    plt.close(fig)
    return outputs


def build_shared_bc_legend(output_root: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(196 / 96, 18 / 96), facecolor="none")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles,
        [GROUP_LABEL[group] for group in GROUP_ORDER],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        fontsize=5.4,
        handlelength=0.60,
        handletextpad=0.12,
        columnspacing=0.42,
        borderaxespad=0,
        frameon=False,
    )
    return _save_shared_legend(fig, output_root, dpi)


def _write_vector_manifests(output_root: Path) -> None:
    rows = []
    for panel, stem in PANEL_STEM.items():
        path = output_root / "ppt_vector" / f"{stem}.svg"
        rows.append(
            {
                "role": panel,
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "sha256": R7.R6.R5.BASE.sha256(path),
                "bytes": path.stat().st_size,
                "text_mode": "outlined paths for PowerPoint rendering only",
            }
        )
    shared = output_root / "ppt_vector" / f"{SHARED_LEGEND_STEM}.svg"
    rows.append(
        {
            "role": "bc_shared_legend",
            "path": str(shared.relative_to(output_root)).replace("\\", "/"),
            "sha256": R7.R6.R5.BASE.sha256(shared),
            "bytes": shared.stat().st_size,
            "text_mode": "outlined paths for PowerPoint rendering only",
        }
    )
    with (output_root / "PPT_VECTOR_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    shared_rows = []
    for path in sorted((output_root / "shared_assets").glob(f"{SHARED_LEGEND_STEM}.*")):
        shared_rows.append(
            {
                "role": "bc_shared_legend",
                "path": str(path.relative_to(output_root)).replace("\\", "/"),
                "sha256": R7.R6.R5.BASE.sha256(path),
                "bytes": path.stat().st_size,
                "format": path.suffix.lstrip("."),
            }
        )
    with (output_root / "SHARED_ASSET_MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=shared_rows[0].keys())
        writer.writeheader()
        writer.writerows(shared_rows)


def _update_caption(output_root: Path) -> None:
    path = output_root / "CAPTION_NOTES.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    d_text = (
        "**d,** Direct signed difference in simulated exploration efficiency: "
        "the Return-day simulation using its own policy means minus the same Return-day "
        "simulation after one policy was fixed to the Naive-day common-support mean. "
        "Estimates and synchronized union-of-mice cluster-bootstrap 95% confidence "
        "intervals were $P_{SF}$ -0.4 pp [-0.8, +0.0], $P_{SA}$ -2.7 pp "
        "[-3.2, -2.2], $P_{BF}$ +8.0 pp [+7.2, +8.8], and $P_{BS}$ +6.9 pp "
        "[+5.8, +7.8]. Negative values indicate higher efficiency after the Naive-day "
        "substitution; positive values indicate higher efficiency with the unmodified "
        "Return-day policy. One-at-a-time substitutions are nonlinear, non-additive, "
        "and not causal components; donor means were fixed and Monte Carlo simulation "
        "noise was not propagated."
    )
    for index, line in enumerate(lines):
        if line.startswith("# Extended Data Fig. 2"):
            lines[index] = "# Extended Data Fig. 2 round-8 caption notes"
        elif line.startswith("**d,**"):
            lines[index] = d_text
        elif line.startswith("**b,**"):
            lines[index] = line.replace("Filled symbols denote", "Larger symbols denote")
    lines.extend(
        [
            "",
            "Round-8 display note: panels b and c use one separate shared experience "
            "legend centered across both panels. Naive day, Return day, and Second day "
            "are encoded consistently by hollow circles and their maintained colors in "
            "b, c, e, f, g, and i. Panel e's two-item key was raised slightly without "
            "changing its distance-from-Return estimand. Editable SVG/PDF and 600-dpi "
            "PNG/TIFF exports are retained; outlined-text SVGs are used only for PowerPoint.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    R7.PANEL_STEM = PANEL_STEM
    R7.configure_style = configure_style
    R7.build_policy_scatter = build_policy_scatter
    R7.build_panel_d = build_panel_d
    R7.build_panel_e = build_panel_e
    R7.build_panel_g = build_panel_g
    # Round 7's caption assumes mathematical absolute-value columns.  Keep the
    # signed Round 6 caption as the base and write the exact Round 8 wording below.
    R7._update_caption = lambda output_root: None
    R7._write_ppt_vector_manifest = lambda output_root: None

    result = R7.render(record_root, output_root, dpi)
    shared_outputs = build_shared_bc_legend(output_root, dpi)
    _write_vector_manifests(output_root)
    _update_caption(output_root)

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 8
    manifest["presentation_changes"] = [
        "a dedicated vector legend shared by panels b and c",
        "one hollow-circle marker grammar for experience groups in b, c, e, f, g, and i",
        "panel d restored to the accepted signed exploration-efficiency difference",
        "panel e experience legend moved slightly upward",
        "panel letters lowered into existing panel whitespace in the PowerPoint assembly",
    ]
    manifest["accepted_round3_display_derivations"] = {
        "e": (
            "deliberate replacement of the accepted signed shift-from-Naive summary "
            "with absolute Naive-day and Second-day distances from Return-day group "
            "means to display policy-profile proximity"
        )
    }
    manifest["round7_supersession"] = (
        "Round 7's mathematical absolute-value display of panel d is rejected; "
        "Round 8 restores the accepted signed contrast without changing source values."
    )
    manifest["marker_contract"] = {
        "experiences": ["Naive day", "Return day", "Second day"],
        "shape": "hollow circle",
        "identity_channel": "maintained experience color",
        "panels": ["b", "c", "e", "f", "g", "i"],
    }
    manifest["shared_bc_legend"] = {
        "editable_asset": f"shared_assets/{SHARED_LEGEND_STEM}.svg",
        "powerpoint_asset": f"ppt_vector/{SHARED_LEGEND_STEM}.svg",
        "frame_720": {"left": 342, "top": 36, "width": 196, "height": 18},
        "object_name": "ED2 shared b-c experience legend",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result.update(
        {
            "review_round": 8,
            "panel_count": 9,
            "shared_asset_count": 1,
            "shared_bc_legend": manifest["shared_bc_legend"],
            "shared_asset_outputs": [
                str(path.relative_to(output_root)).replace("\\", "/")
                for path in shared_outputs
            ],
            "ppt_vector_manifest": "PPT_VECTOR_MANIFEST.csv",
            "shared_asset_manifest": "SHARED_ASSET_MANIFEST.csv",
        }
    )
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
