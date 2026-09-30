"""Render the superseding ED2 slide-3 Round 9 review.

Round 9 preserves the Round 8 science and source tables while completing the
page-wide hollow-circle marker contract, reducing oversized b/c mean symbols,
opening the shared b/c legend spacing, and using the former row gaps as panel
height.  Panel letters sit inside each panel's existing top whitespace.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND8_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_shared_legend_round8.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp" / "ed02_round9_marker_consistency_2026-08-08" / "panels"
)

PANEL_SIZE_IN = {
    "a": (148 / 96, 192 / 96),
    "b": (258 / 96, 192 / 96),
    "c": (258 / 96, 192 / 96),
    "d": (160 / 96, 234 / 96),
    "e": (128 / 96, 234 / 96),
    "f": (400 / 96, 234 / 96),
    "g": (176 / 96, 234 / 96),
    "h": (112 / 96, 234 / 96),
    "i": (400 / 96, 234 / 96),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_round9",
    "b": "ed02b_naive_vs_return_policy_scatter_round9",
    "c": "ed02c_return_vs_second_day_policy_scatter_round9",
    "d": "ed02d_signed_policy_contributions_round9",
    "e": "ed02e_distance_from_return_round9",
    "f": "ed02f_four_policy_calibration_round9",
    "g": "ed02g_depth_pbs_profile_round9",
    "h": "ed02h_four_vs_depth_reconstruction_round9",
    "i": "ed02i_depth_calibration_round9",
}

SHARED_LEGEND_STEM = "ed02_shared_bc_experience_legend_round9"


def _load_round8() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round8_renderer", ROUND8_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND8_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R8 = _load_round8()
ROUND8_BUILD_POLICY_SCATTER = R8.build_policy_scatter
ROUND7_BUILD_PANEL_H = R8.R7.build_panel_h


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND8_BUILD_POLICY_SCATTER(
        renderer, points, contrasts, groups, panel
    )
    # Every marker has the same s=11 area.  Fill alone identifies the
    # session-weighted mean; individual sessions remain hollow.
    for ax in fig.axes:
        current_group_color = None
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            sizes = collection.get_sizes()
            edge = collection.get_edgecolors()
            if len(sizes) and float(np.max(sizes)) <= 20:
                if edge is not None and len(edge):
                    current_group_color = edge[0]
                continue
            if len(sizes) and float(np.max(sizes)) > 20:
                collection.set_sizes(np.full_like(sizes, 11.0, dtype=float))
                if current_group_color is None:
                    raise RuntimeError("Cannot resolve b/c mean marker group color")
                collection.set_facecolor(current_group_color)
                collection.set_edgecolor(current_group_color)
    return fig, tables


def build_panel_h(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND7_BUILD_PANEL_H(renderer, frame)
    ax = fig.axes[0]
    for container in ax.containers:
        lines = getattr(container, "lines", None)
        if not lines:
            continue
        marker_line = lines[0]
        if not isinstance(marker_line, Line2D):
            continue
        marker_line.set_marker("o")
        marker_line.set_markerfacecolor("white")
        marker_line.set_markeredgecolor(marker_line.get_color())
        marker_line.set_markeredgewidth(0.90)
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            edge = collection.get_edgecolors()
            collection.set_facecolor("white")
            if edge is not None and len(edge):
                collection.set_edgecolor(edge)
    return fig, tables


def build_shared_bc_legend(output_root: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(196 / 96, 18 / 96), facecolor="none")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=R8.GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in R8.GROUP_ORDER
    ]
    fig.legend(
        handles,
        [R8.GROUP_LABEL[group] for group in R8.GROUP_ORDER],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        fontsize=5.4,
        handlelength=0.60,
        handletextpad=0.50,
        columnspacing=0.90,
        borderaxespad=0,
        frameon=False,
    )
    return R8._save_shared_legend(fig, output_root, dpi)


def _update_round9_metadata(output_root: Path) -> dict[str, Any]:
    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8").replace(
        "# Extended Data Fig. 2 round-8 caption notes",
        "# Extended Data Fig. 2 round-9 caption notes",
        1,
    )
    caption += (
        "\nRound-9 display note: b/c session-weighted means are solid circles with "
        "exactly the same marker area as the hollow individual-session circles; "
        "the dedicated shared b/c key increases marker-to-text padding to visually match "
        "panel e. "
        "Panel h now uses hollow circles for both model estimates, so shape and fill no "
        "longer introduce a competing category code. Row-2/3 panels absorb the former "
        "inter-row whitespace, and a-i letters sit inside existing panel-top whitespace.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 9
    manifest["presentation_changes"] = [
        "b/c group means changed to solid circles at the individual-session size s=11",
        "shared b/c legend marker-to-text padding increased to match panel e visually",
        "panel h model estimates standardized to hollow circles",
        "row-2 and row-3 panel heights increased from 224 to 234 slide units",
        "a-i panel letters moved to the matching panel-frame top",
    ]
    manifest["marker_contract"]["panels"] = ["b", "c", "e", "f", "g", "h", "i"]
    manifest["marker_contract"]["scope"] = (
        "all colored compared-category markers, including panel-h model estimates"
    )
    manifest["marker_contract"]["aggregate_exception"] = {
        "panels": ["b", "c"],
        "role": "session-weighted group mean",
        "shape": "circle",
        "size": 11,
        "fill": "solid maintained group color",
        "individuals": "hollow circles at the same size 11",
    }
    manifest["shared_bc_legend"]["editable_asset"] = (
        f"shared_assets/{SHARED_LEGEND_STEM}.svg"
    )
    manifest["shared_bc_legend"]["powerpoint_asset"] = (
        f"ppt_vector/{SHARED_LEGEND_STEM}.svg"
    )
    manifest["shared_bc_legend"]["frame_720"] = {
        "left": 342,
        "top": 52,
        "width": 196,
        "height": 18,
    }
    manifest["shared_bc_legend"]["entry_spacing"] = {
        "handletextpad": 0.50,
        "columnspacing": 0.90,
    }
    manifest["layout_contract_round9"] = {
        "row2_frames": {"top": 230, "height": 234, "bottom": 464},
        "row3_frames": {"top": 474, "height": 234, "bottom": 708},
        "inter_row_gaps": [10, 10],
        "bottom_margin": 12,
        "panel_letter_tops": [28, 230, 474],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    R8.PANEL_STEM = PANEL_STEM
    R8.SHARED_LEGEND_STEM = SHARED_LEGEND_STEM
    R8.build_policy_scatter = build_policy_scatter
    R8.build_shared_bc_legend = build_shared_bc_legend
    R8.R7.build_panel_h = build_panel_h
    R8.R7.R6.PANEL_SIZE_IN = PANEL_SIZE_IN

    result = R8.render(record_root, output_root, dpi)
    manifest = _update_round9_metadata(output_root)
    result.update(
        {
            "review_round": 9,
            "layout_contract": manifest["layout_contract_round9"],
            "shared_bc_legend": manifest["shared_bc_legend"],
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
