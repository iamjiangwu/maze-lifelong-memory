"""Render ED2 Round 12 with proportional panel-a artists and compact panel g.

Round 12 preserves the accepted nine-panel inventory, source tables, panel
frames, crop boxes, typography contract, colors, marker grammar, and row-wise
axis baselines from Round 11.  Only two display defects are corrected:

* panel a scales every point-sized schematic artist by the same 0.8 factor as
  the grid, including arrow strokes, arrowheads, and text;
* panel g compresses the shallow/middle/deep x positions and increases the
  left label gutter so both y-axis titles clear their tick labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND11_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_aligned_round11.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round12_vector_vor"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round12_fix_2026-08-09" / "panels"
SCHEMATIC_ARTIST_SCALE = 0.8
PANEL_G_GROUP_SCALE = 0.72
PANEL_G_LEFT = 0.27
PANEL_G_YLABEL_X = -0.205


def _load_round11() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round11_renderer", ROUND11_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND11_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R11 = _load_round11()
ROUND11_BUILD_PANEL_G = R11.build_panel_g


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Keep the 0.8x grid and scale every point-sized artist with it."""

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
                    "text": "preceding turn",
                    "xy": (2.90, 4.17),
                    "ha": "right",
                    "va": "center",
                }
            ]
    style = {
        "cell_linewidth": 0.9 * SCHEMATIC_ARTIST_SCALE,
        "arrow_linewidth": 1.25 * SCHEMATIC_ARTIST_SCALE,
        "previous_arrow_linewidth": 1.25 * SCHEMATIC_ARTIST_SCALE,
        "arrow_mutation_scale": 7.5 * SCHEMATIC_ARTIST_SCALE,
        "title_fontsize": 6.9 * SCHEMATIC_ARTIST_SCALE,
        "title_fontweight": "normal",
        "or_fontsize": 6.4 * SCHEMATIC_ARTIST_SCALE,
        "annotation_fontsize": 6.2 * SCHEMATIC_ARTIST_SCALE,
    }
    fig, ax, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={
            "panel_size": R11.PANEL_SIZE_IN["a"],
            "tight_layout": False,
            "xlim": (-0.35, 3.15),
            "ylim": (-0.49375, 4.44375),
        },
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style=style,
        text=text,
        save=False,
    )
    ax.set_xlim(-0.35, 3.15)
    ax.set_ylim(-0.49375, 4.44375)
    fig.set_size_inches(*R11.PANEL_SIZE_IN["a"], forward=True)
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


def _compress_data_x(ax: plt.Axes, *, center: float, scale: float) -> None:
    """Compress data artists/ticks around ``center`` without changing data values."""

    transform = lambda values: center + (np.asarray(values, dtype=float) - center) * scale
    original_xlim = ax.get_xlim()
    original_ticks = ax.get_xticks()
    original_ticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
    for line in ax.lines:
        if line.get_transform() == ax.transData:
            line.set_xdata(transform(line.get_xdata(orig=False)))
    for collection in ax.collections:
        if collection.get_transform() != ax.transData:
            continue
        if isinstance(collection, LineCollection):
            segments = collection.get_segments()
            for segment in segments:
                segment[:, 0] = transform(segment[:, 0])
            collection.set_segments(segments)
        elif isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                path.vertices[:, 0] = transform(path.vertices[:, 0])
    ax.set_xticks(transform(original_ticks), original_ticklabels)
    ax.set_xlim(original_xlim)


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    # Start from the complete accepted Round-11 artist/legend/marker grammar.
    fig, tables = ROUND11_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    ax, ax_delta = fig.axes[:2]
    for target in (ax, ax_delta):
        _compress_data_x(target, center=1.0, scale=PANEL_G_GROUP_SCALE)

    fig.subplots_adjust(
        left=PANEL_G_LEFT,
        right=0.98,
        bottom=0.256662900,
        top=0.89,
        hspace=0.44,
    )
    for target in (ax, ax_delta):
        target.yaxis.set_label_coords(PANEL_G_YLABEL_X, 0.5)
    return fig, tables


def _update_metadata(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "review_candidate_not_current_version_record"
    manifest["review_round"] = 12
    manifest["presentation_changes"] = list(manifest.get("presentation_changes", [])) + [
        "scaled panel-a arrow strokes, arrowheads, labels, or-text, and preceding-turn text by the same 0.8 factor as the grid",
        "compressed panel-g shallow/middle/deep positions and enlarged its left label gutter without changing source values",
    ]
    manifest["panel_a_display_contract_round12"] = {
        "grid_scale_vs_round10": 0.8,
        "point_artist_scale_vs_round11": SCHEMATIC_ARTIST_SCALE,
        "style": {
            "cell_linewidth": 0.72,
            "arrow_linewidth": 1.0,
            "previous_arrow_linewidth": 1.0,
            "arrow_mutation_scale": 6.0,
            "title_fontsize": 5.52,
            "or_fontsize": 5.12,
            "annotation_fontsize": 4.96,
        },
        "preceding_turn": {
            "text": "preceding turn",
            "xy": [2.90, 4.17],
            "must_not_overlap_grid": True,
        },
    }
    manifest["panel_g_display_contract_round12"] = {
        "group_position_scale_vs_round11": PANEL_G_GROUP_SCALE,
        "group_center": 1.0,
        "group_positions": [0.28, 1.0, 1.72],
        "subplot_left": PANEL_G_LEFT,
        "ylabel_axis_x": PANEL_G_YLABEL_X,
        "source_values_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8").replace(
        "# Extended Data Fig. 2 round-11 caption notes",
        "# Extended Data Fig. 2 round-12 caption notes",
        1,
    )
    caption_path.write_text(caption, encoding="utf-8")
    return manifest


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    original_a = R11.build_panel_a
    original_g = R11.build_panel_g
    try:
        R11.build_panel_a = build_panel_a
        R11.build_panel_g = build_panel_g
        result = R11.render(record_root, output_root, dpi)
    finally:
        R11.build_panel_a = original_a
        R11.build_panel_g = original_g
    manifest = _update_metadata(output_root)
    result.update(
        {
            "review_round": 12,
            "panel_a_display_contract": manifest["panel_a_display_contract_round12"],
            "panel_g_display_contract": manifest["panel_g_display_contract_round12"],
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
