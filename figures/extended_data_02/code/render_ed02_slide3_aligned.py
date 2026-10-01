"""Render the axis-aligned ED2 slide-3 Round 11 review.

Round 11 preserves the Round 10 scientific tables, panel inventory, source
crops, PowerPoint frames, typography, colors, and marker grammar.  It changes
only requested display geometry: a smaller route schematic with a clear
preceding-turn label, aligned row-wise x-axis baselines, tighter categorical
spacing in panels d/h, and panel letters one character higher.
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
import matplotlib.pyplot as plt
import pandas as pd


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND10_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_compact_round10.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp" / "ed02_round11_alignment_2026-08-09" / "panels"
)


def _load_round10() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round10_renderer", ROUND10_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND10_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R10 = _load_round10()
PANEL_SIZE_IN = dict(R10.PANEL_SIZE_IN)
CROP_BBOX_IN = dict(R10.CROP_BBOX_IN)
PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_round11",
    "b": "ed02b_naive_vs_return_policy_scatter_round11",
    "c": "ed02c_return_vs_second_day_policy_scatter_round11",
    "d": "ed02d_signed_policy_contributions_round11",
    "e": "ed02e_distance_from_return_round11",
    "f": "ed02f_four_policy_calibration_round11",
    "g": "ed02g_depth_pbs_profile_round11",
    "h": "ed02h_four_vs_depth_reconstruction_round11",
    "i": "ed02i_depth_calibration_round11",
}
SHARED_LEGEND_STEM = "ed02_shared_bc_experience_legend_round11"

ROUND10_BUILD_PANEL_D = R10.R9.R8.build_panel_d
ROUND10_BUILD_PANEL_E = R10.R9.R8.build_panel_e
ROUND10_BUILD_PANEL_G = R10.R9.R8.build_panel_g
ROUND10_BUILD_PANEL_H = R10.ROUND9_BUILD_PANEL_H
ROUND10_UPDATE_METADATA = R10._update_metadata

LABEL_FRAMES_720 = {
    "a": {"left": 4, "top": 27.5, "width": 18, "height": 16},
    "b": {"left": 154, "top": 27.5, "width": 18, "height": 16},
    "c": {"left": 438, "top": 27.5, "width": 18, "height": 16},
    "d": {"left": 4, "top": 229.5, "width": 18, "height": 16},
    "e": {"left": 145.476654, "top": 229.5, "width": 18, "height": 16},
    "f": {"left": 291.175063, "top": 229.5, "width": 18, "height": 16},
    "g": {"left": 4, "top": 473.5, "width": 18, "height": 16},
    "h": {"left": 177.175063, "top": 473.5, "width": 18, "height": 16},
    "i": {"left": 291.175063, "top": 473.5, "width": 18, "height": 16},
}

AXIS_ALIGNMENT_720 = {
    "row2": {
        "panels": ["d", "e", "f"],
        "x_axis_baseline": 411.225080,
        "matched_envelope_panels": ["d", "e"],
        "matched_axes_top": 232.345648,
        "matched_axes_height": 178.879433,
    },
    "row3": {
        "panels": ["g", "h", "i"],
        "x_axis_baseline": 649.986769,
        "matched_envelope_panels": ["g", "h"],
        "matched_axes_top": 501.645690,
        "matched_axes_height": 148.341080,
    },
}


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Render the canonical schematic at exactly 0.8x with a clear annotation."""

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
    fig, ax, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={
            "panel_size": PANEL_SIZE_IN["a"],
            "tight_layout": False,
            "xlim": (-0.35, 3.15),
            "ylim": (-0.49375, 4.44375),
        },
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 0.9,
            "arrow_linewidth": 1.25,
            "previous_arrow_linewidth": 1.25,
            "arrow_mutation_scale": 7.5,
            "title_fontsize": 6.9,
            "title_fontweight": "normal",
            "or_fontsize": 6.4,
            "annotation_fontsize": 6.2,
        },
        text=text,
        save=False,
    )
    ax.set_xlim(-0.35, 3.15)
    ax.set_ylim(-0.49375, 4.44375)
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


def build_panel_d(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND10_BUILD_PANEL_D(renderer, frame)
    ax = fig.axes[0]
    ax.set_ylim(-0.9, 3.9)
    fig.subplots_adjust(
        left=0.28,
        right=0.985,
        bottom=0.206887380,
        top=0.949497558,
    )
    return fig, tables


def build_panel_e(
    renderer: Any, rows: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND10_BUILD_PANEL_E(renderer, rows)
    fig.subplots_adjust(
        left=0.302,
        right=0.96,
        bottom=0.239070128,
        top=0.92,
    )
    return fig, tables


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND10_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    fig.subplots_adjust(
        left=0.21,
        right=0.98,
        bottom=0.256662900,
        top=0.89,
        hspace=0.44,
    )
    # Hold both stacked y labels at the accepted Round-10 horizontal anchor.
    # Matplotlib otherwise repositions the top label from the new tick extents
    # and lets its leftmost antialiasing touch the source crop boundary.
    for ax in fig.axes[:2]:
        ax.yaxis.set_label_coords(-0.143095, 0.5)
    return fig, tables


def build_panel_h(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND10_BUILD_PANEL_H(renderer, frame)
    ax = fig.axes[0]
    ax.set_ylim(-0.65, 1.65)
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.247288341,
        top=0.881224438,
    )
    return fig, tables


def _update_metadata(output_root: Path) -> dict[str, Any]:
    manifest = ROUND10_UPDATE_METADATA(output_root)

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.replace(
        "# Extended Data Fig. 2 round-10 caption notes",
        "# Extended Data Fig. 2 round-11 caption notes",
        1,
    )
    caption += (
        "\nRound-11 display note: the 720x720 slide, all panel frames, source crop "
        "boxes, scientific values, and marker/color/font contracts are unchanged from "
        "Round 10. Panel a's canonical route-policy geometry is displayed at exactly "
        "0.8x and its preceding-turn label is clear of the grid. Row-wise x-axis "
        "baselines are aligned for d/e/f and g/h/i; d/e and g/h use matched axes "
        "envelopes; categorical spacing is reduced in d and h. The a-i panel letters "
        "are row-aligned one character higher.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    manifest["review_round"] = 11
    manifest.pop("layout_contract_round10", None)
    manifest["presentation_changes"] = [
        "kept the 720x720 slide and every Round 10 panel/legend frame unchanged",
        "displayed the canonical panel-a grid geometry at exactly 0.8x",
        "moved the preceding-turn annotation clear of every panel-a grid stroke",
        "aligned d/e/f and g/h/i x-axis baselines at source-render level",
        "matched d/e and g/h axes-envelope heights",
        "reduced categorical spacing in panels d and h",
        "moved all a-i panel letters one 10.5-pt character upward and row-aligned them",
    ]
    manifest["layout_contract_round11"] = {
        "slide_size_720": [720, 720],
        "panel_frames_720": {},
        "panel_label_frames_720": LABEL_FRAMES_720,
        "row_regions_720": {
            "row1": {"top": 20.178340, "bottom": 222.298281},
            "row2": {"top": 224.298282, "bottom": 461.059971},
            "row3": {"top": 463.059971, "bottom": 699.821660},
        },
        "frame_gaps": [2, 2],
        "crop_mode": "source-level explicit Bbox; PowerPoint crop remains zero",
    }
    # Preserve the exact Round 10 picture geometry rather than reconstructing it
    # from source-crop aspects.
    manifest["layout_contract_round11"]["panel_frames_720"] = {
        "a": {"left": 2, "top": 20.178340, "width": 148, "height": 202.119941},
        "b": {"left": 152, "top": 80, "width": 282, "height": 139.333824},
        "c": {"left": 436, "top": 80, "width": 282, "height": 139.333824},
        "d": {"left": 2, "top": 224.298282, "width": 139.476654, "height": 236.761689},
        "e": {"left": 143.476654, "top": 224.298282, "width": 143.698408, "height": 236.761689},
        "f": {"left": 289.175063, "top": 224.298282, "width": 428.824937, "height": 236.761689},
        "g": {"left": 2, "top": 497.697222, "width": 171.175063, "height": 202.124437},
        "h": {"left": 175.175063, "top": 482.202860, "width": 112, "height": 217.618800},
        "i": {"left": 289.175063, "top": 463.059971, "width": 428.824937, "height": 236.761689},
    }
    manifest["panel_a_display_contract_round11"] = {
        "xlim": [-0.35, 3.15],
        "ylim": [-0.49375, 4.44375],
        "grid_scale_vs_round10": 0.8,
        "preceding_turn": {
            "text": "preceding turn",
            "xy": [2.90, 4.17],
            "horizontal_alignment": "right",
            "minimum_grid_clearance_source_pt": 3.5,
        },
    }
    manifest["axis_alignment_contract_round11"] = AXIS_ALIGNMENT_720
    manifest["category_spacing_contract_round11"] = {
        "d": {
            "round10_slide_units": 47.00084,
            "round11_slide_units": 37.26655,
            "reduction_fraction": 0.2071,
        },
        "h": {
            "round10_slide_units": 85.19419,
            "round11_slide_units": 64.49612,
            "reduction_fraction": 0.2430,
        },
    }
    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    # Configure the inherited rendering chain with the Round 11 source builders.
    R10.PANEL_SIZE_IN = PANEL_SIZE_IN
    R10.PANEL_STEM = PANEL_STEM
    R10.SHARED_LEGEND_STEM = SHARED_LEGEND_STEM
    R10.R9.PANEL_SIZE_IN = PANEL_SIZE_IN
    R10.R9.PANEL_STEM = PANEL_STEM
    R10.R9.SHARED_LEGEND_STEM = SHARED_LEGEND_STEM
    R10.R9.build_policy_scatter = R10.build_policy_scatter
    R10.R9.build_shared_bc_legend = R10.build_shared_bc_legend
    R10.R9.build_panel_h = build_panel_h
    R10.R9.R8.build_panel_d = build_panel_d
    R10.R9.R8.build_panel_e = build_panel_e
    R10.R9.R8.build_panel_g = build_panel_g
    R10.R9.R8.R7.build_panel_a = build_panel_a
    R10.R9.R8.R7.save_figure = R10.save_figure
    R10.R9.R8.R7.R6.PANEL_SIZE_IN = PANEL_SIZE_IN

    result = R10.R9.render(record_root, output_root, dpi)
    manifest = _update_metadata(output_root)
    result.update(
        {
            "review_round": 11,
            "layout_contract": manifest["layout_contract_round11"],
            "axis_alignment_contract": manifest["axis_alignment_contract_round11"],
            "panel_a_display_contract": manifest["panel_a_display_contract_round11"],
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
