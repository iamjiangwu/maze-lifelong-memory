"""Render the ED2 slide-3 round-6 visible-ink-aligned review candidate.

The accepted round-3 record remains the scientific authority and the supplied
PowerPoint remains the inventory authority (exactly panels a-i).  This review
pass changes destination aspect ratios, whitespace, subplot margins, and
support-panel label placement only.  Values, limits, uncertainty, cohorts,
estimands, colors, and panel roles are unchanged.
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
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND5_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_layout_optimized_round5.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round6_layout_2026-08-08" / "panels"


# Exact destination frames at 96 px/in.  The dense calibration pair receives
# the right-hand 400-px hero column; e/h are compact supporting summaries.
PANEL_SIZE_IN = {
    "a": (148 / 96, 192 / 96),
    "b": (258 / 96, 192 / 96),
    "c": (258 / 96, 192 / 96),
    "d": (160 / 96, 224 / 96),
    "e": (128 / 96, 224 / 96),
    "f": (400 / 96, 224 / 96),
    "g": (176 / 96, 224 / 96),
    "h": (112 / 96, 224 / 96),
    "i": (400 / 96, 224 / 96),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_visible_alignment_round6",
    "b": "ed02b_naive_vs_return_policy_scatter_visible_alignment_round6",
    "c": "ed02c_return_vs_second_day_policy_scatter_visible_alignment_round6",
    "d": "ed02d_four_policy_contributions_visible_alignment_round6",
    "e": "ed02e_distance_from_return_visible_alignment_round6",
    "f": "ed02f_four_policy_calibration_visible_alignment_round6",
    "g": "ed02g_depth_pbs_profile_visible_alignment_round6",
    "h": "ed02h_four_vs_depth_reconstruction_visible_alignment_round6",
    "i": "ed02i_depth_calibration_visible_alignment_round6",
}


def load_round5() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round5_renderer", ROUND5_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND5_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R5 = load_round5()
ROUND5_BUILD_PANEL_D = R5.build_panel_d
ROUND5_BUILD_PANEL_E = R5.build_panel_e
ROUND5_BUILD_CALIBRATION = R5.build_calibration
ROUND5_BUILD_PANEL_G = R5.build_panel_g
ROUND4_BUILD_PANEL_A = R5.BASE.build_panel_a
ROUND4_BUILD_POLICY_SCATTER = R5.BASE.build_policy_scatter
ROUND4_BUILD_PANEL_H = R5.BASE.build_panel_h


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND4_BUILD_PANEL_A(renderer)
    shift = 4.0 / 192.0
    for ax in fig.axes:
        position = ax.get_position()
        ax.set_position(
            [position.x0, position.y0 - shift, position.width, position.height]
        )
    return fig, tables


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND4_BUILD_POLICY_SCATTER(
        renderer, points, contrasts, groups, panel
    )
    # Use the top-row canvas more fully so the next row does not appear
    # detached; values, axes, limits, and square aspect remain unchanged.
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.13, top=0.80, wspace=0.34)
    return fig, tables


def build_panel_d(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND5_BUILD_PANEL_D(renderer, frame)
    # Lower the directional key and extend the plot toward the x label so the
    # complete visible panel aligns with f rather than floating above it.
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.16, top=0.94)
    return fig, tables


def build_panel_e(renderer: Any, rows: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND5_BUILD_PANEL_E(renderer, rows)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    labels = ["Day 2" if label == "Second day" else label for label in labels]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.015),
        ncol=2,
        handletextpad=0.12,
        columnspacing=0.28,
        borderaxespad=0,
    )
    ax.set_xlim(-0.30, 3.30)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.12, top=0.94)
    return fig, tables


def build_calibration(
    renderer: Any,
    session_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    model: str,
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND5_BUILD_CALIBRATION(
        renderer, session_frame, summary_frame, model, panel
    )
    for ax in fig.axes:
        ax.set_anchor("N")
    if fig._supxlabel is not None:
        fig._supxlabel.set_y(0.060)
    if fig._supylabel is not None:
        fig._supylabel.set_x(0.010)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.90, wspace=0.12)
    return fig, tables


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND5_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    # The legends remain inside the same scientific roles, but the complete
    # two-axis panel is lowered and extended to match i's visible ink span.
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.15, top=0.88, hspace=0.43)
    return fig, tables


def build_panel_h(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND4_BUILD_PANEL_H(renderer, frame)
    ax = fig.axes[0]
    # Direct labels recover plotting width in the narrow support slot without
    # removing either model identity or changing the plotted estimates.
    ax.set_yticks([])
    ax.text(
        0.006,
        1,
        "Four policy",
        ha="left",
        va="center",
        fontsize=5.0,
        color=renderer.INK,
    )
    ax.text(
        0.006,
        0,
        r"Depth $P_{BS}$",
        ha="left",
        va="center",
        fontsize=5.0,
        color=renderer.INK,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.94)
    return fig, tables


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    R5.PANEL_SIZE_IN = PANEL_SIZE_IN
    R5.PANEL_STEM = PANEL_STEM
    R5.build_panel_d = build_panel_d
    R5.build_panel_e = build_panel_e
    R5.build_calibration = build_calibration
    R5.build_panel_g = build_panel_g
    R5.BASE.build_panel_a = build_panel_a
    R5.BASE.build_policy_scatter = build_policy_scatter
    R5.BASE.build_panel_h = build_panel_h
    result = R5.render(record_root, output_root, dpi)

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8").replace(
        "# Extended Data Fig. 2 round-5 caption notes",
        "# Extended Data Fig. 2 round-6 caption notes",
        1,
    )
    caption_path.write_text(caption, encoding="utf-8")

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 6
    manifest["presentation_changes"] = [
        "larger matched calibration panels f and i",
        "narrower compact support panels e and h",
        "reduced inter-row and bottom whitespace",
        "row-2 and row-3 visible-ink alignment",
        "direct in-axis model labels in h to preserve plotting width",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    result["review_round"] = 6
    result["layout_contract"] = "visible-ink-aligned rows; f/i hero pair"
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
