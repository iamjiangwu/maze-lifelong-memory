"""Render ED2 with restored panel-g depth labels and a raised g panel letter.

Round 13 preserves the accepted Round-12 scientific data, estimands, axes,
frames, source crops, typography, colors, markers, and figure size.  It fixes
one shared-x display regression: the compression pass had replaced the bottom
depth-category labels with the intentionally blank upper-axis labels.  The
external panel letter ``g`` also moves upward by exactly one 10.5-pt character.
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
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASE = Path(__file__).with_name("render_ed02_slide3_schematic_g_fix_round12.py")
BASE_RENDERER = (
    LOCAL_BASE
    if LOCAL_BASE.is_file() and LOCAL_BASE.resolve() != Path(__file__).resolve()
    else REPO_ROOT / "scripts" / "render_ed02_slide3_schematic_g_fix_round12.py"
)
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-09_round14_schematic_g_fix_vor"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round13_g_labels_2026-08-09" / "panels"
DEPTH_LABELS = ("Shallow\n1-2", "Middle\n3-4", "Deep\n5")
PANEL_G_LABEL_TOP_720 = 463.0


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round12_renderer", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _compress_artist_x(ax: plt.Axes, *, center: float, scale: float) -> None:
    """Compress data artists only; do not mutate shared tick labels."""

    transform = lambda values: center + (np.asarray(values, dtype=float) - center) * scale
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


def build_panel_g(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Apply the accepted compression once and restore bottom depth labels."""

    fig, tables = BASE.ROUND11_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    ax, ax_delta = fig.axes[:2]
    original_ticks = np.asarray(ax_delta.get_xticks(), dtype=float)
    if len(original_ticks) != 3:
        raise RuntimeError(f"Expected three panel-g depth ticks, found {original_ticks}")
    for target in (ax, ax_delta):
        _compress_artist_x(
            target,
            center=1.0,
            scale=BASE.PANEL_G_GROUP_SCALE,
        )

    compressed_ticks = 1.0 + (original_ticks - 1.0) * BASE.PANEL_G_GROUP_SCALE
    # Shared axes reuse locator/formatter state.  Set the authoritative labels
    # on the bottom axis once, then hide only the upper-axis label artists.
    ax_delta.set_xticks(compressed_ticks, DEPTH_LABELS)
    ax.tick_params(axis="x", labelbottom=False)
    ax_delta.tick_params(axis="x", labelbottom=True)

    fig.subplots_adjust(
        left=BASE.PANEL_G_LEFT,
        right=0.98,
        bottom=0.256662900,
        top=0.89,
        hspace=0.44,
    )
    for target in (ax, ax_delta):
        target.yaxis.set_label_coords(BASE.PANEL_G_YLABEL_X, 0.5)
    return fig, tables


def _update_metadata(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "review_candidate_not_current_version_record"
    manifest["review_round"] = 13
    manifest["presentation_changes"] = list(manifest.get("presentation_changes", [])) + [
        "restored panel-g shallow/middle/deep categorical x labels erased by the shared-x compression pass",
        "moved only the external panel-g letter upward by exactly one 10.5-pt character height",
    ]
    labels = manifest["layout_contract_round11"]["panel_label_frames_720"]
    previous_top = float(labels["g"]["top"])
    labels["g"]["top"] = PANEL_G_LABEL_TOP_720
    manifest["panel_g_label_contract_round13"] = {
        "labels": list(DEPTH_LABELS),
        "group_positions": [0.28, 1.0, 1.72],
        "upper_axis_labels_hidden": True,
        "bottom_axis_labels_visible": True,
        "source_values_changed": False,
    }
    manifest["panel_g_letter_contract_round13"] = {
        "previous_top_720": previous_top,
        "new_top_720": PANEL_G_LABEL_TOP_720,
        "delta_720": PANEL_G_LABEL_TOP_720 - previous_top,
        "font_size_pt": 10.5,
        "other_panel_letters_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.replace(
        "# Extended Data Fig. 2 round-12 caption notes",
        "# Extended Data Fig. 2 round-13 caption notes",
        1,
    )
    caption += (
        "\nRound-13 display note: panel g again shows the maintained depth categories "
        "Shallow 1-2, Middle 3-4, and Deep 5. The source values, axes, frame, and "
        "crop are unchanged. Only the external panel letter g moves upward by 10.5 pt.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")
    return manifest


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    original_g = BASE.build_panel_g
    try:
        BASE.build_panel_g = build_panel_g
        result = BASE.render(record_root, output_root, dpi)
    finally:
        BASE.build_panel_g = original_g
    manifest = _update_metadata(output_root)
    result.update(
        {
            "review_round": 13,
            "layout_contract": manifest["layout_contract_round11"],
            "panel_g_label_contract": manifest["panel_g_label_contract_round13"],
            "panel_g_letter_contract": manifest["panel_g_letter_contract_round13"],
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
