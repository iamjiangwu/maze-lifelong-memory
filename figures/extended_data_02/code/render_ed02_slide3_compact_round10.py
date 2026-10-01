"""Render the compact ED2 slide-3 Round 10 review.

Round 10 preserves the Round 9 scientific tables and visible roles while
removing source-canvas whitespace that remained after the b/c legend became a
separate object.  The compared-experience observations stay hollow circles;
the b/c session-weighted means are opaque same-size color disks.
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
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND9_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_marker_consistent_round9.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-08_round3_slide_inventory"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tmp" / "ed02_round10_compact_2026-08-08" / "panels"
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

# Explicit source-level crop boxes in inches, measured from the unshifted
# Round-9 vector figures.  B/c share one box; f/i use their union box so paired
# axes remain physically identical.  The boxes include a 2.5-pt safety reserve.
CROP_BBOX_IN = {
    "a": (0.273611, 0.168611, 1.413056, 1.724722),
    "b": (0.000289, 0.198611, 2.653868, 1.509722),
    "c": (0.000289, 0.198611, 2.653868, 1.509722),
    "d": (0.255278, 0.0, 1.666667, 2.395833),
    "e": (0.0, 0.120331, 1.333333, 2.317169),
    "f": (0.066944, 0.147007, 4.166667, 2.410534),
    "g": (0.051944, 0.106993, 1.833333, 2.210466),
    "h": (0.0, 0.083652, 1.166667, 2.350514),
    "i": (0.066944, 0.147007, 4.166667, 2.410534),
}

PANEL_STEM = {
    "a": "ed02a_four_policy_schematic_round10",
    "b": "ed02b_naive_vs_return_policy_scatter_round10",
    "c": "ed02c_return_vs_second_day_policy_scatter_round10",
    "d": "ed02d_signed_policy_contributions_round10",
    "e": "ed02e_distance_from_return_round10",
    "f": "ed02f_four_policy_calibration_round10",
    "g": "ed02g_depth_pbs_profile_round10",
    "h": "ed02h_four_vs_depth_reconstruction_round10",
    "i": "ed02i_depth_calibration_round10",
}

SHARED_LEGEND_STEM = "ed02_shared_bc_experience_legend_round10"


def _load_round9() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round9_renderer", ROUND9_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROUND9_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R9 = _load_round9()
ROUND9_BUILD_PANEL_A = R9.R8.R7.build_panel_a
ROUND9_BUILD_POLICY_SCATTER = R9.build_policy_scatter
ROUND9_BUILD_PANEL_H = R9.build_panel_h


def build_panel_a(renderer: Any) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    return ROUND9_BUILD_PANEL_A(renderer)


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = ROUND9_BUILD_POLICY_SCATTER(
        renderer, points, contrasts, groups, panel
    )
    # Means are true opaque disks.  Individual sessions remain hollow with the
    # maintained 0.64 alpha inherited from the accepted renderer.
    for ax in fig.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            face = collection.get_facecolors()
            edge = collection.get_edgecolors()
            if len(face) and len(edge) and not np.allclose(face[0, :3], 1.0):
                collection.set_alpha(1.0)
                opaque = np.array(face, copy=True)
                opaque[:, 3] = 1.0
                collection.set_facecolors(opaque)
    return fig, tables


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    """Save one consistently cropped editable/vector/raster panel payload."""

    panel = stem.name[4]
    if panel not in CROP_BBOX_IN:
        raise KeyError(f"No Round-10 crop box for {stem.name}")
    crop = Bbox.from_extents(*CROP_BBOX_IN[panel])
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in R9.R8.R7.R6.R5.BASE.FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {
            "bbox_inches": crop,
            "pad_inches": 0,
            "facecolor": "white",
        }
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
        fig.savefig(
            ppt_path,
            bbox_inches=crop,
            pad_inches=0,
            facecolor="white",
        )
    plt.close(fig)
    return outputs


def build_panel_h(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    return ROUND9_BUILD_PANEL_H(renderer, frame)


def build_shared_bc_legend(output_root: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(196 / 96, 18 / 96), facecolor="none")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=R9.R8.GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in R9.R8.GROUP_ORDER
    ]
    fig.legend(
        handles,
        [R9.R8.GROUP_LABEL[group] for group in R9.R8.GROUP_ORDER],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        fontsize=5.4,
        handlelength=0.60,
        handletextpad=0.81,
        columnspacing=0.90,
        borderaxespad=0,
        frameon=False,
    )
    return R9.R8._save_shared_legend(fig, output_root, dpi)


def _update_metadata(output_root: Path) -> dict[str, Any]:
    panel_manifest_path = output_root / "PANEL_MANIFEST.csv"
    panel_manifest = pd.read_csv(panel_manifest_path)
    for panel, bounds in CROP_BBOX_IN.items():
        x0, y0, x1, y1 = bounds
        panel_manifest.loc[panel_manifest["panel"].eq(panel), "width_in"] = x1 - x0
        panel_manifest.loc[panel_manifest["panel"].eq(panel), "height_in"] = y1 - y0
    panel_manifest.to_csv(panel_manifest_path, index=False)

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.replace(
        "# Extended Data Fig. 2 round-9 caption notes",
        "# Extended Data Fig. 2 round-10 caption notes",
        1,
    )
    caption = caption.replace(
        "Larger symbols denote session-weighted group means;",
        "Solid symbols denote session-weighted group means; hollow symbols denote individual sessions;",
    )
    caption += (
        "\nRound-10 display note: the former b/c embedded-legend reserve and explicit "
        "inter-row bands were removed at source-render level. B/c means are opaque solid "
        "group-color disks with the same s=11 area as hollow individual-session circles. "
        "The shared b/c marker-to-text gap matches panel e; a-i letters are moved "
        "approximately one 10.5-pt character lower than Round 9 (panel d uses a "
        "9-pt shift to preserve ink clearance). Scientific values and roles are "
        "unchanged from the Round-9 review.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 10
    manifest.pop("layout_contract_round9", None)
    manifest["presentation_changes"] = [
        "removed the obsolete b/c embedded-legend source-canvas reserve",
        "removed explicit inter-row bands and used the reclaimed height in row panels",
        "moved a-i letters about one 10.5-pt character lower than Round 9; d uses 9 pt for ink clearance",
        "matched the shared b/c marker-to-text gap to panel e",
        "made b/c same-size mean disks fully opaque",
    ]
    manifest["marker_contract"]["aggregate_exception"].update(
        {"fill": "opaque solid maintained group color", "alpha": 1.0}
    )
    manifest["shared_bc_legend"] = {
        "editable_asset": f"shared_assets/{SHARED_LEGEND_STEM}.svg",
        "powerpoint_asset": f"ppt_vector/{SHARED_LEGEND_STEM}.svg",
        "frame_720": {
            "left": 337,
            "top": 57,
            "width": 196,
            "height": 18,
        },
        "object_name": "ED2 shared b-c experience legend",
        "entry_spacing": {"handletextpad": 0.81, "columnspacing": 0.90},
    }
    manifest["layout_contract_round10"] = {
        "panel_frames_720": {
            "a": {"left": 2, "top": 20.178340, "width": 148, "height": 202.119941},
            "b": {"left": 152, "top": 80, "width": 282, "height": 139.333824},
            "c": {"left": 436, "top": 80, "width": 282, "height": 139.333824},
            "d": {"left": 2, "top": 224.298282, "width": 139.476654, "height": 236.761689},
            "e": {"left": 143.476654, "top": 224.298282, "width": 143.698408, "height": 236.761689},
            "f": {"left": 289.175063, "top": 224.298282, "width": 428.824937, "height": 236.761689},
            "g": {"left": 2, "top": 497.697222, "width": 171.175063, "height": 202.124437},
            "h": {"left": 175.175063, "top": 482.202860, "width": 112, "height": 217.618800},
            "i": {"left": 289.175063, "top": 463.059971, "width": 428.824937, "height": 236.761689},
        },
        "panel_label_frames_720": {
            "a": {"left": 4, "top": 38, "width": 18, "height": 16},
            "b": {"left": 154, "top": 38, "width": 18, "height": 16},
            "c": {"left": 438, "top": 38, "width": 18, "height": 16},
            "d": {"left": 4, "top": 239, "width": 18, "height": 16},
            "e": {"left": 145.476654, "top": 240, "width": 18, "height": 16},
            "f": {"left": 291.175063, "top": 240, "width": 18, "height": 16},
            "g": {"left": 4, "top": 484, "width": 18, "height": 16},
            "h": {"left": 177.175063, "top": 484, "width": 18, "height": 16},
            "i": {"left": 291.175063, "top": 484, "width": 18, "height": 16},
        },
        "row_regions_720": {
            "row1": {"top": 20.178340, "bottom": 222.298281},
            "row2": {"top": 224.298282, "bottom": 461.059971},
            "row3": {"top": 463.059971, "bottom": 699.821660},
        },
        "frame_gaps": [2, 2],
        "crop_mode": "source-level explicit Bbox; PowerPoint crop remains zero",
    }
    manifest["source_crop_contract"] = {
        panel: {
            "bbox_inches_bottom_left": list(bounds),
            "width_in": bounds[2] - bounds[0],
            "height_in": bounds[3] - bounds[1],
        }
        for panel, bounds in CROP_BBOX_IN.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    R9.PANEL_SIZE_IN = PANEL_SIZE_IN
    R9.PANEL_STEM = PANEL_STEM
    R9.SHARED_LEGEND_STEM = SHARED_LEGEND_STEM
    R9.build_policy_scatter = build_policy_scatter
    R9.build_shared_bc_legend = build_shared_bc_legend
    R9.build_panel_h = build_panel_h
    R9.R8.R7.build_panel_a = build_panel_a
    R9.R8.R7.save_figure = save_figure
    R9.R8.R7.R6.PANEL_SIZE_IN = PANEL_SIZE_IN

    result = R9.render(record_root, output_root, dpi)
    manifest = _update_metadata(output_root)
    result.update(
        {
            "review_round": 10,
            "layout_contract": manifest["layout_contract_round10"],
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
