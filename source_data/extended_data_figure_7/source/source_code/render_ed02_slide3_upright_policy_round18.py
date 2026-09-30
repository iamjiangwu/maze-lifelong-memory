"""Render ED2 round 18 with upright policy notation and refined annotations.

The accepted round-13 renderer remains the scientific and layout authority.
This append-only wrapper makes three display-only changes: every visible
turning-bias ``P`` label is upright and sized by its figure role, panel a uses
a two-line preceding-turn annotation beside the grey arrow, and the lower
panel-g contrast key is raised by another exact 10 pixels at 600 dpi.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.text import Text
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO_ROOT / "scripts" / "render_ed02_slide3_panel_g_labels_round13.py"
DEFAULT_RECORD_ROOT = (
    REPO_ROOT
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-11_round17_contrast_labels"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "ed02_round18_upright_policy_2026-08-11" / "panels"
PANEL_G_CONTRAST_ANCHOR = (0.58, 1.09737360)
PANEL_G_ADDITIONAL_RAISE_PX_600 = 10.0

UPRIGHT_TEXT = {
    r"$P_{SF}$": r"$\mathrm{P}_{\mathrm{SF}}$",
    r"$P_{SA}$": r"$\mathrm{P}_{\mathrm{SA}}$",
    r"$P_{BF}$": r"$\mathrm{P}_{\mathrm{BF}}$",
    r"$P_{BS}$": r"$\mathrm{P}_{\mathrm{BS}}$",
    r"$|\Delta P|$ from Return day": r"$|\Delta \mathrm{P}|$ from Return day",
    r"$\Delta P_{BS}$": r"$\Delta \mathrm{P}_{\mathrm{BS}}$",
    r"Depth-varying $P_{BS}$ model": r"Depth-varying $\mathrm{P}_{\mathrm{BS}}$ model",
    r"Depth $P_{BS}$": r"Depth $\mathrm{P}_{\mathrm{BS}}$",
}


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round13_for_round18", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
ROUND12 = BASE.BASE
ROUND11 = ROUND12.R11
ROUND10 = ROUND11.R10
PARENT_BUILD_PANEL_G = BASE.build_panel_g
PARENT_BUILD_PANEL_D = ROUND11.build_panel_d
PARENT_BUILD_PANEL_E = ROUND11.build_panel_e
PARENT_ROUND10_BUILD_PANEL_D = ROUND11.ROUND10_BUILD_PANEL_D
PARENT_ROUND10_BUILD_PANEL_E = ROUND11.ROUND10_BUILD_PANEL_E
PARENT_SAVE_FIGURE = ROUND10.save_figure


def build_panel_a(renderer: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Use the accepted 0.8x geometry with an upright, legible label policy."""

    text = deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_TEXT)
    display_title_map = {
        "stem_to_branch": r"$\mathrm{P}_{\mathrm{SA}}$",
        "stem_forward": r"$\mathrm{P}_{\mathrm{SF}}$",
        "branch_to_stem": r"$\mathrm{P}_{\mathrm{BS}}$",
        "branch_forward": r"$\mathrm{P}_{\mathrm{BF}}$",
    }
    source_title_map = {
        "stem_to_branch": r"$P_{SA}$",
        "stem_forward": r"$P_{SF}$",
        "branch_to_stem": r"$P_{BS}$",
        "branch_forward": r"$P_{BF}$",
    }
    for row in text["rows"]:
        row["title"] = display_title_map[row["key"]]
        if row["key"] == "stem_to_branch":
            row["annotations"] = [
                {
                    "text": "preceding\nturn",
                    "xy": (3.285, 3.82),
                    "ha": "right",
                    "va": "center",
                }
            ]
    fig, ax, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={
            "panel_size": ROUND11.PANEL_SIZE_IN["a"],
            "tight_layout": False,
            "xlim": (-0.35, 3.15),
            "ylim": (-0.49375, 4.44375),
        },
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 0.9 * ROUND12.SCHEMATIC_ARTIST_SCALE,
            "arrow_linewidth": 1.25 * ROUND12.SCHEMATIC_ARTIST_SCALE,
            "previous_arrow_linewidth": 1.25 * ROUND12.SCHEMATIC_ARTIST_SCALE,
            "arrow_mutation_scale": 7.5 * ROUND12.SCHEMATIC_ARTIST_SCALE,
            "title_fontsize": 6.2,
            "title_fontweight": "normal",
            "or_fontsize": 6.4 * ROUND12.SCHEMATIC_ARTIST_SCALE,
            "annotation_fontsize": 4.70,
            "annotation_lineheight": 1.05,
        },
        text=text,
        save=False,
    )
    ax.set_xlim(-0.35, 3.15)
    ax.set_ylim(-0.49375, 4.44375)
    fig.set_size_inches(*ROUND11.PANEL_SIZE_IN["a"], forward=True)
    rows = pd.DataFrame(
        [
            {
                "window_minutes": 30,
                "helper": "render_grid_route_choice_schematic",
                "row_key": row["key"],
                # Preserve the accepted plot-ready label text byte-for-byte.
                "policy_label": source_title_map[row["key"]],
            }
            for row in schematic["text"]["rows"]
        ]
    )
    return fig, {"schematic_rows": rows}


def build_panel_g(renderer, policy_rows, contrast_frame):
    fig, tables = PARENT_BUILD_PANEL_G(renderer, policy_rows, contrast_frame)
    if len(fig.axes) < 2:
        raise RuntimeError("Panel g no longer exposes its two maintained axes")
    lower_axis = fig.axes[1]
    legend = lower_axis.get_legend()
    if legend is None:
        raise RuntimeError("Panel-g contrast legend is missing")
    legend.set_bbox_to_anchor(PANEL_G_CONTRAST_ANCHOR, transform=lower_axis.transAxes)
    return fig, tables


def build_panel_d(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Normalize categorical policy labels before Round 11 hands off the figure."""

    fig, tables = PARENT_BUILD_PANEL_D(*args, **kwargs)
    _normalize_policy_text(fig, "d")
    return fig, tables


def build_panel_e(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Normalize categorical policy labels before Round 11 hands off the figure."""

    fig, tables = PARENT_BUILD_PANEL_E(*args, **kwargs)
    _normalize_policy_text(fig, "e")
    return fig, tables


def build_panel_d_source(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Normalize the Round-10 source before the Round-11 alignment wrapper."""

    fig, tables = PARENT_ROUND10_BUILD_PANEL_D(*args, **kwargs)
    _normalize_policy_text(fig, "d")
    return fig, tables


def build_panel_e_source(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Normalize the Round-10 source before the Round-11 alignment wrapper."""

    fig, tables = PARENT_ROUND10_BUILD_PANEL_E(*args, **kwargs)
    _normalize_policy_text(fig, "e")
    return fig, tables


def _normalize_policy_text(fig: Any, panel: str) -> None:
    """Apply upright notation and the main-figure role hierarchy before export."""

    for text in fig.findobj(match=Text):
        raw = text.get_text()
        replacement = UPRIGHT_TEXT.get(raw)
        if replacement is not None:
            text.set_text(replacement)

    if panel in {"b", "c"}:
        for ax in fig.axes:
            for label in (ax.xaxis.label, ax.yaxis.label):
                if "P" in label.get_text() or "mathrm{P}" in label.get_text():
                    label.set_fontsize(6.0)
    elif panel in {"d", "e"}:
        for ax in fig.axes:
            for axis_name, ticks, labels in (
                (
                    "x",
                    ax.get_xticks(),
                    [UPRIGHT_TEXT.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()],
                ),
                (
                    "y",
                    ax.get_yticks(),
                    [UPRIGHT_TEXT.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()],
                ),
            ):
                if any("\\mathrm{P}" in label for label in labels):
                    if axis_name == "x":
                        ax.set_xticks(ticks, labels)
                    else:
                        ax.set_yticks(ticks, labels)
            for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                if "P" in label.get_text() or "mathrm{P}" in label.get_text():
                    label.set_fontsize(5.4)
        if panel == "e":
            for ax in fig.axes:
                if "mathrm{P}" in ax.yaxis.label.get_text():
                    ax.yaxis.label.set_fontsize(6.0)
    elif panel == "g":
        for ax in fig.axes[:2]:
            if "mathrm{P}" in ax.yaxis.label.get_text():
                ax.yaxis.label.set_fontsize(6.0)
    elif panel == "h":
        for text in fig.findobj(match=Text):
            if "mathrm{P}" in text.get_text():
                text.set_fontsize(5.4)
    elif panel == "i":
        for text in fig.findobj(match=Text):
            if "mathrm{P}" in text.get_text():
                text.set_fontsize(6.2)


def save_figure(fig: Any, stem: Path, dpi: int) -> list[Path]:
    panel = stem.name[4]
    _normalize_policy_text(fig, panel)
    return PARENT_SAVE_FIGURE(fig, stem, dpi)


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    original_a = ROUND12.build_panel_a
    original_g = BASE.build_panel_g
    original_d = ROUND11.build_panel_d
    original_e = ROUND11.build_panel_e
    original_source_d = ROUND11.ROUND10_BUILD_PANEL_D
    original_source_e = ROUND11.ROUND10_BUILD_PANEL_E
    original_save = ROUND10.save_figure
    try:
        ROUND12.build_panel_a = build_panel_a
        BASE.build_panel_g = build_panel_g
        ROUND11.build_panel_d = build_panel_d
        ROUND11.build_panel_e = build_panel_e
        ROUND11.ROUND10_BUILD_PANEL_D = build_panel_d_source
        ROUND11.ROUND10_BUILD_PANEL_E = build_panel_e_source
        ROUND10.save_figure = save_figure
        result = BASE.render(record_root, output_root, dpi)
    finally:
        ROUND12.build_panel_a = original_a
        BASE.build_panel_g = original_g
        ROUND11.build_panel_d = original_d
        ROUND11.build_panel_e = original_e
        ROUND11.ROUND10_BUILD_PANEL_D = original_source_d
        ROUND11.ROUND10_BUILD_PANEL_E = original_source_e
        ROUND10.save_figure = original_save

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["review_round"] = 18
    manifest["status"] = "review_candidate_not_current_version_record"
    manifest["upright_policy_contract_round18"] = {
        "notation": "upright P and upright SF/SA/BF/BS subscripts in every visible role",
        "role_font_sizes_pt": {
            "panel_a_schematic": 6.2,
            "panel_b_c_axes": 6.0,
            "panel_d_e_categories": 5.4,
            "panel_e_ylabel": 6.0,
            "panel_g_ylabels": 6.0,
            "panel_h_direct_label": 5.4,
            "panel_i_model_title": 6.2,
        },
        "plot_ready_values_or_labels_changed": False,
    }
    manifest["panel_a_annotation_contract_round18"] = {
        "text": "preceding\\nturn",
        "xy": [3.285, 3.82],
        "horizontal_alignment": "right",
        "vertical_alignment": "center",
        "line_spacing": 1.05,
        "must_not_overlap_grid_or_arrow": True,
    }
    manifest["panel_a_display_contract_round12"]["style"]["title_fontsize"] = 6.2
    manifest["panel_a_display_contract_round12"]["style"]["annotation_fontsize"] = 4.70
    manifest["panel_a_display_contract_round12"]["preceding_turn"] = {
        "text": "preceding\nturn",
        "xy": [3.285, 3.82],
        "horizontal_alignment": "right",
        "vertical_alignment": "center",
        "line_spacing": 1.05,
        "must_not_overlap_grid_or_arrow": True,
    }
    manifest["panel_g_contrast_key_round18"] = {
        "previous_anchor_axes": [0.58, 1.06513460],
        "target_anchor_axes": list(PANEL_G_CONTRAST_ANCHOR),
        "additional_raise_px_at_600dpi": PANEL_G_ADDITIONAL_RAISE_PX_600,
        "upper_legend_changed": False,
        "axes_or_scientific_artists_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    caption_path = output_root / "CAPTION_NOTES.md"
    caption = caption_path.read_text(encoding="utf-8")
    caption = caption.replace(
        "# Extended Data Fig. 2 round-13 caption notes",
        "# Extended Data Fig. 2 round-18 caption notes",
        1,
    )
    for old, new in UPRIGHT_TEXT.items():
        caption = caption.replace(old, new)
    caption += (
        "\nRound-18 display note: all visible turning-bias symbols use upright P "
        "notation at main-figure-aligned role sizes. Panel a's preceding-turn label "
        "is split across two lines and placed beside, but clear of, the grey arrow. "
        "Panel g's lower contrast key is raised by another 10 pixels at 600 dpi. "
        "Scientific values, tables, axes, and panel frames are unchanged.\n"
    )
    caption_path.write_text(caption, encoding="utf-8")

    result["review_round"] = 18
    result["upright_policy_contract_round18"] = manifest[
        "upright_policy_contract_round18"
    ]
    result["panel_a_annotation_contract_round18"] = manifest[
        "panel_a_annotation_contract_round18"
    ]
    result["panel_a_display_contract"] = manifest["panel_a_display_contract_round12"]
    result["panel_g_contrast_key_round18"] = manifest[
        "panel_g_contrast_key_round18"
    ]
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
