"""Render the append-only Extended Data Figure 2 round-23 refinement candidate.

The round-18 renderer remains the scientific authority.  This wrapper changes
only annotation positions, panel-e spine length, and calibration-title spacing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


REPO = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO / "scripts" / "render_ed02_slide3_upright_policy_round18.py"
DEFAULT_RECORD_ROOT = (
    REPO
    / "docs"
    / "figure_version_records"
    / "extended_data_fig02_2026-08-11_round20_upright_policy_clearance"
)
DEFAULT_OUTPUT_ROOT = (
    REPO
    / "docs"
    / "figure_refinement_round2_2026-08-11"
    / "extended_data_1_5"
    / "ed02"
)
PANEL_A_ANNOTATION_BASE_X = 3.42
# Preserve the requested full two-character left shift while clearing the grey
# arrow; at the previous y=3.51 the shifted two-line text intersected it.
PANEL_A_ANNOTATION_Y = 3.80
PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE = "nn"
PANEL_A_CROP_SHIFT_IN = 0.05
PANEL_G_UPPER_ANCHOR = (0.52, 1.30598)
PANEL_G_LOWER_ANCHOR = (0.52, 1.34090)
MODEL_TITLE_Y = 0.94
PANEL_G_TOP_RESERVE_IN = 0.14
REVIEW_ROUND = 23


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("ed02_round18_for_round21", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R18 = _load_base()
R7 = R18.ROUND10.R9.R8.R7
PARENT_BUILD_A = R18.build_panel_a
PARENT_BUILD_G = R18.PARENT_BUILD_PANEL_G
PARENT_BUILD_E = R18.build_panel_e
PARENT_BUILD_E_SOURCE = R18.build_panel_e_source
PARENT_CALIBRATION = R7.build_calibration
LAST_A_GEOMETRY: dict[str, Any] = {}
LAST_G_SAVE_GEOMETRY: dict[str, Any] = {}
LAST_E_GEOMETRY: dict[str, Any] = {}


def build_panel_a(renderer: Any):
    global LAST_A_GEOMETRY
    fig, tables = PARENT_BUILD_A(renderer)
    matches = [text for text in fig.axes[0].texts if text.get_text() == "preceding\nturn"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one preceding-turn annotation, found {len(matches)}")
    annotation = matches[0]
    annotation.set_position((PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y))
    fig.canvas.draw()
    renderer_px = fig.canvas.get_renderer()
    two_character_width_px = float(
        renderer_px.get_text_width_height_descent(
            PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE,
            annotation.get_fontproperties(),
            ismath=False,
        )[0]
    )
    axis = fig.axes[0]
    base_display = axis.transData.transform(
        (PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y)
    )
    shifted_data = axis.transData.inverted().transform(
        (base_display[0] - two_character_width_px, base_display[1])
    )
    annotation.set_position((float(shifted_data[0]), PANEL_A_ANNOTATION_Y))
    fig.canvas.draw()
    renderer_px = fig.canvas.get_renderer()
    text_box = annotation.get_window_extent(renderer_px)
    grey_arrows = [
        patch
        for patch in fig.axes[0].patches
        if patch.__class__.__name__ == "FancyArrowPatch"
        and all(abs(channel - 0.6549019608) < 1e-6 for channel in patch.get_edgecolor()[:3])
    ]
    boxes = [
        patch
        for patch in fig.axes[0].patches
        if patch.__class__.__name__ == "Rectangle"
    ]
    if len(grey_arrows) != 1 or not boxes:
        raise RuntimeError("Panel-a arrow/box artist contract changed")

    def clearance(first: Any, second: Any) -> float:
        dx = max(second.x0 - first.x1, first.x0 - second.x1, 0.0)
        dy = max(second.y0 - first.y1, first.y0 - second.y1, 0.0)
        return math.hypot(dx, dy)

    box_clearances = [
        (clearance(text_box, box.get_extents()), list(box.get_extents().bounds))
        for box in boxes
    ]
    nearest_box_clearance, nearest_box_bbox = min(box_clearances, key=lambda row: row[0])
    LAST_A_GEOMETRY = {
        "base_xy_data": [PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y],
        "final_xy_data": [float(shifted_data[0]), PANEL_A_ANNOTATION_Y],
        "two_character_sample": PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE,
        "two_character_shift_px": two_character_width_px,
        "actual_left_shift_px": float(base_display[0] - axis.transData.transform(shifted_data)[0]),
        "text_bbox_px": list(text_box.bounds),
        "grey_arrow_bbox_px": list(grey_arrows[0].get_extents().bounds),
        "grey_arrow_clearance_px": clearance(text_box, grey_arrows[0].get_extents()),
        "box_clearance_px": nearest_box_clearance,
        "nearest_box_bbox_px": nearest_box_bbox,
    }
    if LAST_A_GEOMETRY["grey_arrow_clearance_px"] < 1.0 or LAST_A_GEOMETRY["box_clearance_px"] < 1.0:
        raise RuntimeError(f"Panel-a annotation clearance failed: {LAST_A_GEOMETRY}")
    # Keep each alternative label inside its middle grid cell, clear of arrows.
    # SA moves left at the same height; the other three follow their diagonal.
    or_positions = {
        "stem_to_branch": ((1.5096, 2.99), (1.4751335, 2.99)),
        "stem_forward": ((1.2124, 2.7928), (1.19268, 2.81252)),
        "branch_to_stem": ((1.8228, 0.718), (1.76712, 0.77368)),
        "branch_forward": ((1.166, 0.7064), (1.21298, 0.65942)),
    }
    or_labels = [text for text in axis.texts if text.get_text() == "or"]
    if len(or_labels) != 4:
        raise RuntimeError(f"Expected four alternative labels, found {len(or_labels)}")
    LAST_A_GEOMETRY["or_label_positions"] = []
    for key, (before, after) in or_positions.items():
        matches = [text for text in or_labels if all(
            math.isclose(value, expected, abs_tol=1e-8)
            for value, expected in zip(text.get_position(), before)
        )]
        if len(matches) != 1:
            raise RuntimeError(f"Panel-a alternative-label anchor changed for {key}")
        matches[0].set_position(after)
        LAST_A_GEOMETRY["or_label_positions"].append({
            "key": key, "before_xy_data": list(before), "after_xy_data": list(after),
        })
    return fig, tables


def build_panel_g(renderer: Any, policy_rows: Any, contrast_frame: Any):
    fig, tables = PARENT_BUILD_G(renderer, policy_rows, contrast_frame)
    if len(fig.axes) < 2:
        raise RuntimeError("Panel g no longer exposes its two maintained axes")
    upper = fig.axes[0].get_legend()
    lower = fig.axes[1].get_legend()
    if upper is None or lower is None:
        raise RuntimeError("Panel-g direct label groups are missing")
    upper.set_bbox_to_anchor(PANEL_G_UPPER_ANCHOR, transform=fig.axes[0].transAxes)
    lower.set_bbox_to_anchor(PANEL_G_LOWER_ANCHOR, transform=fig.axes[1].transAxes)
    return fig, tables


def save_figure(fig: Any, stem: Path, dpi: int):
    global LAST_G_SAVE_GEOMETRY
    panel = stem.name[4]
    R18._normalize_policy_text(fig, panel)
    if panel == "g":
        old_width, old_height = fig.get_size_inches()
        positions = [axis.get_position().frozen() for axis in fig.axes]
        new_height = old_height + PANEL_G_TOP_RESERVE_IN
        fig.set_size_inches(old_width, new_height, forward=True)
        for axis, position in zip(fig.axes, positions, strict=True):
            axis.set_position(
                (
                    position.x0,
                    position.y0 * old_height / new_height,
                    position.width,
                    position.height * old_height / new_height,
                )
            )
        fig.canvas.draw()
        rendered = fig.canvas.get_renderer()
        upper = fig.axes[0].get_legend()
        upper_bbox_in = [
            value / fig.dpi for value in upper.get_window_extent(rendered).bounds
        ]
        crop = R18.ROUND10.CROP_BBOX_IN["g"]
        crop_top = max(crop[3], upper_bbox_in[1] + upper_bbox_in[3] + 3.0 / 72)
        R18.ROUND10.CROP_BBOX_IN["g"] = (crop[0], crop[1], crop[2], crop_top)
        LAST_G_SAVE_GEOMETRY = {
            "figure_size_in": list(fig.get_size_inches()),
            "upper_legend_bbox_in": upper_bbox_in,
            "crop_bbox_in": list(R18.ROUND10.CROP_BBOX_IN["g"]),
            "upper_text_top_clearance_pt": (
                crop_top - upper_bbox_in[1] - upper_bbox_in[3]
            ) * 72,
        }
    return R18.PARENT_SAVE_FIGURE(fig, stem, dpi)


def _shorten_panel_e_spine(fig: Any) -> None:
    global LAST_E_GEOMETRY
    if not fig.axes:
        raise RuntimeError("Panel e has no maintained axis")
    axis = fig.axes[0]
    finite_y: list[float] = []
    for collection in axis.collections:
        offsets = collection.get_offsets()
        if getattr(offsets, "shape", (0, 0))[1:] == (2,):
            finite_y.extend(float(value) for value in offsets[:, 1] if math.isfinite(value))
    for line in axis.lines:
        finite_y.extend(float(value) for value in line.get_ydata() if math.isfinite(value))
    if not finite_y:
        raise RuntimeError("Panel e exposes no finite data artists")
    data_max = max(finite_y)
    low, high = axis.get_ylim()
    span = high - low
    spine_top = min(high - 0.04 * span, data_max + 0.04 * span)
    if not data_max < spine_top < high:
        raise RuntimeError(
            f"Cannot place panel-e spine above data and below annotations: "
            f"data_max={data_max}, ylim={(low, high)}"
        )
    axis.spines["left"].set_bounds(low, spine_top)
    LAST_E_GEOMETRY = {
        "data_max": data_max,
        "ylim": [low, high],
        "left_spine_bounds": [low, spine_top],
        "data_to_spine_clearance": spine_top - data_max,
        "spine_to_annotation_band_clearance": high - spine_top,
    }


def build_panel_e(*args: Any, **kwargs: Any):
    fig, tables = PARENT_BUILD_E(*args, **kwargs)
    _shorten_panel_e_spine(fig)
    return fig, tables


def build_panel_e_source(*args: Any, **kwargs: Any):
    fig, tables = PARENT_BUILD_E_SOURCE(*args, **kwargs)
    _shorten_panel_e_spine(fig)
    return fig, tables


def build_calibration(renderer: Any, session_frame: Any, summary_frame: Any, model: str, panel: str):
    fig, tables = PARENT_CALIBRATION(renderer, session_frame, summary_frame, model, panel)
    if fig._suptitle is None:
        raise RuntimeError("Calibration model title is missing")
    fig._suptitle.set_y(MODEL_TITLE_Y)
    # Match b/c: one opaque, session-weighted coordinate mean per group.
    simulated = "four_own_half_cover_eff" if model == "four_bias" else "depth_own_half_cover_eff"
    points = tables["session_points"]
    means = []
    for axis, group in zip(fig.axes, renderer.GROUP_ORDER):
        rows = points.loc[points["group"].eq(group)]
        x, y = float(rows["real_half_cover_eff"].mean()), float(rows[simulated].mean())
        color = renderer.GROUP_COLOR[group]
        marker = axis.scatter([x], [y], s=11, marker="o", facecolors=color,
                              edgecolors=color, linewidths=0.75, alpha=1.0, zorder=4)
        marker.set_gid(f"ed02_{panel}_{group}_session_mean")
        means.append({"panel": panel, "model": model, "group": group,
                      "n_session_rows": len(rows), "n_mice": rows["animal_id"].nunique(),
                      "observed_mean": x, "simulated_mean": y,
                      "x_column": "real_half_cover_eff", "y_column": simulated,
                      "marker_area_pt2": 11, "alpha": 1.0})
    tables["display_group_means"] = renderer.pd.DataFrame(means)
    return fig, tables


def render(record_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    original_a = R18.build_panel_a
    original_g = R18.build_panel_g
    original_e = R18.build_panel_e
    original_e_source = R18.build_panel_e_source
    original_calibration = R7.build_calibration
    original_save = R18.save_figure
    original_crop = dict(R18.ROUND10.CROP_BBOX_IN)
    try:
        R18.build_panel_a = build_panel_a
        R18.build_panel_g = build_panel_g
        R18.build_panel_e = build_panel_e
        R18.build_panel_e_source = build_panel_e_source
        R7.build_calibration = build_calibration
        R18.save_figure = save_figure
        a_crop = original_crop["a"]
        R18.ROUND10.CROP_BBOX_IN["a"] = (
            a_crop[0] + PANEL_A_CROP_SHIFT_IN,
            a_crop[1],
            a_crop[2] + PANEL_A_CROP_SHIFT_IN,
            a_crop[3],
        )
        g_crop = original_crop["g"]
        R18.ROUND10.CROP_BBOX_IN["g"] = (
            g_crop[0],
            g_crop[1],
            g_crop[2],
            g_crop[3],
        )
        result = R18.render(record_root, output_root, dpi)
    finally:
        R18.build_panel_a = original_a
        R18.build_panel_g = original_g
        R18.build_panel_e = original_e
        R18.build_panel_e_source = original_e_source
        R7.build_calibration = original_calibration
        R18.save_figure = original_save
        R18.ROUND10.CROP_BBOX_IN.clear()
        R18.ROUND10.CROP_BBOX_IN.update(original_crop)

    manifest_path = output_root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    g_frame = manifest["layout_contract_round11"]["panel_frames_720"]["g"]
    g_bottom = float(g_frame["top"]) + float(g_frame["height"])
    g_crop = LAST_G_SAVE_GEOMETRY["crop_bbox_in"]
    source_aspect = (g_crop[2] - g_crop[0]) / (g_crop[3] - g_crop[1])
    g_frame["height"] = float(g_frame["width"]) / source_aspect
    g_frame["top"] = g_bottom - float(g_frame["height"])
    manifest["review_round"] = REVIEW_ROUND
    manifest["status"] = "working_candidate_not_current_version_record"
    manifest["round23_display_only_changes"] = {
        "panel_a_preceding_turn_base_xy": [
            PANEL_A_ANNOTATION_BASE_X,
            PANEL_A_ANNOTATION_Y,
        ],
        "panel_a_preceding_turn_final_xy": LAST_A_GEOMETRY["final_xy_data"],
        "panel_a_preceding_turn_left_shift_basis": (
            "two rendered lowercase n glyph widths at the annotation font"
        ),
        "panel_a_crop_shift_right_in": PANEL_A_CROP_SHIFT_IN,
        "panel_a_geometry_at_source_dpi": dict(LAST_A_GEOMETRY),
        "panel_g_upper_anchor_axes": list(PANEL_G_UPPER_ANCHOR),
        "panel_g_lower_anchor_axes": list(PANEL_G_LOWER_ANCHOR),
        "panel_g_upper_lower_circle_x_aligned": (
            PANEL_G_UPPER_ANCHOR[0] == PANEL_G_LOWER_ANCHOR[0]
        ),
        "panel_g_added_figure_top_reserve_pt": PANEL_G_TOP_RESERVE_IN * 72,
        "panel_g_save_geometry": dict(LAST_G_SAVE_GEOMETRY),
        "panel_g_composite_frame_720": dict(g_frame),
        "panel_e_geometry": dict(LAST_E_GEOMETRY),
        "panel_f_i_model_title_y": MODEL_TITLE_Y,
        "scientific_values_or_tables_changed": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result["review_round"] = REVIEW_ROUND
    result["round23_display_only_changes"] = manifest["round23_display_only_changes"]
    (output_root / "RENDER_RESULT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(render(args.record_root, args.output_root, args.dpi), indent=2))


if __name__ == "__main__":
    main()
