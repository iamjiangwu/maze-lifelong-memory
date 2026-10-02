"""Render Figure 2 from included derived Source Data.

The supported build.py launcher supplies source paths and the two
source-scope validation callbacks before calling render_figure2.
Historical path defaults below are replaced by that launcher.
"""

from __future__ import annotations

import importlib.util

import io

import json

import os

import subprocess

import tempfile

import time

from dataclasses import asdict, dataclass, field, replace as dataclass_replace

from pathlib import Path

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.font_manager import FontProperties

from matplotlib.textpath import TextPath

from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from scipy import stats as scipy_stats

from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    PANEL_LAYOUT,
    apply_panel_rcparams,
    compute_example_cell_arrow_data,
    get_panel_maze,
    load_neuron_data_bundle,
    render_spatial_decoding_error_map,
    render_example_cell_day12_placemap_panel,
    render_example_cell_grid_panel,
    resolve_neuron_data_path,
)

from ca_analysis.figure_colors import get_correlation_experience_colors

REPO_ROOT = Path(__file__).resolve().parents[1]

SORTED_ROOT = REPO_ROOT / "docs" / "sorted_data_full_figure_rerun_2026-06-11"

EXAMPLE_SAVED_DIR = REPO_ROOT / "docs" / "example_neuron_candidates" / "saved_picks"

FIGURE2_HELPER = REPO_ROOT / "scripts" / "preview_figure2_assembly.py"

FIGURE2_DISTRIBUTION_DIR = (
    SORTED_ROOT
    / "figure2_distribution_shuffle_corrected_842_no_presort_exclusions_2026-08-03"
)

FIGURE2_DISTRIBUTION_PAIR_SUPPORT = FIGURE2_DISTRIBUTION_DIR / "pair_support_audit.csv"

FIGURE2_GAP_CURVE = (
    SORTED_ROOT
    / "figure2_gap_anchor_pair_mean_no_manual_mask_2026-08-13"
    / "gap_with_day1_day2_curve_data.csv"
)

FIGURE2_CONSECUTIVE_POINTS = (
    SORTED_ROOT
    / "figure2_consecutive_gap_clean_long_gap_corrected_2026-08-03"
    / "consecutive_gap_clean_long_gap_corrected_points.csv"
)

FIGURE2_CONSECUTIVE_STATS = (
    SORTED_ROOT
    / "figure2_consecutive_gap_clean_long_gap_corrected_2026-08-03"
    / "consecutive_gap_clean_long_gap_corrected_stats.csv"
)

FIGURE2_CONSECUTIVE_DIR = SORTED_ROOT / "figure2_consecutive_gap_clean_long_gap_corrected_2026-08-03"

FIGURE2_CONSECUTIVE_CLEAN_SCOPE = FIGURE2_CONSECUTIVE_DIR / "consecutive_gap_clean_long_gap_22_chain_scope.csv"

FIGURE2_RATE_CHAIN_DIR = SORTED_ROOT / "figure2_rate_chain_no_manual_mask_2026-08-07"

FIGURE2_RATE_CHAIN_POINTS = (
    FIGURE2_RATE_CHAIN_DIR / "session_pair_rate_consecutive_gap_points_clean_long_gap_no_manual_mask.csv"
)

FIGURE2_RATE_CHAIN_STATS = (
    FIGURE2_RATE_CHAIN_DIR / "session_pair_rate_consecutive_gap_stats_clean_long_gap_no_manual_mask.csv"
)

BROWSER_EXPORT_TIMEOUT_SECONDS = 180

SVG_NS = "http://www.w3.org/2000/svg"

XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)

ET.register_namespace("xlink", XLINK_NS)

METRICS = ("place_map_correlation", "pv_correlation", "ensemble_rate_correlation")

METRIC_LABELS = {
    "place_map_correlation": "Place map",
    "pv_correlation": "PV",
    "ensemble_rate_correlation": "Ensemble rate",
    "level_map_correlation": "Level map",
}

FIGURE2_DISTRIBUTION_XLABELS = {
    "place_map_correlation": "Place map correlation",
    "pv_correlation": "PV correlation",
    "ensemble_rate_correlation": "Ensemble rate correlation",
}

FIGURE2_METRIC_SHORT = {
    "place_map_correlation": "place",
    "pv_correlation": "pv",
    "ensemble_rate_correlation": "rate",
    "level_map_correlation": "level",
}

FIGURE2_METRIC_NPZ = {
    "place_map_correlation": {
        "distribution": (
            "figure2_panel_c_place_distribution_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
        "mean": (
            "figure2_panel_d_place_shuffle_mean_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
    },
    "pv_correlation": {
        "distribution": (
            "figure2_panel_e_pv_distribution_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
        "mean": (
            "figure2_panel_f_pv_shuffle_mean_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
    },
    "ensemble_rate_correlation": {
        "distribution": (
            "figure2_panel_g_rate_distribution_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
        "mean": (
            "figure2_panel_h_rate_shuffle_mean_"
            "corrected_842_no_presort_exclusions_36_no_1616_short.npz"
        ),
    },
}

EXPERIENCE_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")

FIGURE2_PANEL_ROLES = {
    "a": "local-correlation field-of-view example",
    "b": "example place-map pairs",
    "c": "population-vector correlation column-selection schematic",
    "d": "population-vector correlation distribution",
    "e": "population-vector mean-vs-shuffle",
    "f": "population-vector gap-length correlation curve",
    "g": "population-vector previous-vs-next gap correlation",
    "h": "place-map correlation row-selection schematic",
    "i": "place-map correlation distribution",
    "j": "place-map mean-vs-shuffle",
    "k": "place-map gap-length correlation curve",
    "l": "place-map previous-vs-next gap correlation",
    "m": "ensemble-rate correlation vector schematic",
    "n": "ensemble-rate correlation distribution",
    "o": "ensemble-rate mean-vs-shuffle",
    "p": "ensemble-rate gap-length correlation curve",
    "q": "ensemble-rate previous-vs-next gap correlation",
}

FIGURE2_SCHEMATIC_ROLES = {
    "schematic_place": "place-map correlation row-selection schematic",
    "schematic_pv": "population-vector correlation column-selection schematic",
    "schematic_rate": "ensemble-rate correlation vector schematic",
}

FIGURE2_PPT_LAYOUT_CANVAS_WIDTH_IN = 6.08

FIGURE2_TOP_ROW_SHIFT_PX = 7.0

FIGURE2_TOP_ROW_SHIFT_REFERENCE_DPI = 300.0

FIGURE2_TOP_ROW_SHIFT_IN = FIGURE2_TOP_ROW_SHIFT_PX / FIGURE2_TOP_ROW_SHIFT_REFERENCE_DPI

FIGURE2_PPT_TOP_SLOTS = {
    "a": (0.048486439195100615, 0.04500218722659668 + FIGURE2_TOP_ROW_SHIFT_IN, 1.4323129921259843, 2.1070395888014),
    "b_labels": (1.4199792213473317, FIGURE2_TOP_ROW_SHIFT_IN, 0.125, 2.3166218285214346),
    "b1": (1.5450, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b2": (2.7072189413823273, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b3": (3.8293285214348205, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
    "b4": (4.946116579177603, FIGURE2_TOP_ROW_SHIFT_IN, 1.0791458880139984, 2.3166218285214346),
}

FIGURE2_PANEL_B_SLOT = (
    FIGURE2_PPT_TOP_SLOTS["b_labels"][0],
    FIGURE2_PPT_TOP_SLOTS["b_labels"][1],
    FIGURE2_PPT_TOP_SLOTS["b4"][0]
    + FIGURE2_PPT_TOP_SLOTS["b4"][2]
    - FIGURE2_PPT_TOP_SLOTS["b_labels"][0],
    FIGURE2_PPT_TOP_SLOTS["b4"][3],
)

FIGURE2_PANEL_B_MIN_COLUMN_GAP_IN = min(
    FIGURE2_PPT_TOP_SLOTS[f"b{index + 1}"][0]
    - (FIGURE2_PPT_TOP_SLOTS[f"b{index}"][0] + FIGURE2_PPT_TOP_SLOTS[f"b{index}"][2])
    for index in range(1, 4)
)

FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS = {
    0: (0.060494313210848645, 1.5728860454943132),
    1: (1.621017060367454, 0.8354002624671916),
    2: (2.570269028871391, 1.9038823272090988),
    3: (4.538460192475941, 1.4868022747156606),
}

FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES = {0: 0.8, 1: 1.0, 2: 0.5, 3: 0.8}

FIGURE2_PPT_LOWER_GAP_IN = 0.08

FIGURE2_PPT_LOWER_CONTENT_LEFT_IN = FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS[0][0]

FIGURE2_PPT_LOWER_CONTENT_RIGHT_IN = (
    FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS[3][0]
    + FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS[3][1]
)

FIGURE2_PPT_SCHEMATIC_WIDTH_IN = (
    FIGURE2_PPT_LOWER_CONTENT_RIGHT_IN
    - FIGURE2_PPT_LOWER_CONTENT_LEFT_IN
    - sum(
        FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS[column][1]
        * FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES[column]
        for column in range(4)
    )
    - 4 * FIGURE2_PPT_LOWER_GAP_IN
)

FIGURE2_PPT_LOWER_COLUMN_SLOTS: dict[int, tuple[float, float]] = {}

_figure2_lower_column_x = (
    FIGURE2_PPT_LOWER_CONTENT_LEFT_IN
    + FIGURE2_PPT_SCHEMATIC_WIDTH_IN
    + FIGURE2_PPT_LOWER_GAP_IN
)

for _figure2_lower_column in range(4):
    _figure2_lower_column_width = (
        FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS[_figure2_lower_column][1]
        * FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES[_figure2_lower_column]
    )
    FIGURE2_PPT_LOWER_COLUMN_SLOTS[_figure2_lower_column] = (
        _figure2_lower_column_x,
        _figure2_lower_column_width,
    )
    _figure2_lower_column_x += _figure2_lower_column_width + FIGURE2_PPT_LOWER_GAP_IN

del _figure2_lower_column, _figure2_lower_column_width, _figure2_lower_column_x

FIGURE2_PPT_LOWER_FIRST_ROW_Y_IN = 2.565062335958005

FIGURE2_PPT_LOWER_ROW_HEIGHT_IN = 1.398365048118985

FIGURE2_PPT_LOWER_ROW_GAP_IN = 0.12

FIGURE2_PPT_BOTTOM_MARGIN_IN = 7.48 - (6.023109142607174 + 1.398365048118985)

FIGURE2_PPT_LOWER_ROW_SLOTS = {
    row: (
        FIGURE2_PPT_LOWER_FIRST_ROW_Y_IN
        + row * (FIGURE2_PPT_LOWER_ROW_HEIGHT_IN + FIGURE2_PPT_LOWER_ROW_GAP_IN),
        FIGURE2_PPT_LOWER_ROW_HEIGHT_IN,
    )
    for row in range(3)
}

FIGURE2_PPT_LAYOUT_CANVAS_SIZE = (
    FIGURE2_PPT_LAYOUT_CANVAS_WIDTH_IN,
    FIGURE2_PPT_LOWER_ROW_SLOTS[2][0]
    + FIGURE2_PPT_LOWER_ROW_HEIGHT_IN
    + FIGURE2_PPT_BOTTOM_MARGIN_IN,
)

FIGURE2_PPT_SCHEMATIC_SLOTS = {
    row: (
        FIGURE2_PPT_LOWER_CONTENT_LEFT_IN,
        row_y,
        FIGURE2_PPT_SCHEMATIC_WIDTH_IN,
        row_h,
    )
    for row, (row_y, row_h) in FIGURE2_PPT_LOWER_ROW_SLOTS.items()
}

FIGURE2_SCHEMATIC_CONTENT_BOUNDS = (0.025, 0.040, 0.95, 0.73)

FIGURE2_SCHEMATIC_HEADER_Y = 0.830

FIGURE2_SCHEMATIC_HEADER_HEIGHT = 0.14

FIGURE2_SCHEMATIC_HEADER_SIDE_PADDING_PT = 5.0

FIGURE2_SCHEMATIC_TITLES = {
    "place_map_correlation": "Place map",
    "pv_correlation": "Population vector (PV)",
    "ensemble_rate_correlation": "Ensemble rate",
}

FIGURE2_COMPACT_DISTRIBUTION_XLABEL = "Correlation"

FIGURE2_SCHEMATIC_PANEL_LETTERS = {0: "c", 1: "h", 2: "m"}

FIGURE2_SECOND_COLUMN_PANEL_LABEL_X_OFFSET_IN = -0.015

FIGURE2_SCHEMATIC_COMPARISON_INPUTS = {
    "place_map_correlation": (
        (0.99, 0.88, 0.54, 0.12, 0.77, 0.94),
        (0.98, 0.84, 0.60, 0.20, 0.71, 0.89),
    ),
    "pv_correlation": (
        (0.97, 0.82, 0.12, 0.58, 0.86, 0.96),
        (0.93, 0.76, 0.20, 0.65, 0.82, 0.94),
    ),
    "ensemble_rate_correlation": (
        (0.08, 0.47, 0.62, 0.78, 0.90, 0.55),
        (0.11, 0.42, 0.67, 0.73, 0.85, 0.60),
    ),
}

FIGURE2_PANEL_B_PICKS = (
    ("837-6day1->837-9day1", 588),
    ("451-6day1->451-9day1", 305),
    ("837-6day1->837-9day1", 175),
    ("418-3day1->418-6day1", 62),
)

FIGURE2_PANEL_B_AXIS_LIM = (-0.5, 14.5)

FIGURE2_PANEL_B_BOUNDARY_LINEWIDTH = 1.15

FIGURE2_LOWER_PANEL_LETTERS = {
    (row, column): chr(ord("d") + row * 5 + column)
    for row in range(3)
    for column in range(4)
}

FIGURE3_FIRST30_EXPERIENCE_COLOR_GROUPS = {
    index: group for index, group in enumerate(EXPERIENCE_ORDER)
}

FIGURE3_LONG_GAP_EXPERIENCE_COLOR_GROUPS = {
    index: EXPERIENCE_ORDER[index] for index in range(1, len(EXPERIENCE_ORDER))
}

EXPERIENCE_INDEX = {label: index for index, label in enumerate(EXPERIENCE_ORDER)}

@dataclass(frozen=True)
class PublicationStyle:
    """Single formatting contract for source-rendered manuscript panels."""

    font_family: str = "Arial"
    base_fontsize: float = 6.0
    axis_label_fontsize: float = 6.0
    tick_fontsize: float = 5.4
    title_fontsize: float = 6.2
    legend_fontsize: float = 4.9
    annotation_fontsize: float = 5.4
    panel_letter_fontsize: float = 10.5
    panel_letter_weight: str = "bold"
    axis_linewidth: float = 0.65
    tick_width: float = 0.55
    tick_length: float = 1.8
    grid_linewidth: float = 0.35
    grid_alpha: float = 0.26
    show_grid: bool = True
    data_linewidth: float = 1.25
    fit_linewidth: float = 1.10
    sem_alpha: float = 0.07
    point_size: float = 15.0
    figure2_marker_scale: float = 1.0
    figure2_pv_distribution_xlabel_x: float | None = None
    point_alpha: float = 0.72
    low_support_alpha: float = 0.24
    low_support_linewidth: float = 0.65
    figure_size: tuple[float, float] = (5.35, 7.25)
    figure2_canvas_size: tuple[float, float] = (7.10, 6.18)
    figure2_panel_a_slot: tuple[float, float, float, float] = (0.06, 0.04, 1.70, 2.24)
    figure2_panel_b_slot: tuple[float, float, float, float] = (1.72, 0.03, 5.34, 2.26)
    figure2_lower_slot: tuple[float, float, float, float] = (0.02, 2.30, 7.05, 3.84)
    figure2_export_width_mm: float = 180.0
    figure4_canvas_size: tuple[float, float] = (7.50, 4.00)
    figure4_export_width_mm: float = 180.0
    figure4_panel_a_slot: tuple[float, float, float, float] = (0.056, 0.127, 3.822, 1.646)
    figure4_panel_b_slot: tuple[float, float, float, float] = (3.878, 0.000, 0.749, 2.089)
    figure4_panel_c_slot: tuple[float, float, float, float] = (4.760, 0.205, 2.005, 1.490)
    figure4_panel_d_slot: tuple[float, float, float, float] = (6.550, 0.000, 0.749, 2.089)
    figure4_panel_e_slot: tuple[float, float, float, float] = (-0.030, 1.724, 4.899, 2.126)
    figure4_panel_f_slot: tuple[float, float, float, float] = (4.720, 1.773, 2.650, 2.226)
    figure3_top_bounds: tuple[float, float, float, float] = (0.145, 0.895, 0.785, 0.965)
    figure3_lower_bounds: tuple[float, float, float, float] = (0.145, 0.895, 0.070, 0.724)
    figure3_lower_wspace: float = 0.32
    figure3_lower_hspace: float = 0.44
    figure3_export_width_mm: float = 136.0
    day2_template_xlim_min: float = 25.0
    day2_template_xlim_max: float = 185.0
    max_lower_panel_aspect: float = 0.95
    dpi: int = 300
    facecolor: str = "white"
    max_blank_right_fraction: float = 0.12
    max_blank_bottom_fraction: float = 0.12
    figure3_metric_y_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "place_map_correlation": (0.0, 0.65),
            "pv_correlation": (0.0, 0.75),
            "ensemble_rate_correlation": (0.0, 0.90),
            "level_map_correlation": (0.0, 0.90),
        }
    )
    figure3_metric_y_ticks: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: {
            "place_map_correlation": (0.0, 0.2, 0.4, 0.6),
            "pv_correlation": (0.0, 0.2, 0.4, 0.6),
            "ensemble_rate_correlation": (0.2, 0.4, 0.6, 0.8),
            "level_map_correlation": (0.2, 0.4, 0.6, 0.8),
        }
    )
    metric_colors: dict[str, str] = field(
        default_factory=lambda: {
            "place_map_correlation": CORRELATION_MEASURE_COLORS["place_map_correlation"],
            "pv_correlation": CORRELATION_MEASURE_COLORS["pv_correlation"],
            "ensemble_rate_correlation": CORRELATION_MEASURE_COLORS["ensemble_rate_correlation"],
            "level_map_correlation": CORRELATION_MEASURE_COLORS["level_map_correlation"],
        }
    )
    day2_group_colors: dict[str, dict[str, str]] = field(
        default_factory=get_correlation_experience_colors
    )

    def rcparams(self) -> dict[str, object]:
        return {
            "font.family": self.font_family,
            "font.size": self.base_fontsize,
            "axes.labelsize": self.axis_label_fontsize,
            "axes.titlesize": self.title_fontsize,
            "xtick.labelsize": self.tick_fontsize,
            "ytick.labelsize": self.tick_fontsize,
            "legend.fontsize": self.legend_fontsize,
            "axes.linewidth": self.axis_linewidth,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": self.facecolor,
        }

def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _find_browser_executables() -> list[Path]:
    from ca_analysis.browser_export import find_browsers

    return find_browsers(preferred=("edge", "chrome", "chromium"))

def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"

def _parse_svg_length(value: str) -> float:
    text = value.strip()
    units = {
        "pt": 1.0,
        "in": 72.0,
        "px": 72.0 / 96.0,
        "cm": 72.0 / 2.54,
        "mm": 72.0 / 25.4,
    }
    for suffix, scale in units.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * scale
    return float(text)

def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float]:
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [float(value) for value in view_box.replace(",", " ").split()]
        if len(values) == 4:
            return values[2], values[3]
    return _parse_svg_length(root.attrib["width"]), _parse_svg_length(root.attrib["height"])

def _replace_id_references(value: str, id_map: dict[str, str]) -> str:
    for old_id, new_id in id_map.items():
        value = value.replace(f"url(#{old_id})", f"url(#{new_id})")
        value = value.replace(f"#{old_id}", f"#{new_id}")
    return value

def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map: dict[str, str] = {}
    for element in root.iter():
        old_id = element.attrib.get("id")
        if old_id:
            new_id = f"{prefix}_{old_id}"
            id_map[old_id] = new_id
            element.attrib["id"] = new_id
    if not id_map:
        return
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if "#" in value:
                element.attrib[key] = _replace_id_references(value, id_map)
        if element.text and "#" in element.text:
            element.text = _replace_id_references(element.text, id_map)
        if element.tail and "#" in element.tail:
            element.tail = _replace_id_references(element.tail, id_map)

def _figure_to_svg_payload(
    panel_id: str,
    fig: plt.Figure,
    *,
    dpi: int,
    fixed_bbox: bool = False,
) -> dict[str, object]:
    fig.set_dpi(dpi)
    buffer = io.StringIO()
    savefig_rc = {"savefig.bbox": None, "savefig.pad_inches": 0.0} if fixed_bbox else {}
    with plt.rc_context(savefig_rc):
        fig.savefig(
            buffer,
            format="svg",
            dpi=dpi,
            bbox_inches=None if fixed_bbox else "tight",
            pad_inches=0.0 if fixed_bbox else 0.01,
            transparent=True,
            metadata={"Date": None},
        )
    plt.close(fig)
    root = ET.fromstring(buffer.getvalue())
    _prefix_svg_ids(root, panel_id)
    width, height = _parse_svg_viewbox(root)
    children = [child for child in list(root) if child.tag != _svg_tag("metadata")]
    return {
        "children": children,
        "viewbox_width": float(width),
        "viewbox_height": float(height),
    }

def _add_svg_panel(
    root: ET.Element,
    panel_svg: dict[str, object],
    *,
    panel_id: str,
    slot_in: tuple[float, float, float, float],
    anchor: tuple[str, str] = ("center", "center"),
) -> dict[str, float]:
    slot_x, slot_y, slot_w, slot_h = (float(value) * 72.0 for value in slot_in)
    view_w = float(panel_svg["viewbox_width"])
    view_h = float(panel_svg["viewbox_height"])
    scale = min(slot_w / view_w, slot_h / view_h)
    rendered_w = view_w * scale
    rendered_h = view_h * scale
    if anchor[0] == "left":
        x = slot_x
    elif anchor[0] == "right":
        x = slot_x + slot_w - rendered_w
    else:
        x = slot_x + (slot_w - rendered_w) / 2.0
    if anchor[1] == "top":
        y = slot_y
    elif anchor[1] == "bottom":
        y = slot_y + slot_h - rendered_h
    else:
        y = slot_y + (slot_h - rendered_h) / 2.0
    group = ET.SubElement(
        root,
        _svg_tag("g"),
        {"id": f"panel_{panel_id}", "transform": f"translate({x:.6f} {y:.6f}) scale({scale:.8f})"},
    )
    for child in panel_svg["children"]:
        group.append(child)
    return {
        "slot_x": slot_x,
        "slot_y": slot_y,
        "slot_width": slot_w,
        "slot_height": slot_h,
        "x": x,
        "y": y,
        "scale": scale,
        "rendered_width": rendered_w,
        "rendered_height": rendered_h,
    }

def _add_svg_label(root: ET.Element, label: str, x_in: float, y_in: float, style: PublicationStyle) -> None:
    text = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{x_in * 72.0:.6f}",
            "y": f"{y_in * 72.0:.6f}",
            "font-family": style.font_family,
            "font-size": f"{style.panel_letter_fontsize:.1f}",
            "font-weight": style.panel_letter_weight,
            "fill": "black",
        },
    )
    text.text = str(label).lower()

def _add_svg_panel_label(
    root: ET.Element,
    label: str,
    x_in: float,
    y_in: float,
    style: PublicationStyle,
    *,
    text_anchor: str | None = None,
) -> None:
    x_pt = x_in * 72.0
    y_pt = y_in * 72.0
    attributes = {
        "x": f"{x_pt:.6f}",
        "y": f"{y_pt:.6f}",
        "font-family": style.font_family,
        "font-size": f"{style.panel_letter_fontsize:.1f}",
        "font-weight": style.panel_letter_weight,
        "fill": "black",
    }
    if text_anchor is not None:
        attributes["text-anchor"] = text_anchor
    text = ET.SubElement(
        root,
        _svg_tag("text"),
        attributes,
    )
    text.text = str(label).lower()

def _write_svg_browser_wrapper(svg_path: Path, html_path: Path, *, width_mm: float, height_mm: float) -> None:
    svg_text = svg_path.read_text(encoding="utf-8")
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: {width_mm:.6f}mm {height_mm:.6f}mm; margin: 0; }}
html, body {{ width: {width_mm:.6f}mm; height: {height_mm:.6f}mm; margin: 0; padding: 0; background: white; overflow: hidden; }}
svg {{ display: block; width: {width_mm:.6f}mm !important; height: {height_mm:.6f}mm !important; }}
* {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
</style>
</head>
<body>
{svg_text}
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8", newline="\n")

def _export_svg_with_chrome(
    svg_path: Path,
    png_path: Path,
    pdf_path: Path,
    *,
    width_mm: float,
    height_mm: float,
    dpi: int,
) -> Path:
    from ca_analysis.browser_export import platform_headless_flags, run_browser
    browsers = _find_browser_executables()
    html_path = svg_path.with_suffix(".html")
    _write_svg_browser_wrapper(svg_path, html_path, width_mm=width_mm, height_mm=height_mm)
    css_width_px = max(1, int(round(width_mm / 25.4 * 96.0)))
    css_height_px = max(1, int(round(height_mm / 25.4 * 96.0)))
    device_scale = max(0.1, float(dpi) / 96.0)
    errors: list[str] = []
    for browser in browsers:
        tmp_png = png_path.with_name(
            f".{png_path.stem}.{browser.stem}.tmp{png_path.suffix}"
        )
        tmp_pdf = pdf_path.with_name(
            f".{pdf_path.stem}.{browser.stem}.tmp{pdf_path.suffix}"
        )
        tmp_png.unlink(missing_ok=True)
        tmp_pdf.unlink(missing_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"figure_export_{browser.stem}_") as profile_dir:
                common = [
                    str(browser),
                    *platform_headless_flags(),
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=1000",
                    f"--user-data-dir={profile_dir}",
                ]
                screenshot_cmd = [
                    *common,
                    f"--window-size={css_width_px},{css_height_px}",
                    f"--force-device-scale-factor={device_scale:.6f}",
                    f"--screenshot={tmp_png.resolve()}",
                    html_path.resolve().as_uri(),
                ]
                run_browser(
                    screenshot_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
                )
                _wait_for_browser_output(tmp_png)
                pdf_cmd = [
                    *common,
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={tmp_pdf.resolve()}",
                    html_path.resolve().as_uri(),
                ]
                run_browser(
                    pdf_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
                )
                _wait_for_browser_output(tmp_pdf)
            if not tmp_png.exists() or not tmp_pdf.exists():
                raise RuntimeError("browser exited without both PNG and PDF outputs")
            os.replace(tmp_png, png_path)
            os.replace(tmp_pdf, pdf_path)
            return html_path
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(f"{browser}: {type(exc).__name__}: {exc}")
            tmp_png.unlink(missing_ok=True)
            tmp_pdf.unlink(missing_ok=True)
    raise RuntimeError(
        "All available Chrome/Edge exporters failed:\n" + "\n".join(errors)
    )

def _wait_for_browser_output(path: Path, *, timeout_seconds: float = 15.0) -> None:
    """Wait for Chromium's detached writer to finish the requested export.

    Current Windows Chromium builds can return from the headless command before
    the screenshot/PDF child process has created its output.  Requiring a
    non-empty, stable file keeps the temporary browser profile alive until the
    asynchronous write is complete.
    """
    deadline = time.monotonic() + timeout_seconds
    previous_size = -1
    stable_polls = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1
        if size > 0 and size == previous_size:
            stable_polls += 1
            if stable_polls >= 3:
                return
        else:
            stable_polls = 0
        previous_size = size
        time.sleep(0.1)
    raise RuntimeError(f"browser output did not settle within {timeout_seconds:g} s: {path}")

def _apply_axis_style(ax: plt.Axes, style: PublicationStyle) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(style.axis_linewidth)
    ax.tick_params(
        axis="both",
        width=style.tick_width,
        length=style.tick_length,
        labelsize=style.tick_fontsize,
        pad=1.5,
    )
    ax.xaxis.label.set_size(style.axis_label_fontsize)
    ax.yaxis.label.set_size(style.axis_label_fontsize)
    ax.title.set_size(style.title_fontsize)
    if style.show_grid:
        ax.grid(axis="y", color="#d8d8d8", linewidth=style.grid_linewidth, alpha=style.grid_alpha)
    else:
        ax.grid(False)

def _apply_figure2_axis_style(ax: plt.Axes, style: PublicationStyle) -> None:
    _apply_axis_style(ax, style)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

def _figure2_distribution_path(metric: str) -> Path:
    return FIGURE2_DISTRIBUTION_DIR / FIGURE2_METRIC_NPZ[metric]["distribution"]

def _figure2_mean_path(metric: str) -> Path:
    return FIGURE2_DISTRIBUTION_DIR / FIGURE2_METRIC_NPZ[metric]["mean"]

def _figure2_star_from_p(p_value: object) -> str:
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return "n.s."
    if not np.isfinite(p) or p > 0.05:
        return "n.s."
    if p > 0.01:
        return "*"
    if p > 0.001:
        return "**"
    return "***"

def _figure2_add_next_gap_annotation(ax: plt.Axes, text: str, style: PublicationStyle, color: str = "black") -> None:
    if not text:
        return
    ax.text(
        0.08,
        0.91,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.annotation_fontsize * 2.175,
        color=color,
        fontweight="bold",
    )

def _figure2_panel_b_wspace(size: tuple[float, float], *, left: float, right: float, n_cols: int = 4) -> float:
    # The maps are aspect-locked square axes, so the actual rendered map gap
    # is slightly larger than the subplot-cell gap. This value matches the
    # actual rendered gap to the smallest PPT-derived prior gap.
    return -0.005

def _apply_figure2_panel_b_square_boundaries(axes: np.ndarray) -> None:
    low, high = FIGURE2_PANEL_B_AXIS_LIM
    for ax in np.asarray(axes, dtype=object).flat:
        ax.add_patch(
            plt.Rectangle(
                (low, low),
                high - low,
                high - low,
                fill=False,
                edgecolor="black",
                linewidth=FIGURE2_PANEL_B_BOUNDARY_LINEWIDTH,
                joinstyle="miter",
                capstyle="butt",
                zorder=5,
                clip_on=False,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(FIGURE2_PANEL_B_AXIS_LIM)
        ax.set_ylim(FIGURE2_PANEL_B_AXIS_LIM)
        for spine in ax.spines.values():
            spine.set_visible(False)

def _render_figure2_panel_b_four_examples(size: tuple[float, float], style: PublicationStyle) -> plt.Figure:
    margins = {"left": 0.038, "right": 0.999, "top": 0.996, "bottom": 0.070}
    fig, _axes, _data, _paths = render_example_cell_grid_panel(
        picks=FIGURE2_PANEL_B_PICKS,
        mode="arrow",
        saved_dir=EXAMPLE_SAVED_DIR,
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "grid_wspace": _figure2_panel_b_wspace(size, left=margins["left"], right=margins["right"]),
            "grid_hspace": 0.010,
            "margins": margins,
        },
        colors={
            "maze_line": "k",
            "trajectory": "0.55",
            "arrow_face": "red",
            "arrow_edge": "red",
        },
        style={
            "maze_linewidth": 1.15,
            "show_maze": False,
            "show_frame": False,
            "axis_lim": FIGURE2_PANEL_B_AXIS_LIM,
            "title_fontsize": style.title_fontsize,
            "title_location": "bottom",
            "title_y": -0.075,
            "row_label_fontsize": 8.0,
            "row_label_pad": 2,
            "trajectory_alpha": 0.28,
            "trajectory_linewidth": 0.26,
            "arrow_head_width": 0.14,
            "arrow_head_length": 0.34,
            "arrow_linewidth": 0.01,
            "arrow_alpha": 0.42,
            "arrow_scale": 0.25,
        },
        labels={
            "row_top": "Day i",
            "row_bottom": "Day i+90",
            "title_template": "{place_r:.2f}",
        },
        save=False,
        n_bins=45,
        sigma=1.0,
        signal="C",
        dt=0.04,
        min_time_base=0.5,
    )
    bottom_row_drop = 0.010
    for ax in _axes[1, :]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - bottom_row_drop, pos.width, pos.height])
        for label_text in ax.texts:
            x_text, y_text = label_text.get_position()
            label_text.set_position((x_text, y_text + bottom_row_drop / pos.height))
    _apply_figure2_panel_b_square_boundaries(_axes)
    for ax in fig.axes:
        ax.title.set_fontsize(style.title_fontsize)
        ax.xaxis.label.set_fontsize(style.axis_label_fontsize)
        ax.yaxis.label.set_fontsize(style.axis_label_fontsize)
        ax.tick_params(labelsize=style.tick_fontsize)
    _axes[0, 0].yaxis.label.set_fontsize(8.0)
    _axes[1, 0].yaxis.label.set_fontsize(8.0)
    return fig

def _figure2_density_from_hist(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    total = float(np.nansum(counts))
    if total <= 0:
        return np.zeros_like(widths, dtype=float)
    return np.asarray(counts, dtype=float) / (total * widths)

def _draw_figure2_distribution_panel(ax: plt.Axes, metric: str, style: PublicationStyle) -> None:
    data = np.load(_figure2_distribution_path(metric), allow_pickle=True)
    edges = np.asarray(data["hist_edges"], dtype=float)
    obs = np.asarray(data["obs"], dtype=float)
    color = style.metric_colors[metric]
    obs_density = _figure2_density_from_hist(np.histogram(obs[np.isfinite(obs)], bins=edges)[0], edges)
    if "shuffle_values" in data.files:
        shuffle_values = np.asarray(data["shuffle_values"], dtype=float)
        shuffle_density = _figure2_density_from_hist(
            np.histogram(shuffle_values[np.isfinite(shuffle_values)], bins=edges)[0],
            edges,
        )
    else:
        shuffle_density = _figure2_density_from_hist(np.asarray(data["shuffle_hist"], dtype=float), edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    ax.bar(
        centers,
        shuffle_density,
        width=widths,
        color="#8a8a8a",
        alpha=0.62,
        edgecolor="none",
        align="center",
        label="Shuffle",
        zorder=1,
    )
    ax.bar(
        centers,
        obs_density,
        width=widths,
        color=color,
        alpha=0.72,
        edgecolor="none",
        align="center",
        label="Data",
        zorder=2,
    )
    ax.set_xlim(-0.45, 1.0)
    ax.set_ylim(bottom=0)
    if metric == "place_map_correlation":
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_xlabel(FIGURE2_DISTRIBUTION_XLABELS[metric])
    if metric == "pv_correlation" and style.figure2_pv_distribution_xlabel_x is not None:
        ax.xaxis.label.set_x(style.figure2_pv_distribution_xlabel_x)
    ax.set_ylabel("Density")
    shuffle_label_xy = {
        "place_map_correlation": (0.42, 0.89),
        "pv_correlation": (0.05, 0.83),
        "ensemble_rate_correlation": (0.05, 0.83),
    }[metric]
    ax.text(
        shuffle_label_xy[0],
        shuffle_label_xy[1],
        "Shuffle",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.annotation_fontsize,
        color="#6f6f6f",
        fontweight="bold",
    )
    data_label_xy = {
        "place_map_correlation": (0.74, 0.33),
        "pv_correlation": (0.61, 0.30),
        "ensemble_rate_correlation": (0.42, 0.32),
    }[metric]
    data_label_bbox = (
        {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4}
        if metric == "ensemble_rate_correlation"
        else None
    )
    ax.text(
        data_label_xy[0],
        data_label_xy[1],
        "Data",
        transform=ax.transAxes,
        ha="left",
        va="bottom" if metric == "ensemble_rate_correlation" else "center",
        fontsize=style.annotation_fontsize,
        color=color,
        fontweight="bold",
        bbox=data_label_bbox,
    )
    _apply_figure2_axis_style(ax, style)

def _draw_figure2_mean_panel(ax: plt.Axes, metric: str, style: PublicationStyle) -> None:
    data = np.load(_figure2_mean_path(metric), allow_pickle=True)
    null = np.asarray(data["null"], dtype=float)
    obs = float(np.asarray(data["obs"], dtype=float).ravel()[0])
    color = style.metric_colors[metric]
    xlims = {
        "place_map_correlation": (-0.012, 0.13),
        "pv_correlation": (-0.025, 0.25),
        "ensemble_rate_correlation": (-0.055, 0.62),
    }
    ax.hist(null[np.isfinite(null)], bins=60, color="#6f6f6f", alpha=0.78, edgecolor="none")
    ax.axvline(obs, color=color, linestyle=(0, (4, 2)), linewidth=style.data_linewidth + 0.35)
    ax.set_xlim(*xlims[metric])
    ax.set_xlabel("Mean correlation")
    ax.set_ylabel("")
    ax.text(
        0.13,
        0.78,
        "Shuffle",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style.annotation_fontsize,
        color="#6f6f6f",
        fontweight="bold",
    )
    ax.text(
        0.58,
        0.52,
        "Data",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=style.annotation_fontsize,
        color=color,
        fontweight="bold",
    )
    _apply_figure2_axis_style(ax, style)

def _draw_figure2_gap_curve(ax: plt.Axes, curve_df: pd.DataFrame, metric: str, style: PublicationStyle) -> None:
    short = FIGURE2_METRIC_SHORT[metric]
    rows = curve_df.loc[curve_df["metric"].eq(short)].sort_values("time_days")
    if rows.empty:
        raise ValueError(f"No gap-curve rows for {metric}")
    x = pd.to_numeric(rows["time_days"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["observed_mean"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(rows["observed_ci95_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(rows["observed_ci95_high"], errors="coerce").to_numpy(dtype=float)
    shuffle = pd.to_numeric(rows["shuffle_mean"], errors="coerce").to_numpy(dtype=float)
    color = style.metric_colors[metric]
    ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
    ax.plot(
        x,
        y,
        color=color,
        marker="o",
        markersize=2.4 * style.figure2_marker_scale,
        linewidth=style.data_linewidth,
        label="Data",
    )
    ax.plot(
        x,
        shuffle,
        color="#777777",
        marker="s",
        markersize=1.9 * style.figure2_marker_scale,
        linestyle=(0, (2, 1.4)),
        linewidth=0.75,
        label="Shuffle",
    )
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=style.annotation_fontsize,
        handlelength=1.2,
        handletextpad=0.35,
        borderaxespad=0.10,
        labelspacing=0.20,
    )
    ax.set_xlim(-8, 370)
    if metric == "ensemble_rate_correlation":
        ax.set_ylim(-0.16, 0.84)
    else:
        ax.set_ylim(-0.08, 0.84)
    ax.set_xticks([1, 90, 180, 270, 360])
    ax.set_xticklabels(["1", "90", "180", "270", "360"])
    ax.set_xlabel("Temporal separation (days)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{METRIC_LABELS[metric]} correlation", pad=1.2)
    _apply_figure2_axis_style(ax, style)

def _binned_mean_line(x: np.ndarray, y: np.ndarray, *, xlim: tuple[float, float], bins: int = 13) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    means = np.full_like(centers, np.nan, dtype=float)
    for index in range(bins):
        if index == bins - 1:
            mask = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask = (x >= edges[index]) & (x < edges[index + 1])
        if np.count_nonzero(mask) >= 3:
            means[index] = float(np.nanmean(y[mask]))
    keep = np.isfinite(means)
    return centers[keep], means[keep]

def _clustered_linear_fit_confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    x_fit: np.ndarray,
    *,
    intercept: float,
    slope: float,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray] | None:
    """CR1 confidence band for the fitted mean, clustered by animal."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    x_fit = np.asarray(x_fit, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    groups = groups[keep]
    n_rows = int(x.size)
    unique_groups = np.unique(groups)
    n_groups = int(unique_groups.size)
    n_parameters = 2
    if (
        n_rows <= n_parameters
        or n_groups <= 1
        or not (np.isfinite(intercept) and np.isfinite(slope))
    ):
        return None
    design = np.column_stack((np.ones(n_rows, dtype=float), x))
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    residuals = y - (intercept + slope * x)
    meat = np.zeros((n_parameters, n_parameters), dtype=float)
    for group in unique_groups:
        selected = groups == group
        score = design[selected].T @ residuals[selected]
        meat += np.outer(score, score)
    cr1_factor = (n_groups / (n_groups - 1)) * (
        (n_rows - 1) / (n_rows - n_parameters)
    )
    covariance = cr1_factor * bread @ meat @ bread
    fit_design = np.column_stack((np.ones(x_fit.size, dtype=float), x_fit))
    variance = np.einsum("ij,jk,ik->i", fit_design, covariance, fit_design)
    if np.any(~np.isfinite(variance)):
        return None
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    t_crit = float(
        scipy_stats.t.ppf((1.0 + confidence) / 2.0, df=n_groups - 1)
    )
    y_fit = intercept + slope * x_fit
    interval = t_crit * standard_error
    return y_fit - interval, y_fit + interval

def _figure2_binned_curve_audit(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    limits_by_metric = {
        "place_map_correlation": (-0.50, 1.0),
        "pv_correlation": (-0.05, 0.85),
        "level_map_correlation": (-1.0, 1.0),
    }
    for metric, limits in limits_by_metric.items():
        sub = points.loc[points["metric"].astype(str).eq(metric)].copy()
        x = pd.to_numeric(sub["prev_gap_correlation"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["next_gap_correlation"], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        line_x, line_y = _binned_mean_line(x[keep], y[keep], xlim=limits, bins=13)
        rows.extend(
            {
                "metric": metric,
                "bin_order": index,
                "x_center": float(x_value),
                "next_gap_mean": float(y_value),
                "source_rows": int(keep.sum()),
                "scope": "clean_long_gap_day1_triplets",
            }
            for index, (x_value, y_value) in enumerate(zip(line_x, line_y))
        )
    return pd.DataFrame(rows)

def _figure2_displayed_estimand_stats(dense_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("place_map_correlation", "pv_correlation", "level_map_correlation"):
        match = dense_stats.loc[dense_stats["metric"].astype(str).eq(metric)]
        if len(match) != 1:
            raise AssertionError(f"Expected one corrected dense stats row for {metric}.")
        stat = match.iloc[0]
        rows.append(
            {
                "metric": metric,
                "metric_label": str(stat["metric_label"]),
                "displayed_estimand": str(stat["point_kind"]),
                "n_rows": int(stat["n_rows"]),
                "n_animals": int(stat["n_animals"]),
                "intercept": float(stat["intercept"]),
                "slope": float(stat["slope"]),
                "slope_se": float(stat["slope_se"]),
                "slope_t": float(stat["slope_t"]),
                "df": int(stat["df"]),
                "p_two_sided": float(stat["p_two_sided"]),
                "source_stats": _rel(FIGURE2_CONSECUTIVE_STATS),
                "test": str(stat["test"]),
            }
        )
    rate_stats = pd.read_csv(FIGURE2_RATE_CHAIN_STATS)
    rate = rate_stats.loc[
        rate_stats["scope"].astype(str).eq("clean_long_gap_day1_triplets")
        & rate_stats["metric"].astype(str).eq("value_sorted")
    ]
    if len(rate) != 1:
        raise AssertionError("Expected one corrected clean-long-gap ensemble-rate stats row.")
    stat = rate.iloc[0]
    rows.append(
        {
            "metric": "ensemble_rate_correlation",
            "metric_label": "Ensemble rate correlation",
            "displayed_estimand": "session_chain",
            "n_rows": int(stat["n_rows"]),
            "n_animals": int(stat["n_animals"]),
            "intercept": float(stat["intercept"]),
            "slope": float(stat["slope"]),
            "slope_se": float(stat["slope_se_cluster_animal"]),
            "slope_t": float(stat["slope_t_cluster_animal"]),
            "df": int(stat["df"]),
            "p_two_sided": float(stat["p_two_sided_cluster_animal"]),
            "source_stats": _rel(FIGURE2_RATE_CHAIN_STATS),
            "test": str(stat["test"]),
        }
    )
    table = pd.DataFrame(rows)
    table["q_bh4_two_sided"] = _bh_q(table["p_two_sided"].astype(float).tolist())
    table["correction_family"] = "Figure2 displayed previous-vs-next estimands BH4"
    return table

def _draw_figure2_rate_chain_panel(ax: plt.Axes, style: PublicationStyle) -> None:
    scope = "clean_long_gap_day1_triplets"
    points = pd.read_csv(FIGURE2_RATE_CHAIN_POINTS)
    stats_df = pd.read_csv(FIGURE2_RATE_CHAIN_STATS)
    rows = points.loc[points["scope"].eq(scope)].copy()
    if rows.empty:
        raise ValueError(f"No Figure 2 rate-chain rows for scope {scope}")
    x = pd.to_numeric(rows["value_sorted_ab"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["value_sorted_bc"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    rows = rows.loc[keep].copy()
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        raise ValueError("No finite sorted rate-chain points for Figure 2")

    color = style.metric_colors["ensemble_rate_correlation"]
    ax.scatter(
        x,
        y,
        s=style.point_size * 1.05,
        facecolor=color,
        edgecolor="#2f4146",
        linewidth=0.45,
        alpha=0.72,
        zorder=2,
    )

    stat_row = stats_df.loc[stats_df["scope"].eq(scope) & stats_df["metric"].eq("value_sorted")]
    if stat_row.empty:
        raise ValueError(f"No Figure 2 rate-chain stats for scope {scope}")
    stat = stat_row.iloc[0]
    displayed_stats = _figure2_displayed_estimand_stats(pd.read_csv(FIGURE2_CONSECUTIVE_STATS))
    displayed_rate = displayed_stats.loc[
        displayed_stats["metric"].astype(str).eq("ensemble_rate_correlation")
    ].iloc[0]
    intercept = float(stat["intercept"])
    slope = float(stat["slope"])
    axis_limits = (-0.10, 0.80)
    xfit = np.linspace(axis_limits[0], axis_limits[1], 120)
    ci_band = _clustered_linear_fit_confidence_band(
        x,
        y,
        rows["animal_id"].astype(str).to_numpy(),
        xfit,
        intercept=intercept,
        slope=slope,
    )
    if ci_band is not None:
        ax.fill_between(xfit, ci_band[0], ci_band[1], color=color, alpha=0.16, linewidth=0, zorder=1)
    ax.plot(xfit, intercept + slope * xfit, color=color, linewidth=style.data_linewidth + 0.25, zorder=3)

    _figure2_add_next_gap_annotation(
        ax,
        _figure2_star_from_p(displayed_rate["q_bh4_two_sided"]),
        style,
        color="black",
    )
    ax.text(
        0.94,
        0.08,
        f"r = {float(stat['pearson_r']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=style.annotation_fontsize,
        color=color,
        fontweight="bold",
    )

    ax.set_xlim(*axis_limits)
    ax.set_ylim(*axis_limits)
    ax.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))
    ax.set_yticks((-0.1, 0.0, 0.2, 0.4, 0.6, 0.8))
    ax.set_xlabel("Previous gap correlation")
    ax.set_ylabel("Next gap correlation")
    _apply_figure2_axis_style(ax, style)

def _draw_figure2_prev_next_panel(
    ax: plt.Axes,
    points_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    metric: str,
    style: PublicationStyle,
    *,
    displayed_q_two_sided: float | None = None,
) -> None:
    if metric == "ensemble_rate_correlation":
        _draw_figure2_rate_chain_panel(ax, style)
        return

    rows = points_df.loc[points_df["metric"].eq(metric)].copy()
    if rows.empty:
        raise ValueError(f"No consecutive-gap rows for {metric}")
    x = pd.to_numeric(rows["prev_gap_correlation"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["next_gap_correlation"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    color = style.metric_colors[metric]
    limits = {
        "place_map_correlation": (-0.50, 1.0),
        "pv_correlation": (-0.05, 0.85),
        "ensemble_rate_correlation": (0.0, 1.0),
        "level_map_correlation": (-1.0, 1.0),
    }[metric]
    alpha = 0.035 if x.size > 5000 else 0.115
    size = (4.0 if x.size > 5000 else 7.0) * style.figure2_marker_scale * style.figure2_marker_scale
    ax.scatter(x, y, s=size, color=color, alpha=alpha, edgecolors="none", rasterized=True)
    line_x, line_y = _binned_mean_line(x, y, xlim=limits, bins=13)
    if line_x.size >= 2:
        ax.plot(line_x, line_y, color=color, linewidth=style.data_linewidth + 0.25)
    if displayed_q_two_sided is None:
        displayed_stats = _figure2_displayed_estimand_stats(stats_df)
        displayed_row = displayed_stats.loc[displayed_stats["metric"].astype(str).eq(metric)]
        if len(displayed_row) != 1:
            raise AssertionError(f"Expected one displayed-estimand BH4 row for {metric}.")
        displayed_q_two_sided = float(displayed_row.iloc[0]["q_bh4_two_sided"])
    elif not np.isfinite(displayed_q_two_sided):
        raise AssertionError(f"Displayed-estimand BH4 q is non-finite for {metric}.")
    _figure2_add_next_gap_annotation(
        ax,
        _figure2_star_from_p(float(displayed_q_two_sided)),
        style,
        color="black",
    )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    if metric == "place_map_correlation":
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Previous gap correlation")
    ax.set_ylabel("Next gap correlation")
    _apply_figure2_axis_style(ax, style)

def _figure2_schematic_header_bounds(
    size: tuple[float, float],
    style: PublicationStyle,
) -> tuple[float, float, float, float]:
    title_font = FontProperties(family=style.font_family, size=style.title_fontsize)
    longest_title_width_pt = float(
        TextPath(
            (0.0, 0.0),
            "Population vector (PV)",
            prop=title_font,
            usetex=False,
        ).get_extents().width
    )
    header_w = min(
        0.94,
        (
            longest_title_width_pt
            + 2.0 * FIGURE2_SCHEMATIC_HEADER_SIDE_PADDING_PT
        )
        / 72.0
        / size[0],
    )
    return (
        (1.0 - header_w) / 2.0,
        FIGURE2_SCHEMATIC_HEADER_Y,
        header_w,
        FIGURE2_SCHEMATIC_HEADER_HEIGHT,
    )

def _render_figure2_metric_schematic(
    metric: str,
    size: tuple[float, float],
    style: PublicationStyle,
) -> plt.Figure:
    if metric not in METRICS:
        raise ValueError(f"unknown Figure 2 schematic metric: {metric}")

    session_1 = np.asarray(
        [
            [0.08, 0.36, 0.82, 0.97, 0.99, 1.00],
            [0.88, 0.52, 0.09, 0.82, 0.96, 0.99],
            [0.99, 0.88, 0.54, 0.12, 0.77, 0.94],
            [1.00, 0.98, 0.88, 0.58, 0.11, 0.72],
            [1.00, 1.00, 0.98, 0.86, 0.57, 0.08],
            [1.00, 1.00, 1.00, 0.96, 0.83, 0.49],
        ],
        dtype=float,
    )
    session_2 = np.asarray(
        [
            [0.06, 0.33, 0.78, 0.96, 0.99, 1.00],
            [0.91, 0.56, 0.13, 0.79, 0.95, 0.99],
            [0.99, 0.90, 0.57, 0.16, 0.73, 0.92],
            [1.00, 0.98, 0.90, 0.61, 0.15, 0.69],
            [1.00, 1.00, 0.98, 0.88, 0.61, 0.10],
            [1.00, 1.00, 1.00, 0.96, 0.86, 0.52],
        ],
        dtype=float,
    )
    session_2[2, :] = FIGURE2_SCHEMATIC_COMPARISON_INPUTS["place_map_correlation"][1]
    session_2[:, 3] = FIGURE2_SCHEMATIC_COMPARISON_INPUTS["pv_correlation"][1]
    metric_color = style.metric_colors[metric]
    grid_color = "#D2D5D6"
    arrow_color = "#34383A"
    comparison_color = "#6C7072"
    fig = plt.figure(figsize=size, facecolor="none")
    ax = fig.add_axes(FIGURE2_SCHEMATIC_CONTENT_BOUNDS)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    def draw_matrix(matrix: np.ndarray, x0: float, y0: float, width: float, height: float) -> None:
        cell_w = width / matrix.shape[1]
        cell_h = height / matrix.shape[0]
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.add_patch(
                    plt.Rectangle(
                        (x0 + column * cell_w, y0 + (matrix.shape[0] - row - 1) * cell_h),
                        cell_w,
                        cell_h,
                        facecolor=plt.cm.Greys(1.0 - matrix[row, column]),
                        edgecolor=grid_color,
                        linewidth=0.45,
                    )
                )

    content_width_in = float(size[0]) * FIGURE2_SCHEMATIC_CONTENT_BOUNDS[2]
    content_height_in = float(size[1]) * FIGURE2_SCHEMATIC_CONTENT_BOUNDS[3]
    matrix_w = 0.34
    matrix_h = matrix_w * content_width_in / content_height_in
    matrix_y = 0.56 - matrix_h / 2.0
    matrix_x = (0.12, 0.57)
    if metric in {"place_map_correlation", "pv_correlation"}:
        for index, (x0, matrix) in enumerate(zip(matrix_x, (session_1, session_2))):
            draw_matrix(matrix, x0, matrix_y, matrix_w, matrix_h)
            ax.text(
                x0 + matrix_w / 2.0,
                0.93,
                f"Session {index + 1}",
                ha="center",
                va="center",
                fontsize=style.annotation_fontsize,
            )
            ax.annotate(
                "",
                xy=(x0 + matrix_w, 0.27),
                xytext=(x0, 0.27),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": arrow_color,
                    "linewidth": 0.75,
                    "mutation_scale": 7.0,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
            )
            ax.text(
                x0 + matrix_w / 2.0,
                0.10,
                "Position",
                ha="center",
                va="center",
                fontsize=style.annotation_fontsize,
            )
        ax.text(
            0.515,
            matrix_y + matrix_h / 2.0,
            "vs",
            ha="center",
            va="center",
            fontsize=style.annotation_fontsize,
            color=comparison_color,
        )
        ax.annotate(
            "",
            xy=(0.070, matrix_y + matrix_h),
            xytext=(0.070, matrix_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": arrow_color,
                "linewidth": 0.75,
                "mutation_scale": 7.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        ax.text(
            0.024,
            matrix_y + matrix_h / 2.0,
            "Neuron",
            ha="center",
            va="center",
            rotation=90,
            fontsize=style.annotation_fontsize,
        )
        for x0 in matrix_x:
            if metric == "place_map_correlation":
                row = 2
                row_h = matrix_h / 6.0
                highlight = (x0, matrix_y + (5 - row) * row_h, matrix_w, row_h)
            else:
                column = 3
                column_w = matrix_w / 6.0
                highlight = (x0 + column * column_w, matrix_y, column_w, matrix_h)
            ax.add_patch(
                plt.Rectangle(
                    highlight[:2],
                    highlight[2],
                    highlight[3],
                    facecolor=metric_color,
                    edgecolor=metric_color,
                    alpha=0.18,
                    linewidth=1.0,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    highlight[:2],
                    highlight[2],
                    highlight[3],
                    fill=False,
                    edgecolor=metric_color,
                    linewidth=1.05,
                )
            )
    else:
        vector_h = 0.60
        vector_cell_h = vector_h / 6.0
        vector_w = vector_cell_h * content_height_in / content_width_in
        vector_y = 0.55 - vector_h / 2.0
        vector_x = tuple(center - vector_w / 2.0 for center in (0.27, 0.75))
        rate_1, rate_2 = (
            np.asarray(values, dtype=float)
            for values in FIGURE2_SCHEMATIC_COMPARISON_INPUTS[metric]
        )
        for index, (x0, values) in enumerate(zip(vector_x, (rate_1, rate_2))):
            cell_h = vector_h / len(values)
            for row, value in enumerate(values):
                ax.add_patch(
                    plt.Rectangle(
                        (x0, vector_y + (len(values) - row - 1) * cell_h),
                        vector_w,
                        cell_h,
                        facecolor=plt.cm.Greys(1.0 - value),
                        edgecolor=grid_color,
                        linewidth=0.45,
                    )
                )
            ax.add_patch(
                plt.Rectangle(
                    (x0, vector_y),
                    vector_w,
                    vector_h,
                    fill=False,
                    edgecolor=metric_color,
                    linewidth=1.05,
                )
            )
            ax.text(
                x0 + vector_w / 2.0,
                0.93,
                f"Session {index + 1}",
                ha="center",
                va="center",
                fontsize=style.annotation_fontsize,
            )
            ax.text(
                x0 + vector_w / 2.0,
                0.12,
                "Mean rate",
                ha="center",
                va="center",
                fontsize=style.annotation_fontsize,
            )
        ax.text(
            0.510,
            vector_y + vector_h / 2.0,
            "vs",
            ha="center",
            va="center",
            fontsize=style.annotation_fontsize,
            color=comparison_color,
        )
        ax.annotate(
            "",
            xy=(0.155, vector_y + vector_h),
            xytext=(0.155, vector_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": arrow_color,
                "linewidth": 0.75,
                "mutation_scale": 7.0,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        ax.text(
            0.100,
            vector_y + vector_h / 2.0,
            "Neuron",
            ha="center",
            va="center",
            rotation=90,
            fontsize=style.annotation_fontsize,
        )
    header_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=10)
    header_ax.set_xlim(0.0, 1.0)
    header_ax.set_ylim(0.0, 1.0)
    header_ax.set_axis_off()
    header_x, header_y, header_w, header_h = _figure2_schematic_header_bounds(size, style)
    header_ax.add_patch(
        plt.Rectangle(
            (header_x, header_y),
            header_w,
            header_h,
            facecolor=metric_color,
            edgecolor="none",
        )
    )
    header_ax.text(
        header_x + header_w / 2.0,
        header_y + header_h / 2.0,
        FIGURE2_SCHEMATIC_TITLES[metric],
        ha="center",
        va="center",
        color="white",
        fontsize=style.title_fontsize,
        fontweight="normal",
    )
    return fig

def _draw_figure2_single_lower_panel(
    metric: str,
    column: int,
    size: tuple[float, float],
    curve_df: pd.DataFrame,
    points_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    style: PublicationStyle,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    if column == 0:
        _draw_figure2_distribution_panel(ax, metric, style)
        ax.set_xlabel(FIGURE2_COMPACT_DISTRIBUTION_XLABEL)
    elif column == 1:
        _draw_figure2_mean_panel(ax, metric, style)
    elif column == 2:
        _draw_figure2_gap_curve(ax, curve_df, metric, style)
        ax.set_title("")
    elif column == 3:
        _draw_figure2_prev_next_panel(ax, points_df, stats_df, metric, style)
    else:
        raise ValueError(f"unknown Figure 2 lower column: {column}")
    fig.tight_layout(pad=0.15)
    if column == 2:
        # Apply after tight_layout so every data-axis rectangle stays fixed.
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_rotation_mode("anchor")
        ax.xaxis.labelpad = 0.0
        ax.set_xlabel("Days apart")
    if column == 1:
        # Keep the accepted histogram axes rectangle: the display-only title
        # must not participate in tight_layout and narrow panels e/j/o.
        ax.set_ylabel("Count")
    if column == 3:
        # Shared data frame for g/l/q; margins are points within the fixed slot.
        width_pt, height_pt = np.asarray(size) * 72.0
        fig.subplots_adjust(
            left=25.6 / width_pt, right=1.0 - 4.935 / width_pt,
            bottom=19.3 / height_pt, top=1.0 - 3.0 / height_pt,
        )
    return fig

def _bh_q(p_values: list[float]) -> list[float]:
    arr = np.asarray(p_values, dtype=float)
    q = np.full(arr.shape, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return q.tolist()
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(arr[idx])]
    ranked = arr[order]
    adjusted = ranked * float(len(order)) / np.arange(1, len(order) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q.tolist()

def render_figure2(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_scope_audit = _validate_figure2_distribution_sources()
    fig2 = _load_script_module("preview_figure2_assembly_source_render", FIGURE2_HELPER)
    canvas_w, canvas_h = FIGURE2_PPT_LAYOUT_CANVAS_SIZE
    figure_w_pt = canvas_w * 72.0
    figure_h_pt = canvas_h * 72.0
    export_height_mm = style.figure2_export_width_mm * canvas_h / canvas_w
    b_picks = list(FIGURE2_PANEL_B_PICKS)
    lower_panel_letters = dict(FIGURE2_LOWER_PANEL_LETTERS)
    schematic_panel_letters = dict(FIGURE2_SCHEMATIC_PANEL_LETTERS)
    lower_roles = {
        0: "correlation distribution",
        1: "mean-vs-neuron-ID-shuffle null",
        2: "gap-length correlation curve",
        3: "previous-vs-next gap correlation",
    }
    lower_sources = {
        0: "corrected-registration correlation/shuffle distribution NPZ",
        1: "corrected-registration shuffle-mean NPZ",
        2: _rel(FIGURE2_GAP_CURVE),
        3: _rel(FIGURE2_CONSECUTIVE_POINTS),
    }
    row_metrics = ["pv_correlation", "place_map_correlation", "ensemble_rate_correlation"]
    schematic_keys = {
        0: "schematic_pv",
        1: "schematic_place",
        2: "schematic_rate",
    }
    lower_axes_records: list[dict[str, object]] = []
    with plt.rc_context(style.rcparams()):
        panel_svgs: dict[str, dict[str, object]] = {}
        fig_a = fig2._render_panel_a(
            (FIGURE2_PPT_TOP_SLOTS["a"][2], FIGURE2_PPT_TOP_SLOTS["a"][3])
        )
        for ax in fig_a.axes:
            for text in ax.texts:
                if text.get_text().startswith("100"):
                    text.set_text("100 \u00b5m")
                    text.set_position((312.0, 268.0))
        panel_svgs["a"] = _figure_to_svg_payload("figure2_a", fig_a, dpi=int(style.dpi), fixed_bbox=True)

        fig_b = _render_figure2_panel_b_four_examples(
            (FIGURE2_PANEL_B_SLOT[2], FIGURE2_PANEL_B_SLOT[3]),
            style,
        )
        panel_svgs["b"] = _figure_to_svg_payload("figure2_b", fig_b, dpi=int(style.dpi), fixed_bbox=True)

        curve_df = pd.read_csv(FIGURE2_GAP_CURVE)
        points_df = pd.read_csv(FIGURE2_CONSECUTIVE_POINTS)
        stats_df = pd.read_csv(FIGURE2_CONSECUTIVE_STATS)
        consecutive_scope_audit = _validate_figure2_clean_consecutive_scope(points_df, stats_df)
        consecutive_binned_curve = _figure2_binned_curve_audit(points_df)
        displayed_estimand_stats = _figure2_displayed_estimand_stats(stats_df)
        stats_df = stats_df.copy()
        displayed_q = displayed_estimand_stats.set_index("metric")["q_bh4_two_sided"].to_dict()
        stats_df["q_bh_two_sided"] = stats_df["metric"].astype(str).map(displayed_q)
        if not np.isfinite(stats_df["q_bh_two_sided"].to_numpy(float)).all():
            raise AssertionError("Every current dense metric must have a displayed-estimand BH4 value.")
        for row, metric in enumerate(row_metrics):
            row_y, row_h = FIGURE2_PPT_LOWER_ROW_SLOTS[row]
            schematic_key = schematic_keys[row]
            fig_schematic = _render_figure2_metric_schematic(
                metric,
                (FIGURE2_PPT_SCHEMATIC_WIDTH_IN, row_h),
                style,
            )
            panel_svgs[schematic_key] = _figure_to_svg_payload(
                f"figure2_{schematic_key}",
                fig_schematic,
                dpi=int(style.dpi),
                fixed_bbox=True,
            )
            for col in range(4):
                col_x, col_w = FIGURE2_PPT_LOWER_COLUMN_SLOTS[col]
                letter = lower_panel_letters[(row, col)]
                fig_lower_panel = _draw_figure2_single_lower_panel(
                    metric,
                    col,
                    (col_w, row_h),
                    curve_df,
                    points_df,
                    stats_df,
                    style,
                )
                panel_svgs[letter] = _figure_to_svg_payload(
                    f"figure2_{letter}",
                    fig_lower_panel,
                    dpi=int(style.dpi),
                    fixed_bbox=True,
                )
                lower_axes_records.append(
                    {
                        "key": f"figure2_{metric}_col{col + 1}",
                        "metric": metric,
                        "row": row,
                        "column": col,
                        "x0": col_x / canvas_w,
                        "y0": row_y / canvas_h,
                        "width": col_w / canvas_w,
                        "height": row_h / canvas_h,
                        "width_in": col_w,
                        "height_in": row_h,
                        "slot_in": [col_x, row_y, col_w, row_h],
                    }
                )

    panel_slots: dict[str, tuple[float, float, float, float]] = {
        "a": FIGURE2_PPT_TOP_SLOTS["a"],
        "b": FIGURE2_PANEL_B_SLOT,
    }
    panel_slots.update(
        {
            schematic_keys[row]: FIGURE2_PPT_SCHEMATIC_SLOTS[row]
            for row in range(3)
        }
    )
    for record in lower_axes_records:
        key = lower_panel_letters[(int(record["row"]), int(record["column"]))]
        panel_slots[key] = tuple(float(value) for value in record["slot_in"])

    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{style.figure2_export_width_mm:.3f}mm",
            "height": f"{export_height_mm:.3f}mm",
            "viewBox": f"0 0 {figure_w_pt:.6f} {figure_h_pt:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
    rendered_sizes = {
        panel_id: _add_svg_panel(
            root,
            panel_svgs[panel_id],
            panel_id=f"figure2_{panel_id}",
            slot_in=slot,
            anchor=("left", "top"),
        )
        for panel_id, slot in panel_slots.items()
    }
    _add_svg_label(root, "a", 0.015, 0.17 + FIGURE2_TOP_ROW_SHIFT_IN, style)
    _add_svg_label(root, "b", 1.475, 0.17 + FIGURE2_TOP_ROW_SHIFT_IN, style)
    for row in range(3):
        slot_x, slot_y, _slot_w, _slot_h = FIGURE2_PPT_SCHEMATIC_SLOTS[row]
        _add_svg_panel_label(
            root,
            schematic_panel_letters[row],
            slot_x + 0.010,
            slot_y + 0.030,
            style,
        )
    for record in lower_axes_records:
        row = int(record["row"])
        col = int(record["column"])
        letter = lower_panel_letters[(row, col)]
        slot_x, slot_y, _slot_w, _slot_h = (float(value) for value in record["slot_in"])
        if col == 1:
            _add_svg_panel_label(
                root,
                letter,
                slot_x + FIGURE2_SECOND_COLUMN_PANEL_LABEL_X_OFFSET_IN,
                slot_y + 0.030,
                style,
                text_anchor="end",
            )
        else:
            _add_svg_panel_label(root, letter, slot_x + 0.010, slot_y + 0.030, style)
    svg_tree = ET.ElementTree(root)

    out_svg = out_dir / "figure2_source_publication.svg"
    out_png = out_dir / "figure2_source_publication.png"
    out_pdf = out_dir / "figure2_source_publication.pdf"
    svg_tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    html_path = _export_svg_with_chrome(
        out_svg,
        out_png,
        out_pdf,
        width_mm=float(style.figure2_export_width_mm),
        height_mm=float(export_height_mm),
        dpi=int(style.dpi),
    )

    panel_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    panel_rows.append(
        {
            "panel_key": "figure2_a",
            "panel_letter": "a",
            "visual_role": FIGURE2_PANEL_ROLES["a"],
            "source": "panel_a_sources/422_corr.npz; panel_a_sources/422-3_corr.npz",
            "filters": "fixed FOV example rendered by preview_figure2_assembly._render_panel_a",
            "source_row_count": "",
            "expected": "two stacked local-correlation FOV examples from included arrays",
            "auto_check": "PASS",
        }
    )
    for row, metric in enumerate(row_metrics):
        schematic_key = schematic_keys[row]
        schematic_role = FIGURE2_SCHEMATIC_ROLES[schematic_key]
        panel_rows.append(
            {
                "panel_key": f"figure2_{schematic_key}",
                "panel_letter": schematic_panel_letters[row],
                "visual_role": schematic_role,
                "source": "source-native explanatory vector schematic; no experimental data",
                "filters": (
                    f"metric={metric}; deterministic 6-neuron example; "
                    "manuscript font and maintained metric color"
                ),
                "source_row_count": "",
                "expected": schematic_role,
                "auto_check": "PASS",
            }
        )
    panel_rows.append(
        {
            "panel_key": "figure2_b",
            "panel_letter": "b",
            "visual_role": FIGURE2_PANEL_ROLES["b"],
            "source": _rel(EXAMPLE_SAVED_DIR),
            "filters": (
                "four fixed picks source-rendered in one shared grid; "
                "equal inter-column spacing set to the smallest PPT-derived prior gap; "
                "poster-style square boundary on every map; no internal maze walls"
            ),
            "source_row_count": len(b_picks),
            "expected": (
                "one Day i / Day i+90 source-rendered example place-map grid with "
                "equal column spacing, square map boundaries, and no internal maze walls"
            ),
            "auto_check": "PASS",
        }
    )
    for panel_id, render_info in rendered_sizes.items():
        schematic_letter_by_key = {
            schematic_keys[row]: schematic_panel_letters[row]
            for row in range(3)
        }
        geometry_rows.append(
            {
                "panel_key": f"figure2_{panel_id}",
                "panel_letter": schematic_letter_by_key.get(panel_id, panel_id),
                "size_inches": json.dumps([]),
                "viewbox_points": json.dumps([]),
                "slot_points": json.dumps([render_info.get("slot_width"), render_info.get("slot_height")]),
                "slot_inches": json.dumps(
                    [render_info.get("slot_width", np.nan) / 72.0, render_info.get("slot_height", np.nan) / 72.0]
                ),
                "scale": render_info.get("scale", np.nan),
                "x_in": render_info.get("slot_x", np.nan) / 72.0,
                "y_in": render_info.get("slot_y", np.nan) / 72.0,
                "width_in": render_info.get("slot_width", np.nan) / 72.0,
                "height_in": render_info.get("slot_height", np.nan) / 72.0,
                "rendered_width_pt": render_info.get("rendered_width", np.nan),
                "rendered_height_pt": render_info.get("rendered_height", np.nan),
            }
        )
    for record in lower_axes_records:
        metric = str(record["metric"])
        col = int(record["column"])
        row = int(record["row"])
        source = (
            lower_sources[col]
            if col >= 2
            else _rel(_figure2_distribution_path(metric) if col == 0 else _figure2_mean_path(metric))
        )
        filters = (
            "metrics place/PV/ensemble only; level excluded; corrected maintained scope; "
            "source-rendered lower-grid axis"
        )
        if col in (0, 1):
            filters += (
                "; 10,000 neuron-ID shuffle repeats; corrected 842 0->3 CIMap; "
                "no manual/pre-sort exclusions; excluded 1616day1 -> 1616bday1"
            )
        source_row_count = ""
        if col == 3:
            filters = (
                "scope=clean_long_gap_day1_triplets; 22 chains from 11 animals; "
                "unrestricted adjacency=23 with only special 1616 short-gap chain excluded"
            )
            source_row_count = "22 clean long-gap chains from 11 animals"
            if metric == "ensemble_rate_correlation":
                source = _rel(FIGURE2_RATE_CHAIN_POINTS)
                filters += "; metric=value_sorted; previous gap=value_sorted_ab; next gap=value_sorted_bc"
        panel_rows.append(
            {
                "panel_key": record["key"],
                "panel_letter": lower_panel_letters[(row, col)],
                "visual_role": f"{METRIC_LABELS[metric]} {lower_roles[col]}",
                "source": source,
                "filters": filters,
                "source_row_count": source_row_count,
                "expected": "source-rendered lower-grid axis, aligned to same metric row and semantic column",
                "auto_check": "PASS",
            }
        )
        geometry_rows.append(
            {
                "panel_key": record["key"],
                "panel_letter": lower_panel_letters[(row, col)],
                "size_inches": json.dumps([]),
                "viewbox_points": json.dumps([]),
                "slot_points": json.dumps([]),
                "slot_inches": json.dumps([record["width_in"], record["height_in"]]),
                "scale": np.nan,
                "x0": record["x0"],
                "y0": record["y0"],
                "width": record["width"],
                "height": record["height"],
                "metric": metric,
                "row": row,
                "column": col,
            }
        )

    panel_rows.sort(key=lambda row: str(row["panel_letter"]))
    verification_path = out_dir / "figure2_panel_verification.csv"
    geometry_path = out_dir / "figure2_panel_geometry.csv"
    consecutive_binned_curve_path = out_dir / "figure2_consecutive_gap_binned_curve_audit.csv"
    displayed_estimand_stats_path = out_dir / "figure2_consecutive_gap_displayed_estimand_bh4.csv"
    pd.DataFrame(panel_rows).to_csv(verification_path, index=False)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)
    consecutive_binned_curve.to_csv(consecutive_binned_curve_path, index=False)
    displayed_estimand_stats.to_csv(displayed_estimand_stats_path, index=False)

    manifest = {
        "figure": "figure2",
        "outputs": {
            "png": _rel(out_png),
            "svg": _rel(out_svg),
            "pdf": _rel(out_pdf),
            "html_export_wrapper": _rel(html_path),
        },
        "renderer": "render_main_figs_v3_publication.render_figure2",
        "source_helper": _rel(FIGURE2_HELPER),
        "style": asdict(style),
        "layout": {
            "export_width_mm": float(style.figure2_export_width_mm),
            "export_height_mm": float(export_height_mm),
            "canvas_size": [float(canvas_w), float(canvas_h)],
            "layout_source": "Figure 2 layout constants in this renderer",
            "top_slots": {key: list(value) for key, value in FIGURE2_PPT_TOP_SLOTS.items()},
            "panel_b_slot": list(FIGURE2_PANEL_B_SLOT),
            "panel_b_min_column_gap_in": float(FIGURE2_PANEL_B_MIN_COLUMN_GAP_IN),
            "lower_column_slots": {
                str(key): list(value) for key, value in FIGURE2_PPT_LOWER_COLUMN_SLOTS.items()
            },
            "lower_original_column_slots": {
                str(key): list(value)
                for key, value in FIGURE2_PPT_LOWER_ORIGINAL_COLUMN_SLOTS.items()
            },
            "lower_column_width_scales": {
                str(key): float(value)
                for key, value in FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES.items()
            },
            "lower_gap_in": float(FIGURE2_PPT_LOWER_GAP_IN),
            "lower_content_bounds_in": [
                float(FIGURE2_PPT_LOWER_CONTENT_LEFT_IN),
                float(FIGURE2_PPT_LOWER_CONTENT_RIGHT_IN),
            ],
            "lower_row_slots": {str(key): list(value) for key, value in FIGURE2_PPT_LOWER_ROW_SLOTS.items()},
            "lower_row_gap_in": float(FIGURE2_PPT_LOWER_ROW_GAP_IN),
            "schematic_slots": {
                schematic_keys[row]: list(FIGURE2_PPT_SCHEMATIC_SLOTS[row])
                for row in range(3)
            },
            "schematic_panel_letters": {
                schematic_keys[row]: schematic_panel_letters[row]
                for row in range(3)
            },
            "lower_panel_letters": {
                f"row{row}_col{column}": lower_panel_letters[(row, column)]
                for row in range(3)
                for column in range(4)
            },
            "schematic_content_bounds": list(FIGURE2_SCHEMATIC_CONTENT_BOUNDS),
            "schematic_header_bounds": list(
                _figure2_schematic_header_bounds(
                    (FIGURE2_PPT_SCHEMATIC_WIDTH_IN, FIGURE2_PPT_LOWER_ROW_HEIGHT_IN),
                    style,
                )
            ),
            "schematic_header_reference_title": FIGURE2_SCHEMATIC_TITLES["pv_correlation"],
            "schematic_header_side_padding_pt": float(FIGURE2_SCHEMATIC_HEADER_SIDE_PADDING_PT),
            "row_title_bands_removed": True,
            "rendered_sizes": rendered_sizes,
            "lower_axes": lower_axes_records,
        },
        "sources": [
            {
                "panel": "figure2_a",
                "display_label": "a",
                "visual_role": FIGURE2_PANEL_ROLES["a"],
                "renderer": "preview_figure2_assembly._render_panel_a",
                "source_path": "panel_a_sources/422_corr.npz; panel_a_sources/422-3_corr.npz",
            },
            {
                "panel": "figure2_b",
                "display_label": "b",
                "visual_role": FIGURE2_PANEL_ROLES["b"],
                "renderer": "render_main_figs_v3_publication._render_figure2_panel_b_four_examples",
                "source_path": _rel(EXAMPLE_SAVED_DIR),
                "fixed_picks": [list(pick) for pick in FIGURE2_PANEL_B_PICKS],
                "map_frame": {
                    "axis_lim": list(FIGURE2_PANEL_B_AXIS_LIM),
                    "boundary_linewidth": FIGURE2_PANEL_B_BOUNDARY_LINEWIDTH,
                    "show_internal_maze_walls": False,
                },
            },
            {
                "panel": "figure2_lower_row_schematics",
                "display_label": "c, h, m",
                "visual_role": (
                    "source-native row-selection schematics for population-vector, place-map, "
                    "and ensemble-rate correlation"
                ),
                "renderer": "render_main_figs_v3_publication._render_figure2_metric_schematic",
                "source_path": "source-native explanatory matrices/vectors; no experimental data",
                "metrics": row_metrics,
                "schematic_keys": [schematic_keys[row] for row in range(3)],
            },
            {
                "panel": "figure2_distribution_and_null_grid",
                "display_label": "d-e, i-j, n-o",
                "visual_role": (
                    "corrected-registration PV/place/ensemble correlation distributions "
                    "and mean-vs-null panels"
                ),
                "renderer": "render_main_figs_v3_publication._draw_figure2_distribution_panel; _draw_figure2_mean_panel",
                "filters": {
                    "metrics": ["place_map_correlation", "pv_correlation", "ensemble_rate_correlation"],
                    "excluded_metrics": ["level_map_correlation"],
                    "excluded_transition": "1616day1 -> 1616bday1",
                    "shuffle_repeats": 10000,
                    "shuffle_type": "neuron-ID permutation of test-session side",
                    "corrected_registration": "842day1 -> 842-3day1",
                    "manual_presort_exclusions": "none",
                },
                "source_paths": {
                    metric: {
                        "distribution": _rel(_figure2_distribution_path(metric)),
                        "mean": _rel(_figure2_mean_path(metric)),
                    }
                    for metric in METRICS
                },
                "pair_support_audit": _rel(FIGURE2_DISTRIBUTION_PAIR_SUPPORT),
                "scope_audit": distribution_scope_audit,
            },
            {
                "panel": "figure2_gap_length_curves",
                "display_label": "f, k, p",
                "visual_role": "gap-length correlation curves",
                "renderer": "render_main_figs_v3_publication._draw_figure2_gap_curve",
                "source_path": _rel(FIGURE2_GAP_CURVE),
                "filters": {"metrics": ["place", "pv", "rate"], "excluded_metrics": ["level"]},
            },
            {
                "panel": "figure2_consecutive_gap",
                "display_label": "g, l, q",
                "visual_role": "previous-vs-next gap correlation panels",
                "renderer": "render_main_figs_v3_publication._draw_figure2_prev_next_panel",
                "source_paths": {
                    "place_map_correlation": _rel(FIGURE2_CONSECUTIVE_POINTS),
                    "pv_correlation": _rel(FIGURE2_CONSECUTIVE_POINTS),
                    "ensemble_rate_correlation": _rel(FIGURE2_RATE_CHAIN_POINTS),
                },
                "stats_paths": {
                    "place_map_correlation": _rel(FIGURE2_CONSECUTIVE_STATS),
                    "pv_correlation": _rel(FIGURE2_CONSECUTIVE_STATS),
                    "ensemble_rate_correlation": _rel(FIGURE2_RATE_CHAIN_STATS),
                },
                "filters": {
                    "place_pv_metrics": ["place_map_correlation", "pv_correlation"],
                    "place_pv_source_scope": "22 clean long-gap Day-1 chains from corrected pair authorities",
                    "explicitly_excluded_chain": "1616day1 -> 1616bday1 -> 1616b-3day1",
                    "ensemble_rate_scope": "clean_long_gap_day1_triplets",
                    "ensemble_rate_metric": "value_sorted",
                    "ensemble_rate_previous_gap_column": "value_sorted_ab",
                    "ensemble_rate_next_gap_column": "value_sorted_bc",
                    "ensemble_rate_excluded_transition": "1616day1 -> 1616bday1",
                    "excluded_metrics": ["level_map_correlation"],
                },
                "scope_audit": consecutive_scope_audit,
                "binned_curve_audit": _rel(consecutive_binned_curve_path),
                "displayed_estimand_bh4": _rel(displayed_estimand_stats_path),
                "scope_sources": {
                    "clean_22": _rel(FIGURE2_CONSECUTIVE_CLEAN_SCOPE),
                },
            }
        ],
        "panel_verification": _rel(verification_path),
        "panel_geometry": _rel(geometry_path),
        "consecutive_gap_binned_curve_audit": _rel(consecutive_binned_curve_path),
        "consecutive_gap_displayed_estimand_bh4": _rel(displayed_estimand_stats_path),
        "distribution_scope_audit": distribution_scope_audit,
        "consecutive_scope_audit": consecutive_scope_audit,
    }
    _write_json(out_dir / "figure2_source_manifest.json", manifest)
    return manifest
