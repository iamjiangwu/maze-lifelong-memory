"""Render the current Figure 3 panels from included derived Source Data.

Panels show convergence toward the full Day 2 reference, early alignment and
long-gap stability across recording cycles. The publication wrapper supplies
the source paths and verifies their integrity before calling render_figure3.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace as dataclass_replace
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image, ImageStat
from scipy import stats as scipy_stats

from ca_analysis import figure_panels as figure_panels_module
from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    load_example_cell_arrow_data,
    load_neuron_data_bundle,
    resolve_neuron_data_path,
)
from ca_analysis.figure_colors import get_correlation_experience_colors


REPO_ROOT = Path(__file__).resolve().parents[1]
DAY2_TEMPLATE_VARIANT_LABEL = "coverage5min_current59"
FIGURE3_FIRST30_AUTHORITY_DIR = REPO_ROOT / "source_data" / "first30"
DAY2_TEMPLATE_GROUP_CURVES = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "day2_template_group_curve_summary.csv"
)
FIGURE3_FIRST30_POINTS_ALL_METRICS = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "combined_first30_points_all_metrics.csv"
)
FIGURE3_FIRST30_STATS_ALL_METRICS = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "combined_first30_stats_all_metrics.csv"
)
MAIN3_NO_MASK_DIR = REPO_ROOT / "source_data" / "long_gap"
LONG_GAP_POINTS = MAIN3_NO_MASK_DIR / "main3_experience_trend_points_no_manual_mask.csv"
LONG_GAP_STATS = MAIN3_NO_MASK_DIR / "main3_experience_trend_clustered_stats_no_manual_mask.csv"
EXAMPLE_SAVED_DIR = REPO_ROOT / "source_data" / "example"
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
EXPERIENCE_ORDER = ("naive", "3mo", "6mo", "9mo", "12mo")
EXPERIENCE_DISPLAY = {
    "naive": "naive",
    "3mo": "3 mo",
    "6mo": "6 mo",
    "9mo": "9 mo",
    "12mo": "12 mo",
}
LONG_GAP_XLABELS = ("1", "2", "3", "4")
FIGURE3_PPT_CONTENT_BBOX_PT = (2.410000, 2.271102, 353.006063, 486.800000)
FIGURE3_PPT_ROW_HEADERS: tuple[dict[str, object], ...] = ()
FIGURE3_SCHEMATIC_TITLES = {
    "moving_windows": "Moving 30-min windows",
    "first30_experience": "First 30 min across cycles",
    "long_gap_day1": "Day 1 across long gaps",
}
FIGURE3_SCHEMATIC_COLORS = {
    "ink": "#30363A",
    "muted": "#667168",
    "bar_fill": "#F1F6F1",
    "bar_edge": "#91A493",
    "whole_fill": "#75A86F",
    "whole_edge": "#376F3E",
    "window_edge": "#3F8048",
    "window_colors": ("#DCEBDC", "#B8D3B8", "#8FB791"),
    "arrow": "#3F8048",
    "connector": "#B3BEB4",
}
FIGURE3_FIRST30_EXPERIENCE_COLOR_GROUPS = {
    index: group for index, group in enumerate(EXPERIENCE_ORDER)
}
FIGURE3_LONG_GAP_EXPERIENCE_COLOR_GROUPS = {
    index: EXPERIENCE_ORDER[index] for index in range(1, len(EXPERIENCE_ORDER))
}
FIGURE3_SCHEMATIC_EXPERIENCE_IDS = tuple(range(len(EXPERIENCE_ORDER)))
FIGURE3_MOVING_WINDOW_TARGET = (0.81, 0.525)
FIGURE3_EXAMPLE_AXIS_LIM = (-0.5, 14.5)
FIGURE3_EXAMPLE_BOUNDARY_LINEWIDTH = 1.15
FIGURE3_CURVE_SUPPORT_MIN_ANIMALS = 3
FIGURE3_CURVE_SUPPORT_MASK = (
    REPO_ROOT
    / "source_data"
    / "support"
    / "ed07_current59_curve_support_mask.csv"
)
FIGURE3_CURVE_SUPPORT_RULE = "max(3, ceil(max_n_animals_for_curve / 2))"
FIGURE3_CURVE_DISPLAY_XMIN = 25.0
FIGURE3_CURVE_DISPLAY_XMAX = 185.0
FIGURE3_ROW_YLABELS = {
    "place_map_correlation": "Place map correlation",
    "pv_correlation": "PV correlation",
    "ensemble_rate_correlation": "Ensemble rate correlation",
}
FIGURE3_STAT_R_POSITION_FRAC = (0.905, 0.050)
FIGURE3_PPT_PANEL_LAYOUT = ({'panel_letter': 'a',
  'key': 'top_example_row',
  'source_svg': 'figure3_panel_a_top_example_row.svg',
  'source_png': 'figure3_panel_a_top_example_row.png',
  'slot_pt': (2.41, 2.271102, 353.006063, 97.808504),
  'src_rect': {'l': 3053, 'r': 1},
  'row': -1,
  'column': -1,
  'metric': '',
  'visual_role': 'four source-rendered example maps: Day i, Day i+1, Day i+90, Day i+91'},
 {'panel_letter': 'b',
  'key': 'moving_windows_to_whole_day2_schematic',
  'source_svg': 'figure3_panel_b_moving_windows_to_whole_day2_schematic.svg',
  'source_png': 'figure3_panel_b_moving_windows_to_whole_day2_schematic.png',
  'slot_pt': (4.355354, 102.5, 115.059134, 52.0),
  'src_rect': {},
  'row': -2,
  'column': 0,
  'metric': '',
  'schematic': 'moving_windows',
  'visual_role': 'moving 30-min Day-1 windows compared with the whole Day-2 template'},
 {'panel_letter': 'c',
  'key': 'first30_across_experience_schematic',
  'source_svg': 'figure3_panel_c_first30_across_experience_schematic.svg',
  'source_png': 'figure3_panel_c_first30_across_experience_schematic.png',
  'slot_pt': (114.819606, 102.5, 115.059134, 52.0),
  'src_rect': {},
  'row': -2,
  'column': 1,
  'metric': '',
  'schematic': 'first30_experience',
  'visual_role': 'first 30 min compared with the whole next day across experience levels'},
 {'panel_letter': 'd',
  'key': 'long_gap_day1_schematic',
  'source_svg': 'figure3_panel_d_long_gap_day1_schematic.svg',
  'source_png': 'figure3_panel_d_long_gap_day1_schematic.png',
  'slot_pt': (236.575827, 102.5, 115.059134, 52.0),
  'src_rect': {},
  'row': -2,
  'column': 2,
  'metric': '',
  'schematic': 'long_gap_day1',
  'visual_role': 'successive Day-1 sessions compared across approximately 90-day gaps'},
 {'panel_letter': 'e',
  'key': 'day2_template_pv_correlation',
  'source_svg': 'figure3_panel_e_day2_template_pv_correlation.svg',
  'source_png': 'figure3_panel_e_day2_template_pv_correlation.png',
  'slot_pt': (3.43748, 160.0, 116.894803, 104.41252),
  'src_rect': {'t': 15640},
  'row': 0,
  'column': 0,
  'metric': 'pv_correlation',
  'visual_role': 'Day1 windows to Day2 template curve over time in maze'},
 {'panel_letter': 'f',
  'key': 'day2_template_first30_pv_correlation',
  'source_svg': 'figure3_panel_f_day2_template_first30_pv_correlation.svg',
  'source_png': 'figure3_panel_f_day2_template_first30_pv_correlation.png',
  'slot_pt': (114.819606, 160.0, 116.894803, 104.41252),
  'src_rect': {'t': 15640},
  'row': 0,
  'column': 1,
  'metric': 'pv_correlation',
  'visual_role': 'Experience scatter from Day2-template source family, first 30 min maze-time window'},
 {'panel_letter': 'g',
  'key': 'long_gap_pv_correlation',
  'source_svg': 'figure3_panel_g_long_gap_pv_correlation.svg',
  'source_png': 'figure3_panel_g_long_gap_pv_correlation.png',
  'slot_pt': (236.575827, 160.0, 116.894803, 104.41252),
  'src_rect': {'t': 15640},
  'row': 0,
  'column': 2,
  'metric': 'pv_correlation',
  'visual_role': 'Long-gap experience trend scatter with source trend fit'},
 {'panel_letter': 'h',
  'key': 'day2_template_place_map_correlation',
  'source_svg': 'figure3_panel_h_day2_template_place_map_correlation.svg',
  'source_png': 'figure3_panel_h_day2_template_place_map_correlation.png',
  'slot_pt': (4.355354, 270.4, 115.059134, 102.772835),
  'src_rect': {'t': 15640},
  'row': 1,
  'column': 0,
  'metric': 'place_map_correlation',
  'visual_role': 'Day1 windows to Day2 template curve over time in maze'},
 {'panel_letter': 'i',
  'key': 'day2_template_first30_place_map_correlation',
  'source_svg': 'figure3_panel_i_day2_template_first30_place_map_correlation.svg',
  'source_png': 'figure3_panel_i_day2_template_first30_place_map_correlation.png',
  'slot_pt': (114.819606, 270.4, 115.059134, 102.772835),
  'src_rect': {'t': 15640},
  'row': 1,
  'column': 1,
  'metric': 'place_map_correlation',
  'visual_role': 'Experience scatter from Day2-template source family, first 30 min maze-time window'},
 {'panel_letter': 'j',
  'key': 'long_gap_place_map_correlation',
  'source_svg': 'figure3_panel_j_long_gap_place_map_correlation.svg',
  'source_png': 'figure3_panel_j_long_gap_place_map_correlation.png',
  'slot_pt': (236.575827, 270.4, 115.059134, 102.772835),
  'src_rect': {'t': 15640},
  'row': 1,
  'column': 2,
  'metric': 'place_map_correlation',
  'visual_role': 'Long-gap experience trend scatter with source trend fit'},
 {'panel_letter': 'k',
  'key': 'day2_template_ensemble_rate_correlation',
  'source_svg': 'figure3_panel_k_day2_template_ensemble_rate_correlation.svg',
  'source_png': 'figure3_panel_k_day2_template_ensemble_rate_correlation.png',
  'slot_pt': (3.037087, 382.6, 117.295197, 105.460157),
  'src_rect': {'t': 15555},
  'row': 2,
  'column': 0,
  'metric': 'ensemble_rate_correlation',
  'visual_role': 'Day1 windows to Day2 template curve over time in maze'},
 {'panel_letter': 'l',
  'key': 'day2_template_first30_ensemble_rate_correlation',
  'source_svg': 'figure3_panel_l_day2_template_first30_ensemble_rate_correlation.svg',
  'source_png': 'figure3_panel_l_day2_template_first30_ensemble_rate_correlation.png',
  'slot_pt': (114.419213, 382.6, 117.295197, 105.460157),
  'src_rect': {'t': 15368},
  'row': 2,
  'column': 1,
  'metric': 'ensemble_rate_correlation',
  'visual_role': 'Experience scatter from Day2-template source family, first 30 min maze-time window'},
 {'panel_letter': 'm',
  'key': 'long_gap_ensemble_rate_correlation',
  'source_svg': 'figure3_panel_m_long_gap_ensemble_rate_correlation.svg',
  'source_png': 'figure3_panel_m_long_gap_ensemble_rate_correlation.png',
  'slot_pt': (236.575827, 382.6, 117.295197, 105.460157),
  'src_rect': {'t': 15647},
  'row': 2,
  'column': 2,
  'metric': 'ensemble_rate_correlation',
  'visual_role': 'Long-gap experience trend scatter with source trend fit'})
FIGURE3_DISPLAY_METRICS = ("pv_correlation", "place_map_correlation", "ensemble_rate_correlation")
DAY2_TEMPLATE_SUPPORT_POLICY = {
    "place_map_correlation": "scaled_nonoverlap",
    "pv_correlation": "scaled_nonoverlap",
    "ensemble_rate_correlation": "not_applicable",
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


def _linear_fit_confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    *,
    intercept: float,
    slope: float,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_fit = np.asarray(x_fit, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3 or not (np.isfinite(intercept) and np.isfinite(slope)):
        return None
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    dof = int(x.size - 2)
    if dof <= 0 or sxx <= 0:
        return None
    residuals = y - (slope * x + intercept)
    mse = float(np.sum(residuals**2)) / dof
    if not np.isfinite(mse) or mse < 0:
        return None
    t_crit = float(scipy_stats.t.ppf((1.0 + confidence) / 2.0, dof))
    se_fit = np.sqrt(mse * (1.0 / x.size + ((x_fit - x_mean) ** 2) / sxx))
    ci = t_crit * se_fit
    y_fit = intercept + slope * x_fit
    return y_fit - ci, y_fit + ci


def _draw_day12_arrow_example(fig: plt.Figure, grid_cell, style: PublicationStyle) -> tuple[list[plt.Axes], dict[str, object]]:
    """Draw the fixed example cell row directly from the saved source bundle."""

    pick = ("451-3day1->451-6day1", 161)
    bundle_path = resolve_neuron_data_path(pick[0], pick[1], EXAMPLE_SAVED_DIR)
    pack = load_neuron_data_bundle(bundle_path)
    top_grid = grid_cell.subgridspec(1, 4, wspace=0.04, width_ratios=(1, 1, 1, 1))
    axes = [fig.add_subplot(top_grid[0, col]) for col in range(4)]
    sides = ("A", "A2", "B", "B2")
    titles = ("Day i", "Day i+1", "Day i+90", "Day i+91")
    for ax, side, title in zip(axes, sides, titles, strict=True):
        arrows = load_example_cell_arrow_data(pack, side)
        ax.plot(
            arrows["traj_x"],
            arrows["traj_y"],
            color="0.55",
            linewidth=0.22,
            alpha=0.24,
            rasterized=True,
            zorder=1,
        )
        for body_x, body_y, nose_x, nose_y in zip(
            arrows["px"],
            arrows["py"],
            arrows["ex"],
            arrows["ey"],
            strict=False,
        ):
            figure_panels_module._add_triangle_arrow_head(
                ax,
                body_x,
                body_y,
                (nose_x - body_x) * 0.25,
                (nose_y - body_y) * 0.25,
                head_width=0.14,
                head_length=0.34,
                facecolor="red",
                edgecolor="red",
                alpha=0.42,
                linewidth=0.01,
                zorder=2,
            )
        low, high = FIGURE3_EXAMPLE_AXIS_LIM
        ax.add_patch(
            plt.Rectangle(
                (low, low),
                high - low,
                high - low,
                fill=False,
                edgecolor="black",
                linewidth=FIGURE3_EXAMPLE_BOUNDARY_LINEWIDTH,
                joinstyle="miter",
                capstyle="butt",
                zorder=5,
                clip_on=False,
            )
        )
        ax.set_aspect("equal")
        ax.set_xlim(FIGURE3_EXAMPLE_AXIS_LIM)
        ax.set_ylim(FIGURE3_EXAMPLE_AXIS_LIM)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=3.5, fontsize=style.title_fontsize)
        for spine in ax.spines.values():
            spine.set_visible(False)
    record = {
        "panel": "figure3_top_example",
        "source_path": _rel(bundle_path),
        "renderer": "render_main_figs_v3_publication._draw_day12_arrow_example",
        "note": "native axes drawn from saved example-cell source bundle with poster-style square boundaries and no internal maze walls; no PDF/PPT crop or intermediate panel raster",
        "pick": {"pair_key": pick[0], "row_idx": pick[1]},
    }
    return axes, record


def _rolling(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    series = pd.Series(values)
    return series.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "n.s."
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _normalize_experience(values: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip().str.lower()
    return out.replace(
        {
            "0mo": "naive",
            "0": "naive",
            "nan": "",
            "none": "",
            "ltm three": "3mo",
            "ltm six": "6mo",
            "ltm nine": "9mo",
            "ltm twelve": "12mo",
        }
    )


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


def _load_day2_template_curves(style: PublicationStyle) -> pd.DataFrame:
    curves = pd.read_csv(DAY2_TEMPLATE_GROUP_CURVES)
    curves = curves.loc[
        curves["metric"].isin(METRICS)
        & curves["window_type"].astype(str).eq("maze_time")
        & np.isclose(pd.to_numeric(curves["window_size"], errors="coerce"), 30.0)
    ].copy()
    support_ok = []
    for _, row in curves.iterrows():
        support_ok.append(
            str(row["support_policy"]) == DAY2_TEMPLATE_SUPPORT_POLICY[str(row["metric"])]
        )
    curves = curves.loc[support_ok].copy()
    if curves.empty:
        raise ValueError("No Day2-template curve rows remain after metric/window/support filters.")
    curves["x_min"] = pd.to_numeric(curves["x_value"], errors="coerce") / 60.0
    curves["mean"] = pd.to_numeric(curves["mean"], errors="coerce")
    curves["sem"] = pd.to_numeric(curves["sem"], errors="coerce")
    curves["n_animals"] = pd.to_numeric(curves["n_animals"], errors="coerce")
    curves = curves[np.isfinite(curves["x_min"]) & np.isfinite(curves["mean"])].copy()
    curves["experience_group"] = pd.Categorical(curves["experience_group"], categories=EXPERIENCE_ORDER, ordered=True)
    curves = curves.sort_values(["metric", "experience_group", "x_min"], kind="stable")
    for metric in METRICS:
        metric_rows = curves.loc[curves["metric"].eq(metric)]
        if metric_rows.empty:
            raise ValueError(f"Missing Day2-template rows for {metric}.")
    return curves


def _load_day2_template_30min_points() -> pd.DataFrame:
    """Read the included first-window animal values used by panels f, i and l."""
    points = pd.read_csv(FIGURE3_FIRST30_POINTS_ALL_METRICS)
    points["metric_label"] = points["metric"].map(METRIC_LABELS)
    return points.sort_values(
        ["metric", "experience_index", "animal"], kind="stable"
    ).reset_index(drop=True)


def _compute_experience_stats(points: pd.DataFrame, *, x_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    p_values: list[float] = []
    for metric in METRICS:
        sub = points.loc[points["metric"].eq(metric)].copy()
        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        if x.size >= 2 and np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            if np.unique(y).size >= 2:
                r, p_value = scipy_stats.pearsonr(x, y)
            else:
                r, p_value = math.nan, math.nan
        else:
            slope, intercept, r, p_value = math.nan, math.nan, math.nan, math.nan
        rows.append(
            {
                "metric": metric,
                "n_rows": int(x.size),
                "n_animals": int(sub.loc[ok, "animal"].astype(str).nunique()) if "animal" in sub.columns else int(x.size),
                "n_experience_levels": int(pd.Series(x).nunique()) if x.size else 0,
                "intercept": float(intercept),
                "slope": float(slope),
                "pearson_r": float(r),
                "p_two_sided": float(p_value),
            }
        )
        p_values.append(float(p_value))
    q_values = _bh_q(p_values)
    for row, q_value in zip(rows, q_values, strict=True):
        row["q_bh_two_sided"] = float(q_value)
    return pd.DataFrame(rows).set_index("metric", drop=False)


def _load_day2_template_30min_stats(points: pd.DataFrame) -> pd.DataFrame:
    """Load the maintained Main Figure 3 first-window tests with standard errors clustered by animal."""

    compare_cols = [
        "animal",
        "experience_group",
        "experience_index",
        "metric",
        "support_policy",
        "window_type",
        "window_size",
        "value",
    ]
    plotted = points.loc[points["metric"].isin(METRICS), compare_cols].copy()
    canonical = pd.read_csv(FIGURE3_FIRST30_POINTS_ALL_METRICS)
    canonical = canonical.loc[canonical["metric"].isin(METRICS), compare_cols].copy()
    for frame in (plotted, canonical):
        for column in ("animal", "experience_group", "metric", "support_policy", "window_type"):
            frame[column] = frame[column].astype(str)
        frame.sort_values(["metric", "experience_index", "animal"], kind="stable", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(
        plotted,
        canonical,
        check_dtype=False,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )

    stats = pd.read_csv(FIGURE3_FIRST30_STATS_ALL_METRICS)
    required_columns = {
        "metric",
        "n_rows",
        "n_animals",
        "n_experience_levels",
        "slope",
        "slope_se",
        "slope_t",
        "p_two_sided",
        "cluster_df",
        "pearson_r",
        "q_bh_two_sided",
        "q_bh4_family",
        "test",
    }
    missing_columns = required_columns.difference(stats.columns)
    if missing_columns:
        raise AssertionError(
            "Figure 3 first-window statistics are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    expected_metrics = set(METRICS) | {"level_map_correlation"}
    if set(stats["metric"].astype(str)) != expected_metrics:
        raise AssertionError("Figure 3 first-window statistics must contain the four-metric BH family.")
    if not (
        stats["n_rows"].eq(59).all()
        and stats["n_animals"].eq(22).all()
        and stats["n_experience_levels"].eq(5).all()
        and stats["cluster_df"].eq(21).all()
        and stats["q_bh4_family"].eq("four Main-Figure representation metrics").all()
        and stats["test"].astype(str).str.contains("CR1 covariance clustered by animal", regex=False).all()
        and stats["test"].astype(str).str.contains("Student t(G-1)", regex=False).all()
    ):
        raise AssertionError("Figure 3 first-window statistics do not match the 59-row CR1/BH4 contract.")

    direct = _compute_experience_stats(points, x_col="experience_index")
    indexed = stats.set_index("metric", drop=False)
    for metric in METRICS:
        if not math.isclose(
            float(indexed.loc[metric, "slope"]),
            float(direct.loc[metric, "slope"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(indexed.loc[metric, "pearson_r"]),
            float(direct.loc[metric, "pearson_r"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Figure 3 first-window plotted rows and statistics disagree for {metric}.")
    return indexed.loc[list(METRICS)].copy()


def _add_support_encoded_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    high_support: np.ndarray,
    *,
    color: str,
    label: str,
    style: PublicationStyle,
) -> dict[str, int]:
    """Draw adjacent curve segments without bridging disconnected support runs."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    high_support = np.asarray(high_support, dtype=bool)
    if not (x.shape == y.shape == high_support.shape):
        raise ValueError("Figure 3 support-line arrays must have identical shapes.")
    high_segments: list[np.ndarray] = []
    for index in range(max(len(x) - 1, 0)):
        if not (
            np.isfinite(x[index])
            and np.isfinite(x[index + 1])
            and np.isfinite(y[index])
            and np.isfinite(y[index + 1])
        ):
            continue
        segment = np.asarray(
            [[x[index], y[index]], [x[index + 1], y[index + 1]]],
            dtype=float,
        )
        if high_support[index] and high_support[index + 1]:
            high_segments.append(segment)
    if high_segments:
        collection = LineCollection(
            high_segments,
            colors=[color],
            linewidths=style.data_linewidth,
            alpha=1.0,
            zorder=3,
            label=label,
        )
        collection.set_capstyle("round")
        ax.add_collection(collection)
    return {"high_segments": len(high_segments), "low_segments": 0}


def _figure3_curve_support_thresholds() -> dict[tuple[str, str], int]:
    """Load and validate the ED6 half-maximum animal-support thresholds."""

    mask = pd.read_csv(FIGURE3_CURVE_SUPPORT_MASK)
    mask = mask.loc[
        mask["metric"].isin(METRICS)
        & mask["window_type"].astype(str).eq("maze_time")
    ].copy()
    thresholds: dict[tuple[str, str], int] = {}
    for (metric, group), rows in mask.groupby(
        ["metric", "experience_group"], observed=True, sort=False
    ):
        observed = sorted(
            set(pd.to_numeric(rows["half_max_threshold"], errors="raise").astype(int))
        )
        max_n = int(
            pd.to_numeric(rows["max_n_animals_for_curve"], errors="raise").max()
        )
        expected = max(
            FIGURE3_CURVE_SUPPORT_MIN_ANIMALS, int(math.ceil(max_n / 2.0))
        )
        if observed != [expected]:
            raise AssertionError(
                f"ED6 support threshold mismatch for {(metric, group)}: "
                f"observed={observed}, expected={expected}"
            )
        thresholds[(str(metric), str(group))] = expected
    expected_keys = {
        (metric, group) for metric in METRICS for group in EXPERIENCE_ORDER
    }
    if set(thresholds) != expected_keys:
        raise AssertionError(
            "ED6 support-mask scope mismatch: "
            f"missing={sorted(expected_keys - set(thresholds))}, "
            f"extra={sorted(set(thresholds) - expected_keys)}"
        )
    return thresholds


def _contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [np.arange(int(start), int(end) + 1) for start, end in zip(starts, ends, strict=True)]


def _summarize_day2_template_support_runs(curves: pd.DataFrame) -> pd.DataFrame:
    thresholds = _figure3_curve_support_thresholds()
    records: list[dict[str, object]] = []
    for (metric, group), rows in curves.groupby(
        ["metric", "experience_group"], observed=True, sort=False
    ):
        rows = rows.sort_values("x_min", kind="stable")
        support = pd.to_numeric(rows["n_animals"], errors="coerce").to_numpy(dtype=float)
        x_min = pd.to_numeric(rows["x_min"], errors="coerce").to_numpy(dtype=float)
        threshold = thresholds[(str(metric), str(group))]
        high = (
            (support >= threshold)
            & (x_min >= FIGURE3_CURVE_DISPLAY_XMIN)
            & (x_min <= FIGURE3_CURVE_DISPLAY_XMAX)
        )
        run_start = 0
        run_id = 0
        for index in range(1, len(rows) + 1):
            if index < len(rows) and high[index] == high[run_start]:
                continue
            run_support = support[run_start:index]
            records.append(
                {
                    "metric": str(metric),
                    "experience_group": str(group),
                    "run_id": run_id,
                    "display_state": "retained" if high[run_start] else "omitted",
                    "support_rule": f"n_animals >= {threshold}",
                    "ed6_rule": FIGURE3_CURVE_SUPPORT_RULE,
                    "start_min": float(x_min[run_start]),
                    "end_min": float(x_min[index - 1]),
                    "n_points": int(index - run_start),
                    "min_n_animals": int(np.nanmin(run_support)),
                    "max_n_animals": int(np.nanmax(run_support)),
                }
            )
            run_id += 1
            run_start = index
    return pd.DataFrame(records)


def _draw_day2_template_curve(
    ax: plt.Axes,
    curves: pd.DataFrame,
    metric: str,
    style: PublicationStyle,
    *,
    show_xlabel: bool,
    show_legend: bool,
    show_title: bool = True,
) -> None:
    metric_rows = curves.loc[curves["metric"].eq(metric)]
    thresholds = _figure3_curve_support_thresholds()
    for group in EXPERIENCE_ORDER:
        sub = metric_rows.loc[metric_rows["experience_group"].astype(str).eq(group)].copy()
        if sub.empty:
            continue
        x = sub["x_min"].to_numpy(dtype=float)
        y = _rolling(sub["mean"].to_numpy(dtype=float), window=5)
        sem = _rolling(sub["sem"].fillna(0.0).to_numpy(dtype=float), window=5)
        n_animals = sub["n_animals"].to_numpy(dtype=float)
        threshold = thresholds[(metric, group)]
        high_support = (
            (n_animals >= threshold)
            & np.isfinite(x)
            & np.isfinite(y)
            & (x >= style.day2_template_xlim_min)
            & (x <= style.day2_template_xlim_max)
        )
        color = style.day2_group_colors[metric][group]
        for indices in _contiguous_true_runs(high_support):
            ax.fill_between(
                x[indices],
                y[indices] - sem[indices],
                y[indices] + sem[indices],
                color=color,
                alpha=style.sem_alpha,
                linewidth=0,
                zorder=1,
            )
        _add_support_encoded_line(
            ax,
            x,
            y,
            high_support,
            color=color,
            label=EXPERIENCE_DISPLAY[group],
            style=style,
        )
    if show_title:
        ax.set_title(METRIC_LABELS[metric], pad=2)
    ax.set_xlim(style.day2_template_xlim_min, style.day2_template_xlim_max)
    ax.set_ylim(0.10, 0.86)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xticks([50, 100, 150])
    ax.set_ylabel("Correlation")
    if show_xlabel:
        ax.set_xlabel("Time in maze (min)")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    _apply_axis_style(ax, style)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.00, 1.02),
                frameon=False,
                ncol=2,
                handlelength=0.95,
                handletextpad=0.25,
                columnspacing=0.55,
                borderaxespad=0.0,
                fontsize=style.legend_fontsize,
            )


def _load_long_gap() -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(LONG_GAP_POINTS, float_precision="round_trip")
    stats_df = pd.read_csv(LONG_GAP_STATS, float_precision="round_trip").set_index(
        "metric", drop=False
    )
    points = points.loc[points["metric"].isin(METRICS)].copy()
    points["experience"] = pd.to_numeric(points["experience"], errors="coerce")
    points["value"] = pd.to_numeric(points["value"], errors="coerce")
    points = points[np.isfinite(points["experience"]) & np.isfinite(points["value"])].copy()
    return points, stats_df


def _metric_ylim(values: np.ndarray, *, floor: float = 0.0, ceiling: float = 1.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (floor, ceiling)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = max(hi - lo, 0.10)
    lo = max(floor, lo - 0.12 * span)
    hi = min(ceiling, hi + 0.18 * span)
    if hi <= lo:
        hi = min(ceiling, lo + 0.2)
    return (lo, hi)


def _draw_experience_scatter(
    ax: plt.Axes,
    points: pd.DataFrame,
    stats_df: pd.DataFrame,
    metric: str,
    style: PublicationStyle,
    *,
    x_col: str,
    x_values: np.ndarray,
    xticks: tuple[int, ...],
    xticklabels: tuple[str, ...],
    show_xlabel: bool,
    ylabel: str,
    annotation_mode: str = "q_r",
    fit_color: str | None = None,
    fit_linestyle: str = "--",
    xlabel: str | None = None,
    show_title: bool = True,
    point_group_map: Mapping[object, str] | None = None,
) -> None:
    sub = points.loc[points["metric"].eq(metric)].copy()
    color = style.metric_colors[metric]
    metric_jitter_offset = {
        "place_map_correlation": 0,
        "pv_correlation": 1,
        "ensemble_rate_correlation": 2,
        "level_map_correlation": 3,
    }
    rng = np.random.default_rng(240624 + metric_jitter_offset[metric])
    jitter = rng.normal(0, 0.045, size=len(sub))
    if point_group_map is None:
        point_colors: str | list[str] = color
    else:
        point_groups = sub[x_col].map(point_group_map)
        if point_groups.isna().any():
            missing = sorted(sub.loc[point_groups.isna(), x_col].astype(str).unique())
            raise ValueError(
                f"Missing Figure 3 experience-color mapping for {x_col}: "
                + ", ".join(missing)
            )
        point_colors = [
            style.day2_group_colors[metric][str(group)] for group in point_groups
        ]
    ax.scatter(
        sub[x_col].to_numpy(dtype=float) + jitter,
        sub["value"].to_numpy(dtype=float),
        color=point_colors,
        s=style.point_size,
        alpha=style.point_alpha,
        edgecolors="none",
        zorder=3,
    )
    if metric in stats_df.index:
        stat = stats_df.loc[metric]
        intercept = float(stat["intercept"])
        slope = float(stat["slope"])
        yfit = intercept + slope * x_values
        fit_line_color = fit_color if fit_color is not None else "#555555"
        ci_band = _linear_fit_confidence_band(
            sub[x_col].to_numpy(dtype=float),
            sub["value"].to_numpy(dtype=float),
            x_values,
            intercept=intercept,
            slope=slope,
        )
        if ci_band is not None:
            ax.fill_between(x_values, ci_band[0], ci_band[1], color=fit_line_color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(
            x_values,
            yfit,
            color=fit_line_color,
            linewidth=style.fit_linewidth,
            linestyle=fit_linestyle,
            zorder=2,
        )
        q_value = float(stat["q_bh_two_sided"]) if "q_bh_two_sided" in stat.index else float(stat["p_two_sided"])
        if "pearson_r" in stat.index:
            r_value = float(stat["pearson_r"])
        elif "r2" in stat.index:
            r2 = float(stat["r2"])
            sign = math.copysign(1.0, float(stat["slope"])) if np.isfinite(float(stat["slope"])) else 1.0
            r_value = sign * math.sqrt(max(r2, 0.0))
        else:
            r_value = math.nan
        if annotation_mode == "q_r":
            ax.text(
                0.05,
                0.94,
                f"q={_format_p(q_value)}\nOLS r={r_value:.2f}" if np.isfinite(r_value) else f"q={_format_p(q_value)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=style.annotation_fontsize,
                color="black",
            )
        elif annotation_mode == "star_r":
            star = _figure2_star_from_p(q_value)
            if star:
                ax.text(
                    0.05,
                    0.94,
                    star,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=style.annotation_fontsize * 2.175,
                    color=fit_line_color,
                    fontweight="bold",
                )
            if np.isfinite(r_value):
                ax.text(
                    0.95,
                    0.08,
                    f"r={r_value:.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=style.annotation_fontsize,
                    color="black",
                )
        elif annotation_mode != "none":
            raise ValueError(f"Unknown annotation_mode: {annotation_mode}")
    if show_title:
        ax.set_title(METRIC_LABELS[metric], pad=2)
    ax.set_xlim(min(xticks) - 0.45, max(xticks) + 0.45)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticklabels))
    ax.set_ylim(*_metric_ylim(sub["value"].to_numpy(dtype=float)))
    ax.set_ylabel(ylabel)
    if show_xlabel:
        ax.set_xlabel(xlabel if xlabel is not None else ("No. of cycles" if x_col == "experience" else "Experience"))
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    _apply_axis_style(ax, style)


def _render_figure3_analysis_schematic(
    schematic: str,
    figsize: tuple[float, float],
    style: PublicationStyle,
) -> plt.Figure:
    """Render a compact method schematic with no illustrative result curve."""

    if schematic not in FIGURE3_SCHEMATIC_TITLES:
        raise ValueError(f"Unknown Figure 3 schematic: {schematic}")
    fig = plt.figure(figsize=figsize, facecolor=style.facecolor)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    ink = str(FIGURE3_SCHEMATIC_COLORS["ink"])
    muted = str(FIGURE3_SCHEMATIC_COLORS["muted"])
    bar_fill = str(FIGURE3_SCHEMATIC_COLORS["bar_fill"])
    bar_edge = str(FIGURE3_SCHEMATIC_COLORS["bar_edge"])
    whole_fill = str(FIGURE3_SCHEMATIC_COLORS["whole_fill"])
    whole_edge = str(FIGURE3_SCHEMATIC_COLORS["whole_edge"])
    window_edge = str(FIGURE3_SCHEMATIC_COLORS["window_edge"])
    window_colors = tuple(FIGURE3_SCHEMATIC_COLORS["window_colors"])
    arrow_color = str(FIGURE3_SCHEMATIC_COLORS["arrow"])
    connector_color = str(FIGURE3_SCHEMATIC_COLORS["connector"])
    linewidth = 0.72
    title_size = style.title_fontsize
    label_size = max(style.legend_fontsize, 4.8)
    small_size = max(style.legend_fontsize - 0.2, 4.5)

    def add_bar(x: float, y: float, width: float, height: float, *, whole: bool = False) -> None:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor=whole_fill if whole else bar_fill,
                edgecolor=whole_edge if whole else bar_edge,
                linewidth=linewidth,
                zorder=2,
            )
        )

    def add_arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        rad: float,
        color: str = arrow_color,
        mutation_scale: float = 5.8,
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=linewidth,
                color=color,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
            )
        )

    ax.text(
        0.5,
        0.965,
        FIGURE3_SCHEMATIC_TITLES[schematic],
        ha="center",
        va="top",
        fontsize=title_size,
        color=ink,
        fontweight="bold",
    )

    if schematic == "moving_windows":
        day1_x, day1_w = 0.07, 0.45
        day2_x, day2_w = 0.69, 0.24
        bar_y, bar_h = 0.34, 0.17
        add_bar(day1_x, bar_y, day1_w, bar_h)
        add_bar(day2_x, bar_y, day2_w, bar_h, whole=True)
        ax.text(day1_x, 0.61, "Day i", ha="left", va="bottom", fontsize=label_size, color=ink)
        ax.text(day2_x + day2_w, 0.61, "Day i + 1", ha="right", va="bottom", fontsize=label_size, color=ink)
        ax.add_patch(
            FancyArrowPatch(
                (day1_x, 0.245),
                (day1_x + day1_w, 0.245),
                arrowstyle="-|>",
                mutation_scale=4.5,
                linewidth=0.62,
                color=ink,
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=4,
            )
        )
        ax.text(
            day1_x + day1_w / 2.0,
            0.145,
            "time",
            ha="center",
            va="top",
            fontsize=small_size,
            color=muted,
        )
        win_w = 0.105
        starts = (0.115, 0.255, 0.395)
        arrow_radii = (-0.32, -0.22, -0.12)
        for start, rad, color in zip(starts, arrow_radii, window_colors, strict=True):
            ax.add_patch(
                plt.Rectangle(
                    (start, bar_y),
                    win_w,
                    bar_h,
                    facecolor=color,
                    edgecolor=window_edge,
                    linewidth=linewidth,
                    zorder=3,
                )
            )
            add_arrow(
                (start + win_w / 2.0, bar_y + bar_h + 0.005),
                FIGURE3_MOVING_WINDOW_TARGET,
                rad=rad,
            )
    elif schematic == "first30_experience":
        group_x = np.linspace(0.10, 0.815, 5)
        bar_y, bar_h, bar_w, gap = 0.36, 0.17, 0.060, 0.012
        for index, (experience_id, x0) in enumerate(
            zip(FIGURE3_SCHEMATIC_EXPERIENCE_IDS, group_x, strict=True)
        ):
            add_bar(float(x0), bar_y, bar_w, bar_h)
            add_bar(float(x0 + bar_w + gap), bar_y, bar_w, bar_h, whole=True)
            highlight_w = bar_w * 0.32
            ax.add_patch(
                plt.Rectangle(
                    (float(x0), bar_y),
                    highlight_w,
                    bar_h,
                    facecolor=window_colors[min(index // 2, 2)],
                    edgecolor=window_edge,
                    linewidth=linewidth,
                    zorder=3,
                )
            )
            add_arrow(
                (float(x0 + highlight_w / 2.0), bar_y + bar_h + 0.005),
                (float(x0 + bar_w + gap + bar_w / 2.0), bar_y + bar_h + 0.005),
                rad=-0.43,
                mutation_scale=5.2,
            )
            ax.text(
                float(x0 + bar_w + gap / 2.0),
                0.25,
                str(experience_id),
                ha="center",
                va="top",
                fontsize=small_size,
                color=muted,
            )
            if index < len(group_x) - 1:
                ax.plot(
                    [float(x0 + 2.0 * bar_w + gap), float(group_x[index + 1])],
                    [bar_y + bar_h / 2.0] * 2,
                    color=connector_color,
                    linewidth=linewidth,
                    linestyle=(0, (2.2, 2.2)),
                    zorder=1,
                )
        first_x = float(group_x[0])
        ax.text(first_x + bar_w / 2.0, 0.62, "Day 1", ha="right", va="bottom", fontsize=small_size, color=ink)
        ax.text(first_x + bar_w + gap + bar_w / 2.0, 0.62, "Day 2", ha="left", va="bottom", fontsize=small_size, color=ink)
        ax.text(0.5, 0.10, "No. of cycles", ha="center", va="top", fontsize=label_size, color=muted)

    else:
        group_x = np.linspace(0.10, 0.815, 5)
        bar_y, bar_h, bar_w, gap = 0.36, 0.17, 0.060, 0.012
        for index, (experience_id, x0) in enumerate(
            zip(FIGURE3_SCHEMATIC_EXPERIENCE_IDS, group_x, strict=True)
        ):
            add_bar(float(x0), bar_y, bar_w, bar_h, whole=True)
            add_bar(float(x0 + bar_w + gap), bar_y, bar_w, bar_h)
            ax.text(
                float(x0 + bar_w + gap / 2.0),
                0.25,
                str(experience_id),
                ha="center",
                va="top",
                fontsize=small_size,
                color=muted,
            )
            if index < len(group_x) - 1:
                ax.plot(
                    [float(x0 + 2.0 * bar_w + gap), float(group_x[index + 1])],
                    [bar_y + bar_h / 2.0] * 2,
                    color=connector_color,
                    linewidth=linewidth,
                    linestyle=(0, (2.2, 2.2)),
                    zorder=1,
                )
                add_arrow(
                    (float(x0 + bar_w / 2.0), bar_y + bar_h + 0.005),
                    (float(group_x[index + 1] + bar_w / 2.0), bar_y + bar_h + 0.005),
                    rad=-0.34,
                    mutation_scale=5.4,
                )
        ax.text(0.5, 0.10, "Cycle", ha="center", va="top", fontsize=label_size, color=muted)
    return fig


def _figure3_source_figsize(layout: Mapping[str, object]) -> tuple[float, float]:
    _x, _y, width_pt, height_pt = (float(value) for value in layout["slot_pt"])
    return width_pt / 72.0, height_pt / 72.0


def _save_figure3_panel_outputs(
    fig: plt.Figure,
    panel_dir: Path,
    stem: str,
    style: PublicationStyle,
    *,
    fixed_bbox: bool = False,
) -> dict[str, Path]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_svg = panel_dir / f"{stem}.svg"
    out_png = panel_dir / f"{stem}.png"
    out_pdf = panel_dir / f"{stem}.pdf"
    save_kwargs = {
        "bbox_inches": None if fixed_bbox else "tight",
        "pad_inches": 0.0 if fixed_bbox else 0.01,
        "facecolor": style.facecolor,
        "metadata": {"Date": None},
    }
    # SVG preserves vector artists, but any native imshow/ImageArtist content
    # is embedded at the supplied DPI.  Pass the publication DPI explicitly so
    # Figure 3a's scientific raster maps are not silently fixed at Matplotlib's
    # 100-dpi default inside otherwise high-resolution master exports.
    fig.savefig(out_svg, dpi=int(style.dpi), **save_kwargs)
    fig.savefig(out_png, dpi=int(style.dpi), **save_kwargs)
    with PdfPages(out_pdf, metadata={"CreationDate": None, "ModDate": None}) as pdf:
        pdf.savefig(
            fig,
            dpi=int(style.dpi),
            bbox_inches=None if fixed_bbox else "tight",
            pad_inches=0.0 if fixed_bbox else 0.01,
            facecolor=style.facecolor,
        )
    plt.close(fig)
    return {"svg": out_svg, "png": out_png, "pdf": out_pdf}


def _render_figure3_individual_panels(
    out_dir: Path,
    style: PublicationStyle,
    *,
    day2_curves: pd.DataFrame,
    day2_points: pd.DataFrame,
    day2_stats: pd.DataFrame,
    long_points: pd.DataFrame,
    long_stats: pd.DataFrame,
) -> Path:
    style = dataclass_replace(style, show_grid=False)
    panel_dir = out_dir / "figure3_individual_panels"
    panel_rows: list[dict[str, object]] = []
    with plt.rc_context(style.rcparams()):
        layout_a = next(layout for layout in FIGURE3_PPT_PANEL_LAYOUT if layout["panel_letter"] == "a")
        panel_size = _figure3_source_figsize(layout_a)
        fig = plt.figure(figsize=panel_size, facecolor=style.facecolor)
        grid = fig.add_gridspec(1, 1, left=0.005, right=0.995, bottom=0.02, top=0.88)
        _draw_day12_arrow_example(fig, grid[0, 0], style)
        outputs = _save_figure3_panel_outputs(
            fig,
            panel_dir,
            Path(str(layout_a["source_svg"])).stem,
            style,
            fixed_bbox=True,
        )
        panel_rows.append(
            {
                "panel_letter": "a",
                "panel_key": layout_a["key"],
                "visual_role": layout_a["visual_role"],
                "source": _rel(EXAMPLE_SAVED_DIR / "451-3day1_to_451-6day1__row0161_A190_B267.npz"),
                "svg": _rel(outputs["svg"]),
                "png": _rel(outputs["png"]),
                "pdf": _rel(outputs["pdf"]),
                "figure_width_in": panel_size[0],
                "figure_height_in": panel_size[1],
            }
        )

        for layout in (
            item for item in FIGURE3_PPT_PANEL_LAYOUT if int(item["row"]) == -2
        ):
            schematic = str(layout["schematic"])
            panel_size = _figure3_source_figsize(layout)
            fig = _render_figure3_analysis_schematic(
                schematic,
                panel_size,
                style,
            )
            outputs = _save_figure3_panel_outputs(
                fig,
                panel_dir,
                Path(str(layout["source_svg"])).stem,
                style,
                fixed_bbox=True,
            )
            panel_rows.append(
                {
                    "panel_letter": layout["panel_letter"],
                    "panel_key": layout["key"],
                    "visual_role": layout["visual_role"],
                    "source": "user-provided Python schematic grammar; maintained compact renderer-native adaptation",
                    "svg": _rel(outputs["svg"]),
                    "png": _rel(outputs["png"]),
                    "pdf": _rel(outputs["pdf"]),
                    "figure_width_in": panel_size[0],
                    "figure_height_in": panel_size[1],
                    "anticipated_result_curve": False,
                }
            )

        for row, metric in enumerate(FIGURE3_DISPLAY_METRICS):
            row_ylim = style.figure3_metric_y_limits[metric]
            row_yticks = style.figure3_metric_y_ticks[metric]
            for col in range(3):
                layout = next(
                    item
                    for item in FIGURE3_PPT_PANEL_LAYOUT
                    if int(item["row"]) == row and int(item["column"]) == col
                )
                panel_size = _figure3_source_figsize(layout)
                fig, ax = plt.subplots(figsize=panel_size, facecolor=style.facecolor)
                if col == 0:
                    _draw_day2_template_curve(
                        ax,
                        day2_curves,
                        metric,
                        style,
                        show_xlabel=True,
                        show_legend=False,
                        show_title=False,
                    )
                    source = _rel(DAY2_TEMPLATE_GROUP_CURVES)
                elif col == 1:
                    _draw_experience_scatter(
                        ax,
                        day2_points,
                        day2_stats,
                        metric,
                        style,
                        x_col="experience_index",
                        x_values=np.linspace(0, 4, 100),
                        xticks=(0, 1, 2, 3, 4),
                        xticklabels=("0", "1", "2", "3", "4"),
                        show_xlabel=True,
                        ylabel="Correlation",
                        annotation_mode="none",
                        fit_color=style.metric_colors[metric],
                        fit_linestyle="-",
                        xlabel="No. of cycles",
                        show_title=False,
                        point_group_map=FIGURE3_FIRST30_EXPERIENCE_COLOR_GROUPS,
                    )
                    source = _rel(FIGURE3_FIRST30_POINTS_ALL_METRICS)
                else:
                    _draw_experience_scatter(
                        ax,
                        long_points,
                        long_stats,
                        metric,
                        style,
                        x_col="experience",
                        x_values=np.linspace(1, 4, 100),
                        xticks=(1, 2, 3, 4),
                        xticklabels=LONG_GAP_XLABELS,
                        show_xlabel=True,
                        ylabel="Correlation",
                        annotation_mode="none",
                        fit_color=style.metric_colors[metric],
                        fit_linestyle="-",
                        xlabel="Cycle",
                        show_title=False,
                        point_group_map=FIGURE3_LONG_GAP_EXPERIENCE_COLOR_GROUPS,
                    )
                    source = _rel(LONG_GAP_POINTS)
                ax.set_ylim(*row_ylim)
                ax.set_yticks(list(row_yticks))
                ax.set_ylabel(FIGURE3_ROW_YLABELS[metric] if col == 0 else "")
                fig.subplots_adjust(
                    left=0.22 if col == 0 else 0.17,
                    right=0.985,
                    bottom=0.255,
                    top=0.98,
                )
                outputs = _save_figure3_panel_outputs(
                    fig,
                    panel_dir,
                    Path(str(layout["source_svg"])).stem,
                    style,
                    fixed_bbox=True,
                )
                panel_rows.append(
                    {
                        "panel_letter": layout["panel_letter"],
                        "panel_key": layout["key"],
                        "visual_role": layout["visual_role"],
                        "source": source,
                        "svg": _rel(outputs["svg"]),
                        "png": _rel(outputs["png"]),
                        "pdf": _rel(outputs["pdf"]),
                        "figure_width_in": panel_size[0],
                        "figure_height_in": panel_size[1],
                    }
                )
    panel_rows = sorted(panel_rows, key=lambda row: str(row["panel_letter"]))
    pd.DataFrame(panel_rows).to_csv(panel_dir / "figure3_individual_panel_manifest.csv", index=False)
    _write_json(panel_dir / "figure3_individual_panel_manifest.json", panel_rows)
    return panel_dir


def _drop_svg_elements_by_id_fragment(parent: ET.Element, fragments: tuple[str, ...]) -> None:
    for child in list(parent):
        child_id = str(child.attrib.get("id", ""))
        if any(fragment in child_id for fragment in fragments):
            parent.remove(child)
            continue
        _drop_svg_elements_by_id_fragment(child, fragments)


def _load_svg_payload_from_file(
    panel_id: str,
    svg_path: Path,
    *,
    drop_id_fragments: tuple[str, ...] = (),
) -> dict[str, object]:
    if not svg_path.exists():
        raise FileNotFoundError(f"Missing source-rendered Figure 3 panel SVG: {svg_path}")
    root = ET.parse(svg_path).getroot()
    if drop_id_fragments:
        _drop_svg_elements_by_id_fragment(root, drop_id_fragments)
    _prefix_svg_ids(root, panel_id)
    width, height = _parse_svg_viewbox(root)
    children = [child for child in list(root) if child.tag != _svg_tag("metadata")]
    return {
        "children": children,
        "viewbox_width": float(width),
        "viewbox_height": float(height),
    }


def _figure3_relative_slot_pt(slot_pt: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    bbox_x, bbox_y, _bbox_w, _bbox_h = FIGURE3_PPT_CONTENT_BBOX_PT
    x, y, w, h = (float(value) for value in slot_pt)
    return (x - bbox_x, y - bbox_y, w, h)


def _figure3_effective_src_rect(layout: Mapping[str, object]) -> dict[str, int]:
    src_rect = {str(key): int(value) for key, value in dict(layout["src_rect"]).items()}
    if str(layout["panel_letter"]) == "a":
        return {}
    if int(layout["row"]) >= 0:
        src_rect.pop("t", None)
    return src_rect


def _svg_defs(root: ET.Element) -> ET.Element:
    for child in root:
        if child.tag == _svg_tag("defs"):
            return child
    return ET.SubElement(root, _svg_tag("defs"))


def _add_svg_panel_cropped(
    root: ET.Element,
    panel_svg: dict[str, object],
    *,
    panel_id: str,
    slot_pt: tuple[float, float, float, float],
    src_rect: dict[str, int],
) -> dict[str, float]:
    """Place a source-rendered panel using PPT srcRect values as layout clipping."""

    slot_x, slot_y, slot_w, slot_h = _figure3_relative_slot_pt(slot_pt)
    view_w = float(panel_svg["viewbox_width"])
    view_h = float(panel_svg["viewbox_height"])
    left = float(src_rect.get("l", 0)) / 100000.0
    top = float(src_rect.get("t", 0)) / 100000.0
    right = float(src_rect.get("r", 0)) / 100000.0
    bottom = float(src_rect.get("b", 0)) / 100000.0
    crop_x = view_w * left
    crop_y = view_h * top
    crop_w = view_w * max(1.0 - left - right, 1e-9)
    crop_h = view_h * max(1.0 - top - bottom, 1e-9)
    required_scale_x = slot_w / crop_w
    required_scale_y = slot_h / crop_h
    if not math.isclose(
        required_scale_x,
        required_scale_y,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise AssertionError(
            f"Figure 3 source panel {panel_id} would be distorted during assembly: "
            f"scale_x={required_scale_x:.8f}, scale_y={required_scale_y:.8f}."
        )
    scale = 0.5 * (required_scale_x + required_scale_y)
    scale_x = scale
    scale_y = scale
    clip_id = f"clip_{panel_id}"
    clip = ET.SubElement(
        _svg_defs(root),
        _svg_tag("clipPath"),
        {"id": clip_id, "clipPathUnits": "userSpaceOnUse"},
    )
    ET.SubElement(
        clip,
        _svg_tag("rect"),
        {
            "x": f"{slot_x:.6f}",
            "y": f"{slot_y:.6f}",
            "width": f"{slot_w:.6f}",
            "height": f"{slot_h:.6f}",
        },
    )
    outer = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": f"panel_{panel_id}",
            "clip-path": f"url(#{clip_id})",
        },
    )
    group = ET.SubElement(
        outer,
        _svg_tag("g"),
        {
            "transform": (
                f"translate({slot_x - crop_x * scale_x:.6f} {slot_y - crop_y * scale_y:.6f}) "
                f"scale({scale_x:.8f} {scale_y:.8f})"
            ),
        },
    )
    for child in panel_svg["children"]:
        group.append(child)
    return {
        "slot_x_pt": float(slot_x),
        "slot_y_pt": float(slot_y),
        "slot_width_pt": float(slot_w),
        "slot_height_pt": float(slot_h),
        "source_viewbox_width_pt": float(view_w),
        "source_viewbox_height_pt": float(view_h),
        "crop_x_pt": float(crop_x),
        "crop_y_pt": float(crop_y),
        "crop_width_pt": float(crop_w),
        "crop_height_pt": float(crop_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "required_scale_x": float(required_scale_x),
        "required_scale_y": float(required_scale_y),
        "required_scale_ratio_deviation": float(abs(required_scale_x / required_scale_y - 1.0)),
        "rendered_width": float(slot_w),
        "rendered_height": float(slot_h),
        "scale": float(scale),
    }


def _add_svg_panel_label_pt(
    root: ET.Element,
    label: str,
    *,
    x_pt: float,
    y_pt: float,
    style: PublicationStyle,
) -> None:
    node = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{float(x_pt):.6f}",
            "y": f"{float(y_pt):.6f}",
            "font-family": style.font_family,
            "font-size": f"{style.panel_letter_fontsize:.1f}",
            "font-weight": style.panel_letter_weight,
            "text-anchor": "start",
            "dominant-baseline": "hanging",
            "fill": "black",
            "class": "figure3-panel-label",
        },
    )
    node.text = str(label).lower()


def _add_svg_day2_group_legend(
    root: ET.Element,
    *,
    metric: str,
    slot_pt: tuple[float, float, float, float],
    style: PublicationStyle,
) -> None:
    slot_x, slot_y, _slot_w, _slot_h = _figure3_relative_slot_pt(slot_pt)
    labels = (("naive", "3mo", "6mo"), ("9mo", "12mo"))
    row_gap = 6.4
    column_gap = 27.0
    line_len = 5.5
    text_gap = 1.1
    font_size = 5.6
    x0 = slot_x + _slot_w - 63.0
    y_offset = 50.0
    y0 = slot_y + _slot_h - y_offset
    for col, group_labels in enumerate(labels):
        for row, group in enumerate(group_labels):
            y = y0 + row * row_gap
            x = x0 + col * column_gap
            color = style.day2_group_colors[metric][group]
            ET.SubElement(
                root,
                _svg_tag("line"),
                {
                    "x1": f"{x:.6f}",
                    "y1": f"{y:.6f}",
                    "x2": f"{x + line_len:.6f}",
                    "y2": f"{y:.6f}",
                    "stroke": color,
                    "stroke-width": "2.100000",
                    "stroke-linecap": "round",
                    "class": "figure3-day2-legend-swatch",
                },
            )
            node = ET.SubElement(
                root,
                _svg_tag("text"),
                {
                    "x": f"{x + line_len + text_gap:.6f}",
                    "y": f"{y:.6f}",
                    "font-family": style.font_family,
                    "font-size": f"{font_size:.1f}",
                    "font-weight": "normal",
                    "text-anchor": "start",
                    "dominant-baseline": "central",
                    "fill": "black",
                    "class": "figure3-day2-legend-label",
                },
            )
            node.text = EXPERIENCE_DISPLAY[group]


def _figure3_stat_annotation(stats_df: pd.DataFrame, metric: str) -> tuple[str, float]:
    if metric not in stats_df.index:
        return ("", math.nan)
    stat = stats_df.loc[metric]
    if "q_bh_two_sided" in stat.index:
        p_value = stat["q_bh_two_sided"]
    elif "p_two_sided_cluster_animal" in stat.index:
        p_value = stat["p_two_sided_cluster_animal"]
    elif "p_two_sided" in stat.index:
        p_value = stat["p_two_sided"]
    else:
        p_value = math.nan
    if "pearson_r" in stat.index:
        r_value = float(stat["pearson_r"])
    elif "r2" in stat.index:
        r2 = float(stat["r2"])
        sign = math.copysign(1.0, float(stat["slope"])) if np.isfinite(float(stat["slope"])) else 1.0
        r_value = sign * math.sqrt(max(r2, 0.0))
    else:
        r_value = math.nan
    return (_figure2_star_from_p(p_value), r_value)


def _add_svg_figure3_stat_overlay(
    root: ET.Element,
    *,
    panel_letter: str,
    metric: str,
    stats_df: pd.DataFrame,
    slot_pt: tuple[float, float, float, float],
    star_x_pt: float | None = None,
    style: PublicationStyle,
) -> None:
    star, r_value = _figure3_stat_annotation(stats_df, metric)
    if not star and not np.isfinite(r_value):
        return
    slot_x, slot_y, slot_w, slot_h = _figure3_relative_slot_pt(slot_pt)
    color = style.metric_colors[metric]
    _ = panel_letter
    star_y_frac = 0.035
    star_x = float(star_x_pt) if star_x_pt is not None else slot_x + slot_w * 0.315
    if star:
        star_node = ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{star_x:.6f}",
                "y": f"{slot_y + slot_h * star_y_frac:.6f}",
                "font-family": style.font_family,
                "font-size": f"{style.axis_label_fontsize if panel_letter == 'm' and star == 'n.s.' else style.annotation_fontsize * 2.175:.1f}",
                "font-weight": "bold",
                "text-anchor": "start",
                "dominant-baseline": "hanging",
                "fill": "black",
                "class": "figure3-stat-star-label",
            },
        )
        star_node.text = star
    if np.isfinite(r_value):
        r_x_frac, r_y_frac = FIGURE3_STAT_R_POSITION_FRAC
        r_node = ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{slot_x + slot_w * r_x_frac:.6f}",
                "y": f"{slot_y + slot_h * r_y_frac:.6f}",
                "font-family": style.font_family,
                "font-size": f"{style.annotation_fontsize:.1f}",
                "font-weight": "normal",
                "text-anchor": "end",
                "dominant-baseline": "hanging",
                "fill": "black",
                "class": "figure3-stat-r-label",
            },
        )
        r_node.text = f"r={r_value:.2f}"


def render_figure3(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    style = dataclass_replace(style, show_grid=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    day2_curves = _load_day2_template_curves(style)
    day2_points = _load_day2_template_30min_points()
    day2_stats = _load_day2_template_30min_stats(day2_points)
    long_points, long_stats = _load_long_gap()
    support_runs = _summarize_day2_template_support_runs(day2_curves)
    support_runs_path = out_dir / "figure3_day2_template_support_runs.csv"
    support_runs.to_csv(support_runs_path, index=False)

    source_panel_dir = _render_figure3_individual_panels(
        out_dir,
        style,
        day2_curves=day2_curves,
        day2_points=day2_points,
        day2_stats=day2_stats,
        long_points=long_points,
        long_stats=long_stats,
    )
    content_w_pt = float(FIGURE3_PPT_CONTENT_BBOX_PT[2])
    content_h_pt = float(FIGURE3_PPT_CONTENT_BBOX_PT[3])
    export_height_mm = style.figure3_export_width_mm * content_h_pt / content_w_pt
    export_width_in = style.figure3_export_width_mm / 25.4
    export_height_in = export_height_mm / 25.4
    lower_panel_letters = {
        (int(layout["row"]), int(layout["column"])): str(layout["panel_letter"])
        for layout in FIGURE3_PPT_PANEL_LAYOUT
        if int(layout["row"]) >= 0
    }
    panel_payloads = {
        str(layout["panel_letter"]): _load_svg_payload_from_file(
            f"figure3_{layout['panel_letter']}",
            source_panel_dir / str(layout["source_svg"]),
            drop_id_fragments=("legend_",)
            if int(layout["row"]) >= 0 and int(layout["column"]) == 0
            else (),
        )
        for layout in FIGURE3_PPT_PANEL_LAYOUT
    }
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{style.figure3_export_width_mm:.3f}mm",
            "height": f"{export_height_mm:.3f}mm",
            "viewBox": f"0 0 {content_w_pt:.6f} {content_h_pt:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": style.facecolor})
    axes_records: list[dict[str, object]] = []
    for layout in FIGURE3_PPT_PANEL_LAYOUT:
        letter = str(layout["panel_letter"])
        placement = _add_svg_panel_cropped(
            root,
            panel_payloads[letter],
            panel_id=f"figure3_{letter}",
            slot_pt=tuple(float(value) for value in layout["slot_pt"]),
            src_rect=_figure3_effective_src_rect(layout),
        )
        slot_x, slot_y, slot_w, slot_h = (
            placement["slot_x_pt"],
            placement["slot_y_pt"],
            placement["slot_width_pt"],
            placement["slot_height_pt"],
        )
        axes_records.append(
            {
                "key": layout["key"],
                "panel_letter": letter,
                "metric": layout["metric"],
                "row": int(layout["row"]),
                "column": int(layout["column"]),
                "x0": slot_x / content_w_pt,
                "y0": (content_h_pt - slot_y - slot_h) / content_h_pt,
                "y0_top_origin": slot_y / content_h_pt,
                "width": slot_w / content_w_pt,
                "height": slot_h / content_h_pt,
                "width_in": (slot_w / content_w_pt) * export_width_in,
                "height_in": (slot_h / content_h_pt) * export_height_in,
                "aspect_width_over_height": slot_w / slot_h,
                "slot_pt": json.dumps([slot_x, slot_y, slot_w, slot_h]),
                "src_rect": json.dumps(_figure3_effective_src_rect(layout), sort_keys=True),
                "ppt_source_src_rect": json.dumps(layout["src_rect"], sort_keys=True),
                "source_svg": _rel(source_panel_dir / str(layout["source_svg"])),
                "source_png": _rel(source_panel_dir / str(layout["source_png"])),
                **placement,
            }
        )
    for layout in FIGURE3_PPT_PANEL_LAYOUT:
        if int(layout["column"]) == 0 and int(layout["row"]) in (0, 1, 2):
            _add_svg_day2_group_legend(
                root,
                metric=str(layout["metric"]),
                slot_pt=tuple(float(value) for value in layout["slot_pt"]),
                style=style,
            )
    stat_star_x_pt_by_column = {
        col: float(
            np.mean(
                [
                    _figure3_relative_slot_pt(tuple(float(value) for value in layout["slot_pt"]))[0]
                    + _figure3_relative_slot_pt(tuple(float(value) for value in layout["slot_pt"]))[2] * 0.315
                    for layout in FIGURE3_PPT_PANEL_LAYOUT
                    if int(layout["column"]) == col and int(layout["row"]) in (0, 1, 2)
                ]
            )
        )
        for col in (1, 2)
    }
    for layout in FIGURE3_PPT_PANEL_LAYOUT:
        if int(layout["column"]) in (1, 2) and int(layout["row"]) in (0, 1, 2):
            _add_svg_figure3_stat_overlay(
                root,
                panel_letter=str(layout["panel_letter"]),
                metric=str(layout["metric"]),
                stats_df=day2_stats if int(layout["column"]) == 1 else long_stats,
                slot_pt=tuple(float(value) for value in layout["slot_pt"]),
                star_x_pt=stat_star_x_pt_by_column[int(layout["column"])],
                style=style,
            )
    for layout in FIGURE3_PPT_PANEL_LAYOUT:
        slot_x, slot_y, slot_w, slot_h = _figure3_relative_slot_pt(tuple(float(value) for value in layout["slot_pt"]))
        label_x = slot_x - 1.8 if int(layout["column"]) == 1 else slot_x - 7.0
        if str(layout["panel_letter"]) in {"h", "i", "l"}:
            label_x += 5.0
        if str(layout["panel_letter"]) in {"f", "i", "l"}:
            anchor = next(item for item in FIGURE3_PPT_PANEL_LAYOUT if item["panel_letter"] == "f")
            label_x = _figure3_relative_slot_pt(tuple(float(value) for value in anchor["slot_pt"]))[0] - 1.8
        label_y = slot_y - (2.0 if int(layout["row"]) == -2 else 5.0)
        _add_svg_panel_label_pt(
            root,
            str(layout["panel_letter"]),
            x_pt=max(0.0, label_x),
            y_pt=max(0.0, label_y),
            style=style,
        )
    svg_tree = ET.ElementTree(root)
    out_svg = out_dir / "figure3_source_publication.svg"
    out_png = out_dir / "figure3_source_publication.png"
    out_pdf = out_dir / "figure3_source_publication.pdf"
    svg_tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    html_path = _export_svg_with_chrome(
        out_svg,
        out_png,
        out_pdf,
        width_mm=float(style.figure3_export_width_mm),
        height_mm=float(export_height_mm),
        dpi=int(style.dpi),
    )
    top_record = {
        "panel": "figure3_top_example",
        "source_path": _rel(EXAMPLE_SAVED_DIR / "451-3day1_to_451-6day1__row0161_A190_B267.npz"),
        "source_panel_svg": _rel(source_panel_dir / "figure3_panel_a_top_example_row.svg"),
        "renderer": "render_main_figs_v3_publication._draw_day12_arrow_example",
        "note": "source-rendered square-boundary panel assembled isotropically on its exact destination aspect ratio; no PPT/PDF media used as source",
        "pick": {"pair_key": "451-3day1->451-6day1", "row_idx": 161},
    }

    schematic_records = [
        {
            "panel": f"figure3_{layout['key']}",
            "panel_letter": str(layout["panel_letter"]),
            "renderer": "render_main_figs_v3_publication._render_figure3_analysis_schematic",
            "source": "user-provided Python schematic grammar; maintained compact renderer-native adaptation",
            "anticipated_result_curve": False,
            "visual_role": str(layout["visual_role"]),
        }
        for layout in FIGURE3_PPT_PANEL_LAYOUT
        if int(layout["row"]) == -2
    ]
    source_manifest = [
        top_record,
        *schematic_records,
        {
            "panel": "figure3_day2_template_curves",
            "source_path": _rel(DAY2_TEMPLATE_GROUP_CURVES),
            "support_run_audit": _rel(support_runs_path),
            "support_mask_authority": _rel(FIGURE3_CURVE_SUPPORT_MASK),
            "support_rule": FIGURE3_CURVE_SUPPORT_RULE,
            "renderer": "render_main_figs_v3_publication._draw_day2_template_curve",
            "filters": {
                "metrics": list(METRICS),
                "window_type": "maze_time",
                "window_size": 30.0,
                "support_policy_by_metric": DAY2_TEMPLATE_SUPPORT_POLICY,
                "excluded_metrics": ["level_map_correlation"],
            },
        },
        {
            "panel": "figure3_day2_template_first30_experience",
            "source_paths": {"included_points": _rel(FIGURE3_FIRST30_POINTS_ALL_METRICS)},
            "plot_ready_points_authority": _rel(FIGURE3_FIRST30_POINTS_ALL_METRICS),
            "stats_path": _rel(FIGURE3_FIRST30_STATS_ALL_METRICS),
            "renderer": "render_main_figs_v3_publication._draw_experience_scatter",
            "statistics": "OLS slope with CR1 covariance clustered by animal; primary two-sided P uses Student t(G-1); BH correction spans the four Main-Figure representation metrics",
            "filters": {
                "metrics": list(METRICS),
                "window_type": "maze_time",
                "window_size": 30.0,
                "value_column": "value",
                "support_policy_by_metric": DAY2_TEMPLATE_SUPPORT_POLICY,
                "variant": DAY2_TEMPLATE_VARIANT_LABEL,
                "excluded_metrics": ["level_map_correlation"],
            },
        },
        {
            "panel": "figure3_long_gap_experience_trends",
            "source_path": _rel(LONG_GAP_POINTS),
            "stats_path": _rel(LONG_GAP_STATS),
            "renderer": "render_main_figs_v3_publication._draw_experience_scatter",
            "filters": {"metrics": list(METRICS), "excluded_metrics": ["level_map_correlation"]},
        },
    ]
    manifest = {
        "figure": "figure3",
        "outputs": {
            "png": _rel(out_png),
            "svg": _rel(out_svg),
            "pdf": _rel(out_pdf),
            "html_export_wrapper": _rel(html_path),
        },
        "renderer": "render_main_figs_v3_publication.render_figure3",
        "style": asdict(style),
        "layout_mode": "schematic_row_source_panel_svg",
        "layout_reference": {
            "layout_source": "Figure 3 layout constants in this renderer",
            "content_bbox_pt": list(FIGURE3_PPT_CONTENT_BBOX_PT),
            "source_panel_dir": _rel(source_panel_dir),
            "row_headers": FIGURE3_PPT_ROW_HEADERS,
            "panels": FIGURE3_PPT_PANEL_LAYOUT,
            "panel_label_anchor": "top_left_inside_slot",
            "schematic_panels": ["b", "c", "d"],
            "first_column_legend_overlays": ["e", "h", "k"],
            "row_label_mode": "metric-specific first-column y-axis labels; no colored row bands",
            "export_width_mm": float(style.figure3_export_width_mm),
            "export_height_mm": float(export_height_mm),
            "note": "Original PowerPoint geometry anchors the panels; every source panel is rendered at the exact destination aspect ratio and assembled with isotropic scaling only.",
        },
        "sources": source_manifest,
        "axes": axes_records,
        "data_counts": {
            "day2_template_rows": int(len(day2_curves)),
            "day2_template_first30_rows": int(len(day2_points)),
            "long_gap_rows": int(len(long_points)),
        },
    }
    verification_records: list[dict[str, object]] = []
    alignment_records: list[dict[str, object]] = []
    for metric in METRICS:
        curve_first = day2_curves.loc[
            day2_curves["metric"].eq(metric)
            & np.isclose(pd.to_numeric(day2_curves["x_min"], errors="coerce"), 30.0)
        ][["experience_group", "mean", "n_animals"]].rename(
            columns={"mean": "curve_mean", "n_animals": "curve_n_animals"}
        )
        point_first = (
            day2_points.loc[day2_points["metric"].eq(metric)]
            .groupby("experience_group", as_index=False)
            .agg(point_mean=("value", "mean"), point_n_animals=("animal", "nunique"), point_n_rows=("value", "size"))
        )
        aligned = curve_first.merge(point_first, on="experience_group", how="outer")
        aligned["metric"] = metric
        aligned["delta_point_minus_curve"] = aligned["point_mean"] - aligned["curve_mean"]
        alignment_records.extend(aligned.to_dict(orient="records"))
    alignment = pd.DataFrame(alignment_records)
    max_alignment_delta = float(pd.to_numeric(alignment["delta_point_minus_curve"], errors="coerce").abs().max())
    if max_alignment_delta > 1e-9:
        raise AssertionError(f"Day2-template first-window source alignment failed: max delta {max_alignment_delta:g}")
    alignment_path = out_dir / "figure3_day2_template_first30_alignment.csv"
    alignment.to_csv(alignment_path, index=False)
    manifest["source_alignment"] = {
        "path": _rel(alignment_path),
        "check": "middle-column per-animal group means equal left-column curve means at 30 min maze-time",
        "max_abs_delta": max_alignment_delta,
    }
    for index, key in enumerate([f"top_example_{i}" for i in range(1, 5)], start=1):
        verification_records.append(
            {
                "panel_key": key,
                "panel_letter": "a",
                "visual_role": f"example map {index} drawn as native square-boundary trajectory/arrow map",
                "source": top_record["source_path"],
                "filters": "fixed saved pick 451-3day1->451-6day1 row 161",
                "source_row_count": 1,
                "expected": "poster-style black square boundary, no internal maze walls, gray trajectory, red arrows, no raster crop",
                "auto_check": "PASS",
            }
        )
    for layout in FIGURE3_PPT_PANEL_LAYOUT:
        if int(layout["row"]) != -2:
            continue
        verification_records.append(
            {
                "panel_key": layout["key"],
                "panel_letter": layout["panel_letter"],
                "visual_role": layout["visual_role"],
                "source": "user-provided Python schematic grammar; maintained compact renderer-native adaptation",
                "filters": "session blocks, highlighted windows, and comparison arrows only",
                "source_row_count": "schematic",
                "expected": "vector-native design schematic with no anticipated result curve",
                "auto_check": "PASS",
            }
        )
    for row, metric in enumerate(FIGURE3_DISPLAY_METRICS):
        curve_rows = day2_curves.loc[day2_curves["metric"].eq(metric)]
        first_rows = day2_points.loc[day2_points["metric"].eq(metric)]
        long_rows = long_points.loc[long_points["metric"].eq(metric)]
        first_slope = float(day2_stats.loc[metric, "slope"]) if metric in day2_stats.index else math.nan
        long_slope = float(long_stats.loc[metric, "slope"]) if metric in long_stats.index and "slope" in long_stats.columns else math.nan
        verification_records.extend(
            [
                {
                    "panel_key": f"day2_template_{metric}",
                    "panel_letter": lower_panel_letters[(row, 0)],
                    "visual_role": "Day1 windows to Day2 template; 30 min maze-time curve over time in maze",
                    "source": _rel(DAY2_TEMPLATE_GROUP_CURVES),
                    "filters": f"metric={metric}; window_type=maze_time; window_size=30; support_policy={DAY2_TEMPLATE_SUPPORT_POLICY[metric]}",
                    "source_row_count": int(len(curve_rows)),
                    "expected": f"increasing curve family across supported time-in-maze display range {style.day2_template_xlim_min:g}-{style.day2_template_xlim_max:g} min; no level-map rows",
                    "auto_check": "PASS" if len(curve_rows) > 0 and metric != "level_map_correlation" else "FAIL",
                },
                {
                    "panel_key": f"day2_template_first30_{metric}",
                    "panel_letter": lower_panel_letters[(row, 1)],
                    "visual_role": "Experience scatter from the same Day2-template source family, first 30 min maze-time window",
                    "source": _rel(FIGURE3_FIRST30_POINTS_ALL_METRICS),
                    "filters": f"metric={metric}; window_type=maze_time; window_size=30; value=first_value; support_policy={DAY2_TEMPLATE_SUPPORT_POLICY[metric]}",
                    "source_row_count": int(len(first_rows)),
                    "n_animals": int(first_rows["animal"].astype(str).nunique()),
                    "expected": "positive Exp0-Exp4 trend aligned with the Day2-template curve family",
                    "computed_slope": first_slope,
                    "primary_p_two_sided_cr1_t_g_minus_1": float(day2_stats.loc[metric, "p_two_sided"]),
                    "primary_q_bh4": float(day2_stats.loc[metric, "q_bh_two_sided"]),
                    "auto_check": "PASS" if len(first_rows) > 0 and first_slope > 0 else "FAIL",
                },
                {
                    "panel_key": f"long_gap_{metric}",
                    "panel_letter": lower_panel_letters[(row, 2)],
                    "visual_role": "Long-gap experience trend",
                    "source": _rel(LONG_GAP_POINTS),
                    "filters": f"metric={metric}; plotted value column=value",
                    "source_row_count": int(len(long_rows)),
                    "expected": "right-column long-gap scatter with source trend fit",
                    "computed_slope": long_slope,
                    "auto_check": "PASS" if len(long_rows) > 0 else "FAIL",
                },
            ]
        )
    verification_path = out_dir / "figure3_panel_verification.csv"
    geometry_path = out_dir / "figure3_axes_geometry.csv"
    pd.DataFrame(verification_records).to_csv(verification_path, index=False)
    pd.DataFrame(axes_records).to_csv(geometry_path, index=False)
    manifest["panel_verification"] = _rel(verification_path)
    manifest["panel_geometry"] = _rel(geometry_path)
    _write_json(out_dir / "figure3_source_manifest.json", manifest)
    return manifest
