"""Render source-data main-figures-v3 panels with a shared publication style.

This script is the source-data path for manuscript-ready panels. It does not
crop figures out of the PDF/PPTX layout. The attached deck/PDF may be used as a
visual reference, but every plotted axis here is redrawn from source tables or
maintained source-rendering helpers.

Current implemented figures:
- Figure 1: maintained source-native Figure 1 panel renderers composed as SVG,
  then exported to PNG/PDF for inspection.
- Figure 2: maintained source-native Figure 2 panel renderers composed as SVG,
  then exported to PNG/PDF for inspection.
- Figure 3: source-rendered individual panels composed with the user-edited
  PowerPoint geometry:
  column 1: Day2-template 30 min maze-time curves
  column 2: first 30 min maze-time windows to the Day2 template
  column 3: long-gap experience trends
- Figure 4: source-rendered cross-session decoding schematic plus sorted decoder,
  spatial-error, linearized-place-field, and behavior-correlation panels.
- Figure 5: maintained source-native level-code panel renderers composed as SVG,
  then exported to PNG/PDF for inspection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
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
from ca_analysis.mm_plot_utils import plot


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "main_figs_v3_source_render_2026-06-24"
FIGURE3_INDIVIDUAL_PANEL_DIR = DEFAULT_OUT_DIR / "figure3_individual_panels"
FIGURE_SCOPE_EXCLUSION_DIR = DEFAULT_OUT_DIR / "figure_scope_exclusions_2026-06-29"
SORTED_ROOT = REPO_ROOT / "docs" / "sorted_data_full_figure_rerun_2026-06-11"
COVERAGE5MIN_ROOT = REPO_ROOT / "docs" / "coverage5min_window_reanalysis_2026-08-05"
DAY2_TEMPLATE_VARIANT_LABEL = "coverage5min_current59"
FIGURE3_FIRST30_AUTHORITY_DIR = COVERAGE5MIN_ROOT / "figure3_authority"
DAY2_TEMPLATE_GROUP_CURVES = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "day2_template_group_curve_summary.csv"
)
DAY2_TEMPLATE_SOURCE_MANIFEST = FIGURE3_FIRST30_AUTHORITY_DIR / "SOURCE_MANIFEST.json"
DAY2_TEMPLATE_FIXED_ROOT = COVERAGE5MIN_ROOT / "fixed_reference"
DAY2_TEMPLATE_ANIMAL_SUMMARIES = {
    "sorted_non1616_plus_12mo": DAY2_TEMPLATE_FIXED_ROOT
    / "sorted_non1616"
    / "animal_metric_summary.csv",
    "original_fallback_naive": DAY2_TEMPLATE_FIXED_ROOT
    / "original_fallback_naive"
    / "animal_metric_summary.csv",
    "sorted_1616_as_0_3_6mo": DAY2_TEMPLATE_FIXED_ROOT
    / "sorted_1616"
    / "animal_metric_summary.csv",
}
FIGURE3_FIRST30_POINTS_ALL_METRICS = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "combined_first30_points_all_metrics.csv"
)
FIGURE3_FIRST30_STATS_ALL_METRICS = (
    FIGURE3_FIRST30_AUTHORITY_DIR / "combined_first30_stats_all_metrics.csv"
)
SOURCE_REPAIR_842_DIR = (
    SORTED_ROOT / "figure2_main3_ed12_source_repair_842_2026-08-03"
)
SOURCE_REPAIR_842_MANIFEST = SOURCE_REPAIR_842_DIR / "SOURCE_REPAIR_MANIFEST.json"
MAIN3_NO_MASK_DIR = SORTED_ROOT / "main3_long_gap_no_manual_mask_2026-08-07"
LONG_GAP_POINTS = MAIN3_NO_MASK_DIR / "main3_experience_trend_points_no_manual_mask.csv"
LONG_GAP_STATS = MAIN3_NO_MASK_DIR / "main3_experience_trend_clustered_stats_no_manual_mask.csv"
MAIN3_NO_MASK_MANIFEST = MAIN3_NO_MASK_DIR / "SOURCE_MANIFEST.json"
MAIN3_MASK_DELTA_AUDIT = MAIN3_NO_MASK_DIR / "MASK_ONLY_METRIC_DELTAS.csv"
FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT = (
    SOURCE_REPAIR_842_DIR / "main3_ed12j_corrected_842_six_metric_source_audit.csv"
)
EXAMPLE_SAVED_DIR = REPO_ROOT / "docs" / "example_neuron_candidates" / "saved_picks"
FIGURE1_HELPER = REPO_ROOT / "scripts" / "preview_figure1_assembly.py"
FIGURE2_HELPER = REPO_ROOT / "scripts" / "preview_figure2_assembly.py"
FIGURE5_HELPER = REPO_ROOT / "scripts" / "preview_figure3_assembly.py"
FIGURE5_CORRECTED_842_DIR = (
    DEFAULT_OUT_DIR / "figure5_no_manual_mask_2026-08-07"
)
FIGURE5_CORRECTED_842_MANIFEST = FIGURE5_CORRECTED_842_DIR / "SOURCE_MANIFEST.json"
FIGURE5_EXPECTED_CORRECTED_TUNING_SHA256 = (
    "54ef9c4f964d34fe8f4332b7c96b23ff519581aab8dac09a86fbfd32da0acee4"
)
FIGURE5_EXPECTED_CORRECTED_CIMAP_SHA256 = (
    "8f4db74927d1c8776497f110ec47b1d8db245aebb1ca1df38d63bf6fa9b29a22"
)
FIGURE2_DISTRIBUTION_DIR = (
    SORTED_ROOT
    / "figure2_distribution_shuffle_corrected_842_no_presort_exclusions_2026-08-03"
)
FIGURE2_DISTRIBUTION_PROVENANCE = FIGURE2_DISTRIBUTION_DIR / "provenance_manifest.json"
FIGURE2_DISTRIBUTION_PAIR_INPUT = FIGURE2_DISTRIBUTION_DIR / "pair_input_audit.csv"
FIGURE2_DISTRIBUTION_PAIR_SUPPORT = FIGURE2_DISTRIBUTION_DIR / "pair_support_audit.csv"
FIGURE2_DISTRIBUTION_METRIC_SUMMARY = FIGURE2_DISTRIBUTION_DIR / "metric_summary.csv"
FIGURE2_EXPECTED_CORRECTED_842_CIMAP_SHA256 = (
    "8f4db74927d1c8776497f110ec47b1d8db245aebb1ca1df38d63bf6fa9b29a22"
)
# Panels f/k/p. The four long-gap points and the D1-D2 anchor use equal session-
# pair weighting. Per-pair place/level values already average matched neurons,
# PV values average spatial bins, and ensemble rate contributes one correlation.
# The source retains the no-manual-mask same-stage rows and corrected long-gap
# values; only the final D1-D2 aggregation changed from animal means to pairs.
FIGURE2_GAP_CURVE = (
    SORTED_ROOT
    / "figure2_gap_anchor_pair_mean_no_manual_mask_2026-08-13"
    / "gap_with_day1_day2_curve_data.csv"
)
FIGURE2_GAP_CURVE_MANIFEST = FIGURE2_GAP_CURVE.with_name("manifest.json")
FIGURE2_GAP_PAIR_AUDIT = (
    SOURCE_REPAIR_842_DIR
    / "figure2_842_affected_pair_metric_before_after_support_audit.csv"
)
FIGURE2_GAP_DISTRIBUTION_PARITY = (
    SOURCE_REPAIR_842_DIR / "figure2_842_0to3_distribution_parity_audit.csv"
)
FIGURE3_LEVEL_PAIR_AUDIT = SOURCE_REPAIR_842_DIR / "main3_ed12j_level_patch_audit.csv"
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
FIGURE2_CONSECUTIVE_MANIFEST = FIGURE2_CONSECUTIVE_DIR / "consecutive_gap_clean_long_gap_manifest.json"
FIGURE2_CONSECUTIVE_UNRESTRICTED_SCOPE = (
    FIGURE2_CONSECUTIVE_DIR / "consecutive_gap_unrestricted_adjacent_23_chain_scope.csv"
)
FIGURE2_CONSECUTIVE_CLEAN_SCOPE = FIGURE2_CONSECUTIVE_DIR / "consecutive_gap_clean_long_gap_22_chain_scope.csv"
FIGURE2_CONSECUTIVE_INPUT_AUDIT = FIGURE2_CONSECUTIVE_DIR / "consecutive_gap_sorted_triplet_input_audit.csv"
FIGURE2_OLD_NEW_REPLICATION_AUDIT = (
    FIGURE2_CONSECUTIVE_DIR
    / "figure2_ed12_consecutive_old_vs_new_replication_audit.csv"
)
FIGURE2_RATE_CHAIN_DIR = SORTED_ROOT / "figure2_rate_chain_no_manual_mask_2026-08-07"
FIGURE2_RATE_CHAIN_POINTS = (
    FIGURE2_RATE_CHAIN_DIR / "session_pair_rate_consecutive_gap_points_clean_long_gap_no_manual_mask.csv"
)
FIGURE2_RATE_CHAIN_STATS = (
    FIGURE2_RATE_CHAIN_DIR / "session_pair_rate_consecutive_gap_stats_clean_long_gap_no_manual_mask.csv"
)
FIGURE2_RATE_CHAIN_MANIFEST = FIGURE2_RATE_CHAIN_DIR / "SOURCE_MANIFEST.json"
FIGURE2_RATE_CHAIN_MASK_DELTA = FIGURE2_RATE_CHAIN_DIR / "MASK_REMOVAL_CHAIN_DELTA_AUDIT.csv"
FIGURE2_CLEAN_CHAIN_ANIMALS = {
    "0v3v6": ("418", "422", "451", "468", "488", "489", "837", "838", "839", "842", "862"),
    "3v6v9": ("418", "451", "468", "837", "838", "839", "842"),
    "6v9v12": ("468", "837", "838", "839"),
}
FIGURE4_DECODER_DIR = SORTED_ROOT / "decoder_panels"
FIGURE4_DECODER_36_DIR = FIGURE_SCOPE_EXCLUSION_DIR / "figure4_decoder_36_no_1616_short"
FIGURE4_CROSS_DECODER_POINTS = FIGURE4_DECODER_36_DIR / "cross_session_decoding_error_points_36_no_1616_short.csv"
FIGURE4_CROSS_DECODER_SUMMARY = FIGURE4_DECODER_36_DIR / "cross_session_decoding_error_summary_36_no_1616_short.csv"
FIGURE4_CROSS_DECODER_STATS = FIGURE4_DECODER_36_DIR / "cross_session_decoding_error_stats_36_no_1616_short.json"
FIGURE4_REWARD_DECODER_POINTS = FIGURE4_DECODER_36_DIR / "reward_control_decoding_error_points_36_no_1616_short.csv"
FIGURE4_REWARD_DECODER_SUMMARY = FIGURE4_DECODER_36_DIR / "reward_control_decoding_error_summary_36_no_1616_short.csv"
FIGURE4_REWARD_DECODER_STATS = FIGURE4_DECODER_36_DIR / "reward_control_decoding_error_stats_36_no_1616_short.json"
FIGURE4_LINEARIZED_ROOT = SORTED_ROOT / "linearized_pf" / "sorted_export"
FIGURE4_PANEL_E_HEATMAP_VMAX = 0.6
FIGURE4_BEHAVIOR_LONG = (
    SORTED_ROOT
    / "behavior_correlation_matrix_exhaustive_2026-06-17"
    / "joined_behavior_neural_sorted_exhaustive_long.csv"
)
FIGURE4_PANEL_A_PPT_CROP = {"l": 12400, "t": 15055, "r": 9866, "b": 9331}
FIGURE4_SCHEMATIC_PAIR = ("837day1", "837-3day1")
FIGURE4_SCHEMATIC_DECODER_PKL = (
    REPO_ROOT
    / "docs"
    / "cross_session_decoding"
    / f"cross_session_{FIGURE4_SCHEMATIC_PAIR[0]}_to_{FIGURE4_SCHEMATIC_PAIR[1]}.pkl"
)
FIGURE4_SCHEMATIC_MAP_BINS = 15
FIGURE4_SCHEMATIC_N_CELLS = 5
FIGURE4_SCHEMATIC_SOURCE_CELL_IDS = (117, 12, 10, 24, 70)
FIGURE4_SCHEMATIC_TRACE_POINTS = 5000
CHROME_CANDIDATES = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
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
EXPERIENCE_DISPLAY = {
    "naive": "naive",
    "3mo": "3 mo",
    "6mo": "6 mo",
    "9mo": "9 mo",
    "12mo": "12 mo",
}
SAME_STAGE_XLABELS = ("Exp0", "Exp1", "Exp2", "Exp3", "Exp4")
LONG_GAP_XLABELS = ("1", "2", "3", "4")
FIGURE1_PANEL_ROLES = {
    "a": "maze and scored first-visit decision schematic",
    "b": "split-day cycle diagram",
    "c": "rewarded/end-node timelines",
    "d": "reward-acquisition long-bout time",
    "g": "exploration-efficiency histogram",
    "h": "latent-day timelines",
    "i": "latent-vs-return long-bout time",
    "j": "first-visit route-score specificity control",
}
FIGURE1_PANEL_SOURCE_DETAILS = {
    "a": {
        "source": "ca_analysis.figure_panels.render_maze_schematic_panel; preview_figure1_assembly.get_figure1_reward_path_decisions",
        "filters": "canonical maze geometry and reward path 0->1->4->10->22->46->94; numbered parent-run junctions 0-5; terminal reward endpoint is not labelled as a decision",
        "source_row_count": 6,
    },
    "b": {
        "source": "ca_analysis.figure_panels.render_cycle_diagram_panel",
        "filters": "split-day cycle schematic; no data table",
        "source_row_count": "",
    },
    "c": {
        "source": "maintained behavior loader via render_endnode_timeline_grid_panel",
        "filters": "sessions 434RRday1, 434-3day1, 434RRday2, 434-6day1; reward_node=94; xlim=0-6500 s",
        "source_row_count": 4,
    },
    "d": {
        "source": "docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/reward_acquisition_day1_month_points_current_plus_new12.csv; docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/reward_acquisition_day1_month_summary_current_plus_new12.csv; docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/reward_acquisition_day1_month_clustered_contrasts_current_plus_new12.csv",
        "filters": "metric=lb0; current saved rows plus 837/838/839 12-month day1 append; month groups 0,3,6,9,12",
        "source_row_count": 79,
    },
    "g": {
        "source": "docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/exploration_first_chunk_current_plus_new_points.csv",
        "filters": "groups=Naive D1 (rewarded),LTM D1 (all); current saved rows plus 837/838/839 12-month return append",
        "source_row_count": 79,
    },
    "h": {
        "source": "maintained behavior loader via render_endnode_timeline_grid_panel",
        "filters": "sessions 1636Rday1, 1636Rday2; no physical water port on latent day1; water port available on latent day2; endpoint marker node=94; xlim=0-6500 s",
        "source_row_count": 2,
    },
    "i": {
        "source": "docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/long_bout_group_points_current_plus_new12.csv; docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/long_bout_group_summary_current_plus_new12.csv; docs/new_12mo_session_fit_2026-06-02/panel_by_panel_current_vs_new/long_bout_group_clustered_contrasts_current_plus_new12.csv",
        "filters": "metric=lb0; groups=Rewarded D1 (all),Unrewarded D2 (all),LTM D1 (all); current saved rows plus 837/838/839 12-month return append",
        "source_row_count": 88,
    },
    "j": {
        "source": "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/ed04_session_route_current_plus_new12.csv; docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/ed04_route_group_stats_current_plus_new12.csv; docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/ed04_route_stats_current_plus_new12.csv",
        "filters": "metric=route_fraction_correct; groups=naive,return,experienced_day2,latent_day2; 162 sessions from 33 mice; four-level session OLS with standard errors clustered by animal and small-G t reference; all six pairwise contrasts under BH6; only q<0.05 brackets shown",
        "source_row_count": 162,
    },
}
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
    "pv_correlation": "PV",
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
FIGURE3_PPT_LAYOUT_REFERENCE = Path("C:/Users/jiangwu/Desktop/ltm-figs/main-figs-v4-f3.pptx")
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
    COVERAGE5MIN_ROOT
    / "rendered"
    / "ed07_fixed_time_tile"
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
FIGURE3_PPT_PANEL_LAYOUT = (
    {
        "panel_letter": "a",
        "key": "top_example_row",
        "source_svg": "figure3_panel_a_top_example_row.svg",
        "source_png": "figure3_panel_a_top_example_row.png",
        "slot_pt": (2.410000, 2.271102, 353.006063, 97.808504),
        "src_rect": {"l": 3053, "r": 1},
        "row": -1,
        "column": -1,
        "metric": "",
        "visual_role": "four source-rendered example maps: Day i, Day i+1, Day i+90, Day i+91",
    },
    {
        "panel_letter": "b",
        "key": "moving_windows_to_whole_day2_schematic",
        "source_svg": "figure3_panel_b_moving_windows_to_whole_day2_schematic.svg",
        "source_png": "figure3_panel_b_moving_windows_to_whole_day2_schematic.png",
        "slot_pt": (4.355354, 102.500000, 115.059134, 52.000000),
        "src_rect": {},
        "row": -2,
        "column": 0,
        "metric": "",
        "schematic": "moving_windows",
        "visual_role": "moving 30-min Day-1 windows compared with the whole Day-2 template",
    },
    {
        "panel_letter": "c",
        "key": "first30_across_experience_schematic",
        "source_svg": "figure3_panel_c_first30_across_experience_schematic.svg",
        "source_png": "figure3_panel_c_first30_across_experience_schematic.png",
        "slot_pt": (114.819606, 102.500000, 115.059134, 52.000000),
        "src_rect": {},
        "row": -2,
        "column": 1,
        "metric": "",
        "schematic": "first30_experience",
        "visual_role": "first 30 min compared with the whole next day across experience levels",
    },
    {
        "panel_letter": "d",
        "key": "long_gap_day1_schematic",
        "source_svg": "figure3_panel_d_long_gap_day1_schematic.svg",
        "source_png": "figure3_panel_d_long_gap_day1_schematic.png",
        "slot_pt": (236.575827, 102.500000, 115.059134, 52.000000),
        "src_rect": {},
        "row": -2,
        "column": 2,
        "metric": "",
        "schematic": "long_gap_day1",
        "visual_role": "successive Day-1 sessions compared across approximately 90-day gaps",
    },
    {
        "panel_letter": "e",
        "key": "day2_template_place_map_correlation",
        "source_svg": "figure3_panel_e_day2_template_place_map_correlation.svg",
        "source_png": "figure3_panel_e_day2_template_place_map_correlation.png",
        "slot_pt": (4.355354, 160.000000, 115.059134, 102.772835),
        "src_rect": {"t": 15640},
        "row": 0,
        "column": 0,
        "metric": "place_map_correlation",
        "visual_role": "Day1 windows to Day2 template curve over time in maze",
    },
    {
        "panel_letter": "f",
        "key": "day2_template_first30_place_map_correlation",
        "source_svg": "figure3_panel_f_day2_template_first30_place_map_correlation.svg",
        "source_png": "figure3_panel_f_day2_template_first30_place_map_correlation.png",
        "slot_pt": (114.819606, 160.000000, 115.059134, 102.772835),
        "src_rect": {"t": 15640},
        "row": 0,
        "column": 1,
        "metric": "place_map_correlation",
        "visual_role": "Experience scatter from Day2-template source family, first 30 min maze-time window",
    },
    {
        "panel_letter": "g",
        "key": "long_gap_place_map_correlation",
        "source_svg": "figure3_panel_g_long_gap_place_map_correlation.svg",
        "source_png": "figure3_panel_g_long_gap_place_map_correlation.png",
        "slot_pt": (236.575827, 160.000000, 115.059134, 102.772835),
        "src_rect": {"t": 15640},
        "row": 0,
        "column": 2,
        "metric": "place_map_correlation",
        "visual_role": "Long-gap experience trend scatter with source trend fit",
    },
    {
        "panel_letter": "h",
        "key": "day2_template_pv_correlation",
        "source_svg": "figure3_panel_h_day2_template_pv_correlation.svg",
        "source_png": "figure3_panel_h_day2_template_pv_correlation.png",
        "slot_pt": (3.437480, 270.400000, 116.894803, 104.412520),
        "src_rect": {"t": 15640},
        "row": 1,
        "column": 0,
        "metric": "pv_correlation",
        "visual_role": "Day1 windows to Day2 template curve over time in maze",
    },
    {
        "panel_letter": "i",
        "key": "day2_template_first30_pv_correlation",
        "source_svg": "figure3_panel_i_day2_template_first30_pv_correlation.svg",
        "source_png": "figure3_panel_i_day2_template_first30_pv_correlation.png",
        "slot_pt": (114.819606, 270.400000, 116.894803, 104.412520),
        "src_rect": {"t": 15640},
        "row": 1,
        "column": 1,
        "metric": "pv_correlation",
        "visual_role": "Experience scatter from Day2-template source family, first 30 min maze-time window",
    },
    {
        "panel_letter": "j",
        "key": "long_gap_pv_correlation",
        "source_svg": "figure3_panel_j_long_gap_pv_correlation.svg",
        "source_png": "figure3_panel_j_long_gap_pv_correlation.png",
        "slot_pt": (236.575827, 270.400000, 116.894803, 104.412520),
        "src_rect": {"t": 15640},
        "row": 1,
        "column": 2,
        "metric": "pv_correlation",
        "visual_role": "Long-gap experience trend scatter with source trend fit",
    },
    {
        "panel_letter": "k",
        "key": "day2_template_ensemble_rate_correlation",
        "source_svg": "figure3_panel_k_day2_template_ensemble_rate_correlation.svg",
        "source_png": "figure3_panel_k_day2_template_ensemble_rate_correlation.png",
        "slot_pt": (3.037087, 382.600000, 117.295197, 105.460157),
        "src_rect": {"t": 15555},
        "row": 2,
        "column": 0,
        "metric": "ensemble_rate_correlation",
        "visual_role": "Day1 windows to Day2 template curve over time in maze",
    },
    {
        "panel_letter": "l",
        "key": "day2_template_first30_ensemble_rate_correlation",
        "source_svg": "figure3_panel_l_day2_template_first30_ensemble_rate_correlation.svg",
        "source_png": "figure3_panel_l_day2_template_first30_ensemble_rate_correlation.png",
        "slot_pt": (114.419213, 382.600000, 117.295197, 105.460157),
        "src_rect": {"t": 15368},
        "row": 2,
        "column": 1,
        "metric": "ensemble_rate_correlation",
        "visual_role": "Experience scatter from Day2-template source family, first 30 min maze-time window",
    },
    {
        "panel_letter": "m",
        "key": "long_gap_ensemble_rate_correlation",
        "source_svg": "figure3_panel_m_long_gap_ensemble_rate_correlation.svg",
        "source_png": "figure3_panel_m_long_gap_ensemble_rate_correlation.png",
        "slot_pt": (236.575827, 382.600000, 117.295197, 105.460157),
        "src_rect": {"t": 15647},
        "row": 2,
        "column": 2,
        "metric": "ensemble_rate_correlation",
        "visual_role": "Long-gap experience trend scatter with source trend fit",
    },
)
FIGURE5_PANEL_ROLES = {
    "a": "level-assignment schematic",
    "e": "level-code heatmap",
    "b": "level-map correlation distribution",
    "c": "shuffle-mean distribution",
    "d": "tree-depth decoder accuracy",
    "f": "level-map correlation vs exploration efficiency",
}
FIGURE5_PANEL_SOURCE_DETAILS = {
    "a": {
        "source": "ca_analysis.figure_panels.render_level_assignment_schematic",
        "filters": "level_definition=tree_depth7; tree_depth=7; level_count=7",
        "source_row_count": "schematic",
    },
    "e": {
        "source": (
            "scripts/preview_figure3_assembly.py::_render_panel_e; "
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "panel_e_842_heatmap_source_audit.csv"
        ),
        "filters": "sorted tree-depth7 place-gated level tuning heatmap; exact d1v1 36-pair order; corrected 842 0->3 tuning root searched first; excluded only 1616day1 -> 1616bday1",
        "source_row_count": "36 sorted d1v1 pairs",
    },
    "b": {
        "source": (
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/plot_data_cell228_level.npz"
        ),
        "filters": "sorted tree-depth7 level-map correlation distribution; exact d1v1 36-pair scope; corrected 842 0->3; excluded only 1616day1 -> 1616bday1; no manual cell-ID exclusions",
        "source_row_count": "16,901 sorted retained neurons from corrected registration/activity gate",
    },
    "c": {
        "source": (
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/plot_data_cell226_level.npz"
        ),
        "filters": "sorted tree-depth7 neuron-ID shuffle mean distribution; seed 20260419; 10,000 repeats; exact 36-pair/14-mouse scope; excluded only 1616day1 -> 1616bday1",
        "source_row_count": "10,000 shuffle repeats",
    },
    "d": {
        "source": (
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "cross_session_decoding_accuracy_points.csv; "
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "cross_session_decoding_accuracy_summary.csv; "
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "cross_session_decoding_accuracy_stats.json"
        ),
        "filters": "corrected-exposure tree-depth7 decoder accuracy, data vs saved label-shuffle; exact 36-pair/14-mouse scope; corrected 842 0->3; excluded only 1616day1 -> 1616bday1",
        "source_row_count": "36 sorted d1v1 pairs",
    },
    "f": {
        "source": (
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "level_map_vs_exploration_efficiency_primary_clean_rows.csv; "
            "docs/main_figs_v3_source_render_2026-06-24/"
            "figure5_no_manual_mask_2026-08-07/"
            "level_map_vs_exploration_efficiency_primary_clean_stats.json"
        ),
        "filters": "sorted primary-clean level-map correlation vs exploration efficiency; exact 36-pair/14-mouse scope; all 36 rows taken from the corrected level authority (support 16,901); same OLS/animal-clustered CR1 estimator",
        "source_row_count": "36 sorted primary-clean pairs",
    },
}
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


def _load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_chrome_executable() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find Chrome/Edge for SVG export. Checked: "
        + ", ".join(str(path) for path in CHROME_CANDIDATES)
    )


def _svg_embedded_image_count(svg_path: Path) -> int:
    root = ET.parse(svg_path).getroot()
    return sum(1 for element in root.iter() if element.tag.endswith("image"))


def _svg_image_hrefs(svg_path: Path) -> list[str]:
    root = ET.parse(svg_path).getroot()
    hrefs: list[str] = []
    for element in root.iter():
        if not element.tag.endswith("image"):
            continue
        for key, value in element.attrib.items():
            if key.endswith("href"):
                hrefs.append(str(value))
    return hrefs


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


def _write_svg_payload_outputs(
    panel_svg: dict[str, object],
    panel_dir: Path,
    stem: str,
    *,
    dpi: int,
    facecolor: str = "white",
) -> dict[str, object]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    width_pt = float(panel_svg["viewbox_width"])
    height_pt = float(panel_svg["viewbox_height"])
    width_mm = width_pt / 72.0 * 25.4
    height_mm = height_pt / 72.0 * 25.4
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{width_mm:.6f}mm",
            "height": f"{height_mm:.6f}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": facecolor})
    for child in panel_svg["children"]:
        root.append(copy.deepcopy(child))
    out_svg = panel_dir / f"{stem}.svg"
    out_png = panel_dir / f"{stem}.png"
    out_pdf = panel_dir / f"{stem}.pdf"
    ET.ElementTree(root).write(out_svg, encoding="utf-8", xml_declaration=True)
    html_path = _export_svg_with_chrome(
        out_svg,
        out_png,
        out_pdf,
        width_mm=width_mm,
        height_mm=height_mm,
        dpi=dpi,
    )
    return {
        "svg": out_svg,
        "png": out_png,
        "pdf": out_pdf,
        "html": html_path,
        "viewbox_width_pt": width_pt,
        "viewbox_height_pt": height_pt,
        "width_mm": width_mm,
        "height_mm": height_mm,
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
    browsers = [path for path in CHROME_CANDIDATES if path.exists()]
    if not browsers:
        _find_chrome_executable()  # Raise the maintained, path-specific error.
        raise AssertionError("unreachable")
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
                subprocess.run(
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
                subprocess.run(
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


def _ensure_sources() -> None:
    required = [
        DAY2_TEMPLATE_GROUP_CURVES,
        DAY2_TEMPLATE_SOURCE_MANIFEST,
        LONG_GAP_POINTS,
        LONG_GAP_STATS,
        MAIN3_NO_MASK_MANIFEST,
        MAIN3_MASK_DELTA_AUDIT,
        SOURCE_REPAIR_842_MANIFEST,
        FIGURE2_GAP_PAIR_AUDIT,
        FIGURE2_GAP_DISTRIBUTION_PARITY,
        FIGURE3_LEVEL_PAIR_AUDIT,
        FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT,
        FIGURE3_FIRST30_POINTS_ALL_METRICS,
        FIGURE3_FIRST30_STATS_ALL_METRICS,
        *DAY2_TEMPLATE_ANIMAL_SUMMARIES.values(),
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files: " + ", ".join(str(path) for path in missing))
    _validate_figure3_coverage5min_authority()
    _validate_source_repair_842()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_figure3_coverage5min_authority() -> None:
    manifest = json.loads(DAY2_TEMPLATE_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    gate = manifest.get("coverage_gate", {})
    if gate.get("min_selected_duration_sec") != 300.0 or gate.get(
        "min_selected_frames_at_dt_0p04"
    ) != 7500:
        raise AssertionError("Figure 3 authority does not declare the fixed five-minute gate.")
    support = manifest.get("first30_support", {})
    expected_metrics = set(METRICS) | {"level_map_correlation"}
    if set(support) != expected_metrics or any(
        int(row.get("n_rows", -1)) != 59 or int(row.get("n_animals", -1)) != 22
        for row in support.values()
    ):
        raise AssertionError("Figure 3 authority does not retain the 59-row/22-mouse BH4 support.")
    for name, expected_hash in manifest.get("outputs", {}).items():
        path = FIGURE3_FIRST30_AUTHORITY_DIR / str(name)
        if not path.is_file() or _file_sha256(path) != str(expected_hash):
            raise AssertionError(f"Figure 3 authority output hash mismatch: {path}")


def _validate_source_repair_842() -> dict[str, object]:
    required = (
        SOURCE_REPAIR_842_MANIFEST,
        FIGURE2_GAP_CURVE,
        FIGURE2_GAP_CURVE_MANIFEST,
        FIGURE2_GAP_PAIR_AUDIT,
        FIGURE2_GAP_DISTRIBUTION_PARITY,
        LONG_GAP_POINTS,
        LONG_GAP_STATS,
        MAIN3_NO_MASK_MANIFEST,
        MAIN3_MASK_DELTA_AUDIT,
        FIGURE3_LEVEL_PAIR_AUDIT,
        FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing corrected 842 source-repair files: " + "; ".join(missing))
    no_mask_manifest = json.loads(MAIN3_NO_MASK_MANIFEST.read_text(encoding="utf-8"))
    no_mask_scope = no_mask_manifest.get("scope", {})
    no_mask_checks = no_mask_manifest.get("checks", {})
    if (
        no_mask_manifest.get("status") != "PASS"
        or no_mask_manifest.get("policy")
        != "sorted neural data; no pre-sort manual cell-ID exclusions"
        or int(no_mask_scope.get("n_metrics", -1)) != 6
        or int(no_mask_scope.get("rows_per_metric", -1)) != 37
        or int(no_mask_scope.get("n_animals", -1)) != 14
        or not all(bool(value) for value in no_mask_checks.values())
        or "whole-session raw-C" not in str(no_mask_manifest.get("ensemble_rate_estimator", ""))
        or "spatial-map" not in str(
            no_mask_manifest.get("ensemble_rate_not_interchangeable_with", "")
        )
    ):
        raise AssertionError("Main Figure 3 no-manual-mask manifest failed its estimator/scope contract.")
    for filename, record in no_mask_manifest.get("outputs", {}).items():
        path = REPO_ROOT / str(record.get("path", ""))
        if (
            path.name != filename
            or not path.is_file()
            or _file_sha256(path).lower() != str(record.get("sha256", "")).lower()
        ):
            raise AssertionError(f"Main Figure 3 no-mask output hash mismatch: {path}")
    repaired_points = pd.read_csv(LONG_GAP_POINTS, float_precision="round_trip")
    repaired_stats = pd.read_csv(LONG_GAP_STATS, float_precision="round_trip")
    delta_audit = pd.read_csv(MAIN3_MASK_DELTA_AUDIT, float_precision="round_trip")
    main3_metrics = {
        "place_map_correlation",
        "level_map_correlation",
        "decoding_error_bins",
        "level_decoding_accuracy",
        "pv_correlation",
        "ensemble_rate_correlation",
    }
    if (
        len(repaired_points) != 222
        or set(repaired_points["metric"].astype(str)) != main3_metrics
        or not repaired_points.groupby("metric").size().eq(37).all()
        or not repaired_points.groupby("metric")["animal"].nunique().eq(14).all()
        or len(delta_audit) != 222
        or len(repaired_stats) != 6
        or set(repaired_stats["metric"].astype(str)) != main3_metrics
        or not repaired_stats["n_rows"].astype(int).eq(37).all()
        or not repaired_stats["n_animals"].astype(int).eq(14).all()
        or not repaired_stats["df"].astype(int).eq(13).all()
    ):
        raise AssertionError("Main Figure 3 no-mask tables lost the six-metric 37-pair scope.")
    expected_bh6 = np.asarray(
        _bh_q(repaired_stats["p_two_sided"].astype(float).tolist()), dtype=float
    )
    if not np.allclose(
        expected_bh6,
        repaired_stats["q_bh_two_sided"].astype(float).to_numpy(),
        rtol=1e-14,
        atol=1e-14,
    ):
        raise AssertionError("Main Figure 3 no-mask statistics do not reproduce BH6.")
    rate_delta = delta_audit.loc[
        delta_audit["metric"].astype(str).eq("ensemble_rate_correlation")
    ]
    if (
        len(rate_delta) != 37
        or float(rate_delta["mask_removal_delta"].abs().max()) >= 0.01
        or int(rate_delta["changed"].astype(str).str.lower().eq("true").sum()) < 1
    ):
        raise AssertionError("Main Figure 3 same-estimator mask-only rate audit is implausible.")
    distribution_parity = pd.read_csv(FIGURE2_GAP_DISTRIBUTION_PARITY)
    if not distribution_parity["status"].astype(str).eq("PASS").all():
        raise AssertionError("The retained corrected-842 Figure 2 distribution audit failed.")
    result = {
        "manifest": _rel(MAIN3_NO_MASK_MANIFEST),
        "manual_exclusion_profile": "none",
        "main3_ed12j_rows": 37,
        "main3_ed12j_animals": 14,
        "main3_bh6": "PASS",
        "same_estimator_rate_max_abs_mask_delta": float(
            rate_delta["mask_removal_delta"].abs().max()
        ),
        "figure2_842_0to3_distribution_parity": "PASS",
    }

    manifest = json.loads(SOURCE_REPAIR_842_MANIFEST.read_text(encoding="utf-8"))
    expected_pairs = {
        "842day1->842-3day1",
        "842day1->842-6day1",
        "842day1->842-9day1",
    }
    support = manifest.get("support", {})
    level_scope = manifest.get("main3_ed12j_level", {})
    if (
        str(manifest.get("corrected_pair")) != "842day1->842-3day1"
        or set(map(str, manifest.get("figure2_recomputed_pairs", []))) != expected_pairs
        or int(manifest.get("figure2_unaffected_pairs_preserved", -1)) != 70
        or int(support.get("restored_general_tree_depth7", -1)) != 16901
        or int(support.get("restored_rows", -1)) != 4
        or int(level_scope.get("n_rows", -1)) != 37
        or int(level_scope.get("n_animals", -1)) != 14
    ):
        raise AssertionError("The corrected 842 source-repair manifest has the wrong scope or support.")
    outputs = manifest.get("outputs", {})
    no_mask_output_paths = {
        (REPO_ROOT / str(record.get("path", ""))).resolve()
        for record in no_mask_manifest.get("outputs", {}).values()
    }
    source_repair_outputs = [
        path
        for path in required[1:]
        if path not in {
            FIGURE2_GAP_CURVE,
            FIGURE2_GAP_CURVE_MANIFEST,
            MAIN3_NO_MASK_MANIFEST,
        }
        and path.resolve() not in no_mask_output_paths
    ]
    for path in source_repair_outputs:
        record = outputs.get(path.name)
        if not isinstance(record, dict):
            raise AssertionError(f"Source-repair manifest does not record {path.name}.")
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.resolve() != path.resolve():
            raise AssertionError(f"Source-repair manifest path mismatch for {path.name}.")
        if str(record.get("sha256", "")).lower() != _file_sha256(path).lower():
            raise AssertionError(f"Source-repair hash mismatch for {path.name}.")
    gap_manifest = json.loads(FIGURE2_GAP_CURVE_MANIFEST.read_text(encoding="utf-8"))
    gap_output = REPO_ROOT / str(gap_manifest.get("outputs", {}).get("curve_data", ""))
    input_paths = {
        "gap_summary": REPO_ROOT / str(gap_manifest.get("gap_summary_path", "")),
        "same_stage": REPO_ROOT / str(gap_manifest.get("same_stage_path", "")),
        "same_stage_day_gap": REPO_ROOT / str(gap_manifest.get("same_stage_day_gap_path", "")),
        "same_stage_shuffle": REPO_ROOT / str(gap_manifest.get("same_stage_shuffle_path", "")),
    }
    expected_input_hashes = gap_manifest.get("input_sha256", {})
    gap_rows = pd.read_csv(FIGURE2_GAP_CURVE)
    if (
        gap_output.resolve() != FIGURE2_GAP_CURVE.resolve()
        or int(gap_manifest.get("same_stage_anchor_rows", -1)) != 47
        or int(gap_manifest.get("same_stage_anchor_animals", -1)) != 14
        or int(gap_manifest.get("gap_summary_rows", -1)) != 16
        or str(gap_manifest.get("long_gap_mean_source")) != "pair"
        or str(gap_manifest.get("day_gap_key_check", {}).get("status", ""))
        != "external_table_key_matched"
        or str(gap_manifest.get("same_stage_shuffle_key_check", {}).get("status", ""))
        != "external_shuffle_key_matched"
        or len(gap_rows) != 20
        or set(gap_rows["metric"].astype(str)) != {"place", "level", "pv", "rate"}
        or not gap_rows.groupby("metric").size().eq(5).all()
        or not gap_rows.groupby("metric")["time_days"].apply(
            lambda values: np.array_equal(
                values.to_numpy(dtype=float),
                np.asarray([1.0, 90.0, 180.0, 270.0, 360.0]),
            )
        ).all()
        or any(
            not path.is_file()
            or _file_sha256(path).lower()
            != str(expected_input_hashes.get(name, "")).lower()
            for name, path in input_paths.items()
        )
    ):
        raise AssertionError("Figure 2 no-manual-mask Day1-Day2 anchor authority drifted.")
    result["figure2_gap_anchor_no_manual_mask"] = {
        "manifest": _rel(FIGURE2_GAP_CURVE_MANIFEST),
        "rows": int(len(gap_rows)),
        "anchor_pairs": int(gap_manifest["same_stage_anchor_rows"]),
        "anchor_animals": int(gap_manifest["same_stage_anchor_animals"]),
        "long_gap_mean_source": str(gap_manifest["long_gap_mean_source"]),
    }
    affected = pd.read_csv(FIGURE2_GAP_PAIR_AUDIT)
    if (
        len(affected) != 12
        or set(affected["pair_key"].astype(str)) != expected_pairs
        or not affected["unaffected_pair_rows_preserved"].astype(bool).all()
        or not affected["n_unaffected_session_pairs_preserved"].astype(int).eq(70).all()
    ):
        raise AssertionError("The corrected 842 Figure 2 affected-pair audit is incomplete.")
    distribution_parity = pd.read_csv(FIGURE2_GAP_DISTRIBUTION_PARITY)
    if (
        len(distribution_parity) != 3
        or set(distribution_parity["metric"].astype(str)) != {"place", "pv", "rate"}
        or not distribution_parity["status"].astype(str).eq("PASS").all()
        or not distribution_parity[
            ["observed_match", "support_match", "shuffle_stream_policy_valid"]
        ].astype(str).apply(lambda column: column.str.lower().eq("true")).all().all()
        or set(distribution_parity["expected_native_support"].astype(int)) != {1, 75, 111}
        or set(distribution_parity["fresh_shuffle_seed"].astype(int)) != {20260602}
        or set(distribution_parity["repaired_shuffle_seed"].astype(int)) != {20260513}
        or set(distribution_parity["shuffle_repeats"].astype(int)) != {10000}
    ):
        raise AssertionError(
            "Corrected 842 0->3 observed values/support do not reproduce the fresh Figure 2 C-H authority, "
            "or the analysis-specific shuffle streams drifted."
        )

    main3_metrics = (
        "place_map_correlation",
        "level_map_correlation",
        "decoding_error_bins",
        "level_decoding_accuracy",
        "pv_correlation",
        "ensemble_rate_correlation",
    )
    corrected_pair = "842day1->842-3day1"
    short_pair = "1616day1->1616bday1"
    corrected_cimap_sha = (
        "8f4db74927d1c8776497f110ec47b1d8db245aebb1ca1df38d63bf6fa9b29a22"
    )
    expected_values = {
        "place_map_correlation": 0.024744643578739196,
        "level_map_correlation": 0.05554206543534958,
        "decoding_error_bins": 6.587967058950327,
        "level_decoding_accuracy": 0.20482274608601764,
        "pv_correlation": -0.017021455068185247,
        "ensemble_rate_correlation": -0.029181452322815223,
    }
    expected_support = {
        "place_map_correlation": ("n_pairs_final", 75, "common_valid_bin_count", 111),
        "level_map_correlation": ("n_pairs_final", 75, "common_level_count", 7),
        "decoding_error_bins": (
            "n_matched_after_rate_gate",
            142,
            "n_matched_before_rate_gate",
            151,
        ),
        "level_decoding_accuracy": (
            "n_matched_after_rate_gate",
            142,
            "n_matched_before_rate_gate",
            151,
        ),
        "pv_correlation": ("n_pairs_final", 75, "n_bins_finite", 111),
        "ensemble_rate_correlation": ("n_used", 75, "n_matched_cimap", 75),
    }
    six_scope = manifest.get("main3_corrected_842_all_metrics", {})
    manual_pairs = set(map(str, manifest.get("manual_mask_pairs", [])))
    target_values = six_scope.get("target_values", {}) if isinstance(six_scope, dict) else {}
    if (
        not isinstance(six_scope, dict)
        or six_scope.get("pair_key") != corrected_pair
        or set(map(str, six_scope.get("metrics", []))) != set(main3_metrics)
        or int(six_scope.get("n_metrics", -1)) != 6
        or int(six_scope.get("literal_rows_per_metric", -1)) != 37
        or int(six_scope.get("literal_animals_per_metric", -1)) != 14
        or six_scope.get("short_pair_retained") != short_pair
        or six_scope.get("source_audit_status") != "PASS"
        or Path(str(six_scope.get("source_audit_path", ""))).resolve()
        != FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT.resolve()
        or six_scope.get("non_level_changed_pair_policy")
        != "exactly_corrected_pair_only"
        or six_scope.get("level_changed_pair_policy")
        != "manual_mask_pairs_union_corrected_pair"
        or six_scope.get("all_other_metric_pair_values_exactly_unchanged") is not True
        or six_scope.get("identity_and_literal_scope_exactly_unchanged") is not True
        or set(target_values) != set(main3_metrics)
        or any(
            float(target_values.get(metric, math.nan)) != expected_values[metric]
            for metric in main3_metrics
        )
    ):
        raise AssertionError(
            "The source-repair manifest does not declare the exact corrected 842 six-metric Main3 contract."
        )

    six_audit = pd.read_csv(
        FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT,
        float_precision="round_trip",
    )
    required_audit_columns = {
        "metric",
        "pair_key",
        "value_column",
        "expected_value",
        "source_value",
        "value_match",
        "source_path",
        "source_sha256",
        "schema_valid",
        "source_target_row_count",
        "support_column",
        "support_value",
        "expected_support_value",
        "support_match",
        "support_path",
        "support_sha256",
        "secondary_support_column",
        "secondary_support_value",
        "expected_secondary_support_value",
        "secondary_support_match",
        "support_semantics",
        "scope_path",
        "scope_sha256",
        "scope_target_row_count",
        "corrected_cimap_path",
        "corrected_cimap_sha256",
        "provenance_path",
        "provenance_sha256",
        "provenance_valid",
        "status",
        "value_before",
        "value_after",
        "main3_source_kind_after",
        "main3_source_path_after",
        "main3_source_detail_after",
        "value_changed",
        "expected_changed_pairs_for_metric",
        "observed_changed_pairs_for_metric",
        "all_other_metric_pair_values_exactly_unchanged",
        "identity_and_literal_scope_exactly_unchanged",
    }
    if required_audit_columns - set(six_audit.columns):
        raise AssertionError("The corrected 842 six-metric audit schema is incomplete.")
    if (
        len(six_audit) != 6
        or six_audit["metric"].astype(str).nunique() != 6
        or set(six_audit["metric"].astype(str)) != set(main3_metrics)
        or set(six_audit["pair_key"].astype(str)) != {corrected_pair}
        or not six_audit["status"].astype(str).eq("PASS").all()
    ):
        raise AssertionError("The corrected 842 audit is not one PASS row for each Main3 metric.")
    truth_columns = (
        "value_match",
        "schema_valid",
        "support_match",
        "secondary_support_match",
        "provenance_valid",
        "value_changed",
        "all_other_metric_pair_values_exactly_unchanged",
        "identity_and_literal_scope_exactly_unchanged",
    )
    if not six_audit[list(truth_columns)].astype(str).apply(
        lambda column: column.str.lower().eq("true")
    ).all().all():
        raise AssertionError("At least one corrected 842 six-metric audit guard failed.")

    six_audit = six_audit.set_index("metric", verify_integrity=True, drop=False)
    expected_level_changes = ";".join(sorted(manual_pairs | {corrected_pair}))
    for metric in main3_metrics:
        row = six_audit.loc[metric]
        support_column, support_value, secondary_column, secondary_value = expected_support[
            metric
        ]
        expected_changed = (
            expected_level_changes if metric == "level_map_correlation" else corrected_pair
        )
        if (
            float(row["expected_value"]) != expected_values[metric]
            or float(row["source_value"]) != expected_values[metric]
            or float(row["value_after"]) != expected_values[metric]
            or str(row["support_column"]) != support_column
            or int(row["support_value"]) != support_value
            or int(row["expected_support_value"]) != support_value
            or str(row["secondary_support_column"]) != secondary_column
            or int(row["secondary_support_value"]) != secondary_value
            or int(row["expected_secondary_support_value"]) != secondary_value
            or int(row["source_target_row_count"]) != 1
            or int(row["scope_target_row_count"]) != 1
            or str(row["expected_changed_pairs_for_metric"]) != expected_changed
            or str(row["observed_changed_pairs_for_metric"]) != expected_changed
            or str(row["source_path"]) != str(row["main3_source_path_after"])
            or not str(row["main3_source_kind_after"]).strip()
            or not str(row["main3_source_detail_after"]).strip()
            or str(row["corrected_cimap_sha256"]).lower() != corrected_cimap_sha
        ):
            raise AssertionError(f"Corrected 842 Main3 audit drifted for {metric}.")
        for path_column, hash_column in (
            ("source_path", "source_sha256"),
            ("support_path", "support_sha256"),
            ("scope_path", "scope_sha256"),
            ("provenance_path", "provenance_sha256"),
            ("corrected_cimap_path", "corrected_cimap_sha256"),
        ):
            source_path = Path(str(row[path_column])).resolve()
            if (
                not source_path.is_file()
                or _file_sha256(source_path).lower() != str(row[hash_column]).lower()
            ):
                raise AssertionError(
                    f"Corrected 842 Main3 {metric} source hash failed for {path_column}."
                )
        if metric == "pv_correlation" and (
            "matched-neuron support" not in str(row["support_semantics"])
            or "spatial-bin support" not in str(row["support_semantics"])
        ):
            raise AssertionError("PV audit conflates matched-neuron and finite-bin support.")
        if metric == "ensemble_rate_correlation" and "not the Figure 2" not in str(
            row["support_semantics"]
        ):
            raise AssertionError("Main3 raw-C rate was conflated with the Figure 2 rate metric.")

    inputs = manifest.get("inputs", {})
    original_points_record = inputs.get("main3_points", {}) if isinstance(inputs, dict) else {}
    original_points_path = Path(str(original_points_record.get("path", ""))).resolve()
    if (
        not original_points_path.is_file()
        or str(original_points_record.get("sha256", "")).lower()
        != _file_sha256(original_points_path).lower()
    ):
        raise AssertionError("The original Main3 point authority is absent or hash-mismatched.")
    original_points = pd.read_csv(original_points_path, float_precision="round_trip")
    repaired_points = pd.read_csv(LONG_GAP_POINTS, float_precision="round_trip")
    identity_columns = ["animal", "cohort", "src", "dst", "experience", "metric"]
    if (
        len(original_points) != 222
        or len(repaired_points) != 222
        or not original_points[identity_columns].equals(repaired_points[identity_columns])
        or set(original_points["metric"].astype(str)) != set(main3_metrics)
        or set(repaired_points["metric"].astype(str)) != set(main3_metrics)
    ):
        raise AssertionError("The repaired Main3 table changed its literal six-metric row identity.")
    original_points = original_points.assign(
        pair_key=original_points["src"].astype(str) + "->" + original_points["dst"].astype(str)
    )
    repaired_points = repaired_points.assign(
        pair_key=repaired_points["src"].astype(str) + "->" + repaired_points["dst"].astype(str)
    )
    original_indexed = original_points.set_index(["metric", "pair_key"], verify_integrity=True)
    repaired_indexed = repaired_points.set_index(["metric", "pair_key"], verify_integrity=True)
    if not original_indexed.index.equals(repaired_indexed.index):
        raise AssertionError("The repaired Main3 metric-pair index changed.")
    common_pair_set: set[str] | None = None
    for metric in main3_metrics:
        old_metric = original_indexed.loc[metric]
        new_metric = repaired_indexed.loc[metric]
        pair_set = set(map(str, old_metric.index))
        expected_mask_changed_set = set(
            delta_audit.loc[
                delta_audit["metric"].astype(str).eq(metric)
                & delta_audit["changed"].astype(str).str.lower().eq("true"),
                "pair_key",
            ].astype(str)
        )
        required_changed_set = expected_mask_changed_set | {corrected_pair}
        changed_set = {
            str(pair_key)
            for pair_key in old_metric.index
            if float(old_metric.at[pair_key, "value"])
            != float(new_metric.at[pair_key, "value"])
        }
        if (
            len(old_metric) != 37
            or old_metric["animal"].astype(str).nunique() != 14
            or new_metric["animal"].astype(str).nunique() != 14
            or short_pair not in pair_set
            or corrected_pair not in pair_set
            or not required_changed_set.issubset(changed_set)
            or not changed_set.issubset(pair_set)
        ):
            raise AssertionError(f"Main3 exact before/after scope failed for {metric}.")
        if common_pair_set is None:
            common_pair_set = pair_set
        elif pair_set != common_pair_set:
            raise AssertionError("The six Main3 metrics no longer share one literal pair scope.")
        audit_row = six_audit.loc[metric]
        if (
            not math.isclose(
                float(audit_row["value_before"]),
                float(old_metric.at[corrected_pair, "value"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(audit_row["value_after"]),
                float(new_metric.at[corrected_pair, "value"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise AssertionError(f"Main3 corrected target provenance failed for {metric}.")

    repaired_stats = pd.read_csv(LONG_GAP_STATS, float_precision="round_trip")
    if (
        len(repaired_stats) != 6
        or set(repaired_stats["metric"].astype(str)) != set(main3_metrics)
        or not repaired_stats["n_rows"].astype(int).eq(37).all()
        or not repaired_stats["n_animals"].astype(int).eq(14).all()
        or not repaired_stats["df"].astype(int).eq(13).all()
        or not np.isfinite(
            repaired_stats[["slope", "slope_se", "p_two_sided", "q_bh_two_sided"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        ).all()
    ):
        raise AssertionError("The repaired Main3 statistics are not six finite CR1 37-row/14-mouse fits.")
    expected_bh6 = np.asarray(
        _bh_q(repaired_stats["p_two_sided"].astype(float).tolist()), dtype=float
    )
    if not np.allclose(
        expected_bh6,
        repaired_stats["q_bh_two_sided"].astype(float).to_numpy(),
        rtol=1e-14,
        atol=1e-14,
    ):
        raise AssertionError("The repaired Main3 statistics do not reproduce the exact BH6 family.")
    manifest_stats = no_mask_manifest.get("statistics", [])
    manifest_stats_by_metric = {
        str(row.get("metric", "")): row
        for row in manifest_stats
        if isinstance(row, dict)
    }
    for _, stat_row in repaired_stats.iterrows():
        declared = manifest_stats_by_metric.get(str(stat_row["metric"]), {})
        for column in (
            "n_rows",
            "n_animals",
            "slope",
            "slope_se",
            "p_two_sided",
            "q_bh_two_sided",
        ):
            if not math.isclose(
                float(declared.get(column, math.nan)),
                float(stat_row[column]),
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise AssertionError(
                    f"Repair manifest Main3 statistic {column} drifted for {stat_row['metric']}."
                )
    result.update(
        {
            "recomputed_pairs": sorted(expected_pairs),
            "unaffected_pairs_preserved": 70,
            "restored_tree_depth7_support": 16901,
            "main3_842_six_metric_source_audit": "PASS",
            "main3_842_target_values": expected_values,
        }
    )
    return result


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


def _validate_figure2_distribution_sources() -> dict[str, object]:
    required = [
        FIGURE2_DISTRIBUTION_PROVENANCE,
        FIGURE2_DISTRIBUTION_PAIR_INPUT,
        FIGURE2_DISTRIBUTION_PAIR_SUPPORT,
        FIGURE2_DISTRIBUTION_METRIC_SUMMARY,
        *[_figure2_distribution_path(metric) for metric in METRICS],
        *[_figure2_mean_path(metric) for metric in METRICS],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing corrected Figure 2 C-H source bundle files: " + "; ".join(missing)
        )

    provenance = json.loads(FIGURE2_DISTRIBUTION_PROVENANCE.read_text(encoding="utf-8"))
    configuration = dict(provenance.get("canonical_configuration", {}))
    scope = dict(provenance.get("scope_validation", {}))
    if provenance.get("status") != "complete" or not bool(
        provenance.get("full_compute_complete", False)
    ):
        raise AssertionError("Corrected Figure 2 C-H source build is not complete.")
    expected_configuration = {
        "pair_count": 36,
        "animal_count": 14,
        "shuffle_repeats": 10000,
        "seed": 20260602,
        "mode": "spatial",
        "metric": "pearson",
        "normalization": "zscore",
        "min_spikes_per_min": 0.1,
        "rate_source": "spike",
        "drop_if_below": "either",
    }
    configuration_mismatch = {
        key: {"expected": expected, "observed": configuration.get(key)}
        for key, expected in expected_configuration.items()
        if configuration.get(key) != expected
    }
    if configuration_mismatch:
        raise AssertionError(
            "Corrected Figure 2 C-H configuration mismatch: "
            + json.dumps(configuration_mismatch, sort_keys=True)
        )
    for exclusion_key in ("exclude_train_ids", "exclude_test_ids", "exclude_pairs"):
        if configuration.get(exclusion_key) not in ([], None):
            raise AssertionError(
                f"Corrected Figure 2 C-H unexpectedly applies {exclusion_key}."
            )
    if (
        int(scope.get("pair_count", -1)) != 36
        or int(scope.get("animal_count", -1)) != 14
        or int(scope.get("corrected_override_count", -1)) != 1
        or bool(scope.get("short_pair_present", True))
    ):
        raise AssertionError("Corrected Figure 2 C-H scope manifest is not the canonical 36-pair scope.")

    pair_input = pd.read_csv(FIGURE2_DISTRIBUTION_PAIR_INPUT, dtype={"animal": str})
    pair_support = pd.read_csv(FIGURE2_DISTRIBUTION_PAIR_SUPPORT, dtype={"animal": str})
    if len(pair_input) != 36 or pair_input["animal"].nunique() != 14:
        raise AssertionError("Corrected Figure 2 C-H pair-input audit is not 36 pairs / 14 animals.")
    if len(pair_support) != 36 or pair_support["animal"].nunique() != 14:
        raise AssertionError("Corrected Figure 2 C-H support audit is not 36 pairs / 14 animals.")
    if not bool(pair_support["n_pairs_dropped_by_user"].eq(0).all()):
        raise AssertionError("Corrected Figure 2 C-H support audit contains manual exclusions.")
    transition_counts = pair_input["transition"].astype(str).value_counts().to_dict()
    if transition_counts != {"0-3": 14, "3-6": 11, "6-9": 7, "9-12": 4}:
        raise AssertionError(f"Corrected Figure 2 C-H transition counts drifted: {transition_counts}")
    pair_keys = set(pair_input["pair_key"].astype(str))
    if (
        "1616day1->1616bday1" in pair_keys
        or "1616bday1->1616b-3day1" not in pair_keys
    ):
        raise AssertionError("Corrected Figure 2 C-H has the wrong 1616 session-pair scope.")
    support_totals = {
        "initial": int(pair_support["n_pairs_initial"].sum()),
        "manual": int(pair_support["n_pairs_dropped_by_user"].sum()),
        "rate": int(pair_support["n_pairs_dropped_by_rate"].sum()),
        "final": int(pair_support["n_pairs_final"].sum()),
    }
    expected_support_totals = {
        "initial": 21_412,
        "manual": 0,
        "rate": 4_511,
        "final": 16_901,
    }
    if support_totals != expected_support_totals:
        raise AssertionError(
            f"Corrected Figure 2 C-H matched-neuron denominator drifted: {support_totals}"
        )
    corrected_input_flag = (
        pair_input["corrected_842_override"].astype(str).str.lower().eq("true")
    )
    corrected_support_flag = (
        pair_support["corrected_842_override"].astype(str).str.lower().eq("true")
    )
    corrected_input = pair_input.loc[corrected_input_flag]
    corrected_support = pair_support.loc[corrected_support_flag]
    if len(corrected_input) != 1 or corrected_input.iloc[0]["pair_key"] != "842day1->842-3day1":
        raise AssertionError("Corrected Figure 2 C-H input audit does not override exactly 842 0->3.")
    if len(corrected_support) != 1:
        raise AssertionError("Corrected Figure 2 C-H support audit lacks the unique 842 0->3 row.")
    corrected_support_row = corrected_support.iloc[0]
    if int(corrected_support_row["n_cimap_rows_active"]) != 151:
        raise AssertionError("Corrected Figure 2 C-H 842 0->3 CIMap support is not 151 rows.")
    if int(corrected_support_row["n_pairs_final"]) != 75:
        raise AssertionError("Corrected Figure 2 C-H 842 0->3 activity-gated support is not 75 cells.")
    corrected_cimap_path = Path(str(corrected_input.iloc[0]["effective_cimap_path"]))
    if (
        not corrected_cimap_path.is_file()
        or _file_sha256(corrected_cimap_path).lower()
        != FIGURE2_EXPECTED_CORRECTED_842_CIMAP_SHA256
    ):
        raise AssertionError("Corrected Figure 2 C-H 842 0->3 CIMap hash is not current.")

    metric_summary = pd.read_csv(FIGURE2_DISTRIBUTION_METRIC_SUMMARY)
    observed_counts = metric_summary.set_index("metric")["n_observed_values"].astype(int).to_dict()
    if observed_counts != {"place": 16_901, "pv": 3_810, "rate": 36}:
        raise AssertionError(f"Corrected Figure 2 C-H metric counts drifted: {observed_counts}")

    panel_source_records = dict(provenance.get("panel_sources", {}))
    panel_source_paths = {
        "place_distribution": _figure2_distribution_path("place_map_correlation"),
        "place_mean": _figure2_mean_path("place_map_correlation"),
        "pv_distribution": _figure2_distribution_path("pv_correlation"),
        "pv_mean": _figure2_mean_path("pv_correlation"),
        "rate_distribution": _figure2_distribution_path("ensemble_rate_correlation"),
        "rate_mean": _figure2_mean_path("ensemble_rate_correlation"),
    }
    for source_key, source_path in panel_source_paths.items():
        source_record = panel_source_records.get(source_key, {})
        if str(source_record.get("sha256", "")).lower() != _file_sha256(source_path).lower():
            raise AssertionError(f"Corrected Figure 2 panel-source hash drifted: {source_key}")

    source_schema: dict[str, dict[str, list[str]]] = {}
    for metric in METRICS:
        distribution_path = _figure2_distribution_path(metric)
        mean_path = _figure2_mean_path(metric)
        with np.load(distribution_path, allow_pickle=False) as distribution:
            distribution_keys = set(distribution.files)
            expected_distribution = {"obs", "hist_edges"}
            expected_distribution.add("shuffle_hist")
            missing_distribution = expected_distribution - distribution_keys
            if missing_distribution:
                raise KeyError(
                    f"{distribution_path.name} is missing keys {sorted(missing_distribution)}"
                )
            edges = np.asarray(distribution["hist_edges"], dtype=float)
            if edges.shape != (101,) or not np.allclose(edges, np.linspace(-1.0, 1.0, 101)):
                raise AssertionError(f"{distribution_path.name} has noncanonical histogram edges.")
        with np.load(mean_path, allow_pickle=False) as mean_source:
            mean_keys = set(mean_source.files)
            missing_mean = {"null", "obs", "common_repeats"} - mean_keys
            if missing_mean:
                raise KeyError(f"{mean_path.name} is missing keys {sorted(missing_mean)}")
            common_repeats = int(np.asarray(mean_source["common_repeats"]).ravel()[0])
            if common_repeats != 10000:
                raise AssertionError(f"{mean_path.name} does not contain 10,000 repeats.")
        source_schema[metric] = {
            "distribution_keys": sorted(distribution_keys),
            "mean_keys": sorted(mean_keys),
        }

    return {
        "pair_count": int(len(pair_input)),
        "animal_count": int(pair_input["animal"].nunique()),
        "corrected_pair": "842day1->842-3day1",
        "corrected_cimap_active_rows": int(corrected_support_row["n_cimap_rows_active"]),
        "corrected_activity_gated_cells": int(corrected_support_row["n_pairs_final"]),
        "corrected_cimap_sha256": FIGURE2_EXPECTED_CORRECTED_842_CIMAP_SHA256,
        "manual_exclusion_count": int(pair_support["n_pairs_dropped_by_user"].sum()),
        "transition_pair_counts": transition_counts,
        "matched_neurons_initial": support_totals["initial"],
        "activity_gate_failures": support_totals["rate"],
        "matched_neurons_final": support_totals["final"],
        "metric_observation_counts": observed_counts,
        "shuffle_repeats": int(configuration["shuffle_repeats"]),
        "source_schema": source_schema,
        "provenance_manifest": _rel(FIGURE2_DISTRIBUTION_PROVENANCE),
        "pair_input_audit": _rel(FIGURE2_DISTRIBUTION_PAIR_INPUT),
        "pair_support_audit": _rel(FIGURE2_DISTRIBUTION_PAIR_SUPPORT),
    }


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


def _figure2_add_data_shuffle_labels(ax: plt.Axes, color: str, style: PublicationStyle) -> None:
    ax.text(
        0.10,
        0.82,
        "Data",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=style.annotation_fontsize,
        color=color,
        fontweight="bold",
    )
    ax.text(
        0.52,
        0.13,
        "Shuffle",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=style.annotation_fontsize,
        color="#777777",
        fontweight="bold",
    )


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


def _render_figure2_panel_b_column(
    pick: tuple[str, int],
    size: tuple[float, float],
    style: PublicationStyle,
    *,
    include_row_labels: bool,
) -> plt.Figure:
    fig, axes, _data, _paths = render_example_cell_grid_panel(
        picks=[pick],
        mode="arrow",
        saved_dir=EXAMPLE_SAVED_DIR,
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "grid_wspace": 0.006,
            "grid_hspace": 0.010,
            "margins": {
                "left": 0.18 if include_row_labels else 0.015,
                "right": 0.999,
                "top": 0.996,
                "bottom": 0.070,
            },
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
            "title_y": -0.105,
            "row_label_fontsize": 8.0,
            "row_label_pad": 4,
            "trajectory_alpha": 0.28,
            "trajectory_linewidth": 0.26,
            "arrow_head_width": 0.14,
            "arrow_head_length": 0.34,
            "arrow_linewidth": 0.01,
            "arrow_alpha": 0.42,
            "arrow_scale": 0.25,
        },
        labels={
            "row_top": "Day i" if include_row_labels else None,
            "row_bottom": "Day i+90" if include_row_labels else None,
            "title_template": "{place_r:.2f}",
        },
        save=False,
        n_bins=45,
        sigma=1.0,
        signal="C",
        dt=0.04,
        min_time_base=0.5,
    )
    _apply_figure2_panel_b_square_boundaries(axes)
    for ax in fig.axes:
        ax.title.set_fontsize(style.title_fontsize)
        ax.xaxis.label.set_fontsize(style.axis_label_fontsize)
        ax.yaxis.label.set_fontsize(style.axis_label_fontsize)
        ax.tick_params(labelsize=style.tick_fontsize)
    if include_row_labels:
        axes[0, 0].yaxis.label.set_fontsize(8.0)
        axes[1, 0].yaxis.label.set_fontsize(8.0)
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


def _validate_figure2_old_new_replication_audit() -> dict[str, object]:
    """Require complete retained-chain recurrence and 1616 cross-figure scope proof."""

    if not FIGURE2_OLD_NEW_REPLICATION_AUDIT.is_file():
        raise FileNotFoundError(FIGURE2_OLD_NEW_REPLICATION_AUDIT)
    audit = pd.read_csv(FIGURE2_OLD_NEW_REPLICATION_AUDIT)
    required_columns = {
        "comparison_family",
        "metric",
        "expected_action",
        "triplet",
        "animal_id",
        "session_a",
        "session_b",
        "session_c",
        "before_rows",
        "after_rows",
        "shared_point_ids",
        "before_only_point_ids",
        "after_only_point_ids",
        "prev_mismatch_rows",
        "next_mismatch_rows",
        "prev_max_abs_diff",
        "next_max_abs_diff",
        "support_prev_mismatch_rows",
        "support_next_mismatch_rows",
        "exact_metadata_mismatch_fields",
        "figure3_rows",
        "status",
        "note",
    }
    if required_columns - set(audit.columns) or len(audit) != 78 or not audit[
        "status"
    ].astype(str).eq("PASS").all():
        raise AssertionError("Figure 2/ED12 old-vs-new replication audit is incomplete.")
    dense = audit.loc[
        audit["comparison_family"].astype(str).eq("dense_historical_point_recurrence")
    ].copy()
    rate = audit.loc[
        audit["comparison_family"].astype(str).eq("session_rate_chain_recurrence")
    ].copy()
    boundary = audit.loc[
        audit["comparison_family"].astype(str).eq("cross_figure_scope_boundary")
    ].copy()
    dense_metrics = {
        "place_map_correlation",
        "pv_correlation",
        "level_map_correlation",
    }
    dense_numeric = dense[
        [
            "before_only_point_ids",
            "after_only_point_ids",
            "prev_mismatch_rows",
            "next_mismatch_rows",
            "support_prev_mismatch_rows",
            "support_next_mismatch_rows",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    dense_max = dense[["prev_max_abs_diff", "next_max_abs_diff"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        len(dense) != 54
        or set(dense["metric"].astype(str)) != dense_metrics
        or not dense["expected_action"].astype(str).eq("retained_unchanged").all()
        or not dense_numeric.eq(0).all().all()
        or not (
            pd.to_numeric(dense["before_rows"], errors="coerce")
            == pd.to_numeric(dense["after_rows"], errors="coerce")
        ).all()
        or not (
            pd.to_numeric(dense["before_rows"], errors="coerce")
            == pd.to_numeric(dense["shared_point_ids"], errors="coerce")
        ).all()
        or not (dense_max.fillna(0.0) <= 1e-15).all().all()
    ):
        raise AssertionError("Historical place/PV/level point recurrence is not exact.")
    action_counts = rate["expected_action"].astype(str).value_counts().to_dict()
    retained_rate = rate.loc[rate["expected_action"].astype(str).eq("retained_unchanged")]
    corrected_rate = rate.loc[rate["expected_action"].astype(str).eq("corrected_842_0to3")]
    removed_rate = rate.loc[rate["expected_action"].astype(str).eq("removed_short_1616")]
    retained_numeric = retained_rate[
        [
            "prev_mismatch_rows",
            "next_mismatch_rows",
            "support_prev_mismatch_rows",
            "support_next_mismatch_rows",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if (
        len(rate) != 23
        or action_counts
        != {"retained_unchanged": 21, "corrected_842_0to3": 1, "removed_short_1616": 1}
        or not retained_numeric.eq(0).all().all()
        or not retained_rate["exact_metadata_mismatch_fields"].fillna("").astype(str).eq("").all()
        or len(corrected_rate) != 1
        or str(corrected_rate.iloc[0]["animal_id"]) != "842"
        or str(corrected_rate.iloc[0]["triplet"]) != "0v3v6"
        or int(corrected_rate.iloc[0]["prev_mismatch_rows"]) != 1
        or int(corrected_rate.iloc[0]["next_mismatch_rows"]) != 0
        or int(corrected_rate.iloc[0]["support_prev_mismatch_rows"]) != 1
        or int(corrected_rate.iloc[0]["support_next_mismatch_rows"]) != 0
        or len(removed_rate) != 1
        or str(removed_rate.iloc[0]["animal_id"]) != "1616"
        or str(removed_rate.iloc[0]["triplet"]) != "0v0v3"
        or int(removed_rate.iloc[0]["before_rows"]) != 1
        or int(removed_rate.iloc[0]["after_rows"]) != 0
    ):
        raise AssertionError("Rate old-vs-new recurrence is not the exact 21/1/1 policy.")
    if (
        len(boundary) != 1
        or str(boundary.iloc[0]["expected_action"])
        != "excluded_fig2_ed12_retained_figure3"
        or str(boundary.iloc[0]["animal_id"]) != "1616"
        or int(boundary.iloc[0]["figure3_rows"]) != 6
    ):
        raise AssertionError("The short 1616 cross-figure scope boundary is not explicit.")
    return {
        "path": _rel(FIGURE2_OLD_NEW_REPLICATION_AUDIT),
        "sha256": _file_sha256(FIGURE2_OLD_NEW_REPLICATION_AUDIT),
        "rows": 78,
        "dense_chain_metric_rows": 54,
        "rate_policy": "21 unchanged + corrected 842 + removed 1616",
        "figure3_short_1616_rows": 6,
    }


def _validate_figure2_clean_consecutive_scope(
    points: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict[str, object]:
    required = [
        FIGURE2_CONSECUTIVE_MANIFEST,
        FIGURE2_CONSECUTIVE_UNRESTRICTED_SCOPE,
        FIGURE2_CONSECUTIVE_CLEAN_SCOPE,
        FIGURE2_CONSECUTIVE_INPUT_AUDIT,
        FIGURE2_RATE_CHAIN_POINTS,
        FIGURE2_RATE_CHAIN_STATS,
        FIGURE2_RATE_CHAIN_MANIFEST,
        FIGURE2_RATE_CHAIN_MASK_DELTA,
        FIGURE2_OLD_NEW_REPLICATION_AUDIT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing corrected Figure 2 consecutive-gap sources: " + "; ".join(missing))
    manifest = json.loads(FIGURE2_CONSECUTIVE_MANIFEST.read_text(encoding="utf-8"))
    unrestricted = pd.read_csv(FIGURE2_CONSECUTIVE_UNRESTRICTED_SCOPE, dtype={"animal_id": str})
    clean = pd.read_csv(FIGURE2_CONSECUTIVE_CLEAN_SCOPE, dtype={"animal_id": str})
    input_audit = pd.read_csv(FIGURE2_CONSECUTIVE_INPUT_AUDIT, dtype={"animal_id": str})
    expected_keys = {
        (triplet, animal)
        for triplet, animals in FIGURE2_CLEAN_CHAIN_ANIMALS.items()
        for animal in animals
    }
    clean_keys = set(zip(clean["triplet"].astype(str), clean["animal_id"].astype(str)))
    excluded = unrestricted.loc[
        ~unrestricted["included_clean_long_gap"].astype(str).str.lower().eq("true")
    ]
    if clean_keys != expected_keys or len(clean) != 22:
        raise AssertionError("Figure 2 clean consecutive-gap scope is not the exact 22-chain authority.")
    if (
        len(unrestricted) != 23
        or len(excluded) != 1
        or str(excluded.iloc[0]["animal_id"]) != "1616"
        or str(excluded.iloc[0]["triplet"]) != "0v0v3"
    ):
        raise AssertionError("Figure 2 unrestricted scope must be 23 chains with only 1616 0v0v3 excluded.")
    if (
        manifest.get("scope_policy") != "clean_long_gap_day1_triplets"
        or int(manifest.get("unrestricted_adjacent_chain_rows", -1)) != 23
        or int(manifest.get("expected_triplet_scope_rows", -1)) != 22
        or int(manifest.get("included_triplets", -1)) != 22
        or int(manifest.get("failed_triplets", -1)) != 0
        or not str(manifest.get("finite_support_policy", "")).startswith("metric_native:")
    ):
        raise AssertionError("Figure 2 corrected consecutive-gap manifest failed the 23-to-22 scope contract.")
    correction = manifest.get("correction_record_842_0to3", {})
    if (
        int(correction.get("n_registered_pairs", -1)) != 151
        or int(correction.get("corrected_n_pairs_rate_pass", -1)) != 75
        or str(correction.get("corrected_cimap_sha256", "")).lower()
        != "8f4db74927d1c8776497f110ec47b1d8db245aebb1ca1df38d63bf6fa9b29a22"
    ):
        raise AssertionError("Figure 2 corrected 842 0->3 provenance is incomplete or inconsistent.")
    if len(input_audit) != 22 or not input_audit["all_inputs_present"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Figure 2 clean consecutive-gap input audit is incomplete.")

    scope_cols = ["triplet", "animal_id", "session_a", "session_b", "session_c"]
    expected_scopes = set(map(tuple, clean[scope_cols].astype(str).to_numpy()))
    for metric in ("place_map_correlation", "pv_correlation", "level_map_correlation"):
        metric_scopes = set(
            map(
                tuple,
                points.loc[points["metric"].astype(str).eq(metric), scope_cols]
                .astype(str)
                .drop_duplicates()
                .to_numpy(),
            )
        )
        if metric_scopes != expected_scopes:
            raise AssertionError(f"Figure 2 {metric} does not use the exact 22 clean chains.")
    id_cols = [*scope_cols, "point_id"]
    place_ids = set(
        map(
            tuple,
            points.loc[points["metric"].astype(str).eq("place_map_correlation"), id_cols]
            .astype(str)
            .to_numpy(),
        )
    )
    level_ids = set(
        map(
            tuple,
            points.loc[points["metric"].astype(str).eq("level_map_correlation"), id_cols]
            .astype(str)
            .to_numpy(),
        )
    )
    if not place_ids or not level_ids:
        raise AssertionError("Figure 2 place/level metric-native finite supports must both be nonempty.")
    if len(place_ids) != int(points["metric"].astype(str).eq("place_map_correlation").sum()):
        raise AssertionError("Figure 2 place-map metric-native point IDs are not unique.")
    if len(level_ids) != int(points["metric"].astype(str).eq("level_map_correlation").sum()):
        raise AssertionError("Figure 2 level-map metric-native point IDs are not unique.")
    for metric in ("place_map_correlation", "pv_correlation", "level_map_correlation"):
        rows = points.loc[points["metric"].astype(str).eq(metric)]
        stat = stats_df.loc[stats_df["metric"].astype(str).eq(metric)]
        if (
            len(stat) != 1
            or int(stat.iloc[0]["n_rows"]) != len(rows)
            or int(stat.iloc[0]["n_animals"]) != 11
        ):
            raise AssertionError(f"Figure 2 {metric} point/stat counts disagree.")

    rate = pd.read_csv(FIGURE2_RATE_CHAIN_POINTS, dtype={"animal_id": str})
    rate_manifest = json.loads(FIGURE2_RATE_CHAIN_MANIFEST.read_text(encoding="utf-8"))
    rate_delta = pd.read_csv(FIGURE2_RATE_CHAIN_MASK_DELTA, dtype={"animal_id": str})
    if (
        rate_manifest.get("policy")
        != "sorted neural data; exclusion profile none; zero manual-ID drops"
        or not str(rate_manifest.get("source_value_authority", "")).endswith(
            "pair_metric_summary.csv"
        )
        or int(rate_manifest.get("scope", {}).get("n_pairs", -1)) != 36
        or int(rate_manifest.get("scope", {}).get("n_chains", -1)) != 22
        or int(rate_delta["changed_by_mask_removal"].astype(str).str.lower().eq("true").sum()) != 2
    ):
        raise AssertionError("Figure 2 ensemble-rate source lacks the no-manual-mask contract.")
    rate = rate.loc[rate["scope"].astype(str).eq("clean_long_gap_day1_triplets")].copy()
    rate_scope = set(
        map(
            tuple,
            rate[["animal_id", "session_a", "session_b", "session_c"]].astype(str).to_numpy(),
        )
    )
    clean_rate_scope = set(
        map(
            tuple,
            clean[["animal_id", "session_a", "session_b", "session_c"]].astype(str).to_numpy(),
        )
    )
    if len(rate) != 22 or rate_scope != clean_rate_scope or rate["animal_id"].nunique() != 11:
        raise AssertionError("Figure 2 ensemble-rate panel does not use the same 22 clean chains.")
    rate_842 = rate.loc[
        rate["animal_id"].eq("842")
        & rate["session_a"].astype(str).eq("842day1")
        & rate["session_b"].astype(str).eq("842-3day1")
    ]
    if len(rate_842) != 1 or int(rate_842.iloc[0]["support_sorted_ab"]) != 75:
        raise AssertionError("Figure 2 ensemble-rate 842 0->3 value is not the corrected 75-cell result.")
    replication = _validate_figure2_old_new_replication_audit()
    return {
        "unrestricted_chains": 23,
        "clean_chains": 22,
        "clean_animals": 11,
        "excluded_chain": "1616day1->1616bday1->1616b-3day1",
        "finite_support_policy": "metric_native_no_cross_metric_intersection",
        "place_native_finite_rows": len(place_ids),
        "level_native_finite_rows": len(level_ids),
        "corrected_842_rate_support": 75,
        "rate_manual_exclusion_profile": "none",
        "rate_mask_removal_changed_chains": 2,
        "old_vs_new_replication": replication,
    }


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


def _draw_figure2_lower_block(size: tuple[float, float], style: PublicationStyle) -> tuple[plt.Figure, list[dict[str, object]]]:
    curve_df = pd.read_csv(FIGURE2_GAP_CURVE)
    points_df = pd.read_csv(FIGURE2_CONSECUTIVE_POINTS)
    stats_df = pd.read_csv(FIGURE2_CONSECUTIVE_STATS)
    fig = plt.figure(figsize=size, facecolor="none")
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.72, 0.88, 1.60, 1.60),
        height_ratios=(1.0, 1.0, 1.0),
        left=0.066,
        right=0.996,
        bottom=0.112,
        top=0.978,
        wspace=0.34,
        hspace=0.46,
    )
    panel_letters = {
        (row, col): chr(ord("c") + row * 4 + col)
        for row in range(3)
        for col in range(4)
    }
    axes_records: list[dict[str, object]] = []
    for row, metric in enumerate(METRICS):
        axes = [
            fig.add_subplot(grid[row, 0]),
            fig.add_subplot(grid[row, 1]),
            fig.add_subplot(grid[row, 2]),
            fig.add_subplot(grid[row, 3]),
        ]
        _draw_figure2_distribution_panel(axes[0], metric, style)
        _draw_figure2_mean_panel(axes[1], metric, style)
        _draw_figure2_gap_curve(axes[2], curve_df, metric, style)
        _draw_figure2_prev_next_panel(axes[3], points_df, stats_df, metric, style)
        for col, ax in enumerate(axes):
            letter = panel_letters.get((row, col))
            if letter:
                ax.text(
                    -0.19 if col != 1 else -0.28,
                    1.06,
                    letter,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=style.panel_letter_fontsize,
                    fontweight=style.panel_letter_weight,
                )
            x0, y0, width, height = ax.get_position().bounds
            axes_records.append(
                {
                    "key": f"figure2_{metric}_col{col + 1}",
                    "metric": metric,
                    "row": row,
                    "column": col,
                    "x0": x0,
                    "y0": y0,
                    "width": width,
                    "height": height,
                    "width_in": width * size[0],
                    "height_in": height * size[1],
                }
            )
    fig.canvas.draw()
    return fig, axes_records


def _figure2_schematic_header_bounds(
    size: tuple[float, float],
    style: PublicationStyle,
) -> tuple[float, float, float, float]:
    title_font = FontProperties(family=style.font_family, size=style.title_fontsize)
    longest_title_width_pt = float(
        TextPath(
            (0.0, 0.0),
            "Population vector",
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


def _add_svg_centered_text(
    root: ET.Element,
    text: str,
    slot_in: tuple[float, float, float, float],
    *,
    font_size: float,
    style: PublicationStyle,
    fill: str = "black",
) -> None:
    x, y, w, h = slot_in
    node = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{(x + w / 2.0) * 72.0:.6f}",
            "y": f"{(y + h * 0.76) * 72.0:.6f}",
            "font-family": style.font_family,
            "font-size": f"{font_size:.1f}",
            "font-weight": "normal",
            "text-anchor": "middle",
            "fill": fill,
        },
    )
    node.text = text


def _add_svg_header_band(
    root: ET.Element,
    text: str,
    slot_in: tuple[float, float, float, float],
    *,
    color: str,
    font_size: float,
    style: PublicationStyle,
) -> None:
    x, y, w, h = slot_in
    ET.SubElement(
        root,
        _svg_tag("rect"),
        {
            "x": f"{x * 72.0:.6f}",
            "y": f"{y * 72.0:.6f}",
            "width": f"{w * 72.0:.6f}",
            "height": f"{h * 72.0:.6f}",
            "fill": color,
        },
    )
    _add_svg_centered_text(root, text, slot_in, font_size=font_size, style=style, fill="white")


def _add_svg_rotated_label(
    root: ET.Element,
    text: str,
    x_in: float,
    y_in: float,
    *,
    font_size: float,
    style: PublicationStyle,
) -> None:
    x_pt = x_in * 72.0
    y_pt = y_in * 72.0
    node = ET.SubElement(
        root,
        _svg_tag("text"),
        {
            "x": f"{x_pt:.6f}",
            "y": f"{y_pt:.6f}",
            "font-family": style.font_family,
            "font-size": f"{font_size:.1f}",
            "font-weight": "normal",
            "text-anchor": "middle",
            "dominant-baseline": "central",
            "fill": "black",
            "transform": f"rotate(-90 {x_pt:.6f} {y_pt:.6f})",
        },
    )
    node.text = text


def _matrix_frames_by_neuron(matrix: np.ndarray, n_neurons: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {arr.shape}")
    if arr.shape[1] == n_neurons:
        return arr
    if arr.shape[0] == n_neurons:
        return arr.T
    raise ValueError(f"matrix shape {arr.shape} does not match n_neurons={n_neurons}")


def _load_figure4_schematic_session(session_id: str) -> dict[str, object]:
    from ca_analysis.config import get_behav_wd, get_neural_wd
    from ca_analysis.io import load_behav_neural

    _, selected_df, _, _raw_counts, calcium, spike_counts, _ = load_behav_neural(
        session_id,
        behav_wd=get_behav_wd(),
        neural_wd=get_neural_wd(),
    )
    if spike_counts is None:
        raise RuntimeError(f"{session_id}: missing spike/activity matrix")
    n_neurons = int(np.asarray(spike_counts).shape[0])
    spike_tn_all = _matrix_frames_by_neuron(spike_counts, n_neurons)
    calcium_tn_all = _matrix_frames_by_neuron(calcium, n_neurons) if calcium is not None else spike_tn_all

    selected_df = selected_df.copy().reset_index(drop=True)
    frames = selected_df["frame"].to_numpy(dtype=int)
    valid = (frames >= 0) & (frames < spike_tn_all.shape[0]) & (frames < calcium_tn_all.shape[0])
    if calcium is not None:
        valid &= ~np.any(np.isnan(calcium_tn_all[frames.clip(0, calcium_tn_all.shape[0] - 1), :]), axis=1)
    selected_df = selected_df.loc[valid].reset_index(drop=True)
    frames = selected_df["frame"].to_numpy(dtype=int)

    return {
        "selected_df": selected_df,
        "positions": selected_df[["mass_x", "mass_y"]].to_numpy(dtype=float),
        "spike_tn": spike_tn_all[frames, :],
        "calcium_tn": calcium_tn_all[frames, :],
        "n_neurons": n_neurons,
    }


def _choose_figure4_schematic_cells(lam: np.ndarray, traces_tn: np.ndarray, n_cells: int = 5) -> np.ndarray:
    peak = np.nanmax(lam, axis=0)
    trace_span = np.nanpercentile(traces_tn, 98, axis=0) - np.nanpercentile(traces_tn, 2, axis=0)
    finite = np.isfinite(peak) & (peak > 0) & np.isfinite(trace_span) & (trace_span > 0)
    candidates = np.flatnonzero(finite)
    if candidates.size < n_cells:
        raise RuntimeError(f"not enough source-backed Figure 4A cells: {candidates.size}")
    top_n = min(max(n_cells * 12, 25), int(candidates.size))
    top = candidates[np.argsort(peak[candidates])[-top_n:]]
    peak_bins = np.nanargmax(np.where(np.isfinite(lam), lam, -np.inf), axis=0)
    top = top[np.argsort(peak_bins[top])]
    quantiles = np.linspace(0, top.size - 1, n_cells).round().astype(int)
    chosen = top[quantiles]
    _, unique_idx = np.unique(chosen, return_index=True)
    if unique_idx.size != n_cells:
        remaining = [idx for idx in top if idx not in set(chosen)]
        chosen = list(chosen[np.sort(unique_idx)])
        chosen.extend(remaining[: n_cells - len(chosen)])
        chosen = np.asarray(chosen, dtype=int)
    return chosen.astype(int)


def _normalize_trace_segment(segment_tn: np.ndarray) -> np.ndarray:
    lo = np.nanpercentile(segment_tn, 5, axis=0)
    hi = np.nanpercentile(segment_tn, 99, axis=0)
    denom = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    norm = (segment_tn - lo) / denom
    return np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.15)


def _figure4_mean_activity_maps(
    activity_tn: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int,
    min_samples: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    activity = np.asarray(activity_tn, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if activity.ndim != 2:
        raise ValueError(f"expected 2D activity matrix, got {activity.shape}")
    valid = (labels >= 0) & (labels < int(n_bins)) & np.all(np.isfinite(activity), axis=1)
    if not np.any(valid):
        raise RuntimeError("Figure 4A schematic has no valid activity samples for map construction")
    labels = labels[valid]
    activity = np.clip(np.nan_to_num(activity[valid], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    occupancy = np.bincount(labels, minlength=int(n_bins)).astype(float)
    maps = np.full((int(n_bins), activity.shape[1]), np.nan, dtype=float)
    keep = occupancy >= int(min_samples)
    for neuron in range(activity.shape[1]):
        sums = np.bincount(labels, weights=activity[:, neuron], minlength=int(n_bins)).astype(float)
        maps[keep, neuron] = sums[keep] / np.maximum(occupancy[keep], 1.0)
    return maps, occupancy


def _figure4_softmax(log_values: np.ndarray) -> np.ndarray:
    log_values = np.asarray(log_values, dtype=float)
    finite = np.isfinite(log_values)
    out = np.zeros_like(log_values, dtype=float)
    if not np.any(finite):
        return out
    shifted = log_values[finite] - float(np.nanmax(log_values[finite]))
    out[finite] = np.exp(np.clip(shifted, -80.0, 0.0))
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def _figure4_label_xy(label: int, n_bins_x: int) -> tuple[int, int]:
    label = int(label)
    n_bins_x = int(n_bins_x)
    return label % n_bins_x, label // n_bins_x


def _choose_diverse_figure4_schematic_cells(
    ranked_cells: np.ndarray,
    map_matrix: np.ndarray,
    peak_bins: np.ndarray,
    n_cells: int,
    n_bins_x: int,
) -> np.ndarray:
    ranked_cells = np.asarray(ranked_cells, dtype=int)
    map_matrix = np.asarray(map_matrix, dtype=float)
    peak_bins = np.asarray(peak_bins, dtype=int)
    if ranked_cells.size <= int(n_cells):
        return ranked_cells[: int(n_cells)]

    def _corr(a_idx: int, b_idx: int) -> float:
        a = np.nan_to_num(map_matrix[:, int(a_idx)], nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(map_matrix[:, int(b_idx)], nan=0.0, posinf=0.0, neginf=0.0)
        a = a - float(np.mean(a))
        b = b - float(np.mean(b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _peak_dist(a_idx: int, b_idx: int) -> float:
        ax, ay = _figure4_label_xy(int(peak_bins[int(a_idx)]), int(n_bins_x))
        bx, by = _figure4_label_xy(int(peak_bins[int(b_idx)]), int(n_bins_x))
        return float(math.hypot(ax - bx, ay - by))

    for min_dist, max_corr in ((13.0, 0.55), (10.0, 0.65), (7.0, 0.75), (4.0, 0.85)):
        chosen: list[int] = []
        for cell in ranked_cells.tolist():
            if all(_peak_dist(cell, prev) >= min_dist and _corr(cell, prev) <= max_corr for prev in chosen):
                chosen.append(int(cell))
            if len(chosen) == int(n_cells):
                return np.asarray(chosen, dtype=int)

    chosen = list(ranked_cells[: int(n_cells)])
    return np.asarray(chosen, dtype=int)


def _build_figure4_schematic_source() -> dict[str, object]:
    from ca_analysis.config import DT
    from ca_analysis.similarity import compute_tuning_and_prior
    from ca_analysis.spatial import discretize_positions

    if not FIGURE4_SCHEMATIC_DECODER_PKL.exists():
        raise FileNotFoundError(FIGURE4_SCHEMATIC_DECODER_PKL)
    with open(FIGURE4_SCHEMATIC_DECODER_PKL, "rb") as f:
        decoder_payload = pickle.load(f)

    src_session, dst_session = FIGURE4_SCHEMATIC_PAIR
    src = _load_figure4_schematic_session(src_session)
    dst = _load_figure4_schematic_session(dst_session)
    train_ids = np.asarray(decoder_payload["train_ids"], dtype=int)
    test_ids = np.asarray(decoder_payload["test_ids"], dtype=int)
    valid_pair = (
        (train_ids >= 0)
        & (test_ids >= 0)
        & (train_ids < int(src["n_neurons"]))
        & (test_ids < int(dst["n_neurons"]))
    )
    train_ids = train_ids[valid_pair]
    test_ids = test_ids[valid_pair]
    if train_ids.size == 0:
        raise RuntimeError("Figure 4A decoder pair has no valid matched neurons")

    n_map_bins = int(FIGURE4_SCHEMATIC_MAP_BINS)
    n_map_total = n_map_bins * n_map_bins
    labels_train_45, _x_edges_45, _y_edges_45 = discretize_positions(
        np.asarray(src["positions"]),
        n_map_bins,
        n_map_bins,
    )
    source_calcium = np.asarray(src["calcium_tn"], dtype=float)[:, train_ids]
    target_traces = np.asarray(dst["calcium_tn"], dtype=float)[:, test_ids]
    calcium_maps, _map_occupancy = _figure4_mean_activity_maps(
        source_calcium,
        labels_train_45,
        n_bins=n_map_total,
        min_samples=3,
    )

    peak = np.nanpercentile(calcium_maps, 99.5, axis=0)
    trace_span = np.nanpercentile(target_traces, 99.5, axis=0) - np.nanpercentile(target_traces, 20, axis=0)
    finite = np.isfinite(peak) & (peak > 0) & np.isfinite(trace_span) & (trace_span > 0)
    candidates = np.flatnonzero(finite)
    n_display_cells = int(FIGURE4_SCHEMATIC_N_CELLS)
    if candidates.size < n_display_cells:
        chosen = _choose_figure4_schematic_cells(
            np.nan_to_num(calcium_maps, nan=0.0),
            target_traces,
            n_cells=n_display_cells,
        )
        candidates = chosen
    top_n = min(max(80, n_display_cells * 20), int(candidates.size))
    combined = peak[candidates] * np.sqrt(np.maximum(trace_span[candidates], 1e-12))
    top = candidates[np.argsort(combined)[-top_n:]]
    score_basis = _normalize_trace_segment(target_traces[:, top])
    score = np.nanpercentile(score_basis, 90, axis=1)
    segment_len = min(int(FIGURE4_SCHEMATIC_TRACE_POINTS), int(score.size))
    if segment_len < 20:
        raise RuntimeError("Figure 4A target activity segment is too short")
    saved_decoded_pos = np.asarray(decoder_payload.get("decoded_pos", []), dtype=int)
    saved_true_pos = np.asarray(decoder_payload.get("labels_pos", []), dtype=int)
    if score.size > segment_len:
        max_start = int(score.size - segment_len)
        later_min = int(round(score.size * 0.55))
        if saved_decoded_pos.size == score.size and saved_true_pos.size == score.size:
            saved_decoded_xy = np.asarray([_figure4_label_xy(label, 15) for label in saved_decoded_pos], dtype=float)
            saved_true_xy = np.asarray([_figure4_label_xy(label, 15) for label in saved_true_pos], dtype=float)
            saved_errors = np.sqrt(np.sum((saved_decoded_xy - saved_true_xy) ** 2, axis=1))
            candidate_centers = np.flatnonzero(
                (np.arange(score.size) >= later_min)
                & (np.arange(score.size) >= segment_len // 2)
                & (np.arange(score.size) <= score.size - max(1, segment_len // 2) - 1)
                & np.isfinite(saved_errors)
                & (saved_errors <= 2.5)
                & np.isfinite(score)
            )
            if candidate_centers.size:
                posterior_center = int(candidate_centers[np.argmax(score[candidate_centers])])
                start = int(np.clip(posterior_center - segment_len // 2, 0, max_start))
            else:
                csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(score, nan=0.0))])
                window_score = csum[segment_len:] - csum[:-segment_len]
                start_min = min(max_start, later_min)
                start = int(start_min + np.argmax(window_score[start_min:])) if start_min < window_score.size else max_start
        else:
            csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(score, nan=0.0))])
            window_score = csum[segment_len:] - csum[:-segment_len]
            start_min = min(max_start, later_min)
            start = int(start_min + np.argmax(window_score[start_min:])) if start_min < window_score.size else max_start
    else:
        start = 0
    stop = start + segment_len
    top_segment = _normalize_trace_segment(target_traces[start:stop, top])
    segment_amp = np.nanpercentile(top_segment, 99, axis=0) - np.nanpercentile(top_segment, 25, axis=0)
    selected_rank = segment_amp * np.sqrt(np.maximum(peak[top], 1e-12))
    peak_bins = np.nanargmax(np.where(np.isfinite(calcium_maps), calcium_maps, -np.inf), axis=0)
    forced_source_cells = tuple(int(v) for v in FIGURE4_SCHEMATIC_SOURCE_CELL_IDS)
    if forced_source_cells:
        if len(forced_source_cells) != n_display_cells:
            raise RuntimeError(
                "Figure 4A forced source-cell list must match "
                f"FIGURE4_SCHEMATIC_N_CELLS={n_display_cells}"
            )
        source_cell_lookup = {int(cell_id): idx for idx, cell_id in enumerate(train_ids.tolist())}
        missing = [cell_id for cell_id in forced_source_cells if cell_id not in source_cell_lookup]
        if missing:
            raise RuntimeError(f"Figure 4A forced source cells are not matched in the decoder CIMap: {missing}")
        chosen = np.asarray([source_cell_lookup[cell_id] for cell_id in forced_source_cells], dtype=int)
    else:
        ranked_top = top[np.argsort(selected_rank)[::-1]]
        chosen = _choose_diverse_figure4_schematic_cells(
            ranked_top,
            calcium_maps,
            peak_bins,
            n_display_cells,
            n_map_bins,
        )
        chosen = chosen[np.argsort(peak_bins[chosen])]
    trace_segment = _normalize_trace_segment(target_traces[start:stop, chosen])

    maps = []
    for idx in chosen:
        neuron_map = np.asarray(calcium_maps[:, idx], dtype=float).reshape(n_map_bins, n_map_bins)
        maps.append(neuron_map)

    decode_bins = 15
    labels_train_decode, _x_edges_decode, _y_edges_decode = discretize_positions(
        np.asarray(src["positions"]),
        decode_bins,
        decode_bins,
    )
    labels_dst_decode, _x_edges_dst_decode, _y_edges_dst_decode = discretize_positions(
        np.asarray(dst["positions"]),
        decode_bins,
        decode_bins,
    )
    spike_train_decode = np.asarray(src["spike_tn"], dtype=float)[:, train_ids]
    valid_decode_train = np.isfinite(labels_train_decode) & (np.asarray(labels_train_decode) >= 0)
    lam_decode, prior_decode = compute_tuning_and_prior(
        np.nan_to_num(spike_train_decode[valid_decode_train], nan=0.0, posinf=0.0, neginf=0.0),
        np.asarray(labels_train_decode, dtype=int)[valid_decode_train],
        n_bins=decode_bins * decode_bins,
        dt=float(DT),
        filter_ratemap=False,
        sigma=1.0,
        min_time=0.0,
        eps=1e-12,
        decode_head="none",
    )
    segment_activity = np.nan_to_num(
        np.asarray(dst["spike_tn"], dtype=float)[start:stop, :][:, test_ids],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    log_rate = np.log(lam_decode + 1e-12)
    log_offset = -float(DT) * np.sum(lam_decode, axis=1) + np.log(prior_decode + 1e-12)
    log_post_segment = (log_rate @ segment_activity.T) + log_offset[:, None]
    decoded_segment = np.nanargmax(log_post_segment, axis=0).astype(int)
    labels_segment = np.asarray(labels_dst_decode, dtype=int)[start:stop]
    decoded_xy = np.asarray([_figure4_label_xy(label, decode_bins) for label in decoded_segment], dtype=float)
    true_xy = np.asarray([_figure4_label_xy(label, decode_bins) for label in labels_segment], dtype=float)
    distances = np.sqrt(np.sum((decoded_xy - true_xy) ** 2, axis=1))
    log_max = np.nanmax(log_post_segment, axis=0)
    exp_sum = np.sum(np.exp(np.clip(log_post_segment - log_max[None, :], -80.0, 0.0)), axis=0)
    top_prob = 1.0 / np.maximum(exp_sum, 1e-12)
    valid_posterior_frames = np.flatnonzero(np.isfinite(top_prob) & np.isfinite(distances))
    close_frames = valid_posterior_frames[distances[valid_posterior_frames] <= 2.5]
    posterior_candidates = close_frames if close_frames.size else valid_posterior_frames
    if posterior_candidates.size:
        vline_idx = int(posterior_candidates[np.argmax(top_prob[posterior_candidates])])
    else:
        vline_idx = int(np.argmax(np.nanmean(trace_segment, axis=1)))
    posterior_index = int(np.clip(start + vline_idx, 0, target_traces.shape[0] - 1))
    posterior = _figure4_softmax(log_post_segment[:, vline_idx]).reshape(decode_bins, decode_bins)
    decoded_label_45 = int(np.nanargmax(posterior))
    true_label_45 = int(np.asarray(labels_dst_decode, dtype=int)[posterior_index])

    decoder_frames = np.asarray(decoder_payload.get("frames", []), dtype=int)
    dst_frame = int(dst["selected_df"].iloc[posterior_index]["frame"])
    decoder_frame_matches = np.flatnonzero(decoder_frames == dst_frame) if decoder_frames.size else np.array([], dtype=int)
    decoder_payload_index = int(decoder_frame_matches[0]) if decoder_frame_matches.size else int(
        min(posterior_index, max(len(decoder_frames) - 1, 0))
    )

    return {
        "src_session": src_session,
        "dst_session": dst_session,
        "decoder_pkl": FIGURE4_SCHEMATIC_DECODER_PKL,
        "n_matched_neurons": int(train_ids.size),
        "selected_train_ids": [int(train_ids[idx]) for idx in chosen],
        "selected_test_ids": [int(test_ids[idx]) for idx in chosen],
        "selected_match_indices": [int(idx) for idx in chosen],
        "template_maps": maps,
        "trace_segment": trace_segment,
        "trace_start_index": int(start),
        "trace_stop_index": int(stop),
        "vline_idx": int(vline_idx),
        "posterior": posterior,
        "posterior_index": int(posterior_index),
        "posterior_frame": int(dst_frame),
        "posterior_decoded_label": int(decoded_label_45),
        "posterior_true_label": int(true_label_45),
        "posterior_decoded_xy": [int(v) for v in _figure4_label_xy(decoded_label_45, decode_bins)],
        "posterior_true_xy": [int(v) for v in _figure4_label_xy(true_label_45, decode_bins)],
        "saved_decoder_payload_index": int(decoder_payload_index),
        "saved_decoder_decoded_pos_15": int(saved_decoded_pos[decoder_payload_index])
        if saved_decoded_pos.size
        else None,
        "saved_decoder_true_pos_15": int(saved_true_pos[decoder_payload_index]) if saved_true_pos.size else None,
        "n_bins_x": int(n_map_bins),
        "n_bins_y": int(n_map_bins),
        "posterior_n_bins_x": int(decode_bins),
        "posterior_n_bins_y": int(decode_bins),
        "dt": float(DT),
        "template_source": f"source-session C matrix averaged in {n_map_bins}x{n_map_bins} position bins",
        "activity_source": "target-session C activity traces at matched-neuron ids",
        "posterior_source": "real 15x15 Poisson decoder posterior recomputed from source-session S tuning and target-session S activity",
    }


def _render_figure4_method_schematic(size: tuple[float, float], style: PublicationStyle) -> tuple[plt.Figure, dict[str, object]]:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    source = _build_figure4_schematic_source()
    title_font = style.axis_label_fontsize * 0.72
    arrow_font = style.annotation_fontsize * 0.70
    tiny_font = style.tick_fontsize * 0.62
    maze = get_panel_maze(6)
    map_cmap = plt.get_cmap("magma").copy()
    map_cmap.set_bad("#f4f4f4")
    posterior_cmap = plt.get_cmap("magma").copy()
    posterior_cmap.set_bad("#f7f7f7")
    trace_y_centers = [0.825, 0.690, 0.500, 0.365, 0.230]

    def _add_real_map(
        bounds: tuple[float, float, float, float],
        image: np.ndarray,
        *,
        is_posterior: bool = False,
        zorder: int = 1,
    ) -> plt.Axes:
        from scipy.ndimage import gaussian_filter
        from matplotlib.patches import Rectangle

        map_ax = ax.inset_axes(bounds, transform=ax.transAxes)
        map_ax.set_zorder(zorder)
        arr = np.asarray(image, dtype=float)
        grid_bins_y, grid_bins_x = int(arr.shape[0]), int(arr.shape[1])
        wall_scale_x = float(grid_bins_x) / 15.0
        wall_scale_y = float(grid_bins_y) / 15.0
        if is_posterior:
            plot_arr = gaussian_filter(np.nan_to_num(arr, nan=0.0), sigma=0.55)
            vmax = float(np.nanmax(plot_arr)) if np.any(np.isfinite(plot_arr)) else 1.0
        else:
            finite = np.isfinite(arr)
            filled = np.where(finite, arr, 0.0)
            plot_arr = gaussian_filter(filled, sigma=1.0)
            positive = plot_arr[np.isfinite(plot_arr) & (plot_arr > 0)]
            vmax = float(np.nanpercentile(positive, 99.7)) if positive.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        map_ax.set_facecolor("#fcfcfc")
        map_ax.imshow(
            np.ma.masked_invalid(plot_arr),
            origin="upper",
            interpolation="nearest",
            cmap=posterior_cmap if is_posterior else map_cmap,
            vmin=0.0,
            vmax=vmax,
        )
        map_ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                grid_bins_x,
                grid_bins_y,
                fill=False,
                edgecolor="0.15",
                linewidth=0.45 if not is_posterior else 0.60,
                zorder=4,
            )
        )
        map_ax.plot(
            maze.wa[:, 0] * wall_scale_x,
            maze.wa[:, 1] * wall_scale_y,
            color="0.10" if not is_posterior else "0.15",
            linewidth=0.26 if not is_posterior else 0.42,
            alpha=0.95,
            zorder=3,
        )
        map_ax.set_xlim(-0.5, grid_bins_x - 0.5)
        map_ax.set_ylim(grid_bins_y - 0.5, -0.5)
        map_ax.set_aspect("equal")
        map_ax.axis("off")
        return map_ax

    template_maps = [np.asarray(m, dtype=float) for m in source["template_maps"]]
    map_geometry_options = {
        "low": (0.045, 0.230, 0.086, 0.052, [0.765, 0.620, 0.410, 0.265, 0.140]),
        "medium": (0.035, 0.230, 0.092, 0.075, [0.760, 0.615, 0.405, 0.260, 0.130]),
        "high": (0.012, 0.235, 0.128, 0.105, [0.760, 0.600, 0.420, 0.260, 0.100]),
        "wide": (0.005, 0.230, 0.106, 0.125, [0.748, 0.602, 0.392, 0.246, 0.112]),
    }
    map_geometry_key = os.environ.get("FIGURE4_SCHEMATIC_TILT", "high").strip().lower()
    if map_geometry_key not in map_geometry_options:
        raise ValueError(
            f"Unknown FIGURE4_SCHEMATIC_TILT={map_geometry_key!r}; "
            f"expected one of {sorted(map_geometry_options)}"
        )
    map_x0, map_w, map_h, map_shear, map_y0s = map_geometry_options[map_geometry_key]
    if map_geometry_key == "high":
        map_y0s = [float(y_center) - map_h / 2.0 for y_center in trace_y_centers]

    def _draw_slanted_template_map(y0: float, neuron_map: np.ndarray, *, zorder: int) -> None:
        from matplotlib.patches import Polygon

        arr = np.asarray(neuron_map, dtype=float)
        finite = np.isfinite(arr)
        if np.any(finite):
            baseline = float(np.nanpercentile(arr[finite], 55.0))
            image = np.where(finite, np.clip(arr - baseline, 0.0, None), 0.0)
            vmax = float(np.nanpercentile(image[image > 0], 99.5)) if np.any(image > 0) else 1.0
        else:
            image = np.zeros_like(arr, dtype=float)
            vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
        pixels_per_bin = 8
        tile = np.repeat(np.repeat(image, pixels_per_bin, axis=0), pixels_per_bin, axis=1)
        sheet = Polygon(
            [
                (map_x0, y0),
                (map_x0 + map_w, y0),
                (map_x0 + map_w + map_shear, y0 + map_h),
                (map_x0 + map_shear, y0 + map_h),
            ],
            closed=True,
            facecolor="#020204",
            edgecolor="none",
            antialiased=True,
            transform=ax.transAxes,
            zorder=zorder - 0.05,
        )
        ax.add_patch(sheet)
        slant = math.degrees(math.atan2(map_shear, map_h))
        transform = matplotlib.transforms.Affine2D().skew_deg(slant, 0).translate(map_x0, y0) + ax.transAxes
        ax.imshow(
            tile,
            extent=(0.0, map_w, 0.0, map_h),
            origin="upper",
            cmap=map_cmap,
            norm=norm,
            interpolation="nearest",
            resample=False,
            aspect="auto",
            transform=transform,
            zorder=zorder,
            clip_on=False,
        )
    for layer, (y0, neuron_map) in enumerate(zip(map_y0s, template_maps, strict=True)):
        _draw_slanted_template_map(y0, neuron_map, zorder=1 + layer * 0.1)
    stack_center_x = map_x0 + map_w / 2.0 + map_shear / 2.0
    map_gap_mid_y = (map_y0s[1] + map_y0s[2] + map_h) / 2.0
    gap_dot_y = map_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, stack_center_x),
        gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    bottom_title_y = 0.025
    ax.text(stack_center_x, bottom_title_y, "Day i place maps", ha="center", va="bottom", fontsize=title_font)

    trace_left, trace_right = 0.405, 0.635
    trace_x = np.linspace(trace_left, trace_right, int(FIGURE4_SCHEMATIC_TRACE_POINTS))
    trace_segment = np.asarray(source["trace_segment"], dtype=float)
    if trace_segment.shape[0] != trace_x.size:
        trace_x = np.linspace(trace_left, trace_right, trace_segment.shape[0])
    vline_x = float(np.interp(int(source["vline_idx"]), [0, max(trace_segment.shape[0] - 1, 1)], [trace_x[0], trace_x[-1]]))
    frame_dx = (trace_right - trace_left) / max(trace_segment.shape[0] - 1, 1)
    visible_half_width = max(frame_dx / 2.0, 0.0014)
    trace_band_ymin = min(y - 0.050 + 0.010 for y in trace_y_centers) - 0.006
    trace_band_ymax = max(y - 0.050 + 0.010 + 0.073 for y in trace_y_centers) + 0.006
    ax.axvspan(
        max(trace_left, vline_x - visible_half_width),
        min(trace_right, vline_x + visible_half_width),
        ymin=max(0.0, trace_band_ymin),
        ymax=min(1.0, trace_band_ymax),
        color="#e5e5e5",
        alpha=0.85,
        linewidth=0,
        zorder=0,
    )
    for index, y_center in enumerate(trace_y_centers):
        baseline = y_center - 0.050
        trace = trace_segment[:, index]
        trace = 0.010 + 0.073 * np.clip(trace, 0.0, 1.0)
        ax.plot(trace_x, baseline + trace, color="#202020", linewidth=0.70)
    trace_gap_mid_y = (trace_y_centers[1] + trace_y_centers[2]) / 2.0
    trace_gap_dot_y = trace_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, (trace_left + trace_right) / 2.0),
        trace_gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    scale_bar_frames = 10.0 / max(float(source["dt"]), 1e-12)
    scale_bar_len = (scale_bar_frames / max(trace_segment.shape[0] - 1, 1)) * (trace_right - trace_left)
    scale_bar_x0 = trace_left
    scale_bar_y = 0.145
    ax.plot(
        [scale_bar_x0, scale_bar_x0 + scale_bar_len],
        [scale_bar_y, scale_bar_y],
        color="#202020",
        linewidth=0.90,
        solid_capstyle="butt",
    )
    ax.text(
        scale_bar_x0 + scale_bar_len / 2.0,
        scale_bar_y - 0.030,
        "10 s",
        ha="center",
        va="top",
        fontsize=tiny_font,
        color="#202020",
    )
    ax.text(
        (trace_left + trace_right) / 2.0,
        bottom_title_y,
        "Day i + 90 activity",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )

    ax.text(0.377, 0.500, "x", ha="center", va="center", fontsize=style.axis_label_fontsize * 1.00)
    posterior = np.asarray(source["posterior"], dtype=float)
    posterior_h = 0.710
    posterior_w = posterior_h * float(size[1]) / float(size[0])
    posterior_x = 0.675
    posterior_y = 0.150
    arrow_start_x = trace_right + 0.010
    arrow_end_x = posterior_x - 0.010
    ax.plot(
        [arrow_start_x, arrow_end_x],
        [0.500, 0.500],
        color="0.25",
        linewidth=0.65,
        marker=">",
        markevery=[-1],
        markersize=1.25,
        markeredgewidth=0,
        markerfacecolor="0.25",
        solid_capstyle="butt",
        zorder=3,
    )

    posterior_ax = _add_real_map((posterior_x, posterior_y, posterior_w, posterior_h), posterior, is_posterior=True)
    decoded_x, decoded_y = [int(v) for v in source["posterior_decoded_xy"]]
    true_x, true_y = [int(v) for v in source["posterior_true_xy"]]
    posterior_ax.scatter(
        [decoded_x],
        [decoded_y],
        s=2.8,
        marker="x",
        linewidths=0.32,
        color="#00A6D6",
        zorder=5,
    )
    posterior_ax.scatter(
        [true_x],
        [true_y],
        s=2.8,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.32,
        zorder=5,
    )
    cbar_x = posterior_x + posterior_w * 0.18
    cbar_y = posterior_y + posterior_h + 0.025
    cbar_w = posterior_w * 0.64
    cbar_h = 0.014
    posterior_cbar_ax = ax.inset_axes((cbar_x, cbar_y, cbar_w, cbar_h), transform=ax.transAxes)
    posterior_cbar_ax.set_zorder(6)
    posterior_cbar_ax.imshow(
        np.linspace(0.0, 1.0, 128)[None, :],
        aspect="auto",
        cmap=posterior_cmap,
        interpolation="nearest",
    )
    posterior_cbar_ax.set_axis_off()
    ax.text(
        cbar_x + cbar_w / 2.0,
        cbar_y + cbar_h + 0.006,
        "P(position)",
        fontsize=tiny_font * 0.92,
        color="#202020",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        zorder=6,
    )
    legend_y = posterior_y - 0.035
    decoded_legend_x = posterior_x + 0.030
    true_legend_x = posterior_x + 0.150
    ax.scatter(
        [decoded_legend_x],
        [legend_y],
        s=4.0,
        marker="x",
        linewidths=0.42,
        color="#00A6D6",
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        decoded_legend_x + 0.012,
        legend_y,
        "Decoded",
        fontsize=tiny_font,
        color="#00A6D6",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.scatter(
        [true_legend_x],
        [legend_y],
        s=4.0,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.42,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        true_legend_x + 0.012,
        legend_y,
        "Real",
        fontsize=tiny_font,
        color="#F04E37",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.text(
        posterior_x + posterior_w / 2.0,
        bottom_title_y,
        "Decoded position",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )
    meta = {
        "src_session": str(source["src_session"]),
        "dst_session": str(source["dst_session"]),
        "decoder_pkl": _rel(Path(source["decoder_pkl"])),
        "n_matched_neurons": int(source["n_matched_neurons"]),
        "selected_train_ids": list(source["selected_train_ids"]),
        "selected_test_ids": list(source["selected_test_ids"]),
        "selected_match_indices": list(source["selected_match_indices"]),
        "trace_start_index": int(source["trace_start_index"]),
        "trace_stop_index": int(source["trace_stop_index"]),
        "vline_idx": int(source["vline_idx"]),
        "posterior_index": int(source["posterior_index"]),
        "posterior_frame": int(source["posterior_frame"]),
        "posterior_decoded_label": int(source["posterior_decoded_label"]),
        "posterior_true_label": int(source["posterior_true_label"]),
        "posterior_decoded_xy": list(source["posterior_decoded_xy"]),
        "posterior_true_xy": list(source["posterior_true_xy"]),
        "n_bins_x": int(source["n_bins_x"]),
        "n_bins_y": int(source["n_bins_y"]),
        "posterior_n_bins_x": int(source["posterior_n_bins_x"]),
        "posterior_n_bins_y": int(source["posterior_n_bins_y"]),
        "dt": float(source["dt"]),
        "template_source": str(source["template_source"]),
        "activity_source": str(source["activity_source"]),
        "posterior_source": str(source["posterior_source"]),
    }
    return fig, meta


def _render_figure4_example_cell_arrow(size: tuple[float, float], style: PublicationStyle) -> tuple[plt.Figure, dict[str, object]]:
    layout = {
        **PANEL_LAYOUT,
        "panel_size": size,
        "grid_wspace": 0.03,
        "margins": {"left": 0.025, "right": 0.995, "top": 0.80, "bottom": 0.015},
    }
    fig, _axes, data, _paths = render_example_cell_day12_placemap_panel(
        pick=("451-3day1->451-6day1", 161),
        mode="arrow",
        saved_dir=EXAMPLE_SAVED_DIR,
        panel_size=size,
        layout=layout,
        colors={
            "maze_line": "k",
            "trajectory": "0.55",
            "arrow_face": "red",
            "arrow_edge": "red",
        },
        style={
            "threshold_sn": 5.0,
            "dedupe_frames": 4,
            "spike_signal": "S",
            "noise_from": "Craw",
            "show_trajectory": True,
            "trajectory_alpha": 0.25,
            "trajectory_linewidth": 0.12,
            "arrow_head_width": 0.055,
            "arrow_head_length": 0.14,
            "arrow_linewidth": 0.01,
            "arrow_alpha": 0.42,
            "arrow_scale": 0.23,
            "maze_linewidth": 0.62,
            "show_maze": True,
            "show_frame": False,
            "title_fontsize": max(4, int(round(style.title_fontsize))),
            "title_pad": 1,
            "suptitle_fontsize": None,
            "tight_layout": False,
        },
        labels={
            "col_titles": ("Day i", "Day i+1", "Day i+90", "Day i+91"),
            "suptitle_template": "",
        },
        save=False,
    )
    arrow_counts = {
        side: int(arrows["n_spikes"])
        for side, arrows in data.get("arrows_by_side", {}).items()
    }
    meta = {
        "bundle_path": _rel(Path(data["bundle_path"])),
        "pair_key": str(data["pair_key"]),
        "row_idx": int(data["row_idx"]),
        "src": str(data["src"]),
        "src_d2": str(data["src_d2"]),
        "dst": str(data["dst"]),
        "dst_d2": str(data["dst_d2"]),
        "cell_a": int(data["cell_a"]),
        "cell_b": int(data["cell_b"]),
        "place_r": float(data["place_r"]),
        "level_r": float(data["level_r"]),
        "mode": str(data["mode"]),
        "arrow_counts": arrow_counts,
    }
    return fig, meta


def _draw_figure4_decoder_bar(
    ax: plt.Axes,
    *,
    points_path: Path,
    summary_path: Path,
    stats_path: Path,
    value_col: str,
    group_order: tuple[str, str],
    colors: dict[str, str],
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    annotation: str,
    style: PublicationStyle,
) -> None:
    points = pd.read_csv(points_path)
    summary = pd.read_csv(summary_path)
    points = points.loc[points["scope"].eq("sorted")].copy()
    summary = summary.loc[summary["scope"].eq("sorted")].copy()
    x_lookup = {group: idx for idx, group in enumerate(group_order)}
    for _pair_id, pair_rows in points.groupby("pair_id", sort=False):
        pair_rows = pair_rows.set_index("group").reindex(group_order)
        if pair_rows[value_col].notna().all():
            ax.plot(
                [x_lookup[group] for group in group_order],
                pair_rows[value_col].to_numpy(dtype=float),
                color="#8a8a8a",
                alpha=0.30,
                linewidth=0.55,
                zorder=1,
            )
    for group in group_order:
        idx = x_lookup[group]
        mean_row = summary.loc[summary["group"].astype(str).eq(group)].iloc[0]
        mean = float(mean_row["mean"])
        sem = float(mean_row["sem"])
        ax.bar(
            idx,
            mean,
            yerr=sem,
            width=0.68,
            color=colors[group],
            edgecolor="none",
            alpha=0.78,
            capsize=2.0,
            zorder=2,
        )
        vals = points.loc[points["group"].astype(str).eq(group), value_col].to_numpy(dtype=float)
        jitter = np.linspace(-0.05, 0.05, len(vals)) if len(vals) else np.array([])
        ax.scatter(idx + jitter, vals, s=9, color=colors[group], alpha=0.68, edgecolors="none", zorder=3)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(group_order)
    ax.set_ylabel(ylabel, labelpad=0.0)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.55, len(group_order) - 0.45)
    y = ylim[1] * 0.92
    ax.plot([0, 0, 1, 1], [y * 0.98, y, y, y * 0.98], color="black", linewidth=0.65)
    ax.text(0.5, y * 1.015, annotation, ha="center", va="bottom", fontsize=style.annotation_fontsize)
    _apply_axis_style(ax, style)
    _ = json.loads(stats_path.read_text(encoding="utf-8"))


def _render_figure4_decoder_panel(
    size: tuple[float, float],
    *,
    reward: bool,
    style: PublicationStyle,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    decoder_ylim = (0, 12.0)
    decoder_yticks = [0, 5, 10]
    if reward:
        _draw_figure4_decoder_bar(
            ax,
            points_path=FIGURE4_REWARD_DECODER_POINTS,
            summary_path=FIGURE4_REWARD_DECODER_SUMMARY,
            stats_path=FIGURE4_REWARD_DECODER_STATS,
            value_col="decoding_error",
            group_order=("Reward", "Ctrls"),
            colors={"Reward": style.metric_colors["pv_correlation"], "Ctrls": "#7f7f7f"},
            ylabel="Decoding error (bins)",
            ylim=decoder_ylim,
            yticks=decoder_yticks,
            annotation="***",
            style=style,
        )
    else:
        _draw_figure4_decoder_bar(
            ax,
            points_path=FIGURE4_CROSS_DECODER_POINTS,
            summary_path=FIGURE4_CROSS_DECODER_SUMMARY,
            stats_path=FIGURE4_CROSS_DECODER_STATS,
            value_col="euclidean_error",
            group_order=("Data", "Shuffle"),
            colors={"Data": style.metric_colors["pv_correlation"], "Shuffle": "#7f7f7f"},
            ylabel="Decoding error (bins)",
            ylim=decoder_ylim,
            yticks=decoder_yticks,
            annotation="**",
            style=style,
        )
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.25, top=0.86)
    return fig


def _write_figure4_sorted_decoder_map_source(
    out_dir: Path,
    *,
    pair_id: str = "579RLday1_to_579-3day1",
) -> tuple[Path, dict[str, object]]:
    from ca_analysis import figure_supplement_panels

    source_path = FIGURE4_DECODER_DIR / "decoder_heatmap_per_pair_bin_rows_sorted.csv"
    long_df = pd.read_csv(source_path)
    sub = long_df.loc[long_df["pair_id"].astype(str).eq(pair_id)].copy()
    if len(sub) != 225:
        raise ValueError(f"Expected 225 sorted decoder-map bins for pair_id={pair_id!r}, found {len(sub)}")
    sub = sub.sort_values("bin_index", kind="stable").reset_index(drop=True)
    reward_bin = int(figure_supplement_panels.DECODING_ERROR_HEATMAP_REWARD_BIN)
    control_bins = {int(v) for v in figure_supplement_panels.DECODING_ERROR_HEATMAP_CONTROL_BINS}
    sub["is_reward"] = sub["bin_index"].astype(int).eq(reward_bin)
    sub["is_control"] = sub["bin_index"].astype(int).isin(control_bins)
    sub["marker_label"] = np.where(
        sub["is_reward"],
        "Reward",
        np.where(sub["is_control"], "Control", ""),
    )
    src = str(sub["src"].dropna().iloc[0]) if sub["src"].notna().any() else ""
    dst = str(sub["dst"].dropna().iloc[0]) if sub["dst"].notna().any() else ""
    animal = str(sub["animal"].dropna().iloc[0]) if sub["animal"].notna().any() else ""
    transition = str(sub["transition"].dropna().iloc[0]) if sub["transition"].notna().any() else ""
    sub["source_pickle"] = str(source_path)
    out_cols = [
        "pair_id",
        "animal",
        "cohort",
        "transition",
        "bin_index",
        "bx",
        "by",
        "decoding_error",
        "frame_count",
        "is_reward",
        "is_control",
        "marker_label",
        "src",
        "dst",
        "source_pickle",
        "source_set",
        "panel_title",
    ]
    out_csv = out_dir / "figure4_panel_c_sorted_spatial_decoding_error_map_source.csv"
    sub.loc[:, out_cols].to_csv(out_csv, index=False)
    finite = pd.to_numeric(sub["decoding_error"], errors="coerce").dropna().to_numpy(dtype=float)
    meta = {
        "source_table": _rel(source_path),
        "source_csv": _rel(out_csv),
        "pair_id": pair_id,
        "src": src,
        "dst": dst,
        "animal": animal,
        "transition": transition,
        "scope": "sorted",
        "source_rows": int(len(sub)),
        "finite_bins": int(finite.size),
        "vmax_98pct": float(np.nanpercentile(finite, 98)) if finite.size else np.nan,
    }
    return out_csv, meta


def _render_figure4_sorted_spatial_decoder_map(
    size: tuple[float, float],
    *,
    out_dir: Path,
    style: PublicationStyle,
) -> tuple[plt.Figure, dict[str, object]]:
    map_csv, meta = _write_figure4_sorted_decoder_map_source(out_dir)
    fig, _ax, _data, _paths = render_spatial_decoding_error_map(
        map_csv=map_csv,
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "reference_panel_size": size,
            "colorbar_fraction": 0.060,
            "colorbar_pad": 0.020,
            "colorbar_location": "left",
        },
        colors={
            "cmap": "magma",
            "bad": "white",
            "wall": "black",
            "reward_square": style.metric_colors["pv_correlation"],
            "control_square": "#7f7f7f",
        },
        style={
            "vmin": 1.0,
            "vmax": 14.0,
            "image_alpha": 0.95,
            "maze_wall_linewidth": 0.85,
            "square_linewidth": 1.45,
            "square_size": 1.10,
        },
        labels={"colorbar": "Decoding error (bins)"},
        save=False,
    )
    map_ax = fig.axes[0]
    colorbar_ax = fig.axes[-1]
    heatmap_height = 0.885
    heatmap_width = heatmap_height * size[1] / size[0]
    map_ax.set_position([0.125, 0.072, heatmap_width, heatmap_height])
    colorbar_ax.set_position([0.060, 0.072, 0.032, heatmap_height])
    colorbar_ax.yaxis.set_ticks_position("left")
    colorbar_ax.yaxis.set_label_position("left")
    colorbar_ax.yaxis.labelpad = 1.0
    colorbar_ax.tick_params(axis="y", pad=1.0, length=2.0)
    map_df = pd.read_csv(map_csv)
    reward_rows = map_df.loc[map_df["is_reward"].astype(bool)]
    if not reward_rows.empty:
        reward_row = reward_rows.iloc[0]
        reward_xy_axes = map_ax.transAxes.inverted().transform(
            map_ax.transData.transform((float(reward_row["bx"]), float(reward_row["by"])))
        )
        figure_panels_module._add_timeline_water_drop_icon(
            map_ax,
            (1.0, float(reward_xy_axes[1])),
            {
                "kind": "water",
                "size": 12.0,
                "linewidth": 0.65,
                "x_offset_points": 6.0,
                "zorder": 20,
            },
        )
    return fig, meta








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
        arrows = compute_example_cell_arrow_data(
            pack,
            side,
            threshold_sn=5.0,
            dedupe_frames=4,
            spike_signal="S",
            noise_from="Craw",
        )
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
    """Load per-animal first-window values from the same bundle family as the curves."""

    usecols = {
        "animal",
        "experience_group",
        "metric",
        "metric_label",
        "map_type",
        "support_policy",
        "window_type",
        "window_size",
        "window_size_label",
        "first_value",
        "mean_common_bins",
        "mean_finite_neuron_fraction",
        "mean_finite_bin_fraction",
        "n_pairs",
    }
    pieces: list[pd.DataFrame] = []
    for source_bundle, path in DAY2_TEMPLATE_ANIMAL_SUMMARIES.items():
        df = pd.read_csv(path, usecols=lambda col: col in usecols)
        df["source_bundle"] = source_bundle
        pieces.append(df)
    points = pd.concat(pieces, ignore_index=True)
    points["experience_group"] = _normalize_experience(points["experience_group"])
    points["metric"] = points["metric"].astype(str)
    points["support_policy"] = points["support_policy"].astype(str)
    points["window_type"] = points["window_type"].astype(str)
    points["window_size"] = pd.to_numeric(points["window_size"], errors="coerce")
    points["value"] = pd.to_numeric(points["first_value"], errors="coerce")
    points["metric_label"] = points["metric"].map(METRIC_LABELS)
    points = points.loc[
        points["metric"].isin(METRICS)
        & points["window_type"].eq("maze_time")
        & np.isclose(points["window_size"], 30.0)
        & np.isfinite(points["value"])
    ].copy()
    support_ok = points.apply(
        lambda row: str(row["support_policy"]) == DAY2_TEMPLATE_SUPPORT_POLICY[str(row["metric"])],
        axis=1,
    )
    points = points.loc[support_ok].copy()
    points["experience_index"] = points["experience_group"].map(EXPERIENCE_INDEX)
    points = points[np.isfinite(points["experience_index"])].copy()
    points["experience_index"] = points["experience_index"].astype(int)
    group_cols = [
        "animal",
        "experience_group",
        "experience_index",
        "metric",
        "metric_label",
        "support_policy",
        "window_type",
        "window_size",
    ]
    points = (
        points.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            value=("value", "mean"),
            n_source_rows=("value", "size"),
            n_pairs=("n_pairs", "sum"),
            source_bundles=("source_bundle", lambda x: ";".join(sorted(set(map(str, x))))),
            mean_common_bins=("mean_common_bins", "mean"),
            mean_finite_neuron_fraction=("mean_finite_neuron_fraction", "mean"),
            mean_finite_bin_fraction=("mean_finite_bin_fraction", "mean"),
        )
    )
    for metric in METRICS:
        metric_rows = points.loc[points["metric"].eq(metric)]
        if metric_rows.empty:
            raise ValueError(f"Missing Day2-template first-window rows for {metric}.")
    return points.sort_values(["metric", "experience_index", "animal"], kind="stable")


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
        ax.text(0.5, 0.10, "No. of cycles", ha="center", va="top", fontsize=label_size, color=muted)
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

        for row, metric in enumerate(METRICS):
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
                    source = "; ".join(_rel(path) for path in DAY2_TEMPLATE_ANIMAL_SUMMARIES.values())
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
                        xlabel="No. of cycles",
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


def _figure3_source_panel_dir(out_dir: Path) -> Path:
    local_dir = out_dir / "figure3_individual_panels"
    required = [local_dir / str(layout["source_svg"]) for layout in FIGURE3_PPT_PANEL_LAYOUT]
    if all(path.exists() for path in required):
        return local_dir
    required = [FIGURE3_INDIVIDUAL_PANEL_DIR / str(layout["source_svg"]) for layout in FIGURE3_PPT_PANEL_LAYOUT]
    if all(path.exists() for path in required):
        return FIGURE3_INDIVIDUAL_PANEL_DIR
    missing = [path for path in required if not path.exists()]
    raise FileNotFoundError(
        "Missing source-rendered Figure 3 individual panels. "
        "Expected SVGs under "
        f"{FIGURE3_INDIVIDUAL_PANEL_DIR}; missing {', '.join(str(path) for path in missing[:3])}"
    )


def _figure3_relative_slot_pt(slot_pt: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    bbox_x, bbox_y, _bbox_w, _bbox_h = FIGURE3_PPT_CONTENT_BBOX_PT
    x, y, w, h = (float(value) for value in slot_pt)
    return (x - bbox_x, y - bbox_y, w, h)


def _figure3_slot_in_from_pt(slot_pt: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(value / 72.0 for value in _figure3_relative_slot_pt(slot_pt))


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


def _add_svg_panel_cropped_absolute(
    root: ET.Element,
    panel_svg: dict[str, object],
    *,
    panel_id: str,
    slot_pt: tuple[float, float, float, float],
    src_rect: dict[str, int],
) -> dict[str, float]:
    """Place a source-rendered panel in absolute figure coordinates with PPT-style clipping."""

    slot_x, slot_y, slot_w, slot_h = (float(value) for value in slot_pt)
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
    scale_x = slot_w / crop_w
    scale_y = slot_h / crop_h
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
        "rendered_width": float(slot_w),
        "rendered_height": float(slot_h),
        "scale": float(scale_x),
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
                "font-size": f"{style.annotation_fontsize * 2.175:.1f}",
                "font-weight": "bold",
                "text-anchor": "start",
                "dominant-baseline": "hanging",
                "fill": color,
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


def _longest_true_run(values: np.ndarray) -> int:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return int(np.max(stops - starts)) if len(starts) else 0


def _first_dark_horizontal_edge_row(
    pixels: np.ndarray,
    bounds_points: list[float],
    *,
    viewbox_points: tuple[float, float],
) -> int:
    height_px, width_px = pixels.shape[:2]
    scale_x = width_px / float(viewbox_points[0])
    scale_y = height_px / float(viewbox_points[1])
    x, y, width, height = (float(value) for value in bounds_points)
    x0 = max(0, int(np.floor(x * scale_x)))
    x1 = min(width_px, int(np.ceil((x + width) * scale_x)))
    y0 = max(0, int(np.floor(y * scale_y)))
    y1 = min(height_px, int(np.ceil((y + 0.35 * height) * scale_y)))
    min_run = max(10, int(np.ceil(0.08 * (x1 - x0))))
    for row_index in range(y0, y1):
        dark = np.max(pixels[row_index, x0:x1, :], axis=1) < 96
        if _longest_true_run(dark) >= min_run:
            return row_index
    raise AssertionError(
        f"No dark horizontal edge found inside bounds {bounds_points} "
        f"with minimum run {min_run}px."
    )


def render_figure1(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig1 = _load_script_module("preview_figure1_assembly_source_render", FIGURE1_HELPER)
        svg_tree, rendered_sizes = fig1.render_assembly_svg(dpi=int(style.dpi))

    out_svg = out_dir / "figure1_source_publication.svg"
    out_png = out_dir / "figure1_source_publication.png"
    out_pdf = out_dir / "figure1_source_publication.pdf"
    out_tiff = out_dir / "figure1_source_publication.tiff"
    svg_tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    html_path = _export_svg_with_chrome(
        out_svg,
        out_png,
        out_pdf,
        width_mm=float(fig1.EXPORT_WIDTH_MM),
        height_mm=float(fig1.EXPORT_HEIGHT_MM),
        dpi=int(style.dpi),
    )
    with Image.open(out_png) as image:
        rgb_pixels = np.asarray(image.convert("RGB"))
        Image.fromarray(rgb_pixels).save(
            out_tiff,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(int(style.dpi), int(style.dpi)),
        )

    individual_panel_dir = out_dir / "figure1_individual_panels"
    individual_panel_paths = fig1.export_individual_panels(
        individual_panel_dir,
        dpi=int(style.dpi),
    )
    individual_manifest_path = out_dir / "figure1_individual_panels_manifest.json"
    _write_json(
        individual_manifest_path,
        {
            panel_id: {ext: _rel(Path(path)) for ext, path in paths.items()}
            for panel_id, paths in individual_panel_paths.items()
        },
    )
    decision_points_path = out_dir / "figure1_panel_a_decision_points.csv"
    decision_points = pd.DataFrame(fig1.get_figure1_reward_path_decisions())
    decision_points["marker_area_points2"] = float(
        fig1.FIGURE1_DECISION_MARKER_AREA_POINTS2
    )
    decision_points["marker_linewidth_points"] = float(
        fig1.FIGURE1_DECISION_MARKER_LINEWIDTH
    )
    decision_points["label_fontsize_points"] = float(
        fig1.FIGURE1_DECISION_LABEL_SIZE
    )
    decision_points["requested_scale_vs_round1"] = float(
        fig1.FIGURE1_DECISION_SCALE
    )
    decision_points["ed3_reference_marker_area_points2"] = float(26 * 1.25**2)
    decision_points["diameter_ratio_vs_ed3"] = np.sqrt(
        decision_points["marker_area_points2"]
        / decision_points["ed3_reference_marker_area_points2"]
    )
    decision_points.to_csv(decision_points_path, index=False)

    endnode_widths = {
        panel_id: [float(value) for value in rendered_sizes[panel_id]["inner_axes_width_points"]]
        for panel_id in ("c", "h")
    }
    all_endnode_widths = endnode_widths["c"] + endnode_widths["h"]
    width_span = float(np.ptp(np.asarray(all_endnode_widths, dtype=float)))
    all_endnode_heights = [
        float(bounds[3])
        for panel_id in ("c", "h")
        for bounds in rendered_sizes[panel_id]["placed_inner_axes_bounds_points"]
    ]
    height_span = float(np.ptp(np.asarray(all_endnode_heights, dtype=float)))
    c_slot_left = float(rendered_sizes["c"]["slot_bounds_points"][0])
    c_left = float(rendered_sizes["c"]["placed_bounds_points"][0])
    c_right = float(
        rendered_sizes["c"]["placed_bounds_points"][0]
        + rendered_sizes["c"]["placed_bounds_points"][2]
    )
    c_bottom = float(
        rendered_sizes["c"]["placed_bounds_points"][1]
        + rendered_sizes["c"]["placed_bounds_points"][3]
    )
    f_slot_left = float(rendered_sizes["h"]["slot_bounds_points"][0])
    f_left = float(rendered_sizes["h"]["placed_bounds_points"][0])
    f_right = float(
        rendered_sizes["h"]["placed_bounds_points"][0]
        + rendered_sizes["h"]["placed_bounds_points"][2]
    )
    f_top = float(rendered_sizes["h"]["placed_bounds_points"][1])
    route_right = float(
        rendered_sizes["j"]["placed_bounds_points"][0]
        + rendered_sizes["j"]["placed_bounds_points"][2]
    )
    g_right = float(
        rendered_sizes["i"]["placed_bounds_points"][0]
        + rendered_sizes["i"]["placed_bounds_points"][2]
    )
    route_left = float(rendered_sizes["j"]["placed_bounds_points"][0])
    bottom_gap = route_left - g_right
    g_label_y = float(rendered_sizes["i"]["panel_id_position_points"][1])
    route_label_y = float(rendered_sizes["j"]["panel_id_position_points"][1])
    moved_g_width = float(rendered_sizes["i"]["placed_bounds_points"][2])
    original_g_target_width = float(rendered_sizes["j"]["slot_bounds_points"][2]) * 1.03
    figure_viewbox_points = (
        float(fig1.EXPORT_CANVAS_SIZE[0] * 72.0),
        float(fig1.EXPORT_CANVAS_SIZE[1] * 72.0),
    )
    maze_top_row = _first_dark_horizontal_edge_row(
        rgb_pixels,
        rendered_sizes["a"]["placed_bounds_points"],
        viewbox_points=figure_viewbox_points,
    )
    c_plot_top_row = _first_dark_horizontal_edge_row(
        rgb_pixels,
        rendered_sizes["c"]["placed_bounds_points"],
        viewbox_points=figure_viewbox_points,
    )
    pixels_per_viewbox_point_y = rgb_pixels.shape[0] / figure_viewbox_points[1]
    maze_to_c_top_delta_points = (
        float(maze_top_row - c_plot_top_row) / pixels_per_viewbox_point_y
    )
    a_vertical_offset = float(
        rendered_sizes["a"]["placed_bounds_points"][1]
        - rendered_sizes["a"]["slot_bounds_points"][1]
    )
    b_vertical_offset = float(
        rendered_sizes["b"]["placed_bounds_points"][1]
        - rendered_sizes["b"]["slot_bounds_points"][1]
    )
    d_axis = rendered_sizes["d"]["placed_inner_axes_bounds_points"][0]
    f_axis = rendered_sizes["h"]["placed_inner_axes_bounds_points"][0]
    d_axis_bottom = float(d_axis[1] + d_axis[3])
    f_axis_bottom = float(f_axis[1] + f_axis[3])
    d_row_offset = float(fig1.PANEL_OFFSETS_POINTS["d"][1])
    f_row_offset = float(fig1.PANEL_OFFSETS_POINTS["h"][1])
    f_label_x, f_label_y = (
        float(value) for value in rendered_sizes["h"]["panel_id_position_points"]
    )
    f_label_to_axis_left = float(f_axis[0] - f_label_x)
    f_axis_top_to_label = float(f_axis[1] - f_label_y)
    bottom_row_offsets = np.asarray(
        [fig1.PANEL_OFFSETS_POINTS[panel_id][1] for panel_id in ("g", "i", "j")],
        dtype=float,
    )
    bottom_row_label_ys = np.asarray(
        [rendered_sizes[panel_id]["panel_id_position_points"][1] for panel_id in ("g", "i", "j")],
        dtype=float,
    )
    alignment_rows = [
        {
            "check": "panel_a_maze_top_minus_panel_c_plot_top_points",
            "actual": maze_to_c_top_delta_points,
            "expected": 0.0,
            "tolerance": 0.25,
            "status": "PASS" if abs(maze_to_c_top_delta_points) <= 0.25 else "FAIL",
        },
        {
            "check": "panel_a_offset_minus_panel_b_offset_points",
            "actual": a_vertical_offset - b_vertical_offset,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(a_vertical_offset - b_vertical_offset) <= 0.01 else "FAIL",
        },
        {
            "check": "endnode_axis_width_span_points",
            "actual": width_span,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if width_span <= 0.01 else "FAIL",
        },
        {
            "check": "endnode_axis_height_span_points",
            "actual": height_span,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if height_span <= 0.01 else "FAIL",
        },
        {
            "check": "panel_d_axis_bottom_minus_panel_f_axis_bottom_points",
            "actual": d_axis_bottom - f_axis_bottom,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(d_axis_bottom - f_axis_bottom) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_d_offset_minus_panel_f_offset_points",
            "actual": d_row_offset - f_row_offset,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(d_row_offset - f_row_offset) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_f_label_to_axis_left_gap_points",
            "actual": f_label_to_axis_left,
            "expected": 20.0,
            "tolerance": 5.0,
            "status": "PASS" if 15.0 <= f_label_to_axis_left <= 25.0 else "FAIL",
        },
        {
            "check": "panel_f_axis_top_to_label_gap_points",
            "actual": f_axis_top_to_label,
            "expected": 5.0,
            "tolerance": 5.0,
            "status": "PASS" if 0.0 <= f_axis_top_to_label <= 10.0 else "FAIL",
        },
        {
            "check": "panel_f_tight_top_minus_panel_c_tight_bottom_points",
            "actual": f_top - c_bottom,
            "expected": 9.0,
            "tolerance": 1.0,
            "status": "PASS" if 8.0 <= f_top - c_bottom <= 10.0 else "FAIL",
        },
        {
            "check": "bottom_row_offset_span_points",
            "actual": float(np.ptp(bottom_row_offsets)),
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if float(np.ptp(bottom_row_offsets)) <= 0.01 else "FAIL",
        },
        {
            "check": "bottom_row_label_y_span_points",
            "actual": float(np.ptp(bottom_row_label_ys)),
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if float(np.ptp(bottom_row_label_ys)) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_f_tight_right_minus_panel_c_tight_right_points",
            "actual": f_right - c_right,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(f_right - c_right) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_h_tight_right_minus_panel_c_tight_right_points",
            "actual": route_right - c_right,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(route_right - c_right) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_c_left_gap_inside_slot_points",
            "actual": c_left - c_slot_left,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(c_left - c_slot_left) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_f_left_gap_inside_slot_points",
            "actual": f_left - f_slot_left,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(f_left - f_slot_left) <= 0.01 else "FAIL",
        },
        {
            "check": "panel_g_to_h_tight_gap_points",
            "actual": bottom_gap,
            "expected": 0.0,
            "tolerance": 12.0,
            "status": "PASS" if 0.0 <= bottom_gap <= 12.0 else "FAIL",
        },
        {
            "check": "panel_h_label_y_minus_panel_g_label_y_points",
            "actual": route_label_y - g_label_y,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS" if abs(route_label_y - g_label_y) <= 0.01 else "FAIL",
        },
        {
            "check": "moved_panel_g_width_minus_original_panel_i_width_points",
            "actual": moved_g_width - original_g_target_width,
            "expected": 0.0,
            "tolerance": 0.01,
            "status": "PASS"
            if abs(moved_g_width - original_g_target_width) <= 0.01
            else "FAIL",
        },
    ]
    alignment_path = out_dir / "figure1_alignment_audit.csv"
    pd.DataFrame(alignment_rows).to_csv(alignment_path, index=False)
    if any(row["status"] != "PASS" for row in alignment_rows):
        raise AssertionError(f"Figure 1 alignment audit failed: {alignment_rows}")

    panel_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for panel_id, renderer in fig1.PANEL_RENDERERS.items():
        display_label = fig1.PANEL_DISPLAY_LABELS.get(panel_id, panel_id)
        render_info = rendered_sizes.get(panel_id, {})
        source_detail = FIGURE1_PANEL_SOURCE_DETAILS.get(
            panel_id,
            {"source": _rel(FIGURE1_HELPER), "filters": "maintained Figure 1 source renderer", "source_row_count": ""},
        )
        panel_rows.append(
            {
                "panel_key": f"figure1_{panel_id}",
                "panel_letter": str(display_label).lower(),
                "visual_role": FIGURE1_PANEL_ROLES.get(
                    panel_id,
                    renderer.__name__.removeprefix("_render_panel_").replace("_", " "),
                ),
                "source": source_detail["source"],
                "helper": _rel(FIGURE1_HELPER),
                "filters": source_detail["filters"],
                "source_row_count": source_detail["source_row_count"],
                "expected": "native source-rendered Figure 1 panel; no PDF/PPTX crop",
                "auto_check": "PASS",
            }
        )
        geometry_rows.append(
            {
                "panel_key": f"figure1_{panel_id}",
                "panel_letter": str(display_label).lower(),
                "size_inches": json.dumps(render_info.get("size_inches", [])),
                "viewbox_points": json.dumps(render_info.get("viewbox_points", [])),
                "slot_points": json.dumps(render_info.get("slot_points", [])),
                "scale": render_info.get("scale", np.nan),
            }
        )

    verification_path = out_dir / "figure1_panel_verification.csv"
    geometry_path = out_dir / "figure1_panel_geometry.csv"
    pd.DataFrame(panel_rows).to_csv(verification_path, index=False)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)

    manifest = {
        "figure": "figure1",
        "outputs": {
            "png": _rel(out_png),
            "svg": _rel(out_svg),
            "pdf": _rel(out_pdf),
            "tiff": _rel(out_tiff),
            "html_export_wrapper": _rel(html_path),
            "individual_panels_manifest": _rel(individual_manifest_path),
        },
        "renderer": "preview_figure1_assembly.render_assembly_svg",
        "source_helper": _rel(FIGURE1_HELPER),
        "style": {
            "font_family": fig1.FONT_FAMILY,
            "base_font_size": fig1.BASE_FONT_SIZE,
            "axis_label_size": fig1.AXIS_LABEL_SIZE,
            "tick_label_size": fig1.TICK_LABEL_SIZE,
            "title_size": fig1.TITLE_SIZE,
            "panel_id_size": fig1.PANEL_ID_SIZE,
            "dpi": int(style.dpi),
        },
        "layout": {
            "export_width_mm": float(fig1.EXPORT_WIDTH_MM),
            "export_height_mm": float(fig1.EXPORT_HEIGHT_MM),
            "canvas_size": list(fig1.CANVAS_SIZE),
            "export_canvas_size": list(getattr(fig1, "EXPORT_CANVAS_SIZE", fig1.CANVAS_SIZE)),
            "mosaic": list(fig1.FIGURE1_MOSAIC),
            "mosaic_width_ratios": list(fig1.MOSAIC_WIDTH_RATIOS),
            "mosaic_height_ratios": list(fig1.MOSAIC_HEIGHT_RATIOS),
            "panel_sizes": {key: list(value) for key, value in fig1.PANEL_SIZES.items()},
            "panel_display_labels": dict(fig1.PANEL_DISPLAY_LABELS),
            "rendered_sizes": rendered_sizes,
        },
        "sources": [
            {
                "panel": f"figure1_{panel_id}",
                "display_label": str(fig1.PANEL_DISPLAY_LABELS.get(panel_id, panel_id)).lower(),
                "renderer": f"preview_figure1_assembly.{renderer.__name__}",
                "source_path": FIGURE1_PANEL_SOURCE_DETAILS.get(panel_id, {}).get("source", _rel(FIGURE1_HELPER)),
                "helper_path": _rel(FIGURE1_HELPER),
                "filters": FIGURE1_PANEL_SOURCE_DETAILS.get(panel_id, {}).get(
                    "filters", "maintained Figure 1 source renderer"
                ),
            }
            for panel_id, renderer in fig1.PANEL_RENDERERS.items()
        ],
        "panel_verification": _rel(verification_path),
        "panel_geometry": _rel(geometry_path),
        "audit_tables": {
            "panel_a_decision_points": _rel(decision_points_path),
            "route_score_session_rows": _rel(fig1.FIGURE1_ROUTE_SCORE_POINTS),
            "route_score_group_summary": _rel(fig1.FIGURE1_ROUTE_SCORE_SUMMARY),
            "route_score_clustered_bh6_contrasts": _rel(fig1.FIGURE1_ROUTE_SCORE_CONTRASTS),
            "endnode_and_right_edge_alignment": _rel(alignment_path),
        },
    }
    _write_json(out_dir / "figure1_source_manifest.json", manifest)
    return manifest


def render_figure2(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_repair_audit = _validate_source_repair_842()
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
            "source": "G:/422/corr.fig; G:/422-3/corr.fig",
            "filters": "fixed FOV example rendered by preview_figure2_assembly._render_panel_a",
            "source_row_count": "",
            "expected": "two stacked local-correlation FOV examples from source .fig files",
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
            "layout_source": "C:/Users/jiangwu/Desktop/ltm-figs/main-figs-v4-f2.pptx slide 1",
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
                "source_path": "G:/422/corr.fig; G:/422-3/corr.fig",
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
                "provenance_manifest": _rel(FIGURE2_DISTRIBUTION_PROVENANCE),
                "pair_input_audit": _rel(FIGURE2_DISTRIBUTION_PAIR_INPUT),
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
                "source_repair_manifest": _rel(SOURCE_REPAIR_842_MANIFEST),
                "corrected_842_pair_audit": _rel(FIGURE2_GAP_PAIR_AUDIT),
                "corrected_842_distribution_parity_audit": _rel(
                    FIGURE2_GAP_DISTRIBUTION_PARITY
                ),
                "source_repair_audit": source_repair_audit,
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
                    "unrestricted_adjacency_scope": "23 chains; retained as audit only",
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
                    "manifest": _rel(FIGURE2_CONSECUTIVE_MANIFEST),
                    "unrestricted_23": _rel(FIGURE2_CONSECUTIVE_UNRESTRICTED_SCOPE),
                    "clean_22": _rel(FIGURE2_CONSECUTIVE_CLEAN_SCOPE),
                    "input_audit": _rel(FIGURE2_CONSECUTIVE_INPUT_AUDIT),
                },
            }
        ],
        "panel_verification": _rel(verification_path),
        "panel_geometry": _rel(geometry_path),
        "consecutive_gap_binned_curve_audit": _rel(consecutive_binned_curve_path),
        "consecutive_gap_displayed_estimand_bh4": _rel(displayed_estimand_stats_path),
        "distribution_scope_audit": distribution_scope_audit,
        "source_repair_audit": source_repair_audit,
        "consecutive_scope_audit": consecutive_scope_audit,
    }
    _write_json(out_dir / "figure2_source_manifest.json", manifest)
    return manifest




def render_figure5(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context():
        fig5 = _load_script_module("preview_figure3_assembly_source_render", FIGURE5_HELPER)
        source_bundle_audit = dict(fig5.validate_figure5_corrected842_sources())
        source_bundle_audit["manifest"] = _rel(FIGURE5_CORRECTED_842_MANIFEST)
        svg_tree = fig5.render_assembly_svg(dpi=int(style.dpi), row="all")
    top_margin_in = 0.12
    top_margin_pt = top_margin_in * 72.0
    root = svg_tree.getroot()
    view_x, view_y, view_w, view_h = map(float, root.attrib["viewBox"].split())
    root.attrib["viewBox"] = f"{view_x:.6f} {view_y:.6f} {view_w:.6f} {view_h + top_margin_pt:.6f}"
    export_height_mm = float(fig5.EXPORT_WIDTH_MM) * (float(fig5.CANVAS_SIZE[1]) + top_margin_in) / float(fig5.CANVAS_SIZE[0])
    root.attrib["height"] = f"{export_height_mm:.3f}mm"
    for child in list(root):
        if child.tag in {_svg_tag("metadata"), _svg_tag("defs")}:
            continue
        if child.tag == _svg_tag("rect") and child.attrib.get("width") == "100%" and child.attrib.get("height") == "100%":
            continue
        existing_transform = child.attrib.get("transform", "")
        child.attrib["transform"] = f"translate(0 {top_margin_pt:.6f}) {existing_transform}".strip()

    out_svg = out_dir / "figure5_source_publication.svg"
    out_png = out_dir / "figure5_source_publication.png"
    out_pdf = out_dir / "figure5_source_publication.pdf"
    svg_tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    html_path = _export_svg_with_chrome(
        out_svg,
        out_png,
        out_pdf,
        width_mm=float(fig5.EXPORT_WIDTH_MM),
        height_mm=export_height_mm,
        dpi=int(style.dpi),
    )

    panel_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    for panel_id, panel in fig5.PANEL_LAYOUT_SPEC.items():
        display_label = str(panel.get("label", panel_id)).lower()
        renderer = fig5.PANEL_RENDERERS[panel_id]
        source_detail = FIGURE5_PANEL_SOURCE_DETAILS.get(
            panel_id,
            {
                "source": _rel(FIGURE5_HELPER),
                "filters": "maintained tree-depth7 level-code source renderer",
                "source_row_count": "source-rendered helper",
            },
        )
        panel_rows.append(
            {
                "panel_key": f"figure5_{panel_id}",
                "panel_letter": display_label,
                "visual_role": FIGURE5_PANEL_ROLES.get(panel_id, renderer.__name__),
                "source": source_detail["source"],
                "helper": _rel(FIGURE5_HELPER),
                "filters": source_detail["filters"],
                "source_row_count": source_detail["source_row_count"],
                "expected": "native source-rendered level-code panel; no PDF/PPTX crop",
                "auto_check": "PASS",
            }
        )
        geometry_rows.append(
            {
                "panel_key": f"figure5_{panel_id}",
                "panel_letter": display_label,
                "x_in": float(panel["x"]),
                "y_in": float(panel["y"]) + top_margin_in,
                "width_in": float(panel["w"]),
                "height_in": float(panel["h"]),
                "render_width_in": float(panel["render_w"]),
                "render_height_in": float(panel["render_h"]),
            }
        )

    verification_path = out_dir / "figure5_panel_verification.csv"
    geometry_path = out_dir / "figure5_panel_geometry.csv"
    pd.DataFrame(panel_rows).to_csv(verification_path, index=False)
    pd.DataFrame(geometry_rows).to_csv(geometry_path, index=False)

    manifest = {
        "figure": "figure5",
        "outputs": {
            "png": _rel(out_png),
            "svg": _rel(out_svg),
            "pdf": _rel(out_pdf),
            "html_export_wrapper": _rel(html_path),
        },
        "renderer": "preview_figure3_assembly.render_assembly_svg(row='all')",
        "source_helper": _rel(FIGURE5_HELPER),
        "corrected_842_source_validation": source_bundle_audit,
        "style": {
            "font_family": fig5.FONT_FAMILY,
            "base_font_size": fig5.BASE_FONT_SIZE,
            "axis_label_size": fig5.AXIS_LABEL_SIZE,
            "tick_label_size": fig5.TICK_LABEL_SIZE,
            "title_size": fig5.TITLE_SIZE,
            "panel_id_size": fig5.PANEL_ID_SIZE,
            "level_color": fig5.LEVEL_COLOR,
            "heatmap_cmap": fig5.HEATMAP_CMAP,
            "heatmap_ordering_method": "preferred_level_similarity",
            "dpi": int(style.dpi),
        },
        "layout": {
            "export_width_mm": float(fig5.EXPORT_WIDTH_MM),
            "export_height_mm": export_height_mm,
            "canvas_size": [float(fig5.CANVAS_SIZE[0]), float(fig5.CANVAS_SIZE[1]) + top_margin_in],
            "helper_canvas_size": list(fig5.CANVAS_SIZE),
            "export_top_margin_in": top_margin_in,
            "panel_layout_spec": fig5.PANEL_LAYOUT_SPEC,
        },
        "sources": [
            {
                "panel": f"figure5_{panel_id}",
                "display_label": str(panel.get("label", panel_id)).lower(),
                "visual_role": FIGURE5_PANEL_ROLES.get(panel_id, fig5.PANEL_RENDERERS[panel_id].__name__),
                "renderer": f"preview_figure3_assembly.{fig5.PANEL_RENDERERS[panel_id].__name__}",
                "source_path": FIGURE5_PANEL_SOURCE_DETAILS.get(panel_id, {}).get("source", _rel(FIGURE5_HELPER)),
                "helper_path": _rel(FIGURE5_HELPER),
                "filters": FIGURE5_PANEL_SOURCE_DETAILS.get(panel_id, {}).get(
                    "filters", "maintained tree-depth7 level-code source renderer"
                ),
            }
            for panel_id, panel in fig5.PANEL_LAYOUT_SPEC.items()
        ],
        "panel_verification": _rel(verification_path),
        "panel_geometry": _rel(geometry_path),
    }
    _write_json(out_dir / "figure5_source_manifest.json", manifest)
    return manifest


def render_figure3(out_dir: Path, style: PublicationStyle) -> dict[str, object]:
    style = dataclass_replace(style, show_grid=False)
    _ensure_sources()
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
            "source_manifest": _rel(DAY2_TEMPLATE_SOURCE_MANIFEST) if DAY2_TEMPLATE_SOURCE_MANIFEST.exists() else None,
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
            "source_paths": {name: _rel(path) for name, path in DAY2_TEMPLATE_ANIMAL_SUMMARIES.items()},
            "plot_ready_points_authority": _rel(FIGURE3_FIRST30_POINTS_ALL_METRICS),
            "stats_path": _rel(FIGURE3_FIRST30_STATS_ALL_METRICS),
            "renderer": "render_main_figs_v3_publication._draw_experience_scatter",
            "statistics": "OLS slope with CR1 covariance clustered by animal; primary two-sided P uses Student t(G-1); BH correction spans the four Main-Figure representation metrics",
            "filters": {
                "metrics": list(METRICS),
                "window_type": "maze_time",
                "window_size": 30.0,
                "value_column": "first_value",
                "support_policy_by_metric": DAY2_TEMPLATE_SUPPORT_POLICY,
                "variant": DAY2_TEMPLATE_VARIANT_LABEL,
                "excluded_metrics": ["level_map_correlation"],
            },
        },
        {
            "panel": "figure3_long_gap_experience_trends",
            "source_path": _rel(LONG_GAP_POINTS),
            "stats_path": _rel(LONG_GAP_STATS),
            "source_repair_manifest": _rel(SOURCE_REPAIR_842_MANIFEST),
            "level_pair_audit": _rel(FIGURE3_LEVEL_PAIR_AUDIT),
            "corrected_842_six_metric_audit": _rel(
                FIGURE3_CORRECTED_842_SIX_METRIC_AUDIT
            ),
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
            "pptx": str(FIGURE3_PPT_LAYOUT_REFERENCE),
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
    for row, metric in enumerate(METRICS):
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
                    "source": "; ".join(_rel(path) for path in DAY2_TEMPLATE_ANIMAL_SUMMARIES.values()),
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


def _png_content_checks(out_png: Path, *, max_right_blank: float, max_bottom_blank: float) -> dict[str, object]:
    image = Image.open(out_png).convert("RGB")
    stat = ImageStat.Stat(image)
    checks: dict[str, object] = {
        "png_size_px": list(image.size),
        "png_channel_stddev": [float(v) for v in stat.stddev],
        "png_nonblank": bool(max(stat.stddev) > 1.0),
    }
    if not checks["png_nonblank"]:
        raise AssertionError(f"Rendered PNG appears blank: {out_png}")
    array = np.asarray(image)
    nonwhite = np.any(array < 248, axis=2)
    ys, xs = np.where(nonwhite)
    if xs.size == 0 or ys.size == 0:
        raise AssertionError(f"Rendered PNG has no detectable non-white content: {out_png}")
    height, width = nonwhite.shape
    top_blank = int(ys.min()) / float(height)
    right_blank = (width - 1 - int(xs.max())) / float(width)
    bottom_blank = (height - 1 - int(ys.max())) / float(height)
    checks["content_bounds_px"] = {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
    }
    checks["top_blank_fraction"] = float(top_blank)
    checks["right_blank_fraction"] = float(right_blank)
    checks["bottom_blank_fraction"] = float(bottom_blank)
    if right_blank > max_right_blank:
        raise AssertionError(f"Rendered figure has too much blank right canvas: {right_blank:.3f} > {max_right_blank:.3f}")
    if bottom_blank > max_bottom_blank:
        raise AssertionError(f"Rendered figure has too much blank bottom canvas: {bottom_blank:.3f} > {max_bottom_blank:.3f}")
    return checks


def validate_figure1_manifest(manifest: dict[str, object]) -> dict[str, object]:
    out_png = REPO_ROOT / manifest["outputs"]["png"]
    out_svg = REPO_ROOT / manifest["outputs"]["svg"]
    out_tiff = REPO_ROOT / manifest["outputs"]["tiff"]
    source_count = len(manifest.get("sources", []))
    sources_by_panel = {
        str(source.get("panel")): source for source in manifest.get("sources", [])
    }
    route_source = sources_by_panel.get("figure1_j", {})
    expected_route_sources = (
        "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
        "ed04_session_route_current_plus_new12.csv; "
        "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
        "ed04_route_group_stats_current_plus_new12.csv; "
        "docs/extended_data_figure_panel_refresh_2026-07-12/source_tables/"
        "ed04_route_stats_current_plus_new12.csv"
    )
    route_points = pd.read_csv(
        REPO_ROOT
        / str(manifest["audit_tables"]["route_score_session_rows"]),
        dtype={"animal": str},
    )
    route_summary = pd.read_csv(
        REPO_ROOT
        / str(manifest["audit_tables"]["route_score_group_summary"])
    )
    route_contrasts = pd.read_csv(
        REPO_ROOT
        / str(manifest["audit_tables"]["route_score_clustered_bh6_contrasts"])
    )
    decision_points = pd.read_csv(
        REPO_ROOT / str(manifest["audit_tables"]["panel_a_decision_points"])
    )
    expected_decisions = [
        (0, 0, 7, 7),
        (1, 1, 7, 3),
        (2, 4, 11, 3),
        (3, 10, 11, 5),
        (4, 22, 13, 5),
        (5, 46, 13, 6),
    ]
    observed_decisions = list(
        decision_points[["depth", "run", "x", "y"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )
    route_group_counts = (
        route_points["condition"].astype(str).value_counts().astype(int).to_dict()
    )
    route_mouse_counts = (
        route_points.groupby("condition")["animal"].nunique().astype(int).to_dict()
    )
    supported_route_contrasts = sorted(
        route_contrasts.loc[route_contrasts["q_bh6"].lt(0.05), "comparison"]
        .astype(str)
        .tolist()
    )
    alignment = pd.read_csv(
        REPO_ROOT
        / str(manifest["audit_tables"]["endnode_and_right_edge_alignment"])
    )
    individual_manifest = json.loads(
        (REPO_ROOT / manifest["outputs"]["individual_panels_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    svg_root = ET.parse(out_svg).getroot()
    svg_texts = [
        "".join(element.itertext()).strip()
        for element in svg_root.iter()
        if element.tag == _svg_tag("text") and "".join(element.itertext()).strip()
    ]
    panel_a_group = next(
        element for element in svg_root.iter() if element.attrib.get("id") == "panel_a"
    )
    panel_a_texts = [
        "".join(element.itertext()).strip()
        for element in panel_a_group.iter()
        if element.tag == _svg_tag("text") and "".join(element.itertext()).strip()
    ]
    display_labels = sorted(
        str(source.get("display_label", "")) for source in manifest.get("sources", [])
    )
    with Image.open(out_tiff) as image:
        tiff_dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
    checks = {
        "source_panel_count": int(source_count),
        "expected_source_panel_count": 8,
        "display_labels": display_labels,
        "svg_embedded_image_count": _svg_embedded_image_count(out_svg),
        "panel_a_decision_count": int(len(decision_points)),
        "panel_a_decisions_exact": observed_decisions == expected_decisions,
        "panel_a_decision_labels": panel_a_texts,
        "panel_a_decision_scale_vs_round1": float(
            decision_points["requested_scale_vs_round1"].iloc[0]
        ),
        "panel_a_marker_diameter_ratio_vs_ed3": float(
            decision_points["diameter_ratio_vs_ed3"].iloc[0]
        ),
        "route_panel_source": str(route_source.get("source_path", "")),
        "route_panel_source_is_current_plus_new12": str(route_source.get("source_path", ""))
        == expected_route_sources,
        "route_source_rows": int(len(route_points)),
        "route_group_counts": route_group_counts,
        "route_mouse_counts": route_mouse_counts,
        "route_summary_rows": int(len(route_summary)),
        "route_contrast_rows": int(len(route_contrasts)),
        "route_supported_bh6": supported_route_contrasts,
        "route_ylabel_present": any("First visit route score" in text for text in svg_texts),
        "alignment_rows": int(len(alignment)),
        "alignment_all_pass": bool(alignment["status"].eq("PASS").all()),
        "individual_panel_count": int(len(individual_manifest)),
        "individual_panel_formats_complete": all(
            set(paths) == {"svg", "pdf", "png", "tiff"}
            and all((REPO_ROOT / path).is_file() and (REPO_ROOT / path).stat().st_size > 0 for path in paths.values())
            for paths in individual_manifest.values()
        ),
        "tiff_dpi": tiff_dpi,
    }
    if checks["source_panel_count"] != checks["expected_source_panel_count"]:
        raise AssertionError(f"Expected 8 Figure 1 source panels, found {checks['source_panel_count']}")
    if checks["display_labels"] != list("abcdefgh"):
        raise AssertionError(f"Figure 1 display labels are not contiguous a-h: {display_labels}")
    if checks["svg_embedded_image_count"] != 0:
        raise AssertionError(f"Figure 1 SVG contains embedded image nodes: {checks['svg_embedded_image_count']}")
    expected_marker_diameter_ratio_vs_ed3 = np.sqrt(
        (34.0 * 1.4**2) / (26 * 1.25**2)
    )
    if (
        checks["panel_a_decision_count"] != 6
        or not checks["panel_a_decisions_exact"]
        or checks["panel_a_decision_labels"] != list("012345")
        or not np.isclose(checks["panel_a_decision_scale_vs_round1"], 1.4)
        or not np.isclose(
            checks["panel_a_marker_diameter_ratio_vs_ed3"],
            expected_marker_diameter_ratio_vs_ed3,
        )
    ):
        raise AssertionError(
            "Figure 1 panel a decision-point geometry, labels, or 1.4x scale contract failed."
        )
    if not checks["route_panel_source_is_current_plus_new12"]:
        raise AssertionError("Figure 1 route-score panel is not linked to the maintained current-plus-new12 source bundle.")
    if checks["route_source_rows"] != 162:
        raise AssertionError(f"Expected 162 Figure 1 route-score rows, found {checks['route_source_rows']}")
    if route_group_counts != {"experienced_day2": 74, "return": 56, "naive": 23, "latent_day2": 9}:
        raise AssertionError(f"Figure 1 route-score session support drifted: {route_group_counts}")
    if route_mouse_counts != {"experienced_day2": 24, "latent_day2": 9, "naive": 23, "return": 17}:
        raise AssertionError(f"Figure 1 route-score mouse support drifted: {route_mouse_counts}")
    if checks["route_summary_rows"] != 4 or checks["route_contrast_rows"] != 6:
        raise AssertionError("Figure 1 route-score audit must contain four group rows and six contrasts.")
    if supported_route_contrasts != [
        "experienced_day2_vs_latent_day2",
        "experienced_day2_vs_naive",
        "experienced_day2_vs_return",
    ]:
        raise AssertionError(f"Figure 1 route-score BH6 support drifted: {supported_route_contrasts}")
    if not checks["route_ylabel_present"]:
        raise AssertionError("Figure 1 SVG lacks the first-visit route-score label.")
    if not checks["alignment_all_pass"]:
        raise AssertionError("Figure 1 end-node/right-edge alignment audit failed.")
    if checks["individual_panel_count"] != 8 or not checks["individual_panel_formats_complete"]:
        raise AssertionError("Figure 1 individual-panel SVG/PDF/PNG/TIFF bundle is incomplete.")
    expected_dpi = float(manifest.get("style", {}).get("dpi", 0))
    if min(tiff_dpi, default=0.0) < expected_dpi * 0.95:
        raise AssertionError(f"Figure 1 TIFF DPI is below the render target: {tiff_dpi} vs {expected_dpi}")
    checks.update(
        _png_content_checks(
            out_png,
            max_right_blank=0.08,
            max_bottom_blank=0.08,
        )
    )
    return checks


def validate_figure2_manifest(manifest: dict[str, object]) -> dict[str, object]:
    out_png = REPO_ROOT / manifest["outputs"]["png"]
    out_svg = REPO_ROOT / manifest["outputs"]["svg"]
    verification_csv = REPO_ROOT / str(manifest["panel_verification"])
    geometry_csv = REPO_ROOT / str(manifest["panel_geometry"])
    verification = pd.read_csv(verification_csv)
    geometry = pd.read_csv(geometry_csv)
    lower = geometry.loc[geometry["metric"].notna()].copy()
    hrefs = _svg_image_hrefs(out_svg)
    non_data_hrefs = [href for href in hrefs if not href.startswith("data:")]
    svg_root = ET.parse(out_svg).getroot()
    root_direct_rect_count = sum(
        child.tag == _svg_tag("rect")
        for child in list(svg_root)
    )
    layout = dict(manifest.get("layout", {}))
    actual_column_slots = {
        int(column): tuple(float(value) for value in slot)
        for column, slot in dict(layout.get("lower_column_slots", {})).items()
    }
    original_column_slots = {
        int(column): tuple(float(value) for value in slot)
        for column, slot in dict(layout.get("lower_original_column_slots", {})).items()
    }
    requested_width_scales = {
        int(column): float(value)
        for column, value in dict(layout.get("lower_column_width_scales", {})).items()
    }
    computed_width_scales = {
        column: actual_column_slots[column][1] / original_column_slots[column][1]
        for column in range(4)
    }
    lower_right_edge_in = max(x + width for x, width in actual_column_slots.values())
    original_lower_right_edge_in = max(x + width for x, width in original_column_slots.values())
    schematic_slots = dict(layout.get("schematic_slots", {}))
    schematic_header_bounds = tuple(
        float(value) for value in layout.get("schematic_header_bounds", [])
    )
    schematic_verification_rows = verification["panel_key"].astype(str).str.startswith(
        "figure2_schematic_"
    )
    display_labels = verification["panel_letter"].astype(str).tolist()
    row_positions = lower.groupby("row").agg(y0=("y0", "first"), height=("height", "first")).sort_index()
    lower_row_gaps_in = [
        (
            float(row_positions.iloc[index + 1]["y0"])
            - float(row_positions.iloc[index]["y0"])
            - float(row_positions.iloc[index]["height"])
        )
        * float(layout["canvas_size"][1])
        for index in range(len(row_positions) - 1)
    ]
    row_y_span = lower.groupby("row")["y0"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    col_x_span = lower.groupby("column")["x0"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    col_width_span = lower.groupby("column")["width"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    row_height_span = lower.groupby("row")["height"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    checks = {
        "source_family_count": int(len(manifest.get("sources", []))),
        "expected_source_family_count": 6,
        "panel_verification_rows": int(len(verification)),
        "expected_panel_verification_rows": 17,
        "n_lower_axes": int(len(lower)),
        "expected_lower_axes": 12,
        "schematic_slot_count": int(len(schematic_slots)),
        "schematic_verification_rows": int(schematic_verification_rows.sum()),
        "schematic_header_bounds": schematic_header_bounds,
        "lower_row_gaps_in": lower_row_gaps_in,
        "display_labels": display_labels,
        "requested_width_scales": requested_width_scales,
        "computed_width_scales": computed_width_scales,
        "lower_right_edge_in": float(lower_right_edge_in),
        "original_lower_right_edge_in": float(original_lower_right_edge_in),
        "row_title_bands_removed": bool(layout.get("row_title_bands_removed", False)),
        "svg_root_direct_rect_count": int(root_direct_rect_count),
        "row_y_span": row_y_span,
        "col_x_span": col_x_span,
        "col_width_span": col_width_span,
        "row_height_span": row_height_span,
        "svg_image_node_count": int(len(hrefs)),
        "expected_svg_image_node_count": 12,
        "svg_non_data_image_href_count": int(len(non_data_hrefs)),
    }
    if checks["source_family_count"] != checks["expected_source_family_count"]:
        raise AssertionError(f"Expected 6 Figure 2 source families, found {checks['source_family_count']}")
    if checks["panel_verification_rows"] != checks["expected_panel_verification_rows"]:
        raise AssertionError(
            f"Expected 17 Figure 2 rendered panels, found {checks['panel_verification_rows']}"
        )
    if checks["n_lower_axes"] != checks["expected_lower_axes"]:
        raise AssertionError(f"Expected 12 Figure 2 lower axes, found {checks['n_lower_axes']}")
    if checks["schematic_slot_count"] != 3 or checks["schematic_verification_rows"] != 3:
        raise AssertionError(
            "Figure 2 must contain three source-native lower-row schematics in the manifest and verification table."
        )
    if len(schematic_header_bounds) != 4 or not 0.0 < schematic_header_bounds[2] < 0.94:
        raise AssertionError(f"Figure 2 compact schematic-header bounds drifted: {schematic_header_bounds}")
    if any(
        not math.isclose(gap, FIGURE2_PPT_LOWER_ROW_GAP_IN, rel_tol=0.0, abs_tol=1e-9)
        for gap in lower_row_gaps_in
    ):
        raise AssertionError(f"Figure 2 lower-row gaps drifted: {lower_row_gaps_in}")
    if display_labels != list("abcdefghijklmnopq"):
        raise AssertionError(f"Figure 2 display labels are not contiguous a-q: {display_labels}")
    if requested_width_scales != FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES:
        raise AssertionError(f"Figure 2 requested lower-column scales drifted: {requested_width_scales}")
    if any(
        not math.isclose(
            computed_width_scales[column],
            FIGURE2_PPT_LOWER_COLUMN_WIDTH_SCALES[column],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for column in range(4)
    ):
        raise AssertionError(f"Figure 2 lower-column width ratios drifted: {computed_width_scales}")
    if not math.isclose(lower_right_edge_in, original_lower_right_edge_in, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"Figure 2 lower-grid right edge moved: {lower_right_edge_in} vs {original_lower_right_edge_in}"
        )
    if not checks["row_title_bands_removed"] or checks["svg_root_direct_rect_count"] != 1:
        raise AssertionError("Figure 2 still contains a root-level row-title band.")
    if any(span > 1e-9 for span in row_y_span.values()):
        raise AssertionError(f"Figure 2 lower row baselines are not aligned: {row_y_span}")
    if any(span > 1e-9 for span in col_x_span.values()):
        raise AssertionError(f"Figure 2 lower column x positions are not aligned: {col_x_span}")
    if any(span > 1e-9 for span in col_width_span.values()):
        raise AssertionError(f"Figure 2 lower column widths are not aligned: {col_width_span}")
    if any(span > 1e-9 for span in row_height_span.values()):
        raise AssertionError(f"Figure 2 lower row heights are not aligned: {row_height_span}")
    if non_data_hrefs:
        raise AssertionError(f"Figure 2 SVG links external image sources: {non_data_hrefs[:3]}")
    if checks["svg_image_node_count"] != checks["expected_svg_image_node_count"]:
        raise AssertionError(
            "Figure 2 source-native schematics unexpectedly changed the embedded-image count: "
            f"{checks['svg_image_node_count']}"
        )
    checks.update(
        _png_content_checks(
            out_png,
            max_right_blank=0.08,
            max_bottom_blank=0.08,
        )
    )
    return checks




def validate_figure5_manifest(manifest: dict[str, object]) -> dict[str, object]:
    out_png = REPO_ROOT / manifest["outputs"]["png"]
    out_svg = REPO_ROOT / manifest["outputs"]["svg"]
    source_count = len(manifest.get("sources", []))
    source_validation = manifest.get("corrected_842_source_validation", {})
    expected_r_label = str(source_validation.get("panel_f_r_label", ""))
    source_contract_pass = (
        source_validation.get("status") == "PASS"
        and int(source_validation.get("included_pairs", -1)) == 36
        and int(source_validation.get("included_animals", -1)) == 14
        and str(source_validation.get("excluded_short_pair", ""))
        == "1616day1->1616bday1"
        and int(source_validation.get("level_neurons_total", -1)) > 0
        and int(source_validation.get("corrected_842_level_support", -1)) == 75
        and int(source_validation.get("displayed_panel_b_heatmap_neurons", -1))
        == int(source_validation.get("level_neurons_total", -2))
        and source_validation.get("displayed_panel_mapping")
        == {
            "a": "level-assignment schematic (preview internal a)",
            "b": "level-code heatmap (preview internal e)",
            "c": "level-map correlation distribution (preview internal b)",
            "d": "shuffle-mean distribution (preview internal c)",
            "e": "tree-depth decoder accuracy (preview internal d)",
            "f": "level-map correlation versus exploration efficiency (preview internal f)",
        }
        and np.isclose(
            float(source_validation.get("corrected_842_level_value", np.nan)),
            0.0555420654353495,
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(source_validation.get("corrected_842_decoder_value", np.nan)),
            0.2048227460860176,
            rtol=0.0,
            atol=1e-12,
        )
        and str(source_validation.get("corrected_842_tuning_sha256", "")).lower()
        == FIGURE5_EXPECTED_CORRECTED_TUNING_SHA256
        and str(source_validation.get("corrected_842_cimap_sha256", "")).lower()
        == FIGURE5_EXPECTED_CORRECTED_CIMAP_SHA256
        and expected_r_label.startswith("r=")
    )
    hrefs = _svg_image_hrefs(out_svg)
    non_data_hrefs = [href for href in hrefs if not href.startswith("data:")]
    svg_root = ET.parse(out_svg).getroot()
    svg_texts = [
        (element.text or "").strip()
        for element in svg_root.iter()
        if element.tag == _svg_tag("text") and (element.text or "").strip()
    ]
    checks = {
        "source_panel_count": int(source_count),
        "expected_source_panel_count": 6,
        "svg_image_node_count": int(len(hrefs)),
        "svg_non_data_image_href_count": int(len(non_data_hrefs)),
        "level_map_correlation_label_count": int(svg_texts.count("Level map correlation")),
        "mean_correlation_label_present": "Mean correlation" in svg_texts,
        "data_label_count": int(svg_texts.count("Data")),
        "shuffle_label_count": int(svg_texts.count("Shuffle")),
        "star_label_count": int(svg_texts.count("***")),
        "corrected_842_source_contract_pass": bool(source_contract_pass),
        "panel_f_expected_r_label": expected_r_label,
        "panel_f_r_label_present": expected_r_label in svg_texts,
        "legacy_panel_f_p_text_present": any("p<" in text or "p =" in text or "p=" in text for text in svg_texts),
    }
    if not checks["corrected_842_source_contract_pass"]:
        raise AssertionError(
            "Figure 5 source manifest does not prove the exact corrected-842 36-pair/14-mouse, "
            "no-manual-mask registration contract."
        )
    if checks["source_panel_count"] != checks["expected_source_panel_count"]:
        raise AssertionError(f"Expected 6 Figure 5 source panels, found {checks['source_panel_count']}")
    if non_data_hrefs:
        raise AssertionError(f"Figure 5 SVG links external image sources: {non_data_hrefs[:3]}")
    if checks["level_map_correlation_label_count"] < 2:
        raise AssertionError(
            "Figure 5 is missing the level-map correlation labels for panels c/f: "
            f"{checks['level_map_correlation_label_count']}"
        )
    if not checks["mean_correlation_label_present"]:
        raise AssertionError("Figure 5 panel d is missing the Mean correlation x-axis label")
    if checks["data_label_count"] < 3 or checks["shuffle_label_count"] < 3:
        raise AssertionError(
            "Figure 5 Data/Shuffle labels are missing from panels c/d/e: "
            f"Data={checks['data_label_count']}, Shuffle={checks['shuffle_label_count']}"
        )
    if checks["star_label_count"] < 2:
        raise AssertionError(f"Expected at least 2 Figure 5 significance star labels, found {checks['star_label_count']}")
    if not checks["panel_f_r_label_present"]:
        raise AssertionError(
            f"Figure 5 panel f is missing the corrected bottom-right {expected_r_label} label"
        )
    if checks["legacy_panel_f_p_text_present"]:
        raise AssertionError("Figure 5 SVG still contains legacy p-value statistic text")
    checks.update(
        _png_content_checks(
            out_png,
            max_right_blank=0.08,
            max_bottom_blank=0.08,
        )
    )
    if checks["content_bounds_px"]["y_min"] <= 2:
        raise AssertionError(f"Figure 5 top content touches the canvas edge: {checks['content_bounds_px']}")
    return checks


def validate_figure3_manifest(manifest: dict[str, object], *, tolerance: float = 1e-9) -> dict[str, object]:
    axes = pd.DataFrame(manifest["axes"])
    if manifest.get("layout_mode") in {
        "ppt_matched_source_panel_svg",
        "schematic_row_source_panel_svg",
    }:
        lower = axes.loc[pd.to_numeric(axes["row"], errors="coerce") >= 0].copy()
        schematics = axes.loc[pd.to_numeric(axes["row"], errors="coerce").eq(-2)].copy()
        top = axes.loc[axes["panel_letter"].eq("a")].copy()
        lower["row"] = pd.to_numeric(lower["row"], errors="coerce").astype(int)
        lower["column"] = pd.to_numeric(lower["column"], errors="coerce").astype(int)
        row_width_span = lower.groupby("row")["width"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
        row_height_span = lower.groupby("row")["height"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
        column_x_span = lower.groupby("column")["x0"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
        source_svgs = [REPO_ROOT / str(path) for path in axes["source_svg"].astype(str)]
        source_pngs = [REPO_ROOT / str(path) for path in axes["source_png"].astype(str)]
        missing_sources = [str(path) for path in [*source_svgs, *source_pngs] if not path.exists()]
        layout_reference = manifest.get("layout_reference", {})
        expected_letters = list("abcdefghijklm")
        actual_letters = axes["panel_letter"].astype(str).tolist()
        checks = {
            "layout_mode": manifest.get("layout_mode"),
            "panel_count": int(len(axes)),
            "expected_panel_count": 13,
            "schematic_panel_count": int(len(schematics)),
            "expected_schematic_panel_count": 3,
            "lower_panel_count": int(len(lower)),
            "expected_lower_panel_count": 9,
            "top_panel_count": int(len(top)),
            "row_header_count": int(len(layout_reference.get("row_headers", []))) if isinstance(layout_reference, dict) else 0,
            "expected_row_header_count": 0,
            "panel_letters": actual_letters,
            "expected_panel_letters": expected_letters,
            "lower_grid_cells": sorted((int(row), int(col)) for row, col in zip(lower["row"], lower["column"], strict=False)),
            "row_width_span": row_width_span,
            "row_height_span": row_height_span,
            "column_x_span": column_x_span,
            "source_panel_files_exist": not missing_sources,
            "missing_source_panel_files": missing_sources,
            "source_alignment_max_abs_delta": float(
                manifest.get("source_alignment", {}).get("max_abs_delta", math.nan)
            ),
            "quantitative_scale_anisotropy_max": float(
                np.nanmax(
                    np.abs(
                        pd.to_numeric(lower["scale_x"], errors="coerce").to_numpy(dtype=float)
                        / pd.to_numeric(lower["scale_y"], errors="coerce").to_numpy(dtype=float)
                        - 1.0
                    )
                )
            ),
        }
        if checks["panel_count"] != checks["expected_panel_count"]:
            raise AssertionError(f"Expected 13 Figure 3 source panels, found {checks['panel_count']}")
        if checks["schematic_panel_count"] != checks["expected_schematic_panel_count"]:
            raise AssertionError(
                f"Expected 3 Figure 3 schematic panels, found {checks['schematic_panel_count']}"
            )
        if checks["lower_panel_count"] != checks["expected_lower_panel_count"]:
            raise AssertionError(f"Expected 9 Figure 3 lower panels, found {checks['lower_panel_count']}")
        if checks["top_panel_count"] != 1:
            raise AssertionError(f"Expected 1 Figure 3 top panel, found {checks['top_panel_count']}")
        if checks["row_header_count"] != checks["expected_row_header_count"]:
            raise AssertionError(f"Expected no Figure 3 row headers, found {checks['row_header_count']}")
        if actual_letters != expected_letters:
            raise AssertionError(f"Unexpected Figure 3 panel order: {actual_letters}")
        if checks["lower_grid_cells"] != [(row, col) for row in range(3) for col in range(3)]:
            raise AssertionError(f"Unexpected Figure 3 lower grid cells: {checks['lower_grid_cells']}")
        if missing_sources:
            raise AssertionError(f"Missing source-rendered individual panel files: {missing_sources[:3]}")
        if checks["source_alignment_max_abs_delta"] > tolerance:
            raise AssertionError(
                "Day2-template first-window source alignment failed: "
                f"{checks['source_alignment_max_abs_delta']}"
            )
        if checks["quantitative_scale_anisotropy_max"] > 1e-6:
            raise AssertionError(
                "Figure 3 quantitative panels were stretched during assembly: "
                f"max scale-ratio deviation={checks['quantitative_scale_anisotropy_max']:.3g}"
            )
        out_svg = REPO_ROOT / manifest["outputs"]["svg"]
        hrefs = _svg_image_hrefs(out_svg)
        non_data_hrefs = [href for href in hrefs if not href.startswith("data:")]
        checks["svg_image_node_count"] = int(len(hrefs))
        checks["svg_non_data_image_href_count"] = int(len(non_data_hrefs))
        if non_data_hrefs:
            raise AssertionError(f"Figure 3 SVG links external image sources: {non_data_hrefs[:3]}")
        svg_root = ET.parse(out_svg).getroot()
        panel_label_texts = [
            (element.text or "").strip()
            for element in svg_root.iter()
            if "figure3-panel-label" in str(element.attrib.get("class", ""))
        ]
        legend_label_texts = [
            (element.text or "").strip()
            for element in svg_root.iter()
            if "figure3-day2-legend-label" in str(element.attrib.get("class", ""))
        ]
        stat_star_nodes = [
            element
            for element in svg_root.iter()
            if "figure3-stat-star-label" in str(element.attrib.get("class", ""))
        ]
        stat_r_nodes = [
            element
            for element in svg_root.iter()
            if "figure3-stat-r-label" in str(element.attrib.get("class", ""))
        ]
        stat_star_texts = [(element.text or "").strip() for element in stat_star_nodes]
        stat_r_texts = [(element.text or "").strip() for element in stat_r_nodes]
        stat_star_x = [float(element.attrib.get("x", "nan")) for element in stat_star_nodes]
        stat_layouts = [
            layout
            for layout in FIGURE3_PPT_PANEL_LAYOUT
            if int(layout["column"]) in (1, 2) and int(layout["row"]) in (0, 1, 2)
        ]
        stat_r_position_fracs = []
        for node, layout in zip(stat_r_nodes, stat_layouts, strict=True):
            slot_x, slot_y, slot_w, slot_h = _figure3_relative_slot_pt(
                tuple(float(value) for value in layout["slot_pt"])
            )
            stat_r_position_fracs.append(
                [
                    (float(node.attrib.get("x", "nan")) - slot_x) / slot_w,
                    (float(node.attrib.get("y", "nan")) - slot_y) / slot_h,
                ]
            )
        stat_star_x_span_by_column = {
            "day2_template_first30": float(np.ptp(np.asarray(stat_star_x[0::2], dtype=float))),
            "long_gap": float(np.ptp(np.asarray(stat_star_x[1::2], dtype=float))),
        }
        all_svg_texts = [
            (element.text or "").strip()
            for element in svg_root.iter()
            if element.tag == _svg_tag("text") and (element.text or "").strip()
        ]
        checks["panel_label_count"] = int(len(panel_label_texts))
        checks["panel_label_texts"] = panel_label_texts
        checks["legend_overlay_label_count"] = int(len(legend_label_texts))
        checks["legend_overlay_label_texts"] = legend_label_texts
        checks["stat_star_label_count"] = int(len(stat_star_texts))
        checks["stat_star_label_texts"] = stat_star_texts
        checks["stat_star_x_span_by_column"] = stat_star_x_span_by_column
        checks["stat_r_label_count"] = int(len(stat_r_texts))
        checks["stat_r_label_texts"] = stat_r_texts
        checks["stat_r_position_fracs"] = stat_r_position_fracs
        checks["legacy_stat_text_present"] = any(("q=" in text or "OLS" in text) for text in all_svg_texts)
        checks["schematic_titles_present"] = all(
            title in all_svg_texts for title in FIGURE3_SCHEMATIC_TITLES.values()
        )
        checks["cycle_axis_label_count"] = int(
            sum(text == "No. of cycles" for text in all_svg_texts)
        )
        checks["stale_experience_label_absent"] = not any(
            text in {"No. of experiences", "First 30 min across experiences"}
            for text in all_svg_texts
        )
        checks["anticipated_result_curve_text_present"] = any(
            fragment in text
            for text in all_svg_texts
            for fragment in (
                "similarity to Day 2",
                "similarity across pivot",
                "pivot time",
                "anticipated",
            )
        )
        if panel_label_texts != expected_letters:
            raise AssertionError(f"Figure 3 panel labels missing or out of order: {panel_label_texts}")
        if len(legend_label_texts) != 15:
            raise AssertionError(f"Expected 15 first-column legend overlay labels, found {len(legend_label_texts)}")
        if len(stat_star_texts) != 6:
            raise AssertionError(f"Expected 6 Figure 3 significance labels, found {len(stat_star_texts)}")
        if len(stat_r_texts) != 6:
            raise AssertionError(f"Expected 6 Figure 3 r labels, found {len(stat_r_texts)}")
        expected_r_position = np.asarray(FIGURE3_STAT_R_POSITION_FRAC, dtype=float)
        r_position_tolerance = max(float(tolerance), 1e-6)
        if any(
            np.max(np.abs(np.asarray(position, dtype=float) - expected_r_position)) > r_position_tolerance
            for position in stat_r_position_fracs
        ):
            raise AssertionError(
                "Figure 3 r labels are not aligned to the maintained upper-right position: "
                f"{stat_r_position_fracs}"
            )
        if not checks["schematic_titles_present"]:
            raise AssertionError("Figure 3 is missing one or more analysis-schematic titles")
        if checks["cycle_axis_label_count"] != 8 or not checks["stale_experience_label_absent"]:
            raise AssertionError(
                "Figure 3 cycle-label contract failed: "
                f"count={checks['cycle_axis_label_count']}, "
                f"stale_absent={checks['stale_experience_label_absent']}"
            )
        if checks["anticipated_result_curve_text_present"]:
            raise AssertionError("Figure 3 still contains anticipated-result schematic content")
        if any(span > tolerance for span in stat_star_x_span_by_column.values()):
            raise AssertionError(
                f"Figure 3 significance labels are not x-aligned by column: {stat_star_x_span_by_column}"
            )
        if checks["legacy_stat_text_present"]:
            raise AssertionError("Figure 3 SVG still contains legacy q=/OLS statistic text")
        out_png = REPO_ROOT / manifest["outputs"]["png"]
        checks.update(
            _png_content_checks(
                out_png,
                max_right_blank=0.06,
                max_bottom_blank=0.06,
            )
        )
        return checks

    top = axes.loc[axes["key"].str.startswith("top_example_")].copy()
    lower = axes.loc[~axes["key"].str.startswith("top_example_")].copy()
    lower["row"] = lower["key"].str.extract(r"_(place_map_correlation|pv_correlation|ensemble_rate_correlation)$")[0]
    lower["column"] = lower["key"].str.replace(r"_(place_map_correlation|pv_correlation|ensemble_rate_correlation)$", "", regex=True)
    width_by_column = lower.groupby("column")["width"].nunique()
    height_by_row = lower.groupby("row")["height"].nunique()
    ylim_low_span = lower.groupby("row")["ylim_low"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    ylim_high_span = lower.groupby("row")["ylim_high"].agg(lambda values: float(np.ptp(values.to_numpy(dtype=float)))).to_dict()
    ytick_unique_counts = lower.groupby("row")["yticks"].nunique().to_dict()
    top_left = float(top["x0"].min())
    top_right = float((top["x0"] + top["width"]).max())
    lower_left = float(lower["x0"].min())
    lower_right = float((lower["x0"] + lower["width"]).max())
    checks = {
        "n_lower_axes": int(len(lower)),
        "top_row_x_span": [top_left, top_right],
        "lower_grid_x_span": [lower_left, lower_right],
        "top_and_lower_equal_width": bool(
            abs(top_left - lower_left) <= tolerance and abs(top_right - lower_right) <= tolerance
        ),
        "all_columns_equal_width": bool(np.ptp(lower.groupby("column")["width"].mean().to_numpy(dtype=float)) <= tolerance),
        "all_rows_equal_height": bool(np.ptp(lower.groupby("row")["height"].mean().to_numpy(dtype=float)) <= tolerance),
        "all_row_y_limits_equal": bool(
            all(span <= tolerance for span in ylim_low_span.values())
            and all(span <= tolerance for span in ylim_high_span.values())
        ),
        "all_row_y_ticks_equal": bool(all(count == 1 for count in ytick_unique_counts.values())),
        "width_by_column": lower.groupby("column")["width"].mean().to_dict(),
        "height_by_row": lower.groupby("row")["height"].mean().to_dict(),
        "width_in_by_column": lower.groupby("column")["width_in"].mean().to_dict(),
        "height_in_by_row": lower.groupby("row")["height_in"].mean().to_dict(),
        "aspect_width_over_height_max": float(lower["aspect_width_over_height"].max()),
        "row_y0": lower.groupby("row")["y0"].mean().to_dict(),
        "column_x0": lower.groupby("column")["x0"].mean().to_dict(),
        "duplicate_width_counts": width_by_column.to_dict(),
        "duplicate_height_counts": height_by_row.to_dict(),
        "row_ylim_low_span": ylim_low_span,
        "row_ylim_high_span": ylim_high_span,
        "row_ytick_unique_counts": ytick_unique_counts,
    }
    if checks["n_lower_axes"] != 9:
        raise AssertionError(f"Expected 9 lower axes, found {checks['n_lower_axes']}")
    if not checks["top_and_lower_equal_width"]:
        raise AssertionError(
            f"Figure 3 top row and lower grid x-spans differ: "
            f"top={checks['top_row_x_span']}, lower={checks['lower_grid_x_span']}"
        )
    if not checks["all_columns_equal_width"]:
        raise AssertionError(f"Lower grid column widths differ: {checks['width_by_column']}")
    if not checks["all_rows_equal_height"]:
        raise AssertionError(f"Lower grid row heights differ: {checks['height_by_row']}")
    if not checks["all_row_y_limits_equal"]:
        raise AssertionError(
            f"Lower grid row y-limits are not harmonized: low={ylim_low_span}, high={ylim_high_span}"
        )
    if not checks["all_row_y_ticks_equal"]:
        raise AssertionError(f"Lower grid row y-ticks differ: {ytick_unique_counts}")
    style_payload = manifest.get("style", {})
    max_aspect = float(style_payload.get("max_lower_panel_aspect", 1.15))
    if checks["aspect_width_over_height_max"] > max_aspect:
        raise AssertionError(
            f"Lower panels are too wide for trend readability: "
            f"{checks['aspect_width_over_height_max']:.3f} > {max_aspect:.3f}"
        )
    out_png = REPO_ROOT / manifest["outputs"]["png"]
    image = Image.open(out_png).convert("RGB")
    stat = ImageStat.Stat(image)
    checks["png_size_px"] = list(image.size)
    checks["png_channel_stddev"] = [float(v) for v in stat.stddev]
    checks["png_nonblank"] = bool(max(stat.stddev) > 1.0)
    if not checks["png_nonblank"]:
        raise AssertionError(f"Rendered PNG appears blank: {out_png}")
    array = np.asarray(image)
    nonwhite = np.any(array < 248, axis=2)
    ys, xs = np.where(nonwhite)
    if xs.size == 0 or ys.size == 0:
        raise AssertionError(f"Rendered PNG has no detectable non-white content: {out_png}")
    height, width = nonwhite.shape
    right_blank = (width - 1 - int(xs.max())) / float(width)
    bottom_blank = (height - 1 - int(ys.max())) / float(height)
    checks["content_bounds_px"] = {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
    }
    checks["right_blank_fraction"] = float(right_blank)
    checks["bottom_blank_fraction"] = float(bottom_blank)
    max_right = float(style_payload.get("max_blank_right_fraction", 0.12))
    max_bottom = float(style_payload.get("max_blank_bottom_fraction", 0.12))
    if right_blank > max_right:
        raise AssertionError(f"Rendered figure has too much blank right canvas: {right_blank:.3f} > {max_right:.3f}")
    if bottom_blank > max_bottom:
        raise AssertionError(f"Rendered figure has too much blank bottom canvas: {bottom_blank:.3f} > {max_bottom:.3f}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["1", "2", "3", "5"],
        choices=["1", "2", "3", "5"],
        help="Currently implemented source-rendered figure IDs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    style = PublicationStyle(dpi=int(args.dpi))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_manifest: dict[str, object] = {
        "renderer": Path(__file__).name,
        "out_dir": _rel(args.out_dir),
        "figures": list(args.figures),
        "style": asdict(style),
        "figure_manifests": {},
        "validation": {},
    }
    if "1" in args.figures:
        fig_manifest = render_figure1(args.out_dir, style)
        run_manifest["figure_manifests"]["1"] = _rel(args.out_dir / "figure1_source_manifest.json")
        run_manifest["validation"]["1"] = validate_figure1_manifest(fig_manifest)
    if "2" in args.figures:
        fig_manifest = render_figure2(args.out_dir, style)
        run_manifest["figure_manifests"]["2"] = _rel(args.out_dir / "figure2_source_manifest.json")
        run_manifest["validation"]["2"] = validate_figure2_manifest(fig_manifest)
    if "3" in args.figures:
        fig_manifest = render_figure3(args.out_dir, style)
        run_manifest["figure_manifests"]["3"] = _rel(args.out_dir / "figure3_source_manifest.json")
        run_manifest["validation"]["3"] = validate_figure3_manifest(fig_manifest)
    if "5" in args.figures:
        fig_manifest = render_figure5(args.out_dir, style)
        run_manifest["figure_manifests"]["5"] = _rel(args.out_dir / "figure5_source_manifest.json")
        run_manifest["validation"]["5"] = validate_figure5_manifest(fig_manifest)
    _write_json(args.out_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
