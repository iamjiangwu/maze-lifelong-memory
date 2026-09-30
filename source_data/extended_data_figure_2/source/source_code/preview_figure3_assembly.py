from __future__ import annotations

import argparse
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis.compat import bootstrap_environment
import ca_analysis.figure_panels as figure_panels_module
from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    DEFAULT_PANEL_OUTPUT_DIR,
    PANEL_LAYOUT,
    apply_panel_rcparams,
    get_level_assignment_schematic_colors,
    get_level_assignment_schematic_geometry,
    get_level_assignment_schematic_layout,
    get_level_assignment_schematic_style,
    get_level_assignment_schematic_text,
    render_correlation_shuffle_distribution_panel,
    render_level_assignment_schematic,
    render_level_code_heatmap_panel,
    render_level_exploration_panel,
    render_paired_bar_scatter_panel,
    render_shuffle_mean_distribution_panel,
)

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


_COMPAT = bootstrap_environment()
REPO_ROOT = Path(_COMPAT.repo_root)

DEFAULT_OUTPUT_DIR = Path("docs/figure_panel_previews")
DEFAULT_BASENAME = "figure3_assembly_preview_current"
SORTED_ROOT = REPO_ROOT / "docs" / "sorted_data_full_figure_rerun_2026-06-11"
MAIN_FIGURE_SOURCE_ROOT = (
    REPO_ROOT
    / "docs"
    / "main_figs_v3_source_render_2026-06-24"
)
FIGURE5_CORRECTED_842_DIR = (
    MAIN_FIGURE_SOURCE_ROOT / "figure5_no_manual_mask_2026-08-07"
)
FIGURE5_SOURCE_MANIFEST = FIGURE5_CORRECTED_842_DIR / "SOURCE_MANIFEST.json"
FIGURE5_LEVEL_DISTRIBUTION_DIR = FIGURE5_CORRECTED_842_DIR
FIGURE5_DECODER_DIR = FIGURE5_CORRECTED_842_DIR
SORTED_LEVEL_DECODER_DIR = SORTED_ROOT / "figure3_decoder_tree_depth7_sorted"
SORTED_LAYOUT_SOURCE_TABLE_DIR = SORTED_ROOT / "layout_panel_exports_sorted_only" / "source_tables"
SORTED_LEVEL_EXPLORATION_DIR = FIGURE5_CORRECTED_842_DIR
# External authority for the level denominator: 21,412 registered pairs minus
# 4,511 activity-gate failures. Read, never typed, so adding a session moves it.
FIGURE5_REGISTRATION_SUMMARY = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "extended_data_figures"
    / "extended_data_fig_05"
    / "04_source_tables"
    / "registration_quality_summary.json"
)
CORRECTED_842_LEVEL_HEATMAP_ROOT = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "rerun_current"
    / "842_0to3_corrected"
    / "level_sorted_tree_depth7"
    / "mode_level"
)
CORRECTED_842_LEVEL_HEATMAP_PKL = (
    CORRECTED_842_LEVEL_HEATMAP_ROOT / "tuning_842day1_to_842-3day1.pkl"
)
CORRECTED_842_CIMAP = (
    MAIN_FIGURE_SOURCE_ROOT
    / "registration_quality_exact_36_2026-07-09"
    / "corrected_cimaps"
    / "842_cell_to_index_map_03_zero_reinserted.mat"
)
CORRECTED_PAIR = "842day1->842-3day1"
SHORT_PAIR = "1616day1->1616bday1"
EXPECTED_PAIR_SET_SHA256 = "50ba0725039671ce86f0964d09c247e681199d3e8c29aad4cd8383ad33d85bc9"
EXPECTED_CORRECTED_842_TUNING_SHA256 = "54ef9c4f964d34fe8f4332b7c96b23ff519581aab8dac09a86fbfd32da0acee4"
EXPECTED_CORRECTED_842_CIMAP_SHA256 = "8f4db74927d1c8776497f110ec47b1d8db245aebb1ca1df38d63bf6fa9b29a22"
EXPECTED_CORRECTED_842_LEVEL_VALUE = 0.0555420654353495
EXPECTED_CORRECTED_842_DECODER_VALUE = 0.2048227460860176
EXPECTED_HEATMAP_PAIR_ORDER_SHA256 = "6cce62531ba2a3456b8a737b1e27f9cb516ec32b70e546ec9991714eda930a47"
LEVEL_HEATMAP_BASE_ROOTS = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "rerun_current"
    / "manual_exclusion_lifted_2026-08-07",
    *(REPO_ROOT / "docs" / f"sorted_neuron_{animal}_correlation_comparison_2026-06-11" for animal in ("418", "422", "434", "451", "468", "488", "489", "579", "837", "838", "839")),
    CORRECTED_842_LEVEL_HEATMAP_ROOT.parent.parent,
    REPO_ROOT / "docs" / "sorted_neuron_842_correlation_comparison_2026-06-11",
    REPO_ROOT / "docs" / "sorted_neuron_862_correlation_comparison_2026-06-11",
    REPO_ROOT / "docs" / "sorted_neuron_1616_correlation_comparison_2026-06-11",
)
LEVEL_HEATMAP_DATA_ROOTS = tuple(
    root / "level_sorted_tree_depth7" / "mode_level" for root in LEVEL_HEATMAP_BASE_ROOTS
)
FONT_FAMILY = "Arial"
EXPORT_WIDTH_MM = 180.0
CANVAS_SIZE = (8.65, 5.85)
FIRST_ROW_CANVAS_SIZE = (8.65, 2.76)

BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
TITLE_SIZE = 8.0
PANEL_ID_SIZE = 13.0
LEVEL_COLOR = CORRELATION_MEASURE_COLORS["level_map_correlation"]
DATA_RED = "#DA2323"
SHUFFLE_GRAY = "#7f7f7f"
HEATMAP_CMAP = "magma"
DIRECT_LABEL_FONT_SIZE = 6.8


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_set_sha256(values) -> str:
    keys = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


@lru_cache(maxsize=1)
def validate_figure5_corrected842_sources() -> dict[str, object]:
    """Fail closed before any Main Figure 5 or poster panel is rendered."""
    if not FIGURE5_SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing corrected Main Figure 5 source manifest: {FIGURE5_SOURCE_MANIFEST}"
        )
    manifest = json.loads(FIGURE5_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    scope = manifest.get("scope", {})
    correction = manifest.get("correction_record_842_0to3", {})
    displayed_mapping = manifest.get("displayed_panel_mapping", {})
    displayed_heatmap = manifest.get("displayed_panel_b_heatmap", {})
    # Anchor the support total to the registration/activity-gate authority rather
    # than to the bundle's own manifest. Every other support check in this function
    # compares against expected_level_support, so if that came from the bundle the
    # bundle could agree with itself at a wrong value and still pass. Deriving it
    # here makes all of them externally anchored, and it survives adding a session
    # because nothing is a typed literal.
    registration_summary = json.loads(
        FIGURE5_REGISTRATION_SUMMARY.read_text(encoding="utf-8")
    )
    registered = int(registration_summary["n_registered_pairs_before_activity_gate"])
    activity_drops = int(
        registration_summary["activity_gate_drops_using_corrected_cimaps"]
    )
    authority_level_support = int(
        registration_summary["corrected_figure2_metric_observations_after_activity_gate"]
    )
    if registered - activity_drops != authority_level_support:
        raise RuntimeError(
            "Corrected registration/activity-gate denominator does not close: "
            f"{registered} - {activity_drops} != {authority_level_support}"
        )
    expected_level_support = int(scope.get("level_neurons_total", -1))
    if expected_level_support != authority_level_support:
        raise RuntimeError(
            "Main Figure 5 manifest level support disagrees with the registration "
            f"authority: manifest {expected_level_support} vs authority "
            f"{authority_level_support}."
        )
    expected_displayed_mapping = {
        "a": "level-assignment schematic (preview internal a)",
        "b": "level-code heatmap (preview internal e)",
        "c": "level-map correlation distribution (preview internal b)",
        "d": "shuffle-mean distribution (preview internal c)",
        "e": "tree-depth decoder accuracy (preview internal d)",
        "f": "level-map correlation versus exploration efficiency (preview internal f)",
    }
    if (
        manifest.get("status") != "complete"
        or displayed_mapping != expected_displayed_mapping
        or int(displayed_heatmap.get("included_pairs", -1)) != 36
        or int(displayed_heatmap.get("included_animals", -1)) != 14
        or expected_level_support <= 0
        or int(displayed_heatmap.get("level_neurons_total", -1)) != expected_level_support
        or int(displayed_heatmap.get("corrected_842_0to3_neurons", -1)) != 75
        or str(displayed_heatmap.get("pair_order_sha256", ""))
        != EXPECTED_HEATMAP_PAIR_ORDER_SHA256
        or str(displayed_heatmap.get("pair_source_audit", ""))
        != "displayed_panel_b_heatmap_pair_source_audit.csv"
        or int(scope.get("eligible_d1v1_pairs", -1)) != 37
        or int(scope.get("included_pairs", -1)) != 36
        or int(scope.get("included_animals", -1)) != 14
        or str(scope.get("mask_policy", ""))
        != "none; exact corrected registration/activity-gate support"
        or str(scope.get("exact_included_pair_set_sha256", "")).lower()
        != EXPECTED_PAIR_SET_SHA256
        or list(scope.get("excluded_pairs", [])) != [SHORT_PAIR]
        or str(scope.get("retained_long_1616_pair", ""))
        != "1616bday1->1616b-3day1"
    ):
        raise RuntimeError("Corrected Main Figure 5 manifest has the wrong scope or exclusion policy.")
    if (
        str(correction.get("pair_key")) != CORRECTED_PAIR
        or int(correction.get("level_support", -1)) != 75
        or not np.isclose(
            float(correction.get("level_map_correlation", np.nan)),
            EXPECTED_CORRECTED_842_LEVEL_VALUE,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(correction.get("level_decoder_accuracy", np.nan)),
            EXPECTED_CORRECTED_842_DECODER_VALUE,
            rtol=0.0,
            atol=1e-12,
        )
        or str(correction.get("corrected_tuning_sha256", "")).lower()
        != EXPECTED_CORRECTED_842_TUNING_SHA256
        or str(correction.get("corrected_cimap_sha256", "")).lower()
        != EXPECTED_CORRECTED_842_CIMAP_SHA256
    ):
        raise RuntimeError("Corrected Main Figure 5 manifest has an invalid 842 correction record.")
    if _file_sha256(CORRECTED_842_LEVEL_HEATMAP_PKL) != EXPECTED_CORRECTED_842_TUNING_SHA256:
        raise RuntimeError("Main Figure 5 corrected 842 heatmap tuning pickle hash has drifted.")
    if _file_sha256(CORRECTED_842_CIMAP) != EXPECTED_CORRECTED_842_CIMAP_SHA256:
        raise RuntimeError("Main Figure 5 corrected 842 CIMap hash has drifted.")

    required_outputs = {
        "plot_data_cell228_level.npz",
        "plot_data_cell226_level.npz",
        "level_shuffle_pair_summary.csv",
        "cross_session_decoding_accuracy_points.csv",
        "cross_session_decoding_accuracy_summary.csv",
        "cross_session_decoding_accuracy_stats.json",
        "level_map_vs_exploration_efficiency_primary_clean_rows.csv",
        "level_map_vs_exploration_efficiency_primary_clean_stats.json",
        "figure5_pair_scope_audit.csv",
        "panel_e_842_heatmap_source_audit.csv",
        "displayed_panel_b_heatmap_pair_source_audit.csv",
    }
    outputs = manifest.get("output_files", {})
    if not required_outputs.issubset(outputs):
        raise RuntimeError(
            "Corrected Main Figure 5 manifest lacks required panel files: "
            + ", ".join(sorted(required_outputs - set(outputs)))
        )
    for filename, record in outputs.items():
        if str(record.get("filename")) != str(filename):
            raise RuntimeError(f"Main Figure 5 output filename record drifted for {filename}.")
        path = FIGURE5_CORRECTED_842_DIR / filename
        if not path.is_file() or _file_sha256(path) != str(record.get("sha256", "")).lower():
            raise RuntimeError(f"Main Figure 5 output hash mismatch: {path}")

    pair_scope = pd.read_csv(FIGURE5_CORRECTED_842_DIR / "figure5_pair_scope_audit.csv")
    pair_scope["pair_key"] = pair_scope["pair_key"].astype(str)
    if (
        len(pair_scope) != 36
        or pair_scope["animal_id"].astype(str).nunique() != 14
        or _pair_set_sha256(pair_scope["pair_key"]) != EXPECTED_PAIR_SET_SHA256
        or SHORT_PAIR in set(pair_scope["pair_key"])
        or int(pd.to_numeric(pair_scope["n_neurons"], errors="raise").sum()) != expected_level_support
        or not _as_bool(pair_scope["level_pair_present"]).all()
        or not _as_bool(pair_scope["decoder_pair_present"]).all()
        or not _as_bool(pair_scope["exploration_pair_present"]).all()
    ):
        raise RuntimeError("Main Figure 5 pair-scope audit failed exact cross-panel parity.")
    target = pair_scope.loc[pair_scope["pair_key"].eq(CORRECTED_PAIR)]
    if len(target) != 1 or int(target.iloc[0]["n_neurons"]) != 75 or not np.isclose(
        float(target.iloc[0]["observed_pair_mean"]),
        EXPECTED_CORRECTED_842_LEVEL_VALUE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Main Figure 5 pair-scope audit lacks the corrected 842 level row.")

    heatmap_pairs = pd.read_csv(
        FIGURE5_CORRECTED_842_DIR / "displayed_panel_b_heatmap_pair_source_audit.csv"
    ).sort_values("display_order", kind="stable")
    ordered_hash = hashlib.sha256(
        "\n".join(heatmap_pairs["pair_key"].astype(str)).encode("utf-8")
    ).hexdigest()
    if (
        len(heatmap_pairs) != 36
        or ordered_hash != EXPECTED_HEATMAP_PAIR_ORDER_SHA256
        or _pair_set_sha256(heatmap_pairs["pair_key"]) != EXPECTED_PAIR_SET_SHA256
        or int(pd.to_numeric(heatmap_pairs["n_neurons"], errors="raise").sum()) != expected_level_support
        or not heatmap_pairs["status"].astype(str).eq("PASS").all()
    ):
        raise RuntimeError("Displayed Main Figure 5 panel-b heatmap disagrees with registration support.")
    for row in heatmap_pairs.itertuples(index=False):
        src, dst = str(row.pair_key).split("->", 1)
        filename = f"tuning_{src}_to_{dst}.pkl"
        matches = [root / filename for root in LEVEL_HEATMAP_DATA_ROOTS if (root / filename).is_file()]
        selected = matches[0] if matches else None
        audited = Path(str(row.selected_pkl_path))
        if not audited.is_absolute():
            audited = REPO_ROOT / audited
        if (
            selected is None
            or selected.resolve() != audited.resolve()
            or _file_sha256(selected) != str(row.selected_pkl_sha256).lower()
        ):
            raise RuntimeError(f"Displayed Main Figure 5 heatmap source resolution drifted for {row.pair_key}.")

    decoder = pd.read_csv(
        FIGURE5_CORRECTED_842_DIR / "cross_session_decoding_accuracy_points.csv"
    )
    decoder["pair_key"] = decoder["src"].astype(str) + "->" + decoder["dst"].astype(str)
    decoder_target = decoder.loc[
        decoder["pair_key"].eq(CORRECTED_PAIR) & decoder["group"].astype(str).eq("Data")
    ]
    if len(decoder_target) != 1 or not np.isclose(
        float(decoder_target.iloc[0]["decoding_accuracy"]),
        EXPECTED_CORRECTED_842_DECODER_VALUE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Main Figure 5 decoder source lacks the corrected 842 accuracy.")

    exploration = pd.read_csv(
        FIGURE5_CORRECTED_842_DIR
        / "level_map_vs_exploration_efficiency_primary_clean_rows.csv"
    )
    exploration["pair_key"] = (
        exploration["src_session"].astype(str) + "->" + exploration["dst_session"].astype(str)
    )
    exploration = exploration.loc[_as_bool(exploration["included_in_primary"])].copy()
    exploration_target = exploration.loc[exploration["pair_key"].eq(CORRECTED_PAIR)]
    if (
        len(exploration) != 36
        or exploration["animal_id"].astype(str).nunique() != 14
        or _pair_set_sha256(exploration["pair_key"]) != EXPECTED_PAIR_SET_SHA256
        or len(exploration_target) != 1
        or int(exploration_target.iloc[0]["level_map_support"]) != 75
        or not np.isclose(
            float(exploration_target.iloc[0]["level_map_correlation"]),
            EXPECTED_CORRECTED_842_LEVEL_VALUE,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise RuntimeError("Main Figure 5 exploration source has the wrong corrected scope/value.")
    exploration_stats = json.loads(
        (
            FIGURE5_CORRECTED_842_DIR
            / "level_map_vs_exploration_efficiency_primary_clean_stats.json"
        ).read_text(encoding="utf-8")
    )
    # Panel f must sit on the same corrected level authority as panels b/c/d.
    # It previously byte-preserved 35 legacy rows, which silently kept the fit
    # line, the r label and the star on the stale 16,897-neuron support while
    # b/c/d had already moved to the corrected pool. Assert the relation rather
    # than a literal count, so adding a session cannot silently invalidate it.
    panel_cd_support = int(
        pd.read_csv(FIGURE5_CORRECTED_842_DIR / "level_shuffle_pair_summary.csv")[
            "n_neurons"
        ].sum()
    )
    panel_f_support = int(exploration["level_map_support"].sum())
    if (
        int(exploration_stats.get("n_pairs", -1)) != 36
        or int(exploration_stats.get("n_animals", -1)) != 14
        or exploration_stats.get("legacy_estimator_reproduction") != "PASS"
        or panel_f_support != panel_cd_support
        or int(exploration_stats.get("level_support_total", -1)) != panel_cd_support
    ):
        raise RuntimeError(
            "Main Figure 5 exploration statistics failed estimator/support checks "
            f"(panel f support {panel_f_support} vs panels b/c/d {panel_cd_support})."
        )

    heatmap_audit = pd.read_csv(
        FIGURE5_CORRECTED_842_DIR / "panel_e_842_heatmap_source_audit.csv"
    )
    if (
        len(heatmap_audit) != 1
        or str(heatmap_audit.iloc[0]["pair_key"]) != CORRECTED_PAIR
        or int(heatmap_audit.iloc[0]["after_n_neurons"]) != 75
        or str(heatmap_audit.iloc[0]["after_sha256"]).lower()
        != EXPECTED_CORRECTED_842_TUNING_SHA256
        or str(heatmap_audit.iloc[0]["status"]) != "PASS"
    ):
        raise RuntimeError("Main Figure 5 heatmap source audit is incomplete.")
    return {
        "status": "PASS",
        "manifest": str(FIGURE5_SOURCE_MANIFEST),
        "included_pairs": 36,
        "included_animals": 14,
        "excluded_short_pair": SHORT_PAIR,
        "level_neurons_total": expected_level_support,
        "corrected_842_level_support": 75,
        "corrected_842_level_value": EXPECTED_CORRECTED_842_LEVEL_VALUE,
        "corrected_842_decoder_value": EXPECTED_CORRECTED_842_DECODER_VALUE,
        "corrected_842_tuning_sha256": EXPECTED_CORRECTED_842_TUNING_SHA256,
        "corrected_842_cimap_sha256": EXPECTED_CORRECTED_842_CIMAP_SHA256,
        "panel_f_pearson_r": float(exploration_stats["pearson_r"]),
        "panel_f_cluster_ols_p": float(exploration_stats["cluster_ols_p"]),
        "panel_f_r_label": f"r={float(exploration_stats['pearson_r']):.2f}",
        "displayed_panel_mapping": displayed_mapping,
        "displayed_panel_b_heatmap_neurons": expected_level_support,
    }


def _star_from_p(p_value: object) -> str:
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


def _axes_dx_for_chars(panel_size: tuple[float, float], *, left: float, right: float, chars: float) -> float:
    axis_width_pt = panel_size[0] * 72.0 * (right - left)
    return chars * (0.55 * DIRECT_LABEL_FONT_SIZE) / axis_width_pt


def _axes_dy_for_text_heights(panel_size: tuple[float, float], *, bottom: float, top: float, units: float) -> float:
    axis_height_pt = panel_size[1] * 72.0 * (top - bottom)
    return units * DIRECT_LABEL_FONT_SIZE / axis_height_pt


def _add_axes_text_label(ax, x: float, y: float, text: str, *, color: str, va: str = "top") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=DIRECT_LABEL_FONT_SIZE,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=color,
    )


PANEL_LAYOUT_SPEC = {
    "a": {
        "x": 0.03,
        "y": 0.05,
        "w": 3.37,
        "h": 2.58,
        "render_w": 3.37,
        "render_h": 2.58,
        "label": "a",
    },
    "e": {
        "x": 3.44,
        "y": 0.05,
        "w": 5.20,
        "h": 2.58,
        "render_w": 5.20,
        "render_h": 2.58,
        "label": "b",
    },
    "b": {
        "x": 0.03,
        "y": 3.05,
        "w": 2.72,
        "h": 2.62,
        "render_w": 2.72,
        "render_h": 2.62,
        "label": "c",
    },
    "c": {
        "x": 2.86,
        "y": 3.05,
        "w": 1.30,
        "h": 2.62,
        "render_w": 1.30,
        "render_h": 2.62,
        "label": "d",
    },
    "d": {
        "x": 4.27,
        "y": 3.05,
        "w": 1.10,
        "h": 2.62,
        "render_w": 1.10,
        "render_h": 2.62,
        "label": "e",
    },
    "f": {
        "x": 5.48,
        "y": 3.05,
        "w": 3.16,
        "h": 2.62,
        "render_w": 3.16,
        "render_h": 2.62,
        "label": "f",
    },
}

FIRST_ROW_LAYOUT_SPEC = {
    "a": {
        "x": 0.03,
        "y": 0.05,
        "w": 3.37,
        "h": 2.58,
        "render_w": 3.37,
        "render_h": 2.58,
        "label": "a",
    },
    "e": {
        "x": 3.44,
        "y": 0.05,
        "w": 5.20,
        "h": 2.58,
        "render_w": 5.20,
        "render_h": 2.58,
        "label": "b",
    },
}


def _set_common_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
        }
    )


def _standardize_axes_text(fig, *, font_scale: float = 1.0) -> None:
    for ax in fig.axes:
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontsize(TITLE_SIZE * font_scale)
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE * font_scale)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE * font_scale)
        ax.tick_params(labelsize=TICK_LABEL_SIZE * font_scale)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(FONT_FAMILY)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(FONT_FAMILY)
                text.set_fontsize(TICK_LABEL_SIZE * font_scale)


def _apply_rcparams(rcparams: dict, size: tuple[float, float], *, layout_extra: dict | None = None) -> None:
    layout = {**PANEL_LAYOUT, "panel_size": size}
    if layout_extra:
        layout.update(layout_extra)
    apply_panel_rcparams(rcparams, panel_size=size, layout=layout)


def _render_panel_a(size: tuple[float, float]):
    row_scale = float(size[1]) / 4.4
    _set_common_rc()
    layout = get_level_assignment_schematic_layout(
        {
            "panel_size": size,
            "width_ratios": (0.45, 1.00),
            "grid_wspace": 0.02,
        }
    )
    geometry = get_level_assignment_schematic_geometry(
        {
            "tree_depth": 7,
            "level_count": 7,
            "tree_width": 12.55,
            "tree_height": 3.6,
            "tree_vertical_pad": 0.10,
            "root_line_length": 1.15,
            "tree_left_pad": 0.80,
            "tree_right_pad": 0.10,
            "maze_margin": 0.15,
        }
    )
    colors = get_level_assignment_schematic_colors(
        {
            "cmap_name": HEATMAP_CMAP,
            "cell_edge": "none",
            "maze_outline": "black",
        }
    )
    style = get_level_assignment_schematic_style(
        {
            "tree_linewidth": 2.0 * row_scale,
            "root_linewidth": 2.0 * row_scale,
            "cell_linewidth": 0.0,
            "maze_outline_linewidth": 1.55 * row_scale,
            "show_tree_nodes": False,
            "show_leaf_markers": True,
            "leaf_marker_size": 17 * row_scale,
            "label_fontsize": 12 * row_scale,
        }
    )
    text = get_level_assignment_schematic_text(
        {
            "level_label": {
                "position": "below_zero",
                "dx": 0.5,
                "dy": 0.0,
                "gap_below_zero": 5,
                "ha": "center",
                "va": "top",
            },
            "root_label_t": 0.52,
            "root_label_offset": 0.40,
            "edge_label_t": 0.56,
            "edge_label_t_by_level": {4: 0.50, 5: 0.44, 6: 0.36},
            "edge_label_offset": 0.40,
            "leaf_label_dx": 0.0,
            "leaf_label_dy": -0.50,
            "show_leaf_label": False,
        }
    )
    fig, _axes, _data, _paths = render_level_assignment_schematic(
        maze_id=6,
        level_definition="tree_depth7",
        panel_size=size,
        layout=layout,
        geometry=geometry,
        colors=colors,
        style=style,
        text=text,
        save=False,
    )
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.08, top=0.94, wspace=0.02)
    return fig


def _render_panel_b(size: tuple[float, float]):
    validate_figure5_corrected842_sources()
    _set_common_rc()
    fig, _ax, _data, _paths = render_correlation_shuffle_distribution_panel(
        npz_path=FIGURE5_LEVEL_DISTRIBUTION_DIR / "plot_data_cell228_level.npz",
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={"data": LEVEL_COLOR, "shuffle": SHUFFLE_GRAY},
        style={
            "data_alpha": 0.72,
            "shuffle_alpha": 0.40,
            "legend": False,
            "show_counts": False,
            "direct_labels": False,
            "direct_label_alpha": 0.98,
            "direct_label_fontsize": 7.2,
            "direct_label_fontweight": "bold",
            "direct_label_transform": "axes",
            "xlim": (-1.0, 1.0),
        },
        labels={"xlabel": "Level map correlation", "ylabel": "Density", "data": "Data", "shuffle": "Shuffle"},
        save=False,
    )
    panel_c_shuffle_y = 0.90 - _axes_dy_for_text_heights(size, bottom=0.24, top=0.94, units=3)
    panel_c_data_x = 0.52 + _axes_dx_for_chars(size, left=0.16, right=0.965, chars=4)
    _add_axes_text_label(_ax, 0.30, panel_c_shuffle_y, "Shuffle", color=SHUFFLE_GRAY)
    _add_axes_text_label(_ax, panel_c_data_x, panel_c_shuffle_y, "Data", color=LEVEL_COLOR)
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.24, top=0.94)
    return fig


def _render_panel_c(size: tuple[float, float]):
    validate_figure5_corrected842_sources()
    _set_common_rc()
    fig, _ax, _data, _paths = render_shuffle_mean_distribution_panel(
        npz_path=FIGURE5_LEVEL_DISTRIBUTION_DIR / "plot_data_cell226_level.npz",
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={"shuffle": SHUFFLE_GRAY, "shuffle_edge": "#555555", "data": LEVEL_COLOR},
        style={
            "bins": 80,
            "hist_alpha": 0.75,
            "line_width": 1.2,
            "line_style": "--",
            "legend": False,
            "show_counts": False,
            "show_pvalue": False,
            "direct_labels": False,
            "direct_label_alpha": 0.98,
            "direct_label_fontsize": 6.6,
            "direct_label_fontweight": "bold",
            "direct_label_transform": "axes",
            "xlim": (0.0, 0.30),
        },
        labels={
            "xlabel": "Mean correlation",
            "ylabel": "Density",
            "shuffle": "Shuffle",
            "data": "Data",
        },
        save=False,
    )
    panel_d_label_dx = _axes_dx_for_chars(size, left=0.32, right=0.95, chars=1)
    _add_axes_text_label(_ax, 0.13 + panel_d_label_dx, 0.78, "Shuffle", color=SHUFFLE_GRAY)
    _add_axes_text_label(_ax, 0.58 - panel_d_label_dx, 0.52, "Data", color=LEVEL_COLOR, va="center")
    _standardize_axes_text(fig, font_scale=0.92)
    fig.subplots_adjust(left=0.32, right=0.95, bottom=0.24, top=0.94)
    return fig


def _render_panel_d(size: tuple[float, float]):
    validate_figure5_corrected842_sources()
    _set_common_rc()
    fig, _ax, _data, _paths = render_paired_bar_scatter_panel(
        points_csv=FIGURE5_DECODER_DIR / "cross_session_decoding_accuracy_points.csv",
        summary_csv=FIGURE5_DECODER_DIR / "cross_session_decoding_accuracy_summary.csv",
        stats_json=FIGURE5_DECODER_DIR / "cross_session_decoding_accuracy_stats.json",
        value_col="decoding_accuracy",
        group_order=("Data", "Shuffle"),
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={
            "Data": LEVEL_COLOR,
            "Shuffle": SHUFFLE_GRAY,
            "pair_line": "#8a8a8a",
            "bar_edge": "none",
            "point_edge": "none",
        },
        style={
            "bar_width": 0.66,
            "bar_alpha": 0.78,
            "point_size": 13,
            "point_alpha": 0.65,
            "point_jitter": 0.03,
            "pair_line_alpha": 0.28,
            "pair_line_width": 0.65,
            "annotation": True,
            "annotation_p_path": ("cluster_ols", "p_greater"),
            "annotation_y": 0.585,
            "annotation_fontsize": 8.0,
            "ylim": (0, 0.66),
            "yticks": [0.0, 0.2, 0.4],
        },
        labels={"ylabel": "Decoding Accuracy", "group_labels": {"Data": "Data", "Shuffle": "Shuffle"}},
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.22, top=0.94)
    return fig


def _render_panel_e(size: tuple[float, float]):
    validate_figure5_corrected842_sources()
    _set_common_rc()
    # First-match-wins: corrected 842 0->3 precedes the historical 842 root,
    # which remains available for the unchanged 3->6 and 6->9 comparisons.
    data_roots = [str(path) for path in LEVEL_HEATMAP_DATA_ROOTS]
    pair_groups = {
        "once": [
            ("418RRday1", "418-3day1"),
            ("422RRday1", "422-3day1"),
            ("434RRday1", "434-3day1"),
            ("451Rday1", "451-3day1"),
            ("468Lday1", "468-3day1"),
            ("488Rday1", "488-3day1"),
            ("489Lday1", "489-3day1"),
            ("579RLday1", "579-3day1"),
            ("837day1", "837-3day1"),
            ("838day1", "838-3day1"),
            ("839day1", "839-3day1"),
            ("842day1", "842-3day1"),
            ("862day1", "862-3day1"),
        ],
        "twice": [
            ("1616bday1", "1616b-3day1"),
            ("418-3day1", "418-6day1"),
            ("422-3day1", "422-6day1"),
            ("451-3day1", "451-6day1"),
            ("468-3day1", "468-6day1"),
            ("488-3day1", "488-6day1"),
            ("489-3day1", "489-6day1"),
            ("837-3day1", "837-6day1"),
            ("838-3day1", "838-6day1"),
            ("839-3day1", "839-6day1"),
            ("842-3day1", "842-6day1"),
            ("862-3day1", "862-6day1"),
        ],
        "three times": [
            ("418-6day1", "418-9day1"),
            ("451-6day1", "451-9day1"),
            ("468-6day1", "468-9day1"),
            ("837-6day1", "837-9day1"),
            ("838-6day1", "838-9day1"),
            ("839-6day1", "839-9day1"),
            ("842-6day1", "842-9day1"),
        ],
        "four times": [
            ("468-9day1", "468-12day1"),
            ("837-9day1", "837-12day1"),
            ("838-9day1", "838-12day1"),
            ("839-9day1", "839-12day1"),
        ],
    }
    pair_order = [pair for group_pairs in pair_groups.values() for pair in group_pairs]
    fig, axes, _data, _paths = render_level_code_heatmap_panel(
        data_roots=data_roots,
        base_groups=pair_groups,
        variant="d1v1",
        pair_order=pair_order,
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size, "grid_wspace": 0.06},
        colors={"heatmap": HEATMAP_CMAP},
        style={
            "normalize_mode": "l2",
            "ordering_method": "hierarchical",
            "lam_index": 1.0,
            "lam_sim": 1.0,
            "refine_eps": 1e-6,
            "tick_labelsize": 7.0,
            "label_fontsize": 9.0,
            "title_fontsize": 8.0,
            "hide_right_yticklabels": True,
            "shared_color_scale": True,
        },
        labels={
            "xlabel": "Levels",
            "ylabel": "Neuron",
            "left_title": "Day i",
            "right_title": "Day i+90",
        },
        xticks=list(range(7)),
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.08, top=0.94, wspace=0.06)
    colorbar = fig.colorbar(
        axes[1].images[0],
        ax=list(axes),
        location="right",
        fraction=0.028,
        pad=0.018,
    )
    colorbar.set_label("Normalized rate", fontsize=7.0, fontfamily=FONT_FAMILY)
    colorbar.ax.tick_params(labelsize=6.5, length=2.0)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
    return fig


def _render_panel_f(size: tuple[float, float]):
    validate_figure5_corrected842_sources()
    _set_common_rc()
    base_size = (3.16, 2.62)
    style_scale = float(size[1]) / base_size[1]
    rows = pd.read_csv(SORTED_LEVEL_EXPLORATION_DIR / "level_map_vs_exploration_efficiency_primary_clean_rows.csv")
    rows = rows.loc[rows["included_in_primary"].astype(bool)].copy()
    stats = json.loads(
        (SORTED_LEVEL_EXPLORATION_DIR / "level_map_vs_exploration_efficiency_primary_clean_stats.json").read_text(
            encoding="utf-8"
        )
    )
    x = pd.to_numeric(rows["exploration_efficiency"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["level_map_correlation"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(x, y, s=18 * style_scale * style_scale, color=LEVEL_COLOR, alpha=0.75, edgecolors="none", zorder=2)
    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
    slope = float(stats["ols_slope"])
    intercept = float(stats["ols_intercept"])
    yfit = intercept + slope * xfit
    ax.plot(xfit, yfit, color=LEVEL_COLOR, linestyle="-", linewidth=1.1 * style_scale, zorder=3)
    if x.size > 2:
        resid = y - (intercept + slope * x)
        s_err = np.sqrt(np.sum(resid**2) / max(x.size - 2, 1))
        x_mean = float(np.mean(x))
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx > 0:
            se = s_err * np.sqrt(1.0 / x.size + (xfit - x_mean) ** 2 / sxx)
            ci = 1.96 * se
            ax.fill_between(xfit, yfit - ci, yfit + ci, color=LEVEL_COLOR, alpha=0.18, linewidth=0, zorder=1)
    ax.text(
        0.08,
        0.92,
        _star_from_p(stats.get("cluster_ols_p", stats.get("pearson_p", np.nan))),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0 * style_scale,
        fontweight="bold",
        color=LEVEL_COLOR,
    )
    ax.text(
        0.94,
        0.08,
        f"r={float(stats['pearson_r']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2 * style_scale,
        color="black",
    )
    ax.set_xlabel("Exploration efficiency")
    ax.set_ylabel("Level map correlation")
    ax.set_position([0.18, 0.22, 0.77, 0.72])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75 * style_scale)
    ax.tick_params(width=0.75 * style_scale, length=3.5 * style_scale, pad=3.5 * style_scale)
    _standardize_axes_text(fig, font_scale=style_scale)
    return fig


PANEL_RENDERERS = {
    "a": _render_panel_a,
    "b": _render_panel_b,
    "c": _render_panel_c,
    "d": _render_panel_d,
    "e": _render_panel_e,
    "f": _render_panel_f,
}


def _svg_tag(tag: str) -> str:
    return f"{{http://www.w3.org/2000/svg}}{tag}"


def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float]:
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        _x0, _y0, width, height = map(float, viewbox.split())
        return width, height
    return float(root.attrib["width"].rstrip("pt")), float(root.attrib["height"].rstrip("pt"))


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map = {}
    for element in root.iter():
        old_id = element.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            element.attrib["id"] = new_id
    if not id_map:
        return
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for old_id, new_id in id_map.items():
                value = value.replace(f"#{old_id}", f"#{new_id}")
            element.attrib[key] = value


def _render_panel_svg(panel_id: str, size: tuple[float, float], *, dpi: int, bbox: str = "standard") -> dict:
    fig = PANEL_RENDERERS[panel_id](size)
    fig.set_dpi(dpi)
    buffer = io.StringIO()
    bbox_inches = "tight" if bbox == "tight" else None
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches=bbox_inches,
            pad_inches=0,
            transparent=True,
            metadata={"Date": None},
        )
    plt.close(fig)
    root = ET.fromstring(buffer.getvalue())
    _prefix_svg_ids(root, f"{panel_id}_")
    width, height = _parse_svg_viewbox(root)
    children = [child for child in list(root) if child.tag != _svg_tag("metadata")]
    return {"children": children, "viewbox_width": width, "viewbox_height": height}


def render_assembly_svg(*, dpi: int = 300, row: str = "all") -> ET.ElementTree:
    panel_spec = FIRST_ROW_LAYOUT_SPEC if row == "first" else PANEL_LAYOUT_SPEC
    canvas_size = FIRST_ROW_CANVAS_SIZE if row == "first" else CANVAS_SIZE
    figure_width = canvas_size[0] * 72.0
    figure_height = canvas_size[1] * 72.0
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{EXPORT_WIDTH_MM:.3f}mm",
            "height": f"{(EXPORT_WIDTH_MM * canvas_size[1] / canvas_size[0]):.3f}mm",
            "viewBox": f"0 0 {figure_width:.6f} {figure_height:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
    for panel_id, panel in panel_spec.items():
        size = (float(panel["render_w"]), float(panel["render_h"]))
        panel_svg = _render_panel_svg(panel_id, size, dpi=dpi, bbox=str(panel.get("bbox", "standard")))
        slot = {
            "x": float(panel["x"]) * 72.0,
            "y": float(panel["y"]) * 72.0,
            "width": float(panel["w"]) * 72.0,
            "height": float(panel["h"]) * 72.0,
        }
        scale = min(slot["width"] / panel_svg["viewbox_width"], slot["height"] / panel_svg["viewbox_height"])
        group = ET.SubElement(
            root,
            _svg_tag("g"),
            {
                "id": f"panel_{panel_id}",
                "transform": f"translate({slot['x']:.6f} {slot['y']:.6f}) scale({scale:.8f})",
            },
        )
        for child in panel_svg["children"]:
            group.append(child)
        ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{slot['x']:.6f}",
                "y": f"{max(8.0, slot['y'] - 3.0):.6f}",
                "font-family": FONT_FAMILY,
                "font-size": f"{PANEL_ID_SIZE:.1f}",
                "font-weight": "700",
                "fill": "black",
            },
        ).text = str(panel.get("label", panel_id))
    return ET.ElementTree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--row", choices=("all", "first"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree = render_assembly_svg(dpi=int(args.dpi), row=str(args.row))
    out_path = args.output_dir / f"{args.basename}.svg"
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print({"saved_paths": [str(out_path)]})


if __name__ == "__main__":
    main()
