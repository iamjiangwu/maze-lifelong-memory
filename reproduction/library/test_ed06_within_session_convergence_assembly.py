"""Focused contract tests for the Extended Data Figure 6 assembly.

Guards the things most likely to regress silently: the five-minute coverage gate as the
data authority, the tile-change x axis whose ``x_plot`` arrives in thousands and is scaled
exactly once for display, and the axis titles shared with Extended Data Figure 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "render_ed06_within_session_convergence_assembly.py"
COVERAGE_ROOT = REPO_ROOT / "docs" / "coverage5min_window_reanalysis_2026-08-05"
FIXED_CURVES = (
    COVERAGE_ROOT / "rendered" / "ed07_fixed_time_tile" / "ed07_current59_group_curve_summary.csv"
)
PIVOT_STATS = (
    COVERAGE_ROOT
    / "rendered"
    / "ed07_pivot"
    / "ed07_pivot_day_specific_early_slope_stats_current59.csv"
)
HISTORICAL_CURVES = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_07"
    / "fixed_reference_time_tile_current59_2026-07-14"
    / "ed07_current59_group_curve_summary.csv"
)

requires_sources = pytest.mark.skipif(
    not (FIXED_CURVES.is_file() and PIVOT_STATS.is_file() and HISTORICAL_CURVES.is_file()),
    reason="coverage5min ED07 bundles unavailable",
)


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_ed06", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_sources
def test_data_authority_is_the_five_minute_gate() -> None:
    module = _load_module()
    assert module.COVERAGE_GATE_SEC == 300.0
    assert module.COVERAGE_GATE_FRAMES == 7500
    assert module.FIXED_CURVES == FIXED_CURVES
    assert module.PIVOT_STATS == PIVOT_STATS
    manifest = json.loads(
        (COVERAGE_ROOT / "figure3_authority" / "SOURCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    gate = manifest["coverage_gate"]
    assert gate["min_selected_duration_sec"] == module.COVERAGE_GATE_SEC
    assert gate["min_selected_frames_at_dt_0p04"] == module.COVERAGE_GATE_FRAMES


@requires_sources
def test_tile_x_axis_is_scaled_exactly_once_for_display() -> None:
    """Source stays in thousands; the axis draws raw transitions, so scale is x1,000.

    Applying the factor twice, or not at all, silently changes what the ED1 title
    "Maze-tile transitions" is naming.
    """

    module = _load_module()
    curves = pd.read_csv(FIXED_CURVES)
    tile = curves[curves["window_type"].eq("tile_change")]
    assert set(tile["x_plot_unit"]) == {module.EXPECTED_X_UNITS["tile_change"]}
    assert 1.0 <= tile["x_plot"].min() and tile["x_plot"].max() < 100.0
    maze = curves[curves["window_type"].eq("maze_time")]
    assert set(maze["x_plot_unit"]) == {module.EXPECTED_X_UNITS["maze_time"]}

    scales = {
        column["window_type"]: column["x_display_scale"] for column in module.FIXED_COLUMNS
    }
    assert scales["maze_time"] == 1.0
    assert scales["tile_change"] == 1000.0
    assert module.DRAWN_X_UNITS["tile_change"] == "maze tile transition"


@requires_sources
def test_axis_titles_follow_extended_data_figure1() -> None:
    module = _load_module()
    assert module.X_LABEL_TIME == "Time in maze (min)"
    assert module.X_LABEL_TILES == "Maze tile transitions"
    assert {column["x_label"] for column in module.FIXED_COLUMNS} == {
        module.X_LABEL_TIME,
        module.X_LABEL_TILES,
    }
    # One y title per metric, no "(r)", shared by the fixed-template and pivot panels.
    labels = [spec["y_label"] for spec in module.METRICS]
    assert labels == [
        "Place map correlation",
        "Population vector correlation",
        "Ensemble rate correlation",
    ]
    assert not any("stability" in label or "(r)" in label for label in labels)
    assert module.DAY_I_LABEL == r"Day $\mathrm{i}$"
    assert module.DAY_IP1_LABEL == r"Day $\mathrm{i}+1$"
    assert [column["title"] for column in module.PIVOT_COLUMNS] == [
        module.DAY_I_LABEL,
        module.DAY_IP1_LABEL,
    ]


@requires_sources
def test_gate_moves_only_the_twenty_minute_column() -> None:
    module = _load_module()
    new = pd.read_csv(FIXED_CURVES)
    old = pd.read_csv(HISTORICAL_CURVES)
    keys = ["metric", "window_type", "window_size", "experience_group", "x_plot"]
    merged = new.merge(old, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)

    def moved(window_type: str, window_size: float) -> bool:
        subset = merged[
            merged["window_type"].eq(window_type) & merged["window_size"].eq(window_size)
        ]
        both = subset[subset["_merge"].eq("both")]
        changed = bool((both["mean_new"] - both["mean_old"]).abs().gt(0.0).any())
        return changed or bool((subset["_merge"] == "left_only").any())

    assert not moved("maze_time", 10.0), "10-min already required 5 min of selected data"
    assert moved("maze_time", 20.0), "20-min eligibility must change under the fixed gate"
    assert not moved("tile_change", 3000.0), "tile windows carry no time-coverage gate"


@requires_sources
def test_pivot_thirty_minute_scope_and_null() -> None:
    module = _load_module()
    stats = pd.read_csv(PIVOT_STATS)
    thirty = stats[stats["window_size_min"].eq(module.PIVOT_WINDOW_MIN)]
    assert set(thirty["n_pair_rows"]) == {module.PIVOT_EXPECTED_PAIR_ROWS}
    assert set(thirty["n_animal_clusters"]) == {module.EXPECTED_ANIMALS}
    assert set(thirty["correction_family_n"]) == {module.PIVOT_BH_FAMILY}
    assert set(thirty["pivot_timebase"]) == {"maze_time"}
    # Place-map Day 2 is the figure's one deliberately null displayed claim.
    place_day2 = thirty[
        thirty["metric"].eq("place_map_correlation") & thirty["day"].eq("day2")
    ].iloc[0]
    assert place_day2["q_bh_two_sided"] > 0.05
    for metric in ("pv_correlation", "ensemble_rate_correlation"):
        row = thirty[thirty["metric"].eq(metric) & thirty["day"].eq("day2")].iloc[0]
        assert row["q_bh_two_sided"] < 0.05


@requires_sources
def test_assembly_renders_and_passes_its_contract(tmp_path: Path) -> None:
    module = _load_module()
    manifest = module.render(tmp_path, round_number=99)
    assert manifest["status"] == "pass"
    assert manifest["validation_status"] == "pass"
    assert [row for row in manifest["validation"] if row["status"] == "FAIL"] == []
    assert sorted(manifest["panels"]) == list("abcdefgh")
    assert len(manifest["displayed_statistics"]) == 15
    for fixed_letter, pivot_letter in zip("ceg", "dfh"):
        panels = manifest["panels"]
        assert panels[fixed_letter]["y_label"] == panels[pivot_letter]["y_label"]
        assert set(panels[pivot_letter]["x_labels"]) == {module.X_LABEL_TIME}
        # One y range per metric row, spanning the fixed-template and pivot blocks.
        assert (
            panels[fixed_letter]["shared_y_limits"]
            == panels[pivot_letter]["shared_y_limits"]
        )
        assert panels[pivot_letter]["superseded_canonical_y_limits"] != []
    assert manifest["panels"]["c"]["x_labels"][-1] == module.X_LABEL_TILES
    gate = manifest["gate_impact_on_displayed_columns"]
    assert gate["10-min window"]["changed_under_five_minute_gate"] is False
    assert gate["20-min window"]["changed_under_five_minute_gate"] is True
    assert gate["3,000-tile window"]["changed_under_five_minute_gate"] is False
    for extension in ("svg", "pdf", "png", "tiff"):
        assert (tmp_path / f"{module.STEM}_round99.{extension}").is_file()
