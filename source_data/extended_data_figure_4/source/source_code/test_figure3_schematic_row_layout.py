from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import FancyArrowPatch, Rectangle
from conftest import require_local_data


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "scripts" / "render_main_figs_v3_publication.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("figure3_schematic_row_test", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_figure3_layout_has_three_schematics_and_contiguous_labels() -> None:
    renderer = _load_renderer()
    layout = renderer.FIGURE3_PPT_PANEL_LAYOUT
    assert [item["panel_letter"] for item in layout] == list("abcdefghijklm")
    assert renderer.FIGURE3_PPT_ROW_HEADERS == ()

    schematics = [item for item in layout if item["row"] == -2]
    quantitative = [item for item in layout if item["row"] >= 0]
    assert [item["panel_letter"] for item in schematics] == ["b", "c", "d"]
    assert [item["schematic"] for item in schematics] == [
        "moving_windows",
        "first30_experience",
        "long_gap_day1",
    ]
    assert [item["panel_letter"] for item in quantitative] == list("efghijklm")
    assert {(item["row"], item["column"]) for item in quantitative} == {
        (row, column) for row in range(3) for column in range(3)
    }

    top = layout[0]["slot_pt"]
    schematic_top = min(item["slot_pt"][1] for item in schematics)
    schematic_bottom = max(item["slot_pt"][1] + item["slot_pt"][3] for item in schematics)
    first_quant_top = min(item["slot_pt"][1] for item in quantitative if item["row"] == 0)
    assert top[1] + top[3] < schematic_top
    assert schematic_bottom < first_quant_top


def test_figure3_schematics_are_vector_native_and_have_no_result_curves() -> None:
    renderer = _load_renderer()
    assert renderer.FIGURE3_SCHEMATIC_TITLES["first30_experience"] == (
        "First 30 min across cycles"
    )
    style = renderer.PublicationStyle(dpi=72)
    forbidden = {
        "similarity to Day 2",
        "similarity across pivot",
        "pivot time",
        "anticipated",
    }
    for schematic, title in renderer.FIGURE3_SCHEMATIC_TITLES.items():
        fig = renderer._render_figure3_analysis_schematic(schematic, (1.62, 0.74), style)
        try:
            assert len(fig.axes) == 1
            ax = fig.axes[0]
            assert not ax.images
            texts = {text.get_text() for text in ax.texts}
            assert title in texts
            assert texts.isdisjoint(forbidden)
            assert sum(isinstance(patch, Rectangle) for patch in ax.patches) >= 5
            assert sum(isinstance(patch, FancyArrowPatch) for patch in ax.patches) >= 3
        finally:
            plt.close(fig)


def test_figure3_schematic_palette_and_comparison_grammar() -> None:
    renderer = _load_renderer()
    style = renderer.PublicationStyle(dpi=72)
    schematic_colors = {
        str(value).lower()
        for value in renderer.FIGURE3_SCHEMATIC_COLORS.values()
        if isinstance(value, str)
    }
    schematic_colors.update(
        str(value).lower()
        for value in renderer.FIGURE3_SCHEMATIC_COLORS["window_colors"]
    )
    assert schematic_colors.isdisjoint(
        {str(value).lower() for value in style.metric_colors.values()}
    )
    representation_colors = list(style.metric_colors.values()) + [
        color
        for metric_colors in style.day2_group_colors.values()
        for color in metric_colors.values()
    ]
    for schematic_key in ("whole_fill", "whole_edge", "window_edge", "arrow"):
        schematic_rgb = np.asarray(to_rgb(renderer.FIGURE3_SCHEMATIC_COLORS[schematic_key]))
        assert min(
            np.linalg.norm(schematic_rgb - np.asarray(to_rgb(color)))
            for color in representation_colors
        ) > 0.13

    figures = {
        name: renderer._render_figure3_analysis_schematic(name, (1.62, 0.74), style)
        for name in renderer.FIGURE3_SCHEMATIC_TITLES
    }
    try:
        moving_ax = figures["moving_windows"].axes[0]
        moving_arrows = [
            patch for patch in moving_ax.patches if isinstance(patch, FancyArrowPatch)
        ]
        assert len(moving_arrows) == 4
        assert not moving_ax.lines
        comparison_arrows = [
            patch
            for patch in moving_arrows
            if np.allclose(patch._posA_posB[1], renderer.FIGURE3_MOVING_WINDOW_TARGET)
        ]
        assert len(comparison_arrows) == 3
        assert all(
            np.allclose(patch._posA_posB[1], renderer.FIGURE3_MOVING_WINDOW_TARGET)
            for patch in comparison_arrows
        )
        moving_texts = {text.get_text() for text in moving_ax.texts}
        assert {"Day i", "Day i + 1", "time"}.issubset(moving_texts)
        assert "30-min windows" not in moving_texts
        assert "whole Day 2" not in moving_texts

        first30_ax = figures["first30_experience"].axes[0]
        long_gap_ax = figures["long_gap_day1"].axes[0]
        for ax in (first30_ax, long_gap_ax):
            assert sum(line.is_dashed() for line in ax.lines) == 4
            texts = {text.get_text() for text in ax.texts}
            assert {"0", "1", "2", "3", "4"}.issubset(texts)
        first30_texts = {text.get_text() for text in first30_ax.texts}
        assert {"Day 1", "Day 2", "No. of cycles"}.issubset(first30_texts)
        assert "No. of experiences" not in first30_texts
        assert {"D1", "D2"}.isdisjoint(first30_texts)
        long_gap_texts = {text.get_text() for text in long_gap_ax.texts}
        assert "No. of cycles" in long_gap_texts
        assert "No. of experiences" not in long_gap_texts
        assert {"D1", "D2", "Day 1", "Day 2", "~90 d"}.isdisjoint(long_gap_texts)

        def backbone_positions(ax):
            return sorted(
                (
                    round(float(patch.get_x()), 6),
                    round(float(patch.get_y()), 6),
                    round(float(patch.get_width()), 6),
                    round(float(patch.get_height()), 6),
                )
                for patch in ax.patches
                if isinstance(patch, Rectangle)
                and np.isclose(float(patch.get_width()), 0.060)
            )

        assert backbone_positions(first30_ax) == backbone_positions(long_gap_ax)
        assert len(backbone_positions(long_gap_ax)) == 10
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_figure3_source_canvases_match_destination_aspect_ratios() -> None:
    renderer = _load_renderer()
    for layout in renderer.FIGURE3_PPT_PANEL_LAYOUT:
        width_in, height_in = renderer._figure3_source_figsize(layout)
        _x, _y, width_pt, height_pt = (float(value) for value in layout["slot_pt"])
        assert np.isclose(width_in / height_in, width_pt / height_pt)


def test_figure3_panel_a_uses_square_boundaries_without_internal_maze_walls() -> None:
    renderer = _load_renderer()
    # The example-neuron bundles are .npz files, which are never committed.
    require_local_data("docs/example_neuron_candidates/saved_picks")
    style = renderer.PublicationStyle(dpi=72)
    layout = next(
        item for item in renderer.FIGURE3_PPT_PANEL_LAYOUT if item["panel_letter"] == "a"
    )
    fig = plt.figure(figsize=renderer._figure3_source_figsize(layout))
    grid = fig.add_gridspec(1, 1, left=0.005, right=0.995, bottom=0.02, top=0.88)
    try:
        axes, _record = renderer._draw_day12_arrow_example(fig, grid[0, 0], style)
        assert len(axes) == 4
        low, high = renderer.FIGURE3_EXAMPLE_AXIS_LIM
        for ax in axes:
            boundaries = [
                patch
                for patch in ax.patches
                if isinstance(patch, Rectangle)
                and np.isclose(float(patch.get_x()), low)
                and np.isclose(float(patch.get_y()), low)
                and np.isclose(float(patch.get_width()), high - low)
                and np.isclose(float(patch.get_height()), high - low)
            ]
            assert len(boundaries) == 1
            assert np.allclose(ax.get_xlim(), (low, high))
            assert np.allclose(ax.get_ylim(), (low, high))
            assert all(str(line.get_color()) != "k" for line in ax.lines)
        fig.canvas.draw()
        renderer_px = fig.canvas.get_renderer()
        for ax in axes:
            boundary_top_px = float(
                ax.transData.transform((renderer.FIGURE3_EXAMPLE_AXIS_LIM[0], high))[1]
            )
            title_bottom_px = float(ax.title.get_window_extent(renderer_px).y0)
            clearance_pt = (title_bottom_px - boundary_top_px) * 72.0 / fig.dpi
            assert clearance_pt >= 1.5
    finally:
        plt.close(fig)


def test_figure3_support_lines_use_only_adjacent_source_segments() -> None:
    renderer = _load_renderer()
    style = renderer.PublicationStyle(dpi=72)
    x = np.arange(7, dtype=float)
    y = np.asarray([0.1, 0.2, 0.3, 0.35, 0.5, 0.55, 0.6])
    high = np.asarray([True, True, False, False, True, True, False])
    fig, ax = plt.subplots()
    try:
        counts = renderer._add_support_encoded_line(
            ax,
            x,
            y,
            high,
            color="#123456",
            label="example",
            style=style,
        )
        assert counts == {"high_segments": 2, "low_segments": 4}
        collections = [item for item in ax.collections if isinstance(item, LineCollection)]
        assert len(collections) == 2
        segments = [segment for collection in collections for segment in collection.get_segments()]
        assert len(segments) == len(x) - 1
        assert all(np.isclose(segment[1, 0] - segment[0, 0], 1.0) for segment in segments)
    finally:
        plt.close(fig)


def test_figure3_scatter_points_reuse_metric_experience_gradients() -> None:
    renderer = _load_renderer()
    style = renderer.PublicationStyle(dpi=72)
    metric = "place_map_correlation"
    cases = (
        (
            "experience_index",
            np.arange(5, dtype=float),
            renderer.FIGURE3_FIRST30_EXPERIENCE_COLOR_GROUPS,
        ),
        (
            "experience",
            np.arange(1, 5, dtype=float),
            renderer.FIGURE3_LONG_GAP_EXPERIENCE_COLOR_GROUPS,
        ),
    )
    for x_col, x_values, point_group_map in cases:
        points = pd.DataFrame(
            {
                "metric": [metric] * len(x_values),
                x_col: x_values,
                "value": np.linspace(0.2, 0.5, len(x_values)),
            }
        )
        fig, ax = plt.subplots()
        try:
            renderer._draw_experience_scatter(
                ax,
                points,
                pd.DataFrame(),
                metric,
                style,
                x_col=x_col,
                x_values=np.linspace(float(x_values.min()), float(x_values.max()), 20),
                xticks=tuple(int(value) for value in x_values),
                xticklabels=tuple(str(int(value)) for value in x_values),
                show_xlabel=True,
                ylabel="Correlation",
                point_group_map=point_group_map,
            )
            actual = ax.collections[0].get_facecolors()
            expected = np.asarray(
                [
                    to_rgba(
                        style.day2_group_colors[metric][point_group_map[value]],
                        alpha=style.point_alpha,
                    )
                    for value in x_values
                ]
            )
            assert np.allclose(actual, expected)
        finally:
            plt.close(fig)


def test_figure3_row_identity_moves_to_first_column_ylabels() -> None:
    renderer = _load_renderer()
    assert renderer.FIGURE3_ROW_YLABELS == {
        "place_map_correlation": "Place map correlation",
        "pv_correlation": "Population vector correlation",
        "ensemble_rate_correlation": "Ensemble rate correlation",
    }
    assert all("\n" not in label for label in renderer.FIGURE3_ROW_YLABELS.values())


def test_figure3_r_annotations_use_one_upper_right_slot_position() -> None:
    renderer = _load_renderer()
    style = renderer.PublicationStyle(dpi=72)
    metric = "place_map_correlation"
    layout = next(item for item in renderer.FIGURE3_PPT_PANEL_LAYOUT if item["panel_letter"] == "f")
    stats_df = pd.DataFrame(
        {"q_bh_two_sided": [0.01], "pearson_r": [0.62]},
        index=[metric],
    )
    root = ET.Element(renderer._svg_tag("svg"))
    renderer._add_svg_figure3_stat_overlay(
        root,
        panel_letter="f",
        metric=metric,
        stats_df=stats_df,
        slot_pt=layout["slot_pt"],
        style=style,
    )

    r_node = next(
        node for node in root.iter() if node.attrib.get("class") == "figure3-stat-r-label"
    )
    slot_x, slot_y, slot_w, slot_h = renderer._figure3_relative_slot_pt(layout["slot_pt"])
    r_x_frac, r_y_frac = renderer.FIGURE3_STAT_R_POSITION_FRAC
    assert float(r_node.attrib["x"]) == pytest.approx(slot_x + slot_w * r_x_frac)
    assert float(r_node.attrib["y"]) == pytest.approx(slot_y + slot_h * r_y_frac)
    assert r_x_frac > 0.85
    assert r_y_frac < 0.10
    assert r_node.attrib["text-anchor"] == "end"
    assert r_node.attrib["dominant-baseline"] == "hanging"
