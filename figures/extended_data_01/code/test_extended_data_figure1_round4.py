from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "render_extended_data_figure1_round4.py"
CANDIDATE_VALIDATION = (
    REPO
    / "docs"
    / "figure_resolution_repair_2026-08-12"
    / "ed01"
    / "RENDER_VALIDATION_ROUND12.json"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location(
        "extended_data_figure1_round4_renderer",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_round_uses_source_data_not_rendered_panels() -> None:
    renderer = load_renderer()
    assert renderer.ROUND == 10
    assert len(renderer.SOURCE_FILES) == 14
    assert all(path.is_file() for path, _ in renderer.SOURCE_FILES.values())
    assert all(
        path.suffix.lower() not in {".svg", ".png", ".tif", ".tiff"}
        for path, _ in renderer.SOURCE_FILES.values()
    )


def test_latest_round_first_panel_is_full_through_first_reward() -> None:
    renderer = load_renderer()
    trajectory = pd.read_csv(renderer.TRAJECTORY, low_memory=False)
    day1 = trajectory.loc[trajectory["nickname"].astype(str).eq("434RRday1")]
    assert len(day1) > 20_000
    assert set(day1["clip_mode"].astype(str)) == {
        "all_tracked_in_maze_segments_through_first_reward_node_reach"
    }
    assert set(day1["reward_node"].astype(int)) == {94}


def test_panel_f_retains_brackets_with_restrained_larger_stars_and_arial_math() -> None:
    renderer = load_renderer()
    renderer.configure_style()
    assert renderer.F_BRACKET_STEP_FRACTION == 0.200
    assert renderer.F_STAR_OFFSET_FRACTION == 0.006
    assert renderer.F_STAR_FONTSIZE_PT == 4.5
    assert plt.rcParams["font.sans-serif"] == ["Arial", "Liberation Sans"]
    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["mathtext.rm"] == "Arial"


def test_panel_i_uses_coverage_round_wording() -> None:
    renderer = load_renderer()
    data = renderer.read_inputs()
    fig, _bboxes, _derived = renderer.build_figure(data)
    try:
        texts = [text.get_text() for axis in fig.axes for text in axis.texts]
        titles = [axis.get_title() for axis in fig.axes]
        xlabels = [axis.get_xlabel() for axis in fig.axes]
        assert titles.count("Within-round coverage") == 1
        assert titles.count("Coverage-round efficiency") == 1
        assert xlabels.count("Coverage round") == 1
        assert "Within-cycle coverage" not in texts + titles + xlabels
        assert "Cycle efficiency" not in texts + titles + xlabels
        assert "Cycle" not in xlabels
    finally:
        plt.close(fig)


def test_latest_round_render_validation_passes() -> None:
    renderer = load_renderer()
    report = json.loads(CANDIDATE_VALIDATION.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["checks"]["aligned_full_height_row_groups"]
    assert report["checks"]["panel_g_square"]
    assert report["checks"]["panel_h_axes_match_panel_g_physical_height"]
    assert report["checks"]["panels_g_h_vertically_centered_in_f_row"]
    assert report["checks"]["panel_h_legend_lower_right"]
    assert report["checks"]["panel_h_legend_inside_lower_right_quadrant"]
    assert report["checks"]["panel_h_legend_clear_of_data"]
    assert report["checks"]["svg_contains_no_raster_images"]
    assert report["checks"]["pdf_contains_no_raster_images"]
    assert report["checks"]["pdf_contains_vector_drawings"]
    assert report["checks"]["pdf_contains_selectable_text"]
    assert report["checks"]["single_canvas_no_nonuniform_panel_scaling"]
    assert report["checks"]["caption_level_prose_absent"]
    assert report["checks"]["rounds_share_one_working_folder"]
    assert report["checks"]["panel_a_day1_uses_no_jump_mask"]
    assert report["checks"]["panel_a_vertical_colorbar"]
    assert report["checks"]["panel_a_colorbar_label_not_cut_off"]
    assert report["checks"]["panel_a_shifted_left_and_row_gap_reduced"]
    assert report["checks"]["panel_a_label_moved_down_inside_canvas"]
    assert report["checks"]["panels_b_to_e_moved_up_by_annotated_gap"]
    assert report["checks"]["background_grids_disabled_by_default"]
    assert report["checks"]["panels_b_to_e_share_ordered_legend_row"]
    assert report["checks"]["panel_f_all_six_axes_have_tests"]
    assert report["checks"]["panel_f_annotation_headroom"]
    assert report["checks"]["panel_f_stars_close_to_own_separated_brackets"]
    assert report["checks"]["panel_i_pair_supports_five_cycles"]
    assert report["checks"]["panel_i_shared_x_axes_and_cycle_boundaries"]
    assert report["checks"]["panel_i_column1_truncated_to_naive_two_cycles"]
    assert report["checks"]["panel_i_efficiency_axis_contains_all_points"]
    assert report["checks"]["panel_i_vertical_condition_labels"]
    assert report["checks"]["panel_i_condition_labels_moved_right"]
    assert report["checks"]["panel_labels_g_h_j_k_outside_top_left_and_i_left"]
    assert report["checks"]["requested_axis_labels_and_end_node_terms"]
