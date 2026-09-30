"""Render ED5 with Figure-2-matched 15 x 15 correlations in panel a."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from prepare_ed05_context_only_all_outside_sources import (
    ANALYSIS_MANIFEST,
    ANALYSIS_REPORT,
    FEASIBILITY_SOURCE,
    MEANS_OUTPUT,
    PAIR_OUTPUT,
    PAIR_SOURCE,
    REPEATS_SOURCE,
    SESSION_POOLS_SOURCE,
    SOURCE_SET,
    STATS_OUTPUT,
    STATS_SOURCE,
    VALIDATION_OUTPUT,
    prepare,
)


REPO = Path(__file__).resolve().parents[1]
BASE_RENDERER = REPO / "scripts/render_ed05_example_place_maps_assembly.py"
OUTPUT = REPO / "docs/figure_refinement_2026-09-05/ed05_round18_15x15_labels"
ROUND = 18


def _load_base():
    spec = importlib.util.spec_from_file_location("ed05_assembly_for_round18", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_RENDERER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    prepare()
    renderer = _load_base()
    renderer.ROUND = ROUND
    renderer.EQUAL_PAIRS = PAIR_OUTPUT
    renderer.EQUAL_MEANS = MEANS_OUTPUT
    renderer.EQUAL_STATS = STATS_OUTPUT
    renderer.EQUAL_SOURCE_SET_EXPECTED = {SOURCE_SET}
    renderer.EQUAL_FEASIBILITY_SET_EXPECTED = {SOURCE_SET}
    renderer.EQUAL_EXTRA_SOURCES = (
        ("context_only_all_outside_original_pair_summary", PAIR_SOURCE),
        ("context_only_all_outside_original_clustered_stats", STATS_SOURCE),
        ("context_only_all_outside_repeat_results", REPEATS_SOURCE),
        ("context_only_all_outside_pair_feasibility", FEASIBILITY_SOURCE),
        ("context_only_all_outside_session_frame_pools", SESSION_POOLS_SOURCE),
        ("context_only_all_outside_analysis_manifest", ANALYSIS_MANIFEST),
        ("context_only_all_outside_analysis_report", ANALYSIS_REPORT),
        ("context_only_all_outside_source_preparation_validation", VALIDATION_OUTPUT),
    )
    renderer.EQUAL_CAPTION_PANEL_NOTE = (
        "- **e.** Duration-matched ensemble-rate Outside versus Maze for "
        "Day1→Day1. All calcium-valid outside-maze frames were used and an "
        "equal number of calcium-valid in-maze frames was sampled without "
        "position or speed filtering (36 transition rows / 14 mice; CR1 means "
        "± 95% CI; 20 in-maze draws)."
    )
    renderer.EQUAL_CAPTION_CORRECTION_NOTE = (
        "- Panel e uses all calcium-valid outside-maze frames and an equal "
        "number of calcium-valid annotated in-maze frames, without position or "
        "speed filtering."
    )
    renderer.EQUAL_CAPTION_STATS_LABEL = "Duration-matched Day1→Day1 maze−outside"
    renderer.EQUAL_MANIFEST_CORRECTION = (
        "panel e uses all outside frames and equal-size unfiltered in-maze draws; "
        "height, axes and style remain matched to b-d"
    )
    renderer.main(["--output-dir", str(OUTPUT)])


if __name__ == "__main__":
    main()
