"""Focused checks for the ED5 panel-a correlation-label estimator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RENDERER = REPO / "scripts/render_ed05_example_place_maps_assembly.py"


def _load_renderer():
    spec = importlib.util.spec_from_file_location("ed05_label_renderer", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_a_labels_use_current_15x15_analysis_values() -> None:
    renderer = _load_renderer()
    selected, maps = renderer.load_shortlist_maps()

    assert renderer.GALLERY_CORRELATION_COLUMN == "place_r_15_analysis"
    assert renderer.GALLERY_CORRELATION_GRID == (15, 15)
    assert renderer.GALLERY_DISPLAY_GRID == (45, 45)
    assert len(selected) == len(maps) == 32
    assert np.isfinite(selected["place_r_15_analysis"].to_numpy(float)).all()

    anchors = selected.loc[selected["fixed_main_fig2b"].astype(bool)].sort_values(
        "anchor_order", kind="stable"
    )
    actual = anchors["place_r_15_analysis"].to_numpy(float)
    assert len(actual) == 4
    assert np.array_equal(np.round(actual, 2), np.array([0.63, 0.42, 0.18, -0.20]))
    # Figure 2 stores final Pearson coefficients as float32; ED5 retains float64.
    expected_float32 = np.array(
        [0.6342973709106445, 0.4175279140472412,
         0.18284372985363007, -0.1984570473432541]
    )
    assert np.array_equal(actual.astype(np.float32).astype(float), expected_float32)
    by_rank = selected.set_index("visual_rank")
    for rank, source, target in ((10, 1126, 1223), (11, 558, 1086), (31, 548, 380)):
        assert (by_rank.loc[rank, "train_id"], by_rank.loc[rank, "test_id"]) == (source, target)
    assert selected["source_id_space"].eq("current_sorted_zero_based").all()


def test_normalized_gallery_pack_correlations_and_ids_are_exact() -> None:
    renderer = _load_renderer()
    selected, maps = renderer.load_shortlist_maps()
    with np.load(renderer.SHORTLIST_NPZ, allow_pickle=False) as pack:
        assert str(pack["schema_version"]) == "current_sorted_normalized_gallery_v1"
        assert np.array_equal(pack["analysis_r15"], selected["place_r_15_analysis"])
        assert np.array_equal(pack["source_neuron_zero_based"], selected["train_id"])
        assert np.array_equal(pack["target_neuron_zero_based"], selected["test_id"])
        assert "smoothed_a" not in pack.files and "field_labels_a" not in pack.files
    assert all(set(value) == {"normalized_a", "normalized_b"} for value in maps.values())
