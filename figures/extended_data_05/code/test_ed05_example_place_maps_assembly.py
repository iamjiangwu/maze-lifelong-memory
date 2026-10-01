"""Focused contract tests for Extended Data Figure 5 assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "render_ed05_example_place_maps_assembly.py"
SHORTLIST_NPZ = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_06"
    / "example_place_map_shortlist_32_clear_fields_2026-08-02"
    / "selected_maps_fields_45_shortlist32.npz"
)
DAY_COMB = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_tables"
    / "ed06_day_combination_animal_values_corrected36.csv"
)


@pytest.mark.skipif(not SHORTLIST_NPZ.is_file(), reason="ED6 shortlist NPZ unavailable")
@pytest.mark.skipif(not DAY_COMB.is_file(), reason="ED6 day-combination table unavailable")
def test_ed05_assembly_round11_layout_and_no_manual_mask_contract(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_ed05", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "assembly"
    manifest = module.assemble(out)
    assert manifest["status"] == "pass"
    assert manifest["round"] == 11
    assert manifest["panel_letters"] == ["a", "b", "c", "d", "e"]
    assert module.MAIN_FIG_PAIR_COLOR == "#B64342"

    width_in, height_in = manifest["canvas_inches"]
    assert width_in < 5.3
    assert height_in < 7.2
    assert module.PAD_LEFT <= 0.05
    assert module.ROW_BE_SHIFT_IN > 0.0
    assert module.DAYCOMB_LEFT > module.GALLERY_LEFT

    validation = pd.read_csv(out / "ED05_VALIDATION_ROUND11.csv")
    assert (validation["status"] == "PASS").all()
    assert "equal_height_matches_bcd" in set(validation["check"])
    assert "place_bh3_significant_brackets" in set(validation["check"])
    assert "pv_bh3_significant_brackets" in set(validation["check"])
    assert "rate_bh3_significant_brackets" in set(validation["check"])
    assert "pv_pair_source_no_manual_mask_drops" in set(validation["check"])
    assert "place_pv_rate_ordered_id_pairs_exact_144" in set(validation["check"])
    assert "place_pv_rate_exact_common_neuron_support" in set(validation["check"])
    assert "gallery_per_side_provenance_complete_and_qualified" in set(validation["check"])
    assert "daycomb_drawn_animal_values_recompute_from_pair_authority" in set(validation["check"])
    assert "daycomb_descriptive_stats_recompute_from_drawn_animals" in set(validation["check"])
    assert "daycomb_displayed_paired_t_stats_recompute_from_drawn_animals" in set(validation["check"])
    assert "equal_time_single_source_vintage" in set(validation["check"])
    assert "pdf_panel_a_effective_dpi_at_least_600" in set(validation["check"])
    assert "svg_panel_a_effective_dpi_at_least_600" in set(validation["check"])

    bh3 = pd.read_csv(out / "ED05_DAY_COMBINATION_BH3_ROUND11.csv")
    bh3 = bh3.loc[
        bh3["family"].eq("d1v1_pairwise")
        & bh3["test"].eq("paired t-test")
    ]
    assert len(bh3) == 9
    assert bh3["p_bh3_within_metric_exact_signflip"].notna().all()
    hits = bh3.groupby("metric")["p_bh3_within_metric_exact_signflip"].apply(
        lambda values: int((values < 0.05).sum())
    )
    assert hits.to_dict() == {
        "ensemble_rate_correlation": 3,
        "place_map_correlation": 0,
        "pv_correlation": 3,
    }

    for extension in ("svg", "png", "tiff"):
        path = out / f"ed05_example_place_maps_assembly_round11.{extension}"
        assert path.is_file() and path.stat().st_size > 10_000

    view_png = out.parent / "extended_data_figure5_source_rerender_round11.png"
    view_tiff = out.parent / "extended_data_figure5_source_rerender_round11.tiff"
    assert view_png.is_file() and view_png.stat().st_size > 10_000
    assert view_tiff.is_file() and view_tiff.stat().st_size > 10_000

    written = json.loads((out / "MANIFEST_ROUND11.json").read_text(encoding="utf-8"))
    assert written["status"] == "pass"
