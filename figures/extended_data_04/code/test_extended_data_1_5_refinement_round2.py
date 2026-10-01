from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def source(name: str) -> str:
    return (REPO / "scripts" / name).read_text(encoding="utf-8")


def test_ed01_round2_language_and_layout_contract() -> None:
    text = source("render_extended_data_figure1_round4.py")
    assert 'colorbar.set_ticklabels(["0", f"{shared_max:.0f}"])' in text
    assert 'colorbar.set_label("Time in maze (min)"' in text
    assert '"End node visits"' in text
    assert '"End nodes"' in text
    assert "column_centers =" in text
    assert '"Return interval (months)"' in text
    assert '"panel_i_left_labels_clear_and_nonoverlapping"' in text


def test_ed02_round21_display_only_contract() -> None:
    text = source("render_ed02_slide3_refinement_round21.py")
    assert "PANEL_A_ANNOTATION_X = 3.52" in text
    assert "PANEL_A_ANNOTATION_Y = 3.48" in text
    assert "PANEL_G_UPPER_ANCHOR = (0.52, 1.30598)" in text
    assert "PANEL_G_LOWER_ANCHOR = (0.58, 1.34090)" in text
    assert 'spines["left"].set_bounds(0.0, 0.10)' in text
    assert "MODEL_TITLE_Y = 0.94" in text


def test_ed04_and_ed05_requested_labels() -> None:
    ed04 = source("render_ed04_registration_quality_assembly.py")
    ed05 = source("render_ed05_example_place_maps_assembly.py")
    assert "ROW_C_TOP = 3.61" in ed04
    assert "ROW_D_TOP = 5.15" in ed04
    assert ed04.count('"No. of cycles"') >= 4
    assert 'set_xlabel("No. of experiences")' not in ed04
    assert '"x_label": "No. of experiences"' not in ed04
    assert 'ax.set_xlabel("Spatial footprint correlation")' in ed04
    assert '"Normalized rate"' in ed05
