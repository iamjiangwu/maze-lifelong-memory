"""Build an unpromoted ED9 display-repair and 600-dpi cloud candidate.

The repair is display-only and artist-local.  Panel b receives a 50-pixel
left canvas extension while its axes remain fixed in composite coordinates.
Panel f retains its accepted bracket/data geometry and uses the 7-point star
size of the promoted Main Figure 1 bracket-scatter composite.  The dense
continuous clouds in display panels g and j are regenerated from their locked
source tables at 600 effective embedded-image dpi; native discrete grids are
left unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PARENT = (
    REPO
    / "docs/figure_version_records/"
    "extended_data_fig09_2026-08-11_round25_panel_e_signal_metadata"
)
MAIN1 = (
    REPO
    / "docs/figure_version_records/"
    "main_figure1_2026-08-11_round7_complete_raster_export_closure"
)
OUTPUT = (
    REPO
    / "docs/figure_refinement_2026-08-12/"
    "ed09_panel_b_margin_panel_f_star_density_600dpi"
)
PANEL_DIR = OUTPUT / "individual_panels"
SOURCE_CODE = PARENT / "source_code"
ROUND6_RENDERER = SOURCE_CODE / "render_ed09_narrow_typography_round6.py"
ROUND12_RENDERER = SOURCE_CODE / "render_ed09_significance_round12.py"
ROUND14_RENDERER = SOURCE_CODE / "render_ed09_panel_c_histogram_round14.py"
ROUND16_RENDERER = SOURCE_CODE / "render_ed09_order_neutral_labels_round16.py"
ROUND3_RENDERER = SOURCE_CODE / "render_ed09_four_row_round3.py"
OLD_RENDERER = SOURCE_CODE / "render_ed09_slide11_panels.py"
PARENT_MANIFEST = (
    PARENT
    / "source_validation/refinement_round2/ED09_REFINEMENT_ROUND2_MANIFEST.json"
)
PANEL_F_SOURCE = PARENT / "source_data/panel_f_inference_round24"
PANEL_F_PARENT_RENDER = (
    PARENT
    / "source_validation/panel_f_inference_round24/"
    "ED09_PANEL_F_INFERENCE_ROUND24_RENDER_MANIFEST.json"
)
MAINTAINED_ED9 = (
    REPO
    / "docs/extended_data_supplement_v2_rebuild_2026-08-05/"
    "ed09_tree_depth_level_controls"
)
MAINTAINED_CORRECTION = MAINTAINED_ED9 / "correction_2026-08-09"
ROUND4_SOURCE = (
    MAINTAINED_CORRECTION
    / "powerpoint_parity_round4_2026-08-09/source_data"
)
ROUND16_SOURCE = (
    MAINTAINED_CORRECTION
    / "order_neutral_labels_round16_2026-08-10/source_data"
)

DPI = 600
CANVAS_PX = (3900, 4010)
PANEL_B_PARENT_SIZE = (750, 1000)
PANEL_B_CANDIDATE_SIZE = (800, 1000)
PANEL_B_LEFT_PAD_PX = 50
PANEL_B_PARENT_X_PX = 1550
PANEL_B_CANDIDATE_X_PX = PANEL_B_PARENT_X_PX - PANEL_B_LEFT_PAD_PX
PANEL_F_STAR_SIZE_PT = 7.0
MIN_PANEL_B_PDF_INSET_PT = 2.0
OPEN_PLACE_MAP_LABEL = "Place map correlation"
OPEN_LEVEL_MAP_LABEL = "Level map correlation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_style(round6: Any) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": round6.FONT["base"],
            "axes.labelsize": round6.FONT["axis_label"],
            "xtick.labelsize": round6.FONT["tick"],
            "ytick.labelsize": round6.FONT["tick"],
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def copy_parent_panels() -> None:
    if OUTPUT.exists():
        existing_manifest = OUTPUT / "CANDIDATE_MANIFEST.json"
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if existing.get("status") == "PASS":
                raise FileExistsError(
                    f"Refusing to overwrite append-only candidate: {OUTPUT}"
                )
        # A failed first render may leave a partial candidate.  It is safe to
        # replace only this exact, unmanifested task-owned directory.
        if OUTPUT.resolve() != (
            REPO
            / "docs/figure_refinement_2026-08-12/"
            "ed09_panel_b_margin_panel_f_star_density_600dpi"
        ).resolve():
            raise RuntimeError(f"Unexpected candidate path: {OUTPUT}")
        shutil.rmtree(OUTPUT)
    PANEL_DIR.mkdir(parents=True)
    for source in (PARENT / "individual_panels").iterdir():
        if source.is_file() and source.suffix.lower() in {".svg", ".pdf", ".png", ".tiff"}:
            shutil.copy2(source, PANEL_DIR / source.name)


def _figure_state(fig: plt.Figure) -> dict[str, Any]:
    axis = fig.axes[0]
    fig.canvas.draw()
    return {
        "figure_size_inches": [float(value) for value in fig.get_size_inches()],
        "axes_bounds_fraction": [float(value) for value in axis.get_position().bounds],
        "axes_bounds_px": [
            float(axis.get_position().x0 * fig.get_figwidth() * DPI),
            float(axis.get_position().y0 * fig.get_figheight() * DPI),
            float(axis.get_position().width * fig.get_figwidth() * DPI),
            float(axis.get_position().height * fig.get_figheight() * DPI),
        ],
        "x_limits": [float(value) for value in axis.get_xlim()],
        "y_limits": [float(value) for value in axis.get_ylim()],
    }


def render_panel_b(round6: Any) -> tuple[dict[str, Path], dict[str, Any]]:
    """Rerender C2 with left padding and unchanged physical axes/artwork."""

    round14 = load_module("ed09_margin_star_round14", ROUND14_RENDERER)
    old = load_module("ed09_margin_star_old_c2", OLD_RENDERER)
    old.ROOT = MAINTAINED_ED9
    old.RENDER_ROOT = PANEL_DIR
    round14.ED9 = MAINTAINED_ED9
    round14.PANEL_B_SIZE = PANEL_B_CANDIDATE_SIZE

    original_draw = old.draw_c2
    original_save = old.save_figure
    state: dict[str, Any] = {}

    def padded_draw(_frame: dict[str, Any]) -> plt.Figure:
        fig = original_draw(
            {
                "w_in": PANEL_B_CANDIDATE_SIZE[0] / DPI,
                "h_in": PANEL_B_CANDIDATE_SIZE[1] / DPI,
            }
        )
        # Parent axes were 0.24..0.96 on a 750-px canvas: 180..720 px.
        # Add 50 px at left and retain the same 540-px axes width.
        fig.subplots_adjust(
            left=(0.24 * PANEL_B_PARENT_SIZE[0] + PANEL_B_LEFT_PAD_PX)
            / PANEL_B_CANDIDATE_SIZE[0],
            right=(0.96 * PANEL_B_PARENT_SIZE[0] + PANEL_B_LEFT_PAD_PX)
            / PANEL_B_CANDIDATE_SIZE[0],
        )
        return fig

    def save_with_fixed_panel_letter(
        fig: plt.Figure, stem: str, description: str
    ) -> dict[str, Path]:
        letters = [text for text in fig.texts if text.get_text().strip() == "b"]
        if len(letters) != 1:
            raise AssertionError(f"Expected one panel-b letter artist, found {len(letters)}")
        parent_letter_x_px = 0.014 * PANEL_B_PARENT_SIZE[0]
        letters[0].set_x(
            (parent_letter_x_px + PANEL_B_LEFT_PAD_PX)
            / PANEL_B_CANDIDATE_SIZE[0]
        )
        state.update(_figure_state(fig))
        return original_save(fig, stem, description)

    old.draw_c2 = padded_draw
    old.save_figure = save_with_fixed_panel_letter
    try:
        paths, audit = round14.render_panel_b(old, round6)
    finally:
        old.draw_c2 = original_draw
        old.save_figure = original_save

    with Image.open(paths["png"]) as image:
        candidate = np.asarray(image.convert("RGB"))
    with Image.open(PARENT / "individual_panels/ED9_C2_shift_distance.png") as image:
        parent = np.asarray(image.convert("RGB"))
    cropped = candidate[:, PANEL_B_LEFT_PAD_PX:, :]
    audit.update(
        {
            "artist_level": True,
            "left_canvas_padding_px": PANEL_B_LEFT_PAD_PX,
            "parent_size_px": list(PANEL_B_PARENT_SIZE),
            "candidate_size_px": list(PANEL_B_CANDIDATE_SIZE),
            "parent_crop_pixel_exact": bool(np.array_equal(cropped, parent)),
            "parent_crop_changed_pixels": int(np.count_nonzero(cropped != parent)),
            "figure_state": state,
            "displayed_source": record(
                MAINTAINED_ED9 / "C_MAXIMAL42_SHIFT_DISTANCE_10000PERM.csv"
            ),
        }
    )
    return paths, audit


def render_panel_f(round6: Any) -> tuple[dict[str, Path], dict[str, Any]]:
    """Rerender B2 with exact 7-point stars and unchanged bracket geometry."""

    round12 = load_module("ed09_margin_star_round12", ROUND12_RENDERER)
    round12.ROUND4_SOURCE = PANEL_F_SOURCE
    old = load_module("ed09_margin_star_old_b2", OLD_RENDERER)
    old.ROOT = PANEL_F_SOURCE
    old.RENDER_ROOT = PANEL_DIR

    original_normalize = round6.normalize_panel
    original_save = old.save_figure
    state: dict[str, Any] = {}

    def normalize_with_main1_star_size(
        fig: plt.Figure, role: str
    ) -> dict[str, Any]:
        result = original_normalize(fig, role)
        if role == "B2":
            stars = [
                text
                for axis in fig.axes
                for text in axis.texts
                if text.get_text().strip() in {"*", "**", "***"}
            ]
            if len(stars) != 2:
                raise AssertionError(f"Expected two panel-f stars, found {len(stars)}")
            for text in stars:
                text.set_fontsize(PANEL_F_STAR_SIZE_PT)
                text.set_fontweight("bold")
                text.set_fontfamily("Arial")
        return result

    def save_with_tick_offset(
        fig: plt.Figure, stem: str, description: str
    ) -> dict[str, Path]:
        axis = fig.axes[0]
        labels = [
            label
            for label in axis.get_xticklabels()
            if label.get_text().startswith("Vertical")
        ]
        if len(labels) != 1:
            raise AssertionError(f"Expected one Vertical tick artist, found {len(labels)}")
        source = labels[0]
        replacement = axis.text(
            *source.get_position(),
            source.get_text(),
            transform=source.get_transform()
            + ScaledTranslation(-3.0 / 72.0, 0.0, fig.dpi_scale_trans),
            ha=source.get_horizontalalignment(),
            va=source.get_verticalalignment(),
            multialignment=source._get_multialignment(),
            fontsize=source.get_fontsize(),
            fontfamily=source.get_fontfamily(),
            color=source.get_color(),
            clip_on=False,
        )
        replacement.set_gid("vertical_tick_offset_minus_3pt")
        source.set_visible(False)
        state.update(_figure_state(fig))
        return original_save(fig, stem, description)

    round6.normalize_panel = normalize_with_main1_star_size
    old.save_figure = save_with_tick_offset
    try:
        paths, audit = round12.render_panel_f(old, round6)
    finally:
        round6.normalize_panel = original_normalize
        old.save_figure = original_save

    audit.update(
        {
            "artist_level": True,
            "star_fontsize_pt": PANEL_F_STAR_SIZE_PT,
            "main_figure1_authority": record(MAIN1 / "figure1_version_record.pdf"),
            "figure_state": state,
        }
    )
    return paths, audit


def save_figure_with_600dpi_rasterized_artists(
    old: Any,
    fig: plt.Figure,
    stem: str,
    description: str,
) -> dict[str, Path]:
    """Save source-native vector panels with rasterized artists at 600 dpi."""

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        suffix: PANEL_DIR / f"{stem}.{suffix}"
        for suffix in ("svg", "pdf", "png", "tiff")
    }
    metadata = {
        "Title": stem,
        "Description": description,
        "Creator": "ED9 audited matplotlib renderer",
    }
    fig.savefig(
        paths["svg"],
        format="svg",
        dpi=DPI,
        facecolor=old.WHITE,
        edgecolor="none",
        metadata=metadata,
    )
    fig.savefig(
        paths["pdf"],
        format="pdf",
        dpi=DPI,
        facecolor=old.WHITE,
        edgecolor="none",
        metadata={
            "Title": stem,
            "Subject": description,
            "Creator": metadata["Creator"],
        },
    )
    fig.savefig(
        paths["png"],
        format="png",
        dpi=DPI,
        facecolor=old.WHITE,
        edgecolor="none",
    )
    plt.close(fig)
    with Image.open(paths["png"]) as image:
        image.convert("RGB").save(
            paths["tiff"], compression="tiff_lzw", dpi=(DPI, DPI)
        )
    return paths


def render_panel_g_dense_cloud(
    round6: Any,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Rerender display panel g (E1) at 600 effective raster dpi."""

    round16 = load_module("ed09_margin_star_round16", ROUND16_RENDERER)
    round3 = load_module("ed09_margin_star_round3_for_e1", ROUND3_RENDERER)
    old = load_module("ed09_margin_star_old_e1", OLD_RENDERER)
    round16.SOURCE_DATA = ROUND16_SOURCE
    round16.ROUND16 = ROUND16_SOURCE.parent
    old.ROOT = MAINTAINED_ED9
    old.RENDER_ROOT = PANEL_DIR
    original_save = old.save_figure

    def save_600(fig: plt.Figure, stem: str, description: str) -> dict[str, Path]:
        axis = fig.axes[0]
        axis.set_xlabel(OPEN_PLACE_MAP_LABEL, color="#000000")
        axis.set_ylabel(OPEN_LEVEL_MAP_LABEL, color="#000000")
        return save_figure_with_600dpi_rasterized_artists(
            old, fig, stem, description
        )

    old.save_figure = save_600
    try:
        paths, audit = round16.render_panel_g(old, round3, round6)
    finally:
        old.save_figure = original_save
    audit.update(
        {
            "artist_level": True,
            "embedded_raster_target_dpi": DPI,
            "source": record(
                ROUND16_SOURCE / "be_roles/E1_MAXIMAL42_CONTINUOUS_CLOUD_INPUT.csv"
            ),
            "axis_labels": {
                "x": OPEN_PLACE_MAP_LABEL,
                "y": OPEN_LEVEL_MAP_LABEL,
                "x_color": "#000000",
                "y_color": "#000000",
            },
        }
    )
    return paths, audit


def render_panel_j_dense_cloud(
    round6: Any,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Rerender display panel j (G1) at 600 effective raster dpi."""

    round3 = load_module("ed09_margin_star_round3_for_g1", ROUND3_RENDERER)
    round3.SOURCE = ROUND4_SOURCE
    old = load_module("ed09_margin_star_old_g1", OLD_RENDERER)
    old.ROOT = ROUND4_SOURCE
    old.RENDER_ROOT = PANEL_DIR
    width_px, height_px = (875, 875)
    fig, audit = round3.draw_g1(
        old,
        {"w_in": width_px / DPI, "h_in": height_px / DPI},
    )
    typography = round6.normalize_panel(fig, "G1")
    figure_state = _figure_state(fig)
    paths = save_figure_with_600dpi_rasterized_artists(
        old,
        fig,
        "ED9_G1_consecutive_gap",
        "ED9 panel j consecutive-gap density cloud at 600 effective dpi",
    )
    audit.update(
        {
            "artist_level": True,
            "embedded_raster_target_dpi": DPI,
            "source_points": record(ROUND4_SOURCE / "G_MAXIMAL26_POINTS.csv"),
            "source_statistics": record(
                ROUND4_SOURCE / "G_REWARDED20_CR1_BH4_STATISTICS.csv"
            ),
            "typography": typography,
            "figure_state": figure_state,
        }
    )
    return paths, audit


def pdf_spans(path: Path, wanted: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip() in wanted:
                            rows.append(
                                {
                                    "page": page_index,
                                    "text": span["text"].strip(),
                                    "size_pt": float(span["size"]),
                                    "font": span["font"],
                                    "bbox_pt": [float(value) for value in span["bbox"]],
                                }
                            )
    return rows


def pdf_text_span(path: Path, text: str) -> dict[str, Any]:
    rows = pdf_spans(path, {text})
    if len(rows) != 1:
        raise AssertionError(f"Expected one PDF span {text!r} in {path}, found {len(rows)}")
    return rows[0]


def pdf_embedded_image_contract(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            for image in page.get_images(full=True):
                xref = int(image[0])
                width_px = int(image[2])
                height_px = int(image[3])
                for rect in page.get_image_rects(xref):
                    dpi_x = width_px / (float(rect.width) / 72.0)
                    dpi_y = height_px / (float(rect.height) / 72.0)
                    rows.append(
                        {
                            "page": page_index,
                            "xref": xref,
                            "pixels": [width_px, height_px],
                            "bbox_pt": [
                                float(rect.x0),
                                float(rect.y0),
                                float(rect.x1),
                                float(rect.y1),
                            ],
                            "effective_dpi": [float(dpi_x), float(dpi_y)],
                            "minimum_effective_dpi": float(min(dpi_x, dpi_y)),
                        }
                    )
    return rows


def _bracket_coordinates(audit: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("x_left", "x_right", "y_base", "tick_height", "y_top", "text_y", "stars")
    return [
        {key: row[key] for key in keys}
        for row in audit["annotation_geometry"]["brackets"]
    ]


def assemble(
    panel_b_paths: dict[str, Path],
    panel_f_paths: dict[str, Path],
    panel_g_paths: dict[str, Path],
    panel_j_paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    rows = copy.deepcopy(parent["panels"])
    changed = {
        "C2": panel_b_paths,
        "B2": panel_f_paths,
        "E1": panel_g_paths,
        "G1": panel_j_paths,
    }
    for row in rows:
        role = row["role"]
        for suffix in ("svg", "pdf", "png", "tiff"):
            target = changed.get(role, {}).get(suffix, PANEL_DIR / Path(row[suffix]).name)
            if not target.is_file():
                raise FileNotFoundError(target)
            row[suffix] = target.relative_to(REPO).as_posix()
            row[f"{suffix}_sha256"] = sha256(target)
            row[f"{suffix}_bytes"] = target.stat().st_size
        if role == "C2":
            row["source_size_px"] = list(PANEL_B_CANDIDATE_SIZE)

    placements = copy.deepcopy(parent["placements"])
    c2 = [row for row in placements if row["role"] == "C2"]
    if len(c2) != 1:
        raise AssertionError("Expected one C2 placement")
    if c2[0]["placed_px"] != {
        "x": PANEL_B_PARENT_X_PX,
        "y": 76,
        "width": PANEL_B_PARENT_SIZE[0],
        "height": PANEL_B_PARENT_SIZE[1],
    }:
        raise AssertionError(f"Unexpected parent C2 placement: {c2[0]['placed_px']}")
    new_box = {
        "x": PANEL_B_CANDIDATE_X_PX,
        "y": 76,
        "width": PANEL_B_CANDIDATE_SIZE[0],
        "height": PANEL_B_CANDIDATE_SIZE[1],
    }
    c2[0]["allocated_cell_px"] = dict(new_box)
    c2[0]["placed_px"] = dict(new_box)

    assembly = load_module("ed09_margin_star_assembly", ROUND12_RENDERER)
    assembly.HERE = REPO
    assembly.CANVAS_PX = CANVAS_PX
    stem = (
        OUTPUT
        / "extended_data_figure9_panel_b_margin_panel_f_star_density_600dpi_candidate"
    )
    outputs = {
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
        "png": stem.with_suffix(".png"),
        "tiff": stem.with_suffix(".tiff"),
        "preview": OUTPUT / f"{stem.name}_300dpi.png",
    }
    assembly.MASTER_SVG = outputs["svg"]
    assembly.MASTER_PDF = outputs["pdf"]
    assembly.MASTER_PNG = outputs["png"]
    assembly.MASTER_TIFF = outputs["tiff"]
    assembly.MASTER_PREVIEW = outputs["preview"]
    assembly.assemble_raster(rows, placements)
    assembly.compose_master_svg(rows, placements)
    assembly.compose_master_pdf(rows, placements)
    # Round-20 opened these compounds in visible text, but the accepted F1
    # SVG retained one historical Dublin Core description.  Normalize only
    # the assembled human-readable metadata so the candidate fails closed on
    # stale terminology without changing any artwork group.
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    for old_term, new_term in (
        ("Place-map", "Place map"),
        ("place-map", "place map"),
        ("Level-map", "Level map"),
        ("level-map", "level map"),
    ):
        svg_text = svg_text.replace(old_term, new_term)
    outputs["svg"].write_text(svg_text, encoding="utf-8", newline="\n")
    return rows, placements, outputs


def other_panel_parity() -> dict[str, Any]:
    changed = {
        "ED9_C2_shift_distance",
        "ED9_B2_matched_coarse_controls",
        "ED9_E1_joint_place_level",
        "ED9_G1_consecutive_gap",
    }
    rows = []
    for parent in sorted((PARENT / "individual_panels").iterdir()):
        if not parent.is_file() or parent.stem in changed:
            continue
        candidate = PANEL_DIR / parent.name
        rows.append(
            {
                "file": parent.name,
                "byte_exact": candidate.is_file()
                and candidate.read_bytes() == parent.read_bytes(),
            }
        )
    return {
        "files": rows,
        "all_byte_exact": bool(rows) and all(row["byte_exact"] for row in rows),
    }


def main() -> int:
    copy_parent_panels()
    round6 = load_module("ed09_margin_star_round6", ROUND6_RENDERER)
    configure_style(round6)
    panel_b_paths, panel_b_audit = render_panel_b(round6)
    panel_f_paths, panel_f_audit = render_panel_f(round6)
    panel_g_paths, panel_g_audit = render_panel_g_dense_cloud(round6)
    panel_j_paths, panel_j_audit = render_panel_j_dense_cloud(round6)
    panel_rows, placements, root_outputs = assemble(
        panel_b_paths,
        panel_f_paths,
        panel_g_paths,
        panel_j_paths,
    )

    panel_b_label = pdf_text_span(panel_b_paths["pdf"], "Fraction of shifted cells")
    parent_composite_label = pdf_text_span(
        PARENT / "ed09_version_record.pdf", "Fraction of shifted cells"
    )
    candidate_composite_label = pdf_text_span(
        root_outputs["pdf"], "Fraction of shifted cells"
    )
    panel_f_stars = pdf_spans(panel_f_paths["pdf"], {"*", "**", "***"})
    composite_stars = pdf_spans(root_outputs["pdf"], {"*", "**", "***"})
    b2_placement = next(row for row in placements if row["role"] == "B2")["placed_px"]
    b2_rect = [
        b2_placement["x"] / DPI * 72.0,
        b2_placement["y"] / DPI * 72.0,
        (b2_placement["x"] + b2_placement["width"]) / DPI * 72.0,
        (b2_placement["y"] + b2_placement["height"]) / DPI * 72.0,
    ]
    panel_f_composite_stars = [
        row
        for row in composite_stars
        if b2_rect[0] <= (row["bbox_pt"][0] + row["bbox_pt"][2]) / 2 <= b2_rect[2]
        and b2_rect[1] <= (row["bbox_pt"][1] + row["bbox_pt"][3]) / 2 <= b2_rect[3]
    ]
    main1_stars = pdf_spans(
        MAIN1 / "figure1_version_record.pdf", {"*", "**", "***"}
    )
    parent_f = json.loads(PANEL_F_PARENT_RENDER.read_text(encoding="utf-8"))[
        "panel_f_audit"
    ]
    parent_brackets = _bracket_coordinates(parent_f)
    candidate_brackets = _bracket_coordinates(panel_f_audit)
    parity = other_panel_parity()
    parent_panel_g = PARENT / "individual_panels/ED9_E1_joint_place_level.png"
    parent_panel_j = PARENT / "individual_panels/ED9_G1_consecutive_gap.png"
    with Image.open(parent_panel_g) as image:
        parent_panel_g_pixels = np.asarray(image.convert("RGB"))
    with Image.open(panel_g_paths["png"]) as image:
        candidate_panel_g_pixels = np.asarray(image.convert("RGB"))
    with Image.open(parent_panel_j) as image:
        parent_panel_j_pixels = np.asarray(image.convert("RGB"))
    with Image.open(panel_j_paths["png"]) as image:
        candidate_panel_j_pixels = np.asarray(image.convert("RGB"))
    panel_g_parent_images = pdf_embedded_image_contract(
        parent_panel_g.with_suffix(".pdf")
    )
    panel_g_candidate_images = pdf_embedded_image_contract(panel_g_paths["pdf"])
    panel_j_parent_images = pdf_embedded_image_contract(
        parent_panel_j.with_suffix(".pdf")
    )
    panel_j_candidate_images = pdf_embedded_image_contract(panel_j_paths["pdf"])
    root_candidate_images = pdf_embedded_image_contract(root_outputs["pdf"])

    panel_b_axes = panel_b_audit["figure_state"]["axes_bounds_px"]
    panel_b_global_axes = [
        PANEL_B_CANDIDATE_X_PX + panel_b_axes[0],
        76 + panel_b_axes[1],
        panel_b_axes[2],
        panel_b_axes[3],
    ]
    expected_parent_global_axes = [
        PANEL_B_PARENT_X_PX + 0.24 * PANEL_B_PARENT_SIZE[0],
        76 + 0.24 * PANEL_B_PARENT_SIZE[1],
        (0.96 - 0.24) * PANEL_B_PARENT_SIZE[0],
        (0.94 - 0.24) * PANEL_B_PARENT_SIZE[1],
    ]

    checks = {
        "panel_b_parent_crop_pixel_exact": panel_b_audit[
            "parent_crop_pixel_exact"
        ],
        "panel_b_individual_ylabel_inset_at_least_2pt": panel_b_label["bbox_pt"][0]
        >= MIN_PANEL_B_PDF_INSET_PT,
        "panel_b_composite_ylabel_position_unchanged": all(
            math.isclose(left, right, abs_tol=0.01)
            for left, right in zip(
                parent_composite_label["bbox_pt"],
                candidate_composite_label["bbox_pt"],
                strict=True,
            )
        ),
        "panel_b_global_axes_geometry_unchanged": all(
            math.isclose(left, right, abs_tol=1e-6)
            for left, right in zip(
                panel_b_global_axes, expected_parent_global_axes, strict=True
            )
        ),
        "panel_b_limits_preserved": panel_b_audit["figure_state"]["x_limits"]
        == [0.75, 6.25]
        and all(
            math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(
                panel_b_audit["figure_state"]["y_limits"],
                [0.039657821138303885, 0.3691356943023306],
                strict=True,
            )
        ),
        "panel_b_neighbor_gaps_remain_25px": (
            PANEL_B_CANDIDATE_X_PX - (75 + 1400) == 25
            and 2325 - (PANEL_B_CANDIDATE_X_PX + PANEL_B_CANDIDATE_SIZE[0])
            == 25
        ),
        "panel_f_individual_two_exact_7pt_stars": len(panel_f_stars) == 2
        and all(math.isclose(row["size_pt"], 7.0, abs_tol=0.01) for row in panel_f_stars),
        "panel_f_composite_two_exact_7pt_stars": len(panel_f_composite_stars) == 2
        and all(
            math.isclose(row["size_pt"], 7.0, abs_tol=0.01)
            for row in panel_f_composite_stars
        ),
        "main1_composite_star_authority_is_7pt": len(main1_stars) >= 2
        and all(math.isclose(row["size_pt"], 7.0, abs_tol=0.01) for row in main1_stars),
        "panel_f_bracket_coordinates_exact": candidate_brackets == parent_brackets,
        "panel_f_axes_and_limits_preserved": all(
            math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(
                panel_f_audit["figure_state"]["axes_bounds_fraction"],
                [0.29, 0.24, 0.67, 0.70],
                strict=True,
            )
        )
        and panel_f_audit["figure_state"]["y_limits"] == [-0.22, 0.84],
        "panel_f_annotation_clearances_pass": all(
            panel_f_audit["annotation_geometry"]["checks"].values()
        ),
        "panel_f_q_values_exact": math.isclose(
            panel_f_audit["contrasts"][0]["q_value"],
            2.8670561454901405e-05,
            abs_tol=1e-18,
        )
        and math.isclose(
            panel_f_audit["contrasts"][1]["q_value"],
            1.6728044637612076e-06,
            abs_tol=1e-18,
        ),
        "panel_f_support_exact": panel_f_audit["source_pairs"] == 32
        and panel_f_audit["source_mice"] == 14,
        "panel_g_native_rerender_dimensions_preserved": (
            parent_panel_g_pixels.shape == candidate_panel_g_pixels.shape
        ),
        "panel_g_source_and_grammar_preserved": panel_g_audit["display_rows"]
        == 16901
        and panel_g_audit["source"]["sha256"]
        == "5d76a9e2c057abfd1e94b17ce53b3d30cea0e2835cf916047f42dbe1ad5b2426"
        and panel_g_audit["data_cloud"]
        == {
            "single_color": "#7976A2",
            "point_alpha": 0.035,
            "point_size": 2.0,
            "rasterized": True,
            "month_encoding": None,
            "legend": None,
        }
        and panel_g_audit["x_limits"] == [-0.65, 1.02]
        and panel_g_audit["y_limits"] == [-0.65, 1.02]
        and panel_g_audit["axis_labels"]
        == {
            "x": OPEN_PLACE_MAP_LABEL,
            "y": OPEN_LEVEL_MAP_LABEL,
            "x_color": "#000000",
            "y_color": "#000000",
        },
        "panel_g_embedded_image_upgraded_to_600dpi": len(
            panel_g_parent_images
        )
        == len(panel_g_candidate_images)
        == 1
        and panel_g_parent_images[0]["minimum_effective_dpi"] < 101.0
        and panel_g_candidate_images[0]["minimum_effective_dpi"] >= 599.0,
        "panel_j_png_pixel_exact_parent": np.array_equal(
            parent_panel_j_pixels, candidate_panel_j_pixels
        ),
        "panel_j_source_and_grammar_preserved": panel_j_audit["display_rows"]
        == 9661
        and panel_j_audit["fit_rows"] == 9661
        and panel_j_audit["fit_animals"] == 11
        and panel_j_audit["figure_state"]["x_limits"] == [-1.0, 1.0]
        and panel_j_audit["figure_state"]["y_limits"] == [-1.0, 1.0],
        "panel_j_embedded_image_upgraded_to_600dpi": len(
            panel_j_parent_images
        )
        == len(panel_j_candidate_images)
        == 1
        and panel_j_parent_images[0]["minimum_effective_dpi"] < 101.0
        and panel_j_candidate_images[0]["minimum_effective_dpi"] >= 599.0,
        "root_retains_only_two_sub600_data_or_cloud_images": sum(
            row["minimum_effective_dpi"] < 599.0
            for row in root_candidate_images
        )
        == 2,
        "all_other_individual_panel_files_byte_exact": parity["all_byte_exact"],
        "root_raster_dimensions_exact": all(
            Image.open(root_outputs[key]).size == CANVAS_PX
            for key in ("png", "tiff")
        ),
        "root_pdf_one_page": fitz.open(root_outputs["pdf"]).page_count == 1,
        "root_svg_editable_text": "<text" in root_outputs["svg"].read_text(
            encoding="utf-8"
        ),
        "root_svg_has_open_map_compounds": OPEN_PLACE_MAP_LABEL
        in root_outputs["svg"].read_text(encoding="utf-8")
        and OPEN_LEVEL_MAP_LABEL
        in root_outputs["svg"].read_text(encoding="utf-8"),
        "root_svg_has_no_forbidden_hyphenated_map_terms": all(
            token not in root_outputs["svg"].read_text(encoding="utf-8")
            for token in ("Place-map", "place-map", "Level-map", "level-map")
        ),
        "root_pdf_has_no_forbidden_hyphenated_map_terms": all(
            token not in "".join(
                page.get_text() for page in fitz.open(root_outputs["pdf"])
            )
            for token in ("Place-map", "place-map", "Level-map", "level-map")
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    manifest = {
        "schema": "ed09-panel-b-margin-panel-f-star-density600-candidate/v1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "promoted": False,
        "parent_record": record(PARENT / "RECORD_HASHES.csv"),
        "renderer": record(Path(__file__)),
        "changes": {
            "C2": {
                "individual_canvas_px": list(PANEL_B_CANDIDATE_SIZE),
                "left_padding_px": PANEL_B_LEFT_PAD_PX,
                "composite_placement_x_px": PANEL_B_CANDIDATE_X_PX,
                "physical_axes_and_composite_coordinates_changed": False,
            },
            "B2": {
                "star_fontsize_pt": PANEL_F_STAR_SIZE_PT,
                "authority": "Main Figure 1 composite bracket-scatter stars",
                "bracket_coordinates_changed": False,
                "axes_or_limits_changed": False,
            },
            "E1": {
                "display_panel": "g",
                "embedded_raster_target_dpi": DPI,
                "scientific_values_or_geometry_changed": False,
            },
            "G1": {
                "display_panel": "j",
                "embedded_raster_target_dpi": DPI,
                "scientific_values_or_geometry_changed": False,
            },
        },
        "panel_b_audit": panel_b_audit,
        "panel_f_audit": panel_f_audit,
        "panel_g_audit": panel_g_audit,
        "panel_j_audit": panel_j_audit,
        "measurements": {
            "panel_b_ylabel_individual": panel_b_label,
            "panel_b_ylabel_parent_composite": parent_composite_label,
            "panel_b_ylabel_candidate_composite": candidate_composite_label,
            "panel_b_global_axes_candidate_px": panel_b_global_axes,
            "panel_b_global_axes_parent_expected_px": expected_parent_global_axes,
            "panel_f_individual_stars": panel_f_stars,
            "panel_f_composite_stars": panel_f_composite_stars,
            "main1_composite_star_authority": main1_stars,
            "panel_f_parent_brackets": parent_brackets,
            "panel_f_candidate_brackets": candidate_brackets,
            "panel_g_parent_embedded_images": panel_g_parent_images,
            "panel_g_candidate_embedded_images": panel_g_candidate_images,
            "panel_j_parent_embedded_images": panel_j_parent_images,
            "panel_j_candidate_embedded_images": panel_j_candidate_images,
            "root_candidate_embedded_images": root_candidate_images,
        },
        "other_panel_parity": parity,
        "panels": panel_rows,
        "placements": placements,
        "outputs": {key: record(path) for key, path in root_outputs.items()},
        "changed_panel_outputs": {
            "C2": {key: record(path) for key, path in panel_b_paths.items()},
            "B2": {key: record(path) for key, path in panel_f_paths.items()},
            "E1": {key: record(path) for key, path in panel_g_paths.items()},
            "G1": {key: record(path) for key, path in panel_j_paths.items()},
        },
        "checks": checks,
    }
    manifest_path = OUTPUT / "CANDIDATE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if status != "PASS":
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"ED9 candidate validation failed: {failed}")
    print(
        json.dumps(
            {
                "status": status,
                "output": OUTPUT.relative_to(REPO).as_posix(),
                "panel_b_ylabel_x0_pt": panel_b_label["bbox_pt"][0],
                "panel_f_star_sizes_pt": [row["size_pt"] for row in panel_f_stars],
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
