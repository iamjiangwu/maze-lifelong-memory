"""Render Extended Data Figure 5 from retained maps and session-pair values.

The gallery contains 32 registered-neuron examples. Quantitative panels compare
the four day combinations for place maps, population vectors and ensemble rates,
and maze-versus-outside activity for 36 transitions from 14 animals. Session pairs
receive equal weight; intervals account for repeated pairs from the same animal.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ed_supplement_common import (  # noqa: E402
    AXIS_COLOR,
    MAIN_AXIS_LABEL_SIZE,
    MUTED_COLOR,
    Canvas,
    Checks,
    sha256,
)
from scripts import ed05_gallery_plotting as gallery

# The gallery heatmaps retain the high-resolution 45 x 45 display grid, but the
# printed effect size must use the same 15 x 15 place-map estimator as Main
# Figure 2 and the quantitative ED5 analyses.
GALLERY_CORRELATION_COLUMN = "place_r_15_analysis"
GALLERY_CORRELATION_GRID = (15, 15)
GALLERY_DISPLAY_GRID = (45, 45)

# Panel geometry uses a top-left page origin, in inches.
SLIDE_W_IN = 7.5
SLIDE_H_IN = 7.5
SLIDE_GALLERY = {
    "left": 0.0,
    "top": -0.004657699037620298,
    "width": 4.981503718285214,
    "height": 5.635324803149606,
}
SLIDE_DAYCOMB = {
    "left": 1.0936132983377078e-06,
    "top": 5.630669291338583,
    "width": 3.75,
    "height": 1.2129024496937884,
}
SLIDE_EQUAL = {
    "left": 3.75,
    "top": 5.357315179352581,
    "width": 1.1309820647419073,
    "height": 1.48625656167979,
}

# Content bbox uses gallery + day-combination band. Panel e is height-matched
# to b-d; the empty 7.5×7.5 page margin is dropped.
_CONTENT_LEFT = min(SLIDE_GALLERY["left"], SLIDE_DAYCOMB["left"], SLIDE_EQUAL["left"])
_CONTENT_RIGHT = max(
    SLIDE_GALLERY["left"] + SLIDE_GALLERY["width"],
    SLIDE_DAYCOMB["left"] + SLIDE_DAYCOMB["width"],
    SLIDE_EQUAL["left"] + SLIDE_EQUAL["width"],
)
_CONTENT_TOP = SLIDE_GALLERY["top"]
_CONTENT_BOTTOM = max(
    SLIDE_GALLERY["top"] + SLIDE_GALLERY["height"],
    SLIDE_DAYCOMB["top"] + SLIDE_DAYCOMB["height"],
)

# Gallery flush left; "correlation" ylabel draws in a local inset of panel b only.
PAD_LEFT = 0.02
PAD_RIGHT = 0.02
# Minimal top margin; panel a sits beside the in-gallery top colorbar.
PAD_TOP = 0.015
# Room under shared b–e x-tick labels (Outside/Maze are taller than d1-d1).
PAD_BOTTOM = 0.08
# Shared y for b–e.
SHARED_YMAX = 0.8
# Must match build_ed05_gallery_figure colorbar axes [..., y0, ..., height].
GALLERY_COLORBAR_Y0 = 0.968
GALLERY_COLORBAR_HEIGHT = 0.008

FIGURE_WIDTH_IN = (_CONTENT_RIGHT - _CONTENT_LEFT) + PAD_LEFT + PAD_RIGHT
FIGURE_HEIGHT_IN = (_CONTENT_BOTTOM - _CONTENT_TOP) + PAD_TOP + PAD_BOTTOM
# Keep the shared Extended Data printed-size typography (0.75× on a 7.5 in width).
FONT_REFERENCE_WIDTH_IN = 7.5
# Nudge b–e one axis-label character right so "correlation" is not clipped.
ROW_BE_SHIFT_IN = MAIN_AXIS_LABEL_SIZE * (FONT_REFERENCE_WIDTH_IN / 10.0) / 72.0
PAIR_CARD_EDGE = "#B8B8B8"
PAIR_CARD_LINEWIDTH = 0.45
# Thicker red outline for the four Main Figure 2b example pairs.
MAIN_FIG_PAIR_COLOR = "#B64342"
MAIN_FIG_PAIR_LINEWIDTH = 1.10
# The gallery is a source-rendered image plate inside otherwise-vector PDF/SVG
# composites.  Matplotlib defaults embedded images in vector exports to 100 dpi;
# explicitly retain publication-resolution sampling at the final placed size.
COMPOSITE_EMBEDDED_RASTER_DPI = 600


def _slide_to_canvas(left: float, top: float, width: float, height: float) -> dict[str, float]:
    """Convert a top-left page rectangle into canvas lower-left inches."""

    return {
        "left": (left - _CONTENT_LEFT) + PAD_LEFT,
        "bottom": FIGURE_HEIGHT_IN
        - PAD_TOP
        - ((top - _CONTENT_TOP) + height),
        "width": width,
        "height": height,
    }


GALLERY_BOX = _slide_to_canvas(**SLIDE_GALLERY)
DAYCOMB_BOX = _slide_to_canvas(**SLIDE_DAYCOMB)
GALLERY_LEFT = GALLERY_BOX["left"]
GALLERY_BOTTOM = GALLERY_BOX["bottom"]
GALLERY_WIDTH = GALLERY_BOX["width"]
GALLERY_HEIGHT = GALLERY_BOX["height"]
DAYCOMB_LEFT = DAYCOMB_BOX["left"] + ROW_BE_SHIFT_IN
DAYCOMB_BOTTOM = DAYCOMB_BOX["bottom"]
DAYCOMB_WIDTH = DAYCOMB_BOX["width"]
DAYCOMB_HEIGHT = DAYCOMB_BOX["height"]
# Panel e shares the b-d band (same bottom/height); keep the slide left/width.
EQUAL_LEFT = (SLIDE_EQUAL["left"] - _CONTENT_LEFT) + PAD_LEFT + ROW_BE_SHIFT_IN
EQUAL_BOTTOM = DAYCOMB_BOTTOM
EQUAL_WIDTH = SLIDE_EQUAL["width"]
EQUAL_HEIGHT = DAYCOMB_HEIGHT
EQUAL_BOX = {
    "left": EQUAL_LEFT,
    "bottom": EQUAL_BOTTOM,
    "width": EQUAL_WIDTH,
    "height": EQUAL_HEIGHT,
}

VARIANT_ORDER = ("d1v1", "d1v2", "d2v1", "d2v2")
VARIANT_TICK_LABELS = {
    "d1v1": "d1-d1",
    "d1v2": "d1-d2",
    "d2v1": "d2-d1",
    "d2v2": "d2-d2",
}
METRIC_SPECS = (
    (
        "place_map_correlation",
        "Place-map correlation",
        "#B95A58",
    ),
    (
        "pv_correlation",
        "PV correlation",
        "#E29957",
    ),
    (
        "ensemble_rate_correlation",
        "Ensemble rate correlation",
        "#4A5E65",
    ),
)

# Change visual slots only. Keep the original metric drawing order below so
# every metric consumes the same random jitter as the preceding source figure.
DAY_COMB_DISPLAY_ORDER = (
    "pv_correlation",
    "place_map_correlation",
    "ensemble_rate_correlation",
)

# The build driver binds these to the included final source tables.
DAY_COMB_STATS: Path
DAY_COMB_DESC: Path
EQUAL_PAIRS: Path
EQUAL_STATS: Path
EQUAL_EXTRA_SOURCES: tuple[tuple[str, Path], ...] = ()
EQUAL_CAPTION_PANEL_NOTE = (
    "- **e.** Equal-time ensemble-rate Outside versus Maze for Day1→Day1 "
    "(36 transition rows / 14 mice; CR1 means ± 95% CI)."
)
EQUAL_CAPTION_STATS_LABEL = "Equal-time Day1→Day1 maze−outside"
EQUAL_MANIFEST_CORRECTION = (
    "panel e height-matched to b-d; ylim before bracket so stars clear the scatter"
)
PLACE_PAIR_SOURCE: Path
PV_PAIR_SOURCE: Path
RATE_PAIR_SOURCE: Path
SELECTED_32: Path
SHORTLIST_NPZ: Path


def load_shortlist_maps() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    selected = pd.read_csv(SELECTED_32, float_precision="round_trip")
    selected = selected.sort_values("visual_rank", kind="mergesort").reset_index(drop=True)
    if len(selected) != 32:
        raise ValueError(f"Expected 32 shortlist pairs; found {len(selected)}")
    required_correlations = {GALLERY_CORRELATION_COLUMN}
    missing_correlations = required_correlations.difference(selected.columns)
    if missing_correlations:
        raise ValueError(
            "Shortlist lacks required correlation columns: "
            f"{sorted(missing_correlations)}"
        )
    if not np.isfinite(selected[list(required_correlations)].to_numpy(float)).all():
        raise ValueError("Shortlist correlation labels contain non-finite values")
    with np.load(SHORTLIST_NPZ, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    npz_ids = [str(x) for x in arrays["candidate_id"]]
    selected_ids = selected["candidate_id"].astype(str).tolist()
    if npz_ids != selected_ids:
        raise ValueError("Shortlist NPZ candidate order does not match selected_32.csv")
    for key in ("normalized_a", "normalized_b"):
        if arrays[key].shape != (32, 45, 45):
            raise ValueError(f"Unexpected gallery array shape for {key}")
    return selected, {
        candidate_id: {
            "normalized_a": arrays["normalized_a"][index],
            "normalized_b": arrays["normalized_b"][index],
        }
        for index, candidate_id in enumerate(selected_ids)
    }


def build_ed05_gallery_figure(
    selected: pd.DataFrame,
    maps: dict[str, dict[str, Any]],
) -> plt.Figure:
    """ED5 gallery: maps + r; red card edges mark Main Figure 2b examples."""

    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    # Axis-label point size matched to the shared ED contract (0.75 × Main Fig 1).
    colorbar_label_size = MAIN_AXIS_LABEL_SIZE * (FONT_REFERENCE_WIDTH_IN / 10.0)
    # Keep figsize aspect matched to the slide gallery slot (fill_slot contract).
    # Thin colorbar sits in the top horizontal band; maps use the remaining card area.
    figure = plt.figure(figsize=(8.0, 9.05), facecolor="white")
    outer = figure.add_gridspec(
        8,
        4,
        left=0.002,
        right=0.998,
        top=0.955,
        bottom=0.008,
        wspace=0.028,
        hspace=0.040,
    )
    for position, row in enumerate(selected.itertuples(index=False)):
        pair_row, pair_column = divmod(position, 4)
        slot = outer[pair_row, pair_column]
        slot_box = slot.get_position(figure)
        header_frac = 0.12
        inset = 0.022 * min(slot_box.width, slot_box.height)
        map_y0 = slot_box.y0 + inset
        map_y1 = slot_box.y1 - header_frac * slot_box.height
        map_height = max(map_y1 - map_y0, 1e-6)
        inner_width = max(slot_box.width - 2.0 * inset, 1e-6)
        gap = 0.012 * slot_box.width
        each_w = (inner_width - gap) / 2.0
        x0 = slot_box.x0 + inset
        values = maps[str(row.candidate_id)]
        for within_pair_column, side_name in enumerate(("a", "b")):
            axis = figure.add_axes(
                [x0 + within_pair_column * (each_w + gap), map_y0, each_w, map_height]
            )
            gallery._draw_map(axis, np.asarray(values[f"normalized_{side_name}"]))
            # Show the complete 45 x 45 map.  The former 2.0--42.5 zoom clipped
            # the outer map bins on every card even though the source arrays
            # were complete.
            axis.set_xlim(-0.5, 44.5)
            axis.set_ylim(44.5, -0.5)
            axis.set_aspect("auto")
            axis.set_clip_on(True)
        header_y = map_y1 + 0.55 * (slot_box.y1 - map_y1)
        figure.text(
            0.5 * (slot_box.x0 + slot_box.x1),
            header_y,
            gallery._format_place_map_correlation(
                getattr(row, GALLERY_CORRELATION_COLUMN)
            ),
            ha="center",
            va="center",
            fontsize=5.0,
            color="#303030",
            clip_on=False,
            zorder=21,
        )
        is_main_figure_pair = bool(getattr(row, "fixed_main_fig2b", False))
        card = Rectangle(
            (slot_box.x0, slot_box.y0),
            slot_box.width,
            slot_box.height,
            transform=figure.transFigure,
            facecolor="none",
            edgecolor=MAIN_FIG_PAIR_COLOR if is_main_figure_pair else PAIR_CARD_EDGE,
            linewidth=(
                MAIN_FIG_PAIR_LINEWIDTH if is_main_figure_pair else PAIR_CARD_LINEWIDTH
            ),
            joinstyle="miter",
            clip_on=False,
            zorder=20,
        )
        figure.add_artist(card)

    # Thin horizontal colorbar in the top empty band (above the 8×4 grid).
    colorbar_axis = figure.add_axes([0.34, 0.968, 0.32, 0.008])
    colorbar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=gallery._display_cmap()),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.ax.xaxis.set_ticks_position("bottom")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(
        labelsize=max(colorbar_label_size, 7.0) * 0.85,
        length=1.5,
        pad=1.0,
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
    )
    colorbar.set_label(
        "Normalized rate",
        fontsize=max(colorbar_label_size, 7.0),
        labelpad=1.5,
    )
    return figure


def render_gallery_plate(output_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Re-render the compact 32-pair plate into svg/png/tiff (+pdf)."""
    selected, maps = load_shortlist_maps()
    # Reorder paired cards only, after validating the original table/NPZ order.
    # The displayed analysis correlation is full precision; candidate_id breaks
    # exact ties deterministically. Each source/target map pair stays together.
    selected = selected.sort_values(
        [GALLERY_CORRELATION_COLUMN, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    derived = output_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    figure = build_ed05_gallery_figure(selected, maps)
    map_axes = [axis for axis in figure.axes if axis.images]
    map_bounds = [
        [float(value) for value in (*axis.get_xlim(), *axis.get_ylim())]
        for axis in map_axes
    ]
    full_map_contract = {
        "map_axis_count": len(map_axes),
        "source_array_shapes": sorted(
            {
                tuple(np.asarray(image.get_array()).shape)
                for axis in map_axes
                for image in axis.images
            }
        ),
        "axis_bounds": map_bounds,
        "all_45x45_sources": all(
            np.asarray(image.get_array()).shape == (45, 45)
            for axis in map_axes
            for image in axis.images
        ),
        "all_maps_fully_shown": all(
            np.allclose(bounds, [-0.5, 44.5, 44.5, -0.5])
            for bounds in map_bounds
        ),
    }
    stem = derived / "ed05_compact_32_grid_8x8"
    outputs: dict[str, Path] = {}
    try:
        for extension in ("svg", "png", "tiff", "pdf"):
            path = stem.with_suffix(f".{extension}")
            kwargs: dict[str, Any] = {"facecolor": "white"}
            if extension in {"png", "tiff"}:
                kwargs["dpi"] = 400
            if extension == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            figure.savefig(path, **kwargs)
            outputs[extension] = path
    finally:
        plt.close(figure)
    selected.to_csv(derived / "ed05_gallery_examples.csv", index=False)
    order_audit = selected[["candidate_id", "visual_rank", GALLERY_CORRELATION_COLUMN]].copy()
    order_audit = order_audit.rename(columns={"visual_rank": "previous_position"})
    order_audit.insert(0, "display_position", np.arange(1, len(order_audit) + 1))
    order_audit["display_row"] = (order_audit["display_position"] - 1) // 4 + 1
    order_audit["display_column"] = (order_audit["display_position"] - 1) % 4 + 1
    order_audit["previous_row"] = (order_audit["previous_position"] - 1) // 4 + 1
    order_audit["previous_column"] = (order_audit["previous_position"] - 1) % 4 + 1
    order_audit["displayed_label"] = selected[GALLERY_CORRELATION_COLUMN].map(
        gallery._format_place_map_correlation
    )
    order_audit_path = derived / "ED05_PANEL_A_DISPLAY_ORDER.csv"
    order_audit.to_csv(order_audit_path, index=False, float_format="%.17g")
    label_audit = selected.reindex(columns=
        [
            "visual_rank",
            "candidate_id",
            "animal_id",
            "pair_key",
            "train_id",
            "test_id",
            "fixed_main_fig2b",
            GALLERY_CORRELATION_COLUMN,
        ]
    ).copy()
    label_audit = label_audit.rename(
        columns={
            GALLERY_CORRELATION_COLUMN: "displayed_r_15x15_analysis",
        }
    )
    label_audit["display_heatmap_grid"] = "45x45"
    label_audit["correlation_grid"] = "15x15"
    label_audit_path = derived / "ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv"
    label_audit.to_csv(label_audit_path, index=False)
    full_map_contract.update(
        {
            "display_grid_shape": list(GALLERY_DISPLAY_GRID),
            "correlation_label_grid_shape": list(GALLERY_CORRELATION_GRID),
            "correlation_label_column": GALLERY_CORRELATION_COLUMN,
            "correlation_label_audit": _repo_rel(label_audit_path),
            "display_order": "decreasing full-precision correlation; left to right, then top to bottom",
            "display_order_tie_break": "candidate_id ascending",
            "display_order_audit": _repo_rel(order_audit_path),
        }
    )
    return outputs, full_map_contract


def _check_composite_embedded_resolution(
    outputs: list[Path], checks: Checks
) -> None:
    """Gate the sole panel-a image at its effective placed PDF/SVG resolution."""

    pdf_path = next(path for path in outputs if path.suffix == ".pdf")
    with fitz.open(pdf_path) as document:
        page = document[0]
        images = page.get_images(full=True)
        checks.record(
            "pdf_only_panel_a_is_raster",
            len(images) == 1,
            f"embedded_images={len(images)}; vector_drawings={len(page.get_drawings())}",
        )
        if len(images) == 1:
            xref = int(images[0][0])
            pixmap = fitz.Pixmap(document, xref)
            rects = page.get_image_rects(xref)
            if len(rects) == 1:
                rect = rects[0]
                dpi_x = float(pixmap.width) / (float(rect.width) / 72.0)
                dpi_y = float(pixmap.height) / (float(rect.height) / 72.0)
                checks.record(
                    "pdf_panel_a_effective_dpi_at_least_600",
                    min(dpi_x, dpi_y) >= COMPOSITE_EMBEDDED_RASTER_DPI - 1.0,
                    (
                        f"{pixmap.width}x{pixmap.height} px in "
                        f"{rect.width:.2f}x{rect.height:.2f} pt; "
                        f"effective_dpi={dpi_x:.3f}x{dpi_y:.3f}"
                    ),
                )
            else:
                checks.record(
                    "pdf_panel_a_effective_dpi_at_least_600",
                    False,
                    f"image_rect_count={len(rects)}",
                )
        else:
            checks.record(
                "pdf_panel_a_effective_dpi_at_least_600",
                False,
                "requires exactly one embedded image",
            )

    svg_path = next(path for path in outputs if path.suffix == ".svg")
    root = ET.parse(svg_path).getroot()
    svg_images = [element for element in root.iter() if element.tag.endswith("}image")]
    checks.record(
        "svg_only_panel_a_is_raster",
        len(svg_images) == 1,
        f"embedded_images={len(svg_images)}",
    )
    if len(svg_images) == 1:
        href = (
            svg_images[0].attrib.get("{http://www.w3.org/1999/xlink}href")
            or svg_images[0].attrib.get("href")
            or ""
        )
        try:
            payload = base64.b64decode(href.split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as embedded:
                px_width, px_height = embedded.size
            width_pt = float(svg_images[0].attrib["width"])
            height_pt = float(svg_images[0].attrib["height"])
            dpi_x = float(px_width) / (width_pt / 72.0)
            dpi_y = float(px_height) / (height_pt / 72.0)
            checks.record(
                "svg_panel_a_effective_dpi_at_least_600",
                min(dpi_x, dpi_y) >= COMPOSITE_EMBEDDED_RASTER_DPI - 1.0,
                (
                    f"{px_width}x{px_height} px in "
                    f"{width_pt:.2f}x{height_pt:.2f} pt; "
                    f"effective_dpi={dpi_x:.3f}x{dpi_y:.3f}"
                ),
            )
        except (IndexError, KeyError, ValueError, OSError) as error:
            checks.record(
                "svg_panel_a_effective_dpi_at_least_600",
                False,
                f"cannot decode embedded gallery: {error}",
            )
    else:
        checks.record(
            "svg_panel_a_effective_dpi_at_least_600",
            False,
            "requires exactly one embedded image",
        )




def _metric_contrast_rows(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    subset = stats.loc[
        stats["metric"].eq(metric)
        & stats["family"].eq("d1v1_pairwise")
        & stats["test"].eq("session-pair contrast; CR1 clustered by animal")
    ].copy()
    return subset


def _load_pair_reaggregation_module():
    path = Path(__file__).resolve().with_name(
        "recompute_ed05_pair_weighted_day_combinations.py"
    )
    spec = importlib.util.spec_from_file_location("ed05_pair_reaggregation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _draw_significant_d1v1_brackets(
    canvas: Canvas,
    ax: plt.Axes,
    stats: pd.DataFrame,
    metric: str,
    *,
    data_top: float,
    raise_above_data: bool = False,
) -> int:
    """Bracket d1v1 versus each other variant when within-metric BH3 p < 0.05."""

    variant_index = {variant: index for index, variant in enumerate(VARIANT_ORDER)}
    rows = _metric_contrast_rows(stats, metric)
    brackets: list[tuple[float, float, float, int]] = []
    for row in rows.itertuples(index=False):
        contrast = str(row.contrast)
        if not contrast.startswith("d1v1_minus_"):
            continue
        other = contrast.replace("d1v1_minus_", "", 1)
        if other not in variant_index:
            continue
        p_value = float(row.p_bh3_within_metric_exact_cluster_signflip)
        if canvas.significance_stars(p_value) == "n.s.":
            continue
        span = variant_index[other] - variant_index["d1v1"]
        brackets.append((float(variant_index["d1v1"]), float(variant_index[other]), p_value, span))
    # Shorter spans sit lower so nested brackets remain readable.
    brackets.sort(key=lambda item: (item[3], item[1]))
    low, high = ax.get_ylim()
    axis_span = high - low
    # Same nest spacing as panel c; stars pulled further down onto the tick line.
    # Double the prior -0.012 offset.
    star_offset_frac = -0.024
    star_fontsize = 8.0 * canvas.font_scale
    step_frac = 0.075
    if raise_above_data:
        # Panel d: raise the stack above scatter; may extend past ymax (clip_on=False).
        base_frac = (float(data_top) - low) / axis_span + 0.045
    else:
        base_frac = min(0.62, max(0.48, (float(data_top) - low) / axis_span + 0.022))
    drawn = 0
    for tier, (x0, x1, p_value, _) in enumerate(brackets):
        # No y_cap: keep uniform step_frac (panel c spacing) even above ymax.
        y_frac = base_frac + step_frac * tier
        if (
            canvas.significance_bracket(
                ax,
                x0,
                x1,
                p_value,
                y_frac=y_frac,
                skip_ns=True,
                star_offset_frac=star_offset_frac,
                star_fontsize=star_fontsize,
            )
            is not None
        ):
            drawn += 1
    return drawn


def draw_day_combination_row(
    canvas: Canvas,
    pair_values: pd.DataFrame,
    descriptive: pd.DataFrame,
    stats: pd.DataFrame,
    checks: Checks,
) -> None:
    # Local inset only on panel b for the shared ylabel; gallery stays flush left.
    ylabel_inset = 0.18
    panel_gap = 0.06
    usable_width = DAYCOMB_WIDTH - ylabel_inset - 0.02
    panel_width = (usable_width - 2.0 * panel_gap) / 3.0
    # Top band for panel letters above the axes (va='bottom').
    letter_room = 0.18
    plot_bottom = DAYCOMB_BOTTOM + 0.11
    plot_height = DAYCOMB_HEIGHT - 0.11 - letter_room
    rng = np.random.default_rng(20260806)
    letters = ("b", "c", "d")
    for index, ((metric, _title, color), letter) in enumerate(
        zip(METRIC_SPECS, letters)
    ):
        display_index = DAY_COMB_DISPLAY_ORDER.index(metric)
        letter = letters[display_index]
        left = DAYCOMB_LEFT + ylabel_inset + display_index * (panel_width + panel_gap)
        ax = canvas.axes(left, plot_bottom, panel_width, plot_height)
        subset = pair_values.loc[pair_values["metric"].eq(metric)]
        data_top = 0.0
        for v_index, variant in enumerate(VARIANT_ORDER):
            values = subset.loc[subset["variant"].eq(variant), "value"].to_numpy(
                dtype=float
            )
            if len(values) != 36:
                raise ValueError(
                    f"{metric}/{variant} expected 36 session pairs; found {len(values)}"
                )
            summary = descriptive.loc[
                descriptive["metric"].eq(metric)
                & descriptive["variant"].eq(variant)
            ]
            if len(summary) != 1:
                raise ValueError(f"Expected one summary row for {metric}/{variant}")
            mean = float(summary["mean"].iloc[0])
            ci_low = float(summary["ci95_low_cr1"].iloc[0])
            ci_high = float(summary["ci95_high_cr1"].iloc[0])
            data_top = max(data_top, float(np.max(values)), float(ci_high))
            canvas.bar_scatter_group(
                ax, v_index, values, mean, ci_low, ci_high, color, rng
            )
        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels([VARIANT_TICK_LABELS[v] for v in VARIANT_ORDER])
        ax.set_ylim(0.0, SHARED_YMAX)
        ax.tick_params(axis="both", length=2.0, pad=0.8)
        for label in ax.get_xticklabels():
            label.set_clip_on(False)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        if display_index == 0:
            ax.set_ylabel("correlation", labelpad=1.0)
        else:
            ax.set_yticklabels([])
        n_brackets = _draw_significant_d1v1_brackets(
            canvas,
            ax,
            stats,
            metric,
            data_top=data_top,
            # Panel d (ensemble rate): raise ticks+stars together above scatter.
            raise_above_data=metric == "ensemble_rate_correlation",
        )
        checks.record(
            f"daycomb_{metric}_n_animals",
            subset["animal_id"].nunique() == 14,
            f"{subset['animal_id'].nunique()} mice",
        )
        checks.record(
            f"daycomb_{metric}_sig_brackets",
            True,
            f"{n_brackets} significant brackets (n.s. omitted)",
        )
        # Bottom of the letter clear of the axes top / 0.8 tick.
        canvas.panel_letter(
            left - 0.05,
            plot_bottom + plot_height + 0.045,
            letter,
            va="bottom",
        )
    checks.record(
        "daycomb_slot_matches_pptx",
        abs(DAYCOMB_WIDTH - SLIDE_DAYCOMB["width"]) < 1e-9
        and abs(DAYCOMB_HEIGHT - SLIDE_DAYCOMB["height"]) < 1e-9,
        f"{DAYCOMB_WIDTH:.4f}x{DAYCOMB_HEIGHT:.4f} in",
    )


def draw_equal_time_panel(
    canvas: Canvas,
    pairs: pd.DataFrame,
    means: pd.DataFrame,
    stats: pd.DataFrame,
    checks: Checks,
) -> None:
    # Height-matched to b-d; share the same y-axis scale and omit a second ylabel.
    letter_room = 0.18
    plot_bottom = DAYCOMB_BOTTOM + 0.11
    plot_height = DAYCOMB_HEIGHT - 0.11 - letter_room
    plot_left = EQUAL_LEFT + 0.12
    ax = canvas.axes(
        plot_left,
        plot_bottom,
        EQUAL_WIDTH - 0.14,
        plot_height,
    )
    d1 = pairs.loc[pairs["variant"].eq("d1v1")].copy()
    if len(d1) != 36:
        raise ValueError(f"Expected 36 d1v1 equal-time rows; found {len(d1)}")
    outside = d1["equal_time_outside_r_mean"].to_numpy(dtype=float)
    maze = d1["equal_time_maze_r_mean"].to_numpy(dtype=float)
    for x0, x1 in zip(outside, maze):
        ax.plot(
            [0.0, 1.0],
            [x0, x1],
            color="#B0B0B0",
            alpha=0.35,
            linewidth=0.55,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(outside)),
        outside,
        s=10.0 * canvas.font_scale**2,
        color="#9A9A9A",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )
    ax.scatter(
        np.ones(len(maze)),
        maze,
        s=10.0 * canvas.font_scale**2,
        color="#4A5E65",
        alpha=0.65,
        edgecolor="none",
        zorder=2,
    )
    for x, condition, color in (
        (0.0, "outside", "#9A9A9A"),
        (1.0, "maze", "#4A5E65"),
    ):
        row = means.loc[
            means["variant"].eq("d1v1") & means["condition"].eq(condition)
        ].iloc[0]
        ax.errorbar(
            x,
            float(row["beta"]),
            yerr=np.array(
                [
                    [float(row["beta"] - row["ci95_low"])],
                    [float(row["ci95_high"] - row["beta"])],
                ]
            ),
            fmt="o",
            color="black",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.2 * canvas.font_scale,
            capsize=2.2 * canvas.font_scale,
            elinewidth=0.9 * canvas.font_scale,
            zorder=4,
        )
    test = stats.loc[
        stats["analysis"].eq("maze_minus_outside") & stats["variant"].eq("d1v1")
    ].iloc[0]
    # Same x-tick grammar as panels b–d (pad/length/clip) so labels are not cut off.
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Outside", "Maze"])
    ax.set_ylim(0.0, SHARED_YMAX)
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=2.0, pad=0.8)
    for label in ax.get_xticklabels():
        label.set_clip_on(False)
    data_top = float(max(np.max(outside), np.max(maze)))
    # Bracket just above the scatter; stars further down (2× prior offset).
    y_frac = min(0.96, data_top / SHARED_YMAX + 0.02)
    canvas.significance_bracket(
        ax,
        0.0,
        1.0,
        float(test["pvalue"]),
        y_frac=y_frac,
        skip_ns=True,
        star_offset_frac=-0.024,
        star_fontsize=8.0 * canvas.font_scale,
    )
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    checks.record(
        "equal_time_d1v1_rows",
        len(d1) == 36 and d1["animal_id"].nunique() == 14,
        f"{len(d1)} rows / {d1['animal_id'].nunique()} mice",
    )
    checks.record(
        "equal_time_d1v1_p",
        float(test["pvalue"]) < 0.05,
        f"p={float(test['pvalue']):.6g}",
    )
    checks.record(
        "equal_height_matches_bcd",
        abs(EQUAL_HEIGHT - DAYCOMB_HEIGHT) < 1e-9
        and abs(EQUAL_BOTTOM - DAYCOMB_BOTTOM) < 1e-9,
        f"{EQUAL_WIDTH:.4f}x{EQUAL_HEIGHT:.4f} in at bottom {EQUAL_BOTTOM:.4f}",
    )
    canvas.panel_letter(
        plot_left - 0.05,
        plot_bottom + plot_height + 0.045,
        "e",
        va="bottom",
    )


def assemble(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = Checks()

    reaggregation = _load_pair_reaggregation_module()
    pair_values, source_checks = reaggregation.load_pair_values(PLACE_PAIR_SOURCE.parent)
    day_stats = pd.read_csv(DAY_COMB_STATS)
    bh_stats_path = output_dir / "ED05_DAY_COMBINATION_BH3.csv"
    day_stats.to_csv(bh_stats_path, index=False)
    day_desc = pd.read_csv(DAY_COMB_DESC)
    pairs = pd.read_csv(EQUAL_PAIRS, dtype={"animal_id": str})
    equal_stats = pd.read_csv(EQUAL_STATS)
    means = equal_stats.loc[equal_stats["analysis"].isin(["outside", "maze"])].rename(
        columns={"analysis": "condition"}
    )
    # Plot the same 432 values derived directly from the three pair authorities.
    checks.record(
        "daycomb_session_pair_values_loaded_from_pair_authority",
        len(pair_values) == 432
        and source_checks["status"].eq("PASS").all(),
        "432 pair/metric values from three matched 144-row authorities",
    )

    desc_recomputed = reaggregation.summarize(pair_values)
    try:
        pd.testing.assert_frame_equal(
            day_desc.reset_index(drop=True),
            desc_recomputed.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=2e-15,
        )
        desc_ok = True
        desc_detail = "12/12 rows exact within 2e-15"
    except AssertionError as error:
        desc_ok = False
        desc_detail = str(error)
    checks.record(
        "daycomb_pair_mean_cr1_descriptive_stats_recomputed",
        desc_ok and len(day_desc) == 12,
        desc_detail,
    )

    stats_recomputed = reaggregation.contrasts(pair_values)
    try:
        pd.testing.assert_frame_equal(
            day_stats.reset_index(drop=True),
            stats_recomputed.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=2e-15,
        )
        stats_ok = True
        stats_detail = "9/9 rows exact within 2e-15"
    except AssertionError as error:
        stats_ok = False
        stats_detail = str(error)
    checks.record(
        "daycomb_pair_weighted_cluster_signflip_cr1_bh3_recomputed",
        stats_ok and len(day_stats) == 9,
        stats_detail,
    )

    gallery_outputs, full_map_contract = render_gallery_plate(output_dir)
    checks.record(
        "gallery_all_64_maps_use_full_45x45_bounds",
        full_map_contract["map_axis_count"] == 64
        and full_map_contract["all_45x45_sources"]
        and full_map_contract["all_maps_fully_shown"],
        json.dumps(full_map_contract, sort_keys=True),
    )
    gallery_path = gallery_outputs["png"]
    for extension, path in gallery_outputs.items():
        checks.record(
            f"gallery_{extension}",
            path.is_file() and path.stat().st_size > 0,
            f"{path.name} {path.stat().st_size} bytes",
        )

    canvas = Canvas(
        FIGURE_WIDTH_IN,
        FIGURE_HEIGHT_IN,
        font_reference_width_in=FONT_REFERENCE_WIDTH_IN,
    )
    placement = canvas.place_image(
        gallery_path,
        GALLERY_LEFT,
        GALLERY_BOTTOM,
        width=GALLERY_WIDTH,
        height=GALLERY_HEIGHT,
        fill_slot=True,
    )
    # Align panel a with the top colorbar band inside the gallery plate.
    letter_a_y = GALLERY_BOTTOM + (
        GALLERY_COLORBAR_Y0 + 0.5 * GALLERY_COLORBAR_HEIGHT
    ) * GALLERY_HEIGHT
    canvas.panel_letter(
        GALLERY_LEFT + 0.01,
        letter_a_y,
        "a",
        va="center",
    )
    draw_day_combination_row(canvas, pair_values, day_desc, day_stats, checks)
    draw_equal_time_panel(canvas, pairs, means, equal_stats, checks)

    checks.check_typography(canvas)
    checks.check_no_grid_and_upright_labels(canvas.fig)
    checks.record(
        "gallery_slot_matches_pptx",
        abs(placement["placed_inches"][2] - SLIDE_GALLERY["width"]) < 5e-4
        and abs(placement["placed_inches"][3] - SLIDE_GALLERY["height"]) < 5e-4,
        str(placement["placed_inches"]),
    )
    checks.record(
        "canvas_is_content_bbox_not_slide_page",
        FIGURE_WIDTH_IN < SLIDE_W_IN - 0.5 and FIGURE_HEIGHT_IN <= SLIDE_H_IN + 0.2,
        f"canvas {FIGURE_WIDTH_IN:.3f}x{FIGURE_HEIGHT_IN:.3f} in vs slide {SLIDE_W_IN}x{SLIDE_H_IN}",
    )
    checks.record(
        "gallery_aspect_preserved",
        bool(placement["aspect_preserved"]),
        str(placement["placed_inches"]),
    )
    checks.record(
        "daycomb_metrics_present",
        set(pair_values["metric"]) >= {m[0] for m in METRIC_SPECS},
        sorted(set(pair_values["metric"])),
    )
    place_sig = sum(
        1
        for row in _metric_contrast_rows(day_stats, "place_map_correlation").itertuples(
            index=False
        )
        if float(row.p_bh3_within_metric_exact_cluster_signflip) < 0.05
    )
    pv_sig = sum(
        1
        for row in _metric_contrast_rows(day_stats, "pv_correlation").itertuples(
            index=False
        )
        if float(row.p_bh3_within_metric_exact_cluster_signflip) < 0.05
    )
    rate_sig = sum(
        1
        for row in _metric_contrast_rows(
            day_stats, "ensemble_rate_correlation"
        ).itertuples(index=False)
        if float(row.p_bh3_within_metric_exact_cluster_signflip) < 0.05
    )
    checks.record("place_bh3_significant_brackets", place_sig == 0, f"n_sig={place_sig}")
    checks.record("pv_bh3_significant_brackets", pv_sig == 3, f"n_sig={pv_sig}")
    checks.record("rate_bh3_significant_brackets", rate_sig == 3, f"n_sig={rate_sig}")

    stem = output_dir / "ed05_example_place_maps_assembly"
    # Vector backends resample imshow artists at figure.dpi.  Preserve the
    # source plate at publication resolution instead of the 100-dpi default.
    canvas.fig.set_dpi(COMPOSITE_EMBEDDED_RASTER_DPI)
    outputs = canvas.save(stem, vector_dpi=COMPOSITE_EMBEDDED_RASTER_DPI)
    _check_composite_embedded_resolution(outputs, checks)
    checks.check_outputs(outputs)
    canvas.close()

    # Always place PNG/TIFF next to the review PPT so the figure is viewable
    # without opening PowerPoint (docs/**/*.png|tiff are gitignored but on disk).
    figure_root = output_dir.parent
    view_png = figure_root / "extended_data_figure5.png"
    view_tiff = figure_root / "extended_data_figure5.tiff"
    png_out = next(path for path in outputs if path.suffix == ".png")
    tiff_out = next(path for path in outputs if path.suffix == ".tiff")
    shutil.copy2(png_out, view_png)
    shutil.copy2(tiff_out, view_tiff)
    checks.record(
        "viewable_png_beside_pptx",
        view_png.is_file() and view_png.stat().st_size > 0,
        f"{view_png.name} {view_png.stat().st_size} bytes",
    )
    checks.record(
        "viewable_tiff_beside_pptx",
        view_tiff.is_file() and view_tiff.stat().st_size > 0,
        f"{view_tiff.name} {view_tiff.stat().st_size} bytes",
    )
    # Touch the image so Windows Explorer refresh picks up the companion copies.
    with Image.open(view_png) as image:
        assert image.size[0] > 0 and image.size[1] > 0

    validation = pd.DataFrame(checks.rows)
    validation_path = output_dir / "ED05_VALIDATION.csv"
    validation.to_csv(validation_path, index=False)

    source_audit = pd.DataFrame(
        [
            {
                "role": "gallery_shortlist_csv",
                "path": _repo_rel(SELECTED_32),
                "sha256": sha256(SELECTED_32),
            },
            {
                "role": "gallery_shortlist_npz",
                "path": _repo_rel(SHORTLIST_NPZ),
                "sha256": sha256(SHORTLIST_NPZ),
            },
            {
                "role": "gallery_15x15_correlation_label_audit",
                "path": full_map_contract["correlation_label_audit"],
                "sha256": sha256(
                    output_dir
                    / "derived"
                    / "ED05_PANEL_A_CORRELATION_LABEL_AUDIT.csv"
                ),
            },
            {
                "role": "day_combination_session_pair_clustered_stats",
                "path": _repo_rel(DAY_COMB_STATS),
                "sha256": sha256(DAY_COMB_STATS),
            },
            {
                "role": "day_combination_bh3_within_metric_render_copy",
                "path": _repo_rel(bh_stats_path),
                "sha256": sha256(bh_stats_path),
            },
            {
                "role": "place_pair_values",
                "path": _repo_rel(PLACE_PAIR_SOURCE),
                "sha256": sha256(PLACE_PAIR_SOURCE),
            },
            {
                "role": "pv_pair_values",
                "path": _repo_rel(PV_PAIR_SOURCE),
                "sha256": sha256(PV_PAIR_SOURCE),
            },
            {
                "role": "rate_pair_values",
                "path": _repo_rel(RATE_PAIR_SOURCE),
                "sha256": sha256(RATE_PAIR_SOURCE),
            },
            {
                "role": "equal_time_pair_rows",
                "path": _repo_rel(EQUAL_PAIRS),
                "sha256": sha256(EQUAL_PAIRS),
            },
            {
                "role": "equal_time_clustered_stats",
                "path": _repo_rel(EQUAL_STATS),
                "sha256": sha256(EQUAL_STATS),
            },
            *[
                {
                    "role": role,
                    "path": _repo_rel(path),
                    "sha256": sha256(path),
                }
                for role, path in EQUAL_EXTRA_SOURCES
            ],
            *[
                {
                    "role": f"gallery_derived_{extension}",
                    "path": _repo_rel(path),
                    "sha256": sha256(path),
                }
                for extension, path in gallery_outputs.items()
            ],
        ]
    )
    source_audit_path = output_dir / "SOURCE_DATA_AUDIT.csv"
    source_audit.to_csv(source_audit_path, index=False)

    caption_notes = output_dir.parent / "CAPTION_NOTES.md"
    caption_notes.write_text(
        "\n".join(
            [
                "# Extended Data Figure 5 — caption notes",
                "",
                "## On-figure content",
                "- **a.** Thirty-two illustrative matched-neuron place-map pairs. Heatmaps use the 45×45 display grid with independent within-map q99 normalization; printed r values use the same 15×15 analysis estimator as Figure 2.",
                "- **b–d.** Session-pair cross-session correlations for population vectors, place maps, and ensemble rate, respectively, across the four Day1/Day2 combinations (36 pairs from 14 mice).",
                EQUAL_CAPTION_PANEL_NOTE,
                "",
                "## Caption-only statistics",
                f"- Day-combination descriptive means: see `{DAY_COMB_DESC.name}`.",
                "- Planned Day1→Day1 contrasts use exact mouse-cluster sign-flip tests of the equal-pair mean with Benjamini--Hochberg correction within each metric across three comparisons; none of the place-map contrasts and all PV and ensemble-rate contrasts survive.",
                f"- {EQUAL_CAPTION_STATS_LABEL}: β={float(equal_stats.loc[equal_stats.variant.eq('d1v1') & equal_stats.analysis.eq('maze_minus_outside'), 'beta'].iloc[0]):.3f}, "
                f"p={float(equal_stats.loc[equal_stats.variant.eq('d1v1') & equal_stats.analysis.eq('maze_minus_outside'), 'pvalue'].iloc[0]):.3g}.",
                "",
                "## Layout",
                f"- Canvas is the content bounding box "
                f"({FIGURE_WIDTH_IN:.3f} × {FIGURE_HEIGHT_IN:.3f} in), not the empty 7.5 × 7.5 slide page.",
                "- Gallery uses the specified panel geometry; panel e is height-matched to b–d.",
                "- Gallery omits ranks, A/B tags, and Fig. 2b labels; colorbar label is `Normalized rate`.",
                "- Day-combination brackets use Main Figure / ED1 ticks/lines and star offset; n.s. brackets are omitted.",
                "- Panel e sets ylim before the significance bracket so stars sit above the scatter cloud.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "figure": "extended_data_figure_5",
        "layout_source": "Panel geometry constants in this renderer",
        "canvas_inches": [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN],
        "slide_page_inches": [SLIDE_W_IN, SLIDE_H_IN],
        "content_bbox_slide_inches": {
            "left": _CONTENT_LEFT,
            "top": _CONTENT_TOP,
            "right": _CONTENT_RIGHT,
            "bottom": _CONTENT_BOTTOM,
        },
        "slot_boxes_canvas_inches": {
            "gallery": GALLERY_BOX,
            "day_combination": DAYCOMB_BOX,
            "equal_time": EQUAL_BOX,
        },
        "outputs": [_repo_rel(path) for path in outputs],
        "viewable_without_powerpoint": {
            "png": _repo_rel(view_png),
            "tiff": _repo_rel(view_tiff),
        },
        "gallery_outputs": {
            extension: _repo_rel(path) for extension, path in gallery_outputs.items()
        },
        "gallery_full_map_contract": full_map_contract,
        "validation": _repo_rel(validation_path),
        "source_audit": _repo_rel(source_audit_path),
        "caption_notes": _repo_rel(caption_notes),
        "status": checks.status,
        "failed_checks": checks.failed,
        "panel_letters": ["a", "b", "c", "d", "e"],
        "corrections": [
            "source-rerender gallery from shortlist npz",
            "panel-a r labels use the Figure-2-matched 15x15 analysis correlations while heatmaps remain 45x45",
            "gallery simplified: no ranks/A-B/Fig2b tags; centered normalized-rate colorbar",
            "panel a and r labels above heatmaps; flush-left gallery; closer larger colorbar",
            "equal-weighted corrected36 session-pair day-combination bars for place/PV/rate with animal-clustered uncertainty",
            "mask-touched sorted-neural rows rebuilt with no manual exclusions and exact common support",
            "significant within-metric BH3 brackets only (n.s. omitted); ED1 star-to-tick gap; shared y='correlation'",
            EQUAL_MANIFEST_CORRECTION,
            "canvas cropped to content bbox",
        ],
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    layout_audit = pd.DataFrame(
        [
            {
                "panel": "a_gallery",
                "slide_left_in": SLIDE_GALLERY["left"],
                "slide_top_in": SLIDE_GALLERY["top"],
                "left_in": GALLERY_LEFT,
                "bottom_in": GALLERY_BOTTOM,
                "width_in": GALLERY_WIDTH,
                "height_in": GALLERY_HEIGHT,
                "pptx_width_in": SLIDE_GALLERY["width"],
                "pptx_height_in": SLIDE_GALLERY["height"],
            },
            {
                "panel": "b_d_day_combination",
                "slide_left_in": SLIDE_DAYCOMB["left"],
                "slide_top_in": SLIDE_DAYCOMB["top"],
                "left_in": DAYCOMB_LEFT,
                "bottom_in": DAYCOMB_BOTTOM,
                "width_in": DAYCOMB_WIDTH,
                "height_in": DAYCOMB_HEIGHT,
                "pptx_width_in": SLIDE_DAYCOMB["width"],
                "pptx_height_in": SLIDE_DAYCOMB["height"],
            },
            {
                "panel": "e_equal_time",
                "slide_left_in": SLIDE_EQUAL["left"],
                "slide_top_in": SLIDE_EQUAL["top"],
                "left_in": EQUAL_LEFT,
                "bottom_in": EQUAL_BOTTOM,
                "width_in": EQUAL_WIDTH,
                "height_in": EQUAL_HEIGHT,
                "pptx_width_in": SLIDE_EQUAL["width"],
                "pptx_height_in": SLIDE_EQUAL["height"],
                "note": "height matched to b-d band; slide slot was taller",
            },
        ]
    )
    layout_audit.to_csv(output_dir / "LAYOUT_AUDIT.csv", index=False)

    if checks.failed:
        raise AssertionError(
            "ED5 assembly checks failed: "
            + "; ".join(f"{row['check']}={row['detail']}" for row in checks.failed)
        )
    return manifest
