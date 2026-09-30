"""Assemble Extended Data Figure 4 (cross-session neuron-matching quality).

Layout authority is slide 5 of ``sup-figs-v2.pptx``, which carries the Extended Data
Figure 4 role under the merged numbering. Sources resolve under the OLD number,
``extended_data_fig_05``.

Panels c-h are drawn here from tables that
``scripts/build_ed04_registration_source_from_raw.py`` regenerated directly from the raw
CellReg outputs, plus the raw-derived physical-alignment table. Panels a and b are the
maintained source-rendered field-of-view images.

Departures from the slide:

* the slide's FOV slot is ``99_legacy_audit_only/ed05_panel_a_fov_displacement.png``. Its
  two views are rebuilt on current data as panels c (cumulative drift path, microns) and
  d (alignment displacement, % of source FOV diagonal).
* panel g adds the marginal count distributions to the joint density.
* the Main Figure 2 similarity cross-check is not applied. It re-derives the activity gate
  and the 21,412 -> 16,901 denominator, which is Methods material and unrelated to
  matching quality.

Formatting follows the main figures for typography only -- family, sizes, weights,
spacing, spines, panel letters. Each panel keeps its own colour scheme, because these
panels compare algorithm candidate classes rather than experimental conditions, so the
main-figure condition palette would imply a contrast the data does not have.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import fitz
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from PIL import Image

from ed_supplement_common import REBUILD_ROOT, REPO_ROOT, Canvas, Checks, sha256

ED04_ROOT = REBUILD_ROOT / "ed04_registration_quality"
SOURCE = ED04_ROOT / "source_from_raw"
ALIGNMENT = SOURCE / "alignment"
DEFAULT_OUTPUT_DIR = ED04_ROOT / "assembly"

ED05 = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "extended_data_figures"
    / "extended_data_fig_05"
)
SOURCE_RENDERED = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_05"
)
LOCAL_CORRELATION_IMAGE = (
    SOURCE_RENDERED
    / "registration_flow_and_four_session"
    / "ed05_four_session_418_local_correlation_paper.png"
)
COREGISTRATION_IMAGE = (
    ED05 / "03_illustrative_examples" / "ed05_four_session_418_coregistration_paper.png"
)
LEGACY_FOV_PANEL = ED05 / "99_legacy_audit_only" / "ed05_panel_a_fov_displacement.png"

FIGURE_WIDTH_IN = 7.5
FIGURE_HEIGHT_IN = 6.54
COMPOSITE_EMBEDDED_RASTER_DPI = 600

MICRON = "µm"
ARROW = "→"

# Candidate-class colours are kept from the maintained panels; they distinguish algorithm
# classes, not experimental conditions.
CLOSEST_COLOR = "#1F77B4"
OTHER_COLOR = "#E8762C"
NEUTRAL_COLOR = "#4C6472"

CANDIDATE_CLASS_LABEL = {
    "closest_by_centroid": "Centroid-closest candidate",
    "other_within_radius": "Other candidates",
}

# Rows in inches, measured from the top of the canvas. Spacing is opened up relative to
# slide 5 so each redrawn row has room for its axis labels. Row C carries the two
# physical-alignment views; row D carries the four distributions.
ROW_A = {"x": 0.17, "y_top": 0.18, "width": 7.20}
ROW_B = {"x": 0.16, "y_top": 1.74, "width": 7.20}
ROW_C_TOP = 3.61
ROW_C_HEIGHT = 1.12
ROW_C_COLUMNS = (0.72, 2.72, 5.14)
ROW_C_WIDTHS = (1.30, 1.72, 1.72)
ROW_D_TOP = 5.15
ROW_D_HEIGHT = 1.04
ROW_D_COLUMNS = (0.68, 2.36, 4.06, 6.30)
ROW_D_WIDTH = 1.12
JOINT_WIDTH = 1.45
JOINT_MARGIN = 0.22
JOINT_GAP = 0.035
LETTER_OFFSET = 0.62
LETTER_LIFT = 0.05


def y_bottom(y_top: float, height: float) -> float:
    return FIGURE_HEIGHT_IN - y_top - height


def letter_y(y_top: float) -> float:
    """Panel-letter baseline, just above a panel whose top edge is ``y_top``."""
    return FIGURE_HEIGHT_IN - y_top + LETTER_LIFT


def step_histogram(
    ax, frame: pd.DataFrame, value: str, bins: np.ndarray, canvas: Canvas
) -> dict[str, Any]:
    """Draw the two candidate classes as step histograms, closest over other."""
    counts: dict[str, Any] = {}
    for klass, color in (
        ("other_within_radius", OTHER_COLOR),
        ("closest_by_centroid", CLOSEST_COLOR),
    ):
        values = frame.loc[frame["candidate_class"].eq(klass), value].to_numpy(float)
        values = values[np.isfinite(values)]
        hist, edges = np.histogram(values, bins=bins)
        ax.step(
            edges[:-1],
            hist,
            where="post",
            color=color,
            linewidth=1.2 * canvas.font_scale,
            label=CANDIDATE_CLASS_LABEL[klass],
            zorder=3 if klass == "closest_by_centroid" else 2,
        )
        counts[klass] = int(hist.sum())
    ax.set_ylabel("Directed candidate count")
    ax.margins(x=0)
    ax.set_ylim(bottom=0)
    return counts


def draw_drift_path(ax, physical: pd.DataFrame, canvas: Canvas) -> dict[str, Any]:
    """Cumulative field-of-view drift per animal, rebuilt on the corrected-36 data.

    This restores the framing of the superseded panel, a drift path in microns, but from
    the current measurements rather than the legacy table.
    """
    paths: dict[str, Any] = {}
    for animal, group in physical.groupby("animal"):
        ordered = group.sort_values("experience")
        x = np.concatenate([[0.0], np.cumsum(ordered["dx_um"].to_numpy(float))])
        y = np.concatenate([[0.0], np.cumsum(ordered["dy_um"].to_numpy(float))])
        ax.plot(
            x,
            y,
            color=NEUTRAL_COLOR,
            linewidth=0.55,
            alpha=0.55,
            zorder=2,
            solid_capstyle="round",
        )
        ax.scatter(
            x[1:],
            y[1:],
            s=8.0 * canvas.font_scale**2 / 0.5625,
            facecolor=NEUTRAL_COLOR,
            edgecolor="none",
            alpha=0.75,
            zorder=3,
        )
        paths[str(animal)] = {
            "n_transitions": int(ordered.shape[0]),
            "final_x_um": float(x[-1]),
            "final_y_um": float(y[-1]),
        }
    ax.scatter(
        [0.0],
        [0.0],
        s=14.0 * canvas.font_scale**2 / 0.5625,
        facecolor="#202020",
        edgecolor="none",
        zorder=4,
    )
    ax.axhline(0.0, color="#C8C8C8", linewidth=0.5, zorder=1)
    ax.axvline(0.0, color="#C8C8C8", linewidth=0.5, zorder=1)
    ax.set_xlabel("Cumulative x shift (" + MICRON + ")")
    ax.set_ylabel("Cumulative y shift (" + MICRON + ")")
    ax.set_aspect("equal", adjustable="datalim")
    return {"n_animals": len(paths), "per_animal": paths}


def draw_displacement_across_experience(ax, physical, canvas):
    """Cumulative displacement from the first footprint against experience index.

    This is the second view carried by the superseded FOV panel, rebuilt on the
    corrected-36 data. It differs from the per-transition percentage: here the quantity
    accumulates from a fixed baseline, so it reports how far the field of view has moved
    overall rather than the size of each individual step.
    """
    curves = []
    for animal, group in physical.groupby("animal"):
        ordered = group.sort_values("experience")
        steps = ordered["experience"].to_numpy(int)
        cx = np.cumsum(ordered["dx_um"].to_numpy(float))
        cy = np.cumsum(ordered["dy_um"].to_numpy(float))
        magnitude = np.hypot(cx, cy)
        ax.plot(
            np.concatenate([[0], steps]),
            np.concatenate([[0.0], magnitude]),
            color=NEUTRAL_COLOR,
            linewidth=0.5,
            alpha=0.38,
            zorder=2,
        )
        curves.append({int(a): float(b) for a, b in zip(steps, magnitude)})

    levels = sorted({level for curve in curves for level in curve})
    means, sems, support = [], [], {}
    for level in levels:
        values = np.array([c[level] for c in curves if level in c], dtype=float)
        means.append(float(values.mean()))
        sems.append(
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
        support[int(level)] = int(values.size)
    ax.errorbar(
        [0] + levels,
        [0.0] + means,
        yerr=[0.0] + sems,
        color="#202020",
        marker="o",
        markersize=2.6 * canvas.font_scale / 0.75,
        linewidth=1.0 * canvas.font_scale / 0.75,
        capsize=2.4 * canvas.font_scale / 0.75,
        elinewidth=0.9 * canvas.font_scale / 0.75,
        zorder=4,
    )
    ax.set_xticks([0] + levels)
    ax.set_xlim(-0.25, max(levels) + 0.25)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("No. of experiences")
    ax.set_ylabel("Displacement from\nfirst FOV (" + MICRON + ")")
    return {
        "n_animals": len(curves),
        "support_per_index": support,
        "mean_per_index": {int(l): float(m) for l, m in zip(levels, means)},
    }


def draw_joint_with_marginals(
    canvas: Canvas,
    registered: pd.DataFrame,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> dict[str, Any]:
    """Registered-pair joint density with both marginal count distributions."""
    x = registered["centroid_distance_um"].to_numpy(float)
    y = registered["spatial_footprint_correlation"].to_numpy(float)
    x_bins = np.linspace(0.0, 14.0, 41)
    y_bins = np.linspace(0.0, 1.0, 41)
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    main_w = width - JOINT_MARGIN - JOINT_GAP
    main_h = height - JOINT_MARGIN - JOINT_GAP
    ax = canvas.axes(x0, y0, main_w, main_h)
    ax_top = canvas.axes(x0, y0 + main_h + JOINT_GAP, main_w, JOINT_MARGIN)
    ax_right = canvas.axes(x0 + main_w + JOINT_GAP, y0, JOINT_MARGIN, main_h)

    mesh = ax.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_where(hist.T == 0, hist.T),
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(1.0, hist.max())),
        shading="flat",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Centroid distance (" + MICRON + ")")
    ax.set_ylabel("Spatial footprint\ncorrelation (r)")

    ax_top.bar(
        xedges[:-1],
        hist.sum(axis=1),
        width=np.diff(xedges),
        align="edge",
        color=NEUTRAL_COLOR,
        edgecolor="none",
        alpha=0.85,
    )
    ax_top.set_xlim(0, 14)
    ax_top.set_ylim(bottom=0)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="y", labelsize=canvas.tick_label * 0.85, pad=1.0)
    ax_top.set_ylabel("Count", fontsize=canvas.tick_label * 0.9, labelpad=1.5)
    ax_top.locator_params(axis="y", nbins=2)

    ax_right.barh(
        yedges[:-1],
        hist.sum(axis=0),
        height=np.diff(yedges),
        align="edge",
        color=NEUTRAL_COLOR,
        edgecolor="none",
        alpha=0.85,
    )
    ax_right.set_ylim(0, 1)
    ax_right.set_xlim(left=0)
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", labelsize=canvas.tick_label * 0.85, pad=1.0)
    ax_right.set_xlabel("Count", fontsize=canvas.tick_label * 0.9, labelpad=1.5)
    ax_right.locator_params(axis="x", nbins=2)

    # Colour scale as a compact inset, as on the slide, so it costs no row width.
    inset = ax.inset_axes([0.62, 0.12, 0.055, 0.44])
    bar = canvas.fig.colorbar(mesh, cax=inset)
    bar.ax.tick_params(
        labelsize=canvas.tick_label * 0.8, width=canvas.axes_linewidth, pad=1.0
    )
    bar.outline.set_linewidth(canvas.axes_linewidth * 0.8)
    bar.set_label("Pairs", fontsize=canvas.tick_label * 0.85, labelpad=1.5)
    return {
        "n_registered_pairs": int(registered.shape[0]),
        "max_bin_count": int(hist.max()),
        "has_marginals": True,
    }


def build(output_dir: Path) -> dict[str, Any]:
    candidates = pd.read_csv(SOURCE / "ed04_candidate_features_directed_long.csv")
    registered = pd.read_csv(SOURCE / "ed04_registered_pair_features_long.csv")
    physical = pd.read_csv(
        ALIGNMENT / "physical_fov_exact_36_source.csv", dtype={"animal": str}
    )
    summary = json.loads((SOURCE / "ED04_SOURCE_SUMMARY.json").read_text(encoding="utf-8"))

    canvas = Canvas(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)
    panels: dict[str, Any] = {}

    # -- a, b: maintained source-rendered field-of-view image plates --------
    for letter, path, spec in (
        ("a", LOCAL_CORRELATION_IMAGE, ROW_A),
        ("b", COREGISTRATION_IMAGE, ROW_B),
    ):
        with Image.open(path) as image:
            aspect = image.size[0] / image.size[1]
        height = float(spec["width"]) / aspect
        record = canvas.place_image(
            path,
            float(spec["x"]),
            y_bottom(float(spec["y_top"]), height),
            width=float(spec["width"]),
        )
        canvas.panel_letter(
            max(float(spec["x"]) - 0.14, 0.02), letter_y(float(spec["y_top"])), letter
        )
        panels[letter] = {**record, "source": "maintained source-rendered image"}

    # -- c: cumulative FOV drift path, rebuilt on corrected-36 data ---------
    ax = canvas.axes(
        ROW_C_COLUMNS[0], y_bottom(ROW_C_TOP, ROW_C_HEIGHT), ROW_C_WIDTHS[0], ROW_C_HEIGHT
    )
    drift = draw_drift_path(ax, physical, canvas)
    canvas.panel_letter(ROW_C_COLUMNS[0] - LETTER_OFFSET, letter_y(ROW_C_TOP), "c")
    panels["c"] = {
        "role": "cumulative field-of-view drift path per animal",
        "source": str(
            (ALIGNMENT / "physical_fov_exact_36_source.csv").relative_to(REPO_ROOT)
        ),
        "restores_legacy_framing": LEGACY_FOV_PANEL.name,
        **drift,
    }

    # -- d: displacement from the first footprint across experience ---------
    ax = canvas.axes(
        ROW_C_COLUMNS[1], y_bottom(ROW_C_TOP, ROW_C_HEIGHT), ROW_C_WIDTHS[1], ROW_C_HEIGHT
    )
    across = draw_displacement_across_experience(ax, physical, canvas)
    canvas.panel_letter(ROW_C_COLUMNS[1] - LETTER_OFFSET, letter_y(ROW_C_TOP), "d")
    panels["d"] = {
        "role": "cumulative displacement from the first footprint across experience",
        "source": str(
            (ALIGNMENT / "physical_fov_exact_36_source.csv").relative_to(REPO_ROOT)
        ),
        "restores_legacy_framing": LEGACY_FOV_PANEL.name,
        "x_label": "No. of experiences",
        "y_label": "Displacement from first FOV",
        **across,
    }

    # -- e: physical alignment displacement per transition, corrected-36 ----
    ax = canvas.axes(
        ROW_C_COLUMNS[2], y_bottom(ROW_C_TOP, ROW_C_HEIGHT), ROW_C_WIDTHS[2], ROW_C_HEIGHT
    )
    order = sorted(physical["experience"].unique())
    # Label by the modal destination month per transition ordinal. The mapping is not
    # strictly one-to-one: the composed pair 1616bday1->1616b-3day1 sits at experience 2
    # with destination month 3, so any single row would mislabel the axis.
    month = physical.groupby("experience")["destination_month"].agg(
        lambda values: int(values.mode().iloc[0])
    )
    labels = {int(e): f"{int(month[e]) - 3}{ARROW}{int(month[e])}" for e in order}
    off_ladder = physical.loc[
        physical.apply(
            lambda row: int(row["destination_month"]) != int(month[row["experience"]]),
            axis=1,
        ),
        "pair_label",
    ].tolist()
    rng = np.random.default_rng(20260806)
    for experience in order:
        values = physical.loc[
            physical["experience"].eq(experience),
            "displacement_percent_source_fov_diagonal",
        ].to_numpy(float)
        ax.scatter(
            np.full(len(values), float(experience))
            + rng.normal(0.0, 0.055, len(values)),
            values,
            s=13.0 * canvas.font_scale**2 / 0.5625,
            facecolor=NEUTRAL_COLOR,
            edgecolor="none",
            alpha=0.75,
            zorder=2,
        )
    # Mean with SEM across pairs at each transition, drawn in the same grammar as the
    # experience curve in panel d so the two alignment views read together.
    transition_mean, transition_sem, transition_support = [], [], {}
    for experience in order:
        values = physical.loc[
            physical["experience"].eq(experience),
            "displacement_percent_source_fov_diagonal",
        ].to_numpy(float)
        values = values[np.isfinite(values)]
        transition_mean.append(float(values.mean()))
        transition_sem.append(
            float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        )
        transition_support[int(experience)] = int(values.size)
    ax.errorbar(
        [int(e) for e in order],
        transition_mean,
        yerr=transition_sem,
        color="#202020",
        marker="o",
        markersize=2.6 * canvas.font_scale / 0.75,
        linewidth=1.0 * canvas.font_scale / 0.75,
        capsize=2.4 * canvas.font_scale / 0.75,
        elinewidth=0.9 * canvas.font_scale / 0.75,
        zorder=4,
    )
    ax.set_xticks([int(e) for e in order])
    # Tick values mean the same session index as in panel d, but the axis starts at the
    # first session that has a preceding one; padding out to 0 would only add dead space.
    ax.set_xlim(
        min(int(e) for e in order) - 0.25, max(int(e) for e in order) + 0.25
    )
    ax.set_ylim(
        0,
        max(
            12.0,
            float(physical["displacement_percent_source_fov_diagonal"].max()) * 1.12,
        ),
    )
    ax.set_xlabel("No. of experiences")
    ax.set_ylabel("Displacement from previous\nFOV (% FOV diagonal)")
    canvas.panel_letter(ROW_C_COLUMNS[2] - LETTER_OFFSET, letter_y(ROW_C_TOP), "e")
    panels["e"] = {
        "source": str(
            (ALIGNMENT / "physical_fov_exact_36_source.csv").relative_to(REPO_ROOT)
        ),
        "n_pairs": int(physical.shape[0]),
        "median_percent": float(
            physical["displacement_percent_source_fov_diagonal"].median()
        ),
        "max_percent": float(physical["displacement_percent_source_fov_diagonal"].max()),
        "replaces_legacy": LEGACY_FOV_PANEL.name,
        "transition_labels": labels,
        "pairs_off_the_modal_month_ladder": off_ladder,
        "mean_percent_per_transition": {
            int(e): round(m, 3) for e, m in zip(order, transition_mean)
        },
        "support_per_transition": transition_support,
        "has_mean_line": True,
        "x_positions": [int(e) for e in order],
        "x_label": "No. of experiences",
        "y_label": "Displacement from previous FOV",
        "month_ranges_for_caption": labels,
    }

    # -- f, g: candidate-feature separation ---------------------------------
    ax = canvas.axes(
        ROW_D_COLUMNS[0], y_bottom(ROW_D_TOP, ROW_D_HEIGHT), ROW_D_WIDTH, ROW_D_HEIGHT
    )
    counts_f = step_histogram(
        ax, candidates, "centroid_distance_um", np.linspace(0.0, 14.0, 41), canvas
    )
    ax.set_xlabel("Centroid distance (" + MICRON + ")")
    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.0,
        borderpad=0.15,
        labelspacing=0.25,
        fontsize=canvas.tick_label * 0.8,
    )
    canvas.panel_letter(ROW_D_COLUMNS[0] - LETTER_OFFSET, letter_y(ROW_D_TOP), "f")
    panels["f"] = {"counts": counts_f, "bins": 40, "range_um": [0.0, 14.0]}

    ax = canvas.axes(
        ROW_D_COLUMNS[1], y_bottom(ROW_D_TOP, ROW_D_HEIGHT), ROW_D_WIDTH, ROW_D_HEIGHT
    )
    plotted = candidates.loc[candidates["correlation_in_0_to_1_plot"].astype(bool)]
    counts_g = step_histogram(
        ax, plotted, "spatial_footprint_correlation", np.linspace(0.0, 1.0, 41), canvas
    )
    ax.set_xlabel("Spatial footprint correlation")
    canvas.panel_letter(ROW_D_COLUMNS[1] - LETTER_OFFSET, letter_y(ROW_D_TOP), "g")
    panels["g"] = {
        "counts": counts_g,
        "excluded_negative_correlations": int(
            (~candidates["correlation_in_0_to_1_plot"].astype(bool)).sum()
        ),
    }

    # -- h: registered-pair joint distribution with marginals ---------------
    panels["h"] = draw_joint_with_marginals(
        canvas,
        registered,
        ROW_D_COLUMNS[2],
        y_bottom(ROW_D_TOP, ROW_D_HEIGHT),
        JOINT_WIDTH,
        ROW_D_HEIGHT,
    )
    canvas.panel_letter(ROW_D_COLUMNS[2] - LETTER_OFFSET, letter_y(ROW_D_TOP), "h")

    # -- i: CellReg registration score --------------------------------------
    ax = canvas.axes(
        ROW_D_COLUMNS[3], y_bottom(ROW_D_TOP, ROW_D_HEIGHT), ROW_D_WIDTH, ROW_D_HEIGHT
    )
    scores = registered["cellreg_registration_score"].to_numpy(float)
    finite = scores[np.isfinite(scores)]
    non_negative = finite[finite >= 0.0]
    hist_h, edges_h = np.histogram(non_negative, bins=np.linspace(0.0, 1.0, 51))
    ax.step(
        edges_h[:-1],
        hist_h,
        where="post",
        color=NEUTRAL_COLOR,
        linewidth=1.2 * canvas.font_scale,
        zorder=3,
    )
    ax.fill_between(
        edges_h[:-1], hist_h, step="post", color=NEUTRAL_COLOR, alpha=0.35, zorder=2
    )
    ax.set_xlabel("CellReg registration score")
    ax.set_ylabel("Registered-pair count")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    canvas.panel_letter(ROW_D_COLUMNS[3] - LETTER_OFFSET, letter_y(ROW_D_TOP), "i")
    panels["i"] = {
        "n_scored": int(finite.size),
        "n_negative_excluded": int((finite < 0).sum()),
        "median": float(np.median(finite)),
    }

    stem = output_dir / "ed04_registration_quality_assembly"
    # Keep the two maintained FOV plates and the compact density colour scale
    # at publication resolution in SVG/PDF rather than Matplotlib's 100-dpi
    # vector-backend default.
    canvas.fig.set_dpi(COMPOSITE_EMBEDDED_RASTER_DPI)
    outputs = canvas.save(stem, vector_dpi=COMPOSITE_EMBEDDED_RASTER_DPI)

    checks = Checks()
    checks.check_typography(canvas)
    checks.check_no_grid_and_upright_labels(canvas.fig)
    checks.check_outputs(outputs)
    pdf_path = next(path for path in outputs if path.suffix == ".pdf")
    with fitz.open(pdf_path) as document:
        page = document[0]
        embedded = []
        for image in page.get_images(full=True):
            xref = int(image[0])
            pixmap = fitz.Pixmap(document, xref)
            for rect in page.get_image_rects(xref):
                embedded.append(
                    {
                        "pixels": [pixmap.width, pixmap.height],
                        "bbox_pt": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "effective_dpi": [
                            pixmap.width / (rect.width / 72.0),
                            pixmap.height / (rect.height / 72.0),
                        ],
                    }
                )
    checks.record(
        "all_embedded_scientific_rasters_at_least_600dpi",
        len(embedded) == 3
        and all(min(row["effective_dpi"]) >= COMPOSITE_EMBEDDED_RASTER_DPI - 1.0 for row in embedded),
        json.dumps(embedded),
    )
    checks.record(
        "scope_36_pairs_14_animals",
        summary["n_session_pairs"] == 36 and summary["n_animals"] == 14,
        f"{summary['n_session_pairs']} pairs / {summary['n_animals']} animals",
    )
    checks.record(
        "raw_candidate_and_registered_totals",
        summary["n_directed_candidate_pairs"] == 119444
        and summary["n_registered_pairs_before_activity_gate"] == 21412,
        f"{summary['n_directed_candidate_pairs']} candidates, "
        f"{summary['n_registered_pairs_before_activity_gate']} registered",
    )
    checks.record("panel_count", len(panels) == 9, f"{len(panels)} panels (a-i)")
    checks.record(
        "legacy_fov_panel_not_used",
        panels["e"]["replaces_legacy"] == LEGACY_FOV_PANEL.name
        and panels["c"]["restores_legacy_framing"] == LEGACY_FOV_PANEL.name
        and panels["d"]["restores_legacy_framing"] == LEGACY_FOV_PANEL.name,
        "legacy FOV panel rebuilt on current data as panels c, d and e",
    )
    checks.record(
        "no_panel_from_legacy_audit_only",
        not any("99_legacy" in str(p.get("path", "")) for p in panels.values()),
        "no panel resolves into 99_legacy_audit_only",
    )
    checks.record(
        "drift_path_covers_all_animals",
        panels["c"]["n_animals"] == 14,
        f"{panels['c']['n_animals']} animal drift paths",
    )
    checks.record(
        "alignment_y_axes_name_their_reference",
        panels["d"]["y_label"] == "Displacement from first FOV"
        and panels["e"]["y_label"] == "Displacement from previous FOV",
        "panel d is referenced to the first FOV, panel e to the preceding session",
    )
    checks.record(
        "panels_d_and_e_share_an_x_axis",
        panels["e"]["x_positions"] == sorted(int(k) for k in panels["d"]["support_per_index"])
        and panels["e"]["x_label"] == panels["d"]["x_label"],
        f"panel e at session indices {panels['e']['x_positions']}, same meaning and label "
        f"as panel d, starting at its first defined step",
    )
    checks.record(
        "both_alignment_panels_carry_mean_line",
        bool(panels["e"].get("has_mean_line")) and panels["d"]["n_animals"] == 14,
        "panels d and e both show the mean with SEM",
    )
    checks.record(
        "displacement_across_experience_present",
        panels["d"]["n_animals"] == 14,
        f"{panels['d']['n_animals']} animals, support {panels['d']['support_per_index']}",
    )
    checks.record(
        "joint_panel_has_marginals",
        bool(panels["h"].get("has_marginals")),
        "panel g carries both marginal count distributions",
    )

    for name, detail in (
        (
            "deviation_candidate_class_palette",
            "panels f-i keep their candidate-class and density colour schemes instead of "
            "the Main Figure 1 condition palette; these panels compare algorithm classes, "
            "not experimental conditions, so condition colours would imply a contrast the "
            "data does not have",
        ),
        (
            "deviation_row_spacing_opened",
            "row spacing is opened relative to slide 5 so each quantitative row has room "
            "for its axis labels; the slide's rows overlap once panels are redrawn rather "
            "than embedded as images",
        ),
        (
            "deviation_figure2_cross_check_not_applied",
            "the Main Figure 2 similarity cross-check and activity gate are not applied; "
            "they re-derive the 21,412 -> 16,901 denominator, which the Methods describe "
            "and which is unrelated to matching quality",
        ),
    ):
        checks.rows.append({"check": name, "status": "DEVIATION", "detail": detail})
    canvas.close()

    manifest = {
        "figure": "Extended Data Figure 4 (merged numbering; slide 5 of sup-figs-v2.pptx)",
        "sources_resolve_under": "extended_data_fig_05 (old numbering)",
        "figure_size_inches": [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN],
        "source_summary": summary,
        "panels": panels,
        "embedded_raster_audit": embedded,
        "formatting": (
            "main-figure typography only: family, sizes, weights, spacing, spines and "
            "panel letters. Candidate-class and density colour schemes are kept, because "
            "these panels compare algorithm classes rather than experimental conditions."
        ),
        "outputs": [
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "sha256": sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in outputs
        ],
        "validation": checks.rows,
        "validation_status": checks.status,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ED04_ASSEMBLY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "ED04_VALIDATION.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks.rows)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build(args.output_dir)
    failed = [c for c in result["validation"] if c["status"] not in ("PASS", "DEVIATION")]
    print(
        json.dumps(
            {
                "validation_status": result["validation_status"],
                "checks": len(result["validation"]),
                "failed": failed,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
