"""Current ED2 plotting and table helpers; no revision-chain imports."""
from __future__ import annotations
from types import SimpleNamespace
from ca_analysis.figure_panels import render_grid_route_choice_schematic
from copy import deepcopy
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from pathlib import Path
from typing import Any
from typing import Iterable
import hashlib
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = '#27313A'

BACKGROUND = '#FFFFFF'

GROUP_LABEL = {'naive': 'Naive day', 'LTM': 'Return day', 'experienced': 'Second day'}

BASE_FONT_SIZE = 6.5

TICK_LABEL_SIZE = 5.8

LEGEND_SIZE = 5.4

GROUP_MARKER = {'naive': 'o', 'LTM': 'o', 'experienced': 'o'}

GROUP_ORDER = ('naive', 'LTM', 'experienced')

LAST_A_GEOMETRY = {}

from ca_analysis.figure_panels import GRID_ROUTE_CHOICE_SCHEMATIC_TEXT

PANEL_SIZE_IN = {'a': (1.5416666666666667, 2.0),
 'b': (2.6875, 2.0),
 'c': (2.6875, 2.0),
 'd': (1.6666666666666667, 2.4375),
 'e': (1.3333333333333333, 2.4375),
 'f': (4.166666666666667, 2.4375),
 'g': (1.8333333333333333, 2.4375),
 'h': (1.1666666666666667, 2.4375),
 'i': (4.166666666666667, 2.4375)}

from ca_analysis.figure_panels import GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY

from ca_analysis.figure_panels import GRID_ROUTE_CHOICE_SCHEMATIC_COLORS

SCHEMATIC_ARTIST_SCALE = 0.8

PANEL_A_ANNOTATION_BASE_X = 3.42

PANEL_A_ANNOTATION_Y = 3.8

PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE = 'nn'

RANDOM_REFERENCE = {'Bf': 0.6666666666666666, 'Ba': 0.5, 'Lf': 0.6666666666666666, 'Lo': 0.5}

REFERENCE = '#6F6F6F'

GROUP_COLOR = {'naive': '#1F90D5', 'LTM': '#D82929', 'experienced': '#5F8C62'}

POLICY_LABEL = {'Bf': '$P_{SF}$', 'Ba': '$P_{SA}$', 'Lf': '$P_{BF}$', 'Lo': '$P_{BS}$'}

POLICY_ORDER = ('Bf', 'Ba', 'Lf', 'Lo')

LIGHT = '#D7DDE2'

UPRIGHT_TEXT = {'$P_{SF}$': '$\\mathrm{P}_{\\mathrm{SF}}$',
 '$P_{SA}$': '$\\mathrm{P}_{\\mathrm{SA}}$',
 '$P_{BF}$': '$\\mathrm{P}_{\\mathrm{BF}}$',
 '$P_{BS}$': '$\\mathrm{P}_{\\mathrm{BS}}$',
 '$|\\Delta P|$ from Return day': '$|\\Delta \\mathrm{P}|$ from Return day',
 '$\\Delta P_{BS}$': '$\\Delta \\mathrm{P}_{\\mathrm{BS}}$',
 'Depth-varying $P_{BS}$ model': 'Depth-varying $\\mathrm{P}_{\\mathrm{BS}}$ model',
 'Depth $P_{BS}$': 'Depth $\\mathrm{P}_{\\mathrm{BS}}$'}

LAST_E_GEOMETRY = {}

DEPTH_SUPPORT_COLUMNS = ('n_Lo_shallow', 'n_Lo_mid', 'n_Lo_deep')

MIN_DEPTH_EVENTS = 10

FOUR_COUNTS = {'naive': 23, 'LTM': 58, 'experienced': 74}

DEPTH_COUNTS = {'naive': 22, 'LTM': 58, 'experienced': 74}

MODEL_TITLE_SIZE = 6.7

MODEL_TITLE_Y = 0.94

DEPTH_ORDER = ('Lo_shallow', 'Lo_mid', 'Lo_deep')

PANEL_G_GROUP_SCALE = 0.72

DEPTH_LABELS = ('Shallow\n1-2', 'Middle\n3-4', 'Deep\n5')

PANEL_G_LEFT = 0.27

PANEL_G_YLABEL_X = -0.205

PANEL_G_UPPER_ANCHOR = (0.52, 1.30598)

PANEL_G_LOWER_ANCHOR = (0.52, 1.3409)

LAST_G_SAVE_GEOMETRY = {}

PANEL_G_TOP_RESERVE_IN = 0.14

CROP_BBOX_IN = {'a': (0.323611, 0.168611, 1.4630560000000001, 1.724722),
 'b': (0.000289, 0.198611, 2.653868, 1.509722),
 'c': (0.000289, 0.198611, 2.653868, 1.509722),
 'd': (0.255278, 0.0, 1.666667, 2.395833),
 'e': (0.0, 0.120331, 1.333333, 2.317169),
 'f': (0.066944, 0.147007, 4.166667, 2.410534),
 'g': (0.051944, 0.106993, 1.833333, 2.210466),
 'h': (0.0, 0.083652, 1.166667, 2.350514),
 'i': (0.066944, 0.147007, 4.166667, 2.410534)}

FORMATS = ('svg', 'pdf', 'png', 'tiff')

PANEL_D_NEUTRAL = '#68737D'

FOUR = '#69737F'

DEPTH = '#D26A3A'

CROP_BBOX_IN_base = {'a': (0.273611, 0.168611, 1.413056, 1.724722),
 'b': (0.000289, 0.198611, 2.653868, 1.509722),
 'c': (0.000289, 0.198611, 2.653868, 1.509722),
 'd': (0.255278, 0.0, 1.666667, 2.395833),
 'e': (0.0, 0.120331, 1.333333, 2.317169),
 'f': (0.066944, 0.147007, 4.166667, 2.410534),
 'g': (0.051944, 0.106993, 1.833333, 2.210466),
 'h': (0.0, 0.083652, 1.166667, 2.350514),
 'i': (0.066944, 0.147007, 4.166667, 2.410534)}

SOURCE_FILES = {'strict_policy_rows': 'primary_30min_strict_joint_equal_junction_policy_rows.csv',
 'strict_policy_contrasts': 'primary_30min_strict_joint_policy_cluster_stats.csv',
 'strict_contributions': 'primary_30min_strict_joint_policy_contribution_summary.csv',
 'strict_gap': 'primary_30min_strict_joint_gap_recapitulation_summary.csv',
 'natural_policy_rows': 'fixed_time_policy_rows.csv',
 'natural_policy_contrasts': 'equal_junction_policy_group_cluster_stats.csv',
 'natural_depth_contrasts': 'equal_junction_depth_group_cluster_stats.csv',
 'natural_calibration': 'equal_junction_calibration_summary.csv',
 'natural_simulation_rows': 'equal_junction_simulation_session_rows.csv'}

WINDOW_MINUTES = 30

POLICY_SHORT = {'Bf': 'enter child vs reverse',
 'Ba': 'alternating vs non-alternating child',
 'Lf': 'leave arrival-child branch vs reverse',
 'Lo': 'parent/stem vs sibling'}

SHARED_LEGEND_STEM = 'ed02_shared_bc_experience_legend'

PANEL_STEM = {'a': 'ed02a_four_policy_schematic',
 'b': 'ed02b_naive_vs_return_policy_scatter',
 'c': 'ed02c_return_vs_second_day_policy_scatter',
 'd': 'ed02d_signed_policy_contributions',
 'e': 'ed02e_distance_from_return',
 'f': 'ed02f_four_policy_calibration',
 'g': 'ed02g_depth_pbs_profile',
 'h': 'ed02h_four_vs_depth_reconstruction',
 'i': 'ed02i_depth_calibration'}

PANEL_ROLE = {'a': 'four branch-policy definitions',
 'b': 'Naive versus Return policy probabilities',
 'c': 'Return versus Second-day policy probabilities',
 'd': 'one-policy-at-a-time contribution test',
 'e': 'policy-wise absolute distance from Return for Naive and Second day',
 'f': 'four-policy walk calibration',
 'g': 'depth-dependent P_BS profile and contrasts',
 'h': 'four-policy versus depth-varying reconstruction',
 'i': 'depth-varying walk calibration'}

PANEL_ORDER = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')


def _configure_source_style() -> None:
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.titlesize": 6.7,
            "axes.labelsize": 6.2,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 5.0,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
        }
    )


def _configure_mathtext_style(renderer: Any) -> None:
    renderer.configure_style()
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _configure_panel_style(renderer: Any) -> None:
    """Use a legible on-slide contract while retaining Main Figure hierarchy."""

    _configure_mathtext_style(renderer)
    renderer.GROUP_LABEL["naive"] = "Naive day"
    renderer.GROUP_LABEL["LTM"] = "Return day"
    renderer.GROUP_LABEL["experienced"] = "Second day"
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 0.70,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_style(renderer: Any) -> None:
    """Set panel typography and use color to identify experience groups."""

    _configure_panel_style(renderer)
    renderer.GROUP_MARKER.update({group: "o" for group in renderer.GROUP_ORDER})


def _draw_policy_schematic(renderer: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Draw the policy schematic at 0.8x scale with upright labels."""

    text = deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_TEXT)
    display_title_map = {
        "stem_to_branch": r"$\mathrm{P}_{\mathrm{SA}}$",
        "stem_forward": r"$\mathrm{P}_{\mathrm{SF}}$",
        "branch_to_stem": r"$\mathrm{P}_{\mathrm{BS}}$",
        "branch_forward": r"$\mathrm{P}_{\mathrm{BF}}$",
    }
    source_title_map = {
        "stem_to_branch": r"$P_{SA}$",
        "stem_forward": r"$P_{SF}$",
        "branch_to_stem": r"$P_{BS}$",
        "branch_forward": r"$P_{BF}$",
    }
    for row in text["rows"]:
        row["title"] = display_title_map[row["key"]]
        if row["key"] == "stem_to_branch":
            row["annotations"] = [
                {
                    "text": "preceding\nturn",
                    "xy": (3.285, 3.82),
                    "ha": "right",
                    "va": "center",
                }
            ]
    fig, ax, schematic, _ = renderer.render_grid_route_choice_schematic(
        layout={
            "panel_size": PANEL_SIZE_IN["a"],
            "tight_layout": False,
            "xlim": (-0.35, 3.15),
            "ylim": (-0.49375, 4.44375),
        },
        geometry=deepcopy(renderer.GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY),
        colors={**renderer.GRID_ROUTE_CHOICE_SCHEMATIC_COLORS},
        style={
            "cell_linewidth": 0.9 * SCHEMATIC_ARTIST_SCALE,
            "arrow_linewidth": 1.25 * SCHEMATIC_ARTIST_SCALE,
            "previous_arrow_linewidth": 1.25 * SCHEMATIC_ARTIST_SCALE,
            "arrow_mutation_scale": 7.5 * SCHEMATIC_ARTIST_SCALE,
            "title_fontsize": 6.2,
            "title_fontweight": "normal",
            "or_fontsize": 6.4 * SCHEMATIC_ARTIST_SCALE,
            "annotation_fontsize": 4.70,
            "annotation_lineheight": 1.05,
        },
        text=text,
        save=False,
    )
    ax.set_xlim(-0.35, 3.15)
    ax.set_ylim(-0.49375, 4.44375)
    fig.set_size_inches(*PANEL_SIZE_IN["a"], forward=True)
    rows = pd.DataFrame(
        [
            {
                "window_minutes": 30,
                "helper": "render_grid_route_choice_schematic",
                "row_key": row["key"],
                # Preserve the included plot-ready label text byte-for-byte.
                "policy_label": source_title_map[row["key"]],
            }
            for row in schematic["text"]["rows"]
        ]
    )
    return fig, {"schematic_rows": rows}


def fit_annotation_clearance(fig, annotation, obstacles, *, minimum_px=1.0):
    """Move only a colliding label by the smallest clear display-space shift."""
    text_box = annotation.get_window_extent(fig.canvas.get_renderer())

    def clear(box, obstacle):
        dx = max(obstacle.x0 - box.x1, box.x0 - obstacle.x1, 0.0)
        dy = max(obstacle.y0 - box.y1, box.y0 - obstacle.y1, 0.0)
        return math.hypot(dx, dy) >= minimum_px

    collisions = [box for box in obstacles if not clear(text_box, box)]
    if not collisions:
        return False
    gap = minimum_px + 0.05
    shifts = []
    for box in collisions:
        shifts.extend(((0.0, box.y1 + gap - text_box.y0),
                       (0.0, box.y0 - gap - text_box.y1),
                       (box.x1 + gap - text_box.x0, 0.0),
                       (box.x0 - gap - text_box.x1, 0.0)))
    for dx, dy in sorted(shifts, key=lambda shift: math.hypot(*shift)):
        moved = text_box.translated(dx, dy)
        inside = (moved.x0 >= fig.bbox.x0 and moved.x1 <= fig.bbox.x1
                  and moved.y0 >= fig.bbox.y0 and moved.y1 <= fig.bbox.y1)
        if inside and all(clear(moved, box) for box in obstacles):
            transform = annotation.get_transform()
            position = transform.transform(annotation.get_position())
            annotation.set_position(transform.inverted().transform(
                (position[0] + dx, position[1] + dy)))
            fig.canvas.draw()
            return True
    raise RuntimeError("Panel-a label cannot clear arrow/boxes within the canvas")


def build_panel_a(renderer: Any):
    global LAST_A_GEOMETRY
    fig, tables = _draw_policy_schematic(renderer)
    matches = [text for text in fig.axes[0].texts if text.get_text() == "preceding\nturn"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one preceding-turn annotation, found {len(matches)}")
    annotation = matches[0]
    annotation.set_position((PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y))
    fig.canvas.draw()
    renderer_px = fig.canvas.get_renderer()
    two_character_width_px = float(
        renderer_px.get_text_width_height_descent(
            PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE,
            annotation.get_fontproperties(),
            ismath=False,
        )[0]
    )
    axis = fig.axes[0]
    base_display = axis.transData.transform(
        (PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y)
    )
    shifted_data = axis.transData.inverted().transform(
        (base_display[0] - two_character_width_px, base_display[1])
    )
    annotation.set_position((float(shifted_data[0]), PANEL_A_ANNOTATION_Y))
    fig.canvas.draw()
    renderer_px = fig.canvas.get_renderer()
    text_box = annotation.get_window_extent(renderer_px)
    grey_arrows = [
        patch
        for patch in fig.axes[0].patches
        if patch.__class__.__name__ == "FancyArrowPatch"
        and all(abs(channel - 0.6549019608) < 1e-6 for channel in patch.get_edgecolor()[:3])
    ]
    boxes = [
        patch
        for patch in fig.axes[0].patches
        if patch.__class__.__name__ == "Rectangle"
    ]
    if len(grey_arrows) != 1 or not boxes:
        raise RuntimeError("Panel-a arrow/box artist contract changed")

    def clearance(first: Any, second: Any) -> float:
        dx = max(second.x0 - first.x1, first.x0 - second.x1, 0.0)
        dy = max(second.y0 - first.y1, first.y0 - second.y1, 0.0)
        return math.hypot(dx, dy)

    if fit_annotation_clearance(
        fig, annotation, [grey_arrows[0].get_extents(), *(box.get_extents() for box in boxes)]
    ):
        shifted_data = annotation.get_position()
        text_box = annotation.get_window_extent(fig.canvas.get_renderer())
    box_clearances = [
        (clearance(text_box, box.get_extents()), list(box.get_extents().bounds))
        for box in boxes
    ]
    nearest_box_clearance, nearest_box_bbox = min(box_clearances, key=lambda row: row[0])
    LAST_A_GEOMETRY = {
        "base_xy_data": [PANEL_A_ANNOTATION_BASE_X, PANEL_A_ANNOTATION_Y],
        "final_xy_data": [float(shifted_data[0]), float(shifted_data[1])],
        "two_character_sample": PANEL_A_ANNOTATION_TWO_CHARACTER_SAMPLE,
        "two_character_shift_px": two_character_width_px,
        "actual_left_shift_px": float(base_display[0] - axis.transData.transform(shifted_data)[0]),
        "text_bbox_px": list(text_box.bounds),
        "grey_arrow_bbox_px": list(grey_arrows[0].get_extents().bounds),
        "grey_arrow_clearance_px": clearance(text_box, grey_arrows[0].get_extents()),
        "box_clearance_px": nearest_box_clearance,
        "nearest_box_bbox_px": nearest_box_bbox,
    }
    if LAST_A_GEOMETRY["grey_arrow_clearance_px"] < 1.0 or LAST_A_GEOMETRY["box_clearance_px"] < 1.0:
        raise RuntimeError(f"Panel-a annotation clearance failed: {LAST_A_GEOMETRY}")
    # Keep each alternative label inside its middle grid cell, clear of arrows.
    # SA moves left at the same height; the other three follow their diagonal.
    or_positions = {
        "stem_to_branch": ((1.5096, 2.99), (1.4751335, 2.99)),
        "stem_forward": ((1.2124, 2.7928), (1.19268, 2.81252)),
        "branch_to_stem": ((1.8228, 0.718), (1.76712, 0.77368)),
        "branch_forward": ((1.166, 0.7064), (1.21298, 0.65942)),
    }
    or_labels = [text for text in axis.texts if text.get_text() == "or"]
    if len(or_labels) != 4:
        raise RuntimeError(f"Expected four alternative labels, found {len(or_labels)}")
    LAST_A_GEOMETRY["or_label_positions"] = []
    for key, (before, after) in or_positions.items():
        matches = [text for text in or_labels if all(
            math.isclose(value, expected, abs_tol=1e-8)
            for value, expected in zip(text.get_position(), before)
        )]
        if len(matches) != 1:
            raise RuntimeError(f"Panel-a alternative-label anchor changed for {key}")
        matches[0].set_position(after)
        LAST_A_GEOMETRY["or_label_positions"].append({
            "key": key, "before_xy_data": list(before), "after_xy_data": list(after),
        })
    return fig, tables


def clean_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def _draw_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    pairs = (("Bf", "Lf"), ("Ba", "Lo"))
    fig, axes = plt.subplots(1, 2, figsize=PANEL_SIZE_IN[panel])
    for ax, (x_metric, y_metric) in zip(axes, pairs):
        ax.axvline(
            renderer.RANDOM_REFERENCE[x_metric],
            color=renderer.REFERENCE,
            linestyle="--",
            linewidth=0.75,
        )
        ax.axhline(
            renderer.RANDOM_REFERENCE[y_metric],
            color=renderer.REFERENCE,
            linestyle="--",
            linewidth=0.75,
        )
        for group in groups:
            sub = points.loc[points["group"].eq(group)]
            color = renderer.GROUP_COLOR[group]
            ax.scatter(
                sub[x_metric],
                sub[y_metric],
                s=11,
                marker=renderer.GROUP_MARKER[group],
                facecolors=renderer.BACKGROUND,
                edgecolors=color,
                linewidths=0.55,
                alpha=0.64,
                label=renderer.GROUP_LABEL[group],
            )
            ax.scatter(
                [sub[x_metric].mean()],
                [sub[y_metric].mean()],
                s=28,
                marker=renderer.GROUP_MARKER[group],
                facecolors=color,
                edgecolors=renderer.BACKGROUND,
                linewidths=0.7,
                zorder=4,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks((0, 0.5, 1.0))
        ax.set_yticks((0, 0.5, 1.0))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(renderer.POLICY_LABEL[x_metric], labelpad=1.0)
        ax.set_ylabel(renderer.POLICY_LABEL[y_metric], labelpad=1.0)
        clean_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.85,
    )
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.17, top=0.80, wspace=0.34)
    return fig, {"session_points": points, "contrasts": contrasts}


def _align_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _draw_policy_scatter(
        renderer, points, contrasts, groups, panel
    )
    # Use the top-row canvas more fully so the next row does not appear
    # detached; values, axes, limits, and square aspect remain unchanged.
    fig.subplots_adjust(left=0.11, right=0.955, bottom=0.13, top=0.80, wspace=0.34)
    return fig, tables


def _style_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _align_policy_scatter(
        renderer, points, contrasts, groups, panel
    )
    # b/c use one shared three-experience legend in b; c has no legend.
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if panel == "b":
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker=renderer.GROUP_MARKER[group],
                markerfacecolor="white",
                markeredgecolor=renderer.GROUP_COLOR[group],
                markeredgewidth=0.7,
                markersize=4.0,
            )
            for group in renderer.GROUP_ORDER
        ]
        fig.legend(
            handles,
            [renderer.GROUP_LABEL[group] for group in renderer.GROUP_ORDER],
            loc="upper center",
            bbox_to_anchor=(0.56, 0.985),
            ncol=3,
            fontsize=LEGEND_SIZE,
            handlelength=0.60,
            handletextpad=0.12,
            columnspacing=0.42,
            borderaxespad=0,
        )
    # Retain a real white safety margin around the shared b/c axes.  The larger
    # tick labels otherwise reached the left SVG/PNG boundary even
    # though they looked acceptable in the composite.
    fig.subplots_adjust(left=0.125, right=0.955, bottom=0.13, top=0.84, wspace=0.34)
    return fig, tables


def hollow_path_collections(fig: plt.Figure) -> None:
    for ax in fig.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            edge = collection.get_edgecolors()
            if edge is None or len(edge) == 0:
                continue
            collection.set_facecolor("white")
            collection.set_linewidth(max(0.75, float(np.max(collection.get_linewidths()))))


def _hollow_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _style_policy_scatter(
        renderer, points, contrasts, groups, panel
    )
    # The experience key is a true figure-level asset shared by b and c.
    for legend in list(fig.legends):
        legend.remove()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    hollow_path_collections(fig)
    return fig, tables


def _mark_policy_scatter_means(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _hollow_policy_scatter(
        renderer, points, contrasts, groups, panel
    )
    # Every marker has the same s=11 area.  Fill alone identifies the
    # session-weighted mean; individual sessions remain hollow.
    for ax in fig.axes:
        current_group_color = None
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            sizes = collection.get_sizes()
            edge = collection.get_edgecolors()
            if len(sizes) and float(np.max(sizes)) <= 20:
                if edge is not None and len(edge):
                    current_group_color = edge[0]
                continue
            if len(sizes) and float(np.max(sizes)) > 20:
                collection.set_sizes(np.full_like(sizes, 11.0, dtype=float))
                if current_group_color is None:
                    raise RuntimeError("Cannot resolve b/c mean marker group color")
                collection.set_facecolor(current_group_color)
                collection.set_edgecolor(current_group_color)
    return fig, tables


def build_policy_scatter(
    renderer: Any,
    points: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _mark_policy_scatter_means(
        renderer, points, contrasts, groups, panel
    )
    # Means are true opaque disks.  Individual sessions remain hollow with the
    # maintained 0.64 alpha used for individual sessions.
    for ax in fig.axes:
        for collection in ax.collections:
            if not isinstance(collection, PathCollection):
                continue
            face = collection.get_facecolors()
            edge = collection.get_edgecolors()
            if len(face) and len(edge) and not np.allclose(face[0, :3], 1.0):
                collection.set_alpha(1.0)
                opaque = np.array(face, copy=True)
                opaque[:, 3] = 1.0
                collection.set_facecolors(opaque)
    return fig, tables


def _draw_policy_distance(renderer: Any, rows: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Draw absolute policy distances from the Return group mean."""
    metrics = list(renderer.POLICY_ORDER)
    natural = rows.copy()
    natural[metrics] = natural[metrics].apply(pd.to_numeric, errors="raise")
    means = natural.groupby("group")[metrics].mean()
    table = pd.DataFrame(
        {
            "window_minutes": 30,
            "metric": metrics,
            "policy_label": [renderer.POLICY_LABEL[key] for key in metrics],
            "return_mean": means.loc["LTM", metrics].to_numpy(dtype=float),
            "naive_mean": means.loc["naive", metrics].to_numpy(dtype=float),
            "second_day_mean": means.loc["experienced", metrics].to_numpy(dtype=float),
        }
    )
    table["abs_naive_minus_return"] = abs(table["naive_mean"] - table["return_mean"])
    table["abs_second_day_minus_return"] = abs(
        table["second_day_mean"] - table["return_mean"]
    )
    table["support"] = "natural primary-policy support"
    table["group_mean_weighting"] = "session weighted"
    counts = (
        natural.groupby("group", as_index=False)
        .agg(n_sessions=("group", "size"), n_mice=("animal_id", "nunique"))
        .assign(window_minutes=30, support="natural primary-policy support")
    )
    group_means = (
        means.reset_index()
        .merge(counts, on="group", how="left", validate="one_to_one")
    )

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["e"])
    x = np.arange(len(metrics), dtype=float)
    naive = table["abs_naive_minus_return"].to_numpy(dtype=float)
    second = table["abs_second_day_minus_return"].to_numpy(dtype=float)
    for xi, first, last in zip(x, naive, second):
        ax.plot(
            (xi - 0.10, xi + 0.10),
            (first, last),
            color=renderer.LIGHT,
            linewidth=0.9,
            zorder=1,
        )
    ax.scatter(
        x - 0.10,
        naive,
        s=19,
        color=renderer.GROUP_COLOR["naive"],
        edgecolor="white",
        linewidth=0.45,
        label="Naive",
        zorder=2,
    )
    ax.scatter(
        x + 0.10,
        second,
        s=19,
        color=renderer.GROUP_COLOR["experienced"],
        edgecolor="white",
        linewidth=0.45,
        label="Second day",
        zorder=2,
    )
    ax.set_xticks(x, table["policy_label"])
    ax.set_ylabel(r"$|\Delta P|$ from Return", labelpad=1.2)
    ax.set_ylim(-0.006, 0.125)
    ax.set_yticks((0, 0.05, 0.10))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, 1.005),
        ncol=2,
        handletextpad=0.18,
        columnspacing=0.50,
        borderaxespad=0,
    )
    clean_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.16, top=0.84)
    return fig, {"distance_from_return": table, "group_means": group_means}


def _align_policy_distance_axis(renderer: Any, rows: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _draw_policy_distance(renderer, rows)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    labels = ["Day 2" if label == "Second day" else label for label in labels]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.015),
        ncol=2,
        handletextpad=0.12,
        columnspacing=0.28,
        borderaxespad=0,
    )
    ax.set_xlim(-0.30, 3.30)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.12, top=0.94)
    return fig, tables


def _style_policy_distance(
    renderer: Any, rows: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _align_policy_distance_axis(renderer, rows)
    ax = fig.axes[0]
    handles, labels = ax.get_legend_handles_labels()
    old = ax.get_legend()
    if old is not None:
        old.remove()
    label_map = {"Naive": "Naive day", "Day 2": "Second day", "Second day": "Second day"}
    ax.legend(
        handles,
        [label_map.get(label, label) for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.46, 1.015),
        ncol=2,
        fontsize=5.3,
        handletextpad=0.06,
        columnspacing=0.12,
        markerscale=0.86,
        borderaxespad=0,
    )
    ax.set_ylabel(r"$|\Delta P|$ from Return day", labelpad=4.0)
    # The narrow panel needs both a larger left canvas reserve and an explicit
    # label coordinate: this leaves the ylabel inside the vector canvas while
    # separating it from the 0.00/0.05/0.10 tick labels.
    ax.yaxis.set_label_coords(-0.333, 0.5)
    fig.subplots_adjust(left=0.302, right=0.96, bottom=0.13, top=0.92)
    return fig, tables


def _mark_policy_distance(
    renderer: Any, rows: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _style_policy_distance(renderer, rows)
    ax = fig.axes[0]
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    # Raise the key only slightly; the axes and estimand remain unchanged.
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=renderer.GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in ("naive", "experienced")
    ]
    ax.legend(
        handles,
        ["Naive day", "Second day"],
        loc="upper center",
        bbox_to_anchor=(0.44, 1.035),
        ncol=2,
        fontsize=5.4,
        handletextpad=0.06,
        columnspacing=0.12,
        markerscale=0.86,
        borderaxespad=0,
    )
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        label = collection.get_label()
        if label in {"Naive", "Naive day"}:
            color = renderer.GROUP_COLOR["naive"]
        elif label == "Second day":
            color = renderer.GROUP_COLOR["experienced"]
        else:
            continue
        collection.set_facecolor("white")
        collection.set_edgecolor(color)
        collection.set_linewidth(0.85)
    return fig, tables


def normalize_policy_text(fig: Any, panel: str) -> None:
    """Apply upright notation and the main-figure role hierarchy before export."""

    for text in fig.findobj(match=Text):
        raw = text.get_text()
        replacement = UPRIGHT_TEXT.get(raw)
        if replacement is not None:
            text.set_text(replacement)

    if panel in {"b", "c"}:
        for ax in fig.axes:
            for label in (ax.xaxis.label, ax.yaxis.label):
                if "P" in label.get_text() or "mathrm{P}" in label.get_text():
                    label.set_fontsize(6.0)
    elif panel in {"d", "e"}:
        for ax in fig.axes:
            for axis_name, ticks, labels in (
                (
                    "x",
                    ax.get_xticks(),
                    [UPRIGHT_TEXT.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()],
                ),
                (
                    "y",
                    ax.get_yticks(),
                    [UPRIGHT_TEXT.get(label.get_text(), label.get_text()) for label in ax.get_yticklabels()],
                ),
            ):
                if any("\\mathrm{P}" in label for label in labels):
                    if axis_name == "x":
                        ax.set_xticks(ticks, labels)
                    else:
                        ax.set_yticks(ticks, labels)
            for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                if "P" in label.get_text() or "mathrm{P}" in label.get_text():
                    label.set_fontsize(5.4)
        if panel == "e":
            for ax in fig.axes:
                if "mathrm{P}" in ax.yaxis.label.get_text():
                    ax.yaxis.label.set_fontsize(6.0)
    elif panel == "g":
        for ax in fig.axes[:2]:
            if "mathrm{P}" in ax.yaxis.label.get_text():
                ax.yaxis.label.set_fontsize(6.0)
    elif panel == "h":
        for text in fig.findobj(match=Text):
            if "mathrm{P}" in text.get_text():
                text.set_fontsize(5.4)
    elif panel == "i":
        for text in fig.findobj(match=Text):
            if "mathrm{P}" in text.get_text():
                text.set_fontsize(6.2)


def _normalize_policy_distance_source(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Set upright policy labels before aligning the panel."""

    fig, tables = _mark_policy_distance(*args, **kwargs)
    normalize_policy_text(fig, "e")
    return fig, tables


def shorten_panel_e_spine(fig: Any) -> None:
    global LAST_E_GEOMETRY
    if not fig.axes:
        raise RuntimeError("Panel e has no maintained axis")
    axis = fig.axes[0]
    finite_y: list[float] = []
    for collection in axis.collections:
        offsets = collection.get_offsets()
        if getattr(offsets, "shape", (0, 0))[1:] == (2,):
            finite_y.extend(float(value) for value in offsets[:, 1] if math.isfinite(value))
    for line in axis.lines:
        finite_y.extend(float(value) for value in line.get_ydata() if math.isfinite(value))
    if not finite_y:
        raise RuntimeError("Panel e exposes no finite data artists")
    data_max = max(finite_y)
    low, high = axis.get_ylim()
    span = high - low
    spine_top = min(high - 0.04 * span, data_max + 0.04 * span)
    if not data_max < spine_top < high:
        raise RuntimeError(
            f"Cannot place panel-e spine above data and below annotations: "
            f"data_max={data_max}, ylim={(low, high)}"
        )
    axis.spines["left"].set_bounds(low, spine_top)
    LAST_E_GEOMETRY = {
        "data_max": data_max,
        "ylim": [low, high],
        "left_spine_bounds": [low, spine_top],
        "data_to_spine_clearance": spine_top - data_max,
        "spine_to_annotation_band_clearance": high - spine_top,
    }


def _fit_policy_distance_source(*args: Any, **kwargs: Any):
    fig, tables = _normalize_policy_distance_source(*args, **kwargs)
    shorten_panel_e_spine(fig)
    return fig, tables


def _place_policy_distance_axis(
    renderer: Any, rows: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _fit_policy_distance_source(renderer, rows)
    fig.subplots_adjust(
        left=0.302,
        right=0.96,
        bottom=0.239070128,
        top=0.92,
    )
    return fig, tables


def _normalize_policy_distance_labels(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, pd.DataFrame]]:
    """Set upright categorical policy labels."""

    fig, tables = _place_policy_distance_axis(*args, **kwargs)
    normalize_policy_text(fig, "e")
    return fig, tables


def build_panel_e(*args: Any, **kwargs: Any):
    fig, tables = _normalize_policy_distance_labels(*args, **kwargs)
    shorten_panel_e_spine(fig)
    return fig, tables


def depth_supported(frame):
    support = pd.Series(True, index=frame.index)
    for column in DEPTH_SUPPORT_COLUMNS:
        support &= pd.to_numeric(frame[column], errors="coerce").ge(
            MIN_DEPTH_EVENTS
        )
    return support


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def normalize_model(value: Any) -> str:
    text = str(value).lower()
    if "depth" in text:
        return "depth_stratified"
    if "four" in text:
        return "four_bias"
    raise ValueError(f"Unrecognized model: {value!r}")


def prepare_calibration(session_frame, summary_frame, model):
    if model not in {"four_bias", "depth_stratified"}:
        raise ValueError(model)
    simulated_column = (
        "four_own_half_cover_eff"
        if model == "four_bias"
        else "depth_own_half_cover_eff"
    )
    points = session_frame.loc[session_frame["group"].isin(GROUP_ORDER)].copy()
    if model == "depth_stratified":
        points = points.loc[depth_supported(points)].copy()
    points = numeric(points, ("real_half_cover_eff", simulated_column))
    points = points.dropna(
        subset=["real_half_cover_eff", simulated_column]
    ).reset_index(drop=True)
    expected = FOUR_COUNTS if model == "four_bias" else DEPTH_COUNTS
    observed = points.groupby("group").size().to_dict()
    if observed != expected:
        raise ValueError(f"{model} calibration rows drifted: {observed}")

    summary = summary_frame.copy()
    summary["model_norm"] = summary["model"].map(normalize_model)
    summary = summary.loc[
        summary["model_norm"].eq(model)
        & summary["group"].isin(GROUP_ORDER)
    ].copy()
    summary = numeric(summary, ("mean_error", "pearson_r", "mae", "slope"))
    if len(summary) != 3:
        raise ValueError(f"{model} calibration requires three summary rows")
    manuscript_columns = [
        "window_minutes",
        "model",
        "group",
        "group_label",
        "n_session_rows",
        "n_mice",
        "mean_error",
        "mae",
        "pearson_r",
        "slope",
        "real_mean",
        "simulated_mean",
        "n_zero_simulated_rows",
        "model_norm",
    ]
    return points, summary[manuscript_columns].reset_index(drop=True), simulated_column


def _draw_calibration(
    renderer: Any,
    session_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    model: str,
    panel: str,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    points, summary, simulated_column = renderer.prepare_calibration(
        session_frame, summary_frame, model
    )
    maximum = float(
        np.nanmax(points[["real_half_cover_eff", simulated_column]].to_numpy())
    )
    limit = max(0.62, np.ceil((maximum + 0.03) / 0.1) * 0.1)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=PANEL_SIZE_IN[panel],
        sharex=True,
        sharey=True,
    )
    for ax, group in zip(axes, renderer.GROUP_ORDER):
        sub = points.loc[points["group"].eq(group)]
        color = renderer.GROUP_COLOR[group]
        ax.plot(
            (0, limit),
            (0, limit),
            color=renderer.LIGHT,
            linestyle="--",
            linewidth=0.8,
        )
        ax.scatter(
            sub["real_half_cover_eff"],
            sub[simulated_column],
            s=13,
            facecolors=renderer.BACKGROUND,
            edgecolors=color,
            linewidths=0.60,
            alpha=0.78,
        )
        ax.set_title(
            renderer.GROUP_LABEL[group],
            color=color,
            pad=0.0,
            fontsize=6.5,
            y=1.13,
        )
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")
        ax.set_xticks((0, 0.4, 0.8))
        ax.set_yticks((0, 0.4, 0.8))
        clean_axes(ax)
    model_title = (
        "Four-policy model"
        if model == "four_bias"
        else r"Depth-varying $P_{BS}$ model"
    )
    fig.suptitle(model_title, y=0.975, fontsize=MODEL_TITLE_SIZE, fontweight="bold")
    fig.supxlabel("Observed efficiency", y=0.075, fontsize=BASE_FONT_SIZE)
    fig.supylabel("Simulated efficiency", x=0.025, fontsize=BASE_FONT_SIZE)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.84, wspace=0.12)
    return fig, {"session_points": points, "calibration_summary": summary}


def build_calibration(renderer: Any, session_frame: Any, summary_frame: Any, model: str, panel: str):
    fig, tables = _draw_calibration(renderer, session_frame, summary_frame, model, panel)
    if fig._suptitle is None:
        raise RuntimeError("Calibration model title is missing")
    fig._suptitle.set_y(MODEL_TITLE_Y)
    # Match b/c: one opaque, session-weighted coordinate mean per group.
    simulated = "four_own_half_cover_eff" if model == "four_bias" else "depth_own_half_cover_eff"
    points = tables["session_points"]
    means = []
    for axis, group in zip(fig.axes, renderer.GROUP_ORDER):
        rows = points.loc[points["group"].eq(group)]
        x, y = float(rows["real_half_cover_eff"].mean()), float(rows[simulated].mean())
        color = renderer.GROUP_COLOR[group]
        marker = axis.scatter([x], [y], s=11, marker="o", facecolors=color,
                              edgecolors=color, linewidths=0.75, alpha=1.0, zorder=4)
        marker.set_gid(f"ed02_{panel}_{group}_session_mean")
        means.append({"panel": panel, "model": model, "group": group,
                      "n_session_rows": len(rows), "n_mice": rows["animal_id"].nunique(),
                      "observed_mean": x, "simulated_mean": y,
                      "x_column": "real_half_cover_eff", "y_column": simulated,
                      "marker_area_pt2": 11, "alpha": 1.0})
    tables["display_group_means"] = renderer.pd.DataFrame(means)
    return fig, tables


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "pass"))


def prepare_depth_profile(
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = policy_rows.loc[policy_rows["group"].isin(GROUP_ORDER)].copy()
    if "primary_depth_support" in rows.columns:
        rows = rows.loc[as_bool(rows["primary_depth_support"])].copy()
    rows = numeric(rows, DEPTH_ORDER)
    melted = rows[["nickname", "animal_id", "group", *DEPTH_ORDER]].melt(
        id_vars=["nickname", "animal_id", "group"],
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"]).reset_index(drop=True)
    summary = (
        melted.groupby(["group", "metric"], as_index=False)["value"]
        .agg(
            mean="mean",
            q25=lambda values: np.quantile(values, 0.25),
            q75=lambda values: np.quantile(values, 0.75),
            n_session_rows="size",
        )
    )
    contrasts = contrast_frame.copy()
    if "support_policy" in contrasts.columns:
        contrasts = contrasts.loc[contrasts["support_policy"].eq("all_depth_strata_n_ge_10")]
    contrasts = contrasts.loc[
        contrasts["contrast_id"].isin(("return_vs_naive", "second_vs_return"))
        & contrasts["metric"].isin(DEPTH_ORDER)
    ].copy()
    contrasts = numeric(
        contrasts,
        (
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "q_bh3_within_window_contrast",
        ),
    )
    if len(summary) != 9 or len(contrasts) != 6:
        raise ValueError(
            f"Depth profile requires 9 group summaries and 6 contrasts; got {len(summary)}, {len(contrasts)}"
        )
    return rows.reset_index(drop=True), summary.reset_index(drop=True), contrasts.reset_index(drop=True)


def _draw_depth_profile(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows, summary, contrasts = renderer.prepare_depth_profile(policy_rows, contrast_frame)
    fig, (ax, ax_delta) = plt.subplots(
        2,
        1,
        figsize=PANEL_SIZE_IN["g"],
        sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0)},
    )
    x = np.arange(3, dtype=float)
    for group in renderer.GROUP_ORDER:
        sub = summary.loc[summary["group"].eq(group)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], renderer.DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        means = sub["mean"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        color = renderer.GROUP_COLOR[group]
        ax.fill_between(x, q25, q75, color=color, alpha=0.09, linewidth=0)
        ax.plot(
            x,
            means,
            marker=renderer.GROUP_MARKER[group],
            color=color,
            markersize=3.6,
            linewidth=1.0,
            label="Day 2" if group == "experienced" else renderer.GROUP_LABEL[group],
        )
    ax.set_ylabel(r"$P_{BS}$", labelpad=1.0)
    ax.set_ylim(0, 0.95)
    ax.set_xticks(x, [])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.24),
        ncol=3,
        handlelength=0.75,
        handletextpad=0.15,
        columnspacing=0.55,
    )
    ax.spines["bottom"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)

    specs = (
        ("return_vs_naive", "Return - Naive", renderer.GROUP_COLOR["LTM"], "o", 0.06),
        (
            "second_vs_return",
            "Day 2 - Return",
            renderer.GROUP_COLOR["experienced"],
            "s",
            -0.06,
        ),
    )
    ax_delta.axhline(0, color=renderer.LIGHT, linewidth=0.75)
    for contrast_id, label, color, marker, offset in specs:
        sub = contrasts.loc[contrasts["contrast_id"].eq(contrast_id)].copy()
        sub["metric"] = pd.Categorical(sub["metric"], renderer.DEPTH_ORDER, ordered=True)
        sub = sub.sort_values("metric")
        mean = sub["beta_target_minus_baseline"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        ax_delta.errorbar(
            x + offset,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            fmt=marker,
            color=color,
            markersize=3.4,
            capsize=1.7,
            elinewidth=0.85,
            label=label,
        )
    ax_delta.set_ylabel(r"$\Delta P_{BS}$", labelpad=1.0)
    ax_delta.set_ylim(-0.025, 0.34)
    ax_delta.set_xticks(x, ("Shallow\n1-2", "Middle\n3-4", "Deep\n5"))
    ax_delta.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.25),
        ncol=2,
        handlelength=0.75,
        handletextpad=0.15,
        columnspacing=0.55,
    )
    clean_axes(ax_delta)
    if "window_minutes" not in summary:
        summary = summary.copy()
        summary.insert(0, "window_minutes", 30)
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.14, top=0.87, hspace=0.47)
    return fig, {
        "supported_session_rows": rows,
        "group_profiles": summary,
        "depth_contrasts": contrasts,
    }


def _align_depth_profile_axis(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _draw_depth_profile(renderer, policy_rows, contrast_frame)
    # The legends remain inside the same scientific roles, but the complete
    # two-axis panel is lowered and extended to match i's visible ink span.
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.15, top=0.88, hspace=0.43)
    return fig, tables


def replace_legend(
    ax: plt.Axes,
    labels: list[str],
    *,
    ncol: int,
    bbox: tuple[float, float],
    fontsize: float,
) -> None:
    handles, _ = ax.get_legend_handles_labels()
    old = ax.get_legend()
    if old is not None:
        old.remove()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=bbox,
        ncol=ncol,
        fontsize=fontsize,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )


def _style_depth_profile(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _align_depth_profile_axis(renderer, policy_rows, contrast_frame)
    ax, ax_delta = fig.axes[:2]
    replace_legend(
        ax,
        ["Naive day", "Return day", "Second day"],
        ncol=2,
        bbox=(0.58, 1.02),
        fontsize=5.4,
    )
    replace_legend(
        ax_delta,
        ["Return day − Naive day", "Second day − Return day"],
        ncol=1,
        bbox=(0.58, 1.02),
        fontsize=5.4,
    )
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.15, top=0.89, hspace=0.44)
    return fig, tables


def _mark_depth_profile(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Use hollow circles for every experience group."""

    fig, tables = _style_depth_profile(renderer, policy_rows, contrast_frame)
    for ax in fig.axes[:2]:
        for line in ax.lines:
            label = line.get_label()
            if label.startswith("_"):
                continue
            if line.get_marker() not in {"None", "", "none"}:
                line.set_marker("o")
                line.set_markerfacecolor("white")
                line.set_markeredgecolor(line.get_color())
                line.set_markeredgewidth(0.80)
        for container in ax.containers:
            lines = getattr(container, "lines", None)
            if not lines:
                continue
            marker_line = lines[0]
            if isinstance(marker_line, Line2D):
                marker_line.set_marker("o")
                marker_line.set_markerfacecolor("white")
                marker_line.set_markeredgecolor(marker_line.get_color())
                marker_line.set_markeredgewidth(0.80)

    ax, ax_delta = fig.axes[:2]
    for target in (ax, ax_delta):
        legend = target.get_legend()
        if legend is not None:
            legend.remove()
    top_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor=renderer.GROUP_COLOR[group], markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in renderer.GROUP_ORDER
    ]
    ax.legend(
        top_handles,
        [renderer.GROUP_LABEL[group] for group in renderer.GROUP_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.58, 1.02),
        ncol=2,
        fontsize=5.4,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )
    contrast_handles = [
        Line2D(
            [], [], linestyle="none", marker="o", markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=0.80, markersize=4.0,
        )
        for color in (renderer.GROUP_COLOR["LTM"], renderer.GROUP_COLOR["experienced"])
    ]
    ax_delta.legend(
        contrast_handles,
        ["Return day − Naive day", "Second day − Return day"],
        loc="upper center",
        bbox_to_anchor=(0.58, 1.02),
        ncol=1,
        fontsize=5.4,
        handlelength=0.65,
        handletextpad=0.12,
        columnspacing=0.32,
        borderaxespad=0,
    )
    return fig, tables


def _place_depth_profile_axis(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _mark_depth_profile(renderer, policy_rows, contrast_frame)
    fig.subplots_adjust(
        left=0.21,
        right=0.98,
        bottom=0.256662900,
        top=0.89,
        hspace=0.44,
    )
    # Hold both stacked y labels at the shared horizontal anchor.
    # Matplotlib otherwise repositions the top label from the new tick extents
    # and lets its leftmost antialiasing touch the source crop boundary.
    for ax in fig.axes[:2]:
        ax.yaxis.set_label_coords(-0.143095, 0.5)
    return fig, tables


def compress_artist_x(ax: plt.Axes, *, center: float, scale: float) -> None:
    """Compress data artists only; do not mutate shared tick labels."""

    transform = lambda values: center + (np.asarray(values, dtype=float) - center) * scale
    for line in ax.lines:
        if line.get_transform() == ax.transData:
            line.set_xdata(transform(line.get_xdata(orig=False)))
    for collection in ax.collections:
        if collection.get_transform() != ax.transData:
            continue
        if isinstance(collection, LineCollection):
            segments = collection.get_segments()
            for segment in segments:
                segment[:, 0] = transform(segment[:, 0])
            collection.set_segments(segments)
        elif isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                path.vertices[:, 0] = transform(path.vertices[:, 0])


def _label_depth_profile(
    renderer: Any,
    policy_rows: pd.DataFrame,
    contrast_frame: pd.DataFrame,
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Compress categorical spacing and show depth labels on the lower axis."""

    fig, tables = _place_depth_profile_axis(renderer, policy_rows, contrast_frame)
    ax, ax_delta = fig.axes[:2]
    original_ticks = np.asarray(ax_delta.get_xticks(), dtype=float)
    if len(original_ticks) != 3:
        raise RuntimeError(f"Expected three panel-g depth ticks, found {original_ticks}")
    for target in (ax, ax_delta):
        compress_artist_x(
            target,
            center=1.0,
            scale=PANEL_G_GROUP_SCALE,
        )

    compressed_ticks = 1.0 + (original_ticks - 1.0) * PANEL_G_GROUP_SCALE
    # Shared axes reuse locator/formatter state.  Set the depth labels
    # on the bottom axis once, then hide only the upper-axis label artists.
    ax_delta.set_xticks(compressed_ticks, DEPTH_LABELS)
    ax.tick_params(axis="x", labelbottom=False)
    ax_delta.tick_params(axis="x", labelbottom=True)

    fig.subplots_adjust(
        left=PANEL_G_LEFT,
        right=0.98,
        bottom=0.256662900,
        top=0.89,
        hspace=0.44,
    )
    for target in (ax, ax_delta):
        target.yaxis.set_label_coords(PANEL_G_YLABEL_X, 0.5)
    return fig, tables


def build_panel_g(renderer: Any, policy_rows: Any, contrast_frame: Any):
    fig, tables = _label_depth_profile(renderer, policy_rows, contrast_frame)
    if len(fig.axes) < 2:
        raise RuntimeError("Panel g no longer exposes its two maintained axes")
    upper = fig.axes[0].get_legend()
    lower = fig.axes[1].get_legend()
    if upper is None or lower is None:
        raise RuntimeError("Panel-g direct label groups are missing")
    upper.set_bbox_to_anchor(PANEL_G_UPPER_ANCHOR, transform=fig.axes[0].transAxes)
    lower.set_bbox_to_anchor(PANEL_G_LOWER_ANCHOR, transform=fig.axes[1].transAxes)
    return fig, tables


def _save_cropped_panel(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    """Save one consistently cropped editable/vector/raster panel payload."""

    panel = stem.name[4]
    if panel not in CROP_BBOX_IN:
        raise KeyError(f"No crop box for {stem.name}")
    crop = Bbox.from_extents(*CROP_BBOX_IN[panel])
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {
            "bbox_inches": crop,
            "pad_inches": 0,
            "facecolor": "white",
        }
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, **kwargs)
        outputs.append(path)

    ppt_path = stem.parent.parent / "ppt_vector" / f"{stem.name}.svg"
    ppt_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(
            ppt_path,
            bbox_inches=crop,
            pad_inches=0,
            facecolor="white",
        )
    plt.close(fig)
    return outputs


def save_figure(fig: Any, stem: Path, dpi: int):
    global LAST_G_SAVE_GEOMETRY
    panel = stem.name[4]
    normalize_policy_text(fig, panel)
    if panel == "g":
        old_width, old_height = fig.get_size_inches()
        positions = [axis.get_position().frozen() for axis in fig.axes]
        new_height = old_height + PANEL_G_TOP_RESERVE_IN
        fig.set_size_inches(old_width, new_height, forward=True)
        for axis, position in zip(fig.axes, positions, strict=True):
            axis.set_position(
                (
                    position.x0,
                    position.y0 * old_height / new_height,
                    position.width,
                    position.height * old_height / new_height,
                )
            )
        fig.canvas.draw()
        rendered = fig.canvas.get_renderer()
        upper = fig.axes[0].get_legend()
        upper_bbox_in = [
            value / fig.dpi for value in upper.get_window_extent(rendered).bounds
        ]
        crop = CROP_BBOX_IN["g"]
        crop_top = max(crop[3], upper_bbox_in[1] + upper_bbox_in[3] + 3.0 / 72)
        CROP_BBOX_IN["g"] = (crop[0], crop[1], crop[2], crop_top)
        LAST_G_SAVE_GEOMETRY = {
            "figure_size_in": list(fig.get_size_inches()),
            "upper_legend_bbox_in": upper_bbox_in,
            "crop_bbox_in": list(CROP_BBOX_IN["g"]),
            "upper_text_top_clearance_pt": (
                crop_top - upper_bbox_in[1] - upper_bbox_in[3]
            ) * 72,
        }
    return _save_cropped_panel(fig, stem, dpi)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _draw_policy_contributions(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    """Draw the signed contrast on the native efficiency scale."""

    rows = frame.copy()
    rows["parameter"] = pd.Categorical(
        rows["parameter"], renderer.POLICY_ORDER, ordered=True
    )
    rows = rows.sort_values("parameter").reset_index(drop=True)
    value_columns = (
        "delta_efficiency_mean",
        "delta_efficiency_ci_low",
        "delta_efficiency_ci_high",
    )
    rows[list(value_columns)] = rows[list(value_columns)].apply(
        pd.to_numeric, errors="raise"
    )

    mean = rows["delta_efficiency_mean"].to_numpy(dtype=float)
    low = rows["delta_efficiency_ci_low"].to_numpy(dtype=float)
    high = rows["delta_efficiency_ci_high"].to_numpy(dtype=float)
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["d"])
    ax.axvline(0, color=renderer.REFERENCE, linewidth=0.85, zorder=0)
    for yi, value, lo, hi in zip(y, mean, low, high):
        ax.hlines(yi, 0, value, color=PANEL_D_NEUTRAL, linewidth=1.25, zorder=1)
        ax.errorbar(
            value,
            yi,
            xerr=[[value - lo], [hi - value]],
            fmt="o",
            color=PANEL_D_NEUTRAL,
            markerfacecolor="white",
            markeredgecolor=PANEL_D_NEUTRAL,
            markeredgewidth=0.90,
            markersize=5.0,
            capsize=2.0,
            elinewidth=1.0,
            zorder=2,
        )
    ax.set_yticks(
        y, [renderer.POLICY_LABEL[str(item)] for item in rows["parameter"]]
    )
    ax.set_xlabel("Exploration efficiency\ndifference", labelpad=1.5)
    ax.set_xlim(-0.04, 0.096)
    ax.set_xticks((-0.04, 0, 0.04, 0.08))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_ylim(-0.55, 3.55)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    fig.subplots_adjust(left=0.28, right=0.985, bottom=0.16, top=0.96)
    return fig, {"policy_contributions": rows}


def build_panel_d(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _draw_policy_contributions(renderer, frame)
    ax = fig.axes[0]
    ax.set_ylim(-0.9, 3.9)
    fig.subplots_adjust(
        left=0.28,
        right=0.985,
        bottom=0.206887380,
        top=0.949497558,
    )
    return fig, tables




def _draw_reconstruction_gap(renderer: Any, frame: pd.DataFrame) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    rows = frame.copy()
    rows["model_norm"] = rows["model"].map(renderer.normalize_model)
    rows["model_norm"] = pd.Categorical(
        rows["model_norm"], ("four_bias", "depth_stratified"), ordered=True
    )
    rows = rows.sort_values("model_norm").reset_index(drop=True)
    for column in (
        "delta_real",
        "delta_simulated",
        "fraction_recapitulated",
        "fraction_ci_low",
        "fraction_ci_high",
    ):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    observed = float(rows["delta_real"].iloc[0])
    rows["delta_simulated_ci_low"] = rows["delta_real"] * rows["fraction_ci_low"]
    rows["delta_simulated_ci_high"] = rows["delta_real"] * rows["fraction_ci_high"]

    fig, ax = plt.subplots(figsize=PANEL_SIZE_IN["h"])
    ax.axvline(observed, color=renderer.INK, linewidth=1.0)
    ax.text(
        observed,
        1.38,
        "Observed",
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=renderer.INK,
    )
    labels = ("Four policy", r"Depth $P_{BS}$")
    colors = (renderer.FOUR, renderer.DEPTH)
    markers = ("o", "s")
    for yi, (_, row), color, marker in zip((1, 0), rows.iterrows(), colors, markers):
        value = float(row["delta_simulated"])
        low = float(row["delta_simulated_ci_low"])
        high = float(row["delta_simulated_ci_high"])
        ax.errorbar(
            value,
            yi,
            xerr=[[value - low], [high - value]],
            fmt=marker,
            color=color,
            markersize=4.8,
            capsize=2.0,
            elinewidth=1.1,
        )
    ax.set_yticks((1, 0), labels)
    ax.set_ylim(-0.48, 1.58)
    ax.set_xlim(0, max(0.34, float(rows["delta_simulated_ci_high"].max()) + 0.04))
    ax.set_xlabel("Return - Naive\nefficiency gap", labelpad=2.0)
    ax.spines["left"].set_visible(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.96, bottom=0.24, top=0.92)
    return fig, {"four_vs_depth_gap": rows}


def _style_reconstruction_gap(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _draw_reconstruction_gap(renderer, frame)
    ax = fig.axes[0]
    ax.set_yticks([])
    rows = tables["four_vs_depth_gap"].copy()
    rows["model_norm"] = pd.Categorical(
        rows["model_norm"], ("four_bias", "depth_stratified"), ordered=True
    )
    rows = rows.sort_values("model_norm").reset_index(drop=True)
    labels = ("Four-policy", r"Depth $P_{BS}$")
    for yi, row, label in zip((1, 0), rows.itertuples(index=False), labels):
        value = float(row.delta_simulated)
        ax.text(
            value - 0.045,
            yi,
            label,
            ha="right",
            va="center",
            fontsize=5.4,
            color=renderer.INK,
        )
    observed = float(rows["delta_real"].iloc[0])
    for text in list(ax.texts):
        if text.get_text() == "Observed":
            text.set_x(observed - 0.020)
            text.set_ha("right")
            text.set_fontsize(5.4)
    ax.set_xlabel("Return day − Naive day\nefficiency gap", labelpad=2.5)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.95)
    return fig, tables


def _mark_reconstruction_gap(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _style_reconstruction_gap(renderer, frame)
    ax = fig.axes[0]
    for container in ax.containers:
        lines = getattr(container, "lines", None)
        if not lines:
            continue
        marker_line = lines[0]
        if not isinstance(marker_line, Line2D):
            continue
        marker_line.set_marker("o")
        marker_line.set_markerfacecolor("white")
        marker_line.set_markeredgecolor(marker_line.get_color())
        marker_line.set_markeredgewidth(0.90)
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            edge = collection.get_edgecolors()
            collection.set_facecolor("white")
            if edge is not None and len(edge):
                collection.set_edgecolor(edge)
    return fig, tables


def build_panel_h(
    renderer: Any, frame: pd.DataFrame
) -> tuple[plt.Figure, dict[str, pd.DataFrame]]:
    fig, tables = _mark_reconstruction_gap(renderer, frame)
    ax = fig.axes[0]
    ax.set_ylim(-0.65, 1.65)
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.247288341,
        top=0.881224438,
    )
    return fig, tables


def _save_matched_panel(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    """Save one consistently cropped editable/vector/raster panel payload."""

    panel = stem.name[4]
    if panel not in CROP_BBOX_IN_base:
        raise KeyError(f"No crop box for {stem.name}")
    crop = Bbox.from_extents(*CROP_BBOX_IN_base[panel])
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {
            "bbox_inches": crop,
            "pad_inches": 0,
            "facecolor": "white",
        }
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, **kwargs)
        outputs.append(path)

    ppt_path = stem.parent.parent / "ppt_vector" / f"{stem.name}.svg"
    ppt_path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(
            ppt_path,
            bbox_inches=crop,
            pad_inches=0,
            facecolor="white",
        )
    plt.close(fig)
    return outputs


def read_30min(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"animal_id": str})
    if "window_minutes" not in frame.columns:
        raise ValueError(f"Quantitative source lacks window_minutes: {path}")
    minutes = pd.to_numeric(frame["window_minutes"], errors="coerce")
    frame = frame.loc[minutes.eq(WINDOW_MINUTES)].copy()
    if frame.empty:
        raise ValueError(f"No 30-minute rows in {path}")
    frame["window_minutes"] = WINDOW_MINUTES
    return frame.reset_index(drop=True)


def validate_source_cohorts(frames):
    strict_counts = frames["strict_policy_rows"].groupby("group").size().to_dict()
    if strict_counts != {"LTM": 58, "naive": 23}:
        raise ValueError(f"Four-policy cohort drifted: {strict_counts}")
    natural_counts = frames["natural_policy_rows"].groupby("group").size().to_dict()
    if natural_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural cohort drifted: {natural_counts}")
    simulation = frames["natural_simulation_rows"]
    simulation_counts = simulation.groupby("group").size().to_dict()
    if simulation_counts != {"LTM": 58, "experienced": 74, "naive": 23}:
        raise ValueError(f"Natural calibration cohort drifted: {simulation_counts}")
    depth = simulation.loc[depth_supported(simulation)]
    depth_counts = depth.groupby("group").size().to_dict()
    if depth_counts != {"LTM": 58, "experienced": 74, "naive": 22}:
        raise ValueError(f"Depth-supported calibration cohort drifted: {depth_counts}")
    for key, frame in frames.items():
        if not pd.to_numeric(frame["window_minutes"], errors="coerce").eq(30).all():
            raise ValueError(f"Non-30-minute row survived in {key}")


def load_sources(
    strict_root: Path,
    natural_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    strict_root = Path(strict_root).resolve()
    natural_root = Path(natural_root).resolve()
    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for key, filename in SOURCE_FILES.items():
        root = strict_root if key.startswith("strict_") else natural_root
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[key] = read_30min(path)
        paths[key] = path
    validate_source_cohorts(frames)
    return frames, paths


def correct_policy_descriptions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "metric" in out.columns:
        out["display_description"] = out["metric"].map(POLICY_SHORT)
        if "description" in out.columns:
            out["description"] = out["display_description"]
    return out


def prepare_two_axis_scatter(
    policy_rows: pd.DataFrame,
    contrasts: pd.DataFrame,
    groups: tuple[str, str],
    contrast_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = policy_rows.loc[policy_rows["group"].isin(groups)].copy()
    points = numeric(points, (*POLICY_ORDER,))
    stats = contrasts.loc[
        contrasts["contrast_id"].eq(contrast_id) & contrasts["metric"].isin(POLICY_ORDER)
    ].copy()
    stats = correct_policy_descriptions(stats)
    stats = numeric(
        stats,
        (
            "beta_target_minus_baseline",
            "ci_low",
            "ci_high",
            "q_bh4_within_window_contrast",
        ),
    )
    stats["metric"] = pd.Categorical(stats["metric"], POLICY_ORDER, ordered=True)
    stats = stats.sort_values("metric").reset_index(drop=True)
    if len(stats) != 4:
        raise ValueError(f"Expected four {contrast_id} policy contrasts")
    return points.reset_index(drop=True), stats


def save_shared_legend(fig: plt.Figure, output_root: Path, dpi: int) -> list[Path]:
    editable_root = output_root / "shared_assets"
    ppt_root = output_root / "ppt_vector"
    editable_root.mkdir(parents=True, exist_ok=True)
    ppt_root.mkdir(parents=True, exist_ok=True)
    stem = editable_root / SHARED_LEGEND_STEM
    outputs: list[Path] = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"bbox_inches": None, "transparent": True}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        if suffix == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, **kwargs)
        outputs.append(path)
    ppt_path = ppt_root / f"{SHARED_LEGEND_STEM}.svg"
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(ppt_path, bbox_inches=None, transparent=True)
    plt.close(fig)
    return outputs


def build_shared_bc_legend(output_root: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(196 / 96, 18 / 96), facecolor="none")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=GROUP_COLOR[group],
            markeredgewidth=0.80,
            markersize=4.0,
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles,
        [GROUP_LABEL[group] for group in GROUP_ORDER],
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=3,
        fontsize=5.4,
        handlelength=0.60,
        handletextpad=0.81,
        columnspacing=0.90,
        borderaxespad=0,
        frameon=False,
    )
    return save_shared_legend(fig, output_root, dpi)

source = SimpleNamespace(BACKGROUND=BACKGROUND, DEPTH=DEPTH, DEPTH_ORDER=DEPTH_ORDER, FOUR=FOUR, GRID_ROUTE_CHOICE_SCHEMATIC_COLORS=GRID_ROUTE_CHOICE_SCHEMATIC_COLORS, GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY=GRID_ROUTE_CHOICE_SCHEMATIC_GEOMETRY, GRID_ROUTE_CHOICE_SCHEMATIC_TEXT=GRID_ROUTE_CHOICE_SCHEMATIC_TEXT, GROUP_COLOR=GROUP_COLOR, GROUP_LABEL=GROUP_LABEL, GROUP_MARKER=GROUP_MARKER, GROUP_ORDER=GROUP_ORDER, INK=INK, LIGHT=LIGHT, POLICY_LABEL=POLICY_LABEL, POLICY_ORDER=POLICY_ORDER, RANDOM_REFERENCE=RANDOM_REFERENCE, REFERENCE=REFERENCE, configure_style=_configure_source_style, load_sources=load_sources, normalize_model=normalize_model, pd=pd, prepare_calibration=prepare_calibration, prepare_depth_profile=prepare_depth_profile, prepare_two_axis_scatter=prepare_two_axis_scatter, render_grid_route_choice_schematic=render_grid_route_choice_schematic)
save_matched_figure = _save_matched_panel
