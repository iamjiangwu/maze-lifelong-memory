"""Source-render the author's Day 1 / Day 91 Figure 1 layout.

Reuse frozen accepted renderers and values. No analysis, selection or current
record is changed. The supplied PNG is embedded byte-for-byte; its surrounding
fully transparent margin is excluded by an SVG viewport, with no pixel edits.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Ellipse, Rectangle
from matplotlib.transforms import Affine2D
from figure1_complete_timeline_source import build_complete_rewarded_timelines
import numpy as np
import pandas as pd
from PIL import Image
import fitz

ROOT = Path(__file__).resolve().parents[1]
VOR = ROOT / "docs/figure_version_records"
MAIN = VOR / "main_figure1_2026-08-13_round12_no_port_mouse1636"
ED1 = VOR / "extended_data_fig01_2026-09-08_round22_panel_h_axis_labels"
NS = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", NS)
ET.register_namespace("xlink", XLINK)
MM = 72 / 25.4
WIDTH, HEIGHT = 180 * MM, 196 * MM
# Actual accepted main-figure export: 180 mm wide, SVG viewBox width 576.
OLD_EXPORT_SCALE = 180 * MM / 576
OLD_TIMELINE_SCALE = 1.0098194403820917 * OLD_EXPORT_SCALE
TIMELINE_AXES_HEIGHT_RATIO = 0.28
VARIANTS = {
    "A_day1_day91": [("434RRday1", "Day 1"), ("434-3day1", "Day 91")],
}
SLOTS_MM = {
    "a": (3, 3.5, 55, 47), "b": (61, 3.5, 116, 47), "c": (3, 52, 174, 14),
    "d": (3, 67, 65, 57),
    "e": (70, 67, 106, 56), "f": (3, 122, 65, 35),
    "g": (3, 159, 65, 35), "h": (70, 122, 106, 35),
    "i": (70, 159, 51, 35), "j": (124, 159, 52, 35),
}
MAPPING = {"d": "a", "c": "b", "e": "c", "f": "d", "g": "g", "h": "h", "i": "i", "j": "j"}
OLD_DISPLAY = {"a": "a", "b": "b", "c": "c", "d": "d", "g": "e", "h": "f", "i": "g", "j": "h"}
DISPLAY_ADJUSTMENTS = {}
AXES_MM = {
    "e": [(74, 71, 47, 19), (128, 71, 47, 19), (74, 95, 47, 19), (128, 95, 47, 19)],
    "f": [(13, 126, 54, 25)],
    "g": [(13, 160, 54, 27)],
    "h": [(74, 126, 47, 19), (128, 126, 47, 19)],
    "i": [(74, 160, 47, 27)],
    "j": [(128, 160, 47, 27)],
}
SIGNIFICANCE_AUDIT = {}


def add_standard_water_drop(ax, xy, coordinates, key, offset=None):
    vertices = [(0,.66),(-.60,.05),(-.48,-.52),(0,-.58),(.48,-.52),(.60,.05),(0,.66),(0,.66)]
    path = MplPath(vertices, [MplPath.MOVETO]+[MplPath.CURVE4]*6+[MplPath.CLOSEPOLY])
    bounds = path.get_extents()
    scale = 3*MM/bounds.height
    drawing = DrawingArea(bounds.width*scale, bounds.height*scale, 0, 0)
    transform = Affine2D().scale(scale).translate(-bounds.x0*scale, -bounds.y0*scale)
    drop = PathPatch(path.transformed(transform), facecolor="#36a3ff", edgecolor="#0866b2", linewidth=.65, joinstyle="round", gid="water_outline_"+key)
    drawing.add_artist(drop)
    highlight = Ellipse(transform.transform((-.13,.02)), width=.15*scale, height=.30*scale, angle=28, facecolor="#c9ecff", edgecolor="none", alpha=.9)
    drawing.add_artist(highlight)
    kwargs = {} if offset is None else {"xybox":offset,"boxcoords":"offset points"}
    ax.add_artist(AnnotationBbox(drawing,xy,xycoords=coordinates,frameon=False,pad=0,box_alignment=(.5,.5),annotation_clip=False,zorder=12,**kwargs))


def standardize_maze_and_water(fig, letter):
    if letter == "d":
        ax = fig.axes[0]
        ax.lines[0].set_linewidth(.62)
        ax.lines[0].set_gid("maze_wall_d")
        for patch in list(ax.patches):
            rgb = np.asarray(patch.get_facecolor()[:3])
            if isinstance(patch, Rectangle) and .5 < rgb[0] < .95 and np.ptp(rgb) < 1e-5:
                patch.set_facecolor("#d8d8d8")
                patch.set_edgecolor("#d8d8d8")
                patch.set_linewidth(.35)
            if np.allclose(rgb, matplotlib.colors.to_rgb("#36a3ff")) or np.allclose(rgb, matplotlib.colors.to_rgb("#c9ecff")):
                patch.remove()
        add_standard_water_drop(ax,(15.18,6.12),ax.transData,"d_0")
    elif letter in {"e","h"}:
        for index,ax in enumerate(fig.axes):
            for artist in list(ax.artists):
                if not isinstance(artist,AnnotationBbox):
                    continue
                water = any(isinstance(child,PathPatch) and np.allclose(child.get_facecolor()[:3],matplotlib.colors.to_rgb("#36a3ff")) for child in artist.offsetbox.get_children())
                if water:
                    xy,offset = artist.xy,artist.xybox
                    artist.remove()
                    add_standard_water_drop(ax,xy,ax.transAxes,f"{letter}_{index}",offset)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tag(name):
    return f"{{{NS}}}{name}"


def panel_at_final_size(preview, fig, letter):
    """Native point-sized artists; no scaling of an exported panel."""
    sx, sy, sw, sh = SLOTS_MM[letter]
    fig.set_layout_engine(None)
    fig.set_size_inches(sw / 25.4, sh / 25.4)
    for ax in fig.axes:
        for text in ax.texts:
            text.set_fontsize(6.5)
        ax.xaxis.label.set_size(6.5)
        ax.yaxis.label.set_size(6.5)
        ax.title.set_size(6.5)
        ax.tick_params(axis="both", labelsize=6.5, pad=2, length=2.5, width=.6)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(6.5)
            if letter == "g":
                # Keep the key clear of i's y-axis title after the columns widen.
                legend.set_bbox_to_anchor((1-2.5/AXES_MM["g"][0][2], 1))
        if letter == "d":
            for text in ax.texts:
                if text.get_text() in {str(i) for i in range(6)}:
                    text.set_fontsize(5.1)
    if letter in AXES_MM:
        for ax, (x, y, w, h) in zip(fig.axes, AXES_MM[letter]):
            ax.set_position([(x-sx)/sw, 1-(y-sy+h)/sh, w/sw, h/sh])
            ax.set_aspect("auto")
            ax.xaxis.labelpad = 3
            ax.yaxis.labelpad = 3
        if letter in {"f", "i", "j"}:
            repack_significance(fig.axes[0], letter, AXES_MM[letter][0][3])
        if letter == "e":
            for ax in fig.axes[:2]:
                ax.tick_params(axis="x", bottom=False)
    elif letter == "d":
        # Fill the shared row down to e's xlabel band rather than leaving
        # that full band empty below the schematic.
        for ax in fig.axes:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            unit = 51/15
            x = 10.5 + (min(xlim)+.5)*unit
            y = 71 + (min(ylim)+.5)*unit
            w, h = abs(xlim[1]-xlim[0])*unit, abs(ylim[1]-ylim[0])*unit
            ax.set_position([(x-sx)/sw, 1-(y-sy+h)/sh, w/sw, h/sh])
    standardize_maze_and_water(fig, letter)
    if letter in {"f", "i", "j"}:
        for ax in fig.axes:
            for idx, points in enumerate(c for c in ax.collections if isinstance(c, PathCollection)):
                points.set_alpha(.72)
                points.set_zorder(2)
                points.set_gid(f"bar_scatter_{letter}_{idx}")
            for idx, container in enumerate(c for c in ax.containers if isinstance(c, ErrorbarContainer)):
                for part, artist in enumerate(container.get_children()):
                    artist.set_zorder(4)
                    artist.set_gid(f"errorbar_{letter}_{idx}_{part}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = []
    for ax in fig.axes:
        box = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        axes.append({"x_mm": sx+box.x0*25.4, "y_mm": sy+sh-box.y1*25.4, "width_mm": box.width*25.4, "height_mm": box.height*25.4})
    native_bounds = fig.get_tightbbox(renderer)
    buffer = io.StringIO()
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(buffer, format="svg", transparent=True, bbox_inches=None, metadata={"Date": None})
    plt.close(fig)
    root = ET.fromstring(buffer.getvalue())
    actual = [float(v) for v in root.attrib["viewBox"].split()]
    assert np.allclose(actual[2:], [sw*MM, sh*MM], atol=1e-5), (letter, actual, sw*MM, sh*MM)
    preview._prefix_svg_ids(root, "final_" + letter + "_")
    return {"children": [n for n in root if n.tag != tag("metadata")],
            "viewbox_width": sw*MM, "viewbox_height": sh*MM,
            "axes_final_mm": axes,
            "content_bounds_mm": [native_bounds.x0*25.4, native_bounds.y0*25.4, native_bounds.x1*25.4, native_bounds.y1*25.4],
            "font_sizes_pt": {"axis_labels_and_titles": 6.5, "ticks_and_legend": 6.5, "stars": 6.5}}


def repack_significance(ax, letter, height_mm):
    brackets = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if len(x) == 4 and x[0] == x[1] and x[2] == x[3] and x[0] != x[2]:
            brackets.append(line)
    stars = [t for t in ax.texts if re.fullmatch(r"\*+", t.get_text())]
    assert len(brackets) == len(stars) == (4 if letter == "f" else 3), (letter, len(brackets), len(stars))
    pairs = [(float(line.get_xdata()[0]), float(line.get_xdata()[2]), text.get_text()) for line, text in zip(brackets, stars)]
    for item in [*brackets, *stars]:
        item.remove()
    ymin,ymax = (0,7500) if letter == "f" else ((0,6500) if letter == "i" else (.22,1.27))
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0, 2000, 4000, 6000] if letter != "j" else [.25,.5,.75,1.0])
    maximum = max(float(np.asarray(c.get_offsets())[:, 1].max()) for c in ax.collections if len(c.get_offsets()) > 1)
    first_height = (maximum-ymin) / (ymax-ymin) * height_mm + 1.35
    rows = []
    previous = None
    for idx, (x1, x2, star) in enumerate(pairs):
        line_mm = first_height + idx * 1.8
        y = line_mm / height_mm
        cap = .30 / height_mm
        start, stop = x1, x2
        ax.plot([start, start, stop, stop], [y-cap, y, y, y-cap], transform=ax.get_xaxis_transform(), color="black", lw=.65, clip_on=False, gid=f"sig_{letter}_bracket_{idx}")
        glyph = TextPath((0, 0), star, size=6.5, prop=FontProperties(family="Arial", weight="bold")).get_extents()
        # Position the visible asterisk glyph, not its empty font descent box.
        baseline = y + (.30 - glyph.y0/MM)/height_mm
        ax.text((x1+x2)/2, baseline, star, transform=ax.get_xaxis_transform(), ha="center", va="baseline", fontsize=6.5, fontweight="bold", color="black", clip_on=False, gid=f"sig_{letter}_star_{idx}")
        glyph_top = line_mm + .30 + glyph.height/MM
        assert glyph_top < height_mm, (letter, glyph_top, height_mm)
        rows.append({"groups": [x1, x2], "stars": star, "bracket_height_mm": line_mm, "star_gap_mm": .30, "visible_star_height_mm": glyph.height/MM, "pitch_mm": None if previous is None else line_mm-previous, "endpoints_at_bar_centers": True})
        previous = line_mm
    SIGNIFICANCE_AUDIT[letter] = {"original_comparisons_and_star_tokens": pairs, "placements": rows, "axis_ylim": [ymin, ymax], "data_unchanged": True}


def compress_timeline(renderer, size, key, audit):
    fig = renderer(size)
    fig.canvas.draw()
    old_size = fig.get_size_inches().copy()
    boxes = [ax.get_position().frozen() for ax in fig.axes]
    old_axis_h = boxes[0].height * old_size[1]
    new_h = old_axis_h * TIMELINE_AXES_HEIGHT_RATIO
    bottom, top, gap = 0.32, 0.16, 0.25
    nrows = 2 if key == "c" else 1
    total = bottom + top + nrows * new_h + (nrows - 1) * gap
    fig.set_size_inches(old_size[0], total)
    rows = sorted({round(box.y0, 5) for box in boxes}, reverse=True)
    for ax, box in zip(fig.axes, boxes):
        row = rows.index(round(box.y0, 5))
        lower = bottom + (nrows - 1 - row) * (new_h + gap)
        ax.set_position([box.x0, lower / total, box.width, new_h / total])
    audit[key] = {
        "old_native_axes_height_inches": float(old_axis_h),
        "new_native_axes_height_inches": float(new_h),
        "axes_height_ratio": TIMELINE_AXES_HEIGHT_RATIO,
        "x_range": [0, 6500], "source_values_changed": False,
        "text_and_marker_sizes_changed": False,
    }
    return fig


def compact_quantitative(renderer, size, key, factor):
    fig = renderer(size)
    old_size = fig.get_size_inches().copy()
    ax = fig.axes[0]
    box = ax.get_position().frozen()
    def values():
        return {
            "limits": [ax.get_xlim(), ax.get_ylim()],
            "collections": [np.asarray(c.get_offsets()).tolist() for c in ax.collections],
            "lines": [[np.asarray(l.get_xdata()).tolist(), np.asarray(l.get_ydata()).tolist()] for l in ax.lines],
            "bars": [[p.get_x(), p.get_y(), p.get_width(), p.get_height()] for p in ax.patches if hasattr(p, "get_width")],
            "text": [t.get_text() for t in ax.texts],
        }
    before = values()
    h = box.height * old_size[1] * factor
    bottom, top = .38, .10
    height = bottom + h + top
    fig.set_size_inches(old_size[0], height)
    ax.set_position([box.x0, bottom / height, box.width, h / height])
    assert before == values()
    DISPLAY_ADJUSTMENTS[key] = {"axes_height_ratio": factor, "values_axes_limits_stars_and_marker_data_unchanged": True}
    return fig


def native_panels(rebuilder, output, audit):
    # The maintained entrypoint binds every path to the frozen record and
    # replaces the timeline loader with the frozen six-session CSV.
    saved_export = rebuilder._export_svg_bundle
    rebuilder._export_svg_bundle = lambda svg, stem: [svg]
    rebuilder._render_main1(MAIN, output / "source_rebuild", {"selected_record": MAIN.relative_to(ROOT).as_posix()})
    rebuilder._export_svg_bundle = saved_export
    preview = sys.modules["record_local_main1_preview"]
    rewarded, timeline_audit = build_complete_rewarded_timelines(MAIN)
    archived = pd.read_csv(MAIN / "plot_ready/figure1_endnode_timelines.csv")
    complete = pd.concat([rewarded, archived.loc[archived.nickname.str.startswith("1636")]], ignore_index=True)
    complete.to_csv(output / "complete_endnode_timelines.csv", index=False)
    (output / "TIMELINE_SOURCE_PARITY.json").write_text(json.dumps(timeline_audit, indent=2)+"\n", encoding="utf-8")
    def complete_timeline(nickname, *, lower_threshold=63, upper_threshold=126):
        rows = complete.loc[complete.nickname.eq(nickname) & complete.endnode.between(lower_threshold, upper_threshold)]
        return rows.endnode.to_numpy(int), rows.time_in_maze_s.to_numpy(float)
    preview.render_endnode_timeline_grid_panel.__globals__["extract_endnode_timeline"] = complete_timeline
    panels = {}
    reverse = {key: letter for letter, key in MAPPING.items()}
    for key, renderer in preview.PANEL_RENDERERS.items():
        if key == "b":
            continue  # Replaced by the author's requested horizontal timeline.
        size = preview.PANEL_SIZES[key]
        fig = renderer(size)
        def values():
            result = []
            for ax in fig.axes:
                lines = []
                for line in ax.lines:
                    x = np.asarray(line.get_xdata()).tolist()
                    y = np.asarray(line.get_ydata()).tolist()
                    if reverse[key] in {"f", "i", "j"} and len(x) == 4 and x[0] == x[1] and x[2] == x[3] and x[0] != x[2]:
                        continue  # Significance placement is audited separately.
                    lines.append([x, y])
                result.append({"xlim": ax.get_xlim(),
                    "offsets": [np.asarray(c.get_offsets()).tolist() for c in ax.collections],
                    "segments": [np.asarray(c.get_segments()).tolist() for c in ax.collections if hasattr(c, "get_segments")],
                    "bars": [[p.get_x(), p.get_y(), p.get_width(), p.get_height()] for p in ax.patches if hasattr(p, "get_x") and hasattr(p, "get_width")],
                    "lines": lines, "annotations": [t.get_text() for t in ax.texts]})
            return json.dumps(result, sort_keys=True, default=lambda value: value.item())
        before = values()
        fig.canvas.draw()
        old_height = fig.get_tightbbox(fig.canvas.get_renderer()).height * 25.4 * OLD_TIMELINE_SCALE
        old_axes_h = fig.axes[0].get_position().height * fig.get_size_inches()[1] * 25.4 * OLD_TIMELINE_SCALE
        panels[key] = panel_at_final_size(preview, fig, reverse[key])
        assert before == values()
        DISPLAY_ADJUSTMENTS[key] = {"data_and_annotation_hash_before": hashlib.sha256(before.encode()).hexdigest(), "data_and_annotation_hash_after": hashlib.sha256(values().encode()).hexdigest(), "data_bars_errorbars_xlimits_annotations_unchanged": True, "export_scale": 1.0, "axes_final_mm": panels[key]["axes_final_mm"], "font_sizes_pt": panels[key]["font_sizes_pt"]}
        if key in {"c", "h"}:
            bounds = panels[key]["content_bounds_mm"]
            new_height = bounds[3] - bounds[1]
            audit[key] = {"old_figure_panel_height_mm": old_height, "new_figure_panel_height_mm": new_height, "overall_height_ratio": new_height/old_height, "old_figure_axes_height_mm": old_axes_h, "new_figure_axes_height_mm": 19, "axes_height_ratio": 19/old_axes_h, "source_values_changed": False, "font_sizes_pt": {"axis_labels_and_titles": 6.5, "ticks": 6.5}, "author_refinement": "Moderate compression of e/h from 22 to 19 mm, not the rejected 10.5 mm."}
    return preview, panels


def trajectories(preview, ed, data, cases, name):
    # Author's sketch places examples above the right-hand timeline column.
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.0, "svg.fonttype": "none", "pdf.fonttype": 42})
    count = len(cases)
    width, height = 116 / 25.4, 47 / 25.4
    fig = plt.figure(figsize=(width, height))
    gap = 1.5 / 25.4
    maze_width = 52 / 25.4
    x0 = 0
    norm = Normalize(0, float(data.elapsed_minutes.max()))
    segment_counts = {}
    for i, (session, title) in enumerate(cases):
        ax = fig.add_axes([(x0 + i * (maze_width + gap)) / width, 0, maze_width / width, 44.5/47])
        rows = data.loc[data.nickname.eq(session)]
        ed.draw_trajectory(ax, rows, title, norm)
        ax.lines[0].set_gid(f"maze_wall_b_{i}")
        ax.set_ylim(14.6, -.6)
        ax.set_title(title, fontsize=7, pad=2)
        segment_counts[session] = int(rows.draw_to_next.iloc[:-1].sum())
    cax = fig.add_axes([107/116, 1/47, 1.1/116, 42/47])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax)
    cb.solids.set_rasterized(False)
    cb.set_ticks([0, norm.vmax])
    cb.set_ticklabels(["0", f"{norm.vmax:.0f}"])
    cb.set_label("Time in maze (min)", fontsize=6.2, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1.5)
    cb.outline.set_linewidth(.4)
    panel = panel_at_final_size(preview, fig, "b")
    return panel, segment_counts


def session_timeline(preview):
    """A true day-coordinate axis with each two-day cycle named explicitly."""
    fig, ax = plt.subplots(figsize=(6.70, .69))
    fig.subplots_adjust(left=.002, right=.998, bottom=.25, top=.95)
    ax.set_xlim(-33, 396)
    ax.set_ylim(-.65, .56)
    ax.axis("off")
    ax.annotate("", xy=(389, 0), xytext=(-28, 0), arrowprops={"arrowstyle": "->", "color": "#858585", "lw": .8, "mutation_scale": 6})
    rows = []
    for n, day in enumerate([1, 91, 181, 271, 361]):
        first = "Naive" if n == 0 else "Return"
        first_color = preview.FIGURE1_NAIVE_COLOR if n == 0 else preview.FIGURE1_RETURN_COLOR
        # Each day-pair glyph is centered at its exact calendar coordinate.
        # Minimum physical width separates the two condition colors clearly.
        day_per_mm = 429/(174*.996)
        y_per_mm = 1.21/(14*.70)
        for part, color in enumerate([first_color, preview.FIGURE1_SECOND_DAY_COLOR]):
            center = day+.5 + (-.45 if part == 0 else .45)*day_per_mm
            ax.add_patch(plt.Rectangle((center-.35*day_per_mm, -1.5*y_per_mm), .7*day_per_mm, 3*y_per_mm,
                         facecolor=color, edgecolor="none", zorder=4, gid=f"session_tick_{day}_{part}"))
        ax.text(day + .5, -.40, f"Days {day}\u2013{day+1}", ha="center", va="center", fontsize=6.5)
        if n < 4:
            ax.text(day + 45.5, .20, "Rest", ha="center", va="center", fontsize=6.5, color="#555555")
        rows.extend([{"day": day, "cycle": n+1, "condition": first}, {"day": day+1, "cycle": n+1, "condition": "Second day"}])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=4.5, markeredgewidth=0, color=color, label=label) for color, label in [(preview.FIGURE1_NAIVE_COLOR, "Naive"), (preview.FIGURE1_RETURN_COLOR, "Return"), (preview.FIGURE1_SECOND_DAY_COLOR, "Second day")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, frameon=False, fontsize=6.5, handletextpad=.35, columnspacing=2, borderaxespad=0, borderpad=0)
    return panel_at_final_size(preview, fig, "c"), rows


def place(root, panel, letter, slot, *, fixed_scale=None):
    x, y, w, h = [v * MM for v in slot]
    scale = 1.0
    pw, ph = panel["viewbox_width"] * scale, panel["viewbox_height"] * scale
    assert pw <= w + .1 and ph <= h + .1, (letter, pw / MM, ph / MM, slot)
    xx, yy = x + (w - pw) / 2, y
    group = ET.SubElement(root, tag("g"), {"id": f"panel_{letter}", "transform": f"translate({xx} {yy}) scale({scale})"})
    for child in panel["children"]:
        group.append(deepcopy(child))
    label(root, letter, x - 2 * MM, y - 1 * MM)
    return {"x_mm": xx / MM, "y_mm": yy / MM, "width_mm": pw / MM, "height_mm": ph / MM, "scale": scale}


def label(root, letter, x, y):
    if letter in {"d","e"}:
        y = 69.5*MM
    elif letter in {"f","h"}:
        y = 124.5*MM
    y = max(y, 3.8 * MM)
    node = ET.SubElement(root, tag("text"), {"x": str(x), "y": str(y), "font-family": "Arial", "font-size": "9", "font-weight": "bold", "fill": "black"})
    node.text = letter


def maze_image(root, image_path, bounds):
    x, y, w, h = [v * MM for v in SLOTS_MM["a"]]
    left, top, right, bottom = bounds
    viewport = ET.SubElement(root, tag("svg"), {
        "id": "panel_a", "x": str(x), "y": str(y), "width": str(w), "height": str(h),
        "viewBox": f"{left} {top} {right-left} {bottom-top}",
        "preserveAspectRatio": "xMidYMid meet", "overflow": "hidden",
    })
    ET.SubElement(viewport, tag("image"), {"width": "12800", "height": "8000", "x": "0", "y": "0", f"{{{XLINK}}}href": "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode()})
    label(root, "a", x - 2 * MM, y - 1 * MM)
    add_panel_a_water_port(root)


def add_panel_a_water_port(root):
    """Place the shared vector icon next to the visible port cylinder."""
    for old in list(root):
        if old.get("id")=="a_water_port":
            root.remove(old)
    fig = plt.figure(figsize=(5/25.4,5/25.4))
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    add_standard_water_drop(ax,(.5,.5),ax.transAxes,"a_0")
    buffer = io.StringIO()
    with matplotlib.rc_context({"savefig.bbox":None,"savefig.pad_inches":0}):
        fig.savefig(buffer,format="svg",transparent=True,bbox_inches=None,metadata={"Date":None})
    plt.close(fig)
    icon = ET.fromstring(buffer.getvalue())
    ids = {node.get("id"):"water_a_"+node.get("id") for node in icon.iter() if node.get("id")}
    for node in icon.iter():
        for attr,value in list(node.attrib.items()):
            if attr=="id":
                node.set(attr,ids[value])
            else:
                for old,new in ids.items():
                    value = value.replace("#"+old+")","#"+new+")")
                    if value=="#"+old:
                        value="#"+new
                node.set(attr,value)
    group = ET.SubElement(root,tag("g"),{"id":"a_water_port","transform":f"translate({47*MM} {32.5*MM})"})
    for child in icon:
        if child.tag!=tag("metadata"):
            group.append(child)


def export(svg, stem):
    # Browser SVG export is the repository precedent: PyMuPDF's SVG importer
    # does not preserve matplotlib CSS font shorthand and nested clipping.
    rebuilder = sys.modules.get("figure1_frozen_rebuilder") or load(VOR / "rebuild_current_main_figures.py", "figure1_frozen_rebuilder")
    pdf = stem.with_suffix(".pdf")
    html = stem.with_suffix(".html")
    html.write_text("<!doctype html><html><head><meta charset='utf-8'><style>"
                    "@page{size:180mm 196mm;margin:0}html,body{margin:0;width:180mm;height:196mm;overflow:hidden}"
                    "body>svg{display:block;width:180mm!important;height:196mm!important}"
                    "</style></head><body>" + svg.read_text(encoding="utf-8") + "</body></html>", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="fig1_pdf_") as profile:
        fresh_pdf = Path(profile) / "fresh_export.pdf"
        command = [str(rebuilder._find_chrome()), "--headless=new", "--disable-gpu", "--no-sandbox", "--disable-software-rasterizer", "--no-pdf-header-footer", "--print-to-pdf-no-header", f"--user-data-dir={Path(profile)/'browser'}", f"--print-to-pdf={fresh_pdf}", html.resolve().as_uri()]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        rebuilder._wait_for_file(fresh_pdf)
        with fitz.open(fresh_pdf) as fresh:
            assert len(fresh)==1
            assert np.allclose([fresh[0].rect.width, fresh[0].rect.height], [WIDTH,HEIGHT], atol=1.0)
        # Replace the file atomically so a preview reader cannot retain a
        # partially rewritten destination or prevent an in-place write.
        with tempfile.NamedTemporaryFile(prefix=stem.name+"_", suffix=".pdf", dir=pdf.parent, delete=False) as handle:
            staged_pdf = Path(handle.name)
        try:
            shutil.copy2(fresh_pdf, staged_pdf)
            staged_pdf.replace(pdf)
        finally:
            staged_pdf.unlink(missing_ok=True)
    with fitz.open(pdf) as doc:
        assert len(doc) == 1
        metadata = doc.metadata
        metadata["subject"] = "Figure 1 source SVG SHA256: " + sha(svg)
        doc.set_metadata(metadata)
        doc.saveIncr()
        page = doc[0]
        for dpi, suffix in [(180, "_preview.png"), (300, "_300dpi.png")]:
            page.get_pixmap(dpi=dpi, alpha=False).save(stem.with_name(stem.name + suffix))
        return {"pdf_pages": len(doc), "pdf_size_points": list(page.rect), "pdf_text_chars": len(page.get_text()), "pdf_images": len(page.get_images()), "export_method": "repository Chrome SVG-to-PDF; PyMuPDF raster preview"}


def finish_comparison(output):
    allpdf = fitz.open()
    for name in VARIANTS:
        with fitz.open(output / ("figure1_" + name + ".pdf")) as doc:
            allpdf.insert_pdf(doc)
    allpdf.save(output / "figure1_three_alternatives.pdf")
    allpdf.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.6), dpi=150)
    labels = ["A | Day 1 + Day 91", "B | Day 1 + Day 2 + Day 91", "C | All four days"]
    for ax, name, title in zip(axes, VARIANTS, labels):
        ax.imshow(plt.imread(output / ("figure1_" + name + "_preview.png")))
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, fontfamily="Arial")
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.95, wspace=.025)
    fig.savefig(output / "figure1_comparison.png", facecolor="white", dpi=180)
    plt.close(fig)


def finish_record(output, manifest):
    manifest["figure_size_mm"] = [180, 196]
    manifest["spacing_refinement"] = {"previous_canvas_height_mm": 202, "canvas_height_mm": 196,
        "schematic_body_side_mm": 51, "previous_schematic_body_side_mm": 44,
        "timeline_strip_height_mm": 14, "timeline_tick_size_mm": [.7, 3],
        "day_pair_glyph_centers": "Exact calendar midpoint; minimum physical glyph width for legibility",
        "endnode_axis_size_mm": [47,19], "right_column_gutter_mm": 7,
        "rest_label_axis_y": .20, "previous_rest_label_axis_y": .38}
    manifest["quantitative_display_adjustments"] = DISPLAY_ADJUSTMENTS
    manifest["significance_layout"] = SIGNIFICANCE_AUDIT
    manifest["shared_style"] = {"maze_wall_pt":.62,"maze_gray":"#d8d8d8","water_drop_height_mm":3,"water_outline_pt":.65,"significance_line_pt":.65,"significance_cap_mm":.3,"significance_star_font_pt":6.5,"bar_scatter_alpha":.72,"bar_scatter_zorder":2,"errorbar_zorder":4}
    manifest["builder"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    timeline_parity = json.loads((output / "TIMELINE_SOURCE_PARITY.json").read_text())
    manifest["source_hashes"].update(timeline_parity["source_hashes"])
    for path in [ROOT / "scripts/figure1_complete_timeline_source.py", ROOT / "src/ca_analysis/trajectory.py"]:
        manifest["source_hashes"][str(path)] = sha(path)
    manifest["status"] = "SINGLE_DAY1_DAY91_DRAFT"
    manifest["timeline_source_parity"] = "TIMELINE_SOURCE_PARITY.json"
    manifest["panel_a_water_port"] = {"center_mm":[49.5,35],"path_height_mm":3,"placement":"Beside the cylinder inserted into the right maze wall","source_image_pixel_edits":False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("""# Figure 1: Day 1 and Day 91

One draft at 180 x 196 mm, following the author's supplied composition.
Panel b includes only Day 1 and Day 91.

| New panel | Content |
| --- | --- |
| a | Supplied physical-maze illustration |
| b | Selected ED1a trajectories through first reward |
| c | Day-coordinate timeline with day pairs, Rest intervals and one condition color key |
| d | Existing maze schematic (old a) |
| e | Compressed rewarded end-node timelines (old c) |
| f | First long bout across return months (old d) |
| g | Exploration-efficiency distribution (old e) |
| h | Compressed latent end-node timelines (old f) |
| i | Naive/latent/return first-long-bout comparison (old g) |
| j | First-visit route score (old h) |

The six end-node plotting frames are 47 x 19 mm: modestly compressed from the
22-mm review, while substantially taller than the rejected 10.5-mm version.
The colored maze d spans the same row as e's complete 2-by-2 scatter grid.
Its maze body fills a 51-mm square, enlarged from 44 mm into previously empty
space. The timeline has a 14-mm strip, with separated 0.7-by-3-mm condition
marks centered on each day pair. Narrower gutters widen the right plot columns.
The first-long-bout plot f shares a row and top axis position with h.
The final three plots g/i/j share top and bottom axis positions and have
27-mm plotting heights. All panels are rendered at final size and assembled
at scale 1. Ordinary text is 6.5 pt, panel letters 9 pt and schematic decision
numerals 5.1 pt. Data, marker sizes, axis ticks and text are not stretched.

In f/i/j, significance brackets start and end at the category bar centers.
They use 1.8-mm spacing, 0.3-mm caps and 0.3-mm visible-glyph clearance.
Comparisons and star strings are unchanged. Display limits provide room for
the annotations above all individual values. Data, bars, error bars and
statistics remain unchanged. The trajectory masks and original shared
0-to-16-minute color scale are preserved.

Maze backgrounds b/d share 0.62-pt walls and the same gray fill. Water-drop
outlines in d/e/h have identical 3-mm path height and 0.65-pt strokes.
All f/i/j significance lines use 0.65-pt strokes. Export uses a fresh PDF
destination and embeds its source SVG hash before generating the preview.
The f/i/j scatter points share alpha 0.72 and z-order 2; black error bars use
z-order 4 to remain visible above the individual points, matching panel j.

The accepted figure shows all end nodes 63-126, but its frozen rewarded CSV
accidentally contained only nodes 95-126. The complete draft-local event table
is recovered from saved trajectories and matched point-for-point and color
for color to the accepted artwork. TIMELINE_SOURCE_PARITY.json documents all
4,380 rewarded events and exact archived-subset parity. Latent events are
unchanged. The accepted record is not modified.

The full supplied PNG is embedded byte-for-byte. Only its fully transparent
outer margin is excluded by the viewport; no visible pixels, contrast or colors
are changed. The exact original is saved under inputs/maze_3d_original.png.
A separate vector water-drop icon sits beside the visible cylindrical water
port in panel a; it uses the same 3-mm shape and style as d/e/h.

This remains a layout draft. ED1, accepted records, manuscript text and
captions are unchanged.

Rebuild from repository root:
`python scripts/build_figure1_layout_alternatives.py --image output/figures/figure1_day1_day91_compact_2026-09-25_refined/inputs/maze_3d_original.png`

Input hashes, panel geometry, timeline labels and trajectory segment counts are
in manifest.json and the companion CSVs. The editable SVG, PDF and 300-dpi PNG
are exported together.
The browser export preserves SVG clipping and editable fonts; the sole raster
image in each PDF is the user-provided 3D view.
""", encoding="utf-8")
    outputs = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "OUTPUT_HASHES.json"}
    (output / "OUTPUT_HASHES.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=Path(r"C:\Users\jiangwu\Downloads\maze_3d_figure_12800x8000 (2).png"))
    ap.add_argument("--output-dir", type=Path, default=ROOT / "output/figures/figure1_day1_day91_compact_2026-09-25_refined")
    ap.add_argument("--export-existing", choices=["all", *VARIANTS], help="Rerender existing source SVGs without repeating native source plotting")
    ap.add_argument("--refresh-record", action="store_true", help="Refresh documentation and hashes without changing rendered artwork")
    ap.add_argument("--add-panel-a-water", action="store_true", help="Add the shared icon to existing artwork without rerendering other panels")
    args = ap.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.add_panel_a_water:
        manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        DISPLAY_ADJUSTMENTS.update(manifest["quantitative_display_adjustments"])
        SIGNIFICANCE_AUDIT.update(manifest["significance_layout"])
        for name in manifest["variants"]:
            stem = output/("figure1_"+name)
            tree = ET.parse(stem.with_suffix(".svg"))
            root = tree.getroot()
            unchanged = [ET.tostring(n) for n in root if n.get("id")!="a_water_port"]
            before = np.asarray(Image.open(stem.with_name(stem.name+"_preview.png"))).copy()
            add_panel_a_water_port(root)
            assert unchanged==[ET.tostring(n) for n in root if n.get("id")!="a_water_port"]
            tree.write(stem.with_suffix(".svg"),encoding="utf-8",xml_declaration=True)
            manifest["variants"][name].update(export(stem.with_suffix(".svg"),stem))
            after = np.asarray(Image.open(stem.with_name(stem.name+"_preview.png")))
            yy,xx = np.where(np.any(before!=after,axis=2))
            bbox = [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
            assert 46*180/25.4 < xx.min() < xx.max() < 53*180/25.4 and 31*180/25.4 < yy.min() < yy.max() < 39*180/25.4, bbox
            manifest["panel_a_only_edit_audit"] = {"original_svg_elements_unchanged":True,"pixels_outside_new_icon_unchanged":True,"changed_pixel_bbox_180dpi":bbox}
        finish_record(output,manifest)
        print(json.dumps(manifest["panel_a_only_edit_audit"],indent=2))
        return
    if args.refresh_record:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        DISPLAY_ADJUSTMENTS.update(manifest["quantitative_display_adjustments"])
        SIGNIFICANCE_AUDIT.update(manifest["significance_layout"])
        finish_record(output, manifest)
        return
    if args.export_existing:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        DISPLAY_ADJUSTMENTS.update(manifest["quantitative_display_adjustments"])
        SIGNIFICANCE_AUDIT.update(manifest["significance_layout"])
        for name in (VARIANTS if args.export_existing == "all" else [args.export_existing]):
            stem = output / ("figure1_" + name)
            result = export(stem.with_suffix(".svg"), stem)
            manifest["variants"][name].update(result)
            print(name, result, flush=True)
        if args.export_existing == "all":
            finish_comparison(output)
            (output / "inputs").mkdir(exist_ok=True)
            if args.image.resolve() != (output / "inputs/maze_3d_original.png").resolve():
                shutil.copy2(args.image, output / "inputs/maze_3d_original.png")
            assert all(sha(p) == h for p, h in manifest["source_hashes"].items())
        finish_record(output, manifest)
        return
    # Read and hash sources before rendering; existing accepted files are read-only.
    source_paths = [args.image, MAIN / "figure1_version_record.svg", ED1 / "drawn_values/PANEL_A_TRAJECTORY_PLOT_DATA_ROUND21.csv.gz", VOR / "CURRENT_MAIN_FIGURES.md", VOR / "CURRENT_EXTENDED_DATA_FIGURES.md"]
    source_paths += sorted((MAIN / "plot_ready").glob("*.csv"))
    source_paths += [MAIN / "source_code/preview_figure1_assembly.py", MAIN / "source_code/figure_panels.py", ED1 / "source_code/render_extended_data_figure1_round4.py", ED1 / "source_code/render_grs2026_first_maze_trial_trajectories.py", VOR / "SOURCE_REBUILD_MAIN_RUNTIME.zip"]
    source_hashes = {str(p.resolve()): sha(p) for p in source_paths}
    Image.MAX_IMAGE_PIXELS = 12800 * 8000 + 1
    with Image.open(args.image) as im:
        assert im.mode == "RGBA" and im.size == (12800, 8000)
        bounds = im.getchannel("A").getbbox()
    audit = {}
    rebuilder = load(VOR / "rebuild_current_main_figures.py", "figure1_frozen_rebuilder")
    preview, panels = native_panels(rebuilder, output, audit)
    panels["b"], timeline_rows = session_timeline(preview)
    pd.DataFrame(timeline_rows).to_csv(output / "session_timeline_labels.csv", index=False)
    data = pd.read_csv(ED1 / "drawn_values/PANEL_A_TRAJECTORY_PLOT_DATA_ROUND21.csv.gz")
    summary = data.groupby("nickname", sort=False).agg(rows=("nickname", "size"), max_maze_minutes=("elapsed_minutes", "max"), drawn_segments=("draw_to_next", "sum"))
    summary.to_csv(output / "trajectory_source_summary.csv")
    ed_builder = load(VOR / "rebuild_current_extended_data_figures.py", "figure1_ed1_frozen_rebuilder")
    results = {}
    with ed_builder.frozen_stage(ED1) as stage:
        ed = ed_builder.load_module(stage / "scripts/render_extended_data_figure1_round4.py", "figure1_ed1_trajectory_renderer")
        for name, cases in VARIANTS.items():
            trajectory, segments = trajectories(preview, ed, data, cases, name)
            root = ET.Element(tag("svg"), {"width": "180mm", "height": "196mm", "viewBox": f"0 0 {WIDTH} {HEIGHT}", "version": "1.1"})
            ET.SubElement(root, tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
            maze_image(root, args.image, bounds)
            geometry = {}
            for letter, key in MAPPING.items():
                geometry[letter] = place(root, panels[key], letter, SLOTS_MM[letter])
            geometry["b"] = place(root, trajectory, "b", SLOTS_MM["b"])
            stem = output / ("figure1_" + name)
            svg = stem.with_suffix(".svg")
            ET.ElementTree(root).write(svg, encoding="utf-8", xml_declaration=True)
            rendered = export(svg, stem)
            results[name] = {"sessions": [s for s, _ in cases], "drawn_segments": segments, "geometry": geometry, **rendered}
            print(name, rendered, flush=True)
    assert all(sha(p) == h for p, h in source_hashes.items())
    manifest = {"status": "DRAFT_FOR_AUTHOR_SELECTION", "source_hashes": source_hashes, "image_original_pixel_size": [12800, 8000], "image_visible_viewport": bounds, "image_pixel_edits": "none; original PNG bytes embedded", "unchanged_source_hashes_verified": True, "timeline_compression": audit, "panel_map": {new: f"old Figure 1{OLD_DISPLAY[key]}" for new, key in MAPPING.items()}, "new_a": "author-supplied physical maze illustration", "new_c": "replacement of old b: true day-coordinate timeline, explicit Naive/Return/Second day labels", "timeline_labels": timeline_rows, "new_b": "ED1a first-reward trajectories, frozen masks and shared original color scale", "layout_reference": "author sketch: physical maze and trajectories above timeline; existing schematic and plots beneath", "variants": results}
    (output / "inputs").mkdir(exist_ok=True)
    if args.image.resolve() != (output / "inputs/maze_3d_original.png").resolve():
        shutil.copy2(args.image, output / "inputs/maze_3d_original.png")
    finish_record(output, manifest)
    print(json.dumps({"output": str(output), "timeline_compression": audit, "source_hashes_verified": len(source_hashes)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
