"""Draw Figure 5 panels from the supplied level-code measurements and fits."""
from __future__ import annotations

import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ca_analysis.figure_panels import (
    CORRELATION_MEASURE_COLORS,
    PANEL_LAYOUT,
    get_level_assignment_schematic_colors,
    get_level_assignment_schematic_geometry,
    get_level_assignment_schematic_layout,
    get_level_assignment_schematic_style,
    get_level_assignment_schematic_text,
    render_correlation_shuffle_distribution_panel,
    render_level_assignment_schematic,
    render_level_code_heatmap_panel,
)

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# The builder supplies these paths after authenticating the Source Data.
FIGURE5_LEVEL_DISTRIBUTION_DIR: Path
FIGURE5_LEVEL_HEATMAP_PLOT_READY: Path
FIGURE5_LEVEL_HEATMAP_PLOT_READY_SHA256: str
SORTED_LEVEL_EXPLORATION_DIR: Path
FONT_FAMILY = "Arial"
EXPORT_WIDTH_MM = 180.0
CANVAS_SIZE = (8.65, 5.85)

BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
TICK_LABEL_SIZE = 7.0
TITLE_SIZE = 8.0
PANEL_ID_SIZE = 13.0
LEVEL_COLOR = CORRELATION_MEASURE_COLORS["level_map_correlation"]
SHUFFLE_GRAY = "#7f7f7f"
HEATMAP_CMAP = "magma"
DIRECT_LABEL_FONT_SIZE = 6.8


def _star_from_p(p_value: object) -> str:
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return "n.s."
    if not np.isfinite(p) or p > 0.05:
        return "n.s."
    if p > 0.01:
        return "*"
    if p > 0.001:
        return "**"
    return "***"


def _axes_dx_for_chars(panel_size: tuple[float, float], *, left: float, right: float, chars: float) -> float:
    axis_width_pt = panel_size[0] * 72.0 * (right - left)
    return chars * (0.55 * DIRECT_LABEL_FONT_SIZE) / axis_width_pt


def _axes_dy_for_text_heights(panel_size: tuple[float, float], *, bottom: float, top: float, units: float) -> float:
    axis_height_pt = panel_size[1] * 72.0 * (top - bottom)
    return units * DIRECT_LABEL_FONT_SIZE / axis_height_pt


def _add_axes_text_label(ax, x: float, y: float, text: str, *, color: str, va: str = "top") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va=va,
        fontsize=DIRECT_LABEL_FONT_SIZE,
        fontfamily=FONT_FAMILY,
        fontweight="bold",
        color=color,
    )


def _set_common_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": TICK_LABEL_SIZE,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
        }
    )


def _standardize_axes_text(fig, *, font_scale: float = 1.0) -> None:
    for ax in fig.axes:
        ax.title.set_fontfamily(FONT_FAMILY)
        ax.title.set_fontsize(TITLE_SIZE * font_scale)
        ax.xaxis.label.set_fontfamily(FONT_FAMILY)
        ax.yaxis.label.set_fontfamily(FONT_FAMILY)
        ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE * font_scale)
        ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE * font_scale)
        ax.tick_params(labelsize=TICK_LABEL_SIZE * font_scale)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontfamily(FONT_FAMILY)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(FONT_FAMILY)
                text.set_fontsize(TICK_LABEL_SIZE * font_scale)


def _render_panel_a(size: tuple[float, float]):
    row_scale = float(size[1]) / 4.4
    _set_common_rc()
    layout = get_level_assignment_schematic_layout(
        {
            "panel_size": size,
            "width_ratios": (0.45, 1.00),
            "grid_wspace": 0.02,
        }
    )
    geometry = get_level_assignment_schematic_geometry(
        {
            "tree_depth": 7,
            "level_count": 7,
            "tree_width": 12.55,
            "tree_height": 3.6,
            "tree_vertical_pad": 0.10,
            "root_line_length": 1.15,
            "tree_left_pad": 0.80,
            "tree_right_pad": 0.10,
            "maze_margin": 0.15,
        }
    )
    colors = get_level_assignment_schematic_colors(
        {
            "cmap_name": HEATMAP_CMAP,
            "cell_edge": "none",
            "maze_outline": "black",
        }
    )
    style = get_level_assignment_schematic_style(
        {
            "tree_linewidth": 2.0 * row_scale,
            "root_linewidth": 2.0 * row_scale,
            "cell_linewidth": 0.0,
            "maze_outline_linewidth": 1.55 * row_scale,
            "show_tree_nodes": False,
            "show_leaf_markers": True,
            "leaf_marker_size": 17 * row_scale,
            "label_fontsize": 12 * row_scale,
        }
    )
    text = get_level_assignment_schematic_text(
        {
            "level_label": {
                "position": "below_zero",
                "dx": 0.5,
                "dy": 0.0,
                "gap_below_zero": 5,
                "ha": "center",
                "va": "top",
            },
            "root_label_t": 0.52,
            "root_label_offset": 0.40,
            "edge_label_t": 0.56,
            "edge_label_t_by_level": {4: 0.50, 5: 0.44, 6: 0.36},
            "edge_label_offset": 0.40,
            "leaf_label_dx": 0.0,
            "leaf_label_dy": -0.50,
            "show_leaf_label": False,
        }
    )
    fig, _axes, _data, _paths = render_level_assignment_schematic(
        maze_id=6,
        level_definition="tree_depth7",
        panel_size=size,
        layout=layout,
        geometry=geometry,
        colors=colors,
        style=style,
        text=text,
        save=False,
    )
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.08, top=0.94, wspace=0.02)
    return fig


def _render_panel_b(size: tuple[float, float]):
    _set_common_rc()
    fig, _ax, _data, _paths = render_correlation_shuffle_distribution_panel(
        npz_path=FIGURE5_LEVEL_DISTRIBUTION_DIR / "plot_data_cell228_level.npz",
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size},
        colors={"data": LEVEL_COLOR, "shuffle": SHUFFLE_GRAY},
        style={
            "data_alpha": 0.72,
            "shuffle_alpha": 0.40,
            "legend": False,
            "show_counts": False,
            "direct_labels": False,
            "direct_label_alpha": 0.98,
            "direct_label_fontsize": 7.2,
            "direct_label_fontweight": "bold",
            "direct_label_transform": "axes",
            "xlim": (-1.0, 1.0),
        },
        labels={"xlabel": "Level map correlation", "ylabel": "Density", "data": "Data", "shuffle": "Shuffle"},
        save=False,
    )
    panel_c_shuffle_y = 0.90 - _axes_dy_for_text_heights(size, bottom=0.24, top=0.94, units=3)
    panel_c_data_x = 0.52 + _axes_dx_for_chars(size, left=0.16, right=0.965, chars=4)
    _add_axes_text_label(_ax, 0.30, panel_c_shuffle_y, "Shuffle", color=SHUFFLE_GRAY)
    _add_axes_text_label(_ax, panel_c_data_x, panel_c_shuffle_y, "Data", color=LEVEL_COLOR)
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.24, top=0.94)
    return fig


def _render_panel_e(
    size: tuple[float, float],
    *,
    refined_order_path: Path | None = None,
    refined_order_sha256: str | None = None,
    return_data: bool = False,
):
    _set_common_rc()
    pair_groups = {
        "once": [
            ("418RRday1", "418-3day1"),
            ("422RRday1", "422-3day1"),
            ("434RRday1", "434-3day1"),
            ("451Rday1", "451-3day1"),
            ("468Lday1", "468-3day1"),
            ("488Rday1", "488-3day1"),
            ("489Lday1", "489-3day1"),
            ("579RLday1", "579-3day1"),
            ("837day1", "837-3day1"),
            ("838day1", "838-3day1"),
            ("839day1", "839-3day1"),
            ("842day1", "842-3day1"),
            ("862day1", "862-3day1"),
        ],
        "twice": [
            ("1616bday1", "1616b-3day1"),
            ("418-3day1", "418-6day1"),
            ("422-3day1", "422-6day1"),
            ("451-3day1", "451-6day1"),
            ("468-3day1", "468-6day1"),
            ("488-3day1", "488-6day1"),
            ("489-3day1", "489-6day1"),
            ("837-3day1", "837-6day1"),
            ("838-3day1", "838-6day1"),
            ("839-3day1", "839-6day1"),
            ("842-3day1", "842-6day1"),
            ("862-3day1", "862-6day1"),
        ],
        "three times": [
            ("418-6day1", "418-9day1"),
            ("451-6day1", "451-9day1"),
            ("468-6day1", "468-9day1"),
            ("837-6day1", "837-9day1"),
            ("838-6day1", "838-9day1"),
            ("839-6day1", "839-9day1"),
            ("842-6day1", "842-9day1"),
        ],
        "four times": [
            ("468-9day1", "468-12day1"),
            ("837-9day1", "837-12day1"),
            ("838-9day1", "838-12day1"),
            ("839-9day1", "839-12day1"),
        ],
    }
    pair_order = [pair for group_pairs in pair_groups.values() for pair in group_pairs]
    fig, axes, data, _paths = render_level_code_heatmap_panel(
        plot_ready_csv=FIGURE5_LEVEL_HEATMAP_PLOT_READY,
        variant="d1v1",
        pair_order=pair_order,
        panel_size=size,
        layout={**PANEL_LAYOUT, "panel_size": size, "grid_wspace": 0.06},
        colors={"heatmap": HEATMAP_CMAP},
        style={
            "normalize_mode": "l2",
            "plot_ready_sha256": FIGURE5_LEVEL_HEATMAP_PLOT_READY_SHA256,
            "refined_order_path": refined_order_path,
            "refined_order_sha256": refined_order_sha256,
            "tick_labelsize": 7.0,
            "label_fontsize": 9.0,
            "title_fontsize": 8.0,
            "hide_right_yticklabels": True,
            "shared_color_scale": True,
        },
        labels={
            "xlabel": "Levels",
            "ylabel": "Neuron",
            "left_title": "Day i",
            "right_title": "Day i+90",
        },
        xticks=list(range(7)),
        save=False,
    )
    _standardize_axes_text(fig)
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.08, top=0.94, wspace=0.06)
    colorbar = fig.colorbar(
        axes[1].images[0],
        ax=list(axes),
        location="right",
        fraction=0.028,
        pad=0.018,
    )
    colorbar.set_label("Normalized rate", fontsize=7.0, fontfamily=FONT_FAMILY)
    colorbar.ax.tick_params(labelsize=6.5, length=2.0)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
    if return_data:
        return fig, data
    return fig


def _render_panel_f(size: tuple[float, float]):
    _set_common_rc()
    base_size = (3.16, 2.62)
    style_scale = float(size[1]) / base_size[1]
    rows = pd.read_csv(SORTED_LEVEL_EXPLORATION_DIR / "level_map_vs_exploration_efficiency_primary_clean_rows.csv")
    rows = rows.loc[rows["included_in_primary"].astype(bool)].copy()
    stats = json.loads(
        (SORTED_LEVEL_EXPLORATION_DIR / "level_map_vs_exploration_efficiency_primary_clean_stats.json").read_text(
            encoding="utf-8"
        )
    )
    x = pd.to_numeric(rows["exploration_efficiency"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(rows["level_map_correlation"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    fig, ax = plt.subplots(figsize=size)
    # Match a 3.66 pt dot diameter after full page scaling.
    ax.scatter(x, y, s=19.952666666666666 * style_scale * style_scale, color=LEVEL_COLOR, alpha=0.75, edgecolors="none", zorder=2)
    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
    slope = float(stats["ols_slope"])
    intercept = float(stats["ols_intercept"])
    yfit = intercept + slope * xfit
    ax.plot(xfit, yfit, color=LEVEL_COLOR, linestyle="-", linewidth=1.1 * style_scale, zorder=3)
    if x.size > 2:
        resid = y - (intercept + slope * x)
        s_err = np.sqrt(np.sum(resid**2) / max(x.size - 2, 1))
        x_mean = float(np.mean(x))
        sxx = float(np.sum((x - x_mean) ** 2))
        if sxx > 0:
            se = s_err * np.sqrt(1.0 / x.size + (xfit - x_mean) ** 2 / sxx)
            ci = 1.96 * se
            ax.fill_between(xfit, yfit - ci, yfit + ci, color=LEVEL_COLOR, alpha=0.18, linewidth=0, zorder=1)
    ax.text(
        0.08,
        0.92,
        _star_from_p(stats.get("cluster_ols_p", stats.get("pearson_p", np.nan))),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0 * style_scale,
        fontweight="bold",
        color="black",
    )
    ax.text(
        0.94,
        0.08,
        f"r={float(stats['pearson_r']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2 * style_scale,
        color="black",
    )
    ax.set_xlabel("Exploration efficiency")
    ax.set_ylabel("Level map correlation")
    ax.set_position([0.18, 0.22, 0.77, 0.72])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75 * style_scale)
    ax.tick_params(width=0.75 * style_scale, length=3.5 * style_scale, pad=3.5 * style_scale)
    _standardize_axes_text(fig, font_scale=style_scale)
    return fig


PANEL_RENDERERS = {
    "a": _render_panel_a,
    "b": _render_panel_b,
    "e": _render_panel_e,
    "f": _render_panel_f,
}


def _svg_tag(tag: str) -> str:
    return f"{{http://www.w3.org/2000/svg}}{tag}"


def _parse_svg_viewbox(root: ET.Element) -> tuple[float, float]:
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        _x0, _y0, width, height = map(float, viewbox.split())
        return width, height
    return float(root.attrib["width"].rstrip("pt")), float(root.attrib["height"].rstrip("pt"))


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    id_map = {}
    for element in root.iter():
        old_id = element.attrib.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            element.attrib["id"] = new_id
    if not id_map:
        return
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for old_id, new_id in id_map.items():
                value = value.replace(f"#{old_id}", f"#{new_id}")
            element.attrib[key] = value


def _render_panel_svg(panel_id: str, size: tuple[float, float], *, dpi: int, bbox: str = "standard") -> dict:
    fig = PANEL_RENDERERS[panel_id](size)
    fig.set_dpi(dpi)
    buffer = io.StringIO()
    bbox_inches = "tight" if bbox == "tight" else None
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches=bbox_inches,
            pad_inches=0,
            transparent=True,
            metadata={"Date": None},
        )
    plt.close(fig)
    root = ET.fromstring(buffer.getvalue())
    _prefix_svg_ids(root, f"{panel_id}_")
    width, height = _parse_svg_viewbox(root)
    children = [child for child in list(root) if child.tag != _svg_tag("metadata")]
    return {"children": children, "viewbox_width": width, "viewbox_height": height}


def render_assembly_svg(*, dpi: int = 300) -> ET.ElementTree:
    panel_spec = PANEL_LAYOUT_SPEC
    canvas_size = CANVAS_SIZE
    figure_width = canvas_size[0] * 72.0
    figure_height = canvas_size[1] * 72.0
    root = ET.Element(
        _svg_tag("svg"),
        {
            "width": f"{EXPORT_WIDTH_MM:.3f}mm",
            "height": f"{(EXPORT_WIDTH_MM * canvas_size[1] / canvas_size[0]):.3f}mm",
            "viewBox": f"0 0 {figure_width:.6f} {figure_height:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, _svg_tag("rect"), {"width": "100%", "height": "100%", "fill": "white"})
    for panel_id, panel in panel_spec.items():
        size = (float(panel["render_w"]), float(panel["render_h"]))
        panel_svg = _render_panel_svg(panel_id, size, dpi=dpi, bbox=str(panel.get("bbox", "standard")))
        slot = {
            "x": float(panel["x"]) * 72.0,
            "y": float(panel["y"]) * 72.0,
            "width": float(panel["w"]) * 72.0,
            "height": float(panel["h"]) * 72.0,
        }
        scale = min(slot["width"] / panel_svg["viewbox_width"], slot["height"] / panel_svg["viewbox_height"])
        group = ET.SubElement(
            root,
            _svg_tag("g"),
            {
                "id": f"panel_{panel_id}",
                "transform": f"translate({slot['x']:.6f} {slot['y']:.6f}) scale({scale:.8f})",
            },
        )
        for child in panel_svg["children"]:
            group.append(child)
        ET.SubElement(
            root,
            _svg_tag("text"),
            {
                "x": f"{slot['x']:.6f}",
                "y": f"{max(8.0, slot['y'] - 3.0):.6f}",
                "font-family": FONT_FAMILY,
                "font-size": f"{PANEL_ID_SIZE:.1f}",
                "font-weight": "700",
                "fill": "black",
            },
        ).text = str(panel.get("label", panel_id))
    return ET.ElementTree(root)
