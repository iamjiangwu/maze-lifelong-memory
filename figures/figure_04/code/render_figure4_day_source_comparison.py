"""Figure 4 decoder comparisons, supported-position map and SVG placement.

build_figure4_standardized_pv supplies the included tables; the shared main
figure driver calls the SVG placement helpers to compose the final figure.
"""

from __future__ import annotations

import copy

from pathlib import Path

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats

PANEL_C_MIN_SOURCE_FRAMES = 5

FIGSIZE = (7.50, 4.00)

WIDTH_MM = 180.0

HEIGHT_MM = WIDTH_MM * FIGSIZE[1] / FIGSIZE[0]

FIGURE4_ORANGE = "#e29957"

COLOR = {"d1": FIGURE4_ORANGE, "d2": FIGURE4_ORANGE, "reference": "#7f7f7f"}

SLOTS = {
    "a": (0.056, 0.127, 3.822, 1.646),
    "b": (3.878, 0.000, 0.749, 2.089),
    "c": (4.760, 0.205, 2.005, 1.490),
    "d": (6.550, 0.000, 0.749, 2.089),
    "e": (-0.030, 1.724, 4.899, 2.126),
    "f": (4.720, 1.773, 2.650, 2.226),
}

LABEL_POSITIONS = {
    "a": (0.02, 0.12),
    "b": (3.80, 0.12),
    "c": (4.60, 0.12),
    "d": (6.48, 0.12),
    "e": (0.02, 1.82),
    "f": (4.86, 1.87),
}

SVG_NS = "http://www.w3.org/2000/svg"

XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)

ET.register_namespace("xlink", XLINK_NS)

def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6.0,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.2,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "legend.fontsize": 4.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

def axis_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.2)

def star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def svg_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    values = [float(value) for value in root.attrib["viewBox"].replace(",", " ").split()]
    if len(values) != 4:
        raise RuntimeError("SVG viewBox must contain four numbers")
    return tuple(values)  # type: ignore[return-value]

def prefix_ids(root: ET.Element, prefix: str) -> None:
    replacements: dict[str, str] = {}
    for element in root.iter():
        old = element.attrib.get("id")
        if old:
            replacements[old] = f"{prefix}_{old}"
            element.attrib["id"] = replacements[old]
    if not replacements:
        return
    patterns = [(f"url(#{old})", f"url(#{new})") for old, new in replacements.items()]
    href_patterns = [(f"#{old}", f"#{new}") for old, new in replacements.items()]
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            updated = value
            for old, new in patterns:
                updated = updated.replace(old, new)
            if key.endswith("href"):
                for old, new in href_patterns:
                    if updated == old:
                        updated = new
            element.attrib[key] = updated
        if element.text:
            updated = element.text
            for old, new in patterns:
                updated = updated.replace(old, new)
            element.text = updated

def add_svg_panel(
    destination: ET.Element,
    payload: bytes,
    *,
    panel_id: str,
    slot: tuple[float, float, float, float],
) -> None:
    source = ET.fromstring(payload)
    prefix_ids(source, panel_id)
    x0, y0, source_w, source_h = svg_viewbox(source)
    slot_x, slot_y, slot_w, slot_h = (value * 72.0 for value in slot)
    scale = min(slot_w / source_w, slot_h / source_h)
    tx = slot_x + (slot_w - source_w * scale) / 2.0 - x0 * scale
    ty = slot_y - y0 * scale
    group = ET.SubElement(
        destination,
        f"{{{SVG_NS}}}g",
        {"id": panel_id, "transform": f"translate({tx:.6f} {ty:.6f}) scale({scale:.9f})"},
    )
    for child in list(source):
        group.append(copy.deepcopy(child))

def paired_panel(
    rows: pd.DataFrame,
    stat_row: pd.Series,
    *,
    variant: str,
    panel: str,
    size: tuple[float, float],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    if panel == "b":
        left = rows["observed_error"].to_numpy(dtype=float)
        right = rows["exact_label_permutation_error"].to_numpy(dtype=float)
        labels = ("Data", "Shuffle")
        # Share panel d's y-range so Data/Shuffle and Reward/Ctrls are comparable.
        ylim = (0.0, 15.0)
        yticks = [0.0, 5.0, 10.0, 15.0]
    else:
        left = rows["reward_eu"].to_numpy(dtype=float)
        right = rows["ctrl_eu_mean"].to_numpy(dtype=float)
        labels = ("Reward", "Ctrls")
        ylim = (0.0, 15.0)
        yticks = [0.0, 5.0, 10.0, 15.0]
    for first, second in zip(left, right, strict=True):
        ax.plot([0, 1], [first, second], color="#8a8a8a", lw=0.55, alpha=0.30, zorder=1)
    jitter = np.linspace(-0.045, 0.045, len(left))
    ax.scatter(jitter, left, s=9, color=COLOR[variant], alpha=0.68, edgecolors="none", zorder=3)
    ax.scatter(
        1 + jitter,
        right,
        s=9,
        color=COLOR["reference"],
        alpha=0.68,
        edgecolors="none",
        zorder=3,
    )
    for x, values, color in ((0, left, COLOR[variant]), (1, right, COLOR["reference"])):
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        ax.bar(
            x,
            mean,
            yerr=sem,
            width=0.68,
            color=color,
            edgecolor="none",
            alpha=0.78,
            capsize=2.0,
            zorder=2,
        )
    p = float(stat_row["cluster_p"])
    bracket_y = ylim[1] * 0.92
    ax.plot(
        [0, 0, 1, 1],
        [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
        color="black",
        lw=0.65,
    )
    ax.text(
        0.5,
        bracket_y * 1.015,
        star(p),
        ha="center",
        va="bottom",
        fontsize=5.4,
    )
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Decoding error (bins)", labelpad=0.0)
    ax.set_xlim(-0.55, 1.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    axis_style(ax)
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.25, top=0.86)
    return fig

def map_panel(
    rows: pd.DataFrame,
    *,
    variant: str,
    size: tuple[float, float],
    source_csv: Path,
) -> plt.Figure:
    from ca_analysis import figure_panels as figure_panels_module
    from ca_analysis.figure_panels import PANEL_LAYOUT, render_spatial_decoding_error_map

    source_rows = rows.rename(
        columns={
            "true_position_bin": "bin_index",
            "mean_decoding_error": "decoding_error",
            "is_reward_proxy": "is_reward",
        }
    ).copy()
    source_rows["marker_label"] = np.where(
        source_rows["is_reward"].astype(bool),
        "Reward",
        np.where(source_rows["is_control"].astype(bool), "Control", ""),
    )
    source_rows["display_support_pass"] = (
        pd.to_numeric(source_rows["n_frames"], errors="coerce")
        >= PANEL_C_MIN_SOURCE_FRAMES
    )
    source_rows.loc[~source_rows["display_support_pass"], "decoding_error"] = np.nan
    source_csv.parent.mkdir(parents=True, exist_ok=True)
    source_rows.to_csv(source_csv, index=False)

    fig, _ax, _data, _paths = render_spatial_decoding_error_map(
        map_csv=source_csv,
        panel_size=size,
        layout={
            **PANEL_LAYOUT,
            "panel_size": size,
            "reference_panel_size": size,
            "colorbar_fraction": 0.060,
            "colorbar_pad": 0.020,
            "colorbar_location": "left",
        },
        colors={
            "cmap": "magma",
            "bad": "white",
            "wall": "black",
            "reward_square": COLOR[variant],
            "control_square": "#7f7f7f",
        },
        style={
            "vmin": 1.0,
            "vmax": 14.0,
            "image_alpha": 0.95,
            "maze_wall_linewidth": 0.85,
            "square_linewidth": 1.45,
            "square_size": 1.10,
        },
        labels={"colorbar": "Decoding error (bins)"},
        save=False,
    )
    map_ax = fig.axes[0]
    colorbar_ax = fig.axes[-1]
    heatmap_height = 0.885
    heatmap_width = heatmap_height * size[1] / size[0]
    map_ax.set_position([0.125, 0.072, heatmap_width, heatmap_height])
    colorbar_ax.set_position([0.060, 0.072, 0.032, heatmap_height])
    colorbar_ax.yaxis.set_ticks_position("left")
    colorbar_ax.yaxis.set_label_position("left")
    colorbar_ax.yaxis.labelpad = 1.0
    colorbar_ax.tick_params(axis="y", pad=1.0, length=2.0)
    reward_rows = source_rows.loc[source_rows["is_reward"].astype(bool)]
    if not reward_rows.empty:
        reward_row = reward_rows.iloc[0]
        reward_xy_axes = map_ax.transAxes.inverted().transform(
            map_ax.transData.transform(
                (float(reward_row["bx"]), float(reward_row["by"]))
            )
        )
        figure_panels_module._add_timeline_water_drop_icon(
            map_ax,
            (1.0, float(reward_xy_axes[1])),
            {
                "kind": "water",
                "size": 12.0,
                "linewidth": 0.65,
                "x_offset_points": 6.0,
                "zorder": 20,
            },
        )
    return fig
