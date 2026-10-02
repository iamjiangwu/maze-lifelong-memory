"""Draw the Figure 4 method schematic and compose its SVG crop.

build_figure4_standardized_pv supplies the schematic data path and style;
the shared main figure driver uses the crop helper during assembly.
"""

from __future__ import annotations

import json

import math

import os

from dataclasses import dataclass

from pathlib import Path

import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

from matplotlib.colors import Normalize

from ca_analysis.figure_panels import get_panel_maze

REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURE4_PANEL_A_PPT_CROP = {"l": 12400, "t": 15055, "r": 9866, "b": 9331}

FIGURE4_SCHEMATIC_TRACE_POINTS = 5000

SVG_NS = "http://www.w3.org/2000/svg"

XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)

ET.register_namespace("xlink", XLINK_NS)

@dataclass(frozen=True)
class PublicationStyle:
    """Single formatting contract for source-rendered manuscript panels."""

    font_family: str = "Arial"
    base_fontsize: float = 6.0
    axis_label_fontsize: float = 6.0
    tick_fontsize: float = 5.4
    title_fontsize: float = 6.2
    legend_fontsize: float = 4.9
    annotation_fontsize: float = 5.4
    axis_linewidth: float = 0.65
    dpi: int = 300
    facecolor: str = "white"

    def rcparams(self) -> dict[str, object]:
        return {
            "font.family": self.font_family,
            "font.size": self.base_fontsize,
            "axes.labelsize": self.axis_label_fontsize,
            "axes.titlesize": self.title_fontsize,
            "xtick.labelsize": self.tick_fontsize,
            "ytick.labelsize": self.tick_fontsize,
            "legend.fontsize": self.legend_fontsize,
            "axes.linewidth": self.axis_linewidth,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": self.facecolor,
        }

def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"

FIGURE4_SCHEMATIC_DATA_ROOT = REPO_ROOT / "illustration_data"

def _build_figure4_schematic_source() -> dict[str, object]:
    # The method diagram needs only its displayed values, not a second set of
    # full neural sessions. The quantitative panels use their own decoder data.
    illustration_root = FIGURE4_SCHEMATIC_DATA_ROOT
    illustration_path = illustration_root / "spatial_decoder_example.npz"
    if illustration_path.is_file():
        metadata = json.loads((illustration_root / "spatial_decoder_example.json").read_text())
        with np.load(illustration_path, allow_pickle=False) as arrays:
            source = dict(metadata)
            source["template_maps"] = np.asarray(arrays["place_maps"], dtype=float)
            source["trace_segment"] = np.asarray(arrays["activity"], dtype=float)
            source["posterior"] = np.asarray(arrays["posterior"], dtype=float)
        return source
    raise FileNotFoundError(f"Missing included schematic input: {illustration_path}")

def _render_figure4_method_schematic(size: tuple[float, float], style: PublicationStyle) -> tuple[plt.Figure, dict[str, object]]:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    source = _build_figure4_schematic_source()
    title_font = style.axis_label_fontsize * 0.72
    arrow_font = style.annotation_fontsize * 0.70
    tiny_font = style.tick_fontsize * 0.62
    maze = get_panel_maze(6)
    map_cmap = plt.get_cmap("magma").copy()
    map_cmap.set_bad("#f4f4f4")
    posterior_cmap = plt.get_cmap("magma").copy()
    posterior_cmap.set_bad("#f7f7f7")
    trace_y_centers = [0.825, 0.690, 0.500, 0.365, 0.230]

    def _add_real_map(
        bounds: tuple[float, float, float, float],
        image: np.ndarray,
        *,
        is_posterior: bool = False,
        zorder: int = 1,
    ) -> plt.Axes:
        from scipy.ndimage import gaussian_filter
        from matplotlib.patches import Rectangle

        map_ax = ax.inset_axes(bounds, transform=ax.transAxes)
        map_ax.set_zorder(zorder)
        arr = np.asarray(image, dtype=float)
        grid_bins_y, grid_bins_x = int(arr.shape[0]), int(arr.shape[1])
        wall_scale_x = float(grid_bins_x) / 15.0
        wall_scale_y = float(grid_bins_y) / 15.0
        if is_posterior:
            plot_arr = gaussian_filter(np.nan_to_num(arr, nan=0.0), sigma=0.55)
            vmax = float(np.nanmax(plot_arr)) if np.any(np.isfinite(plot_arr)) else 1.0
        else:
            finite = np.isfinite(arr)
            filled = np.where(finite, arr, 0.0)
            plot_arr = gaussian_filter(filled, sigma=1.0)
            positive = plot_arr[np.isfinite(plot_arr) & (plot_arr > 0)]
            vmax = float(np.nanpercentile(positive, 99.7)) if positive.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        map_ax.set_facecolor("#fcfcfc")
        map_ax.imshow(
            np.ma.masked_invalid(plot_arr),
            origin="upper",
            interpolation="nearest",
            cmap=posterior_cmap if is_posterior else map_cmap,
            vmin=0.0,
            vmax=vmax,
        )
        map_ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                grid_bins_x,
                grid_bins_y,
                fill=False,
                edgecolor="0.15",
                linewidth=0.45 if not is_posterior else 0.60,
                zorder=4,
            )
        )
        map_ax.plot(
            maze.wa[:, 0] * wall_scale_x,
            maze.wa[:, 1] * wall_scale_y,
            color="0.10" if not is_posterior else "0.15",
            linewidth=0.26 if not is_posterior else 0.42,
            alpha=0.95,
            zorder=3,
        )
        map_ax.set_xlim(-0.5, grid_bins_x - 0.5)
        map_ax.set_ylim(grid_bins_y - 0.5, -0.5)
        map_ax.set_aspect("equal")
        map_ax.axis("off")
        return map_ax

    template_maps = [np.asarray(m, dtype=float) for m in source["template_maps"]]
    map_geometry_options = {
        "low": (0.045, 0.230, 0.086, 0.052, [0.765, 0.620, 0.410, 0.265, 0.140]),
        "medium": (0.035, 0.230, 0.092, 0.075, [0.760, 0.615, 0.405, 0.260, 0.130]),
        "high": (0.012, 0.235, 0.128, 0.105, [0.760, 0.600, 0.420, 0.260, 0.100]),
        "wide": (0.005, 0.230, 0.106, 0.125, [0.748, 0.602, 0.392, 0.246, 0.112]),
    }
    map_geometry_key = os.environ.get("FIGURE4_SCHEMATIC_TILT", "high").strip().lower()
    if map_geometry_key not in map_geometry_options:
        raise ValueError(
            f"Unknown FIGURE4_SCHEMATIC_TILT={map_geometry_key!r}; "
            f"expected one of {sorted(map_geometry_options)}"
        )
    map_x0, map_w, map_h, map_shear, map_y0s = map_geometry_options[map_geometry_key]
    if map_geometry_key == "high":
        map_y0s = [float(y_center) - map_h / 2.0 for y_center in trace_y_centers]

    def _draw_slanted_template_map(y0: float, neuron_map: np.ndarray, *, zorder: int) -> None:
        from matplotlib.patches import Polygon

        arr = np.asarray(neuron_map, dtype=float)
        finite = np.isfinite(arr)
        if np.any(finite):
            baseline = float(np.nanpercentile(arr[finite], 55.0))
            image = np.where(finite, np.clip(arr - baseline, 0.0, None), 0.0)
            vmax = float(np.nanpercentile(image[image > 0], 99.5)) if np.any(image > 0) else 1.0
        else:
            image = np.zeros_like(arr, dtype=float)
            vmax = 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
        pixels_per_bin = 8
        tile = np.repeat(np.repeat(image, pixels_per_bin, axis=0), pixels_per_bin, axis=1)
        sheet = Polygon(
            [
                (map_x0, y0),
                (map_x0 + map_w, y0),
                (map_x0 + map_w + map_shear, y0 + map_h),
                (map_x0 + map_shear, y0 + map_h),
            ],
            closed=True,
            facecolor="#020204",
            edgecolor="none",
            antialiased=True,
            transform=ax.transAxes,
            zorder=zorder - 0.05,
        )
        ax.add_patch(sheet)
        slant = math.degrees(math.atan2(map_shear, map_h))
        transform = matplotlib.transforms.Affine2D().skew_deg(slant, 0).translate(map_x0, y0) + ax.transAxes
        ax.imshow(
            tile,
            extent=(0.0, map_w, 0.0, map_h),
            origin="upper",
            cmap=map_cmap,
            norm=norm,
            interpolation="nearest",
            resample=False,
            aspect="auto",
            transform=transform,
            zorder=zorder,
            clip_on=False,
        )
    for layer, (y0, neuron_map) in enumerate(zip(map_y0s, template_maps, strict=True)):
        _draw_slanted_template_map(y0, neuron_map, zorder=1 + layer * 0.1)
    stack_center_x = map_x0 + map_w / 2.0 + map_shear / 2.0
    map_gap_mid_y = (map_y0s[1] + map_y0s[2] + map_h) / 2.0
    gap_dot_y = map_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, stack_center_x),
        gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    bottom_title_y = 0.025
    ax.text(stack_center_x, bottom_title_y, "Day i place maps", ha="center", va="bottom", fontsize=title_font)

    trace_left, trace_right = 0.405, 0.635
    trace_x = np.linspace(trace_left, trace_right, int(FIGURE4_SCHEMATIC_TRACE_POINTS))
    trace_segment = np.asarray(source["trace_segment"], dtype=float)
    if trace_segment.shape[0] != trace_x.size:
        trace_x = np.linspace(trace_left, trace_right, trace_segment.shape[0])
    vline_x = float(np.interp(int(source["vline_idx"]), [0, max(trace_segment.shape[0] - 1, 1)], [trace_x[0], trace_x[-1]]))
    frame_dx = (trace_right - trace_left) / max(trace_segment.shape[0] - 1, 1)
    visible_half_width = max(frame_dx / 2.0, 0.0014)
    trace_band_ymin = min(y - 0.050 + 0.010 for y in trace_y_centers) - 0.006
    trace_band_ymax = max(y - 0.050 + 0.010 + 0.073 for y in trace_y_centers) + 0.006
    ax.axvspan(
        max(trace_left, vline_x - visible_half_width),
        min(trace_right, vline_x + visible_half_width),
        ymin=max(0.0, trace_band_ymin),
        ymax=min(1.0, trace_band_ymax),
        color="#e5e5e5",
        alpha=0.85,
        linewidth=0,
        zorder=0,
    )
    for index, y_center in enumerate(trace_y_centers):
        baseline = y_center - 0.050
        trace = trace_segment[:, index]
        trace = 0.010 + 0.073 * np.clip(trace, 0.0, 1.0)
        ax.plot(trace_x, baseline + trace, color="#202020", linewidth=0.70)
    trace_gap_mid_y = (trace_y_centers[1] + trace_y_centers[2]) / 2.0
    trace_gap_dot_y = trace_gap_mid_y + np.array([0.009, 0.0, -0.009])
    ax.scatter(
        np.full(3, (trace_left + trace_right) / 2.0),
        trace_gap_dot_y,
        s=0.12,
        color="#202020",
        edgecolors="none",
        linewidths=0.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    scale_bar_frames = 10.0 / max(float(source["dt"]), 1e-12)
    scale_bar_len = (scale_bar_frames / max(trace_segment.shape[0] - 1, 1)) * (trace_right - trace_left)
    scale_bar_x0 = trace_left
    scale_bar_y = 0.145
    ax.plot(
        [scale_bar_x0, scale_bar_x0 + scale_bar_len],
        [scale_bar_y, scale_bar_y],
        color="#202020",
        linewidth=0.90,
        solid_capstyle="butt",
    )
    ax.text(
        scale_bar_x0 + scale_bar_len / 2.0,
        scale_bar_y - 0.030,
        "10 s",
        ha="center",
        va="top",
        fontsize=tiny_font,
        color="#202020",
    )
    ax.text(
        (trace_left + trace_right) / 2.0,
        bottom_title_y,
        "Day i + 90 activity",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )

    ax.text(0.377, 0.500, "x", ha="center", va="center", fontsize=style.axis_label_fontsize * 1.00)
    posterior = np.asarray(source["posterior"], dtype=float)
    posterior_h = 0.710
    posterior_w = posterior_h * float(size[1]) / float(size[0])
    posterior_x = 0.675
    posterior_y = 0.150
    arrow_start_x = trace_right + 0.010
    arrow_end_x = posterior_x - 0.010
    ax.plot(
        [arrow_start_x, arrow_end_x],
        [0.500, 0.500],
        color="0.25",
        linewidth=0.65,
        marker=">",
        markevery=[-1],
        markersize=1.25,
        markeredgewidth=0,
        markerfacecolor="0.25",
        solid_capstyle="butt",
        zorder=3,
    )

    posterior_ax = _add_real_map((posterior_x, posterior_y, posterior_w, posterior_h), posterior, is_posterior=True)
    decoded_x, decoded_y = [int(v) for v in source["posterior_decoded_xy"]]
    true_x, true_y = [int(v) for v in source["posterior_true_xy"]]
    posterior_ax.scatter(
        [decoded_x],
        [decoded_y],
        s=2.8,
        marker="x",
        linewidths=0.32,
        color="#00A6D6",
        zorder=5,
    )
    posterior_ax.scatter(
        [true_x],
        [true_y],
        s=2.8,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.32,
        zorder=5,
    )
    cbar_x = posterior_x + posterior_w * 0.18
    cbar_y = posterior_y + posterior_h + 0.025
    cbar_w = posterior_w * 0.64
    cbar_h = 0.014
    posterior_cbar_ax = ax.inset_axes((cbar_x, cbar_y, cbar_w, cbar_h), transform=ax.transAxes)
    posterior_cbar_ax.set_zorder(6)
    posterior_cbar_ax.imshow(
        np.linspace(0.0, 1.0, 128)[None, :],
        aspect="auto",
        cmap=posterior_cmap,
        interpolation="nearest",
    )
    posterior_cbar_ax.set_axis_off()
    ax.text(
        cbar_x + cbar_w / 2.0,
        cbar_y + cbar_h + 0.006,
        "P(position)",
        fontsize=tiny_font * 0.92,
        color="#202020",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        zorder=6,
    )
    legend_y = posterior_y - 0.035
    decoded_legend_x = posterior_x + 0.030
    true_legend_x = posterior_x + 0.150
    ax.scatter(
        [decoded_legend_x],
        [legend_y],
        s=4.0,
        marker="x",
        linewidths=0.42,
        color="#00A6D6",
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        decoded_legend_x + 0.012,
        legend_y,
        "Decoded",
        fontsize=tiny_font,
        color="#00A6D6",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.scatter(
        [true_legend_x],
        [legend_y],
        s=4.0,
        marker="o",
        facecolor="none",
        edgecolor="#F04E37",
        linewidths=0.42,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        true_legend_x + 0.012,
        legend_y,
        "Real",
        fontsize=tiny_font,
        color="#F04E37",
        ha="left",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )
    ax.text(
        posterior_x + posterior_w / 2.0,
        bottom_title_y,
        "Decoded position",
        ha="center",
        va="bottom",
        fontsize=title_font,
    )
    meta = {
        "src_session": str(source["src_session"]),
        "dst_session": str(source["dst_session"]),
        "n_matched_neurons": int(source["n_matched_neurons"]),
        "selected_train_ids": list(source["selected_train_ids"]),
        "selected_test_ids": list(source["selected_test_ids"]),
        "selected_match_indices": list(source["selected_match_indices"]),
        "trace_start_index": int(source["trace_start_index"]),
        "trace_stop_index": int(source["trace_stop_index"]),
        "vline_idx": int(source["vline_idx"]),
        "posterior_index": int(source["posterior_index"]),
        "posterior_frame": int(source["posterior_frame"]),
        "posterior_decoded_label": int(source["posterior_decoded_label"]),
        "posterior_true_label": int(source["posterior_true_label"]),
        "posterior_decoded_xy": list(source["posterior_decoded_xy"]),
        "posterior_true_xy": list(source["posterior_true_xy"]),
        "n_bins_x": int(source["n_bins_x"]),
        "n_bins_y": int(source["n_bins_y"]),
        "posterior_n_bins_x": int(source["posterior_n_bins_x"]),
        "posterior_n_bins_y": int(source["posterior_n_bins_y"]),
        "dt": float(source["dt"]),
        "template_source": str(source["template_source"]),
        "activity_source": str(source["activity_source"]),
        "posterior_source": str(source["posterior_source"]),
    }
    return fig, meta

def _svg_defs(root: ET.Element) -> ET.Element:
    for child in root:
        if child.tag == _svg_tag("defs"):
            return child
    return ET.SubElement(root, _svg_tag("defs"))

def _add_svg_panel_cropped_absolute(
    root: ET.Element,
    panel_svg: dict[str, object],
    *,
    panel_id: str,
    slot_pt: tuple[float, float, float, float],
    src_rect: dict[str, int],
) -> dict[str, float]:
    """Place a source-rendered panel in absolute figure coordinates with PPT-style clipping."""

    slot_x, slot_y, slot_w, slot_h = (float(value) for value in slot_pt)
    view_w = float(panel_svg["viewbox_width"])
    view_h = float(panel_svg["viewbox_height"])
    left = float(src_rect.get("l", 0)) / 100000.0
    top = float(src_rect.get("t", 0)) / 100000.0
    right = float(src_rect.get("r", 0)) / 100000.0
    bottom = float(src_rect.get("b", 0)) / 100000.0
    crop_x = view_w * left
    crop_y = view_h * top
    crop_w = view_w * max(1.0 - left - right, 1e-9)
    crop_h = view_h * max(1.0 - top - bottom, 1e-9)
    scale_x = slot_w / crop_w
    scale_y = slot_h / crop_h
    clip_id = f"clip_{panel_id}"
    clip = ET.SubElement(
        _svg_defs(root),
        _svg_tag("clipPath"),
        {"id": clip_id, "clipPathUnits": "userSpaceOnUse"},
    )
    ET.SubElement(
        clip,
        _svg_tag("rect"),
        {
            "x": f"{slot_x:.6f}",
            "y": f"{slot_y:.6f}",
            "width": f"{slot_w:.6f}",
            "height": f"{slot_h:.6f}",
        },
    )
    outer = ET.SubElement(
        root,
        _svg_tag("g"),
        {
            "id": f"panel_{panel_id}",
            "clip-path": f"url(#{clip_id})",
        },
    )
    group = ET.SubElement(
        outer,
        _svg_tag("g"),
        {
            "transform": (
                f"translate({slot_x - crop_x * scale_x:.6f} {slot_y - crop_y * scale_y:.6f}) "
                f"scale({scale_x:.8f} {scale_y:.8f})"
            ),
        },
    )
    for child in panel_svg["children"]:
        group.append(child)
    return {
        "slot_x_pt": float(slot_x),
        "slot_y_pt": float(slot_y),
        "slot_width_pt": float(slot_w),
        "slot_height_pt": float(slot_h),
        "source_viewbox_width_pt": float(view_w),
        "source_viewbox_height_pt": float(view_h),
        "crop_x_pt": float(crop_x),
        "crop_y_pt": float(crop_y),
        "crop_width_pt": float(crop_w),
        "crop_height_pt": float(crop_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "rendered_width": float(slot_w),
        "rendered_height": float(slot_h),
        "scale": float(scale_x),
    }
