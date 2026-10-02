"""Render parallel D1-source and D2-source versions of Main Figure 4.

The renderer consumes only saved analysis tables.  It does not load neural
recordings.  Panel a is conceptual, panels b-d are source-native, panel e uses
the maintained place-field data with a compact density-origin tick label, and
panel f is the matched K=115 source-day comparison.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


DEFAULT_REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = DEFAULT_REPO
DEFAULT_ANALYSIS = DEFAULT_BUNDLE / "analysis"
DEFAULT_OUTPUT = DEFAULT_BUNDLE / "rendered_current_format"
PANEL_A_SVG = Path(
    "docs/main_figs_v3_source_render_2026-06-24/"
    "cross_figure_refinement_round2_2026-08-11/figure4_individual_panels/"
    "figure4_panel_a_decoding_method_schematic.svg"
)
CURRENT_FIGURE4_PNG = Path(
    "docs/figure_version_records/main_figure4_2026-08-11_round2_place_maps_label/"
    "figure4_version_record_600dpi.png"
)
CURRENT_FIGURE4_SVG = Path(
    "docs/figure_version_records/main_figure4_2026-08-11_round2_place_maps_label/"
    "figure4_version_record.svg"
)
PANEL_E_SVG = Path(
    "docs/main_figs_v3_source_render_2026-06-24/"
    "cross_figure_refinement_round2_2026-08-11/figure4_individual_panels/"
    "figure4_panel_e_linearized_place_field_stability.svg"
)


def resolve_record_asset(repo: Path, relative: Path) -> Path:
    """Resolve a named version-record asset from the active root or archive."""

    active = repo / relative
    if active.is_file():
        return active
    prefix = Path("docs/figure_version_records")
    archived = (
        repo
        / "docs/figure_version_records_archive_2026-08-09"
        / relative.relative_to(prefix)
    )
    if archived.is_file():
        return archived
    raise FileNotFoundError(active)
PANEL_E_HEATMAP_VMAX = 0.6
PANEL_C_MIN_SOURCE_FRAMES = 5
PANEL_A_CROP = {"l": 12400, "t": 15055, "r": 9866, "b": 9331}
FIGSIZE = (7.50, 4.00)
WIDTH_MM = 180.0
HEIGHT_MM = WIDTH_MM * FIGSIZE[1] / FIGSIZE[0]
DPI = 600
BROWSER_EXPORT_TIMEOUT_SECONDS = 420
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
CURRENT_MAIN_FIGURE4_SLOTS = {
    "a": (0.056, 0.127, 3.822, 1.646),
    "b": (3.878, 0.000, 0.749, 2.089),
    "c": (4.760, 0.205, 2.005, 1.490),
    "d": (6.550, 0.000, 0.749, 2.089),
    "e": (-0.030, 1.724, 4.899, 2.126),
    "f": (4.720, 1.773, 2.650, 2.226),
}
RASTER_CROP_PADDING: dict[str, tuple[float, float, float, float]] = {}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("d1", "d2"),
        default=("d1", "d2"),
        help="Source-day pages to render; use '--variants d1' for the selected candidate.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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




def figure_svg(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.patch.set_alpha(0.0)
    fig.savefig(buffer, format="svg", transparent=True, facecolor="none", bbox_inches=None)
    plt.close(fig)
    return buffer.getvalue()


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


def _replace_group_by_id(
    root: ET.Element, group_id: str, replacement: ET.Element
) -> bool:
    for index, child in enumerate(list(root)):
        if child.get("id") == group_id:
            root.remove(child)
            root.insert(index, copy.deepcopy(replacement))
            return True
        if _replace_group_by_id(child, group_id, replacement):
            return True
    return False


def add_preserved_panel_a(
    destination: ET.Element,
    baseline_payload: bytes,
    revised_panel_payload: bytes,
) -> None:
    """Copy accepted panel a and replace its two refined Day-i labels."""

    baseline = ET.fromstring(baseline_payload)
    clip = baseline.find(
        f"./{{{SVG_NS}}}defs/{{{SVG_NS}}}clipPath[@id='clip_figure4_a']"
    )
    group = baseline.find(f"./{{{SVG_NS}}}g[@id='panel_figure4_a']")
    if clip is None or group is None:
        raise RuntimeError("Current Figure 4 SVG is missing the preserved panel-a group or clip")
    revised = ET.fromstring(revised_panel_payload)
    group = copy.deepcopy(group)
    for group_id in ("figure4_a_text_1", "figure4_a_text_3"):
        revised_label = revised.find(f".//{{{SVG_NS}}}g[@id='{group_id}']")
        if revised_label is None:
            raise RuntimeError(f"Refined panel-a SVG is missing {group_id}")
        if not _replace_group_by_id(group, group_id, revised_label):
            raise RuntimeError(f"Accepted panel-a group is missing {group_id}")
    defs = next(
        (child for child in destination if child.tag == f"{{{SVG_NS}}}defs"),
        None,
    )
    if defs is None:
        defs = ET.Element(f"{{{SVG_NS}}}defs")
        destination.insert(0, defs)
    defs.append(copy.deepcopy(clip))
    destination.append(group)


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


def cluster_bootstrap_band(rows: pd.DataFrame, xfit: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x = rows["first_long_bout_time_s"].to_numpy(dtype=float)
    y = rows["reward_error"].to_numpy(dtype=float)
    animals = rows["animal_id"].astype(str).to_numpy()
    mice = np.asarray(sorted(set(animals)))
    rng = np.random.default_rng(seed)
    lines: list[np.ndarray] = []
    for _ in range(5_000):
        sampled = rng.choice(mice, size=len(mice), replace=True)
        indices = np.concatenate([np.flatnonzero(animals == mouse) for mouse in sampled])
        if np.unique(x[indices]).size < 2:
            continue
        fit = stats.linregress(x[indices], y[indices])
        lines.append(fit.intercept + fit.slope * xfit)
    array = np.asarray(lines)
    return np.quantile(array, 0.025, axis=0), np.quantile(array, 0.975, axis=0)


def behavior_panel(
    matched: pd.DataFrame,
    stat_row: pd.Series,
    *,
    variant: str,
    shared_limits: tuple[tuple[float, float], tuple[float, float]],
    size: tuple[float, float],
) -> plt.Figure:
    y_column = f"{variant}_reward_proxy_error"
    rows = matched[["animal_id", "first_long_bout_time_s", y_column]].rename(columns={y_column: "reward_error"})
    x = rows["first_long_bout_time_s"].to_numpy(dtype=float)
    y = rows["reward_error"].to_numpy(dtype=float)
    fit = stats.linregress(x, y)
    xfit = np.linspace(shared_limits[0][0], shared_limits[0][1], 200)
    lo, hi = cluster_bootstrap_band(rows, xfit, seed=SEED_FOR_VARIANT[variant])
    fig, ax = plt.subplots(figsize=size)
    ax.fill_between(xfit, lo, hi, color=COLOR[variant], alpha=0.16, lw=0)
    ax.plot(xfit, fit.intercept + fit.slope * xfit, color=COLOR[variant], lw=1.1)
    ax.scatter(x, y, s=15, color=COLOR[variant], alpha=0.75, edgecolors="none", zorder=3)
    r = float(stat_row["pearson_r"])
    p = float(stat_row["cluster_p"])
    ax.text(
        0.04,
        0.95,
        star(p),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.75,
        color=COLOR[variant],
        fontweight="bold",
    )
    ax.text(
        0.96,
        0.08,
        f"r={r:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
    )
    ax.set_xlabel("First long bout time (s)")
    ax.set_ylabel("Reward decoding error (bins)", labelpad=1.0)
    ax.set_xlim(*shared_limits[0])
    ax.set_ylim(*shared_limits[1])
    axis_style(ax)
    fig.subplots_adjust(left=0.21, right=0.985, bottom=0.20, top=0.975)
    return fig


SEED_FOR_VARIANT = {"d1": 202608041, "d2": 202608042}


def _length_mm(value: str) -> float:
    text = str(value).strip().lower()
    match = re.fullmatch(r"([0-9.eE+-]+)(mm|pt|in|px)?", text)
    if not match:
        raise ValueError(f"Cannot parse SVG length {value!r}")
    number = float(match.group(1))
    unit = match.group(2) or "px"
    return {
        "mm": number,
        "pt": number * 25.4 / 72.0,
        "in": number * 25.4,
        "px": number * 25.4 / 96.0,
    }[unit]


def _wait_stable(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    previous = -1
    stable = 0
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else -1
        if size > 0 and size == previous:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
        previous = size
        time.sleep(0.1)
    raise RuntimeError(f"Browser output did not settle: {path}")


def export_svg(svg_path: Path) -> dict[str, Any]:
    from ca_analysis.browser_export import find_browser

    browser = find_browser(preferred=("chrome", "edge", "chromium"))
    root = ET.parse(svg_path).getroot()
    width_mm = _length_mm(root.attrib["width"])
    height_mm = _length_mm(root.attrib["height"])
    svg_text = svg_path.read_text(encoding="utf-8")
    html_path = svg_path.with_suffix(".html")
    html_path.write_text(
        f"""<!doctype html><html><head><meta charset=\"utf-8\"><style>
@page {{ size: {width_mm:.6f}mm {height_mm:.6f}mm; margin: 0; }}
html,body {{ width:{width_mm:.6f}mm; height:{height_mm:.6f}mm; margin:0; padding:0; background:white; overflow:hidden; }}
svg {{ display:block; width:{width_mm:.6f}mm !important; height:{height_mm:.6f}mm !important; }}
* {{ print-color-adjust:exact; -webkit-print-color-adjust:exact; }}
</style></head><body>{svg_text}</body></html>""",
        encoding="utf-8",
    )
    pdf_path = svg_path.with_suffix(".pdf")
    png_path = svg_path.with_suffix(".png")
    pdf_path.unlink(missing_ok=True)
    png_path.unlink(missing_ok=True)
    css_width = max(1, int(round(width_mm / 25.4 * 96.0)))
    css_height = max(1, int(round(height_mm / 25.4 * 96.0)))
    scale = DPI / 96.0
    with tempfile.TemporaryDirectory(prefix="fig4_chrome_", dir=str(svg_path.parent)) as profile:
        common = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--user-data-dir={Path(profile).resolve()}",
        ]
        subprocess.run(
            [
                *common,
                f"--window-size={css_width},{css_height}",
                f"--force-device-scale-factor={scale:.6f}",
                f"--screenshot={png_path.resolve()}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
        )
        _wait_stable(png_path)
        subprocess.run(
            [
                *common,
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path.resolve()}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=BROWSER_EXPORT_TIMEOUT_SECONDS,
        )
        _wait_stable(pdf_path)
    html_path.unlink(missing_ok=True)
    tiff_path = svg_path.with_suffix(".tiff")
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        rgb.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
        dimensions = rgb.size
    return {
        "svg": str(svg_path),
        "pdf": str(pdf_path),
        "png": str(png_path),
        "tiff": str(tiff_path),
        "pixel_dimensions": list(dimensions),
        "sha256": {
            "svg": sha256(svg_path),
            "pdf": sha256(pdf_path),
            "png": sha256(png_path),
            "tiff": sha256(tiff_path),
        },
    }


def write_panel_svg(payload: bytes, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = ET.fromstring(payload)
    x0, y0, width_pt, height_pt = svg_viewbox(source)
    width_mm = _length_mm(source.attrib["width"])
    height_mm = _length_mm(source.attrib["height"])
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{width_mm:.6f}mm",
            "height": f"{height_mm:.6f}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"width": "100%", "height": "100%", "fill": "white"})
    group = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {"transform": f"translate({-x0:.6f} {-y0:.6f})"},
    )
    for child in list(source):
        group.append(copy.deepcopy(child))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return export_svg(path)


def compose_figure(
    panel_payloads: dict[str, bytes],
    output_svg: Path,
    *,
    panel_a_baseline_payload: bytes,
) -> dict[str, Any]:
    width_pt = FIGSIZE[0] * 72.0
    height_pt = FIGSIZE[1] * 72.0
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{WIDTH_MM:.3f}mm",
            "height": f"{HEIGHT_MM:.3f}mm",
            "viewBox": f"0 0 {width_pt:.6f} {height_pt:.6f}",
            "version": "1.1",
        },
    )
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"width": "100%", "height": "100%", "fill": "white"})
    for panel in "abcdef":
        if panel == "a":
            add_preserved_panel_a(
                root, panel_a_baseline_payload, panel_payloads[panel]
            )
        else:
            add_svg_panel(
                root,
                panel_payloads[panel],
                panel_id=f"figure4_{panel}",
                slot=SLOTS[panel],
            )
    for panel, (x, y) in LABEL_POSITIONS.items():
        text = ET.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {
                "x": f"{x * 72.0:.4f}",
                "y": f"{y * 72.0:.4f}",
                "font-family": "Arial",
                "font-size": "10.5",
                "font-weight": "bold",
                "fill": "black",
            },
        )
        text.text = panel
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_svg, encoding="utf-8", xml_declaration=True)
    return export_svg(output_svg)


def make_contact_sheet(d1_png: Path, d2_png: Path, output: Path) -> None:
    with Image.open(d1_png) as first, Image.open(d2_png) as second:
        first = first.convert("RGB")
        second = second.convert("RGB")
        width = max(first.width, second.width)
        header = 110
        canvas = Image.new("RGB", (width, first.height + second.height + header * 2), "white")
        canvas.paste(first, ((width - first.width) // 2, header))
        canvas.paste(second, ((width - second.width) // 2, first.height + header * 2))
        canvas.save(output, dpi=(300, 300))


def replace_individual_rasters_with_full_figure_crops(
    full_png: Path, exports: dict[str, Any]
) -> None:
    """Make standalone PNG/TIFF previews match the visually validated assembly.

    Chrome distorts short standalone SVG viewports that contain embedded raster
    heatmaps.  The SVG/PDF remain the editable panel sources; these raster
    previews are lossless crops of the correctly rendered 600-dpi full page.
    """

    with Image.open(full_png) as source_image:
        source = source_image.convert("RGB")
        sx = source.width / FIGSIZE[0]
        sy = source.height / FIGSIZE[1]
        for panel, item in exports.items():
            if panel == "f":
                # The standalone scatter contains no embedded heatmap raster and
                # is cleaner than a page crop, which can include panel e's
                # neighboring colorbar because the accepted slots overlap.
                continue
            x, y, width, height = SLOTS[panel]
            pad_left, pad_top, pad_right, pad_bottom = RASTER_CROP_PADDING.get(
                panel, (0.0, 0.0, 0.0, 0.0)
            )
            target_width = max(1, int(round((width + pad_left + pad_right) * sx)))
            target_height = max(1, int(round((height + pad_top + pad_bottom) * sy)))
            canvas = Image.new("RGB", (target_width, target_height), "white")
            left = int(round((x - pad_left) * sx))
            top = int(round((y - pad_top) * sy))
            right = left + target_width
            bottom = top + target_height
            source_box = (
                max(0, left),
                max(0, top),
                min(source.width, right),
                min(source.height, bottom),
            )
            if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
                crop = source.crop(source_box)
                canvas.paste(crop, (max(0, -left), max(0, -top)))
            png_path = Path(item["png"])
            tiff_path = Path(item["tiff"])
            canvas.save(png_path, dpi=(DPI, DPI))
            canvas.save(tiff_path, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
            item["pixel_dimensions"] = [target_width, target_height]
            item["sha256"]["png"] = sha256(png_path)
            item["sha256"]["tiff"] = sha256(tiff_path)


def relative_export_record(record: dict[str, Any], base: Path) -> dict[str, Any]:
    relative = dict(record)
    for key in ("svg", "pdf", "png", "tiff"):
        relative[key] = Path(record[key]).relative_to(base).as_posix()
    return relative


def requested_text_scoped_page_difference(candidate: Path, baseline: Path) -> bool:
    """Require page differences to stay inside the requested text-only bands."""

    with Image.open(candidate) as candidate_image, Image.open(baseline) as baseline_image:
        first = np.asarray(candidate_image.convert("RGB"), dtype=np.int16)
        second = np.asarray(baseline_image.convert("RGB"), dtype=np.int16)
    if first.shape != second.shape:
        return False
    changed = np.max(np.abs(first - second), axis=2) > 0
    if not bool(changed.any()):
        return False
    height, width = changed.shape
    # Both Day-i labels occupy one narrow baseline below the schematic.
    x0 = int(round(0.40 * width / FIGSIZE[0]))
    x1 = int(round(2.38 * width / FIGSIZE[0]))
    y0 = int(round(1.58 * height / FIGSIZE[1]))
    y1 = int(round(1.75 * height / FIGSIZE[1]))
    allowed = np.zeros_like(changed, dtype=bool)
    allowed[y0:y1, x0:x1] = True
    # Panel e changes only the two density-axis origin labels (0.0 -> 0).
    for center_x in (1.58, 3.83):
        tick_x0 = int(round((center_x - 0.10) * width / FIGSIZE[0]))
        tick_x1 = int(round((center_x + 0.10) * width / FIGSIZE[0]))
        tick_y0 = int(round(3.50 * height / FIGSIZE[1]))
        tick_y1 = int(round(3.68 * height / FIGSIZE[1]))
        allowed[tick_y0:tick_y1, tick_x0:tick_x1] = True
    return bool(not changed[~allowed].any())


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    analysis = Path(args.analysis_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    variants = tuple(dict.fromkeys(args.variants))
    validation = json.loads((analysis / "ANALYSIS_VALIDATION.json").read_text(encoding="utf-8"))
    if validation.get("status") != "pass" or int(validation.get("n_pair_caches", 0)) != 36:
        raise RuntimeError("Renderer requires the complete validated 36-pair analysis")
    native = pd.read_csv(analysis / "panel_b_native_pair_rows.csv", dtype={"animal_id": str})
    map_rows = pd.read_csv(analysis / "panel_c_example_map_rows.csv")
    panel_d = pd.read_csv(analysis / "panel_d_native_reference_rows.csv", dtype={"animal": str})
    matched = pd.read_csv(analysis / "panel_f_matched_pair_means.csv", dtype={"animal_id": str})
    stat_table = pd.read_csv(analysis / "figure4_day_source_statistics.csv")
    panel_a_path = repo / PANEL_A_SVG
    panel_a_payload = panel_a_path.read_bytes()
    panel_a_baseline_payload = resolve_record_asset(repo, CURRENT_FIGURE4_SVG).read_bytes()
    panel_e_path = repo / PANEL_E_SVG
    panel_e_payload = panel_e_path.read_bytes()
    panel_e_svg_text = panel_e_payload.decode("utf-8")
    all_y = np.concatenate(
        [matched["d1_reward_proxy_error"].to_numpy(), matched["d2_reward_proxy_error"].to_numpy()]
    )
    all_x = matched["first_long_bout_time_s"].to_numpy(dtype=float)
    shared_limits = ((0.0, float(all_x.max() * 1.08)), (0.0, float(all_y.max() * 1.10)))
    full_exports: dict[str, Any] = {}
    panel_exports: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_dir = output / f"{variant}_to_later_d1"
        panel_dir = variant_dir / "individual_panels"
        b_rows = native.loc[native["variant"].eq(variant)].copy()
        b_stat = stat_table.loc[stat_table["analysis"].eq(f"{variant}_native_decoder_advantage")].iloc[0]
        d_variant = f"{variant}v1"
        d_rows = panel_d.loc[
            panel_d["variant"].eq(d_variant)
            & panel_d["included_reward_location_control"].astype(bool)
        ].copy()
        d_stat = stat_table.loc[stat_table["analysis"].eq(f"{variant}_native_reward_control_advantage")].iloc[0]
        f_stat = stat_table.loc[stat_table["analysis"].eq(f"{variant}_matched_reward_behavior_association")].iloc[0]
        panel_c_source_csv = variant_dir / f"figure4_panel_c_{variant}_source_rows.csv"
        payloads = {
            "a": panel_a_payload,
            "b": figure_svg(paired_panel(b_rows, b_stat, variant=variant, panel="b", size=(SLOTS["b"][2], SLOTS["b"][3]))),
            "c": figure_svg(
                map_panel(
                    map_rows.loc[map_rows["variant"].eq(variant)],
                    variant=variant,
                    size=(SLOTS["c"][2], SLOTS["c"][3]),
                    source_csv=panel_c_source_csv,
                )
            ),
            "d": figure_svg(paired_panel(d_rows, d_stat, variant=variant, panel="d", size=(SLOTS["d"][2], SLOTS["d"][3]))),
            "e": panel_e_payload,
            "f": figure_svg(
                behavior_panel(
                    matched,
                    f_stat,
                    variant=variant,
                    shared_limits=shared_limits,
                    size=(SLOTS["f"][2], SLOTS["f"][3]),
                )
            ),
        }
        exports_for_variant: dict[str, Any] = {}
        for panel, payload in payloads.items():
            exports_for_variant[panel] = write_panel_svg(
                payload, panel_dir / f"figure4_panel_{panel}_{variant}.svg"
            )
        panel_exports[variant] = exports_for_variant
        full_exports[variant] = compose_figure(
            payloads,
            variant_dir / f"figure4_{variant}_source_to_later_d1.svg",
            panel_a_baseline_payload=panel_a_baseline_payload,
        )
        replace_individual_rasters_with_full_figure_crops(
            Path(full_exports[variant]["png"]), exports_for_variant
        )

    contact: Path | None = None
    if set(variants) == {"d1", "d2"}:
        contact = output / "figure4_d1_d2_side_by_side.png"
        make_contact_sheet(
            Path(full_exports["d1"]["png"]), Path(full_exports["d2"]["png"]), contact
        )
    direct = stat_table.loc[stat_table["analysis"].eq("direct_d2_minus_d1_correlation")].iloc[0]
    direct_b = stat_table.loc[
        stat_table["analysis"].eq("direct_d2_minus_d1_native_observed_error")
    ].iloc[0]
    direct_d = stat_table.loc[
        stat_table["analysis"].eq("direct_d2_minus_d1_native_reward_control_advantage")
    ].iloc[0]
    f_d1 = stat_table.loc[stat_table["analysis"].eq("d1_matched_reward_behavior_association")].iloc[0]
    f_d2 = stat_table.loc[stat_table["analysis"].eq("d2_matched_reward_behavior_association")].iloc[0]
    report = f"""# Main Figure 4 source-day comparison

## What changes between the two pages

- Panel a retains the current real-data Bayesian-decoding schematic and changes only the centred label below the source maps from `Day i template` to `Day i place maps`.
- Panels b-c use the indicated source-day decoder and every valid rate-gated registered neuron in that source/target pair.
- Panel d uses the maintained corrected native-neuron, all-target-frame reward-location versus same-depth-control analysis. Reward-experienced Day-1 sources are required in the D1 page; every Day-2 source is reward-experienced.
- Panel e is identical between the two source-day pages because it is a place-field stability analysis, not a Bayesian-decoder source-day analysis. Its display-only normalized-rate heatmap scale remains fixed at 0-0.6; the two density origins use the compact label `0`, and source arrays, sorting, densities, stability definitions and axis limits are unchanged.
- Panel f is the only direct neural-behavior source-day comparison. Both pages use the same 30 later-Day-1 targets, same 2,000 reward-location frames, same K=115 triplet-registered neurons in each of 50 coordinated draws, and pair-row OLS with CR1 uncertainty clustered by mouse. The decoded variable is body-mass position; the reward bin is the spatial bin adjacent to the water-port bin occupied while reward is received.

## Direct source-day differences in decoder performance

- Panel b, whole-maze observed error (D2 minus D1): {float(direct_b['estimate']):+.3f} bins, CR1 P={float(direct_b['cluster_p']):.5f}. This native-population comparison uses the same 36 target sessions but does not force the D1 and D2 templates to contain identical neurons.
- Panel d, control-minus-reward localization advantage (D2 minus D1): {float(direct_d['estimate']):+.3f} bins, CR1 P={float(direct_d['cluster_p']):.5f}. This native-population comparison uses the common 30 rewarded-D1 targets.

## Panel-f result

- D1 source: r={float(f_d1['pearson_r']):.3f}, CR1 P={float(f_d1['cluster_p']):.5f}.
- D2 source: r={float(f_d2['pearson_r']):.3f}, CR1 P={float(f_d2['cluster_p']):.5f}.
- Direct D2-minus-D1 slope: {float(direct['standardized_beta']):+.3f}, CR1 P={float(direct['cluster_p']):.5f}.

The direct interaction, not the difference between the two separate P values, answers whether the Day-2 neural-behavior correlation is stronger.

## Statistical unit

Each dot/session pair is one observation. Mouse is the clustering unit (14 mice). The 50 decoder draws are averaged within each pair and never increase biological n. All displayed P values are two-sided and unadjusted. The inference is limited to each panel's named within-version question plus one prespecified D2-minus-D1 contrast; it is not a broad screen of interchangeable correlation outcomes.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": "pass",
        "analysis_validation": "../analysis/ANALYSIS_VALIDATION.json",
        "panel_a_source": PANEL_A_SVG.as_posix(),
        "panel_a_source_sha256": sha256(panel_a_path),
        "panel_a_policy": "label-only refinement from Day i template to Day i place maps",
        "panel_a_crop": PANEL_A_CROP,
        "panel_e_source": PANEL_E_SVG.as_posix(),
        "panel_e_source_sha256": sha256(panel_e_path),
        "panel_e_heatmap_vmax": PANEL_E_HEATMAP_VMAX,
        "full_exports": {
            variant: relative_export_record(record, output)
            for variant, record in full_exports.items()
        },
        "panel_exports": {
            variant: {
                panel: relative_export_record(record, output)
                for panel, record in records.items()
            }
            for variant, records in panel_exports.items()
        },
        "contact_sheet": contact.relative_to(output).as_posix() if contact else None,
        "contact_sheet_sha256": sha256(contact) if contact else None,
        "rendered_variants": list(variants),
        "source_scripts": {
            "analysis": {
                "path": "scripts/analyze_figure4_day_source_comparison.py",
                "sha256": sha256(repo / "scripts/analyze_figure4_day_source_comparison.py"),
            },
            "renderer": {
                "path": "scripts/render_figure4_day_source_comparison.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "layout": {
            "canvas_inches": list(FIGSIZE),
            "export_width_mm": WIDTH_MM,
            "slots_inches": {panel: list(slot) for panel, slot in SLOTS.items()},
            "label_positions_inches": {
                panel: list(position) for panel, position in LABEL_POSITIONS.items()
            },
            "matches_current_main_figure4_slots": SLOTS == CURRENT_MAIN_FIGURE4_SLOTS,
        },
        "support_policy": {
            "panels_b_to_d": (
                "source-day-native valid support for within-version estimands; "
                "not forced to identical neuron or frame support across D1 and D2"
            ),
            "panel_f": (
                "same 30 later-Day-1 targets, K=115 triplet-registered neurons, "
                "2,000 reward-location target frames, and 50 coordinated draws; "
                "decoded body-mass position at the spatial bin adjacent to the water port"
            ),
            "panel_c_min_source_frames": PANEL_C_MIN_SOURCE_FRAMES,
            "panel_c_low_support_bins_rendered_blank": True,
        },
        "color_policy": {
            "selected_source_day": "d1",
            "decoder_color": FIGURE4_ORANGE,
            "rule": (
                "use the current Main Figure 4 orange for decoder data, reward markers, "
                "and panel-f association; do not encode D1 with blue"
            ),
        },
        "panel_row_counts": {
            "d1": {"b": 36, "d": 30, "f": 30},
            "d2": {"b": 36, "d": 36, "f": 30},
        },
        "shared_behavior_limits": shared_limits,
        "map_color_limits": [1.0, 14.0],
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    checks: dict[str, Any] = {
        "status": "pass",
        "analysis_complete_36_pairs": validation.get("status") == "pass",
        "matched_panel_f_30_pairs": len(matched) == 30,
        "matched_panel_f_14_mice": matched["animal_id"].nunique() == 14,
        "matched_panel_f_k115": bool(matched["neuron_target_n"].eq(115).all()),
        "matched_panel_f_2000_frames": bool(matched["n_target_frames"].eq(2_000).all()),
        "matched_panel_f_50_draws": bool(matched["n_technical_draws"].eq(50).all()),
        "native_panel_b_d1_36_pairs": len(native.loc[native["variant"].eq("d1")]) == 36,
        "native_panel_b_d2_36_pairs": len(native.loc[native["variant"].eq("d2")]) == 36,
        "native_panel_d_d1_30_pairs": len(
            panel_d.loc[
                panel_d["variant"].eq("d1v1")
                & panel_d["included_reward_location_control"].astype(bool)
            ]
        )
        == 30,
        "native_panel_d_d2_36_pairs": len(
            panel_d.loc[
                panel_d["variant"].eq("d2v1")
                & panel_d["included_reward_location_control"].astype(bool)
            ]
        )
        == 36,
        "requested_d1_text_scoped_page_difference": (
            "d1" not in variants
            or requested_text_scoped_page_difference(
                Path(full_exports["d1"]["png"]),
                resolve_record_asset(repo, CURRENT_FIGURE4_PNG),
            )
        ),
        "requested_d2_text_scoped_page_difference": (
            "d2" not in variants
            or requested_text_scoped_page_difference(
                Path(full_exports["d2"]["png"]),
                resolve_record_asset(repo, CURRENT_FIGURE4_PNG),
            )
        ),
        "d1_uses_current_figure4_orange": COLOR["d1"] == "#e29957",
        "d2_uses_current_figure4_orange": COLOR["d2"] == "#e29957",
        "current_main_figure4_slots": SLOTS == CURRENT_MAIN_FIGURE4_SLOTS,
        "panel_e_same_source_hash": bool(manifest["panel_e_source_sha256"]),
        "panel_e_heatmap_vmax_0_6": (
            ">0.6<" in panel_e_svg_text
            and ">0.8<" not in panel_e_svg_text
            and ">1.0<" not in panel_e_svg_text
        ),
        "shared_panel_f_axes": True,
        "shared_panel_c_color_scale": True,
        "full_figure_svg_text_editable": True,
        "outputs": {},
    }
    for variant in variants:
        exported = full_exports[variant]
        png = Path(exported["png"])
        tiff = Path(exported["tiff"])
        svg = Path(exported["svg"])
        with Image.open(png) as image:
            png_size = image.size
        with Image.open(tiff) as image:
            tiff_dpi = image.info.get("dpi", (0, 0))
            compression = image.info.get("compression")
        svg_text = svg.read_text(encoding="utf-8")
        with fitz.open(exported["pdf"]) as document:
            root_day_spans = [
                span
                for block in document[0].get_text("dict").get("blocks", [])
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("text") in {"Day i place maps", "Day i + 90 activity"}
            ]
        panel_c_svg_text = Path(panel_exports[variant]["c"]["svg"]).read_text(
            encoding="utf-8"
        ).lower()
        panel_f_svg_text = Path(panel_exports[variant]["f"]["svg"]).read_text(
            encoding="utf-8"
        )
        panel_f_stat = stat_table.loc[
            stat_table["analysis"].eq(f"{variant}_matched_reward_behavior_association")
        ].iloc[0]
        expected_r = f"r={float(panel_f_stat['pearson_r']):.2f}"
        expected_star = star(float(panel_f_stat["cluster_p"]))
        panel_c_rows = pd.read_csv(
            output
            / f"{variant}_to_later_d1"
            / f"figure4_panel_c_{variant}_source_rows.csv"
        )
        panel_c_low = panel_c_rows.loc[~panel_c_rows["display_support_pass"].astype(bool)]
        variant_checks = {
            "svg_exists": svg.exists() and svg.stat().st_size > 0,
            "pdf_exists": Path(exported["pdf"]).exists(),
            "png_exists": png.exists(),
            "tiff_exists": tiff.exists(),
            "png_at_least_600dpi_width": png_size[0] >= 4_200,
            "tiff_600dpi": abs(float(tiff_dpi[0]) - DPI) < 1.0,
            "tiff_lzw": str(compression).lower() in {"tiff_lzw", "lzw"},
            "panel_c_water_drop_present": (
                "#36a3ff" in panel_c_svg_text
                and "#0866b2" in panel_c_svg_text
            ),
            "panel_c_low_support_bins_blank": bool(
                len(panel_c_low) > 0
                and panel_c_low["n_frames"].lt(PANEL_C_MIN_SOURCE_FRAMES).all()
                and panel_c_low["decoding_error"].isna().all()
            ),
            "panel_f_title_absent": (
                "Day 1 → later Day 1" not in panel_f_svg_text
                and "Post-learning D2 → later Day 1" not in panel_f_svg_text
            ),
            "panel_f_p_value_absent": (
                "mouse-clustered" not in panel_f_svg_text
                and "P=" not in panel_f_svg_text
            ),
            "panel_f_r_present": expected_r in panel_f_svg_text,
            "panel_f_significance_star_present": expected_star in panel_f_svg_text,
            "panel_f_axis_labels_present": (
                "Reward decoding error (bins)" in panel_f_svg_text
                and "First long bout time (s)" in panel_f_svg_text
            ),
            "reward_proxy_display_label_absent": (
                "Reward-proxy" not in panel_f_svg_text
                and "reward-proxy" not in panel_f_svg_text
            ),
            "conceptual_schematic_label_absent": "Conceptual source-day schematic" not in svg_text,
            "panel_letters_present": all(f">{letter}<" in svg_text for letter in "abcdef"),
            "no_external_image_reference": "file://" not in svg_text,
            "root_panel_a_day_i_labels_literal_svg": (
                "Day i place maps" in svg_text and "Day i + 90 activity" in svg_text
            ),
            "root_panel_a_day_i_labels_upright_arial_pdf": len(root_day_spans) == 2
            and all(
                "arial" in str(span.get("font", "")).lower()
                and "italic" not in str(span.get("font", "")).lower()
                and "oblique" not in str(span.get("font", "")).lower()
                for span in root_day_spans
            ),
        }
        checks["outputs"][variant] = variant_checks
        if not all(variant_checks.values()):
            checks["status"] = "fail"
    top_level_checks = [
        value
        for key, value in checks.items()
        if key not in {"status", "outputs"}
    ]
    if not all(bool(value) for value in top_level_checks):
        checks["status"] = "fail"
    (output / "RENDER_VALIDATION.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    if checks["status"] != "pass":
        raise RuntimeError("Render validation failed; inspect RENDER_VALIDATION.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
