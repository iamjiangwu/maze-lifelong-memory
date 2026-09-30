"""Render animal 418 four-session local-correlation image panels for ED5.

The renderer reads the image payloads directly from the MATLAB ``corr.fig``
sources and uses Python/matplotlib for every visual output.  It leaves the
existing four-session CellReg footprint panel unchanged and writes a matching
local-correlation row plus an optional two-row source-rendered composite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from scipy.io import loadmat

try:
    from scripts.render_ed05_registration_flow_and_four_session import (
        prepare_four_session_example,
    )
except ModuleNotFoundError:  # pragma: no cover - direct scripts/ execution
    from render_ed05_registration_flow_and_four_session import (  # type: ignore
        prepare_four_session_example,
    )


REPO = Path(__file__).resolve().parents[1]
DEFAULT_FOOTPRINT_ROOT = Path(r"F:\footprint-new\418")
DEFAULT_OUTPUT_DIR = (
    REPO
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_05"
    / "registration_flow_and_four_session"
)
DEFAULT_DPI = 600
MICRONS_PER_PIXEL = 1.8
SCALE_BAR_UM = 100.0
SCALE_BAR_PX = SCALE_BAR_UM / MICRONS_PER_PIXEL
DISPLAY_CONTRAST_PERCENTILES = (0.5, 99.5)

CORRELATION_SPECS = (
    ("day1", "Day 1", "0", Path(r"G:\418\corr.fig")),
    ("day91", "Day 91", "3", Path(r"G:\418-3\corr.fig")),
    (
        "day181",
        "Day 181",
        "6",
        Path(r"G:\418-6\all\4x_smoothed_cat\corr.fig"),
    ),
    ("day271", "Day 271", "9", Path(r"G:\418-9\corr.fig")),
)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paper_rc() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.0,
        "axes.titlesize": 6.2,
        "legend.fontsize": 4.9,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


def _load_matlab_local_correlation_figure(fig_path: Path) -> dict[str, object]:
    """Extract the main two-dimensional image from a MATLAB HG2 figure."""

    if not fig_path.exists():
        raise FileNotFoundError(fig_path)
    data = loadmat(
        str(fig_path),
        variable_names=["hgS_070000"],
        squeeze_me=True,
        struct_as_record=False,
    )
    if "hgS_070000" not in data:
        raise KeyError(f"{fig_path} does not contain MATLAB figure variable hgS_070000")
    matlab_fig = data["hgS_070000"]
    figure_children = list(np.atleast_1d(matlab_fig.children).flat)
    axes_candidates = [
        child for child in figure_children if getattr(child, "type", None) == "axes"
    ]
    if not axes_candidates:
        raise ValueError(f"No image axes found in {fig_path}")

    selected: tuple[object, object] | None = None
    selected_size = -1
    for axes_obj in axes_candidates:
        for child in np.atleast_1d(axes_obj.children).flat:
            if getattr(child, "type", None) != "image":
                continue
            cdata = np.asarray(child.properties.CData)
            if cdata.ndim == 2 and min(cdata.shape) > 1 and cdata.size > selected_size:
                selected = (axes_obj, child)
                selected_size = int(cdata.size)
    if selected is None:
        raise ValueError(f"No two-dimensional image payload found in {fig_path}")
    axes_obj, image_obj = selected

    colorbar_candidates = [
        child
        for child in figure_children
        if "colorbar" in str(getattr(child, "type", "")).lower()
    ]
    colormap = getattr(axes_obj.properties, "Colormap", None)
    if colormap is None and colorbar_candidates:
        colormap = getattr(colorbar_candidates[0].properties, "Colormap", None)
    if colormap is None:
        raise ValueError(f"No colormap found in {fig_path}")

    image = np.asarray(image_obj.properties.CData, dtype=np.float64)
    if not np.isfinite(image).all():
        raise ValueError(
            f"Local-correlation image contains nonfinite values: {fig_path}"
        )
    text_children = [
        child
        for child in np.atleast_1d(axes_obj.children).flat
        if getattr(child, "type", None) == "text"
    ]
    return {
        "image": image,
        "xdata": np.asarray(image_obj.properties.XData, dtype=float),
        "ydata": np.asarray(image_obj.properties.YData, dtype=float),
        "clim": np.asarray(axes_obj.properties.CLim, dtype=float),
        "colormap": np.asarray(colormap, dtype=float),
        "title_text": (
            str(text_children[0].properties.String) if text_children else ""
        ),
    }


def _load_all_correlation_panels() -> tuple[list[dict[str, object]], pd.DataFrame]:
    panels: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for panel_key, display_title, month_code, source in CORRELATION_SPECS:
        panel = _load_matlab_local_correlation_figure(source)
        image = np.asarray(panel["image"], dtype=float)
        panels.append(panel)
        rows.append(
            {
                "animal": "418",
                "panel_key": panel_key,
                "display_title": display_title,
                "month_code": month_code,
                "corr_fig_source": str(source.resolve()),
                "corr_fig_sha256": _sha256(source),
                "source_title": panel["title_text"],
                "height_px": int(image.shape[0]),
                "width_px": int(image.shape[1]),
                "n_finite_pixels": int(np.isfinite(image).sum()),
                "data_min": float(image.min()),
                "data_max": float(image.max()),
                "source_clim_min": float(np.asarray(panel["clim"])[0]),
                "source_clim_max": float(np.asarray(panel["clim"])[1]),
                "xdata_start": float(np.asarray(panel["xdata"])[0]),
                "xdata_end": float(np.asarray(panel["xdata"])[-1]),
                "ydata_start": float(np.asarray(panel["ydata"])[0]),
                "ydata_end": float(np.asarray(panel["ydata"])[-1]),
            }
        )

    shapes = {np.asarray(panel["image"]).shape for panel in panels}
    if shapes != {(288, 360)}:
        raise AssertionError(f"Unexpected local-correlation image shapes: {shapes}")
    reference_cmap = np.asarray(panels[0]["colormap"], dtype=float)
    if not all(
        np.array_equal(np.asarray(panel["colormap"], dtype=float), reference_cmap)
        for panel in panels[1:]
    ):
        raise AssertionError(
            "The four local-correlation sources use different colormaps"
        )
    summary = pd.DataFrame(rows)
    summary["shared_clim_min"] = float(summary["data_min"].min())
    summary["shared_clim_max"] = float(summary["data_max"].max())
    pooled_values = np.concatenate(
        [np.asarray(panel["image"], dtype=float).ravel() for panel in panels]
    )
    display_vmin, display_vmax = np.percentile(
        pooled_values, DISPLAY_CONTRAST_PERCENTILES
    )
    summary["display_percentile_low"] = DISPLAY_CONTRAST_PERCENTILES[0]
    summary["display_percentile_high"] = DISPLAY_CONTRAST_PERCENTILES[1]
    summary["display_clim_min"] = float(display_vmin)
    summary["display_clim_max"] = float(display_vmax)
    summary["n_pixels_below_display_clim"] = [
        int(np.count_nonzero(np.asarray(panel["image"]) < display_vmin))
        for panel in panels
    ]
    summary["n_pixels_above_display_clim"] = [
        int(np.count_nonzero(np.asarray(panel["image"]) > display_vmax))
        for panel in panels
    ]
    summary["fraction_pixels_below_display_clim"] = (
        summary["n_pixels_below_display_clim"] / summary["n_finite_pixels"]
    )
    summary["fraction_pixels_above_display_clim"] = (
        summary["n_pixels_above_display_clim"] / summary["n_finite_pixels"]
    )
    summary[
        "display_orientation"
    ] = "native CData; origin upper; no transpose/flip/rotation"
    summary["display_interpolation"] = "nearest"
    summary["microns_per_pixel"] = MICRONS_PER_PIXEL
    return panels, summary


def _draw_image(
    ax: plt.Axes,
    panel: dict[str, object],
    *,
    vmin: float,
    vmax: float,
    show_scale_bar: bool,
) -> None:
    image = np.asarray(panel["image"], dtype=float)
    ax.imshow(
        image,
        cmap=ListedColormap(np.asarray(panel["colormap"], dtype=float)),
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_axis_off()
    if show_scale_bar:
        x1 = image.shape[1] - 12.0
        x0 = x1 - SCALE_BAR_PX
        y = image.shape[0] - 12.0
        ax.plot(
            [x0, x1],
            [y, y],
            color="white",
            linewidth=1.4,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2.0,
            y - 5.0,
            "100 \N{MICRO SIGN}m",
            color="white",
            ha="center",
            va="bottom",
            fontsize=5.0,
        )


def _save_bundle(fig: plt.Figure, stem: Path, *, dpi: int) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        suffix: stem.with_suffix(f".{suffix}")
        for suffix in ("png", "svg", "pdf", "tiff")
    }
    common = {"facecolor": "white", "bbox_inches": None, "pad_inches": 0}
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(paths["png"], dpi=dpi, **common)
        fig.savefig(paths["svg"], metadata={"Date": None}, **common)
        fig.savefig(
            paths["pdf"], metadata={"CreationDate": None, "ModDate": None}, **common
        )
    with Image.open(paths["png"]) as image:
        image.convert("RGB").save(
            paths["tiff"],
            dpi=(dpi, dpi),
            compression="tiff_lzw",
        )
    return {suffix: _rel(path) for suffix, path in paths.items()}


def _render_correlation_row(
    panels: list[dict[str, object]], summary: pd.DataFrame
) -> plt.Figure:
    vmin = float(summary["display_clim_min"].iloc[0])
    vmax = float(summary["display_clim_max"].iloc[0])
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.48), facecolor="white")
        for index, (ax, panel, spec) in enumerate(
            zip(axes, panels, CORRELATION_SPECS, strict=True)
        ):
            _draw_image(
                ax,
                panel,
                vmin=vmin,
                vmax=vmax,
                show_scale_bar=(index == len(panels) - 1),
            )
            ax.set_title(spec[1], pad=2.0, fontweight="normal")
        fig.subplots_adjust(
            left=0.004, right=0.996, top=0.88, bottom=0.015, wspace=0.025
        )
    return fig


def _render_individual(
    panel: dict[str, object], title: str, *, vmin: float, vmax: float
) -> plt.Figure:
    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(1.8, 1.48), facecolor="white")
        _draw_image(ax, panel, vmin=vmin, vmax=vmax, show_scale_bar=True)
        ax.set_title(title, pad=2.0, fontweight="normal")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.015)
    return fig


def _draw_footprint_image(
    ax: plt.Axes, rgb: np.ndarray, *, show_scale_bar: bool
) -> None:
    ax.imshow(rgb, interpolation="nearest", origin="upper")
    ax.set_axis_off()
    if show_scale_bar:
        x1 = rgb.shape[1] - 12.0
        x0 = x1 - SCALE_BAR_PX
        y = rgb.shape[0] - 12.0
        ax.plot([x0, x1], [y, y], color="white", linewidth=1.4, solid_capstyle="butt")
        ax.text(
            (x0 + x1) / 2.0,
            y - 5.0,
            "100 \N{MICRO SIGN}m",
            color="white",
            ha="center",
            va="bottom",
            fontsize=5.0,
        )


def _render_combined(
    footprint_rgbs: list[np.ndarray],
    correlation_panels: list[dict[str, object]],
    summary: pd.DataFrame,
) -> plt.Figure:
    vmin = float(summary["display_clim_min"].iloc[0])
    vmax = float(summary["display_clim_max"].iloc[0])
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.05), facecolor="white")
        for column, (rgb, panel, spec) in enumerate(
            zip(footprint_rgbs, correlation_panels, CORRELATION_SPECS, strict=True)
        ):
            _draw_footprint_image(axes[0, column], rgb, show_scale_bar=(column == 3))
            axes[0, column].set_title(spec[1], pad=2.0, fontweight="normal")
            _draw_image(
                axes[1, column],
                panel,
                vmin=vmin,
                vmax=vmax,
                show_scale_bar=(column == 3),
            )

        fig.text(
            0.006, 0.685, "CellReg footprints", rotation=90, ha="left", va="center"
        )
        fig.text(0.006, 0.275, "Local correlation", rotation=90, ha="left", va="center")
        handles = [
            Patch(
                facecolor="#00ff31",
                edgecolor="none",
                label="Registered across all four sessions",
            ),
            Patch(facecolor="#e5e5e5", edgecolor="none", label="Other detected cells"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.005),
            handlelength=1.0,
        )
        fig.subplots_adjust(
            left=0.025, right=0.996, top=0.95, bottom=0.085, wspace=0.025, hspace=0.035
        )
    return fig


def _image_audit(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return {
            "path": _rel(path),
            "width_px": int(image.width),
            "height_px": int(image.height),
            "nonblank": bool(np.ptp(rgb.reshape(-1, 3), axis=0).max() > 1),
            "sha256": _sha256(path),
        }


def render_all(
    *, footprint_root: Path, output_dir: Path, dpi: int
) -> dict[str, object]:
    panels, summary = _load_all_correlation_panels()
    (
        footprint_rgbs,
        footprint_stacks,
        _common,
        footprint_summary,
    ) = prepare_four_session_example(footprint_root)
    footprint_shapes = [tuple(stack.shape[1:]) for stack in footprint_stacks]
    correlation_shapes = [tuple(np.asarray(panel["image"]).shape) for panel in panels]
    if footprint_shapes != correlation_shapes:
        raise AssertionError(
            f"Footprint/local-correlation FOV mismatch: {footprint_shapes} versus {correlation_shapes}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_dir / "ed05_four_session_418_local_correlation_source_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    arrays_path = output_dir / "ed05_four_session_418_local_correlation_source_data.npz"
    np.savez_compressed(
        arrays_path,
        **{
            f"{spec[0]}_cdata": np.asarray(panel["image"], dtype=np.float64)
            for spec, panel in zip(CORRELATION_SPECS, panels, strict=True)
        },
        source_shared_clim=np.asarray(
            [summary["shared_clim_min"].iloc[0], summary["shared_clim_max"].iloc[0]],
            dtype=np.float64,
        ),
        display_clim=np.asarray(
            [summary["display_clim_min"].iloc[0], summary["display_clim_max"].iloc[0]],
            dtype=np.float64,
        ),
        colormap=np.asarray(panels[0]["colormap"], dtype=np.float64),
    )

    row_fig = _render_correlation_row(panels, summary)
    row_outputs = _save_bundle(
        row_fig,
        output_dir / "ed05_four_session_418_local_correlation_paper",
        dpi=dpi,
    )
    plt.close(row_fig)

    individual_outputs: dict[str, dict[str, str]] = {}
    vmin = float(summary["display_clim_min"].iloc[0])
    vmax = float(summary["display_clim_max"].iloc[0])
    for panel, spec in zip(panels, CORRELATION_SPECS, strict=True):
        fig = _render_individual(panel, spec[1], vmin=vmin, vmax=vmax)
        individual_outputs[spec[0]] = _save_bundle(
            fig,
            output_dir / f"ed05_four_session_418_local_correlation_{spec[0]}",
            dpi=dpi,
        )
        plt.close(fig)

    combined_fig = _render_combined(footprint_rgbs, panels, summary)
    combined_outputs = _save_bundle(
        combined_fig,
        output_dir / "ed05_four_session_418_coregistration_and_local_correlation_paper",
        dpi=dpi,
    )
    plt.close(combined_fig)

    png_paths = [
        REPO / row_outputs["png"],
        REPO / combined_outputs["png"],
        *[REPO / outputs["png"] for outputs in individual_outputs.values()],
    ]
    manifest = {
        "script": "scripts/render_ed05_418_local_correlation.py",
        "figure": "Extended Data Figure 5",
        "backend": "python_matplotlib",
        "figure_contract": {
            "core_conclusion": "The native local-correlation image remains directly inspectable for each session in the animal 418 four-session CellReg example.",
            "archetype": "microscopy_image_plate",
            "evidence_chain": [
                "animal 418 Day 1/91/181/271 session mapping",
                "direct extraction of each corr.fig two-dimensional CData payload",
                "native 288 x 360 orientation and common field-of-view geometry",
                "modestly contrast-enhanced shared intensity scale and calibrated 100 micrometre scale bar",
            ],
            "review_risks": [
                "wrong longitudinal session folder",
                "transpose, flip, rotation, crop, or smoothing of source CData",
                "session-specific contrast rescaling that impedes comparison",
                "mixing a CellReg diagnostic correlation plot with the CNMF-E local-correlation image",
            ],
        },
        "animal": "418",
        "sessions": summary.to_dict(orient="records"),
        "footprint_source_summary": footprint_summary.to_dict(orient="records"),
        "source_data": _rel(arrays_path),
        "source_summary": _rel(summary_path),
        "outputs": {
            "local_correlation_row": row_outputs,
            "individual_sessions": individual_outputs,
            "coregistration_and_local_correlation_composite": combined_outputs,
        },
        "image_integrity": {
            "source": "MATLAB figure CData; no screenshot or raster crop",
            "orientation": "native array with origin upper; no transpose, flip, or rotation",
            "interpolation": "nearest",
            "crop": "none; complete 288 x 360 field of view",
            "contrast": "one shared pooled-session 0.5th-to-99.5th percentile window across all four sessions",
            "source_shared_full_range": [
                float(summary["shared_clim_min"].iloc[0]),
                float(summary["shared_clim_max"].iloc[0]),
            ],
            "display_percentiles": list(DISPLAY_CONTRAST_PERCENTILES),
            "display_clim": [vmin, vmax],
            "clipping_audit_fields": [
                "n_pixels_below_display_clim",
                "n_pixels_above_display_clim",
                "fraction_pixels_below_display_clim",
                "fraction_pixels_above_display_clim",
            ],
            "local_selective_adjustment": False,
            "scale": f"{SCALE_BAR_UM:g} micrometres at {MICRONS_PER_PIXEL:g} micrometres per pixel",
        },
        "validation": {
            "correlation_shapes_yx": [list(shape) for shape in correlation_shapes],
            "footprint_shapes_yx": [list(shape) for shape in footprint_shapes],
            "fov_shapes_match": footprint_shapes == correlation_shapes,
            "display_clim_is_inside_source_range": bool(
                float(summary["shared_clim_min"].iloc[0]) < vmin < vmax
                < float(summary["shared_clim_max"].iloc[0])
            ),
            "all_sources_have_50000_frame_title": bool(
                summary["source_title"]
                .eq("CNMF-E local correlation image (50000 frames)")
                .all()
            ),
            "all_outputs_nonblank": all(
                bool(_image_audit(path)["nonblank"]) for path in png_paths
            ),
            "png_audit": [_image_audit(path) for path in png_paths],
        },
    }
    if not manifest["validation"]["all_sources_have_50000_frame_title"]:
        raise AssertionError("At least one corr.fig source has an unexpected title")
    if not manifest["validation"]["all_outputs_nonblank"]:
        raise AssertionError("At least one rendered PNG is blank")
    manifest_path = output_dir / "ed05_four_session_418_local_correlation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = _rel(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--footprint-root", type=Path, default=DEFAULT_FOOTPRINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = render_all(
        footprint_root=args.footprint_root,
        output_dir=args.output_dir,
        dpi=int(args.dpi),
    )
    print(
        json.dumps(
            {
                "local_correlation_row": manifest["outputs"]["local_correlation_row"],
                "combined": manifest["outputs"][
                    "coregistration_and_local_correlation_composite"
                ],
                "manifest": manifest["manifest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
