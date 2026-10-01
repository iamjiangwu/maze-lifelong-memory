"""Render ED5 registration denominator flow and a four-session example.

The example is regenerated from animal 418 footprint stacks and the maintained
0/3/6/9-month common CIMap.  The denominator panel uses the exact corrected
36-pair registration audit and keeps every session pair visible.
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
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from PIL import Image

try:
    from scripts.render_poster_figure2a_418_0to9_footprint_variants import (
        SESSION_SPECS,
        _cellreg_projection_rgb,
        _load_all_session_map,
        _load_footprint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct scripts/ execution
    from render_poster_figure2a_418_0to9_footprint_variants import (  # type: ignore
        SESSION_SPECS,
        _cellreg_projection_rgb,
        _load_all_session_map,
        _load_footprint,
    )


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(r"F:\footprint-new\418")
REGISTRATION_AUDIT_DIR = (
    REPO
    / "docs"
    / "main_figs_v3_source_render_2026-06-24"
    / "registration_quality_exact_36_2026-07-09"
)
PAIR_AUDIT = REGISTRATION_AUDIT_DIR / "pair_source_and_count_audit.csv"
REGISTRATION_SUMMARY = REGISTRATION_AUDIT_DIR / "registration_quality_summary.json"
DEFAULT_OUTPUT_DIR = (
    REPO
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_05"
    / "registration_flow_and_four_session"
)

TRANSITION_COLORS = {
    "0-3": "#4C78A8",
    "3-6": "#F28E2B",
    "6-9": "#59A14F",
    "9-12": "#B07AA1",
}
DISPLAY_TRANSITION = {key: key.replace("-", "→") + " mo" for key in TRANSITION_COLORS}

for _font_path in (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
):
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))


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


def _save_bundle(fig: plt.Figure, stem: Path, *, dpi: int = 600) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        suffix: stem.with_suffix(f".{suffix}")
        for suffix in ("png", "svg", "pdf", "tiff")
    }
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig.savefig(paths["png"], dpi=dpi, facecolor="white", bbox_inches="tight")
        fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight")
        fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight")
    plt.close(fig)
    with Image.open(paths["png"]) as image:
        image.convert("RGB").save(paths["tiff"], dpi=(dpi, dpi), compression="tiff_lzw")
    return {key: _rel(path) for key, path in paths.items()}


def _image_audit(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        size = image.size
    return {
        "path": _rel(path),
        "width_px": int(size[0]),
        "height_px": int(size[1]),
        "nonblank": bool(max(high - low for low, high in extrema) > 1),
        "sha256": _sha256(path),
    }


def _paper_rc() -> dict[str, object]:
    return {
        "font.family": "Arial",
        "font.size": 6.0,
        "axes.titlesize": 6.2,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.4,
        "ytick.labelsize": 5.4,
        "legend.fontsize": 4.9,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }


def prepare_four_session_example(
    source_root: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    common = _load_all_session_map(source_root)
    rgbs: list[np.ndarray] = []
    stacks: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for key, display_title, map_column, footprint_name, _color in SESSION_SPECS:
        source = source_root / footprint_name
        stack = _load_footprint(source)
        stacks.append(stack)
        common_ids = common[map_column].to_numpy(dtype=int)
        rgb = _cellreg_projection_rgb(stack, common_ids)
        rgbs.append(rgb)
        rows.append(
            {
                "session_key": key,
                "display_title": display_title,
                "session_code_month": map_column,
                "footprint_source": str(source.resolve()),
                "footprint_source_sha256": _sha256(source),
                "n_detected_cells": int(stack.shape[0]),
                "height_px": int(stack.shape[1]),
                "width_px": int(stack.shape[2]),
                "n_registered_all_four": int(len(common)),
                "common_cell_id_min": int(common_ids.min()),
                "common_cell_id_max": int(common_ids.max()),
                "green_signal_pixels": int(
                    np.count_nonzero(
                        (rgb[..., 1] > 0.5)
                        & (rgb[..., 0] <= 0.5)
                        & (rgb[..., 2] <= 0.5)
                    )
                ),
                "white_signal_pixels": int(
                    np.count_nonzero(
                        (rgb[..., 0] > 0.5) & (rgb[..., 1] > 0.5) & (rgb[..., 2] > 0.5)
                    )
                ),
            }
        )
    return rgbs, stacks, common, pd.DataFrame(rows)


def _normalized_cell_image(stack: np.ndarray, one_based_id: int) -> np.ndarray:
    image = np.asarray(stack[int(one_based_id) - 1], dtype=np.float32)
    maximum = float(np.nanmax(image))
    if not np.isfinite(maximum) or maximum <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return np.nan_to_num(image / maximum, copy=False).astype(np.float32)


def _centroid_and_area(image: np.ndarray) -> tuple[float, float, int]:
    mask = image >= 0.5
    area = int(mask.sum())
    if area == 0:
        y, x = np.unravel_index(int(np.nanargmax(image)), image.shape)
        return float(y), float(x), 0
    weights = np.where(mask, image, 0.0)
    total = float(weights.sum())
    yy, xx = np.indices(image.shape)
    return (
        float((yy * weights).sum() / total),
        float((xx * weights).sum() / total),
        area,
    )


def _centered_crop(image: np.ndarray, y: float, x: float, size: int) -> np.ndarray:
    radius = size // 2
    cy = int(round(y))
    cx = int(round(x))
    output = np.zeros((size, size), dtype=np.float32)
    src_y0 = max(0, cy - radius)
    src_y1 = min(image.shape[0], cy + radius + 1)
    src_x0 = max(0, cx - radius)
    src_x1 = min(image.shape[1], cx + radius + 1)
    dst_y0 = src_y0 - (cy - radius)
    dst_x0 = src_x0 - (cx - radius)
    output[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return output


def prepare_matched_cell_montage(
    stacks: list[np.ndarray],
    common: pd.DataFrame,
    source_root: Path,
    *,
    crop_size: int = 31,
) -> tuple[list[list[np.ndarray]], pd.DataFrame]:
    anchors = (
        (0.22, 0.25),
        (0.50, 0.25),
        (0.78, 0.25),
        (0.22, 0.72),
        (0.50, 0.72),
        (0.78, 0.72),
    )
    candidates: list[dict[str, object]] = []
    for common_row, row in common.iterrows():
        session_props: list[tuple[float, float, int]] = []
        for stack, (_key, _title, map_column, _source, _color) in zip(
            stacks, SESSION_SPECS, strict=True
        ):
            image = _normalized_cell_image(stack, int(row[map_column]))
            session_props.append(_centroid_and_area(image))
        day1_y, day1_x, _ = session_props[0]
        height, width = stacks[0].shape[1:]
        if min(prop[2] for prop in session_props) < 4:
            continue
        candidates.append(
            {
                "common_row": int(common_row),
                "day1_x_fraction": float(day1_x / (width - 1)),
                "day1_y_fraction": float(day1_y / (height - 1)),
                "session_props": session_props,
            }
        )
    if len(candidates) < len(anchors):
        raise ValueError("Too few all-four-session cells pass the footprint-area gate")

    selected: list[dict[str, object]] = []
    remaining = candidates.copy()
    for anchor_x, anchor_y in anchors:
        chosen_source = min(
            remaining,
            key=lambda item: (
                (float(item["day1_x_fraction"]) - anchor_x) ** 2
                + (float(item["day1_y_fraction"]) - anchor_y) ** 2,
                int(item["common_row"]),
            ),
        )
        remaining.remove(chosen_source)
        chosen = dict(chosen_source)
        chosen["anchor_x_fraction"] = anchor_x
        chosen["anchor_y_fraction"] = anchor_y
        selected.append(chosen)

    crops: list[list[np.ndarray]] = [[] for _ in SESSION_SPECS]
    audit_rows: list[dict[str, object]] = []
    for example_index, chosen in enumerate(selected, start=1):
        common_row = int(chosen["common_row"])
        row = common.iloc[common_row]
        session_props = list(chosen["session_props"])
        for session_index, (
            stack,
            (session_key, display_title, map_column, footprint_name, _color),
            (y, x, area),
        ) in enumerate(zip(stacks, SESSION_SPECS, session_props, strict=True)):
            cell_id = int(row[map_column])
            image = _normalized_cell_image(stack, cell_id)
            crops[session_index].append(_centered_crop(image, y, x, crop_size))
            audit_rows.append(
                {
                    "example_index": example_index,
                    "selection_rule": "nearest unused all-four cell to fixed Day1 normalized-FOV anchor; minimum thresholded area 4 pixels in every session",
                    "common_map_row_zero_based": common_row,
                    "anchor_x_fraction": float(chosen["anchor_x_fraction"]),
                    "anchor_y_fraction": float(chosen["anchor_y_fraction"]),
                    "session_key": session_key,
                    "display_title": display_title,
                    "map_column": map_column,
                    "footprint_source": str((source_root / footprint_name).resolve()),
                    "cell_id_one_based": cell_id,
                    "centroid_x_px": x,
                    "centroid_y_px": y,
                    "thresholded_area_px": area,
                    "crop_size_px": crop_size,
                    "crop_centering": "each footprint centered on its own session-specific centroid",
                }
            )
    return crops, pd.DataFrame(audit_rows)


def render_matched_cell_montage(crops: list[list[np.ndarray]]) -> plt.Figure:
    n_rows = len(crops)
    n_columns = len(crops[0])
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(
            n_rows, n_columns, figsize=(5.5, 2.85), facecolor="white"
        )
        for row_index, (session_crops, session_spec) in enumerate(
            zip(crops, SESSION_SPECS, strict=True)
        ):
            for column_index, crop in enumerate(session_crops):
                ax = axes[row_index, column_index]
                rgb = np.zeros((*crop.shape, 3), dtype=np.float32)
                rgb[..., 1] = np.power(np.clip(crop, 0.0, 1.0), 0.75)
                ax.imshow(rgb, interpolation="nearest", origin="upper", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#B0B0B0")
                    spine.set_linewidth(0.35)
                if row_index == 0:
                    ax.set_title(
                        f"Example {column_index + 1}", pad=2.0, fontweight="normal"
                    )
                if column_index == 0:
                    ax.set_ylabel(
                        session_spec[1],
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=7.0,
                    )

        scale_um = 20.0
        scale_px = scale_um / 1.8
        last_ax = axes[-1, -1]
        x1 = crops[-1][-1].shape[1] - 3.5
        x0 = x1 - scale_px
        y = crops[-1][-1].shape[0] - 4.0
        last_ax.plot(
            [x0, x1], [y, y], color="white", linewidth=1.2, solid_capstyle="butt"
        )
        last_ax.text(
            (x0 + x1) / 2,
            y - 2.0,
            "20 μm",
            color="white",
            ha="center",
            va="bottom",
            fontsize=4.3,
        )
        fig.suptitle(
            "Centroid-centered footprints of deterministically sampled all-four-session cells",
            y=0.995,
            fontsize=6.2,
            fontweight="normal",
        )
        fig.subplots_adjust(
            left=0.095, right=0.995, top=0.89, bottom=0.02, wspace=0.08, hspace=0.08
        )
    return fig


def render_four_session_example(
    rgbs: list[np.ndarray], summary: pd.DataFrame
) -> plt.Figure:
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.72), facecolor="white")
        for ax, rgb, (_, row) in zip(axes, rgbs, summary.iterrows(), strict=True):
            ax.imshow(rgb, interpolation="nearest", origin="upper")
            ax.set_title(str(row["display_title"]), pad=2.0, fontweight="normal")
            ax.set_axis_off()

        # One calibrated bar is sufficient because all panels share 1.8 um/pixel.
        last_ax = axes[-1]
        scale_um = 100.0
        microns_per_pixel = 1.8
        scale_px = scale_um / microns_per_pixel
        image_h = int(summary.iloc[-1]["height_px"])
        image_w = int(summary.iloc[-1]["width_px"])
        x1 = image_w - 12.0
        x0 = x1 - scale_px
        y = image_h - 12.0
        last_ax.plot(
            [x0, x1], [y, y], color="white", linewidth=1.4, solid_capstyle="butt"
        )
        last_ax.text(
            (x0 + x1) / 2.0,
            y - 5.0,
            "100 μm",
            color="white",
            ha="center",
            va="bottom",
            fontsize=5.0,
        )

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
            left=0.004, right=0.996, top=0.88, bottom=0.13, wspace=0.025
        )
    return fig


def prepare_denominator_rows(
    pair_audit: Path, summary_json: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(pair_audit)
    required = {
        "animal",
        "pair_label",
        "transition",
        "n_registered_pairs",
        "corrected_n_pairs_dropped_by_rate",
        "corrected_n_pairs_rate_pass",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise KeyError(f"Pair audit is missing required columns: {missing}")
    pairs = pairs[
        [
            "animal",
            "pair_label",
            "transition",
            "n_registered_pairs",
            "corrected_n_pairs_dropped_by_rate",
            "corrected_n_pairs_rate_pass",
            "cimap_match_status",
            "n_registered_pair_ids_changed_by_coordinate_correction",
        ]
    ].copy()
    pairs["retained_fraction"] = (
        pairs["corrected_n_pairs_rate_pass"] / pairs["n_registered_pairs"]
    )
    pairs["transition_order"] = pairs["transition"].map(
        {key: idx for idx, key in enumerate(TRANSITION_COLORS)}
    )
    pairs = pairs.sort_values(
        ["transition_order", "n_registered_pairs", "pair_label"]
    ).reset_index(drop=True)

    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    total_registered = int(pairs["n_registered_pairs"].sum())
    total_dropped = int(pairs["corrected_n_pairs_dropped_by_rate"].sum())
    total_retained = int(pairs["corrected_n_pairs_rate_pass"].sum())
    assertions = {
        "n_session_pairs": int(len(pairs)),
        "n_animals": int(pairs["animal"].nunique()),
        "n_registered_pairs_before_activity_gate": total_registered,
        "n_dropped_by_activity_gate": total_dropped,
        "n_rate_qualified_matches": total_retained,
        "retained_fraction": total_retained / total_registered,
        "dropped_fraction": total_dropped / total_registered,
        "registered_equals_dropped_plus_retained": bool(
            total_registered == total_dropped + total_retained
        ),
        "matches_registration_summary_registered": bool(
            total_registered
            == int(saved_summary["n_registered_pairs_before_activity_gate"])
        ),
        "matches_registration_summary_retained": bool(
            total_retained
            == int(
                saved_summary[
                    "corrected_figure2_metric_observations_after_activity_gate"
                ]
            )
        ),
    }
    if not all(
        assertions[key]
        for key in (
            "registered_equals_dropped_plus_retained",
            "matches_registration_summary_registered",
            "matches_registration_summary_retained",
        )
    ):
        raise AssertionError(
            f"Registration denominator reconciliation failed: {assertions}"
        )
    summary = pd.DataFrame([assertions])
    return pairs, summary


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.65)
    ax.tick_params(width=0.65, length=2.2)


def render_denominator_panel(pairs: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    row = summary.iloc[0]
    with plt.rc_context(_paper_rc()):
        fig = plt.figure(figsize=(7.2, 2.55), facecolor="white")
        gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.35], wspace=0.31)
        flow_ax = fig.add_subplot(gs[0, 0])
        scatter_ax = fig.add_subplot(gs[0, 1])

        flow_ax.set_xlim(0, 1)
        flow_ax.set_ylim(0, 1)
        flow_ax.axis("off")
        box_specs = [
            (
                0.06,
                0.64,
                0.88,
                0.23,
                "CellReg-matched neuron pairs",
                int(row["n_registered_pairs_before_activity_gate"]),
                "#DCE8F5",
            ),
            (
                0.06,
                0.12,
                0.88,
                0.23,
                "Rate-qualified neuron pairs",
                int(row["n_rate_qualified_matches"]),
                "#DDEEDB",
            ),
        ]
        for x, y, width, height, label, count, color in box_specs:
            flow_ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    facecolor=color,
                    edgecolor="#4A4A4A",
                    linewidth=0.65,
                )
            )
            flow_ax.text(
                x + width / 2,
                y + 0.15,
                f"{count:,}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            flow_ax.text(
                x + width / 2, y + 0.055, label, ha="center", va="center", fontsize=5.6
            )
        flow_ax.add_patch(
            FancyArrowPatch(
                (0.50, 0.62),
                (0.50, 0.37),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#4A4A4A",
            )
        )
        flow_ax.text(
            0.54,
            0.495,
            f"Activity gate removes {int(row['n_dropped_by_activity_gate']):,}\n({100 * float(row['dropped_fraction']):.1f}%)",
            ha="left",
            va="center",
            fontsize=5.2,
        )
        flow_ax.text(
            0.5,
            0.94,
            "Corrected registration denominator",
            ha="center",
            va="center",
            fontsize=6.2,
        )
        flow_ax.text(
            0.5,
            0.015,
            f"{int(row['n_session_pairs'])} consecutive 3-month pairs; {int(row['n_animals'])} mice",
            ha="center",
            va="bottom",
            fontsize=5.2,
        )

        max_value = float(pairs["n_registered_pairs"].max()) * 1.06
        scatter_ax.plot(
            [0, max_value],
            [0, max_value],
            color="#B5B5B5",
            linewidth=0.7,
            linestyle="--",
            zorder=1,
        )
        for transition, color in TRANSITION_COLORS.items():
            subset = pairs.loc[pairs["transition"] == transition]
            scatter_ax.scatter(
                subset["n_registered_pairs"],
                subset["corrected_n_pairs_rate_pass"],
                s=16,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                label=DISPLAY_TRANSITION[transition],
                zorder=2,
            )
        scatter_ax.set_xlim(0, max_value)
        scatter_ax.set_ylim(0, max_value)
        scatter_ax.set_xlabel("Registered pairs before activity gate")
        scatter_ax.set_ylabel("Rate-qualified neuron pairs")
        scatter_ax.set_title("Every session pair", pad=3.0, fontweight="normal")
        scatter_ax.legend(
            loc="upper left", ncol=2, handletextpad=0.35, columnspacing=0.7
        )
        scatter_ax.text(
            0.98,
            0.04,
            f"Overall retained: {100 * float(row['retained_fraction']):.1f}%",
            transform=scatter_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
        )
        _style_axes(scatter_ax)
        fig.subplots_adjust(left=0.02, right=0.985, top=0.92, bottom=0.19)
    return fig


def render_all(*, source_root: Path, output_dir: Path, dpi: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rgbs, stacks, common_map, example_summary = prepare_four_session_example(
        source_root
    )
    common_path = output_dir / "ed05_four_session_418_common_cell_map.csv"
    example_summary_path = output_dir / "ed05_four_session_418_source_summary.csv"
    common_map.to_csv(common_path, index=False)
    example_summary.to_csv(example_summary_path, index=False)
    example_outputs = _save_bundle(
        render_four_session_example(rgbs, example_summary),
        output_dir / "ed05_four_session_418_coregistration_paper",
        dpi=dpi,
    )
    matched_crops, matched_cell_audit = prepare_matched_cell_montage(
        stacks, common_map, source_root
    )
    matched_cell_audit_path = (
        output_dir / "ed05_four_session_418_matched_cell_montage_source.csv"
    )
    matched_cell_audit.to_csv(matched_cell_audit_path, index=False)
    matched_cell_outputs = _save_bundle(
        render_matched_cell_montage(matched_crops),
        output_dir / "ed05_four_session_418_matched_cell_montage",
        dpi=dpi,
    )

    pair_rows, denominator_summary = prepare_denominator_rows(
        PAIR_AUDIT, REGISTRATION_SUMMARY
    )
    pair_rows_path = (
        output_dir / "ed05_registration_denominator_pair_rows_corrected36.csv"
    )
    denominator_summary_path = (
        output_dir / "ed05_registration_denominator_summary_corrected36.csv"
    )
    pair_rows.to_csv(pair_rows_path, index=False)
    denominator_summary.to_csv(denominator_summary_path, index=False)
    denominator_outputs = _save_bundle(
        render_denominator_panel(pair_rows, denominator_summary),
        output_dir / "ed05_registration_denominator_flow_corrected36",
        dpi=dpi,
    )

    png_paths = [
        REPO / example_outputs["png"],
        REPO / matched_cell_outputs["png"],
        REPO / denominator_outputs["png"],
    ]
    manifest = {
        "script": _rel(Path(__file__)),
        "figure": "Extended Data Figure 5",
        "backend": "python_matplotlib",
        "figure_contract": {
            "core_conclusion": (
                "Stable four-session fields of view and an explicit corrected registration-to-activity-gate "
                "denominator support defensible longitudinal neuron matching."
            ),
            "archetype": "image_plate_plus_quantitative_validation",
            "evidence_chain": [
                "animal 418 native-FOV footprint example across 0/3/6/9 months",
                "all-four-session CellReg membership",
                "deterministically sampled centroid-centered matched-cell footprints",
                "exact corrected36 registered-pair denominator",
                "per-session-pair activity-gate retention",
            ],
            "review_risks": [
                "poster composite provenance",
                "CellReg score versus P_same confusion",
                "21412 registered pairs versus 16901 rate-qualified matches",
                "illustrative single-cell montage being mistaken for independent accuracy validation",
                "long-gap direct registrations being confused with consecutive analysis registrations",
            ],
        },
        "four_session_example": {
            "animal": "418",
            "session_labels": example_summary["display_title"].tolist(),
            "n_registered_all_four": int(len(common_map)),
            "source_root": str(source_root.resolve()),
            "common_cimap_source": str(
                (
                    source_root
                    / "common_cimap_nearby_interpolated"
                    / "common_cimap_nearby_interpolated.mat"
                ).resolve()
            ),
            "common_cimap_sha256": _sha256(
                source_root
                / "common_cimap_nearby_interpolated"
                / "common_cimap_nearby_interpolated.mat"
            ),
            "source_summary": _rel(example_summary_path),
            "common_cell_map": _rel(common_path),
            "outputs": example_outputs,
            "image_integrity": {
                "projection": "per-cell footprint normalized to its maximum and thresholded at 0.5",
                "pseudo_color": "green for neurons registered in all four sessions; white for other detected cells",
                "alignment": "native FOV per session; no image warping in the displayed four-panel row",
                "scale": "100 μm bar using 1.8 μm per pixel; shown once because all panels share calibration",
                "local_selective_adjustment": False,
            },
        },
        "matched_cell_montage": {
            "animal": "418",
            "n_example_cells": int(matched_cell_audit["example_index"].nunique()),
            "session_labels": example_summary["display_title"].tolist(),
            "selection": (
                "nearest unused all-four-session cell to each of six fixed normalized-FOV anchors in Day 1, "
                "after requiring at least four footprint pixels at >=0.5 of the cell maximum in every session"
            ),
            "display_alignment": (
                "each raw footprint is centered on its own session-specific centroid; this montage shows "
                "footprint shape continuity and does not display global FOV displacement"
            ),
            "statistical_role": "deterministic illustrative example, not an independent registration-accuracy test",
            "source_table": _rel(matched_cell_audit_path),
            "outputs": matched_cell_outputs,
            "image_integrity": {
                "crop": "31 x 31 pixels centered on each session-specific footprint centroid",
                "pseudo_color": "green intensity from each independently normalized native footprint",
                "scale": "20 μm bar using 1.8 μm per pixel; inset from the lower-right crop boundary",
                "local_selective_adjustment": False,
            },
        },
        "denominator_flow": {
            "scope": "final sorted Figure 2 corrected36, excluding the short 1616 transition",
            "pair_audit_source": _rel(PAIR_AUDIT),
            "pair_audit_sha256": _sha256(PAIR_AUDIT),
            "registration_summary_source": _rel(REGISTRATION_SUMMARY),
            "registration_summary_sha256": _sha256(REGISTRATION_SUMMARY),
            "pair_rows": _rel(pair_rows_path),
            "summary": _rel(denominator_summary_path),
            "outputs": denominator_outputs,
            "statistical_role": "descriptive denominator and inclusion audit; no inferential p-value",
        },
        "style": "Arial 6 pt base, 6.2 pt titles, 5.4 pt ticks, 0.65 pt axes; editable SVG/PDF",
        "png_validation": [_image_audit(path) for path in png_paths],
    }
    if not all(bool(row["nonblank"]) for row in manifest["png_validation"]):
        raise RuntimeError("At least one ED5 output is blank")
    manifest_path = output_dir / "ed05_registration_flow_and_four_session_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    manifest = render_all(
        source_root=args.source_root, output_dir=args.output_dir, dpi=int(args.dpi)
    )
    print(
        json.dumps(
            {
                "output_dir": _rel(args.output_dir),
                "n_registered_all_four": manifest["four_session_example"][
                    "n_registered_all_four"
                ],
                "denominator_summary": manifest["denominator_flow"]["summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
