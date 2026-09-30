"""Render merged Extended Data Figure 10 from the three slide-12 source roles.

The quantitative inputs are frozen plot-ready tables. This script changes only
the display: it follows the supplied PowerPoint hierarchy and keeps caption-level
methods, cohort details, and statistics off the panels.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from ca_analysis.figure_colors import get_correlation_measure_colors

from _ed10_neural_eligibility import audit_pair_universe, load_neural_eligibility


REPO = Path(__file__).resolve().parents[1]
SUPPLEMENT = REPO / "docs" / "extended_data_supplement_v2_rebuild_2026-08-05"
ROOT = SUPPLEMENT / "ed10_reward_level_behavior_bridge"
SYNTHESIS = (
    REPO
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_tables"
    / "extended_data_fig_11"
    / "fixed_neuron_correlation_synthesis_2026-08-09"
)
TABLE_A = SYNTHESIS / "POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv"
TABLE_B = SYNTHESIS / "REWARD_LEVEL_COMMONALITY.csv"
PROFILE_TABLE = SYNTHESIS / "REWARD_LEVEL_PROFILE_TESTS.csv"
SYNTHESIS_REPORT = SYNTHESIS / "REPORT.md"
SYNTHESIS_CAPTION = SYNTHESIS / "ED10_CAPTION_AND_SOURCE_GUIDE.md"
PAIR_AUDIT = SYNTHESIS / "CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
SYNTHESIS_VALIDATION = SYNTHESIS / "VALIDATION.json"
SYNTHESIS_MANIFEST = SYNTHESIS / "SOURCE_MANIFEST.csv"
NEURAL_ELIGIBILITY_CONTRACT = SYNTHESIS / "NEURAL_ELIGIBILITY_CONTRACT.json"
ROUND6_SOURCE_VALIDATION = ROOT / "ED10_ROUND6_SOURCE_VALIDATION.json"
TABLE_C = (
    REPO
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_tables"
    / "extended_data_fig_11"
    / "level_policy_bridge_join_corrected36.csv"
)
STATS_C = TABLE_C.with_name("level_policy_bridge_stats_corrected36.csv")
EXCLUSION_CONTRACT = SUPPLEMENT / "EXCLUDED_SESSION_CONTRACT.json"
MAIN_FIGURE5_RECORD = (
    REPO
    / "docs"
    / "figure_version_records"
    / "main_figure5_2026-08-11_round2_normalized_rate"
)
MAIN_FIGURE5_MANIFEST = MAIN_FIGURE5_RECORD / "figure5_source_manifest.json"
MAIN_FIGURE5_SVG = MAIN_FIGURE5_RECORD / "figure5_version_record.svg"
MAIN_FIGURE5_RENDERER = (
    MAIN_FIGURE5_RECORD / "source_code" / "preview_figure3_assembly.py"
)

COLORS = get_correlation_measure_colors()
REWARD = COLORS["pv_correlation"]
PLACE = COLORS["place_map_correlation"]
LEVEL = COLORS["level_map_correlation"]
PV = COLORS["pv_correlation"]
RATE = COLORS["ensemble_rate_correlation"]
SHARED = "#7F7F7F"
BOOTSTRAP_SEED = 20260809
ROUND = "round8"
ROLE_ASPECTS = {"a": 685.44 / 261.62, "b": 685.44 / 167.97, "c": 685.44 / 202.21}
ROLE_HEIGHTS_720 = {"a": 261.62, "b": 167.97, "c": 202.21}
ROUND4_INTERROW_GAPS_720 = {"a_b": 17.52, "b_c": 20.83}
ROUND5_INTERROW_GAPS_720 = {
    key: value / 2 for key, value in ROUND4_INTERROW_GAPS_720.items()
}
TARGET_INTERROW_GAPS_720 = {
    "a_b": ROUND5_INTERROW_GAPS_720["a_b"] / 2,
    "b_c": ROUND5_INTERROW_GAPS_720["b_c"],
}
CONTENT_HEIGHT_720 = (
    sum(ROLE_HEIGHTS_720.values()) + sum(TARGET_INTERROW_GAPS_720.values())
)
CONTENT_MARGIN_720 = (720 - CONTENT_HEIGHT_720) / 2
ROLE_TOPS_720 = {
    "a": CONTENT_MARGIN_720,
    "b": CONTENT_MARGIN_720
    + ROLE_HEIGHTS_720["a"]
    + TARGET_INTERROW_GAPS_720["a_b"],
    "c": CONTENT_MARGIN_720
    + ROLE_HEIGHTS_720["a"]
    + TARGET_INTERROW_GAPS_720["a_b"]
    + ROLE_HEIGHTS_720["b"]
    + TARGET_INTERROW_GAPS_720["b_c"],
}
SLIDE_ROLE_RECTS = {
    role: [
        17.28 / 720,
        (720 - ROLE_TOPS_720[role] - ROLE_HEIGHTS_720[role]) / 720,
        685.44 / 720,
        ROLE_HEIGHTS_720[role] / 720,
    ]
    for role in ("a", "b", "c")
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def resolve_manifest_source(path: Path, expected_sha256: str, expected_bytes: int) -> Path:
    """Resolve an immutable-record path after administrative record archival."""

    candidates = [path]
    parts = list(path.parts)
    record_indices = [
        index
        for index, part in enumerate(parts)
        if part in {"figure_version_records", "figure_version_records_archive_2026-08-09"}
    ]
    if record_indices:
        index = record_indices[-1]
        tail = Path(*parts[index + 2 :])
        for root in (
            REPO / "docs" / "figure_version_records",
            REPO / "docs" / "figure_version_records_archive_2026-08-09",
        ):
            candidates.extend(root.glob(f"*/{tail.as_posix()}"))
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size == expected_bytes
            and sha256(candidate) == expected_sha256
        ):
            return candidate
    raise FileNotFoundError(path)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
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


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=3.5)


RAW_DECODER_COLUMNS = {
    "reward_location": "reward_error_fixed2000",
    "whole_decoder": "whole_error_fixed2000",
}
RAW_DECODER_LABELS = {
    "reward_location": "Reward decoding error",
    "whole_decoder": "Whole maze decoding error",
}
DISPLAY_OUTCOME_COLUMNS = {
    "whole_session_reward_approach_speed_advantage": "log_reward_over_pooled",
    "first_long_bout_time": "first_long_bout_time_s",
    "exploration_efficiency": "exploration_efficiency_raw",
}


def raw_decoder_error_display_table(
    maintained: pd.DataFrame,
    pair_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Flip only decoder-strength rows after proving them against raw errors."""

    display = maintained.copy()
    display["outcome_label"] = display["outcome_label"].replace(
        {"Reward-approach speed advantage": "Reward approach speed advantage"}
    )
    rows: list[dict[str, object]] = []
    for block, decoder_column in RAW_DECODER_COLUMNS.items():
        decoder = pd.to_numeric(pair_rows[decoder_column], errors="raise").to_numpy(float)
        for outcome, behavior_column in DISPLAY_OUTCOME_COLUMNS.items():
            behavior = pd.to_numeric(pair_rows[behavior_column], errors="raise").to_numpy(float)
            raw_r = float(np.corrcoef(decoder, behavior)[0, 1])
            mask = display["block"].eq(block) & display["outcome"].eq(outcome)
            if int(mask.sum()) != 1:
                raise AssertionError(f"Expected one ED10 matrix row for {block}/{outcome}")
            maintained_r = float(display.loc[mask, "pooled_pearson_r"].iloc[0])
            if not np.isclose(raw_r, -maintained_r, atol=1e-12, rtol=0.0):
                raise AssertionError(
                    f"Raw-error sign audit failed for {block}/{outcome}: "
                    f"raw={raw_r}, maintained={maintained_r}"
                )
            display.loc[mask, "pooled_pearson_r"] = raw_r
            display.loc[mask, "block_label"] = RAW_DECODER_LABELS[block]
            display.loc[mask, "display_role"] = (
                "raw_decoder_error_descriptive_no_cellwise_inference"
            )
            rows.append(
                {
                    "block": block,
                    "outcome": outcome,
                    "decoder_column": decoder_column,
                    "behavior_column": behavior_column,
                    "maintained_negative_error_r": maintained_r,
                    "display_raw_error_r": raw_r,
                    "exact_sign_reverse": bool(
                        np.isclose(raw_r, -maintained_r, atol=1e-12, rtol=0.0)
                    ),
                }
            )

    unchanged_columns = [column for column in maintained.columns if column != "outcome_label"]
    unchanged = maintained.loc[
        ~maintained["block"].isin(RAW_DECODER_COLUMNS), unchanged_columns
    ].reset_index(drop=True)
    unchanged_display = display.loc[
        ~display["block"].isin(RAW_DECODER_COLUMNS), unchanged_columns
    ].reset_index(drop=True)
    if not unchanged.equals(unchanged_display):
        raise AssertionError("A non-decoder ED10 matrix row changed")
    if display["outcome_label"].astype(str).str.contains("Reward-approach", regex=False).any():
        raise AssertionError("A dashed reward-approach label survived in the display table")
    audit = {
        "schema": "ed10-raw-decoder-error-display-sign-audit/v1",
        "status": "PASS",
        "cohort": {"n_pairs": int(len(pair_rows)), "n_animals": int(pair_rows["animal_id"].nunique())},
        "higher_decoder_value_means": "poorer decoding",
        "changed_cells": rows,
        "checks": {
            "six_decoder_cells_are_exact_sign_reversals": all(
                bool(row["exact_sign_reverse"]) for row in rows
            ) and len(rows) == 6,
            "other_twelve_cells_exact": True,
            "display_outcome_labels_use_open_compounds": True,
            "cohort_30_pairs_14_mice": len(pair_rows) == 30
            and pair_rows["animal_id"].nunique() == 14,
        },
    }
    return display, audit


def heatmap_panel(ax: plt.Axes, cax: plt.Axes, table: pd.DataFrame) -> None:
    row_order = [
        "reward_location",
        "whole_decoder",
        "place_map",
        "structure",
        "pv_map",
        "rate",
    ]
    col_order = [
        "whole_session_reward_approach_speed_advantage",
        "first_long_bout_time",
        "exploration_efficiency",
    ]
    matrix = (
        table.pivot(index="block", columns="outcome", values="pooled_pearson_r")
        .loc[row_order, col_order]
        .to_numpy(float)
    )
    norm = mpl.colors.Normalize(vmin=-0.8, vmax=0.8)
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_xticks(
        range(3),
        ["Reward approach\nspeed advantage", "First long\nbout time", "Exploration\nefficiency"],
    )
    ax.set_yticks(
        range(6),
        ["Reward decoding error", "Whole maze decoding error", "Place map correlation", "Level map correlation", "Population vector correlation", "Ensemble rate correlation"],
    )
    for label in ax.get_yticklabels():
        label.set_color("black")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="white" if abs(value) >= 0.55 else "black",
                fontweight="bold" if abs(value) >= 0.55 else "normal",
            )
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Draw the continuous key as vector cells so the SVG/PDF contains no
    # embedded raster payloads.
    edges = np.linspace(-0.8, 0.8, 129)
    centers = ((edges[:-1] + edges[1:]) / 2)[:, None]
    cax.pcolormesh(
        [0, 1],
        edges,
        centers,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    cax.set_xlim(0, 1)
    cax.set_ylim(-0.8, 0.8)
    cax.set_xticks([])
    cax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Pearson correlation")


def commonality_panel(axes: list[plt.Axes], table: pd.DataFrame) -> None:
    pooled = table.loc[table["specification"].eq("pooled")].copy()
    outcomes = ["earlier_first_long", "exploration_efficiency"]
    titles = ["First long bout time", "Exploration efficiency"]
    labels = ["Reward decoder unique", "Shared", "Level map unique"]
    for ax, outcome, title in zip(axes, outcomes, titles):
        row = pooled.loc[pooled["outcome"].eq(outcome)]
        if len(row) != 1:
            raise AssertionError(f"Expected one pooled commonality row for {outcome}")
        row = row.iloc[0]
        values = 100 * np.array([row["unique_a"], row["shared"], row["unique_b"]], float)
        left = 0.0
        for value, color in zip(values, [REWARD, SHARED, LEVEL]):
            ax.barh([0], [value], left=left, color=color, edgecolor="white", linewidth=0.5, height=0.52)
            if value >= 6:
                ax.text(
                    left + value / 2,
                    0,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white" if color in (REWARD, LEVEL) else "black",
                    fontweight="bold",
                )
            left += value
        ax.set_xlim(0, 60)
        ax.set_yticks([])
        ax.set_title(title, fontweight="bold", pad=2)
        ax.set_xlabel("Explained variance (%)")
        clean_axis(ax)
    handles = [mpl.patches.Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip([REWARD, SHARED, LEVEL], labels)]
    axes[0].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.34), ncol=3, handlelength=1.0, columnspacing=0.8)


def cluster_bootstrap_band(
    frame: pd.DataFrame, x: str, y: str, xfit: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    use = frame[[x, y, "animal_id"]].replace([np.inf, -np.inf], np.nan).dropna()
    xv = use[x].to_numpy(float)
    yv = use[y].to_numpy(float)
    animal = use["animal_id"].astype(str).to_numpy()
    mice = np.asarray(sorted(set(animal)))
    rng = np.random.default_rng(seed)
    lines: list[np.ndarray] = []
    for _ in range(5_000):
        sampled = rng.choice(mice, size=len(mice), replace=True)
        indices = np.concatenate([np.flatnonzero(animal == mouse) for mouse in sampled])
        if np.unique(xv[indices]).size < 2:
            continue
        slope, intercept = np.polyfit(xv[indices], yv[indices], 1)
        lines.append(intercept + slope * xfit)
    array = np.asarray(lines)
    return np.quantile(array, 0.025, axis=0), np.quantile(array, 0.975, axis=0)


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value) or q_value >= 0.05:
        return "n.s."
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    return "*"


def _point_to_bbox_clearance_px(
    points_px: np.ndarray, bbox: mpl.transforms.Bbox, marker_radius_px: float
) -> float:
    dx = np.maximum.reduce(
        [bbox.x0 - points_px[:, 0], np.zeros(len(points_px)), points_px[:, 0] - bbox.x1]
    )
    dy = np.maximum.reduce(
        [bbox.y0 - points_px[:, 1], np.zeros(len(points_px)), points_px[:, 1] - bbox.y1]
    )
    return float(np.min(np.hypot(dx, dy)) - marker_radius_px)


def add_collision_aware_r_label(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, pearson_r: float, column: str
) -> mpl.text.Text:
    candidates = [(0.94, 0.08), (0.94, 0.16), (0.82, 0.08), (0.82, 0.16), (0.94, 0.24)]
    points_px = ax.transData.transform(np.column_stack([x, y]))
    marker_radius_px = (np.sqrt(10.0) / 2.0) * ax.figure.dpi / 72.0
    best: tuple[float, mpl.text.Text] | None = None
    for x_axes, y_axes in candidates:
        label = ax.text(
            x_axes,
            y_axes,
            f"r={pearson_r:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="black",
            zorder=5,
        )
        label.set_gid(f"policy_pearson_r_{column}")
        ax.figure.canvas.draw()
        bbox = label.get_window_extent(ax.figure.canvas.get_renderer())
        clearance = _point_to_bbox_clearance_px(points_px, bbox, marker_radius_px)
        if best is None or clearance > best[0]:
            if best is not None:
                best[1].remove()
            best = (clearance, label)
        else:
            label.remove()
        if clearance >= 2.0:
            break
    if best is None or best[0] < 2.0:
        raise AssertionError(
            f"Could not place lower-right r label clear of scatter points for {column}: "
            f"best clearance={None if best is None else best[0]:.3f}px"
        )
    best[1]._ed10_point_clearance_px = best[0]
    return best[1]


def policy_scatter_panel(
    axes: list[plt.Axes], frame: pd.DataFrame, stats_frame: pd.DataFrame
) -> None:
    specs = [
        ("Lf", r"$\mathrm{P}_{\mathrm{BF}}$"),
        ("Bf", r"$\mathrm{P}_{\mathrm{SF}}$"),
        ("Ba", r"$\mathrm{P}_{\mathrm{SA}}$"),
        ("Lo_depth_weighted", r"Depth-weighted $\mathrm{P}_{\mathrm{BS}}$"),
    ]
    for index, (ax, (column, label)) in enumerate(zip(axes, specs)):
        use = frame[["level_mean_all", column, "animal_id"]].dropna().copy()
        if len(use) != 36 or use["animal_id"].astype(str).nunique() != 14:
            raise AssertionError(f"Unexpected ED10 policy support for {column}")
        x = use["level_mean_all"].to_numpy(float)
        y = use[column].to_numpy(float)
        xfit = np.linspace(float(x.min()), float(x.max()), 100)
        lo, hi = cluster_bootstrap_band(use, "level_mean_all", column, xfit, seed=BOOTSTRAP_SEED + index)
        slope, intercept = np.polyfit(x, y, 1)
        ax.fill_between(xfit, lo, hi, color=LEVEL, alpha=0.16, linewidth=0)
        ax.scatter(x, y, s=10, color=LEVEL, alpha=0.78, edgecolor="white", linewidth=0.25)
        ax.plot(xfit, intercept + slope * xfit, color=LEVEL, linewidth=0.9)
        ax.set_xlim(-0.05, 0.55)
        ax.set_xticks([0.0, 0.2, 0.4])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel(label)
        clean_axis(ax)
        stat = stats_frame.loc[stats_frame["y"].eq(column)]
        if len(stat) != 1:
            raise AssertionError(f"Expected one ED10 policy-stat row for {column}")
        stat = stat.iloc[0]
        q_value = float(stat["cluster_q_bh_window_family"])
        if not np.isfinite(q_value):
            q_value = float(stat["cluster_p"])
        annotation = significance_stars(q_value)
        if annotation != "n.s.":
            bottom, top = ax.get_ylim()
            span = top - bottom
            ax.set_ylim(bottom, max(top, float(np.max(y)) + 0.20 * span))
        star = ax.text(
            0.08,
            0.92,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12.0,
            fontweight="normal" if annotation == "n.s." else "bold",
            color=LEVEL,
            zorder=5,
        )
        star.set_gid(f"policy_significance_{column}")
        annotation_clearance = float("-inf")
        for _ in range(6):
            ax.figure.canvas.draw()
            annotation_bbox = star.get_window_extent(ax.figure.canvas.get_renderer())
            annotation_clearance = _point_to_bbox_clearance_px(
                ax.transData.transform(np.column_stack([x, y])),
                annotation_bbox,
                (np.sqrt(10.0) / 2.0) * ax.figure.dpi / 72.0,
            )
            if annotation_clearance >= 2.0:
                break
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top + 0.05 * (top - bottom))
        if annotation_clearance < 2.0:
            raise AssertionError(
                f"Main-figure-style annotation overlaps scatter points for {column}: "
                f"clearance={annotation_clearance:.3f}px"
            )
        star._ed10_point_clearance_px = annotation_clearance
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        add_collision_aware_r_label(
            ax, x, y, float(stat["pearson_r"]), column
        )


def save_panel(fig: plt.Figure, stem: Path, *, exact_canvas: bool = False) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(
            path,
            facecolor="white",
            bbox_inches=None if exact_canvas else "tight",
            **kwargs,
        )
        if suffix in {"png", "tiff"}:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                save_kwargs: dict[str, object] = {"dpi": (600, 600)}
                if suffix == "tiff":
                    save_kwargs["compression"] = "tiff_lzw"
                rgb.save(path, **save_kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _rect(role: list[float], local: list[float]) -> list[float]:
    return [
        role[0] + local[0] * role[2],
        role[1] + local[1] * role[3],
        local[2] * role[2],
        local[3] * role[3],
    ]


def add_role_a(fig: plt.Figure, role: list[float], table: pd.DataFrame) -> list[plt.Axes]:
    heat = fig.add_axes(_rect(role, [0.205, 0.20, 0.690, 0.69]))
    cax = fig.add_axes(_rect(role, [0.910, 0.20, 0.018, 0.69]))
    heatmap_panel(heat, cax, table)
    fig.text(
        role[0] + 0.006 * role[2],
        role[1] + 0.972 * role[3],
        "a",
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    return [heat, cax]


def add_role_b(fig: plt.Figure, role: list[float], table: pd.DataFrame) -> list[plt.Axes]:
    axes = [
        fig.add_axes(_rect(role, [0.070, 0.23, 0.405, 0.54])),
        fig.add_axes(_rect(role, [0.555, 0.23, 0.405, 0.54])),
    ]
    commonality_panel(axes, table)
    fig.text(
        role[0] + 0.006 * role[2],
        role[1] + 0.972 * role[3],
        "b",
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    return axes


def add_role_c(
    fig: plt.Figure,
    role: list[float],
    table: pd.DataFrame,
    stats_table: pd.DataFrame,
) -> list[plt.Axes]:
    left, right, gap = 0.090, 0.985, 0.080
    width = (right - left - 3 * gap) / 4
    axes = [
        fig.add_axes(_rect(role, [left + i * (width + gap), 0.245, width, 0.640]))
        for i in range(4)
    ]
    policy_scatter_panel(axes, table, stats_table)
    for ax in axes:
        ax.set_xlabel("")
    fig.text(
        role[0] + 0.53 * role[2],
        role[1] + 0.060 * role[3],
        "Level map correlation",
        fontsize=8.0,
        ha="center",
        va="bottom",
    )
    fig.text(
        role[0] + 0.006 * role[2],
        role[1] + 0.972 * role[3],
        "c",
        fontsize=13.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    return axes


def rendered_role_qa(
    fig: plt.Figure, axes: list[plt.Axes], role: str
) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_artists = [
        artist
        for artist in fig.findobj(match=mpl.text.Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    text_boxes = [artist.get_window_extent(renderer) for artist in text_artists]
    collisions: list[dict[str, str]] = []
    for left_index, left in enumerate(text_boxes):
        for right_index in range(left_index + 1, len(text_boxes)):
            right = text_boxes[right_index]
            overlap_width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
            overlap_height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
            if overlap_width * overlap_height > 4.0:
                collisions.append(
                    {
                        "left": text_artists[left_index].get_text(),
                        "right": text_artists[right_index].get_text(),
                    }
                )
    axes_bounds = [axis.get_position().bounds for axis in axes]
    data_axes = axes if role != "a" else axes[:1]
    data_bounds = [axis.get_position().bounds for axis in data_axes]
    panel_a_y_label_weights: dict[str, str] = {}
    if role == "a":
        panel_a_y_label_weights = {
            label.get_text(): str(label.get_fontweight())
            for label in axes[0].get_yticklabels()
            if label.get_text().strip()
        }
    same_row_alignment = True
    if role in {"b", "c"}:
        bottoms = [bounds[1] for bounds in data_bounds]
        heights = [bounds[3] for bounds in data_bounds]
        same_row_alignment = max(bottoms) - min(bottoms) <= 1e-9 and max(
            heights
        ) - min(heights) <= 1e-9
    neighbor_axis_clearance = True
    neighbor_axis_clearance_px: list[float] = []
    r_label_point_clearance_px: list[float] = []
    r_label_axes_positions: list[list[float]] = []
    star_contract: list[dict[str, object]] = []
    annotation_point_clearance_px: list[float] = []
    if role == "c":
        axis_boxes = [axis.get_window_extent(renderer) for axis in axes]
        for index in range(1, len(axes)):
            labels = [axes[index].yaxis.label, *axes[index].get_yticklabels()]
            label_boxes = [
                label.get_window_extent(renderer)
                for label in labels
                if label.get_visible() and label.get_text().strip()
            ]
            clearance = min(box.x0 for box in label_boxes) - axis_boxes[index - 1].x1
            neighbor_axis_clearance_px.append(float(clearance))
            neighbor_axis_clearance &= clearance >= 2.0
        for axis in axes:
            r_labels = [
                text
                for text in axis.texts
                if str(text.get_gid() or "").startswith("policy_pearson_r_")
            ]
            if len(r_labels) != 1:
                r_label_point_clearance_px.append(float("-inf"))
                r_label_axes_positions.append([float("nan"), float("nan")])
            else:
                label = r_labels[0]
                r_label_point_clearance_px.append(
                    float(getattr(label, "_ed10_point_clearance_px", float("-inf")))
                )
                r_label_axes_positions.append([float(value) for value in label.get_position()])
            for text in axis.texts:
                if str(text.get_gid() or "").startswith("policy_significance_"):
                    point_clearance = float(
                        getattr(text, "_ed10_point_clearance_px", float("-inf"))
                    )
                    annotation_point_clearance_px.append(point_clearance)
                    star_contract.append(
                        {
                            "text": text.get_text(),
                            "position": [float(value) for value in text.get_position()],
                            "fontsize": float(text.get_fontsize()),
                            "fontweight": str(text.get_fontweight()),
                            "color": str(mpl.colors.to_hex(text.get_color())).upper(),
                            "horizontalalignment": text.get_horizontalalignment(),
                            "verticalalignment": text.get_verticalalignment(),
                            "point_clearance_px": point_clearance,
                        }
                    )
    return {
        "all_text_inside_canvas": all(
            box.x0 >= canvas.x0 - 1
            and box.y0 >= canvas.y0 - 1
            and box.x1 <= canvas.x1 + 1
            and box.y1 <= canvas.y1 + 1
            for box in text_boxes
        ),
        "no_text_text_collision": not collisions,
        "text_collisions": collisions,
        "same_row_axes_aligned": bool(same_row_alignment),
        "neighbor_axis_text_clearance": bool(neighbor_axis_clearance),
        "neighbor_axis_text_clearance_px": neighbor_axis_clearance_px,
        "r_label_point_clearance_px": r_label_point_clearance_px,
        "r_label_axes_positions": r_label_axes_positions,
        "r_labels_clear_of_points": role != "c"
        or (
            len(r_label_point_clearance_px) == 4
            and min(r_label_point_clearance_px) >= 2.0
            and all(x >= 0.70 and y <= 0.30 for x, y in r_label_axes_positions)
        ),
        "annotation_point_clearance_px": annotation_point_clearance_px,
        "annotations_clear_of_points": role != "c"
        or (
            len(annotation_point_clearance_px) == 4
            and min(annotation_point_clearance_px) >= 2.0
        ),
        "main_figure5_star_contract": star_contract,
        "panel_a_y_label_weights": panel_a_y_label_weights,
        "axes_bounds": [list(bounds) for bounds in axes_bounds],
        "n_text_artists": len(text_artists),
    }


def render_individuals(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    stats_c: pd.DataFrame,
    out: Path,
) -> tuple[list[Path], dict[str, dict[str, object]]]:
    outputs: list[Path] = []
    qa: dict[str, dict[str, object]] = {}
    for role, table, builder, stem in (
        ("a", a, add_role_a, "a_correlation_matrix"),
        ("b", b, add_role_b, "b_variation_partition"),
        ("c", c, add_role_c, "c_level_policy_associations"),
    ):
        width = 7.15
        fig = plt.figure(figsize=(width, width / ROLE_ASPECTS[role]))
        axes = (
            builder(fig, [0, 0, 1, 1], table, stats_c)
            if role == "c"
            else builder(fig, [0, 0, 1, 1], table)
        )
        qa[role] = rendered_role_qa(fig, axes, role)
        outputs += save_panel(
            fig, out / "panels" / stem, exact_canvas=True
        )
    return outputs, qa


def render_assembly(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    stats_c: pd.DataFrame,
    stem: Path,
) -> tuple[list[Path], dict[str, dict[str, object]]]:
    fig = plt.figure(figsize=(7.5, 7.5))
    axes = {
        "a": add_role_a(fig, SLIDE_ROLE_RECTS["a"], a),
        "b": add_role_b(fig, SLIDE_ROLE_RECTS["b"], b),
        "c": add_role_c(fig, SLIDE_ROLE_RECTS["c"], c, stats_c),
    }
    qa = {role: rendered_role_qa(fig, role_axes, role) for role, role_axes in axes.items()}
    outputs = save_panel(fig, stem, exact_canvas=True)
    return outputs, qa


def write_plain_language_docs(
    a: pd.DataFrame,
    b: pd.DataFrame,
    stats_c: pd.DataFrame,
    profile: pd.DataFrame,
    *,
    output_root: Path | None = None,
) -> list[Path]:
    document_root = output_root.resolve() if output_root is not None else ROOT
    pooled = profile.loc[
        profile["claim"].eq("reward_level_profile_crossover")
        & profile["specification"].eq("pooled")
        & profile["weighting"].eq("pair_weighted")
    ].iloc[0]
    stage = profile.loc[
        profile["claim"].eq("reward_level_profile_crossover")
        & profile["specification"].eq("stage_adjusted")
        & profile["weighting"].eq("pair_weighted")
    ].iloc[0]
    pooled_b = b.loc[b["specification"].eq("pooled")].set_index("outcome")
    early = pooled_b.loc["earlier_first_long"]
    explore = pooled_b.loc["exploration_efficiency"]
    whole_decoder = a.loc[a["block"].eq("whole_decoder")].set_index("outcome")
    depth_policy = stats_c.loc[stats_c["y"].eq("Lo_depth_weighted")].iloc[0]
    report = f"""# Extended Data Figure 10: resolved result and final display

## Plain-language answer

The result did not collapse. The weaker-looking draft silently changed what was
shown: it replaced the PowerPoint's pooled Pearson correlations with adjusted
regression coefficients, changed the speed endpoint from the whole session to
the first 30 minutes, removed the population vector row, and omitted the
policy panel. The current figure restores the original three-part hierarchy
while keeping the stricter analyses in Source Data. Round 8 retains all five
PowerPoint readouts and adds whole maze decoding error as a visible general decoder
quality control; no inherited row is replaced.

Panel a now displays raw decoder error and the current no-manual-mask K = 166
map-stability values against reward approach speed advantage, elapsed first
long bout time, and exploration efficiency. Higher decoder error means poorer
decoding. These are descriptive correlations, not eighteen independent
significance claims. Whole maze decoding error correlates
{whole_decoder.loc['whole_session_reward_approach_speed_advantage', 'pooled_pearson_r']:.3f},
{whole_decoder.loc['first_long_bout_time', 'pooled_pearson_r']:.3f}, and
{whole_decoder.loc['exploration_efficiency', 'pooled_pearson_r']:.3f} with the
three displayed behaviors, respectively. It uses the same denoised-C K = 166
draws and exactly 2,000 target frames as reward decoding. It is a 15 x 15
whole maze position decoder, not an all-frame or occupancy-balanced decoder.

Panel b shows the pooled reward decoder/level map variance partition. For first
long bout time, explained variance is {100*early.unique_a:.1f}% reward decoder unique,
{100*early.shared:.1f}% shared and {100*early.unique_b:.1f}% level map unique.
For exploration it is {100*explore.unique_a:.1f}%, {100*explore.shared:.1f}% and
{100*explore.unique_b:.1f}%, respectively. Explained-variance partitions are
unchanged when decoder error or elapsed bout time is multiplied by -1. Here Reward decoder
unique means reward decoder unique relative to the level map; it
does not mean unique relative to whole maze decoding.

Panel c retains the separate clean36 level-policy bridge. The depth-weighted
P_BS association is positive (r = {depth_policy.pearson_r:.3f}; full policy-by-
window q = {depth_policy.cluster_q_bh_window_family:.3g}; ***), whereas the other three displayed policy
associations are nonsignificant controls and are labeled `n.s.`. This family is supporting evidence from
a different cohort, not an independent replication of panels a/b. Each scatter
now reports its descriptive Pearson r at the lower right; `n.s.` and the significant star
use the promoted main-figure upper-left purple grammar, with normal weight for `n.s.`.

## Data and exclusion resolution

- Panels a/b use 30 session pairs from 14 mice and average 50 technical K = 166
  draws within pair.
- Decoder and raw-map calculations use the same selected neuron tuples.
- The live neural-eligibility contract is applied before QC. Animals 840 and
  841 are behavior-only; 488-9, 579-6 and every other locked neural-excluded
  session are absent. Earlier valid 488 and 579 intervals remain only when they
  do not touch those excluded sessions.
- The six-month Day1/Day2 session pairs for animals 1636, 1637, 1639, 325
  and 347 are absent because they belong to a different experiment.
- The first-30-min matched-terminal speed result is retained as a labeled
  sensitivity; it is not substituted into panel a.

## Claim boundary

The defensible conclusion is that stronger Day-1 spatial map quality covaries
with more efficient later navigation across return stages. The data do not
establish a unique reward code, mediation, causality, an independent
replication of Main Fig. 4f, or a stage-independent double dissociation.

## Figure and PowerPoint QA contract

The final source figure preserves the PowerPoint's three-band hierarchy and adds
whole maze decoding error as a prespecified general decoder quality control directly
below reward decoding, yielding a six-by-three matrix. Two commonality bars and
four aligned policy axes are retained. The a-b gap is exactly half its round-5
value while the b-c gap, row heights and source-role aspects are unchanged. The three roles are separate
zero-crop vector/raster assets. Panel methods, cohort counts and inferential
statistics are kept in the caption rather than printed over the plots, except
for the requested `n.s.`/`***` annotations and descriptive Pearson correlations in panel c.

Detailed statistical and conversion audit: `{SYNTHESIS_REPORT.as_posix()}`.
"""
    caption = f"""# Extended Data Figure 10 caption notes

**Extended Data Fig. 10 | Spatial code stability and later navigation.**
**a,** Pair-level Pearson correlations between six Day-1-source neural
readouts and three target-session behaviors for 30 session pairs from 14 mice.
The original five rows are retained and whole maze decoding error is added immediately
below reward decoding error as a general decoder quality control. Both decoder rows
use denoised C calcium activity, the same K = 166 neuron draws and exactly 2,000
target frames per pair; whole maze decoding is not an all-frame or occupancy-
balanced estimator. Values are raw decoder errors, so higher values indicate
poorer decoding. The columns show reward approach speed advantage, elapsed
first long bout time, and exploration efficiency; lower bout time denotes
earlier onset. Neural
readouts use the current exclusion-clean, no-manual-mask K = 166 authority;
technical draws are averaged within session pair. Cells are descriptive and no
cellwise significance threshold was applied. The first-30-min matched-terminal
endpoint is reported as a sensitivity in Source Data. **b,** Pooled commonality
partition for reward-proximal decoding and level map stability. Explained-
variance partitions are unchanged by using elapsed first-long-bout time rather
than its sign-reversed earlier-oriented form. In panel b, Reward decoder unique
means reward decoder unique relative to the level map,
not relative to whole maze decoding. **c,** Level map correlation versus four fixed 30-min turning-policy measures
in the separate clean36 cohort (36 pairs, 14 mice). Points are pairs; lines are
ordinary least-squares fits and bands are 95% mouse-cluster bootstrap intervals.
The depth-weighted P_BS association was supported after the maintained policy-
by-window family correction (q = {depth_policy.cluster_q_bh_window_family:.3g}; ***); the other three measures are nonsignificant controls and are labeled `n.s.`. Each scatter displays descriptive Pearson r at lower right. Association
does not imply mediation or causality.

Animals 840 and 841 and all neural-excluded sessions, including 488-9 and the
sick-animal 579-6 session, were removed before neural QC. The six-month
Day1/Day2 session pairs for animals 1636, 1637, 1639, 325 and 347 are absent
because they belong to a different experiment.

[Sources]
{SYNTHESIS_VALIDATION.as_posix()}
{TABLE_A.as_posix()}
{TABLE_B.as_posix()}
{TABLE_C.as_posix()}
{STATS_C.as_posix()}
{EXCLUSION_CONTRACT.as_posix()}
{NEURAL_ELIGIBILITY_CONTRACT.as_posix()}
{ROUND6_SOURCE_VALIDATION.as_posix()}
{SYNTHESIS_CAPTION.as_posix()}
[/Sources]
"""
    report_path = document_root / "AUDIT_REPORT.md"
    caption_path = document_root / "ED10_CAPTION_NOTES.md"
    report_path.write_text(report, encoding="utf-8")
    caption_path.write_text(caption, encoding="utf-8")
    return [report_path, caption_path]


def main(*, output_root: Path | None = None, output_round: str | None = None) -> None:
    configure_style()
    required = [
        TABLE_A,
        TABLE_B,
        PROFILE_TABLE,
        SYNTHESIS_REPORT,
        SYNTHESIS_CAPTION,
        TABLE_C,
        STATS_C,
        EXCLUSION_CONTRACT,
        PAIR_AUDIT,
        SYNTHESIS_VALIDATION,
        SYNTHESIS_MANIFEST,
        MAIN_FIGURE5_MANIFEST,
        MAIN_FIGURE5_SVG,
        MAIN_FIGURE5_RENDERER,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    eligibility = load_neural_eligibility(REPO)
    for authority_path in (
        eligibility["neural_path"],
        eligibility["biological_path"],
        eligibility["ed08_path"],
    ):
        if authority_path not in required:
            required.append(authority_path)
    exclusion = json.loads(EXCLUSION_CONTRACT.read_text(encoding="utf-8"))
    forbidden = sorted(eligibility["excluded_sessions"])
    synthesis_validation = json.loads(
        SYNTHESIS_VALIDATION.read_text(encoding="utf-8")
    )
    if synthesis_validation.get("status") != "PASS":
        raise AssertionError("Current ED10 synthesis validation is not PASS")
    if not all(
        Path(path).exists() and sha256(Path(path)) == str(digest)
        for path, digest in synthesis_validation.get("outputs", {}).items()
    ):
        raise AssertionError("A current ED10 synthesis output hash is stale")
    source_manifest = pd.read_csv(SYNTHESIS_MANIFEST)
    for row in source_manifest.itertuples(index=False):
        path = Path(str(row.path))
        try:
            resolve_manifest_source(path, str(row.sha256), int(row.bytes))
        except FileNotFoundError as exc:
            raise AssertionError(
                f"Stale ED10 synthesis source manifest row: {path}"
            ) from exc
    a = pd.read_csv(TABLE_A)
    b = pd.read_csv(TABLE_B)
    profile = pd.read_csv(PROFILE_TABLE)
    c = pd.read_csv(TABLE_C, dtype={"animal_id": str})
    stats_c = pd.read_csv(STATS_C)
    pair_audit = pd.read_csv(PAIR_AUDIT, dtype=str)
    if len(a) != 18 or int(a["n_pairs"].unique()[0]) != 30 or int(a["n_animals"].unique()[0]) != 14:
        raise AssertionError("ED10 correlation matrix is not the frozen 30-pair/14-mouse source")
    expected_blocks = [
        "reward_location",
        "whole_decoder",
        "place_map",
        "structure",
        "pv_map",
        "rate",
    ]
    observed_blocks = list(dict.fromkeys(a["block"].astype(str)))
    if observed_blocks != expected_blocks:
        raise AssertionError(f"ED10 matrix row order mismatch: {observed_blocks}")
    if len(pair_audit) != 30 or pair_audit["pair_key"].nunique() != 30:
        raise AssertionError("ED10 30-pair membership audit is incomplete")
    a, decoder_sign_audit = raw_decoder_error_display_table(a, pair_audit)
    if len(c) != 36 or c["pair_key"].nunique() != 36 or c["animal_id"].nunique() != 14:
        raise AssertionError("ED10 policy source is not clean36/14")
    if len(stats_c) != 4:
        raise AssertionError("ED10 policy statistics source must contain four displayed roles")
    main5_manifest = json.loads(MAIN_FIGURE5_MANIFEST.read_text(encoding="utf-8"))
    main5_style = main5_manifest["style"]
    main5_svg = MAIN_FIGURE5_SVG.read_text(encoding="utf-8")
    main5_renderer = MAIN_FIGURE5_RENDERER.read_text(encoding="utf-8")
    if not (
        float(main5_style["base_font_size"]) == 8.0
        and float(main5_style["axis_label_size"]) == 8.0
        and float(main5_style["tick_label_size"]) == 7.0
        and float(main5_style["title_size"]) == 8.0
        and int(main5_style["panel_id_size"]) == 13
        and "r=0.75" in main5_svg
        and ">***</text>" in main5_svg
        and "0.08," in main5_renderer
        and "0.92," in main5_renderer
        and "0.94," in main5_renderer
        and "fontsize=7.2 * style_scale" in main5_renderer
    ):
        raise AssertionError("Promoted Main Figure 5 typography/annotation authority drifted")
    pair_audit["source_session"] = pair_audit["pair_key"].astype(str).str.split("->").str[0]
    pair_audit["target_session"] = pair_audit["pair_key"].astype(str).str.split("->").str[1]
    ab_eligibility = audit_pair_universe(
        pair_audit,
        eligibility,
        pair_column="pair_key",
        animal_column="animal_id",
    )
    c_eligibility = audit_pair_universe(
        c,
        eligibility,
        pair_column="pair_key",
        animal_column="animal_id",
    )
    if not ab_eligibility["eligibility_decision"].eq("USE_NEURAL").all():
        raise AssertionError("ED10 a/b contains a neural-ineligible row")
    if not c_eligibility["eligibility_decision"].eq("USE_NEURAL").all():
        raise AssertionError("ED10 c contains a neural-ineligible row")
    membership_text = "\n".join(
        pair_audit[["pair_key", "source_session", "target_session"]].fillna("").astype(str).agg("|".join, axis=1).tolist()
        + c[["pair_key", "src", "dst"]].fillna("").astype(str).agg("|".join, axis=1).tolist()
    )
    present_forbidden = sorted(token for token in forbidden if token in membership_text)
    if present_forbidden:
        raise AssertionError(f"Forbidden ED10 sessions found: {present_forbidden}")

    render_root = output_root.resolve() if output_root is not None else ROOT
    render_round = output_round or ROUND
    output_dir = render_root / "assembly" / render_round
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = render_root / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    membership = pd.concat(
        [
            pair_audit[["pair_key", "source_session", "target_session"]].assign(role="a_b_30pair"),
            c[["pair_key", "src", "dst"]].rename(columns={"src": "source_session", "dst": "target_session"}).assign(role="c_clean36"),
        ],
        ignore_index=True,
    )
    membership.to_csv(source_data / "ED10_COHORT_MEMBERSHIP.csv", index=False)
    raw_decoder_table_path = source_data / "ED10_RAW_DECODER_ERROR_CORRELATION_MATRIX.csv"
    decoder_sign_audit_path = source_data / "ED10_RAW_DECODER_ERROR_SIGN_AUDIT.json"
    a.to_csv(raw_decoder_table_path, index=False)
    decoder_sign_audit_path.write_text(
        json.dumps(decoder_sign_audit, indent=2) + "\n", encoding="utf-8"
    )

    outputs = [raw_decoder_table_path, decoder_sign_audit_path]
    outputs += write_plain_language_docs(
        a, b, stats_c, profile, output_root=render_root
    )
    individual_outputs, role_qa = render_individuals(a, b, c, stats_c, output_dir)
    outputs += individual_outputs
    assembly_stem = output_dir / f"extended_data_figure10_source_rerender_{render_round}"
    assembly_outputs, assembly_qa = render_assembly(a, b, c, stats_c, assembly_stem)
    outputs += assembly_outputs
    root_outputs = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        root_path = render_root / f"extended_data_figure10_source_rerender_{render_round}.{suffix}"
        shutil.copyfile(assembly_stem.with_suffix(f".{suffix}"), root_path)
        root_outputs.append(root_path)
    outputs += root_outputs
    root_png = render_root / f"extended_data_figure10_source_rerender_{render_round}.png"

    with Image.open(root_png) as image:
        if (
            image.size != (4500, 4500)
            or image.mode != "RGB"
            or float(np.std(np.asarray(image.convert("L")))) < 3
        ):
            raise AssertionError(f"ED10 root PNG failed raster QA: {image.size}")
    svg_text = assembly_stem.with_suffix(".svg").read_text(encoding="utf-8")
    forbidden_panel_text = ("cluster t", "window-family", "n =", "pair p", "not supported", "supported")
    if any(token in svg_text for token in forbidden_panel_text):
        raise AssertionError("Caption-level ED10 text survived in the assembly")
    if "<image" in svg_text:
        raise AssertionError("ED10 assembly SVG contains an embedded raster payload")
    expected_annotations = ["n.s.", "n.s.", "n.s.", "***"]
    for scope, audit in (("role", role_qa["c"]), ("assembly", assembly_qa["c"])):
        observed_annotations = [
            row["text"] for row in audit["main_figure5_star_contract"]
        ]
        if observed_annotations != expected_annotations:
            raise AssertionError(
                f"ED10 panel-c {scope} annotation contract drifted: "
                f"{observed_annotations}"
            )
    if "Pearson correlation" not in svg_text or "Pooled Pearson r" in svg_text:
        raise AssertionError("ED10 panel-a colorbar title is not literal")
    forbidden_open_compound_hyphens = (
        "Reward-approach",
        "Whole-maze",
        "Place-map",
        "Level-map",
        "Population-vector",
        "Ensemble-rate",
        "Reward-decoder",
    )
    if any(term in svg_text for term in forbidden_open_compound_hyphens):
        raise AssertionError("ED10 contains a prohibited dashed display term")
    for scope, audit in (("role", role_qa["a"]), ("assembly", assembly_qa["a"])):
        weights = audit["panel_a_y_label_weights"]
        if len(weights) != 6 or any(weight != "normal" for weight in weights.values()):
            raise AssertionError(
                f"ED10 panel-a {scope} neural-label weights are not uniformly normal: "
                f"{weights}"
            )
    if not all(
        bool(role_qa[role]["all_text_inside_canvas"])
        and bool(role_qa[role]["no_text_text_collision"])
        and bool(role_qa[role]["same_row_axes_aligned"])
        and bool(role_qa[role]["neighbor_axis_text_clearance"])
        and bool(role_qa[role]["r_labels_clear_of_points"])
        and bool(role_qa[role]["annotations_clear_of_points"])
        for role in ("a", "b", "c")
    ):
        raise AssertionError(f"ED10 role overlap/clipping QA failed: {role_qa}")
    if not all(
        bool(assembly_qa[role]["all_text_inside_canvas"])
        and bool(assembly_qa[role]["no_text_text_collision"])
        and bool(assembly_qa[role]["same_row_axes_aligned"])
        and bool(assembly_qa[role]["neighbor_axis_text_clearance"])
        and bool(assembly_qa[role]["r_labels_clear_of_points"])
        and bool(assembly_qa[role]["annotations_clear_of_points"])
        for role in ("a", "b", "c")
    ):
        raise AssertionError(f"ED10 assembly overlap/clipping QA failed: {assembly_qa}")

    validation = {
        "schema": "ed10_slide12_powerpoint_matched_validation/v7",
        "status": "PASS",
        "source_slide": 12,
        "panel_inventory": [
            "a correlation matrix",
            "b reward-level variance partition",
            "c four level-policy associations",
        ],
        "layout_contract": "top matrix / middle two bars / bottom four scatters",
        "caption_only_statistics": True,
        "forbidden_sessions": forbidden,
        "forbidden_sessions_present": present_forbidden,
        "checks": {
            "a_scope_30_pairs_14_mice": True,
            "b_scope_30_pairs_14_mice": True,
            "c_scope_36_pairs_14_mice": True,
            "live_24_session_2_animal_neural_eligibility_applied_before_qc": True,
            "no_forbidden_session_in_membership": True,
            "maintained_source_tables_not_recomputed": True,
            "panel_a_raw_decoder_error_sign_audit_pass": all(
                decoder_sign_audit["checks"].values()
            ),
            "panel_a_other_twelve_cells_exact": True,
            "open_compound_display_terms": True,
            "root_png_4500_square": True,
            "caption_level_statistics_absent_from_panels": True,
            "assembly_svg_has_no_embedded_raster": True,
            "individual_role_text_contained_and_nonoverlapping": True,
            "assembly_text_contained_and_nonoverlapping": True,
            "same_row_axes_aligned": True,
            "synthesis_nested_source_and_output_hashes_live": True,
            "whole_session_powerpoint_endpoint_restored": True,
            "whole_maze_decoder_control_added_without_replacing_powerpoint_rows": True,
            "panel_a_exact_6x3_inventory": True,
            "panel_a_neural_measure_labels_all_black": True,
            "panel_a_neural_measure_labels_all_normal_weight": True,
            "panel_a_uses_elapsed_first_long_bout_time": True,
            "panel_b_reward_decoder_unique_label_and_elapsed_time_title": True,
            "panel_c_true_policy_subscripts_and_two_decimal_ticks": True,
            "panel_c_corrected_depth_weighted_q_shown_as_three_stars": True,
            "panel_c_first_three_controls_labeled_nonsignificant": True,
            "panel_a_colorbar_title_is_pearson_correlation": True,
            "main_figure5_typography_contract_applied": True,
            "panel_c_main_figure5_star_contract_applied": True,
            "panel_c_four_lower_right_r_labels_clear_of_points": True,
            "panel_c_four_annotations_clear_of_points": True,
            "a_b_gap_exactly_half_round5": True,
            "b_c_gap_unchanged_from_round5": True,
            "row_heights_and_aspects_unchanged": True,
        },
        "interrow_geometry_720": {
            "round4_gaps": ROUND4_INTERROW_GAPS_720,
            "round5_gaps": ROUND5_INTERROW_GAPS_720,
            "round8_gaps": TARGET_INTERROW_GAPS_720,
            "round8_tops": ROLE_TOPS_720,
            "round8_outer_margin": CONTENT_MARGIN_720,
            "gap_ratios": {
                key: TARGET_INTERROW_GAPS_720[key] / ROUND5_INTERROW_GAPS_720[key]
                for key in ROUND5_INTERROW_GAPS_720
            },
        },
        "main_figure5_annotation_authority": {
            "manifest": record(MAIN_FIGURE5_MANIFEST, "display_authority"),
            "svg": record(MAIN_FIGURE5_SVG, "display_authority"),
            "renderer": record(MAIN_FIGURE5_RENDERER, "display_authority"),
            "font_contract": {
                "base": 8.0,
                "axis": 8.0,
                "ticks": 7.0,
                "title": 8.0,
                "panel_id": 13.0,
            },
            "annotation": {
                "position": [0.08, 0.92],
                "fontsize": 12.0,
                "color": LEVEL,
                "significant_fontweight": "bold",
                "nonsignificant_fontweight": "normal",
            },
            "r": {"preferred_position": [0.94, 0.08], "fontsize": 7.2, "color": "black"},
        },
        "role_geometry": {
            role: {"slide_rect_720pt": [value * 720 for value in rect], "aspect": ROLE_ASPECTS[role]}
            for role, rect in SLIDE_ROLE_RECTS.items()
        },
        "role_qa": role_qa,
        "assembly_qa": assembly_qa,
        "neural_eligibility": {
            "path": str(eligibility["neural_path"]),
            "sha256": eligibility["neural_sha256"],
            "excluded_animals": sorted(eligibility["excluded_animals"]),
            "excluded_sessions": sorted(eligibility["excluded_sessions"]),
        },
        "inputs": [record(path, "input") for path in required],
        "outputs": [record(path, "output") for path in outputs],
        "renderer": record(Path(__file__).resolve(), "renderer"),
    }
    validation_path = render_root / "AUDIT_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "validation": validation_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
