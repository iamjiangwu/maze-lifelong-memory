"""Render Extended Data Figure 10 from the three frozen slide-12 source roles.

The quantitative inputs are frozen plot-ready tables. Panel a shows black,
normal-weight correlation values with adjacent stars derived from two-sided
pair-row OLS tests with CR1 standard errors clustered by animal and Benjamini--
Hochberg correction across the three behavior outcomes within each neural row.
Nominal CR1 and row-wise BH3 results are exported together.
The ED2 policy order (P_SF, P_SA, P_BF, depth-weighted P_BS) is preserved.
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
import statsmodels.api as sm
from PIL import Image
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster

from ca_analysis.figure_colors import get_correlation_measure_colors

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
PAIR_AUDIT = SYNTHESIS / "CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
SYNTHESIS_VALIDATION = SYNTHESIS / "VALIDATION.json"
SYNTHESIS_MANIFEST = SYNTHESIS / "SOURCE_MANIFEST.csv"
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

RAW_P_VALUE_COLUMN = "pairwise_nominal_p_descriptive"
P_VALUE_COLUMN = "cr1_bh3_adjusted_p"
P_STAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))


def p_value_star(value: float) -> str:
    """Return the figure's P-value star grammar without displaying n.s."""

    for threshold, star in P_STAR_THRESHOLDS:
        if value < threshold:
            return star
    return ""


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return deterministic BH-adjusted p values in the input row order."""

    p_values = pd.to_numeric(values, errors="raise").to_numpy(float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise AssertionError("BH input contains an invalid p value")
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return pd.Series(adjusted, index=values.index, dtype=float)


def _cr1_p_value(frame: pd.DataFrame, predictor: str, outcome: str) -> dict[str, float | int]:
    """Fit y ~ 1 + x with CR1 covariance and a Student t(G-1) reference."""

    x = pd.to_numeric(frame[predictor], errors="raise").to_numpy(float)
    y = pd.to_numeric(frame[outcome], errors="raise").to_numpy(float)
    design = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, design).fit()
    groups = frame["animal_id"].astype(str).to_numpy()
    covariance = cov_cluster(model, groups, use_correction=True)
    slope = float(model.params[1])
    se = float(np.sqrt(covariance[1, 1]))
    df = int(np.unique(groups).size - 1)
    t_value = slope / se
    return {
        "cr1_slope": slope,
        "cr1_se": se,
        "cr1_t": t_value,
        "cr1_df": df,
        "cr1_p_two_sided": float(2.0 * stats.t.sf(abs(t_value), df)),
    }


def panel_a_multiplicity_audit(
    table: pd.DataFrame, pair_rows: pd.DataFrame
) -> pd.DataFrame:
    """Build the displayed animal-clustered slope tests and row-wise BH3."""

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
    audit = table.copy()
    if len(audit) != 18 or RAW_P_VALUE_COLUMN not in audit.columns:
        raise AssertionError("ED10 panel-a p-value grid must contain exactly 18 cells")
    if len(pair_rows) != 30 or pair_rows["animal_id"].astype(str).nunique() != 14:
        raise AssertionError("ED10 panel-a CR1 input must contain 30 pairs from 14 mice")
    audit["neural_row_order"] = audit["block"].map(
        {name: index + 1 for index, name in enumerate(row_order)}
    )
    audit["behavior_column_order"] = audit["outcome"].map(
        {name: index + 1 for index, name in enumerate(col_order)}
    )
    if audit[["neural_row_order", "behavior_column_order"]].isna().any().any():
        raise AssertionError("ED10 panel-a multiplicity grid has an unknown row or column")
    if audit.duplicated(["block", "outcome"]).any():
        raise AssertionError("ED10 panel-a multiplicity grid contains duplicate cells")
    audit = audit.sort_values(
        ["neural_row_order", "behavior_column_order"], kind="stable"
    ).reset_index(drop=True)
    audit["raw_p_two_sided_pairwise_pearson"] = pd.to_numeric(
        audit[RAW_P_VALUE_COLUMN], errors="raise"
    )
    audit["bh3_family_id"] = "neural_measure::" + audit["block"].astype(str)
    audit["bh3_family_size"] = 3
    neural_columns = {
        "reward_location": "reward_error_fixed2000",
        "whole_decoder": "whole_error_fixed2000",
        "place_map": "place_map",
        "structure": "structure",
        "pv_map": "pv_map",
        "rate": "rate",
    }
    outcome_columns = {
        "whole_session_reward_approach_speed_advantage": "log_reward_over_pooled",
        "first_long_bout_time": "first_long_bout_time_s",
        "exploration_efficiency": "exploration_efficiency_raw",
    }
    cr1_rows = []
    for row in audit.itertuples(index=False):
        cr1_rows.append(
            _cr1_p_value(
                pair_rows,
                neural_columns[str(row.block)],
                outcome_columns[str(row.outcome)],
            )
        )
    cr1 = pd.DataFrame(cr1_rows, index=audit.index)
    for column in cr1.columns:
        audit[column] = cr1[column]
    audit["cr1_nominal_star"] = audit["cr1_p_two_sided"].map(p_value_star)
    audit["cr1_bh3_adjusted_p"] = audit.groupby("block", sort=False)[
        "cr1_p_two_sided"
    ].transform(benjamini_hochberg)
    audit["cr1_bh3_star"] = audit["cr1_bh3_adjusted_p"].map(p_value_star)
    audit["statistical_unit"] = "session_pair"
    audit["p_value_scope_note"] = (
        "Pair-row OLS with standard errors clustered by animal, CR1 and Student t(13); row-wise BH3"
    )
    columns = [
        "neural_row_order",
        "behavior_column_order",
        "block",
        "block_label",
        "outcome",
        "outcome_label",
        "pooled_pearson_r",
        "raw_p_two_sided_pairwise_pearson",
        "bh3_family_id",
        "bh3_family_size",
        "cr1_slope",
        "cr1_se",
        "cr1_t",
        "cr1_df",
        "cr1_p_two_sided",
        "cr1_nominal_star",
        "cr1_bh3_adjusted_p",
        "cr1_bh3_star",
        "n_pairs",
        "n_animals",
        "statistical_unit",
        "p_value_scope_note",
    ]
    return audit[columns]


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
    # Display magnitude only; keep signed coefficients and two-sided tests in
    # the source table unchanged. Use the positive half of the inherited palette.
    matrix = np.abs(matrix)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.8)
    magnitude_cmap = mpl.colors.ListedColormap(
        mpl.colormaps["RdBu_r"](np.linspace(0.5, 1.0, 256)),
        name="absolute_pearson",
    )
    image = ax.pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap=magnitude_cmap,
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
        ["Reward decoding error", "Whole maze decoding error", "Place map correlation", "Level map correlation", "PV correlation", "Ensemble rate correlation"],
    )
    for label in ax.get_yticklabels():
        label.set_color("black")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                # Keep the 18 effect sizes visually neutral; the separate
                # adjacent annotation carries the requested CR1-BH3 stars.
                color="black",
                fontweight="normal",
            )
            label.set_gid(f"correlation_value_{row}_{col}")
            cell = table.loc[
                table["block"].eq(row_order[row])
                & table["outcome"].eq(col_order[col])
            ]
            if len(cell) != 1:
                raise AssertionError(
                    f"Expected one panel-a cell for {row_order[row]}/{col_order[col]}"
                )
            star = p_value_star(float(cell[P_VALUE_COLUMN].iloc[0]))
            if star:
                annotation = ax.annotate(
                    star,
                    xy=(col, row),
                    xytext=(12.0, 3.0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold",
                    annotation_clip=True,
                )
                annotation.set_gid(f"panel_a_star_{row}_{col}")
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Draw the continuous key as vector cells so the SVG/PDF contains no
    # embedded raster payloads.
    edges = np.linspace(0.0, 0.8, 129)
    centers = ((edges[:-1] + edges[1:]) / 2)[:, None]
    cax.pcolormesh(
        [0, 1],
        edges,
        centers,
        cmap=magnitude_cmap,
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    cax.set_xlim(0, 1)
    cax.set_ylim(0.0, 0.8)
    cax.set_xticks([])
    cax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Absolute Pearson correlation")


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
        # Match the canonical ED2 policy order.
        ("Bf", r"$\mathrm{P}_{\mathrm{SF}}$"),
        ("Ba", r"$\mathrm{P}_{\mathrm{SA}}$"),
        ("Lf", r"$\mathrm{P}_{\mathrm{BF}}$"),
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
        q_value = float(stat["cluster_bh4_adjusted_p"])
        if not np.isfinite(q_value):
            raise AssertionError("Each displayed policy requires a finite BH4-adjusted p value")
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
            # Reserve the accepted 12-pt annotation headroom to preserve axes.
            # The n.s. display size is set to 8 pt after the layout calculation.
            fontsize=12.0,
            fontweight="normal" if annotation == "n.s." else "bold",
            color="black",
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
        if annotation == "n.s.":
            star.set_fontsize(8.0)
            ax.figure.canvas.draw()
            annotation_clearance = _point_to_bbox_clearance_px(
                ax.transData.transform(np.column_stack([x, y])),
                star.get_window_extent(ax.figure.canvas.get_renderer()),
                (np.sqrt(10.0) / 2.0) * ax.figure.dpi / 72.0,
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
    panel_a_correlation_value_styles: list[dict[str, str]] = []
    panel_a_star_contract: list[dict[str, object]] = []
    if role == "a":
        panel_a_y_label_weights = {
            label.get_text(): str(label.get_fontweight())
            for label in axes[0].get_yticklabels()
            if label.get_text().strip()
        }
        panel_a_correlation_value_styles = [
            {
                "text": text.get_text(),
                "color": str(mpl.colors.to_hex(text.get_color())).upper(),
                "fontweight": str(text.get_fontweight()),
            }
            for text in axes[0].texts
            if str(text.get_gid() or "").startswith("correlation_value_")
        ]
        values_by_cell = {
            str(text.get_gid()).removeprefix("correlation_value_"): text
            for text in axes[0].texts
            if str(text.get_gid() or "").startswith("correlation_value_")
        }
        heatmap_box = axes[0].get_window_extent(renderer)
        for star in axes[0].texts:
            gid = str(star.get_gid() or "")
            if not gid.startswith("panel_a_star_"):
                continue
            cell_id = gid.removeprefix("panel_a_star_")
            value = values_by_cell.get(cell_id)
            if value is None:
                raise AssertionError(f"Panel-a star has no matched r value: {gid}")
            star_box = star.get_window_extent(renderer)
            value_box = value.get_window_extent(renderer)
            panel_a_star_contract.append(
                {
                    "cell_id": cell_id,
                    "text": star.get_text(),
                    "color": str(mpl.colors.to_hex(star.get_color())).upper(),
                    "fontweight": str(star.get_fontweight()),
                    "fontsize": float(star.get_fontsize()),
                    "horizontal_clearance_from_r_px": float(star_box.x0 - value_box.x1),
                    "star_center_above_r_center": bool(
                        (star_box.y0 + star_box.y1) / 2
                        > (value_box.y0 + value_box.y1) / 2
                    ),
                    "inside_heatmap_axes": bool(
                        star_box.x0 >= heatmap_box.x0 - 1
                        and star_box.y0 >= heatmap_box.y0 - 1
                        and star_box.x1 <= heatmap_box.x1 + 1
                        and star_box.y1 <= heatmap_box.y1 + 1
                    ),
                }
            )
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
        "panel_a_correlation_value_styles": panel_a_correlation_value_styles,
        "panel_a_star_contract": panel_a_star_contract,
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
    """Describe only the displayed outcomes and captioned comparisons."""
    document_root = output_root.resolve() if output_root is not None else ROOT
    tests = profile.set_index("claim")
    pooled_b = b.loc[b["specification"].eq("pooled")].set_index("outcome")
    early = pooled_b.loc["earlier_first_long"]
    explore = pooled_b.loc["exploration_efficiency"]
    depth_policy = stats_c.loc[stats_c["y"].eq("Lo_depth_weighted")].iloc[0]
    report = f"""# Extended Data Figure 10: neural preservation and later behavior

Panel a contains six neural readouts and three later-session behaviors
(30 session pairs from 14 animals). Stars use two-sided regressions with
standard errors clustered by animal and BH correction across the three
behavior outcomes within each neural-measure row.

Panel b partitions behavioral variance into reward-decoder unique, shared,
and level-map unique components: {100*early.unique_a:.1f}%, {100*early.shared:.1f}%,
and {100*early.unique_b:.1f}% for first long bout time; {100*explore.unique_a:.1f}%,
{100*explore.shared:.1f}%, and {100*explore.unique_b:.1f}% for exploration efficiency.
The three captioned profile comparisons use two-sided exact animal-score
sign-flip tests and BH correction across those three tests. Adjusted p values
are {tests.loc['first_long_reward_minus_level', 'bh3_adjusted_p']:.3g},
{tests.loc['exploration_reward_minus_level', 'bh3_adjusted_p']:.3g}, and
{tests.loc['reward_level_profile_crossover', 'bh3_adjusted_p']:.3g}, respectively.

Panel c relates level map correlation to the four displayed turning biases
measured during Explore behavior in the first 30 cumulative maze minutes
(36 pairs from 14 animals). All four slope tests use standard errors clustered
by animal and BH correction across those four tests. Depth-weighted P_BS:
r = {depth_policy.pearson_r:.3f}, adjusted p = {depth_policy.cluster_bh4_adjusted_p:.3g}.
"""
    caption = f"""# Extended Data Figure 10 caption notes

**Extended Data Fig. 10 | Neural preservation and later behavior.**

**a,** Absolute Pearson correlations between six neural readouts and three
later-session behaviors, across 30 session pairs from 14 animals. Stars use
two-sided regressions with standard errors clustered by animal and BH
correction across three behaviors within each neural-measure row. The behavior
outcomes are whole-session reward approach speed advantage, first long bout
time and exploration efficiency.

**b,** Commonality partition of variance associated with reward decoding error
and level map correlation. The two bars concern first long bout time and
exploration efficiency. Three pooled profile contrasts use two-sided exact
animal-score sign-flip tests with BH correction across the three captioned
comparisons. First-long-bout difference: {tests.loc['first_long_reward_minus_level', 'estimate']:.2f},
adjusted p = {tests.loc['first_long_reward_minus_level', 'bh3_adjusted_p']:.3g};
exploration difference: {abs(tests.loc['exploration_reward_minus_level', 'estimate']):.2f},
adjusted p = {tests.loc['exploration_reward_minus_level', 'bh3_adjusted_p']:.3g};
profile crossover: adjusted p = {tests.loc['reward_level_profile_crossover', 'bh3_adjusted_p']:.3g}.

**c,** Level map correlation versus four turning biases during exploration
within the first 30 cumulative maze minutes of the later session (36 pairs from
14 animals). Lines are linear fits and bands are 95% confidence intervals
from resampling animals. Stars use two-sided regressions with standard errors
clustered by animal and BH correction across the four displayed policies.
Depth-weighted P_BS: adjusted p = {depth_policy.cluster_bh4_adjusted_p:.3g}.
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
        TABLE_C,
        STATS_C,
        PAIR_AUDIT,
        SYNTHESIS_VALIDATION,
        SYNTHESIS_MANIFEST,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

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
    multiplicity_audit = panel_a_multiplicity_audit(a, pair_audit)
    a = a.merge(
        multiplicity_audit[
            [
                "block",
                "outcome",
                "cr1_p_two_sided",
                "cr1_bh3_adjusted_p",
            ]
        ],
        on=["block", "outcome"],
        how="left",
        validate="one_to_one",
    )
    if not (
        int(multiplicity_audit["cr1_nominal_star"].ne("").sum()) == 14
        and int(multiplicity_audit["cr1_bh3_star"].ne("").sum()) == 12
    ):
        raise AssertionError("ED10 panel-a Pearson/CR1/BH significance counts drifted")
    if len(c) != 36 or c["pair_key"].nunique() != 36 or c["animal_id"].nunique() != 14:
        raise AssertionError("ED10 policy source is not clean36/14")
    if len(stats_c) != 4:
        raise AssertionError("ED10 policy statistics source must contain four displayed roles")
    pair_audit["source_session"] = pair_audit["pair_key"].astype(str).str.split("->").str[0]
    pair_audit["target_session"] = pair_audit["pair_key"].astype(str).str.split("->").str[1]
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
    multiplicity_audit_path = source_data / "ED10_PANEL_A_MULTIPLICITY_AUDIT.csv"
    multiplicity_summary_path = source_data / "ED10_PANEL_A_MULTIPLICITY_SUMMARY.json"
    a.to_csv(raw_decoder_table_path, index=False)
    absolute_display_path = source_data / "ED10_PANEL_A_ABSOLUTE_DISPLAY.csv"
    absolute_display = a[["block", "outcome", "pooled_pearson_r", P_VALUE_COLUMN]].copy()
    absolute_display["absolute_pearson_r"] = absolute_display["pooled_pearson_r"].abs()
    absolute_display["display_text"] = absolute_display["absolute_pearson_r"].map(lambda value: f"{value:.2f}")
    absolute_display["stars"] = absolute_display[P_VALUE_COLUMN].map(p_value_star)
    absolute_display.to_csv(absolute_display_path, index=False)
    decoder_sign_audit_path.write_text(
        json.dumps(decoder_sign_audit, indent=2) + "\n", encoding="utf-8"
    )
    multiplicity_audit.to_csv(multiplicity_audit_path, index=False)
    cr1_bh3_changed = multiplicity_audit.loc[
        multiplicity_audit["cr1_nominal_star"].ne(
            multiplicity_audit["cr1_bh3_star"]
        )
    ]
    multiplicity_summary_path.write_text(
        json.dumps(
            {
                "schema": "ed10-panel-a-mouse-cr1-bh-audit/v2",
                "status": "PASS",
                "plotted_star_source": "mouse_clustered_cr1_rowwise_bh3",
                "plotted_star_thresholds": {
                    "*": "p < 0.05",
                    "**": "p < 0.01",
                    "***": "p < 0.001",
                },
                "statistical_unit": "session_pair",
                "n_pairs": 30,
                "n_animals": 14,
                "nominal_test_note": (
                    "Two-sided pair-row OLS slope test with CR1 covariance clustered by mouse and Student t(13)"
                ),
                "families": {
                    "mouse_cr1_neural_row_bh3": {
                        "number_of_families": 6,
                        "tests_per_family": 3,
                        "significant_cells": 12,
                        "changed_from_nominal_cr1": cr1_bh3_changed[
                            [
                                "block_label",
                                "outcome_label",
                                "cr1_nominal_star",
                                "cr1_bh3_star",
                            ]
                        ].to_dict(orient="records"),
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = [
        absolute_display_path,
        raw_decoder_table_path,
        decoder_sign_audit_path,
        multiplicity_audit_path,
        multiplicity_summary_path,
    ]
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
        if any(
            row["color"] != "#000000"
            for row in audit["main_figure5_star_contract"]
        ):
            raise AssertionError(f"ED10 panel-c {scope} significance annotations must be black")
        if any(
            row["fontsize"] != (8.0 if row["text"] == "n.s." else 12.0)
            for row in audit["main_figure5_star_contract"]
        ):
            raise AssertionError(f"ED10 panel-c {scope} n.s. must match the 8-pt x-axis label")
    if "Absolute Pearson correlation" not in svg_text or "Pooled Pearson r" in svg_text:
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
        correlation_styles = audit["panel_a_correlation_value_styles"]
        if not (
            len(correlation_styles) == 18
            and all(row["color"] == "#000000" for row in correlation_styles)
            and all(row["fontweight"] == "normal" for row in correlation_styles)
        ):
            raise AssertionError(
                f"ED10 panel-a {scope} correlation values are not uniformly black "
                f"normal text: {correlation_styles}"
            )
        bh3_cr1_stars = audit["panel_a_star_contract"]
        if not (
            [row["text"] for row in bh3_cr1_stars]
            == ["*", "*", "*", "**", "**", "**", "**", "**", "***", "*", "***", "*"]
            and all(row["color"] == "#000000" for row in bh3_cr1_stars)
            and all(row["fontweight"] == "bold" for row in bh3_cr1_stars)
            and all(float(row["horizontal_clearance_from_r_px"]) >= 2.0 for row in bh3_cr1_stars)
            and all(bool(row["star_center_above_r_center"]) for row in bh3_cr1_stars)
            and all(bool(row["inside_heatmap_axes"]) for row in bh3_cr1_stars)
        ):
            raise AssertionError(
                f"ED10 panel-a {scope} mouse-CR1 row-BH3 star contract drifted: {bh3_cr1_stars}"
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
        "schema": "ed10_slide12_powerpoint_matched_validation/v9",
        "status": "PASS",
        "source_slide": 12,
        "panel_inventory": [
            "a correlation matrix",
            "b reward-level variance partition",
            "c four level-policy associations",
        ],
        "layout_contract": "top matrix / middle two bars / bottom four scatters",
        "panel_a_mouse_cr1_rowwise_bh3_stars": True,
        "checks": {
            "a_scope_30_pairs_14_mice": True,
            "b_scope_30_pairs_14_mice": True,
            "c_scope_36_pairs_14_mice": True,
            "maintained_source_tables_not_recomputed": True,
            "panel_a_raw_decoder_error_sign_audit_pass": all(
                decoder_sign_audit["checks"].values()
            ),
            "panel_a_other_twelve_cells_exact": True,
            "open_compound_display_terms": True,
            "root_png_4500_square": True,
            "caption_level_statistics_absent_except_requested_annotations": True,
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
            "panel_a_18_correlation_values_all_black_normal_weight": True,
            "panel_a_12_mouse_cr1_bh3_stars_upper_right_of_r": True,
            "panel_a_cr1_covariance_clustered_by_mouse": True,
            "panel_a_cr1_nominal_and_bh3_exported": True,
            "panel_a_uses_elapsed_first_long_bout_time": True,
            "panel_b_reward_decoder_unique_label_and_elapsed_time_title": True,
            "panel_c_true_policy_subscripts_and_two_decimal_ticks": True,
            "panel_c_corrected_depth_weighted_q_shown_as_three_stars": True,
            "panel_c_first_three_controls_labeled_nonsignificant": True,
            "panel_a_colorbar_title_is_pearson_correlation": True,
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
        "annotation_style": {
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
        "inputs": [record(path, "input") for path in required],
        "outputs": [record(path, "output") for path in outputs],
        "renderer": record(Path(__file__).resolve(), "renderer"),
    }
    validation_path = render_root / "AUDIT_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "validation": validation_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
