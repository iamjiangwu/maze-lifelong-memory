"""Render Extended Data Figure 10 from its three plot-ready source tables.

The quantitative inputs are frozen plot-ready tables. Panel a shows black,
normal-weight correlation values with adjacent stars derived from two-sided
pair-row OLS tests with CR1 standard errors clustered by animal and Benjamini--
Hochberg correction across the three behavior outcomes within each neural row.
Nominal CR1 and row-wise BH3 results are exported together.
The ED2 policy order (P_SF, P_SA, P_BF, depth-weighted P_BS) is preserved.
"""

from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLE_A = DATA / "POWERPOINT_WHOLE_SESSION_CORRELATION_MATRIX.csv"
TABLE_B = DATA / "REWARD_LEVEL_COMMONALITY.csv"
PAIR_VALUES = DATA / "CURRENT_FIXED_NEURON_PAIR_ROWS.csv"
TABLE_C = DATA / "level_policy_bridge_join_corrected36.csv"
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
INTERROW_GAPS_720 = {"a_b": 4.38, "b_c": 10.415}
CONTENT_HEIGHT_720 = (
    sum(ROLE_HEIGHTS_720.values()) + sum(INTERROW_GAPS_720.values())
)
CONTENT_MARGIN_720 = (720 - CONTENT_HEIGHT_720) / 2
ROLE_TOPS_720 = {
    "a": CONTENT_MARGIN_720,
    "b": CONTENT_MARGIN_720
    + ROLE_HEIGHTS_720["a"]
    + INTERROW_GAPS_720["a_b"],
    "c": CONTENT_MARGIN_720
    + ROLE_HEIGHTS_720["a"]
    + INTERROW_GAPS_720["a_b"]
    + ROLE_HEIGHTS_720["b"]
    + INTERROW_GAPS_720["b_c"],
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


def correlation_statistics(
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
) -> pd.DataFrame:
    """Flip only decoder-strength rows after proving them against raw errors."""

    display = maintained.copy()
    display["outcome_label"] = display["outcome_label"].replace(
        {"Reward-approach speed advantage": "Reward approach speed advantage"}
    )
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
    return display


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
    marker_radius_px = (np.sqrt(10.0) / 2.0) * ax.figure.dpi / 72.0
    best: tuple[float, mpl.text.Text] | None = None
    # When no lower-right candidate clears the points by 2 px, lower the y-axis floor
    # in 5% steps and retry the same candidates. Data that clear on the first pass
    # render exactly as before.
    floor_extensions = 0
    for floor_extensions in range(7):
        if floor_extensions:
            best[1].remove()
            best = None
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom - 0.05 * (top - bottom), top)
        points_px = ax.transData.transform(np.column_stack([x, y]))
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
        if best is not None and best[0] >= 2.0:
            break
    if best is None or best[0] < 2.0:
        raise AssertionError(
            f"Could not place lower-right r label clear of scatter points for {column}: "
            f"best clearance={None if best is None else best[0]:.3f}px"
        )
    if floor_extensions:
        # A lower floor moves every point; check the clearance of
        # the upper-left significance annotation and fail closed if too close.
        ax.figure.canvas.draw()
        points_px = ax.transData.transform(np.column_stack([x, y]))
        for text in ax.texts:
            if str(text.get_gid() or "").startswith("policy_significance_"):
                clearance = _point_to_bbox_clearance_px(
                    points_px, text.get_window_extent(ax.figure.canvas.get_renderer()), marker_radius_px
                )
                if clearance < 2.0:
                    raise AssertionError(
                        f"Lowered floor brings points within 2 px of the annotation for {column}: {clearance:.3f}px"
                    )
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


def render_individuals(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    stats_c: pd.DataFrame,
    out: Path,
) -> list[Path]:
    """Draw and save each panel from the prepared plotting tables."""
    outputs: list[Path] = []
    for role, table, builder, stem in (
        ("a", a, add_role_a, "a_correlation_matrix"),
        ("b", b, add_role_b, "b_variation_partition"),
        ("c", c, add_role_c, "c_level_policy_associations"),
    ):
        width = 7.15
        fig = plt.figure(figsize=(width, width / ROLE_ASPECTS[role]))
        if role == "c":
            builder(fig, [0, 0, 1, 1], table, stats_c)
        else:
            builder(fig, [0, 0, 1, 1], table)
        fig.canvas.draw()
        outputs += save_panel(fig, out / "panels" / stem, exact_canvas=True)
    return outputs


def render_assembly(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    stats_c: pd.DataFrame,
    stem: Path,
) -> list[Path]:
    """Place the three panels on the figure page without altering their values."""
    fig = plt.figure(figsize=(7.5, 7.5))
    add_role_a(fig, SLIDE_ROLE_RECTS["a"], a)
    add_role_b(fig, SLIDE_ROLE_RECTS["b"], b)
    add_role_c(fig, SLIDE_ROLE_RECTS["c"], c, stats_c)
    fig.canvas.draw()
    return save_panel(fig, stem, exact_canvas=True)


def main(*, output_root: Path | None = None, output_round: str | None = None) -> dict[str, object]:
    """Load the supplied measurements, prepare plotted statistics and export."""
    configure_style()
    for path in (TABLE_A, TABLE_B, TABLE_C, STATS_C, PAIR_VALUES):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    a = pd.read_csv(TABLE_A)
    b = pd.read_csv(TABLE_B)
    c = pd.read_csv(TABLE_C, dtype={"animal_id": str})
    stats_c = pd.read_csv(STATS_C)
    pair_rows = pd.read_csv(PAIR_VALUES, dtype=str)
    if len(a) != 18 or int(a["n_pairs"].unique()[0]) != 30 or int(a["n_animals"].unique()[0]) != 14:
        raise AssertionError("ED10 correlation matrix requires 30 pairs from 14 animals")
    expected_blocks = [
        "reward_location", "whole_decoder", "place_map", "structure", "pv_map", "rate",
    ]
    if list(dict.fromkeys(a["block"].astype(str))) != expected_blocks:
        raise AssertionError("ED10 correlation matrix row order mismatch")
    if len(pair_rows) != 30 or pair_rows["pair_key"].nunique() != 30:
        raise AssertionError("ED10 correlation inputs require 30 distinct pairs")
    a = raw_decoder_error_display_table(a, pair_rows)
    statistics = correlation_statistics(a, pair_rows)
    a = a.merge(
        statistics[["block", "outcome", "cr1_p_two_sided", "cr1_bh3_adjusted_p"]],
        on=["block", "outcome"],
        how="left",
        validate="one_to_one",
    )
    if not (
        int(statistics["cr1_nominal_star"].ne("").sum()) == 13
        and int(statistics["cr1_bh3_star"].ne("").sum()) == 13
    ):
        raise AssertionError("ED10 correlation significance counts differ from the source")
    if len(c) != 36 or c["pair_key"].nunique() != 36 or c["animal_id"].nunique() != 14:
        raise AssertionError("ED10 policy inputs require 36 pairs from 14 animals")
    if len(stats_c) != 4:
        raise AssertionError("ED10 policy statistics require four displayed measures")

    render_root = output_root.resolve() if output_root is not None else ROOT
    render_round = output_round or ROUND
    output_dir = render_root / "assembly" / render_round
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = render_root / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)

    # The wrapper reads this table to check the displayed cells.
    absolute_display_path = source_data / "ED10_PANEL_A_ABSOLUTE_DISPLAY.csv"
    absolute_display = a[["block", "outcome", "pooled_pearson_r", P_VALUE_COLUMN]].copy()
    absolute_display["absolute_pearson_r"] = absolute_display["pooled_pearson_r"].abs()
    absolute_display["display_text"] = absolute_display["absolute_pearson_r"].map(lambda value: f"{value:.2f}")
    absolute_display["stars"] = absolute_display[P_VALUE_COLUMN].map(p_value_star)
    absolute_display.to_csv(absolute_display_path, index=False)

    statistics_path = source_data / "correlation_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    outputs = [absolute_display_path, statistics_path]
    outputs += render_individuals(a, b, c, stats_c, output_dir)
    assembly_stem = output_dir / f"extended_data_figure10_source_rerender_{render_round}"
    outputs += render_assembly(a, b, c, stats_c, assembly_stem)
    result = {"status": "PASS", "outputs": [path.as_posix() for path in outputs]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
