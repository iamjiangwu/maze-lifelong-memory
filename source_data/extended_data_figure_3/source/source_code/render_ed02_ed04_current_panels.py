"""Render current paper-style source panels for Extended Data Figures 2-4.

Only saved, validated analysis tables are consumed. The separate currentness
refresh script performs the targeted raw ED2 append and saved-table-only ED3
simulation; this renderer does not load trajectories or rerun simulations.
Every plotted table and statistic is exported beside a scope manifest.

Important scope locks
---------------------
* The exact first-long-bout tile-transition table has 153 sessions after the six
  appended 12-month sessions pass strict recovered-frame lb0/slb0 parity.
* The four-policy, depth-varying, and parameter-by-parameter decomposition
  panels all use validated current-plus-new12 saved-table inputs.
* Latent-return pre-reward effects use the corrected three-claim family; the
  broad effect is borderline after correction and the specificity contrast is
  not significant. The post-port direct-approach effect is the robust result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFRESH_ROOT = (
    REPO_ROOT / "docs" / "extended_data_figure_panel_refresh_2026-07-12"
)
CURRENT_ROOT = (
    REPO_ROOT
    / "docs"
    / "behavior_panel_sensitivity_840_841_with_new12_baseline_2026-06-06"
)
NEW12_ROOT = (
    REPO_ROOT
    / "docs"
    / "new_12mo_session_fit_2026-06-02"
    / "panel_by_panel_current_vs_new"
)
ROUTE_ROOT = (
    REPO_ROOT
    / "docs"
    / "turn_sequence_retention_2026-08-11_direct_entry_reconstruction"
)

FORMATS = ("png", "svg", "pdf", "tiff")

COLORS = {
    "naive": "#8BBDEB",
    "naive_mark": "#1F90D5",
    "return": "#DA2323",
    "return_mark": "#D82929",
    "experienced": "#5F8C62",
    "latent": "#B3A99B",
    "axis": "#202020",
    "grid": "#D8D8D8",
    "neutral": "#7E878F",
    "legacy": "#8A8A8A",
}
MONTH_COLORS = {0: "#8BBDEB", 3: "#F5A18E", 6: "#F37B6D", 9: "#DB4B4B", 12: "#9B0E13"}
GROUP_LABELS = {
    "naive": "Naive day",
    "LTM": "Return day",
    "experienced": "Second day",
    "latent": "Latent day 2",
    "latent_day2": "Latent day 2",
    "experienced_day2": "Second day",
    "return": "Return day",
}


def parse_required_binary(value: Any) -> bool:
    """Parse a required Boolean without treating missing values as truthy."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
        if float(value) in {0.0, 1.0}:
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "t", "1"}:
        return True
    if text in {"false", "f", "0"}:
        return False
    raise ValueError(f"Unresolved or invalid route correctness value: {value!r}")

POLICY_SPECS = [
    ("Bf", r"$P_{SF}$", "Continue into branch"),
    ("Ba", r"$P_{SA}$", "Alternate side"),
    ("Lf", r"$P_{BF}$", "Exit sampled subtree"),
    ("Lo", r"$P_{BS}$", "Exit toward stem"),
]
PARAMETER_TO_POLICY = {
    "Bf": r"$P_{SF}$",
    "Ba": r"$P_{SA}$",
    "Lf": r"$P_{BF}$",
    "Lo": r"$P_{BS}$",
}
ROUTE_GROUP_ORDER = ["naive", "return", "experienced_day2", "latent_day2"]
ROUTE_DEPTH_PANEL_ORDER = [
    "naive",
    "experienced_day2",
    "return",
    "latent_day2",
]
ROUTE_DEPTH_PANEL_LABELS = {
    "naive": "Naive day 1",
    "experienced_day2": "Second day",
    "return": "Return day 1",
    "latent_day2": "Latent day 2",
}
ROUTE_DEPTH_PANEL_COLORS = {
    "naive": COLORS["naive"],
    "experienced_day2": COLORS["experienced"],
    "return": COLORS["return"],
    "latent_day2": COLORS["latent"],
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 4.9,
            "legend.frameon": False,
            "lines.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def sem(values: pd.Series | np.ndarray) -> float:
    array = (
        pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    )
    return float(stats.sem(array)) if len(array) >= 2 else np.nan


def bh_q(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if not len(finite_idx):
        return pd.Series(q, index=values.index)
    order = np.argsort(p[finite_idx])
    ranked = p[finite_idx][order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[finite_idx[order]] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=values.index)


def p_text(value: float, prefix: str = "P") -> str:
    if not np.isfinite(value):
        return f"{prefix}=NA"
    if value < 0.001:
        return f"{prefix}<0.001"
    return f"{prefix}={value:.3f}"


def animal_id_from_session(value: Any) -> str:
    match = re.match(r"^(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot recover animal ID from {value!r}")
    return match.group(1)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in FORMATS:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": 0.03,
            "facecolor": "white",
        }
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def cluster_ols(
    y: np.ndarray, xmat: np.ndarray, clusters: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    xmat = np.asarray(xmat, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    finite = np.isfinite(y) & np.all(np.isfinite(xmat), axis=1)
    y, xmat, clusters = y[finite], xmat[finite], clusters[finite]
    n, k = xmat.shape
    unique_clusters = np.unique(clusters)
    if n <= k or len(unique_clusters) < 3:
        raise ValueError(
            "Cluster OLS requires more rows than parameters and at least three clusters"
        )
    xtx_inv = np.linalg.pinv(xmat.T @ xmat)
    beta = xtx_inv @ xmat.T @ y
    residual = y - xmat @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = xmat[mask].T @ residual[mask][:, None]
        meat += score @ score.T
    g = len(unique_clusters)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = scale * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = beta / se
    p_value = 2.0 * stats.t.sf(np.abs(t_value), df=g - 1)
    return {
        "n_rows": int(n),
        "n_clusters": int(g),
        "df_cluster": int(g - 1),
        "beta": beta,
        "covariance": covariance,
        "cluster_se": se,
        "t_cluster": t_value,
        "p_cluster": p_value,
    }


def two_group_cluster_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    target: str,
    baseline: str,
    cluster_col: str,
) -> dict[str, Any]:
    subset = df.loc[df[group_col].isin([target, baseline])].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=[value_col, group_col, cluster_col])
    indicator = subset[group_col].eq(target).astype(float).to_numpy()
    fit = cluster_ols(
        subset[value_col].to_numpy(dtype=float),
        np.column_stack([np.ones(len(subset)), indicator]),
        subset[cluster_col].astype(str).to_numpy(),
    )
    return {
        "target": target,
        "baseline": baseline,
        "n_target_rows": int(subset[group_col].eq(target).sum()),
        "n_baseline_rows": int(subset[group_col].eq(baseline).sum()),
        "n_target_animals": int(
            subset.loc[subset[group_col].eq(target), cluster_col].nunique()
        ),
        "n_baseline_animals": int(
            subset.loc[subset[group_col].eq(baseline), cluster_col].nunique()
        ),
        "beta_target_minus_baseline": float(fit["beta"][1]),
        "cluster_se": float(fit["cluster_se"][1]),
        "t_cluster": float(fit["t_cluster"][1]),
        "p_cluster": float(fit["p_cluster"][1]),
        "n_clusters": fit["n_clusters"],
    }


ROUTE_PAIRWISE_SPECS = (
    ("experienced_day2", "naive"),
    ("return", "latent_day2"),
    ("experienced_day2", "return"),
    ("experienced_day2", "latent_day2"),
    ("return", "naive"),
    ("latent_day2", "naive"),
)


def four_level_cluster_route_stats(
    df: pd.DataFrame,
    *,
    value_col: str = "route_fraction_correct",
    group_col: str = "condition",
    cluster_col: str = "animal",
    group_order: tuple[str, ...] = tuple(ROUTE_GROUP_ORDER),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one saturated condition model and return BH6 contrasts plus group CIs."""
    model_df = df[[value_col, group_col, cluster_col]].copy()
    model_df[value_col] = pd.to_numeric(model_df[value_col], errors="coerce")
    model_df = model_df.dropna(subset=[value_col, group_col, cluster_col])
    model_df[group_col] = model_df[group_col].astype(str)
    model_df[cluster_col] = model_df[cluster_col].astype(str)
    observed_groups = set(model_df[group_col])
    if observed_groups != set(group_order):
        raise ValueError(
            f"Route model groups are {sorted(observed_groups)}; expected {list(group_order)}"
        )

    xmat = np.column_stack(
        [model_df[group_col].eq(group).astype(float).to_numpy() for group in group_order]
    )
    fit = cluster_ols(
        model_df[value_col].to_numpy(dtype=float),
        xmat,
        model_df[cluster_col].to_numpy(),
    )
    beta = np.asarray(fit["beta"], dtype=float)
    covariance = np.asarray(fit["covariance"], dtype=float)
    df_cluster = int(fit["df_cluster"])
    t_critical = float(stats.t.ppf(0.975, df=df_cluster))
    group_to_index = {group: index for index, group in enumerate(group_order)}

    group_rows: list[dict[str, Any]] = []
    for group in group_order:
        index = group_to_index[group]
        estimate = float(beta[index])
        se = float(np.sqrt(max(covariance[index, index], 0.0)))
        subset = model_df.loc[model_df[group_col].eq(group)]
        group_rows.append(
            {
                "condition": group,
                "condition_label": GROUP_LABELS[group],
                "mean_route_fraction_correct": estimate,
                "cluster_se": se,
                "cluster_ci95_low": estimate - t_critical * se,
                "cluster_ci95_high": estimate + t_critical * se,
                "n_sessions": int(len(subset)),
                "n_animals": int(subset[cluster_col].nunique()),
                "n_clusters_total": int(fit["n_clusters"]),
                "df_cluster": df_cluster,
                "model": "four_level_session_ols_cr1_mouse_clustered_small_g_t",
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for target, baseline in ROUTE_PAIRWISE_SPECS:
        contrast = np.zeros(len(group_order), dtype=float)
        contrast[group_to_index[target]] = 1.0
        contrast[group_to_index[baseline]] = -1.0
        estimate = float(contrast @ beta)
        se = float(np.sqrt(max(contrast @ covariance @ contrast, 0.0)))
        t_value = estimate / se
        p_value = float(2.0 * stats.t.sf(abs(t_value), df=df_cluster))
        target_sub = model_df.loc[model_df[group_col].eq(target)]
        baseline_sub = model_df.loc[model_df[group_col].eq(baseline)]
        pair_rows.append(
            {
                "plot_metric": value_col,
                "source_metric": "four_level_session_ols_cr1_mouse_clustered_small_g_t",
                "comparison": f"{target}_vs_{baseline}",
                "target_condition": target,
                "baseline_condition": baseline,
                "n_sessions_target": int(len(target_sub)),
                "n_sessions_baseline": int(len(baseline_sub)),
                "n_animals_target": int(target_sub[cluster_col].nunique()),
                "n_animals_baseline": int(baseline_sub[cluster_col].nunique()),
                "mean_target": float(beta[group_to_index[target]]),
                "mean_baseline": float(beta[group_to_index[baseline]]),
                "mean_diff_target_minus_baseline": estimate,
                "cluster_se": se,
                "cluster_ci95_low": estimate - t_critical * se,
                "cluster_ci95_high": estimate + t_critical * se,
                "t_cluster": t_value,
                "df_cluster": df_cluster,
                "p_cluster": p_value,
                "n_rows_total": int(fit["n_rows"]),
                "n_clusters_total": int(fit["n_clusters"]),
                "correction_family": "all_six_route_score_pairwise_contrasts_bh6",
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    if len(pairwise) != 6 or pairwise["comparison"].nunique() != 6:
        raise AssertionError("Route-score inference must contain exactly six unique pairs")
    pairwise["q_bh6"] = bh_q(pairwise["p_cluster"])
    pairwise["supported_bh6"] = pairwise["q_bh6"].lt(0.05)
    return pairwise, pd.DataFrame(group_rows)


def continuous_cluster_test(
    df: pd.DataFrame,
    value_col: str,
    predictor_col: str,
    cluster_col: str,
    scale: float = 1.0,
) -> dict[str, Any]:
    subset = df[[value_col, predictor_col, cluster_col]].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset[predictor_col] = pd.to_numeric(subset[predictor_col], errors="coerce")
    subset = subset.dropna()
    predictor = subset[predictor_col].to_numpy(dtype=float) / scale
    fit = cluster_ols(
        subset[value_col].to_numpy(dtype=float),
        np.column_stack([np.ones(len(subset)), predictor]),
        subset[cluster_col].astype(str).to_numpy(),
    )
    return {
        "n_rows": fit["n_rows"],
        "n_animals": fit["n_clusters"],
        "predictor_scale": scale,
        "beta_per_scale": float(fit["beta"][1]),
        "cluster_se": float(fit["cluster_se"][1]),
        "t_cluster": float(fit["t_cluster"][1]),
        "p_cluster": float(fit["p_cluster"][1]),
    }


def add_strip_summary(
    ax: plt.Axes,
    values_by_group: list[np.ndarray],
    colors: list[str],
    rng: np.random.Generator,
) -> None:
    for index, (values, color) in enumerate(zip(values_by_group, colors)):
        values = np.asarray(values, dtype=float)
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=9,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            alpha=0.82,
            zorder=2,
        )
        ax.errorbar(
            index,
            float(np.mean(values)),
            yerr=sem(values),
            color=color,
            marker="o",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=1.8,
            elinewidth=0.8,
            zorder=3,
        )


def prepare_ed02(
    source_tables: Path,
    *,
    input_refresh_root: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, Any]]:
    input_root = (input_refresh_root or source_tables.parent).resolve()
    exact_refresh_dir = (
        input_root / "rerun_current" / "ed02_lb0_tiles_current_plus_new12"
    )
    exact_path = (
        exact_refresh_dir
        / "ed02_lb0_matched_tile_transitions_per_session_current_plus_new12.csv"
    )
    parity_path = exact_refresh_dir / "ed02_lb0_tiles_new12_six_parity_audit.csv"
    exact_manifest_path = exact_refresh_dir / "manifest.json"
    exact = pd.read_csv(exact_path, dtype={"animal_id": str}).copy()
    parity = pd.read_csv(parity_path)
    exact_manifest = json.loads(exact_manifest_path.read_text(encoding="utf-8"))
    if len(exact) != 153 or not exact["load_ok"].all():
        raise ValueError(
            "Exact first-long-bout tile table must contain 153 successfully loaded sessions"
        )
    if len(parity) != 6 or not parity["include_after_all_parity_gates"].all():
        raise ValueError(
            "All six appended ED2 sessions must pass the strict parity gates"
        )
    if exact_manifest.get("status") != "current_plus_new12_validated":
        raise ValueError("ED2 currentness manifest is not validated")
    exact["scope_status"] = "current_plus_new12_exact_estimand_validated"
    expected_group_counts = {"naive": 23, "return": 56, "experienced_day2": 74}
    found_counts = exact.groupby("group_key", observed=True).size().to_dict()
    if found_counts != expected_group_counts:
        raise ValueError(f"Unexpected exact-estimand group counts: {found_counts}")
    appended_new12 = {
        f"{animal}-12day{day}" for animal in (837, 838, 839) for day in (1, 2)
    }
    if not appended_new12.issubset(set(exact["session"])):
        raise ValueError(
            "Current exact-estimand table is missing appended new12 sessions"
        )

    month_source = exact.loc[exact["day"].eq(1)].copy()
    month_summary = (
        month_source.groupby("month", as_index=False)
        .agg(
            n_sessions=("session", "nunique"),
            n_animals=("animal_id", "nunique"),
            mean_tiles=("nose_tiles_to_first_long_bout", "mean"),
            sem_tiles=("nose_tiles_to_first_long_bout", sem),
            median_tiles=("nose_tiles_to_first_long_bout", "median"),
            mean_time_s=("lb0_time_in_maze_s", "mean"),
        )
        .sort_values("month")
    )
    month_tests: list[dict[str, Any]] = []
    test_source = month_source.copy()
    test_source["month_label"] = test_source["month"].astype(str)
    test_source["log1p_nose_tiles_to_first_long_bout"] = np.log1p(
        pd.to_numeric(test_source["nose_tiles_to_first_long_bout"], errors="coerce")
    )
    for month in [3, 6, 9, 12]:
        row = two_group_cluster_test(
            test_source,
            "log1p_nose_tiles_to_first_long_bout",
            "month_label",
            str(month),
            "0",
            "animal_id",
        )
        row["contrast"] = f"{month}mo_return_minus_naive"
        row["analysis_scale"] = "log1p_tile_transition_count"
        row["ratio_one_plus_count"] = float(np.exp(row["beta_target_minus_baseline"]))
        target_raw = test_source.loc[
            test_source["month_label"].eq(str(month)),
            "nose_tiles_to_first_long_bout",
        ]
        baseline_raw = test_source.loc[
            test_source["month_label"].eq("0"),
            "nose_tiles_to_first_long_bout",
        ]
        row["raw_mean_target"] = float(target_raw.mean())
        row["raw_mean_baseline"] = float(baseline_raw.mean())
        row["raw_mean_difference"] = float(target_raw.mean() - baseline_raw.mean())
        month_tests.append(row)
    tile_stats = pd.DataFrame(month_tests)
    tile_stats["q_bh4"] = bh_q(tile_stats["p_cluster"])

    pearson = stats.pearsonr(
        exact["lb0_time_in_maze_s"].to_numpy(dtype=float),
        exact["nose_tiles_to_first_long_bout"].to_numpy(dtype=float),
    )
    spearman = stats.spearmanr(
        exact["lb0_time_in_maze_s"].to_numpy(dtype=float),
        exact["nose_tiles_to_first_long_bout"].to_numpy(dtype=float),
    )
    tile_time_stats = pd.DataFrame(
        [
            {
                "scope": "current_plus_new12_exact_estimand_153_sessions",
                "n_sessions": len(exact),
                "n_animals": exact["animal_id"].nunique(),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        ]
    )

    exploration_summary = pd.read_csv(
        CURRENT_ROOT
        / "exploration_efficiency_cycles_pooled_current_plus_new12_summary.csv"
    )
    exploration_cycles = pd.read_csv(
        CURRENT_ROOT
        / "exploration_efficiency_cycles_pooled_current_plus_new12_cycles.csv"
    )
    first_chunk = pd.read_csv(
        CURRENT_ROOT / "exploration_first_chunk_current_plus_new12_points.csv"
    )
    first_chunk["animal_id"] = first_chunk["animal_id"].astype(str)
    first_chunk["group_key"] = np.where(
        first_chunk["group"].str.contains("LTM"), "return", "naive"
    )
    exploration_test = pd.DataFrame(
        [
            two_group_cluster_test(
                first_chunk,
                "first_chunk_efficiency",
                "group_key",
                "return",
                "naive",
                "animal_id",
            )
        ]
    )
    first_cycle_counts = (
        exploration_summary.loc[exploration_summary["cycle_index"].eq(1)]
        .set_index("group_key")["n_sessions"]
        .to_dict()
    )
    if first_cycle_counts != {"naive": 23, "return": 56, "experienced_day2": 73}:
        raise ValueError(
            f"Unexpected current-plus-new12 first-cycle counts: {first_cycle_counts}"
        )

    speed_plot = pd.read_csv(
        CURRENT_ROOT / "bodymass_speed_by_mode_first60_plot_data_current_plus_new12.csv"
    )
    speed_sessions = pd.read_csv(
        CURRENT_ROOT / "bodymass_speed_current_plus_new12_by_session.csv",
        dtype={"animal_id": str},
    )
    speed_rows: list[dict[str, Any]] = []
    for mode in ["explore", "reward"]:
        for day in ["day1", "day2"]:
            subset = speed_sessions.loc[
                speed_sessions["day"].eq(day) & speed_sessions["analysis_mode"].eq(mode)
            ].copy()
            subset["month"] = pd.to_numeric(
                subset["month_label"].astype(str).str.extract(r"(\d+)", expand=False),
                errors="coerce",
            )
            result = continuous_cluster_test(
                subset,
                "observed_pair_speed_cm_s",
                "month",
                "animal_id",
                scale=3.0,
            )
            result["mode"] = mode
            result["day"] = day
            result["estimand"] = "speed_change_per_3_months"
            speed_rows.append(result)
    speed_stats = pd.DataFrame(speed_rows)

    tables = {
        "exact": exact,
        "month_summary": month_summary,
        "tile_stats": tile_stats,
        "tile_time_stats": tile_time_stats,
        "exploration_summary": exploration_summary,
        "exploration_cycles": exploration_cycles,
        "exploration_test": exploration_test,
        "speed_plot": speed_plot,
        "speed_stats": speed_stats,
    }
    paths = {
        "exact": write_csv(
            exact,
            source_tables / "ed02_lb0_tile_transitions_exact_current_plus_new12.csv",
        ),
        "parity": write_csv(
            parity, source_tables / "ed02_lb0_tile_new12_append_parity_audit.csv"
        ),
        "month_summary": write_csv(
            month_summary,
            source_tables / "ed02_lb0_tile_month_summary_current_plus_new12.csv",
        ),
        "tile_stats": write_csv(
            tile_stats,
            source_tables / "ed02_lb0_tile_month_cluster_stats_current_plus_new12.csv",
        ),
        "tile_time_stats": write_csv(
            tile_time_stats,
            source_tables / "ed02_lb0_tile_time_stats_current_plus_new12.csv",
        ),
        "exploration_summary": write_csv(
            exploration_summary,
            source_tables / "ed02_exploration_efficiency_cycles_current_plus_new12.csv",
        ),
        "exploration_test": write_csv(
            exploration_test,
            source_tables
            / "ed02_exploration_first_cycle_cluster_stats_current_plus_new12.csv",
        ),
        "speed_plot": write_csv(
            speed_plot,
            source_tables / "ed02_bodymass_speed_plot_data_current_plus_new12.csv",
        ),
        "speed_stats": write_csv(
            speed_stats,
            source_tables / "ed02_bodymass_speed_cluster_stats_current_plus_new12.csv",
        ),
    }
    audit = {
        "exact_estimator_scope": "validated current-plus-new12 153-session table",
        "exact_estimator_group_counts": found_counts,
        "exact_estimator_appended_sessions": sorted(appended_new12),
        "exact_estimator_parity_manifest": str(exact_manifest_path),
        "all_appended_sessions_passed_parity": True,
        "current_plus_new12_exploration_first_cycle_counts": first_cycle_counts,
        "tile_month_statistical_unit": "eligible Day 1 session row",
        "tile_month_dependence": "CR1 covariance clustered by animal_id with Student t(G-1) calibration",
        "tile_month_analysis_scale": "log1p nose tile-transition count",
        "tile_month_multiplicity": "BH across the four return-month-versus-naive contrasts",
    }
    return tables, paths, audit


def plot_ed02(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    rng = np.random.default_rng(20260712)

    exact = tables["exact"]
    first_day = exact.loc[exact["day"].eq(1)].copy()
    summary = tables["month_summary"].set_index("month")
    tile_stats = tables["tile_stats"].copy()
    tile_stats["month"] = (
        tile_stats["contrast"].astype(str).str.extract(r"^(\d+)mo")[0].astype(int)
    )
    tile_stats = tile_stats.set_index("month")
    months = [0, 3, 6, 9, 12]
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    for index, month in enumerate(months):
        values = first_day.loc[
            first_day["month"].eq(month), "nose_tiles_to_first_long_bout"
        ].to_numpy(dtype=float)
        ax.scatter(
            index + rng.uniform(-0.08, 0.08, len(values)),
            values,
            s=8,
            facecolor="white",
            edgecolor=MONTH_COLORS[month],
            linewidth=0.5,
            alpha=0.75,
        )
        ax.errorbar(
            index,
            summary.loc[month, "mean_tiles"],
            yerr=summary.loc[month, "sem_tiles"],
            color=MONTH_COLORS[month],
            marker="o",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=1.6,
            elinewidth=0.8,
            zorder=3,
        )
        ax.text(
            index,
            16,
            f"n={int(summary.loc[month, 'n_sessions'])}",
            ha="center",
            va="bottom",
            fontsize=4.4,
        )
        if month != 0:
            ax.text(
                index,
                0.97,
                p_text(float(tile_stats.loc[month, "q_bh4"]), prefix="q"),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=4.5,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(months)), ["Naive", "3 mo", "6 mo", "9 mo", "12 mo"])
    ax.set_ylabel("Tile transitions to first long bout")
    ax.set_title(
        "First-long-bout-matched traversal", loc="left", fontweight="bold", pad=2.0
    )
    ax.text(
        0.99,
        0.02,
        "Tests: log(1 + count), CR1 clustered by mouse; BH4",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.2,
        color=COLORS["legacy"],
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4, which="both")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.88)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed02a_lb0_tile_transitions_exact_current_plus_new12"
        )
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    for key, color, label in [
        ("naive", COLORS["naive_mark"], "Naive day"),
        ("return", COLORS["return_mark"], "Return day"),
        ("experienced_day2", COLORS["experienced"], "Second day"),
    ]:
        subset = exact.loc[exact["group_key"].eq(key)]
        ax.scatter(
            subset["lb0_time_in_maze_s"],
            subset["nose_tiles_to_first_long_bout"],
            s=8,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            alpha=0.72,
            label=label,
        )
    x = exact["lb0_time_in_maze_s"].to_numpy(dtype=float)
    y = exact["nose_tiles_to_first_long_bout"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.linspace(0, float(x.max()), 100)
    ax.plot(fit_x, slope * fit_x + intercept, color=COLORS["axis"], linewidth=0.85)
    stat_row = tables["tile_time_stats"].iloc[0]
    ax.text(
        0.03,
        0.97,
        f"r={stat_row['pearson_r']:.2f}, {p_text(stat_row['pearson_p'])}\nn={int(stat_row['n_sessions'])} sessions",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.8,
    )
    ax.set_xlabel("Time to first long bout (s)")
    ax.set_ylabel("Tile transitions to first long bout")
    ax.set_title("Matched tile-time diagnostic", loc="left", fontweight="bold", pad=2.0)
    ax.legend(loc="lower right", handletextpad=0.3)
    ax.grid(color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.88)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed02b_lb0_tile_time_diagnostic_exact_current_plus_new12"
        )
    )

    cycle_summary = tables["exploration_summary"]
    fig, ax = plt.subplots(figsize=(3.55, 2.05))
    for group, color, label in [
        ("naive", COLORS["naive_mark"], "Naive day"),
        ("return", COLORS["return_mark"], "Return day"),
        ("experienced_day2", COLORS["experienced"], "Second day"),
    ]:
        subset = cycle_summary.loc[
            cycle_summary["group_key"].eq(group) & cycle_summary["cycle_index"].le(10)
        ]
        ax.plot(
            subset["cycle_index"],
            subset["eff_mean"],
            color=color,
            marker="o",
            markersize=2.4,
            label=f"{label} (cycle 1 n={int(subset.iloc[0]['n_sessions'])})",
        )
        ax.fill_between(
            subset["cycle_index"],
            subset["eff_mean"] - subset["eff_sem"],
            subset["eff_mean"] + subset["eff_sem"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    p_value = float(tables["exploration_test"].iloc[0]["p_cluster"])
    ax.text(
        0.98,
        0.97,
        f"Cycle 1 return vs naive\nclustered {p_text(p_value)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
    )
    ax.set_xlabel("Successive 32-end-node coverage cycle")
    ax.set_ylabel(r"Exploration efficiency ($32/N_{32}$)")
    ax.set_title("Reduced redundant sampling", loc="left", fontweight="bold", pad=2.0)
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.legend(loc="lower right")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.88)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed02c_exploration_efficiency_cycles_current_plus_new12"
        )
    )

    speed = tables["speed_plot"]
    stat_specs = [
        ("day1", "Day 1", COLORS["axis"]),
        ("day2", "Day 2", COLORS["experienced"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.18), sharey=True)
    for ax, mode, title in zip(
        axes, ["explore", "reward"], ["Exploration", "Reward-directed"]
    ):
        day1 = speed.loc[
            speed["mode_label"].eq(mode) & speed["day"].eq("day1")
        ].sort_values("month")
        ax.plot(
            day1["month"],
            day1["speed_mean_cm_s"],
            color=COLORS["axis"],
            linewidth=0.8,
            zorder=1,
        )
        for _, row in day1.iterrows():
            point_color = (
                COLORS["naive_mark"]
                if int(row["month"]) == 0
                else COLORS["return_mark"]
            )
            ax.errorbar(
                row["month"],
                row["speed_mean_cm_s"],
                yerr=row["speed_sem_cm_s"],
                color=point_color,
                marker="o",
                markersize=3.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=0,
                capsize=1.5,
                elinewidth=0.7,
                zorder=3,
            )
        day2 = speed.loc[
            speed["mode_label"].eq(mode) & speed["day"].eq("day2")
        ].sort_values("month")
        ax.errorbar(
            day2["month"],
            day2["speed_mean_cm_s"],
            yerr=day2["speed_sem_cm_s"],
            color=COLORS["experienced"],
            marker="s",
            markersize=3.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=0.9,
            capsize=1.5,
            elinewidth=0.7,
            zorder=2,
        )
        for row_index, (day, day_label, color) in enumerate(stat_specs):
            stat_row = (
                tables["speed_stats"]
                .loc[
                    tables["speed_stats"]["mode"].eq(mode)
                    & tables["speed_stats"]["day"].eq(day)
                ]
                .iloc[0]
            )
            ax.text(
                0.03,
                0.96 - 0.095 * row_index,
                f"{day_label}: β/3 mo={stat_row['beta_per_scale']:.2f}; "
                f"{p_text(stat_row['p_cluster'])}",
                color=color,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.45,
            )
        day1_counts = (
            speed.loc[speed["mode_label"].eq(mode) & speed["day"].eq("day1")]
            .set_index("month")["n_animals"]
            .astype(int)
        )
        day2_counts = (
            speed.loc[speed["mode_label"].eq(mode) & speed["day"].eq("day2")]
            .set_index("month")["n_animals"]
            .astype(int)
        )
        months = [0, 3, 6, 9, 12]
        tick_labels = [
            f"{'Naive' if month == 0 else month}\n{day1_counts.loc[month]}/{day2_counts.loc[month]}"
            for month in months
        ]
        ax.set_title(title, loc="left", fontweight="bold", pad=2.0)
        ax.set_xticks(months, tick_labels)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    axes[0].set_ylabel("Body-mass tracking speed (cm/s)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["naive_mark"],
            markeredgecolor="white",
            label="Naive day",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["axis"],
            markerfacecolor=COLORS["return_mark"],
            markeredgecolor="white",
            label="Return day",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLORS["experienced"],
            markerfacecolor=COLORS["experienced"],
            markeredgecolor="white",
            label="Second day",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.53, 0.995),
    )
    fig.text(
        0.54,
        0.035,
        "Retention interval (months); n = Day 1/Day 2",
        ha="center",
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.84, wspace=0.14)
    outputs.extend(
        save_figure(fig, output_dir / "ed02d_bodymass_speed_control_current_plus_new12")
    )
    return outputs


def prepare_ed03(
    source_tables: Path,
    *,
    input_refresh_root: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, Any]]:
    input_root = (input_refresh_root or source_tables.parent).resolve()
    points_expanded = pd.read_csv(
        CURRENT_ROOT / "turning_bias_chunk_points_current_plus_new12.csv"
    )
    points_expanded = points_expanded.loc[points_expanded["n_chunks"].eq(32)].copy()
    # Match the Main Fig. 1g estimand: every displayed policy must have at
    # least ten qualifying events in the same early window.  The broader
    # 23-naive/56-return table remains an explicit support sensitivity rather
    # than silently defining a different ED3 primary.
    support_columns = ["n_Bf", "n_Ba", "n_Lf", "n_Lo"]
    points = points_expanded.loc[
        points_expanded[support_columns].ge(10).all(axis=1)
    ].copy()
    points["animal_id"] = points["nickname"].map(animal_id_from_session)
    points_expanded["animal_id"] = points_expanded["nickname"].map(
        animal_id_from_session
    )
    expected_counts = {"LTM": 55, "experienced": 72, "naive": 21}
    found_counts = points.groupby("group", observed=True).size().to_dict()
    if found_counts != expected_counts:
        raise ValueError(f"Unexpected current-plus-new12 policy counts: {found_counts}")
    raw_summary = points.groupby("group", as_index=False).agg(
        n_sessions=("nickname", "nunique"),
        n_animals=("animal_id", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric, _, _ in POLICY_SPECS},
        **{f"{metric}_sem": (metric, sem) for metric, _, _ in POLICY_SPECS},
    )
    test_rows: list[dict[str, Any]] = []
    for metric, policy_label, description in POLICY_SPECS:
        for target, baseline, family in [
            ("LTM", "naive", "return_vs_naive"),
            ("experienced", "LTM", "second_vs_return"),
        ]:
            row = two_group_cluster_test(
                points, metric, "group", target, baseline, "animal_id"
            )
            row.update(
                {
                    "metric": metric,
                    "policy_label": policy_label,
                    "description": description,
                    "family": family,
                }
            )
            test_rows.append(row)
    raw_stats = pd.DataFrame(test_rows)
    raw_stats["q_bh4_within_family"] = np.nan
    for _, index in raw_stats.groupby("family").groups.items():
        raw_stats.loc[index, "q_bh4_within_family"] = bh_q(
            raw_stats.loc[index, "p_cluster"]
        )

    decomposition_refresh_dir = (
        input_root
        / "rerun_current"
        / "ed03_policy_decomposition_current_plus_new12"
    )
    decomposition_path = (
        decomposition_refresh_dir
        / "policy_decomposition_ltm_to_naive_current_plus_new12_chunk32.csv"
    )
    decomposition_manifest_path = decomposition_refresh_dir / "manifest.json"
    decomposition_manifest = json.loads(
        decomposition_manifest_path.read_text(encoding="utf-8")
    )
    decomposition = pd.read_csv(decomposition_path)
    if set(decomposition["parameter"]) != {"Bf", "Ba", "Lf", "Lo"}:
        raise ValueError(
            "Current decomposition must contain all four policy parameters"
        )
    if (
        decomposition_manifest.get("status") != "current_plus_new12_validated"
        or decomposition_manifest.get("n_naive") != 23
        or decomposition_manifest.get("n_ltm") != 57
    ):
        raise ValueError(
            "ED3 currentness manifest has an unexpected status or cohort size"
        )
    decomposition["scope_status"] = "current_plus_new12_n_naive23_n_ltm57"
    decomposition["policy_label"] = decomposition["parameter"].map(PARAMETER_TO_POLICY)

    simulator_summary_path = (
        input_root
        / "source_rendered_new"
        / "extended_data_fig_03"
        / "chunk_window_uncertainty_current_plus_new12_2026-07-14"
        / "ed03_simulator_window_mouse_cluster_summary_current_plus_new12.csv"
    )
    simulator_summary = pd.read_csv(simulator_summary_path)
    simulator_row = simulator_summary.loc[simulator_summary["n_chunks"].eq(32)]
    if len(simulator_row) != 1:
        raise ValueError("Expected one July-14 simulator summary row for n_chunks=32")
    simulator_row = simulator_row.iloc[0]
    gap = pd.DataFrame(
        [
            {
                "model": "four_bias",
                "model_label": "4 biases",
                "n_chunks": 32,
                "n_naive": int(simulator_row["n_naive_rows"]),
                "n_LTM": int(simulator_row["n_return_rows"]),
                "percent_explained_mean": 100
                * float(simulator_row["fraction_four_bias_at_ltm_length"]),
                "percent_explained_ci_lo": 100
                * float(
                    simulator_row["fraction_four_bias_at_ltm_length_cluster_ci_low"]
                ),
                "percent_explained_ci_hi": 100
                * float(
                    simulator_row["fraction_four_bias_at_ltm_length_cluster_ci_high"]
                ),
            },
            {
                "model": "depth_stratified",
                "model_label": "Depth-stratified P_BS",
                "n_chunks": 32,
                "n_naive": int(simulator_row["n_naive_rows"]),
                "n_LTM": int(simulator_row["n_return_rows"]),
                "percent_explained_mean": 100
                * float(simulator_row["fraction_depth_stratified_at_ltm_length"]),
                "percent_explained_ci_lo": 100
                * float(
                    simulator_row[
                        "fraction_depth_stratified_at_ltm_length_cluster_ci_low"
                    ]
                ),
                "percent_explained_ci_hi": 100
                * float(
                    simulator_row[
                        "fraction_depth_stratified_at_ltm_length_cluster_ci_high"
                    ]
                ),
            },
        ]
    )
    gap["scope"] = "current_plus_new12_support_matched"
    gap["scope_status"] = "primary_diagnostic_ratio"
    gap["uncertainty"] = "mouse-cluster bootstrap"
    gap["source_csv"] = str(simulator_summary_path)
    current_values = gap.set_index("model")["percent_explained_mean"].to_dict()
    if not (
        52.0 < current_values["four_bias"] < 52.8
        and 85.0 < current_values["depth_stratified"] < 85.9
    ):
        raise ValueError(
            f"Unexpected current-plus-new12 coverage fractions: {current_values}"
        )

    depth = pd.read_csv(
        CURRENT_ROOT
        / "turning_bias_depth_pbs_shift_selected_chunks_current_plus_new12_plot_data.csv"
    )
    depth = depth.loc[depth["n_chunks"].eq(32)].copy()
    tables = {
        "points": points,
        "points_expanded_support_sensitivity": points_expanded,
        "raw_summary": raw_summary,
        "raw_stats": raw_stats,
        "decomposition": decomposition,
        "gap": gap,
        "depth": depth,
    }
    paths = {
        key: write_csv(value, source_tables / f"ed03_{key}_current_source.csv")
        for key, value in tables.items()
    }
    audit = {
        "policy_scope": "current_plus_new12",
        "policy_session_counts": found_counts,
        "policy_support_rule": "n_Bf, n_Ba, n_Lf, and n_Lo all >= 10 in the first 1/32 Explore window",
        "expanded_policy_session_counts": points_expanded.groupby(
            "group", observed=True
        )
        .size()
        .to_dict(),
        "gap_current_plus_new12": current_values,
        "gap_uncertainty": "July-14 mouse-cluster bootstrap",
        "decomposition_scope": "current_plus_new12 n_naive=23 n_LTM=57",
        "decomposition_manifest": str(decomposition_manifest_path),
        "manuscript_values_note": (
            "Legacy manuscript wording 54%/85% should be refreshed to approximately "
            "52%/85%; these are diagnostic recapitulation ratios, not causal proportions."
        ),
    }
    return tables, paths, audit


def plot_ed03(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    points = tables["points"]
    rng = np.random.default_rng(20260713)
    groups = ["naive", "LTM", "experienced"]
    colors = [COLORS["naive_mark"], COLORS["return_mark"], COLORS["experienced"]]
    fig, axes = plt.subplots(1, 4, figsize=(7.01, 1.95), sharey=True)
    for ax, (metric, policy_label, description) in zip(axes, POLICY_SPECS):
        values_by_group = [
            points.loc[points["group"].eq(group), metric].to_numpy(dtype=float)
            for group in groups
        ]
        add_strip_summary(ax, values_by_group, colors, rng)
        reference = 0.5 if metric in {"Ba", "Lo"} else 2.0 / 3.0
        ax.axhline(reference, color="#A0A0A0", linewidth=0.55, linestyle="--", zorder=0)
        ax.set_xticks(range(3), ["Naive", "Return", "Second"], rotation=30, ha="right")
        ax.set_ylim(0.25, 1.02)
        ax.set_title(f"{policy_label}  {description}", loc="left", pad=2.0)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    axes[0].set_ylabel("Observed branch-choice probability")
    fig.text(
        0.995,
        0.015,
        "All four policy supports >=10; n=21 naive, 55 return, 72 second-day sessions",
        ha="right",
        fontsize=4.4,
        color=COLORS["legacy"],
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.28, top=0.88, wspace=0.22)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed03a_observed_branch_policies_current_plus_new12"
        )
    )

    decomposition = tables["decomposition"].copy()
    order = ["Bf", "Ba", "Lf", "Lo"]
    decomposition["parameter"] = pd.Categorical(
        decomposition["parameter"], order, ordered=True
    )
    decomposition = decomposition.sort_values("parameter")
    fig, ax = plt.subplots(figsize=(3.2, 2.05))
    x = np.arange(len(decomposition))
    values = decomposition["delta_p_mean"].to_numpy(dtype=float)
    low = values - decomposition["delta_p_ci_lo"].to_numpy(dtype=float)
    high = decomposition["delta_p_ci_hi"].to_numpy(dtype=float) - values
    bar_colors = [COLORS["neutral"] if value >= 0 else "#B95A58" for value in values]
    ax.bar(x, values, color=bar_colors, width=0.65, alpha=0.9)
    ax.errorbar(
        x,
        values,
        yerr=np.vstack([low, high]),
        fmt="none",
        color=COLORS["axis"],
        capsize=1.8,
        elinewidth=0.7,
    )
    ax.axhline(0, color=COLORS["axis"], linewidth=0.65)
    ax.set_xticks(
        x,
        [
            PARAMETER_TO_POLICY[str(parameter)]
            for parameter in decomposition["parameter"]
        ],
    )
    ax.set_ylabel(r"Simulated $Delta$ exploration efficiency")
    ax.set_title("Single-policy contribution", loc="left", fontweight="bold", pad=2.0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.88)
    outputs.extend(
        save_figure(fig, output_dir / "ed03b_policy_contributions_current_plus_new12")
    )

    gap = tables["gap"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    model_order = ["four_bias", "depth_stratified"]
    x = np.arange(2, dtype=float)
    subset = gap.set_index("model").reindex(model_order)
    mean = subset["percent_explained_mean"].to_numpy(dtype=float)
    low = mean - subset["percent_explained_ci_lo"].to_numpy(dtype=float)
    high = subset["percent_explained_ci_hi"].to_numpy(dtype=float) - mean
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([low, high]),
        fmt="o",
        markerfacecolor=COLORS["return_mark"],
        markeredgecolor=COLORS["return_mark"],
        color=COLORS["return_mark"],
        markersize=4.0,
        markeredgewidth=0.8,
        capsize=2.0,
        elinewidth=0.8,
    )
    for x_value, y_value in zip(x, mean):
        ax.text(
            x_value,
            y_value + 5.5,
            f"{y_value:.0f}%",
            ha="center",
            fontsize=4.7,
            color=COLORS["return_mark"],
        )
    ax.axhline(100, color="#A0A0A0", linewidth=0.55, linestyle="--")
    ax.set_xticks(x, ["Four policies", r"Depth-varying $P_{BS}$"])
    ax.set_ylabel("Observed efficiency gap recapitulated (%)")
    ax.set_ylim(20, 115)
    ax.set_title(
        "Policy-swap recapitulation diagnostic", loc="left", fontweight="bold", pad=2.0
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.text(
        0.97,
        0.015,
        "Mouse-cluster bootstrap CI; n=21 naive, 57 return sessions; not a causal proportion",
        ha="right",
        fontsize=4.4,
        color=COLORS["legacy"],
    )
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.25, top=0.88)
    outputs.extend(
        save_figure(fig, output_dir / "ed03c_coverage_gap_explained_current_plus_new12")
    )

    depth = (
        tables["depth"]
        .set_index("stratum")
        .reindex(["Lo_shallow", "Lo_mid", "Lo_deep"])
    )
    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    values = depth["delta_mean"].to_numpy(dtype=float)
    low = values - depth["delta_ci_lo"].to_numpy(dtype=float)
    high = depth["delta_ci_hi"].to_numpy(dtype=float) - values
    ax.errorbar(
        np.arange(3),
        values,
        yerr=np.vstack([low, high]),
        color=COLORS["return_mark"],
        marker="o",
        markersize=4.0,
        markeredgecolor="white",
        markeredgewidth=0.4,
        capsize=2.0,
        elinewidth=0.8,
    )
    ax.axhline(0, color="#A0A0A0", linewidth=0.55, linestyle="--")
    ax.set_xticks(
        [0, 1, 2], ["Shallow\n(levels 1-2)", "Middle\n(levels 3-4)", "Deep\n(level 5)"]
    )
    ax.set_ylabel(r"Return − naive $P_{BS}$")
    ax.set_title(
        "Stemward-exit shift by tree depth", loc="left", fontweight="bold", pad=2.0
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.25, top=0.88)
    outputs.extend(
        save_figure(fig, output_dir / "ed03d_depth_varying_pbs_current_plus_new12")
    )
    return outputs


def prepare_ed04(
    source_tables: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, Any]]:
    effects = pd.read_csv(
        CURRENT_ROOT / "latent_return_junction_claim_effect_current_plus_new12.csv"
    )
    effects = effects.loc[effects["primary_claim"].notna()].copy()
    expected_claims = {
        "broad_pre_reward_explore",
        "pre_reward_direct_chain_specificity_check",
        "reward_approach_after_reward",
    }
    if set(effects["primary_claim"]) != expected_claims:
        raise ValueError(
            "ED4 effect table does not contain the corrected three-claim family"
        )
    sessions = pd.read_csv(
        CURRENT_ROOT / "latent_return_junction_claim_session_current_plus_new12.csv"
    )
    sessions = sessions.loc[sessions["stratum"].isin(effects["stratum"])].copy()
    animal_dwell = sessions.groupby(
        ["stratum", "condition", "animal"], as_index=False
    ).agg(
        mean_decision_time_s=("mean_decision_time_s", "mean"),
        n_sessions=("session_id", "nunique"),
    )

    old_route = pd.read_csv(ROUTE_ROOT / "reward_path_first_turns.csv")
    old_route["source_set"] = "current_saved"
    new_route = pd.read_csv(
        NEW12_ROOT / "reward_sequence_memory_new12_reward_path_first_turns.csv"
    )
    new_route["source_set"] = "new12_saved"
    overlap = set(old_route["session_id"]).intersection(set(new_route["session_id"]))
    if overlap:
        raise ValueError(f"Current and new12 route sources overlap: {sorted(overlap)}")
    route_events = pd.concat([old_route, new_route], ignore_index=True, sort=False)
    if len(route_events) != 972:
        raise ValueError(
            f"Expected 972 current-plus-new12 reward-path events; found {len(route_events)}"
        )
    route_events["animal"] = route_events["animal"].astype(str)

    route_events["reward_correct"] = route_events["reward_correct"].map(
        parse_required_binary
    )
    if "choice_origin" not in route_events:
        route_events["choice_origin"] = "observed"
    route_events["choice_origin"] = route_events["choice_origin"].fillna("").astype(str)
    route_events.loc[route_events["choice_origin"].eq(""), "choice_origin"] = "observed"
    allowed_origins = {"observed", "inferred"}
    invalid_origins = sorted(set(route_events["choice_origin"]) - allowed_origins)
    if invalid_origins:
        raise ValueError(f"Unexpected route choice origins: {invalid_origins}")
    route_events["choice_inferred"] = route_events["choice_origin"].eq("inferred")
    if "directly_observed" not in route_events:
        route_events["directly_observed"] = ~route_events["choice_inferred"]
    else:
        missing_observation_flag = route_events["directly_observed"].isna()
        route_events.loc[missing_observation_flag, "directly_observed"] = ~route_events.loc[
            missing_observation_flag, "choice_inferred"
        ]
        route_events["directly_observed"] = route_events["directly_observed"].map(
            parse_required_binary
        )
    if (route_events["directly_observed"] == route_events["choice_inferred"]).any():
        raise ValueError("Every scored route choice must be exactly one of observed or inferred")
    session_route = route_events.groupby(
        ["session_id", "animal", "condition", "condition_label"], as_index=False
    ).agg(
        route_fraction_correct=("reward_correct", "mean"),
        n_depths=("paper_depth", "nunique"),
        n_decisions_observed=("directly_observed", "sum"),
        n_decisions_inferred=("choice_inferred", "sum"),
    )
    session_route["n_decisions_scored"] = session_route["n_depths"]
    session_route["all6_observed"] = session_route["n_decisions_observed"].eq(6)
    session_route["all6_resolved"] = session_route["n_decisions_scored"].eq(6)
    if not session_route["all6_resolved"].all():
        unresolved = session_route.loc[~session_route["all6_resolved"], "session_id"].tolist()
        raise ValueError(f"Unresolved current-plus-new12 route sessions: {unresolved}")
    provenance_counts = route_events["choice_origin"].value_counts().to_dict()
    if provenance_counts != {"observed": 970, "inferred": 2}:
        raise ValueError(f"Unexpected route-choice provenance counts: {provenance_counts}")
    reconstructed = session_route.loc[session_route["n_decisions_inferred"].gt(0)]
    if len(reconstructed) != 1 or reconstructed.iloc[0]["session_id"] != "489-3day1":
        raise ValueError("Direct-entry reconstruction must be confined to 489-3day1")
    if not (
        int(reconstructed.iloc[0]["n_decisions_observed"]) == 4
        and int(reconstructed.iloc[0]["n_decisions_inferred"]) == 2
        and math.isclose(float(reconstructed.iloc[0]["route_fraction_correct"]), 4.0 / 6.0)
    ):
        raise ValueError("489-3day1 direct-entry reconstruction does not resolve to 4/6")
    animal_route = session_route.groupby(
        ["animal", "condition", "condition_label"], as_index=False
    ).agg(
        route_fraction_correct=("route_fraction_correct", "mean"),
        n_sessions=("session_id", "nunique"),
    )
    route_counts = (
        session_route.groupby("condition", observed=True)["session_id"]
        .nunique()
        .to_dict()
    )
    expected_route_counts = {
        "experienced_day2": 74,
        "latent_day2": 9,
        "naive": 23,
        "return": 56,
    }
    if route_counts != expected_route_counts:
        raise ValueError(
            f"Unexpected current-plus-new12 route session counts: {route_counts}"
        )

    animal_depth = route_events.groupby(
        [
            "animal",
            "condition",
            "condition_label",
            "paper_depth",
            "required_transition_type",
        ],
        as_index=False,
    ).agg(
        reward_correct=("reward_correct", "mean"), n_sessions=("session_id", "nunique")
    )
    depth_summary = animal_depth.groupby(
        ["condition", "condition_label", "paper_depth", "required_transition_type"],
        as_index=False,
    ).agg(
        n_animals=("animal", "nunique"),
        mean_reward_correct=("reward_correct", "mean"),
        sem_reward_correct=("reward_correct", sem),
    )
    transition_animal = route_events.groupby(
        ["animal", "condition", "condition_label", "required_transition_type"],
        as_index=False,
    ).agg(
        mean_reward_correct=("reward_correct", "mean"),
        n_sessions=("session_id", "nunique"),
        n_decisions=("reward_correct", "size"),
    )
    transition_tests: list[dict[str, Any]] = []
    for transition in ["alternate_required", "repeat_required"]:
        subset = transition_animal.loc[
            transition_animal["required_transition_type"].eq(transition)
            & transition_animal["condition"].isin(["return", "latent_day2"])
        ]
        return_values = pd.to_numeric(
            subset.loc[subset["condition"].eq("return"), "mean_reward_correct"],
            errors="coerce",
        ).to_numpy(dtype=float)
        latent_values = pd.to_numeric(
            subset.loc[subset["condition"].eq("latent_day2"), "mean_reward_correct"],
            errors="coerce",
        ).to_numpy(dtype=float)
        test = stats.mannwhitneyu(return_values, latent_values, alternative="two-sided")
        transition_tests.append(
            {
                "transition_type": transition,
                "comparison": "return_vs_latent_day2",
                "n_return_animals": len(return_values),
                "n_latent_animals": len(latent_values),
                "mean_return": float(np.mean(return_values)),
                "mean_latent": float(np.mean(latent_values)),
                "mean_difference": float(
                    np.mean(return_values) - np.mean(latent_values)
                ),
                "mannwhitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    transition_stats = pd.DataFrame(transition_tests)
    transition_stats["q_bh2"] = bh_q(transition_stats["p_value"])

    route_stats, route_group_stats = four_level_cluster_route_stats(
        session_route,
    )
    second_latent = route_stats.loc[
        route_stats["comparison"].eq("experienced_day2_vs_latent_day2")
    ].iloc[0]
    if not (
        math.isclose(float(second_latent["mean_diff_target_minus_baseline"]), 0.162162162162162, rel_tol=1e-10)
        and math.isclose(float(second_latent["q_bh6"]), 0.0435439412958498, rel_tol=1e-8)
    ):
        raise ValueError(
            "Current ED4 Second-versus-Latent route contrast drifted from the locked BH6 model"
        )
    route_definition = pd.read_csv(ROUTE_ROOT / "reward_path_definition.csv")
    depth_small_multiples = depth_summary.merge(
        route_definition[
            [
                "paper_depth",
                "tree_depth",
                "parent_node",
                "reward_child",
                "reward_side",
                "required_transition_type",
            ]
        ],
        on=["paper_depth", "required_transition_type"],
        how="left",
        validate="many_to_one",
    )
    depth_small_multiples["panel_order"] = depth_small_multiples[
        "condition"
    ].map({group: index for index, group in enumerate(ROUTE_DEPTH_PANEL_ORDER)})
    depth_small_multiples["display_label"] = depth_small_multiples[
        "condition"
    ].map(ROUTE_DEPTH_PANEL_LABELS)
    depth_small_multiples["n_sessions"] = (
        depth_small_multiples["condition"].map(route_counts).astype(int)
    )
    depth_small_multiples["mean_definition"] = (
        "equal-animal mean of within-animal condition/depth first-visit correctness"
    )
    depth_small_multiples["interval_definition"] = (
        "SEM across animal-condition depth rates"
    )
    depth_small_multiples = depth_small_multiples[
        [
            "panel_order",
            "condition",
            "display_label",
            "paper_depth",
            "tree_depth",
            "parent_node",
            "reward_child",
            "reward_side",
            "required_transition_type",
            "n_sessions",
            "n_animals",
            "mean_reward_correct",
            "sem_reward_correct",
            "mean_definition",
            "interval_definition",
        ]
    ].sort_values(["panel_order", "tree_depth"])
    expected_depth_support = {
        "naive": (23, 23),
        "experienced_day2": (74, 24),
        "return": (56, 17),
        "latent_day2": (9, 9),
    }
    observed_depth_support = {
        group: (
            int(subset["n_sessions"].iloc[0]),
            int(subset["n_animals"].iloc[0]),
        )
        for group, subset in depth_small_multiples.groupby(
            "condition", observed=True
        )
    }
    if observed_depth_support != expected_depth_support:
        raise ValueError(
            "Unexpected ED4 depth-panel sessions/mice support: "
            f"{observed_depth_support}"
        )
    if len(depth_small_multiples) != 24 or any(
        set(subset["tree_depth"].astype(int)) != set(range(6))
        for _, subset in depth_small_multiples.groupby("condition", observed=True)
    ):
        raise ValueError("ED4 depth small-multiple table is not four complete depths 0-5")

    tables = {
        "effects": effects,
        "animal_dwell": animal_dwell,
        "route_events": route_events,
        "session_route": session_route,
        "animal_route": animal_route,
        "animal_depth": animal_depth,
        "depth_summary": depth_summary,
        "depth_small_multiples": depth_small_multiples,
        "transition_animal": transition_animal,
        "transition_stats": transition_stats,
        "route_stats": route_stats,
        "route_group_stats": route_group_stats,
        "route_definition": route_definition,
    }
    paths = {
        key: write_csv(value, source_tables / f"ed04_{key}_current_plus_new12.csv")
        for key, value in tables.items()
    }
    audit = {
        "latent_return_scope": "current_plus_new12 corrected three-claim family",
        "route_session_counts": route_counts,
        "route_animal_counts": animal_route.groupby("condition")["animal"]
        .nunique()
        .to_dict(),
        "pre_reward_primary_q": effects.set_index("primary_claim")[
            "bh_q_primary_3_claims"
        ].to_dict(),
        "route_inference": {
            "model": "four-level session OLS; CR1 covariance clustered by mouse; Student t(df=32)",
            "correction": "BH across all six two-sided pairwise condition contrasts",
            "choice_provenance_counts": provenance_counts,
            "direct_entry_reconstruction": {
                "session_id": "489-3day1",
                "assumption": "direct home to first valid location",
                "first_valid_run_node": 16,
                "first_valid_abs_frame": 378,
                "run_path": [0, 1, 3, 7, 16],
                "resolved_score": "4/6",
                "directly_observed_choices": 4,
                "inferred_choices": 2,
            },
            "supported_comparisons": route_stats.loc[
                route_stats["supported_bh6"], "comparison"
            ].tolist(),
        },
        "route_depth_small_multiples": {
            "display_axis": "reward-path parent tree depth 0-5",
            "mean": "equal-animal mean of within-animal condition/depth first-visit correctness",
            "interval": "SEM across animal-condition depth rates",
            "support_sessions_mice": observed_depth_support,
        },
        "interpretation_lock": (
            "Pre-reward broad dwell is borderline after correction; direct-minus-other specificity is "
            "not supported; post-reward direct approach is robust; route and alternation panels are "
            "negative/specificity controls, not evidence for an exact motor sequence."
        ),
    }
    return tables, paths, audit


def add_route_score_bh6_brackets(ax: plt.Axes, route_stats: pd.DataFrame) -> None:
    bracket_specs = (
        ("experienced_day2_vs_return", 1, 2, 1.025),
        ("experienced_day2_vs_latent_day2", 2, 3, 1.075),
        ("experienced_day2_vs_naive", 0, 2, 1.125),
    )
    expected = {comparison for comparison, *_ in bracket_specs}
    supported = set(
        route_stats.loc[route_stats["q_bh6"].lt(0.05), "comparison"].astype(str)
    )
    if supported != expected:
        raise ValueError(
            f"Unexpected supported ED4 route comparisons: {sorted(supported)}"
        )
    lookup = route_stats.set_index("comparison")
    for comparison, x_left, x_right, y_base in bracket_specs:
        q_value = float(lookup.loc[comparison, "q_bh6"])
        stars = "***" if q_value < 0.001 else "**" if q_value < 0.01 else "*"
        height = 0.014
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_base, y_base + height, y_base + height, y_base],
            color=COLORS["axis"],
            linewidth=0.6,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            (x_left + x_right) / 2,
            y_base + height + 0.005,
            stars,
            ha="center",
            va="bottom",
            fontsize=5.2,
            fontweight="bold",
            color=COLORS["axis"],
            clip_on=False,
            zorder=6,
        )


def plot_ed04(tables: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    rng = np.random.default_rng(20260714)
    effects = tables["effects"].set_index("primary_claim")
    animal_dwell = tables["animal_dwell"]

    pre_specs = [
        ("broad_pre_reward_explore", "All internal junctions", "Mean dwell (s)"),
        (
            "pre_reward_direct_chain_specificity_check",
            "Direct − other junctions",
            "Dwell contrast (s)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.0))
    for ax, (claim, title, ylabel) in zip(axes, pre_specs):
        effect = effects.loc[claim]
        subset = animal_dwell.loc[animal_dwell["stratum"].eq(effect["stratum"])]
        values = [
            subset.loc[
                subset["condition"].eq(condition), "mean_decision_time_s"
            ].to_numpy(dtype=float)
            for condition in ["latent", "return"]
        ]
        add_strip_summary(ax, values, [COLORS["latent"], COLORS["return_mark"]], rng)
        ax.set_xticks([0, 1], ["Latent day 2", "Return day"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=2.0)
        annotation = (
            f"β={effect['animal_balanced_beta_return_minus_latent_s']:.3f} s\n"
            f"clustered {p_text(effect['animal_balanced_cluster_p'])}; "
            f"q={effect['bh_q_primary_3_claims']:.3f}"
        )
        ax.text(
            0.98,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.7},
            zorder=5,
        )
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.text(
        0.01,
        0.01,
        "Pre-reward exploratory estimates: borderline / not significant after the three-claim correction.",
        ha="left",
        fontsize=4.5,
        color=COLORS["legacy"],
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.88, wspace=0.35)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed04a_pre_reward_dwell_borderline_current_plus_new12"
        )
    )

    claim = "reward_approach_after_reward"
    effect = effects.loc[claim]
    subset = animal_dwell.loc[animal_dwell["stratum"].eq(effect["stratum"])]
    values = [
        subset.loc[subset["condition"].eq(condition), "mean_decision_time_s"].to_numpy(
            dtype=float
        )
        for condition in ["latent", "return"]
    ]
    fig, ax = plt.subplots(figsize=(2.35, 1.95))
    add_strip_summary(ax, values, [COLORS["latent"], COLORS["return_mark"]], rng)
    ax.set_xticks([0, 1], ["Latent day 2", "Return day"])
    ax.set_ylabel("Direct-path junction dwell (s)")
    ax.set_title("Post-port direct approach", loc="left", fontweight="bold", pad=2.0)
    ax.text(
        0.98,
        0.97,
        f"β={effect['animal_balanced_beta_return_minus_latent_s']:.3f} s\n"
        f"clustered {p_text(effect['animal_balanced_cluster_p'])}; q={effect['bh_q_primary_3_claims']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.22, top=0.88)
    outputs.extend(
        save_figure(
            fig, output_dir / "ed04b_post_port_direct_approach_current_plus_new12"
        )
    )

    definition = tables["route_definition"].sort_values("paper_depth")
    fig, ax = plt.subplots(figsize=(5.1, 1.35))
    x = np.arange(len(definition) + 1)
    path_nodes = [
        int(definition.iloc[0]["parent_node"]),
        *definition["reward_child"].astype(int).tolist(),
    ]
    ax.plot(x, np.zeros_like(x), color=COLORS["axis"], linewidth=1.1, zorder=1)
    ax.scatter(
        x,
        np.zeros_like(x),
        s=32,
        color=["white"] * (len(x) - 1) + [COLORS["return_mark"]],
        edgecolor=COLORS["axis"],
        linewidth=0.7,
        zorder=2,
    )
    for index, node in enumerate(path_nodes):
        ax.text(index, 0, str(node), ha="center", va="center", fontsize=4.6, zorder=3)
    for index, row in definition.iterrows():
        transition = (
            "start"
            if row["required_transition_type"] == "start"
            else (
                "alternate"
                if row["required_transition_type"] == "alternate_required"
                else "repeat"
            )
        )
        ax.text(
            index + 0.5,
            0.13,
            str(row["reward_side"]).upper(),
            ha="center",
            va="bottom",
            fontsize=5.1,
            fontweight="bold",
        )
        ax.text(
            index + 0.5,
            -0.14,
            transition,
            ha="center",
            va="top",
            fontsize=4.3,
            color=COLORS["legacy"],
        )
    ax.text(
        len(x) - 1,
        0.2,
        "Water port",
        ha="center",
        va="bottom",
        fontsize=4.8,
        color=COLORS["return_mark"],
    )
    ax.set_xlim(-0.35, len(x) - 0.65)
    ax.set_ylim(-0.28, 0.32)
    ax.axis("off")
    ax.set_title(
        "Six first-visit reward-path decisions (L-R-R-L-R-L)",
        loc="left",
        fontweight="bold",
        pad=1.0,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.82)
    outputs.extend(
        save_figure(fig, output_dir / "ed04c_reward_path_first_visit_schematic")
    )

    depth = tables["depth_summary"]
    fig, ax = plt.subplots(figsize=(3.8, 2.25))
    group_colors = {
        "naive": COLORS["naive_mark"],
        "return": COLORS["return_mark"],
        "experienced_day2": COLORS["experienced"],
        "latent_day2": COLORS["latent"],
    }
    for group in ROUTE_GROUP_ORDER:
        subset = depth.loc[depth["condition"].eq(group)].sort_values("paper_depth")
        ax.errorbar(
            subset["paper_depth"],
            subset["mean_reward_correct"],
            yerr=subset["sem_reward_correct"],
            color=group_colors[group],
            marker="o",
            markersize=2.8,
            capsize=1.4,
            elinewidth=0.65,
            label=f"{GROUP_LABELS[group]} (n={int(subset['n_animals'].max())})",
        )
    ax.axhline(0.5, color="#A0A0A0", linewidth=0.55, linestyle="--")
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Reward-path decision depth")
    ax.set_ylabel("First choice toward reward")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        "Depth-resolved first-visit route choices",
        loc="left",
        fontweight="bold",
        pad=2.0,
    )
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handletextpad=0.3)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.2, top=0.88)
    outputs.extend(
        save_figure(fig, output_dir / "ed04d_route_depth_current_plus_new12")
    )

    depth_panels = tables["depth_small_multiples"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharey=True)
    for index, (ax, group) in enumerate(zip(axes, ROUTE_DEPTH_PANEL_ORDER)):
        subset = depth_panels.loc[
            depth_panels["condition"].eq(group)
        ].sort_values("tree_depth")
        if len(subset) != 6:
            raise ValueError(f"Expected six ED4 depth bars for {group}; found {len(subset)}")
        x = subset["tree_depth"].to_numpy(dtype=int)
        values = subset["mean_reward_correct"].to_numpy(dtype=float)
        errors = subset["sem_reward_correct"].to_numpy(dtype=float)
        color = ROUTE_DEPTH_PANEL_COLORS[group]
        ax.axhline(0.5, color="#A0A0A0", linewidth=0.5, linestyle="--", zorder=0)
        ax.bar(x, values, width=0.76, color=color, alpha=0.92, zorder=2)
        ax.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            color=COLORS["axis"],
            capsize=1.4,
            elinewidth=0.6,
            capthick=0.6,
            zorder=3,
        )
        ax.set_xticks(range(6))
        ax.set_xlim(-0.55, 5.55)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_xlabel(
            ROUTE_DEPTH_PANEL_LABELS[group],
            color=color,
            fontweight="bold",
            labelpad=4.0,
        )
        ax.text(
            0.5,
            -0.29,
            f"{int(subset['n_sessions'].iloc[0])} sessions / "
            f"{int(subset['n_animals'].iloc[0])} mice",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.5,
            color=COLORS["legacy"],
        )
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.35)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel("First choice toward reward")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(
        "First-visit reward-path choices by parent depth",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=6.4,
        fontweight="bold",
    )
    fig.text(
        0.99,
        0.975,
        "Equal-animal mean ± SEM",
        ha="right",
        va="top",
        fontsize=4.6,
        color=COLORS["legacy"],
    )
    fig.text(
        0.535,
        0.02,
        "Reward-path parent-node tree depth",
        ha="center",
        fontsize=5.2,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.30,
        top=0.86,
        wspace=0.18,
    )
    outputs.extend(
        save_figure(
            fig,
            output_dir
            / "ed04g_route_depth_small_multiples_current_plus_new12",
        )
    )

    session_route = tables["session_route"]
    route_group_stats = tables["route_group_stats"].set_index("condition")
    route_stats = tables["route_stats"]
    route_rng = np.random.default_rng(20260716)
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    tick_labels: list[str] = []
    for index, group in enumerate(ROUTE_GROUP_ORDER):
        values = session_route.loc[
            session_route["condition"].eq(group), "route_fraction_correct"
        ].to_numpy(dtype=float)
        color = group_colors[group]
        jitter = route_rng.uniform(-0.14, 0.14, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=7,
            facecolor="white",
            edgecolor=color,
            linewidth=0.42,
            alpha=0.48,
            zorder=2,
        )
        summary = route_group_stats.loc[group]
        mean = float(summary["mean_route_fraction_correct"])
        ci_low = float(summary["cluster_ci95_low"])
        ci_high = float(summary["cluster_ci95_high"])
        ax.errorbar(
            index,
            mean,
            yerr=np.array([[mean - ci_low], [ci_high - mean]]),
            color=color,
            marker="o",
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=1.8,
            elinewidth=0.8,
            zorder=4,
        )
        tick_labels.append(
            f"{GROUP_LABELS[group].replace(' day', '').replace(' 2', '')}\n"
            f"{int(summary['n_sessions'])}/{int(summary['n_animals'])}"
        )
    ax.axhline(0.5, color="#A0A0A0", linewidth=0.55, linestyle="--")
    ax.set_xticks(range(4), tick_labels)
    ax.set_xlabel("Condition (sessions/mice)")
    ax.set_ylabel("First-visit route score")
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0.22, 1.18)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_title(
        "No precise route recovery on return", loc="left", fontweight="bold", pad=2.0
    )
    add_route_score_bh6_brackets(ax, route_stats)
    ax.text(
        0.98,
        0.025,
        "CR1 clustered by mouse; BH across 6 contrasts\nonly q < 0.05 shown",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.2,
        color=COLORS["legacy"],
    )
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.88)
    outputs.extend(
        save_figure(fig, output_dir / "ed04e_route_score_current_plus_new12")
    )

    transition = tables["transition_animal"]
    transition_stats = tables["transition_stats"].set_index("transition_type")
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0), sharey=True)
    for ax, transition_type, title in zip(
        axes,
        ["alternate_required", "repeat_required"],
        ["Alternation-compatible choices", "Repeat-required choice"],
    ):
        subset = transition.loc[
            transition["required_transition_type"].eq(transition_type)
            & transition["condition"].isin(["latent_day2", "return"])
        ]
        values = [
            subset.loc[subset["condition"].eq(group), "mean_reward_correct"].to_numpy(
                dtype=float
            )
            for group in ["latent_day2", "return"]
        ]
        add_strip_summary(ax, values, [COLORS["latent"], COLORS["return_mark"]], rng)
        ax.axhline(0.5, color="#A0A0A0", linewidth=0.55, linestyle="--")
        ax.set_xticks([0, 1], ["Latent day 2", "Return day"])
        stat_row = transition_stats.loc[transition_type]
        ax.text(
            0.98,
            0.97,
            f"MW {p_text(stat_row['p_value'])}\nBH2 q={stat_row['q_bh2']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.7,
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=2.0)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
    axes[0].set_ylabel("First choice toward reward")
    axes[0].set_ylim(-0.03, 1.05)
    fig.text(
        0.99,
        0.01,
        "The repeat-required estimate contains one route depth; the alternation-compatible subset is the primary negative control.",
        ha="right",
        fontsize=4.4,
        color=COLORS["legacy"],
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.88, wspace=0.2)
    outputs.extend(
        save_figure(
            fig,
            output_dir / "ed04f_alternation_latent_negative_control_current_plus_new12",
        )
    )
    return outputs


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    configure_style()
    refresh_root = args.refresh_root.resolve()
    input_refresh_root = Path(
        getattr(args, "input_refresh_root", DEFAULT_REFRESH_ROOT)
    ).resolve()
    source_tables = refresh_root / "source_tables"
    source_tables.mkdir(parents=True, exist_ok=True)
    output_dirs = {
        2: refresh_root / "source_rendered_new" / "extended_data_fig_02",
        3: refresh_root / "source_rendered_new" / "extended_data_fig_03",
        4: refresh_root / "source_rendered_new" / "extended_data_fig_04",
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    ed02_tables, ed02_table_paths, ed02_audit = prepare_ed02(
        source_tables,
        input_refresh_root=input_refresh_root,
    )
    ed03_tables, ed03_table_paths, ed03_audit = prepare_ed03(
        source_tables,
        input_refresh_root=input_refresh_root,
    )
    ed04_tables, ed04_table_paths, ed04_audit = prepare_ed04(source_tables)
    ed02_outputs = plot_ed02(ed02_tables, output_dirs[2])
    ed03_outputs = plot_ed03(ed03_tables, output_dirs[3])
    ed04_outputs = plot_ed04(ed04_tables, output_dirs[4])

    style = {
        "backend": "Python/matplotlib only",
        "font": "Arial",
        "base_font_pt": 6.0,
        "tick_font_pt": 5.4,
        "axes_linewidth_pt": 0.65,
        "raster_dpi": 600,
        "svg_text_editable": True,
        "pdf_fonttype": 42,
        "formats": list(FORMATS),
    }
    manifests: dict[int, Path] = {}
    figure_payloads = {
        2: {
            "core_conclusion": (
                "First-long-bout-matched tile transitions decrease across returns, while repeated "
                "end-node coverage becomes less redundant; speed changes do not substitute for the "
                "count-based traversal control."
            ),
            "scope_audit": ed02_audit,
            "source_tables": [str(path) for path in ed02_table_paths.values()],
            "outputs": [str(path) for path in ed02_outputs],
            "review_risk": (
                "The six appended 12-month sessions are included only after exact recovered-frame "
                "lb0/slb0 parity; the parity audit must travel with this 153-session panel."
            ),
        },
        3: {
            "core_conclusion": (
                "On the Main-Figure-matched support gate, current-plus-new12 local branch policies "
                "recapitulate about 52% of the observed coverage gap, rising to about 85% when the "
                "stemward-exit policy varies by depth. These are diagnostic ratios, while the "
                "single-policy decomposition is a broader-scope descriptive sensitivity."
            ),
            "scope_audit": ed03_audit,
            "source_tables": [str(path) for path in ed03_table_paths.values()],
            "outputs": [str(path) for path in ed03_outputs],
            "review_risk": (
                "Observed policy contrasts use the all-four-policy support gate (n=21 naive / 55 "
                "return); the simulator gap uses its depth-complete n=21 / 57 scope, and the "
                "single-policy decomposition uses n=23 / 57. Keep these estimands distinct."
            ),
        },
        4: {
            "core_conclusion": (
                "Return animals show a robust post-port direct-approach advantage, but the latent-"
                "controlled pre-reward estimate is borderline and first-visit choices do not support "
                "recovery of a precise fixed turn sequence."
            ),
            "scope_audit": ed04_audit,
            "source_tables": [str(path) for path in ed04_table_paths.values()],
            "outputs": [str(path) for path in ed04_outputs],
            "review_risk": (
                "Route correctness is partly alternation-compatible; the latent comparison is a "
                "negative specificity control and must not be interpreted as exact route memory."
            ),
        },
    }
    for figure, payload in figure_payloads.items():
        payload["script"] = str(Path(__file__).resolve())
        payload["style"] = style
        manifest_path = source_tables / f"ed0{figure}_current_render_manifest.json"
        manifests[figure] = write_manifest(manifest_path, payload)
    master = {
        "script": str(Path(__file__).resolve()),
        "refresh_root": str(refresh_root),
        "style": style,
        "figure_manifests": {str(key): str(value) for key, value in manifests.items()},
        "all_outputs": [
            str(path) for path in [*ed02_outputs, *ed03_outputs, *ed04_outputs]
        ],
        "scope_policy": (
            "Current-plus-new12 validated tables are used for ED2 exact traversal, ED3 branch-policy "
            "decomposition, and all other primary panels. No audit composite was copied."
        ),
    }
    master_path = write_manifest(
        source_tables / "ed02_ed04_current_render_manifest.json", master
    )
    master["manifest_path"] = str(master_path)
    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-root", type=Path, default=DEFAULT_REFRESH_ROOT)
    parser.add_argument(
        "--input-refresh-root",
        type=Path,
        default=DEFAULT_REFRESH_ROOT,
        help="Read preserved ED2 currentness inputs from this root while writing to --refresh-root.",
    )
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
