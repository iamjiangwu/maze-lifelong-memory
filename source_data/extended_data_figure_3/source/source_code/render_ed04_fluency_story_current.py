"""Render the current ED4 pre/post fluency story from saved session rows.

The four displayed claims are deliberately aligned with the four panels:

1. broad exploration-mode junction fluency before first reward;
2. broad exploration-mode junction fluency after first reward;
3. pre-reward direct-minus-other junction dwell specificity;
4. post-first-reward reward-approach dwell on the direct reward chain.

Every eligible session row is retained. Inference uses CR1 covariance clustered
by mouse with Student-t calibration at G-1 degrees of freedom. Benjamini-
Hochberg correction is applied separately to the four raw-scale displayed
claims and to their four log1p specification checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    REPO_ROOT
    / "docs"
    / "behavior_panel_sensitivity_840_841_with_new12_baseline_2026-06-06"
    / "latent_return_junction_claim_session_current_plus_new12.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_04"
    / "ed04_fluency_story_2026-07-16"
)

FORMATS = ("png", "svg", "pdf", "tiff")
CONDITION_ORDER = ("latent", "return")
CONDITION_LABEL = {"latent": "Latent day2", "return": "Return"}
COLOR = {
    "latent": "#91877A",
    "return": "#C53B38",
    "axis": "#222222",
    "grid": "#D9D9D9",
    "neutral": "#6E7478",
}

CLAIMS = (
    {
        "claim_id": "ed4_broad_pre_reward_explore",
        "stratum": "pre_explore_all_junctions",
        "title": "Broad pre-reward fluency",
        "ylabel": "Mean junction dwell (s)",
        "stem": "ed04a_broad_pre_reward_fluency_current",
        "footer": "exploration mode; all internal junctions before first reward",
    },
    {
        "claim_id": "ed4_broad_post_reward_explore",
        "stratum": "all_post_explore_all_junctions",
        "title": "Broad post-reward fluency",
        "ylabel": "Mean junction dwell (s)",
        "stem": "ed04b_broad_post_reward_fluency_current",
        "footer": "exploration mode; all internal junctions after first reward",
    },
    {
        "claim_id": "ed4_pre_reward_direct_chain_specificity_check",
        "stratum": "pre_explore_direct_minus_other",
        "title": "Pre-reward path specificity",
        "ylabel": "Reward-path minus other dwell (s)",
        "stem": "ed04c_pre_reward_path_specificity_current",
        "footer": "exploration mode; reward-path minus other-junction dwell before first reward",
    },
    {
        "claim_id": "ed4_reward_approach_after_reward",
        "stratum": "all_post_reward_approach_direct_chain",
        "title": "Reward-path approaches after reward",
        "ylabel": "Mean reward-path junction dwell (s)",
        "stem": "ed04d_post_port_direct_approach_current",
        "footer": "reward-approach mode; reward-path junctions after first reward",
    },
)

EXPECTED_SCOPE = {
    "ed4_broad_pre_reward_explore": {
        "n_rows": 63,
        "n_mice": 20,
        "n_latent": 8,
        "n_return": 55,
    },
    "ed4_broad_post_reward_explore": {
        "n_rows": 65,
        "n_mice": 21,
        "n_latent": 9,
        "n_return": 56,
    },
    "ed4_pre_reward_direct_chain_specificity_check": {
        "n_rows": 61,
        "n_mice": 20,
        "n_latent": 8,
        "n_return": 53,
    },
    "ed4_reward_approach_after_reward": {
        "n_rows": 65,
        "n_mice": 21,
        "n_latent": 9,
        "n_return": 56,
    },
}


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.4,
            "ytick.labelsize": 5.4,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_q(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[finite[order]] = np.clip(adjusted, 0.0, 1.0)
    return q


def cluster_fit(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """CR1 sandwich fit with a t(G-1) reference distribution."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    if x.ndim == 1:
        x = x[:, None]
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    finite = (
        np.isfinite(y)
        & np.all(np.isfinite(x), axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    y, x, clusters, weights = y[finite], x[finite], clusters[finite], weights[finite]
    n, k = x.shape
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if n <= k or g < 3:
        raise ValueError(f"Cluster fit needs n>k and >=3 mice; got n={n}, k={k}, G={g}")
    xtwx_inv = np.linalg.pinv(x.T @ (weights[:, None] * x))
    beta = xtwx_inv @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = x[mask].T @ (weights[mask] * residual[mask])
        meat += np.outer(score, score)
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = scale * xtwx_inv @ meat @ xtwx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = beta / se
    p_two = 2.0 * stats.t.sf(np.abs(t_value), df=g - 1)
    return {
        "n_rows": int(n),
        "n_mice": int(g),
        "df": int(g - 1),
        "beta": beta,
        "se": se,
        "t": t_value,
        "p_two_sided": p_two,
        "covariance": covariance,
    }


def reconstruct_post_reward_broad(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_strata = (
        "all_post_explore_direct_chain",
        "all_post_explore_other_junctions",
    )
    components = source.loc[
        source["stratum"].isin(component_strata)
        & source["condition"].isin(CONDITION_ORDER)
    ].copy()
    for column in ("n_events", "mean_decision_time_s"):
        components[column] = pd.to_numeric(components[column], errors="coerce")
    components = components.dropna(
        subset=["condition", "animal", "session_id", "n_events", "mean_decision_time_s"]
    )
    if components.duplicated(["condition", "animal", "session_id", "stratum"]).any():
        raise AssertionError("Post-reward component table has duplicate session-stratum rows")
    components["weighted_dwell_sum_s"] = (
        components["n_events"] * components["mean_decision_time_s"]
    )
    keys = ["condition", "animal", "session_id"]
    combined = (
        components.groupby(keys, dropna=False)
        .agg(
            n_events=("n_events", "sum"),
            weighted_dwell_sum_s=("weighted_dwell_sum_s", "sum"),
            n_component_strata=("stratum", "nunique"),
            first_event_after_reward_s=("first_event_after_reward_s", "min"),
            last_event_after_reward_s=("last_event_after_reward_s", "max"),
        )
        .reset_index()
    )
    combined["mean_decision_time_s"] = (
        combined["weighted_dwell_sum_s"] / combined["n_events"]
    )
    component_wide = components.pivot(
        index=keys,
        columns="stratum",
        values=["n_events", "mean_decision_time_s"],
    ).reset_index()
    component_wide.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in component_wide.columns
    ]
    for key in keys:
        component_wide = component_wide.rename(columns={f"{key}__": key})
    audit = combined.merge(component_wide, on=keys, how="left", validate="one_to_one")
    if not combined["n_component_strata"].eq(2).all():
        bad = combined.loc[~combined["n_component_strata"].eq(2), keys + ["n_component_strata"]]
        raise AssertionError(f"Every broad post-reward row needs both path components:\n{bad}")
    direct = audit["mean_decision_time_s__all_post_explore_direct_chain"]
    other = audit["mean_decision_time_s__all_post_explore_other_junctions"]
    lower = np.minimum(direct, other)
    upper = np.maximum(direct, other)
    if not ((audit["mean_decision_time_s"] >= lower - 1e-12) & (audit["mean_decision_time_s"] <= upper + 1e-12)).all():
        raise AssertionError("Weighted broad post-reward means fall outside component bounds")
    combined["stratum"] = "all_post_explore_all_junctions"
    combined["claim_window"] = "all_post"
    combined["mode_label"] = "explore"
    combined["reward_epoch"] = "post_first_reward"
    combined["path_class"] = "all"
    combined["median_decision_time_s"] = np.nan
    return combined, audit


def prepare_claim_rows(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = source.copy()
    source["animal"] = source["animal"].astype(str)
    source["condition"] = source["condition"].astype(str)
    source["session_id"] = source["session_id"].astype(str)
    source["mean_decision_time_s"] = pd.to_numeric(
        source["mean_decision_time_s"], errors="coerce"
    )
    post_broad, audit = reconstruct_post_reward_broad(source)
    source = pd.concat([source, post_broad], ignore_index=True, sort=False)
    rows: list[pd.DataFrame] = []
    for claim in CLAIMS:
        subset = source.loc[
            source["stratum"].eq(claim["stratum"])
            & source["condition"].isin(CONDITION_ORDER)
        ].copy()
        subset = subset.dropna(
            subset=["mean_decision_time_s", "animal", "session_id"]
        )
        subset["claim_id"] = claim["claim_id"]
        subset["panel_title"] = claim["title"]
        subset["analysis_unit"] = "eligible session row"
        subset["included_for_displayed_primary"] = True
        subset = subset.sort_values(["condition", "animal", "session_id"])
        rows.append(subset)
    result = pd.concat(rows, ignore_index=True, sort=False)
    validate_scope(result)
    return result, audit


def validate_scope(rows: pd.DataFrame) -> None:
    for claim_id, expected in EXPECTED_SCOPE.items():
        subset = rows.loc[rows["claim_id"].eq(claim_id)]
        observed = {
            "n_rows": int(len(subset)),
            "n_mice": int(subset["animal"].nunique()),
            "n_latent": int(subset["condition"].eq("latent").sum()),
            "n_return": int(subset["condition"].eq("return").sum()),
        }
        if observed != expected:
            raise AssertionError(f"Unexpected current scope for {claim_id}: {observed} != {expected}")


def fit_claim(
    rows: pd.DataFrame,
    *,
    scale: str,
    inverse_session_weighting: bool = False,
) -> dict[str, Any]:
    work = rows.copy()
    raw = pd.to_numeric(work["mean_decision_time_s"], errors="coerce")
    if scale == "raw_seconds":
        work["analysis_value"] = raw
    elif scale == "log1p_seconds":
        work = work.loc[raw.ge(0)].copy()
        work["analysis_value"] = np.log1p(raw.loc[work.index])
    else:
        raise ValueError(scale)
    work = work.dropna(subset=["analysis_value", "condition", "animal"])
    indicator = work["condition"].eq("return").astype(float).to_numpy()
    x = np.column_stack([np.ones(len(work)), indicator])
    weights = None
    if inverse_session_weighting:
        counts = (
            work.groupby(["condition", "animal"])["session_id"]
            .transform("size")
            .astype(float)
        )
        weights = (1.0 / counts).to_numpy(dtype=float)
    fit = cluster_fit(
        work["analysis_value"].to_numpy(dtype=float),
        x,
        work["animal"].astype(str).to_numpy(),
        weights=weights,
    )
    estimate = float(fit["beta"][1])
    se = float(fit["se"][1])
    critical = float(stats.t.ppf(0.975, fit["df"]))
    return {
        "scale": scale,
        "comparison": "return_vs_latent",
        "n_rows": int(fit["n_rows"]),
        "n_mice": int(fit["n_mice"]),
        "n_return_rows": int(work["condition"].eq("return").sum()),
        "n_latent_rows": int(work["condition"].eq("latent").sum()),
        "n_return_mice": int(work.loc[work["condition"].eq("return"), "animal"].nunique()),
        "n_latent_mice": int(work.loc[work["condition"].eq("latent"), "animal"].nunique()),
        "estimate_return_minus_latent": estimate,
        "cluster_se": se,
        "ci95_low": estimate - critical * se,
        "ci95_high": estimate + critical * se,
        "t_cluster": float(fit["t"][1]),
        "df": int(fit["df"]),
        "p_cluster_two_sided": float(fit["p_two_sided"][1]),
        "weighting": "inverse_session_count" if inverse_session_weighting else "session_row_unit_weight",
    }


def build_claim_statistics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for claim in CLAIMS:
        subset = rows.loc[rows["claim_id"].eq(claim["claim_id"])]
        for scale in ("raw_seconds", "log1p_seconds"):
            primary = fit_claim(subset, scale=scale)
            sensitivity = fit_claim(
                subset,
                scale=scale,
                inverse_session_weighting=True,
            )
            records.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_label": claim["title"],
                    "stratum": claim["stratum"],
                    **primary,
                    "inverse_session_count_beta": sensitivity[
                        "estimate_return_minus_latent"
                    ],
                    "inverse_session_count_cluster_p": sensitivity[
                        "p_cluster_two_sided"
                    ],
                }
            )
    result = pd.DataFrame(records)
    result["q_bh4_displayed_story"] = np.nan
    result["inverse_session_count_bh4_q"] = np.nan
    for scale in ("raw_seconds", "log1p_seconds"):
        mask = result["scale"].eq(scale)
        if int(mask.sum()) != 4:
            raise AssertionError(f"Displayed {scale} family must contain exactly four claims")
        result.loc[mask, "q_bh4_displayed_story"] = bh_q(
            result.loc[mask, "p_cluster_two_sided"].to_numpy()
        )
        result.loc[mask, "inverse_session_count_bh4_q"] = bh_q(
            result.loc[mask, "inverse_session_count_cluster_p"].to_numpy()
        )
    return result


def build_interaction_tables(
    claim_rows: pd.DataFrame,
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "animal", "session_id"]
    claim_wide = claim_rows.pivot(
        index=keys,
        columns="claim_id",
        values="mean_decision_time_s",
    ).reset_index()
    same_path = source.loc[
        source["stratum"].eq("all_post_explore_direct_chain")
        & source["condition"].isin(CONDITION_ORDER),
        keys + ["mean_decision_time_s"],
    ].copy()
    same_path = same_path.rename(
        columns={"mean_decision_time_s": "post_reward_same_path_explore_s"}
    )
    wide = claim_wide.merge(same_path, on=keys, how="left", validate="one_to_one")
    pre = "ed4_broad_pre_reward_explore"
    post = "ed4_broad_post_reward_explore"
    approach = "ed4_reward_approach_after_reward"
    specs = (
        (
            "broad_post_minus_pre_x_condition",
            "Attenuation of broad fluency after reward",
            post,
            pre,
            "exact_interaction",
        ),
        (
            "approach_minus_broad_post_x_condition",
            "Visible-panel contrast: approach versus broad post-reward exploration",
            approach,
            post,
            "descriptive_mixed_mode_and_path",
        ),
        (
            "approach_minus_same_path_explore_x_condition",
            "Approach advantage versus same-path post-reward exploration",
            approach,
            "post_reward_same_path_explore_s",
            "exact_interaction",
        ),
    )
    session_records: list[pd.DataFrame] = []
    stat_records: list[dict[str, Any]] = []
    for interaction_id, label, high_col, low_col, family_role in specs:
        subset = wide[keys + [high_col, low_col]].dropna().copy()
        for scale in ("raw_seconds", "log1p_seconds"):
            if scale == "raw_seconds":
                subset["interaction_value"] = subset[high_col] - subset[low_col]
            else:
                subset["interaction_value"] = np.log1p(subset[high_col]) - np.log1p(
                    subset[low_col]
                )
            subset["interaction_id"] = interaction_id
            subset["interaction_label"] = label
            subset["scale"] = scale
            subset["family_role"] = family_role
            session_records.append(
                subset[
                    keys
                    + [
                        "interaction_id",
                        "interaction_label",
                        "scale",
                        "family_role",
                        high_col,
                        low_col,
                        "interaction_value",
                    ]
                ].rename(columns={high_col: "first_component", low_col: "second_component"})
            )
            fit_rows = subset.rename(
                columns={"interaction_value": "mean_decision_time_s"}
            )
            result = fit_claim(fit_rows, scale="raw_seconds")
            stat_records.append(
                {
                    "interaction_id": interaction_id,
                    "interaction_label": label,
                    "scale": scale,
                    "family_role": family_role,
                    **{key: value for key, value in result.items() if key != "scale"},
                }
            )
    stats_table = pd.DataFrame(stat_records)
    stats_table["q_bh2_exact_interactions"] = np.nan
    for scale in ("raw_seconds", "log1p_seconds"):
        mask = stats_table["scale"].eq(scale) & stats_table["family_role"].eq(
            "exact_interaction"
        )
        if int(mask.sum()) != 2:
            raise AssertionError(f"Exact {scale} interaction family must contain two rows")
        stats_table.loc[mask, "q_bh2_exact_interactions"] = bh_q(
            stats_table.loc[mask, "p_cluster_two_sided"].to_numpy()
        )
    return pd.concat(session_records, ignore_index=True), stats_table


def one_group_estimates(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        subset = rows.loc[rows["condition"].eq(condition)].copy()
        fit = cluster_fit(
            subset["mean_decision_time_s"].to_numpy(dtype=float),
            np.ones((len(subset), 1), dtype=float),
            subset["animal"].astype(str).to_numpy(),
        )
        mean = float(fit["beta"][0])
        se = float(fit["se"][0])
        critical = float(stats.t.ppf(0.975, fit["df"]))
        records.append(
            {
                "condition": condition,
                "mean": mean,
                "ci95_low": mean - critical * se,
                "ci95_high": mean + critical * se,
            }
        )
    return pd.DataFrame(records)


def deterministic_jitter(session_id: str, claim_id: str) -> float:
    value = int(
        hashlib.sha256(f"{claim_id}|{session_id}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    return ((value / 0xFFFFFFFF) - 0.5) * 0.20


def draw_claim_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    statistic: pd.Series,
    claim: dict[str, str],
    *,
    panel_label: str | None = None,
) -> None:
    estimates = one_group_estimates(rows).set_index("condition")
    for index, condition in enumerate(CONDITION_ORDER):
        subset = rows.loc[rows["condition"].eq(condition)]
        x = np.array(
            [
                index + deterministic_jitter(str(session), claim["claim_id"])
                for session in subset["session_id"]
            ]
        )
        y = subset["mean_decision_time_s"].to_numpy(dtype=float)
        ax.scatter(
            x,
            y,
            s=8.0,
            facecolor="white",
            edgecolor=COLOR[condition],
            linewidth=0.45,
            alpha=0.62,
            zorder=2,
        )
        estimate = estimates.loc[condition]
        ax.errorbar(
            index,
            estimate["mean"],
            yerr=np.array(
                [
                    [estimate["mean"] - estimate["ci95_low"]],
                    [estimate["ci95_high"] - estimate["mean"]],
                ]
            ),
            color=COLOR[condition],
            marker="D",
            markerfacecolor=COLOR[condition],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=3.8,
            capsize=1.8,
            elinewidth=0.8,
            zorder=4,
        )
    ax.set_xticks(range(2), [CONDITION_LABEL[c] for c in CONDITION_ORDER], rotation=18, ha="right")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(claim["ylabel"])
    ax.set_title(claim["title"], loc="left", fontweight="bold", pad=2.0)
    ax.grid(axis="y", color=COLOR["grid"], linewidth=0.35, alpha=0.55, zorder=0)
    if claim["claim_id"] == "ed4_pre_reward_direct_chain_specificity_check":
        ax.axhline(0.0, color=COLOR["neutral"], linewidth=0.55, linestyle="--", zorder=1)
    p_value = float(statistic["p_cluster_two_sided"])
    q_value = float(statistic["q_bh4_displayed_story"])
    p_text = f"P={p_value:.3f}" if p_value >= 0.001 else f"P={p_value:.2g}"
    ax.text(
        0.98,
        0.97,
        f"beta={statistic['estimate_return_minus_latent']:.3f} s\nCR1 {p_text}; BH4 q={q_value:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6},
        zorder=6,
    )
    if panel_label is not None:
        ax.text(
            -0.17,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
        )


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


def render_figures(
    rows: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    raw_stats = statistics.loc[statistics["scale"].eq("raw_seconds")].set_index(
        "claim_id"
    )
    outputs: list[Path] = []
    for claim in CLAIMS:
        subset = rows.loc[rows["claim_id"].eq(claim["claim_id"])]
        fig, ax = plt.subplots(figsize=(2.55, 2.18))
        draw_claim_panel(ax, subset, raw_stats.loc[claim["claim_id"]], claim)
        fig.text(
            0.23,
            0.018,
            f"n={len(subset)} sessions, {subset['animal'].nunique()} mice; {claim['footer']}",
            ha="left",
            va="bottom",
            fontsize=4.2,
            color=COLOR["neutral"],
        )
        fig.subplots_adjust(left=0.23, right=0.97, bottom=0.31, top=0.88)
        outputs += save_figure(fig, output_dir / claim["stem"])

    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.35))
    for panel_label, ax, claim in zip(("a", "b", "c", "d"), axes.flat, CLAIMS):
        subset = rows.loc[rows["claim_id"].eq(claim["claim_id"])]
        draw_claim_panel(
            ax,
            subset,
            raw_stats.loc[claim["claim_id"]],
            claim,
            panel_label=panel_label,
        )
    # Panels a and b show the same all-junction dwell metric before versus
    # after reward, so their visual comparison requires an identical y-axis.
    top_ymin = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    top_ymax = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    shared_ticks = axes[0, 0].get_yticks()
    shared_ticks = shared_ticks[
        (shared_ticks >= top_ymin) & (shared_ticks <= top_ymax)
    ]
    for ax in axes[0, :]:
        ax.set_yticks(shared_ticks)
        ax.set_ylim(top_ymin, top_ymax)
    if not (
        np.allclose(axes[0, 0].get_ylim(), axes[0, 1].get_ylim())
        and np.allclose(axes[0, 0].get_yticks(), axes[0, 1].get_yticks())
    ):
        raise AssertionError("Panels a and b must share identical y-limits and ticks")
    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        bottom=0.13,
        top=0.96,
        wspace=0.40,
        hspace=0.50,
    )
    outputs += save_figure(
        fig,
        output_dir / "ed04_fluency_four_panel_story_current",
    )
    return outputs


def validate_outputs(paths: list[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty output: {path}")
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "suffix": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "status": "ok",
        }
        if path.suffix.lower() in {".png", ".tiff"}:
            with Image.open(path) as image:
                gray = np.asarray(image.convert("L"))
                record["width_px"] = int(image.width)
                record["height_px"] = int(image.height)
                record["pixel_sd"] = float(np.std(gray))
                if float(np.std(gray)) < 1.0:
                    raise AssertionError(f"Raster appears blank: {path}")
        if path.suffix.lower() == ".svg":
            text = path.read_text(encoding="utf-8")
            record["editable_text"] = "<text" in text
            record["arial_declared"] = "Arial" in text
            if not record["editable_text"] or not record["arial_declared"]:
                raise AssertionError(f"SVG text/font QA failed: {path}")
        records.append(record)
    return pd.DataFrame(records)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_readme(
    output_dir: Path,
    statistics: pd.DataFrame,
    interactions: pd.DataFrame,
) -> Path:
    raw = statistics.loc[statistics["scale"].eq("raw_seconds")].set_index("claim_id")
    log = statistics.loc[statistics["scale"].eq("log1p_seconds")].set_index("claim_id")
    raw_interactions = interactions.loc[interactions["scale"].eq("raw_seconds")].set_index(
        "interaction_id"
    )
    lines = [
        "# ED4 pre/post fluency story",
        "",
        "## Figure contract",
        "",
        "- **Core conclusion:** return sessions show strong broad pre-reward fluency, no supported pre-reward direct-path specificity, an attenuated post-reward exploratory advantage, and robust reward-directed approach fluency after the first reward encounter.",
        "- **Statistical unit:** every eligible session row; CR1 covariance clustered by mouse; Student t with G-1 df.",
        "- **Displayed family:** BH4 across exactly the four current panels, separately for raw seconds and log1p seconds.",
        "- **Time anchor:** before versus after that session's first reward encounter.",
        "",
        "## Displayed results",
        "",
    ]
    for claim in CLAIMS:
        raw_row = raw.loc[claim["claim_id"]]
        log_row = log.loc[claim["claim_id"]]
        lines.append(
            f"- {claim['title']}: beta={raw_row['estimate_return_minus_latent']:.4f} s; "
            f"CR1 P={raw_row['p_cluster_two_sided']:.6g}; BH4 q={raw_row['q_bh4_displayed_story']:.6g}; "
            f"log1p BH4 q={log_row['q_bh4_displayed_story']:.6g}; "
            f"n={int(raw_row['n_rows'])} sessions/{int(raw_row['n_mice'])} mice."
        )
    lines.extend(
        [
            "",
            "## Exact interaction controls",
            "",
        ]
    )
    for interaction_id in (
        "broad_post_minus_pre_x_condition",
        "approach_minus_broad_post_x_condition",
        "approach_minus_same_path_explore_x_condition",
    ):
        row = raw_interactions.loc[interaction_id]
        q_value = row["q_bh2_exact_interactions"]
        correction_text = (
            f"BH2 q={q_value:.6g}"
            if np.isfinite(q_value)
            else "descriptive only; changes both mode and path class"
        )
        lines.append(
            f"- {row['interaction_label']}: beta={row['estimate_return_minus_latent']:.4f} s; "
            f"P={row['p_cluster_two_sided']:.6g}; {correction_text}."
        )
    lines.extend(
        [
            "",
            "## Interpretation lock",
            "",
            "- Broad post-reward exploratory fluency is smaller than before reward and does not cross 0.05 under the displayed BH4 family; this is borderline evidence for a residual effect, not evidence that the effect is absent.",
            "- The pre-reward direct-minus-other control is unsupported, so the strong pre-reward advantage should be described as broad rather than direct-path-specific.",
            "- The approach advantage is larger than broad post-reward exploration overall, but not larger than exploration at the same direct-chain junctions. Use `during reward-directed approaches`, not `uniquely reward-approach-specific`.",
            "- The approach metric is direct-chain junction dwell/commitment time in reward-approach mode, not full home-to-water run duration.",
        ]
    )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path) -> dict[str, Any]:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_names = [
        "fluency_union_bh4_sensitivity.csv",
        *[
            f"ed04_fluency_three_panel_story_current.{extension}"
            for extension in FORMATS
        ],
        *[
            f"ed04c_post_port_direct_approach_current.{extension}"
            for extension in FORMATS
        ],
    ]
    for stale_name in stale_names:
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    source = pd.read_csv(SOURCE_PATH, dtype={"animal": str}, low_memory=False)
    claim_rows, reconstruction_audit = prepare_claim_rows(source)
    claim_statistics = build_claim_statistics(claim_rows)
    interaction_rows, interaction_statistics = build_interaction_tables(
        claim_rows,
        source,
    )

    correction_registry = claim_statistics[
        [
            "claim_id",
            "claim_label",
            "scale",
            "p_cluster_two_sided",
            "q_bh4_displayed_story",
            "inverse_session_count_cluster_p",
            "inverse_session_count_bh4_q",
        ]
    ].copy()
    table_map = {
        "fluency_plot_rows": claim_rows,
        "post_reward_broad_reconstruction_audit": reconstruction_audit,
        "fluency_statistics": claim_statistics,
        "fluency_correction_registry": correction_registry,
        "fluency_interaction_session_rows": interaction_rows,
        "fluency_interaction_statistics": interaction_statistics,
    }
    table_paths: list[Path] = []
    for name, table in table_map.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        table_paths.append(path)

    figure_paths = render_figures(claim_rows, claim_statistics, output_dir)
    qa = validate_outputs(figure_paths)
    qa_path = output_dir / "output_qa.csv"
    qa.to_csv(qa_path, index=False)
    table_paths.append(qa_path)
    readme_path = write_readme(
        output_dir,
        claim_statistics,
        interaction_statistics,
    )

    raw_stats = claim_statistics.loc[claim_statistics["scale"].eq("raw_seconds")]
    manifest = {
        "status": "validated_current_plus_new12_source_render",
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir.resolve()),
        "figure_contract": {
            "core_conclusion": (
                "Return sessions show strong broad pre-reward fluency without supported direct-path "
                "specificity, an attenuated post-reward exploratory advantage, and robust "
                "post-reward reward-directed approach fluency."
            ),
            "archetype": "four_panel_quantitative_grid",
            "backend": "Python/matplotlib only",
            "formats": list(FORMATS),
            "raster_dpi": 600,
            "svg_text_editable": True,
            "shared_axis_groups": [
                "panels a-b: mean junction dwell, identical y-limits and ticks"
            ],
        },
        "analysis_contract": {
            "row_unit": "eligible session",
            "dependence": "CR1 covariance clustered by mouse",
            "calibration": "Student t with df=number of mouse clusters minus 1",
            "primary_weighting": "all eligible session rows retained with unit weight",
            "sensitivity_weighting": "inverse session count within mouse",
            "displayed_multiplicity": (
                "BH4 across broad pre-reward exploration, broad post-reward exploration, "
                "pre-reward direct-minus-other specificity, and post-reward direct-chain "
                "approach; raw and log1p scales corrected separately"
            ),
            "interaction_multiplicity": (
                "BH2 across broad pre/post attenuation and reward-approach versus same-path "
                "exploration; the visible-panel approach-versus-broad contrast is descriptive "
                "because it changes both mode and path class"
            ),
            "reward_node": 94,
            "reward_epoch_anchor": "that session's first reward encounter",
        },
        "scope": {
            row["claim_id"]: {
                "n_sessions": int(row["n_rows"]),
                "n_mice": int(row["n_mice"]),
                "n_return_sessions": int(row["n_return_rows"]),
                "n_latent_sessions": int(row["n_latent_rows"]),
            }
            for _, row in raw_stats.iterrows()
        },
        "source": {
            "path": str(SOURCE_PATH.resolve()),
            "bytes": SOURCE_PATH.stat().st_size,
            "sha256": sha256(SOURCE_PATH),
        },
        "source_tables": [str(path.resolve()) for path in table_paths],
        "figures": [str(path.resolve()) for path in figure_paths],
        "readme": str(readme_path.resolve()),
        "qa": {
            "n_outputs": int(len(qa)),
            "all_nonempty": True,
            "all_rasters_nonblank": True,
            "all_svg_text_editable_and_arial_declared": True,
        },
        "interpretation_lock": (
            "Broad post-reward exploration is borderline and does not cross 0.05 after displayed BH4 correction; nonsignificance is not absence. "
            "The reward-approach effect is not uniquely mode-specific relative to exploration at "
            "the same direct-chain junctions."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(json_ready(manifest), indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args.output_dir.resolve())
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
