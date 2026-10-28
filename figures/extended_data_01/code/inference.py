"""Animal-clustered endpoint contrasts and log-log regression."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from release_rendering.statistics import sem, significance_stars, cr1_ols, linear_contrast, benjamini_hochberg
GROUP_LABELS = {
    "naive": "Naive day",
    "return": "Return day",
    "experienced_day2": "Second day",
}

def _endpoint_statistics(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 18 displayed time/tile animal-clustered pairwise contrasts."""

    endpoint_order = ("first_reward", "tenth_reward", "first_long_bout")
    axis_specs = (
        ("maze_time_s", "Time in maze (s)"),
        ("maze_tiles", "Maze tile steps"),
    )
    pairs = (
        ("naive", "return"),
        ("naive", "experienced_day2"),
        ("return", "experienced_day2"),
    )
    rows: list[dict[str, object]] = []
    for endpoint_key in endpoint_order:
        endpoint = endpoints.loc[endpoints["endpoint_key"].eq(endpoint_key)]
        endpoint_label = str(endpoint["endpoint_label"].iloc[0])
        for axis_key, axis_label in axis_specs:
            for group_a, group_b in pairs:
                use = endpoint.loc[
                    endpoint["group_key"].isin((group_a, group_b))
                ].copy()
                outcome = pd.to_numeric(use[axis_key], errors="coerce").to_numpy(float)
                design = np.column_stack(
                    (
                        np.ones(len(use), dtype=float),
                        use["group_key"].eq(group_b).to_numpy(float),
                    )
                )
                valid = np.isfinite(outcome)
                if np.any(valid & (outcome <= 0)):
                    invalid_sessions = use.loc[
                        valid & (outcome <= 0), "session"
                    ].astype(str).tolist()
                    raise ValueError(
                        "Natural-log endpoint outcomes must be positive for "
                        f"{endpoint_key}/{axis_key}: {invalid_sessions}"
                    )
                estimate = standard_error = t_value = p_value = float("nan")
                degrees = 0
                n_clusters = int(use.loc[valid, "animal_id"].nunique())
                if valid.sum() > 2 and n_clusters >= 2:
                    fit = cr1_ols(
                        np.log(outcome),
                        design,
                        use["animal_id"].astype(str).tolist(),
                    )
                    contrast = linear_contrast(fit, [0.0, 1.0])
                    estimate = contrast.estimate
                    standard_error = contrast.standard_error
                    t_value = contrast.t_value
                    p_value = contrast.p_two_sided
                    degrees = contrast.degrees_of_freedom
                    n_clusters = fit.n_clusters
                group_a_values = outcome[
                    use["group_key"].eq(group_a).to_numpy() & valid
                ]
                group_b_values = outcome[
                    use["group_key"].eq(group_b).to_numpy() & valid
                ]
                rows.append(
                    {
                        "endpoint_key": endpoint_key,
                        "endpoint_label": endpoint_label,
                        "axis_key": axis_key,
                        "axis_label": axis_label,
                        "group_a": group_a,
                        "group_b": group_b,
                        "group_a_label": GROUP_LABELS[group_a],
                        "group_b_label": GROUP_LABELS[group_b],
                        "n_sessions_group_a": int(group_a_values.size),
                        "n_sessions_group_b": int(group_b_values.size),
                        "n_sessions": int(valid.sum()),
                        "n_animals": n_clusters,
                        "mean_group_a": float(np.mean(group_a_values)),
                        "mean_group_b": float(np.mean(group_b_values)),
                        "raw_mean_difference_b_minus_a": float(
                            np.mean(group_b_values) - np.mean(group_a_values)
                        ),
                        "transform": "log",
                        "coefficient_b_minus_a": estimate,
                        "standard_error": standard_error,
                        "cluster_t": t_value,
                        "cluster_p_two_sided": p_value,
                        "cluster_df": degrees,
                    }
                )
    statistics = pd.DataFrame(rows)
    statistics["p_bh18"] = benjamini_hochberg(
        statistics["cluster_p_two_sided"].to_numpy(float)
    )
    return statistics

def compute_long_bout_relation(
    long_onsets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the log-log relation between time and transitions at bout onset."""

    sessions = long_onsets.copy()
    sessions["log_time"] = np.log(
        pd.to_numeric(sessions["first_long_bout_time_s"], errors="coerce")
    )
    sessions["log_tiles"] = np.log(
        pd.to_numeric(
            sessions["first_long_bout_tile_transitions"],
            errors="coerce",
        )
    )
    valid = np.isfinite(sessions["log_time"]) & np.isfinite(sessions["log_tiles"])
    use = sessions.loc[valid].copy()
    if len(use) <= 2 or use["animal_id"].nunique() < 2:
        raise ValueError(
            "The acquisition endpoint relation needs at least three sessions "
            "from two animals"
        )
    design = np.column_stack((np.ones(len(use)), use["log_time"].to_numpy(float)))
    fit = cr1_ols(
        use["log_tiles"].to_numpy(float),
        design,
        use["animal_id"].astype(str).tolist(),
    )
    slope = linear_contrast(fit, [0.0, 1.0])
    x_line = np.geomspace(
        float(use["first_long_bout_time_s"].min()),
        float(use["first_long_bout_time_s"].max()),
        100,
    )
    line = pd.DataFrame(
        {
            "row_kind": "fit_line",
            "first_long_bout_time_s": x_line,
            "first_long_bout_tile_transitions": np.exp(
                fit.beta[0] + fit.beta[1] * np.log(x_line)
            ),
        }
    )
    use["row_kind"] = "session_point"
    statistics = pd.DataFrame(
        [
            {
                "row_kind": "model_statistic",
                "n_sessions": fit.n_observations,
                "n_animals": fit.n_clusters,
                "intercept_log_scale": float(fit.beta[0]),
                "slope_loglog": slope.estimate,
                "cluster_se": slope.standard_error,
                "cluster_t": slope.t_value,
                "cluster_df": slope.degrees_of_freedom,
                "p_two_sided": slope.p_two_sided,
                "ci95_slope_low": slope.confidence_low,
                "ci95_slope_high": slope.confidence_high,
                "formula": "ln(steps) ~ 1 + ln(in-maze seconds)",
            }
        ]
    )
    return (
        use,
        line,
        statistics,
        pd.concat((use, line, statistics), ignore_index=True, sort=False),
    )
