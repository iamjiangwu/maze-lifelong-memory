"""Recompute ED5 b-d summaries from the maintained session-pair rows.

This is an aggregation-only analysis. It does not recompute maps, CIMaps, or
per-pair correlations. Each of the 36 long-gap transitions receives equal
weight in the displayed means. Repeated transitions from the same mouse are
handled with CR1 cluster-robust uncertainty and Student-t calibration with
G-1 degrees of freedom. The three d1-d1 contrasts within each neural metric
are adjusted together with Benjamini-Hochberg.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


VARIANT_ORDER = ("d1v1", "d1v2", "d2v1", "d2v2")
METRICS = {
    "place_map_correlation": (
        "Place map",
        "place_pair_summary_corrected36_all_variants.csv",
        "place_map_mean_all",
    ),
    "pv_correlation": (
        "Population vector",
        "pv_pair_summary_corrected36_all_variants.csv",
        "data_mean",
    ),
    "ensemble_rate_correlation": (
        "Ensemble rate",
        "rate_pair_summary_corrected36_all_variants.csv",
        "pearson_raw_r",
    ),
}


def bh_adjust(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(raw) / np.arange(1, len(raw) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def clustered_intercept(values: np.ndarray, clusters: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values are not allowed")
    n = len(values)
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    if n < 2 or g < 2:
        raise ValueError("At least two observations and two clusters are required")
    estimate = float(values.mean())
    residual = values - estimate
    meat = sum(float(residual[clusters == cluster].sum()) ** 2 for cluster in unique_clusters)
    variance = (1.0 / n) ** 2 * meat * g / (g - 1)
    se = float(np.sqrt(max(variance, 0.0)))
    df = g - 1
    t_value = estimate / se if se > 0 else np.inf
    p_two_sided = float(2.0 * stats.t.sf(abs(t_value), df=df))
    tcrit = float(stats.t.ppf(0.975, df=df))
    return {
        "estimate": estimate,
        "se_cr1": se,
        "ci95_low": estimate - tcrit * se,
        "ci95_high": estimate + tcrit * se,
        "t": t_value,
        "df": df,
        "p_two_sided": p_two_sided,
        "n_rows": n,
        "n_animals": g,
    }


def exact_cluster_signflip(values: np.ndarray, clusters: np.ndarray) -> float:
    """Two-sided exact sign-flip p for a pair-weighted clustered mean."""

    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters, dtype=str)
    unique_clusters = np.unique(clusters)
    cluster_sums = np.asarray(
        [values[clusters == cluster].sum() for cluster in unique_clusters], dtype=float
    )
    observed = abs(float(cluster_sums.sum()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(unique_clusters))))
    permuted = np.abs(signs @ cluster_sums)
    return float(np.mean(permuted >= observed - 1e-15))


def load_pair_values(plot_ready: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined: list[pd.DataFrame] = []
    source_checks: list[dict[str, object]] = []
    reference_keys: set[tuple[str, str, str]] | None = None
    for metric, (label, filename, value_column) in METRICS.items():
        path = plot_ready / filename
        frame = pd.read_csv(path, dtype={"animal_id": str})
        counts = frame.groupby("variant", observed=True).size().to_dict()
        if len(frame) != 144 or counts != {variant: 36 for variant in VARIANT_ORDER}:
            raise AssertionError(f"Unexpected ED5 pair support for {metric}: {counts}")
        if frame.duplicated(["animal_id", "group", "variant"]).any():
            raise AssertionError(f"Duplicate animal/group/variant rows for {metric}")
        keys = set(map(tuple, frame[["animal_id", "group", "variant"]].astype(str).to_numpy()))
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise AssertionError(f"Transition support differs for {metric}")
        values = pd.to_numeric(frame[value_column], errors="raise")
        if not np.isfinite(values).all():
            raise AssertionError(f"Nonfinite pair values for {metric}")
        selected = frame[
            ["animal_id", "group", "variant", "pair_key", "src_session", "dst_session"]
        ].copy()
        selected.insert(0, "metric_label", label)
        selected.insert(0, "metric", metric)
        selected["transition_id"] = selected["animal_id"] + "::" + selected["group"].astype(str)
        selected["value"] = values.to_numpy(float)
        selected["source_file"] = filename
        selected["source_value_column"] = value_column
        combined.append(selected)
        source_checks.append(
            {
                "metric": metric,
                "source_file": filename,
                "source_value_column": value_column,
                "n_rows": len(frame),
                "n_animals": frame["animal_id"].nunique(),
                "n_variants": frame["variant"].nunique(),
                "rows_per_variant": 36,
                "duplicate_transition_variant_rows": 0,
                "status": "PASS",
            }
        )
    result = pd.concat(combined, ignore_index=True)
    result = result.sort_values(
        ["metric", "variant", "animal_id", "group"], kind="stable"
    ).reset_index(drop=True)
    return result, pd.DataFrame(source_checks)


def summarize(pair_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (metric, label, variant), frame in pair_values.groupby(
        ["metric", "metric_label", "variant"], observed=True, sort=False
    ):
        result = clustered_intercept(frame["value"].to_numpy(float), frame["animal_id"])
        values = frame["value"].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "variant": variant,
                "n_session_pairs": len(frame),
                "n_animals": frame["animal_id"].nunique(),
                "mean": float(values.mean()),
                "sd_across_pairs": float(values.std(ddof=1)),
                "sem_across_pairs": float(values.std(ddof=1) / np.sqrt(len(values))),
                "median_across_pairs": float(np.median(values)),
                "se_cr1": result["se_cr1"],
                "ci95_low_cr1": result["ci95_low"],
                "ci95_high_cr1": result["ci95_high"],
                "cluster_df": result["df"],
                "aggregation": "equal-weighted session-pair mean",
                "interval": "CR1 by animal; Student t(G-1)",
            }
        )
    order = {value: index for index, value in enumerate(VARIANT_ORDER)}
    output = pd.DataFrame(rows)
    output["variant_order"] = output["variant"].map(order)
    return output.sort_values(["metric", "variant_order"], kind="stable").drop(
        columns="variant_order"
    )


def contrasts(pair_values: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, frame in pair_values.groupby("metric", observed=True, sort=False):
        wide = frame.pivot(
            index=["transition_id", "animal_id"], columns="variant", values="value"
        )
        if len(wide) != 36 or wide[list(VARIANT_ORDER)].isna().any(axis=None):
            raise AssertionError(f"Incomplete paired day-combination matrix for {metric}")
        for other in VARIANT_ORDER[1:]:
            difference = (wide["d1v1"] - wide[other]).to_numpy(float)
            clusters = wide.index.get_level_values("animal_id").to_numpy(str)
            result = clustered_intercept(difference, clusters)
            p_exact = exact_cluster_signflip(difference, clusters)
            rows.append(
                {
                    "metric": metric,
                    "family": "d1v1_pairwise",
                    "test": "session-pair contrast; CR1 clustered by animal",
                    "contrast": f"d1v1_minus_{other}",
                    "mean_difference": result["estimate"],
                    "se_cr1": result["se_cr1"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "statistic": result["t"],
                    "cluster_df": result["df"],
                    "p_value": result["p_two_sided"],
                    "p_exact_cluster_signflip": p_exact,
                    "n_session_pairs": result["n_rows"],
                    "n_animals": result["n_animals"],
                    "covariance": "CR1",
                    "reference_distribution": "Student t(G-1)",
                }
            )
    output = pd.DataFrame(rows)
    output["p_bh3_within_metric_cr1"] = np.nan
    output["p_bh3_within_metric_exact_cluster_signflip"] = np.nan
    for _, indices in output.groupby("metric", sort=False).groups.items():
        indices = list(indices)
        output.loc[indices, "p_bh3_within_metric_cr1"] = bh_adjust(
            output.loc[indices, "p_value"].to_numpy(float)
        )
        output.loc[
            indices, "p_bh3_within_metric_exact_cluster_signflip"
        ] = bh_adjust(output.loc[indices, "p_exact_cluster_signflip"].to_numpy(float))
    return output


def aggregation_comparison(pair_values: pd.DataFrame, descriptive: pd.DataFrame) -> pd.DataFrame:
    animal = (
        pair_values.groupby(["metric", "metric_label", "animal_id", "variant"], observed=True)[
            "value"
        ]
        .mean()
        .reset_index()
    )
    animal_summary = (
        animal.groupby(["metric", "metric_label", "variant"], observed=True)
        .agg(animal_equal_mean=("value", "mean"), n_animals=("animal_id", "nunique"))
        .reset_index()
    )
    comparison = descriptive[
        ["metric", "metric_label", "variant", "mean", "n_session_pairs"]
    ].merge(
        animal_summary,
        on=["metric", "metric_label", "variant"],
        validate="one_to_one",
    )
    comparison = comparison.rename(columns={"mean": "session_pair_equal_mean"})
    comparison["pair_minus_animal_mean"] = (
        comparison["session_pair_equal_mean"] - comparison["animal_equal_mean"]
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = args.record.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pair_values, source_checks = load_pair_values(record / "plot_ready")
    descriptive = summarize(pair_values)
    contrast = contrasts(pair_values)
    comparison = aggregation_comparison(pair_values, descriptive)

    paths = {
        "pair_values": output / "ed05_day_combination_session_pair_values.csv",
        "descriptive": output / "ed05_day_combination_session_pair_descriptive_cr1.csv",
        "contrasts": output / "ed05_day_combination_session_pair_contrasts_cr1_bh3.csv",
        "aggregation_comparison": output / "ed05_day_combination_aggregation_comparison.csv",
        "source_validation": output / "ed05_day_combination_source_validation.csv",
    }
    pair_values.to_csv(paths["pair_values"], index=False)
    descriptive.to_csv(paths["descriptive"], index=False)
    contrast.to_csv(paths["contrasts"], index=False)
    comparison.to_csv(paths["aggregation_comparison"], index=False)
    source_checks.to_csv(paths["source_validation"], index=False)

    report = {
        "status": "PASS",
        "analysis": "ED5 b-d equal-weighted session-pair reaggregation",
        "source_record": record.name,
        "maps_or_cimaps_recomputed": False,
        "n_session_pairs_per_metric_variant": 36,
        "n_animals": 14,
        "point_estimate": "equal-weighted mean of session-pair correlations",
        "uncertainty": "CR1 clustered by animal with Student t(G-1)",
        "multiplicity": "BH3 across the three d1v1 contrasts separately within each metric",
        "primary_contrast_test": "two-sided exact mouse-cluster sign flip of the equal-pair mean difference",
        "contrast_sensitivity": "CR1 clustered by animal with Student t(G-1)",
        "outputs": {key: path.name for key, path in paths.items()},
    }
    (output / "ANALYSIS_MANIFEST.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
