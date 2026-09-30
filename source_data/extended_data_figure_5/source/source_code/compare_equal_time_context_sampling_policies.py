"""Compare context-only fixed-duration and all-outside sampling policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


KEYS = ["variant", "group", "animal_id", "src_session", "dst_session"]
METRICS = {
    "outside": "equal_time_outside_r_mean",
    "maze": "equal_time_maze_r_mean",
    "maze_minus_outside": "equal_time_maze_minus_outside_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed5", type=Path, required=True)
    parser.add_argument("--all-outside", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed = pd.read_csv(args.fixed5, dtype={"animal_id": str})
    all_outside = pd.read_csv(args.all_outside, dtype={"animal_id": str})
    left = fixed[KEYS + list(METRICS.values())].rename(
        columns={value: f"fixed5_{name}" for name, value in METRICS.items()}
    )
    right = all_outside[KEYS + list(METRICS.values())].rename(
        columns={value: f"all_outside_{name}" for name, value in METRICS.items()}
    )
    joined = left.merge(right, on=KEYS, validate="one_to_one", how="outer", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("The two context-only runs do not have identical pair inventories")
    rows = []
    for variant in ["d1v1", "d1v2", "d2v1", "d2v2"]:
        subset = joined.loc[joined["variant"].eq(variant)].copy()
        n_animals = int(subset["animal_id"].nunique())
        cluster_df = n_animals - 1
        critical = float(stats.t.ppf(0.975, df=cluster_df))
        for metric in METRICS:
            difference = f"all_outside_minus_fixed5_{metric}"
            subset[difference] = subset[f"all_outside_{metric}"] - subset[f"fixed5_{metric}"]
            joined.loc[subset.index, difference] = subset[difference]
            model = smf.ols(f"{difference} ~ 1", data=subset).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": subset["animal_id"],
                    "use_correction": True,
                    "df_correction": True,
                },
            )
            estimate = float(model.params["Intercept"])
            se = float(model.bse["Intercept"])
            t_value = float(model.tvalues["Intercept"])
            rows.append(
                {
                    "variant": variant,
                    "estimand": metric,
                    "n_pair_rows": int(len(subset)),
                    "n_animals": n_animals,
                    "mean_all_outside_minus_fixed5": estimate,
                    "cluster_se": se,
                    "ci95_low": estimate - critical * se,
                    "ci95_high": estimate + critical * se,
                    "t_stat": t_value,
                    "pvalue": float(2.0 * (1.0 - stats.t.cdf(abs(t_value), df=cluster_df))),
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out_dir / "context_only_sampling_policy_pair_comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(
        args.out_dir / "context_only_sampling_policy_clustered_comparison.csv", index=False
    )


if __name__ == "__main__":
    main()
