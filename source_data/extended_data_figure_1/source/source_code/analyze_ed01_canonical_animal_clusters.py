"""Recompute ED1f/g inference after canonicalizing biological animal IDs.

The accepted Round 18 artwork was generated from 153 session rows, but its two
clustered-statistics tables treated recording aliases (for example ``451`` and
``451R``) as different animals.  This script keeps the session-level outcomes
and estimands unchanged, canonicalizes each animal to its leading numeric ID,
and freezes the corrected 27-contrast endpoint family and the single tile-time
log-log model for an append-only successor record.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
PARENT_NAME = "extended_data_fig01_2026-08-15_round18_group80_matched153"
PARENT = REPO / "docs/figure_version_records" / PARENT_NAME
OUT = (
    REPO
    / "docs/figure_version_candidates"
    / "ed01_canonical_animal_clusters_2026-08-17"
)
ENDPOINT_ROWS_NAME = "ed01_endpoint_axis_session_rows_canonical29.csv"
ENDPOINT_STATS_NAME = "ed01_endpoint_axis_cluster_stats_canonical29_bh27.csv"
TILE_ROWS_NAME = "ed01_tile_time_loglog_rows_canonical29.csv"
TILE_STATS_NAME = "ed01_tile_time_loglog_stats_canonical29.csv"
ALIAS_MAP_NAME = "ed01_animal_alias_map.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_animal(value: object) -> str:
    match = re.match(r"\d+", str(value).strip())
    if match is None:
        raise ValueError(f"Cannot canonicalize animal ID {value!r}")
    return str(int(match.group(0)))


def canonicalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(
        result.columns.get_loc("animal_id") + 1,
        "animal_id_alias",
        result["animal_id"].astype(str),
    )
    result["animal_id"] = result["animal_id_alias"].map(canonical_animal)
    return result


def clustered_ols(y: np.ndarray, design: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    keep = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    y, design, clusters = y[keep], design[keep], clusters[keep]
    n, k = design.shape
    unique = np.unique(clusters)
    g = len(unique)
    if n <= k or g < 3:
        raise ValueError(f"Insufficient clustered-model support: n={n}, k={k}, G={g}")
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique:
        mask = clusters == cluster
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    cr1 = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = cr1 * inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    tested = 1
    t_value = float(beta[tested] / standard_error[tested])
    p_value = float(2.0 * stats.t.sf(abs(t_value), df=g - 1))
    return {
        "beta": beta,
        "standard_error": standard_error,
        "t": t_value,
        "p": p_value,
        "n": int(n),
        "k": int(k),
        "g": int(g),
        "df": int(g - 1),
        "covariance": covariance,
    }


def bh_adjust(values: pd.Series) -> np.ndarray:
    p = pd.to_numeric(values, errors="raise").to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def star(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def recompute_endpoints(rows: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for _, original in template.iterrows():
        endpoint = str(original["endpoint_key"])
        axis = str(original["axis_key"])
        group_a = str(original["group_a"])
        group_b = str(original["group_b"])
        subset = rows.loc[
            rows["endpoint_key"].astype(str).eq(endpoint)
            & rows["group_key"].astype(str).isin([group_a, group_b])
        ].copy()
        subset[axis] = pd.to_numeric(subset[axis], errors="coerce")
        subset = subset.loc[np.isfinite(subset[axis]) & subset[axis].ge(0)].copy()
        response = np.log1p(subset[axis].to_numpy(float))
        indicator = subset["group_key"].astype(str).eq(group_b).astype(float).to_numpy()
        design = np.column_stack((np.ones(len(subset)), indicator))
        fit = clustered_ols(response, design, subset["animal_id"].to_numpy(str))
        values_a = subset.loc[subset["group_key"].astype(str).eq(group_a), axis].to_numpy(float)
        values_b = subset.loc[subset["group_key"].astype(str).eq(group_b), axis].to_numpy(float)
        coefficient = float(fit["beta"][1])
        row = original.to_dict()
        row.update(
            {
                "n_sessions_group_a": int(len(values_a)),
                "n_sessions_group_b": int(len(values_b)),
                "n_sessions": int(len(subset)),
                "n_mice": int(fit["g"]),
                "mean_group_a": float(np.mean(values_a)),
                "mean_group_b": float(np.mean(values_b)),
                "raw_mean_diff_group_b_minus_a": float(np.mean(values_b) - np.mean(values_a)),
                "transform": "log1p",
                "coef_log1p_group_b_minus_a": coefficient,
                "effect_ratio_plus_one": float(np.exp(coefficient)),
                "cluster_t": float(fit["t"]),
                "cluster_p_two_sided": float(fit["p"]),
                "cluster_df": int(fit["df"]),
                "test": "OLS on log1p outcome; CR1 covariance clustered by canonical biological animal; t(G-1)",
            }
        )
        output.append(row)
    result = pd.DataFrame(output)
    result["q_bh27"] = bh_adjust(result["cluster_p_two_sided"])
    return result[list(template.columns)]


def recompute_tile_time(rows: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    included = rows["included_in_loglog_model"].astype(str).str.lower().eq("true")
    model = rows.loc[included].copy()
    x = pd.to_numeric(model["log_elapsed_time_s"], errors="raise").to_numpy(float)
    y = pd.to_numeric(model["log_tile_transitions"], errors="raise").to_numpy(float)
    design = np.column_stack((np.ones(len(model)), x))
    fit = clustered_ols(y, design, model["animal_id"].to_numpy(str))
    beta = np.asarray(fit["beta"], dtype=float)
    se = float(fit["standard_error"][1])
    critical = float(stats.t.ppf(0.975, df=fit["df"]))
    fitted = design @ beta
    residual = y - fitted
    ordinary_r_squared = float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2))
    pearson_r = float(stats.pearsonr(np.exp(x), np.exp(y)).statistic)
    original = template.iloc[0].to_dict()
    original.update(
        {
            "dependence": "CR1 covariance clustered by canonical biological animal",
            "n_sessions": int(len(model)),
            "n_mice": int(fit["g"]),
            "df_t": int(fit["df"]),
            "intercept_log_scale": float(beta[0]),
            "slope_loglog": float(beta[1]),
            "cluster_se": se,
            "t_cluster": float(fit["t"]),
            "p_cluster_t_two_sided": float(fit["p"]),
            "ci95_slope_low": float(beta[1] - critical * se),
            "ci95_slope_high": float(beta[1] + critical * se),
            "raw_pearson_r_descriptive_only": pearson_r,
            "ordinary_ols_r_squared_descriptive_only": ordinary_r_squared,
            "n_excluded": int(len(rows) - len(model)),
        }
    )
    return pd.DataFrame([original], columns=template.columns)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_plot = PARENT / "plot_ready"
    endpoint_source = parent_plot / "ed01_endpoint_axis_session_rows_current153.csv"
    endpoint_stats_source = parent_plot / "ed01_endpoint_axis_cluster_stats_current153.csv"
    tile_source = parent_plot / "ed02_tile_time_loglog_rows_current153.csv"
    tile_stats_source = parent_plot / "ed02_tile_time_loglog_stats_current153.csv"
    sources = (endpoint_source, endpoint_stats_source, tile_source, tile_stats_source)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    endpoint_raw = pd.read_csv(endpoint_source, dtype={"animal_id": str})
    tile_raw = pd.read_csv(tile_source, dtype={"animal_id": str})
    endpoint_template = pd.read_csv(endpoint_stats_source)
    tile_template = pd.read_csv(tile_stats_source)
    endpoint_rows = canonicalize_rows(endpoint_raw)
    tile_rows = canonicalize_rows(tile_raw)
    endpoint_stats = recompute_endpoints(endpoint_rows, endpoint_template)
    tile_stats = recompute_tile_time(tile_rows, tile_template)

    aliases = pd.concat(
        [
            endpoint_rows[["animal_id_alias", "animal_id"]],
            tile_rows[["animal_id_alias", "animal_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    aliases = aliases.rename(columns={"animal_id": "canonical_animal_id"})
    aliases["alias_changed"] = aliases["animal_id_alias"].ne(aliases["canonical_animal_id"])
    aliases = aliases.sort_values(
        ["canonical_animal_id", "animal_id_alias"],
        key=lambda series: series.map(lambda value: int(value) if str(value).isdigit() else str(value)),
        kind="stable",
    )

    endpoint_rows.to_csv(OUT / ENDPOINT_ROWS_NAME, index=False)
    endpoint_stats.to_csv(OUT / ENDPOINT_STATS_NAME, index=False)
    tile_rows.to_csv(OUT / TILE_ROWS_NAME, index=False)
    tile_stats.to_csv(OUT / TILE_STATS_NAME, index=False)
    aliases.to_csv(OUT / ALIAS_MAP_NAME, index=False)

    displayed_old = endpoint_template.loc[
        endpoint_template["axis_key"].isin(["maze_time_s", "maze_tiles"])
    ].copy()
    displayed_new = endpoint_stats.loc[
        endpoint_stats["axis_key"].isin(["maze_time_s", "maze_tiles"])
    ].copy()
    exact_unchanged_columns = (
        "n_sessions_group_a",
        "n_sessions_group_b",
        "n_sessions",
        "mean_group_a",
        "mean_group_b",
        "raw_mean_diff_group_b_minus_a",
        "coef_log1p_group_b_minus_a",
        "effect_ratio_plus_one",
    )
    endpoint_point_estimates_unchanged = all(
        np.allclose(
            pd.to_numeric(endpoint_template[column], errors="raise"),
            pd.to_numeric(endpoint_stats[column], errors="raise"),
            rtol=0,
            atol=1e-12,
        )
        for column in exact_unchanged_columns
    )
    endpoint_stars_unchanged = [star(value) for value in displayed_old["q_bh27"]] == [
        star(value) for value in displayed_new["q_bh27"]
    ]
    tile_point_estimates_unchanged = np.allclose(
        tile_template.iloc[0][["intercept_log_scale", "slope_loglog"]].astype(float),
        tile_stats.iloc[0][["intercept_log_scale", "slope_loglog"]].astype(float),
        rtol=0,
        atol=1e-12,
    )
    split_groups = aliases.groupby("canonical_animal_id")["animal_id_alias"].nunique()
    checks = {
        "endpoint_153_sessions": endpoint_rows["session"].nunique() == 153,
        "tile_time_153_sessions": tile_rows["session"].nunique() == 153,
        "endpoint_29_biological_animals": endpoint_rows["animal_id"].nunique() == 29,
        "tile_time_29_biological_animals": tile_rows["animal_id"].nunique() == 29,
        "raw_alias_count_36": endpoint_rows["animal_id_alias"].nunique() == 36,
        "seven_biological_animals_had_split_aliases": int(split_groups.gt(1).sum()) == 7,
        "endpoint_family_has_27_tests": len(endpoint_stats) == 27,
        "endpoint_family_has_18_displayed_tests": len(displayed_new) == 18,
        "endpoint_all_displayed_remain_significant": bool(displayed_new["q_bh27"].lt(0.05).all()),
        "endpoint_all_displayed_stars_unchanged": endpoint_stars_unchanged,
        "endpoint_point_estimates_unchanged": endpoint_point_estimates_unchanged,
        "endpoint_family_max_adjusted_p_exact": np.isclose(
            float(endpoint_stats["q_bh27"].max()), 0.0002471109894525664, rtol=0, atol=1e-16
        ),
        "endpoint_caption_max_adjusted_p_rounds_to_1_8e_4": np.isclose(
            float(displayed_new["q_bh27"].max()), 1.8e-4, rtol=0.06
        ),
        "tile_time_point_estimates_unchanged": tile_point_estimates_unchanged,
        "tile_time_slope_rounds_to_0_964": round(float(tile_stats.iloc[0]["slope_loglog"]), 3) == 0.964,
        "tile_time_caption_p_rounds_to_2_7e_35": np.isclose(
            float(tile_stats.iloc[0]["p_cluster_t_two_sided"]), 2.7e-35, rtol=0.06
        ),
        "tile_time_caption_ci_rounds_to_0_94_0_99": (
            round(float(tile_stats.iloc[0]["ci95_slope_low"]), 2) == 0.94
            and round(float(tile_stats.iloc[0]["ci95_slope_high"]), 2) == 0.99
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    output_paths = [
        OUT / ENDPOINT_ROWS_NAME,
        OUT / ENDPOINT_STATS_NAME,
        OUT / TILE_ROWS_NAME,
        OUT / TILE_STATS_NAME,
        OUT / ALIAS_MAP_NAME,
    ]
    validation = {
        "schema": "ed01-canonical-animal-clusters/v1",
        "status": status,
        "parent_record": PARENT_NAME,
        "checks": {key: bool(value) for key, value in checks.items()},
        "endpoint_family": {
            "tests": int(len(endpoint_stats)),
            "displayed_tests": int(len(displayed_new)),
            "maximum_family_q_bh27": float(endpoint_stats["q_bh27"].max()),
            "maximum_displayed_q_bh27": float(displayed_new["q_bh27"].max()),
            "old_maximum_displayed_q_bh27": float(displayed_old["q_bh27"].max()),
            "display_stars_unchanged": bool(endpoint_stars_unchanged),
        },
        "tile_time": tile_stats.iloc[0].astype(object).where(pd.notna(tile_stats.iloc[0]), None).to_dict(),
        "identity": {
            "raw_aliases": int(endpoint_rows["animal_id_alias"].nunique()),
            "canonical_biological_animals": int(endpoint_rows["animal_id"].nunique()),
            "biological_animals_with_multiple_aliases": int(split_groups.gt(1).sum()),
        },
        "source_hashes": {
            path.relative_to(REPO).as_posix(): sha256(path) for path in sources
        },
        "output_hashes": {
            path.name: sha256(path) for path in output_paths
        },
    }
    write_json(OUT / "VALIDATION.json", validation)
    with (OUT / "SOURCE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("role", "path", "sha256", "bytes"))
        for role, path in zip(
            ("endpoint_rows", "endpoint_parent_stats", "tile_time_rows", "tile_time_parent_stats"),
            sources,
            strict=True,
        ):
            writer.writerow((role, path.relative_to(REPO).as_posix(), sha256(path), path.stat().st_size))
    (OUT / "REPORT.md").write_text(
        "# ED1 canonical-animal clustered inference\n\n"
        f"Status: **{status}**. The 153 sessions contain 36 recording aliases but 29 "
        "biological animals. Seven animals were represented by two aliases. The endpoint "
        f"analysis retains one 27-test Benjamini--Hochberg family; all 18 displayed stars "
        f"are unchanged and the corrected maximum displayed adjusted p value is "
        f"{float(displayed_new['q_bh27'].max()):.6g} (maximum across the complete family, "
        f"{float(endpoint_stats['q_bh27'].max()):.6g}). The tile--time slope is unchanged "
        f"({float(tile_stats.iloc[0]['slope_loglog']):.9f}); canonical clustering gives "
        f"p={float(tile_stats.iloc[0]['p_cluster_t_two_sided']):.6g}, df="
        f"{int(tile_stats.iloc[0]['df_t'])}.\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
