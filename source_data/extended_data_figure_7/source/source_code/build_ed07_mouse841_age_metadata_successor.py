"""Build the ED7 first-window age-model successor after mouse 841 DOB recovery.

The current Main Figure 3 first-window table is the value authority.  This
script verifies that its 58 rows shared with the frozen ED7 record are exact,
reproduces the frozen 21-mouse models, attaches birth dates to all 59 current
recording cycles, and refits the three displayed experience-adjusted age
models with mouse-clustered CR1 covariance and a BH3 correction family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts import render_ed09_all_eligible_age_closure as age_science  # noqa: E402


PARENT_RECORD_NAME = "extended_data_fig07_2026-08-16_round16_upright_i_axis"
PARENT_RECORD = next(
    path
    for path in (
        ROOT / "docs" / "figure_version_records" / PARENT_RECORD_NAME,
        ROOT / "docs" / "figure_version_records_archive_2026-08-09" / PARENT_RECORD_NAME,
    )
    if path.is_dir()
)
MAIN3_RECORD = (
    ROOT
    / "docs"
    / "figure_version_records"
    / "main_figure3_2026-08-16_round9_available_window_slopes"
)
PARENT_POINTS = PARENT_RECORD / "plot_ready" / "ED07_FIRST30_PLOT_READY_MAINTAINED58.csv"
PARENT_MODELS = PARENT_RECORD / "plot_ready" / "ED07_FIRST30_AGE_MODELS_MAINTAINED58.csv"
MAIN3_POINTS = MAIN3_RECORD / "source_data" / "first30" / "combined_first30_points_all_metrics.csv"
PROVENANCE = (
    ROOT
    / "docs"
    / "extended_data_figure_panel_refresh_2026-07-12"
    / "source_rendered_new"
    / "extended_data_fig_09"
    / "all_eligible_age_closure_current58_2026-07-14"
    / "ed09_all_eligible_pair_provenance.csv"
)
DEMOGRAPHICS = ROOT / "docs" / "animal_metadata" / "animal_birth_gender_surgery.csv"
DEFAULT_OUT = ROOT / "docs" / "ed07_mouse841_age_metadata_2026-08-18"

DISPLAY_METRICS = (
    ("place_map_correlation", "Place map"),
    ("pv_correlation", "Population vector"),
    ("ensemble_rate_correlation", "Ensemble rate"),
)
METRIC_LABEL = dict(DISPLAY_METRICS)
METRIC_ORDER = {metric: index for index, (metric, _) in enumerate(DISPLAY_METRICS)}
MODEL_NUMERIC_COLUMNS = (
    "beta_per_30d",
    "cluster_se_cr1",
    "t_cluster",
    "p_cluster_two_sided",
    "ci95_low",
    "ci95_high",
    "r_squared",
    "cluster_df",
    "cr1_scale",
    "q_bh3_displayed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return relative(value)
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


def fit_age_family(frame: pd.DataFrame, specification: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, label in DISPLAY_METRICS:
        sub = frame.loc[frame["metric"].eq(metric)].copy()
        model_input = sub[["animal_id", "experience_index", "age_day1_days", "value"]].copy()
        fit = age_science.clustered_model(model_input, "value", specification)
        fit["metric"] = metric
        fit["metric_label"] = label
        fit["model_role"] = "age_plus_categorical_analysis_experience"
        rows.append(fit)
    result = pd.DataFrame(rows)
    result["q_bh3_displayed"] = age_science.bh_adjust(result["p_cluster_two_sided"])
    result["metric_order"] = result["metric"].map(METRIC_ORDER)
    return result.sort_values("metric_order").drop(columns="metric_order").reset_index(drop=True)


def load_parent() -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(PARENT_POINTS, dtype={"animal_id": str})
    for column in ("experience_index", "age_day1_days", "value"):
        points[column] = pd.to_numeric(points[column], errors="raise")
    models = pd.read_csv(PARENT_MODELS)
    return points, models


def reproduce_parent(points: pd.DataFrame, saved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reproduced = fit_age_family(points, "frozen58_reproduction")
    comparison = reproduced.merge(
        saved,
        on="metric",
        suffixes=("_recomputed", "_saved"),
        validate="one_to_one",
    )
    for column in MODEL_NUMERIC_COLUMNS:
        left = pd.to_numeric(comparison[f"{column}_recomputed"], errors="raise")
        right = pd.to_numeric(comparison[f"{column}_saved"], errors="raise")
        comparison[f"absolute_delta_{column}"] = np.abs(left - right)
        if not np.allclose(left, right, rtol=0.0, atol=1.0e-12):
            raise AssertionError(f"Frozen ED7 model reproduction failed for {column}")
    if set(reproduced["n_rows"].astype(int)) != {58}:
        raise AssertionError("Frozen ED7 reproduction no longer has 58 rows per metric")
    if set(reproduced["n_mice"].astype(int)) != {21}:
        raise AssertionError("Frozen ED7 reproduction no longer has 21 mice")
    return reproduced, comparison


def load_current59(parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(MAIN3_POINTS, dtype={"animal": str})
    values = values.loc[values["metric"].isin(METRIC_ORDER)].copy()
    values = values.rename(columns={"animal": "animal_id"})
    values["animal_id"] = values["animal_id"].astype(str)
    values["experience_index"] = pd.to_numeric(values["experience_index"], errors="raise").astype(int)
    values["value"] = pd.to_numeric(values["value"], errors="raise")
    if len(values) != 59 * len(DISPLAY_METRICS):
        raise AssertionError(f"Current Main Figure 3 first-window source has {len(values)} displayed rows")
    if values["animal_id"].nunique() != 22:
        raise AssertionError("Current Main Figure 3 first-window source is not 22 mice")
    if values.duplicated(["animal_id", "experience_index", "metric"]).any():
        raise AssertionError("Current Main Figure 3 first-window keys are not unique")

    shared = values.loc[values["animal_id"].ne("841")].merge(
        parent[["animal_id", "experience_index", "metric", "value"]],
        on=["animal_id", "experience_index", "metric"],
        suffixes=("_main3", "_parent"),
        validate="one_to_one",
    )
    shared["absolute_value_delta"] = np.abs(shared["value_main3"] - shared["value_parent"])
    if len(shared) != 58 * len(DISPLAY_METRICS) or not np.allclose(
        shared["value_main3"], shared["value_parent"], rtol=0.0, atol=0.0
    ):
        raise AssertionError("The 58 shared Main Figure 3 values differ from frozen ED7")

    provenance = pd.read_csv(PROVENANCE, dtype={"animal_id": str})
    provenance_keep = [
        "animal_id",
        "experience_index",
        "session_day1",
        "session_day2",
        "day1_session_date",
        "day2_session_date",
        "joint_deconvolution_shared_component_identity",
        "matching_basis",
        "neural_source_policy",
        "n_neurons_after_rate_gate",
        "support_match_target_n",
        "support_target_eligible",
        "support_target_exclusion_reason",
    ]
    provenance = provenance[provenance_keep].drop_duplicates(["animal_id", "experience_index"])
    provenance["experience_index"] = pd.to_numeric(
        provenance["experience_index"], errors="raise"
    ).astype(int)
    values = values.merge(
        provenance,
        on=["animal_id", "experience_index"],
        how="left",
        validate="many_to_one",
    )
    if values[["session_day1", "session_day2", "day1_session_date"]].isna().any().any():
        raise AssertionError("Current Main Figure 3 values did not fully join session provenance")

    demographics = pd.read_csv(DEMOGRAPHICS, dtype={"animal_id": str})
    values = values.merge(
        demographics[["animal_id", "birth_date", "gender", "metadata_status", "source"]],
        on="animal_id",
        how="left",
        validate="many_to_one",
    )
    values["birth_date_dt"] = pd.to_datetime(values["birth_date"], errors="coerce")
    values["day1_session_date_dt"] = pd.to_datetime(values["day1_session_date"], errors="coerce")
    values["age_day1_days"] = (
        values["day1_session_date_dt"] - values["birth_date_dt"]
    ).dt.days
    if values["age_day1_days"].isna().any():
        missing = sorted(values.loc[values["age_day1_days"].isna(), "animal_id"].unique())
        raise AssertionError(f"Current59 age metadata remain incomplete: {missing}")

    shared_age = values.loc[values["animal_id"].ne("841")].merge(
        parent[["animal_id", "experience_index", "metric", "age_day1_days"]],
        on=["animal_id", "experience_index", "metric"],
        suffixes=("_current", "_parent"),
        validate="one_to_one",
    )
    if not np.allclose(
        shared_age["age_day1_days_current"],
        shared_age["age_day1_days_parent"],
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("Ages for the 58 shared observations drifted")
    mouse841 = values.loc[values["animal_id"].eq("841")]
    if len(mouse841) != len(DISPLAY_METRICS):
        raise AssertionError("Mouse 841 is not represented once per displayed metric")
    if set(mouse841["birth_date"].astype(str)) != {"2025-02-06"}:
        raise AssertionError("Mouse 841 birth date is not 2025-02-06")
    if set(mouse841["gender"].astype(str)) != {"F"}:
        raise AssertionError("Mouse 841 sex is not F")
    if set(mouse841["age_day1_days"].astype(int)) != {120}:
        raise AssertionError("Mouse 841 Day 1 age is not 120 days")
    return values, shared


def make_plot_ready(values: pd.DataFrame) -> pd.DataFrame:
    result = values.copy()
    result["analysis_experience_index"] = result["experience_index"].astype(int)
    result["metric_label"] = result["metric"].map(METRIC_LABEL)
    result["source_scope"] = "accepted_main_figure3_first30_fixed_reference"
    result["source_vintage"] = (
        "2026-08-16 current Main Figure 3 first-window authority; "
        "mouse 841 birth date added 2026-08-18"
    )
    result["manual_neuron_exclusion_profile"] = (
        "none; estimator uses all joint component indices"
    )
    result["ultm_prior_gap_indicator"] = 0
    result["inherited_579_exclusion_indicator"] = 0
    result["is_targeted_addition"] = False
    result["metadata_update_indicator"] = result["animal_id"].eq("841")
    result["primary_scope"] = "maintained59; current Figure 3 values and complete birth dates"
    result["timebase_contract"] = (
        "Day1 maze_time [0,1800); selected-row exposure; full Day2 reference"
    )
    result["representation_contract"] = (
        "place=15x15 neuron tuning mean; PV=binwise population vector; "
        "rate=matched-neuron raw-C rate vector"
    )
    result["matching_contract"] = "adjacent-day joint-deconvolution index identity"
    result["reward_context"] = (
        "rewarded/unrewarded context inherited from current Figure 3 pair manifest; "
        "not a model covariate"
    )
    columns = [
        "animal_id",
        "session_day1",
        "session_day2",
        "analysis_experience_index",
        "experience_index",
        "age_day1_days",
        "metric",
        "metric_label",
        "value",
        "source_scope",
        "source_vintage",
        "window_type",
        "window_size",
        "support_policy",
        "source_bundles",
        "manual_neuron_exclusion_profile",
        "ultm_prior_gap_indicator",
        "inherited_579_exclusion_indicator",
        "is_targeted_addition",
        "metadata_update_indicator",
        "primary_scope",
        "timebase_contract",
        "representation_contract",
        "matching_contract",
        "reward_context",
    ]
    result["metric_order"] = result["metric"].map(METRIC_ORDER)
    result = result.sort_values(
        ["metric_order", "experience_index", "animal_id", "session_day1", "session_day2"],
        kind="stable",
    )
    return result[columns].reset_index(drop=True)


def make_support_annotation(values: pd.DataFrame) -> pd.DataFrame:
    row = values.loc[values["animal_id"].eq("841")].iloc[0]
    return pd.DataFrame(
        [
            {
                "animal_id": "841",
                "session_day1": row["session_day1"],
                "session_day2": row["session_day2"],
                "ed7_first_window_included": True,
                "age_day1_days": int(row["age_day1_days"]),
                "n_neurons_after_rate_gate": int(float(row["n_neurons_after_rate_gate"])),
                "separate_matched_support_audit_target": int(
                    float(row["support_match_target_n"])
                ),
                "separate_matched_support_audit_pass": False,
                "annotation": (
                    "Included in the current first-window analysis. The retained population "
                    "is 139 cells, below the separate 164-cell matched-support audit threshold."
                ),
                "scientific_basis": (
                    "Current Main Figure 3 accepts this recording cycle under the first-window "
                    "available-support estimator; the 164-cell threshold belongs to a separate "
                    "same-stage matched-support audit."
                ),
            }
        ]
    )


def write_outputs(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    parent_points, parent_models = load_parent()
    frozen_models, frozen_comparison = reproduce_parent(parent_points, parent_models)
    current, shared_values = load_current59(parent_points)
    plot_ready = make_plot_ready(current)
    models = fit_age_family(plot_ready, "maintained59_primary")
    support = make_support_annotation(current)
    cohort = (
        plot_ready.drop_duplicates(["animal_id", "experience_index", "session_day1", "session_day2"])
        [[
            "animal_id",
            "experience_index",
            "session_day1",
            "session_day2",
            "age_day1_days",
            "metadata_update_indicator",
            "source_bundles",
        ]]
        .sort_values(["experience_index", "animal_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(cohort) != 59 or cohort["animal_id"].nunique() != 22:
        raise AssertionError("Successor cohort is not 59 observations from 22 mice")
    if set(models["n_rows"].astype(int)) != {59} or set(models["n_mice"].astype(int)) != {22}:
        raise AssertionError("Successor models do not use 59 observations from 22 mice")

    paths = {
        "plot_ready": out / "ED07_FIRST30_PLOT_READY_MAINTAINED59.csv",
        "models": out / "ED07_FIRST30_AGE_MODELS_MAINTAINED59.csv",
        "cohort": out / "ED07_FIRST30_COHORT_MAINTAINED59.csv",
        "support": out / "ED07_MOUSE841_SUPPORT_ANNOTATION.csv",
        "frozen_models": out / "ED07_FROZEN58_MODEL_REPRODUCTION.csv",
        "frozen_comparison": out / "ED07_FROZEN58_MODEL_REPRODUCTION_CHECK.csv",
        "shared_values": out / "ED07_SHARED58_VALUE_IDENTITY_CHECK.csv",
    }
    plot_ready.to_csv(paths["plot_ready"], index=False, lineterminator="\n")
    models.to_csv(paths["models"], index=False, lineterminator="\n")
    cohort.to_csv(paths["cohort"], index=False, lineterminator="\n")
    support.to_csv(paths["support"], index=False, lineterminator="\n")
    frozen_models.to_csv(paths["frozen_models"], index=False, lineterminator="\n")
    frozen_comparison.to_csv(paths["frozen_comparison"], index=False, lineterminator="\n")
    shared_values.to_csv(paths["shared_values"], index=False, lineterminator="\n")

    checks = {
        "frozen58_models_reproduced_to_1e_12": True,
        "shared58_values_byte_numeric_identical": True,
        "shared58_ages_identical": True,
        "mouse841_birth_date": "2025-02-06",
        "mouse841_sex": "F",
        "mouse841_day1_age_days": 120,
        "successor_observations_per_metric": 59,
        "successor_mice": 22,
        "successor_model_family": "BH3 across the three displayed measures",
        "mouse841_first_window_included": True,
        "mouse841_separate_matched_support_audit": "139 retained cells; below target 164",
    }
    inputs = {
        label: {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for label, path in {
            "parent_points": PARENT_POINTS,
            "parent_models": PARENT_MODELS,
            "current_main3_points": MAIN3_POINTS,
            "session_provenance": PROVENANCE,
            "animal_demographics": DEMOGRAPHICS,
        }.items()
    }
    outputs = {
        label: {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for label, path in paths.items()
    }
    validation = {
        "schema": "ed07_mouse841_age_metadata_successor/v1",
        "status": "PASS",
        "estimand": (
            "first 30 min of maze time on Day i versus the full Day i+1 template, "
            "with age and categorical experience stage"
        ),
        "statistical_unit": "recording cycle; CR1 covariance clustered by biological mouse",
        "representation_space": {
            "place_map_correlation": "15x15 grid-position tuning maps",
            "pv_correlation": "grid-position population vectors",
            "ensemble_rate_correlation": "matched-neuron ensemble-rate vectors",
        },
        "reward_context": "inherited from the current Main Figure 3 pair manifest; not a model covariate",
        "checks": checks,
        "inputs": inputs,
        "outputs": outputs,
        "models": models.to_dict(orient="records"),
    }
    validation_path = out / "VALIDATION.json"
    validation_path.write_text(
        json.dumps(json_ready(validation), indent=2) + "\n", encoding="utf-8"
    )
    readme = out / "README.md"
    readme.write_text(
        "# ED7 mouse 841 age metadata update\n\n"
        "Mouse 841 was born on 2025-02-06 and was 120 days old at the first "
        "recording. The current Main Figure 3 first-window values already include "
        "this recording cycle. The 58 shared values and their frozen ED7 models "
        "were reproduced before the three displayed age models were refitted with "
        "59 observations from 22 mice. Panels d to f have no dependency on this "
        "metadata update.\n\n"
        "Mouse 841 has 139 cells after the separate rate gate, below the 164-cell "
        "matched-support audit target. This does not exclude it from the current "
        "first-window available-support estimator.\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "out_dir": relative(out),
        "validation": relative(validation_path),
        "models": models[
            [
                "metric",
                "n_rows",
                "n_mice",
                "beta_per_30d",
                "cluster_se_cr1",
                "p_cluster_two_sided",
                "q_bh3_displayed",
            ]
        ].to_dict(orient="records"),
    }


def validate_frozen_only() -> dict[str, Any]:
    points, models = load_parent()
    reproduced, comparison = reproduce_parent(points, models)
    return {
        "status": "PASS",
        "observations_per_metric": int(reproduced["n_rows"].iloc[0]),
        "mice": int(reproduced["n_mice"].iloc[0]),
        "maximum_absolute_numeric_delta": float(
            comparison[
                [column for column in comparison.columns if column.startswith("absolute_delta_")]
            ].to_numpy(dtype=float).max()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frozen-only", action="store_true")
    args = parser.parse_args()
    result = validate_frozen_only() if args.frozen_only else write_outputs(args.out_dir.resolve())
    print(json.dumps(json_ready(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
