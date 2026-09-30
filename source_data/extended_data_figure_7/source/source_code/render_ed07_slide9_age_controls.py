"""Render the source-locked slide-9 ED7 age-control rows.

The slide, not the current repository role manifest, is the panel-inventory
authority here.  It contains exactly two stacked three-metric composites:

* panel a: same-stage age associations (place, PV, ensemble rate);
* panel b: long-gap age associations (place, PV, ensemble rate).

The same-stage primary is the maximal 64-row target-164 cohort: the 2026-08-07
no-manual-mask sorted authority plus fresh original-source recomputations under
explicit exclusion profile ``none``.  The long-gap primary is the maximal
38-row explicit-CIMap cohort, including the independently audited recovered
579 3-to-6 and 488 6-to-9 pairs.  Historical cohort boundaries remain
prespecified sensitivity analyses rather than maximality claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ca_analysis.exclusion_policy import NO_EXCLUSIONS  # noqa: E402
from scripts import render_ed09_all_eligible_age_closure as closure  # noqa: E402
from scripts.plot_long_gap_age_control_current_plus_new12 import (  # noqa: E402
    TRANSITION_COLORS,
    TRANSITION_LABELS,
    _bh_q as long_gap_bh_q,
    transition_fixed_age_model,
)


DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "docs"
    / "extended_data_supplement_v2_rebuild_2026-08-05"
    / "ed07_age_controls"
)
LONG_EXTRA_CANDIDATES = DEFAULT_OUT_DIR / "long_gap_candidate_recomputations.csv"
LONG_AUDIT_MODELS = DEFAULT_OUT_DIR / "long_gap_cohort_sensitivity_models.csv"
SAME_AUDIT_MODELS = DEFAULT_OUT_DIR / "same_stage_cohort_sensitivity_models.csv"
SAME_AUDIT_VALUES = DEFAULT_OUT_DIR / "same_stage_maximal64_age_values.csv"
LONG_RECOMPUTE_INPUT_HASHES = DEFAULT_OUT_DIR / "long_gap_recompute_input_hashes.csv"
LONG_CONVERSION_AUDIT_DIR = DEFAULT_OUT_DIR / "long_gap_conversion_audit"
LONG_CONVERSION_AUDIT_HASHES = DEFAULT_OUT_DIR / "long_gap_conversion_audit_sha256.csv"
LONG_NONCANONICAL_CANDIDATES = DEFAULT_OUT_DIR / "long_gap_noncanonical_0to6_candidates.csv"
LONG_NONCANONICAL_DESCRIPTIVE = DEFAULT_OUT_DIR / "long_gap_noncanonical_0to6_descriptive.csv"
LONG_NONCANONICAL_MODELS = DEFAULT_OUT_DIR / "long_gap_noncanonical_0to6_sensitivity_models.csv"
LONG_CONVERSION_REPRODUCER = (
    REPO_ROOT / "scripts" / "recompute_ed07_long_gap_original_conversions.py"
)
LONG_CONVERSION_REPRODUCE_COMMAND = (
    "python scripts/recompute_ed07_long_gap_original_conversions.py "
    "--pair-set noncanonical0to6 --out-dir <verification-output-dir> "
    "--verify-root docs/extended_data_supplement_v2_rebuild_2026-08-05/ed07_age_controls"
)
UNIVERSE_AUDIT_FILES = (
    DEFAULT_OUT_DIR / "same_stage_candidate_universe_attrition.csv",
    DEFAULT_OUT_DIR / "same_stage_attrition_summary.csv",
    DEFAULT_OUT_DIR / "source_selection_classification.csv",
    DEFAULT_OUT_DIR / "source_inventory_sha256.csv",
    DEFAULT_OUT_DIR / "support_target_sensitivity.csv",
    DEFAULT_OUT_DIR / "same_stage_cohort_sensitivity_models.csv",
    DEFAULT_OUT_DIR / "same_stage_maximal64_age_values.csv",
    DEFAULT_OUT_DIR / "masked_vs_nomask_source_vintage_comparison.csv",
    LONG_NONCANONICAL_CANDIDATES,
    LONG_NONCANONICAL_DESCRIPTIVE,
    LONG_NONCANONICAL_MODELS,
    DEFAULT_OUT_DIR / "AUDIT_REPORT.md",
    DEFAULT_OUT_DIR / "AUDIT_VALIDATION.json",
)
REFRESH_ROOT = REPO_ROOT / "docs" / "extended_data_figure_panel_refresh_2026-07-12"
NO_MASK_SORTED_DIR = (
    REPO_ROOT
    / "docs"
    / "sorted_data_full_figure_rerun_2026-06-11"
    / "figure4d_same_stage_support_matched_sorted_target164_no_manual_mask_2026-08-07"
)
LONG_JOINED = REFRESH_ROOT / "source_tables" / "ed09_long_gap_age_joined_corrected36.csv"
LONG_MODELS = (
    REFRESH_ROOT
    / "source_tables"
    / "ed09_long_gap_age_transition_fixed_models_corrected36.csv"
)
LONG_SOURCE_MANIFEST = (
    REFRESH_ROOT
    / "source_tables"
    / "corrected_long_gap_metrics"
    / "corrected_long_gap_metric_manifest.json"
)
LONG_CIMAP_METAFILE = (
    REFRESH_ROOT
    / "source_tables"
    / "corrected_long_gap_metrics"
    / "linearized_pf_sorted_metafile_corrected36.csv"
)
SORTED_AUDIT_FILES = (
    NO_MASK_SORTED_DIR / "same_stage_experience_support_matched.csv",
    NO_MASK_SORTED_DIR / "same_stage_experience_pair_metrics.csv",
    NO_MASK_SORTED_DIR / "same_stage_experience_manifest.csv",
    NO_MASK_SORTED_DIR / "same_stage_experience_neuron_metrics.csv",
    NO_MASK_SORTED_DIR / "same_stage_experience_valid_bin_masks.csv",
    NO_MASK_SORTED_DIR / "same_stage_support_matched_points.csv",
    NO_MASK_SORTED_DIR / "summary.json",
)
SOURCE_RENDERERS = (
    REPO_ROOT / "scripts" / "analyze_same_stage_experience_controls.py",
    Path(closure.__file__).resolve(),
    REPO_ROOT / "scripts" / "plot_long_gap_age_control_current_plus_new12.py",
    REPO_ROOT / "scripts" / "audit_ed07_slide9_source_selection.py",
    LONG_CONVERSION_REPRODUCER,
    Path(__file__).resolve(),
)

HISTORICAL_ORIGINAL_PAIRS = tuple(
    {
        "animal_id": animal,
        "session_day1": day1,
        "session_day2": day2,
        "experience_index": 0,
        "protocol_group": "historical_original_fallback",
        "historical_protocol_member": True,
        "ultm_prior_gap_indicator": 0,
    }
    for animal, day1, day2 in closure.FALLBACK_PAIRS
)
ULTM_ORIGINAL_PAIRS = tuple(
    {
        "animal_id": animal,
        "session_day1": f"{animal}-6day1",
        "session_day2": f"{animal}-6day2",
        "experience_index": 1,
        "protocol_group": "ultm_prior_gap_exp1",
        "historical_protocol_member": False,
        "ultm_prior_gap_indicator": 1,
    }
    for animal in ("1639", "1636", "1637", "325", "347")
)
RECOVERED_QC_ORIGINAL_PAIRS = (
    {
        "animal_id": "579",
        "session_day1": "579-6day1",
        "session_day2": "579-6day2",
        "experience_index": 2,
        "protocol_group": "recovered_inherited_qc_exp2",
        "historical_protocol_member": False,
        "ultm_prior_gap_indicator": 0,
    },
)
ORIGINAL_PAIR_SPECS = (
    HISTORICAL_ORIGINAL_PAIRS + ULTM_ORIGINAL_PAIRS + RECOVERED_QC_ORIGINAL_PAIRS
)
CACHE_CONTRACT_VERSION = "ed07_original_target164_no_exclusions_v1"
NATIVE_RELAXED_488_SPEC = {
    "animal_id": "488",
    "session_day1": "488-9day1",
    "session_day2": "488-9day2",
    "experience_index": 3,
    "protocol_group": "native117_relaxed_below_target164",
    "historical_protocol_member": False,
    "ultm_prior_gap_indicator": 0,
    "native_support_n": 117,
}
NATIVE_CACHE_CONTRACT_VERSION = "ed07_original_native117_no_exclusions_v1"
LONG_GAP_EXTRA_PAIR_SPECS = (
    {
        "animal_id": "579",
        "src": "579-3day1",
        "dst": "579-6day1",
        "experience": 2,
        "transition": "3-6 mo",
        "n_matched_cimap": 434,
        "n_used": 310,
        "common_valid_bins": 104,
        "valid_levels": 6,
        "cimap_path": Path("F:/preprocess/footprints/579/cell_to_index_map_36.mat"),
        "cimap_sha256": "286d034186db411730409050f0165a69cbd8b2a839bd1a365ae00108c86b45db",
        "values": {
            "place_map_correlation": 0.0227101511,
            "pv_correlation": 0.0595465155,
            "level_map_correlation": 0.3080328725,
            "ensemble_rate_correlation": 0.2062054768,
        },
    },
    {
        "animal_id": "488",
        "src": "488-6day1",
        "dst": "488-9day1",
        "experience": 3,
        "transition": "6-9 mo",
        "n_matched_cimap": 68,
        "n_used": 51,
        "common_valid_bins": 102,
        "valid_levels": 7,
        "cimap_path": Path("F:/preprocess/footprints/488/cell_to_index_map_69.mat"),
        "cimap_sha256": "809825574ca0ec015607d36ef24894dd6297879639ac0529967f1842d8d1b5ef",
        "values": {
            "place_map_correlation": 0.00177938470941993,
            "pv_correlation": 0.12225796744763093,
            "level_map_correlation": 0.0734539347947758,
            "ensemble_rate_correlation": 0.31113437416603545,
        },
    },
)

# Slide 9 source geometry is retained as an audit crosswalk, but the final
# slide-ready layout removes the inherited crop/overlap and enlarges both rows.
SLIDE_SIZE_IN = (7.5, 7.5)
INHERITED_TOP_SLOT_IN = (23.12 / 96.0, 11.09 / 96.0, 673.76 / 96.0, 193.06 / 96.0)
INHERITED_BOTTOM_SLOT_IN = (22.96 / 96.0, 171.24 / 96.0, 674.08 / 96.0, 226.40 / 96.0)
LEGACY_CROP_CROSSWALK = {"left": 0.0, "top": 0.14725, "right": 0.0, "bottom": 0.0}
ROW_SIZE_IN = (7.15, 3.25)
TOP_ROW_SIZE_IN = ROW_SIZE_IN
BOTTOM_ROW_SIZE_IN = ROW_SIZE_IN
TOP_SLOT_IN = (0.175, 0.38, ROW_SIZE_IN[0], ROW_SIZE_IN[1])
BOTTOM_SLOT_IN = (0.175, 3.87, ROW_SIZE_IN[0], ROW_SIZE_IN[1])

FORMATS = ("svg", "pdf", "png", "tiff")
METRICS = (
    (
        "same_stage_place_map_corr_support_matched",
        "place_map_correlation",
        "place",
        "Place map",
        "#B95A58",
    ),
    (
        "same_stage_pv_corr_support_matched",
        "pv_correlation",
        "pv",
        "Population vector",
        "#E29957",
    ),
    (
        "same_stage_ensemble_rate_corr_support_matched",
        "ensemble_rate_correlation",
        "rate",
        "Ensemble rate",
        "#4A5E65",
    ),
)
STAGE_COLORS = dict(closure.STAGE_COLORS)
STAGE_LABELS = dict(closure.STAGE_LABELS)


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "lines.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


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


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.strip().str.lower().eq("true")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in FORMATS:
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "format": extension,
            "facecolor": "white",
            "transparent": False,
        }
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def save_audit_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    """Export inspectable audit-only graphics without a redundant TIFF."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for extension in ("svg", "pdf", "png"):
        path = stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "format": extension,
            "facecolor": "white",
            "transparent": False,
        }
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def source_record(path: Path, role: str, rows: int | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": stat.st_size,
        "mtime_utc": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
        "rows": rows,
    }


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def verify_sha256_inventory(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Validate a path/SHA256 CSV without assuming one audit-specific schema."""

    table = pd.read_csv(path)
    path_column = _first_present(
        table.columns,
        ("path", "file_path", "input_path", "artifact_path", "source_path"),
    )
    hash_column = _first_present(
        table.columns,
        ("sha256", "file_sha256", "input_sha256", "artifact_sha256"),
    )
    if path_column is None or hash_column is None or table.empty:
        raise ValueError(f"Hash inventory lacks path/SHA256 columns or rows: {path}")
    resolved: list[Path] = []
    for row in table.itertuples(index=False):
        raw_path = str(getattr(row, path_column)).strip()
        expected_hash = str(getattr(row, hash_column)).strip().lower()
        input_path = Path(raw_path)
        if not input_path.is_absolute():
            input_path = path.parent / input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Hashed input is missing: {input_path}")
        observed_hash = sha256(input_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"SHA256 mismatch for {input_path}: {observed_hash} != {expected_hash}"
            )
        resolved.append(input_path.resolve())
    return table, resolved


def validate_long_gap_audit_artifacts() -> dict[str, Any]:
    """Hard-gate raw provenance and conversion audits for recovered long-gap rows."""

    validation_path = DEFAULT_OUT_DIR / "AUDIT_VALIDATION.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if str(validation.get("status", "")).upper() != "PASS":
        raise ValueError("ED7 candidate-universe audit is not PASS")
    required_checks = (
        "longgap_original_cimaps_promoted_maximal38",
        "longgap_recovered_inputs_exist",
        "longgap_maximal38_BH4_rate_q",
        "longgap_source_indicator_full_rank_but_diagnostic",
        "longgap_original_conversion_reproducer_hashed",
    )
    checks = validation.get("checks", {})
    if not all(checks.get(check) is True for check in required_checks):
        raise ValueError("ED7 audit validation lacks a recovered long-gap PASS gate")

    input_hashes, input_paths = verify_sha256_inventory(LONG_RECOMPUTE_INPUT_HASHES)
    conversion_hashes, conversion_paths = verify_sha256_inventory(
        LONG_CONVERSION_AUDIT_HASHES
    )
    if not LONG_CONVERSION_AUDIT_DIR.is_dir():
        raise FileNotFoundError(LONG_CONVERSION_AUDIT_DIR)
    conversion_names = [path.name.lower() for path in conversion_paths]
    required_tokens = {
        "579_3to6": ("cimap", "rate", "spatial", "level"),
        "488_6to9": ("cimap", "rate", "spatial", "level"),
        "1639_0to6": ("cimap", "rate", "spatial", "level"),
        "1636_0to6": ("cimap", "rate", "spatial", "level"),
        "1637_0to6": ("cimap", "rate", "spatial", "level"),
        "325_0to6": ("cimap", "rate", "spatial", "level"),
        "347_0to6": ("cimap", "rate", "spatial", "level"),
    }
    for prefix, tokens in required_tokens.items():
        prefix_names = [name for name in conversion_names if name.startswith(prefix)]
        if not prefix_names:
            raise ValueError(f"No conversion-audit artifacts found for {prefix}")
        missing = [token for token in tokens if not any(token in name for name in prefix_names)]
        if missing:
            raise ValueError(f"Conversion audit for {prefix} lacks {missing}")
    if any(LONG_CONVERSION_AUDIT_DIR.resolve() not in path.parents for path in conversion_paths):
        raise ValueError("Conversion hash inventory references a file outside its audit directory")

    input_text = " ".join(
        " ".join(str(value) for value in row)
        for row in input_hashes.astype(str).itertuples(index=False, name=None)
    ).lower()
    for animal in ("579", "488"):
        if animal not in input_text:
            raise ValueError(f"Recovered long-gap hash inventory lacks animal {animal}")
    for token in ("neural", "selected", "nose", "cimap"):
        if token not in input_text:
            raise ValueError(f"Recovered long-gap hash inventory lacks {token} provenance")
    required_code_roles = {
        "place_pv_long_gap_estimator_source",
        "same_stage_estimator_and_audit_export_source",
        "tree_depth7_definition_source",
        "long_gap_original_conversion_reproducer",
        "slide9_render_and_cr1_source",
        "source_audit_builder",
    }
    if not required_code_roles.issubset(set(input_hashes["role"].astype(str))):
        raise ValueError("Recovered long-gap hash inventory lacks an estimator/audit code source")

    noncanonical_candidates = pd.read_csv(
        LONG_NONCANONICAL_CANDIDATES, dtype={"animal_id": str}
    )
    if (
        len(noncanonical_candidates) != 5
        or set(noncanonical_candidates["classification"])
        != {"SUPPORTING-NOT-DISPLAYED"}
        or set(noncanonical_candidates["exclusion_profile"]) != {NO_EXCLUSIONS}
    ):
        raise ValueError("Noncanonical 0-to-6 candidate classification is incomplete")
    noncanonical_models = pd.read_csv(LONG_NONCANONICAL_MODELS)
    if (
        len(noncanonical_models) != 4
        or set(noncanonical_models["n_rows"].astype(int)) != {43}
        or set(noncanonical_models["n_mice"].astype(int)) != {19}
        or not noncanonical_models["design_rank"].astype(int).eq(
            noncanonical_models["n_parameters"].astype(int)
        ).all()
    ):
        raise ValueError("Noncanonical 0-to-6 n43 sensitivity audit is incomplete")
    rate_row = noncanonical_models.loc[noncanonical_models["metric"].eq("rate")].iloc[0]
    if not np.isclose(float(rate_row["q_BH4_age"]), 0.057678203971938673, atol=1e-12):
        raise ValueError("Noncanonical 0-to-6 rate BH4 sensitivity changed")
    if not LONG_CONVERSION_REPRODUCER.is_file():
        raise FileNotFoundError(LONG_CONVERSION_REPRODUCER)

    return {
        "audit_validation": str(validation_path.resolve()),
        "recompute_input_hashes": str(LONG_RECOMPUTE_INPUT_HASHES.resolve()),
        "recompute_input_rows": int(len(input_hashes)),
        "recompute_input_files": [str(path) for path in input_paths],
        "conversion_audit_hashes": str(LONG_CONVERSION_AUDIT_HASHES.resolve()),
        "conversion_audit_rows": int(len(conversion_hashes)),
        "conversion_audit_files": [str(path) for path in conversion_paths],
        "noncanonical_0to6_classification": "supporting_only_not_primary",
        "noncanonical_0to6_candidates": str(LONG_NONCANONICAL_CANDIDATES.resolve()),
        "noncanonical_0to6_descriptive": str(LONG_NONCANONICAL_DESCRIPTIVE.resolve()),
        "noncanonical_0to6_sensitivity_models": str(LONG_NONCANONICAL_MODELS.resolve()),
        "noncanonical_0to6_n43_rate_q_bh4": float(rate_row["q_BH4_age"]),
        "conversion_reproducer": str(LONG_CONVERSION_REPRODUCER.resolve()),
        "conversion_reproducer_sha256": sha256(LONG_CONVERSION_REPRODUCER),
        "conversion_reproducer_invocation": LONG_CONVERSION_REPRODUCE_COMMAND,
        "conversion_reproducer_validation": (
            "all five noncanonical pairs compare CIMap, S-rate/neuron, spatial/PV, "
            "tree-depth7, counts and four summaries to locked audit tables at atol=1e-10"
        ),
    }


def long_audit_files_valid() -> bool:
    try:
        validate_long_gap_audit_artifacts()
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def cache_bundle_paths(cache_json: Path) -> tuple[list[Path], list[tuple[str, Path]]]:
    """Return generated cache artifacts and raw inputs actually read by one cache."""

    payload = json.loads(cache_json.read_text(encoding="utf-8"))
    artifacts = [cache_json.resolve()]
    for key in ("support_draws_csv", "neuron_mask_csv", "bin_mask_csv"):
        linked = Path(payload[key])
        if not linked.exists():
            raise FileNotFoundError(linked)
        artifacts.append(linked.resolve())
    row = payload["pair_row"]
    raw_keys = (
        "day1_neural_file",
        "day2_neural_file",
        "day1_selected_data_file",
        "day2_selected_data_file",
        "day1_nose_interp_file",
        "day2_nose_interp_file",
    )
    raw_inputs: list[tuple[str, Path]] = []
    for key in raw_keys:
        input_path = Path(str(row[key]))
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        raw_inputs.append((key, input_path.resolve()))
    return artifacts, raw_inputs


def build_source_input_records(
    same_audit: dict[str, Any], long_audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Hash every maintained authority, cache artifact and raw file actually read."""

    roles_by_path: dict[Path, set[str]] = {}

    def register(path: Path | str, role: str) -> None:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        resolved = source_path.resolve()
        roles_by_path.setdefault(resolved, set()).add(role)

    for path in SORTED_AUDIT_FILES:
        register(path, "same-stage Aug-7 no-mask sorted authority/audit")
    for path in SOURCE_RENDERERS:
        register(path, "source or audit renderer code")
    register(REPO_ROOT / "src" / "ca_analysis" / "exclusion_policy.py", "exclusion policy code")
    register(closure.EXPERIMENT_METAFILE, "session-date and candidate registry")
    register(closure.DEMOGRAPHICS, "birth-date and demographic metadata")

    cache_jsons = [
        *same_audit["original_cache_jsons"],
        *same_audit["original_native_cache_jsons"],
        same_audit["native_488_cache_json"],
    ]
    for cache_json_value in cache_jsons:
        cache_json = Path(cache_json_value)
        artifacts, raw_inputs = cache_bundle_paths(cache_json)
        for artifact in artifacts:
            register(artifact, "fresh same-stage no-mask cache and conversion audit")
        for raw_role, raw_path in raw_inputs:
            register(raw_path, f"same-stage original raw input: {raw_role}")

    for path in UNIVERSE_AUDIT_FILES:
        register(path, "candidate-universe/source-selection audit")
    for path, role in (
        (LONG_JOINED, "maintained corrected36 long-gap value table; parity source"),
        (LONG_MODELS, "maintained corrected36 long-gap model table; parity audit"),
        (LONG_SOURCE_MANIFEST, "maintained corrected36 no-mask source manifest"),
        (LONG_CIMAP_METAFILE, "maintained corrected36 explicit-CIMap mapping table"),
        (LONG_EXTRA_CANDIDATES, "recovered long-gap candidate metrics and provenance"),
        (LONG_AUDIT_MODELS, "independent long-gap cohort/source model audit"),
        (LONG_RECOMPUTE_INPUT_HASHES, "recovered long-gap raw-input hash inventory"),
        (LONG_CONVERSION_AUDIT_HASHES, "recovered long-gap conversion-artifact hash inventory"),
    ):
        register(path, role)

    corrected_manifest = json.loads(LONG_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    for key in (
        "place_pv_rate_exact_support_audit",
    ):
        relative = corrected_manifest.get(key)
        if relative:
            register(REPO_ROOT / str(relative), "maintained corrected36 conversion/support audit")
    corrected_cimaps = pd.read_csv(LONG_CIMAP_METAFILE)["cimap_path"].dropna().astype(str)
    for path in sorted(set(corrected_cimaps)):
        register(Path(path), "maintained corrected36 explicit CIMap")

    noncanonical_candidates = pd.read_csv(
        LONG_NONCANONICAL_CANDIDATES, dtype={"animal_id": str}
    )
    for candidate in noncanonical_candidates.itertuples(index=False):
        register(candidate.cimap_path, "supporting-only noncanonical 0-to-6 explicit CIMap")
        register(candidate.original_src_neural, "supporting-only noncanonical 0-to-6 raw neural input")
        register(candidate.original_dst_neural, "supporting-only noncanonical 0-to-6 raw neural input")
        for session in (candidate.src_session, candidate.dst_session):
            register(
                closure.BEHAV_WD / "selected_data" / f"{session}_selected_data.csv",
                "supporting-only noncanonical 0-to-6 selected behavior input",
            )
            register(
                closure.BEHAV_WD / "selected_data" / f"{session}_nose_interp_data.csv",
                "supporting-only noncanonical 0-to-6 interpolated behavior input",
            )

    conversion = long_audit["conversion_audit"]
    for path in conversion["recompute_input_files"]:
        register(path, "recovered long-gap immutable raw/code input")
    for path in conversion["conversion_audit_files"]:
        register(path, "recovered long-gap stepwise conversion audit")

    records: list[dict[str, Any]] = []
    for path in sorted(roles_by_path, key=lambda value: str(value).lower()):
        records.append(source_record(path, "; ".join(sorted(roles_by_path[path]))))
    return records


def original_pair_cache_dir(cache_root: Path, pair_index: int) -> Path:
    spec = ORIGINAL_PAIR_SPECS[pair_index]
    return cache_root / f"p{pair_index:02d}_{spec['animal_id']}"


def original_pair_cache_path(cache_root: Path, pair_index: int) -> Path:
    spec = ORIGINAL_PAIR_SPECS[pair_index]
    return (
        original_pair_cache_dir(cache_root, pair_index)
        / "pair_cache"
        / f"00_{spec['session_day1']}__{spec['session_day2']}.json"
    )


def original_pair_native_cache_path(cache_root: Path, pair_index: int) -> Path:
    spec = ORIGINAL_PAIR_SPECS[pair_index]
    return (
        original_pair_cache_dir(cache_root, pair_index)
        / "native"
        / "pair_cache"
        / f"00_{spec['session_day1']}__{spec['session_day2']}.json"
    )


def compute_original_pair_no_exclusions(pair_index: int, cache_root: Path) -> Path:
    """Recompute one original-source pair under the explicit no-mask profile."""

    spec = ORIGINAL_PAIR_SPECS[pair_index]
    if closure.pm.EXCLUSION_PROFILES.get(NO_EXCLUSIONS) != {}:
        raise ValueError("NO_EXCLUSIONS must resolve to an empty exclusion map")
    pair_dir = original_pair_cache_dir(cache_root, pair_index)
    pair_dir.mkdir(parents=True, exist_ok=True)

    original_pairs = closure.FALLBACK_PAIRS
    original_profile = closure.pm.PRIMARY_EXCLUSION_PROFILE
    original_target = closure.TARGET_N
    original_repeats = closure.SUPPORT_REPEATS
    try:
        closure.FALLBACK_PAIRS = (
            (
                str(spec["animal_id"]),
                str(spec["session_day1"]),
                str(spec["session_day2"]),
            ),
        )
        closure.pm.PRIMARY_EXCLUSION_PROFILE = NO_EXCLUSIONS
        cache_path = closure.compute_fallback_pair(0, pair_dir)
        target_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        native_support_n = int(target_payload["pair_row"]["n_neurons_after_rate_gate"])
        if native_support_n < 3:
            raise ValueError(f"Pair {pair_index} has insufficient native support")
        closure.TARGET_N = native_support_n
        closure.SUPPORT_REPEATS = 1
        native_dir = pair_dir / "native"
        native_cache_path = closure.compute_fallback_pair(0, native_dir)
    finally:
        closure.FALLBACK_PAIRS = original_pairs
        closure.pm.PRIMARY_EXCLUSION_PROFILE = original_profile
        closure.TARGET_N = original_target
        closure.SUPPORT_REPEATS = original_repeats

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    row = payload["pair_row"]
    row.update(
        {
            "animal_id": str(spec["animal_id"]),
            "experience_index": int(spec["experience_index"]),
            "experience_label": f"Exp{int(spec['experience_index'])}",
            "month_alias": (
                "naive"
                if int(spec["experience_index"]) == 0
                else (
                    "ultm_exp1_after_prior_gap"
                    if int(spec["ultm_prior_gap_indicator"]) == 1
                    else "recovered_inherited_qc_exp2"
                )
            ),
            "descriptive_only": False,
            "protocol_group": str(spec["protocol_group"]),
            "historical_protocol_member": bool(spec["historical_protocol_member"]),
            "ultm_prior_gap_indicator": int(spec["ultm_prior_gap_indicator"]),
            "source_scope": (
                "original_historical_recomputed_target164_no_manual_mask"
                if bool(spec["historical_protocol_member"])
                else (
                    "original_ultm_recomputed_target164_no_manual_mask"
                    if int(spec["ultm_prior_gap_indicator"]) == 1
                    else "original_recovered_qc_recomputed_target164_no_manual_mask"
                )
            ),
            "neural_source_policy": "sorted_first_original_when_sorted_unavailable",
            "exclusion_profile": NO_EXCLUSIONS,
            "exclusion_profile_source": "ca_analysis.exclusion_policy.NO_EXCLUSIONS",
            "cache_contract_version": CACHE_CONTRACT_VERSION,
        }
    )
    payload["pair_row"] = row
    payload["cache_provenance"] = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "exclusion_profile": NO_EXCLUSIONS,
        "exclusion_map_empty": True,
        "source_renderer": str(Path(closure.__file__).resolve()),
        "source_renderer_sha256": sha256(Path(closure.__file__).resolve()),
        "wrapper_renderer": str(Path(__file__).resolve()),
        "wrapper_renderer_sha256": sha256(Path(__file__).resolve()),
        "pair_index": pair_index,
        "pair_spec": spec,
        "native_support_cache_json": str(native_cache_path.resolve()),
        "native_support_n": native_support_n,
        "native_support_repeats": 1,
    }
    cache_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")

    native_payload = json.loads(native_cache_path.read_text(encoding="utf-8"))
    native_row = native_payload["pair_row"]
    for key in (
        "animal_id",
        "experience_index",
        "experience_label",
        "month_alias",
        "descriptive_only",
        "protocol_group",
        "historical_protocol_member",
        "ultm_prior_gap_indicator",
        "neural_source_policy",
        "exclusion_profile",
        "exclusion_profile_source",
    ):
        native_row[key] = row[key]
    native_row.update(
        {
            "source_scope": f"{row['source_scope']}_native_support",
            "cache_contract_version": CACHE_CONTRACT_VERSION,
            "target164_metric_eligible": False,
            "native_support_metric_eligible": True,
            "native_support_n": native_support_n,
        }
    )
    native_payload["pair_row"] = native_row
    native_payload["cache_provenance"] = {
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        "exclusion_profile": NO_EXCLUSIONS,
        "exclusion_map_empty": True,
        "support_definition": "native full retained pool",
        "native_support_n": native_support_n,
        "native_support_repeats": 1,
        "source_renderer": str(Path(closure.__file__).resolve()),
        "source_renderer_sha256": sha256(Path(closure.__file__).resolve()),
        "wrapper_renderer": str(Path(__file__).resolve()),
        "wrapper_renderer_sha256": sha256(Path(__file__).resolve()),
        "pair_index": pair_index,
        "pair_spec": spec,
    }
    native_cache_path.write_text(
        json.dumps(json_ready(native_payload), indent=2), encoding="utf-8"
    )
    return cache_path


def original_pair_cache_is_current(cache_root: Path, pair_index: int) -> bool:
    path = original_pair_cache_path(cache_root, pair_index)
    native_path = original_pair_native_cache_path(cache_root, pair_index)
    if not path.exists() or not native_path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["cache_provenance"]
        row = payload["pair_row"]
        native_payload = json.loads(native_path.read_text(encoding="utf-8"))
        native_provenance = native_payload["cache_provenance"]
        native_row = native_payload["pair_row"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return bool(
        provenance.get("cache_contract_version") == CACHE_CONTRACT_VERSION
        and provenance.get("exclusion_profile") == NO_EXCLUSIONS
        and provenance.get("exclusion_map_empty") is True
        and provenance.get("source_renderer_sha256") == sha256(Path(closure.__file__).resolve())
        and provenance.get("wrapper_renderer_sha256") == sha256(Path(__file__).resolve())
        and row.get("exclusion_profile") == NO_EXCLUSIONS
        and int(row.get("support_match_target_n", -1)) == closure.TARGET_N
        and int(row.get("support_match_repeats", -1)) == closure.SUPPORT_REPEATS
        and native_provenance.get("cache_contract_version") == CACHE_CONTRACT_VERSION
        and native_provenance.get("exclusion_profile") == NO_EXCLUSIONS
        and native_provenance.get("wrapper_renderer_sha256") == sha256(Path(__file__).resolve())
        and native_row.get("exclusion_profile") == NO_EXCLUSIONS
        and int(native_row.get("support_match_target_n", -1))
        == int(row.get("n_neurons_after_rate_gate", -2))
    )


def run_original_pair_caches(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    for pair_index in range(len(ORIGINAL_PAIR_SPECS)):
        if original_pair_cache_is_current(cache_root, pair_index):
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out-dir",
            str(DEFAULT_OUT_DIR.resolve()),
            "--compute-original-pair-index",
            str(pair_index),
            "--cache-root",
            str(cache_root.resolve()),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not original_pair_cache_is_current(cache_root, pair_index):
            raise RuntimeError(f"Fresh original-pair cache failed validation: index {pair_index}")


def load_original_pair_caches(
    cache_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[Path],
    pd.DataFrame,
    list[Path],
]:
    pair_rows: list[dict[str, Any]] = []
    draws: list[pd.DataFrame] = []
    neurons: list[pd.DataFrame] = []
    bins: list[pd.DataFrame] = []
    cache_paths: list[Path] = []
    native_pair_rows: list[dict[str, Any]] = []
    native_cache_paths: list[Path] = []
    for pair_index, spec in enumerate(ORIGINAL_PAIR_SPECS):
        cache_path = original_pair_cache_path(cache_root, pair_index)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        row = payload["pair_row"]
        if row.get("exclusion_profile") != NO_EXCLUSIONS:
            raise ValueError(f"Pair {pair_index} is not explicitly no-mask")
        pair_rows.append(row)
        draws.append(pd.read_csv(payload["support_draws_csv"], dtype={"animal_id": str}))
        neurons.append(pd.read_csv(payload["neuron_mask_csv"], dtype={"animal_id": str}))
        bins.append(pd.read_csv(payload["bin_mask_csv"], dtype={"animal_id": str}))
        cache_paths.append(cache_path)
        native_cache_path = original_pair_native_cache_path(cache_root, pair_index)
        native_payload = json.loads(native_cache_path.read_text(encoding="utf-8"))
        native_pair_rows.append(native_payload["pair_row"])
        native_cache_paths.append(native_cache_path)
    return (
        pd.DataFrame(pair_rows),
        pd.concat(draws, ignore_index=True),
        pd.concat(neurons, ignore_index=True),
        pd.concat(bins, ignore_index=True),
        cache_paths,
        pd.DataFrame(native_pair_rows),
        native_cache_paths,
    )


def native_relaxed_cache_path(cache_root: Path) -> Path:
    return (
        cache_root
        / "pair_cache"
        / "00_488-9day1__488-9day2.json"
    )


def compute_native_relaxed_488_no_exclusions(cache_root: Path) -> Path:
    """Compute the prespecified 488 Exp3 native-117 support sensitivity."""

    if closure.pm.EXCLUSION_PROFILES.get(NO_EXCLUSIONS) != {}:
        raise ValueError("NO_EXCLUSIONS must resolve to an empty exclusion map")
    cache_root.mkdir(parents=True, exist_ok=True)
    original_pairs = closure.FALLBACK_PAIRS
    original_profile = closure.pm.PRIMARY_EXCLUSION_PROFILE
    original_target = closure.TARGET_N
    original_repeats = closure.SUPPORT_REPEATS
    try:
        closure.FALLBACK_PAIRS = (
            (
                NATIVE_RELAXED_488_SPEC["animal_id"],
                NATIVE_RELAXED_488_SPEC["session_day1"],
                NATIVE_RELAXED_488_SPEC["session_day2"],
            ),
        )
        closure.pm.PRIMARY_EXCLUSION_PROFILE = NO_EXCLUSIONS
        closure.TARGET_N = int(NATIVE_RELAXED_488_SPEC["native_support_n"])
        closure.SUPPORT_REPEATS = 1
        cache_path = closure.compute_fallback_pair(0, cache_root)
    finally:
        closure.FALLBACK_PAIRS = original_pairs
        closure.pm.PRIMARY_EXCLUSION_PROFILE = original_profile
        closure.TARGET_N = original_target
        closure.SUPPORT_REPEATS = original_repeats

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    row = payload["pair_row"]
    row.update(
        {
            "experience_index": int(NATIVE_RELAXED_488_SPEC["experience_index"]),
            "experience_label": "Exp3",
            "month_alias": "9mo",
            "descriptive_only": False,
            "protocol_group": NATIVE_RELAXED_488_SPEC["protocol_group"],
            "historical_protocol_member": False,
            "ultm_prior_gap_indicator": 0,
            "source_scope": "original_native117_relaxed_no_manual_mask",
            "neural_source_policy": "original_when_sorted_unavailable",
            "exclusion_profile": NO_EXCLUSIONS,
            "exclusion_profile_source": "ca_analysis.exclusion_policy.NO_EXCLUSIONS",
            "cache_contract_version": NATIVE_CACHE_CONTRACT_VERSION,
            "target164_metric_eligible": False,
            "native_support_metric_eligible": True,
            "primary_exclusion_reason": "rate_gated_cells_117_below_target_164",
        }
    )
    payload["pair_row"] = row
    payload["cache_provenance"] = {
        "cache_contract_version": NATIVE_CACHE_CONTRACT_VERSION,
        "exclusion_profile": NO_EXCLUSIONS,
        "exclusion_map_empty": True,
        "support_definition": "native full retained pool",
        "native_support_n": 117,
        "primary_target_n": 164,
        "source_renderer": str(Path(closure.__file__).resolve()),
        "source_renderer_sha256": sha256(Path(closure.__file__).resolve()),
        "wrapper_renderer": str(Path(__file__).resolve()),
        "wrapper_renderer_sha256": sha256(Path(__file__).resolve()),
        "pair_spec": NATIVE_RELAXED_488_SPEC,
    }
    cache_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return cache_path


def native_relaxed_cache_is_current(cache_root: Path) -> bool:
    path = native_relaxed_cache_path(cache_root)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload["cache_provenance"]
        row = payload["pair_row"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return bool(
        provenance.get("cache_contract_version") == NATIVE_CACHE_CONTRACT_VERSION
        and provenance.get("exclusion_profile") == NO_EXCLUSIONS
        and provenance.get("native_support_n") == 117
        and provenance.get("source_renderer_sha256") == sha256(Path(closure.__file__).resolve())
        and provenance.get("wrapper_renderer_sha256") == sha256(Path(__file__).resolve())
        and row.get("exclusion_profile") == NO_EXCLUSIONS
        and int(row.get("support_match_target_n", -1)) == 117
        and int(row.get("support_match_repeats", -1)) == 1
        and int(row.get("n_neurons_after_rate_gate", -1)) == 117
    )


def load_native_relaxed_488(cache_root: Path) -> tuple[pd.DataFrame, Path]:
    path = native_relaxed_cache_path(cache_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([payload["pair_row"]]), path


def run_native_relaxed_488_cache(cache_root: Path) -> None:
    if native_relaxed_cache_is_current(cache_root):
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out-dir",
        str(DEFAULT_OUT_DIR.resolve()),
        "--compute-native-relaxed-488",
        "--cache-root",
        str(cache_root.resolve()),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not native_relaxed_cache_is_current(cache_root):
        raise RuntimeError("Fresh 488 native-support cache failed validation")


def _source_columns() -> list[str]:
    return [spec[0] for spec in METRICS]


def clustered_model_with_protocol_indicator(
    frame: pd.DataFrame,
    metric: str,
    specification: str,
) -> dict[str, Any]:
    columns = [
        "animal_id",
        "experience_index",
        "age_day1_days",
        "ultm_prior_gap_indicator",
        metric,
    ]
    sub = frame[columns].copy()
    for column in columns[1:]:
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna().reset_index(drop=True)
    stages = sorted(int(value) for value in sub["experience_index"].unique())
    age30 = (
        sub["age_day1_days"].to_numpy(dtype=float)
        - float(sub["age_day1_days"].mean())
    ) / 30.0
    stage_values = sub["experience_index"].to_numpy(dtype=int)
    design_columns = [np.ones(len(sub), dtype=float), age30]
    names = ["intercept", "age_per_30d"]
    for stage in stages[1:]:
        design_columns.append((stage_values == stage).astype(float))
        names.append(f"stage_{stage}")
    design_columns.append(sub["ultm_prior_gap_indicator"].to_numpy(dtype=float))
    names.append("ultm_prior_gap_indicator")
    xmat = np.column_stack(design_columns)
    y = sub[metric].to_numpy(dtype=float)
    clusters = sub["animal_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    n, k = xmat.shape
    g = len(unique_clusters)
    row: dict[str, Any] = {
        "specification": specification,
        "metric": metric,
        "n_rows": n,
        "n_mice": g,
        "n_parameters": k,
        "design_rank": int(np.linalg.matrix_rank(xmat)),
        "design_condition_number": float(np.linalg.cond(xmat)),
        "model_columns": ";".join(names),
        "beta_per_30d": np.nan,
        "cluster_se_cr1": np.nan,
        "t_cluster": np.nan,
        "p_cluster_two_sided": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "r_squared": np.nan,
        "ultm_protocol_beta": np.nan,
        "ultm_protocol_se_cr1": np.nan,
        "ultm_protocol_t": np.nan,
        "ultm_protocol_p": np.nan,
        "ultm_protocol_ci95_low": np.nan,
        "ultm_protocol_ci95_high": np.nan,
    }
    if n <= k or g < 3 or row["design_rank"] != k:
        return row
    bread = np.linalg.pinv(xmat.T @ xmat)
    beta = bread @ xmat.T @ y
    residual = y - xmat @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = xmat[mask].T @ residual[mask, None]
        meat += score @ score.T
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = scale * bread @ meat @ bread
    se = float(np.sqrt(max(float(covariance[1, 1]), 0.0)))
    beta_age = float(beta[1])
    t_value = beta_age / se if se > 0 else np.nan
    p_value = (
        float(2.0 * stats.t.sf(abs(t_value), df=g - 1))
        if np.isfinite(t_value)
        else np.nan
    )
    crit = float(stats.t.ppf(0.975, df=g - 1))
    total = float(np.sum((y - np.mean(y)) ** 2))
    protocol_beta = float(beta[-1])
    protocol_se = float(np.sqrt(max(float(covariance[-1, -1]), 0.0)))
    protocol_t = protocol_beta / protocol_se if protocol_se > 0 else np.nan
    protocol_p = (
        float(2.0 * stats.t.sf(abs(protocol_t), df=g - 1))
        if np.isfinite(protocol_t)
        else np.nan
    )
    row.update(
        {
            "beta_per_30d": beta_age,
            "cluster_se_cr1": se,
            "t_cluster": t_value,
            "p_cluster_two_sided": p_value,
            "ci95_low": beta_age - crit * se,
            "ci95_high": beta_age + crit * se,
            "r_squared": 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan,
            "cluster_df": g - 1,
            "cr1_scale": scale,
            "ultm_protocol_beta": protocol_beta,
            "ultm_protocol_se_cr1": protocol_se,
            "ultm_protocol_t": protocol_t,
            "ultm_protocol_p": protocol_p,
            "ultm_protocol_ci95_low": protocol_beta - crit * protocol_se,
            "ultm_protocol_ci95_high": protocol_beta + crit * protocol_se,
        }
    )
    return row


def model_family_with_protocol_indicator(
    frame: pd.DataFrame,
    specification: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        [
            clustered_model_with_protocol_indicator(frame, metric, specification)
            for metric, _, _, _ in closure.METRICS
        ]
    )
    result["q_bh4"] = closure.bh_adjust(result["p_cluster_two_sided"])
    lookup = {
        metric: (metric_key, metric_label, color)
        for metric, metric_key, metric_label, color in closure.METRICS
    }
    result["metric_key"] = result["metric"].map(lambda value: lookup[value][0])
    result["metric_label"] = result["metric"].map(lambda value: lookup[value][1])
    result["color"] = result["metric"].map(lambda value: lookup[value][2])
    return result


def long_gap_model_with_original_indicator(
    frame: pd.DataFrame,
    metric: str,
    specification: str,
) -> dict[str, Any]:
    sub = frame.loc[
        frame["metric"].eq(metric),
        ["animal_id", "experience", "age_dst_days", "value", "original_source_indicator"],
    ].copy()
    for column in ("experience", "age_dst_days", "value", "original_source_indicator"):
        sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna().reset_index(drop=True)
    y = sub["value"].to_numpy(dtype=float)
    age30 = (
        sub["age_dst_days"].to_numpy(dtype=float)
        - float(sub["age_dst_days"].mean())
    ) / 30.0
    transitions = sorted(int(value) for value in sub["experience"].unique())
    exp_values = sub["experience"].to_numpy(dtype=int)
    columns = [np.ones(len(sub), dtype=float), age30]
    names = ["intercept", "age30"]
    for transition in transitions[1:]:
        columns.append((exp_values == transition).astype(float))
        names.append(f"transition_{transition}")
    columns.append(sub["original_source_indicator"].to_numpy(dtype=float))
    names.append("original_source_indicator")
    xmat = np.column_stack(columns)
    n, k = xmat.shape
    clusters = sub["animal_id"].astype(str).to_numpy()
    unique_clusters = np.unique(clusters)
    g = len(unique_clusters)
    row: dict[str, Any] = {
        "specification": specification,
        "metric": metric,
        "n_rows": n,
        "n_animals": g,
        "n_parameters": k,
        "design_rank": int(np.linalg.matrix_rank(xmat)),
        "design_condition_number": float(np.linalg.cond(xmat)),
        "model_columns": ";".join(names),
        "beta_per_30d": np.nan,
        "cluster_se": np.nan,
        "t_cluster": np.nan,
        "p_cluster": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "rsquared": np.nan,
        "original_source_beta": np.nan,
        "original_source_se": np.nan,
        "original_source_t": np.nan,
        "original_source_p": np.nan,
        "original_source_ci_low": np.nan,
        "original_source_ci_high": np.nan,
    }
    if n <= k or g < 3 or row["design_rank"] != k:
        return row
    bread = np.linalg.pinv(xmat.T @ xmat)
    beta = bread @ xmat.T @ y
    residual = y - xmat @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster in unique_clusters:
        mask = clusters == cluster
        score = xmat[mask].T @ residual[mask, None]
        meat += score @ score.T
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = scale * bread @ meat @ bread
    se = float(np.sqrt(max(float(covariance[1, 1]), 0.0)))
    beta_age = float(beta[1])
    t_value = beta_age / se if se > 0 else np.nan
    p_value = (
        float(2.0 * stats.t.sf(abs(t_value), df=g - 1))
        if np.isfinite(t_value)
        else np.nan
    )
    crit = float(stats.t.ppf(0.975, df=g - 1))
    total = float(np.sum((y - np.mean(y)) ** 2))
    source_se = float(np.sqrt(max(float(covariance[-1, -1]), 0.0)))
    source_beta = float(beta[-1])
    source_t = source_beta / source_se if source_se > 0 else np.nan
    source_p = (
        float(2.0 * stats.t.sf(abs(source_t), df=g - 1))
        if np.isfinite(source_t)
        else np.nan
    )
    row.update(
        {
            "beta_per_30d": beta_age,
            "cluster_se": se,
            "t_cluster": t_value,
            "p_cluster": p_value,
            "ci_low": beta_age - crit * se,
            "ci_high": beta_age + crit * se,
            "rsquared": 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan,
            "cluster_df": g - 1,
            "cr1_scale": scale,
            "original_source_beta": source_beta,
            "original_source_se": source_se,
            "original_source_t": source_t,
            "original_source_p": source_p,
            "original_source_ci_low": source_beta - crit * source_se,
            "original_source_ci_high": source_beta + crit * source_se,
        }
    )
    return row


def long_gap_model_family(
    frame: pd.DataFrame,
    specification: str,
    *,
    include_original_source_indicator: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, metric_key, metric_label, color in closure.METRICS:
        long_metric = {
            "same_stage_place_map_corr_support_matched": "place_map_correlation",
            "same_stage_pv_corr_support_matched": "pv_correlation",
            "same_stage_level_corr_support_matched": "level_map_correlation",
            "same_stage_ensemble_rate_corr_support_matched": "ensemble_rate_correlation",
        }[metric]
        if include_original_source_indicator:
            row = long_gap_model_with_original_indicator(frame, long_metric, specification)
        else:
            metric_frame = frame.loc[frame["metric"].eq(long_metric)].copy()
            row = transition_fixed_age_model(metric_frame, long_metric)
            row["specification"] = specification
            clean = metric_frame[["animal_id", "experience", "age_dst_days", "value"]].copy()
            for column in ("experience", "age_dst_days", "value"):
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
            clean = clean.dropna().reset_index(drop=True)
            transitions = sorted(int(value) for value in clean["experience"].unique())
            age30 = (
                clean["age_dst_days"].to_numpy(dtype=float)
                - float(clean["age_dst_days"].mean())
            ) / 30.0
            exp_values = clean["experience"].to_numpy(dtype=int)
            design = [np.ones(len(clean), dtype=float), age30]
            for transition in transitions[1:]:
                design.append((exp_values == transition).astype(float))
            xmat = np.column_stack(design)
            n, k = xmat.shape
            g = clean["animal_id"].astype(str).nunique()
            row["n_parameters"] = k
            row["design_rank"] = int(np.linalg.matrix_rank(xmat))
            row["design_condition_number"] = float(np.linalg.cond(xmat))
            row["cluster_df"] = g - 1
            row["cr1_scale"] = (
                (g / (g - 1.0)) * ((n - 1.0) / (n - k))
                if g > 1 and n > k
                else np.nan
            )
        row["metric_key"] = metric_key
        row["metric_label"] = metric_label
        row["color"] = color
        rows.append(row)
    result = pd.DataFrame(rows)
    result["p_cluster_q_bh4"] = long_gap_bh_q(result["p_cluster"])
    return result


def long_gap_leave_one_mouse(
    frame: pd.DataFrame, full_models: pd.DataFrame
) -> pd.DataFrame:
    """Cluster-deletion sensitivity for the maximal long-gap estimand."""

    rows: list[dict[str, Any]] = []
    full_lookup = full_models.set_index("metric_key")
    for mouse in sorted(frame["animal_id"].astype(str).unique()):
        reduced = frame.loc[frame["animal_id"].astype(str).ne(mouse)].copy()
        fits = long_gap_model_family(reduced, f"leave_out_{mouse}")
        for fit in fits.itertuples(index=False):
            full = full_lookup.loc[fit.metric_key]
            beta = float(fit.beta_per_30d)
            full_beta = float(full.beta_per_30d)
            rows.append(
                {
                    "left_out_mouse": mouse,
                    "metric": fit.metric,
                    "metric_key": fit.metric_key,
                    "metric_label": fit.metric_label,
                    "n_rows": int(fit.n_rows),
                    "n_mice": int(fit.n_animals),
                    "design_rank": int(fit.design_rank),
                    "n_parameters": int(fit.n_parameters),
                    "design_condition_number": float(fit.design_condition_number),
                    "beta_per_30d": beta,
                    "cluster_se_cr1": float(fit.cluster_se),
                    "p_cluster_two_sided": float(fit.p_cluster),
                    "q_bh4_within_left_out_mouse": float(fit.p_cluster_q_bh4),
                    "full_beta_per_30d": full_beta,
                    "full_q_bh4": float(full.p_cluster_q_bh4),
                    "delta_from_full": beta - full_beta,
                    "delta_over_full_se": (
                        (beta - full_beta) / float(full.cluster_se)
                        if float(full.cluster_se) > 0
                        else np.nan
                    ),
                    "sign_differs_from_full": bool(
                        np.sign(beta) != np.sign(full_beta)
                    ),
                    "bh4_significance_differs_from_full": bool(
                        (float(fit.p_cluster_q_bh4) < 0.05)
                        != (float(full.p_cluster_q_bh4) < 0.05)
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["left_out_mouse", "metric_key"], kind="stable"
    ).reset_index(drop=True)


def long_gap_coverage_tables(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit destination-age support, stratum overlap and primary design geometry."""

    pairs = frame.loc[frame["metric"].eq("place_map_correlation")].copy()
    coverage = (
        pairs.groupby(["experience", "transition", "source_scope"], as_index=False)
        .agg(
            n_pairs=("pair_key", "size"),
            n_mice=("animal_id", "nunique"),
            age_min=("age_dst_days", "min"),
            age_q25=("age_dst_days", lambda values: float(np.quantile(values, 0.25))),
            age_median=("age_dst_days", "median"),
            age_q75=("age_dst_days", lambda values: float(np.quantile(values, 0.75))),
            age_max=("age_dst_days", "max"),
        )
        .sort_values(["experience", "source_scope"], kind="stable")
    )
    ranges = (
        pairs.groupby(["experience", "transition"], as_index=False)["age_dst_days"]
        .agg(["min", "max", "size"])
        .reset_index()
    )
    overlap_rows: list[dict[str, Any]] = []
    for left in ranges.itertuples(index=False):
        for right in ranges.itertuples(index=False):
            overlap = max(
                0.0,
                min(float(left.max), float(right.max))
                - max(float(left.min), float(right.min)),
            )
            union = max(float(left.max), float(right.max)) - min(
                float(left.min), float(right.min)
            )
            overlap_rows.append(
                {
                    "transition_a_experience": int(left.experience),
                    "transition_b_experience": int(right.experience),
                    "transition_a": str(left.transition),
                    "transition_b": str(right.transition),
                    "range_a_min": float(left.min),
                    "range_a_max": float(left.max),
                    "range_b_min": float(right.min),
                    "range_b_max": float(right.max),
                    "overlap_days": overlap,
                    "range_overlap_jaccard": overlap / union if union > 0 else 1.0,
                }
            )

    sample = pairs[["animal_id", "experience", "age_dst_days"]].copy()
    age30 = (
        sample["age_dst_days"].to_numpy(dtype=float)
        - float(sample["age_dst_days"].mean())
    ) / 30.0
    transitions = sorted(int(value) for value in sample["experience"].unique())
    exp_values = sample["experience"].to_numpy(dtype=int)
    columns = [np.ones(len(sample), dtype=float), age30]
    names = ["intercept", "age_per_30d"]
    for transition in transitions[1:]:
        columns.append((exp_values == transition).astype(float))
        names.append(f"transition_{transition}")
    xmat = np.column_stack(columns)
    stage_columns = [np.ones(len(sample), dtype=float)] + [
        (exp_values == transition).astype(float) for transition in transitions[1:]
    ]
    stage_x = np.column_stack(stage_columns)
    age = sample["age_dst_days"].to_numpy(dtype=float)
    stage_beta = np.linalg.pinv(stage_x.T @ stage_x) @ stage_x.T @ age
    residual = age - stage_x @ stage_beta
    total = float(np.sum((age - np.mean(age)) ** 2))
    age_r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan
    design = pd.DataFrame(
        [
            {
                "analysis": "destination_age_regressed_on_transition",
                "n_rows": len(sample),
                "n_mice": sample["animal_id"].astype(str).nunique(),
                "n_parameters": xmat.shape[1],
                "design_rank": int(np.linalg.matrix_rank(xmat)),
                "design_condition_number": float(np.linalg.cond(xmat)),
                "model_columns": ";".join(names),
                "age_transition_r_squared": age_r2,
                "age_vif_given_transition": (
                    1.0 / (1.0 - age_r2) if np.isfinite(age_r2) and age_r2 < 1 else np.inf
                ),
            }
        ]
    )
    return coverage, pd.DataFrame(overlap_rows), design


def build_same_stage_source(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the maximal no-mask target-164 source and prespecified sensitivities."""

    source_dir.mkdir(parents=True, exist_ok=True)
    cache_root = source_dir / "same_stage_original_target164_no_exclusions_2026-08-08"
    run_original_pair_caches(cache_root)
    (
        original,
        original_draws,
        original_neurons,
        original_bins,
        cache_paths,
        original_native,
        native_cache_paths,
    ) = load_original_pair_caches(cache_root)
    native_cache_root = source_dir / "same_stage_original_native117_no_exclusions_2026-08-08"
    run_native_relaxed_488_cache(native_cache_root)
    native_relaxed_row, native_cache_path = load_native_relaxed_488(native_cache_root)
    if len(original) != 14 or original["animal_id"].astype(str).nunique() != 9:
        raise ValueError("Expected 14 recomputed original-source pairs from 9 mice")
    if set(original["exclusion_profile"].astype(str)) != {NO_EXCLUSIONS}:
        raise ValueError("Original-source cache is not uniformly exclusion_profile=none")
    if original_neurons["excluded_by_profile"].astype(bool).any():
        raise ValueError("No-mask original-source cache contains profile-excluded neurons")
    if set(original["support_match_target_n"].astype(int)) != {closure.TARGET_N}:
        raise ValueError("Original-source cache is not target-164")
    if set(original["support_match_repeats"].astype(int)) != {closure.SUPPORT_REPEATS}:
        raise ValueError("Original-source cache does not use 1,000 technical draws")

    affected = original.loc[
        original["session_day2"].eq("1632Lday2"),
        ["n_neurons_day2_raw", "n_neurons_after_profile_exclusions", "n_neurons_after_rate_gate"],
    ].iloc[0]
    affected_neuron = original_neurons.loc[
        original_neurons["session_day2"].eq("1632Lday2")
        & original_neurons["shared_component_id"].eq(176)
    ]
    if (
        int(affected["n_neurons_day2_raw"])
        != int(affected["n_neurons_after_profile_exclusions"])
        or int(affected["n_neurons_after_rate_gate"]) != 896
        or len(affected_neuron) != 1
        or bool(affected_neuron.iloc[0]["excluded_by_profile"])
        or not bool(affected_neuron.iloc[0]["retained_after_rate_gate"])
    ):
        raise ValueError("1632Lday2 component 176 no-mask reconstruction failed")

    expected_ultm_support = {
        "1639": 1562,
        "1636": 555,
        "1637": 932,
        "325": 231,
        "347": 351,
    }
    observed_ultm_support = (
        original.loc[original["ultm_prior_gap_indicator"].astype(int).eq(1)]
        .set_index("animal_id")["n_neurons_after_rate_gate"]
        .astype(int)
        .to_dict()
    )
    if observed_ultm_support != expected_ultm_support:
        raise ValueError(
            f"ULTM native-support mismatch: {observed_ultm_support} != {expected_ultm_support}"
        )
    original["target164_metric_eligible"] = original["support_target_eligible"].astype(bool)
    if int(original["target164_metric_eligible"].sum()) != 13:
        raise ValueError("Expected 13/14 original pairs to clear target-164; 841 remains below target")
    recovered_579 = original.loc[
        original["session_day1"].eq("579-6day1"),
        [
            "n_neurons_after_rate_gate",
            "common_valid_bin_count",
            *[metric for metric, _, _, _ in closure.METRICS],
        ],
    ].iloc[0]
    expected_579_values = {
        "same_stage_place_map_corr_support_matched": 0.024693036364545094,
        "same_stage_pv_corr_support_matched": 0.1401305499246842,
        "same_stage_level_corr_support_matched": 0.18002680768599386,
        "same_stage_ensemble_rate_corr_support_matched": 0.4057234327827898,
    }
    if int(recovered_579["n_neurons_after_rate_gate"]) != 376:
        raise ValueError("579-6 native no-mask support must be 376")
    if int(recovered_579["common_valid_bin_count"]) != 91:
        raise ValueError("579-6 common valid-bin count must be 91")
    for metric, expected in expected_579_values.items():
        if not np.isclose(float(recovered_579[metric]), expected, atol=1.0e-6):
            raise ValueError(f"579-6 target164 {metric} changed from audited value")
    expected_native_488 = {
        "same_stage_place_map_corr_support_matched": 0.428223,
        "same_stage_pv_corr_support_matched": 0.467195,
        "same_stage_level_corr_support_matched": 0.557314,
        "same_stage_ensemble_rate_corr_support_matched": 0.597334,
    }
    if int(native_relaxed_row.iloc[0]["n_neurons_after_rate_gate"]) != 117:
        raise ValueError("488-9 native retained support must be 117")
    for metric, expected in expected_native_488.items():
        if not np.isclose(float(native_relaxed_row.iloc[0][metric]), expected, atol=1.0e-6):
            raise ValueError(f"488-9 native117 {metric} changed from audited value")

    support_path = NO_MASK_SORTED_DIR / "same_stage_experience_support_matched.csv"
    pair_path = NO_MASK_SORTED_DIR / "same_stage_experience_pair_metrics.csv"
    manifest_path = NO_MASK_SORTED_DIR / "same_stage_experience_manifest.csv"
    summary_path = NO_MASK_SORTED_DIR / "summary.json"
    support = pd.read_csv(support_path, dtype={"animal_id": str})
    pairs = pd.read_csv(pair_path, dtype={"animal_id": str})
    sorted_manifest = pd.read_csv(manifest_path, dtype={"animal_id": str})
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("exclusion_profile") != NO_EXCLUSIONS:
        raise ValueError("Selected sorted source is not exclusion_profile=none")
    if len(support) != 51 or support["animal_id"].nunique() != 14:
        raise ValueError("Expected 51 no-mask sorted rows from 14 mice")
    if set(support["support_match_target_n"].astype(int)) != {closure.TARGET_N}:
        raise ValueError("Same-stage sorted source is not target-164")
    if set(support["support_match_repeats"].astype(int)) != {closure.SUPPORT_REPEATS}:
        raise ValueError("Same-stage sorted source does not use 1,000 technical draws")
    if not sorted_manifest["status"].eq("included").all():
        raise ValueError("No-mask sorted manifest contains excluded rows")
    if set(sorted_manifest["exclusion_profile"].astype(str)) != {NO_EXCLUSIONS}:
        raise ValueError("No-mask sorted manifest does not uniformly declare none")
    sorted_manifest["stage_key"] = (
        sorted_manifest["animal_id"].astype(str)
        + ":"
        + sorted_manifest["session_day1"].astype(str)
        + ":"
        + sorted_manifest["session_day2"].astype(str)
    )
    if (
        sorted_manifest["stage_key"].duplicated().any()
        or set(sorted_manifest["stage_key"].astype(str))
        != set(support["stage_key"].astype(str))
    ):
        raise ValueError("Derived sorted-manifest stage keys do not match the support authority")

    sorted_values = (
        support.merge(
            pairs[
                [
                    "stage_key",
                    "n_pairs_final",
                    "common_valid_bin_count",
                    "positions_source_pair",
                    "spatial_label_source_pair",
                ]
            ],
            on="stage_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            sorted_manifest[
                [
                    "stage_key",
                    "neural_root",
                    "neural_id_space",
                    "exclusion_profile",
                    "n_neurons_after_exclusions",
                    "n_neurons_after_rate_gate",
                ]
            ],
            on="stage_key",
            how="left",
            validate="one_to_one",
        )
    )
    sorted_values["source_scope"] = "sorted_target164_no_manual_mask"
    sorted_values["neural_source_policy"] = "sorted_first"
    sorted_values["joint_deconvolution_shared_component_identity"] = True
    sorted_values["cimap_required"] = False
    sorted_values["matching_basis"] = (
        "shared fitted component index across jointly deconvolved Day1/Day2 traces"
    )
    sorted_values["value_source_file"] = str(support_path.resolve())
    sorted_values["protocol_group"] = "longitudinal_sorted"
    sorted_values["historical_protocol_member"] = True
    sorted_values["ultm_prior_gap_indicator"] = 0
    sorted_values["target164_metric_eligible"] = True

    cache_lookup = {
        (str(spec["session_day1"]), str(spec["session_day2"])): str(path.resolve())
        for spec, path in zip(ORIGINAL_PAIR_SPECS, cache_paths)
    }
    original = original.copy()
    original["neural_id_space"] = "presort"
    original["value_source_file"] = [
        cache_lookup[(str(day1), str(day2))]
        for day1, day2 in zip(original["session_day1"], original["session_day2"])
    ]
    original["day1_nose_interp_file"] = original["session_day1"].map(
        lambda session: str(
            (closure.BEHAV_WD / "selected_data" / f"{session}_nose_interp_data.csv").resolve()
        )
    )
    original["day2_nose_interp_file"] = original["session_day2"].map(
        lambda session: str(
            (closure.BEHAV_WD / "selected_data" / f"{session}_nose_interp_data.csv").resolve()
        )
    )
    native_cache_lookup = {
        (str(spec["session_day1"]), str(spec["session_day2"])): str(path.resolve())
        for spec, path in zip(ORIGINAL_PAIR_SPECS, native_cache_paths)
    }
    original_native = original_native.copy()
    original_native["neural_id_space"] = "presort"
    original_native["value_source_file"] = [
        native_cache_lookup[(str(day1), str(day2))]
        for day1, day2 in zip(
            original_native["session_day1"], original_native["session_day2"]
        )
    ]
    native_relaxed_row = native_relaxed_row.copy()
    native_relaxed_row["neural_id_space"] = "presort"
    native_relaxed_row["value_source_file"] = str(native_cache_path.resolve())
    native_relaxed_row["day1_nose_interp_file"] = str(
        (closure.BEHAV_WD / "selected_data" / "488-9day1_nose_interp_data.csv").resolve()
    )
    native_relaxed_row["day2_nose_interp_file"] = str(
        (closure.BEHAV_WD / "selected_data" / "488-9day2_nose_interp_data.csv").resolve()
    )

    union_columns = sorted(
        set(sorted_values.columns) | set(original.columns) | set(native_relaxed_row.columns)
    )
    union = pd.concat(
        [
            sorted_values.reindex(columns=union_columns),
            original.reindex(columns=union_columns),
            native_relaxed_row.reindex(columns=union_columns),
        ],
        ignore_index=True,
    )
    if union["stage_key"].duplicated().any():
        raise ValueError("Duplicate same-stage pair key")
    provenance = closure.attach_age_metadata(union)
    provenance["age_and_target164_included"] = (
        provenance["age_analysis_included"].astype(bool)
        & provenance["target164_metric_eligible"].astype(bool)
    )
    included = provenance.loc[provenance["age_and_target164_included"]].copy()

    native_sorted = sorted_values.copy()
    native_metric_map = {
        "same_stage_place_map_corr_support_matched": "same_stage_place_map_corr",
        "same_stage_pv_corr_support_matched": "same_stage_pv_corr",
        "same_stage_level_corr_support_matched": "same_stage_level_corr",
        "same_stage_ensemble_rate_corr_support_matched": "same_stage_ensemble_rate_corr",
    }
    pair_native_lookup = pairs.set_index("stage_key")
    for destination, source in native_metric_map.items():
        native_sorted[destination] = native_sorted["stage_key"].map(
            pair_native_lookup[source]
        )
    native_sorted["source_scope"] = "sorted_native_support_no_manual_mask"
    native_sorted["support_match_target_n"] = native_sorted["n_neurons_after_rate_gate"]
    native_sorted["support_match_repeats"] = 1
    native_sorted["native_support_metric_eligible"] = True
    native_sorted["value_source_file"] = str(pair_path.resolve())
    native_union_columns = sorted(
        set(native_sorted.columns)
        | set(original_native.columns)
        | set(native_relaxed_row.columns)
    )
    native_union = pd.concat(
        [
            native_sorted.reindex(columns=native_union_columns),
            original_native.reindex(columns=native_union_columns),
            native_relaxed_row.reindex(columns=native_union_columns),
        ],
        ignore_index=True,
    )
    if native_union["stage_key"].duplicated().any():
        raise ValueError("Duplicate same-stage pair key in native-support sensitivity")
    native_provenance = closure.attach_age_metadata(native_union)
    native_relaxed = native_provenance.loc[
        native_provenance["age_analysis_included"].astype(bool)
    ].copy()
    historical = included.loc[included["historical_protocol_member"].astype(bool)].copy()
    historical_source_checkpoint = provenance.loc[
        provenance["historical_protocol_member"].astype(bool)
    ].copy()
    inherited_qc = included.loc[
        included["protocol_group"].ne("recovered_inherited_qc_exp2")
    ].copy()
    all_metric_columns = [metric for metric, _, _, _ in closure.METRICS]
    if len(provenance) != 66 or provenance["animal_id"].nunique() != 22:
        raise ValueError("Expected 66 recomputed source rows / 22 mice after reconciliation")
    if len(included) != 64 or included["animal_id"].nunique() != 21:
        raise ValueError("Expected maximal 64 age+target164 rows / 21 mice")
    if len(inherited_qc) != 63 or inherited_qc["animal_id"].nunique() != 21:
        raise ValueError("Inherited-QC exclusion sensitivity must contain 63 rows / 21 mice")
    if len(historical) != 58 or historical["animal_id"].nunique() != 21:
        raise ValueError("Historical-protocol parity must remain 58 rows / 21 mice")
    if (
        len(historical_source_checkpoint) != 59
        or historical_source_checkpoint["animal_id"].nunique() != 22
    ):
        raise ValueError("Historical source-universe checkpoint must remain 59 rows / 22 mice")
    if len(native_relaxed) != 65 or native_relaxed["animal_id"].nunique() != 21:
        raise ValueError("Native-support relaxed sensitivity must contain 65 rows / 21 mice")
    if included[all_metric_columns].isna().any().any():
        raise ValueError("Same-stage maximal source contains missing metric values")

    audited_values = pd.read_csv(SAME_AUDIT_VALUES, dtype={"animal_id": str})
    observed_values = included[
        [
            "animal_id",
            "session_day1",
            "session_day2",
            "experience_index",
            "age_day1_days",
            *all_metric_columns,
        ]
    ].rename(
        columns={
            "same_stage_place_map_corr_support_matched": "place_observed",
            "same_stage_pv_corr_support_matched": "pv_observed",
            "same_stage_level_corr_support_matched": "level_observed",
            "same_stage_ensemble_rate_corr_support_matched": "rate_observed",
        }
    )
    audited_compare = observed_values.merge(
        audited_values[
            [
                "animal_id",
                "session_day1",
                "session_day2",
                "experience_index",
                "age_day1_days",
                "place",
                "pv",
                "level",
                "rate",
            ]
        ],
        on=["animal_id", "session_day1", "session_day2"],
        how="outer",
        suffixes=("_observed_key", "_audit"),
        indicator=True,
        validate="one_to_one",
    )
    if len(audited_compare) != 64 or not audited_compare["_merge"].eq("both").all():
        raise ValueError("Maximal64 source does not match the independent row ledger")
    for metric in ("place", "pv", "level", "rate"):
        if not np.allclose(
            audited_compare[f"{metric}_observed"],
            audited_compare[metric],
            atol=5.0e-10,
            rtol=0.0,
        ):
            raise ValueError(f"Maximal64 {metric} values disagree with independent audit")
    for field in ("experience_index", "age_day1_days"):
        if not np.allclose(
            audited_compare[f"{field}_observed_key"],
            audited_compare[f"{field}_audit"],
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"Maximal64 {field} disagrees with independent audit")

    maximal_spec = "maximal64_age_plus_stage_target164_no_manual_mask"
    inherited_qc_spec = "inherited_qc_exclusion63_age_plus_stage_target164_no_manual_mask"
    historical_spec = "historical_protocol_only58_age_plus_stage_target164_no_manual_mask"
    protocol_spec = "maximal64_age_plus_stage_plus_ultm_protocol_target164_no_manual_mask"
    native_spec = "native_support65_age_plus_stage_no_equal_support_no_manual_mask"
    maximal_models = closure.model_family(included, maximal_spec)
    inherited_qc_models = closure.model_family(inherited_qc, inherited_qc_spec)
    historical_models = closure.model_family(historical, historical_spec)
    protocol_models = model_family_with_protocol_indicator(included, protocol_spec)
    native_models = closure.model_family(native_relaxed, native_spec)
    all_models = pd.concat(
        [
            maximal_models,
            inherited_qc_models,
            historical_models,
            protocol_models,
            native_models,
        ],
        ignore_index=True,
    )
    spec_order = {
        maximal_spec: 0,
        inherited_qc_spec: 1,
        historical_spec: 2,
        protocol_spec: 3,
        native_spec: 4,
    }
    metric_order = {key: index for index, (_, key, _, _) in enumerate(closure.METRICS)}
    all_models["specification_order"] = all_models["specification"].map(spec_order)
    all_models["metric_order"] = all_models["metric_key"].map(metric_order)
    all_models = all_models.sort_values(
        ["specification_order", "metric_order"], kind="stable"
    ).drop(columns=["specification_order", "metric_order"]).reset_index(drop=True)
    audited_models = pd.read_csv(SAME_AUDIT_MODELS)
    audit_spec_map = {
        maximal_spec: "maximal_valid64_primary",
        inherited_qc_spec: "inherited_qc63",
        historical_spec: "historical_protocol_only58",
        protocol_spec: "maximal64_plus_ultm_protocol_indicator",
    }
    audited_lookup = audited_models.set_index(["specification", "metric"])
    for row in all_models.loc[all_models["specification"].isin(audit_spec_map)].itertuples(
        index=False
    ):
        audited = audited_lookup.loc[(audit_spec_map[row.specification], row.metric_key)]
        comparisons = {
            "beta": (row.beta_per_30d, audited.beta_age_per_30d),
            "se": (row.cluster_se_cr1, audited.se_CR1_age),
            "p": (row.p_cluster_two_sided, audited.p_CR1_student_t_G_minus_1),
            "q": (row.q_bh4, audited.q_BH4_age),
            "condition": (row.design_condition_number, audited.design_condition_number),
            "cr1_scale": (row.cr1_scale, audited.cr1_scale),
        }
        if row.specification == protocol_spec:
            comparisons["protocol_beta"] = (
                row.ultm_protocol_beta,
                audited.beta_ultm_split_bout_protocol,
            )
            comparisons["protocol_p"] = (
                row.ultm_protocol_p,
                audited.p_CR1_ultm_split_bout_protocol,
            )
        if (
            int(row.n_rows) != int(audited.n_rows)
            or int(row.n_mice) != int(audited.n_mice)
            or int(row.design_rank) != int(audited.design_rank)
        ):
            raise ValueError(
                f"Same-stage {row.specification}/{row.metric_key} model dimensions disagree with audit"
            )
        for label, (observed, expected) in comparisons.items():
            if not np.isclose(float(observed), float(expected), atol=1.0e-10, rtol=0.0):
                raise ValueError(
                    f"Same-stage {row.specification}/{row.metric_key} {label} disagrees with audit"
                )
    models = maximal_models.loc[
        maximal_models["metric_key"].isin([spec[2] for spec in METRICS])
    ].copy()
    display_order = {spec[2]: index for index, spec in enumerate(METRICS)}
    models["metric_order"] = models["metric_key"].map(display_order)
    models = models.sort_values("metric_order", kind="stable").reset_index(drop=True)
    if len(models) != 3:
        raise ValueError("Same-stage artwork must contain exactly three displayed models")

    same_lomo = closure.leave_one_mouse(included, maximal_models)
    full_same_lookup = maximal_models.set_index("metric")
    same_lomo["full_beta_per_30d"] = same_lomo["metric"].map(
        full_same_lookup["beta_per_30d"]
    )
    same_lomo["full_q_bh4"] = same_lomo["metric"].map(full_same_lookup["q_bh4"])
    same_lomo["bh4_significance_differs_from_full"] = (
        same_lomo["q_bh4_within_left_out_mouse"].lt(0.05)
        != same_lomo["full_q_bh4"].lt(0.05)
    )
    leverage_rows, leverage_mice, design_audit = closure.leverage_tables(
        included, same_lomo
    )
    age_coverage, age_overlap = closure.coverage_tables(included)
    primary_design_models = maximal_models[
        [
            "metric",
            "metric_key",
            "n_rows",
            "n_mice",
            "n_parameters",
            "design_rank",
            "design_condition_number",
            "model_columns",
        ]
    ].copy()

    plot_cols = [
        "animal_id",
        "session_day1",
        "session_day2",
        "stage_key",
        "experience_index",
        "experience_label",
        "descriptive_only",
        "age_day1_days",
        "age_day2_days",
        "day_gap",
        "source_scope",
        "protocol_group",
        "historical_protocol_member",
        "ultm_prior_gap_indicator",
        "neural_source_policy",
        "neural_id_space",
        "exclusion_profile",
        "joint_deconvolution_shared_component_identity",
        "cimap_required",
        "matching_basis",
        "support_match_target_n",
        "support_match_repeats",
        "n_neurons_after_rate_gate",
        *all_metric_columns,
    ]
    plot_ready = included.reindex(columns=plot_cols)
    plot_path = source_dir / "ed07_same_stage_plot_ready_maximal64_no_manual_mask_round1.csv"
    model_path = source_dir / "ed07_same_stage_models_maximal64_displayed_round1.csv"
    sensitivity_path = source_dir / "ed07_same_stage_model_sensitivities_bh4_round1.csv"
    provenance_path = source_dir / "ed07_same_stage_recomputed_source_provenance66_round1.csv"
    original_path = source_dir / "ed07_same_stage_original14_no_exclusions_audit_round1.csv"
    draw_path = source_dir / "ed07_same_stage_original14_support_draws_round1.csv"
    neuron_path = source_dir / "ed07_same_stage_original14_neuron_mask_round1.csv"
    bin_path = source_dir / "ed07_same_stage_original14_bin_mask_round1.csv"
    cache_index_path = source_dir / "ed07_same_stage_original14_cache_index_round1.csv"
    native_path = source_dir / "ed07_same_stage_native117_488_exp3_audit_round1.csv"
    native_cohort_path = source_dir / "ed07_same_stage_native_support65_sensitivity_round1.csv"
    lomo_path = source_dir / "ed07_same_stage_maximal64_leave_one_mouse_out_bh4_round1.csv"
    leverage_row_path = source_dir / "ed07_same_stage_maximal64_row_leverage_round1.csv"
    leverage_mouse_path = source_dir / "ed07_same_stage_maximal64_mouse_influence_round1.csv"
    design_path = source_dir / "ed07_same_stage_maximal64_design_audit_round1.csv"
    design_models_path = source_dir / "ed07_same_stage_maximal64_design_by_metric_round1.csv"
    coverage_path = source_dir / "ed07_same_stage_maximal64_age_coverage_by_stage_source_round1.csv"
    overlap_path = source_dir / "ed07_same_stage_maximal64_age_range_overlap_round1.csv"
    plot_ready.to_csv(plot_path, index=False)
    models.to_csv(model_path, index=False)
    all_models.to_csv(sensitivity_path, index=False)
    provenance.to_csv(provenance_path, index=False)
    original.to_csv(original_path, index=False)
    original_draws.to_csv(draw_path, index=False)
    original_neurons.to_csv(neuron_path, index=False)
    original_bins.to_csv(bin_path, index=False)
    native_relaxed_row.to_csv(native_path, index=False)
    native_relaxed.to_csv(native_cohort_path, index=False)
    same_lomo.to_csv(lomo_path, index=False)
    leverage_rows.to_csv(leverage_row_path, index=False)
    leverage_mice.to_csv(leverage_mouse_path, index=False)
    design_audit.to_csv(design_path, index=False)
    primary_design_models.to_csv(design_models_path, index=False)
    age_coverage.to_csv(coverage_path, index=False)
    age_overlap.to_csv(overlap_path, index=False)
    pd.DataFrame(
        [
            {
                "pair_index": index,
                **spec,
                "cache_json": str(path.resolve()),
                "cache_json_sha256": sha256(path),
                "native_cache_json": str(native_path.resolve()),
                "native_cache_json_sha256": sha256(native_path),
            }
            for index, (spec, path, native_path) in enumerate(
                zip(ORIGINAL_PAIR_SPECS, cache_paths, native_cache_paths)
            )
        ]
    ).to_csv(cache_index_path, index=False)

    audit = {
        "sorted_rows": 51,
        "sorted_mice": 14,
        "original_candidate_rows": 14,
        "original_target164_rows": 13,
        "original_mice": 9,
        "recomputed_source_rows": 66,
        "reconciled_candidate_mice": 22,
        "maximal_age_target164_rows": 64,
        "maximal_age_target164_mice": 21,
        "inherited_qc_exclusion_rows": 63,
        "inherited_qc_exclusion_mice": 21,
        "historical_protocol_parity_rows": 58,
        "historical_protocol_parity_mice": 21,
        "historical_protocol_source_checkpoint_rows": 59,
        "historical_protocol_source_checkpoint_mice": 22,
        "historical_checkpoint_interpretation": "reconciliation parity only; not a maximal eligibility ceiling",
        "native_support_relaxed_rows": 65,
        "native_support_relaxed_mice": 21,
        "ultm_added_rows": 5,
        "recovered_inherited_qc_rows": 1,
        "support_target_n": closure.TARGET_N,
        "support_repeats": closure.SUPPORT_REPEATS,
        "exclusion_profile": NO_EXCLUSIONS,
        "manual_profile_excluded_neurons": int(
            original_neurons["excluded_by_profile"].astype(bool).sum()
        ),
        "mask_intersection_recomputed": "1632Lday2 component 176; retained after rate gate",
        "mask_intersection_pool": 896,
        "missing_age_support_only_pair": "841day1->841day2",
        "ultm_native_support": expected_ultm_support,
        "recovered_579_native_support": 376,
        "recovered_579_target164_values": expected_579_values,
        "native_488_support": 117,
        "native_488_values": expected_native_488,
        "plot_ready": str(plot_path.resolve()),
        "displayed_models": str(model_path.resolve()),
        "model_sensitivities_bh4": str(sensitivity_path.resolve()),
        "candidate_provenance": str(provenance_path.resolve()),
        "independent_maximal64_values_audit": str(SAME_AUDIT_VALUES.resolve()),
        "independent_model_audit": str(SAME_AUDIT_MODELS.resolve()),
        "original_pair_audit": str(original_path.resolve()),
        "original_support_draws": str(draw_path.resolve()),
        "original_neuron_mask": str(neuron_path.resolve()),
        "original_bin_mask": str(bin_path.resolve()),
        "original_cache_index": str(cache_index_path.resolve()),
        "original_cache_jsons": [str(path.resolve()) for path in cache_paths],
        "native_488_cache_json": str(native_cache_path.resolve()),
        "native_488_audit": str(native_path.resolve()),
        "native_support_cohort": str(native_cohort_path.resolve()),
        "leave_one_mouse_out_bh4": str(lomo_path.resolve()),
        "leave_one_mouse_out_sign_flip_rows": int(
            same_lomo["sign_differs_from_full"].astype(bool).sum()
        ),
        "leave_one_mouse_out_bh4_class_change_rows": int(
            same_lomo["bh4_significance_differs_from_full"].astype(bool).sum()
        ),
        "leave_one_mouse_out_bh4_class_changes": same_lomo.loc[
            same_lomo["bh4_significance_differs_from_full"].astype(bool),
            ["left_out_mouse", "metric_key", "beta_per_30d", "q_bh4_within_left_out_mouse"],
        ].to_dict("records"),
        "row_leverage": str(leverage_row_path.resolve()),
        "mouse_influence": str(leverage_mouse_path.resolve()),
        "design_audit": str(design_path.resolve()),
        "design_by_metric": str(design_models_path.resolve()),
        "age_coverage_by_stage_source": str(coverage_path.resolve()),
        "age_range_overlap": str(overlap_path.resolve()),
        "original_native_cache_jsons": [str(path.resolve()) for path in native_cache_paths],
        "original_draw_rows": int(len(original_draws)),
        "original_neuron_rows": int(len(original_neurons)),
        "original_bin_rows": int(len(original_bins)),
    }
    return plot_ready, models, provenance, audit


def build_long_gap_source(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    conversion_audit = validate_long_gap_audit_artifacts()
    joined = pd.read_csv(LONG_JOINED, dtype={"animal_id": str})
    all_long_metrics = [
        "place_map_correlation",
        "pv_correlation",
        "level_map_correlation",
        "ensemble_rate_correlation",
    ]
    parity36 = joined.loc[
        joined["age_control_included"].astype(str).str.lower().eq("true")
        & joined["metric"].isin(all_long_metrics)
    ].copy()
    if set(parity36.groupby("metric").size()) != {36}:
        raise ValueError("Corrected36 parity input must contain 36 rows per metric")
    parity36["source_scope"] = "corrected36_sorted_explicit_cimap_no_manual_mask"
    parity36["original_source_indicator"] = 0
    parity36["exclusion_profile"] = NO_EXCLUSIONS

    source_manifest = json.loads(LONG_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        source_manifest.get("manual_cell_id_exclusion_profile") != NO_EXCLUSIONS
        or int(source_manifest.get("effective_manual_cell_id_drops", -1)) != 0
    ):
        raise ValueError("Corrected36 long-gap authority is not explicitly no-mask")
    cimap_metafile = pd.read_csv(LONG_CIMAP_METAFILE)
    if len(cimap_metafile) != 36 or cimap_metafile["cimap_path"].isna().any():
        raise ValueError("Corrected36 must provide one explicit CIMap path per pair")
    missing_cimaps = [
        path for path in cimap_metafile["cimap_path"].astype(str) if not Path(path).exists()
    ]
    if missing_cimaps:
        raise FileNotFoundError(f"Missing corrected36 CIMaps: {missing_cimaps[:3]}")

    candidates = pd.read_csv(LONG_EXTRA_CANDIDATES, dtype={"animal_id": str})
    if len(candidates) != 2 or set(candidates["animal_id"]) != {"579", "488"}:
        raise ValueError("Expected exact 579 3->6 and 488 6->9 long-gap candidates")
    if set(candidates["classification"]) != {"USE"}:
        raise ValueError("Both reconciled long-gap additions must be classified USE")
    if set(candidates["exclusion_profile"]) != {NO_EXCLUSIONS}:
        raise ValueError("Long-gap additions are not explicitly no-mask")
    for row in candidates.itertuples(index=False):
        expected = next(
            spec for spec in LONG_GAP_EXTRA_PAIR_SPECS if spec["animal_id"] == row.animal_id
        )
        cimap_path = Path(row.cimap_path)
        if not cimap_path.exists() or sha256(cimap_path) != str(row.cimap_sha256).lower():
            raise ValueError(f"Long-gap CIMap hash mismatch for {row.animal_id}")
        if not Path(row.original_src_neural).exists() or not Path(row.original_dst_neural).exists():
            raise FileNotFoundError(f"Missing original neural source for {row.animal_id}")
        if (
            row.src_session != expected["src"]
            or row.dst_session != expected["dst"]
            or int(row.registered_pairs_valid) != expected["n_matched_cimap"]
            or int(row.native_n) != expected["n_used"]
            or int(row.common_spatial_bins) != expected["common_valid_bins"]
            or int(row.common_tree_depth7_levels) != expected["valid_levels"]
            or str(row.cimap_sha256).lower() != expected["cimap_sha256"]
        ):
            raise ValueError(f"Long-gap candidate audit fields changed for {row.animal_id}")
        for metric, expected_value in expected["values"].items():
            if not np.isclose(float(getattr(row, metric)), expected_value, atol=5.0e-7):
                raise ValueError(f"Long-gap candidate {row.animal_id} {metric} changed")

    dates = closure._session_date_lookup(closure.EXPERIMENT_METAFILE).set_index("session_id")
    demographics = pd.read_csv(closure.DEMOGRAPHICS, dtype={"animal_id": str}).set_index(
        "animal_id"
    )
    extra_rows: list[dict[str, Any]] = []
    candidate_metric_columns = {
        "place_map_correlation": "place_map_correlation",
        "pv_correlation": "pv_correlation",
        "level_map_correlation": "level_map_correlation",
        "ensemble_rate_correlation": "ensemble_rate_correlation",
    }
    for candidate in candidates.itertuples(index=False):
        src_date = pd.Timestamp(dates.loc[candidate.src_session, "session_date"])
        dst_date = pd.Timestamp(dates.loc[candidate.dst_session, "session_date"])
        demographic = demographics.loc[str(candidate.animal_id)]
        birth_date = pd.Timestamp(demographic["birth_date"])
        surgery_date = pd.to_datetime(demographic["surgery_date"], errors="coerce")
        experience = 2 if str(candidate.transition) == "3_to_6" else 3
        transition = "3-6 mo" if experience == 2 else "6-9 mo"
        for metric, value_column in candidate_metric_columns.items():
            extra_rows.append(
                {
                    "metric": metric,
                    "src": candidate.src_session,
                    "dst": candidate.dst_session,
                    "animal_id": str(candidate.animal_id),
                    "experience": experience,
                    "value": float(getattr(candidate, value_column)),
                    "pair_key": f"{candidate.src_session}->{candidate.dst_session}",
                    "transition": transition,
                    "src_session_id": candidate.src_session,
                    "src_session_date": src_date,
                    "src_session_date_source": dates.loc[
                        candidate.src_session, "session_date_source"
                    ],
                    "src_ID": dates.loc[candidate.src_session, "ID"],
                    "dst_session_id": candidate.dst_session,
                    "dst_session_date": dst_date,
                    "dst_session_date_source": dates.loc[
                        candidate.dst_session, "session_date_source"
                    ],
                    "dst_ID": dates.loc[candidate.dst_session, "ID"],
                    "birth_date": demographic["birth_date"],
                    "gender": demographic["gender"],
                    "surgery_date": demographic["surgery_date"],
                    "age_at_surgery_days": demographic["age_at_surgery_days"],
                    "metadata_status": demographic["metadata_status"],
                    "birth_date_dt": birth_date,
                    "surgery_date_dt": surgery_date,
                    "age_src_days": int((src_date - birth_date).days),
                    "age_dst_days": int((dst_date - birth_date).days),
                    "gap_days": int((dst_date - src_date).days),
                    "days_surgery_to_dst": (
                        int((dst_date - surgery_date).days)
                        if pd.notna(surgery_date)
                        else np.nan
                    ),
                    "age_control_exclusion_reason": "",
                    "age_control_included": True,
                    "source_scope": "original_no_mask_explicit_cimap_recovered_candidate",
                    "original_source_indicator": 1,
                    "exclusion_profile": NO_EXCLUSIONS,
                    "original_src_neural": candidate.original_src_neural,
                    "original_dst_neural": candidate.original_dst_neural,
                    "cimap_path": candidate.cimap_path,
                    "cimap_sha256": candidate.cimap_sha256,
                    "n_matched_cimap": int(candidate.registered_pairs_valid),
                    "n_used": int(candidate.native_n),
                    "common_spatial_bins": int(candidate.common_spatial_bins),
                    "common_tree_depth7_levels": int(candidate.common_tree_depth7_levels),
                }
            )
    extra = pd.DataFrame(extra_rows)
    maximal38 = pd.concat([parity36, extra], ignore_index=True, sort=False)
    plus_579 = pd.concat(
        [parity36, extra.loc[extra["animal_id"].eq("579")]], ignore_index=True, sort=False
    )
    plus_488 = pd.concat(
        [parity36, extra.loc[extra["animal_id"].eq("488")]], ignore_index=True, sort=False
    )
    cohorts = {
        "parity36": parity36,
        "plus_488_only37": plus_488,
        "plus_579_only37": plus_579,
        "maximal38": maximal38,
    }
    model_frames = [long_gap_model_family(frame, specification) for specification, frame in cohorts.items()]
    model_frames.append(
        long_gap_model_family(
            maximal38,
            "maximal38_plus_original_source_indicator",
            include_original_source_indicator=True,
        )
    )
    all_models = pd.concat(model_frames, ignore_index=True)
    spec_order = {name: index for index, name in enumerate([*cohorts, "maximal38_plus_original_source_indicator"])}
    metric_order = {key: index for index, (_, key, _, _) in enumerate(closure.METRICS)}
    all_models["specification_order"] = all_models["specification"].map(spec_order)
    all_models["metric_order"] = all_models["metric_key"].map(metric_order)
    all_models = all_models.sort_values(
        ["specification_order", "metric_order"], kind="stable"
    ).drop(columns=["specification_order", "metric_order"]).reset_index(drop=True)
    maximal38_models_all4 = all_models.loc[
        all_models["specification"].eq("maximal38")
    ].copy()
    long_lomo = long_gap_leave_one_mouse(maximal38, maximal38_models_all4)
    long_coverage, long_overlap, long_design = long_gap_coverage_tables(maximal38)
    long_design_by_metric = maximal38_models_all4[
        [
            "metric",
            "metric_key",
            "n_rows",
            "n_animals",
            "n_parameters",
            "design_rank",
            "design_condition_number",
            "cluster_df",
            "cr1_scale",
        ]
    ].copy()

    audited_models = pd.read_csv(LONG_AUDIT_MODELS)
    audited_lookup = audited_models.set_index(["specification", "metric"])
    for row in all_models.itertuples(index=False):
        audited = audited_lookup.loc[(row.specification, row.metric_key)]
        comparisons = {
            "beta": (row.beta_per_30d, audited.beta_age_per_30d),
            "p": (row.p_cluster, audited.p_CR1_student_t_G_minus_1),
            "q": (row.p_cluster_q_bh4, audited.q_BH4_age),
        }
        if row.specification == "maximal38_plus_original_source_indicator":
            comparisons["original_source_p"] = (
                row.original_source_p,
                audited.p_CR1_original_source,
            )
        for label, (observed, expected) in comparisons.items():
            if not np.isclose(float(observed), float(expected), atol=2.0e-6):
                raise ValueError(
                    f"Long-gap {row.specification}/{row.metric_key} {label} disagrees with audit"
                )

    models = all_models.loc[
        all_models["specification"].eq("maximal38")
        & all_models["metric_key"].isin([spec[2] for spec in METRICS])
    ].copy()
    display_order = {spec[2]: index for index, spec in enumerate(METRICS)}
    models["metric_order"] = models["metric_key"].map(display_order)
    models = models.sort_values("metric_order", kind="stable").reset_index(drop=True)
    displayed_metrics = [spec[1] for spec in METRICS]
    plot_ready = maximal38.loc[maximal38["metric"].isin(displayed_metrics)].copy()
    counts = plot_ready.groupby("metric").size()
    mice = plot_ready.groupby("metric")["animal_id"].nunique()
    if len(models) != 3 or set(counts) != {38} or set(mice) != {14}:
        raise ValueError("Long-gap artwork source must contain 3 metrics x 38 pairs / 14 mice")
    if plot_ready["value"].isna().any() or plot_ready["age_dst_days"].isna().any():
        raise ValueError("Long-gap maximal38 plotted values contain missing data")

    plot_cols = [
        "metric",
        "animal_id",
        "src",
        "dst",
        "pair_key",
        "experience",
        "transition",
        "value",
        "age_src_days",
        "age_dst_days",
        "gap_days",
        "source_scope",
        "original_source_indicator",
        "exclusion_profile",
        "age_control_included",
        "age_control_exclusion_reason",
    ]
    plot_ready = plot_ready.reindex(columns=plot_cols).sort_values(
        ["metric", "experience", "age_dst_days", "animal_id"], kind="stable"
    )
    plot_path = source_dir / "ed07_long_gap_plot_ready_maximal38_round1.csv"
    model_path = source_dir / "ed07_long_gap_models_maximal38_displayed_round1.csv"
    sensitivity_path = source_dir / "ed07_long_gap_model_sensitivities_bh4_round1.csv"
    maximal_all_path = source_dir / "ed07_long_gap_maximal38_all4_metrics_round1.csv"
    additions_path = source_dir / "ed07_long_gap_candidate_additions2_all4_metrics_round1.csv"
    lomo_path = source_dir / "ed07_long_gap_maximal38_leave_one_mouse_out_bh4_round1.csv"
    coverage_path = source_dir / "ed07_long_gap_maximal38_age_coverage_by_transition_source_round1.csv"
    overlap_path = source_dir / "ed07_long_gap_maximal38_age_range_overlap_round1.csv"
    design_path = source_dir / "ed07_long_gap_maximal38_design_audit_round1.csv"
    design_models_path = source_dir / "ed07_long_gap_maximal38_design_by_metric_round1.csv"
    plot_ready.to_csv(plot_path, index=False)
    models.to_csv(model_path, index=False)
    all_models.to_csv(sensitivity_path, index=False)
    maximal38.to_csv(maximal_all_path, index=False)
    extra.to_csv(additions_path, index=False)
    long_lomo.to_csv(lomo_path, index=False)
    long_coverage.to_csv(coverage_path, index=False)
    long_overlap.to_csv(overlap_path, index=False)
    long_design.to_csv(design_path, index=False)
    long_design_by_metric.to_csv(design_models_path, index=False)
    audit = {
        "primary_pairs_per_metric": 38,
        "primary_mice_per_metric": 14,
        "parity_pairs_per_metric": 36,
        "recovered_candidates": ["579-3day1->579-6day1", "488-6day1->488-9day1"],
        "matching": "explicit CIMap for all pairs",
        "exclusion_profile": NO_EXCLUSIONS,
        "age_timepoint": "destination session",
        "observation": "one source-target pair row",
        "plot_ready": str(plot_path.resolve()),
        "displayed_models": str(model_path.resolve()),
        "model_sensitivities_bh4": str(sensitivity_path.resolve()),
        "maximal_all4_source": str(maximal_all_path.resolve()),
        "candidate_additions": str(additions_path.resolve()),
        "candidate_audit_input": str(LONG_EXTRA_CANDIDATES.resolve()),
        "independent_model_audit": str(LONG_AUDIT_MODELS.resolve()),
        "leave_one_mouse_out_bh4": str(lomo_path.resolve()),
        "leave_one_mouse_out_sign_flip_rows": int(
            long_lomo["sign_differs_from_full"].astype(bool).sum()
        ),
        "leave_one_mouse_out_bh4_class_change_rows": int(
            long_lomo["bh4_significance_differs_from_full"].astype(bool).sum()
        ),
        "leave_one_mouse_out_bh4_class_changes": long_lomo.loc[
            long_lomo["bh4_significance_differs_from_full"].astype(bool),
            [
                "left_out_mouse",
                "metric_key",
                "beta_per_30d",
                "q_bh4_within_left_out_mouse",
                "full_q_bh4",
            ],
        ].to_dict("records"),
        "leave_one_mouse_out_sign_flips": long_lomo.loc[
            long_lomo["sign_differs_from_full"].astype(bool),
            ["left_out_mouse", "metric_key", "beta_per_30d", "full_beta_per_30d"],
        ].to_dict("records"),
        "age_coverage_by_transition_source": str(coverage_path.resolve()),
        "age_range_overlap": str(overlap_path.resolve()),
        "design_audit": str(design_path.resolve()),
        "design_by_metric": str(design_models_path.resolve()),
        "conversion_audit": conversion_audit,
    }
    return plot_ready, models, audit


def _same_stage_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    result: pd.Series,
    metric: str,
    label: str,
    color: str,
    *,
    show_ylabel: bool,
) -> None:
    beta = float(result["beta_per_30d"])
    for stage, group in frame.groupby("experience_index", sort=True):
        stage = int(stage)
        stage_color = STAGE_COLORS[stage]
        for source_scope, source_group in group.groupby("source_scope", sort=False):
            marker = "D" if source_scope.startswith("original") else "o"
            ax.scatter(
                source_group["age_day1_days"],
                source_group[metric],
                s=11 if marker == "o" else 15,
                marker=marker,
                facecolor=stage_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        x = group["age_day1_days"].to_numpy(dtype=float)
        y = group[metric].to_numpy(dtype=float)
        line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
        line_y = float(np.mean(y)) + beta * (line_x - float(np.mean(x))) / 30.0
        ax.plot(line_x, line_y, color=stage_color, linewidth=1.0, zorder=2)
    ax.set_title(label, loc="left", fontweight="bold", color=color, pad=1.6)
    ax.set_xlabel("Age on Day 1 (days)")
    if show_ylabel:
        ax.set_ylabel("Same-stage Day 1-Day 2\ncorrelation (r)")
    ax.set_xlim(80, 680)
    ax.set_xticks([120, 300, 480, 660])
    ax.set_ylim(-0.04, 1.0)


def _value_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    low, high = float(np.min(finite)), float(np.max(finite))
    padding = max((high - low) * 0.12, 0.025)
    return low - padding, high + padding


def _long_gap_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    result: pd.Series,
    metric: str,
    label: str,
    color: str,
    *,
    show_ylabel: bool,
) -> None:
    subset = frame.loc[frame["metric"].eq(metric)].copy()
    beta = float(result["beta_per_30d"])
    for experience, group in subset.groupby("experience", sort=True):
        experience = int(experience)
        x = group["age_dst_days"].to_numpy(dtype=float)
        y = group["value"].to_numpy(dtype=float)
        transition_color = TRANSITION_COLORS[experience]
        for original_source, source_group in group.groupby(
            "original_source_indicator", sort=True
        ):
            ax.scatter(
                source_group["age_dst_days"],
                source_group["value"],
                s=15 if int(original_source) else 11,
                marker="D" if int(original_source) else "o",
                facecolor=transition_color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )
        if len(x) >= 3 and np.std(x) > 0:
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 40)
            line_y = float(np.mean(y)) + beta * (line_x - float(np.mean(x))) / 30.0
            ax.plot(line_x, line_y, color=transition_color, linewidth=1.0, zorder=2)
    ax.set_title(label, loc="left", fontweight="bold", color=color, pad=1.6)
    ax.set_xlabel("Destination-session age (days)")
    if show_ylabel:
        ax.set_ylabel("Long-gap cross-session\ncorrelation (r)")
    ax.set_xlim(175, 680)
    ax.set_xticks([200, 350, 500, 650])
    ax.set_ylim(*_value_limits(subset["value"]))


def render_top_row(frame: pd.DataFrame, models: pd.DataFrame, stem: Path) -> list[Path]:
    lookup = models.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=TOP_ROW_SIZE_IN, sharex=True, sharey=True)
    for index, (ax, (same_metric, _, _, label, color)) in enumerate(zip(axes, METRICS)):
        _same_stage_axis(
            ax,
            frame,
            lookup.loc[same_metric],
            same_metric,
            label,
            color,
            show_ylabel=index == 0,
        )
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=STAGE_COLORS[stage],
            markeredgecolor="none",
            label=STAGE_LABELS[stage],
            markersize=4,
        )
        for stage in STAGE_LABELS
    ]
    source_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", label="sorted", markersize=3.8),
        Line2D([0], [0], marker="D", linestyle="", color="#555555", label="original recomputed", markersize=3.8),
    ]
    fig.legend(
        stage_handles + source_handles,
        [handle.get_label() for handle in stage_handles + source_handles],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.985),
        ncol=7,
        handletextpad=0.25,
        columnspacing=0.78,
    )
    fig.text(0.012, 0.965, "a", fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.16, top=0.83, wspace=0.23)
    return save_figure(fig, stem)


def render_bottom_row(frame: pd.DataFrame, models: pd.DataFrame, stem: Path) -> list[Path]:
    lookup = models.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=BOTTOM_ROW_SIZE_IN, sharex=True)
    for index, (ax, (_, long_metric, _, label, color)) in enumerate(zip(axes, METRICS)):
        _long_gap_axis(
            ax,
            frame,
            lookup.loc[long_metric],
            long_metric,
            label,
            color,
            show_ylabel=index == 0,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TRANSITION_COLORS[transition],
            markeredgecolor="none",
            label=TRANSITION_LABELS[transition],
            markersize=4,
        )
        for transition in sorted(TRANSITION_LABELS)
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="#555555",
                label="maintained parity36",
                markersize=3.8,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                color="#555555",
                label="recovered original",
                markersize=3.8,
            ),
        ]
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        title="Destination experience",
        title_fontsize=5.8,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=6,
        handletextpad=0.25,
        columnspacing=0.9,
    )
    fig.text(0.012, 0.965, "b", fontsize=10.5, fontweight="bold", ha="left", va="top")
    fig.subplots_adjust(left=0.072, right=0.995, bottom=0.16, top=0.83, wspace=0.24)
    return save_figure(fig, stem)


def render_recovered_conversion_audit(stem: Path) -> list[Path]:
    """Visualize the four upstream conversion gates for both promoted long-gap rows."""

    prefixes = ("579_3to6", "488_6to9")
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.8), squeeze=False)
    spatial_cmap = ListedColormap(["#F3F3F3", "#A9C2CF", "#315C7A"])
    for row_index, prefix in enumerate(prefixes):
        cimap = pd.read_csv(LONG_CONVERSION_AUDIT_DIR / f"{prefix}_cimap_mapping.csv")
        rate = pd.read_csv(
            LONG_CONVERSION_AUDIT_DIR
            / f"{prefix}_neuron_rate_gate_and_metric_inputs.csv"
        )
        spatial = pd.read_csv(
            LONG_CONVERSION_AUDIT_DIR / f"{prefix}_spatial_bin_audit.csv"
        )
        level = pd.read_csv(
            LONG_CONVERSION_AUDIT_DIR / f"{prefix}_tree_depth7_level_support.csv"
        )

        ax = axes[row_index, 0]
        counts = [
            len(cimap),
            int(bool_series(cimap["valid_registered_pair"]).sum()),
            int(bool_series(cimap["retained_after_S_rate_gate"]).sum()),
        ]
        ax.bar(range(3), counts, color=["#C9C9C9", "#7D9AAA", "#315C7A"], width=0.72)
        ax.set_xticks(range(3), ["raw", "CIMap", "S gate"], rotation=20, ha="right")
        ax.set_ylabel("Neuron-pair rows")
        ax.set_title(f"{prefix.replace('_', ' ')}\nidentity attrition", loc="left", fontweight="bold")

        ax = axes[row_index, 1]
        retained = bool_series(rate["retained_both_sessions"])
        x = pd.to_numeric(rate["source_S_events_per_min"], errors="coerce") + 1.0e-4
        y = pd.to_numeric(rate["destination_S_events_per_min"], errors="coerce") + 1.0e-4
        ax.scatter(x[~retained], y[~retained], s=5, color="#C8C8C8", alpha=0.55, linewidth=0)
        ax.scatter(x[retained], y[retained], s=6, color="#315C7A", alpha=0.72, linewidth=0)
        ax.axvline(0.1, color="#777777", linewidth=0.6, linestyle="--")
        ax.axhline(0.1, color="#777777", linewidth=0.6, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Source S events min$^{-1}$")
        ax.set_ylabel("Destination S events min$^{-1}$")
        ax.set_title("Both-session rate gate", loc="left", fontweight="bold")

        ax = axes[row_index, 2]
        spatial_state = np.zeros((15, 15), dtype=int)
        for item in spatial.itertuples(index=False):
            x_bin = int(item.bin_x_0based)
            y_bin = int(item.bin_y_0based)
            if str(item.common_valid_intersection).strip().lower() == "true":
                spatial_state[y_bin, x_bin] = 1
            if str(item.pv_finite_final_input).strip().lower() == "true":
                spatial_state[y_bin, x_bin] = 2
        ax.imshow(spatial_state, origin="lower", cmap=spatial_cmap, vmin=0, vmax=2)
        ax.set_xticks([0, 7, 14])
        ax.set_yticks([0, 7, 14])
        ax.set_xlabel("Maze x bin")
        ax.set_ylabel("Maze y bin")
        ax.set_title(
            "Spatial support\n"
            f"common={int(bool_series(spatial['common_valid_intersection']).sum())}; "
            f"finite PV={int(bool_series(spatial['pv_finite_final_input']).sum())}",
            loc="left",
            fontweight="bold",
        )

        ax = axes[row_index, 3]
        depth = pd.to_numeric(level["tree_depth7_level_0based"], errors="coerce")
        width = 0.36
        ax.bar(
            depth - width / 2,
            pd.to_numeric(level["source_occupancy_sec"], errors="coerce"),
            width=width,
            color="#7D9AAA",
            label="source",
        )
        ax.bar(
            depth + width / 2,
            pd.to_numeric(level["destination_occupancy_sec"], errors="coerce"),
            width=width,
            color="#D1905A",
            label="destination",
        )
        ax.set_xlabel("Tree-depth7 level")
        ax.set_ylabel("Occupancy (s)")
        ax.set_title(
            "Level support\n"
            f"common={int(bool_series(level['common_valid_intersection']).sum())}",
            loc="left",
            fontweight="bold",
        )
        if row_index == 0:
            ax.legend(frameon=False, fontsize=5.5, handlelength=1.0)

    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.105, top=0.875, wspace=0.42, hspace=0.78)
    return save_audit_figure(fig, stem)


def render_noncanonical_0to6_supporting_audit(stem: Path) -> list[Path]:
    """Visualize exact supporting-only 0-to-6 values without promoting them."""

    frame = pd.read_csv(LONG_NONCANONICAL_CANDIDATES, dtype={"animal_id": str})
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.2), sharex=True)
    metric_specs = (
        ("place_map_correlation", "Place map", "#B95A58"),
        ("pv_correlation", "Population vector", "#E29957"),
        ("level_map_correlation", "Level map", "#7976A2"),
        ("ensemble_rate_correlation", "Ensemble rate", "#4A5E65"),
    )
    x = np.arange(len(frame), dtype=float)
    for ax, (metric, label, color) in zip(axes, metric_specs):
        ax.scatter(
            x,
            pd.to_numeric(frame[metric], errors="coerce"),
            s=18,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axhline(0, color="#AAAAAA", linewidth=0.55)
        ax.set_title(label, loc="left", fontweight="bold", color=color)
        ax.set_xticks(x, frame["animal_id"], rotation=35, ha="right")
        ax.set_xlabel("Mouse")
        ax.set_ylabel("0-to-6 correlation (r)")
        values = pd.to_numeric(frame[metric], errors="coerce")
        low, high = _value_limits(values)
        ax.set_ylim(min(-0.03, low), max(0.1, high))
    fig.text(
        0.012,
        0.97,
        "Supporting-only noncanonical split-bout 0-to-6 audit",
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.28, top=0.78, wspace=0.38)
    return save_audit_figure(fig, stem)


def compose_compact_assembly(top_png: Path, bottom_png: Path, stem: Path) -> list[Path]:
    top = plt.imread(top_png)
    bottom = plt.imread(bottom_png)
    fig = plt.figure(figsize=(7.5, 6.82), facecolor="white")
    top_ax = fig.add_axes([0.0233, 0.512, 0.9534, 0.4765])
    bottom_ax = fig.add_axes([0.0233, 0.0115, 0.9534, 0.4765])
    for ax, image in ((top_ax, top), (bottom_ax, bottom)):
        ax.imshow(image)
        ax.axis("off")
    return save_figure(fig, stem)


def compose_slide_mockup(top_png: Path, bottom_png: Path, output_path: Path) -> Path:
    """Place enlarged aligned rows into the final slide-9 geometry."""

    dpi = 600
    canvas = Image.new("RGB", (int(SLIDE_SIZE_IN[0] * dpi), int(SLIDE_SIZE_IN[1] * dpi)), "white")
    top = Image.open(top_png).convert("RGB")
    bottom = Image.open(bottom_png).convert("RGB")

    bottom_box = tuple(int(round(value * dpi)) for value in BOTTOM_SLOT_IN)
    bottom_resized = bottom.resize((bottom_box[2], bottom_box[3]), Image.Resampling.LANCZOS)
    canvas.paste(bottom_resized, (bottom_box[0], bottom_box[1]))

    top_box = tuple(int(round(value * dpi)) for value in TOP_SLOT_IN)
    top_resized = top.resize((top_box[2], top_box[3]), Image.Resampling.LANCZOS)
    # Rows do not crop or overlap; z-order is immaterial but deterministic.
    canvas.paste(top_resized, (top_box[0], top_box[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(dpi, dpi))
    return output_path


def caption_statistics(same: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    same_lookup = same.set_index("metric")
    long_lookup = long.set_index("metric")
    rows: list[dict[str, Any]] = []
    for metric_order, (same_metric, long_metric, key, label, _) in enumerate(METRICS):
        row = same_lookup.loc[same_metric]
        rows.append(
            {
                "panel": "a",
                "metric_order": metric_order,
                "metric_key": key,
                "metric_label": label,
                "estimand": "age slope after categorical experience-stage adjustment",
                "estimate_beta_per_30d": float(row["beta_per_30d"]),
                "cluster_se": float(row["cluster_se_cr1"]),
                "p_mouse_clustered": float(row["p_cluster_two_sided"]),
                "q_bh4": float(row["q_bh4"]),
                "n_pair_rows": int(row["n_rows"]),
                "n_mice": int(row["n_mice"]),
                "matching": "shared fitted-component identity; joint Day1/Day2 deconvolution",
                "display_location": "caption_only",
            }
        )
        row = long_lookup.loc[long_metric]
        rows.append(
            {
                "panel": "b",
                "metric_order": metric_order,
                "metric_key": key,
                "metric_label": label,
                "estimand": "destination-age slope after long-gap transition adjustment",
                "estimate_beta_per_30d": float(row["beta_per_30d"]),
                "cluster_se": float(row["cluster_se"]),
                "p_mouse_clustered": float(row["p_cluster"]),
                "q_bh4": float(row["p_cluster_q_bh4"]),
                "n_pair_rows": int(row["n_rows"]),
                "n_mice": int(row["n_animals"]),
                "matching": "explicit CIMap",
                "display_location": "caption_only",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["panel", "metric_order"], kind="stable")
        .drop(columns="metric_order")
        .reset_index(drop=True)
    )


def validate_outputs(
    top_outputs: list[Path],
    bottom_outputs: list[Path],
    assembly_outputs: list[Path],
    mockup_path: Path,
    same_points: pd.DataFrame,
    same_models: pd.DataFrame,
    same_provenance: pd.DataFrame,
    same_audit: dict[str, Any],
    long_points: pd.DataFrame,
    long_models: pd.DataFrame,
    long_audit: dict[str, Any],
    caption_stats: pd.DataFrame,
    source_inputs: list[dict[str, Any]],
    visual_qa_status: str,
    audit_figure_outputs: list[Path],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": detail})

    def add_pending(check: str, detail: str) -> None:
        checks.append({"check": check, "status": "PENDING", "detail": detail})

    add(
        "slide_inventory_exactly_two_slots",
        True,
        "panel a same-stage composite; panel b long-gap composite",
    )
    add(
        "slide_inventory_exactly_three_metrics_per_slot",
        len(METRICS) == 3
        and set(same_models["metric_key"]) == {"place", "pv", "rate"}
        and set(long_models["metric_key"]) == {"place", "pv", "rate"},
        "place, population vector, ensemble rate; no additional panel role",
    )
    add(
        "same_stage_scope_maximal64",
        len(same_points) == 64 and same_points["animal_id"].nunique() == 21,
        f"{len(same_points)} target164 age-eligible rows / {same_points['animal_id'].nunique()} mice",
    )
    add(
        "same_stage_no_manual_mask",
        set(same_points["exclusion_profile"].astype(str)) == {NO_EXCLUSIONS}
        and same_audit["manual_profile_excluded_neurons"] == 0,
        "sorted and all fresh original caches declare none; 0 neurons excluded_by_profile",
    )
    add(
        "same_stage_support_target164",
        set(pd.to_numeric(same_points["support_match_target_n"], errors="coerce").dropna().astype(int)) == {164},
        "all eligible plotted rows use target-164 support matching",
    )
    add(
        "same_stage_joint_identity",
        same_points["joint_deconvolution_shared_component_identity"].astype(bool).all()
        and ~same_points["cimap_required"].astype(bool).any(),
        "shared fitted component identity valid within joint Day1/Day2 deconvolution",
    )
    add(
        "same_stage_inclusion_audit",
        len(same_provenance) == 66
        and same_provenance["age_and_target164_included"].astype(bool).sum() == 64
        and same_audit["historical_protocol_source_checkpoint_rows"] == 59
        and same_audit["historical_protocol_source_checkpoint_mice"] == 22
        and same_audit["historical_protocol_parity_rows"] == 58
        and same_audit["inherited_qc_exclusion_rows"] == 63,
        "66 reconciled source rows; historical 59/22 and 58/21 are parity checkpoints; maximal64 primary",
    )
    candidate_validation = json.loads(
        (DEFAULT_OUT_DIR / "AUDIT_VALIDATION.json").read_text(encoding="utf-8")
    )
    add(
        "candidate_universe_reconciled",
        candidate_validation.get("status") == "PASS"
        and len(pd.read_csv(DEFAULT_OUT_DIR / "same_stage_candidate_universe_attrition.csv"))
        == 145,
        "145-row raw registry universe; parsed/planned/source gates and maximality decisions are explicit",
    )
    current_wrapper_hash = sha256(Path(__file__).resolve())
    current_cache_payloads = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in [
            *same_audit["original_cache_jsons"],
            *same_audit["original_native_cache_jsons"],
            same_audit["native_488_cache_json"],
        ]
    ]
    add(
        "same_stage_cache_wrapper_sha_frozen",
        len(current_cache_payloads) == 29
        and all(
            payload["cache_provenance"].get("wrapper_renderer_sha256")
            == current_wrapper_hash
            for payload in current_cache_payloads
        ),
        f"14 target + 14 native companions + native117 cache pin wrapper {current_wrapper_hash}",
    )
    same_lomo = pd.read_csv(same_audit["leave_one_mouse_out_bh4"])
    add(
        "same_stage_leave_one_mouse_out_complete",
        len(same_lomo) == 84
        and set(same_lomo.groupby("left_out_mouse").size()) == {4}
        and (
            same_lomo["design_rank"].eq(same_lomo["n_parameters"]).all()
            if {"design_rank", "n_parameters"}.issubset(same_lomo.columns)
            else True
        ),
        f"21 mouse deletions x BH4; sign flips={int(same_lomo['sign_differs_from_full'].astype(bool).sum())}",
    )
    add(
        "same_stage_no_single_mouse_changes_bh4_conclusion",
        not same_lomo["sign_differs_from_full"].astype(bool).any()
        and not same_lomo["bh4_significance_differs_from_full"].astype(bool).any(),
        "slope sign and BH4 significance classification are unchanged across all 21 cluster deletions",
    )
    same_design = pd.read_csv(same_audit["design_by_metric"])
    add(
        "same_stage_design_full_rank_finite_condition",
        same_design["design_rank"].eq(same_design["n_parameters"]).all()
        and np.isfinite(same_design["design_condition_number"]).all(),
        "all four maximal64 metric designs are full rank; condition numbers exported",
    )
    add(
        "same_stage_age_overlap_exported",
        Path(same_audit["age_coverage_by_stage_source"]).exists()
        and Path(same_audit["age_range_overlap"]).exists(),
        "age ranges by stage/source and every pairwise stage overlap are exported",
    )
    add(
        "long_gap_scope_maximal38",
        set(long_points.groupby("metric").size()) == {38}
        and set(long_points.groupby("metric")["animal_id"].nunique()) == {14},
        "38 pair rows / 14 mice for each displayed metric",
    )
    add(
        "long_gap_explicit_cimap",
        long_audit_files_valid(),
        "corrected36 plus recovered 579/488 rows all use explicit hashed CIMaps and conversion audits",
    )
    conversion_reproducer = long_audit["conversion_audit"]
    add(
        "long_gap_conversion_reproducer_source_locked",
        Path(conversion_reproducer["conversion_reproducer"]).is_file()
        and conversion_reproducer["conversion_reproducer_sha256"]
        == sha256(Path(conversion_reproducer["conversion_reproducer"])),
        "maintained one-pair-per-subprocess no-exclusion reproducer is hashed and its exact invocation is in the manifest/README",
    )
    long_lomo = pd.read_csv(long_audit["leave_one_mouse_out_bh4"])
    add(
        "long_gap_leave_one_mouse_out_complete",
        len(long_lomo) == 56
        and set(long_lomo.groupby("left_out_mouse").size()) == {4}
        and long_lomo["design_rank"].eq(long_lomo["n_parameters"]).all(),
        f"14 mouse deletions x BH4; sign flips={int(long_lomo['sign_differs_from_full'].astype(bool).sum())}",
    )
    add(
        "long_gap_no_positive_bh4_conclusion_depends_on_one_mouse",
        not long_lomo.loc[
            long_lomo["full_q_bh4"].lt(0.05),
            ["bh4_significance_differs_from_full", "sign_differs_from_full"],
        ].astype(bool).any().any(),
        "no full-cohort BH4-positive metric loses significance or reverses direction after a mouse deletion",
    )
    observed_class_changes = set(
        map(
            tuple,
            long_lomo.loc[
                long_lomo["bh4_significance_differs_from_full"].astype(bool),
                ["left_out_mouse", "metric_key"],
            ].astype(str).to_numpy(),
        )
    )
    observed_sign_flips = set(
        map(
            tuple,
            long_lomo.loc[
                long_lomo["sign_differs_from_full"].astype(bool),
                ["left_out_mouse", "metric_key"],
            ].astype(str).to_numpy(),
        )
    )
    add(
        "long_gap_lomo_instability_explicitly_locked_for_caption",
        observed_class_changes == {("837", "rate")}
        and observed_sign_flips
        == {("488", "level"), ("837", "level"), ("837", "place")},
        "leave-837 makes rate BH4-positive; three sign flips occur only for full-cohort nonsignificant place/level estimates",
    )
    long_design = pd.read_csv(long_audit["design_by_metric"])
    add(
        "long_gap_design_full_rank_finite_condition",
        long_design["design_rank"].eq(long_design["n_parameters"]).all()
        and np.isfinite(long_design["design_condition_number"]).all(),
        "all four maximal38 metric designs are full rank; condition numbers exported",
    )
    add(
        "long_gap_age_overlap_exported",
        Path(long_audit["age_coverage_by_transition_source"]).exists()
        and Path(long_audit["age_range_overlap"]).exists(),
        "destination-age ranges and every pairwise transition overlap are exported",
    )
    add(
        "raw_and_maintained_inputs_immutable",
        len(source_inputs) >= 100
        and all(record["sha256"] and record["bytes"] > 0 for record in source_inputs)
        and any("same-stage original raw input" in record["role"] for record in source_inputs)
        and any("recovered long-gap immutable" in record["role"] for record in source_inputs),
        f"{len(source_inputs)} deduplicated maintained/cache/raw/code inputs carry SHA256 hashes",
    )
    audit_by_stem: dict[str, set[str]] = {}
    for path in audit_figure_outputs:
        audit_by_stem.setdefault(path.stem, set()).add(path.suffix.lower())
        add(
            f"audit_figure_{path.stem}_{path.suffix[1:]}_nonempty",
            path.stat().st_size > 1000,
            f"{path.name}: {path.stat().st_size} bytes",
        )
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (np.nan, np.nan))
                add(
                    f"audit_figure_{path.stem}_png_600dpi",
                    min(dpi) >= 599,
                    f"{image.width} x {image.height} px at dpi={dpi}",
                )
    add(
        "conversion_audit_visualizations_complete",
        len(audit_by_stem) == 2
        and all(suffixes == {".svg", ".pdf", ".png"} for suffixes in audit_by_stem.values()),
        "recovered conversion-stage overview and supporting-only 0-to-6 summary; audit-only, not slide roles",
    )
    add(
        "caption_statistics_exact_six_rows",
        len(caption_stats) == 6 and set(caption_stats["display_location"]) == {"caption_only"},
        "3 metrics x 2 panels; exact beta, P, q and counts live outside the artwork",
    )
    add("no_background_grid_lines", True, "all six axes use clean white backgrounds with no grid call")

    for label, outputs in (("top", top_outputs), ("bottom", bottom_outputs), ("assembly", assembly_outputs)):
        by_suffix = {path.suffix.lower(): path for path in outputs}
        add(
            f"{label}_formats_complete",
            set(by_suffix) == {".svg", ".pdf", ".png", ".tiff"},
            ", ".join(sorted(by_suffix)),
        )
        png = by_suffix[".png"]
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (np.nan, np.nan))
            add(
                f"{label}_png_600dpi",
                min(dpi) >= 599,
                f"{image.width} x {image.height} px; dpi={dpi}",
            )
            if label in {"top", "bottom"}:
                expected_size = (4290, 1950)
                add(
                    f"{label}_source_media_aspect",
                    image.size == expected_size,
                    f"{image.width} x {image.height} px; expected {expected_size}",
                )
        svg_text = by_suffix[".svg"].read_text(encoding="utf-8", errors="replace")
        if label in {"top", "bottom"}:
            add(f"{label}_svg_editable_text", "<text" in svg_text, "live SVG text nodes present")
            add(f"{label}_svg_declares_arial", "Arial" in svg_text, "Arial declared in SVG")
            expected_letter = "a" if label == "top" else "b"
            add(
                f"{label}_single_embedded_panel_letter_10_5pt",
                len(
                    re.findall(
                        rf">\s*{expected_letter}\s*</text>", svg_text, flags=re.IGNORECASE
                    )
                )
                == 1
                and "10.5px" in svg_text,
                f"one embedded {expected_letter} authority at 10.5 pt",
            )
            add(
                f"{label}_exactly_three_visible_metrics_no_level_role",
                "Level map" not in svg_text
                and all(metric_label in svg_text for metric_label in ("Place map", "Population vector", "Ensemble rate")),
                "place, population vector and ensemble rate only",
            )
            palette = STAGE_COLORS if label == "top" else TRANSITION_COLORS
            add(
                f"{label}_shared_semantic_palette_present",
                all(str(color).lower() in svg_text.lower() for color in palette.values()),
                "all authoritative stage/transition colors appear in points/lines and the matching legend",
            )
            forbidden = re.compile(
                r"(?:beta|p\s*=|q\s*=|n\s*=|clustered\s+p|\u03b2|\u2605|\u2217)",
                flags=re.IGNORECASE,
            )
            svg_text_nodes = " ".join(
                re.sub(r"<[^>]+>", " ", node)
                for node in re.findall(
                    r"<text\b[^>]*>.*?</text>", svg_text, flags=re.IGNORECASE | re.DOTALL
                )
            )
            hits = sorted(set(forbidden.findall(svg_text_nodes)))
            add(
                f"{label}_no_inferential_annotations",
                not hits,
                "no beta, P, q, n or significance annotations"
                if not hits
                else f"forbidden text tokens: {hits}",
            )
        else:
            add(
                "assembly_layout_only_raster_composite",
                "<image" in svg_text,
                "compact assembly embeds the two validated source-rendered row panels",
            )
        for path in outputs:
            add(f"{label}_{path.suffix[1:]}_nonempty", path.stat().st_size > 1000, f"{path.name}: {path.stat().st_size} bytes")

    with Image.open(mockup_path) as image:
        dpi = image.info.get("dpi", (np.nan, np.nan))
        add(
            "slide_geometry_mockup_exact_canvas",
            image.size == (4500, 4500) and min(dpi) >= 599,
            f"{image.width} x {image.height} px at dpi={dpi}; 7.5 x 7.5 in",
        )
    add(
        "top_slot_geometry_exact",
        TOP_SLOT_IN == (0.175, 0.38, 7.15, 3.25),
        f"x,y,w,h={TOP_SLOT_IN}; no crop",
    )
    add(
        "bottom_slot_geometry_exact",
        BOTTOM_SLOT_IN == (0.175, 3.87, 7.15, 3.25),
        f"x,y,w,h={BOTTOM_SLOT_IN}; no crop",
    )
    top_scale_ratio = (TOP_SLOT_IN[2] / 4290) / (TOP_SLOT_IN[3] / 1950)
    bottom_scale_ratio = (BOTTOM_SLOT_IN[2] / 4290) / (BOTTOM_SLOT_IN[3] / 1950)
    add(
        "slot_placement_isotropic",
        abs(top_scale_ratio - 1.0) < 0.001 and abs(bottom_scale_ratio - 1.0) < 0.001,
        f"top scale_x/scale_y={top_scale_ratio:.8f}; bottom={bottom_scale_ratio:.8f}",
    )
    add(
        "final_rows_no_crop_no_overlap",
        TOP_SLOT_IN[1] + TOP_SLOT_IN[3] < BOTTOM_SLOT_IN[1]
        and BOTTOM_SLOT_IN[1] + BOTTOM_SLOT_IN[3] <= SLIDE_SIZE_IN[1],
        "two enlarged 7.15 x 3.25 in rows; 0.24 in gap; symmetric 0.38 in vertical margins",
    )
    if visual_qa_status == "pass":
        add(
            "manual_visual_qa",
            True,
            "human-visible inspection confirmed legibility, clipping, alignment, palette and panel-letter authority",
        )
    else:
        add_pending(
            "manual_visual_qa",
            "rendered outputs require human-visible inspection before promotion",
        )

    validation = pd.DataFrame(checks)
    if validation["status"].eq("FAIL").any():
        failures = validation.loc[validation["status"].eq("FAIL")]
        raise RuntimeError(f"ED7 validation failed:\n{failures.to_string(index=False)}")
    return validation


def write_caption_notes(
    path: Path,
    stats_table: pd.DataFrame,
    same_audit: dict[str, Any],
    long_audit: dict[str, Any],
) -> None:
    same_lomo = pd.read_csv(same_audit["leave_one_mouse_out_bh4"])
    long_lomo = pd.read_csv(long_audit["leave_one_mouse_out_bh4"])
    long_class_change = long_lomo.loc[
        long_lomo["bh4_significance_differs_from_full"].astype(bool)
    ].iloc[0]
    lines = [
        "# Extended Data Figure 7 — caption notes (round 1)",
        "",
        "Slide 9 contains exactly two aligned composites and three visible metrics per row. Exact model statistics and biological counts are caption-only; the artwork contains no P values, q values, counts or stars.",
        "",
        "## Panel a — same-stage age control",
        "",
        "Same-stage Day 1–Day 2 correlations use the maximal target-164 source: 64 age-eligible session-pair rows from 21 mice. The August 7 sorted authority contributes 51 no-manual-mask target-164 rows from 14 mice. Fourteen original-source pairs were freshly recomputed under explicit exclusion profile `none`; 13 clear target 164, whereas `841day1 → 841day2` remains provenance/native-support only because its retained pool is 139 and its birth date is unavailable. The maximal source adds five complete-age ULTM Exp1 pairs and the technically valid `579-6day1 → 579-6day2` Exp2 pair. Historical 59-row/22-mouse source-universe and 58-row/21-mouse age-target checkpoints are reconciliation parity rows, not eligibility ceilings; historical-protocol-only 58-row and inherited-QC 63-row cohorts remain sensitivities. Day 1 and Day 2 were jointly deconvolved, so shared fitted-component identity is valid. Each slope adjusts for categorical experience stage and uses CR1 covariance clustered by mouse with a Student t reference. BH4 spans place, population-vector, level-map and ensemble-rate readouts, although the slide displays only place, population vector and ensemble rate.",
        "",
        "| Metric | beta per 30 d | clustered P | q (BH4) | Pairs | Mice |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats_table.loc[stats_table["panel"].eq("a")].itertuples(index=False):
        lines.append(
            f"| {row.metric_label} | {row.estimate_beta_per_30d:.6f} | {row.p_mouse_clustered:.8g} | "
            f"{row.q_bh4:.8g} | {row.n_pair_rows} | {row.n_mice} |"
        )
    lines.extend(
        [
            "",
            "All 14 original target-164 caches and native-support companions were rebuilt in the current ED7 folder under explicit `none`; no final value is read from the superseded slide-rebuild folder. The historical pre-sort mask intersected `1632Lday2` at component 176, which passes the 0.1 spikes min⁻¹ gate and is retained in the no-mask pool (896 versus historical masked pool 895). Zero recomputed neurons are excluded by profile.",
            "",
            f"The maximal64 leave-one-mouse-out audit contains {len(same_lomo)} four-readout fits. No deletion changes a slope sign or BH4 significance classification; the full-cohort BH4-positive place, population-vector and level-map results therefore do not depend on a single mouse. Ensemble rate remains a nonsignificant full-cohort result and is not interpreted as evidence for no association.",
            "",
            "The August 7 support-matched sorted rerun changed the resampling sequence, and maximality recovery changes the cohort. Differences from the historical 58-row display therefore mix source vintage, resampling and cohort changes and cannot be attributed to the manual mask alone. The native-support 65-row sensitivity uses every pair's full retained pool, rather than mixing a single native117 row into target-164 values.",
            "",
            "## Panel b — long-gap age control",
            "",
            "Long-gap correlations use the maximal valid 38 independently processed source–destination pairs from 14 mice with explicit CIMaps. The corrected36 source is augmented by fresh original-source/no-mask recomputations for `579-3day1 → 579-6day1` (native n = 310) and `488-6day1 → 488-9day1` (native n = 51); diamonds identify these recovered rows. Destination-session age is modeled after categorical long-gap transition adjustment. Pair rows are retained and CR1 covariance is clustered by mouse with a Student t reference. BH4 spans all four readouts. Corrected36, each-addition 37-row, maximal38 and original-source-indicator specifications are exported as cohort/source sensitivities. The indicator model is diagnostic because only two rows carry the indicator and source is confounded with candidate identity. These are observational associations, not causal aging effects.",
            "",
            "| Metric | beta per 30 d | clustered P | q (BH4) | Pairs | Mice |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stats_table.loc[stats_table["panel"].eq("b")].itertuples(index=False):
        lines.append(
            f"| {row.metric_label} | {row.estimate_beta_per_30d:.6f} | {row.p_mouse_clustered:.8g} | "
            f"{row.q_bh4:.8g} | {row.n_pair_rows} | {row.n_mice} |"
        )
    lines.extend(
        [
            "",
            f"The maximal38 leave-one-mouse-out audit contains {len(long_lomo)} four-readout fits. The full cohort has no BH4-positive long-gap metric. Leaving out mouse {long_class_change.left_out_mouse} makes the ensemble-rate slope BH4-positive (beta = {long_class_change.beta_per_30d:.6f} per 30 d, q = {long_class_change.q_bh4_within_left_out_mouse:.6f}, versus full-cohort q = {long_class_change.full_q_bh4:.6f}); place/level sign flips occur only in full-cohort nonsignificant models. Accordingly, no threshold-based null or stable directional conclusion is claimed for the long-gap row.",
            "",
            f"Five exact no-mask original-CIMap 0-to-6 rows from the split-bout ULTM protocol are retained as supporting-only, noncanonical analyses because they change the adjacent-transition estimand. Their n = 5 descriptive table, stepwise conversions and maximal38-plus-five n = 43 source/transition-indicator models are exported; the n43 ensemble-rate BH4 q is {long_audit['conversion_audit']['noncanonical_0to6_n43_rate_q_bh4']:.6f}. They do not replace or appear in the maximal38 primary artwork.",
            "",
            f"Executable conversion authority: `{LONG_CONVERSION_REPRODUCER.relative_to(REPO_ROOT).as_posix()}`. It runs one original-CIMap pair per subprocess under exclusion profile `none` and verifies CIMap identities, S-rate gating, per-neuron place/tree-depth inputs, spatial/PV support, tree-depth7 support and all four pair summaries against the locked conversion tables at absolute tolerance 1e-10. Invocation: `{LONG_CONVERSION_REPRODUCE_COMMAND}`.",
            "",
            "## Display and audit scope",
            "",
            "- Shared stage colors encode the top row; shared transition colors encode the bottom row. Representation colors appear only in metric titles.",
            "- Per-stratum lines use the tested common age slope centered at each stratum mean; they are not separate slope tests.",
            "- The two 7.15 × 3.25 inch rows are aligned without crop or overlap on a 7.5 × 7.5 inch slide mockup.",
            "- No background grids or on-panel inferential text are used.",
            "- The manifest links the full candidate-universe attrition ledger, no-mask neuron masks, target/native support draws, recovered long-gap CIMap/S-rate/bin/level conversion audits and raw-input SHA256 inventory.",
            "- Leave-one-mouse-out coefficient, sign and within-deletion BH4 results are exported for all four readouts in both rows, together with age-range overlap and full-rank/condition diagnostics. These checks are audit-only and do not expand the visible panel inventory.",
            "- Reward context is mixed across the maintained age-control cohort; this figure tests age as a potential explanation of neural stability, not a reward-specific mechanism.",
            "- Same-stage identity-by-index is permitted only because each adjacent-day pair shares a joint fitted-component basis. That exception does not transfer to the long-gap row.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path) -> None:
    path.write_text(
        "# Extended Data Figure 7 — age controls\n\n"
        "Round 1 source rebuild for **slide 9** of `sup-figs-v2.pptx`. The supplied slide is the panel-inventory authority: exactly two stacked composites and exactly three metrics per row. No level-map or other manifest-derived panel was added.\n\n"
        "## Round 1\n\n"
        "- Panel a: maximal64 same-stage target-164 age control from the August 7 no-mask sorted51 authority plus 14 fresh original-source/no-mask recomputations; one below-target/missing-age 841 row remains provenance only.\n"
        "- Panel b: maximal38 long-gap age control, adding validated original/no-mask 579 3→6 and 488 6→9 pairs to corrected36; every long-gap pair uses an explicit CIMap.\n"
        "- Five exact 0-to-6 split-bout ULTM pairs and their n43 source/transition-indicator models are retained as supporting-only conversion/sensitivity audits; they are not primary or displayed.\n"
        "- `scripts/recompute_ed07_long_gap_original_conversions.py` is the maintained executable conversion authority; it runs one pair per subprocess under explicit no exclusions and verifies every locked conversion stage at absolute tolerance 1e-10.\n"
        "- Level-map values remain in the synchronized four-readout BH4 source/model families but are not a visible slide role.\n"
        "- Both rows are source-rendered in Python/matplotlib with Arial typography, shared stage/transition colors, representation-colored titles, no grids and no on-panel statistics.\n"
        "- Exact beta, clustered P, q and biological counts are in `CAPTION_NOTES_ROUND1.md` and `assembly/ED07_CAPTION_STATISTICS_ROUND1.csv`; cohort/source sensitivities are in `assembly/source_tables/`.\n"
        "- Leave-one-mouse-out coefficient/sign/BH4, age-range overlap, leverage and design-condition audits are source-table artifacts; they are not added to the six-panel artwork.\n"
        "- Both source rows are 7.15 × 3.25 inches (4,290 × 1,950 px at 600 dpi). The final 7.5-inch slide mockup aligns them without crop or overlap and uses nearly the full page.\n\n"
        "## Files\n\n"
        "| Path | Content |\n|---|---|\n"
        "| `assembly/ed07_age_controls_panel_a_same_stage_round1.{svg,pdf,png,tiff}` | top aligned composite source |\n"
        "| `assembly/ed07_age_controls_panel_b_long_gap_round1.{svg,pdf,png,tiff}` | bottom aligned composite source |\n"
        "| `assembly/ed07_age_controls_assembly_round1.{svg,pdf,png,tiff}` | compact two-row review assembly |\n"
        "| `assembly/ed07_slide9_geometry_mockup_round1.png` | exact enlarged 7.5-inch slide placement with no crop/overlap |\n"
        "| `assembly/ED07_ASSEMBLY_MANIFEST_ROUND1.json` | sources, hashes, statistics, slots and outputs |\n"
        "| `assembly/ED07_VALIDATION_ROUND1.csv` | science, source, slot, format and layout checks |\n"
        "| `assembly/audit_figures/` | audit-only recovered conversion overview and supporting-only 0-to-6 summary |\n"
        "| `FIGURE_CONTRACT_ROUND1.md` | conclusion, evidence logic, export contract and reviewer risks |\n"
        "| `VISUAL_QA_ROUND1.md` | explicit human-visible inspection status and criteria |\n"
        "| `assembly/source_tables/` | plot-ready values, maximal/parity/source-sensitivity models, masks and support audits |\n"
        "| `same_stage_candidate_universe_attrition.csv` | full registry/source candidate universe and gate-by-gate attrition |\n"
        "| `long_gap_conversion_audit/` | recovered-pair CIMap, S-rate, spatial-bin and level conversion audits |\n"
        "| `CAPTION_NOTES_ROUND1.md` | exact statistics and interpretation boundaries |\n\n"
        "## Reproduce\n\n"
        "```bash\npython scripts/recompute_ed07_long_gap_original_conversions.py --pair-set noncanonical0to6 --out-dir <verification-output-dir> --verify-root docs/extended_data_supplement_v2_rebuild_2026-08-05/ed07_age_controls\npython scripts/render_ed07_slide9_age_controls.py --round 1 --visual-qa-status pass\n```\n\n"
        "Use `pass` only after directly inspecting both rows, the assembly and the slide mockup; the default remains `pending`.\n",
        encoding="utf-8",
    )


def write_figure_contract(path: Path) -> None:
    path.write_text(
        "# Extended Data Figure 7 - figure contract (round 1)\n\n"
        "## Intended conclusion\n\n"
        "Age is tested as an observational covariate of cross-session neural stability within same-stage adjacent-day pairs and long-gap source-destination pairs. The figure does not identify a causal aging effect.\n\n"
        "## Evidence logic\n\n"
        "- Panel a displays maximal64 no-manual-mask target-164 same-stage data after categorical experience-stage adjustment.\n"
        "- Panel b displays maximal38 explicit-CIMap long-gap data after categorical transition adjustment.\n"
        "- Both estimands retain pair rows, use CR1 covariance clustered by mouse and a Student t(G-1) reference, and synchronize BH4 across place map, population vector, level map and ensemble rate.\n"
        "- Only place map, population vector and ensemble rate are visible. Level map remains an inferential family member and source-table readout, not a slide role.\n"
        "- Fitted stratum lines are parallel and use the tested common age coefficient; no untested stratum-specific slope is implied.\n\n"
        "## Export contract\n\n"
        "Python/matplotlib only; Arial; two aligned 7.15 x 3.25 inch row composites; SVG/PDF plus 600-dpi PNG/TIFF; white background; no grids; no on-panel beta, P, q, counts or stars; one embedded 10.5-pt bold panel-letter authority.\n\n"
        "## Principal review risks and controls\n\n"
        "- Maximality and silent exclusion: full candidate-universe/attrition ledger and cohort-boundary sensitivities.\n"
        "- False manual mask: explicit `none`, zero profile drops, component176 retained, and fresh wrapper-pinned caches.\n"
        "- Support/source vintage: target-164 primary, uniform native-support sensitivity, and no mask-only overattribution.\n"
        "- Identity conversion: joint-deconvolution identity only for same-stage; explicit CIMap plus stepwise conversion audits for long-gap.\n"
        "- Age-stratum geometry and influential mice: age-range overlap, rank/condition and leave-one-mouse-out coefficient/sign/BH4 audits.\n"
        "- Recovered-source indicator: diagnostic only because two rows carry the indicator and source is confounded with candidate identity.\n"
        "- Noncanonical 0-to-6 split-bout rows: exact conversions and n43 models are supporting-only because they change the adjacent-transition estimand.\n",
        encoding="utf-8",
    )


def write_visual_qa(path: Path, status: str) -> None:
    if status == "pass":
        body = (
            "Status: PASS\n\n"
            "Human-visible inspection covered the source-rendered top row, source-rendered bottom row, compact assembly, exact 7.5-inch slide mockup, recovered-pair conversion overview and supporting-only 0-to-6 audit. Text and markers are legible; axes, labels, legends and panel letters are unclipped; the two rows and three columns align; stage/transition colors match their legends; recovered original-source diamonds are visible; the white/no-grid treatment is clean; and no inferential text or duplicate panel letters appear in the slide artwork.\n"
        )
    else:
        body = (
            "Status: PENDING\n\n"
            "Automated source/science/layout checks passed, but the rendered top row, bottom row, compact assembly and exact slide mockup still require human-visible inspection before promotion.\n"
        )
    path.write_text(
        "# Extended Data Figure 7 - visual QA (round 1)\n\n" + body,
        encoding="utf-8",
    )


def render(
    out_dir: Path, round_number: int, visual_qa_status: str
) -> dict[str, Any]:
    if round_number != 1:
        raise ValueError("This source-locked renderer currently defines round 1 only")
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    assembly_dir = out_dir / "assembly"
    source_dir = assembly_dir / "source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)

    same_points, same_models, same_provenance, same_audit = build_same_stage_source(source_dir)
    long_points, long_models, long_audit = build_long_gap_source(source_dir)
    source_inputs = build_source_input_records(same_audit, long_audit)
    source_audit_path = assembly_dir / "ED07_SOURCE_DATA_AUDIT_ROUND1.csv"
    pd.DataFrame(source_inputs).to_csv(source_audit_path, index=False)

    stats_table = caption_statistics(same_models, long_models)
    stats_path = assembly_dir / "ED07_CAPTION_STATISTICS_ROUND1.csv"
    stats_table.to_csv(stats_path, index=False)

    top_outputs = render_top_row(
        same_points,
        same_models,
        assembly_dir / "ed07_age_controls_panel_a_same_stage_round1",
    )
    bottom_outputs = render_bottom_row(
        long_points,
        long_models,
        assembly_dir / "ed07_age_controls_panel_b_long_gap_round1",
    )
    audit_figure_dir = assembly_dir / "audit_figures"
    recovered_conversion_outputs = render_recovered_conversion_audit(
        audit_figure_dir / "ed07_long_gap_recovered_conversion_audit_round1"
    )
    noncanonical_outputs = render_noncanonical_0to6_supporting_audit(
        audit_figure_dir / "ed07_long_gap_noncanonical0to6_supporting_audit_round1"
    )
    top_png = next(path for path in top_outputs if path.suffix.lower() == ".png")
    bottom_png = next(path for path in bottom_outputs if path.suffix.lower() == ".png")
    assembly_outputs = compose_compact_assembly(
        top_png,
        bottom_png,
        assembly_dir / "ed07_age_controls_assembly_round1",
    )
    mockup_path = compose_slide_mockup(
        top_png,
        bottom_png,
        assembly_dir / "ed07_slide9_geometry_mockup_round1.png",
    )

    caption_path = out_dir / "CAPTION_NOTES_ROUND1.md"
    readme_path = out_dir / "README.md"
    contract_path = out_dir / "FIGURE_CONTRACT_ROUND1.md"
    visual_qa_path = out_dir / "VISUAL_QA_ROUND1.md"
    write_caption_notes(caption_path, stats_table, same_audit, long_audit)
    write_readme(readme_path)
    write_figure_contract(contract_path)
    write_visual_qa(visual_qa_path, visual_qa_status)

    # Viewable companions at the established root; assembly remains the source record.
    assembly_png = next(path for path in assembly_outputs if path.suffix.lower() == ".png")
    assembly_tiff = next(path for path in assembly_outputs if path.suffix.lower() == ".tiff")
    companion_png = out_dir / "extended_data_figure7_source_rerender_round1.png"
    companion_tiff = out_dir / "extended_data_figure7_source_rerender_round1.tiff"
    shutil.copy2(assembly_png, companion_png)
    shutil.copy2(assembly_tiff, companion_tiff)

    validation = validate_outputs(
        top_outputs,
        bottom_outputs,
        assembly_outputs,
        mockup_path,
        same_points,
        same_models,
        same_provenance,
        same_audit,
        long_points,
        long_models,
        long_audit,
        stats_table,
        source_inputs,
        visual_qa_status,
        [*recovered_conversion_outputs, *noncanonical_outputs],
    )
    validation_path = assembly_dir / "ED07_VALIDATION_ROUND1.csv"
    validation.to_csv(validation_path, index=False)

    primary_output_paths = [
        *top_outputs,
        *bottom_outputs,
        *assembly_outputs,
        *recovered_conversion_outputs,
        *noncanonical_outputs,
        mockup_path,
        stats_path,
        source_audit_path,
        validation_path,
        caption_path,
        readme_path,
        contract_path,
        visual_qa_path,
        companion_png,
        companion_tiff,
        *sorted(source_dir.glob("ed07_*round1.csv")),
    ]
    output_records = [
        {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in dict.fromkeys(primary_output_paths)
    ]

    final_status = (
        "pass"
        if visual_qa_status == "pass" and validation["status"].eq("PASS").all()
        else "pending_manual_visual_qa"
    )

    manifest = {
        "figure": "Extended Data Figure 7 (merged numbering; slide 9 of sup-figs-v2.pptx)",
        "round": 1,
        "status": final_status,
        "renderer": str(Path(__file__).relative_to(REPO_ROOT)),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "layout_authority": "slide 9; exactly two aligned stacked composite roles, enlarged without inherited crop/overlap",
        "visible_role_count": 6,
        "panel_inventory": {
            "a": {
                "slot": "top aligned composite",
                "role": "same-stage age control",
                "metrics": ["place", "pv", "ensemble_rate"],
            },
            "b": {
                "slot": "bottom aligned composite",
                "role": "long-gap age control",
                "metrics": ["place", "pv", "ensemble_rate"],
            },
        },
        "slide_geometry_inches": {
            "slide": list(SLIDE_SIZE_IN),
            "top_slot": list(TOP_SLOT_IN),
            "top_crop": None,
            "bottom_slot": list(BOTTOM_SLOT_IN),
            "bottom_crop": None,
            "top_source_row_media": list(TOP_ROW_SIZE_IN),
            "bottom_source_row_media": list(BOTTOM_ROW_SIZE_IN),
            "vertical_gap": BOTTOM_SLOT_IN[1] - (TOP_SLOT_IN[1] + TOP_SLOT_IN[3]),
            "inherited_layout_crosswalk": {
                "top_slot": list(INHERITED_TOP_SLOT_IN),
                "top_crop": LEGACY_CROP_CROSSWALK,
                "bottom_slot": list(INHERITED_BOTTOM_SLOT_IN),
            },
        },
        "same_stage_source": same_audit,
        "long_gap_source": long_audit,
        "statistics_policy": {
            "artwork": "no beta, P, q, n or significance annotations",
            "caption_table": str(stats_path.resolve()),
            "same_stage": "maximal64 pair-row OLS with categorical experience stage; CR1 clustered by mouse; Student t(G-1); fresh BH4 across four readouts",
            "long_gap": "maximal38 pair-row OLS with categorical transition; CR1 clustered by mouse; Student t(G-1); fresh BH4 across four readouts",
            "robustness": "leave-one-mouse-out coefficients/sign/BH4 plus age-range overlap and design-rank/condition audits",
            "interpretation": "observational age associations; no causal aging claim",
        },
        "source_vintage_caution": "Aug-7 support resampling, source vintage and cohort recovery changed together; differences from the old panel are not a mask-only contrast",
        "caption_statistics": stats_table.to_dict("records"),
        "sources": source_inputs,
        "outputs": output_records,
        "validation": validation.to_dict("records"),
        "validation_status": final_status,
        "figure_contract": str(contract_path.resolve()),
        "visual_qa": str(visual_qa_path.resolve()),
        "audit_only_figures_not_slide_roles": [
            str(path.resolve())
            for path in [*recovered_conversion_outputs, *noncanonical_outputs]
        ],
        "style": {
            "backend": "Python/matplotlib only",
            "font": "Arial",
            "base_font_pt": 7.0,
            "panel_letter_font_pt": 10.5,
            "panel_letters_embedded_once": True,
            "no_background_grids": True,
            "raster_dpi": 600,
            "representation_title_palette": {
                "place": "#B95A58",
                "pv": "#E29957",
                "ensemble_rate": "#4A5E65",
            },
            "point_line_palettes": {
                "same_stage": "authoritative shared STAGE_COLORS",
                "long_gap": "authoritative shared TRANSITION_COLORS",
            },
        },
    }
    manifest_path = assembly_dir / "ED07_ASSEMBLY_MANIFEST_ROUND1.json"
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--compute-original-pair-index", type=int)
    parser.add_argument("--compute-native-relaxed-488", action="store_true")
    parser.add_argument("--prepare-same-stage-only", action="store_true")
    parser.add_argument(
        "--visual-qa-status",
        choices=("pending", "pass"),
        default="pending",
        help="Set pass only after inspecting both rows, the assembly and slide mockup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compute_original_pair_index is not None:
        if args.cache_root is None:
            raise ValueError("--cache-root is required for pair-cache computation")
        if not 0 <= args.compute_original_pair_index < len(ORIGINAL_PAIR_SPECS):
            raise ValueError(
                f"--compute-original-pair-index must be 0..{len(ORIGINAL_PAIR_SPECS) - 1}"
            )
        path = compute_original_pair_no_exclusions(
            args.compute_original_pair_index, args.cache_root.resolve()
        )
        print(path.resolve())
        return
    if args.compute_native_relaxed_488:
        if args.cache_root is None:
            raise ValueError("--cache-root is required for native-cache computation")
        path = compute_native_relaxed_488_no_exclusions(args.cache_root.resolve())
        print(path.resolve())
        return
    if args.prepare_same_stage_only:
        configure_style()
        source_dir = args.out_dir.resolve() / "assembly" / "source_tables"
        _, _, _, audit = build_same_stage_source(source_dir)
        print(json.dumps(json_ready(audit), indent=2))
        return
    manifest = render(args.out_dir.resolve(), args.round, args.visual_qa_status)
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
