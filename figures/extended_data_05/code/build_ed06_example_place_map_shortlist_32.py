"""Build the 32-pair, 30%-level-overlap ED6 shortlist from frozen v3.

The script performs no raw-session or cache recomputation.  It derives a
display-level activity profile from each independently q99-normalized 45 x 45
map, selects 32 examples under explicit representation-coverage and cross-pair
morphology constraints, and renders an ED6-ready 8 x 8 map grid in frozen v3
visual-quality order.  An optional correction mode locks a reviewed shortlist,
removes only named display ranks, and chooses replacements with clearly bounded,
multi-field activity at the final figure scale.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping
import uuid

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ed06_shortlist32_mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for source_root in (REPO_ROOT, SRC_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ca_analysis.ed06_example_candidates import CORRELATION_BIN_LABELS
from ca_analysis.ed06_level_preservation import (
    DEFAULT_ACTIVITY_FLOOR,
    DEFAULT_RELIABLE_FINE_BINS,
    compute_level_activity_profile,
    encoded_level_jaccard,
    level_profile_overlap,
    ordered_level_earth_mover,
    preservation_stratum,
)
from ca_analysis.ed06_representation_diversity import build_fine_maze_crosswalk
from ca_analysis.ed06_representation_selection import (
    select_representation_diverse_candidates,
)
from scripts import build_ed06_example_place_map_candidates_v3 as v3


DEFAULT_PARENT = REPO_ROOT / "outputs" / "ed05_candidates"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "ed05_shortlist"
TARGET_N = 32
VISUAL_RETENTION = 0.97
DIVERSITY_RETENTION = 0.98
LEVEL_BIAS_WEIGHT = 0.30
MORPHOLOGY_ACTIVITY_FLOOR = 0.20
MORPHOLOGY_POWER = 1.40
MORPHOLOGY_GAUSSIAN_SIGMA = 1.25
MORPHOLOGY_SIDE_SIMILARITY_THRESHOLD = 0.85
MORPHOLOGY_FIELD_OVERLAP_SIMILARITY_THRESHOLD = 0.75
MORPHOLOGY_FIELD_DICE_THRESHOLD = 0.80
PROFILE_SENSITIVITY_FLOORS = (0.10, 0.30, 0.40)
PAIR_CARD_EDGE_COLOR = "#B8B8B8"
MAIN_FIG_PAIR_COLOR = "#B64342"
PAIR_CARD_LINEWIDTH = 0.45
MAIN_FIG_PAIR_LINEWIDTH = 1.10
FIELD_READABILITY_MAX_EFFECTIVE_BINS = 24.0
FIELD_READABILITY_MIN_EXPORTED_FIELDS = 2
FIELD_READABILITY_MIN_PROMINENT_FIELDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", default=str(DEFAULT_PARENT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--solver-time-limit", type=float, default=180.0)
    parser.add_argument(
        "--reference-shortlist-dir",
        help="Reviewed 32-pair bundle whose accepted rows should be locked.",
    )
    parser.add_argument(
        "--replace-visual-ranks",
        default="11,12,16,23",
        help="Comma-separated display ranks to replace from the reference bundle.",
    )
    parser.add_argument("--skip-rendering", action="store_true")
    parser.add_argument("--skip-individuals", action="store_true")
    return parser.parse_args()


def _json_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        decoded = json.loads(text) if text.startswith("[") else [text]
    elif isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
        decoded = value
    else:
        decoded = [value]
    return [str(token) for token in decoded]


def _int_set(value: object) -> set[int]:
    return {int(token) for token in _json_tokens(value)}


def _primary_run_token(value: object) -> str:
    """Return the strongest field's run token from a frozen-v3 token list."""

    tokens = _json_tokens(value)
    if not tokens:
        return "none"
    parts = str(tokens[0]).split(":")
    if len(parts) < 2 or not parts[1].startswith("R"):
        raise ValueError(f"malformed dominant-run token: {tokens[0]!r}")
    return parts[1]


def _dice_coefficient(first: np.ndarray, second: np.ndarray) -> float:
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    denominator = int(first_bool.sum() + second_bool.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(first_bool & second_bool) / denominator)


def _gini_coefficient(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    if finite.size == 0 or float(finite.sum()) <= 0.0:
        return 0.0
    ordered = np.sort(finite)
    count = ordered.size
    return float(
        2.0
        * np.dot(np.arange(1, count + 1, dtype=float), ordered)
        / (count * ordered.sum())
        - (count + 1.0) / count
    )


def _field_readability_side_record(
    normalized: np.ndarray,
    common_mask: np.ndarray,
    field_labels: np.ndarray,
) -> dict[str, float | int]:
    """Summarize whether multiple exported fields remain crisp at panel scale."""

    values = np.asarray(normalized, dtype=float)
    common = np.asarray(common_mask, dtype=bool) & np.isfinite(values)
    labels = np.asarray(field_labels, dtype=int)
    activity = np.clip(values[common], 0.0, None)
    total_mass = float(activity.sum())
    field_mask = common & (labels > 0)
    field_mass = float(np.clip(values[field_mask], 0.0, None).sum())
    field_records: list[dict[str, float]] = []
    for field_id in sorted(set(labels[field_mask].tolist())):
        use = field_mask & (labels == int(field_id))
        field_values = np.clip(values[use], 0.0, None)
        mass = float(field_values.sum())
        squared_mass = float(np.square(field_values).sum())
        field_records.append(
            {
                "peak": float(field_values.max(initial=0.0)),
                "mass_share": mass / total_mass if total_mass > 0.0 else 0.0,
                "high_bins": float(np.count_nonzero(field_values >= 0.50)),
                "effective_bins": (
                    mass * mass / squared_mass if squared_mass > 0.0 else 0.0
                ),
            }
        )
    prominent = [
        record
        for record in field_records
        if record["peak"] >= 0.50
        and record["mass_share"] >= 0.015
        and record["high_bins"] >= 1.0
    ]
    field_peaks = sorted(
        (record["peak"] for record in field_records), reverse=True
    )
    effective_bins = [record["effective_bins"] for record in field_records]
    return {
        "field_readability_gini": _gini_coefficient(activity),
        "field_readability_field_mass_share": (
            field_mass / total_mass if total_mass > 0.0 else 0.0
        ),
        "field_readability_second_field_peak": (
            float(field_peaks[1]) if len(field_peaks) >= 2 else 0.0
        ),
        "field_readability_prominent_fields": len(prominent),
        "field_readability_median_effective_bins": (
            float(np.median(effective_bins)) if effective_bins else 0.0
        ),
        "field_readability_max_effective_bins": (
            float(max(effective_bins)) if effective_bins else 0.0
        ),
        "field_readability_exported_fields": len(field_records),
    }


def add_field_readability_columns(
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Add a field-size/readability audit without rewarding fewer fields."""

    work = frame.copy().reset_index(drop=True)
    if len(work) != np.asarray(arrays["candidate_id"]).shape[0]:
        raise ValueError("field-readability arrays do not match candidate rows")
    records: list[dict[str, object]] = []
    for index, row in enumerate(work.itertuples(index=False)):
        record: dict[str, object] = {"candidate_id": str(row.candidate_id)}
        for side in ("a", "b"):
            side_record = _field_readability_side_record(
                arrays[f"normalized_{side}"][index],
                arrays["common_mask"][index],
                arrays[f"display_field_labels_{side}"][index],
            )
            record.update(
                {f"{name}_{side}": value for name, value in side_record.items()}
            )
        records.append(record)
    metrics = pd.DataFrame(records)
    work = work.merge(metrics, on="candidate_id", validate="one_to_one")
    for side in ("a", "b"):
        exported = work[f"field_readability_exported_fields_{side}"].astype(int)
        expected = work[f"field_count_{side}"].astype(int)
        if not exported.eq(expected).all():
            raise ValueError(f"field-readability labels disagree with side {side}")
        for metric in (
            "gini",
            "field_mass_share",
            "second_field_peak",
        ):
            column = f"field_readability_{metric}_{side}"
            work[f"field_readability_percentile_{metric}_{side}"] = work[
                column
            ].rank(method="average", pct=True)
        for metric in ("median_effective_bins", "max_effective_bins"):
            column = f"field_readability_{metric}_{side}"
            work[f"field_readability_percentile_compact_{metric}_{side}"] = (
                -work[column]
            ).rank(method="average", pct=True)
        work[f"field_readability_score_{side}"] = (
            0.45
            * work[
                f"field_readability_percentile_compact_max_effective_bins_{side}"
            ]
            + 0.20
            * work[
                f"field_readability_percentile_compact_median_effective_bins_{side}"
            ]
            + 0.15
            * work[f"field_readability_percentile_second_field_peak_{side}"]
            + 0.10 * work[f"field_readability_percentile_gini_{side}"]
            + 0.10
            * work[f"field_readability_percentile_field_mass_share_{side}"]
        )
    work["pair_field_readability_min"] = work[
        ["field_readability_score_a", "field_readability_score_b"]
    ].min(axis=1)
    work["pair_field_readability_mean"] = work[
        ["field_readability_score_a", "field_readability_score_b"]
    ].mean(axis=1)
    work["pair_field_readability_score"] = (
        0.75 * work["pair_field_readability_min"]
        + 0.25 * work["pair_field_readability_mean"]
    )
    work["pair_field_readability_max_effective_bins"] = work[
        [
            "field_readability_max_effective_bins_a",
            "field_readability_max_effective_bins_b",
        ]
    ].max(axis=1)
    work["field_readability_replacement_eligible"] = (
        (work["field_count_a"] >= FIELD_READABILITY_MIN_EXPORTED_FIELDS)
        & (work["field_count_b"] >= FIELD_READABILITY_MIN_EXPORTED_FIELDS)
        & (
            work["field_readability_prominent_fields_a"]
            >= FIELD_READABILITY_MIN_PROMINENT_FIELDS
        )
        & (
            work["field_readability_prominent_fields_b"]
            >= FIELD_READABILITY_MIN_PROMINENT_FIELDS
        )
        & (
            work["pair_field_readability_max_effective_bins"]
            <= FIELD_READABILITY_MAX_EFFECTIVE_BINS
        )
    )
    work["replacement_utility_lambda_0"] = work[
        "pair_field_readability_score"
    ]
    work["replacement_utility_lambda_0.10"] = (
        0.90 * work["pair_field_readability_score"]
        + 0.10 * work["level_profile_overlap"]
    )
    work["replacement_utility_lambda_0.30"] = (
        (1.0 - LEVEL_BIAS_WEIGHT) * work["pair_field_readability_score"]
        + LEVEL_BIAS_WEIGHT * work["level_profile_overlap"]
    )
    return work


def _morphology_conflict_reason(blurred_cosine: float, field_dice: float) -> str:
    if float(blurred_cosine) >= MORPHOLOGY_SIDE_SIMILARITY_THRESHOLD:
        return "high_blurred_cosine"
    if (
        float(blurred_cosine) >= MORPHOLOGY_FIELD_OVERLAP_SIMILARITY_THRESHOLD
        and float(field_dice) >= MORPHOLOGY_FIELD_DICE_THRESHOLD
    ):
        return "moderate_cosine_high_field_dice"
    return ""


def _blurred_map_embedding(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Return a normalized, tolerance-smoothed map vector for visual matching."""

    map_values = np.asarray(values, dtype=float)
    valid = np.asarray(support, dtype=bool)
    if map_values.shape != (45, 45) or valid.shape != map_values.shape:
        raise ValueError("morphology inputs must be 45 x 45")
    focused = np.where(
        valid,
        np.clip(
            (map_values - MORPHOLOGY_ACTIVITY_FLOOR)
            / (1.0 - MORPHOLOGY_ACTIVITY_FLOOR),
            0.0,
            1.0,
        )
        ** MORPHOLOGY_POWER,
        0.0,
    )
    blurred = gaussian_filter(focused, sigma=MORPHOLOGY_GAUSSIAN_SIGMA)
    vector = np.where(valid, blurred, 0.0).ravel()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector


def build_pair_morphology_audit(
    frame: pd.DataFrame, arrays: Mapping[str, np.ndarray]
) -> pd.DataFrame:
    """Compare every candidate pair using all four individual-map matchings.

    A candidate-pair conflict requires one sidewise comparison to exceed the
    strict continuous blurred-map threshold, or a moderate blurred-map match
    together with high display-field overlap. This prevents an unrelated partner
    map or diffuse tails from hiding a repeated individual map.
    """

    candidate_ids = frame["candidate_id"].astype(str).tolist()
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("morphology audit requires unique candidate ids")
    normalized = {
        "A": np.asarray(arrays["normalized_a"], dtype=float),
        "B": np.asarray(arrays["normalized_b"], dtype=float),
    }
    labels = {
        "A": np.asarray(arrays["display_field_labels_a"]),
        "B": np.asarray(arrays["display_field_labels_b"]),
    }
    support = np.asarray(arrays["common_mask"], dtype=bool)
    expected_shape = (len(frame), 45, 45)
    if support.shape != expected_shape or any(
        values.shape != expected_shape
        for values in (*normalized.values(), *labels.values())
    ):
        raise ValueError("morphology arrays do not match candidate ordering")

    embeddings = {
        side: np.stack(
            [
                _blurred_map_embedding(values[index], support[index])
                for index in range(len(frame))
            ]
        )
        for side, values in normalized.items()
    }
    activity_masks = {
        side: support & (values >= MORPHOLOGY_ACTIVITY_FLOOR)
        for side, values in normalized.items()
    }
    field_masks = {side: values > 0 for side, values in labels.items()}

    records: list[dict[str, object]] = []
    for first in range(len(frame)):
        for second in range(first + 1, len(frame)):
            comparisons: list[dict[str, object]] = []
            for side_first in ("A", "B"):
                for side_second in ("A", "B"):
                    blurred_cosine = float(
                        embeddings[side_first][first] @ embeddings[side_second][second]
                    )
                    field_dice = _dice_coefficient(
                        field_masks[side_first][first],
                        field_masks[side_second][second],
                    )
                    activity_dice = _dice_coefficient(
                        activity_masks[side_first][first],
                        activity_masks[side_second][second],
                    )
                    conflict_reason = _morphology_conflict_reason(
                        blurred_cosine, field_dice
                    )
                    is_conflict = bool(conflict_reason)
                    comparisons.append(
                        {
                            "side_first": side_first,
                            "side_second": side_second,
                            "blurred_cosine": blurred_cosine,
                            "field_dice": field_dice,
                            "activity_dice": activity_dice,
                            "is_conflict": is_conflict,
                            "conflict_reason": conflict_reason,
                        }
                    )
            conflicts = [item for item in comparisons if item["is_conflict"]]
            best = max(
                conflicts or comparisons,
                key=lambda item: (
                    float(item["blurred_cosine"]),
                    float(item["field_dice"]),
                    float(item["activity_dice"]),
                    str(item["side_first"]),
                    str(item["side_second"]),
                ),
            )
            records.append(
                {
                    "candidate_id_first": candidate_ids[first],
                    "candidate_id_second": candidate_ids[second],
                    "parent_visual_rank_first": int(
                        frame.iloc[first]["parent_visual_rank"]
                    ),
                    "parent_visual_rank_second": int(
                        frame.iloc[second]["parent_visual_rank"]
                    ),
                    "morphology_conflict": bool(conflicts),
                    "conflict_side_first": (
                        str(best["side_first"]) if conflicts else ""
                    ),
                    "conflict_side_second": (
                        str(best["side_second"]) if conflicts else ""
                    ),
                    "conflict_blurred_cosine": (
                        float(best["blurred_cosine"]) if conflicts else np.nan
                    ),
                    "conflict_field_dice": (
                        float(best["field_dice"]) if conflicts else np.nan
                    ),
                    "conflict_activity_dice": (
                        float(best["activity_dice"]) if conflicts else np.nan
                    ),
                    "conflict_reason": (
                        str(best["conflict_reason"]) if conflicts else ""
                    ),
                    "max_blurred_cosine": max(
                        float(item["blurred_cosine"]) for item in comparisons
                    ),
                    "max_field_dice": max(
                        float(item["field_dice"]) for item in comparisons
                    ),
                    "max_activity_dice": max(
                        float(item["activity_dice"]) for item in comparisons
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def build_morphology_conflict_components(
    frame: pd.DataFrame, audit: pd.DataFrame
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Return direct conflict edges and auditable graph components."""

    adjacency: dict[str, set[str]] = {
        candidate_id: set() for candidate_id in frame["candidate_id"].astype(str)
    }
    edge_rows = audit.loc[audit["morphology_conflict"].astype(bool)]
    for row in edge_rows.itertuples(index=False):
        first = str(row.candidate_id_first)
        second = str(row.candidate_id_second)
        adjacency[first].add(second)
        adjacency[second].add(first)

    rank_lookup = frame.set_index("candidate_id")["parent_visual_rank"].to_dict()
    fixed_lookup = frame.set_index("candidate_id")["fixed_main_fig2b"].to_dict()
    visited: set[str] = set()
    conflict_pairs: list[tuple[str, str]] = []
    component_rows: list[dict[str, object]] = []
    component_index = 0
    for start in sorted(adjacency, key=lambda item: (rank_lookup[item], item)):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        members: set[str] = set()
        while stack:
            candidate_id = stack.pop()
            if candidate_id in visited:
                continue
            visited.add(candidate_id)
            members.add(candidate_id)
            stack.extend(sorted(adjacency[candidate_id] - visited))
        ordered = sorted(members, key=lambda item: (rank_lookup[item], item))
        component_index += 1
        for first in ordered:
            component_rows.append(
                {
                    "component_id": f"MC{component_index:02d}",
                    "component_size": len(ordered),
                    "candidate_id": first,
                    "parent_visual_rank": int(rank_lookup[first]),
                    "fixed_main_fig2b": bool(fixed_lookup[first]),
                    "direct_conflict_neighbors": json.dumps(
                        sorted(
                            adjacency[first],
                            key=lambda item: (rank_lookup[item], item),
                        )
                    ),
                }
            )
    conflict_pairs = sorted(
        {
            tuple(sorted((str(row.candidate_id_first), str(row.candidate_id_second))))
            for row in edge_rows.itertuples(index=False)
        }
    )
    return conflict_pairs, pd.DataFrame.from_records(component_rows)


def _required_parent_paths(parent_dir: Path) -> dict[str, Path]:
    names = (
        "selected_100.csv",
        "selected_display_fields.csv",
        "selected_maps_fields_45_v3.npz",
        "selected_map_bins_topology_45_v3.csv.gz",
        "selection_manifest.json",
    )
    paths = {name: parent_dir / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen v3 parent artifacts: {missing}")
    return paths


def validate_parent_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    manifest = json.loads(paths["selection_manifest.json"].read_text(encoding="utf-8"))
    artifact_records = {
        str(record["path"]): record for record in manifest.get("artifacts", [])
    }
    for name in (
        "selected_100.csv",
        "selected_display_fields.csv",
        "selected_maps_fields_45_v3.npz",
        "selected_map_bins_topology_45_v3.csv.gz",
    ):
        record = artifact_records.get(name)
        if record is None:
            raise ValueError(f"parent manifest does not register {name}")
        path = paths[name]
        if int(record["bytes"]) != path.stat().st_size:
            raise ValueError(f"parent artifact size differs from manifest: {name}")
        if str(record["sha256"]) != v3.sha256_file(path):
            raise ValueError(f"parent artifact hash differs from manifest: {name}")
    checks = manifest.get("checks", {})
    required_checks = {
        "n_selected": 100,
        "n_session_pairs": 36,
        "duplicate_neuron_nodes": 0,
        "n_fixed_main_fig2b": 4,
        "image_conflicts_at_threshold": 0,
        "field_counts_match_exported_labels": True,
        "visual_rank_is_strict_quality_order": True,
    }
    failures = {
        key: {"expected": expected, "actual": checks.get(key)}
        for key, expected in required_checks.items()
        if checks.get(key) != expected
    }
    if failures:
        raise ValueError(f"parent manifest validation checks failed: {failures}")
    return manifest


def load_parent_bundle(
    parent_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    paths = _required_parent_paths(parent_dir)
    validate_parent_manifest(paths)
    frame = pd.read_csv(paths["selected_100.csv"])
    if len(frame) != 100 or frame["candidate_id"].nunique() != 100:
        raise ValueError("frozen v3 parent must contain 100 unique candidates")
    if frame["visual_rank"].astype(int).tolist() != list(range(1, 101)):
        raise ValueError("frozen v3 visual ranks must be exactly 1 through 100")
    if not frame["visual_score_v3"].is_monotonic_decreasing:
        raise ValueError("frozen v3 parent is not ordered by visual_score_v3")
    frame = frame.rename(columns={"visual_rank": "parent_visual_rank"})

    fields = pd.read_csv(paths["selected_display_fields.csv"])
    with np.load(paths["selected_maps_fields_45_v3.npz"], allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}
    ids = arrays["candidate_id"].astype(str).tolist()
    if ids != frame["candidate_id"].astype(str).tolist():
        raise ValueError("parent CSV and NPZ candidate ordering differs")
    for key in (
        "normalized_a",
        "normalized_b",
        "smoothed_c_raw_a",
        "smoothed_c_raw_b",
        "occupancy_mask_a",
        "occupancy_mask_b",
        "common_mask",
        "display_field_labels_a",
        "display_field_labels_b",
    ):
        if arrays[key].shape != (100, 45, 45):
            raise ValueError(
                f"unexpected parent NPZ shape for {key}: {arrays[key].shape}"
            )
    return frame, fields, arrays


def _parse_replacement_ranks(value: str) -> tuple[int, ...]:
    ranks = tuple(
        sorted(
            {
                int(token.strip())
                for token in str(value).split(",")
                if token.strip()
            }
        )
    )
    if not ranks:
        raise ValueError("at least one replacement visual rank is required")
    if any(rank < 1 or rank > TARGET_N for rank in ranks):
        raise ValueError(f"replacement ranks must be between 1 and {TARGET_N}")
    return ranks


def apply_reference_replacement_policy(
    frame: pd.DataFrame,
    *,
    reference_dir: Path,
    replacement_ranks: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Lock reviewed rows and expose only readable candidates as replacements."""

    reference_path = reference_dir / "selected_32.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference shortlist is missing {reference_path}")
    reference = pd.read_csv(reference_path)
    if len(reference) != TARGET_N or reference["candidate_id"].nunique() != TARGET_N:
        raise ValueError("reference shortlist must contain 32 unique candidates")
    if reference["visual_rank"].astype(int).tolist() != list(
        range(1, TARGET_N + 1)
    ):
        raise ValueError("reference shortlist visual ranks must be exactly 1 through 32")
    replacement_ranks = tuple(sorted(set(int(rank) for rank in replacement_ranks)))
    rejected = reference.loc[
        reference["visual_rank"].astype(int).isin(replacement_ranks)
    ].copy()
    if len(rejected) != len(replacement_ranks):
        raise ValueError("not every requested replacement rank exists in the reference")
    if rejected["fixed_main_fig2b"].astype(bool).any():
        raise ValueError("Figure 2b anchors cannot be removed in correction mode")
    parent_ids = set(frame["candidate_id"].astype(str))
    reference_ids = set(reference["candidate_id"].astype(str))
    if not reference_ids.issubset(parent_ids):
        missing = sorted(reference_ids - parent_ids)
        raise ValueError(f"reference shortlist contains non-parent candidates: {missing}")

    rejected_ids = set(rejected["candidate_id"].astype(str))
    locked_ids = reference_ids - rejected_ids
    work = frame.copy()
    work["locked_reference"] = work["candidate_id"].astype(str).isin(locked_ids)
    work["rejected_reference"] = work["candidate_id"].astype(str).isin(rejected_ids)
    work["replacement_candidate_pool"] = (
        ~work["rejected_reference"]
        & (
            work["locked_reference"]
            | work["field_readability_replacement_eligible"].astype(bool)
        )
    )
    pool = work.loc[work["replacement_candidate_pool"]].copy()
    if int(pool["locked_reference"].sum()) != TARGET_N - len(replacement_ranks):
        raise ValueError("reference lock count does not match requested replacements")

    audit_columns = [
        "candidate_id",
        "visual_rank",
        "parent_visual_rank",
        "place_r_45_display",
        "level_profile_overlap",
        "field_count_a",
        "field_count_b",
    ]
    reference_audit = reference[audit_columns].rename(
        columns={"visual_rank": "reference_visual_rank"}
    )
    metrics = work[
        [
            "candidate_id",
            "pair_field_readability_score",
            "pair_field_readability_max_effective_bins",
            "field_readability_prominent_fields_a",
            "field_readability_prominent_fields_b",
            "field_readability_replacement_eligible",
        ]
    ]
    reference_audit = reference_audit.merge(
        metrics, on="candidate_id", validate="one_to_one"
    )
    reference_audit["reference_action"] = np.where(
        reference_audit["reference_visual_rank"].isin(replacement_ranks),
        "replace",
        "lock",
    )
    reference_audit = reference_audit.sort_values("reference_visual_rank")
    policy: dict[str, object] = {
        "enabled": True,
        "reference_dir": str(reference_dir),
        "reference_selected_32": str(reference_path),
        "replacement_visual_ranks": list(replacement_ranks),
        "locked_candidate_ids": sorted(locked_ids),
        "rejected_candidate_ids": sorted(rejected_ids),
        "replacement_count": len(replacement_ranks),
        "field_readability_max_effective_bins": (
            FIELD_READABILITY_MAX_EFFECTIVE_BINS
        ),
        "minimum_exported_fields_per_side": (
            FIELD_READABILITY_MIN_EXPORTED_FIELDS
        ),
        "minimum_prominent_fields_per_side": (
            FIELD_READABILITY_MIN_PROMINENT_FIELDS
        ),
    }
    return work, pool, reference_audit, policy


def _profile_record(
    candidate_id: str,
    parent_rank: int,
    side: str,
    profile: Any,
    shared: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "parent_visual_rank": parent_rank,
            "side": side,
            "display_level_1based": level,
            "common_support_bins": int(profile.support_bins[level - 1]),
            "support_weight": float(profile.support_weight[level - 1]),
            "density_above_0.20": float(profile.density_above_floor[level - 1]),
            "weighted_density": float(profile.weighted_density[level - 1]),
            "profile_probability": float(profile.probability[level - 1]),
            "shared_profile_mass": float(shared[level - 1]),
        }
        for level in range(1, 8)
    ]


def characterize_level_preservation(
    frame: pd.DataFrame, arrays: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = np.asarray(build_fine_maze_crosswalk().display_level)
    if levels.shape != (45, 45):
        raise ValueError(f"unexpected fine-maze level grid shape: {levels.shape}")
    records: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        common = np.asarray(arrays["common_mask"][index], dtype=bool)
        profile_a = compute_level_activity_profile(
            arrays["normalized_a"][index], common, levels
        )
        profile_b = compute_level_activity_profile(
            arrays["normalized_b"][index], common, levels
        )
        profile_valid_a = bool(np.sum(profile_a.weighted_density) > 0.0)
        profile_valid_b = bool(np.sum(profile_b.weighted_density) > 0.0)
        profile_valid = profile_valid_a and profile_valid_b
        overlap = level_profile_overlap(profile_a.probability, profile_b.probability)
        shared = np.minimum(profile_a.probability, profile_b.probability)
        candidate_id = str(row.candidate_id)
        long_rows.extend(
            _profile_record(
                candidate_id,
                int(row.parent_visual_rank),
                "A",
                profile_a,
                shared,
            )
        )
        long_rows.extend(
            _profile_record(
                candidate_id,
                int(row.parent_visual_rank),
                "B",
                profile_b,
                shared,
            )
        )

        encoded_a = _int_set(row.encoded_level_set_a)
        encoded_b = _int_set(row.encoded_level_set_b)
        raw_a = compute_level_activity_profile(
            arrays["smoothed_c_raw_a"][index],
            common,
            levels,
            activity_floor=0.0,
        )
        raw_b = compute_level_activity_profile(
            arrays["smoothed_c_raw_b"][index],
            common,
            levels,
            activity_floor=0.0,
        )
        record: dict[str, object] = {
            "candidate_id": candidate_id,
            "level_profile_valid_a": profile_valid_a,
            "level_profile_valid_b": profile_valid_b,
            "level_profile_valid": profile_valid,
            "level_profile_overlap": overlap,
            "level_preservation_stratum": preservation_stratum(overlap),
            "encoded_level_jaccard": encoded_level_jaccard(encoded_a, encoded_b),
            "dominant_profile_level_a": (
                int(np.argmax(profile_a.probability) + 1) if profile_valid_a else None
            ),
            "dominant_profile_level_b": (
                int(np.argmax(profile_b.probability) + 1) if profile_valid_b else None
            ),
            "dominant_profile_level_match": bool(
                profile_valid
                and np.argmax(profile_a.probability) == np.argmax(profile_b.probability)
            ),
            "dominant_profile_level_shift": (
                int(np.argmax(profile_b.probability) - np.argmax(profile_a.probability))
                if profile_valid
                else None
            ),
            "ordered_level_earth_mover": (
                ordered_level_earth_mover(profile_a.probability, profile_b.probability)
                if profile_valid
                else np.nan
            ),
            "raw_c_level_profile_overlap": level_profile_overlap(
                raw_a.probability, raw_b.probability
            ),
        }
        record["raw_c_overlap_absolute_difference"] = abs(
            float(record["raw_c_level_profile_overlap"]) - overlap
        )
        for floor in PROFILE_SENSITIVITY_FLOORS:
            sensitivity_a = compute_level_activity_profile(
                arrays["normalized_a"][index],
                common,
                levels,
                activity_floor=floor,
            )
            sensitivity_b = compute_level_activity_profile(
                arrays["normalized_b"][index],
                common,
                levels,
                activity_floor=floor,
            )
            record[f"level_profile_overlap_floor_{floor:.2f}"] = level_profile_overlap(
                sensitivity_a.probability, sensitivity_b.probability
            )
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(long_rows)


def _has_location(tokens: Iterable[str], region: str) -> bool:
    suffix = f":{region}"
    return any(str(token).endswith(suffix) for token in tokens)


def add_selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    set_a = work["encoded_level_set_a"].map(_int_set)
    set_b = work["encoded_level_set_b"].map(_int_set)
    work["phen_exact_level_set"] = [
        first == second for first, second in zip(set_a, set_b, strict=True)
    ]
    work["phen_disjoint_level_set"] = [
        not bool(first & second) for first, second in zip(set_a, set_b, strict=True)
    ]
    work["low_level_preservation"] = work["level_profile_overlap"] < 0.30
    work["low_level_overlap_0.40"] = work["level_profile_overlap"] <= 0.40
    work["high_level_overlap_0.80"] = work["level_profile_overlap"] >= 0.80
    work["near_one_r45"] = work["place_r_45_display"] >= 0.98

    work["primary_run_a"] = work["dominant_run_tokens_a"].map(_primary_run_token)
    work["primary_run_b"] = work["dominant_run_tokens_b"].map(_primary_run_token)
    work["primary_run_pair_signature"] = [
        "|".join(
            sorted(
                (first, second),
                key=lambda token: (
                    token == "none",
                    int(token[1:]) if token.startswith("R") else token,
                ),
            )
        )
        for first, second in zip(
            work["primary_run_a"], work["primary_run_b"], strict=True
        )
    ]
    for signature in sorted(work["primary_run_pair_signature"].unique()):
        safe_signature = str(signature).replace("|", "_")
        work[f"motif_primary_run_pair_{safe_signature}"] = work[
            "primary_run_pair_signature"
        ].eq(signature)

    location_by_side: dict[str, list[list[str]]] = {}
    for side in ("a", "b"):
        location_by_side[side] = (
            work[f"field_location_tokens_{side}"].map(_json_tokens).tolist()
        )
        for region in ("entrance", "junction", "proximal", "middle", "distal"):
            work[f"has_{region}_{side}"] = [
                _has_location(tokens, region) for tokens in location_by_side[side]
            ]

    semantic_rows: list[dict[str, str]] = []
    for index, row in enumerate(work.itertuples(index=False)):
        field_buckets = [
            f"A:{row.selector_field_category_a}",
            f"B:{row.selector_field_category_b}",
        ]
        encoded_levels = [
            f"A:{row.encoded_level_token_a}",
            f"B:{row.encoded_level_token_b}",
        ]
        locations = [
            *(f"field:A:{token}" for token in location_by_side["a"][index]),
            *(f"field:B:{token}" for token in location_by_side["b"][index]),
            *(f"run:A:{token}" for token in _json_tokens(row.dominant_run_tokens_a)),
            *(f"run:B:{token}" for token in _json_tokens(row.dominant_run_tokens_b)),
        ]
        run_relations = [
            "same_run" if bool(row.phen_same_run) else "not_same_run",
            (
                "within_level_branch_switch"
                if bool(row.phen_within_level_branch_switch)
                else "no_within_level_branch_switch"
            ),
            "depth_shift" if bool(row.phen_depth_shift) else "no_depth_shift",
        ]
        phenotype = [
            f"A:{row.level_phenotype_a}",
            f"B:{row.level_phenotype_b}",
            f"transition:{row.phenotype_transition}",
            f"field_change:{row.field_count_direction}",
        ]
        semantic_rows.append(
            {
                "div_field_bucket": json.dumps(field_buckets),
                "div_encoded_level": json.dumps(encoded_levels),
                "div_location": json.dumps(sorted(set(locations))),
                "div_field_transition": json.dumps(
                    [str(row.field_count_transition_v3)]
                ),
                "div_level_relation": json.dumps(
                    [str(row.level_set_transition), str(row.level_relation)]
                ),
                "div_run_relation": json.dumps(run_relations),
                "div_phenotype": json.dumps(phenotype),
            }
        )
    work = pd.concat([work.reset_index(drop=True), pd.DataFrame(semantic_rows)], axis=1)
    work["visual_rank_quality"] = (
        100.0 - work["parent_visual_rank"].astype(float)
    ) / 99.0
    work["selection_utility_lambda_0"] = work["visual_rank_quality"]
    work["selection_utility_lambda_0.10"] = (
        0.90 * work["visual_rank_quality"] + 0.10 * work["level_profile_overlap"]
    )
    work["selection_utility_lambda_0.30"] = (1.0 - LEVEL_BIAS_WEIGHT) * work[
        "visual_rank_quality"
    ] + LEVEL_BIAS_WEIGHT * work["level_profile_overlap"]
    work["selection_utility_biased"] = work["selection_utility_lambda_0.30"]
    return work


def _supported_token_count(values: pd.Series, minimum_support: int = 2) -> int:
    support: Counter[str] = Counter()
    for value in values:
        support.update(set(_json_tokens(value)))
    return sum(count >= minimum_support for count in support.values())


def _primary_run_motif_bounds(
    frame: pd.DataFrame,
    *,
    fixed_col: str = "fixed_main_fig2b",
) -> dict[str, tuple[int | None, int | None]]:
    """Return anchor-safe caps for repeated strongest-field run pairs."""

    bounds: dict[str, tuple[int | None, int | None]] = {}
    for column in sorted(
        item for item in frame.columns if item.startswith("motif_primary_run_pair_")
    ):
        signature = column.removeprefix("motif_primary_run_pair_")
        first, second = signature.split("_", maxsplit=1)
        default_cap = 1 if first == second else 2
        support = int(frame[column].astype(bool).sum())
        fixed = (
            frame[fixed_col].astype(bool)
            if fixed_col in frame.columns
            else pd.Series(False, index=frame.index)
        )
        fixed_count = int((frame[column].astype(bool) & fixed).sum())
        cap = max(default_cap, fixed_count)
        if support > cap:
            bounds[column] = (None, cap)
    return bounds


def _selector_kwargs(
    frame: pd.DataFrame,
    *,
    refinement_score_col: str,
    score_col: str = "visual_rank_quality",
    fixed_col: str = "fixed_main_fig2b",
    conflict_pairs: Iterable[tuple[str, str]],
    solver_time_limit: float,
) -> dict[str, object]:
    level_columns: dict[str, tuple[str, ...]] = {}
    level_minima: dict[str, int] = {}
    for side in ("a", "b"):
        for level in range(1, 8):
            label = f"{side.upper()}:L{level}"
            level_columns[label] = (f"encodes_level_{level}_{side}",)
            level_minima[label] = 6 if level < 7 else 3

    category_bounds = {
        f"{side}:{category}": bounds
        for side in ("A", "B")
        for category, bounds in {
            "1": (2, 5),
            "2": (4, 8),
            "3": (3, 7),
            "4-5": (4, 9),
            "6-8": (4, 9),
            "9+": (4, 9),
        }.items()
    }
    flag_bounds: dict[str, tuple[int | None, int | None]] = {
        "phen_same_run": (13, 22),
        "phen_within_level_branch_switch": (7, 13),
        "phen_depth_shift": (4, 8),
        "phen_field_gain_loss": (12, 29),
        "phen_exact_level_set": (3, 5),
        "phen_disjoint_level_set": (3, 5),
        "low_level_overlap_0.40": (4, 8),
        "high_level_overlap_0.80": (4, None),
        "near_one_r45": (1, None),
    }
    flag_bounds.update(_primary_run_motif_bounds(frame, fixed_col=fixed_col))
    location_floors = {
        "entrance": 3,
        "junction": 3,
        "proximal": 10,
        "middle": 20,
        "distal": 8,
    }
    for side in ("a", "b"):
        for region, lower in location_floors.items():
            flag_bounds[f"has_{region}_{side}"] = (lower, None)

    semantic_columns = (
        "div_field_bucket",
        "div_encoded_level",
        "div_location",
        "div_field_transition",
        "div_level_relation",
        "div_run_relation",
        "div_phenotype",
    )
    family_weights = {
        column: 1.0 / max(1, _supported_token_count(frame[column]))
        for column in semantic_columns
    }
    # Bounded boolean flags are hard safeguards, not another diversity family.
    family_weights["phenotype_flag"] = 0.0
    return {
        "target_n": TARGET_N,
        "required_pairs": (),
        "min_per_pair": 0,
        "max_per_pair": TARGET_N,
        "required_correlation_bins": list(CORRELATION_BIN_LABELS),
        "min_per_correlation_bin": 2,
        "max_per_correlation_bin": 5,
        "min_low_extreme": 2,
        "min_high_extreme": 2,
        "correlation_col": "place_r_45_display",
        "score_col": score_col,
        "refinement_score_col": refinement_score_col,
        "fixed_col": fixed_col,
        "side_category_columns": (
            "selector_field_category_a",
            "selector_field_category_b",
        ),
        "side_category_bounds": category_bounds,
        "single_field_category": None,
        "max_single_field_side_fraction": None,
        "level_membership_columns": level_columns,
        "minimum_sides_per_level": level_minima,
        "flag_bounds": flag_bounds,
        "semantic_token_columns": semantic_columns,
        "phenotype_flag_columns": (),
        "token_family_weights": family_weights,
        "include_side_category_tokens": False,
        "include_level_tokens": False,
        "min_token_candidate_support": 2,
        "max_token_hits": 3,
        "token_decay": 0.50,
        "quality_retention": VISUAL_RETENTION,
        "diversity_retention": DIVERSITY_RETENTION,
        "conflict_pairs": tuple(conflict_pairs),
        "solver_time_limit": float(solver_time_limit),
    }


def select_shortlist(
    frame: pd.DataFrame,
    *,
    conflict_pairs: Iterable[tuple[str, str]],
    solver_time_limit: float,
    score_col: str = "visual_rank_quality",
    fixed_col: str = "fixed_main_fig2b",
    refinement_score_columns: tuple[str, str, str] = (
        "selection_utility_lambda_0",
        "selection_utility_lambda_0.10",
        "selection_utility_lambda_0.30",
    ),
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
]:
    conflict_pairs = tuple(conflict_pairs)
    baseline_score, ten_percent_score, biased_score = refinement_score_columns
    baseline, baseline_diagnostics = select_representation_diverse_candidates(
        frame,
        **_selector_kwargs(
            frame,
            refinement_score_col=baseline_score,
            score_col=score_col,
            fixed_col=fixed_col,
            conflict_pairs=conflict_pairs,
            solver_time_limit=solver_time_limit,
        ),
    )
    ten_percent, ten_percent_diagnostics = select_representation_diverse_candidates(
        frame,
        **_selector_kwargs(
            frame,
            refinement_score_col=ten_percent_score,
            score_col=score_col,
            fixed_col=fixed_col,
            conflict_pairs=conflict_pairs,
            solver_time_limit=solver_time_limit,
        ),
    )
    selected, diagnostics = select_representation_diverse_candidates(
        frame,
        **_selector_kwargs(
            frame,
            refinement_score_col=biased_score,
            score_col=score_col,
            fixed_col=fixed_col,
            conflict_pairs=conflict_pairs,
            solver_time_limit=solver_time_limit,
        ),
    )
    selected = selected.sort_values(
        ["visual_score_v3", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["visual_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    return (
        selected,
        diagnostics,
        baseline,
        baseline_diagnostics,
        ten_percent,
        ten_percent_diagnostics,
    )


def reconstruct_map_lookup(
    parent_frame: pd.DataFrame,
    fields: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    field_lookup = {
        (str(candidate_id), str(side).upper()): group.to_dict("records")
        for (candidate_id, side), group in fields.groupby(
            ["candidate_id", "side"], observed=True, sort=False
        )
    }
    lookup: dict[str, dict[str, Any]] = {}
    for index, candidate_id in enumerate(parent_frame["candidate_id"].astype(str)):
        lookup[candidate_id] = {
            "normalized_a": np.asarray(arrays["normalized_a"][index]),
            "normalized_b": np.asarray(arrays["normalized_b"][index]),
            "map_a": np.asarray(arrays["smoothed_c_raw_a"][index]),
            "map_b": np.asarray(arrays["smoothed_c_raw_b"][index]),
            "mask_a": np.asarray(arrays["occupancy_mask_a"][index], dtype=bool),
            "mask_b": np.asarray(arrays["occupancy_mask_b"][index], dtype=bool),
            "common_mask": np.asarray(arrays["common_mask"][index], dtype=bool),
            "field_labels_a": np.asarray(
                arrays["display_field_labels_a"][index], dtype=np.int16
            ),
            "field_labels_b": np.asarray(
                arrays["display_field_labels_b"][index], dtype=np.int16
            ),
            "q99_scale_a": float(arrays["q99_scale_a"][index]),
            "q99_scale_b": float(arrays["q99_scale_b"][index]),
            "fields_a": field_lookup.get((candidate_id, "A"), []),
            "fields_b": field_lookup.get((candidate_id, "B"), []),
        }
    return lookup


def export_selected_maps(
    selected: pd.DataFrame,
    parent_arrays: Mapping[str, np.ndarray],
    parent_frame: pd.DataFrame,
    parent_dir: Path,
    output_dir: Path,
) -> list[str]:
    parent_index = {
        str(candidate_id): index
        for index, candidate_id in enumerate(parent_frame["candidate_id"])
    }
    indices = np.asarray(
        [parent_index[str(candidate_id)] for candidate_id in selected["candidate_id"]],
        dtype=int,
    )
    npz_path = output_dir / "selected_maps_fields_45_shortlist32.npz"
    np.savez_compressed(
        npz_path,
        candidate_id=np.asarray(selected["candidate_id"].astype(str), dtype="U"),
        visual_rank=selected["visual_rank"].to_numpy(dtype=int),
        parent_visual_rank=selected["parent_visual_rank"].to_numpy(dtype=int),
        normalized_a=parent_arrays["normalized_a"][indices],
        normalized_b=parent_arrays["normalized_b"][indices],
        smoothed_c_raw_a=parent_arrays["smoothed_c_raw_a"][indices],
        smoothed_c_raw_b=parent_arrays["smoothed_c_raw_b"][indices],
        occupancy_mask_a=parent_arrays["occupancy_mask_a"][indices],
        occupancy_mask_b=parent_arrays["occupancy_mask_b"][indices],
        common_mask=parent_arrays["common_mask"][indices],
        display_field_labels_a=parent_arrays["display_field_labels_a"][indices],
        display_field_labels_b=parent_arrays["display_field_labels_b"][indices],
        q99_scale_a=parent_arrays["q99_scale_a"][indices],
        q99_scale_b=parent_arrays["q99_scale_b"][indices],
    )

    topology = pd.read_csv(parent_dir / "selected_map_bins_topology_45_v3.csv.gz")
    topology = topology.loc[
        topology["candidate_id"].astype(str).isin(set(selected["candidate_id"]))
    ].copy()
    rank_lookup = selected.set_index("candidate_id")["visual_rank"].to_dict()
    parent_rank_lookup = selected.set_index("candidate_id")[
        "parent_visual_rank"
    ].to_dict()
    topology["parent_visual_rank"] = topology["candidate_id"].map(parent_rank_lookup)
    topology["visual_rank"] = topology["candidate_id"].map(rank_lookup)
    topology = topology.sort_values(
        ["visual_rank", "side", "bin_y", "bin_x"], kind="mergesort"
    )
    topology_path = output_dir / "selected_map_bins_topology_45_shortlist32.csv.gz"
    v3._export_frame(topology, topology_path, compression="gzip")
    return [npz_path.name, topology_path.name]


def build_compact_grid_key(selected: pd.DataFrame) -> pd.DataFrame:
    """Return the author-facing lookup for the 8 x 8 map grid."""

    if len(selected) != TARGET_N:
        raise ValueError(f"compact grid requires exactly {TARGET_N} pairs")
    ranks = selected["visual_rank"].astype(int).tolist()
    if ranks != list(range(1, TARGET_N + 1)):
        raise ValueError("compact grid requires visual ranks 1 through 32 in order")

    key = selected.copy()
    zero_based = key["visual_rank"].astype(int) - 1
    key.insert(1, "grid_row_1based", zero_based // 4 + 1)
    key.insert(2, "pair_column_1based", zero_based % 4 + 1)
    key.insert(3, "map_column_a_1based", 2 * key["pair_column_1based"] - 1)
    key.insert(4, "map_column_b_1based", 2 * key["pair_column_1based"])
    preferred = [
        "visual_rank",
        "grid_row_1based",
        "pair_column_1based",
        "map_column_a_1based",
        "map_column_b_1based",
        "candidate_id",
        "fixed_main_fig2b",
        "parent_visual_rank",
        "visual_score_v3",
        "level_profile_overlap",
        "place_r_45_display",
        "correlation_bin",
        "pair_key",
        "group",
        "animal_id",
        "src_session",
        "dst_session",
        "train_id",
        "test_id",
        "node_a",
        "node_b",
        "field_count_a",
        "field_count_b",
        "encoded_level_token_a",
        "encoded_level_token_b",
        "field_location_tokens_a",
        "field_location_tokens_b",
        "phen_same_run",
        "phen_within_level_branch_switch",
        "phen_depth_shift",
        "phen_field_gain_loss",
        "phen_exact_level_set",
        "phen_disjoint_level_set",
    ]
    missing = [column for column in preferred if column not in key.columns]
    if missing:
        raise ValueError(f"compact grid key is missing required columns: {missing}")
    return key.loc[:, preferred]


def _compact_grid_geometry(overlay_fields: bool) -> dict[str, float]:
    return {
        "left": 0.006,
        "right": 0.994,
        "top": 0.997,
        "bottom": 0.075 if overlay_fields else 0.055,
        "pair_wspace": 0.040,
        "within_pair_wspace": 0.004,
        "row_hspace": 0.040,
        "header_ratio": 0.060,
        "header_map_hspace": 0.010,
    }


def _format_place_map_correlation(value: float) -> str:
    return f"r = {float(value):.2f}".replace("-", "\N{MINUS SIGN}")


def build_compact_grid_figure(
    selected: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    *,
    overlay_fields: bool,
) -> plt.Figure:
    """Build one ED6-ready page with 32 adjacent A/B pairs and 64 maps."""

    grid_key = build_compact_grid_key(selected)
    missing_maps = sorted(set(grid_key["candidate_id"].astype(str)) - set(maps))
    if missing_maps:
        raise ValueError(
            f"maps are missing for compact-grid candidates: {missing_maps}"
        )

    geometry = _compact_grid_geometry(overlay_fields)
    figure = plt.figure(figsize=(8.0, 9.05), facecolor="white")
    outer = figure.add_gridspec(
        8,
        4,
        left=geometry["left"],
        right=geometry["right"],
        top=geometry["top"],
        bottom=geometry["bottom"],
        wspace=geometry["pair_wspace"],
        hspace=geometry["row_hspace"],
    )
    map_axes: list[plt.Axes] = []
    pair_card_patches: list[Rectangle] = []
    pair_header_texts: list[matplotlib.text.Text] = []
    for position, row in enumerate(selected.itertuples(index=False)):
        pair_row, pair_column = divmod(position, 4)
        slot = outer[pair_row, pair_column]
        pair_grid = slot.subgridspec(
            2,
            2,
            height_ratios=[geometry["header_ratio"], 1.0],
            hspace=geometry["header_map_hspace"],
            wspace=geometry["within_pair_wspace"],
        )
        values = maps[str(row.candidate_id)]
        current_map_axes: list[plt.Axes] = []
        for within_pair_column, side in enumerate(("a", "b")):
            axis = figure.add_subplot(pair_grid[1, within_pair_column])
            v3._draw_map(axis, np.asarray(values[f"normalized_{side}"]))
            if overlay_fields:
                v3._draw_fields(
                    axis,
                    np.asarray(values[f"field_labels_{side}"]),
                    values[f"fields_{side}"],
                )
            axis.text(
                0.022,
                0.978,
                side.upper(),
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=5.2,
                weight="bold",
                clip_on=True,
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.52,
                    "pad": 0.28,
                },
            )
            axis.set_gid(f"pair-{int(row.visual_rank):02d}-{side.upper()}")
            map_axes.append(axis)
            current_map_axes.append(axis)

        slot_box = slot.get_position(figure)
        map_top = max(axis.get_position().y1 for axis in current_map_axes)
        header_y = map_top + 0.50 * (slot_box.y1 - map_top)
        pair_rank = int(row.visual_rank)
        pair_header_texts.append(
            figure.text(
                slot_box.x0 + 0.025 * slot_box.width,
                header_y,
                str(pair_rank),
                ha="left",
                va="center",
                fontsize=5.5,
                weight="bold",
                color="#303030",
            )
        )
        pair_header_texts.append(
            figure.text(
                0.5 * (slot_box.x0 + slot_box.x1),
                header_y,
                _format_place_map_correlation(row.place_r_45_display),
                ha="center",
                va="center",
                fontsize=5.5,
                color="#303030",
            )
        )
        is_main_figure_pair = bool(row.fixed_main_fig2b)
        edge_color = (
            MAIN_FIG_PAIR_COLOR if is_main_figure_pair else PAIR_CARD_EDGE_COLOR
        )
        edge_width = (
            MAIN_FIG_PAIR_LINEWIDTH if is_main_figure_pair else PAIR_CARD_LINEWIDTH
        )
        card = Rectangle(
            (slot_box.x0, slot_box.y0),
            slot_box.width,
            slot_box.height,
            transform=figure.transFigure,
            facecolor="none",
            edgecolor=edge_color,
            linewidth=edge_width,
            joinstyle="miter",
            clip_on=False,
            zorder=20,
        )
        card.set_gid(f"pair-card-{pair_rank:02d}")
        figure.add_artist(card)
        pair_card_patches.append(card)
        if is_main_figure_pair:
            anchor_order = int(round(float(row.anchor_order)))
            pair_header_texts.append(
                figure.text(
                    slot_box.x1 - 0.025 * slot_box.width,
                    header_y,
                    f"Fig. 2b\N{EN DASH}{anchor_order}",
                    ha="right",
                    va="center",
                    fontsize=4.4,
                    weight="bold",
                    color="white",
                    bbox={
                        "facecolor": MAIN_FIG_PAIR_COLOR,
                        "edgecolor": "none",
                        "pad": 0.45,
                    },
                )
            )

    colorbar_axis = figure.add_axes([0.310, 0.020, 0.280, 0.0075])
    colorbar_axis.set_gid("shared-q99-colorbar")
    colorbar = figure.colorbar(
        matplotlib.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=v3._display_cmap()),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0.0, 0.5, 1.0],
    )
    colorbar.set_label("Denoised C (each map: q99 = 1)", fontsize=5.0, labelpad=0.8)
    colorbar.ax.tick_params(labelsize=4.4, length=1.4, pad=0.7)

    figure.text(
        0.012,
        0.024,
        "r: place-map correlation",
        ha="left",
        va="center",
        fontsize=4.8,
        color="#303030",
    )
    main_figure_key = Rectangle(
        (0.748, 0.018),
        0.018,
        0.012,
        transform=figure.transFigure,
        facecolor="none",
        edgecolor=MAIN_FIG_PAIR_COLOR,
        linewidth=MAIN_FIG_PAIR_LINEWIDTH,
        clip_on=False,
    )
    main_figure_key.set_gid("main-figure-pair-key")
    figure.add_artist(main_figure_key)
    figure.text(
        0.772,
        0.024,
        "shown in Fig. 2b",
        ha="left",
        va="center",
        fontsize=4.8,
        color="#303030",
    )

    if overlay_fields:
        handles = [
            plt.Line2D([0], [0], color=color, lw=1.4, label=f"L{level}")
            for level, color in enumerate(v3.LEVEL_COLORS, start=1)
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=7,
            frameon=False,
            fontsize=4.0,
            handlelength=1.2,
            handletextpad=0.25,
            columnspacing=0.65,
        )

    # These attributes make structural QA possible without pixel inference.
    figure._ed06_map_axes = tuple(map_axes)  # type: ignore[attr-defined]
    figure._ed06_pair_card_patches = tuple(pair_card_patches)  # type: ignore[attr-defined]
    figure._ed06_pair_header_texts = tuple(pair_header_texts)  # type: ignore[attr-defined]
    figure._ed06_colorbar_axis = colorbar_axis  # type: ignore[attr-defined]
    figure._ed06_grid_shape = (8, 8)  # type: ignore[attr-defined]
    return figure


def render_compact_grid(
    selected: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    overlay_fields: bool,
) -> list[str]:
    stem = (
        "ed06_compact_32_grid_8x8_field_overlay_audit"
        if overlay_fields
        else "ed06_compact_32_grid_8x8"
    )
    figure = build_compact_grid_figure(selected, maps, overlay_fields=overlay_fields)
    paths: list[str] = []
    try:
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            for extension in ("png", "svg", "pdf"):
                path = output_dir / f"{stem}.{extension}"
                kwargs = {"dpi": 400} if extension == "png" else {}
                figure.savefig(path, facecolor="white", **kwargs)
                paths.append(path.name)
    finally:
        plt.close(figure)
    return paths


def _location_token_count(frame: pd.DataFrame, side: str) -> int:
    tokens: set[str] = set()
    for value in frame[f"field_location_tokens_{side}"].dropna():
        tokens.update(_json_tokens(value))
    return len(tokens)


def build_coverage_table(parent: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(
        dimension: str, label: str, parent_count: int, shortlist_count: int
    ) -> None:
        rows.append(
            {
                "dimension": dimension,
                "label": label,
                "parent_100_count": int(parent_count),
                "shortlist_32_count": int(shortlist_count),
            }
        )

    for pair in sorted(parent["pair_key"].astype(str).unique()):
        add(
            "session_pair",
            pair,
            int(parent["pair_key"].astype(str).eq(pair).sum()),
            int(selected["pair_key"].astype(str).eq(pair).sum()),
        )
    for label in CORRELATION_BIN_LABELS:
        add(
            "r45_bin",
            label,
            int(parent["correlation_bin"].astype(str).eq(label).sum()),
            int(selected["correlation_bin"].astype(str).eq(label).sum()),
        )
    for side in ("a", "b"):
        column = f"selector_field_category_{side}"
        for category in ("1", "2", "3", "4-5", "6-8", "9+"):
            label = f"{side.upper()}:{category}"
            add(
                "field_count_bucket",
                label,
                int(parent[column].astype(str).eq(label).sum()),
                int(selected[column].astype(str).eq(label).sum()),
            )
        for level in range(1, 8):
            level_column = f"encodes_level_{level}_{side}"
            add(
                "encoded_level",
                f"{side.upper()}:L{level}",
                int(parent[level_column].astype(bool).sum()),
                int(selected[level_column].astype(bool).sum()),
            )
        for region in ("entrance", "junction", "proximal", "middle", "distal"):
            region_column = f"has_{region}_{side}"
            add(
                "within_level_region",
                f"{side.upper()}:{region}",
                int(parent[region_column].astype(bool).sum()),
                int(selected[region_column].astype(bool).sum()),
            )
        add(
            "unique_level_branch_location_tokens",
            side.upper(),
            _location_token_count(parent, side),
            _location_token_count(selected, side),
        )
    for column in (
        "phen_same_run",
        "phen_within_level_branch_switch",
        "phen_depth_shift",
        "phen_field_gain_loss",
        "phen_exact_level_set",
        "phen_disjoint_level_set",
    ):
        add(
            "representation_transition",
            column,
            int(parent[column].astype(bool).sum()),
            int(selected[column].astype(bool).sum()),
        )
    for label in (
        "low_<0.30",
        "intermediate_0.30-<0.75",
        "high_>=0.75",
    ):
        add(
            "level_profile_overlap_stratum",
            label,
            int(parent["level_preservation_stratum"].astype(str).eq(label).sum()),
            int(selected["level_preservation_stratum"].astype(str).eq(label).sum()),
        )
    return pd.DataFrame(rows)


def build_constraint_table(diagnostics: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for detail in diagnostics["constraints"]["details"]:
        lower = detail["lower"]
        upper = detail["upper"]
        achieved = float(detail["achieved"])
        passed = (lower is None or achieved >= float(lower) - 1e-7) and (
            upper is None or achieved <= float(upper) + 1e-7
        )
        rows.append({**detail, "pass": bool(passed)})
    return pd.DataFrame(rows)


def build_bias_ablation(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    selected_diagnostics: Mapping[str, Any],
    baseline: pd.DataFrame,
    baseline_diagnostics: Mapping[str, Any],
    ten_percent: pd.DataFrame,
    ten_percent_diagnostics: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_baseline = set(baseline["candidate_id"].astype(str))
    id_ten_percent = set(ten_percent["candidate_id"].astype(str))
    id_selected = set(selected["candidate_id"].astype(str))
    membership = frame[
        [
            "candidate_id",
            "parent_visual_rank",
            "visual_score_v3",
            "level_profile_overlap",
            "place_r_45_display",
        ]
    ].copy()
    membership["selected_lambda_0"] = membership["candidate_id"].isin(id_baseline)
    membership["selected_lambda_0.10"] = membership["candidate_id"].isin(id_ten_percent)
    membership["selected_lambda_0.30"] = membership["candidate_id"].isin(id_selected)
    membership["change"] = np.select(
        [
            membership["selected_lambda_0.30"] & ~membership["selected_lambda_0.10"],
            membership["selected_lambda_0.10"] & ~membership["selected_lambda_0.30"],
        ],
        ["entered_at_30pct_vs_10pct", "exited_at_30pct_vs_10pct"],
        default="unchanged",
    )

    summary_rows: list[dict[str, object]] = []
    for label, use, diagnostics in (
        ("lambda_0", baseline, baseline_diagnostics),
        ("lambda_0.10", ten_percent, ten_percent_diagnostics),
        ("lambda_0.30", selected, selected_diagnostics),
    ):
        summary_rows.append(
            {
                "selection": label,
                "n": len(use),
                "raw_visual_score_sum": float(use["visual_score_v3"].sum()),
                "raw_visual_score_mean": float(use["visual_score_v3"].mean()),
                "visual_rank_quality_sum": float(use["visual_rank_quality"].sum()),
                "mean_parent_visual_rank": float(use["parent_visual_rank"].mean()),
                "mean_level_profile_overlap": float(
                    use["level_profile_overlap"].mean()
                ),
                "median_level_profile_overlap": float(
                    use["level_profile_overlap"].median()
                ),
                "high_overlap_count": int((use["level_profile_overlap"] >= 0.75).sum()),
                "low_overlap_count": int((use["level_profile_overlap"] < 0.30).sum()),
                "diversity_score": float(diagnostics["diversity_achieved"]),
                "covered_diversity_tokens": int(
                    diagnostics["covered_diversity_tokens"]
                ),
            }
        )
    return pd.DataFrame(summary_rows), membership


def validate_shortlist(
    parent: pd.DataFrame,
    selected: pd.DataFrame,
    ten_percent: pd.DataFrame,
    grid_key: pd.DataFrame,
    selected_fields: pd.DataFrame,
    selected_level_profiles: pd.DataFrame,
    selected_topology: pd.DataFrame,
    selected_morphology: pd.DataFrame,
    morphology_components: pd.DataFrame,
    conflicts: pd.DataFrame,
    constraints: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
    baseline_diagnostics: Mapping[str, Any],
    ten_percent_diagnostics: Mapping[str, Any],
    render_paths: Iterable[str],
    replacement_policy: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    selected_ids = set(selected["candidate_id"].astype(str))
    ten_percent_ids = set(ten_percent["candidate_id"].astype(str))
    expected_ids = selected["candidate_id"].astype(str).tolist()
    npz_ids = np.asarray(arrays["candidate_id"]).astype(str).tolist()
    npz_visual_ranks = np.asarray(arrays["visual_rank"], dtype=int).tolist()
    npz_parent_ranks = np.asarray(arrays["parent_visual_rank"], dtype=int).tolist()
    field_group_counts = selected_fields.groupby(
        ["candidate_id", "side"], observed=True
    ).size()
    per_candidate_field_counts_match = True
    field_label_ids_match = True
    for index, row in enumerate(selected.itertuples(index=False)):
        for side in ("a", "b"):
            expected = int(getattr(row, f"field_count_{side}"))
            exported = int(
                field_group_counts.get((str(row.candidate_id), side.upper()), 0)
            )
            labels = np.asarray(arrays[f"display_field_labels_{side}"][index])
            positive = sorted(set(labels[labels > 0].astype(int).tolist()))
            per_candidate_field_counts_match &= exported == expected
            field_label_ids_match &= positive == list(range(1, expected + 1))

    node_values = [
        str(value)
        for row in selected[["node_a", "node_b"]].itertuples(index=False)
        for value in row
    ]
    topology_counts = selected_topology.groupby(
        ["candidate_id", "side"], observed=True
    ).size()
    topology_unique_bins = (
        selected_topology.drop_duplicates(["candidate_id", "side", "bin_y", "bin_x"])
        .groupby(["candidate_id", "side"], observed=True)
        .size()
    )
    topology_rank_lookup = selected.set_index("candidate_id")["visual_rank"].to_dict()
    topology_identity_ok = (
        len(selected_topology) == TARGET_N * 2 * 45 * 45
        and set(selected_topology["candidate_id"].astype(str)) == selected_ids
        and topology_counts.eq(45 * 45).all()
        and topology_unique_bins.eq(45 * 45).all()
        and selected_topology["visual_rank"]
        .astype(int)
        .eq(selected_topology["candidate_id"].map(topology_rank_lookup).astype(int))
        .all()
    )
    expected_rows = ((selected["visual_rank"].astype(int) - 1) // 4 + 1).tolist()
    expected_pair_columns = ((selected["visual_rank"].astype(int) - 1) % 4 + 1).tolist()
    expected_map_columns_a = [2 * value - 1 for value in expected_pair_columns]
    expected_map_columns_b = [2 * value for value in expected_pair_columns]
    grid_key_ok = bool(
        len(grid_key) == TARGET_N
        and grid_key["candidate_id"].astype(str).tolist() == expected_ids
        and grid_key["visual_rank"].astype(int).tolist() == list(range(1, TARGET_N + 1))
        and grid_key["grid_row_1based"].astype(int).tolist() == expected_rows
        and grid_key["pair_column_1based"].astype(int).tolist() == expected_pair_columns
        and grid_key["map_column_a_1based"].astype(int).tolist()
        == expected_map_columns_a
        and grid_key["map_column_b_1based"].astype(int).tolist()
        == expected_map_columns_b
        and all(
            right == left + 1
            for left, right in zip(
                grid_key["map_column_a_1based"].astype(int),
                grid_key["map_column_b_1based"].astype(int),
                strict=True,
            )
        )
    )
    render_names = set(render_paths)
    expected_render_names = {
        f"ed06_compact_32_grid_8x8.{extension}" for extension in ("png", "svg", "pdf")
    } | {
        f"ed06_compact_32_grid_8x8_field_overlay_audit.{extension}"
        for extension in ("png", "svg", "pdf")
    }
    motif_counts = selected["primary_run_pair_signature"].astype(str).value_counts()
    same_run_counts = motif_counts[
        [len(set(signature.split("|"))) == 1 for signature in motif_counts.index]
    ]
    mixed_run_counts = motif_counts[
        [len(set(signature.split("|"))) > 1 for signature in motif_counts.index]
    ]
    selected_parent_ranks = set(selected["parent_visual_rank"].astype(int))
    replacement_policy = dict(replacement_policy or {"enabled": False})
    correction_enabled = bool(replacement_policy.get("enabled", False))
    locked_ids = set(replacement_policy.get("locked_candidate_ids", ()))
    rejected_ids = set(replacement_policy.get("rejected_candidate_ids", ()))
    replacement_ids = selected_ids - locked_ids if correction_enabled else set()
    replacement_rows = selected.loc[
        selected["candidate_id"].astype(str).isin(replacement_ids)
    ]
    rejected_rows = parent.loc[
        parent["candidate_id"].astype(str).isin(rejected_ids)
    ]
    checks: dict[str, bool] = {
        "exactly_32_unique_candidates": (
            len(selected) == TARGET_N and selected["candidate_id"].nunique() == TARGET_N
        ),
        "strict_visual_quality_order": bool(
            selected["visual_score_v3"].is_monotonic_decreasing
            and selected["visual_rank"].tolist() == list(range(1, TARGET_N + 1))
        ),
        "all_10_r45_bins": selected["correlation_bin"].nunique() == 10,
        "r45_low_extreme": int((selected["place_r_45_display"] <= -0.25).sum()) >= 2,
        "r45_high_extreme": int((selected["place_r_45_display"] >= 0.90).sum()) >= 2,
        "r45_near_one": int((selected["place_r_45_display"] >= 0.98).sum()) >= 1,
        "all_four_fig2b_anchors": int(selected["fixed_main_fig2b"].sum()) == 4,
        "all_pair_correlations_finite": bool(
            np.isfinite(selected["place_r_45_display"].astype(float)).all()
        ),
        "fig2b_anchor_orders_are_one_through_four": sorted(
            selected.loc[
                selected["fixed_main_fig2b"].astype(bool), "anchor_order"
            ].astype(int)
        )
        == [1, 2, 3, 4],
        "all_hard_constraints": bool(constraints["pass"].all()),
        "quality_reference_proven_optimal": bool(
            diagnostics["quality_reference_proven_optimal"]
        ),
        "diversity_reference_proven_optimal": bool(
            diagnostics["diversity_reference_proven_optimal"]
        ),
        "final_refinement_proven_optimal": bool(
            diagnostics["final_refinement_proven_optimal"]
        ),
        "baseline_all_solver_stages_proven_optimal": all(
            bool(baseline_diagnostics[key])
            for key in (
                "quality_reference_proven_optimal",
                "diversity_reference_proven_optimal",
                "final_refinement_proven_optimal",
            )
        ),
        "ten_percent_all_solver_stages_proven_optimal": all(
            bool(ten_percent_diagnostics[key])
            for key in (
                "quality_reference_proven_optimal",
                "diversity_reference_proven_optimal",
                "final_refinement_proven_optimal",
            )
        ),
        "thirty_percent_bias_changes_membership_vs_ten_percent": selected_ids
        != ten_percent_ids,
        "thirty_percent_bias_improves_mean_overlap_vs_ten_percent": float(
            selected["level_profile_overlap"].mean()
        )
        > float(ten_percent["level_profile_overlap"].mean()),
        "no_duplicate_neuron_nodes": len(node_values) == len(set(node_values)),
        "no_individual_map_morphology_conflicts": len(conflicts) == 0,
        "same_primary_run_pair_motif_cap": bool(
            same_run_counts.empty or same_run_counts.max() <= 1
        ),
        "mixed_primary_run_pair_motif_cap": bool(
            mixed_run_counts.empty or mixed_run_counts.max() <= 2
        ),
        "user_reported_redundancy_clusters_separated": (
            len(selected_parent_ranks & {5, 8, 9, 22}) <= 1
            and len(selected_parent_ranks & {21, 51}) <= 1
        ),
        "low_overlap_examples_retained": int(
            (selected["level_profile_overlap"] <= 0.40).sum()
        )
        >= 4,
        "compact_grid_key_order_and_adjacency": grid_key_ok,
        "selected_level_profile_row_count": len(selected_level_profiles)
        == TARGET_N * 2 * 7,
        "compact_render_bundle_complete_or_skipped": not render_names
        or expected_render_names.issubset(render_names),
        "selected_fields_match_counts": len(selected_fields)
        == int(selected["field_count_a"].sum() + selected["field_count_b"].sum()),
        "per_candidate_side_field_counts_match": per_candidate_field_counts_match,
        "field_label_ids_match_descriptors": field_label_ids_match,
        "topology_identity_and_row_counts": bool(topology_identity_ok),
        "parent_subset_only": selected_ids.issubset(
            set(parent["candidate_id"].astype(str))
        ),
        "reference_accepted_28_locked_exactly": (
            not correction_enabled or locked_ids.issubset(selected_ids)
        ),
        "reference_rejected_candidates_absent": (
            not correction_enabled or not bool(selected_ids & rejected_ids)
        ),
        "exact_replacement_count": (
            not correction_enabled
            or len(replacement_ids)
            == int(replacement_policy.get("replacement_count", 0))
        ),
        "all_replacements_pass_field_readability_screen": (
            not correction_enabled
            or bool(
                replacement_rows["field_readability_replacement_eligible"]
                .astype(bool)
                .all()
            )
        ),
        "replacements_have_smaller_effective_fields_than_rejected": (
            not correction_enabled
            or float(
                replacement_rows[
                    "pair_field_readability_max_effective_bins"
                ].max()
            )
            < float(
                rejected_rows[
                    "pair_field_readability_max_effective_bins"
                ].min()
            )
        ),
        "replacements_have_higher_readability_than_rejected": (
            not correction_enabled
            or float(replacement_rows["pair_field_readability_score"].min())
            > float(rejected_rows["pair_field_readability_score"].max())
        ),
        "npz_candidate_order": npz_ids == expected_ids,
        "npz_visual_rank_order": npz_visual_ranks == list(range(1, TARGET_N + 1)),
        "npz_parent_rank_order": npz_parent_ranks
        == selected["parent_visual_rank"].astype(int).tolist(),
        "subset_npz_shapes": all(
            np.asarray(arrays[key]).shape == (TARGET_N, 45, 45)
            for key in (
                "normalized_a",
                "normalized_b",
                "smoothed_c_raw_a",
                "smoothed_c_raw_b",
                "occupancy_mask_a",
                "occupancy_mask_b",
                "common_mask",
                "display_field_labels_a",
                "display_field_labels_b",
            )
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "checks": checks,
        "all_pass": not failures,
        "failures": failures,
        "selected_count": len(selected),
        "session_pair_count": int(selected["pair_key"].nunique()),
        "max_examples_per_session_pair": int(
            selected["pair_key"].astype(str).value_counts().max()
        ),
        "r45_min": float(selected["place_r_45_display"].min()),
        "r45_max": float(selected["place_r_45_display"].max()),
        "multi_field_side_fraction": float(
            (
                (selected["field_count_a"] > 1).sum()
                + (selected["field_count_b"] > 1).sum()
            )
            / (2 * len(selected))
        ),
        "mean_level_profile_overlap": float(selected["level_profile_overlap"].mean()),
        "ten_percent_mean_level_profile_overlap": float(
            ten_percent["level_profile_overlap"].mean()
        ),
        "thirty_vs_ten_membership_swap_count": int(
            len(selected_ids.symmetric_difference(ten_percent_ids)) // 2
        ),
        "median_level_profile_overlap": float(
            selected["level_profile_overlap"].median()
        ),
        "replacement_parent_visual_ranks": sorted(
            replacement_rows["parent_visual_rank"].astype(int).tolist()
        ),
        "replacement_candidate_ids": sorted(
            replacement_rows["candidate_id"].astype(str).tolist()
        ),
        "replacement_mean_level_profile_overlap": (
            float(replacement_rows["level_profile_overlap"].mean())
            if correction_enabled
            else None
        ),
        "rejected_mean_level_profile_overlap": (
            float(rejected_rows["level_profile_overlap"].mean())
            if correction_enabled
            else None
        ),
        "replacement_max_effective_field_bins": (
            float(
                replacement_rows[
                    "pair_field_readability_max_effective_bins"
                ].max()
            )
            if correction_enabled
            else None
        ),
        "rejected_min_effective_field_bins": (
            float(
                rejected_rows[
                    "pair_field_readability_max_effective_bins"
                ].min()
            )
            if correction_enabled
            else None
        ),
        "location_tokens_a": _location_token_count(selected, "a"),
        "location_tokens_b": _location_token_count(selected, "b"),
        "morphology_parent_edge_count": int(
            morphology_components["direct_conflict_neighbors"]
            .map(_json_tokens)
            .map(len)
            .sum()
            // 2
        )
        if not morphology_components.empty
        else 0,
        "morphology_component_count": int(
            morphology_components["component_id"].nunique()
        )
        if not morphology_components.empty
        else 0,
        "max_selected_pair_blurred_cosine": float(
            selected_morphology["max_blurred_cosine"].max()
        ),
        "max_same_primary_run_pair_count": int(same_run_counts.max())
        if not same_run_counts.empty
        else 0,
        "max_mixed_primary_run_pair_count": int(mixed_run_counts.max())
        if not mixed_run_counts.empty
        else 0,
    }


def write_readme(
    output_dir: Path,
    selected: pd.DataFrame,
    ablation: pd.DataFrame,
    validation: Mapping[str, Any],
    render_paths: Iterable[str],
    replacement_policy: Mapping[str, Any] | None = None,
) -> None:
    baseline = ablation.set_index("selection").loc["lambda_0"]
    ten_percent = ablation.set_index("selection").loc["lambda_0.10"]
    biased = ablation.set_index("selection").loc["lambda_0.30"]
    render_paths = list(render_paths)
    replacement_policy = dict(replacement_policy or {"enabled": False})
    correction_note = ""
    if replacement_policy.get("enabled", False):
        correction_note = f"""
## Reviewed-shortlist correction

The previous display ranks {replacement_policy['replacement_visual_ranks']} were
removed because their fields were diffuse at the final card size. The other
{TARGET_N - int(replacement_policy['replacement_count'])} candidate IDs are
locked exactly. New rows must contain at least
{FIELD_READABILITY_MIN_EXPORTED_FIELDS} exported and
{FIELD_READABILITY_MIN_PROMINENT_FIELDS} visibly prominent fields on each side,
and no single exported field may have more than
{FIELD_READABILITY_MAX_EFFECTIVE_BINS:.0f} effective activity bins. Effective
field size is `(sum activity)^2 / sum(activity^2)`, so it detects long diffuse
fields without rewarding a neuron merely for having fewer fields. The final
replacement utility is `0.70 x field readability + 0.30 x level-profile
overlap`; the complete accepted/rejected/replacement crosswalk is exported.

The selected replacement parent ranks are
{validation['replacement_parent_visual_ranks']}. Their maximum effective field
size is {validation['replacement_max_effective_field_bins']:.2f} bins, below the
smallest rejected value of {validation['rejected_min_effective_field_bins']:.2f};
replacement mean level overlap is
{validation['replacement_mean_level_profile_overlap']:.3f}, versus
{validation['rejected_mean_level_profile_overlap']:.3f} for the removed rows.
"""
    render_note = (
        "- `ed06_compact_32_grid_8x8.{png,svg,pdf}`: primary compact plate; "
        "eight rows, four adjacent A/B pairs per row.\n"
        "- `ed06_compact_32_grid_8x8_field_overlay_audit.{png,svg,pdf}`: "
        "the same order with field contours and level labels for audit only."
        if render_paths
        else "- Compact plates and individual previews were intentionally skipped for this run."
    )
    text = f"""# ED6 compact 32-pair place-map shortlist (30% level-overlap bias)

This is a review shortlist derived only from the frozen, validated 100-example
v3 gallery. It does not rerun raw sessions or replace the current Extended Data
Figure 6 until the examples are chosen by the author.

## How the 32 were chosen

The inclusion problem first maximizes the existing visual rank, then retains at
least {100 * VISUAL_RETENTION:.1f}% of that constrained optimum while maximizing
family-balanced semantic diversity, and finally retains at least
{100 * DIVERSITY_RETENTION:.1f}% of the semantic optimum while maximizing
`0.70 x visual-rank quality + 0.30 x level-profile overlap`. The delivered rows
are then reordered strictly by the original visual score; correlation never
enters the ranking objective.

Session-pair coverage is neither constrained nor rewarded: the selector is free
to take the strongest examples regardless of which session pair they came from.
All ten fixed r45 bins (including at least one r45 >= 0.98), the four Figure 2b
anchors, every field-count class,
levels L1-L7 on both sides, within-level entrance/junction/proximal/middle/distal
locations, and a mixture of same-run, branch-switch, depth-shift, field gain/loss,
identical level-set, disjoint level-set, and low-overlap examples are protected by
hard constraints.

Cross-pair repetition is also a hard constraint, not a post-hoc warning. Every
A/B map is compared with both sides of every other candidate after applying a
small Gaussian tolerance to activity above {MORPHOLOGY_ACTIVITY_FLOOR:.2f}. A
direct conflict edge is added when one individual-map match has blurred cosine
>= {MORPHOLOGY_SIDE_SIMILARITY_THRESHOLD:.2f}, or blurred cosine >=
{MORPHOLOGY_FIELD_OVERLAP_SIMILARITY_THRESHOLD:.2f} together with display-field
Dice >= {MORPHOLOGY_FIELD_DICE_THRESHOLD:.2f}; no directly connected pair can be
selected. The strongest field on each side supplies a primary run-pair motif.
Repeated same-run motifs are capped at one example and mixed-run motifs at two,
with fixed Figure 2b anchors taking precedence if ever necessary.

{correction_note}

## Level-profile overlap

For each map and level, the metric averages activity above {DEFAULT_ACTIVITY_FLOOR:.2f}
on the independently q99-normalized 45 x 45 map over two-session common support.
Per-level means prevent deeper levels from winning merely because they contain
more branches. Levels with fewer than {DEFAULT_RELIABLE_FINE_BINS} common fine
bins are downweighted. The seven values are normalized to sum to one, and the
pair score is the shared mass `sum(min(profile A, profile B))`, from 0 to 1.

This is a descriptive metric of the displayed place maps, not an inferential
level-map stability statistic. Encoded-level Jaccard, dominant-level shift,
ordered level Earth-mover distance, threshold sensitivities, raw-C overlap, and
run/location changes remain separate audits. Gallery labels therefore say
`level overlap`, not `stable`.

## Result

- {len(selected)} pairs from {selected['pair_key'].nunique()} session pairs.
- r45 range {selected['place_r_45_display'].min():.3f} to {selected['place_r_45_display'].max():.3f}.
- {100 * validation['multi_field_side_fraction']:.1f}% of map sides are multi-field.
- Level-profile overlap mean/median: {selected['level_profile_overlap'].mean():.3f}/{selected['level_profile_overlap'].median():.3f}.
- Location tokens retained: A={validation['location_tokens_a']}, B={validation['location_tokens_b']}.
- Selected morphology conflicts: 0 across {validation['morphology_component_count']}
  parent conflict component(s); maximum repeated same-run/mixed-run primary motif
  counts are {validation['max_same_primary_run_pair_count']}/
  {validation['max_mixed_primary_run_pair_count']}.
- Mean level overlap is {baseline['mean_level_profile_overlap']:.3f} at 0%,
  {ten_percent['mean_level_profile_overlap']:.3f} at 10%, and
  {biased['mean_level_profile_overlap']:.3f} at 30%; corresponding mean raw
  visual scores are {baseline['raw_visual_score_mean']:.5f},
  {ten_percent['raw_visual_score_mean']:.5f}, and
  {biased['raw_visual_score_mean']:.5f}.

## Main files

- `selected_32.csv`: complete shortlist metadata and selection scores.
- `compact_grid_key.csv`: row/pair/map-column positions and concise source metadata.
- `all_100_level_preservation_scores.csv`: all parent candidates with inclusion flags.
- `level_profiles_long.csv`: level-by-level profile inputs and support weights
  for all 100 scored candidates, with shortlist inclusion flags.
- `selected_level_profiles_long.csv`: the corresponding 32-candidate subset.
- `coverage_100_vs_32.csv` and `constraint_audit.csv`: achieved coverage.
- `bias_ablation_summary.csv` and `bias_ablation_membership.csv`: 0%, 10%, and 30% comparison.
- `all_100_field_readability_scores.csv` and `reference_replacement_audit.csv`:
  compact-field inputs and the exact 28-lock/four-replacement crosswalk.
- `all_100_pair_morphology_audit.csv.gz`, `selected_pair_morphology_audit.csv`,
  and `morphology_conflict_components.csv`: individual-map redundancy inputs,
  selected-pair audit, and direct-edge conflict-graph membership.
- `selected_maps_fields_45_shortlist32.npz`: plot-ready raw/normalized maps,
  masks, field labels, q99 scales, and both rank systems.
- `selected_display_fields.csv` and
  `selected_map_bins_topology_45_shortlist32.csv.gz`: field/bin source data.
{render_note}

All maps use independent within-map q99 normalization on common support. The
colorbar is fixed from 0 to 1 and explicitly labels q99 = 1; raw amplitudes and
q99 scales remain in the source arrays/tables.

The primary plate deliberately omits session names, visual scores, and field
labels so the 64 heatmaps remain large. Each thin grey card encloses one adjacent
A/B pair; its header gives the visual rank and raw-map place correlation. The
four reused Figure 2b examples have a thicker red card plus a direct `Fig. 2b-N`
tag, so provenance is not encoded by color alone. Full labels are in the grid key.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _source_manifest(
    parent_paths: Mapping[str, Path],
    *,
    extra_paths: Iterable[Path] = (),
) -> pd.DataFrame:
    paths = [
        *parent_paths.values(),
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "ca_analysis" / "ed06_level_preservation.py",
        REPO_ROOT / "src" / "ca_analysis" / "ed06_representation_diversity.py",
        REPO_ROOT / "src" / "ca_analysis" / "ed06_representation_selection.py",
        REPO_ROOT / "src" / "ca_analysis" / "ed06_example_candidates.py",
        REPO_ROOT / "scripts" / "build_ed06_example_place_map_candidates_v2.py",
        REPO_ROOT / "scripts" / "build_ed06_example_place_map_candidates_v3.py",
        *(Path(path) for path in extra_paths),
    ]
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    return pd.DataFrame(
        [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": v3.sha256_file(path),
            }
            for path in paths
        ]
    )


def _build_bundle(args: argparse.Namespace, *, canonical_output_dir: Path) -> int:
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_paths = _required_parent_paths(parent_dir)

    print("Loading frozen validated v3 gallery...", flush=True)
    parent, fields, parent_arrays = load_parent_bundle(parent_dir)
    preservation, level_profiles = characterize_level_preservation(
        parent, parent_arrays
    )
    if not preservation["level_profile_valid"].astype(bool).all():
        invalid = (
            preservation.loc[
                ~preservation["level_profile_valid"].astype(bool), "candidate_id"
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"undefined level profiles in frozen v3 parent: {invalid}")
    characterized = parent.merge(preservation, on="candidate_id", validate="one_to_one")
    characterized = add_selection_columns(characterized)
    characterized = add_field_readability_columns(characterized, parent_arrays)

    reference_audit = pd.DataFrame()
    replacement_policy: dict[str, object] = {"enabled": False}
    selection_frame = characterized
    selector_score_col = "visual_rank_quality"
    selector_fixed_col = "fixed_main_fig2b"
    selector_refinement_columns = (
        "selection_utility_lambda_0",
        "selection_utility_lambda_0.10",
        "selection_utility_lambda_0.30",
    )
    if args.reference_shortlist_dir:
        reference_dir = Path(args.reference_shortlist_dir).expanduser().resolve()
        replacement_ranks = _parse_replacement_ranks(args.replace_visual_ranks)
        (
            characterized,
            selection_frame,
            reference_audit,
            replacement_policy,
        ) = apply_reference_replacement_policy(
            characterized,
            reference_dir=reference_dir,
            replacement_ranks=replacement_ranks,
        )
        selector_score_col = "pair_field_readability_score"
        selector_fixed_col = "locked_reference"
        selector_refinement_columns = (
            "replacement_utility_lambda_0",
            "replacement_utility_lambda_0.10",
            "replacement_utility_lambda_0.30",
        )

    print("Building individual-map morphology conflicts...", flush=True)
    morphology_audit = build_pair_morphology_audit(characterized, parent_arrays)
    (
        morphology_conflict_pairs,
        morphology_components,
    ) = build_morphology_conflict_components(characterized, morphology_audit)

    print("Solving visual/diversity/preservation shortlist...", flush=True)
    (
        selected,
        diagnostics,
        baseline,
        baseline_diagnostics,
        ten_percent,
        ten_percent_diagnostics,
    ) = select_shortlist(
        selection_frame,
        conflict_pairs=morphology_conflict_pairs,
        solver_time_limit=float(args.solver_time_limit),
        score_col=selector_score_col,
        fixed_col=selector_fixed_col,
        refinement_score_columns=selector_refinement_columns,
    )
    selected_ids = set(selected["candidate_id"].astype(str))
    baseline_ids = set(baseline["candidate_id"].astype(str))
    ten_percent_ids = set(ten_percent["candidate_id"].astype(str))
    all_100 = characterized.copy()
    all_100["selected_lambda_0"] = all_100["candidate_id"].isin(baseline_ids)
    all_100["selected_lambda_0.10"] = all_100["candidate_id"].isin(ten_percent_ids)
    all_100["selected_lambda_0.30"] = all_100["candidate_id"].isin(selected_ids)
    all_100["final_selection_action"] = "unselected"
    if replacement_policy.get("enabled", False):
        locked_ids = set(replacement_policy["locked_candidate_ids"])
        rejected_ids = set(replacement_policy["rejected_candidate_ids"])
        replacement_ids = selected_ids - locked_ids
        all_100.loc[
            all_100["candidate_id"].astype(str).isin(locked_ids),
            "final_selection_action",
        ] = "locked_from_reference"
        all_100.loc[
            all_100["candidate_id"].astype(str).isin(rejected_ids),
            "final_selection_action",
        ] = "rejected_from_reference"
        all_100.loc[
            all_100["candidate_id"].astype(str).isin(replacement_ids),
            "final_selection_action",
        ] = "selected_replacement"
        reference_audit["final_selected"] = reference_audit[
            "candidate_id"
        ].astype(str).isin(selected_ids)
        replacement_audit_rows = selected.loc[
            selected["candidate_id"].astype(str).isin(replacement_ids),
            [
                "candidate_id",
                "visual_rank",
                "parent_visual_rank",
                "place_r_45_display",
                "level_profile_overlap",
                "field_count_a",
                "field_count_b",
                "pair_field_readability_score",
                "pair_field_readability_max_effective_bins",
                "field_readability_prominent_fields_a",
                "field_readability_prominent_fields_b",
                "field_readability_replacement_eligible",
            ],
        ].copy()
        replacement_audit_rows = replacement_audit_rows.rename(
            columns={"visual_rank": "final_visual_rank"}
        )
        replacement_audit_rows["reference_visual_rank"] = np.nan
        replacement_audit_rows["reference_action"] = "selected_replacement"
        replacement_audit_rows["final_selected"] = True
        reference_audit = pd.concat(
            [reference_audit, replacement_audit_rows],
            ignore_index=True,
            sort=False,
        )
        final_rank_lookup = selected.set_index("candidate_id")["visual_rank"].to_dict()
        reference_audit["final_visual_rank"] = reference_audit[
            "candidate_id"
        ].map(final_rank_lookup)

    selected_fields = fields.loc[
        fields["candidate_id"].astype(str).isin(selected_ids)
    ].copy()
    rank_lookup = selected.set_index("candidate_id")["visual_rank"].to_dict()
    parent_rank_lookup = selected.set_index("candidate_id")[
        "parent_visual_rank"
    ].to_dict()
    selected_fields["parent_visual_rank"] = selected_fields["candidate_id"].map(
        parent_rank_lookup
    )
    selected_fields["visual_rank"] = selected_fields["candidate_id"].map(rank_lookup)
    selected_fields = selected_fields.sort_values(
        ["visual_rank", "side", "field_id"], kind="mergesort"
    )
    level_profiles = level_profiles.copy()
    level_profiles["selected_lambda_0"] = (
        level_profiles["candidate_id"].astype(str).isin(baseline_ids)
    )
    level_profiles["selected_lambda_0.10"] = (
        level_profiles["candidate_id"].astype(str).isin(ten_percent_ids)
    )
    level_profiles["selected_lambda_0.30"] = (
        level_profiles["candidate_id"].astype(str).isin(selected_ids)
    )
    level_profiles["visual_rank"] = level_profiles["candidate_id"].map(rank_lookup)
    level_profiles = level_profiles.sort_values(
        ["parent_visual_rank", "side", "display_level_1based"], kind="mergesort"
    )
    selected_level_profiles = level_profiles.loc[
        level_profiles["selected_lambda_0.30"]
    ].copy()
    selected_level_profiles = selected_level_profiles.sort_values(
        ["visual_rank", "side", "display_level_1based"], kind="mergesort"
    )

    maps = reconstruct_map_lookup(characterized, fields, parent_arrays)
    selected_morphology = morphology_audit.loc[
        morphology_audit["candidate_id_first"].astype(str).isin(selected_ids)
        & morphology_audit["candidate_id_second"].astype(str).isin(selected_ids)
    ].copy()
    conflicts = selected_morphology.loc[
        selected_morphology["morphology_conflict"].astype(bool)
    ].copy()
    constraint_table = build_constraint_table(diagnostics)
    coverage = build_coverage_table(characterized, selected)
    ablation, ablation_membership = build_bias_ablation(
        characterized,
        selected,
        diagnostics,
        baseline,
        baseline_diagnostics,
        ten_percent,
        ten_percent_diagnostics,
    )
    raw_disagreement = characterized[
        [
            "candidate_id",
            "parent_visual_rank",
            "level_profile_overlap",
            "raw_c_level_profile_overlap",
            "raw_c_overlap_absolute_difference",
            "place_r_45_display",
        ]
    ].sort_values(
        ["raw_c_overlap_absolute_difference", "parent_visual_rank"],
        ascending=[False, True],
        kind="mergesort",
    )

    grid_key = build_compact_grid_key(selected)
    v3._export_frame(selected, output_dir / "selected_32.csv")
    v3._export_frame(grid_key, output_dir / "compact_grid_key.csv")
    v3._export_frame(all_100, output_dir / "all_100_level_preservation_scores.csv")
    v3._export_frame(level_profiles, output_dir / "level_profiles_long.csv")
    v3._export_frame(
        selected_level_profiles, output_dir / "selected_level_profiles_long.csv"
    )
    v3._export_frame(selected_fields, output_dir / "selected_display_fields.csv")
    v3._export_frame(constraint_table, output_dir / "constraint_audit.csv")
    v3._export_frame(coverage, output_dir / "coverage_100_vs_32.csv")
    v3._export_frame(ablation, output_dir / "bias_ablation_summary.csv")
    v3._export_frame(ablation_membership, output_dir / "bias_ablation_membership.csv")
    v3._export_frame(raw_disagreement, output_dir / "rawC_disagreement_audit.csv")
    readability_columns = [
        "candidate_id",
        "parent_visual_rank",
        "place_r_45_display",
        "correlation_bin",
        "visual_score_v3",
        "level_profile_overlap",
        "field_count_a",
        "field_count_b",
        "field_readability_prominent_fields_a",
        "field_readability_prominent_fields_b",
        "field_readability_median_effective_bins_a",
        "field_readability_median_effective_bins_b",
        "field_readability_max_effective_bins_a",
        "field_readability_max_effective_bins_b",
        "pair_field_readability_max_effective_bins",
        "field_readability_score_a",
        "field_readability_score_b",
        "pair_field_readability_score",
        "field_readability_replacement_eligible",
        "locked_reference",
        "rejected_reference",
        "replacement_candidate_pool",
        "final_selection_action",
    ]
    for column in ("locked_reference", "rejected_reference", "replacement_candidate_pool"):
        if column not in all_100.columns:
            all_100[column] = False
    v3._export_frame(
        all_100[readability_columns],
        output_dir / "all_100_field_readability_scores.csv",
    )
    v3._export_frame(
        reference_audit,
        output_dir / "reference_replacement_audit.csv",
    )
    v3._export_frame(
        morphology_audit,
        output_dir / "all_100_pair_morphology_audit.csv.gz",
        compression="gzip",
    )
    v3._export_frame(
        selected_morphology, output_dir / "selected_pair_morphology_audit.csv"
    )
    v3._export_frame(
        morphology_components, output_dir / "morphology_conflict_components.csv"
    )
    v3._export_frame(conflicts, output_dir / "selected_image_similarity_conflicts.csv")
    export_selected_maps(selected, parent_arrays, characterized, parent_dir, output_dir)
    selected_topology = pd.read_csv(
        output_dir / "selected_map_bins_topology_45_shortlist32.csv.gz",
        usecols=["candidate_id", "side", "visual_rank", "bin_y", "bin_x"],
    )

    render_paths: list[str] = []
    if not args.skip_rendering:
        print("Rendering compact one-page clean and overlay grids...", flush=True)
        render_paths.extend(
            render_compact_grid(selected, maps, output_dir, overlay_fields=False)
        )
        render_paths.extend(
            render_compact_grid(selected, maps, output_dir, overlay_fields=True)
        )
        if not args.skip_individuals:
            render_paths.extend(v3.render_individuals(selected, maps, output_dir))

    subset_npz_path = output_dir / "selected_maps_fields_45_shortlist32.npz"
    with np.load(subset_npz_path, allow_pickle=False) as subset_data:
        subset_arrays = {key: np.asarray(subset_data[key]) for key in subset_data.files}
        validation = validate_shortlist(
            characterized,
            selected,
            ten_percent,
            grid_key,
            selected_fields,
            selected_level_profiles,
            selected_topology,
            selected_morphology,
            morphology_components,
            conflicts,
            constraint_table,
            subset_arrays,
            diagnostics,
            baseline_diagnostics,
            ten_percent_diagnostics,
            render_paths,
            replacement_policy,
        )
    (output_dir / "validation_checks.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, default=v3._json_default),
        encoding="utf-8",
    )
    (output_dir / "selection_solver_diagnostics.json").write_text(
        json.dumps(
            {
                "lambda_0": baseline_diagnostics,
                "lambda_0.10": ten_percent_diagnostics,
                "lambda_0.30": diagnostics,
            },
            indent=2,
            sort_keys=True,
            default=v3._json_default,
        ),
        encoding="utf-8",
    )
    source_manifest_extra = (
        [Path(replacement_policy["reference_selected_32"])]
        if replacement_policy.get("enabled", False)
        else []
    )
    source_manifest = _source_manifest(
        parent_paths, extra_paths=source_manifest_extra
    )
    v3._export_frame(source_manifest, output_dir / "source_component_manifest.csv")
    write_readme(
        output_dir,
        selected,
        ablation,
        validation,
        render_paths,
        replacement_policy,
    )

    if not validation["all_pass"]:
        raise RuntimeError(f"shortlist validation failed: {validation['failures']}")

    artifact_records = v3._artifact_records(output_dir)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "parent_dir": str(parent_dir),
        "output_dir": str(canonical_output_dir),
        "target_n": TARGET_N,
        "visual_retention": VISUAL_RETENTION,
        "diversity_retention": DIVERSITY_RETENTION,
        "level_bias_weight": LEVEL_BIAS_WEIGHT,
        "session_pair_policy": "audit only; neither constrained nor optimized",
        "display_order": "strict frozen-v3 visual_score_v3 descending",
        "reviewed_shortlist_replacement": replacement_policy,
        "field_readability": {
            "unit": "each exported display field on each 45 x 45 normalized map",
            "effective_field_bins": "(sum activity)^2 / sum(activity^2)",
            "maximum_effective_bins_for_replacement": (
                FIELD_READABILITY_MAX_EFFECTIVE_BINS
            ),
            "minimum_exported_fields_per_side": (
                FIELD_READABILITY_MIN_EXPORTED_FIELDS
            ),
            "minimum_prominent_fields_per_side": (
                FIELD_READABILITY_MIN_PROMINENT_FIELDS
            ),
            "selection_score": (
                "weaker-side weighted readability with compact field sizes, "
                "second-field visibility, within-map concentration, and "
                "field-mass support"
            ),
            "level_bias": "0.70 x readability + 0.30 x level-profile overlap",
        },
        "cross_pair_morphology": {
            "unit": "individual map with all A/B cross-candidate comparisons",
            "activity_floor": MORPHOLOGY_ACTIVITY_FLOOR,
            "power": MORPHOLOGY_POWER,
            "gaussian_sigma": MORPHOLOGY_GAUSSIAN_SIGMA,
            "strict_blurred_cosine_threshold": MORPHOLOGY_SIDE_SIMILARITY_THRESHOLD,
            "field_overlap_blurred_cosine_threshold": (
                MORPHOLOGY_FIELD_OVERLAP_SIMILARITY_THRESHOLD
            ),
            "field_overlap_dice_threshold": MORPHOLOGY_FIELD_DICE_THRESHOLD,
            "activity_mask_dice": "audit only",
            "conflict_policy": "no selected candidate pair may share a direct edge",
            "primary_run_pair_policy": (
                "same-run motif max 1; mixed-run motif max 2; fixed anchors prevail"
            ),
            "parent_conflict_edges": int(morphology_audit["morphology_conflict"].sum()),
            "parent_conflict_components": int(
                morphology_components["component_id"].nunique()
            )
            if not morphology_components.empty
            else 0,
        },
        "compact_grid": {
            "pair_rows": 8,
            "pair_columns": 4,
            "map_rows": 8,
            "map_columns": 8,
            "pair_adjacency": "A immediately left of B",
            "pair_card": "thin neutral outline with rank and place-map r header",
            "main_figure_encoding": (
                "thicker red outline plus direct Fig. 2b-anchor-order tag"
            ),
            "primary_annotations": "A/B in-map labels; pair metadata in card header",
        },
        "level_profile": {
            "normalization": "independent within-map q99",
            "support": "two-session common occupancy",
            "activity_floor": DEFAULT_ACTIVITY_FLOOR,
            "reliable_fine_bins": DEFAULT_RELIABLE_FINE_BINS,
            "metric": "sum_l min(p_A_l, p_B_l)",
            "interpretation": "descriptive display-derived selection metric",
        },
        "rendered_paths": render_paths,
        "validation": validation,
        "selection_diagnostics": diagnostics,
        "baseline_diagnostics": baseline_diagnostics,
        "artifacts": artifact_records,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=v3._json_default),
        encoding="utf-8",
    )
    print(
        f"Done: {len(selected)} examples, {selected['pair_key'].nunique()} pairs, "
        f"r45 {selected['place_r_45_display'].min():.3f} to "
        f"{selected['place_r_45_display'].max():.3f}, "
        f"median level overlap {selected['level_profile_overlap'].median():.3f}.",
        flush=True,
    )
    return 0


def _promote_staged_bundle(staging_dir: Path, final_dir: Path) -> None:
    if staging_dir.parent != final_dir.parent:
        raise ValueError("staging and final output directories must share a parent")
    backup_dir = final_dir.parent / f".{final_dir.name}.backup-{uuid.uuid4().hex}"
    had_previous = final_dir.exists()
    if had_previous:
        os.replace(final_dir, backup_dir)
    try:
        os.replace(staging_dir, final_dir)
    except BaseException:
        if had_previous and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def main() -> int:
    args = parse_args()
    final_output_dir = Path(args.output_dir).expanduser().resolve()
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-",
            dir=final_output_dir.parent,
        )
    ).resolve()
    staged_args = argparse.Namespace(**vars(args))
    staged_args.output_dir = str(staging_dir)
    try:
        result = _build_bundle(staged_args, canonical_output_dir=final_output_dir)
        _promote_staged_bundle(staging_dir, final_output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return result


if __name__ == "__main__":
    raise SystemExit(main())
