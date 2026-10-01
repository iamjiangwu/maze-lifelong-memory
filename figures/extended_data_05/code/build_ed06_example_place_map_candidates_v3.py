"""Build the level-aware ED6 real-place-map candidate gallery.

This v3 workflow starts from the frozen exhaustive v2 registry/cache, measures
thresholded display fields on all 17,845 map pairs, maps every field into the
maze's true tree-depth/run coordinate system, and selects 100 examples in that
scientific representation space.  The output remains an illustrative review
gallery and is not promoted into the Extended Data assembly by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ed06_v3_mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for source_root in (REPO_ROOT, SRC_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ca_analysis.ed06_example_candidates import (
    CORRELATION_BIN_LABELS,
    combine_pair_quality_scores,
    normalize_pair_independent_quantile,
    safe_corr,
)
from ca_analysis.ed06_place_map_cache import (
    PlaceMapCacheParameters,
    resolve_candidate_side_context,
)
from ca_analysis.ed06_representation_diversity import (
    DISPLAY_FIELD_MIN_FINE_BINS,
    DISPLAY_FIELD_THRESHOLD,
    build_fine_maze_crosswalk,
    describe_display_pair,
    field_count_neutral_map_quality,
    maze_wall_to_fine_coordinates,
)
from ca_analysis.ed06_representation_selection import (
    select_representation_diverse_candidates,
)
from ca_analysis.maze import NewMaze
from scripts import build_ed06_example_place_map_candidates_v2 as v2


matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


DISPLAY_QUANTILE = 0.99
COLORBAR_LABEL = "Within-map normalized denoised C (q99 = 1)"
DEFAULT_V2 = Path(
    r"C:\Users\jiangwu\Documents\Codex\2026-07-31\fo\outputs\ed06_place_map_candidates_100_v2"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\jiangwu\Documents\Codex\2026-07-31\fo\outputs\ed06_place_map_candidates_100_v3_levelaware"
)
IMAGE_SIMILARITY_THRESHOLD = 0.95
LEVEL_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", default=str(DEFAULT_V2))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-individuals", action="store_true")
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--reuse-characterization", action="store_true")
    parser.add_argument("--quality-retention", type=float, default=0.98)
    parser.add_argument("--diversity-retention", type=float, default=0.98)
    parser.add_argument("--solver-time-limit", type=float, default=180.0)
    parser.add_argument(
        "--image-similarity-threshold",
        type=float,
        default=IMAGE_SIMILARITY_THRESHOLD,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reliability_scale(value: object) -> float:
    numeric = float(value) if pd.notna(value) else -0.10
    return float(np.clip((numeric + 0.10) / 0.90, 0.0, 1.0))


def _activity_scale(value: object) -> float:
    numeric = float(value) if pd.notna(value) else 0.0
    return float(np.log1p(np.clip(numeric, 0.0, 12.0)) / np.log1p(12.0))


def _field_count_bucket(value: int) -> str:
    count = int(value)
    if count <= 3:
        return str(count)
    if count <= 5:
        return "4-5"
    if count <= 8:
        return "6-8"
    return "9+"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (tuple, set, frozenset, np.ndarray)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, frozenset, np.ndarray)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return value


def _export_frame(
    frame: pd.DataFrame, path: Path, *, compression: str | None = None
) -> None:
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]).columns:
        if (
            output[column]
            .map(
                lambda value: isinstance(
                    value, (dict, list, tuple, set, frozenset, np.ndarray)
                )
            )
            .any()
        ):
            output[column] = output[column].map(_serialize_cell)
    output.to_csv(path, index=False, compression=compression)


def _candidate_raw_maps(
    source: Mapping[str, Any],
    sessions: Mapping[str, Any],
    anchor_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_id = str(source["candidate_id"])
    if bool(source.get("fixed_main_fig2b", False)):
        values = anchor_maps[candidate_id]
        return (
            np.asarray(values["map_a"], dtype=float),
            np.asarray(values["map_b"], dtype=float),
            np.asarray(values["occ_a"]) > 0,
            np.asarray(values["occ_b"]) > 0,
        )
    context_a = resolve_candidate_side_context(source, "A")
    context_b = resolve_candidate_side_context(source, "B")
    map_a, mask_a = v2.cached_node_map(sessions, context_a, int(source["train_id"]))
    map_b, mask_b = v2.cached_node_map(sessions, context_b, int(source["test_id"]))
    return (
        np.asarray(map_a, dtype=float),
        np.asarray(map_b, dtype=float),
        np.asarray(mask_a, dtype=bool),
        np.asarray(mask_b, dtype=bool),
    )


def _flatten_map_summary(summary: Mapping[str, Any], side: str) -> dict[str, Any]:
    suffix = str(side).lower()
    return {f"{key}_{suffix}": value for key, value in summary.items()}


def _field_rows(
    candidate: Mapping[str, Any],
    description: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "candidate_id": str(candidate["candidate_id"]),
        "pair_key": str(candidate["pair_key"]),
        "fixed_main_fig2b": bool(candidate["fixed_main_fig2b"]),
        "place_r_45_display": float(candidate["place_r_45_display"]),
    }
    for side in ("a", "b"):
        for field in description[side]["fields"]:
            record = {**base, "side": side.upper(), **field}
            rows.append(record)
    return rows


def _primary_run_tokens(fields: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        f"L{int(field['dominant_display_level_1based'])}:"
        f"R{int(field['dominant_run_index'])}:"
        f"{field['dominant_within_run_region']}"
        for field in fields
    )


def _semantic_record(
    base: Mapping[str, Any],
    description: Mapping[str, Any],
    normalized_a: np.ndarray,
    normalized_b: np.ndarray,
    common: np.ndarray,
) -> dict[str, Any]:
    summary_a = description["a"]["summary"]
    summary_b = description["b"]["summary"]
    fields_a = description["a"]["fields"]
    fields_b = description["b"]["fields"]
    pair = description["pair_summary"]
    quality_a = field_count_neutral_map_quality(normalized_a, common)
    quality_b = field_count_neutral_map_quality(normalized_b, common)
    map_score_a = (
        0.80 * float(quality_a["image_quality_score"])
        + 0.15 * _reliability_scale(base["splithalf_corr_train"])
        + 0.05 * _activity_scale(base["rate_per_min_train"])
    )
    map_score_b = (
        0.80 * float(quality_b["image_quality_score"])
        + 0.15 * _reliability_scale(base["splithalf_corr_test"])
        + 0.05 * _activity_scale(base["rate_per_min_test"])
    )
    visual_score = float(combine_pair_quality_scores(map_score_a, map_score_b))

    runs_a = {int(field["dominant_run_index"]) for field in fields_a}
    runs_b = {int(field["dominant_run_index"]) for field in fields_b}
    levels_a = set(map(int, summary_a["encoded_level_set"]))
    levels_b = set(map(int, summary_b["encoded_level_set"]))
    count_a = int(summary_a["field_count"])
    count_b = int(summary_b["field_count"])
    bucket_a = _field_count_bucket(count_a)
    bucket_b = _field_count_bucket(count_b)
    if bool(runs_a & runs_b) and levels_a.isdisjoint(levels_b):
        raise AssertionError("shared dominant run cannot coexist with disjoint levels")
    record: dict[str, Any] = dict(base)
    record.update(_flatten_map_summary(summary_a, "a"))
    record.update(_flatten_map_summary(summary_b, "b"))
    record.update(pair)
    for key, value in quality_a.items():
        record[f"field_neutral_{key}_a"] = value
    for key, value in quality_b.items():
        record[f"field_neutral_{key}_b"] = value
    record.update(
        {
            "map_visual_score_v3_a": map_score_a,
            "map_visual_score_v3_b": map_score_b,
            "visual_score_v3": visual_score,
            "field_count_bucket_a": bucket_a,
            "field_count_bucket_b": bucket_b,
            "selector_field_category_a": f"A:{bucket_a}",
            "selector_field_category_b": f"B:{bucket_b}",
            "field_count_transition_v3": f"{bucket_a}->{bucket_b}",
            "dominant_run_tokens_a": _primary_run_tokens(fields_a),
            "dominant_run_tokens_b": _primary_run_tokens(fields_b),
            "phen_same_run": bool(runs_a & runs_b),
            "phen_within_level_branch_switch": bool(levels_a & levels_b)
            and not bool(runs_a & runs_b),
            "phen_depth_shift": bool(levels_a and levels_b)
            and levels_a.isdisjoint(levels_b),
            "phen_field_gain_loss": count_a != count_b,
            "dominant_run_set_a": tuple(sorted(runs_a)),
            "dominant_run_set_b": tuple(sorted(runs_b)),
        }
    )
    return record


def characterize_universe(
    pool: pd.DataFrame,
    anchors: pd.DataFrame,
    scored: pd.DataFrame,
    sessions: Mapping[str, Any],
    anchor_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Characterize all active pairs and the four fixed anchors."""

    lookup = scored.set_index("candidate_id", drop=False)
    crosswalk = build_fine_maze_crosswalk()
    records: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    embeddings: dict[str, np.ndarray] = {}
    sources = [
        *pool.to_dict(orient="records"),
        *anchors.to_dict(orient="records"),
    ]
    for index, source in enumerate(sources, start=1):
        candidate_id = str(source["candidate_id"])
        base = lookup.loc[candidate_id].to_dict()
        map_a, map_b, mask_a, mask_b = _candidate_raw_maps(
            source, sessions, anchor_maps
        )
        common = mask_a & mask_b
        normalized_a, normalized_b, _, _ = normalize_pair_independent_quantile(
            map_a,
            map_b,
            common,
            quantile=DISPLAY_QUANTILE,
        )
        description = describe_display_pair(
            normalized_a,
            map_a,
            normalized_b,
            map_b,
            common,
            crosswalk=crosswalk,
        )
        record = _semantic_record(base, description, normalized_a, normalized_b, common)
        recomputed_r = safe_corr(map_a[common], map_b[common])
        if not np.isclose(recomputed_r, float(base["place_r_45_display"]), atol=1e-6):
            raise AssertionError(f"r45 mismatch for {candidate_id}")
        records.append(record)
        fields.extend(_field_rows(record, description))
        embeddings[candidate_id] = v2.downsample_pair_embedding(
            normalized_a, normalized_b, common
        )
        if index % 1000 == 0 or index == len(sources):
            print(
                f"Characterized display fields: {index:,}/{len(sources):,}",
                flush=True,
            )
    frame = pd.DataFrame(records)
    frame = frame.sort_values(
        ["visual_score_v3", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "full_visual_rank_v3", np.arange(1, len(frame) + 1))
    field_frame = pd.DataFrame(fields)
    if len(frame) != 17_845 or len(field_frame) == 0:
        raise AssertionError("unexpected v3 representation universe size")
    return frame, field_frame, embeddings


def compute_candidate_embeddings(
    frame: pd.DataFrame,
    sessions: Mapping[str, Any],
    anchor_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Recompute compact duplicate-screen embeddings for a bounded candidate set."""

    embeddings: dict[str, np.ndarray] = {}
    for index, source in enumerate(frame.to_dict(orient="records"), start=1):
        map_a, map_b, mask_a, mask_b = _candidate_raw_maps(
            source, sessions, anchor_maps
        )
        common = mask_a & mask_b
        normalized_a, normalized_b, _, _ = normalize_pair_independent_quantile(
            map_a, map_b, common, quantile=DISPLAY_QUANTILE
        )
        candidate_id = str(source["candidate_id"])
        embeddings[candidate_id] = v2.downsample_pair_embedding(
            normalized_a, normalized_b, common
        )
        if index % 1000 == 0 or index == len(frame):
            print(f"Rebuilt embeddings: {index:,}/{len(frame):,}", flush=True)
    return embeddings


def save_embeddings(
    embeddings: Mapping[str, np.ndarray], path: Path, ordered_ids: Iterable[str]
) -> None:
    ids = [str(value) for value in ordered_ids]
    np.savez_compressed(
        path,
        candidate_id=np.asarray(ids, dtype="U"),
        embedding=np.stack([np.asarray(embeddings[value]) for value in ids]),
    )


def load_embeddings(path: Path, expected_ids: Iterable[str]) -> dict[str, np.ndarray]:
    expected = [str(value) for value in expected_ids]
    with np.load(path, allow_pickle=False) as payload:
        ids = payload["candidate_id"].astype(str).tolist()
        values = np.asarray(payload["embedding"], dtype=np.float32)
    if ids != expected or values.shape[0] != len(ids):
        raise ValueError("cached semantic-pool embeddings do not match solver pool")
    return {candidate_id: values[index] for index, candidate_id in enumerate(ids)}


def _selected_similarity_conflicts(
    selected: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    *,
    threshold: float,
) -> pd.DataFrame:
    columns = [
        "candidate_id_1",
        "candidate_id_2",
        "direct_similarity",
        "swapped_similarity",
        "max_similarity",
        "threshold",
    ]
    ids = selected["candidate_id"].astype(str).tolist()
    vectors = np.stack([np.asarray(embeddings[value], dtype=float) for value in ids])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    midpoint = vectors.shape[1] // 2
    swapped = np.concatenate([vectors[:, midpoint:], vectors[:, :midpoint]], axis=1)
    rows: list[dict[str, Any]] = []
    for first in range(len(ids)):
        direct = vectors[first + 1 :] @ vectors[first]
        reverse = swapped[first + 1 :] @ vectors[first]
        maxima = np.maximum(direct, reverse)
        for offset in np.flatnonzero(maxima >= float(threshold)):
            second = first + 1 + int(offset)
            rows.append(
                {
                    "candidate_id_1": ids[first],
                    "candidate_id_2": ids[second],
                    "direct_similarity": float(direct[offset]),
                    "swapped_similarity": float(reverse[offset]),
                    "max_similarity": float(maxima[offset]),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _selector_kwargs(
    frame: pd.DataFrame,
    *,
    required_pairs: list[str],
    conflict_pairs: list[tuple[str, str]],
    quality_retention: float,
    diversity_retention: float,
    solver_time_limit: float,
) -> dict[str, Any]:
    level_columns: dict[str, tuple[str, ...]] = {}
    level_minima: dict[str, int] = {}
    for side in ("a", "b"):
        side_label = side.upper()
        for level in range(1, 8):
            key = f"{side_label}:L{level}"
            level_columns[key] = (f"encodes_level_{level}_{side}",)
            level_minima[key] = 8 if level < 7 else 4
    category_bounds = {
        "A:1": (None, 25),
        "A:2": (15, None),
        "A:3": (12, None),
        "A:4-5": (15, None),
        "A:6-8": (10, None),
        "A:9+": (4, None),
        "B:1": (None, 25),
        "B:2": (15, None),
        "B:3": (12, None),
        "B:4-5": (15, None),
        "B:6-8": (10, None),
        "B:9+": (4, None),
    }
    return {
        "target_n": 100,
        "required_pairs": required_pairs,
        "min_per_pair": 2,
        "max_per_pair": 4,
        "required_correlation_bins": list(CORRELATION_BIN_LABELS),
        "min_per_correlation_bin": 5,
        "max_per_correlation_bin": 15,
        "min_low_extreme": 4,
        "min_high_extreme": 4,
        "correlation_col": "place_r_45_display",
        "score_col": "visual_score_v3",
        "fixed_col": "fixed_main_fig2b",
        "conflict_pairs": conflict_pairs,
        "side_category_columns": (
            "selector_field_category_a",
            "selector_field_category_b",
        ),
        "side_category_bounds": category_bounds,
        "single_field_category": None,
        "max_single_field_side_fraction": None,
        "level_membership_columns": level_columns,
        "minimum_sides_per_level": level_minima,
        "semantic_token_columns": (
            "encoded_level_token_a",
            "encoded_level_token_b",
            "level_phenotype_a",
            "level_phenotype_b",
            "field_count_transition_v3",
            "level_set_transition",
            "phenotype_transition",
            "field_location_tokens_a",
            "field_location_tokens_b",
            "dominant_run_tokens_a",
            "dominant_run_tokens_b",
        ),
        "phenotype_flag_columns": (
            "phen_same_run",
            "phen_within_level_branch_switch",
            "phen_depth_shift",
            "phen_field_gain_loss",
        ),
        "flag_bounds": {
            "phen_same_run": (10, None),
            "phen_within_level_branch_switch": (8, None),
            "phen_depth_shift": (8, None),
            "phen_field_gain_loss": (20, None),
        },
        "token_family_weights": {
            "encoded_level_token_a": 2.0,
            "encoded_level_token_b": 2.0,
            "level_phenotype_a": 2.0,
            "level_phenotype_b": 2.0,
            "field_count_transition_v3": 1.5,
            "level_set_transition": 1.5,
            "phenotype_transition": 1.5,
            "field_location_tokens_a": 1.0,
            "field_location_tokens_b": 1.0,
            "dominant_run_tokens_a": 1.0,
            "dominant_run_tokens_b": 1.0,
        },
        "min_token_candidate_support": 2,
        "max_token_hits": 3,
        "token_decay": 0.50,
        "quality_retention": float(quality_retention),
        "diversity_retention": float(diversity_retention),
        "solver_time_limit": float(solver_time_limit),
    }


def build_semantic_solver_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain high-quality candidates within every required semantic stratum.

    The full 17,845 rows are characterized and exported first.  This bounded
    pool keeps the MILP tractable without returning to a quality-only shortlist.
    """

    eligible = frame.loc[frame["selection_eligible"].astype(bool)].sort_values(
        ["visual_score_v3", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    parts: list[pd.DataFrame] = [
        eligible.groupby(
            ["pair_key", "correlation_bin"], observed=True, sort=True
        ).head(8)
    ]
    for column in ("selector_field_category_a", "selector_field_category_b"):
        parts.append(
            eligible.groupby(["pair_key", column], observed=True, sort=True).head(5)
        )
    for column in (
        "encoded_level_token_a",
        "encoded_level_token_b",
        "level_phenotype_a",
        "level_phenotype_b",
    ):
        parts.append(
            eligible.groupby(["pair_key", column], observed=True, sort=True).head(2)
        )
    for column in (
        "phen_same_run",
        "phen_within_level_branch_switch",
        "phen_depth_shift",
        "phen_field_gain_loss",
    ):
        parts.append(
            eligible.loc[eligible[column].astype(bool)]
            .groupby("pair_key", observed=True, sort=True)
            .head(4)
        )
    # Preserve high-quality exemplars of each concrete run/within-run location,
    # even when that location is rare within a session-pair/r45 cell.
    lookup = eligible.set_index("candidate_id", drop=False)
    for column in (
        "field_location_tokens_a",
        "field_location_tokens_b",
        "dominant_run_tokens_a",
        "dominant_run_tokens_b",
    ):
        exploded = eligible[["candidate_id", "visual_score_v3", column]].explode(column)
        ids = (
            exploded.dropna(subset=[column])
            .sort_values(
                ["visual_score_v3", "candidate_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            .groupby(column, observed=True, sort=True)
            .head(3)["candidate_id"]
        )
        parts.append(lookup.loc[list(dict.fromkeys(map(str, ids)))].copy())
    parts.append(eligible.loc[eligible["fixed_main_fig2b"].astype(bool)])
    pool = pd.concat(parts, ignore_index=True).drop_duplicates(
        "candidate_id", keep="first"
    )
    return pool.sort_values(
        ["visual_score_v3", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_candidates(
    frame: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    *,
    required_pairs: list[str],
    quality_retention: float,
    diversity_retention: float,
    solver_time_limit: float,
    similarity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    eligible = frame.loc[frame["selection_eligible"].astype(bool)].copy()
    conflicts: list[tuple[str, str]] = []
    attempts: list[dict[str, Any]] = []
    final_diagnostics: dict[str, Any] | None = None
    final_selected: pd.DataFrame | None = None
    for iteration in range(1, 7):
        kwargs = _selector_kwargs(
            eligible,
            required_pairs=required_pairs,
            conflict_pairs=conflicts,
            quality_retention=quality_retention,
            diversity_retention=diversity_retention,
            solver_time_limit=solver_time_limit,
        )
        try:
            selected, diagnostics = select_representation_diverse_candidates(
                eligible, **kwargs
            )
        except RuntimeError:
            if float(quality_retention) <= 0.97:
                raise
            kwargs["quality_retention"] = 0.97
            selected, diagnostics = select_representation_diverse_candidates(
                eligible, **kwargs
            )
            diagnostics["quality_retention_fallback_from"] = quality_retention
        found = _selected_similarity_conflicts(
            selected, embeddings, threshold=similarity_threshold
        )
        attempts.append(
            {
                "iteration": iteration,
                "n_input_conflicts": len(conflicts),
                "n_selected_conflicts": len(found),
                "quality_achieved": diagnostics["quality_achieved"],
                "diversity_achieved": diagnostics["diversity_achieved"],
            }
        )
        final_selected = selected
        final_diagnostics = diagnostics
        if len(found) == 0:
            break
        known = {tuple(sorted(edge)) for edge in conflicts}
        for first, second in found[["candidate_id_1", "candidate_id_2"]].itertuples(
            index=False, name=None
        ):
            edge = tuple(sorted((str(first), str(second))))
            if edge not in known:
                conflicts.append(edge)
                known.add(edge)
    if final_selected is None or final_diagnostics is None:
        raise RuntimeError("v3 selection did not run")
    remaining = _selected_similarity_conflicts(
        final_selected, embeddings, threshold=similarity_threshold
    )
    if len(remaining):
        raise RuntimeError("lazy image-conflict selection did not converge")
    final_diagnostics["lazy_image_conflict_iterations"] = attempts
    final_diagnostics["image_similarity_threshold"] = similarity_threshold
    return final_selected, final_diagnostics, pd.DataFrame(attempts)


def materialize_selected(
    selected: pd.DataFrame,
    sessions: Mapping[str, Any],
    anchor_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    maps = v2.materialize_selected_maps(selected, sessions, anchor_maps)
    crosswalk = build_fine_maze_crosswalk()
    for row in selected.itertuples(index=False):
        candidate_id = str(row.candidate_id)
        values = maps[candidate_id]
        description = describe_display_pair(
            np.asarray(values["normalized_a"], dtype=float),
            np.asarray(values["map_a"], dtype=float),
            np.asarray(values["normalized_b"], dtype=float),
            np.asarray(values["map_b"], dtype=float),
            np.asarray(values["common_mask"], dtype=bool),
            crosswalk=crosswalk,
        )
        values["field_labels_a"] = np.asarray(
            description["a"]["field_labels"], dtype=np.int16
        )
        values["field_labels_b"] = np.asarray(
            description["b"]["field_labels"], dtype=np.int16
        )
        values["fields_a"] = description["a"]["fields"]
        values["fields_b"] = description["b"]["fields"]
    return maps


def export_selected_maps(
    selected: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    ordered = [maps[str(value)] for value in selected["candidate_id"]]
    npz_path = output_dir / "selected_maps_fields_45_v3.npz"
    np.savez_compressed(
        npz_path,
        candidate_id=np.asarray(selected["candidate_id"].astype(str), dtype="U"),
        visual_rank=selected["visual_rank"].to_numpy(dtype=int),
        normalized_a=np.stack([value["normalized_a"] for value in ordered]),
        normalized_b=np.stack([value["normalized_b"] for value in ordered]),
        smoothed_c_raw_a=np.stack([value["map_a"] for value in ordered]),
        smoothed_c_raw_b=np.stack([value["map_b"] for value in ordered]),
        occupancy_mask_a=np.stack([value["mask_a"] for value in ordered]),
        occupancy_mask_b=np.stack([value["mask_b"] for value in ordered]),
        common_mask=np.stack([value["common_mask"] for value in ordered]),
        display_field_labels_a=np.stack([value["field_labels_a"] for value in ordered]),
        display_field_labels_b=np.stack([value["field_labels_b"] for value in ordered]),
        q99_scale_a=np.asarray([value["q99_scale_a"] for value in ordered]),
        q99_scale_b=np.asarray([value["q99_scale_b"] for value in ordered]),
    )

    cross = build_fine_maze_crosswalk()
    yy, xx = np.indices((45, 45))
    long_rows: list[pd.DataFrame] = []
    topology = {
        "bin_y": yy.ravel(),
        "bin_x": xx.ravel(),
        "maze_x_15": cross.maze_x.ravel(),
        "maze_y_15": cross.maze_y.ravel(),
        "physical_cell_id": cross.cell_id.ravel(),
        "run_index": cross.run_index.ravel(),
        "tree_depth_0based": cross.tree_depth.ravel(),
        "display_level_1based": cross.display_level.ravel(),
        "branch_ordinal_1based": cross.branch_ordinal_1based.ravel(),
        "branch_path": cross.branch_path.ravel(),
        "coarse_within_run_index_0based": cross.coarse_within_run_index.ravel(),
        "fine_within_run_index_0based": cross.fine_within_run_index.ravel(),
        "within_run_fraction": cross.within_run_fraction.ravel(),
        "within_run_region": cross.within_run_region.ravel(),
        "maze_support": cross.maze_support.ravel().astype(int),
    }
    for row, values in zip(selected.itertuples(index=False), ordered, strict=True):
        for side in ("a", "b"):
            long_rows.append(
                pd.DataFrame(
                    {
                        "visual_rank": int(row.visual_rank),
                        "candidate_id": str(row.candidate_id),
                        "side": side.upper(),
                        **topology,
                        "session_occupied": np.asarray(values[f"mask_{side}"])
                        .ravel()
                        .astype(int),
                        "common_valid": np.asarray(values["common_mask"])
                        .ravel()
                        .astype(int),
                        "smoothed_c_map_raw": np.asarray(values[f"map_{side}"]).ravel(),
                        "normalized_independent_q99": np.asarray(
                            values[f"normalized_{side}"]
                        ).ravel(),
                        "display_field_id": np.asarray(
                            values[f"field_labels_{side}"]
                        ).ravel(),
                        "q99_scale": float(values[f"q99_scale_{side}"]),
                    }
                )
            )
    bins_path = output_dir / "selected_map_bins_topology_45_v3.csv.gz"
    pd.concat(long_rows, ignore_index=True).to_csv(
        bins_path, index=False, compression="gzip"
    )
    return [npz_path.name, bins_path.name]


def _display_cmap() -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    return cmap


def _draw_maze(ax: plt.Axes) -> None:
    maze = NewMaze(6)
    wall = maze_wall_to_fine_coordinates(maze.wa)
    ax.plot(wall[:, 0], wall[:, 1], color="white", lw=0.45, alpha=0.96)
    ax.set_xlim(-0.5, 44.5)
    ax.set_ylim(44.5, -0.5)


def _draw_map(ax: plt.Axes, values: np.ndarray) -> None:
    ax.imshow(
        values,
        origin="upper",
        cmap=_display_cmap(),
        norm=Normalize(0.0, 1.0),
        interpolation="nearest",
    )
    _draw_maze(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_fields(
    ax: plt.Axes,
    labels: np.ndarray,
    fields: Iterable[Mapping[str, Any]],
) -> None:
    for field in fields:
        field_id = int(field["field_id"])
        level = int(field["dominant_display_level_1based"])
        color = LEVEL_COLORS[level - 1]
        mask = np.asarray(labels) == field_id
        if np.any(mask):
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=0.85,
                origin="upper",
            )
        ax.text(
            float(field["peak_col_45"]),
            float(field["peak_row_45"]),
            str(level),
            color="white",
            fontsize=4.3,
            ha="center",
            va="center",
            weight="bold",
            bbox={"facecolor": color, "edgecolor": "none", "pad": 0.25},
        )


def _levels_label(row: pd.Series | Any, side: str) -> str:
    value = getattr(row, f"encoded_level_token_{side}")
    return str(value).replace("|", ",")


def _candidate_title(row: Any) -> str:
    fixed = " | Fig. 2b" if bool(row.fixed_main_fig2b) else ""
    preservation = (
        f" | level overlap={float(row.level_profile_overlap):.2f}"
        if hasattr(row, "level_profile_overlap")
        else ""
    )
    return (
        f"#{int(row.visual_rank):03d}{fixed} | q={float(row.visual_score_v3):.3f} | "
        f"r45={float(row.place_r_45_display):.2f}{preservation}\n"
        f"{row.pair_key} | A{int(row.train_id)} -> B{int(row.test_id)}"
    )


def _render_page(
    rows: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    *,
    page_number: int,
    total_pages: int,
    overlay_fields: bool,
    title_prefix: str = "ED6 level-aware candidates",
) -> plt.Figure:
    fig = plt.figure(figsize=(18.0, 11.2))
    outer = fig.add_gridspec(
        5,
        5,
        left=0.022,
        right=0.988,
        top=0.94,
        bottom=0.085,
        wspace=0.16,
        hspace=0.38,
    )
    for position, row in enumerate(rows.itertuples(index=False)):
        slot = outer[position // 5, position % 5]
        pair_grid = slot.subgridspec(1, 2, wspace=0.025)
        values = maps[str(row.candidate_id)]
        for column, side in enumerate(("a", "b")):
            ax = fig.add_subplot(pair_grid[0, column])
            _draw_map(ax, np.asarray(values[f"normalized_{side}"]))
            if overlay_fields:
                _draw_fields(
                    ax,
                    np.asarray(values[f"field_labels_{side}"]),
                    values[f"fields_{side}"],
                )
            ax.set_title(
                f"{side.upper()}: {int(getattr(row, f'field_count_{side}'))} fields; "
                f"{_levels_label(row, side)}",
                fontsize=5.8,
                pad=1.2,
            )
        title_ax = fig.add_subplot(slot, frame_on=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.set_title(_candidate_title(row), fontsize=6.4, loc="left", pad=12.5)

    mode = "field-level audit" if overlay_fields else "normalized real place maps"
    fig.suptitle(
        f"{title_prefix}: {mode} | page {page_number}/{total_pages}",
        fontsize=12,
        y=0.993,
    )
    colorbar_axis = fig.add_axes([0.35, 0.034, 0.30, 0.014])
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=_display_cmap()),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[0, 0.5, 1],
    )
    colorbar.set_label(COLORBAR_LABEL, fontsize=7, labelpad=2)
    colorbar.ax.tick_params(labelsize=6, length=2)
    if overlay_fields:
        handles = [
            plt.Line2D([0], [0], color=color, lw=2, label=f"level {level}")
            for level, color in enumerate(LEVEL_COLORS, start=1)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.060),
            ncol=7,
            frameon=False,
            fontsize=6,
        )
    return fig


def render_contact_sheets(
    selected: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    *,
    overlay_fields: bool,
) -> list[str]:
    stem = "field_overlay" if overlay_fields else "ranked"
    pdf_path = output_dir / f"{stem}_100_contact_sheets.pdf"
    paths = [pdf_path.name]
    total_pages = int(np.ceil(len(selected) / 25))
    with PdfPages(pdf_path) as pdf:
        for page_number, start in enumerate(range(0, len(selected), 25), start=1):
            rows = selected.iloc[start : start + 25]
            fig = _render_page(
                rows,
                maps,
                page_number=page_number,
                total_pages=total_pages,
                overlay_fields=overlay_fields,
            )
            pdf.savefig(fig, facecolor="white")
            for extension in ("png", "svg"):
                path = output_dir / (
                    f"{stem}_{start + 1:03d}_{start + len(rows):03d}.{extension}"
                )
                kwargs = {"dpi": 300} if extension == "png" else {}
                fig.savefig(path, facecolor="white", **kwargs)
                paths.append(path.name)
            plt.close(fig)
    return paths


def render_individuals(
    selected: pd.DataFrame,
    maps: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    directory = output_dir / "individual_previews"
    directory.mkdir(parents=True, exist_ok=True)
    for stale_preview in directory.glob("rank_*.png"):
        stale_preview.unlink()
    paths: list[str] = []
    for row in selected.itertuples(index=False):
        values = maps[str(row.candidate_id)]
        fig, axes = plt.subplots(1, 2, figsize=(3.8, 2.65))
        for ax, side, session in zip(
            axes, ("a", "b"), (row.src_session, row.dst_session), strict=True
        ):
            _draw_map(ax, np.asarray(values[f"normalized_{side}"]))
            ax.set_title(
                f"{session}\n{int(getattr(row, f'field_count_{side}'))} fields; "
                f"{_levels_label(row, side)}",
                fontsize=6.2,
                pad=2,
            )
        fixed = " | fixed Fig. 2b" if bool(row.fixed_main_fig2b) else ""
        preservation = (
            f" | level overlap={float(row.level_profile_overlap):.2f}"
            if hasattr(row, "level_profile_overlap")
            else ""
        )
        fig.suptitle(
            f"Rank {int(row.visual_rank):03d}{fixed} | "
            f"q={float(row.visual_score_v3):.3f} | "
            f"r45={float(row.place_r_45_display):.3f}{preservation}",
            fontsize=7,
            y=0.99,
        )
        fig.text(
            0.5,
            0.215,
            f"{row.pair_key} | A{int(row.train_id)} -> B{int(row.test_id)}",
            ha="center",
            fontsize=5.8,
        )
        cax = fig.add_axes([0.30, 0.105, 0.40, 0.022])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=Normalize(0.0, 1.0), cmap=_display_cmap()
            ),
            cax=cax,
            orientation="horizontal",
            ticks=[0, 0.5, 1],
        )
        cb.set_label(COLORBAR_LABEL, fontsize=5.2, labelpad=1)
        cb.ax.tick_params(labelsize=5, length=1.5)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.78, bottom=0.34, wspace=0.04)
        name = (
            f"rank_{int(row.visual_rank):03d}__"
            f"{v2.v1.safe_name(row.pair_key)}__A{int(row.train_id)}_B{int(row.test_id)}.png"
        )
        path = directory / name
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)).replace("\\", "/"))
    return paths


def build_audits(
    selected: pd.DataFrame,
    selected_maps: Mapping[str, Mapping[str, Any]],
    selected_fields: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    embeddings: Mapping[str, np.ndarray],
    *,
    similarity_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    pair_coverage = (
        selected.groupby("pair_key", as_index=False)
        .agg(
            n_examples=("candidate_id", "size"),
            best_visual_rank=("visual_rank", "min"),
            min_r45=("place_r_45_display", "min"),
            max_r45=("place_r_45_display", "max"),
        )
        .sort_values("pair_key", kind="mergesort")
    )
    pair_coverage.to_csv(output_dir / "pair_coverage.csv", index=False)
    correlation_coverage = (
        selected.groupby("correlation_bin", observed=False)
        .agg(
            n_examples=("candidate_id", "size"),
            min_r45=("place_r_45_display", "min"),
            max_r45=("place_r_45_display", "max"),
            median_visual_score_v3=("visual_score_v3", "median"),
        )
        .reindex(CORRELATION_BIN_LABELS)
        .reset_index()
    )
    correlation_coverage.to_csv(
        output_dir / "correlation_coverage_r45.csv", index=False
    )

    side_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    for side in ("a", "b"):
        for bucket, count in (
            selected[f"field_count_bucket_{side}"].value_counts().items()
        ):
            side_rows.append(
                {
                    "side": side.upper(),
                    "field_count_bucket": bucket,
                    "n_maps": int(count),
                }
            )
        for level in range(1, 8):
            level_rows.append(
                {
                    "side": side.upper(),
                    "display_level_1based": level,
                    "n_maps": int(selected[f"encodes_level_{level}_{side}"].sum()),
                    "n_same_level_multifield_maps": int(
                        (
                            selected[f"is_same_level_multifield_{side}"].astype(bool)
                            & selected[f"encoded_level_token_{side}"].eq(f"L{level}")
                        ).sum()
                    ),
                }
            )
    field_coverage = pd.DataFrame(side_rows)
    field_coverage.to_csv(output_dir / "field_count_coverage_by_side.csv", index=False)
    level_coverage = pd.DataFrame(level_rows)
    level_coverage.to_csv(
        output_dir / "encoded_level_coverage_by_side.csv", index=False
    )

    phenotype_names = (
        "phen_same_run",
        "phen_within_level_branch_switch",
        "phen_depth_shift",
        "phen_field_gain_loss",
    )
    phenotype_coverage = pd.DataFrame(
        {
            "phenotype": phenotype_names,
            "n_selected": [int(selected[value].sum()) for value in phenotype_names],
        }
    )
    phenotype_coverage.to_csv(output_dir / "pair_phenotype_coverage.csv", index=False)

    run_coverage = (
        selected_fields.groupby(["dominant_display_level_1based"], as_index=False)
        .agg(
            n_fields=("field_id", "size"),
            n_unique_runs=("dominant_run_index", "nunique"),
            n_unique_locations=("field_location_token", "nunique"),
        )
        .rename(columns={"dominant_display_level_1based": "display_level_1based"})
    )
    run_coverage.to_csv(output_dir / "field_run_location_coverage.csv", index=False)
    location_coverage = (
        selected_fields.groupby(
            [
                "side",
                "dominant_display_level_1based",
                "dominant_run_index",
                "dominant_within_run_region",
            ],
            as_index=False,
        )
        .agg(n_fields=("field_id", "size"), n_candidates=("candidate_id", "nunique"))
        .sort_values(["side", "dominant_display_level_1based", "dominant_run_index"])
    )
    location_coverage.to_csv(
        output_dir / "field_location_coverage_long.csv", index=False
    )

    node_rows = pd.concat(
        [
            selected[["candidate_id", "node_a"]].rename(columns={"node_a": "node"}),
            selected[["candidate_id", "node_b"]].rename(columns={"node_b": "node"}),
        ],
        ignore_index=True,
    )
    duplicate_nodes = (
        node_rows.groupby("node", as_index=False)
        .agg(
            n_selected_uses=("candidate_id", "size"),
            selected_candidate_ids=(
                "candidate_id",
                lambda values: ";".join(sorted(map(str, values))),
            ),
        )
        .query("n_selected_uses > 1")
    )
    duplicate_nodes.to_csv(output_dir / "duplicate_neuron_node_audit.csv", index=False)
    similarity = _selected_similarity_conflicts(
        selected, embeddings, threshold=similarity_threshold
    )
    similarity.to_csv(
        output_dir / "selected_image_similarity_conflicts.csv", index=False
    )

    normalization_ok = True
    invalid_nan_ok = True
    field_count_ok = True
    maze_support_ok = True
    cross = build_fine_maze_crosswalk()
    for row in selected.itertuples(index=False):
        values = selected_maps[str(row.candidate_id)]
        common = np.asarray(values["common_mask"], dtype=bool)
        maze_support_ok &= bool(np.all(~common | cross.maze_support))
        for side in ("a", "b"):
            normalized = np.asarray(values[f"normalized_{side}"], dtype=float)
            labels = np.asarray(values[f"field_labels_{side}"], dtype=int)
            normalization_ok &= bool(
                np.isfinite(normalized[common]).all()
                and np.isclose(np.nanmax(normalized), 1.0, atol=1e-6)
            )
            invalid_nan_ok &= bool(np.isnan(normalized[~common]).all())
            field_count_ok &= int(labels.max()) == int(
                getattr(row, f"field_count_{side}")
            )

    field_hierarchy_mismatches = int(
        (
            selected_fields["dominant_tree_depth_0based"].astype(int) + 1
            != selected_fields["dominant_display_level_1based"].astype(int)
        ).sum()
    )
    impossible_branch_ordinals = int(
        (
            (selected_fields["dominant_branch_ordinal_1based"].astype(int) < 1)
            | (
                selected_fields["dominant_branch_ordinal_1based"].astype(int)
                > 2 ** selected_fields["dominant_tree_depth_0based"].astype(int)
            )
        ).sum()
    )
    expected_location_prefix = (
        "L"
        + selected_fields["dominant_display_level_1based"].astype(int).astype(str)
        + ":B"
        + selected_fields["dominant_branch_ordinal_1based"].astype(int).astype(str)
        + ":"
    )
    location_token_mismatches = int(
        sum(
            not str(token).startswith(prefix)
            for token, prefix in zip(
                selected_fields["field_location_token"],
                expected_location_prefix,
                strict=True,
            )
        )
    )
    contradictory_same_run_depth_shift = int(
        (
            selected["phen_same_run"].astype(bool)
            & selected["phen_depth_shift"].astype(bool)
        ).sum()
    )

    side_counts = {
        f"{side.upper()}:{bucket}": int(
            selected[f"field_count_bucket_{side}"].eq(bucket).sum()
        )
        for side in ("a", "b")
        for bucket in ("1", "2", "3", "4-5", "6-8", "9+")
    }
    level_counts = {
        f"{side.upper()}:L{level}": int(selected[f"encodes_level_{level}_{side}"].sum())
        for side in ("a", "b")
        for level in range(1, 8)
    }
    checks: dict[str, Any] = {
        "n_selected": int(len(selected)),
        "n_fixed_main_fig2b": int(selected["fixed_main_fig2b"].sum()),
        "n_session_pairs": int(selected["pair_key"].nunique()),
        "pair_min": int(pair_coverage["n_examples"].min()),
        "pair_max": int(pair_coverage["n_examples"].max()),
        "correlation_bin_min": int(correlation_coverage["n_examples"].min()),
        "correlation_bin_max": int(correlation_coverage["n_examples"].max()),
        "n_r45_le_minus_0_25": int((selected["place_r_45_display"] <= -0.25).sum()),
        "n_r45_ge_0_90": int((selected["place_r_45_display"] >= 0.90).sum()),
        "r45_min": float(selected["place_r_45_display"].min()),
        "r45_max": float(selected["place_r_45_display"].max()),
        "duplicate_neuron_nodes": int(len(duplicate_nodes)),
        "image_conflicts_at_threshold": int(len(similarity)),
        "visual_rank_is_strict_quality_order": bool(
            selected["visual_score_v3"].is_monotonic_decreasing
        ),
        "all_normalized_maps_valid": bool(normalization_ok),
        "all_invalid_bins_nan": bool(invalid_nan_ok),
        "field_counts_match_exported_labels": bool(field_count_ok),
        "all_common_bins_map_to_physical_maze": bool(maze_support_ok),
        "field_level_run_hierarchy_mismatches": field_hierarchy_mismatches,
        "impossible_branch_ordinals": impossible_branch_ordinals,
        "field_location_token_mismatches": location_token_mismatches,
        "contradictory_same_run_depth_shift": contradictory_same_run_depth_shift,
        "quality_achieved": float(diagnostics["quality_achieved"]),
        "quality_floor": float(diagnostics["quality_floor"]),
        "diversity_achieved": float(diagnostics["diversity_achieved"]),
        "diversity_floor": float(diagnostics["diversity_floor"]),
        "field_count_side_counts": side_counts,
        "encoded_level_side_counts": level_counts,
        "phenotype_counts": {
            value: int(selected[value].sum()) for value in phenotype_names
        },
        "fraction_multifield_map_sides": float(
            (
                (selected["field_count_a"] >= 2).sum()
                + (selected["field_count_b"] >= 2).sum()
            )
            / 200.0
        ),
    }
    scalar_required = {
        "n_selected": 100,
        "n_fixed_main_fig2b": 4,
        "n_session_pairs": 36,
        "duplicate_neuron_nodes": 0,
        "image_conflicts_at_threshold": 0,
        "field_level_run_hierarchy_mismatches": 0,
        "impossible_branch_ordinals": 0,
        "field_location_token_mismatches": 0,
        "contradictory_same_run_depth_shift": 0,
    }
    for key, expected in scalar_required.items():
        if checks[key] != expected:
            raise AssertionError(checks)
    if checks["pair_min"] < 2 or checks["pair_max"] > 4:
        raise AssertionError(checks)
    if checks["correlation_bin_min"] < 5 or checks["correlation_bin_max"] > 15:
        raise AssertionError(checks)
    if checks["n_r45_le_minus_0_25"] < 4 or checks["n_r45_ge_0_90"] < 4:
        raise AssertionError(checks)
    if checks["quality_achieved"] + 1e-8 < checks["quality_floor"]:
        raise AssertionError(checks)
    if checks["diversity_achieved"] + 1e-8 < checks["diversity_floor"]:
        raise AssertionError(checks)
    if not all(
        checks[key]
        for key in (
            "visual_rank_is_strict_quality_order",
            "all_normalized_maps_valid",
            "all_invalid_bins_nan",
            "field_counts_match_exported_labels",
            "all_common_bins_map_to_physical_maze",
        )
    ):
        raise AssertionError(checks)
    for side in ("A", "B"):
        if side_counts[f"{side}:1"] > 25:
            raise AssertionError(checks)
        for bucket, minimum in {
            "2": 15,
            "3": 12,
            "4-5": 15,
            "6-8": 10,
            "9+": 4,
        }.items():
            if side_counts[f"{side}:{bucket}"] < minimum:
                raise AssertionError(checks)
        for level in range(1, 8):
            minimum = 8 if level < 7 else 4
            if level_counts[f"{side}:L{level}"] < minimum:
                raise AssertionError(checks)
    for name, minimum in {
        "phen_same_run": 10,
        "phen_within_level_branch_switch": 8,
        "phen_depth_shift": 8,
        "phen_field_gain_loss": 20,
    }.items():
        if checks["phenotype_counts"][name] < minimum:
            raise AssertionError(checks)
    (output_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return checks


def write_readme(
    output_dir: Path,
    checks: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    overlap_v2: int,
) -> None:
    quality_reference = (
        "a proven constrained quality optimum"
        if bool(diagnostics["quality_reference_proven_optimal"])
        else "a constraint-verified quality reference"
    )
    diversity_reference = (
        "a proven diversity optimum"
        if bool(diagnostics["diversity_reference_proven_optimal"])
        else "a constraint-verified time-limit incumbent, not a proven global optimum"
    )
    if bool(diagnostics["final_quality_refinement_fell_back_to_stage2"]):
        final_refinement = (
            "Stage 3 did not yield a separately verified feasible refinement, so "
            "the final gallery uses the verified Stage 2 solution."
        )
    elif bool(diagnostics["final_quality_refinement_proven_optimal"]):
        final_refinement = "Stage 3 then found the proven quality-refined solution."
    else:
        final_refinement = (
            "Stage 3 yielded a verified refinement without proving global optimality."
        )
    text = f"""# ED6 level-aware real place-map candidates: v3 ranked 100

This is an **illustrative candidate gallery**, not a population estimate or an
inferential place-field analysis. It was not promoted into the Extended Data
assembly. The gallery retains the four Main Figure 2b pairs and chooses 96
additional examples after characterizing all 17,845 available pairs.

## What changed from v2

V2 used whole-image similarity plus an appearance score that explicitly
favoured one to three compact components. That made the selected maps look
clean, but it underrepresented the maze's multi-field and level-like codes.
V3 defines diversity in maze coordinates: display-field count, encoded
tree-depth level 1-7, run/branch, coarse outward within-run region, and the
A-to-B representation transition. Exact peak and mass-weighted within-run
positions are exported for audit but are not optimization tokens. Only
{overlap_v2}/100 v2 choices remain.

## Result

- Selected pairs: {checks['n_selected']}; fixed Figure 2b pairs: {checks['n_fixed_main_fig2b']}.
- Session pairs: all {checks['n_session_pairs']}, with {checks['pair_min']}-{checks['pair_max']} examples per pair.
- Multi-field map sides: {100 * checks['fraction_multifield_map_sides']:.1f}%.
- Recomputed r45 range: {checks['r45_min']:.3f} to {checks['r45_max']:.3f}; every fixed r45 bin contains {checks['correlation_bin_min']}-{checks['correlation_bin_max']} examples.
- Reused neuron nodes: {checks['duplicate_neuron_nodes']}.
- Near-duplicate map pairs at cosine similarity >= {IMAGE_SIMILARITY_THRESHOLD:.2f}: {checks['image_conflicts_at_threshold']}.

## Display fields and encoded levels

Each map is normalized independently by its own q99 on the two-session common
occupancy mask and clipped to 0-1. White is outside common occupancy. A
**display field** is an 8-connected component at normalized C >=
{DISPLAY_FIELD_THRESHOLD:.2f} containing at least
{DISPLAY_FIELD_MIN_FINE_BINS} fine 45x45 bins. This rule was selected by a
threshold/minimum-area visual sensitivity audit. Display fields are not
shuffle-significant place fields and should not be used as a biological field
count endpoint.

Array row/column maps exactly to maze `(x, y)` as `x=column//3`,
`y=row//3`. Each field is assigned to its dominant raw-C tree-depth level,
run/branch, and outward within-run position. Complete level/run mass shares are
also exported, so fields spanning junctions are not forced to a single label
in the audit trail. The current representation is true tree-depth7, displayed
as levels 1-7; it is not the legacy geometric8 definition.

## Quality and selection

The new image-quality score uses contrast, within-active-region coherence, and
a lower visibility gate. It contains no component-count reward, no upper
active-area penalty, and no largest-component dominance term. Within-session
reliability contributes 15% and activity 5%; correlation, spatial information,
entropy, symmetry, field count, and encoded level never enter the quality
score. The delivered 100 are ordered strictly by descending `visual_score_v3`.

Selection uses a three-stage MILP target. Stage 1 found {quality_reference}
over the {int(diagnostics['candidate_count']):,}-candidate semantic solver pool;
all 17,845 pairs were characterized before pool construction. Stage 2 retained at least
{100 * float(diagnostics['quality_retention']):.1f}% of that quality while
maximizing diminishing-return coverage of level, run, field-location, and
transition tokens; its diversity reference was {diversity_reference}.
{final_refinement} It achieves quality
{float(diagnostics['quality_achieved']):.3f} against the required floor
{float(diagnostics['quality_floor']):.3f}, and diversity
{float(diagnostics['diversity_achieved']):.3f} against the retained-reference
floor {float(diagnostics['diversity_floor']):.3f}. All hard selection
constraints were rechecked on the delivered rows.

## Review files

Start with `ranked_100_contact_sheets.pdf`. The same examples with field
outlines and level numbers are in `field_overlay_100_contact_sheets.pdf`.
`selected_100.csv`, `selected_display_fields.csv`, and the coverage tables
explain why each candidate was retained. Full-universe candidate and field
descriptors are gzip CSVs. Raw/normalized map arrays, field labels, common
support, q99 scales, and bin-level topology are in
`selected_maps_fields_45_v3.npz` and
`selected_map_bins_topology_45_v3.csv.gz`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _artifact_records(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "selection_manifest.json":
            continue
        if any(part.startswith(".mplconfig") for part in path.parts):
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    v2_dir = Path(args.v2_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else v2_dir / "intermediate_context_maps_v3"
    )
    required_v2 = (
        "full_scored_candidates_17845.csv",
        "selection_manifest.json",
    )
    missing = [name for name in required_v2 if not (v2_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen v2 inputs: {missing}")

    print("Loading frozen exhaustive v2 registry/cache...", flush=True)
    pool, reconciliation, source_components = v2.v1.build_exact_ed6_pool()
    anchors, anchor_maps = v2.v1.build_anchor_rows()
    scored_v2 = pd.read_csv(v2_dir / "full_scored_candidates_17845.csv")
    sessions = v2.load_cached_sessions(
        pool,
        cache_dir=cache_dir,
        parameters=PlaceMapCacheParameters(),
    )
    representation_path = output_dir / "full_representation_candidates_17845.csv.gz"
    fields_path = output_dir / "full_display_fields_17845_pairs.csv.gz"
    solver_pool_path = output_dir / "semantic_solver_pool.csv.gz"
    embeddings_path = output_dir / "semantic_solver_pool_embeddings.npz"
    if bool(args.reuse_characterization):
        for path in (representation_path, fields_path, solver_pool_path):
            if not path.is_file():
                raise FileNotFoundError(f"cannot reuse missing artifact: {path}")
        print("Reusing complete v3 characterization tables...", flush=True)
        representation = pd.read_csv(representation_path)
        field_frame = pd.read_csv(fields_path)
        solver_pool = pd.read_csv(solver_pool_path)
        if embeddings_path.is_file():
            embeddings = load_embeddings(
                embeddings_path, solver_pool["candidate_id"].astype(str)
            )
        else:
            embeddings = compute_candidate_embeddings(
                solver_pool, sessions, anchor_maps
            )
            save_embeddings(
                embeddings,
                embeddings_path,
                solver_pool["candidate_id"].astype(str),
            )
    else:
        representation, field_frame, embeddings = characterize_universe(
            pool, anchors, scored_v2, sessions, anchor_maps
        )
        _export_frame(representation, representation_path, compression="gzip")
        _export_frame(field_frame, fields_path, compression="gzip")
        unconstrained = representation.loc[
            representation["selection_eligible"].astype(bool)
        ].head(100)
        _export_frame(unconstrained, output_dir / "unconstrained_quality_top_100.csv")
        solver_pool = build_semantic_solver_pool(representation)
        _export_frame(solver_pool, solver_pool_path, compression="gzip")
        embeddings = {
            str(candidate_id): embeddings[str(candidate_id)]
            for candidate_id in solver_pool["candidate_id"]
        }
        save_embeddings(
            embeddings,
            embeddings_path,
            solver_pool["candidate_id"].astype(str),
        )

    print("Selecting representation-diverse candidates...", flush=True)
    selected, diagnostics, attempts = select_candidates(
        solver_pool,
        embeddings,
        required_pairs=sorted(pool["pair_key"].astype(str).unique()),
        quality_retention=float(args.quality_retention),
        diversity_retention=float(args.diversity_retention),
        solver_time_limit=float(args.solver_time_limit),
        similarity_threshold=float(args.image_similarity_threshold),
    )
    _export_frame(selected, output_dir / "selected_100.csv")
    attempts.to_csv(output_dir / "selection_iterations.csv", index=False)
    (output_dir / "selection_solver_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    rank_lookup = selected.set_index("candidate_id")["visual_rank"]
    selected_fields = field_frame.loc[
        field_frame["candidate_id"].isin(set(selected["candidate_id"]))
    ].copy()
    selected_fields.insert(
        0,
        "visual_rank",
        selected_fields["candidate_id"].map(rank_lookup).astype(int),
    )
    selected_fields = selected_fields.sort_values(
        ["visual_rank", "side", "field_id"], kind="mergesort"
    )
    _export_frame(selected_fields, output_dir / "selected_display_fields.csv")

    selected_maps = materialize_selected(selected, sessions, anchor_maps)
    map_paths = export_selected_maps(selected, selected_maps, output_dir)
    checks = build_audits(
        selected,
        selected_maps,
        selected_fields,
        diagnostics,
        embeddings,
        similarity_threshold=float(args.image_similarity_threshold),
        output_dir=output_dir,
    )
    contact_paths = render_contact_sheets(
        selected, selected_maps, output_dir, overlay_fields=False
    )
    overlay_paths: list[str] = []
    if not bool(args.skip_overlays):
        overlay_paths = render_contact_sheets(
            selected, selected_maps, output_dir, overlay_fields=True
        )
    individual_paths: list[str] = []
    if not bool(args.skip_individuals):
        individual_paths = render_individuals(selected, selected_maps, output_dir)

    v2_selected = pd.read_csv(v2_dir / "selected_100.csv")
    overlap_v2 = int(
        selected["candidate_id"].isin(set(v2_selected["candidate_id"])).sum()
    )
    comparison = selected[
        [
            "candidate_id",
            "visual_rank",
            "visual_score_v3",
            "field_count_a",
            "field_count_b",
            "encoded_level_token_a",
            "encoded_level_token_b",
        ]
    ].merge(
        v2_selected[["candidate_id", "visual_rank", "visual_score"]].rename(
            columns={
                "visual_rank": "v2_visual_rank",
                "visual_score": "v2_visual_score",
            }
        ),
        on="candidate_id",
        how="left",
    )
    comparison["selected_in_v2"] = comparison["v2_visual_rank"].notna()
    _export_frame(comparison, output_dir / "v2_v3_selection_comparison.csv")
    reconciliation.to_csv(output_dir / "pair_reconciliation.csv", index=False)
    source_components.to_csv(output_dir / "source_component_manifest.csv", index=False)
    anchors.to_csv(output_dir / "fixed_fig2b_anchor_provenance.csv", index=False)
    write_readme(output_dir, checks, diagnostics, overlap_v2=overlap_v2)

    source_paths = {
        "frozen_v2_selection_manifest": v2_dir / "selection_manifest.json",
        "frozen_v2_full_scored": v2_dir / "full_scored_candidates_17845.csv",
        "frozen_v2_cache_index": cache_dir / "cache_index_complete.csv",
        "representation_descriptor_module": REPO_ROOT
        / "src"
        / "ca_analysis"
        / "ed06_representation_diversity.py",
        "representation_selector_module": REPO_ROOT
        / "src"
        / "ca_analysis"
        / "ed06_representation_selection.py",
        "builder": Path(__file__).resolve(),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "illustrative_level_aware_candidate_gallery_not_cohort_inference",
        "assembly_promotion": False,
        "script": str(Path(__file__).relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_v2_directory": str(v2_dir),
        "cache_directory": str(cache_dir),
        "pool": {
            "active_rows": int(len(pool)),
            "fixed_anchor_rows": int(len(anchors)),
            "full_characterized_rows": int(len(representation)),
            "display_field_rows": int(len(field_frame)),
            "session_pairs": int(pool["pair_key"].nunique()),
        },
        "display": {
            "signal": "denoised C",
            "bins": [45, 45],
            "normalization": "independent_per_map_q99_on_common_support_clipped_0_1",
            "colorbar_label": COLORBAR_LABEL,
            "outside_common_support": "white_NaN",
            "maze_wall_affine": "fine=(wall+0.5)*3-0.5",
        },
        "display_field_definition": {
            "threshold": DISPLAY_FIELD_THRESHOLD,
            "minimum_component_area_fine_bins": DISPLAY_FIELD_MIN_FINE_BINS,
            "connectivity": 8,
            "status": "descriptive_visual_selection_aid_not_shuffle_significant",
        },
        "representation_space": {
            "level_definition": "true_tree_depth7",
            "display_levels": [1, 2, 3, 4, 5, 6, 7],
            "location": "run_branch_and_coarse_outward_within_run_region",
            "exact_peak_and_mass_weighted_positions": "exported_for_audit_not_optimization_tokens",
            "encoded_level": "union_of_display_field_dominant_raw_C_mass_levels",
            "complete_field_level_run_mass_shares_exported": True,
        },
        "quality": {
            "map_score": "0.80 field-count-neutral image + 0.15 within-session reliability + 0.05 activity",
            "pair_score": "0.60 min(A,B) + 0.40 mean(A,B)",
            "order": "descending visual_score_v3 only",
            "excludes": [
                "cross_session_correlation",
                "spatial_information",
                "entropy",
                "symmetry",
                "field_count",
                "encoded_level",
            ],
        },
        "selection": diagnostics,
        "checks": checks,
        "v2_v3_selected_overlap": overlap_v2,
        "source_sha256": {
            key: sha256_file(path)
            for key, path in source_paths.items()
            if path.is_file()
        },
        "expected_primary_artifacts": [
            *map_paths,
            *contact_paths,
            *overlay_paths,
            *individual_paths,
        ],
        "artifacts": _artifact_records(output_dir),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2, sort_keys=True), flush=True)
    print(f"Wrote level-aware v3 gallery to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
