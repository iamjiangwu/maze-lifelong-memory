"""Representation-aware constrained selection for the ED6 example gallery.

This module only selects from candidate-level descriptors computed elsewhere.
It deliberately does not define place fields, maze levels, or display-map
quality.  Those meanings remain explicit inputs to the selector.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


CountBounds = tuple[int | None, int | None]


@dataclass(frozen=True)
class _ConstraintRecord:
    family: str
    name: str
    lower: float
    upper: float


class _ConstraintBuilder:
    def __init__(self, n_variables: int) -> None:
        self.n_variables = int(n_variables)
        self.rows: list[sparse.csr_matrix] = []
        self.records: list[_ConstraintRecord] = []

    def add(
        self,
        coefficients: np.ndarray | Sequence[float],
        *,
        lower: float = -np.inf,
        upper: float = np.inf,
        family: str,
        name: str,
    ) -> None:
        values = np.asarray(coefficients, dtype=float).ravel()
        if values.size != self.n_variables:
            raise ValueError(
                f"constraint {name!r} has {values.size} coefficients; "
                f"expected {self.n_variables}"
            )
        indices = np.flatnonzero(values)
        row = sparse.csr_matrix(
            (
                values[indices],
                (np.zeros(indices.size, dtype=int), indices),
            ),
            shape=(1, self.n_variables),
        )
        self.rows.append(row)
        self.records.append(
            _ConstraintRecord(
                family=str(family),
                name=str(name),
                lower=float(lower),
                upper=float(upper),
            )
        )

    def matrix(self) -> sparse.csr_matrix:
        if not self.rows:
            return sparse.csr_matrix((0, self.n_variables), dtype=float)
        return sparse.vstack(self.rows, format="csr")


def _normalize_bounds(
    bounds: CountBounds,
    *,
    name: str,
) -> tuple[float, float]:
    lower, upper = bounds
    lo = 0.0 if lower is None else float(lower)
    hi = np.inf if upper is None else float(upper)
    if lo < 0 or hi < 0 or lo > hi:
        raise ValueError(f"invalid bounds for {name!r}: {bounds!r}")
    return lo, hi


def _as_bool(series: pd.Series) -> np.ndarray:
    """Coerce common serialized boolean values without treating 'False' as true."""

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).to_numpy(dtype=bool)
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "",
        "none",
        "nan",
        "<na>",
        "nat",
    }
    result = np.zeros(len(series), dtype=bool)
    for position, value in enumerate(series.to_numpy(dtype=object)):
        if _is_missing_scalar(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            result[position] = bool(value)
            continue
        if isinstance(value, (int, np.integer, float, np.floating)):
            result[position] = bool(value)
            continue
        text = str(value).strip().lower()
        if text in true_values:
            result[position] = True
        elif text not in false_values:
            raise ValueError(f"cannot coerce boolean value {value!r}")
    return result


def _valid_text(value: object) -> str | None:
    if _is_missing_scalar(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "nat"}:
        return None
    return text


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _iter_cell_tokens(value: object) -> Iterable[str]:
    """Yield tokens from an iterable, JSON list, or scalar cell value.

    Ordinary strings are kept intact.  This lets a caller distinguish a
    compound level-set token (for example, ``L2|L4``) from a tuple containing
    two independent field-location tokens.
    """

    if _is_missing_scalar(value):
        return
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none"}:
            return
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                for token in decoded:
                    normalized = _valid_text(token)
                    if normalized is not None:
                        yield normalized
                return
        yield text
        return
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series)):
        for token in value:
            normalized = _valid_text(token)
            if normalized is not None:
                yield normalized
        return
    normalized = _valid_text(value)
    if normalized is not None:
        yield normalized


def _add_index_constraint(
    builder: _ConstraintBuilder,
    indices: Iterable[int],
    *,
    lower: float = -np.inf,
    upper: float = np.inf,
    family: str,
    name: str,
) -> None:
    use = np.asarray(sorted(set(int(index) for index in indices)), dtype=int)
    row = sparse.csr_matrix(
        (
            np.ones(use.size, dtype=float),
            (np.zeros(use.size, dtype=int), use),
        ),
        shape=(1, builder.n_variables),
    )
    builder.rows.append(row)
    builder.records.append(
        _ConstraintRecord(
            family=str(family),
            name=str(name),
            lower=float(lower),
            upper=float(upper),
        )
    )


def _solve_binary_milp(
    objective: np.ndarray,
    matrix: sparse.csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    stage: str,
    time_limit: float,
) -> Any:
    n_variables = int(objective.size)
    result = milp(
        c=np.asarray(objective, dtype=float),
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    accepted_incumbent = False
    if not bool(result.success) and result.x is not None and int(result.status) == 1:
        candidate = np.rint(np.asarray(result.x, dtype=float))
        integral = bool(np.max(np.abs(np.asarray(result.x) - candidate)) <= 1e-5)
        achieved = np.asarray(matrix @ candidate, dtype=float).ravel()
        feasible = bool(
            integral
            and np.all(achieved >= np.asarray(lower, dtype=float) - 1e-6)
            and np.all(achieved <= np.asarray(upper, dtype=float) + 1e-6)
        )
        if feasible:
            result.x = candidate
            accepted_incumbent = True
    result["accepted_time_limit_incumbent"] = accepted_incumbent
    if (not bool(result.success) and not accepted_incumbent) or result.x is None:
        raise RuntimeError(
            f"representation selection {stage} failed: "
            f"status={result.status}, {result.message}"
        )
    return result


def _solver_diagnostics(result: Any) -> dict[str, object]:
    return {
        "success": bool(result.success),
        "accepted_time_limit_incumbent": bool(
            result.get("accepted_time_limit_incumbent", False)
        ),
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "mip_gap": (
            None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)
        ),
        "mip_node_count": (
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
    }


def _constraint_diagnostics(
    builder: _ConstraintBuilder,
    chosen: np.ndarray,
) -> dict[str, object]:
    matrix = builder.matrix()
    achieved = np.asarray(matrix @ chosen, dtype=float).ravel()
    details: list[dict[str, object]] = []
    family_rows: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(builder.records):
        family_rows[record.family].append(index)
        if record.family not in {"node_reuse", "conflict"}:
            details.append(
                {
                    "family": record.family,
                    "name": record.name,
                    "lower": None if not np.isfinite(record.lower) else record.lower,
                    "upper": None if not np.isfinite(record.upper) else record.upper,
                    "achieved": float(achieved[index]),
                }
            )

    families: dict[str, dict[str, object]] = {}
    tolerance = 1e-6
    for family, indices in sorted(family_rows.items()):
        values = achieved[indices]
        records = [builder.records[index] for index in indices]
        violations = 0
        for value, record in zip(values, records, strict=True):
            if value < record.lower - tolerance or value > record.upper + tolerance:
                violations += 1
        families[family] = {
            "count": len(indices),
            "violations": int(violations),
            "minimum_achieved": float(np.min(values)),
            "maximum_achieved": float(np.max(values)),
        }
    return {
        "count": len(builder.records),
        "families": families,
        "details": details,
    }


def _build_candidate_tokens(
    work: pd.DataFrame,
    *,
    side_category_columns: Sequence[str],
    level_membership_columns: Mapping[str, Sequence[str]],
    semantic_token_columns: Sequence[str],
    phenotype_flag_columns: Sequence[str],
    token_family_weights: Mapping[str, float],
    include_side_category_tokens: bool,
    include_level_tokens: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    tokens_by_candidate: list[set[str]] = [set() for _ in range(len(work))]
    token_weights: dict[str, float] = {}

    def add(index: int, family: str, value: object) -> None:
        normalized = _valid_text(value)
        if normalized is None:
            return
        token = f"{family}={normalized}"
        tokens_by_candidate[index].add(token)
        weight = float(token_family_weights.get(family, 1.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"invalid token weight for {family!r}: {weight!r}")
        token_weights[token] = weight

    if include_side_category_tokens:
        for column in side_category_columns:
            for index, value in enumerate(work[column].to_numpy(dtype=object)):
                add(index, column, value)
        if len(side_category_columns) == 2:
            first, second = side_category_columns
            for index, (value_a, value_b) in enumerate(
                zip(
                    work[first].to_numpy(dtype=object),
                    work[second].to_numpy(dtype=object),
                    strict=True,
                )
            ):
                a = _valid_text(value_a)
                b = _valid_text(value_b)
                if a is not None and b is not None:
                    add(index, "side_category_pair", f"{a}->{b}")

    if include_level_tokens:
        for level, columns in level_membership_columns.items():
            for column in columns:
                values = _as_bool(work[column])
                for index in np.flatnonzero(values):
                    add(int(index), column, level)

    for column in semantic_token_columns:
        family = str(column)
        for index, value in enumerate(work[column].to_numpy(dtype=object)):
            for token in _iter_cell_tokens(value):
                add(index, family, token)

    for column in phenotype_flag_columns:
        values = _as_bool(work[column])
        for index in np.flatnonzero(values):
            add(int(index), "phenotype_flag", column)

    membership: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(tokens_by_candidate):
        for token in sorted(tokens):
            membership[token].append(index)
    return (
        {
            token: np.asarray(indices, dtype=int)
            for token, indices in sorted(membership.items())
        },
        token_weights,
    )


def _actual_diversity_score(
    chosen: np.ndarray,
    token_membership: Mapping[str, np.ndarray],
    token_weights: Mapping[str, float],
    *,
    max_token_hits: int,
    token_decay: float,
) -> tuple[float, int, dict[int, int]]:
    score = 0.0
    covered = 0
    hit_histogram: Counter[int] = Counter()
    for token, indices in token_membership.items():
        hits = int(np.sum(chosen[indices] > 0.5))
        if hits <= 0:
            continue
        covered += 1
        hit_histogram[hits] += 1
        levels = min(hits, int(max_token_hits))
        score += float(token_weights[token]) * sum(
            float(token_decay) ** level for level in range(levels)
        )
    return float(score), int(covered), dict(sorted(hit_histogram.items()))


def select_representation_diverse_candidates(
    frame: pd.DataFrame,
    *,
    target_n: int = 100,
    required_pairs: Sequence[str] | None = None,
    pair_bounds: Mapping[str, CountBounds] | None = None,
    min_per_pair: int = 2,
    max_per_pair: int = 4,
    required_correlation_bins: Sequence[str] | None = None,
    correlation_bin_bounds: Mapping[str, CountBounds] | None = None,
    min_per_correlation_bin: int = 5,
    max_per_correlation_bin: int = 15,
    min_low_extreme: int = 4,
    low_extreme_threshold: float = -0.25,
    min_high_extreme: int = 4,
    high_extreme_threshold: float = 0.90,
    correlation_col: str = "place_r_45_display",
    score_col: str = "visual_score",
    refinement_score_col: str | None = None,
    fixed_col: str = "fixed_main_fig2b",
    node_columns: Sequence[str] = ("node_a", "node_b"),
    conflict_pairs: Sequence[tuple[str, str]] | None = None,
    side_category_columns: Sequence[str] = (
        "field_count_category_a",
        "field_count_category_b",
    ),
    side_category_bounds: Mapping[str, CountBounds] | None = None,
    single_field_category: str | None = "1",
    max_single_field_side_fraction: float | None = 0.45,
    level_membership_columns: Mapping[str, Sequence[str]] | None = None,
    minimum_sides_per_level: int | Mapping[str, int] | None = None,
    flag_bounds: Mapping[str, CountBounds] | None = None,
    semantic_token_columns: Sequence[str] = (),
    phenotype_flag_columns: Sequence[str] = (),
    token_family_weights: Mapping[str, float] | None = None,
    include_side_category_tokens: bool = True,
    include_level_tokens: bool = True,
    min_token_candidate_support: int = 2,
    max_token_hits: int = 3,
    token_decay: float = 0.5,
    quality_retention: float = 0.98,
    diversity_retention: float = 0.98,
    solver_time_limit: float = 120.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a quality-ranked, representation-diverse candidate set.

    The hard constraints operate only on caller-provided candidate metadata.
    Side-category bounds count both map sides, so a selected row contributes
    zero, one, or two sides to a category.  Level minima use the same side-count
    convention.  ``flag_bounds`` constrain generic candidate-level boolean
    phenotypes; bounded flags also enter token coverage.  No definition of a
    field, level, or phenotype is embedded in this module.

    Optimization is lexicographic in three stages: maximize ``score_col``;
    retain a configurable fraction of that optimum while maximizing
    diminishing-return token coverage; then retain a configurable fraction of
    the diversity optimum while maximizing ``refinement_score_col``.  When no
    refinement column is supplied, the final stage maximizes ``score_col``
    again, preserving the historical quality-refinement behavior.
    """

    if int(target_n) <= 0:
        raise ValueError("target_n must be positive")
    if not 0.0 < float(quality_retention) <= 1.0:
        raise ValueError("quality_retention must be in (0, 1]")
    if not 0.0 < float(diversity_retention) <= 1.0:
        raise ValueError("diversity_retention must be in (0, 1]")
    if int(max_token_hits) <= 0:
        raise ValueError("max_token_hits must be positive")
    if not 0.0 < float(token_decay) <= 1.0:
        raise ValueError("token_decay must be in (0, 1]")
    if int(min_token_candidate_support) <= 0:
        raise ValueError("min_token_candidate_support must be positive")

    level_membership_columns = dict(level_membership_columns or {})
    flag_bounds = dict(flag_bounds or {})
    token_family_weights = dict(token_family_weights or {})
    side_category_bounds = dict(side_category_bounds or {})
    pair_bounds = dict(pair_bounds or {})
    correlation_bin_bounds = dict(correlation_bin_bounds or {})

    phenotype_token_columns = tuple(
        dict.fromkeys([*phenotype_flag_columns, *flag_bounds.keys()])
    )
    required_columns = {
        "candidate_id",
        "pair_key",
        "correlation_bin",
        correlation_col,
        score_col,
        *side_category_columns,
        *semantic_token_columns,
        *phenotype_token_columns,
        *node_columns,
    }
    for columns in level_membership_columns.values():
        required_columns.update(columns)
    missing = sorted(required_columns - set(frame.columns))
    if refinement_score_col is not None and refinement_score_col not in frame.columns:
        missing.append(refinement_score_col)
        missing = sorted(set(missing))
    if missing:
        raise ValueError(f"selection table is missing columns: {missing}")

    work = frame.reset_index(drop=True).copy()
    if work["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")
    if work["candidate_id"].map(_valid_text).isna().any():
        raise ValueError("candidate_id must not be missing")
    if work["pair_key"].map(_valid_text).isna().any():
        raise ValueError("pair_key must not be missing")
    if work["correlation_bin"].map(_valid_text).isna().any():
        raise ValueError("correlation_bin must not be missing")
    n_candidates = len(work)
    if n_candidates < int(target_n):
        raise ValueError(f"only {n_candidates} candidates for target_n={int(target_n)}")

    score_series = pd.to_numeric(work[score_col], errors="coerce")
    missing_score_count = int(score_series.isna().sum())
    scores = score_series.fillna(0.0).to_numpy(dtype=float)
    if np.any(scores < 0.0) or not np.all(np.isfinite(scores)):
        raise ValueError(f"{score_col} must contain finite nonnegative values")

    refinement_col = refinement_score_col or score_col
    refinement_series = pd.to_numeric(work[refinement_col], errors="coerce")
    missing_refinement_count = int(refinement_series.isna().sum())
    refinement_scores = refinement_series.fillna(0.0).to_numpy(dtype=float)
    if np.any(refinement_scores < 0.0) or not np.all(np.isfinite(refinement_scores)):
        raise ValueError(f"{refinement_col} must contain finite nonnegative values")

    fixed = (
        _as_bool(work[fixed_col])
        if fixed_col in work.columns
        else np.zeros(n_candidates, dtype=bool)
    )
    if int(np.sum(fixed)) > int(target_n):
        raise ValueError("fixed candidates exceed target_n")

    builder = _ConstraintBuilder(n_candidates)
    _add_index_constraint(
        builder,
        range(n_candidates),
        lower=float(target_n),
        upper=float(target_n),
        family="target",
        name="selected_rows",
    )
    for index in np.flatnonzero(fixed):
        _add_index_constraint(
            builder,
            [int(index)],
            lower=1.0,
            upper=1.0,
            family="fixed_anchor",
            name=str(work.at[int(index), "candidate_id"]),
        )

    pair_values = work["pair_key"].astype(str)
    if required_pairs is None:
        required_pairs = sorted(pair_values.unique().tolist())
    pair_bound_config: dict[str, CountBounds] = {
        str(pair): (int(min_per_pair), int(max_per_pair)) for pair in required_pairs
    }
    pair_bound_config.update({str(key): value for key, value in pair_bounds.items()})
    for pair, bounds in pair_bound_config.items():
        lower, upper = _normalize_bounds(bounds, name=f"pair:{pair}")
        indices = np.flatnonzero(pair_values.eq(pair).to_numpy())
        if indices.size < lower:
            raise ValueError(
                f"pair {pair!r} has {indices.size} candidates, below floor {lower:g}"
            )
        _add_index_constraint(
            builder,
            indices,
            lower=lower,
            upper=upper,
            family="pair",
            name=pair,
        )

    correlation_bin_values = work["correlation_bin"].astype(str)
    if required_correlation_bins is None:
        required_correlation_bins = sorted(correlation_bin_values.unique().tolist())
    correlation_bound_config: dict[str, CountBounds] = {
        str(label): (int(min_per_correlation_bin), int(max_per_correlation_bin))
        for label in required_correlation_bins
    }
    correlation_bound_config.update(
        {str(key): value for key, value in correlation_bin_bounds.items()}
    )
    for label, bounds in correlation_bound_config.items():
        lower, upper = _normalize_bounds(bounds, name=f"correlation_bin:{label}")
        indices = np.flatnonzero(correlation_bin_values.eq(label).to_numpy())
        if indices.size < lower:
            raise ValueError(
                f"correlation bin {label!r} has {indices.size} candidates, "
                f"below floor {lower:g}"
            )
        _add_index_constraint(
            builder,
            indices,
            lower=lower,
            upper=upper,
            family="correlation_bin",
            name=label,
        )

    correlations = pd.to_numeric(work[correlation_col], errors="coerce").to_numpy(
        dtype=float
    )
    finite_correlation = np.isfinite(correlations)
    if int(min_low_extreme) > 0:
        indices = np.flatnonzero(
            finite_correlation & (correlations <= float(low_extreme_threshold))
        )
        if indices.size < int(min_low_extreme):
            raise ValueError(f"only {indices.size} low-extreme candidates")
        _add_index_constraint(
            builder,
            indices,
            lower=float(min_low_extreme),
            upper=float(target_n),
            family="correlation_extreme",
            name=f"low<={float(low_extreme_threshold):g}",
        )
    if int(min_high_extreme) > 0:
        indices = np.flatnonzero(
            finite_correlation & (correlations >= float(high_extreme_threshold))
        )
        if indices.size < int(min_high_extreme):
            raise ValueError(f"only {indices.size} high-extreme candidates")
        _add_index_constraint(
            builder,
            indices,
            lower=float(min_high_extreme),
            upper=float(target_n),
            family="correlation_extreme",
            name=f"high>={float(high_extreme_threshold):g}",
        )

    node_to_indices: dict[str, set[int]] = defaultdict(set)
    for column in node_columns:
        for index, value in enumerate(work[column].to_numpy(dtype=object)):
            node = _valid_text(value)
            if node is not None:
                node_to_indices[node].add(index)
    for node, indices in sorted(node_to_indices.items()):
        if len(indices) <= 1:
            continue
        _add_index_constraint(
            builder,
            indices,
            lower=0.0,
            upper=1.0,
            family="node_reuse",
            name=node,
        )

    id_to_index = {
        str(candidate_id): int(index)
        for index, candidate_id in enumerate(work["candidate_id"])
    }
    ignored_conflict_edges = 0
    seen_conflicts: set[tuple[int, int]] = set()
    for first, second in conflict_pairs or ():
        if str(first) not in id_to_index or str(second) not in id_to_index:
            ignored_conflict_edges += 1
            continue
        index_a = id_to_index[str(first)]
        index_b = id_to_index[str(second)]
        if index_a == index_b:
            continue
        edge = tuple(sorted((index_a, index_b)))
        if edge in seen_conflicts:
            continue
        seen_conflicts.add(edge)
        _add_index_constraint(
            builder,
            edge,
            lower=0.0,
            upper=1.0,
            family="conflict",
            name=f"{first}|{second}",
        )

    category_values = {
        column: work[column].map(_valid_text).to_numpy(dtype=object)
        for column in side_category_columns
    }
    category_bound_config: dict[str, CountBounds] = {
        str(category): bounds for category, bounds in side_category_bounds.items()
    }
    if single_field_category is not None and max_single_field_side_fraction is not None:
        fraction = float(max_single_field_side_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("max_single_field_side_fraction must be in [0, 1]")
        category = str(single_field_category)
        cap = int(np.floor(2 * int(target_n) * fraction))
        existing_lower, existing_upper = category_bound_config.get(
            category, (None, None)
        )
        upper = cap if existing_upper is None else min(int(existing_upper), cap)
        category_bound_config[category] = (existing_lower, upper)

    for category, bounds in category_bound_config.items():
        lower, upper = _normalize_bounds(bounds, name=f"side_category:{category}")
        coefficients = np.zeros(n_candidates, dtype=float)
        for values in category_values.values():
            coefficients += np.asarray(values == category, dtype=float)
        if float(np.sum(coefficients)) < lower:
            raise ValueError(
                f"side category {category!r} has only "
                f"{int(np.sum(coefficients))} available sides, below floor {lower:g}"
            )
        builder.add(
            coefficients,
            lower=lower,
            upper=upper,
            family="side_category",
            name=category,
        )

    if minimum_sides_per_level is None:
        level_minima: dict[str, int] = {}
    elif isinstance(minimum_sides_per_level, Mapping):
        level_minima = {
            str(level): int(minimum)
            for level, minimum in minimum_sides_per_level.items()
        }
    else:
        level_minima = {
            str(level): int(minimum_sides_per_level)
            for level in level_membership_columns
        }
    unknown_levels = sorted(set(level_minima) - set(level_membership_columns))
    if unknown_levels:
        raise ValueError(f"level minima lack membership columns: {unknown_levels}")
    for level, minimum in level_minima.items():
        if minimum < 0:
            raise ValueError(f"negative level minimum for {level!r}")
        coefficients = np.zeros(n_candidates, dtype=float)
        for column in level_membership_columns[level]:
            coefficients += _as_bool(work[column]).astype(float)
        if float(np.sum(coefficients)) < float(minimum):
            raise ValueError(
                f"level {level!r} has only {int(np.sum(coefficients))} "
                f"available sides, below floor {minimum}"
            )
        builder.add(
            coefficients,
            lower=float(minimum),
            upper=float(2 * int(target_n)),
            family="level",
            name=level,
        )

    for column, bounds in flag_bounds.items():
        lower, upper = _normalize_bounds(bounds, name=f"flag:{column}")
        coefficients = _as_bool(work[column]).astype(float)
        if float(np.sum(coefficients)) < lower:
            raise ValueError(
                f"flag {column!r} has only {int(np.sum(coefficients))} "
                f"available candidates, below floor {lower:g}"
            )
        builder.add(
            coefficients,
            lower=lower,
            upper=upper,
            family="flag",
            name=column,
        )

    base_matrix = builder.matrix()
    base_lower = np.asarray([record.lower for record in builder.records], dtype=float)
    base_upper = np.asarray([record.upper for record in builder.records], dtype=float)

    quality_objective = -scores
    stage1 = _solve_binary_milp(
        quality_objective,
        base_matrix,
        base_lower,
        base_upper,
        stage="stage 1 (quality optimum)",
        time_limit=solver_time_limit,
    )
    chosen_stage1 = (stage1.x > 0.5).astype(float)
    quality_optimum = float(scores @ chosen_stage1)
    quality_floor = float(quality_retention) * quality_optimum

    token_membership, token_weights = _build_candidate_tokens(
        work,
        side_category_columns=side_category_columns,
        level_membership_columns=level_membership_columns,
        semantic_token_columns=semantic_token_columns,
        phenotype_flag_columns=phenotype_token_columns,
        token_family_weights=token_family_weights,
        include_side_category_tokens=include_side_category_tokens,
        include_level_tokens=include_level_tokens,
    )
    token_membership = {
        token: indices
        for token, indices in token_membership.items()
        if indices.size >= int(min_token_candidate_support)
        and token_weights[token] > 0.0
    }
    token_weights = {token: token_weights[token] for token in token_membership}

    auxiliary: list[tuple[str, int, float]] = []
    for token, indices in token_membership.items():
        n_levels = min(int(max_token_hits), int(indices.size), int(target_n))
        for level in range(1, n_levels + 1):
            auxiliary.append(
                (
                    token,
                    level,
                    float(token_weights[token]) * float(token_decay) ** (level - 1),
                )
            )
    n_auxiliary = len(auxiliary)
    n_variables = n_candidates + n_auxiliary
    extended_base = sparse.hstack(
        [base_matrix, sparse.csr_matrix((base_matrix.shape[0], n_auxiliary))],
        format="csr",
    )
    stage2_rows: list[sparse.csr_matrix] = [extended_base]
    stage2_lower: list[np.ndarray] = [base_lower]
    stage2_upper: list[np.ndarray] = [base_upper]

    quality_row = sparse.csr_matrix(
        np.concatenate([scores, np.zeros(n_auxiliary, dtype=float)]).reshape(1, -1)
    )
    stage2_rows.append(quality_row)
    stage2_lower.append(np.asarray([quality_floor - 1e-9], dtype=float))
    stage2_upper.append(np.asarray([np.inf], dtype=float))

    token_to_auxiliary: dict[str, list[tuple[int, int]]] = defaultdict(list)
    token_constraint_rows: list[sparse.csr_matrix] = []
    token_constraint_lower: list[float] = []
    token_constraint_upper: list[float] = []
    for auxiliary_index, (token, level, _weight) in enumerate(auxiliary):
        variable_index = n_candidates + auxiliary_index
        token_to_auxiliary[token].append((level, variable_index))
        indices = token_membership[token]
        columns = np.concatenate([indices, np.asarray([variable_index], dtype=int)])
        values = np.concatenate(
            [np.ones(indices.size, dtype=float), np.asarray([-float(level)])]
        )
        token_constraint_rows.append(
            sparse.csr_matrix(
                (values, (np.zeros(columns.size, dtype=int), columns)),
                shape=(1, n_variables),
            )
        )
        token_constraint_lower.append(0.0)
        token_constraint_upper.append(np.inf)
    for entries in token_to_auxiliary.values():
        entries.sort()
        for (_first_level, first), (_second_level, second) in zip(
            entries[:-1], entries[1:], strict=True
        ):
            token_constraint_rows.append(
                sparse.csr_matrix(
                    (
                        np.asarray([1.0, -1.0]),
                        (np.asarray([0, 0]), np.asarray([first, second])),
                    ),
                    shape=(1, n_variables),
                )
            )
            token_constraint_lower.append(0.0)
            token_constraint_upper.append(np.inf)
    if token_constraint_rows:
        stage2_rows.append(sparse.vstack(token_constraint_rows, format="csr"))
        stage2_lower.append(np.asarray(token_constraint_lower, dtype=float))
        stage2_upper.append(np.asarray(token_constraint_upper, dtype=float))

    stage2_matrix = sparse.vstack(stage2_rows, format="csr")
    stage2_lower_array = np.concatenate(stage2_lower)
    stage2_upper_array = np.concatenate(stage2_upper)
    diversity_weights = np.asarray(
        [weight for _token, _level, weight in auxiliary], dtype=float
    )
    stage2_objective = np.zeros(n_variables, dtype=float)
    if n_auxiliary:
        stage2_objective[n_candidates:] = -diversity_weights
    else:
        stage2_objective[:n_candidates] = quality_objective
    stage2 = _solve_binary_milp(
        stage2_objective,
        stage2_matrix,
        stage2_lower_array,
        stage2_upper_array,
        stage="stage 2 (token diversity optimum)",
        time_limit=solver_time_limit,
    )
    chosen_stage2 = (stage2.x[:n_candidates] > 0.5).astype(float)
    diversity_optimum, _covered_stage2, _histogram_stage2 = _actual_diversity_score(
        chosen_stage2,
        token_membership,
        token_weights,
        max_token_hits=max_token_hits,
        token_decay=token_decay,
    )
    diversity_floor = float(diversity_retention) * diversity_optimum

    stage3_rows = [stage2_matrix]
    stage3_lower = [stage2_lower_array]
    stage3_upper = [stage2_upper_array]
    if n_auxiliary:
        diversity_row = sparse.csr_matrix(
            np.concatenate(
                [np.zeros(n_candidates, dtype=float), diversity_weights]
            ).reshape(1, -1)
        )
        stage3_rows.append(diversity_row)
        stage3_lower.append(np.asarray([diversity_floor - 1e-9], dtype=float))
        stage3_upper.append(np.asarray([np.inf], dtype=float))
    stage3_matrix = sparse.vstack(stage3_rows, format="csr")
    stage3_objective = np.zeros(n_variables, dtype=float)
    stage3_objective[:n_candidates] = -refinement_scores
    stage3_error: str | None = None
    try:
        stage3 = _solve_binary_milp(
            stage3_objective,
            stage3_matrix,
            np.concatenate(stage3_lower),
            np.concatenate(stage3_upper),
            stage="stage 3 (quality within diversity floor)",
            time_limit=solver_time_limit,
        )
        chosen = (stage3.x[:n_candidates] > 0.5).astype(float)
    except RuntimeError as error:
        # Stage-2's candidate vector is itself feasible for the stage-3
        # quality/diversity floors.  scipy.milp does not accept a warm start,
        # so retain that verified vector when the refinement search cannot
        # rediscover an incumbent before its time limit.
        stage3 = None
        stage3_error = str(error)
        chosen = chosen_stage2.copy()
    chosen_indices = np.flatnonzero(chosen > 0.5)
    if chosen_indices.size != int(target_n):
        raise RuntimeError(
            f"solver selected {chosen_indices.size}, expected {int(target_n)}"
        )

    quality_achieved = float(scores @ chosen)
    refinement_achieved = float(refinement_scores @ chosen)
    diversity_achieved, covered_tokens, token_hit_histogram = _actual_diversity_score(
        chosen,
        token_membership,
        token_weights,
        max_token_hits=max_token_hits,
        token_decay=token_decay,
    )
    selected = work.iloc[chosen_indices].copy()
    selected = selected.sort_values(
        [score_col, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(0, "visual_rank", np.arange(1, len(selected) + 1, dtype=int))

    diagnostics: dict[str, object] = {
        "candidate_count": int(n_candidates),
        "target_n": int(target_n),
        "fixed_anchor_count": int(np.sum(fixed)),
        "missing_visual_score_count_filled_with_zero": missing_score_count,
        "refinement_score_col": refinement_col,
        "missing_refinement_score_count_filled_with_zero": (missing_refinement_count),
        "refinement_score_achieved": refinement_achieved,
        "quality_optimum": quality_optimum,
        "quality_reference_proven_optimal": bool(stage1.success),
        "quality_retention": float(quality_retention),
        "quality_floor": quality_floor,
        "quality_achieved": quality_achieved,
        "solver_time_limit_seconds": float(solver_time_limit),
        "diversity_reference": diversity_optimum,
        "diversity_optimum": diversity_optimum if stage2.success else None,
        "diversity_reference_proven_optimal": bool(stage2.success),
        "diversity_retention": float(diversity_retention),
        "diversity_floor": diversity_floor,
        "diversity_achieved": diversity_achieved,
        "final_quality_refinement_proven_optimal": bool(
            refinement_score_col is None and stage3 is not None and stage3.success
        ),
        "final_refinement_proven_optimal": bool(stage3 is not None and stage3.success),
        "final_quality_refinement_fell_back_to_stage2": stage3 is None,
        "final_quality_refinement_error": stage3_error,
        "available_diversity_tokens": int(len(token_membership)),
        "covered_diversity_tokens": covered_tokens,
        "token_hit_histogram": token_hit_histogram,
        "auxiliary_variable_count": int(n_auxiliary),
        "ignored_conflict_edges": int(ignored_conflict_edges),
        "constraints": _constraint_diagnostics(builder, chosen),
        "solver_stages": {
            "quality": _solver_diagnostics(stage1),
            "diversity": _solver_diagnostics(stage2),
            "quality_with_diversity_floor": (
                _solver_diagnostics(stage3) if stage3 is not None else None
            ),
        },
    }
    return selected, diagnostics


__all__ = ["CountBounds", "select_representation_diverse_candidates"]
